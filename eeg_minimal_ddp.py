"""LeJEPA on PEERS EEG (128 channels as views) with optional multi-GPU DDP.

Single GPU : CUDA_VISIBLE_DEVICES=0 python eeg_minimal_ddp.py --bs 48 --epochs 20
8 GPUs     : torchrun --nproc_per_node=8 eeg_minimal_ddp.py --bs 48 --epochs 20

--bs is the GLOBAL (effective) batch; per-rank batch is bs/world. SIGReg uses the
repo's DDP-aware loss (all-reduces the empirical characteristic function over ranks,
so the statistic is over the GLOBAL batch), and the projector BatchNorm is converted
to SyncBatchNorm. At a fixed global batch this makes multi-GPU training mathematically
equivalent (in expectation) to single-GPU, just faster.
"""
import os, json, argparse
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch, torch.nn as nn
import torch.distributed as dist
import wandb, tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.ops import MLP

import lejepa

DEFAULT_CACHE = "/home/ubuntu/eeg-data-proc/eegData/preprocessed_resampled_200hz_peers"
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eeg_checkpoints")


def ddp_setup():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


class ViT1DEncoder(nn.Module):
    """1-D ViT: each view is a single-channel signal [1, window_len]."""

    def __init__(self, proj_dim=16, window_len=6000, patch=50, dim=384, depth=6, heads=6,
                 mlp_ratio=4, emb_dim=512):
        super().__init__()
        assert window_len % patch == 0, "window_len must be divisible by patch"
        num_patches = window_len // patch
        self.patch_embed = nn.Conv1d(1, dim, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * mlp_ratio,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, emb_dim)
        self.proj = MLP(emb_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def encode(self, x):
        x = self.patch_embed(x).transpose(1, 2)
        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        x = self.norm(self.transformer(x))
        return self.head(x[:, 0])

    def forward(self, x):
        N, V = x.shape[:2]
        emb = self.encode(x.flatten(0, 1))
        return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)


class EEGWindowDataset(torch.utils.data.Dataset):
    """Each item is one window [n_views=128, 1, window_len], views = channels."""

    def __init__(self, cache_root, split, window_len=6000, mmap_cache=128):
        self.cache_root = cache_root
        self.window_len = window_len
        self.mmap_cache = mmap_cache
        splits = json.load(open(os.path.join(cache_root, "splits.json")))
        win = pd.read_parquet(
            os.path.join(cache_root, "windows.parquet"),
            columns=["recording_id", "start_sample"],
        )
        keep = win["recording_id"].map(lambda r: splits.get(r) == split)
        win = win[keep].reset_index(drop=True)
        self.recording_id = win["recording_id"].to_numpy()
        self.start = win["start_sample"].to_numpy()
        self._cache = OrderedDict()

    def _get_array(self, recording_id):
        c = self._cache
        arr = c.get(recording_id)
        if arr is not None:
            c.move_to_end(recording_id)
            return arr
        path = os.path.join(self.cache_root, "peers_memory", recording_id + ".npy")
        arr = np.load(path, mmap_mode="r")
        c[recording_id] = arr
        if len(c) > self.mmap_cache:
            c.popitem(last=False)
        return arr

    def __getitem__(self, i):
        L = self.window_len
        start = int(self.start[i])
        arr = self._get_array(self.recording_id[i])
        w = np.asarray(arr[:, start : start + L], dtype=np.float32)
        if w.shape[1] != L:
            padded = np.zeros((w.shape[0], L), dtype=np.float32)
            padded[:, : w.shape[1]] = w
            w = padded
        mu = w.mean(axis=1, keepdims=True)
        sd = w.std(axis=1, keepdims=True)
        w = (w - mu) / (sd + 1e-8)
        return torch.from_numpy(w).unsqueeze(1)

    def __len__(self):
        return len(self.start)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_root", type=str, default=DEFAULT_CACHE)
    p.add_argument("--bs", type=int, default=48, help="GLOBAL batch size")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lamb", type=float, default=0.02)
    p.add_argument("--proj_dim", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window_len", type=int, default=6000)
    p.add_argument("--patch", type=int, default=50)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_batches", type=int, default=50)
    p.add_argument("--max_steps_per_epoch", type=int, default=None)
    p.add_argument("--wandb_project", type=str, default="LeJEPA-EEG")
    p.add_argument("--tag", type=str, default="ddp")
    cfg = p.parse_args()

    rank, world_size, local_rank = ddp_setup()
    device = f"cuda:{local_rank}"
    torch.manual_seed(cfg.seed)
    assert cfg.bs % world_size == 0, "global batch must be divisible by world size"
    per_rank_bs = cfg.bs // world_size
    if is_main():
        os.makedirs(CKPT_DIR, exist_ok=True)
        wandb.init(project=cfg.wandb_project, config=vars(cfg) | {"world_size": world_size})

    train_ds = EEGWindowDataset(cfg.cache_root, "train", cfg.window_len)
    dev_ds = EEGWindowDataset(cfg.cache_root, "dev", cfg.window_len)
    if is_main():
        print(f"train windows: {len(train_ds)} | dev windows: {len(dev_ds)} | "
              f"world={world_size} per_rank_bs={per_rank_bs}", flush=True)

    train_sampler = (
        DistributedSampler(train_ds, shuffle=True, seed=cfg.seed, drop_last=True)
        if world_size > 1 else None
    )
    dev_sampler = (
        DistributedSampler(dev_ds, shuffle=False, drop_last=True)
        if world_size > 1 else None
    )
    train = DataLoader(
        train_ds, batch_size=per_rank_bs, shuffle=(train_sampler is None),
        sampler=train_sampler, drop_last=True, num_workers=cfg.num_workers,
        pin_memory=True, persistent_workers=cfg.num_workers > 0,
    )
    dev = DataLoader(
        dev_ds, batch_size=per_rank_bs, sampler=dev_sampler,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    model = ViT1DEncoder(
        proj_dim=cfg.proj_dim, window_len=cfg.window_len, patch=cfg.patch,
        dim=cfg.dim, depth=cfg.depth, heads=cfg.heads,
    ).to(device)
    if world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    sigreg = lejepa.multivariate.SlicingUnivariateTest(
        univariate_test=lejepa.univariate.EppsPulley(t_max=3, n_points=17),
        num_slices=256,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=5e-2)
    max_steps = cfg.max_steps_per_epoch
    steps_per_epoch = len(train) if max_steps is None else min(len(train), max_steps)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = max(1, min(steps_per_epoch, total_steps - 1))
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=max(1, total_steps - warmup_steps), eta_min=cfg.lr / 1000)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

    def losses(vs):
        emb, proj = model(vs)
        inv = (proj.mean(0) - proj).square().mean()
        sig = sigreg(proj)
        return sig * cfg.lamb + inv * (1 - cfg.lamb), sig, inv

    scaler = GradScaler()
    for epoch in range(cfg.epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        it = tqdm.tqdm(train, total=steps_per_epoch) if is_main() else train
        for step, vs in enumerate(it):
            if max_steps is not None and step >= max_steps:
                break
            with autocast("cuda", dtype=torch.bfloat16):
                vs = vs.to(device, non_blocking=True)
                lejepa_loss, sigreg_loss, inv_loss = losses(vs)
            opt.zero_grad()
            scaler.scale(lejepa_loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            if is_main():
                wandb.log({
                    "train/lejepa": lejepa_loss.item(),
                    "train/sigreg": sigreg_loss.item(),
                    "train/inv": inv_loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                })
                if step % cfg.log_every == 0:
                    print(f"[{cfg.tag}][epoch {epoch} step {step}] "
                          f"lejepa={lejepa_loss.item():.4f} sigreg={sigreg_loss.item():.4f} "
                          f"inv={inv_loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}",
                          flush=True)

        # Dev eval: ALL ranks participate (the SIGReg all_reduce needs every rank).
        model.eval()
        sums = torch.zeros(3, device=device)
        n = 0
        if dev_sampler is not None:
            dev_sampler.set_epoch(epoch)
        with torch.inference_mode():
            for vs in dev:
                if n >= cfg.eval_batches:
                    break
                vs = vs.to(device, non_blocking=True)
                with autocast("cuda", dtype=torch.bfloat16):
                    lj, sg, iv = losses(vs)
                sums += torch.stack([lj.float(), sg.float(), iv.float()])
                n += 1
        if world_size > 1:
            dist.all_reduce(sums, op=dist.ReduceOp.AVG)
        n = max(1, n)
        if is_main():
            lj, sg, iv = (sums / n).tolist()
            wandb.log({"dev/lejepa": lj, "dev/sigreg": sg, "dev/inv": iv, "dev/epoch": epoch})
            print(f"[{cfg.tag}][epoch {epoch}] dev/lejepa={lj:.4f} dev/sigreg={sg:.4f} dev/inv={iv:.4f}",
                  flush=True)
            core = model.module if world_size > 1 else model
            torch.save({"epoch": epoch, "model": core.state_dict(), "config": vars(cfg)},
                       os.path.join(CKPT_DIR, "ddp_last.pt"))
        if world_size > 1:
            dist.barrier()

    if is_main():
        wandb.finish()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
