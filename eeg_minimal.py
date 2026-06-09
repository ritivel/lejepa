import os, json
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch, torch.nn as nn
import wandb, hydra, tqdm
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.ops import MLP

DEFAULT_CACHE = "/home/ubuntu/eeg-data-proc/eegData/preprocessed_resampled_200hz_peers"
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eeg_checkpoints")


class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device="cuda")
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class ViT1DEncoder(nn.Module):
    """1-D ViT: each view is a single-channel signal [1, window_len]."""

    def __init__(
        self,
        proj_dim=16,
        window_len=6000,
        patch=50,
        dim=384,
        depth=6,
        heads=6,
        mlp_ratio=4,
        emb_dim=512,
    ):
        super().__init__()
        assert window_len % patch == 0, "window_len must be divisible by patch"
        num_patches = window_len // patch
        self.patch_embed = nn.Conv1d(1, dim, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * mlp_ratio,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
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


def lejepa_losses(net, sigreg, vs, lamb):
    emb, proj = net(vs)
    inv_loss = (proj.mean(0) - proj).square().mean()
    sigreg_loss = sigreg(proj)
    lejepa_loss = sigreg_loss * lamb + inv_loss * (1 - lamb)
    return lejepa_loss, sigreg_loss, inv_loss


@hydra.main(version_base=None)
def main(cfg: DictConfig):
    wandb.init(project="LeJEPA-EEG", config=dict(cfg))
    torch.manual_seed(0)
    os.makedirs(CKPT_DIR, exist_ok=True)

    cache_root = cfg.get("cache_root", DEFAULT_CACHE)
    window_len = cfg.get("window_len", 6000)
    num_workers = cfg.get("num_workers", 8)
    eval_batches = cfg.get("eval_batches", 50)
    max_steps = cfg.get("max_steps_per_epoch", None)
    log_every = cfg.get("log_every", 100)

    train_ds = EEGWindowDataset(cache_root, "train", window_len)
    dev_ds = EEGWindowDataset(cache_root, "dev", window_len)
    print(f"train windows: {len(train_ds)} | dev windows: {len(dev_ds)}")
    train = DataLoader(
        train_ds,
        batch_size=cfg.bs,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    dev = DataLoader(
        dev_ds, batch_size=cfg.bs, num_workers=num_workers, pin_memory=True
    )

    net = ViT1DEncoder(
        proj_dim=cfg.proj_dim,
        window_len=window_len,
        patch=cfg.get("patch", 50),
        dim=cfg.get("dim", 384),
        depth=cfg.get("depth", 6),
        heads=cfg.get("heads", 6),
    ).to("cuda")
    sigreg = SIGReg().to("cuda")

    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=5e-2)
    steps_per_epoch = len(train) if max_steps is None else min(len(train), max_steps)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = max(1, min(steps_per_epoch, total_steps - 1))
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(
        opt, T_max=max(1, total_steps - warmup_steps), eta_min=cfg.lr / 1000
    )
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

    scaler = GradScaler()
    for epoch in range(cfg.epochs):
        net.train()
        for step, vs in enumerate(tqdm.tqdm(train, total=steps_per_epoch)):
            if max_steps is not None and step >= max_steps:
                break
            with autocast("cuda", dtype=torch.bfloat16):
                vs = vs.to("cuda", non_blocking=True)
                lejepa_loss, sigreg_loss, inv_loss = lejepa_losses(
                    net, sigreg, vs, cfg.lamb
                )
            opt.zero_grad()
            scaler.scale(lejepa_loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            wandb.log(
                {
                    "train/lejepa": lejepa_loss.item(),
                    "train/sigreg": sigreg_loss.item(),
                    "train/inv": inv_loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                }
            )
            if step % log_every == 0:
                print(
                    f"[epoch {epoch} step {step}] "
                    f"lejepa={lejepa_loss.item():.4f} "
                    f"sigreg={sigreg_loss.item():.4f} "
                    f"inv={inv_loss.item():.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )

        # Evaluation on dev split (unlabeled: track the SSL loss itself)
        net.eval()
        tot = {"lejepa": 0.0, "sigreg": 0.0, "inv": 0.0}
        n = 0
        with torch.inference_mode():
            for vs in dev:
                if n >= eval_batches:
                    break
                vs = vs.to("cuda", non_blocking=True)
                with autocast("cuda", dtype=torch.bfloat16):
                    lj, sg, iv = lejepa_losses(net, sigreg, vs, cfg.lamb)
                tot["lejepa"] += lj.item()
                tot["sigreg"] += sg.item()
                tot["inv"] += iv.item()
                n += 1
        n = max(1, n)
        wandb.log(
            {
                "dev/lejepa": tot["lejepa"] / n,
                "dev/sigreg": tot["sigreg"] / n,
                "dev/inv": tot["inv"] / n,
                "dev/epoch": epoch,
            }
        )
        print(
            f"[epoch {epoch}] dev/lejepa={tot['lejepa']/n:.4f} "
            f"dev/sigreg={tot['sigreg']/n:.4f} dev/inv={tot['inv']/n:.4f}"
        )
        ckpt = os.path.join(CKPT_DIR, "last.pt")
        torch.save(
            {"epoch": epoch, "model": net.state_dict(), "config": dict(cfg)}, ckpt
        )
    wandb.finish()


if __name__ == "__main__":
    main()
