"""LeJEPA minimal (image / Imagenette) with optional multi-GPU DDP.

Runs identically as a single process or under torchrun:
  single GPU : CUDA_VISIBLE_DEVICES=1 python minimal_ddp.py --bs 256 --epochs 100
  4 GPUs     : torchrun --nproc_per_node=4 minimal_ddp.py --bs 256 --epochs 100

The global (effective) batch size is --bs in both cases; per-rank batch is bs/world.
SIGReg uses the repo's DDP-aware loss (lejepa.multivariate.SlicingUnivariateTest),
which all-reduces the empirical characteristic function so the statistic is computed
over the GLOBAL batch. The projector BatchNorm is converted to SyncBatchNorm so its
statistics are also global. This makes the multi-GPU run mathematically equivalent
(in expectation) to the single-GPU run at the same global batch.
"""
import os, argparse

import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
import timm, wandb, tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2
from datasets import load_dataset
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.ops import MLP

import lejepa


def ddp_setup():
    """Init process group from torchrun env vars; return (rank, world_size, local_rank)."""
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


class ViTEncoder(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch8_224",
            pretrained=False,
            num_classes=512,
            drop_path_rate=0.1,
            img_size=128,
        )
        self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)


class Model(nn.Module):
    """Encoder + online linear probe in one module so a single DDP wrapper covers both."""

    def __init__(self, proj_dim):
        super().__init__()
        self.encoder = ViTEncoder(proj_dim=proj_dim)
        self.probe = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 10))

    def forward(self, x):
        emb, proj = self.encoder(x)
        yhat = self.probe(emb.detach())
        return emb, proj, yhat


class HFDataset(torch.utils.data.Dataset):
    def __init__(self, split, V=1):
        self.V = V
        self.ds = load_dataset("frgfm/imagenette", "160px", split=split, trust_remote_code=True)
        self.aug = v2.Compose(
            [
                v2.RandomResizedCrop(128, scale=(0.08, 1.0)),
                v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                v2.RandomGrayscale(p=0.2),
                v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),
                v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
                v2.RandomHorizontalFlip(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.test = v2.Compose(
            [
                v2.Resize(128),
                v2.CenterCrop(128),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __getitem__(self, i):
        item = self.ds[i]
        img = item["image"].convert("RGB")
        transform = self.aug if self.V > 1 else self.test
        return torch.stack([transform(img) for _ in range(self.V)]), item["label"]

    def __len__(self):
        return len(self.ds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lamb", type=float, default=0.02)
    p.add_argument("--V", type=int, default=4)
    p.add_argument("--proj_dim", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--bs", type=int, default=256, help="GLOBAL batch size")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--wandb_project", type=str, default="LeJEPA-DDP")
    p.add_argument("--tag", type=str, default="run")
    cfg = p.parse_args()

    rank, world_size, local_rank = ddp_setup()
    device = f"cuda:{local_rank}"
    torch.manual_seed(cfg.seed)
    assert cfg.bs % world_size == 0, "global batch must be divisible by world size"
    per_rank_bs = cfg.bs // world_size

    if is_main():
        wandb.init(project=cfg.wandb_project, config=vars(cfg) | {"world_size": world_size})

    train_ds = HFDataset("train", V=cfg.V)
    test_ds = HFDataset("validation", V=1)
    train_sampler = (
        DistributedSampler(train_ds, shuffle=True, seed=cfg.seed, drop_last=True)
        if world_size > 1
        else None
    )
    train = DataLoader(
        train_ds,
        batch_size=per_rank_bs,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    test = DataLoader(test_ds, batch_size=256, num_workers=cfg.num_workers)

    model = Model(proj_dim=cfg.proj_dim).to(device)
    if world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    core = model.module if world_size > 1 else model

    # Repo's DDP-aware SIGReg (all-reduces the empirical CF over ranks).
    sigreg = lejepa.multivariate.SlicingUnivariateTest(
        univariate_test=lejepa.univariate.EppsPulley(t_max=3, n_points=17),
        num_slices=256,
    ).to(device)

    g1 = {"params": core.encoder.parameters(), "lr": cfg.lr, "weight_decay": 5e-2}
    g2 = {"params": core.probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}
    opt = torch.optim.AdamW([g1, g2])
    steps_per_epoch = len(train)
    total_steps = steps_per_epoch * cfg.epochs
    s1 = LinearLR(opt, start_factor=0.01, total_iters=steps_per_epoch)
    s2 = CosineAnnealingLR(opt, T_max=max(1, total_steps - steps_per_epoch), eta_min=1e-3)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[steps_per_epoch])

    scaler = GradScaler()
    for epoch in range(cfg.epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        it = tqdm.tqdm(train, total=steps_per_epoch) if is_main() else train
        for vs, y in it:
            vs = vs.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast("cuda", dtype=torch.bfloat16):
                emb, proj, yhat = model(vs)
                inv_loss = (proj.mean(0) - proj).square().mean()
                sigreg_loss = sigreg(proj)
                lejepa_loss = sigreg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)
                probe_loss = F.cross_entropy(yhat, y.repeat_interleave(cfg.V))
                loss = lejepa_loss + probe_loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            if is_main():
                wandb.log(
                    {
                        "train/probe": probe_loss.item(),
                        "train/lejepa": lejepa_loss.item(),
                        "train/sigreg": sigreg_loss.item(),
                        "train/inv": inv_loss.item(),
                    }
                )

        # Eval on rank 0 over the full test set (exact, no sampler padding).
        if is_main():
            model.eval()
            correct = 0
            with torch.inference_mode():
                for vs, y in test:
                    vs = vs.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    with autocast("cuda", dtype=torch.bfloat16):
                        _, _, yhat = (
                            core(vs) if world_size > 1 else model(vs)
                        )
                    correct += (yhat.argmax(1) == y).sum().item()
            acc = correct / len(test_ds)
            wandb.log({"test/acc": acc, "test/epoch": epoch})
            print(f"[{cfg.tag}][epoch {epoch}] test/acc = {acc:.4f}", flush=True)
        if world_size > 1:
            dist.barrier()

    if is_main():
        wandb.finish()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
