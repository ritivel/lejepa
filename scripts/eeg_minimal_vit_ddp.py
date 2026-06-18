"""Minimal LeJEPA ImageNette run adapted to EEG-as-1-channel images.

The goal is to keep the original minimal LeJEPA experiment as intact as possible:

* ViT-S/8 backbone
* 128 x 128 transformed views
* V augmented views
* SIGReg + invariance loss
* AdamW + linear warmup + cosine schedule

The necessary EEG-specific adaptation is the dataset: a cached EEG window
``(128 channels, 6000 samples)`` is converted to a 1-channel image
``(1, 128, 6000)`` and then passed through the image-style view transform.
"""

from __future__ import annotations

import os
from functools import lru_cache

import hydra
import numpy as np
import pandas as pd
import timm
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn as nn
import tqdm
import wandb
from omegaconf import DictConfig
from pathlib import Path
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.ops import MLP
from torchvision.transforms import v2


def is_dist() -> bool:
  return dist.is_available() and dist.is_initialized()


def is_rank0() -> bool:
  return not is_dist() or dist.get_rank() == 0


@lru_cache(maxsize=256)
def cached_array(path: str) -> np.ndarray:
  return np.load(path, mmap_mode="r")


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
    A = torch.randn(proj.size(-1), 256, device=proj.device)
    A = A.div_(A.norm(p=2, dim=0))
    if is_dist():
      dist.broadcast(A, src=0)
    x_t = (proj @ A).unsqueeze(-1) * self.t
    err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
    statistic = (err @ self.weights) * proj.size(-2)
    return statistic.mean()


class ViTEncoder(nn.Module):
  def __init__(self, proj_dim=128):
    super().__init__()
    self.backbone = timm.create_model(
      "vit_small_patch8_224",
      pretrained=False,
      num_classes=512,
      drop_path_rate=0.1,
      img_size=128,
      in_chans=1,
    )
    self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

  def forward(self, x):
    num_samples, num_views = x.shape[:2]
    emb = self.backbone(x.flatten(0, 1))
    proj = self.proj(emb).reshape(num_samples, num_views, -1).transpose(0, 1)
    return emb, proj


def gather_with_grad(x: torch.Tensor) -> torch.Tensor:
  if not is_dist():
    return x
  gathered = dist_nn.all_gather(x)
  return torch.cat(tuple(gathered), dim=1)


class EEGImageDataset(Dataset):
  def __init__(self, rows: pd.DataFrame, *, V: int, train: bool):
    self.rows = rows.reset_index(drop=True)
    self.V = int(V)
    self.train = bool(train)
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
        v2.Normalize(mean=[0.5], std=[0.5]),
      ]
    )
    self.test = v2.Compose(
      [
        v2.Resize(128),
        v2.CenterCrop(128),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5], std=[0.5]),
      ]
    )

  def __len__(self):
    return len(self.rows)

  @staticmethod
  def _to_uint8_image(x: np.ndarray) -> torch.Tensor:
    x = x.astype(np.float32, copy=False)
    x = x - x.mean(axis=1, keepdims=True)
    x = x / (x.std(axis=1, keepdims=True) + 1e-6)
    x = np.clip(x, -3.0, 3.0)
    x = ((x + 3.0) / 6.0 * 255.0).astype(np.uint8)
    return torch.from_numpy(x).unsqueeze(0)

  def __getitem__(self, index):
    row = self.rows.iloc[index]
    arr = cached_array(str(row["array_path"]))
    start = int(row["start_sample"])
    stop = int(row["stop_sample"])
    x = np.asarray(arr[:, start:stop], dtype=np.float32)
    img = self._to_uint8_image(x)
    transform = self.aug if self.train else self.test
    return torch.stack([transform(img) for _ in range(self.V)])


def subject_splits(df: pd.DataFrame, seed: int) -> pd.Series:
  rng = np.random.default_rng(seed)
  subjects = np.asarray(sorted(set(df["subject_id_in_dataset"].astype(str))), dtype=object)
  rng.shuffle(subjects)
  n_train = max(1, int(0.9 * len(subjects)))
  train_subjects = set(subjects[:n_train])
  return df["subject_id_in_dataset"].astype(str).isin(train_subjects)


@hydra.main(version_base=None)
def main(cfg: DictConfig):
  local_rank = int(os.environ.get("LOCAL_RANK", 0))
  world_size = int(os.environ.get("WORLD_SIZE", 1))
  if world_size > 1:
    dist.init_process_group("nccl")
  torch.cuda.set_device(local_rank)
  device = torch.device("cuda", local_rank)
  torch.manual_seed(int(getattr(cfg, "seed", 0)))
  np.random.seed(int(getattr(cfg, "seed", 0)) + local_rank)

  if is_rank0():
    wandb.init(
      project="eeg-lejepa-basic",
      name=f"eeg_minimal_vit_1ch_v{cfg.V}_bs{cfg.bs}_g{world_size}",
      config=dict(cfg) | {"world_size": world_size, "global_batch_size": cfg.bs},
    )
  checkpoint_dir = Path(str(getattr(
    cfg, "checkpoint_dir", "/home/ubuntu/lejepa-runs/eeg-minimal-vit/checkpoints"
  )))
  save_every_epochs = int(getattr(cfg, "save_every_epochs", 1))
  if is_rank0():
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

  windows = pd.read_parquet(str(cfg.windows_path))
  windows = windows[windows["dataset_id"].eq(str(cfg.dataset_id))].copy()
  train_mask = subject_splits(windows, int(getattr(cfg, "seed", 0)))
  train_df = windows[train_mask].reset_index(drop=True)
  val_df = windows[~train_mask].reset_index(drop=True)

  if cfg.bs % world_size != 0:
    raise ValueError(f"global batch size {cfg.bs} not divisible by {world_size}")
  local_batch = cfg.bs // world_size
  num_workers = int(getattr(cfg, "num_workers", 4))

  train_ds = EEGImageDataset(train_df, V=cfg.V, train=True)
  val_ds = EEGImageDataset(val_df, V=1, train=False)
  train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if is_dist() else None
  val_sampler = DistributedSampler(val_ds, shuffle=False) if is_dist() else None
  train = DataLoader(
    train_ds,
    batch_size=local_batch,
    shuffle=train_sampler is None,
    sampler=train_sampler,
    drop_last=True,
    num_workers=num_workers,
    pin_memory=True,
  )
  val = DataLoader(
    val_ds,
    batch_size=max(1, 256 // world_size),
    sampler=val_sampler,
    num_workers=num_workers,
    pin_memory=True,
  )

  net_module = nn.SyncBatchNorm.convert_sync_batchnorm(
    ViTEncoder(proj_dim=cfg.proj_dim)
  ).to(device)
  net = DDP(net_module, device_ids=[local_rank])
  sigreg = SIGReg().to(device)

  optimizer = torch.optim.AdamW(
    net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
  )
  warmup_steps = len(train)
  total_steps = len(train) * cfg.epochs
  scheduler = SequentialLR(
    optimizer,
    schedulers=[
      LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps),
      CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-3),
    ],
    milestones=[warmup_steps],
  )

  scaler = GradScaler(enabled=True)
  for epoch in range(cfg.epochs):
    if train_sampler is not None:
      train_sampler.set_epoch(epoch)
    net.train()
    for views in tqdm.tqdm(train, total=len(train), disable=not is_rank0()):
      with autocast("cuda", dtype=torch.bfloat16):
        views = views.to(device, non_blocking=True)
        _, proj = net(views)
        global_proj = gather_with_grad(proj)
        inv_loss = (global_proj.mean(0) - global_proj).square().mean()
        sigreg_loss = sigreg(global_proj)
        lejepa_loss = sigreg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)
        loss = lejepa_loss * world_size

      optimizer.zero_grad()
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
      scheduler.step()
      if is_rank0():
        wandb.log(
          {
            "train/lejepa": lejepa_loss.item(),
            "train/sigreg": sigreg_loss.item(),
            "train/inv": inv_loss.item(),
            "train/epoch": epoch,
          }
        )

    net.eval()
    val_loss = torch.tensor(0.0, device=device)
    val_count = torch.tensor(0, device=device, dtype=torch.long)
    with torch.inference_mode():
      for views in val:
        views = views.to(device, non_blocking=True)
        with autocast("cuda", dtype=torch.bfloat16):
          _, proj = net(views)
          global_proj = gather_with_grad(proj)
          inv_loss = (global_proj.mean(0) - global_proj).square().mean()
          sigreg_loss = sigreg(global_proj)
          lejepa_loss = sigreg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)
        val_loss += lejepa_loss.detach()
        val_count += 1
    if is_dist():
      dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
      dist.all_reduce(val_count, op=dist.ReduceOp.SUM)
    if is_rank0() and val_count.item() > 0:
      wandb.log({"val/lejepa": (val_loss / val_count).item(), "val/epoch": epoch})
      if save_every_epochs > 0 and (
        (epoch + 1) % save_every_epochs == 0 or (epoch + 1) == int(cfg.epochs)
      ):
        state = {
          "epoch": epoch,
          "model": net.module.state_dict(),
          "optimizer": optimizer.state_dict(),
          "scheduler": scheduler.state_dict(),
          "scaler": scaler.state_dict(),
          "config": dict(cfg),
        }
        torch.save(state, checkpoint_dir / f"epoch={epoch:04d}.pt")
        torch.save(state, checkpoint_dir / "last.pt")

  if is_rank0():
    wandb.finish()
  if is_dist():
    dist.destroy_process_group()


if __name__ == "__main__":
  main()
