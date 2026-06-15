"""Minimal DDP LeJEPA pretraining on cached EEG windows.

This intentionally mirrors ``MINIMAL.md`` while replacing ImageNette image
views with augmented EEG-window views. One sample is a full-channel EEG window
``(C, T)`` and the LeJEPA views are stochastic augmentations of that same window.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb
from omegaconf import DictConfig
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.ops import MLP


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


class EEGWindowDataset(Dataset):
  """Cached EEG windows with stochastic augmentation views."""

  def __init__(
    self,
    rows: pd.DataFrame,
    *,
    V: int,
    window_samples: int,
    jitter_samples: int,
    n_global_views: int,
    global_crop_samples: int,
    local_crop_samples: int,
    channel_dropout_p: float,
    time_mask_p: float,
    time_mask_frac: float,
    noise_std: float,
    amp_scale: float,
    train: bool,
  ) -> None:
    self.rows = rows.reset_index(drop=True)
    self.V = int(V)
    self.window_samples = int(window_samples)
    self.jitter_samples = int(jitter_samples)
    self.n_global_views = int(n_global_views)
    self.global_crop_samples = int(global_crop_samples)
    self.local_crop_samples = int(local_crop_samples)
    self.channel_dropout_p = float(channel_dropout_p)
    self.time_mask_p = float(time_mask_p)
    self.time_mask_frac = float(time_mask_frac)
    self.noise_std = float(noise_std)
    self.amp_scale = float(amp_scale)
    self.train = bool(train)

  def __len__(self) -> int:
    return len(self.rows)

  @staticmethod
  def _resize_time(x: np.ndarray, target_samples: int) -> np.ndarray:
    if x.shape[1] == target_samples:
      return x
    tensor = torch.from_numpy(x).unsqueeze(0)
    resized = F.interpolate(
      tensor, size=int(target_samples), mode="linear", align_corners=False
    )
    return resized.squeeze(0).numpy().astype(np.float32, copy=False)

  def _crop(self, row: pd.Series, crop_samples: int) -> np.ndarray:
    arr = cached_array(str(row["array_path"]))
    start = int(row["start_sample"])
    if self.train and self.jitter_samples > 0:
      max_start = max(0, int(arr.shape[1]) - crop_samples)
      low = max(0, start - self.jitter_samples)
      high = min(max_start, start + self.jitter_samples)
      start = int(np.random.randint(low, high + 1)) if high >= low else start
    stop = start + crop_samples
    if stop > arr.shape[1]:
      start = int(arr.shape[1]) - crop_samples
      stop = int(arr.shape[1])
    x = np.asarray(arr[:, start:stop], dtype=np.float32)
    if x.shape[1] != crop_samples:
      raise IndexError(f"bad EEG crop shape {x.shape} from {row['array_path']}")
    return x

  def _augment(self, x: np.ndarray) -> np.ndarray:
    y = x.copy()
    y = y - y.mean(axis=1, keepdims=True)
    y = y / (y.std(axis=1, keepdims=True) + 1e-6)
    if self.train:
      if self.amp_scale > 0:
        scale = np.random.uniform(1.0 - self.amp_scale, 1.0 + self.amp_scale)
        y = y * np.float32(scale)
      if self.noise_std > 0:
        y = y + np.random.normal(0.0, self.noise_std, size=y.shape).astype(np.float32)
      if self.channel_dropout_p > 0:
        keep = np.random.random(y.shape[0]) > self.channel_dropout_p
        if not keep.any():
          keep[np.random.randint(0, y.shape[0])] = True
        y[~keep] = 0.0
      if self.time_mask_p > 0 and np.random.random() < self.time_mask_p:
        width = max(1, int(self.window_samples * self.time_mask_frac))
        start = np.random.randint(0, self.window_samples - width + 1)
        y[:, start:start + width] = 0.0
    return y.astype(np.float32, copy=False)

  def __getitem__(self, index: int):
    row = self.rows.iloc[index]
    views = []
    for view_idx in range(self.V):
      crop_samples = (
        self.global_crop_samples
        if view_idx < self.n_global_views
        else self.local_crop_samples
      )
      x = self._crop(row, crop_samples)
      x = self._resize_time(x, self.window_samples)
      views.append(self._augment(x))
    return torch.from_numpy(np.stack(views))


class EEGEncoder(nn.Module):
  """Small full-window EEG encoder for LeJEPA smoke/pretraining runs."""

  def __init__(self, hidden_size: int = 512, proj_dim: int = 128):
    super().__init__()
    self.temporal = nn.Sequential(
      nn.Conv1d(1, 128, kernel_size=25, stride=10, padding=12),
      nn.GELU(),
      nn.Conv1d(128, hidden_size, kernel_size=15, stride=5, padding=7),
      nn.GELU(),
    )
    self.channel_mixer = nn.Sequential(
      nn.LayerNorm(hidden_size),
      nn.Linear(hidden_size, hidden_size),
      nn.GELU(),
    )
    self.proj = MLP(hidden_size, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

  def forward(self, x: torch.Tensor):
    num_samples, num_views, num_channels, num_time = x.shape
    y = x.reshape(num_samples * num_views * num_channels, 1, num_time)
    y = self.temporal(y).mean(dim=-1)
    y = y.reshape(num_samples * num_views, num_channels, -1).mean(dim=1)
    emb = self.channel_mixer(y)
    proj = self.proj(emb).reshape(num_samples, num_views, -1).transpose(0, 1)
    return emb, proj


def gather_with_grad(x: torch.Tensor) -> torch.Tensor:
  if not is_dist():
    return x
  gathered = dist_nn.all_gather(x)
  return torch.cat(tuple(gathered), dim=1)


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
      name=f"eeg_aug_v{cfg.V}_bs{cfg.bs}_g{world_size}",
      config=dict(cfg) | {"world_size": world_size, "global_batch_size": cfg.bs},
    )

  windows = pd.read_parquet(str(cfg.windows_path))
  windows = windows[windows["dataset_id"].eq(str(cfg.dataset_id))].copy()
  train_mask = subject_splits(windows, int(getattr(cfg, "seed", 0)))
  train_df = windows[train_mask].reset_index(drop=True)
  val_df = windows[~train_mask].reset_index(drop=True)

  if cfg.bs % world_size != 0:
    raise ValueError(f"global batch size {cfg.bs} not divisible by {world_size}")
  local_batch = cfg.bs // world_size
  num_workers = int(getattr(cfg, "num_workers", 4))
  common = dict(
    V=cfg.V,
    window_samples=cfg.window_samples,
    jitter_samples=cfg.jitter_samples,
    n_global_views=cfg.n_global_views,
    global_crop_samples=cfg.global_crop_samples,
    local_crop_samples=cfg.local_crop_samples,
    channel_dropout_p=cfg.channel_dropout_p,
    time_mask_p=cfg.time_mask_p,
    time_mask_frac=cfg.time_mask_frac,
    noise_std=cfg.noise_std,
    amp_scale=cfg.amp_scale,
  )
  train_ds = EEGWindowDataset(train_df, train=True, **common)
  val_ds = EEGWindowDataset(val_df, train=False, **common)
  train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if is_dist() else None
  val_sampler = DistributedSampler(val_ds, shuffle=False) if is_dist() else None
  train_loader = DataLoader(
    train_ds,
    batch_size=local_batch,
    sampler=train_sampler,
    shuffle=train_sampler is None,
    drop_last=True,
    num_workers=num_workers,
    pin_memory=True,
  )
  val_loader = DataLoader(
    val_ds,
    batch_size=max(1, cfg.bs // world_size),
    sampler=val_sampler,
    num_workers=num_workers,
    pin_memory=True,
  )

  model = nn.SyncBatchNorm.convert_sync_batchnorm(
    EEGEncoder(hidden_size=cfg.hidden_size, proj_dim=cfg.proj_dim)
  ).to(device)
  model = DDP(model, device_ids=[local_rank])
  sigreg = SIGReg().to(device)
  optimizer = torch.optim.AdamW(
    model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
  )
  warmup_steps = len(train_loader)
  total_steps = len(train_loader) * cfg.epochs
  scheduler = SequentialLR(
    optimizer,
    schedulers=[
      LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps),
      CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6),
    ],
    milestones=[warmup_steps],
  )
  scaler = GradScaler(enabled=True)

  for epoch in range(cfg.epochs):
    if train_sampler is not None:
      train_sampler.set_epoch(epoch)
    model.train()
    iterator = tqdm.tqdm(train_loader, disable=not is_rank0())
    for views in iterator:
      views = views.to(device, non_blocking=True)
      with autocast("cuda", dtype=torch.bfloat16):
        _, proj = model(views)
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

    model.eval()
    val_loss = torch.tensor(0.0, device=device)
    val_count = torch.tensor(0, device=device, dtype=torch.long)
    with torch.inference_mode():
      for views in val_loader:
        views = views.to(device, non_blocking=True)
        with autocast("cuda", dtype=torch.bfloat16):
          _, proj = model(views)
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

  if is_rank0():
    wandb.finish()
  if is_dist():
    dist.destroy_process_group()


if __name__ == "__main__":
  main()
