"""Profile the two current EEG LeJEPA training loops on one GPU.

This script intentionally does not log to W&B or save checkpoints. It reuses the
dataset/model/loss pieces from the training scripts and measures a short
optimizer loop after warmup so we can compare throughput on identical hardware.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader


def _load_module(path: Path, name: str):
  import importlib.util

  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise ImportError(f"Could not load module from {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _train_rows(windows_path: Path, dataset_id: str, seed: int, subject_splits):
  windows = pd.read_parquet(str(windows_path))
  windows = windows[windows["dataset_id"].eq(dataset_id)].copy()
  train_mask = subject_splits(windows, seed)
  return windows[train_mask].reset_index(drop=True)


def _filter_existing_arrays(rows: pd.DataFrame) -> pd.DataFrame:
  if "array_path" not in rows.columns:
    return rows
  exists = rows["array_path"].map(lambda value: Path(str(value)).exists())
  return rows[exists].reset_index(drop=True)


def _time_steps(
  *,
  name: str,
  model: nn.Module,
  sigreg: nn.Module,
  loader: DataLoader,
  device: torch.device,
  lamb: float,
  lr: float,
  weight_decay: float,
  warmup_steps: int,
  measure_steps: int,
):
  model.train()
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
  scaler = GradScaler(enabled=True)
  iterator = iter(loader)
  torch.cuda.reset_peak_memory_stats(device)
  step_times: list[float] = []
  measured_samples = 0
  measured_views = 0
  batch_size = 0
  views_per_sample = 0

  for step_idx in range(warmup_steps + measure_steps):
    try:
      views = next(iterator)
    except StopIteration:
      iterator = iter(loader)
      views = next(iterator)

    views = views.to(device, non_blocking=True)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with autocast("cuda", dtype=torch.bfloat16):
      _, proj = model(views)
      inv_loss = (proj.mean(0) - proj).square().mean()
      sigreg_loss = sigreg(proj)
      loss = sigreg_loss * lamb + inv_loss * (1 - lamb)

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    if step_idx >= warmup_steps:
      step_times.append(elapsed)
      batch_size = int(views.shape[0])
      views_per_sample = int(views.shape[1])
      measured_samples += int(views.shape[0])
      measured_views += int(views.shape[0] * views.shape[1])

  total_time = float(sum(step_times))
  mean_step = total_time / max(1, len(step_times))
  return {
    "run": name,
    "steps": len(step_times),
    "batch_size": batch_size,
    "views_per_sample": views_per_sample,
    "mean_step_seconds": mean_step,
    "samples_per_second": measured_samples / total_time,
    "views_per_second": measured_views / total_time,
    "max_cuda_memory_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
  }


def profile_conv(args, train_df: pd.DataFrame, device: torch.device):
  mod = _load_module(Path(args.repo_root) / "scripts" / "eeg_lejepa_ddp.py", "eeg_lejepa_ddp")
  dataset = mod.EEGWindowDataset(
    train_df,
    V=args.v,
    window_samples=args.window_samples,
    jitter_samples=2000,
    n_global_views=2,
    global_crop_samples=6000,
    local_crop_samples=800,
    channel_dropout_p=0.4,
    time_mask_p=0.8,
    time_mask_frac=0.2,
    noise_std=0.03,
    amp_scale=0.3,
    train=True,
  )
  loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=args.num_workers,
    pin_memory=True,
  )
  model = nn.SyncBatchNorm.convert_sync_batchnorm(
    mod.EEGEncoder(hidden_size=512, proj_dim=128, channel_pool="mean")
  ).to(device)
  sigreg = mod.SIGReg().to(device)
  return _time_steps(
    name="eeg_harder_conv_mean",
    model=model,
    sigreg=sigreg,
    loader=loader,
    device=device,
    lamb=0.02,
    lr=2e-3,
    weight_decay=5e-2,
    warmup_steps=args.warmup_steps,
    measure_steps=args.measure_steps,
  )


def profile_minimal_vit(args, train_df: pd.DataFrame, device: torch.device):
  mod = _load_module(
    Path(args.repo_root) / "scripts" / "eeg_minimal_vit_ddp.py", "eeg_minimal_vit_ddp"
  )
  dataset = mod.EEGImageDataset(train_df, V=args.v, train=True)
  loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=args.num_workers,
    pin_memory=True,
  )
  model = nn.SyncBatchNorm.convert_sync_batchnorm(mod.ViTEncoder(proj_dim=16)).to(device)
  sigreg = mod.SIGReg().to(device)
  return _time_steps(
    name="eeg_minimal_vit_1ch",
    model=model,
    sigreg=sigreg,
    loader=loader,
    device=device,
    lamb=0.02,
    lr=2e-3,
    weight_decay=5e-2,
    warmup_steps=args.warmup_steps,
    measure_steps=args.measure_steps,
  )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
  parser.add_argument("--windows-path", required=True)
  parser.add_argument("--dataset-id", default="peers_memory")
  parser.add_argument("--batch-size", type=int, default=256)
  parser.add_argument("--v", type=int, default=4)
  parser.add_argument("--window-samples", type=int, default=6000)
  parser.add_argument("--num-workers", type=int, default=8)
  parser.add_argument("--warmup-steps", type=int, default=5)
  parser.add_argument("--measure-steps", type=int, default=25)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--require-existing-arrays", action="store_true")
  parser.add_argument(
    "--runs",
    nargs="+",
    default=["conv", "minimal_vit"],
    choices=["conv", "minimal_vit"],
  )
  parser.add_argument("--output", default="")
  args = parser.parse_args()

  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  torch.backends.cudnn.benchmark = True
  device = torch.device("cuda", 0)
  torch.cuda.set_device(device)

  conv_mod = _load_module(Path(args.repo_root) / "scripts" / "eeg_lejepa_ddp.py", "splits")
  train_df = _train_rows(
    Path(args.windows_path), args.dataset_id, args.seed, conv_mod.subject_splits
  )
  if args.require_existing_arrays:
    train_df = _filter_existing_arrays(train_df)
  if train_df.empty:
    raise RuntimeError("No training rows available for profiling")

  results = []
  if "conv" in args.runs:
    results.append(profile_conv(args, train_df, device))
  if "minimal_vit" in args.runs:
    results.append(profile_minimal_vit(args, train_df, device))

  payload = {"device": torch.cuda.get_device_name(device), "results": results}
  text = json.dumps(payload, indent=2, sort_keys=True)
  print(text)
  if args.output:
    Path(args.output).write_text(text + "\n")


if __name__ == "__main__":
  main()
