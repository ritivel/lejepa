"""Linear-probe TUAB using checkpoints from ``eeg_lejepa_ddp.py``.

This is intentionally separate from the SeisLM TUAB runner because the EEG
LeJEPA checkpoints save an ``EEGEncoder`` state dict, not a SeisLM/Wav2Vec2
Lightning checkpoint.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset

from eeg_lejepa_ddp import EEGEncoder


PYEDFLIB_UV_TO_V = 1e-6
DEFAULT_WINDOW_SAMPLES = 3001
TUH_TARGET_CHANNELS = (
  "FP1", "FP2",
  "F7", "F3", "FZ", "F4", "F8",
  "T7", "C3", "CZ", "C4", "T8",
  "P7", "P3", "PZ", "P4", "P8",
  "O1", "O2",
)
TUH_LEGACY_ALIASES = {
  "T3": "T7",
  "T4": "T8",
  "T5": "P7",
  "T6": "P8",
}
EGI_128_CHANNELS = tuple(f"E{i}" for i in range(1, 129))
TUH_TO_EGI_128_EQUIVALENTS = {
  "FP1": "E22",
  "FP2": "E9",
  "F7": "E33",
  "F3": "E24",
  "FZ": "E11",
  "F4": "E124",
  "F8": "E122",
  "T7": "E45",
  "C3": "E36",
  "C4": "E104",
  "T8": "E108",
  "P7": "E58",
  "P3": "E52",
  "PZ": "E62",
  "P4": "E92",
  "P8": "E96",
  "O1": "E70",
  "O2": "E83",
}
TUH_CZ_NEAREST_EGI = "E55"


def normalize_label(label: str) -> str:
  name = label.strip().upper()
  name = " ".join(name.split())
  if name.startswith("EEG "):
    name = name[4:]
  name = name.replace(" ", "")
  for suffix in ("-REF", "-LE", "-AVG", "-AR"):
    if name.endswith(suffix):
      name = name[: -len(suffix)]
  return TUH_LEGACY_ALIASES.get(name, name)


def channel_indices(labels: list[str]) -> tuple[list[int], list[str]]:
  by_name: dict[str, int] = {}
  for idx, label in enumerate(labels):
    by_name.setdefault(normalize_label(label), idx)
  missing = [channel for channel in TUH_TARGET_CHANNELS if channel not in by_name]
  if missing:
    return [], missing
  return [by_name[channel] for channel in TUH_TARGET_CHANNELS], []


def zero_fill_egi128(data_19: np.ndarray, *, include_cz: bool) -> np.ndarray:
  out = np.zeros((len(EGI_128_CHANNELS), data_19.shape[1]), dtype=np.float32)
  egi_index = {name: idx for idx, name in enumerate(EGI_128_CHANNELS)}
  tuh_index = {name: idx for idx, name in enumerate(TUH_TARGET_CHANNELS)}
  for tuh_name, egi_name in TUH_TO_EGI_128_EQUIVALENTS.items():
    out[egi_index[egi_name]] = data_19[tuh_index[tuh_name]]
  if include_cz:
    out[egi_index[TUH_CZ_NEAREST_EGI]] = data_19[tuh_index["CZ"]]
  return out


def transform_channels(data_19: np.ndarray, mode: str) -> np.ndarray:
  if mode == "tuh19":
    return data_19
  if mode == "egi128_zero":
    return zero_fill_egi128(data_19, include_cz=False)
  if mode == "egi128_zero_cz":
    return zero_fill_egi128(data_19, include_cz=True)
  raise ValueError("channel_mode must be one of: tuh19, egi128_zero, egi128_zero_cz")


def discover_tuab_records(root: Path, window_samples: int) -> pd.DataFrame:
  rows: list[dict] = []
  labels = {"normal": 0, "abnormal": 1}
  for path in sorted(root.glob("*/*/01_tcp_ar/*.edf")):
    rel = path.relative_to(root)
    split, label_name = rel.parts[0], rel.parts[1]
    if split not in {"train", "eval"} or label_name not in labels:
      continue
    try:
      with pyedflib.EdfReader(str(path)) as reader:
        indices, missing = channel_indices(reader.getSignalLabels())
        if missing:
          raise ValueError("missing_10_20_channels:" + ",".join(missing))
        n_samples = int(reader.getNSamples()[indices[0]])
      status = "ok" if n_samples >= window_samples else "dropped"
      drop_reason = None if status == "ok" else "too_short"
    except Exception as exc:  # noqa: BLE001
      n_samples = 0
      status = "dropped"
      drop_reason = f"{type(exc).__name__}:{exc}"
    rows.append({
      "path": str(path),
      "split": split,
      "label_name": label_name,
      "label": labels[label_name],
      "subject_id": path.stem.split("_")[0],
      "recording_id": path.stem,
      "n_samples": n_samples,
      "status": status,
      "drop_reason": drop_reason,
    })
  return pd.DataFrame(rows)


def load_manifest(root: Path, manifest_path: Path, refresh: bool, window_samples: int) -> pd.DataFrame:
  if manifest_path.exists() and not refresh:
    return pd.read_parquet(manifest_path)
  df = discover_tuab_records(root, window_samples)
  manifest_path.parent.mkdir(parents=True, exist_ok=True)
  df.to_parquet(manifest_path, index=False)
  return df


@dataclass
class SplitRows:
  train: pd.DataFrame
  val: pd.DataFrame
  test: pd.DataFrame


def split_rows(df: pd.DataFrame, val_frac: float, seed: int) -> SplitRows:
  df = df[df["status"].eq("ok")].copy()
  train_all = df[df["split"].eq("train")].copy()
  test = df[df["split"].eq("eval")].copy()
  subjects = np.array(sorted(train_all["subject_id"].unique()))
  rng = np.random.default_rng(seed)
  rng.shuffle(subjects)
  n_val = max(1, int(round(len(subjects) * val_frac)))
  val_subjects = set(subjects[:n_val])
  val = train_all[train_all["subject_id"].isin(val_subjects)].copy()
  train = train_all[~train_all["subject_id"].isin(val_subjects)].copy()
  return SplitRows(train=train, val=val, test=test)


class TuabWindowDataset(Dataset):
  def __init__(
    self,
    rows: pd.DataFrame,
    *,
    window_samples: int,
    encoder_samples: int,
    deterministic: bool,
    channel_mode: str,
    seed: int,
  ) -> None:
    self.rows = rows.reset_index(drop=True)
    self.window_samples = int(window_samples)
    self.encoder_samples = int(encoder_samples)
    self.deterministic = bool(deterministic)
    self.channel_mode = str(channel_mode)
    self.rng = np.random.default_rng(seed)

  def __len__(self) -> int:
    return len(self.rows)

  def __getitem__(self, index: int):
    row = self.rows.iloc[index]
    path = Path(row["path"])
    with pyedflib.EdfReader(str(path)) as reader:
      indices, missing = channel_indices(reader.getSignalLabels())
      if missing:
        raise ValueError("missing_10_20_channels:" + ",".join(missing))
      n_samples = int(reader.getNSamples()[indices[0]])
      max_start = n_samples - self.window_samples
      if max_start < 0:
        raise IndexError(f"{path} has {n_samples} samples < {self.window_samples}")
      start = max_start // 2 if self.deterministic else int(self.rng.integers(0, max_start + 1))
      data = np.stack(
        [
          reader.readSignal(idx, start=start, n=self.window_samples, digital=False).astype(np.float32, copy=False)
          for idx in indices
        ],
        axis=0,
      )
    data = data * PYEDFLIB_UV_TO_V
    data = transform_channels(data, self.channel_mode)
    data = data - data.mean(axis=1, keepdims=True)
    data = data / (data.std(axis=1, keepdims=True) + 1e-6)
    tensor = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32)).unsqueeze(0)
    if data.shape[1] != self.encoder_samples:
      tensor = torch.nn.functional.interpolate(
        tensor, size=self.encoder_samples, mode="linear", align_corners=False
      )
    return tensor.squeeze(0), torch.tensor(int(row["label"]), dtype=torch.long)


class LinearProbe(nn.Module):
  def __init__(self, encoder: EEGEncoder, hidden_size: int, num_classes: int = 2) -> None:
    super().__init__()
    self.encoder = encoder
    for param in self.encoder.parameters():
      param.requires_grad = False
    self.head = nn.Sequential(
      nn.LayerNorm(hidden_size),
      nn.Linear(hidden_size, num_classes),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
      emb, _ = self.encoder(x[:, None])
    return self.head(emb)


def balanced_acc(preds: list[int], labels: list[int]) -> float:
  return float(balanced_accuracy_score(labels, preds))


def run_epoch(model, loader, device, optimizer=None) -> tuple[float, float]:
  train = optimizer is not None
  model.train(train)
  total_loss = 0.0
  preds: list[int] = []
  labels: list[int] = []
  for x, y in loader:
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    if train:
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    total_loss += float(loss.detach().cpu()) * y.numel()
    preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
    labels.extend(y.detach().cpu().tolist())
  return total_loss / max(1, len(labels)), balanced_acc(preds, labels)


def load_encoder(ckpt_path: Path, hidden_size: int, proj_dim: int, device) -> EEGEncoder:
  ckpt = torch.load(ckpt_path, map_location="cpu")
  encoder = EEGEncoder(hidden_size=hidden_size, proj_dim=proj_dim)
  encoder.load_state_dict(ckpt["model"], strict=True)
  encoder.to(device)
  encoder.eval()
  return encoder


def evaluate_checkpoint(
  *,
  ckpt_path: Path,
  splits: SplitRows,
  args: argparse.Namespace,
  device,
) -> dict:
  encoder = load_encoder(ckpt_path, args.hidden_size, args.proj_dim, device)
  model = LinearProbe(encoder, hidden_size=args.hidden_size).to(device)
  loaders = {
    "train": DataLoader(
      TuabWindowDataset(
        splits.train,
        window_samples=args.window_samples,
        encoder_samples=args.encoder_samples,
        deterministic=False,
        channel_mode=args.channel_mode,
        seed=args.seed,
      ),
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers,
      pin_memory=True,
      drop_last=True,
    ),
    "val": DataLoader(
      TuabWindowDataset(
        splits.val,
        window_samples=args.window_samples,
        encoder_samples=args.encoder_samples,
        deterministic=True,
        channel_mode=args.channel_mode,
        seed=args.seed,
      ),
      batch_size=args.batch_size,
      shuffle=False,
      num_workers=args.num_workers,
      pin_memory=True,
    ),
    "test": DataLoader(
      TuabWindowDataset(
        splits.test,
        window_samples=args.window_samples,
        encoder_samples=args.encoder_samples,
        deterministic=True,
        channel_mode=args.channel_mode,
        seed=args.seed,
      ),
      batch_size=args.batch_size,
      shuffle=False,
      num_workers=args.num_workers,
      pin_memory=True,
    ),
  }
  optimizer = torch.optim.AdamW(
    model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay
  )
  best_val = -1.0
  best_test = -1.0
  best_epoch = -1
  for epoch in range(args.epochs):
    train_loss, train_bacc = run_epoch(model, loaders["train"], device, optimizer)
    val_loss, val_bacc = run_epoch(model, loaders["val"], device)
    test_loss, test_bacc = run_epoch(model, loaders["test"], device)
    if val_bacc > best_val:
      best_val = val_bacc
      best_test = test_bacc
      best_epoch = epoch
    if args.wandb:
      wandb.log({
        "epoch": epoch,
        "train/loss": train_loss,
        "train/balanced_acc": train_bacc,
        "val/loss": val_loss,
        "val/balanced_acc": val_bacc,
        "test/loss": test_loss,
        "test/balanced_acc": test_bacc,
      })
  return {
    "checkpoint": str(ckpt_path),
    "checkpoint_name": ckpt_path.name,
    "best_epoch": best_epoch,
    "best_val_balanced_acc": best_val,
    "test_balanced_acc_at_best_val": best_test,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--tuab-root", required=True)
  parser.add_argument("--manifest-path", required=True)
  parser.add_argument("--ckpt", action="append", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument("--channel-mode", default="egi128_zero_cz")
  parser.add_argument("--window-samples", type=int, default=3001)
  parser.add_argument("--encoder-samples", type=int, default=6000)
  parser.add_argument("--hidden-size", type=int, default=512)
  parser.add_argument("--proj-dim", type=int, default=16)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--num-workers", type=int, default=8)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--lr", type=float, default=4e-4)
  parser.add_argument("--weight-decay", type=float, default=0.1)
  parser.add_argument("--val-frac", type=float, default=0.1)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--refresh-manifest", action="store_true")
  parser.add_argument("--wandb", action="store_true")
  parser.add_argument("--wandb-project", default="eeg-lejepa-tuab")
  args = parser.parse_args()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  root = Path(args.tuab_root)
  manifest = Path(args.manifest_path)
  df = load_manifest(root, manifest, args.refresh_manifest, args.window_samples)
  splits = split_rows(df, args.val_frac, args.seed)
  results = []
  for ckpt in args.ckpt:
    ckpt_path = Path(ckpt)
    run = None
    if args.wandb:
      run = wandb.init(
        project=args.wandb_project,
        name=f"tuab_lp_{ckpt_path.stem}",
        config=vars(args) | {"checkpoint": str(ckpt_path)},
        reinit=True,
      )
    result = evaluate_checkpoint(
      ckpt_path=ckpt_path, splits=splits, args=args, device=device
    )
    results.append(result)
    if run is not None:
      wandb.summary.update(result)
      wandb.finish()

  out = Path(args.output)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(results, indent=2, sort_keys=True))
  print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
