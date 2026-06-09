"""Downstream eval of a LeJEPA-EEG checkpoint on TUH Abnormal EEG (TUAB).

Mirrors the seisLM downstream pattern (load pretrained backbone -> task head ->
supervised train -> report test metrics), adapted to the per-channel 1-D ViT:
each EEG channel is encoded independently and the per-channel embeddings are
mean-pooled into a window representation, then a small head classifies
normal vs abnormal.

Linear probe (default, encoder frozen):
  python eeg_downstream.py --ckpt eeg_checkpoints/ddp_last.pt --epochs 30
Finetune (unfreeze encoder):
  python eeg_downstream.py --ckpt eeg_checkpoints/ddp_last.pt --finetune --lr 1e-4
"""
import os, argparse

import numpy as np
import pandas as pd
import torch, torch.nn as nn
import pyedflib
import scipy.signal
import tqdm
import wandb
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from eeg_minimal_ddp import ViT1DEncoder

DEFAULT_TUAB = "/home/ubuntu/eeg-data-proc/downstream_data/tuh_eeg_abnormal/v3.0.1"
PYEDFLIB_UV_TO_V = 1e-6
LABELS = {"normal": 0, "abnormal": 1}

# 10-20 montage used by TUAB (from seisLM tuab_dataloaders.py).
TUH_TARGET_CHANNELS = (
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
    "T7", "C3", "CZ", "C4", "T8", "P7", "P3", "PZ", "P4", "P8", "O1", "O2",
)
TUH_LEGACY_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def _normalize_label(label: str) -> str:
    name = " ".join(label.strip().upper().split())
    if name.startswith("EEG "):
        name = name[4:]
    name = name.replace(" ", "")
    for suffix in ("-REF", "-LE", "-AVG", "-AR"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return TUH_LEGACY_ALIASES.get(name, name)


def _channel_indices(labels):
    by_name = {}
    for idx, label in enumerate(labels):
        by_name.setdefault(_normalize_label(label), idx)
    missing = [c for c in TUH_TARGET_CHANNELS if c not in by_name]
    if missing:
        return [], missing
    return [by_name[c] for c in TUH_TARGET_CHANNELS], []


class TuabDataset(Dataset):
    """One resampled 30 s window per recording: [n_channels=19, window_len]."""

    def __init__(self, rows, window_len=6000, target_sr=200, win_seconds=30,
                 deterministic=False, seed=42):
        self.rows = rows.reset_index(drop=True)
        self.window_len = window_len
        self.win_seconds = win_seconds
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        path = row["path"]
        with pyedflib.EdfReader(path) as r:
            indices, missing = _channel_indices(r.getSignalLabels())
            if missing:
                raise ValueError("missing_10_20_channels:" + ",".join(missing))
            sr = float(r.getSampleFrequency(indices[0]))
            native_win = int(round(self.win_seconds * sr))
            n = int(r.getNSamples()[indices[0]])
            max_start = n - native_win
            if max_start < 0:
                raise IndexError(f"{path}: {n} < {native_win}")
            start = max_start // 2 if self.deterministic else int(self.rng.integers(0, max_start + 1))
            data = np.stack(
                [r.readSignal(idx, start=start, n=native_win).astype(np.float32)
                 for idx in indices],
                axis=0,
            )
        data = data * PYEDFLIB_UV_TO_V
        # Resample native 30 s window -> window_len samples (== target_sr Hz).
        data = scipy.signal.resample(data, self.window_len, axis=1).astype(np.float32)
        mu = data.mean(axis=1, keepdims=True)
        sd = data.std(axis=1, keepdims=True)
        data = (data - mu) / (sd + 1e-8)
        return (
            torch.from_numpy(np.ascontiguousarray(data)).unsqueeze(1),  # [C, 1, T]
            torch.tensor(int(row["label"]), dtype=torch.long),
        )


def subject_split(train_df, val_frac, seed):
    subjects = np.array(sorted(train_df["subject_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n_val = max(1, int(round(len(subjects) * val_frac)))
    val_subj = set(subjects[:n_val])
    val = train_df[train_df["subject_id"].isin(val_subj)].copy()
    train = train_df[~train_df["subject_id"].isin(val_subj)].copy()
    return train, val


class DownstreamClassifier(nn.Module):
    def __init__(self, encoder, emb_dim=512, num_classes=2, freeze=True):
        super().__init__()
        self.encoder = encoder
        self.freeze = freeze
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, num_classes))

    def forward(self, x):
        B, C, _, T = x.shape  # [B, C, 1, T]
        flat = x.reshape(B * C, 1, T)
        if self.freeze:
            with torch.no_grad():
                feats = self.encoder.encode(flat)
        else:
            feats = self.encoder.encode(flat)
        feats = feats.reshape(B, C, -1).mean(1)  # mean-pool over channels
        return self.head(feats)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    ys, probs, preds = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        p = torch.softmax(logits.float(), dim=1)[:, 1]
        probs.append(p.cpu()); preds.append(logits.argmax(1).cpu()); ys.append(y)
    y = torch.cat(ys).numpy(); prob = torch.cat(probs).numpy(); pred = torch.cat(preds).numpy()
    return {
        "acc": float((pred == y).mean()),
        "bal_acc": float(balanced_accuracy_score(y, pred)),
        "auroc": float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="eeg_checkpoints/ddp_last.pt")
    p.add_argument("--tuab_root", default=DEFAULT_TUAB)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--finetune", action="store_true", help="unfreeze encoder")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_project", default="LeJEPA-EEG-downstream")
    p.add_argument("--tag", default="probe")
    p.add_argument("--out_json", default=None, help="write best-val test metrics here")
    cfg = p.parse_args()

    torch.manual_seed(cfg.seed)
    device = "cuda"
    wandb.init(project=cfg.wandb_project, name=cfg.tag, config=vars(cfg))

    # Load pretrained encoder.
    ck = torch.load(cfg.ckpt, map_location="cpu")
    ecfg = ck["config"]
    encoder = ViT1DEncoder(
        proj_dim=ecfg["proj_dim"], window_len=ecfg["window_len"], patch=ecfg["patch"],
        dim=ecfg["dim"], depth=ecfg["depth"], heads=ecfg["heads"],
    )
    encoder.load_state_dict(ck["model"])
    window_len = ecfg["window_len"]
    print(f"loaded encoder from {cfg.ckpt} (pretrain epoch {ck['epoch']}), "
          f"mode={'finetune' if cfg.finetune else 'linear-probe'}", flush=True)

    # Data.
    m = pd.read_parquet(os.path.join(cfg.tuab_root, "tuab_manifest.parquet"))
    m = m[m["status"] == "ok"].copy()
    train_all = m[m["split"] == "train"].copy()
    test_df = m[m["split"] == "eval"].copy()
    train_df, val_df = subject_split(train_all, cfg.val_frac, cfg.seed)
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}", flush=True)
    dl_kw = dict(num_workers=cfg.num_workers, pin_memory=True,
                 persistent_workers=cfg.num_workers > 0)
    train_loader = DataLoader(
        TuabDataset(train_df, window_len, seed=cfg.seed), batch_size=cfg.bs,
        shuffle=True, drop_last=True, **dl_kw)
    val_loader = DataLoader(
        TuabDataset(val_df, window_len, deterministic=True), batch_size=cfg.bs, **dl_kw)
    test_loader = DataLoader(
        TuabDataset(test_df, window_len, deterministic=True), batch_size=cfg.bs, **dl_kw)

    model = DownstreamClassifier(encoder, num_classes=2, freeze=not cfg.finetune).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = len(train_loader)
    total = steps * cfg.epochs
    warmup = max(1, min(steps, total - 1))
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup)
    s2 = CosineAnnealingLR(opt, T_max=max(1, total - warmup), eta_min=cfg.lr / 1000)
    sched = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup])
    loss_fn = nn.CrossEntropyLoss()

    best_val_auroc, best_test = -1.0, None
    for epoch in range(cfg.epochs):
        model.train()
        if not cfg.finetune:
            model.encoder.eval()
        for x, y in tqdm.tqdm(train_loader, total=steps, desc=f"ep{epoch}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            wandb.log({"train/loss": loss.item(), "train/lr": sched.get_last_lr()[0]})
        val = evaluate(model, val_loader, device)
        test = evaluate(model, test_loader, device)
        wandb.log({f"val/{k}": v for k, v in val.items()} |
                  {f"test/{k}": v for k, v in test.items()} | {"epoch": epoch})
        print(f"[{cfg.tag}][epoch {epoch}] "
              f"val: acc={val['acc']:.4f} bal={val['bal_acc']:.4f} auroc={val['auroc']:.4f} | "
              f"test: acc={test['acc']:.4f} bal={test['bal_acc']:.4f} auroc={test['auroc']:.4f}",
              flush=True)
        if val["auroc"] > best_val_auroc:
            best_val_auroc = val["auroc"]
            best_test = test
    print(f"\n[{cfg.tag}] BEST (by val AUROC) -> test: {best_test}", flush=True)
    wandb.log({f"best_test/{k}": v for k, v in best_test.items()})
    wandb.finish()
    if cfg.out_json is not None:
        import json
        with open(cfg.out_json, "w") as f:
            json.dump({"pretrain_epoch": int(ck["epoch"]), "best_val_auroc": best_val_auroc,
                       **{f"test_{k}": v for k, v in best_test.items()}}, f)


if __name__ == "__main__":
    main()
