# LeJEPA EEG Runs

This file records EEG-specific runs built on top of the minimal LeJEPA DDP
replication work.

## Branch and Code Version

- Repository: `ritivel/lejepa`
- Branch: `eeg-pretraining`
- EEG code commit: `c52794f`
- Parent DDP replication commit: `49831e5`

The `c52794f` commit adds:

- `scripts/eeg_lejepa_ddp.py`
- a DDP EEG pretraining loop modeled after `scripts/minimal_inet10_ddp.py`
- PEERS trainer-cache loading from `windows.parquet`
- full-window EEG view construction
- global/local temporal crop view construction
- EEG augmentations for LeJEPA views
- W&B logging to `eeg-lejepa-basic`

## Code Changes Up To This Version

### Reused From DDP Replication

The EEG script keeps the DDP mechanics from the corrected minimal ImageNette DDP
replication:

- `torch.distributed.run`
- `DistributedSampler`
- `SyncBatchNorm`
- differentiable `all_gather` of projected embeddings before `inv` and `SIGReg`
- shared SIGReg random projection matrix broadcast from rank 0
- `lejepa_loss * world_size` before backward to cancel DDP gradient averaging
- distributed validation with all-reduced validation loss

### EEG Data Loader

The EEG script reads the existing PEERS cache:

```text
/home/ubuntu/eeg-data-proc/eegData/preprocessed_resampled_200hz_peers/windows.parquet
```

Each cache row points to a `.npy` recording array:

```text
(128 channels, n_samples)
```

Each training sample is a 30-second EEG window:

```text
(128, 6000)
```

because the cache is sampled at 200 Hz.

### EEG Encoder

The first EEG encoder is intentionally small and utilitarian:

- shared temporal `Conv1d` over each channel
- mean pooling over time
- mean pooling over channels
- small channel mixer MLP
- LeJEPA projector MLP

This is not intended as the final EEG architecture. It is a minimal model for
testing whether the view construction gives a non-trivial `inv` loss curve.

## Runs

| # | Date UTC | Commit | W&B Project / Run | View construction | Key command settings | Status | Notes |
|---|---|---|---|---|---|---|---|
| EEG-001 | 2026-06-15 | `c52794f` | `eeg-lejepa-basic` / `https://wandb.ai/ritivel/eeg-lejepa-basic/runs/wkksypd2` | 4 augmented full-window views of the same `(128, 6000)` EEG window | `V=4`, `bs=256`, `epochs=200`, `jitter_samples=400`, `channel_dropout_p=0.1`, `time_mask_p=0.5`, `time_mask_frac=0.08`, `noise_std=0.02`, `amp_scale=0.2`, 8xA100 | STOPPED | `inv` saturated early. Views were still too similar because every view saw nearly the full 30-second all-channel window. |
| EEG-002 | 2026-06-15 | `01fa8d6` | `eeg-lejepa-basic` / `https://wandb.ai/ritivel/eeg-lejepa-basic/runs/3mml1jou` | 2 global 30-second views + 2 local 8-second temporal-crop views resized to 30 seconds | `V=4`, `n_global_views=2`, `global_crop_samples=6000`, `local_crop_samples=1600`, `bs=256`, `epochs=200`, `jitter_samples=1200`, `channel_dropout_p=0.2`, `time_mask_p=0.7`, `time_mask_frac=0.15`, `noise_std=0.03`, `amp_scale=0.3`, 8xA100 | STOPPED | First improved global/local temporal-view run. Designed to make the invariance task harder than EEG-001. |
| EEG-003 | 2026-06-15 | `01fa8d6` | `eeg-lejepa-basic` / `https://wandb.ai/ritivel/eeg-lejepa-basic/runs/nj2tldip` | Harder 2 global 30-second views + 2 local 4-second temporal-crop views resized to 30 seconds | `V=4`, `n_global_views=2`, `global_crop_samples=6000`, `local_crop_samples=800`, `bs=256`, `epochs=200`, `jitter_samples=2000`, `channel_dropout_p=0.4`, `time_mask_p=0.8`, `time_mask_frac=0.2`, `noise_std=0.03`, `amp_scale=0.3`, 8xA100 | STOPPED | Harder-view run. Improved training dynamics relative to EEG-001/002, but downstream still peaked early; checkpoints moved to `checkpoints/harder_global_local_v3/`. |
| EEG-004 | 2026-06-16 | `0c20352` | `eeg-lejepa-basic` / `https://wandb.ai/ritivel/eeg-lejepa-basic/runs/bmstrjyl` | Same harder views as EEG-003, but replace mean-over-channels with attention channel pooling | `V=4`, `n_global_views=2`, `global_crop_samples=6000`, `local_crop_samples=800`, `channel_pool=attention`, `bs=256`, `epochs=200`, `jitter_samples=2000`, `channel_dropout_p=0.4`, `time_mask_p=0.8`, `time_mask_frac=0.2`, `noise_std=0.03`, `amp_scale=0.3`, 8xA100 | STOPPED | Did not perform better than EEG-003. Stopped after epoch 6; checkpoints moved to `checkpoints/attention_channel_pool_v4/`. |
| EEG-005 | 2026-06-16 | `356811a` | `eeg-lejepa-basic` / run logged from `eeg_patch_transformer_20260616_160644_8gpu.log` | Same harder views as EEG-003, but replace the Conv1D channel-pooling encoder with an EEG channel-time patch transformer | `encoder_type=patch_transformer`, `patch_channels=8`, `patch_time=400`, `transformer_depth=6`, `transformer_heads=8`, `V=4`, `n_global_views=2`, `global_crop_samples=6000`, `local_crop_samples=800`, `bs=256`, `epochs=200`, harder-view augmentations, 8xA100 | STOPPED / EVAL COMPLETE | Exp4 architecture ablation. Checkpoints saved under `checkpoints/patch_transformer_v4/`; 5-seed TUAB LP completed for epochs 0-24. |
| EEG-006 | 2026-06-17 | `132c443` | `eeg-lejepa-basic` / `https://wandb.ai/ritivel/eeg-lejepa-basic/runs/mhnvy8n3` | Exp5 EEG-as-image minimal ViT-S/8: PEERS `(128, 6000)` windows converted to one-channel images and transformed to `(1, 128, 128)` views with the ImageNette-style augmentation stack | `scripts/eeg_minimal_vit_ddp.py`, `vit_small_patch8_224`, `in_chans=1`, `img_size=128`, `V=4`, `proj_dim=16`, `lamb=0.02`, `lr=2e-3`, `bs=256`, `epochs=200`, 4xH100 | STOPPED / EVAL COMPLETE | Relaunched with checkpoint saving after the first no-checkpoint run. Checkpoints saved under `checkpoints/exp5_minimal_vit_1ch_ckpt/`; 5-seed TUAB LP completed for epochs 0-77. Best 5-seed mean TUAB balanced accuracy was `65.08%` at checkpoint epoch 5, substantially below earlier EEG runs. |

## View Construction Details

### EEG-001: Augmented Full-Window Views

For each PEERS window:

```text
base window: (128, 6000)
views:       4 x augmented (128, 6000)
```

Augmentations:

- per-channel z-score
- temporal jitter of up to 400 samples
- amplitude scaling
- Gaussian noise
- channel dropout
- time masking

This run still saturated early, suggesting the four views were too similar.

### EEG-002: Global/Local Temporal Views

For each PEERS window:

```text
base window: (128, 6000)
global views: 2 x crop 6000 samples
local views:  2 x crop 1600 samples, linearly resized to 6000 samples
```

The local views do not recover missing temporal detail. They stretch an 8-second
crop back to 30 seconds only so the same encoder can be reused.

This run also increases augmentation strength:

- `jitter_samples`: `400 -> 1200`
- `channel_dropout_p`: `0.1 -> 0.2`
- `time_mask_p`: `0.5 -> 0.7`
- `time_mask_frac`: `0.08 -> 0.15`
- `noise_std`: `0.02 -> 0.03`
- `amp_scale`: `0.2 -> 0.3`

## Current Interpretation

The original neighborhood/channel-view EEG LeJEPA runs saturated because the
views were too easy: simultaneous EEG channels or mild full-window augmentations
remain highly correlated.

The current EEG direction is to keep the LeJEPA/DDP machinery fixed and make the
views more like the ImageNette minimal run:

- different corruptions of one underlying sample
- explicit global/local view asymmetry
- harder temporal changes before changing architecture

The main metric to watch first is `train/inv`. If it saturates immediately, the
views are still too similar. If it decreases gradually over many epochs, the view
construction is more promising.

## Planned Architecture Ablation: Attention Channel Pooling

EEG-004 keeps the EEG-003 view construction fixed and changes only the channel
aggregation in the encoder:

```text
mean over channels -> learned attention pooling over channels
```

This tests whether the previous mean-over-channels encoder was washing out
spatial/channel-specific information needed for downstream TUAB performance.

## Exp4 Architecture Ablation: EEG Patch Transformer

EEG-005 keeps the EEG-003/EEG-004 harder view construction fixed and changes
only the encoder:

```text
Conv1D per-channel encoder + channel pooling
-> Conv2D channel-time patch embed + transformer encoder + token mean pool
```

Patch layout:

```text
input:          (128 channels, 6000 samples)
patch size:     (8 channels, 400 samples)
token grid:     16 x 15
tokens/view:    240
```

This is meant to be closer to the ImageNette minimal LeJEPA architecture:

```text
Image: local image patches -> transformer
EEG:   local channel-time patches -> transformer
```

## Preserved Artifacts In S3

Before shutting down the 8xA100 training/eval VM, the important run artifacts were
synced to:

```text
s3://nirmit-dev-vm-storages/lejepa-runs/eeg-basic/
```

### Checkpoints

| Run | Local checkpoint dir on 8xA100 | S3 prefix | Objects | Size |
|---|---|---|---:|---:|
| EEG-002 / `global_local_v2` | `/home/ubuntu/lejepa-runs/eeg-basic/checkpoints/global_local_v2` | `s3://nirmit-dev-vm-storages/lejepa-runs/eeg-basic/checkpoints/global_local_v2/` | 12 | ~0.94 GB |
| EEG-003 / `harder_global_local_v3` | `/home/ubuntu/lejepa-runs/eeg-basic/checkpoints/harder_global_local_v3` | `s3://nirmit-dev-vm-storages/lejepa-runs/eeg-basic/checkpoints/harder_global_local_v3/` | 21 | ~1.65 GB |
| EEG-005 / `patch_transformer_v4` | `/home/ubuntu/lejepa-runs/eeg-basic/checkpoints/patch_transformer_v4` | `s3://nirmit-dev-vm-storages/lejepa-runs/eeg-basic/checkpoints/patch_transformer_v4/` | 26 | ~8.10 GB |

Checkpoint coverage:

```text
global_local_v2:        epoch=0000.pt through epoch=0010.pt, plus last.pt
harder_global_local_v3: epoch=0000.pt through epoch=0019.pt, plus last.pt
patch_transformer_v4:   epoch=0000.pt through epoch=0024.pt, plus last.pt
```

### Training Logs

Training logs from `/home/ubuntu/lejepa-runs/eeg-basic/logs/` were synced to:

```text
s3://nirmit-dev-vm-storages/lejepa-runs/eeg-basic/logs/
```

At preservation time this prefix had 6 log files, including:

```text
eeg_global_local_ckpt_20260615_123209_8gpu.log
eeg_harder_views_20260615_183302_8gpu.log
eeg_patch_transformer_20260616_160644_8gpu.log
```

### Remote Downstream Eval Outputs

The remote TUAB downstream eval output directories from the 8xA100 VM were
synced to:

```text
s3://nirmit-dev-vm-storages/lejepa-runs/eeg-basic/tuab_lp_harder_global_local_v3/
s3://nirmit-dev-vm-storages/lejepa-runs/eeg-basic/tuab_lp_patch_transformer_v4_multiseed/
```

The `patch_transformer_v4` downstream prefix includes raw chunk outputs for:

```text
seeds: 101, 202, 303, 404, 505
epochs: 0000 through 0024
```

The raw files are chunked by seed and checkpoint range, e.g.
`patch_transformer_v4_seed101_chunk0.json`.

### Local Analysis Artifacts

The local result summaries and plots were also copied back to S3:

```text
s3://nirmit-dev-vm-storages/lejepa-runs/eeg-basic/analysis_artifacts/
```

This prefix includes:

```text
TUAB-downstream*.md
tuab_downstream*.json
tuab_downstream*.csv
tuab_downstream*.png
tuab_lp_multiseed_results/
tuab_lp_patch_transformer_v4_multiseed_results/
```

Important analysis files:

```text
tuab_downstream_multiseed_3exp_comparison.png
tuab_downstream_multiseed_summary.json
tuab_downstream_multiseed_summary.csv
tuab_downstream_multiseed_summary_patch_transformer_v4.json
tuab_downstream_multiseed_summary_patch_transformer_v4.csv
```

The previous two multiseed downstream evals (`global_local_v2` and
`harder_global_local_v3`) are preserved under:

```text
s3://nirmit-dev-vm-storages/lejepa-runs/eeg-basic/analysis_artifacts/tuab_lp_multiseed_results/
```

and cover:

```text
seeds: 101, 202, 303, 404, 505
epochs: 0000 through 0008
```

The `patch_transformer_v4` multiseed downstream eval chunks are preserved under:

```text
s3://nirmit-dev-vm-storages/lejepa-runs/eeg-basic/analysis_artifacts/tuab_lp_patch_transformer_v4_multiseed_results/
```

## Single-H100 Profiling

Date: 2026-06-17

VM: `68.209.74.204`, `NVIDIA H100 80GB HBM3`

Workspace:

```text
/home/ubuntu/lejepa-profile-h100/
```

Profiler:

```text
scripts/profile_eeg_runs.py
```

Method:

- PEERS cache synced from `s3://nirmit-dev-vm-storages/eeg-trainer-cache/resampled_200hz_peers/run-20260528T191359Z/final`.
- Short optimizer-loop profile only: no W&B logging, no checkpointing, bf16 autocast, AdamW, SIGReg + invariance loss.
- Each point uses warmup steps followed by measured steps and reports optimizer-step wall time with `torch.cuda.synchronize`.
- `V=4`; samples/sec counts original EEG windows, views/sec counts augmented views.

Results:

| Run | Batch | Step sec | Samples/sec | Views/sec | Max GPU GB |
|---|---:|---:|---:|---:|---:|
| `eeg_harder_conv_mean` | 32 | 0.0654 | 489.6 | 1958.2 | 13.9 |
| `eeg_harder_conv_mean` | 64 | 0.1330 | 481.1 | 1924.4 | 26.3 |
| `eeg_harder_conv_mean` | 128 | 0.2680 | 477.6 | 1910.5 | 45.2 |
| `eeg_harder_conv_mean` | 192 | 0.4019 | 477.8 | 1911.1 | 65.3 |
| `eeg_minimal_vit_1ch` | 32 | 0.0414 | 773.3 | 3093.2 | 5.8 |
| `eeg_minimal_vit_1ch` | 64 | 0.0733 | 873.5 | 3493.8 | 11.1 |
| `eeg_minimal_vit_1ch` | 128 | 0.1373 | 932.2 | 3729.0 | 21.7 |
| `eeg_minimal_vit_1ch` | 256 | 0.2645 | 967.8 | 3871.3 | 42.8 |
| `eeg_minimal_vit_1ch` | 384 | 0.3950 | 972.2 | 3888.9 | 63.9 |

Notes:

- `eeg_harder_conv_mean` OOMed at `batch=256` on one H100. This is expected because the 8-GPU training run's `bs=256` is a global DDP batch, i.e. local batch `32` per GPU.
- The Conv EEG path saturates around `~480 samples/sec` on one H100; increasing batch mostly increases memory without improving throughput.
- The EEG-as-image minimal ViT path reaches `~970 samples/sec` by `batch=256-384`, about `2x` the Conv EEG path in this single-GPU profile.
