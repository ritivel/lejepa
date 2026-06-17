# Exp4 Proposal: EEG Patch-Transformer Encoder

Date: 2026-06-16

## Motivation

The first EEG LeJEPA runs improved view construction but still showed early
saturation and only limited downstream gains. The remaining major suspect is the
encoder architecture.

The current EEG encoder is intentionally minimal:

```text
per-channel temporal Conv1D
mean over time
mean over channels or attention over channels
projector MLP
```

This is useful for quick testing, but it is not close to the original LeJEPA
minimal ImageNette architecture, which uses a ViT-style patch transformer.

Exp4 should test whether moving the EEG encoder closer to the original
ImageNette LeJEPA encoder improves training dynamics and downstream utility.

## What Stays Fixed

Keep the following unchanged from the current stronger EEG setup:

- PEERS 200 Hz trainer cache.
- DDP mechanics from the corrected minimal replication:
  - `torch.distributed.run`
  - `DistributedSampler`
  - `SyncBatchNorm`
  - differentiable global `all_gather` before `inv` and `SIGReg`
  - shared SIGReg random projection matrix broadcast from rank 0
  - `lejepa_loss * world_size` before backward
  - distributed validation all-reduce
- LeJEPA objective:

```text
loss = lambda * SIGReg + (1 - lambda) * inv
```

- `lambda = 0.02`
- `V = 4`
- 2 global views and 2 local views.
- Current harder global/local view construction:

```text
global_crop_samples = 6000   # 30 seconds
local_crop_samples  = 800    # 4 seconds
jitter_samples      = 2000   # 10 seconds
channel_dropout_p   = 0.4
time_mask_p         = 0.8
time_mask_frac      = 0.2
noise_std           = 0.03
amp_scale           = 0.3
```

This keeps attribution clean: exp4 changes only the encoder.

## Original ImageNette Minimal Tokenization

Original minimal LeJEPA uses an image ViT:

```text
image input: 3 x 128 x 128
patch size: 8 x 8
patch grid: 16 x 16
tokens/view: 256
```

Each image patch contains:

```text
3 channels x 8 x 8 = 192 raw values
```

The important design pattern is:

```text
local patch tokens -> transformer global mixing -> one representation
```

## EEG Tokenization Goal

Current EEG input:

```text
128 channels x 6000 samples
```

At 200 Hz:

```text
6000 samples = 30 seconds
```

For an EEG patch transformer, patch over:

```text
channel axis x time axis
```

Token count:

```text
tokens = (128 / patch_channels) * (6000 / patch_time)
```

## Patch Size Comparison

| Modality / Option | Input | Patch size | Token grid | Tokens/view |
|---|---:|---:|---:|---:|
| ImageNette minimal | `128 x 128` | `8 x 8` | `16 x 16` | `256` |
| EEG A | `128 x 6000` | `8 ch x 200 samples` | `16 x 30` | `480` |
| EEG B | `128 x 6000` | `8 ch x 400 samples` | `16 x 15` | `240` |
| EEG C | `128 x 6000` | `16 ch x 200 samples` | `8 x 30` | `240` |
| EEG D | `128 x 6000` | `16 ch x 400 samples` | `8 x 15` | `120` |

## Recommended Exp4 Patch Size

Use:

```text
patch_channels = 8
patch_time     = 400 samples
tokens/view    = 16 x 15 = 240
```

Why:

- It is very close to the ImageNette minimal token count of 256.
- It preserves more channel/spatial resolution than `16 ch x 200 samples`.
- A 2-second time patch is still reasonable for EEG rhythms and avoids the cost
  of 480 tokens/view.
- It should be memory-safe on 8xA100 with current batch size.

## Proposed Encoder

Replace the current Conv1D + channel pooling encoder with:

```text
EEG window: (128, 6000)
-> Conv2D patch embedding
-> 240 patch tokens
-> transformer encoder
-> mean token pooling
-> projector MLP
```

Concrete shape flow:

```text
input view: (128, 6000)
add pseudo image channel: (1, 128, 6000)
Conv2d patch embed kernel=(8, 400), stride=(8, 400)
output: (embed_dim, 16, 15)
flatten: (240, embed_dim)
transformer: (240, embed_dim)
mean pool tokens: (embed_dim)
projector: (proj_dim)
```

Recommended initial hyperparameters:

```text
embed_dim = 512
depth     = 6
heads     = 8
mlp_ratio = 4
pool      = mean over tokens
proj_dim  = 16
```

## Why This Is The Right Next Ablation

It moves the EEG model much closer to the original LeJEPA architecture:

```text
Image: local image patches -> ViT
EEG:   local channel-time patches -> transformer
```

It also addresses the leading remaining hypothesis:

> The current encoder is over-smoothing EEG by reducing each channel over time
> and then pooling over channels too early.

With patch tokens, the model can learn interactions such as:

```text
temporal pattern in frontal channels + temporal pattern in central channels
```

instead of collapsing the full window into a simple channel average.

## Expected Outcomes

If architecture is the bottleneck:

- `train/inv` should remain non-trivial longer.
- downstream TUAB performance may improve beyond epoch 1.
- later checkpoints may degrade less.

If saturation remains:

- view construction is still too easy, or
- the PEERS-to-TUAB transfer signal is weak, or
- we need frequency-domain / semantic EEG augmentations.

## Minimal Risk Controls

For exp4, do not change:

- data
- view construction
- optimizer
- loss
- DDP code
- batch size
- checkpoint/eval protocol

Only change:

```text
encoder = Conv1D channel-pooling encoder -> EEG patch transformer
```

This gives a clean architecture ablation.
