# DDP Run Fixes for Minimal LeJEPA Replication

Date: 2026-06-15

This note records what happened while trying to make the `MINIMAL.md` ImageNette LeJEPA example run as one DDP job across multiple GPUs while matching the single-GPU minimal run.

## Goal

Run the minimal ImageNette LeJEPA experiment as a single distributed experiment across GPUs `0,5,6,7` on the 8xA100 VM, while matching the behavior of the single-GPU reference run.

Reference single-GPU run:

- VM: `68.209.75.159`
- Script: `/home/ubuntu/lejepa-runs/minimal_inet10/mnist.py`
- Command: `+lamb=0.02 +V=4 +proj_dim=16 +lr=2e-3 +bs=256 +epochs=800`
- W&B project: `LeJEPA`
- W&B run: `https://wandb.ai/ritivel/LeJEPA/runs/9qmbhy4y`

## Initial DDP Problem

The first DDP wrapper did not match the single-GPU curves.

Root causes:

1. `SIGReg` was computed independently on each rank's local mini-batch, not on the global batch.
2. The projector uses `BatchNorm1d`, and the first DDP wrapper used per-rank BatchNorm statistics.
3. Even after gathering global projected embeddings, DDP averages parameter gradients across ranks. A loss already computed on the global batch must account for that gradient averaging.
4. The first DDP validation path ran full validation only on rank 0 while other ranks waited at a barrier, causing an NCCL timeout.

## Fixes Added

File:

- `lejepa/scripts/minimal_inet10_ddp.py`

Changes:

1. Convert the encoder/projector to synchronized BatchNorm:

```python
net_module = nn.SyncBatchNorm.convert_sync_batchnorm(
    ViTEncoder(proj_dim=cfg.proj_dim)
).to(device)
net = DDP(net_module, device_ids=[local_rank])
```

2. Gather projected embeddings across ranks before LeJEPA loss:

```python
def gather_with_grad(x: torch.Tensor) -> torch.Tensor:
    if not is_dist():
        return x
    gathered = dist_nn.all_gather(x)
    return torch.cat(tuple(gathered), dim=1)
```

Then:

```python
global_proj = gather_with_grad(proj)
inv_loss = (global_proj.mean(0) - global_proj).square().mean()
sigreg_loss = sigreg(global_proj)
```

3. Broadcast the SIGReg random projection matrix from rank 0:

```python
A = torch.randn(proj.size(-1), 256, device=proj.device)
A = A.div_(A.norm(p=2, dim=0))
if is_dist():
    dist.broadcast(A, src=0)
```

4. Scale the global LeJEPA loss by `world_size` before backward:

```python
loss = (lejepa_loss * world_size) + probe_loss
```

Reason: DDP averages gradients across ranks. `probe_loss` is a local mean, so DDP averaging gives the right global mean gradient. `lejepa_loss` is already computed on the gathered global batch on every rank, so multiplying by `world_size` cancels DDP's gradient averaging and matches single-GPU global-batch gradient scale.

5. Run validation on all ranks and all-reduce `correct` and `total`:

```python
correct = torch.tensor(0, device=device, dtype=torch.long)
total = torch.tensor(0, device=device, dtype=torch.long)
...
if is_dist():
    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
if is_rank0():
    wandb.log({"test/acc": (correct / total).item(), "test/epoch": epoch})
```

This avoids the previous NCCL timeout caused by rank 0 doing full validation while other ranks waited.

## DDP Run Attempts

### Incorrect DDP Run

- W&B run: `https://wandb.ai/ritivel/LeJEPA/runs/pq8gmgds`
- Problem: per-rank `SIGReg`/BatchNorm statistics, not global-batch-equivalent.
- Status: stopped.

### Global Stats DDP Run

- W&B run: `https://wandb.ai/ritivel/LeJEPA/runs/3vconaf4`
- Fixes included: global projected embedding gather, SyncBatchNorm, shared SIGReg projection matrix.
- Problem: rank 0 validation caused NCCL timeout.
- Status: failed/stopped.

### Current Loss-Fixed + Distributed Validation DDP Run

- W&B run: `https://wandb.ai/ritivel/LeJEPA/runs/oaz37b1n`
- Run name: `minimal_inet10_ddp_globalstats_lossfix_v4_bs256_g4`
- GPUs: `0,5,6,7`
- Command:

```bash
CUDA_VISIBLE_DEVICES=0,5,6,7 WANDB_MODE=online \
  /home/ubuntu/venv_minimal_ddp/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  scripts/minimal_inet10_ddp.py \
  +lamb=0.02 +V=4 +proj_dim=16 +lr=2e-3 +bs=256 +epochs=800 +num_workers=4
```

Status at last check:

- Active.
- Each DDP process uses about 13 GB VRAM.
- All four target GPUs are active.

## Remaining Caveats

Even with the fixes above, exact curve identity with a single-GPU run is not guaranteed because:

- Data shuffling order differs under `DistributedSampler`.
- Augmentation RNG streams differ across ranks.
- Floating point all-reduce order and BatchNorm synchronization can introduce small numerical differences.
- The online probe is trained under DDP and may not exactly match the single-process optimizer trajectory.

But the corrected DDP run is now much closer to the intended single-GPU global-batch objective than the initial DDP wrapper.
