"""DDP version of the MINIMAL.md ImageNette LeJEPA example.

The model, augmentations, losses, optimizer, and hyperparameter names mirror the
minimal example. DDP is only used to split one experiment across multiple GPUs.
"""

from __future__ import annotations

import os

import hydra
import timm
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb
from datasets import load_dataset
from omegaconf import DictConfig
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.ops import MLP
from torchvision.transforms import v2


def is_dist() -> bool:
  return dist.is_available() and dist.is_initialized()


def is_rank0() -> bool:
  return not is_dist() or dist.get_rank() == 0


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
    )
    self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

  def forward(self, x):
    num_samples, num_views = x.shape[:2]
    emb = self.backbone(x.flatten(0, 1))
    proj = self.proj(emb).reshape(num_samples, num_views, -1).transpose(0, 1)
    return emb, proj


def gather_with_grad(x: torch.Tensor) -> torch.Tensor:
  """Differentiable all-gather along the sample dimension."""
  if not is_dist():
    return x
  gathered = dist_nn.all_gather(x)
  return torch.cat(tuple(gathered), dim=1)


class HFDataset(torch.utils.data.Dataset):
  def __init__(self, split, V=1):
    self.V = V
    self.ds = load_dataset("frgfm/imagenette", "160px", split=split)
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

  def __getitem__(self, index):
    item = self.ds[index]
    img = item["image"].convert("RGB")
    transform = self.aug if self.V > 1 else self.test
    return torch.stack([transform(img) for _ in range(self.V)]), item["label"]

  def __len__(self):
    return len(self.ds)


@hydra.main(version_base=None)
def main(cfg: DictConfig):
  local_rank = int(os.environ.get("LOCAL_RANK", 0))
  world_size = int(os.environ.get("WORLD_SIZE", 1))
  if world_size > 1:
    dist.init_process_group("nccl")
  torch.cuda.set_device(local_rank)
  device = torch.device("cuda", local_rank)
  torch.manual_seed(0)

  if is_rank0():
    wandb.init(
      project="LeJEPA",
      name=(
        f"minimal_inet10_ddp_globalstats_lossfix_v{cfg.V}_bs{cfg.bs}"
        f"_g{world_size}"
      ),
      config=dict(cfg) | {"world_size": world_size, "global_batch_size": cfg.bs},
    )

  if is_rank0():
    # Populate HuggingFace cache once to avoid all ranks racing on first run.
    load_dataset("frgfm/imagenette", "160px", split="train")
    load_dataset("frgfm/imagenette", "160px", split="validation")
  if is_dist():
    dist.barrier()

  if cfg.bs % world_size != 0:
    raise ValueError(f"global batch size {cfg.bs} not divisible by {world_size}")
  local_batch = cfg.bs // world_size
  num_workers = int(getattr(cfg, "num_workers", 4))

  train_ds = HFDataset("train", V=cfg.V)
  test_ds = HFDataset("validation", V=1)
  train_sampler = (
    DistributedSampler(train_ds, shuffle=True, drop_last=True)
    if is_dist()
    else None
  )
  train = DataLoader(
    train_ds,
    batch_size=local_batch,
    shuffle=train_sampler is None,
    sampler=train_sampler,
    drop_last=True,
    num_workers=num_workers,
    pin_memory=True,
  )
  test = DataLoader(
    test_ds,
    batch_size=max(1, 256 // world_size),
    sampler=DistributedSampler(test_ds, shuffle=False) if is_dist() else None,
    num_workers=num_workers,
    pin_memory=True,
  )

  net_module = nn.SyncBatchNorm.convert_sync_batchnorm(
    ViTEncoder(proj_dim=cfg.proj_dim)
  ).to(device)
  net = DDP(net_module, device_ids=[local_rank])
  probe = DDP(
    nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 10)).to(device),
    device_ids=[local_rank],
  )
  sigreg = SIGReg().to(device)

  g1 = {"params": net.parameters(), "lr": cfg.lr, "weight_decay": 5e-2}
  g2 = {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}
  opt = torch.optim.AdamW([g1, g2])
  warmup_steps = len(train)
  total_steps = len(train) * cfg.epochs
  s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
  s2 = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-3)
  scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

  scaler = GradScaler(enabled=True)
  for epoch in range(cfg.epochs):
    if train_sampler is not None:
      train_sampler.set_epoch(epoch)
    net.train(), probe.train()
    iterator = tqdm.tqdm(train, total=len(train), disable=not is_rank0())
    for vs, y in iterator:
      with autocast("cuda", dtype=torch.bfloat16):
        vs = vs.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        emb, proj = net(vs)
        global_proj = gather_with_grad(proj)
        inv_loss = (global_proj.mean(0) - global_proj).square().mean()
        sigreg_loss = sigreg(global_proj)
        lejepa_loss = sigreg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)
        y_rep = y.repeat_interleave(cfg.V)
        yhat = probe(emb.detach())
        probe_loss = F.cross_entropy(yhat, y_rep)
        # DDP averages parameter gradients across ranks. ``probe_loss`` is a
        # local mean, so that averaging gives the correct global-batch gradient.
        # ``lejepa_loss`` is already computed on the gathered global batch on
        # every rank, so multiply by world_size to cancel DDP's gradient average
        # and match the single-GPU global-batch update.
        loss = (lejepa_loss * world_size) + probe_loss

      opt.zero_grad()
      scaler.scale(loss).backward()
      scaler.step(opt)
      scaler.update()
      scheduler.step()
      if is_rank0():
        wandb.log(
          {
            "train/probe": probe_loss.item(),
            "train/lejepa": lejepa_loss.item(),
            "train/sigreg": sigreg_loss.item(),
            "train/inv": inv_loss.item(),
            "train/epoch": epoch,
          }
        )

    net.eval(), probe.eval()
    correct = torch.tensor(0, device=device, dtype=torch.long)
    total = torch.tensor(0, device=device, dtype=torch.long)
    with torch.inference_mode():
      for vs, y in test:
        vs = vs.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast("cuda", dtype=torch.bfloat16):
          pred = probe(net(vs)[0]).argmax(1)
        correct += (pred == y).sum()
        total += y.numel()
    if is_dist():
      dist.all_reduce(correct, op=dist.ReduceOp.SUM)
      dist.all_reduce(total, op=dist.ReduceOp.SUM)
    if is_rank0():
      wandb.log({"test/acc": (correct / total).item(), "test/epoch": epoch})

  if is_rank0():
    wandb.finish()
  if is_dist():
    dist.destroy_process_group()


if __name__ == "__main__":
  main()
