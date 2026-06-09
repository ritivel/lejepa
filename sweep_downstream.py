"""Sweep linear-probe downstream eval across pretraining checkpoints.

Pretraining only keeps the latest `ddp_last.pt` (overwritten each epoch), so this
driver snapshots each new epoch's checkpoint as it appears, runs a linear probe on
it, records the best-val-AUROC test metrics, and finally plots test balanced
accuracy vs pretraining epoch.
"""
import os, time, json, shutil, subprocess

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "eeg_checkpoints/ddp_last.pt")
EPOCH7 = os.path.join(HERE, "eeg_checkpoints/probe_snapshot.pt")
SWEEP = os.path.join(HERE, "eeg_checkpoints/sweep")
RESULTS = os.path.join(SWEEP, "results.jsonl")
PROBE_EPOCHS = 25
os.makedirs(SWEEP, exist_ok=True)


def pretrain_alive():
    return subprocess.run(
        "pgrep -f eeg_minimal_ddp.py", shell=True, capture_output=True
    ).returncode == 0


def safe_ckpt_epoch(path):
    """Return epoch int, or None if file missing / mid-write / unreadable."""
    if not os.path.exists(path):
        return None
    try:
        return int(torch.load(path, map_location="cpu")["epoch"])
    except Exception:
        return None


def verified_copy(src, dst, tries=5):
    for _ in range(tries):
        shutil.copy(src, dst)
        try:
            torch.load(dst, map_location="cpu")
            return True
        except Exception:
            time.sleep(5)
    return False


def probe(src_ckpt, epoch):
    snap = os.path.join(SWEEP, f"ep{epoch}.pt")
    if not verified_copy(src_ckpt, snap):
        print(f"skip epoch {epoch}: could not get a clean copy", flush=True)
        return False
    out = os.path.join(SWEEP, f"ep{epoch}.json")
    cmd = (
        f"cd {HERE} && . .venv/bin/activate && WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 "
        f"python -u eeg_downstream.py --ckpt {snap} --epochs {PROBE_EPOCHS} "
        f"--num_workers 8 --tag sweep_ep{epoch} --out_json {out}"
    )
    subprocess.run(["bash", "-lc", cmd], check=True)
    rec = json.load(open(out))
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"PROBED epoch {epoch}: {rec}", flush=True)
    return True


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = sorted((json.loads(l) for l in open(RESULTS)), key=lambda r: r["pretrain_epoch"])
    ep = [r["pretrain_epoch"] for r in recs]
    bal = [r["test_bal_acc"] for r in recs]
    auroc = [r["test_auroc"] for r in recs]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ep, bal, "o-", label="test balanced accuracy", color="tab:blue")
    ax.plot(ep, auroc, "s--", label="test AUROC", color="tab:orange", alpha=0.7)
    ax.axhline(0.5, ls=":", color="gray", label="chance")
    ax.set_xlabel("pretraining epoch (checkpoint)")
    ax.set_ylabel("TUAB downstream metric (best-val-AUROC ckpt)")
    ax.set_title("LeJEPA-EEG: TUAB linear-probe vs pretraining progress")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(HERE, "downstream_sweep.png")
    fig.savefig(out_png, dpi=120)
    print(f"saved {out_png}", flush=True)


def main():
    probed = set()
    if os.path.exists(RESULTS):
        for line in open(RESULTS):
            probed.add(json.loads(line)["pretrain_epoch"])

    if 7 not in probed and os.path.exists(EPOCH7):
        if probe(EPOCH7, 7):
            probed.add(7)

    while True:
        ep = safe_ckpt_epoch(CKPT)
        if ep is not None and ep not in probed:
            if probe(CKPT, ep):
                probed.add(ep)
        if not pretrain_alive():
            ep = safe_ckpt_epoch(CKPT)
            if ep is not None and ep not in probed:
                if probe(CKPT, ep):
                    probed.add(ep)
            break
        time.sleep(30)

    plot()
    print("SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
