"""Does a wrong recall point toward another item of the same trial?

For every target of a multi-item validation trial (ForwardRecall for the FIFO
run, BackwardRecall for the LIFO runs), the signed error is the direction of
the mean network output during the hold (skipping the first 80 ms, as in
loss_and_recall_accuracy.py) minus the target's direction. Pull is the signed
error taken positive toward the nearest other item of the trial. Prints the
fraction of targets whose error points toward that item and the mean pull,
overall and split by the distance from the target to that item. Run from
anywhere:
    python examples/stack_task_analysis/error_direction.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "content" / "runs" / "task-trained"
MODELS = ROOT / "content" / "trained_models" / "task-trained"
# (network label, run tag, run name, checkpoint in the run's lightning_logs)
NETWORKS = [
    (
        "FIFO epoch 200",
        "20261007_StackTask_test",
        "max_epochs=200_batch_size=64_seed=0",
        None,
    ),
    (
        "LIFO epoch 200",
        "20261021_StackTaskLIFO",
        "max_epochs=200_batch_size=64_seed=0",
        None,
    ),
    (
        "LIFO epoch 195",
        "20261021_StackTaskLIFO",
        "max_epochs=200_batch_size=64_seed=0",
        "epoch=194-step=9750.ckpt",
    ),
    (
        "LIFO epoch 384",
        "20261027_StackTaskLIFO_400ep",
        "max_epochs=400_batch_size=64_seed=0",
        "epoch=383-step=19200.ckpt",
    ),
]
SKIP_BINS, TOL_DEG = 4, 15
DIST_BINS = [30, 60, 90, 180]


def segments(tgt):
    """Hold periods as (start, stop) bins: runs of Valid=1 with a constant target."""
    valid = tgt[:, 2] > 0.5
    change = np.r_[
        True,
        np.any(np.abs(np.diff(tgt[:, :2], axis=0)) > 1e-9, axis=1)
        | (valid[1:] != valid[:-1]),
    ]
    starts = np.flatnonzero(change)
    stops = np.r_[starts[1:], len(tgt)]
    return [(a, b) for a, b in zip(starts, stops) if valid[a]]


def wrap_deg(d):
    return (d + 180) % 360 - 180


def signed_errors(label, run_tag, run_name, ckpt):
    with open(MODELS / run_tag / run_name / "model.pkl", "rb") as f:
        wrapper = pickle.load(f)
    with open(MODELS / run_tag / run_name / "datamodule_train.pkl", "rb") as f:
        dm = pickle.load(f)
    if ckpt is not None:
        state = torch.load(
            RUNS / run_tag / "lightning_logs" / ckpt,
            map_location="cpu",
            weights_only=False,
        )
        wrapper.load_state_dict(state["state_dict"])
    wrapper = wrapper.to("cpu").eval()
    ics, inputs, targets, _, conds, extra, _, _ = [
        t.cpu() for t in dm.valid_ds.tensors
    ]
    torch.manual_seed(0)  # the model's initial hidden state is noisy
    with torch.no_grad():
        out = wrapper.forward(ics, inputs)["controlled"].numpy()
    targets = targets.numpy()
    end = extra[:, 1].numpy().astype(int)
    rows = []
    for i in range(len(out)):
        segs = segments(targets[i, : end[i]])
        if len(segs) < 2:
            continue
        angles = [np.rad2deg(np.arctan2(*targets[i, a, 1::-1])) for a, _ in segs]
        for k, (a, b) in enumerate(segs):
            mean_xy = out[i, a + SKIP_BINS : b, :2].mean(axis=0)
            signed = wrap_deg(np.rad2deg(np.arctan2(mean_xy[1], mean_xy[0])) - angles[k])
            to_others = [wrap_deg(angles[j] - angles[k]) for j in range(len(segs)) if j != k]
            nearest = min(to_others, key=abs)
            rows.append(
                dict(
                    network=label,
                    n_items=len(segs),
                    item=k + 1,
                    signed=signed,
                    dist=abs(nearest),
                    pull=signed * np.sign(nearest),
                )
            )
    return pd.DataFrame(rows)


def summary(d):
    return pd.Series(
        dict(
            toward=(d.pull > 0).mean(),
            mean_pull=d.pull.mean(),
            abs_err=d.signed.abs().mean(),
            n=len(d),
        )
    )


df = pd.concat([signed_errors(*net) for net in NETWORKS], ignore_index=True)
df["dist_bin"] = pd.cut(df.dist, DIST_BINS)
order = [net[0] for net in NETWORKS]
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:.3f}")
for name, d in [("all multi-item targets", df), (f"targets > {TOL_DEG} deg", df[df.signed.abs() > TOL_DEG])]:
    print(f"\n===== {name}: overall =====")
    print(d.groupby("network").apply(summary).reindex(order))
    print(f"\n===== {name}: by distance to the nearest other item (deg) =====")
    print(d.groupby(["network", "dist_bin"], observed=True).apply(summary).reindex(order, level=0))
