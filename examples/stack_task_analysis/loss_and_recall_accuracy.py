"""Loss curves and recall accuracy of a trained StackTask model.

Plots train/valid loss vs epoch and network output vs target for one trial of
each type, and prints the angular error of the output during each hold period.
"item" in the tables is the output position: the order in which targets are
released, which for BackwardRecall is the reverse of the order they were shown.
Per-target errors are also saved as recall_errors<label>.csv in the run folder.

Run from anywhere. With no options it analyses the FIFO run:
    python examples/stack_task_analysis/loss_and_recall_accuracy.py
LIFO run, final weights, then the epoch-195 checkpoint:
    python examples/stack_task_analysis/loss_and_recall_accuracy.py \\
        --run-tag 20261021_StackTaskLIFO --tasks SingleRecall BackwardRecall
    python examples/stack_task_analysis/loss_and_recall_accuracy.py \\
        --run-tag 20261021_StackTaskLIFO --tasks SingleRecall BackwardRecall \\
        --checkpoint lightning_logs/epoch=194-step=9750.ckpt
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--run-tag", default="20261007_StackTask_test")
parser.add_argument("--run-name", default="max_epochs=200_batch_size=64_seed=0")
parser.add_argument(
    "--tasks",
    nargs=2,
    default=["SingleRecall", "ForwardRecall"],
    metavar=("SINGLE", "MULTI"),
    help="the one-item and the multi-item task names, as in the dataset",
)
parser.add_argument(
    "--checkpoint",
    help="Lightning checkpoint whose weights replace those in model.pkl; "
    "relative paths are taken from the run folder",
)
parser.add_argument(
    "--label", help="suffix for the output files (default: from the checkpoint)"
)
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[2]
RUN_TAG = args.run_tag
RUN = ROOT / "content" / "runs" / "task-trained" / RUN_TAG
MODEL = ROOT / "content" / "trained_models" / "task-trained" / RUN_TAG / args.run_name
SINGLE_TASK, MULTI_TASK = args.tasks
CKPT = None if args.checkpoint is None else RUN / args.checkpoint
LABEL = args.label if args.label is not None else (f"_{CKPT.stem}" if CKPT else "")
BIN_MS, SKIP_BINS, TOL_DEG = 20, 4, 15  # 80 ms = 4 bins
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
LABELS = ["TargetX", "TargetY", "TargetValid"]

# ---------- 1. loss curves ----------
m = pd.read_csv(RUN / "lightning_logs" / "metrics.csv")
tr = m.dropna(subset=["train/loss"])
va = m.dropna(subset=["valid/loss"])
print(
    "train rows",
    len(tr),
    "valid rows",
    len(va),
    "| first/last valid",
    va["valid/loss"].iloc[0],
    va["valid/loss"].iloc[-1],
    "| last train",
    tr["train/loss"].iloc[-1],
)
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(
    tr["epoch"] + 1,
    tr["train/loss"],
    color=COLORS[0],
    lw=1.5,
    label="Train (logged every 100 steps)",
)
ax.plot(
    va["epoch"] + 1,
    va["valid/loss"],
    color=COLORS[1],
    lw=1.5,
    ls="--",
    label="Validation",
)
ax.set_yscale("log")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (log scale)")
ax.set_title(f"{RUN_TAG}: loss over {va['epoch'].max() + 1} epochs")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e6e5e0", lw=0.6)
fig.tight_layout()
fig.savefig(RUN / "planner_loss_curve.png", dpi=150)
plt.close(fig)

# ---------- model + validation data ----------
with open(MODEL / "model.pkl", "rb") as f:
    wrapper = pickle.load(f)
with open(MODEL / "datamodule_train.pkl", "rb") as f:
    dm = pickle.load(f)
if CKPT is not None:
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    wrapper.load_state_dict(ckpt["state_dict"])
    print(f"weights from {CKPT.name} (epoch {ckpt['epoch'] + 1})")
else:
    print("weights from model.pkl (end of training)")
wrapper = wrapper.to("cpu").eval()
ics, inputs, targets, inds, conds, extra, _, true_inputs = [
    t.cpu() for t in dm.valid_ds.tensors
]
print("valid trials", inputs.shape, "tasks", dm.data_env.task_list)
if list(dm.data_env.task_list) != [SINGLE_TASK, MULTI_TASK]:
    raise ValueError(f"--tasks {SINGLE_TASK} {MULTI_TASK} do not match the dataset")
torch.manual_seed(0)  # the model's initial hidden state is noisy
with torch.no_grad():
    out = wrapper.forward(ics, inputs)["controlled"].numpy()
targets = targets.numpy()
true_inputs = true_inputs.numpy()
task = conds.numpy().astype(int).ravel()
go = extra[:, 0].numpy().astype(int)
end = extra[:, 1].numpy().astype(int)
print("valid MSE over all bins:", ((out - targets) ** 2).mean())


def segments(tgt):
    """Hold periods as (start, stop) bins.

    A hold is a run of bins where Valid=1 and the target is constant.
    """
    valid = tgt[:, 2] > 0.5
    change = np.r_[
        True,
        np.any(np.abs(np.diff(tgt[:, :2], axis=0)) > 1e-9, axis=1)
        | (valid[1:] != valid[:-1]),
    ]
    starts = np.flatnonzero(change)
    stops = np.r_[starts[1:], len(tgt)]
    return [(a, b) for a, b in zip(starts, stops) if valid[a]]


def ang_err(o, t):
    d = np.arctan2(o[:, 1], o[:, 0]) - np.arctan2(t[1], t[0])
    return np.abs(np.rad2deg((d + np.pi) % (2 * np.pi) - np.pi))


rows = []
for i in range(len(out)):
    segs = segments(targets[i, : end[i]])
    for k, (a, b) in enumerate(segs):
        hold = slice(a + SKIP_BINS, b)
        o = out[i, hold]
        rows.append(
            dict(
                trial=i,
                task=dm.data_env.task_list[task[i]],
                n_items=len(segs),
                item=k + 1,
                err=ang_err(o, targets[i, a, :2]).mean(),
                err_early=ang_err(out[i, a : a + SKIP_BINS], targets[i, a, :2]).mean(),
                radius=np.linalg.norm(o[:, :2], axis=1).mean(),
                valid=o[:, 2].mean(),
                hold_bins=b - a - SKIP_BINS,
            )
        )
    # off periods: before Go, and after the last trigger
    rows[-1]["tail_xy"] = np.abs(out[i, segs[-1][1] + SKIP_BINS : end[i], :2]).mean()
    rows[-1]["tail_valid"] = out[i, segs[-1][1] + SKIP_BINS : end[i], 2].mean()
    rows[-1]["pre_valid"] = out[i, : go[i], 2].mean()
    rows[-1]["pre_xy"] = np.abs(out[i, : go[i], :2]).mean()
df = pd.DataFrame(rows)
df.to_csv(RUN / f"recall_errors{LABEL}.csv", index=False)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:.3f}")
print("\nmin hold bins:", df.hold_bins.min())
print("\n== per task: mean angular error (deg) in hold, over all targets ==")
print(df.groupby("task").err.agg(["mean", "median", "max", "count"]))
trial = df.groupby(["task", "trial"]).err.max().reset_index()
print("\n== per task: fraction of trials with every target within 15 deg ==")
print(
    trial.assign(ok=trial.err <= TOL_DEG)
    .groupby("task")
    .ok.agg(["mean", "sum", "count"])
)
print("\n== by number of items and item position ==")
print(
    df.groupby(["task", "n_items", "item"]).agg(
        err=("err", "mean"),
        err_median=("err", "median"),
        within15=("err", lambda e: (e <= TOL_DEG).mean()),
        err_first80ms=("err_early", "mean"),
        radius=("radius", "mean"),
        valid=("valid", "mean"),
        n=("err", "count"),
    )
)
t2 = df.groupby(["task", "n_items", "trial"]).err.max().reset_index()
print("\n== fraction of trials all-within-15 by number of items ==")
print(
    t2.assign(ok=t2.err <= TOL_DEG)
    .groupby(["task", "n_items"])
    .ok.agg(["mean", "count"])
)
print("\n== off periods (output should be 0) ==")
print(
    df.dropna(subset=["tail_xy"])
    .groupby("task")[["pre_xy", "pre_valid", "tail_xy", "tail_valid"]]
    .mean()
)
# is a wrong answer another item of the same trial? (order confusion)
bad = df[df.err > TOL_DEG]
print("\nbad targets:", len(bad), "of", len(df))
print(bad.groupby(["task", "n_items", "item"]).size())
conf = 0
for _, r in bad.iterrows():
    i = int(r.trial)
    segs = segments(targets[i, : end[i]])
    a, b = segs[int(r["item"]) - 1]
    others = [
        targets[i, s[0], :2] for j, s in enumerate(segs) if j != int(r["item"]) - 1
    ]
    if (
        others
        and min(ang_err(out[i, a + SKIP_BINS : b], t).mean() for t in others) <= TOL_DEG
    ):
        conf += 1
print("bad targets that match a different item of the same trial:", conf)

# ---------- 2. output vs target for three validation trials ----------
n_items = df.groupby("trial").n_items.first()
picks = [(SINGLE_TASK, 1), (MULTI_TASK, 2), (MULTI_TASK, 3)]
fig, axes = plt.subplots(3, 1, figsize=(9, 9))
for ax, (tname, n) in zip(axes, picks):
    cand = [
        i
        for i in n_items.index
        if n_items[i] == n and dm.data_env.task_list[task[i]] == tname
    ]
    i = cand[0]
    T = end[i]
    time = np.arange(T) * BIN_MS / 1000
    for c in range(3):
        ax.plot(time, targets[i, :T, c], color=COLORS[c], lw=1, ls="--", alpha=0.9)
        ax.plot(time, out[i, :T, c], color=COLORS[c], lw=1.8, label=LABELS[c])
    ax.axvline(go[i] * BIN_MS / 1000, color="#52514e", lw=0.8)
    for on in np.flatnonzero(np.diff(np.r_[0, true_inputs[i, :T, 3]]) > 0.5):
        ax.axvline(on * BIN_MS / 1000, color="#52514e", lw=0.8, ls=":")
    worst = df[df.trial == i].err.max()
    ax.set_title(
        f"{tname}, {n} item{'s' if n > 1 else ''} "
        f"(validation trial {i}; worst hold error {worst:.1f} deg)",
        fontsize=11,
    )
    ax.set_ylim(-1.3, 1.3)
    ax.set_ylabel("Output")
    ax.spines[["top", "right"]].set_visible(False)
axes[-1].set_xlabel("Time (s)")
axes[0].legend(
    loc="upper left",
    ncol=3,
    frameon=False,
    title="Solid: network output   Dashed: target   "
    "Vertical: Go (solid), Trigger (dotted)",
    title_fontsize=8,
)
fig.tight_layout()
fig.savefig(RUN / f"planner_outputs_vs_targets{LABEL}.png", dpi=150)
plt.close(fig)
print("\nsaved figures in", RUN)

# ---------- where is the remaining squared error? ----------
se = (out - targets) ** 2
mask_pad = np.arange(300)[None, :] < end[:, None]
sw = np.zeros_like(mask_pad)
for i in range(len(out)):
    for a, b in segments(targets[i, : end[i]]):
        sw[i, a : a + SKIP_BINS] = True
    sw[i, b : b + SKIP_BINS] = True
print(
    "\nMSE inside trial:",
    se[mask_pad].mean(),
    "| in padding after trial end:",
    se[~mask_pad].mean(),
    "| padding share of bins:",
    (~mask_pad).mean(),
)
print(
    "MSE in first 80 ms after a switch:",
    se[sw & mask_pad].mean(),
    "| elsewhere inside trial:",
    se[mask_pad & ~sw].mean(),
)
print(
    "per-bin MSE at offsets 0..5 after a switch:",
    [
        float(
            np.mean(
                [
                    se[i, a + k].mean()
                    for i in range(len(out))
                    for a, b in segments(targets[i, : end[i]])
                ]
            )
        )
        for k in range(6)
    ],
)
print(
    "mean |output| in padding (XY, Valid):",
    np.abs(out[..., :2])[~mask_pad].mean(),
    out[..., 2][~mask_pad].mean(),
)
print(df.groupby(["task", "n_items", "item"]).err_early.mean())
ld = dict(
    controlled=torch.tensor(out),
    actions=torch.tensor(out),
    targets=torch.tensor(targets),
    inputs=inputs,
    conds=conds,
    extra=extra,
    epoch=199,
)
print("StackLoss on these validation trials:", float(wrapper.loss_func(ld)))
