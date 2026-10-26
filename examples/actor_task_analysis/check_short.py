"""Reaching and loss terms of every trial of an ActorTask run, next to a baseline.

For each trial's final model and one mid-run checkpoint, reports how many reach
trials end within reach_radius and how long the hand takes to arrive; then plots
the validation loss terms over epochs and prints their ratios at the end.
Expects the run's checkpoints and metrics.csv copied into
content/runs/task-trained/<run-tag>/lightning_logs/<run-name>/.
Run from anywhere: python examples/actor_task_analysis/check_short.py
    [--run-tag TAG] [--checkpoint epoch=299-step=2100.ckpt]
"""

import argparse
import pickle
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")

parser = argparse.ArgumentParser()
parser.add_argument("--run-tag", default="20261025_ActorTask_step_short")
parser.add_argument(
    "--checkpoint", help="checkpoint file name; default: the middle saved epoch"
)
parser.add_argument("--baseline", default="20261007_ActorTask_small")
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "content" / "trained_models" / "task-trained"
RUNS = ROOT / "content" / "runs" / "task-trained"
LOGS = RUNS / args.run_tag / "lightning_logs"
COLORS = ["#2a78d6", "#eb6834", "#1baa7a", "#9b59b6"]
GREY = "#52514e"

trials = sorted(p.name for p in LOGS.iterdir() if p.is_dir())
# Label each trial by the settings that differ between trials
parts = [
    {k.strip("_"): v for k, v in re.findall(r"([a-z_]+)=([^_]+)", t)} for t in trials
]
varying = [k for k in parts[0] if len({p.get(k) for p in parts}) > 1]
labels = {
    t: ", ".join(f"{k}={p[k]}" for k in varying) or t for t, p in zip(trials, parts)
}


def middle_checkpoint(log_dir):
    ckpts = sorted(
        log_dir.glob("epoch=*.ckpt"), key=lambda p: int(p.stem.split("-")[0][6:])
    )
    return ckpts[len(ckpts) // 2]


cases = [(args.baseline, next((MODELS / args.baseline).iterdir()), None)]
for t in trials:
    ckpt = (
        LOGS / t / args.checkpoint if args.checkpoint else middle_checkpoint(LOGS / t)
    )
    cases.append((f"{labels[t]} final", MODELS / args.run_tag / t, None))
    cases.append(
        (f"{labels[t]} {ckpt.stem.split('-')[0]}", MODELS / args.run_tag / t, ckpt)
    )

# ---------------- reaching ----------------
print("arrival = first bin within reach_radius of the goal after TargetValid on")
for label, model_dir, ckpt in cases:
    with open(model_dir / "model.pkl", "rb") as f:
        wrapper = pickle.load(f).to("cpu").eval()
    with open(model_dir / "datamodule_train.pkl", "rb") as f:
        datamodule = pickle.load(f)
    if ckpt:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        wrapper.load_state_dict(state["state_dict"])
    # The datamodule holds the training environment with the run's loss settings
    env = datamodule.data_env
    wrapper.set_environment(env)
    torch.manual_seed(0)
    tensors = [t.cpu() for t in datamodule.valid_ds.tensors]
    ics, inputs, targets, _, conds, extra, inputs_to_env, _ = tensors
    with torch.no_grad():
        tip = wrapper.forward(ics, inputs, inputs_to_env)["controlled"].numpy()
    goal = targets[:, -1].numpy()
    valid_on = extra[:, 0].numpy().astype(int)
    reach = np.flatnonzero(~conds.numpy().astype(bool).ravel())
    radius = env.loss_func.reach_radius
    arrive_ms, end_in = [], 0
    for i in reach:
        near = np.linalg.norm(tip[i] - goal[i], axis=-1) < radius
        end_in += near[-1]
        near[: valid_on[i]] = False
        if near.any():
            arrive_ms.append((near.argmax() - valid_on[i]) * env.dt * 1000)
    print(
        f"{label}: end within reach_radius {end_in}/{len(reach)} "
        f"({end_in / len(reach):.1%}); arrive {len(arrive_ms)}/{len(reach)}, "
        f"median {np.median(arrive_ms):.0f} ms "
        f"(IQR {np.percentile(arrive_ms, 25):.0f}-{np.percentile(arrive_ms, 75):.0f})"
    )

# ---------------- loss terms over epochs ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
terms = ["position", "effort", "trigger"]
print("\nfinal-epoch validation loss terms (weighted, as trained)")
for t, color in zip(trials, COLORS):
    metrics = pd.read_csv(LOGS / t / "metrics.csv")
    valid = metrics.dropna(subset=["valid/loss"]).groupby("epoch").last()
    for ax, term in zip(axes, terms):
        ax.semilogy(
            valid.index,
            valid[f"valid/{term}_loss"],
            color=color,
            lw=1.4,
            label=labels[t],
        )
    pos, eff, trig = (valid.iloc[-1][f"valid/{k}_loss"] for k in terms)
    print(
        f"{labels[t]}: position {pos:.2e}, effort {eff:.2e}, trigger {trig:.2e}; "
        f"position:effort:trigger = 1 : {eff / pos:.2f} : {trig / pos:.2f}"
    )
base_metrics = next((RUNS / args.baseline).rglob("metrics.csv"), None)
if base_metrics is not None:
    base = pd.read_csv(base_metrics).dropna(subset=["valid/loss"])
    base = base.groupby("epoch").last()
    axes[0].semilogy(
        base.index,
        base["valid/loss"],
        color=GREY,
        lw=1.2,
        ls="--",
        label=f"{args.baseline}: total loss only",
    )
for ax, title in zip(axes, ["Position", "Effort", "Trigger (weighted)"]):
    ax.set_title(f"{title} loss, validation", fontsize=11)
    ax.set_xlabel("Epoch")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False, fontsize=8.5)
axes[0].set_ylabel("Loss (log scale)")
fig.suptitle(f"{args.run_tag}: loss terms over epochs", fontsize=12)
fig.tight_layout()
out_png = RUNS / args.run_tag / "loss_terms_over_epochs.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print("\nsaved", out_png)
