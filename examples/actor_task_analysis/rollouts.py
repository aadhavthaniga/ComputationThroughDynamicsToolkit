"""What a trained ActorTask model does on its validation trials.

Rolls out the saved model, reports how far and in which direction the hand
moves, the parts of the loss and how well Trigger fires, and plots fingertip
paths and the Trigger output against its label.
Run from anywhere: python examples/actor_task_analysis/rollouts.py
    [--run-tag TAG] [--run-name NAME] [--checkpoint path/to/last.ckpt]
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")

parser = argparse.ArgumentParser()
parser.add_argument("--run-tag", default="20261007_ActorTask_small")
parser.add_argument("--run-name", default="max_epochs=52_batch_size=256_seed=0")
parser.add_argument(
    "--checkpoint", help="Lightning checkpoint whose weights replace model.pkl's"
)
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[2]
RUN_TAG = args.run_tag
MODEL = ROOT / "content" / "trained_models" / "task-trained" / RUN_TAG / args.run_name
RUN = ROOT / "content" / "runs" / "task-trained" / RUN_TAG
N_PLOT = 16
TRIGGER_WINDOW = 10  # bins (100 ms) around label onset that count as on time
COLORS = ["#2a78d6", "#eb6834"]
GREY = "#52514e"

with open(MODEL / "model.pkl", "rb") as f:
    wrapper = pickle.load(f).to("cpu").eval()
with open(MODEL / "datamodule_train.pkl", "rb") as f:
    datamodule = pickle.load(f)

if args.checkpoint:
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    wrapper.load_state_dict(ckpt["state_dict"])

# model.pkl is saved holding the shorter simulation environment; the datamodule
# holds the training environment with the run's own loss settings (runs saved
# before trigger_shape existed get the pulse label)
env = datamodule.data_env
wrapper.set_environment(env)
loss_func = env.loss_func
torch.manual_seed(0)
np.random.seed(0)

tensors = [t.cpu() for t in datamodule.valid_ds.tensors]
ics, inputs, targets, _, conds, extra, inputs_to_env, _ = tensors
with torch.no_grad():
    out = wrapper.forward(ics, inputs, inputs_to_env)
loss_dict = {
    "controlled": out["controlled"],
    "actions": out["actions"],
    "targets": targets,
    "extra": extra,
    "epoch": loss_func.full_trial_epoch,  # score the full trial
}
total = loss_func(loss_dict)
label = loss_func.trigger_label(out["controlled"], targets, extra[:, 0])
trigger = out["actions"][..., loss_func.n_muscles]

tip = out["controlled"].numpy()
goal = targets[:, -1, :].numpy()
center = env.center.numpy()
radius = env.reach_dist
valid_on = extra[:, 0].numpy().astype(int)
catch = conds.numpy().astype(bool).ravel()
dist = np.linalg.norm(tip - center, axis=-1)  # distance from start per timestep

# ---------------- movement ----------------
reach = np.flatnonzero(~catch)
toward_goal = (goal[reach] - center) / radius
end = tip[reach, -1] - center
along = (end * toward_goal).sum(axis=1)
cos_err = along / np.maximum(np.linalg.norm(end, axis=1), 1e-9)
direction_err = np.rad2deg(np.arccos(np.clip(cos_err, -1, 1)))
to_goal = np.linalg.norm(tip[reach, -1] - goal[reach], axis=1)
before_valid = np.array([dist[i, : valid_on[i]].max() for i in reach])

print(f"{len(reach)} reach trials, {catch.sum()} catch trials, R = {radius} m")
print(
    f"reach: distance from start at end {dist[reach, -1].mean():.4f} m "
    f"({dist[reach, -1].mean() / radius:.0%} of R), "
    f"largest during trial {dist[reach].max(axis=1).mean():.4f} m"
)
print(
    f"reach: movement along the target direction {along.mean():.4f} m "
    f"({along.mean() / radius:.0%} of R), "
    f"direction error {direction_err.mean():.0f} deg (90 = unrelated)"
)
print(
    f"reach: distance to goal at end {to_goal.mean():.4f} m, "
    f"{(to_goal < loss_func.reach_radius).sum()} of {len(reach)} trials end "
    f"within reach_radius"
)
print(f"reach: largest distance from start before Valid {before_valid.mean():.4f} m")
print(f"catch: distance from start at end {dist[catch, -1].mean():.4f} m")

# ---------------- loss ----------------
unweighted = ((trigger - label) ** 2).mean().item()
never = loss_func.trigger_loss(torch.zeros_like(label), label).item()
print(f"\nloss on the validation trials (full trial scored): {total.item():.4f}")
for name, value in loss_func.last_terms.items():
    print(f"  {name}: {value.item():.4f}")
print(
    f"Trigger output: mean {trigger.mean().item():.4f}, max "
    f"{trigger.max().item():.4f}; label has a pulse in "
    f"{int((label.sum(dim=1) > 0).sum())} trials"
)
print(f"trigger loss, unweighted MSE (trigger_pos_weight = 1): {unweighted:.4f}")
print(
    f"trigger loss with trigger_pos_weight = {loss_func.trigger_pos_weight} "
    f"({loss_func.labeled_weight(env.n_timesteps):.2f}): "
    f"{loss_func.trigger_loss(trigger, label).item():.4f} "
    f"(never firing would score {never:.4f}, a correct label 0)"
)

# ---------------- trigger ----------------
fired = trigger > 0.5
has_label = label.sum(dim=1) > 0
onset = (label > 0).int().argmax(dim=1)
first_fire = torch.where(fired.any(dim=1), fired.int().argmax(dim=1), -1)
on_time = (first_fire >= 0) & ((first_fire - onset).abs() <= TRIGGER_WINDOW)
inside = label > 0
shape = f"{loss_func.trigger_shape}, {loss_func.trigger_steps * env.dt * 1000:.0f} ms"
if loss_func.trigger_shape == "step_decay":
    shape += f" at 1, then {loss_func.decay_steps * env.dt * 1000:.0f} ms decay"
print(f"\nTrigger label: {shape}")
print(
    f"labeled trials whose Trigger first crosses 0.5 within "
    f"{TRIGGER_WINDOW * env.dt * 1000:.0f} ms of label onset: "
    f"{int(on_time[has_label].sum())} of {int(has_label.sum())} "
    f"({on_time[has_label].float().mean().item():.0%})"
)
print(
    f"mean Trigger inside the label {trigger[inside].mean().item():.4f}, "
    f"outside {trigger[~inside].mean().item():.4f}"
)
catch_fired = fired[torch.as_tensor(catch)].any(dim=1)
print(
    f"catch trials where Trigger exceeds 0.5: {int(catch_fired.sum())} of "
    f"{len(catch_fired)} ({catch_fired.float().mean().item():.0%})"
)

# ---------------- figure ----------------
trigger = trigger.numpy()
label = label.numpy()
fig = plt.figure(figsize=(14, 6.4))
grid = fig.add_gridspec(2, 4, width_ratios=[1.15, 1.15, 1, 1])
ax = fig.add_subplot(grid[:, :2])
theta = np.linspace(0, 2 * np.pi, 200)
ax.plot(
    center[0] + radius * np.cos(theta),
    center[1] + radius * np.sin(theta),
    color=GREY,
    lw=1,
    label=f"Circle of possible goals (R = {radius:.2f} m)",
)
labelled = set()
for i in range(N_PLOT):
    if catch[i]:
        name = None if "catch" in labelled else "Catch trial path"
        labelled.add("catch")
        ax.plot(tip[i, :, 0], tip[i, :, 1], color=GREY, lw=1.2, ls=":", label=name)
        continue
    name = None if "reach" in labelled else "Fingertip path (reach trial)"
    labelled.add("reach")
    ax.plot(
        [tip[i, -1, 0], goal[i, 0]],
        [tip[i, -1, 1], goal[i, 1]],
        color="#c3c2b7",
        lw=0.6,
        zorder=1,
    )
    ax.plot(tip[i, :, 0], tip[i, :, 1], color=COLORS[0], lw=1.4, label=name)
    ax.scatter(
        *tip[i, -1], s=22, color=COLORS[0], zorder=3, edgecolors="#fcfcfb", lw=0.6
    )
ax.scatter(
    *goal[:N_PLOT][~catch[:N_PLOT]].T,
    s=55,
    color=COLORS[1],
    zorder=3,
    edgecolors="#fcfcfb",
    lw=0.8,
    label="Target for that trial",
)
ax.scatter(*center, s=140, marker="+", color="#0b0b0b", zorder=4, label="Start")
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title(
    f"Fingertip paths, {N_PLOT} validation trials\n"
    "(dot = end of path; thin grey line joins it to that trial's target)",
    fontsize=11,
)
ax.legend(
    frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2
)
ax.spines[["top", "right"]].set_visible(False)

time = np.arange(env.n_timesteps) * env.dt
shown = list(np.flatnonzero(~catch[:N_PLOT])[:3])
shown += list(np.flatnonzero(catch[:N_PLOT])[:1])
for k, i in enumerate(shown):
    bx = fig.add_subplot(grid[k // 2, 2 + k % 2])
    bx.plot(time, label[i], color=COLORS[1], lw=1.5, ls="--", label="Label")
    bx.plot(time, trigger[i], color=COLORS[0], lw=1.6, label="Trigger output")
    if valid_on[i] >= 0:
        bx.axvline(valid_on[i] * env.dt, color=GREY, lw=0.8, ls=":")
    kind = "catch" if catch[i] else f"Valid on at {valid_on[i] * 10} ms"
    bx.set_title(
        f"Trial {i}: {kind}\nhand ends {dist[i, -1] * 100:.1f} cm from start",
        fontsize=9.5,
    )
    bx.set_ylim(-0.25, 1.15)
    bx.spines[["top", "right"]].set_visible(False)
    if k >= 2:
        bx.set_xlabel("Time (s)")
    if k % 2 == 0:
        bx.set_ylabel("Trigger")
    if k == 0:
        bx.legend(frameon=False, fontsize=8.5, loc="upper right")
fig.suptitle(f"{RUN_TAG}: what the trained arm does", fontsize=12)
fig.tight_layout()
suffix = f"_{Path(args.checkpoint).stem}" if args.checkpoint else ""
out_png = RUN / f"rollouts_{args.run_name}{suffix}.png"
out_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print("\nsaved", out_png)
