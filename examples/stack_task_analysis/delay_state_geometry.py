"""Geometry of the StackTask delay-period hidden states.

PCA of the pre-Go state in SingleRecall (is it a ring?), and linear decoders
for item 1 and item 2 in two-item ForwardRecall with their subspace overlap.
Run from anywhere: python examples/stack_task_analysis/delay_state_geometry.py
"""

import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import subspace_angles
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
RUN_TAG = "20261007_StackTask_test"
RUN = ROOT / "content" / "runs" / "task-trained" / RUN_TAG
MODEL = (
    ROOT
    / "content"
    / "trained_models"
    / "task-trained"
    / RUN_TAG
    / "max_epochs=200_batch_size=64_seed=0"
)
N = 300
with open(MODEL / "model.pkl", "rb") as f:
    wrapper = pickle.load(f).to("cpu").eval()
with open(MODEL / "datamodule_train.pkl", "rb") as f:
    env = pickle.load(f).data_env
env.set_seed(
    20261007
)  # new seed: trials the network has not seen (training used seed 0)


def collect(task, n_items):
    """Generate trials until N have n_items.

    Returns the pre-Go hidden states and the item angles.
    """
    X = np.zeros((N, env.n_timesteps, 4), dtype=np.float32)
    infos = []
    while len(infos) < N:
        inp, _, _, info = env.generate_trial(task)
        if info["n_items"] == n_items:
            X[len(infos), : info["trial_end"]] = inp
            infos.append(info)
    with torch.no_grad():
        o = wrapper.forward(torch.zeros(N, 3), torch.tensor(X))
    lat, out = o["latents"].numpy(), o["controlled"].numpy()
    go = np.array([i["go"] for i in infos])
    ang = np.array([i["angles"] for i in infos])
    H = lat[np.arange(N), go - 1]
    # sanity: recall error in each hold (skip 4 bins after each switch)
    errs = []
    for k, info in enumerate(infos):
        sw = [info["go"], *info["trigger_on"]]
        for j in range(n_items):
            seg = out[k, sw[j] + 4 : sw[j + 1], :2]
            d = np.arctan2(seg[:, 1], seg[:, 0]) - info["angles"][j]
            errs.append(np.abs(np.rad2deg((d + np.pi) % (2 * np.pi) - np.pi)).mean())
    delay = np.array([(i["go"] - (i["item_on"][-1] + 10)) * 20 for i in infos])
    print(
        f"{task} {n_items}-item: states {H.shape}, "
        f"mean recall error {np.mean(errs):.2f} deg, max {np.max(errs):.1f}, "
        f"delay {delay.min()}-{delay.max()} ms"
    )
    return H, ang


def pca(H):
    Hc = H - H.mean(0)
    U, s, Vt = np.linalg.svd(Hc, full_matrices=False)
    return Hc @ Vt.T, s**2 / np.sum(s**2), Vt


# ---------------- 1. SingleRecall ring ----------------
H1, a1 = collect("SingleRecall", 1)
a1 = a1[:, 0]
Z, var, _ = pca(H1)
print(
    "variance explained PC1-3:",
    np.round(var[:3], 4),
    "sum",
    round(var[:3].sum(), 4),
    "| PC1-2 sum",
    round(var[:2].sum(), 4),
    "| PCs for 99%:",
    int(np.searchsorted(np.cumsum(var), 0.99) + 1),
    "| participation ratio",
    round(1 / np.sum(var**2), 2),
)
r = np.linalg.norm(Z[:, :2], axis=1)
phi = np.arctan2(Z[:, 1], Z[:, 0])
print(
    f"radius in PC1-2 plane: mean {r.mean():.3f}, "
    f"CV {r.std() / r.mean():.3f}, min/max {r.min():.3f}/{r.max():.3f}"
)
# ordered by angle? best rotation/reflection of plane angle onto target angle
best = None
for sgn in (1, -1):
    d = sgn * phi - a1
    off = np.arctan2(np.sin(d).mean(), np.cos(d).mean())
    res = np.rad2deg(np.abs((d - off + np.pi) % (2 * np.pi) - np.pi))
    if best is None or res.mean() < best[0]:
        best = (res.mean(), res.max(), sgn)
print(
    f"plane angle vs target angle after best rotation: "
    f"mean abs deviation {best[0]:.1f} deg, max {best[1]:.1f} deg, "
    f"orientation {'same' if best[2] > 0 else 'mirrored'}"
)
order = np.argsort(a1)
ph = np.unwrap(best[2] * phi[order])
print(
    "plane angle monotonic in target angle: violations",
    int((np.diff(ph) < 0).sum()),
    "of",
    N - 1,
    "| total winding (deg)",
    round(float(np.rad2deg(ph[-1] - ph[0])), 1),
)
Y1 = np.c_[np.cos(a1), np.sin(a1)]
A = np.c_[Z[:, :2], np.ones(N)]
pred = A @ np.linalg.lstsq(A, Y1, rcond=None)[0]
print(
    "R^2 of (cos, sin) from PC1-2 alone:",
    round(r2_score(Y1, pred), 4),
    "| from PC1-3:",
    round(
        r2_score(
            Y1,
            np.c_[Z[:, :3], np.ones(N)]
            @ np.linalg.lstsq(np.c_[Z[:, :3], np.ones(N)], Y1, rcond=None)[0],
        ),
        4,
    ),
)
print(
    "corr of PC3 with cos2a, sin2a:",
    np.round(
        [
            np.corrcoef(Z[:, 2], np.cos(2 * a1))[0, 1],
            np.corrcoef(Z[:, 2], np.sin(2 * a1))[0, 1],
        ],
        3,
    ),
)

fig = plt.figure(figsize=(12, 5.4))
ax = fig.add_subplot(1, 2, 1)
sc = ax.scatter(
    Z[:, 0],
    Z[:, 1],
    c=np.rad2deg(a1),
    cmap="twilight",
    vmin=0,
    vmax=360,
    s=26,
    edgecolors="#52514e",
    linewidths=0.3,
)
ax.set_xlabel(f"PC1 ({var[0]:.0%} of variance)")
ax.set_ylabel(f"PC2 ({var[1]:.0%})")
ax.set_aspect("equal")
ax.set_title("PC1 vs PC2")
ax.spines[["top", "right"]].set_visible(False)
ax3 = fig.add_subplot(1, 2, 2, projection="3d")
ax3.scatter(
    Z[:, 0],
    Z[:, 1],
    Z[:, 2],
    c=np.rad2deg(a1),
    cmap="twilight",
    vmin=0,
    vmax=360,
    s=22,
    edgecolors="#52514e",
    linewidths=0.3,
)
ax3.set_xlabel("PC1")
ax3.set_ylabel("PC2")
ax3.set_zlabel(f"PC3 ({var[2]:.0%})")
ax3.set_title("PC1, PC2, PC3")
ax3.view_init(elev=28, azim=-58)
lim = np.abs(Z[:, :3]).max() * 1.05
ax3.set_xlim(-lim, lim)
ax3.set_ylim(-lim, lim)
ax3.set_zlim(-lim, lim)
cb = fig.colorbar(sc, ax=[ax, ax3], shrink=0.75, pad=0.06, ticks=[0, 90, 180, 270, 360])
cb.set_label("Target angle (deg)")
fig.suptitle(
    f"SingleRecall: hidden state at the last bin before Go ({N} unseen trials)"
)
fig.savefig(RUN / "planner_delay_states_pca.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------- 2. Two-item ForwardRecall decoders ----------------
H2, a2 = collect("ForwardRecall", 2)
Hc = H2 - H2.mean(0)
_, var2, Vt2 = pca(H2)
print(
    "two-item states: PCs for 99% variance:",
    int(np.searchsorted(np.cumsum(var2), 0.99) + 1),
    "| participation ratio",
    round(1 / np.sum(var2**2), 2),
    "| top-6 var",
    np.round(var2[:6], 3),
)
alphas = np.logspace(-6, 2, 17)
kf = KFold(10, shuffle=True, random_state=0)
Ys = [np.c_[np.cos(a2[:, j]), np.sin(a2[:, j])] for j in (0, 1)]
W = []
for j in (0, 1):
    p = cross_val_predict(RidgeCV(alphas=alphas), Hc, Ys[j], cv=kf)
    px = cross_val_predict(
        RidgeCV(alphas=alphas), Hc, Ys[1 - j], cv=kf
    )  # same states, other item's angle
    m = RidgeCV(alphas=alphas).fit(Hc, Ys[j])
    W.append(m.coef_)
    d = np.arctan2(p[:, 1], p[:, 0]) - a2[:, j]
    de = np.abs(np.rad2deg((d + np.pi) % (2 * np.pi) - np.pi))
    print(
        f"item {j + 1}: 10-fold CV R^2 "
        f"cos {r2_score(Ys[j][:, 0], p[:, 0]):.4f}, "
        f"sin {r2_score(Ys[j][:, 1], p[:, 1]):.4f} | "
        f"decoded-angle error mean {de.mean():.2f} deg | ridge alpha {m.alpha_:.1e}"
    )
    # does item j's decoder read out the OTHER item? apply it to the other item's labels
    cross = m.predict(Hc)
    print(
        f"         item {j + 1} decoder scored against item {2 - j}'s angle: "
        f"R^2 {r2_score(Ys[1 - j], cross):.3f}"
    )
ang_dec = np.rad2deg(subspace_angles(W[0].T, W[1].T))
print(
    "principal angles between decoder weight planes (deg):",
    np.round(np.sort(ang_dec), 1),
    "| cosines",
    np.round(np.sort(np.cos(np.deg2rad(ang_dec)))[::-1], 3),
)
# encoding planes: how the state moves with each item's (cos, sin)
D = np.c_[Ys[0], Ys[1]]
B = np.linalg.lstsq(D - D.mean(0), Hc, rcond=None)[0]  # 4 x 128
fit = (D - D.mean(0)) @ B
print(
    "fraction of state variance explained by the 4 linear angle terms:",
    round(1 - ((Hc - fit) ** 2).sum() / (Hc**2).sum(), 4),
)
ang_enc = np.rad2deg(subspace_angles(B[:2].T, B[2:].T))
print(
    "principal angles between encoding planes (deg):",
    np.round(np.sort(ang_enc), 1),
    "| encoding vector norms item1",
    np.round(np.linalg.norm(B[:2], axis=1), 3),
    "item2",
    np.round(np.linalg.norm(B[2:], axis=1), 3),
)
# chance level: random 2-D planes inside the space the states actually occupy
rng = np.random.default_rng(0)
k99 = int(np.searchsorted(np.cumsum(var2), 0.99) + 1)
for name, dim in (("128-D", 128), (f"{k99}-D (99% variance)", k99)):
    sm = np.array(
        [
            np.sort(
                np.rad2deg(
                    subspace_angles(
                        rng.standard_normal((dim, 2)), rng.standard_normal((dim, 2))
                    )
                )
            )
            for _ in range(2000)
        ]
    )
    print(f"random planes in {name}: mean principal angles {np.round(sm.mean(0), 1)}")
print("saved", RUN / "planner_delay_states_pca.png")
