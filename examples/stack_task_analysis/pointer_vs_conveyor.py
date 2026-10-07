"""Where the StackTask model keeps item 2, and how the memory meets the readout.

Applies the pre-Go item decoders through time around the first trigger, and
measures how much of the delay code lies along the TargetX/TargetY readout.
Run from anywhere: python examples/stack_task_analysis/pointer_vs_conveyor.py
"""

import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import orth, subspace_angles
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
N, BIN = 300, 20
with open(MODEL / "model.pkl", "rb") as f:
    wrapper = pickle.load(f).to("cpu").eval()
with open(MODEL / "datamodule_train.pkl", "rb") as f:
    env = pickle.load(f).data_env
env.set_seed(
    20261007
)  # same seed and same order as the previous analysis -> same trials


def collect(task, n_items):
    X = np.zeros((N, env.n_timesteps, 4), dtype=np.float32)
    infos = []
    while len(infos) < N:
        inp, _, _, info = env.generate_trial(task)
        if info["n_items"] == n_items:
            X[len(infos), : info["trial_end"]] = inp
            infos.append(info)
    with torch.no_grad():
        o = wrapper.forward(torch.zeros(N, 3), torch.tensor(X))
    return o["latents"].numpy(), o["controlled"].numpy(), infos


def aerr(pred, ang):
    d = np.arctan2(pred[:, 1], pred[:, 0]) - ang
    return np.abs(np.rad2deg((d + np.pi) % (2 * np.pi) - np.pi))


lat1, out1, inf1 = collect("SingleRecall", 1)
lat2, out2, inf2 = collect("ForwardRecall", 2)
idx = np.arange(N)
go = np.array([i["go"] for i in inf2])
t1 = np.array([i["trigger_on"][0] for i in inf2])
t2 = np.array([i["trigger_on"][1] for i in inf2])
ang = np.array([i["angles"] for i in inf2])
Y = [np.c_[np.cos(ang[:, j]), np.sin(ang[:, j])] for j in (0, 1)]
print("go->trig1 bins min", (t1 - go).min(), "| trig1->trig2 bins min", (t2 - t1).min())

# decoders fit at the pre-Go state (same as before)
H = lat2[idx, go - 1]
mu = H.mean(0)
alphas = np.logspace(-6, 2, 17)
dec = [RidgeCV(alphas=alphas).fit(H - mu, Y[j]) for j in (0, 1)]
print(
    "check, pre-Go in-sample decoded error (deg): item1",
    aerr(dec[0].predict(H - mu), ang[:, 0]).mean().round(2),
    "item2",
    aerr(dec[1].predict(H - mu), ang[:, 1]).mean().round(2),
)

# ---------- 1. time course around the first trigger ----------
rel = np.arange(
    -15, 20
)  # bins relative to trigger 1; all inside Go..trigger 2 for every trial
res = {
    k: []
    for k in [
        "d2_i2",
        "d1_i2",
        "d1_i1",
        "d2_i1",
        "out_i2",
        "out_i1",
        "len1",
        "len2",
        "r2_d2_i2",
        "r2_d1_i2",
        "d2_i2_raw",
        "d1_i2_raw",
    ]
}
for r in rel:
    Ht = lat2[idx, t1 + r]
    Hc_t = Ht - Ht.mean(0)  # remove the across-trial mean at this time bin
    p1, p2 = (
        dec[0].predict(Hc_t) - dec[0].intercept_,
        dec[1].predict(Hc_t) - dec[1].intercept_,
    )
    res["d2_i2"].append(np.median(aerr(p2, ang[:, 1])))
    res["d1_i2"].append(np.median(aerr(p1, ang[:, 1])))
    res["d1_i1"].append(np.median(aerr(p1, ang[:, 0])))
    res["d2_i1"].append(np.median(aerr(p2, ang[:, 0])))
    res["len1"].append(np.linalg.norm(p1, axis=1).mean())
    res["len2"].append(np.linalg.norm(p2, axis=1).mean())
    res["r2_d2_i2"].append(r2_score(Y[1], p2))
    res["r2_d1_i2"].append(r2_score(Y[1], p1))
    o = out2[idx, t1 + r, :2]
    res["out_i2"].append(np.median(aerr(o, ang[:, 1])))
    res["out_i1"].append(np.median(aerr(o, ang[:, 0])))
    pr1, pr2 = dec[0].predict(Ht - mu), dec[1].predict(Ht - mu)  # no re-centring
    res["d2_i2_raw"].append(np.median(aerr(pr2, ang[:, 1])))
    res["d1_i2_raw"].append(np.median(aerr(pr1, ang[:, 1])))
res = {k: np.array(v) for k, v in res.items()}
np.set_printoptions(linewidth=220, precision=1, suppress=True)
show = [0, 5, 10, 13, 14, 15, 16, 17, 18, 19, 20, 22, 25, 30, 34]
print("bins rel. trigger1:", rel[show])
for k in ["d2_i2", "d1_i2", "d1_i1", "d2_i1", "out_i2", "d2_i2_raw", "d1_i2_raw"]:
    print(f"{k:10s} median err deg:", res[k][show])
print("decoded vector length  dec1:", np.round(res["len1"][show], 2))
print("                       dec2:", np.round(res["len2"][show], 2))
print("R^2 item2  dec2:", np.round(res["r2_d2_i2"][show], 2))
print("           dec1:", np.round(res["r2_d1_i2"][show], 2))

# fresh decoders at +10 bins after trigger 1: where do the items live now?
kf = KFold(10, shuffle=True, random_state=0)
Hp = lat2[idx, t1 + 10]
Hpc = Hp - Hp.mean(0)
Wr = wrapper.model.readout.weight.detach().numpy()
br = wrapper.model.readout.bias.detach().numpy()
Q = orth(Wr[:2].T)  # 128 x 2 basis of the TargetX/TargetY readout plane
new = {}
for j in (0, 1):
    p = cross_val_predict(RidgeCV(alphas=alphas), Hpc, Y[j], cv=kf)
    m = RidgeCV(alphas=alphas).fit(Hpc, Y[j])
    new[j] = m.coef_
    print(
        f"fresh decoder at +200 ms after trigger 1, item {j + 1}: "
        f"CV R^2 {r2_score(Y[j], p):.4f}, "
        f"decoded error {aerr(p, ang[:, j]).mean():.2f} deg"
    )


def pa(A, B):
    return np.round(np.sort(np.rad2deg(subspace_angles(A.T, B.T))), 1)


print(
    "principal angles (deg): post-trigger item-2 decoder plane vs pre-Go item-2 plane",
    pa(new[1], dec[1].coef_),
    "| vs pre-Go item-1 plane",
    pa(new[1], dec[0].coef_),
    "| vs readout plane",
    pa(new[1], Wr[:2]),
)
print(
    "                        pre-Go item-1 decoder plane vs readout plane",
    pa(dec[0].coef_, Wr[:2]),
    "| pre-Go item-2 plane vs readout plane",
    pa(dec[1].coef_, Wr[:2]),
)


# encoding planes (how the state moves with each item) pre-Go and post-trigger
def enc(Hc):
    Dm = np.c_[Y[0], Y[1]]
    Dm = Dm - Dm.mean(0)
    B = np.linalg.lstsq(Dm, Hc, rcond=None)[0]
    return B[:2], B[2:]


E1, E2 = enc(H - mu)
P1, P2 = enc(Hpc)
print(
    "encoding planes: item 2 post-trigger vs item 2 pre-Go",
    pa(P2, E2),
    "| vs item 1 pre-Go",
    pa(P2, E1),
    "| item-1 norm pre-Go",
    np.linalg.norm(E1).round(2),
    "post-trigger",
    np.linalg.norm(P1).round(2),
    "| item-2 norm pre",
    np.linalg.norm(E2).round(2),
    "post",
    np.linalg.norm(P2).round(2),
)
# while item 1 is being output (5 bins before trigger 1)
Hd = lat2[idx, t1 - 5]
D1, D2 = enc(Hd - Hd.mean(0))
print(
    "during item-1 output: item 1 plane vs its pre-Go plane",
    pa(D1, E1),
    "| item 2 plane vs its pre-Go plane",
    pa(D2, E2),
    "| item 1 post-Go plane vs item 2 post-trigger plane",
    pa(D1, P2),
)


# ---------- 2. output-potent vs output-null ----------
def potent(M):  # M: rows are state-space vectors (centred states or encoding vectors)
    return float(((M @ Q) ** 2).sum() / (M**2).sum())


Hs = lat1[idx, np.array([i["go"] for i in inf1]) - 1]
Hs_c = Hs - Hs.mean(0)
gs = np.array([i["go"] for i in inf1])
Hs_hold = lat1[idx, gs + 10]
Hs_hold_c = Hs_hold - Hs_hold.mean(0)
rows = [
    ("SingleRecall ring, before Go", potent(Hs_c)),
    ("Two-item: item 1 code, before Go", potent(E1)),
    ("Two-item: item 2 code, before Go", potent(E2)),
    ("SingleRecall, during output (Go + 200 ms)", potent(Hs_hold_c)),
    ("Two-item: item 2 code, during its output", potent(P2)),
]
for n_, v in rows:
    print(f"{n_:45s} output-potent {v:7.2%}  output-null {1 - v:7.2%}")
print("chance (random 2-D plane in 128-D):", 2 / 128)
print(
    "readout row norms X, Y, Valid:",
    np.round(np.linalg.norm(Wr, axis=1), 3),
    "| angle between X and Y readout rows (deg):",
    np.rad2deg(
        np.arccos(Wr[0] @ Wr[1] / np.linalg.norm(Wr[0]) / np.linalg.norm(Wr[1]))
    ).round(1),
)
print(
    "std across trials of the X/Y output: before Go",
    (Hs_c @ Wr[:2].T).std(0).round(4),
    "| during output",
    (Hs_hold_c @ Wr[:2].T).std(0).round(3),
)
print(
    "ring radius (state units) before Go:",
    np.linalg.norm(Hs_c, axis=1).mean().round(3),
    "| its component in the readout plane:",
    np.linalg.norm(Hs_c @ Q, axis=1).mean().round(4),
    "| during output: total",
    np.linalg.norm(Hs_hold_c, axis=1).mean().round(3),
    "in readout plane",
    np.linalg.norm(Hs_hold_c @ Q, axis=1).mean().round(3),
)
# is the pre-Go ring also null for TargetValid?
Q3 = orth(Wr.T)
print(
    "output-potent incl. TargetValid row (3-D): ring before Go",
    round(float(((Hs_c @ Q3) ** 2).sum() / (Hs_c**2).sum()), 4),
)

# ---------- figure ----------
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
fig, (ax, bx) = plt.subplots(
    1, 2, figsize=(13.5, 5), gridspec_kw=dict(width_ratios=[1.25, 1])
)
t_ms = rel * BIN
series = [
    ("d2_i2", "Item-2 decoder reads item 2 (pointer)", C[0], "-"),
    ("d1_i2", "Item-1 decoder reads item 2 (conveyor belt)", C[1], "-"),
    ("d1_i1", "Item-1 decoder reads item 1", C[2], "--"),
    ("out_i2", "Network output vs item 2", C[3], ":"),
]
for k, lab, c, ls in series:
    ax.plot(t_ms, res[k], color=c, ls=ls, lw=2, label=lab)
ax.axhline(90, color="#52514e", lw=0.8, ls=(0, (1, 3)))
ax.text(t_ms[0], 92, "chance (90 deg)", fontsize=8, color="#52514e")
ax.axvline(0, color="#0b0b0b", lw=0.8)
ax.text(-8, 172, "Trigger 1", ha="right", fontsize=9, color="#0b0b0b")
ax.set_ylim(0, 180)
ax.set_yticks([0, 45, 90, 135, 180])
ax.set_xlabel("Time from first trigger (ms)")
ax.set_ylabel("Median decoded-angle error (deg)")
ax.set_title(
    "Where is item 2 after the first trigger?\n"
    "(decoders fit at the last bin before Go)",
    fontsize=11,
)
ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 0.97))
ax.spines[["top", "right"]].set_visible(False)
names = [r[0] for r in rows][::-1]
vals = np.array([r[1] for r in rows][::-1]) * 100
bx.barh(names, vals, color=C[0], height=0.55)
for y, v in enumerate(vals):
    bx.text(
        v + 1.5,
        y,
        f"{v:.1f}% potent, {100 - v:.1f}% null",
        va="center",
        fontsize=9,
        color="#0b0b0b",
    )
bx.set_xlim(0, 100)
bx.set_xlabel("Share of angle-coding variance in the TargetX/TargetY readout plane (%)")
bx.set_title("Output-potent vs output-null", fontsize=11)
bx.spines[["top", "right"]].set_visible(False)
bx.tick_params(axis="y", length=0)
fig.tight_layout()
fig.savefig(RUN / "planner_pointer_vs_conveyor.png", dpi=150)
plt.close(fig)
print("saved", RUN / "planner_pointer_vs_conveyor.png")

# ---------- extra: when does item 2 move? time course aligned to Go ----------
print("\nGo-aligned (bins rel. Go):", np.arange(-3, 16))
g = {k: [] for k in ["d1_i2", "d2_i2", "d1_i1", "len1_i2", "r2"]}
for r in range(-3, 16):
    Ht = lat2[idx, go + r]
    Hc_t = Ht - Ht.mean(0)
    p1, p2 = Hc_t @ dec[0].coef_.T, Hc_t @ dec[1].coef_.T
    g["d1_i2"].append(np.median(aerr(p1, ang[:, 1])))
    g["d2_i2"].append(np.median(aerr(p2, ang[:, 1])))
    g["d1_i1"].append(np.median(aerr(p1, ang[:, 0])))
    g["len1_i2"].append(np.linalg.norm(p1, axis=1).mean())
    g["r2"].append(r2_score(Y[1], p1))
for k, v in g.items():
    print(f"{k:8s}", np.round(v, 1 if "len" not in k and k != "r2" else 2))
print(
    "item-2 encoding plane during item-1 output vs item-1 pre-Go encoding plane:",
    pa(D2, E1),
    "| norms: item 2 during item-1 output",
    np.linalg.norm(D2).round(2),
    "item 1 during its output",
    np.linalg.norm(D1).round(2),
)
# share of item-2 coding variance (during item-1 output) that lies in
# item 1's old plane / its own old plane / the readout plane
for nm, Pl in (
    ("item-1 pre-Go plane", E1),
    ("item-2 pre-Go plane", E2),
    ("readout plane", Wr[:2]),
):
    Qp = orth(Pl.T)
    print(
        f"  item-2 code during item-1 output: share in {nm}: "
        f"{((D2 @ Qp) ** 2).sum() / (D2**2).sum():.1%}   | "
        f"item-2 code after trigger 1: {((P2 @ Qp) ** 2).sum() / (P2**2).sum():.1%}"
    )
