"""Is LIFO's extra recall error explained by holding targets longer?

Joins the per-target errors written by loss_and_recall_accuracy.py
(recall_errors*.csv) with each validation trial's item onsets, Go and Trigger
onsets, read from the noise-free inputs in the saved datamodule. Time held is
from the end of a target's presentation to the start of its output (Go for the
first target released, otherwise the previous trigger). Only the multi-item
tasks are compared: ForwardRecall (FIFO run) vs BackwardRecall (LIFO run).

Plots error vs time held for FIFO and LIFO, fits
log(error) ~ time held * network with standard errors clustered by trial,
and, for LIFO 3-item trials, compares output positions 2 and 3 at matched
time held. Run from anywhere:
    python examples/stack_task_analysis/error_vs_time_held.py
"""

import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "content" / "runs" / "task-trained"
MODELS = ROOT / "content" / "trained_models" / "task-trained"
FIFO_TAG, LIFO_TAG = "20261007_StackTask_test", "20261021_StackTaskLIFO"
LIFO400_TAG = "20261027_StackTaskLIFO_400ep"
RUN_NAMES = {
    FIFO_TAG: "max_epochs=200_batch_size=64_seed=0",
    LIFO_TAG: "max_epochs=200_batch_size=64_seed=0",
    LIFO400_TAG: "max_epochs=400_batch_size=64_seed=0",
}
# (network label, run tag, errors file)
FIFO = ("FIFO epoch 200", FIFO_TAG, "recall_errors.csv")
LIFOS = [
    ("LIFO epoch 200", LIFO_TAG, "recall_errors.csv"),
    ("LIFO epoch 195", LIFO_TAG, "recall_errors_epoch=194-step=9750.csv"),
    ("LIFO epoch 384", LIFO400_TAG, "recall_errors_epoch=383-step=19200.csv"),
]
BIN_MS, ITEM_BINS = 20, 10  # items are shown for 200 ms
MEDIAN_BIN_MS, MIN_PER_BIN = 250, 10
COLORS = {"FIFO": "#2a78d6", "LIFO": "#eb6834"}
MARKERS = {1: "o", 2: "s", 3: "^"}
PROBE_S = [0.5, 1.0, 1.5, 2.0]


def rising_edges(x):
    return np.flatnonzero(np.diff(np.r_[0, (x > 0.5).astype(int)]) > 0)


def trial_events(run_tag):
    """Item onsets, Go and Trigger onsets (bins) of every validation trial."""
    with open(MODELS / run_tag / RUN_NAMES[run_tag] / "datamodule_train.pkl", "rb") as f:
        dm = pickle.load(f)
    tensors = [t.cpu().numpy() for t in dm.valid_ds.tensors]
    extra, true_inputs = tensors[5], tensors[7]
    return [
        dict(
            item_on=rising_edges(np.abs(x[:, :2]).sum(axis=1)),
            go=int(e[0]),
            trigger_on=rising_edges(x[:, 3]),
        )
        for x, e in zip(true_inputs, extra)
    ]


def load(network, run_tag, csv, events):
    df = pd.read_csv(RUNS / run_tag / csv)
    lifo = network.startswith("LIFO")
    hold = []
    for r in df.itertuples():
        ev = events[r.trial]
        n, k = r.n_items, r.item  # k = output position, 1-based
        assert len(ev["item_on"]) == n and len(ev["trigger_on"]) == n, r.trial
        shown = n - k if lifo else k - 1  # presentation index of this target
        out_start = ev["go"] if k == 1 else ev["trigger_on"][k - 2]
        hold.append((out_start - ev["item_on"][shown] - ITEM_BINS) * BIN_MS)
    df["hold_s"] = np.array(hold) / 1000
    df["network"] = network.split()[0]
    df["log_err"] = np.log(df.err)
    df["cluster"] = f"{network}/" + df.trial.astype(str)
    return df[df.task != "SingleRecall"]


def binned_median(d):
    edges = np.arange(0, d.hold_s.max() + 0.25, MEDIAN_BIN_MS / 1000)
    g = d.groupby(pd.cut(d.hold_s, edges), observed=True).err
    stats = g.agg(["median", "count"])
    stats = stats[stats["count"] >= MIN_PER_BIN]
    return np.array([iv.mid for iv in stats.index]), stats["median"].to_numpy()


events = {tag: trial_events(tag) for tag in RUN_NAMES}
fifo = load(*FIFO, events[FIFO_TAG])
lifos = [(lab, load(lab, tag, csv, events[tag])) for lab, tag, csv in LIFOS]

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:.3f}")
print("time held (s) by network and output position")
for lab, d in [(FIFO[0], fifo), *lifos]:
    print(f"\n{lab}")
    print(
        d.groupby(["n_items", "item"]).agg(
            hold_median=("hold_s", "median"),
            hold_min=("hold_s", "min"),
            hold_max=("hold_s", "max"),
            err_median=("err", "median"),
            n=("err", "count"),
        )
    )

# ---------- figure ----------
fig, axes = plt.subplots(1, len(lifos), figsize=(6.5 * len(lifos), 5), sharey=True)
for ax, (lab, lifo) in zip(axes, lifos):
    for name, d in [("FIFO", fifo), ("LIFO", lifo)]:
        for k, mk in MARKERS.items():
            s = d[d["item"] == k]
            ax.scatter(
                s.hold_s,
                s.err,
                s=14,
                marker=mk,
                facecolors="none",
                edgecolors=COLORS[name],
                alpha=0.45,
                lw=0.8,
            )
        x, y = binned_median(d)
        ax.plot(x, y, color=COLORS[name], lw=2.4, label=f"{name}: binned median")
    ax.set_yscale("log")
    ax.set_xlabel("Time held: end of presentation to start of output (s)")
    ax.set_title(f"FIFO epoch 200 vs {lab}", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
axes[0].set_ylabel("Mean angular error during hold (deg, log scale)")
handles, _ = axes[0].get_legend_handles_labels()
handles += [
    plt.Line2D([], [], ls="", marker=mk, mfc="none", mec="#52514e", label=f"output {k}")
    for k, mk in MARKERS.items()
]
axes[0].legend(handles=handles, frameon=False, fontsize=9, loc="upper left")
fig.suptitle(
    "StackTask multi-item recall: error vs time held "
    "(ForwardRecall = FIFO, BackwardRecall = LIFO)",
    fontsize=12,
)
fig.tight_layout()
out_png = RUNS / LIFO_TAG / "error_vs_time_held.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print("\nsaved", out_png)

# ---------- regression: log error ~ time held * network ----------
for lab, lifo in lifos:
    d = pd.concat([fifo, lifo], ignore_index=True)
    fit = smf.ols("log_err ~ hold_s * C(network, Treatment('FIFO'))", d).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d.cluster)[0]}
    )
    print(f"\n===== FIFO epoch 200 vs {lab}: log(err) ~ hold_s * network =====")
    print(fit.summary2().tables[1].iloc[:, [0, 1, 3, 4, 5]])
    lo = max(fifo.hold_s.min(), lifo.hold_s.min())
    hi = min(fifo.hold_s.max(), lifo.hold_s.max())
    print(f"shared range of time held: {lo:.2f}-{hi:.2f} s")
    print("LIFO / FIFO error ratio at equal time held (95% CI, clustered):")
    names = fit.params.index
    for h in PROBE_S:
        if not lo <= h <= hi:
            continue
        c = np.zeros(len(names))
        c[names.str.contains("network") & ~names.str.contains(":")] = 1
        c[names.str.contains(":")] = h
        t = fit.t_test(c)
        est, (a, b) = float(t.effect[0]), t.conf_int()[0]
        print(
            f"  {h:.1f} s: {np.exp(est):.2f}x ({np.exp(a):.2f}-{np.exp(b):.2f}), "
            f"p = {float(t.pvalue):.2g}"
        )

# ---------- LIFO 3-item trials: output position 2 vs 3 at matched hold ----------
for lab, lifo in lifos:
    d = lifo[(lifo.n_items == 3) & lifo["item"].isin([2, 3])]
    p2, p3 = d[d["item"] == 2], d[d["item"] == 3]
    lo, hi = max(p2.hold_s.min(), p3.hold_s.min()), min(p2.hold_s.max(), p3.hold_s.max())
    m = d[d.hold_s.between(lo, hi)]
    fit = smf.ols("log_err ~ hold_s + C(item)", m).fit(
        cov_type="cluster", cov_kwds={"groups": m.trial}
    )
    est = fit.params["C(item)[T.3]"]
    a, b = fit.conf_int().loc["C(item)[T.3]"]
    print(f"\n===== {lab}, 3-item trials: output 3 vs output 2 =====")
    print(
        f"overlap of time held {lo:.2f}-{hi:.2f} s: "
        f"{(m['item'] == 2).sum()} output-2 and {(m['item'] == 3).sum()} output-3 targets"
    )
    print(
        f"output 3 / output 2 error at equal time held: {np.exp(est):.2f}x "
        f"({np.exp(a):.2f}-{np.exp(b):.2f}), p = {fit.pvalues['C(item)[T.3]']:.2g}; "
        f"slope {fit.params['hold_s']:.2f} log-units per s"
    )
    edges = np.arange(np.floor(lo * 4) / 4, hi + 0.25, 0.25)
    print("median error (deg) in 250 ms bins of time held [n]:")
    tab = m.groupby([pd.cut(m.hold_s, edges), "item"], observed=True).err.agg(
        ["median", "count"]
    )
    tab = tab.unstack("item")
    tab = tab[(tab["count"] >= 5).all(axis=1)]
    print(tab)
