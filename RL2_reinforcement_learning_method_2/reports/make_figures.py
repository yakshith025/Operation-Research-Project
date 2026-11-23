"""Figures of the RL implementation report (RL_implementation_report.tex).

Reads only (nothing is written outside reports/figures/):
  ../../RL1_reinforcement_learning_method_1/model/train_log.csv        RL1 server training
  ../../RL1_reinforcement_learning_method_1/results/comparison.csv     RL1 vs MILP vs greedy per instance
  ../code/rl2_params.json                                              RL2 training settings (n weighting)

Usage (from RL2_reinforcement_learning_method_2/reports):  python make_figures.py
"""
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RL1 = os.path.join(HERE, "..", "..", "RL1_reinforcement_learning_method_1")
OUT = os.path.join(HERE, "figures")

# reference palette (light surface), validated: blue, orange, aqua
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e4e3df", "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def training_curve():
    rows = list(csv.DictReader(open(os.path.join(RL1, "model", "train_log.csv"), encoding="utf-8")))
    ep = np.array([int(r["epoch"]) for r in rows])
    by_n = [json.loads(r["val_ratio_by_n"]) for r in rows]
    keep = ep >= 1                                    # epoch 0 = untrained (1.43), off the scale
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for n, col in (("20", BLUE), ("50", ORANGE), ("100", AQUA)):
        y = np.array([b[n] for b in by_n])
        ax.plot(ep[keep], y[keep], color=col, lw=1.6, label=f"n = {n}")
        ax.annotate(f"n = {n}", (ep[keep][-1], y[keep][-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK2, fontsize=8)
    ax.axhline(1.0, color=INK2, lw=1.0, ls="--")
    ax.annotate("greedy cost rule = 1", xy=(75, 1.0), xytext=(60, 1.014), color=INK2, fontsize=8,
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.7})
    for x, txt in ((10, "warm-up ends\n(500 batches)"), (153, "best epoch 153"), (210, "early stop 210")):
        ax.axvline(x, color=MUTED, lw=0.8, ls=":")
        ax.text(x + 2, 1.045, txt, color=INK2, fontsize=7.5, va="top")
    ax.set_xlim(0, 228)
    ax.set_ylim(0.92, 1.05)
    ax.set_xlabel("epoch (50 batches each)")
    ax.set_ylabel("validation cost / greedy cost")
    ax.legend(loc="lower left", ncol=3, fontsize=8)
    fig.savefig(os.path.join(OUT, "rl1_training_curve.pdf"))
    plt.close(fig)


def gap_by_n():
    rows = list(csv.DictReader(open(os.path.join(RL1, "results", "comparison.csv"), encoding="utf-8")))
    ns = sorted({int(r["n"]) for r in rows})
    mean, cnt = [], []
    for n in ns:
        g = [float(r["gap_rl_vs_milp_%"]) for r in rows
             if int(r["n"]) == n and r["rl_valid"] == "yes" and r["milp_valid"] == "yes"
             and r["gap_rl_vs_milp_%"] not in ("", "nan")]
        mean.append(float(np.mean(g)))
        cnt.append(len(g))
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    bars = ax.bar(ns, mean, width=6.0, color=BLUE, edgecolor=SURFACE, linewidth=1.0)
    for b, m in zip(bars, mean):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.03, f"{m:.2f}", ha="center", va="bottom",
                color=INK2, fontsize=7.5)
    ax.set_xticks(ns)
    ax.set_xlabel("number of EVs n")
    ax.set_ylabel("mean gap to MILP (%)")
    ax.set_ylim(0, max(mean) * 1.25)
    ax.grid(axis="x", visible=False)
    fig.savefig(os.path.join(OUT, "rl1_gap_by_n.pdf"))
    plt.close(fig)
    return dict(zip(ns, mean)), dict(zip(ns, cnt))


def n_weighting():
    TR = json.load(open(os.path.join(HERE, "..", "code", "rl2_params.json"), encoding="utf-8"))["training"]
    ns = np.arange(int(TR["n_min"]), int(TR["n_max"]) + 1)
    uni = np.ones_like(ns, dtype=float) / len(ns)
    w = ns.astype(float) ** float(TR["n_weight_power"])
    w /= w.sum()
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ax.step(ns, 100 * uni, where="mid", color=BLUE, lw=1.6, label="RL1: uniform")
    ax.step(ns, 100 * w, where="mid", color=ORANGE, lw=1.6, label="RL2: proportional to n")
    ax.annotate("RL1: uniform", (ns[-1], 100 * uni[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", color=INK2, fontsize=8)
    ax.annotate("RL2: proportional to n", (ns[-1], 100 * w[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", color=INK2, fontsize=8)
    ax.set_xlim(ns[0], ns[-1] + 1)
    ax.set_ylim(0, 100 * w.max() * 1.2)
    ax.set_xlabel("number of EVs n of a training batch")
    ax.set_ylabel("probability (%)")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(os.path.join(OUT, "rl2_n_weighting.pdf"))
    plt.close(fig)
    return float((ns * uni).sum()), float((ns * w).sum()), float(uni[ns >= 60].sum()), float(w[ns >= 60].sum())


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    training_curve()
    gaps, counts = gap_by_n()
    print("RL1 mean gap to MILP by n:", {k: round(v, 2) for k, v in gaps.items()}, counts)
    print("mean n uniform / weighted, share n>=60 uniform / weighted:", n_weighting())
    print("figures in", OUT)
