"""Temporal demand profiles of the EV-Solomon instances (56 Solomon files, 100 EVs each).

Reads dataset/csv/solomon_data.csv and solomon_config.csv and writes into ../results/original_profiles/:
  profile_<class>.png      one panel per file: requests per time bin (bars), kWh per bin (line),
                           cumulative share of requests (dashed, right axis)
  overview_heatmap.png     all 56 files: share of requests per tenth of the planning period T
  overview_cumulative.png  cumulative share of requests vs t/T, one panel per class
  summary.csv              per file: T, first/last/mean/median request, share at t=0, peak bin

Usage: python plot_temporal_demand.py [--n 100] [--bins 20]
  --n     use only the first n EVs of each file (instances are file + first n EVs)
  --bins  number of time bins over [0, T]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # formulation_4/analysis/evening_peak
REPO = ROOT.parent.parent.parent
CSV = REPO / "dataset" / "csv"
CLASSES = ["C1", "C2", "R1", "R2", "RC1", "RC2"]
CLASS_COLOR = {"C1": "#1f77b4", "C2": "#6baed6", "R1": "#d62728", "R2": "#fc8d59",
               "RC1": "#2ca02c", "RC2": "#98df8a"}


def load(n):
    data = pd.read_csv(CSV / "solomon_data.csv")
    cfg = pd.read_csv(CSV / "solomon_config.csv").set_index("instance")
    ev = data[(data["type"] == "ev") & (data["id"] <= n)].copy()
    ev = ev.join(cfg[["class", "T_min"]], on="instance")
    ev["t_rel"] = ev["request_min"] / ev["T_min"]
    return ev, cfg


def profile_figure(ev, cfg, cls, bins, n, out):
    files = [i for i in cfg.index if cfg.loc[i, "class"] == cls]
    ncol = 4
    nrow = int(np.ceil(len(files) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.9 * nrow), squeeze=False)
    T = cfg.loc[files[0], "T_min"]
    edges = np.linspace(0, T, bins + 1)
    width = edges[1] - edges[0]
    color = CLASS_COLOR[cls]
    ymax = 0
    for f in files:
        ymax = max(ymax, np.histogram(ev.loc[ev.instance == f, "request_min"], edges)[0].max())
    for ax, f in zip(axes.flat, files):
        e = ev[ev.instance == f]
        cnt, _ = np.histogram(e["request_min"], edges)
        kwh, _ = np.histogram(e["request_min"], edges, weights=e["energy_kWh"])
        ax.bar(edges[:-1], cnt, width=width, align="edge", color=color, alpha=0.75,
               edgecolor="white", linewidth=0.5, label="requests")
        ax.set_ylim(0, ymax * 1.15)
        ax.set_xlim(0, T)
        at0 = (e["request_min"] == 0).mean()
        ax.set_title(f"{f.replace('EV-', '')}   median {e['request_min'].median():.0f} min,"
                     f" at t=0: {at0:.0%}", fontsize=9)
        ax.tick_params(labelsize=8)
        ax2 = ax.twinx()
        t = np.sort(e["request_min"].to_numpy())
        ax2.step(np.r_[0, t, T], np.r_[0, np.arange(1, len(t) + 1) / len(t), 1], where="post",
                 color="black", lw=1, ls="--", label="cumulative share")
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(labelsize=8)
        ax3 = ax.twinx()
        ax3.plot(edges[:-1] + width / 2, kwh, color="#444444", lw=1, marker=".", ms=3,
                 label="kWh requested")
        ax3.set_ylim(0, max(1, kwh.max()) * 1.15)
        ax3.axis("off")
    for ax in axes.flat[len(files):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("requests per bin", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("request time (min)", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.75),
               plt.Line2D([], [], color="#444444", marker=".", lw=1),
               plt.Line2D([], [], color="black", ls="--", lw=1)]
    fig.legend(handles, ["requests per bin (left axis)", "kWh requested per bin (own scale)",
                         "cumulative share of requests (right axis, 0-1)"],
               loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(f"Temporal demand profile, class {cls}: first {n} EVs per file, "
                 f"T = {T} min, {bins} bins of {width:.0f} min", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def overview_heatmap(ev, cfg, n, out, bins=10):
    files = [i for c in CLASSES for i in cfg.index if cfg.loc[i, "class"] == c]
    edges = np.linspace(0, 1, bins + 1)
    M = np.array([np.histogram(ev.loc[ev.instance == f, "t_rel"].clip(upper=1), edges)[0]
                  for f in files], dtype=float)
    M /= M.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 13))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0)
    ax.set_yticks(range(len(files)))
    ax.set_yticklabels([f"{f.replace('EV-', '')}  (T={cfg.loc[f, 'T_min']})" for f in files], fontsize=7)
    ax.set_xticks(range(bins))
    ax.set_xticklabels([f"{a:.1f}-{b:.1f}" for a, b in zip(edges[:-1], edges[1:])], rotation=45,
                       fontsize=8)
    ax.set_xlabel("request time as a fraction of the planning period T")
    prev = None
    for k, f in enumerate(files):
        c = cfg.loc[f, "class"]
        if prev is not None and c != prev:
            ax.axhline(k - 0.5, color="white", lw=2)
        prev = c
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="share of the file's requests")
    ax.set_title(f"Share of requests per tenth of T (first {n} EVs per file)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def overview_cumulative(ev, cfg, n, out):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for ax, cls in zip(axes.flat, CLASSES):
        files = [i for i in cfg.index if cfg.loc[i, "class"] == cls]
        cmap = plt.get_cmap("viridis", len(files))
        for k, f in enumerate(files):
            t = np.sort(ev.loc[ev.instance == f, "t_rel"].to_numpy())
            ax.step(np.r_[0, t, 1], np.r_[0, np.arange(1, len(t) + 1) / len(t), 1], where="post",
                    color=cmap(k), lw=1.2, label=f.replace("EV-", ""))
        ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=1)
        ax.set_title(f"{cls}  (T = {cfg.loc[files[0], 'T_min']} min)")
        ax.legend(fontsize=7, ncol=2, loc="lower right")
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("t / T")
    for ax in axes[:, 0]:
        ax.set_ylabel("cumulative share of requests")
    fig.suptitle(f"Cumulative request profile per file (first {n} EVs); dotted = uniform arrivals")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def summary(ev, cfg, bins):
    rows = []
    for f in cfg.index:
        e = ev[ev.instance == f]
        T = cfg.loc[f, "T_min"]
        cnt, edges = np.histogram(e["request_min"], np.linspace(0, T, bins + 1))
        k = cnt.argmax()
        rows.append({"instance": f, "class": cfg.loc[f, "class"], "T_min": T, "n_evs": len(e),
                     "first_min": e.request_min.min(), "last_min": e.request_min.max(),
                     "mean_min": round(e.request_min.mean(), 1),
                     "median_min": e.request_min.median(),
                     "share_at_t0": round((e.request_min == 0).mean(), 3),
                     "peak_bin_min": f"{edges[k]:.0f}-{edges[k + 1]:.0f}",
                     "peak_bin_requests": int(cnt[k]),
                     "total_kWh": round(e.energy_kWh.sum(), 1)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--bins", type=int, default=20)
    a = ap.parse_args()
    out = ROOT / "results" / ("original_profiles" if a.n == 100 else f"original_profiles_n{a.n}")
    out.mkdir(exist_ok=True)
    ev, cfg = load(a.n)
    for cls in CLASSES:
        profile_figure(ev, cfg, cls, a.bins, a.n, out / f"profile_{cls}.png")
    overview_heatmap(ev, cfg, a.n, out / "overview_heatmap.png")
    overview_cumulative(ev, cfg, a.n, out / "overview_cumulative.png")
    summary(ev, cfg, a.bins).to_csv(out / "summary.csv", index=False)
    print("written to", out)


if __name__ == "__main__":
    main()
