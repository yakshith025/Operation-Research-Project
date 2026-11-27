"""Before / after: request profile of the original formulation-4 data and of the evening-peak data (../data).

Writes ../results/evening_peak_check_reshaped/before_after_by_class.png (class-mean request density over t/T, n = 100, all
requests including t = 0) and ../results/evening_peak_check_reshaped/before_after_examples.png (one file per class, histogram).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # formulation_4/analysis/evening_peak
REPO = ROOT.parent.parent.parent
F4 = REPO / "formulation_4"
SETS = {"original": F4 / "data", "evening peak": ROOT / "data"}
COL = {"original": "#7f7f7f", "evening peak": "#d62728"}
CLASSES = ["C1", "C2", "R1", "R2", "RC1", "RC2"]
EXAMPLE = {"C1": "EV-C101", "C2": "EV-C204", "R1": "EV-R101", "R2": "EV-R203", "RC1": "EV-RC105",
           "RC2": "EV-RC201"}
OUT = ROOT / "results" / "evening_peak_check_reshaped"


def load(d):
    data = pd.read_csv(d / "solomon_data.csv", low_memory=False)
    cfg = pd.read_csv(d / "solomon_config.csv").set_index("instance")
    ev = data[data.type == "ev"].join(cfg[["class", "T_min"]], on="instance")
    ev["u"] = ev.request_min / ev.T_min
    return ev, cfg


def main():
    OUT.mkdir(exist_ok=True)
    sets = {k: load(v) for k, v in SETS.items()}
    edges = np.linspace(0, 1, 21)
    w = edges[1] - edges[0]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for ax, cls in zip(axes.flat, CLASSES):
        for k, (ev, cfg) in sets.items():
            e = ev[ev["class"] == cls]
            H = np.array([np.histogram(g.u, edges)[0] / len(g) / w for _, g in e.groupby("instance")])
            m = H.mean(0)
            lo, hi = np.percentile(H, [10, 90], axis=0)
            ax.fill_between(edges[:-1], lo, hi, step="post", color=COL[k], alpha=0.15)
            ax.step(edges[:-1], m, where="post", color=COL[k], lw=2, label=f"{k} (class mean)")
        ax.axhline(1, color="tab:blue", ls=":", lw=1)
        ax.axvspan(2 / 3, 1, color="tab:red", alpha=0.06)
        T = sets["original"][1].query("`class` == @cls").T_min.iloc[0]
        ax.set_title(f"{cls}  (T = {T} min)")
        ax.set_ylim(0, 4)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("t / T")
    for ax in axes[:, 0]:
        ax.set_ylabel("request density (1 = uniform)")
    fig.suptitle("Request density per class, n = 100: original vs reshaped (band = 10-90 % of files; "
                 "first bin includes the t = 0 burst, clipped at 4)")
    fig.tight_layout()
    fig.savefig(OUT / "before_after_by_class.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, cls in zip(axes.flat, CLASSES):
        f = EXAMPLE[cls]
        T = sets["original"][1].loc[f, "T_min"]
        be = np.linspace(0, T, 21)
        for k, (ev, cfg) in sets.items():
            ax.hist(ev[ev.instance == f].request_min, be, color=COL[k], alpha=0.55, label=k)
        ax.set_title(f"{f.replace('EV-', '')}  (T = {T} min)")
        ax.set_xlabel("request time (min)")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("requests per bin")
    axes[1, 0].set_ylabel("requests per bin")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Example files, 100 EVs: original vs reshaped request times")
    fig.tight_layout()
    fig.savefig(OUT / "before_after_examples.png", dpi=150)
    plt.close(fig)
    print("written to", OUT)


if __name__ == "__main__":
    main()
