"""Is there an evening peak in the 560 EV-Solomon instances (56 files x first n EVs, n = 10, 20, ..., 100)?

The data has no clock time: requests are minutes in [0, T]. "Evening" is taken as the late part of the planning
period, t/T >= LATE (default 2/3, the last third). For every instance:

  late_share          share of requests with t/T >= LATE (uniform arrivals: 1 - LATE)
  late_ratio          late_share / (1 - LATE)                 > 1 means more late demand than uniform
  late_vs_early       request rate (per unit t/T) in the late window / rate in [0, LATE), requests at t = 0
                      excluded (they are a start-of-period burst, not a time-of-day profile)   > 1 = evening peak
  late_kwh_vs_early   the same with kWh instead of request counts
  peak_t_rel          centre of the busiest sliding window (width WIN of T), t = 0 requests excluded
  peak_in_late        peak_t_rel >= LATE
  last_rel            latest request / T
  evening_peak        late_vs_early > 1 and peak_in_late

Writes ../results/evening_peak_check_original/instances.csv (560 rows), evening_peak/by_class_n.csv, evening_peak/summary.md and figures.
Usage: python evening_peak_check.py [--late 0.6667] [--win 0.1] [--data DIR --out DIR]
  reshaped data: --data ../data --out ../results/evening_peak_check_reshaped
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
SIZES = range(10, 101, 10)


def busiest_window(t, win):
    """Centre (in t/T) of the window of width win holding most requests; ties -> earliest."""
    if len(t) == 0:
        return np.nan, 0
    best, centre = -1, np.nan
    for s in np.sort(t):
        k = int(np.sum((t >= s) & (t < s + win)))
        if k > best:
            best, centre = k, min(s + win / 2, 1.0)
    return centre, best


def metrics(e, T, late, win):
    t = e["request_min"].to_numpy() / T
    kwh = e["energy_kWh"].to_numpy()
    pos = t > 0
    late_m = t >= late
    early_m = pos & (t < late)
    w_early = late - 0.0  # width of [0, late) in t/T
    w_late = 1 - late
    rate_late = late_m.sum() / w_late
    rate_early = early_m.sum() / w_early
    kwh_late = kwh[late_m].sum() / w_late
    kwh_early = kwh[early_m].sum() / w_early
    pk, pk_n = busiest_window(t[pos], win)
    r = {
        "n_requests": len(t),
        "share_at_t0": round(float((~pos).mean()), 3),
        "late_share": round(float(late_m.mean()), 3),
        "late_ratio": round(float(late_m.mean() / w_late), 3),
        "late_vs_early": round(rate_late / rate_early, 3) if rate_early > 0 else np.nan,
        "late_kwh_vs_early": round(kwh_late / kwh_early, 3) if kwh_early > 0 else np.nan,
        "peak_t_rel": round(float(pk), 3),
        "peak_window_requests": pk_n,
        "last_rel": round(float(t.max()), 3),
    }
    r["peak_in_late"] = bool(r["peak_t_rel"] >= late)
    r["evening_peak"] = bool(r["late_vs_early"] > 1 and r["peak_in_late"])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--late", type=float, default=2 / 3)
    ap.add_argument("--win", type=float, default=0.1)
    ap.add_argument("--data", default=str(CSV), help="folder with solomon_data.csv / solomon_config.csv")
    ap.add_argument("--out", default=None, help="output folder (default evening_peak[_late...])")
    a = ap.parse_args()
    out = ROOT / "results" / ("evening_peak_check_original" if abs(a.late - 2 / 3) < 1e-3
                              else f"evening_peak_check_original_late{a.late:g}")
    if a.out:
        out = Path(a.out)
    out.mkdir(exist_ok=True)

    src = Path(a.data)
    data = pd.read_csv(src / "solomon_data.csv")
    cfg = pd.read_csv(src / "solomon_config.csv").set_index("instance")
    ev = data[data["type"] == "ev"]

    rows = []
    for f in cfg.index:
        T = cfg.loc[f, "T_min"]
        ef = ev[ev.instance == f]
        for n in SIZES:
            r = {"file": f.replace("EV-", ""), "class": cfg.loc[f, "class"], "n": n, "T_min": T}
            r.update(metrics(ef[ef["id"] <= n], T, a.late, a.win))
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(out / "instances.csv", index=False)

    g = df.groupby(["class", "n"]).agg(
        instances=("file", "size"),
        evening_peak=("evening_peak", "sum"),
        peak_in_late=("peak_in_late", "sum"),
        mean_late_share=("late_share", "mean"),
        mean_late_vs_early=("late_vs_early", "mean"),
        max_late_vs_early=("late_vs_early", "max"),
        mean_last_rel=("last_rel", "mean")).round(3)
    g.to_csv(out / "by_class_n.csv")

    # --- figures -------------------------------------------------------------------------------------------
    files = [f.replace("EV-", "") for c in CLASSES for f in cfg.index if cfg.loc[f, "class"] == c]
    M = df.pivot(index="file", columns="n", values="late_vs_early").loc[files]
    fig, ax = plt.subplots(figsize=(7.5, 13))
    vmax = max(2.0, np.nanmax(M.values))
    im = ax.imshow(M.values, aspect="auto", cmap="RdBu_r",
                   norm=matplotlib.colors.TwoSlopeNorm(vcenter=1, vmin=0, vmax=vmax))
    ax.set_yticks(range(len(files)))
    ax.set_yticklabels(files, fontsize=7)
    ax.set_xticks(range(len(SIZES)))
    ax.set_xticklabels(list(SIZES))
    ax.set_xlabel("instance size n (first n EVs of the file)")
    for i, f in enumerate(files):
        for j, n in enumerate(SIZES):
            v = M.iloc[i, j]
            ev_pk = df[(df.file == f) & (df.n == n)].evening_peak.iloc[0]
            ax.text(j, i, f"{v:.2f}" + ("*" if ev_pk else ""), ha="center", va="center", fontsize=5.5,
                    color="black")
    prev = None
    for k, f in enumerate(files):
        c = df[df.file == f]["class"].iloc[0]
        if prev is not None and c != prev:
            ax.axhline(k - 0.5, color="black", lw=1.5)
        prev = c
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                 label=f"request rate in last {1 - a.late:.0%} of T / rate before (t=0 excluded)")
    ax.set_title("Evening-to-earlier demand ratio, all 560 instances\n(red > 1 = more late demand; "
                 "* = busiest window also late)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "late_vs_early_heatmap.png", dpi=150)
    plt.close(fig)

    # mean intensity profile per class (n = 100), t = 0 burst shown separately
    edges = np.linspace(0, 1, 21)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for ax, cls in zip(axes.flat, CLASSES):
        fl = [f for f in cfg.index if cfg.loc[f, "class"] == cls]
        H = []
        for f in fl:
            t = ev.loc[ev.instance == f, "request_min"].to_numpy() / cfg.loc[f, "T_min"]
            H.append(np.histogram(t[t > 0], edges)[0] / len(t) / (edges[1] - edges[0]))
        H = np.array(H)
        for h in H:
            ax.step(edges[:-1], h, where="post", color="grey", alpha=0.35, lw=0.8)
        ax.step(edges[:-1], H.mean(0), where="post", color="black", lw=2, label="class mean")
        ax.axhline(1, color="tab:blue", ls=":", lw=1, label="uniform arrivals")
        ax.axvspan(a.late, 1, color="tab:red", alpha=0.08, label="evening window")
        ax.set_title(f"{cls}  (T = {cfg.loc[fl[0], 'T_min']} min)")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("t / T")
    for ax in axes[:, 0]:
        ax.set_ylabel("request density (t > 0)")
    fig.suptitle("Request density over the planning period, n = 100 (grey: files, black: class mean)")
    fig.tight_layout()
    fig.savefig(out / "density_by_class.png", dpi=150)
    plt.close(fig)

    # --- summary -------------------------------------------------------------------------------------------
    tot = len(df)
    lines = [
        "# Evening-peak check over the 560 EV-Solomon instances", "",
        f"Evening window: t/T >= {a.late:.3f}; busiest-window width {a.win:.0%} of T; t = 0 requests excluded "
        "from rates and peak location.", "",
        f"- instances with an evening peak (late rate > earlier rate AND busiest window late): "
        f"**{int(df.evening_peak.sum())} / {tot}**",
        f"- late rate > earlier rate: {int((df.late_vs_early > 1).sum())} / {tot}",
        f"- busiest window in the evening: {int(df.peak_in_late.sum())} / {tot}",
        f"- late_vs_early: median {df.late_vs_early.median():.2f}, max {df.late_vs_early.max():.2f}",
        f"- share of requests in the evening window: mean {df.late_share.mean():.3f} "
        f"(uniform: {1 - a.late:.3f})",
        f"- requests with t/T > 0.9: {int((df.last_rel > 0.9).sum())} instances have any",
        "", "## Per class and size", "", g.reset_index().to_markdown(index=False), "",
        "## Instances flagged", "",
        (df[df.evening_peak][["file", "class", "n", "late_vs_early", "late_share", "peak_t_rel"]]
         .to_markdown(index=False) if df.evening_peak.any() else "none"), "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
