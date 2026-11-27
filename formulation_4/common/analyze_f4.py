"""Segment x income insights of one formulation-4 run (any method: MILP, greedy, RL).

Reads the shared result tables of a results folder (results_io_f4): the runs of one run_id in runs.csv and
their EV rows in plan.csv (which carry segment, theta and stranded). Only valid plans are used. Shares of
"by van" are over FREE-CHOICE EVs (a station within range); stranded EVs must take a van.

Usage:  python analyze_f4.py RESULTS_DIR [RUN_ID]      (default: the MILP run, else the first run id)
The solvers call run() at the end of every run.

Writes RESULTS_DIR/analysis/<run_id>/:
  status_counts.csv          runs per status
  by_segment.csv             per segment: EVs, share by van, queue, van wait, downtime, money paid, perceived cost
  by_segment_income.csv      the same per segment x income band (theta in fifths)
  by_segment_class.csv       the same per segment x Solomon class (C1 ... RC2)
  by_segment_size.csv        the same per segment x instance size n
  threshold.csv              logistic regression P(van | free choice) on theta, per segment, and the income
                             threshold theta* where the van becomes the likelier choice
  summary.md                 headline numbers and every table in Markdown
  fig_van_share_by_income.png, fig_cost_by_income.png   (if matplotlib is installed)
"""
import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import results_io_f4 as rio  # noqa: E402

BANDS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
BAND_NAMES = [f"{lo:.1f}-{hi:.1f}" for lo, hi in BANDS]
SEGS = ("personal", "commercial")


def band_of(theta):
    return BAND_NAMES[max(0, min(int(theta * 5 + 1e-9), 4))]


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def pick_run(results_dir, run_id=None):
    runs = rio.read_runs(results_dir)
    ids = []
    for r in runs:
        if r["run_id"] not in ids:
            ids.append(r["run_id"])
    if run_id:
        return run_id, runs
    milp = [r["run_id"] for r in runs if r["method"] == "MILP"]
    return (milp[0] if milp else (ids[0] if ids else None)), runs


def load(results_dir, run_id=None):
    run_id, runs = pick_run(results_dir, run_id)
    mine = [r for r in runs if r["run_id"] == run_id]
    valid = {rio.key_of(r): r for r in mine if r.get("valid") == "yes"}
    evs = []
    for row in rio.iter_plan(results_dir, set(valid), types={"ev"}):
        run = valid[rio.key_of(row)]
        e = dict(row)
        e["class"], e["n"] = run.get("class", ""), int(float(run["n"]))
        for k in ("theta", "queue_min", "van_wait_min", "drive_min", "drive_km", "charge_min", "downtime_min",
                  "time_to_charged_min", "pay_Rs", "perceived_money_Rs", "time_cost_Rs", "customer_cost_Rs"):
            e[k] = fnum(e.get(k))
        e["van"] = e.get("resource_type") == "MCS"
        e["free"] = e.get("stranded") == "no"
        evs.append(e)
    return run_id, mine, evs


def stats(group):
    free = [e for e in group if e["free"]]
    fcs = [e for e in group if not e["van"]]
    van = [e for e in group if e["van"]]

    def m(vals, d=2):
        vals = [v for v in vals if v == v]
        return round(float(np.mean(vals)), d) if vals else ""
    return {"EVs": len(group), "free_choice": len(free),
            "share_van_free_choice_%": round(100 * sum(e["van"] for e in free) / len(free), 1) if free else "",
            "mean_theta": m([e["theta"] for e in group], 3),
            "mean_queue_FCS_min": m([e["queue_min"] for e in fcs]),
            "mean_van_wait_min": m([e["van_wait_min"] for e in van]),
            "mean_drive_min": m([e["drive_min"] for e in fcs]),
            "mean_downtime_min": m([e["downtime_min"] for e in group]),
            "mean_pay_Rs": m([e["pay_Rs"] for e in group]),
            "mean_perceived_money_Rs": m([e["perceived_money_Rs"] for e in group]),
            "mean_time_cost_Rs": m([e["time_cost_Rs"] for e in group]),
            "mean_customer_cost_Rs": m([e["customer_cost_Rs"] for e in group])}


def table(evs, key, order=None):
    groups = {}
    for e in evs:
        groups.setdefault((e["segment"], key(e)), []).append(e)
    keys = sorted(groups, key=lambda k: (SEGS.index(k[0]) if k[0] in SEGS else 9,
                                         order.index(k[1]) if order and k[1] in order else str(k[1])))
    return [{"segment": s, "group": g, **stats(groups[(s, g)])} for s, g in keys]


def logit(x, y, ridge=1e-6, iters=50):
    """P(y=1) = 1 / (1 + exp(-(b0 + b1 x))) by Newton's method. Returns (b0, b1) or None."""
    if len(x) < 10 or y.min() == y.max():
        return None
    X = np.column_stack([np.ones_like(x), x])
    b = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        g = X.T @ (y - p) - ridge * b
        H = X.T @ (X * (p * (1 - p))[:, None]) + ridge * np.eye(2)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return None
        b = b + step
        if np.abs(step).max() < 1e-9:
            break
    return float(b[0]), float(b[1])


def write_rows(path, rows):
    if not rows:
        return
    cols = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    try:
        with open(path, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=cols)
            wr.writeheader()
            wr.writerows(rows)
    except OSError as e:
        print(f"WARNING: {path} not written ({e})", flush=True)


def md(rows):
    if not rows:
        return "(no data)\n"
    cols = list(rows[0])
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        out += "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n"
    return out


def figures(outdir, evs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    colors = {"personal": "#2a6fb0", "commercial": "#c4612b"}
    mids = [0.5 * (lo + hi) for lo, hi in BANDS]
    for name, val, ylab, only_free in (("fig_van_share_by_income.png", "share_van_free_choice_%",
                                        "share of free-choice EVs served by a van (%)", True),
                                       ("fig_cost_by_income.png", "mean_customer_cost_Rs",
                                        "mean perceived customer cost (Rs)", False)):
        fig, ax = plt.subplots(figsize=(6.0, 3.4))
        for s in SEGS:
            ys = []
            for b in BAND_NAMES:
                grp = [e for e in evs if e["segment"] == s and band_of(e["theta"]) == b]
                st = stats(grp)
                ys.append(st[val] if st[val] != "" and len(grp) >= 5 else float("nan"))
            ax.plot(mids, ys, marker="o", color=colors[s], label=s)
        ax.set_xlabel("income index theta (band mid-point)")
        ax.set_ylabel(ylab)
        ax.set_xlim(0, 1)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, name), dpi=150)
        plt.close(fig)


def run(results_dir, run_id=None):
    run_id, runs, evs = load(results_dir, run_id)
    if run_id is None:
        print("analysis: no runs found", flush=True)
        return None
    outdir = os.path.join(results_dir, "analysis", run_id)
    os.makedirs(outdir, exist_ok=True)
    counts = {}
    for r in runs:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    write_rows(os.path.join(outdir, "status_counts.csv"), [{"status": k, "runs": v} for k, v in sorted(counts.items())])
    seg_rows = [{"segment": s, **stats([e for e in evs if e["segment"] == s])} for s in SEGS]
    inc_rows = table(evs, lambda e: band_of(e["theta"]), BAND_NAMES)
    cls_rows = table(evs, lambda e: e["class"])
    size_rows = table(evs, lambda e: e["n"])
    thr = []
    for s in SEGS:
        free = [e for e in evs if e["segment"] == s and e["free"]]
        fit = logit(np.array([e["theta"] for e in free]), np.array([1.0 if e["van"] else 0.0 for e in free]))
        if fit is None:
            thr.append({"segment": s, "free_choice_EVs": len(free), "b0": "", "b1_theta": "", "theta_star": "",
                        "note": "too few EVs or no variation"})
            continue
        b0, b1 = fit
        ts = -b0 / b1 if abs(b1) > 1e-12 else float("nan")
        note = ("van likelier above theta*" if b1 > 0 else "van likelier below theta*") if 0 <= ts <= 1 else \
            ("van never the likelier choice in [0, 1]" if b0 + max(b1, 0) < 0 else "van likelier at every income")
        thr.append({"segment": s, "free_choice_EVs": len(free), "b0": round(b0, 4), "b1_theta": round(b1, 4),
                    "theta_star": round(ts, 3) if ts == ts else "", "note": note})
    write_rows(os.path.join(outdir, "by_segment.csv"), seg_rows)
    write_rows(os.path.join(outdir, "by_segment_income.csv"), inc_rows)
    write_rows(os.path.join(outdir, "by_segment_class.csv"), cls_rows)
    write_rows(os.path.join(outdir, "by_segment_size.csv"), size_rows)
    write_rows(os.path.join(outdir, "threshold.csv"), thr)
    figures(outdir, evs)
    valid = [r for r in runs if r.get("valid") == "yes"]
    Z = [fnum(r["Z_total"]) for r in valid]
    key = {"run_id": run_id, "runs": len(runs), "valid_plans": len(valid), "status_counts": counts,
           "EVs_in_valid_plans": len(evs), "mean_Z_Rs": round(float(np.mean(Z)), 2) if Z else ""}
    for row in seg_rows:
        key[f"share_van_{row['segment']}_%"] = row["share_van_free_choice_%"]
        key[f"mean_customer_cost_{row['segment']}_Rs"] = row["mean_customer_cost_Rs"]
    with open(os.path.join(outdir, "key_numbers.json"), "w") as fh:
        json.dump(key, fh, indent=1)
    text = (f"# Segment x income analysis of run `{run_id}`\n\n"
            f"Folder `{os.path.abspath(results_dir)}`. {len(runs)} runs, {len(valid)} valid plans, "
            f"{len(evs)} EVs in valid plans. Status: {counts}.\n\n"
            "Shares by van are over free-choice EVs (a station within range).\n\n"
            "## Per segment\n\n" + md(seg_rows) +
            "\n## Income threshold (logistic regression on theta, free-choice EVs)\n\n" + md(thr) +
            "\n## Per segment and income band\n\n" + md(inc_rows) +
            "\n## Per segment and Solomon class\n\n" + md(cls_rows) +
            "\n## Per segment and instance size\n\n" + md(size_rows))
    with open(os.path.join(outdir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"analysis of {run_id} written to {os.path.abspath(outdir)}", flush=True)
    return key


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
