"""Projected run time of the experiments, from the measured times of the baseline runs.

MILP   For every instance of a variant: the baseline instance's measured wall-clock time (server run
       21-22 Nov: 300 s limit, 8 workers x 4 threads, 32 cores), scaled by the model-size ratio
       variant / baseline (binaries + big-M rows of main_formulation_3, counted exactly from the data;
       only the number of vans and of guns change the size), capped at the time limit, plus 2 s to
       start the worker. An instance that hit the limit in the baseline stays at the limit. E3
       (longer horizon): baseline-infeasible instances are counted at the full limit (they may become
       feasible and hard). The instances are then given to W workers in the solver's order (small n
       first) and the finish time of the last one is the projected time.
       Worst case: every instance runs into the limit.
RL1    Inference: the baseline's measured time per instance (RTX 4090), scaled by the number of
       resources (guns + vans), plus 1 min to load torch, the model and the data.
       Retraining (optional method rl1_retrain): the baseline training time (train_meta.json:
       38.7 min for 210 epochs = early stop) scaled the same way; worst case = max_epochs epochs.

Usage (from comparative_results/code)
    python project_timing.py                       # default priorities 1-4, all sizes
    python project_timing.py --all                 # incl. E3
    python project_timing.py --sizes 10 20 30 40 50
    python project_timing.py --all --write         # also writes ../timing_projection.md / .csv
"""
import argparse
import csv
import json
import os
import sys
import time

import common as cm


def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def baseline_times(EXP, project=cm.PROJECT, rl1_folder=None):
    milp_dir = cm.project_path(EXP["baseline"]["milp_results"], project)
    rl1_res = os.path.join(cm.rl1_dir(project, rl1_folder), "results")
    out = {"milp": {}, "rl1": {}, "train_min": None, "train_epochs": None, "max_epochs": None}
    p = os.path.join(milp_dir, "runs.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p, newline="")):
            if r["run_id"] == EXP["baseline"]["milp_run_id"] and r["method"] == "MILP":
                out["milp"][(r["instance"], int(float(r["n"])))] = {
                    "wall": _f(r["wall_clock_s"], 0.0), "build": _f(r["build_s"], 0.0), "status": r["status"]}
    p = os.path.join(rl1_res, "runs.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p, newline="")):
            if r["run_id"] == EXP["baseline"]["rl1_run_id"]:
                out["rl1"][(r["instance"], int(float(r["n"])))] = _f(r["wall_clock_s"], 0.0)
    mdir = os.path.join(cm.rl1_dir(project, rl1_folder), "model")
    try:
        meta = json.load(open(os.path.join(mdir, "train_meta.json")))
        t0 = time.mktime(time.strptime(meta["started"], "%Y-%m-%d %H:%M:%S"))
        t1 = time.mktime(time.strptime(meta["finished"], "%Y-%m-%d %H:%M:%S"))
        out["train_min"] = (t1 - t0) / 60
        out["train_epochs"] = int(meta.get("epoch") or 0) or None
        out["max_epochs"] = int(meta["rl1_params"]["training"]["max_epochs"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return out


def tasks_of(v, EXP, data, sizes=None, only=None):
    """(file, n) pairs of a variant, in the solver's order (small n first, then file name)."""
    src = EXP["instances"]["source"]
    fams = {f.upper() for f in (v.get("families") or [])} if v else set()
    out = []
    for nm in data.names():
        if data.source(nm) != src or (only and nm not in only):
            continue
        if fams and data.family(nm).upper() not in fams:
            continue
        for n in (sizes or EXP["instances"]["sizes"]):
            if 1 <= n <= data.n_evs(nm):
                out.append((nm, n))
    return sorted(out, key=lambda t: (t[1], t[0]))


def model_size(D, M, guns):
    """Binaries and big-M rows of main_formulation_3 for instance dict D with M vans and the given
    guns per station (counted as solve_milp_f3.build_model creates them)."""
    import numpy as np
    n = int(D["n"])
    reach = D["d_ev_fcs"] <= D["R"][:, None]                       # (n, F)
    users = reach.sum(0)
    guns = np.asarray(guns, dtype=int)
    y = int((reach * guns[None, :]).sum())
    share = (reach.astype(np.int32) @ reach.T.astype(np.int32)) > 0
    pairs = int(np.triu(share, 1).sum())
    binaries = y + n * M + (n * n + n) * M + M + pairs
    big_m = int(2 * (guns * users * (users - 1) // 2).sum()) + n * (n - 1) * M + n * M
    return binaries + big_m


def _sizes_for(v, D):
    M0, g0 = int(D["params"]["n_vans"]), D["guns"]
    M, g = M0, g0
    if v and v["kind"] == "config" and v["key"] == "n_vans":
        M = int(v["value"])
    if v and v["kind"] == "guns":
        g = [int(v["value"])] * len(g0)
    return model_size(D, M0, g0), model_size(D, M, g), (int(sum(g0)) + M0), (int(sum(g)) + M)


def makespan(times, workers):
    free = [0.0] * max(1, workers)
    for t in times:
        k = min(range(len(free)), key=free.__getitem__)
        free[k] += t
    return max(free) if times else 0.0


def project(variants, EXP, sizes=None, only=None, workers=None, time_limit=None, project=cm.PROJECT,
            rl1_folder=None, overhead_s=2.0):
    """One dict per variant: tasks, MILP hours (expected, worst), RL1 minutes, retrain minutes."""
    sys.path.insert(0, cm.code_dir(project))
    import make_ev_solomon as mes
    data = mes.EvSolomonData(cm.project_path(EXP["baseline"]["dataset_csv"], project))
    B = baseline_times(EXP, project, rl1_folder)
    if not B["milp"]:
        print("note: the baseline MILP runs.csv was not found, so every instance is counted at the time limit "
              "(expected = worst case)", flush=True)
    workers = workers or EXP["solver"]["milp_workers"]
    TL = float(time_limit or EXP["solver"]["milp_time_limit_s"])
    base_TL = float(EXP["solver"]["milp_time_limit_s"])
    cache = {}
    rows = []
    for v in variants:
        exp_t, worst_t, rl_s, ratio_k = [], [], 0.0, []
        for nm, n in tasks_of(v, EXP, data, sizes, only):
            if (nm, n) not in cache:
                cache[(nm, n)] = data.read(nm, n)
            s0, s1, k0, k1 = _sizes_for(v, cache[(nm, n)])
            r = s1 / s0 if s0 else 1.0
            b = B["milp"].get((nm, n), {"wall": base_TL, "build": 0.0, "status": "unknown"})
            build = b["build"] * r
            solve = max(0.0, b["wall"] - b["build"])
            if b["status"].startswith("time_limit") or solve >= base_TL - 1:
                t = TL
            elif v and v["kind"] == "horizon" and b["status"] == "infeasible":
                t = TL
            else:
                t = min(TL, solve * r)
            exp_t.append(t + build + overhead_s)
            worst_t.append(TL + build + overhead_s)
            rl_s += B["rl1"].get((nm, n), 0.25) * (k1 / k0)
            ratio_k.append(k1 / k0)
        kr = sum(ratio_k) / len(ratio_k) if ratio_k else 1.0
        tr = (B["train_min"] or 40.0) * kr
        tr_worst = tr * (B["max_epochs"] / B["train_epochs"]) if B["train_epochs"] and B["max_epochs"] else tr
        rows.append({"variant": v["id"] if v else cm.BASE_ID, "experiment": v["experiment"] if v else "",
                     "priority": v["priority"] if v else "", "instances": len(exp_t),
                     "milp_h": makespan(exp_t, workers) / 3600, "milp_worst_h": makespan(worst_t, workers) / 3600,
                     "rl1_min": rl_s / 60 + 1.0, "retrain_min": tr + rl_s / 60 + 1.0,
                     "retrain_worst_min": tr_worst + rl_s / 60 + 1.0})
    return rows


def table(rows, methods=("milp", "rl1")):
    """Markdown table of the projection with totals per priority and a running total."""
    show_rt = "rl1_retrain" in methods
    head = ["priority", "variant", "instances", "MILP h (expected)", "MILP h (worst)", "RL1 min"]
    if show_rt:
        head += ["RL1 retrain min (expected)", "RL1 retrain min (worst)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    cum = {"milp": 0.0, "worst": 0.0, "rl": 0.0, "rt": 0.0, "rtw": 0.0}
    by_p = {}
    for r in rows:
        by_p.setdefault(r["priority"], []).append(r)
    for p_, rs in by_p.items():
        for r in rs:
            cells = [str(p_), r["variant"], str(r["instances"]), f"{r['milp_h']:.1f}", f"{r['milp_worst_h']:.1f}",
                     f"{r['rl1_min']:.1f}"]
            if show_rt:
                cells += [f"{r['retrain_min']:.0f}", f"{r['retrain_worst_min']:.0f}"]
            lines.append("| " + " | ".join(cells) + " |")
        s = {k: sum(r[k] for r in rs) for k in ("milp_h", "milp_worst_h", "rl1_min", "retrain_min", "retrain_worst_min")}
        cum["milp"] += s["milp_h"]
        cum["worst"] += s["milp_worst_h"]
        cum["rl"] += s["rl1_min"]
        cum["rt"] += s["retrain_min"]
        cum["rtw"] += s["retrain_worst_min"]
        cells = [f"**{p_}**", f"**priority {p_} total** (running total)", str(sum(r["instances"] for r in rs)),
                 f"**{s['milp_h']:.1f}** ({cum['milp']:.1f})", f"{s['milp_worst_h']:.1f} ({cum['worst']:.1f})",
                 f"{s['rl1_min']:.0f} ({cum['rl']:.0f})"]
        if show_rt:
            cells += [f"{s['retrain_min']:.0f} ({cum['rt']:.0f})", f"{s['retrain_worst_min']:.0f} ({cum['rtw']:.0f})"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines), cum


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--priority", nargs="*", type=int)
    ap.add_argument("--experiments", nargs="*")
    ap.add_argument("--variants", nargs="*")
    ap.add_argument("--all", action="store_true", help="every priority (incl. E3)")
    ap.add_argument("--sizes", nargs="*", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--time-limit", type=float)
    ap.add_argument("--project", default=cm.PROJECT)
    ap.add_argument("--rl1-dir")
    ap.add_argument("--write", action="store_true", help="write ../timing_projection.md and .csv")
    args = ap.parse_args()
    EXP = cm.load_experiments()
    pr = sorted({e["priority"] for e in EXP["experiments"]}) if args.all else args.priority
    sel = cm.select(EXP, pr, args.experiments, args.variants)
    rows = project(sel, EXP, args.sizes, None, args.workers, args.time_limit, args.project, args.rl1_dir)
    md, cum = table(rows, ("milp", "rl1", "rl1_retrain"))
    w = args.workers or EXP["solver"]["milp_workers"]
    head = (f"Projected run time on the lab server ({w} MILP workers x {EXP['solver']['milp_threads']} threads, "
            f"{args.time_limit or EXP['solver']['milp_time_limit_s']:g} s per instance; RL1 on the RTX 4090), "
            f"sizes {args.sizes or 'all (10..100)'}; projected {time.strftime('%Y-%m-%d')}.")
    print(head)
    print(md)
    print(f"\nTotal: MILP {cum['milp']:.1f} h expected ({cum['worst']:.1f} h worst case) + RL1 {cum['rl']:.0f} min "
          f"(+ {cum['rt']:.0f} min if every variant is also retrained)")
    if args.write:
        with open(os.path.join(cm.COMP, "timing_projection.md"), "w") as fh:
            fh.write(head + "\n\n" + md + "\n")
        with open(os.path.join(cm.COMP, "timing_projection.csv"), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows([{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])
        print(f"written {os.path.join(cm.COMP, 'timing_projection.md')}")


if __name__ == "__main__":
    main()
