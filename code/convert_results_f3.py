"""Convert a formulation-3 results folder of the old layout to the shared schema of results_io.py.

Old layout: results.csv, run_settings.json, report.md, results.xlsx, logs/<instance>.log and one
folder per instance (evs.csv, vans.csv, guns.csv, solution.json, row.json, gurobi.log, error.log).
New layout: runs.csv, plan.csv, meta.json, logs.zip (+ analysis/ and report/ as before).

    python convert_results_f3.py [results_dir] [--run-id ID] [--instances DIR] [--params FILE]
    (default ../new_results; run id milp_<time limit>s from run_settings.json)

Nothing is solved. The stored MILP plans (solution.json) are scored again with the shared checker,
which gives the same Z as before (checked for every instance and printed); the greedy plans are
built again (the greedy rule is deterministic) and written as method greedy. The old files are
not touched; the script lists them at the end so they can be deleted after a look at the tables.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_income as inc        # noqa: E402
import results_io as rio         # noqa: E402
import solve_milp_f3 as smf      # noqa: E402

SOLVER_FIELDS = {"gurobi_obj_Rs": "solver_obj", "best_bound_Rs": "best_bound", "runtime_s": "solve_s",
                 "nodes_explored": "nodes", "binaries": "binaries", "continuous": "continuous",
                 "constraints": "constraints", "build_s": "build_s", "wall_clock_s": "wall_clock_s",
                 "gpu_root_LP_bound": "gpu_root_LP_bound", "gpu_LP_time_s": "gpu_LP_s"}


def new_status(old, has_plan):
    """Old status text -> results_io vocabulary (and the error message, if any)."""
    s = str(old or "unknown").strip()
    if s.startswith("error"):
        return "error", s.split(":", 1)[1].strip() if ":" in s else s
    base = s.split(" (")[0].strip().replace(" ", "_")         # 'time limit (no solution)' -> time_limit
    if not has_plan and base != "infeasible":
        base += "_no_plan"
    return base, ""


def close(a, b):
    try:
        return abs(float(a) - float(b)) <= 0.011
    except (TypeError, ValueError):
        return a in ("", None) and b in ("", None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", nargs="?", default=os.path.join(HERE, "..", "new_results"))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--instances", default=inc.DEFAULT_INSTANCES)
    ap.add_argument("--params", default=inc.DEFAULT_PARAMS)
    args = ap.parse_args()
    out = args.results_dir
    try:
        settings = json.load(open(os.path.join(out, "run_settings.json")))
    except (OSError, ValueError):
        settings = {}
    tl = settings.get("time_limit_s", 300.0)
    run_id = args.run_id or f"milp_{float(tl):g}s"
    warm = "--no-warm-start" not in settings.get("argv", [])
    P3 = inc.load_params(args.params)
    files = {os.path.splitext(os.path.basename(p))[0]: p
             for p in glob.glob(os.path.join(args.instances, "*", "EV-*.txt"))}
    folders = [d for d in glob.glob(os.path.join(out, "EV-*_n*"))
               if os.path.isdir(d) and (os.path.exists(os.path.join(d, "row.json"))
                                        or os.path.exists(os.path.join(d, "solution.json")))]

    def order(d):
        base, n = os.path.basename(d).rsplit("_n", 1)
        return int(n), base
    folders.sort(key=order)
    print(f"{len(folders)} instance folders in {os.path.abspath(out)}; MILP run id {run_id}", flush=True)
    logs = os.path.join(out, "_convert_logs")      # staging folder, packed into logs.zip at the end
    os.makedirs(logs, exist_ok=True)
    runs, plan = [], []
    z_bad, g_bad, skipped = [], [], []
    t0 = time.time()
    for k, d in enumerate(folders, 1):
        name = os.path.basename(d)
        base, n = name.rsplit("_n", 1)
        n = int(n)
        try:
            row = json.load(open(os.path.join(d, "row.json")))
        except (OSError, ValueError):
            row = {}
        sol_path = os.path.join(d, "solution.json")
        sol = json.load(open(sol_path)) if os.path.exists(sol_path) else None
        if not row and sol:
            row = {**sol.get("summary", {}), **sol.get("costs", {})}
        if base not in files:
            skipped.append(f"{name} (no EV-Solomon file {base})")
            continue
        J = smf.load_instance(files[base], n, P3, args.instances, greedy=True)
        keys = smf.run_keys(J, run_id, "MILP", 0)
        status, message = new_status(row.get("status"), sol is not None)
        solver = {new: row[old] for old, new in SOLVER_FIELDS.items() if row.get(old, "") != ""}
        if row.get("gpu_used"):
            message = message or f"GPU: {row['gpu_used']}"
        solver["warm_start"] = (f"greedy ({row['greedy_rule']} rule)" if warm and row.get("greedy_rule")
                                else "none")
        if sol is not None:
            p_ = {c: sol[c] for c in ("resource", "S", "C", "van_routes", "route_van")}
            metrics, rows, _ = rio.evaluate(J, p_)
            Z, bound = metrics["Z_exact"], solver.get("best_bound", "")
            gap = max(0.0, round(100 * (Z - float(bound)) / Z, 3)) if bound != "" and Z else ""
            runs.append(rio.run_row(keys, status, metrics, message=message, **solver, **{"gap_%": gap}))
            plan += rio.with_key(keys, rows)
            old_z = row.get("total_Z", row.get("Z_Rs", ""))
            if not close(old_z, metrics["Z_total"]):
                z_bad.append(f"{name}: old {old_z} new {Z}")
        else:
            runs.append(rio.run_row(keys, status, message=message, **solver))
        g_row, g_plan = smf.greedy_run(J)
        runs.append(g_row)
        plan += g_plan
        if not close(row.get("greedy_Z_Rs", ""), g_row["Z_total"]):
            g_bad.append(f"{name}: old {row.get('greedy_Z_Rs', '')} new {g_row['Z_total']}")
        stem = rio.log_stem(run_id, base, n)
        for src, suffix in ((os.path.join(d, "gurobi.log"), ".gurobi.log"),
                            (os.path.join(d, "error.log"), ".error.log"),
                            (os.path.join(out, "logs", f"{name}.log"), ".log")):
            if os.path.exists(src):
                shutil.copy(src, os.path.join(logs, stem + suffix))
        if k % 50 == 0 or k == len(folders):
            print(f"  {k}/{len(folders)} converted ({time.time() - t0:.0f} s)", flush=True)

    rio.prepare(out)
    rio.replace_runs(out, runs, plan)
    rio.update_meta(out, run_id, {"method": "MILP", **settings,
                                  "converted": time.strftime("%Y-%m-%d %H:%M:%S"),
                                  "converted_from": "old layout (one folder per instance)"})
    rio.update_meta(out, "greedy", {"method": "greedy",
                                    "rule": "cost rule, else earliest-finish rule (solve_milp_f3.greedy_f3)",
                                    "built": time.strftime("%Y-%m-%d %H:%M:%S")})
    old_logs = os.path.join(out, "logs")
    packed = rio.pack_logs(out, src=logs)
    milp = [r for r in runs if r["method"] == "MILP"]
    counts = {}
    for r in milp:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"\nruns.csv: {len(runs)} runs ({len(milp)} MILP, {len(runs) - len(milp)} greedy); "
          f"plan.csv: {len(plan)} rows; logs.zip: {packed} files", flush=True)
    print(f"MILP status: {counts}", flush=True)
    with_plan = sum(1 for r in milp if r["Z_total"] != "")
    print(f"MILP Z equal to the old total_Z: {with_plan - len(z_bad)} of the {with_plan} with a plan"
          + (f"; differ: {z_bad[:5]}" if z_bad else ""))
    print(f"greedy Z equal to the old greedy_Z_Rs: {len(milp) - len(g_bad)} of {len(milp)}"
          + (f"; differ: {g_bad[:5]}" if g_bad else ""))
    if skipped:
        print(f"skipped: {skipped}")
    old = [os.path.basename(f) for f in folders]
    extra = [f for f in ("results.csv", "results.pending.csv", "results.xlsx", "report.md", "run_settings.json")
             if os.path.exists(os.path.join(out, f))]
    print(f"\nOld files (not deleted): {len(old)} instance folders (EV-*_n*/)" +
          (f", {', '.join(extra)}" if extra else "") +
          (", logs/ (copied into logs.zip)" if os.path.isdir(old_logs) else ""))


if __name__ == "__main__":
    main()
