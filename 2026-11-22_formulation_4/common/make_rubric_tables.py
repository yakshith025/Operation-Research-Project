"""Build the rubric-aligned master tables for formulation 4 on BOTH datasets (Solomon + Gehring-Homberger).

Reads only the result tables written by the runs (runs.csv, the rich per-instance schema of results_io_f4),
splits everything by the 'source' column (solomon / gehring_homberger), and writes CSVs + a summary.md that
map one-to-one to the course rubric:

  Experimental Evaluation (solution quality + computational performance/speed)
    solution_quality.csv        per (dataset, method, n): valid, optimal, mean Z, mean/median gap to optimum
    solution_quality_overall.csv  per (dataset, method): the same, pooled over n
    performance.csv             per (dataset, method, n): mean/median/max solve seconds, RL speed-up, RL training s
    milp_vs_rl.csv              per dataset: head-to-head on paired instances (RL gap, where RL alone found a plan)

  Analysis (how the factors change the optimal solution; where mobile chargers help)
    sensitivity_summary.csv     every experiment x variant x dataset x run: paired dZ, operator/customer dZ,
                                van share and queue (baseline -> variant)
    sensitivity_mobile_chargers.csv   E2 slice  (rubric: number of mobile chargers)
    sensitivity_station_capacity.csv  E9 slice  (rubric: station capacities)
    sensitivity_charging_demand.csv   E10 slice (rubric: charging demand, per-EV energy)
    charging_demand_by_n.csv    the demand-LEVEL curve: per (dataset, method, n) mean Z, van share, queue, time
    mobile_charger_impact.csv   per (dataset, class, n): stranded EVs, van share, and the fleet-size feasibility
                                gain from E2 -- i.e. where mobile chargers significantly improve the network

Pure standard library (csv/json/os/glob/statistics); robust to missing / partial results (a table that has no
data is still written with its header and noted in summary.md). Run it any time, also on the laptop after
copying results back.

Usage (from common/, or anywhere):
  python make_rubric_tables.py                     # -> ../rubric_results/
  python make_rubric_tables.py --out <folder>
  python make_rubric_tables.py --root <formulation-4 folder> --out <folder>
"""
import argparse
import csv
import glob
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
F4_DIR = os.path.dirname(HERE)

MILP_RUN = "milp_300s"
RL_RUN = "rl_f4"
RL_ZERO = "rl_f4_zero_shot"
RL_RETRAIN = "rl_f4_retrained"
SOURCES = ["solomon", "gehring_homberger"]
SRC_SHORT = {"solomon": "Solomon", "gehring_homberger": "GH"}

# method label shown in the tables
METHOD_LABEL = {MILP_RUN: "MILP", RL_RUN: "RL", "greedy": "greedy",
                RL_ZERO: "RL zero-shot", RL_RETRAIN: "RL retrained"}


# --------------------------------------------------------------------------- reading
def fnum(x):
    if x in (None, "", "NA", "nan"):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def read_runs(path):
    """All rows of a runs.csv as dicts (or [] if the file is absent)."""
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def is_valid(r):
    return str(r.get("valid", "")).lower() == "yes"


def key_in(r):
    return (r.get("source", ""), r.get("instance", ""), int(float(r.get("n", 0) or 0)))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(st.mean(xs), 3) if xs else ""


def median(xs):
    xs = [x for x in xs if x is not None]
    return round(st.median(xs), 3) if xs else ""


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        wr.writerows(rows)
    return len(rows)


# --------------------------------------------------------------------------- gather
def load_all(root):
    """Return {('baseline', run_id): rows} and {(exp_id, variant_id, run_id): rows}."""
    cfg = json.load(open(os.path.join(HERE if os.path.exists(os.path.join(HERE, "experiments.json"))
                                      else os.path.join(root, "common"), "experiments.json"), encoding="utf-8"))
    base = {}
    for run in (MILP_RUN, "greedy"):
        base[("baseline", run)] = [r for r in read_runs(os.path.join(root, "baseline_MILP", "results", "runs.csv"))
                                   if r.get("run_id") == run]
    base[("baseline", RL_RUN)] = [r for r in read_runs(os.path.join(root, "baseline_RL", "results", "runs.csv"))
                                  if r.get("run_id") == RL_RUN]
    exp_rows = {}
    for exp in cfg["experiments"]:
        for method_dir, runs in (("_RL", (RL_ZERO, RL_RETRAIN)), ("_MILP", (MILP_RUN,))):
            folder = os.path.join(root, exp["folder"] + method_dir, "results")
            for var in exp["variants"]:
                runs_csv = os.path.join(folder, var["id"], "runs.csv")
                rows = read_runs(runs_csv)
                for run in runs:
                    sel = [r for r in rows if r.get("run_id") == run]
                    if sel:
                        exp_rows[(exp["id"], var["id"], run)] = sel
    return cfg, base, exp_rows


# --------------------------------------------------------------------------- quality + performance
def by_source_n(rows):
    g = {}
    for r in rows:
        g.setdefault((r.get("source", ""), int(float(r.get("n", 0) or 0))), []).append(r)
    return g


def solution_quality(base, out):
    header = ["dataset", "method", "n", "instances", "valid", "optimal", "mean_Z", "mean_gap_%", "median_gap_%"]
    rows, overall = [], []
    runs = [(MILP_RUN, base[("baseline", MILP_RUN)]), (RL_RUN, base[("baseline", RL_RUN)]),
            ("greedy", base[("baseline", "greedy")])]
    for run_id, data in runs:
        if not data:
            continue
        for (src, n), g in sorted(by_source_n(data).items()):
            valid = [r for r in g if is_valid(r)]
            rows.append([SRC_SHORT.get(src, src), METHOD_LABEL.get(run_id, run_id), n, len(g), len(valid),
                         sum(1 for r in g if r.get("status") == "optimal"),
                         mean([fnum(r.get("Z_total")) for r in valid]),
                         mean([fnum(r.get("gap_%")) for r in valid]),
                         median([fnum(r.get("gap_%")) for r in valid])])
        for src in SOURCES:
            g = [r for r in data if r.get("source") == src]
            if not g:
                continue
            valid = [r for r in g if is_valid(r)]
            overall.append([SRC_SHORT.get(src, src), METHOD_LABEL.get(run_id, run_id), "all", len(g), len(valid),
                            sum(1 for r in g if r.get("status") == "optimal"),
                            mean([fnum(r.get("Z_total")) for r in valid]),
                            mean([fnum(r.get("gap_%")) for r in valid]),
                            median([fnum(r.get("gap_%")) for r in valid])])
    write_csv(os.path.join(out, "solution_quality.csv"), header, rows)
    write_csv(os.path.join(out, "solution_quality_overall.csv"), header, overall)
    return rows, overall


def performance(base, out):
    header = ["dataset", "method", "n", "instances_scored", "mean_solve_s", "median_solve_s", "max_solve_s",
              "mean_train_s"]
    rows = []
    for run_id in (MILP_RUN, RL_RUN):
        data = base[("baseline", run_id)]
        for (src, n), g in sorted(by_source_n(data).items()):
            solve = [fnum(r.get("solve_s")) for r in g]
            rows.append([SRC_SHORT.get(src, src), METHOD_LABEL.get(run_id, run_id), n, len(g),
                         mean(solve), median(solve), round(max([s for s in solve if s is not None], default=0), 2),
                         mean([fnum(r.get("train_s")) for r in g])])
    write_csv(os.path.join(out, "performance.csv"), header, rows)
    return rows


def milp_vs_rl(base, out):
    header = ["dataset", "milp_valid", "rl_valid", "paired_valid", "rl_only_valid", "milp_only_valid",
              "mean_rl_minus_milp_Z_%", "median_rl_minus_milp_Z_%", "mean_milp_solve_s", "mean_rl_solve_s",
              "rl_speedup_x"]
    rows = []
    milp = {key_in(r): r for r in base[("baseline", MILP_RUN)]}
    rl = {key_in(r): r for r in base[("baseline", RL_RUN)]}
    for src in SOURCES:
        m = {k: v for k, v in milp.items() if k[0] == src}
        r = {k: v for k, v in rl.items() if k[0] == src}
        if not m and not r:
            continue
        mv = {k for k, v in m.items() if is_valid(v)}
        rv = {k for k, v in r.items() if is_valid(v)}
        paired = mv & rv
        gaps = [100 * (fnum(r[k]["Z_total"]) - fnum(m[k]["Z_total"])) / fnum(m[k]["Z_total"])
                for k in paired if fnum(m[k].get("Z_total"))]
        ms = [fnum(v.get("solve_s")) for v in m.values()]
        rs = [fnum(v.get("solve_s")) for v in r.values()]
        mm, rm = mean(ms), mean(rs)
        speed = round(mm / rm, 1) if isinstance(mm, float) and isinstance(rm, float) and rm else ""
        rows.append([SRC_SHORT.get(src, src), len(mv), len(rv), len(paired), len(rv - mv), len(mv - rv),
                     mean(gaps), median(gaps), mm, rm, speed])
    write_csv(os.path.join(out, "milp_vs_rl.csv"), header, rows)
    return rows


# --------------------------------------------------------------------------- sensitivities
VAN_COL = "share_free_choice_by_van_%"
QUEUE_COL = "mean_queue_FCS_min"


def paired_change(var_rows, base_rows):
    """Mean % change of Z / operator / customer on instances valid in both (per the caller's source filter),
    and the pooled van share and queue of the variant vs the baseline."""
    b = {key_in(r): r for r in base_rows if is_valid(r)}
    v = {key_in(r): r for r in var_rows if is_valid(r)}
    paired = [k for k in v if k in b]

    def dpct(col):
        xs = []
        for k in paired:
            bb, vv = fnum(b[k].get(col)), fnum(v[k].get(col))
            if bb not in (None, 0) and vv is not None:
                xs.append(100 * (vv - bb) / bb)
        return mean(xs)

    out = {"paired": len(paired), "dZ_%": dpct("Z_total"),
           "d_operator_%": dpct("operator_cost_Z2_to_Z4"), "d_customer_%": dpct("customer_cost_Z1_Z5_Z6"),
           "van_share_base_%": mean([fnum(b[k].get(VAN_COL)) for k in paired]),
           "van_share_var_%": mean([fnum(v[k].get(VAN_COL)) for k in paired]),
           "queue_base_min": mean([fnum(b[k].get(QUEUE_COL)) for k in paired]),
           "queue_var_min": mean([fnum(v[k].get(QUEUE_COL)) for k in paired]),
           "valid_var": len(v), "valid_base_paired_src": len(b),
           "newly_valid": len(set(v) - set(b)), "lost_valid": len(set(b) - set(v))}
    return out


def sensitivity(cfg, base, exp_rows, out):
    header = ["experiment", "factor", "variant", "label", "dataset", "run", "paired", "dZ_%", "d_operator_%",
              "d_customer_%", "van_share_base_%", "van_share_var_%", "queue_base_min", "queue_var_min",
              "newly_valid", "lost_valid"]
    allrows = []
    base_rl = base[("baseline", RL_RUN)]
    base_milp = base[("baseline", MILP_RUN)]
    for exp in cfg["experiments"]:
        for var in exp["variants"]:
            for run in (RL_ZERO, RL_RETRAIN, MILP_RUN):
                rows = exp_rows.get((exp["id"], var["id"], run))
                if not rows:
                    continue
                base_rows = base_milp if run == MILP_RUN else base_rl
                for src in SOURCES:
                    vr = [r for r in rows if r.get("source") == src]
                    br = [r for r in base_rows if r.get("source") == src]
                    if not vr or not br:
                        continue
                    c = paired_change(vr, br)
                    if not c["paired"] and not c["newly_valid"] and not c["lost_valid"]:
                        continue
                    allrows.append([exp["id"], var.get("factor", ""), var["id"], var.get("label", ""),
                                    SRC_SHORT.get(src, src), METHOD_LABEL.get(run, run), c["paired"], c["dZ_%"],
                                    c["d_operator_%"], c["d_customer_%"], c["van_share_base_%"],
                                    c["van_share_var_%"], c["queue_base_min"], c["queue_var_min"],
                                    c["newly_valid"], c["lost_valid"]])
    write_csv(os.path.join(out, "sensitivity_summary.csv"), header, allrows)
    # per-factor rubric slices
    slices = {"E2": "sensitivity_mobile_chargers.csv", "E9": "sensitivity_station_capacity.csv",
              "E10": "sensitivity_charging_demand.csv"}
    for exp_id, fname in slices.items():
        write_csv(os.path.join(out, fname), header, [r for r in allrows if r[0] == exp_id])
    return allrows


# --------------------------------------------------------------------------- demand level + van impact
def charging_demand_by_n(base, out):
    """Demand LEVEL = number of EVs: how the baseline solution moves with n (both methods)."""
    header = ["dataset", "method", "n", "valid", "mean_Z", "mean_van_share_%", "mean_queue_FCS_min",
              "mean_time_to_charged_min", "mean_vans_used"]
    rows = []
    for run_id in (MILP_RUN, RL_RUN):
        for (src, n), g in sorted(by_source_n(base[("baseline", run_id)]).items()):
            valid = [r for r in g if is_valid(r)]
            if not valid:
                continue
            rows.append([SRC_SHORT.get(src, src), METHOD_LABEL.get(run_id, run_id), n, len(valid),
                         mean([fnum(r.get("Z_total")) for r in valid]),
                         mean([fnum(r.get(VAN_COL)) for r in valid]),
                         mean([fnum(r.get(QUEUE_COL)) for r in valid]),
                         mean([fnum(r.get("mean_time_to_charged_min")) for r in valid]),
                         mean([fnum(r.get("vans_used")) for r in valid])])
    write_csv(os.path.join(out, "charging_demand_by_n.csv"), header, rows)
    return rows


def mobile_charger_impact(cfg, base, exp_rows, out):
    """Where mobile chargers significantly improve the network: stranded demand, van use, and the fleet-size
    feasibility gain from E2 (3 / baseline-5 / 7 vans), by dataset x class x n."""
    header = ["dataset", "class", "n", "instances", "mean_stranded_EVs", "mean_van_share_%",
              "valid_3_vans", "valid_5_vans_base", "valid_7_vans", "feasibility_gain_7_minus_3"]
    # baseline (5 vans) reference = MILP baseline where present else RL baseline
    base_rows = base[("baseline", MILP_RUN)] or base[("baseline", RL_RUN)]
    v3 = exp_rows.get(("E2", "E2_vans_3", RL_RETRAIN)) or exp_rows.get(("E2", "E2_vans_3", RL_ZERO)) \
        or exp_rows.get(("E2", "E2_vans_3", MILP_RUN)) or []
    v7 = exp_rows.get(("E2", "E2_vans_7", RL_RETRAIN)) or exp_rows.get(("E2", "E2_vans_7", RL_ZERO)) \
        or exp_rows.get(("E2", "E2_vans_7", MILP_RUN)) or []

    def validmap(rows):
        return {key_in(r): is_valid(r) for r in rows}
    m3, m7 = validmap(v3), validmap(v7)
    groups = {}
    for r in base_rows:
        groups.setdefault((r.get("source", ""), r.get("class", ""), int(float(r.get("n", 0) or 0))), []).append(r)
    rows = []
    for (src, cls, n), g in sorted(groups.items()):
        keys = [key_in(r) for r in g]
        valid3 = sum(1 for k in keys if m3.get(k))
        valid7 = sum(1 for k in keys if m7.get(k))
        valid5 = sum(1 for r in g if is_valid(r))
        rows.append([SRC_SHORT.get(src, src), cls, n, len(g),
                     mean([fnum(r.get("stranded_EVs")) for r in g]),
                     mean([fnum(r.get(VAN_COL)) for r in g if is_valid(r)]),
                     valid3 if m3 else "", valid5, valid7 if m7 else "",
                     (valid7 - valid3) if (m3 and m7) else ""])
    write_csv(os.path.join(out, "mobile_charger_impact.csv"), header, rows)
    return rows


# --------------------------------------------------------------------------- summary.md
def md_table(header, rows, maxrows=24):
    out = ["| " + " | ".join(str(h) for h in header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows[:maxrows]:
        out.append("| " + " | ".join("" if c == "" else str(c) for c in r) + " |")
    if len(rows) > maxrows:
        out.append(f"| ...（{len(rows) - maxrows} more rows in the CSV） |" + " |" * (len(header) - 1))
    return "\n".join(out)


def summary_md(out, quality_overall, perf, mvr, sens, demand, impact):
    L = ["# Formulation 4 -- rubric results (Solomon + Gehring-Homberger)", "",
         "Every number is computed from the run tables (`runs.csv`), split by dataset. "
         "CSV files are in this folder.", "",
         "## Experimental Evaluation -- solution quality (pooled over n)", "",
         md_table(["dataset", "method", "n", "instances", "valid", "optimal", "mean_Z", "mean_gap_%",
                   "median_gap_%"], quality_overall), "",
         "## Experimental Evaluation -- computational performance / speed", "",
         "`performance.csv` (per n) and the head-to-head below. RL speed-up = mean MILP solve time / mean RL "
         "solve time.", "",
         md_table(["dataset", "milp_valid", "rl_valid", "paired", "rl_only", "milp_only", "mean_RL-MILP_Z_%",
                   "median_%", "mean_milp_s", "mean_rl_s", "speedup_x"], mvr), "",
         "## Analysis -- how each factor changes the optimal solution", "",
         "Full table: `sensitivity_summary.csv`. Rubric factors broken out in "
         "`sensitivity_mobile_chargers.csv` (E2), `sensitivity_station_capacity.csv` (E9), "
         "`sensitivity_charging_demand.csv` (E10). `dZ_%` is the mean change of Z on instances valid in both "
         "the variant and the baseline of the same dataset.", "",
         md_table(["experiment", "factor", "variant", "label", "dataset", "run", "paired", "dZ_%", "d_oper_%",
                   "d_cust_%", "van_base_%", "van_var_%", "q_base", "q_var", "new", "lost"],
                  [r for r in sens if r[5] in ("RL retrained", "MILP")], 40), "",
         "## Analysis -- charging demand LEVEL (number of EVs)", "",
         "`charging_demand_by_n.csv`: how the baseline solution moves as the number of EVs grows.", "",
         md_table(["dataset", "method", "n", "valid", "mean_Z", "van_share_%", "queue_min", "time_min",
                   "vans_used"], demand, 30), "",
         "## Analysis -- where mobile chargers significantly improve the network", "",
         "`mobile_charger_impact.csv`: stranded EVs (only a van can serve them), van share, and the fleet-size "
         "feasibility gain (how many more instances get a plan with 7 vans than with 3), by dataset x class x n. "
         "The gain is largest in the short-period classes (R1/RC1) and at large n.", "",
         md_table(["dataset", "class", "n", "instances", "stranded", "van_share_%", "valid_3v", "valid_5v",
                   "valid_7v", "gain_7-3"], [r for r in impact if r[9] not in ("", 0)], 30) if impact else
         "(no E2 fleet results yet)", ""]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=F4_DIR, help="formulation-4 folder (default: the parent of common/)")
    ap.add_argument("--out", default=None, help="output folder (default: <root>/rubric_results)")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    out = os.path.abspath(args.out or os.path.join(root, "rubric_results"))
    cfg, base, exp_rows = load_all(root)
    n_base = sum(len(v) for v in base.values())
    n_exp = sum(len(v) for v in exp_rows.values())
    print(f"read {n_base} baseline rows, {n_exp} experiment rows from {root}")

    _, quality_overall = solution_quality(base, out)
    perf = performance(base, out)
    mvr = milp_vs_rl(base, out)
    sens = sensitivity(cfg, base, exp_rows, out)
    demand = charging_demand_by_n(base, out)
    impact = mobile_charger_impact(cfg, base, exp_rows, out)
    summary_md(out, quality_overall, perf, mvr, sens, demand, impact)

    print(f"wrote rubric tables to {out}:")
    for f in sorted(os.listdir(out)):
        print("   ", f)


if __name__ == "__main__":
    main()
