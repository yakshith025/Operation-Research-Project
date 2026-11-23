"""RL2 against RL1, the MILP and the greedy cost rule: per instance and summarised.

    python compare_rl2.py                    # ../results vs ../../RL1_.../results vs ../../new_results
    python compare_rl2.py --prefix rl1model  # e.g. the ablation run (active search on RL1's model)

Reads runs.csv of the three folders (schema of code/results_io.py) and joins them on (instance, n):
  RL2-EAS   run id <prefix>_eas        RL2 with active search          (this folder)
  RL2       run id <prefix>_sampling   RL2 sampling only (as RL1)      (this folder)
  RL1       run id rl1                                                 (RL1 folder)
  greedy    run id greedy                                              (MILP folder)
  MILP      run id milp_300s           the reference                  (MILP folder)
A method whose runs are missing is left out. Writes into --out (default ../results):
  comparison.csv          one row per (instance, n): status / validity / Z / seconds of every method and
                          its gap to the MILP plan
  comparison_summary.md   one table per method overall, then the mean gap to the MILP by class and by n,
                          and what active search changed (numbers from the tables)
Gap = 100 (Z_method - Z_MILP) / Z_MILP over instances where both plans are valid; negative = cheaper.
"""
import argparse
import os
import sys

import numpy as np

from rl2_data import PROJECT, RL2_DIR
import results_io as rio

RL1_RESULTS = os.path.join(PROJECT, "RL1_reinforcement_learning_method_1", "results")


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def runs_of(folder, run_id):
    return {(r["instance"], str(int(float(r["n"])))): r for r in rio.read_runs(folder) if r.get("run_id") == run_id}


def zval(r):
    return num(r.get("Z_total")) if r and r.get("valid") == "yes" else None


def pct(a, b):
    return 100.0 * (a - b) / b if a is not None and b else None


def fmt(v, d=2):
    return "-" if v is None else f"{v:.{d}f}"


def mean(vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def median(vals):
    v = [x for x in vals if x is not None]
    return float(np.median(v)) if v else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rl2", default=os.path.join(RL2_DIR, "results"))
    ap.add_argument("--rl1", default=RL1_RESULTS)
    ap.add_argument("--milp", default=os.path.join(PROJECT, "new_results"))
    ap.add_argument("--prefix", default="rl2", help="run ids <prefix>_eas and <prefix>_sampling")
    ap.add_argument("--rl1-run-id", default="rl1")
    ap.add_argument("--milp-run-id", default="milp_300s")
    ap.add_argument("--out", default=None, help="output folder (default: the RL2 results folder)")
    args = ap.parse_args()
    out = args.out or args.rl2
    methods = [("RL2-EAS", runs_of(args.rl2, f"{args.prefix}_eas")),
               ("RL2", runs_of(args.rl2, f"{args.prefix}_sampling")),
               ("RL1", runs_of(args.rl1, args.rl1_run_id)),
               ("greedy", runs_of(args.milp, "greedy"))]
    missing = [m for m, d in methods if not d]
    methods = [(m, d) for m, d in methods if d]
    milp = runs_of(args.milp, args.milp_run_id)
    if not any(m.startswith("RL2") for m, _ in methods):
        print(f"no RL2 runs with prefix {args.prefix} in {os.path.abspath(args.rl2)}/runs.csv", flush=True)
        sys.exit(1)
    if missing:
        print(f"note: no runs found for {missing}; left out", flush=True)
    keys = set()
    for m, d in methods:
        if m.startswith("RL2"):
            keys |= set(d)
    rows = []
    for key in sorted(keys, key=lambda k: (k[0], int(k[1]))):
        ref = milp.get(key, {})
        zm = zval(ref)
        cls = next((d[key]["class"] for m, d in methods if key in d), ref.get("class", ""))
        row = {"instance": key[0], "class": cls, "n": int(key[1]), "milp_status": ref.get("status", "not run"),
               "milp_valid": ref.get("valid", ""), "milp_Z": ref.get("Z_total", ""), "milp_s": ref.get("solve_s", "")}
        for m, d in methods:
            r = d.get(key, {})
            row[f"{m}_status"], row[f"{m}_valid"] = r.get("status", ""), r.get("valid", "")
            row[f"{m}_Z"], row[f"{m}_s"] = r.get("Z_total", ""), r.get("solve_s", "")
            g = pct(zval(r), zm)
            row[f"{m}_gap_milp_%"] = round(g, 3) if g is not None else ""
        rows.append(row)
    cols = ["instance", "class", "n", "milp_status", "milp_valid", "milp_Z", "milp_s"]
    for m, _ in methods:
        cols += [f"{m}_status", f"{m}_valid", f"{m}_Z", f"{m}_s", f"{m}_gap_milp_%"]
    rio.write_table(os.path.join(out, "comparison.csv"), cols, rows)

    def method_stats(m, sel):
        both = [r for r in sel if r[f"{m}_valid"] == "yes" and r["milp_valid"] == "yes"]
        opt = [r for r in both if r["milp_status"] == "optimal"]
        g = [num(r[f"{m}_gap_milp_%"]) for r in both]
        gr = []
        if m != "greedy" and any(mm == "greedy" for mm, _ in methods):
            for r in sel:
                a, b = num(r[f"{m}_Z"]) if r[f"{m}_valid"] == "yes" else None, \
                    num(r["greedy_Z"]) if r.get("greedy_valid") == "yes" else None
                if a is not None and b is not None:
                    gr.append(a < b - 1e-6)
        secs = [num(r[f"{m}_s"]) for r in sel]
        return {"valid plans": sum(r[f"{m}_valid"] == "yes" for r in sel),
                "valid, MILP none": sum(r[f"{m}_valid"] == "yes" and r["milp_valid"] != "yes" for r in sel),
                "MILP valid, this not": sum(r["milp_valid"] == "yes" and r[f"{m}_valid"] != "yes" for r in sel),
                "gap to MILP mean %": fmt(mean(g)), "gap to MILP median %": fmt(median(g)),
                "gap to optimum mean %": fmt(mean([num(r[f"{m}_gap_milp_%"]) for r in opt])),
                "cheaper than MILP": sum(x is not None and x < -1e-6 for x in g),
                "cheaper than greedy": f"{sum(gr)}/{len(gr)}" if gr else "-",
                "s per instance": fmt(mean(secs), 2)}

    def table(header, groups):
        keys_ = list(next(iter(groups.values())).keys())
        lines = [f"| {header} | " + " | ".join(keys_) + " |", "|---|" + "---|" * len(keys_)]
        lines += [f"| {g} | " + " | ".join(str(v) for v in s.values()) + " |" for g, s in groups.items()]
        return "\n".join(lines)

    def gap_by(field, values, label):
        groups = {}
        for v in values:
            sel = [r for r in rows if r[field] == v]
            groups[f"{label}{v}"] = {m: fmt(mean([num(r[f"{m}_gap_milp_%"]) for r in sel
                                                  if r[f"{m}_valid"] == "yes" and r["milp_valid"] == "yes"]))
                                     for m, _ in methods}
        return groups

    overall = {m: method_stats(m, rows) for m, _ in methods}
    text = [f"# RL2 vs RL1 vs MILP vs greedy (prefix {args.prefix}, MILP run {args.milp_run_id})", "",
            f"{len(rows)} instances; the MILP has a valid plan in {sum(r['milp_valid'] == 'yes' for r in rows)} "
            f"({sum(r['milp_status'] == 'optimal' for r in rows)} proven optimal).", "",
            "Gap = 100 (Z_method - Z_MILP) / Z_MILP over instances where both plans are valid; negative = cheaper.", "",
            "## Overall", "", table("method", overall), "",
            "## Mean gap to the MILP (%) by class", "", table("class", gap_by("class", sorted({r['class'] for r in rows}), "")), "",
            "## Mean gap to the MILP (%) by number of EVs", "", table("n", gap_by("n", sorted({r['n'] for r in rows}), "n = ")), ""]
    if any(m == "RL2-EAS" for m, _ in methods) and any(m == "RL2" for m, _ in methods):
        ch = [pct(num(r["RL2-EAS_Z"]), num(r["RL2_Z"])) for r in rows
              if r["RL2-EAS_valid"] == "yes" and r["RL2_valid"] == "yes"]
        newly = sum(r["RL2-EAS_valid"] == "yes" and r["RL2_valid"] != "yes" for r in rows)
        text += ["## What active search changed (RL2-EAS vs RL2 sampling)", "",
                 f"Over {len(ch)} instances with both plans valid: Z changed by {fmt(mean(ch))} % on average "
                 f"(median {fmt(median(ch))} %); lower in {sum(c is not None and c < -1e-9 for c in ch)}, equal in "
                 f"{sum(c is not None and abs(c) <= 1e-9 for c in ch)}. Active search made a valid plan out of an "
                 f"invalid one in {newly} instances.", ""]
    with open(os.path.join(out, "comparison_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(text))
    print("\n".join(text[:14]))
    print(f"written: {os.path.abspath(os.path.join(out, 'comparison.csv'))} and comparison_summary.md")


if __name__ == "__main__":
    main()
