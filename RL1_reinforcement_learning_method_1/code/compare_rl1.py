"""RL1 against the MILP and the greedy cost rule: per instance and summarised.

    python compare_rl1.py                                   # ../results vs ../../new_results
    python compare_rl1.py --rl ../results --milp ../../new_results --rl-run-id rl1 --milp-run-id milp_300s

Reads runs.csv of both folders (schema of code/results_io.py; the greedy rows are the run id
`greedy` of the MILP folder) and joins them on (instance, n). Writes into the RL folder:
  comparison.csv          one row per (instance, n): status, validity, Z and seconds of RL1, MILP and
                          greedy; gap of RL1 to the MILP plan and to the greedy plan (both valid),
                          speed-up = MILP seconds / RL1 seconds
  comparison_summary.md   the same summarised by class, by n and overall (numbers from the tables)
Gap = 100 (Z_RL1 - Z_other) / Z_other; negative = RL1 cheaper.
"""
import argparse
import csv
import os
import sys

import numpy as np

from rl1_data import PROJECT, RL1_DIR
import results_io as rio

COLUMNS = ["instance", "class", "n", "rl_status", "rl_valid", "rl_Z", "rl_s", "rl_violations",
           "milp_status", "milp_valid", "milp_Z", "milp_gap_%", "milp_s", "greedy_status", "greedy_Z",
           "gap_rl_vs_milp_%", "gap_rl_vs_greedy_%", "speedup_milp_over_rl"]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def latest(rows, run_id):
    out = {}
    for r in rows:
        if r.get("run_id") == run_id:
            out[(r["instance"], str(int(float(r["n"]))))] = r
    return out


def gap(a, b):
    return round(100.0 * (a - b) / b, 3) if a is not None and b not in (None, 0.0) else ""


def stats(vals):
    v = [x for x in vals if x != "" and x is not None]
    if not v:
        return "-", "-"
    return f"{np.mean(v):.2f}", f"{np.median(v):.2f}"


def summary_rows(rows):
    both = [r for r in rows if r["rl_valid"] == "yes" and r["milp_valid"] == "yes"]
    opt = [r for r in both if r["milp_status"] == "optimal"]
    bg = [r for r in rows if r["rl_valid"] == "yes" and r["greedy_Z"] != ""]
    m_all, med_all = stats([r["gap_rl_vs_milp_%"] for r in both])
    m_opt, _ = stats([r["gap_rl_vs_milp_%"] for r in opt])
    m_g, _ = stats([r["gap_rl_vs_greedy_%"] for r in bg])
    rl_s = [num(r["rl_s"]) for r in rows if num(r["rl_s"]) is not None]
    milp_s = [num(r["milp_s"]) for r in rows if num(r["milp_s"]) is not None]
    return {"instances": len(rows),
            "RL1 valid": sum(r["rl_valid"] == "yes" for r in rows),
            "MILP valid": sum(r["milp_valid"] == "yes" for r in rows),
            "MILP optimal": sum(r["milp_status"] == "optimal" for r in rows),
            "RL1 valid, MILP none": sum(r["rl_valid"] == "yes" and r["milp_valid"] != "yes" for r in rows),
            "MILP valid, RL1 not": sum(r["milp_valid"] == "yes" and r["rl_valid"] != "yes" for r in rows),
            "gap to MILP mean %": m_all, "gap to MILP median %": med_all,
            "gap to optimum mean %": m_opt,
            "RL1 < greedy": f"{sum(num(r['gap_rl_vs_greedy_%']) < 0 for r in bg)}/{len(bg)}" if bg else "-",
            "gap to greedy mean %": m_g,
            "RL1 s mean": f"{np.mean(rl_s):.2f}" if rl_s else "-",
            "MILP s mean": f"{np.mean(milp_s):.1f}" if milp_s else "-"}


def md_table(groups):
    keys = list(next(iter(groups.values())).keys())
    lines = ["| group | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for g, s in groups.items():
        lines.append(f"| {g} | " + " | ".join(str(s[k]) for k in keys) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rl", default=os.path.join(RL1_DIR, "results"))
    ap.add_argument("--milp", default=os.path.join(PROJECT, "new_results"))
    ap.add_argument("--rl-run-id", default="rl1")
    ap.add_argument("--milp-run-id", default="milp_300s")
    args = ap.parse_args()
    rl = latest(rio.read_runs(args.rl), args.rl_run_id)
    if not rl:
        print(f"no runs of {args.rl_run_id} in {os.path.abspath(args.rl)}/runs.csv", flush=True)
        sys.exit(1)
    other = rio.read_runs(args.milp)
    milp, greedy = latest(other, args.milp_run_id), latest(other, "greedy")
    if not milp:
        print(f"note: no {args.milp_run_id} runs in {os.path.abspath(args.milp)}/runs.csv; MILP columns stay blank")
    rows = []
    for key in sorted(rl, key=lambda k: (rl[k]["class"], k[0], int(k[1]))):
        r, m, g = rl[key], milp.get(key, {}), greedy.get(key, {})
        zr = num(r.get("Z_total")) if r.get("valid") == "yes" else None
        zm = num(m.get("Z_total")) if m.get("valid") == "yes" else None
        zg = num(g.get("Z_total")) if g.get("valid") == "yes" else None
        rs, ms = num(r.get("solve_s")), num(m.get("solve_s"))
        rows.append({"instance": key[0], "class": r.get("class", ""), "n": int(key[1]),
                     "rl_status": r.get("status", ""), "rl_valid": r.get("valid", ""), "rl_Z": r.get("Z_total", ""),
                     "rl_s": r.get("solve_s", ""), "rl_violations": r.get("violations", ""),
                     "milp_status": m.get("status", "not run"), "milp_valid": m.get("valid", ""),
                     "milp_Z": m.get("Z_total", ""), "milp_gap_%": m.get("gap_%", ""), "milp_s": m.get("solve_s", ""),
                     "greedy_status": g.get("status", ""), "greedy_Z": g.get("Z_total", ""),
                     "gap_rl_vs_milp_%": gap(zr, zm), "gap_rl_vs_greedy_%": gap(zr, zg),
                     "speedup_milp_over_rl": round(ms / rs, 1) if ms and rs else ""})
    path = os.path.join(args.rl, "comparison.csv")
    rio.write_table(path, COLUMNS, rows)
    by_class = {c: summary_rows([r for r in rows if r["class"] == c]) for c in sorted({r["class"] for r in rows})}
    by_n = {f"n = {n}": summary_rows([r for r in rows if r["n"] == n]) for n in sorted({r["n"] for r in rows})}
    total = summary_rows(rows)
    text = [f"# RL1 vs MILP vs greedy ({args.rl_run_id} vs {args.milp_run_id})", "",
            f"{total['instances']} instances. RL1 gives a valid plan in {total['RL1 valid']}, the MILP in "
            f"{total['MILP valid']} ({total['MILP optimal']} proven optimal). Where both have a valid plan, RL1 "
            f"costs on average {total['gap to MILP mean %']} % more than the MILP plan (median "
            f"{total['gap to MILP median %']} %); against proven optima the mean gap is "
            f"{total['gap to optimum mean %']} %. RL1 beats the greedy rule in {total['RL1 < greedy']} instances "
            f"(mean gap to greedy {total['gap to greedy mean %']} %). RL1 has a valid plan where the MILP has "
            f"none in {total['RL1 valid, MILP none']} instances; the reverse happens in "
            f"{total['MILP valid, RL1 not']}. Mean time per instance: RL1 {total['RL1 s mean']} s, MILP "
            f"{total['MILP s mean']} s.", "",
            "Gap = 100 (Z_RL1 - Z_other) / Z_other, over instances where both plans are valid; negative = RL1 cheaper.",
            "", "## By class", "", md_table(by_class), "", "## By number of EVs", "", md_table(by_n), "",
            "## All", "", md_table({"all": total}), ""]
    with open(os.path.join(args.rl, "comparison_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(text))
    print("\n".join(text[:3]))
    print(f"written: {os.path.abspath(path)} and comparison_summary.md")


if __name__ == "__main__":
    main()
