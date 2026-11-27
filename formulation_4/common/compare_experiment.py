"""Compare the variants of one experiment with the baseline of the same method (formulation 4).

MILP folder: every variant's run milp_300s against methods/milp/results (milp_300s).
RL folder:   every variant's runs rl_f4_zero_shot and rl_f4_retrained against methods/rl/results (rl_f4); if the
             MILP folder of the same experiment has results for the variant, the RL plans are also compared
             with the MILP plans of that variant (gap to MILP).
Only PAIRED instances count in the changes: (instance, n) with a valid plan in both the variant and the baseline.

Writes <experiment>_<METHOD>/results/comparison/:
  comparison.csv            one row per (variant, run id): validity, mean / median change of Z, operator and
                            customer cost, van shares per segment (variant vs baseline), vans and seats used,
                            queues and downtime per segment, money paid per segment, value per unit of the factor
  comparison_by_income.csv  share of free-choice EVs served by a van per segment x income band, per variant
                            and for the baseline (all valid plans)
  comparison.md             the question, both tables and the headline statements
  fig_van_share.png, fig_dZ.png  (if matplotlib is installed)

Usage:  python compare_experiment.py E5 milp [RESULTS_ROOT] [BASELINE_DIR]
        (the experiment folders call it through analyze_<E>_<METHOD>.py)
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
F4_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import paths  # noqa: E402
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import analyze_f4 as an  # noqa: E402
import results_io_f4 as rio  # noqa: E402

SEGS = ("personal", "commercial")


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def rel(path, start=F4_DIR):
    """Relative path when possible (os.path.relpath fails across Windows drives)."""
    try:
        return os.path.relpath(path, start).replace(os.sep, "/")
    except ValueError:
        return os.path.abspath(path)


def runs_by_key(results_dir, run_id):
    out = {}
    for r in rio.read_runs(results_dir):
        if r["run_id"] == run_id:
            out[(r["instance"], rio.key_of(r)[4])] = r
    return out


def pooled_share(rows, seg):
    free = sum(fnum(r.get(f"free_choice_{seg}")) for r in rows if r.get(f"free_choice_{seg}") not in ("", None))
    van = sum(fnum(r.get(f"free_choice_by_van_{seg}")) for r in rows if r.get(f"free_choice_by_van_{seg}") not in ("", None))
    return round(100 * van / free, 2) if free else ""


def mean_col(rows, col, d=2):
    vals = [fnum(r.get(col)) for r in rows]
    vals = [v for v in vals if v == v]
    return round(float(np.mean(vals)), d) if vals else ""


def compare_pair(var_rows, base_rows, label_extra=None):
    """Statistics of a variant run against a baseline run (dicts keyed by (instance, n))."""
    valid_v = {k for k, r in var_rows.items() if r.get("valid") == "yes"}
    valid_b = {k for k, r in base_rows.items() if r.get("valid") == "yes"}
    paired = sorted(valid_v & valid_b)
    V = [var_rows[k] for k in paired]
    B = [base_rows[k] for k in paired]
    dz = [100 * (fnum(v["Z_total"]) - fnum(b["Z_total"])) / fnum(b["Z_total"]) for v, b in zip(V, B)]
    dzr = [fnum(v["Z_total"]) - fnum(b["Z_total"]) for v, b in zip(V, B)]

    def dpct(col):
        x = [100 * (fnum(v[col]) - fnum(b[col])) / fnum(b[col]) for v, b in zip(V, B) if fnum(b[col]) not in (0.0,)]
        x = [y for y in x if y == y]
        return round(float(np.mean(x)), 3) if x else ""
    status = {}
    for r in var_rows.values():
        status[r["status"]] = status.get(r["status"], 0) + 1
    row = {"runs": len(var_rows), "valid": len(valid_v), "baseline_valid": len(valid_b),
           "paired": len(paired), "newly_valid": len(valid_v - valid_b), "no_longer_valid": len(valid_b - valid_v),
           "status_counts": json.dumps(status),
           "mean_dZ_%": round(float(np.mean(dz)), 3) if dz else "",
           "median_dZ_%": round(float(np.median(dz)), 3) if dz else "",
           "mean_dZ_Rs": round(float(np.mean(dzr)), 2) if dzr else "",
           "mean_d_operator_cost_%": dpct("operator_cost_Z2_to_Z4"),
           "mean_d_customer_cost_%": dpct("customer_cost_Z1_Z5_Z6")}
    for seg in SEGS:
        row[f"share_van_{seg}_%"] = pooled_share(V, seg)
        row[f"baseline_share_van_{seg}_%"] = pooled_share(B, seg)
    for col, name in (("vans_used", "vans_used"), ("van_seats_used_%", "van_seats_used_%"),
                      ("mean_queue_personal_min", "mean_queue_personal_min"),
                      ("mean_queue_commercial_min", "mean_queue_commercial_min"),
                      ("mean_downtime_commercial_min", "mean_downtime_commercial_min"),
                      ("mean_pay_personal_Rs", "mean_pay_personal_Rs"),
                      ("mean_pay_commercial_Rs", "mean_pay_commercial_Rs"),
                      ("mean_customer_cost_personal_Rs", "mean_customer_cost_personal_Rs"),
                      ("mean_customer_cost_commercial_Rs", "mean_customer_cost_commercial_Rs"),
                      ("solve_s", "mean_solve_s")):
        row[name] = mean_col(V, col)
        row[f"baseline_{name}"] = mean_col(B, col)
    gaps = [fnum(r.get("gap_%")) for r in var_rows.values() if r.get("gap_%") not in ("", None)]
    if gaps:
        row["optimal"] = sum(r["status"] == "optimal" for r in var_rows.values())
        row["median_gap_%"] = round(float(np.median(gaps)), 3)
    if label_extra:
        row.update(label_extra)
    return row, dzr


def income_table(results_dir, run_id, label):
    if not os.path.exists(os.path.join(results_dir, "runs.csv")):
        return []
    _, _, evs = an.load(results_dir, run_id)
    out = []
    for seg in SEGS:
        for b in an.BAND_NAMES:
            grp = [e for e in evs if e["segment"] == seg and an.band_of(e["theta"]) == b]
            if not grp:
                continue
            st = an.stats(grp)
            out.append({"run": label, "segment": seg, "income_band": b, "EVs": st["EVs"],
                        "free_choice": st["free_choice"], "share_van_free_choice_%": st["share_van_free_choice_%"],
                        "mean_queue_FCS_min": st["mean_queue_FCS_min"], "mean_customer_cost_Rs": st["mean_customer_cost_Rs"]})
    return out


def write_rows(path, rows):
    if not rows:
        return
    cols = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)


def figures(outdir, rows, exp):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    rows = [r for r in rows if r.get("paired")]
    if not rows:
        return
    labels = [f"{r['variant']}\n{r['run_id']}" if r["run_id"] not in ("milp_300s",) else r["variant"] for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(rows)), 3.6))
    for j, (seg, col) in enumerate((("personal", "#2a6fb0"), ("commercial", "#c4612b"))):
        vals = [fnum(r[f"share_van_{seg}_%"]) for r in rows]
        base = [fnum(r[f"baseline_share_van_{seg}_%"]) for r in rows]
        ax.bar(x + (j - 0.5) * 0.38, vals, 0.36, color=col, label=f"{seg} (variant)")
        ax.scatter(x + (j - 0.5) * 0.38, base, color="black", marker="_", s=200, zorder=3,
                   label="baseline (same instances)" if j == 0 else None)
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel("free-choice EVs served by a van (%)")
    ax.set_title(f"{exp['id']}: {exp['name']}", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_van_share.png"), dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(rows)), 3.2))
    ax.bar(x, [fnum(r["mean_dZ_%"]) for r in rows], color="#555555")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel("mean change of Z vs baseline (%)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_dZ.png"), dpi=150)
    plt.close(fig)


def md_table(rows, cols):
    if not rows:
        return "(no results yet)\n"
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        out += "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n"
    return out


def run(exp_id, method, results_root_base=None, baseline_dir=None):
    cfg = json.load(open(os.path.join(HERE, "experiments.json"), encoding="utf-8"))
    exp = next(e for e in cfg["experiments"] if e["id"].upper() == exp_id.upper())
    method = method.lower()
    root = paths.exp_dir(exp["folder"], method, results_root_base)
    res_root = os.path.join(root, "results")
    outdir = os.path.join(res_root, "comparison")
    os.makedirs(outdir, exist_ok=True)
    S, Bc = cfg["solver"], cfg["baseline"]
    if method == "milp":
        base_dir, base_run, var_runs = os.path.join(F4_DIR, Bc["milp_results"]), Bc["milp_run_id"], [S["milp_run_id"]]
    else:
        base_dir, base_run = os.path.join(F4_DIR, Bc["rl_results"]), Bc["rl_run_id"]
        var_runs = [S["rl_zero_shot_run_id"], S["rl_retrained_run_id"]]
    if baseline_dir:                               # e.g. a test baseline
        base_dir = baseline_dir
    base = runs_by_key(base_dir, base_run)
    milp_root = os.path.join(paths.exp_dir(exp["folder"], "milp", results_root_base), "results")
    rows, inc_rows = [], []
    if base:
        inc_rows += income_table(base_dir, base_run, f"baseline ({base_run})")
    for var in exp["variants"]:
        vdir = os.path.join(res_root, var["id"])
        if not os.path.exists(os.path.join(vdir, "runs.csv")):
            continue
        for run_id in var_runs:
            vr = runs_by_key(vdir, run_id)
            if not vr:
                continue
            extra = {}
            if method == "rl":
                mr = runs_by_key(os.path.join(milp_root, var["id"]), S["milp_run_id"])
                if mr:
                    st, _ = compare_pair(vr, mr)
                    extra = {"paired_with_milp": st["paired"], "mean_gap_to_milp_%": st["mean_dZ_%"],
                             "median_gap_to_milp_%": st["median_dZ_%"]}
            st, dzr = compare_pair(vr, base, extra)
            bval = exp.get("baseline_values", {}).get(var.get("factor"))
            if isinstance(var.get("value"), (int, float)) and isinstance(bval, (int, float)) and var["value"] != bval and dzr:
                st["dZ_Rs_per_unit_of_factor"] = round(float(np.mean(dzr)) / (var["value"] - bval), 2)
            rows.append({"experiment": exp["id"], "variant": var["id"], "label": var["label"], "run_id": run_id,
                         "baseline_run": base_run, **st})
            inc_rows += income_table(vdir, run_id, f"{var['id']} ({run_id})")
    write_rows(os.path.join(outdir, "comparison.csv"), rows)
    write_rows(os.path.join(outdir, "comparison_by_income.csv"), inc_rows)
    figures(outdir, rows, exp)
    notes = []
    if not base:
        notes.append(f"The baseline run {base_run} is not in {base_dir} yet: changes cannot be computed until it is.")
    for r in rows:
        if r.get("paired"):
            notes.append(f"{r['variant']} ({r['run_id']}): Z {r['mean_dZ_%']:+} % on average over {r['paired']} paired "
                         f"instances; van share personal {r['baseline_share_van_personal_%']} -> {r['share_van_personal_%']} %, "
                         f"commercial {r['baseline_share_van_commercial_%']} -> {r['share_van_commercial_%']} %; "
                         f"{r['newly_valid']} instance(s) newly solved, {r['no_longer_valid']} lost.")
    thr = exp.get("decision_threshold_pct")
    if thr is not None:
        for r in rows:
            if r.get("paired") and r.get("mean_dZ_%") != "":
                small = abs(fnum(r["mean_dZ_%"])) < thr
                notes.append(f"Decision rule ({exp['id']}): |mean change of Z| = {abs(fnum(r['mean_dZ_%'])):.2f} % is "
                             f"{'below' if small else 'above'} {thr} %: "
                             + ("a larger battery changes little, so van recharging need not be modelled."
                                if small else "the battery binds; modelling van recharging is worth it."))
    cols = ["variant", "run_id", "valid", "paired", "newly_valid", "no_longer_valid", "mean_dZ_%", "median_dZ_%",
            "mean_d_operator_cost_%", "mean_d_customer_cost_%", "baseline_share_van_personal_%", "share_van_personal_%",
            "baseline_share_van_commercial_%", "share_van_commercial_%", "vans_used", "baseline_vans_used"]
    if method == "rl":
        cols += ["mean_gap_to_milp_%"]
    text = (f"# {exp['id']}: {exp['name']} ({method.upper()})\n\n**Question.** {exp['question']}\n\n"
            f"{exp.get('_about', '')}\n\nBaseline: `{base_run}` in `{rel(base_dir)}`. Changes are over "
            "paired instances (valid in both). Van shares are pooled over the free-choice EVs of the paired instances.\n\n"
            "## Variants against the baseline\n\n" + md_table(rows, cols) +
            "\n## Headlines\n\n" + ("\n".join(f"- {n}" for n in notes) if notes else "- (no results yet)") + "\n"
            "\n## Van share by segment and income band (all valid plans)\n\n" +
            md_table(inc_rows, ["run", "segment", "income_band", "EVs", "free_choice", "share_van_free_choice_%",
                                "mean_queue_FCS_min", "mean_customer_cost_Rs"]) +
            "\nFull columns in comparison.csv and comparison_by_income.csv.\n")
    with open(os.path.join(outdir, "comparison.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"comparison of {exp['id']} ({method}) written to {outdir}: {len(rows)} variant run(s)", flush=True)
    return rows


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    run(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None,
        sys.argv[4] if len(sys.argv) > 4 else None)
