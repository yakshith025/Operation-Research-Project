"""Income insights from a main_formulation_3 run: who is served at an FCS and who by a van,
by income theta, by period (peak / off-peak), family and instance size.

Reads the shared result tables (results_io.py): the runs of one run_id in runs.csv (default: the
latest MILP run in meta.json) and their EV rows in plan.csv. Only valid plans are used
(valid == yes; optimal or time-limit incumbents). The EV attributes (theta, stranded, nearest
FCS, period, ...) are rebuilt from the dataset, they are not stored in the results. The choice
analysis uses the free-choice EVs only: a stranded EV (no FCS in range) must take a van
whatever its income.

Usage (from code/):  python analyze_f3.py [results_dir] [run_id]   (default ../new_results, latest MILP run)
solve_milp_f3.py calls run() at the end of every run.

Writes <results_dir>/analysis/:
  ev_level.csv                       every EV of every valid plan (one row each)
  by_income.csv                      per income band: EVs, share by van, waits, costs
  by_income_period.csv               ... split by peak / off-peak
  by_income_layout.csv               ... split by Solomon layout (C, R, RC) and horizon (1 short, 2 long)
  by_income_size.csv                 ... split by instance size n
  logit.csv                          logistic regression P(van) on income, peak, income x peak, ...
  key_numbers.json, insights.md      headline numbers and all tables in Markdown
  fig_share_van_by_income.pdf/.png   share of free-choice EVs served by a van, by income and period
  fig_waits_by_income.pdf/.png       FCS queue and van wait by income
  fig_cost_by_income.pdf/.png        perceived money and time cost by income
"""
import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_ev_solomon as mes  # noqa: E402
import make_income as inc   # noqa: E402
import results_io as rio    # noqa: E402
BANDS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
NUM = ["theta", "mu", "c_D", "c_Q", "c_V", "c_C", "W_min", "reachable_fcs", "nearest_reachable_fcs_km",
       "E_kWh", "r_min", "load_at_request", "start_min", "end_min", "drive_km", "drive_min",
       "queue_min", "van_wait_min", "charge_min", "time_to_charged_min", "pay_Rs",
       "perceived_money_Rs", "time_cost_Rs", "customer_cost_Rs", "operator_energy_Rs"]


BAND_NAMES = [f"{lo:.1f}-{hi:.1f}" for lo, hi in BANDS]
VAN_ONLY = ("stops", "km", "kWh_delivered", "kWh_drawn", "battery_left_kWh", "return_min")


def band_of(theta):
    """Income band of theta: [0, 0.2), [0.2, 0.4), ..., [0.8, 1.0]."""
    if theta != theta:
        return BAND_NAMES[-1]
    return BAND_NAMES[max(0, min(int(theta * 5 + 1e-9), 4))]


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# ------------------------------------------------------------------ data
def pick_run(results_dir, run_id=None, method="MILP"):
    """(run_id, its rows of runs.csv). Default: the latest run of the method in meta.json, else
    the first one in runs.csv."""
    runs = rio.read_runs(results_dir)
    present = {r["run_id"] for r in runs}
    if run_id is None:
        meta = rio.read_meta(results_dir)
        cands = [(v.get("started", ""), k) for k, v in meta.items()
                 if v.get("method") == method and k in present]
        if cands:
            run_id = max(cands)[1]
        else:
            ids = [r["run_id"] for r in runs if r.get("method") == method]
            run_id = ids[0] if ids else None
    return run_id, [r for r in runs if r["run_id"] == run_id]


def load(results_dir, run_id=None):
    """Runs of one run_id, the valid ones, and one row per EV of every valid plan (plan.csv row
    + EV attributes rebuilt from the dataset)."""
    from solve_milp_f3 import load_instance      # here: solve_milp_f3 imports this module too
    run_id, inst = pick_run(results_dir, run_id)
    used_all = [r for r in inst if r.get("valid", "") == "yes"]
    keys = {rio.key_of(r) for r in used_all}
    plan = {}
    for e in rio.iter_plan(results_dir, keys, types={"ev"}):
        plan.setdefault(rio.key_of(e), []).append(e)
    meta = rio.read_meta(results_dir).get(run_id, {})
    instances = meta.get("instances", "")              # dataset folder of the run (CSV or text form)
    data = mes.EvSolomonData(instances) if instances and os.path.isdir(instances) else None
    if data is None or not data.names():
        data = mes.EvSolomonData(inc.DEFAULT_CSV)
    params = meta.get("params", "")
    P3 = inc.load_params(params if params and os.path.isfile(params) else inc.DEFAULT_PARAMS)
    evs, used = [], []
    for r in used_all:
        k = rio.key_of(r)
        if k not in plan or r["instance"] not in data:
            continue
        J = load_instance(r["instance"], int(k[4]), P3, data, greedy=False)
        attrs = {a["id"]: a for a in rio.ev_attributes(J)}
        used.append(r)
        fam = r.get("class", "")
        for p in plan[k]:
            p = {c: v for c, v in p.items() if c not in VAN_ONLY}
            e = {"instance": r["instance"], "n": int(k[4]), **attrs.get(int(p["id"]), {}), **p}
            for c in NUM:
                if c in e:
                    e[c] = to_float(e[c])
            e.update({"n": int(k[4]), "id": int(p["id"]), "EV": int(p["id"]),   # EV: name used before
                      "type": p["resource_type"], "family": fam,
                      "layout": fam[:-1],
                      "horizon": "type 1 (short)" if fam.endswith("1") else "type 2 (long)",
                      "solver_status": r.get("status", ""), "income_band": band_of(e["theta"]),
                      "van": 1 if p["resource_type"] == "MCS" else 0})
            evs.append(e)
    return run_id, inst, used, evs


def mean(vals):
    vals = [v for v in vals if v == v]           # drop NaN
    return float(np.mean(vals)) if vals else float("nan")


def stats(group):
    fcs = [e for e in group if e["type"] == "FCS"]
    van = [e for e in group if e["type"] == "MCS"]
    return {"EVs": len(group), "at_FCS": len(fcs), "by_van": len(van),
            "share_by_van_%": round(100 * len(van) / len(group), 1) if group else "",
            "mean_theta": round(mean([e["theta"] for e in group]), 3),
            "mean_queue_FCS_min": round(mean([e["queue_min"] for e in fcs]), 2) if fcs else "",
            "share_FCS_users_queueing_%": (round(100 * sum(e["queue_min"] > 1e-6 for e in fcs) / len(fcs), 1)
                                           if fcs else ""),
            "mean_van_wait_min": round(mean([e["van_wait_min"] for e in van]), 2) if van else "",
            "mean_drive_km_FCS": round(mean([e["drive_km"] for e in fcs]), 2) if fcs else "",
            "mean_time_to_charged_min": round(mean([e["time_to_charged_min"] for e in group]), 1),
            "mean_pay_Rs": round(mean([e["pay_Rs"] for e in group]), 1),
            "mean_perceived_money_Rs": round(mean([e["perceived_money_Rs"] for e in group]), 1),
            "mean_time_cost_Rs": round(mean([e["time_cost_Rs"] for e in group]), 1),
            "mean_customer_cost_Rs": round(mean([e["customer_cost_Rs"] for e in group]), 1)}


def table(evs, keys):
    """Rows of stats for every combination of the grouping keys (income band always last)."""
    groups = {}
    for e in evs:
        groups.setdefault(tuple(e[k] for k in keys), []).append(e)

    def order(k):
        return tuple(BAND_NAMES.index(v) if v in BAND_NAMES else (v if isinstance(v, (int, float)) else str(v))
                     for v in k)
    rows = []
    for key in sorted(groups, key=order):
        rows.append({**dict(zip(keys, key)), **stats(groups[key])})
    return rows


# ------------------------------------------------------------------ logistic regression
def logit(evs):
    """P(van) = logistic(b0 + b1 theta + b2 peak + b3 theta*peak + b4 E/10 + b5 nearest FCS km),
    free-choice EVs. Newton's method with a tiny ridge (keeps it finite under separation)."""
    rows = [e for e in evs if e["nearest_reachable_fcs_km"] == e["nearest_reachable_fcs_km"]]
    if len(rows) <= 6:                                  # 6 coefficients need more than 6 observations
        return None, f"only {len(rows)} free-choice EVs for 6 coefficients"
    y = np.array([e["van"] for e in rows], dtype=float)
    if y.min() == y.max():
        return None, "every free-choice EV has the same resource"
    th = np.array([e["theta"] for e in rows])
    pk = np.array([1.0 if e["period"] == "peak" else 0.0 for e in rows])
    X = np.column_stack([np.ones(len(rows)), th, pk, th * pk,
                         np.array([e["E_kWh"] for e in rows]) / 10.0,
                         np.array([e["nearest_reachable_fcs_km"] for e in rows])])
    names = ["intercept", "theta (income)", "peak", "theta x peak", "energy E (per 10 kWh)",
             "nearest reachable FCS (km)"]
    lam = 1e-3 * np.eye(X.shape[1])
    lam[0, 0] = 0.0
    b = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -35, 35)))
        g = X.T @ (y - p) - lam @ b
        H = X.T @ (X * (p * (1 - p))[:, None]) + lam
        step = np.linalg.solve(H, g)
        b += step
        if np.max(np.abs(step)) < 1e-8:
            break
    p = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -35, 35)))
    H = X.T @ (X * (p * (1 - p))[:, None]) + lam
    se = np.sqrt(np.clip(np.diag(np.linalg.inv(H)), 0, None))
    out = []
    for k, nm in enumerate(names):
        z = b[k] / se[k] if se[k] > 0 else float("nan")
        pval = math.erfc(abs(z) / math.sqrt(2)) if z == z else float("nan")
        out.append({"term": nm, "coef": round(float(b[k]), 4), "std_err": round(float(se[k]), 4),
                    "z": round(float(z), 2), "p_value": float(f"{pval:.3g}"),
                    "odds_ratio": round(float(math.exp(min(50.0, b[k]))), 4)})
    # average marginal effect of income on P(van): d p / d theta = (b1 + b3 peak) p (1 - p)
    ame = float(np.mean((b[1] + b[3] * pk) * p * (1 - p)))
    ame_peak = float(np.mean(((b[1] + b[3]) * p * (1 - p))[pk == 1])) if pk.any() else float("nan")
    ame_off = float(np.mean((b[1] * p * (1 - p))[pk == 0])) if (pk == 0).any() else float("nan")
    acc = float(np.mean((p >= 0.5) == (y == 1)))
    info = {"observations": len(rows), "share_van": round(float(y.mean()), 4),
            "AME_theta_all": round(ame, 4), "AME_theta_peak": round(ame_peak, 4),
            "AME_theta_offpeak": round(ame_off, 4), "in_sample_accuracy": round(acc, 4),
            "coef": {nm: float(b[k]) for k, nm in enumerate(names)}}
    return out, info


# ------------------------------------------------------------------ figures
def figures(outdir, free, lg_info):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    plt.rcParams.update({"font.size": 9, "font.family": "serif", "mathtext.fontset": "cm",
                         "axes.spines.top": False, "axes.spines.right": False})
    mids = [(lo + hi) / 2 for lo, hi in BANDS]
    made = []

    def share(grp):
        return 100 * np.mean([e["van"] for e in grp]) if grp else np.nan

    # 1. share by van per income band, all / peak / off-peak (+ logit curves)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    styles = [("all free-choice EVs", lambda e: True, "#333333", "o", "-"),
              ("peak", lambda e: e["period"] == "peak", "#C44E52", "s", "--"),
              ("off-peak", lambda e: e["period"] == "off-peak", "#4C72B0", "^", ":")]
    for lab, sel, col, mk, ls in styles:
        vals = [share([e for e in free if e["income_band"] == b and sel(e)]) for b in BAND_NAMES]
        ax.plot(mids, vals, marker=mk, color=col, ls=ls, lw=1.2, ms=4, label=lab)
    ax.set(xlabel=r"income $\theta$ (band midpoint)", ylabel="served by a van (%)", xlim=(0, 1),
           xticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig_share_van_by_income.{ext}"), dpi=200)
    plt.close(fig)
    made.append("fig_share_van_by_income")

    # 2. waits by income
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    q = [mean([e["queue_min"] for e in free if e["income_band"] == b and e["type"] == "FCS"])
         for b in BAND_NAMES]
    qp = [mean([e["queue_min"] for e in free if e["income_band"] == b and e["type"] == "FCS"
                and e["period"] == "peak"]) for b in BAND_NAMES]
    w = [mean([e["van_wait_min"] for e in free if e["income_band"] == b and e["type"] == "MCS"])
         for b in BAND_NAMES]
    ax.plot(mids, q, marker="o", color="#333333", lw=1.2, ms=4, label="FCS queue, all")
    ax.plot(mids, qp, marker="s", color="#C44E52", ls="--", lw=1.2, ms=4, label="FCS queue, peak")
    ax.plot(mids, w, marker="^", color="#55A868", ls=":", lw=1.2, ms=4, label="wait for van")
    ax.set(xlabel=r"income $\theta$ (band midpoint)", ylabel="mean minutes", xlim=(0, 1),
           xticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig_waits_by_income.{ext}"), dpi=200)
    plt.close(fig)
    made.append("fig_waits_by_income")

    # 3. perceived cost split by income
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    money = [mean([e["perceived_money_Rs"] for e in free if e["income_band"] == b]) for b in BAND_NAMES]
    tcost = [mean([e["time_cost_Rs"] for e in free if e["income_band"] == b]) for b in BAND_NAMES]
    pay = [mean([e["pay_Rs"] for e in free if e["income_band"] == b]) for b in BAND_NAMES]
    xs = np.arange(len(BAND_NAMES))
    ax.bar(xs, money, 0.6, color="#4C72B0", label=r"perceived money ($\mu_i\times$paid)")
    ax.bar(xs, tcost, 0.6, bottom=money, color="#DD8452", label="perceived time")
    ax.plot(xs, pay, "k_", ms=18, mew=1.5, label="money actually paid")
    ax.set_xticks(xs)
    ax.set_xticklabels(BAND_NAMES)
    ax.set(xlabel=r"income band $\theta$", ylabel="Rs per EV")
    top = np.nanmax([m_ + t_ for m_, t_ in zip(money, tcost)] + [0.0])
    ax.set_ylim(0, 1.45 * top if top > 0 else 1.0)
    ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig_cost_by_income.{ext}"), dpi=200)
    plt.close(fig)
    made.append("fig_cost_by_income")
    return made


# ------------------------------------------------------------------ main
def write_rows(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as fh:
        if rows:
            wr = csv.DictWriter(fh, fieldnames=keys, restval="")
            wr.writeheader()
            wr.writerows(rows)


def md(rows):
    if not rows:
        return "_(no data)_\n"
    keys = list(rows[0])
    s = "| " + " | ".join(keys) + " |\n|" + "---|" * len(keys) + "\n"
    for r in rows:
        s += "| " + " | ".join(str(r.get(k, "")) for k in keys) + " |\n"
    return s


def run(results_dir, run_id=None):
    outdir = os.path.join(results_dir, "analysis")
    os.makedirs(outdir, exist_ok=True)
    run_id, inst, used, evs = load(results_dir, run_id)
    free = [e for e in evs if e.get("stranded") == "no"]
    print(f"analysis of run {run_id}: {len(used)} valid solutions of {len(inst)} instances, {len(evs)} EVs "
          f"({len(free)} with a free choice)", flush=True)
    write_rows(os.path.join(outdir, "ev_level.csv"), evs)
    t_inc = table(free, ["income_band"])
    t_per = table(free, ["period", "income_band"])
    t_lay = table(free, ["layout", "horizon", "income_band"])
    t_size = table(free, ["n", "income_band"])
    t_str = table(evs, ["stranded"])
    for nm, rows in (("by_income", t_inc), ("by_income_period", t_per),
                     ("by_income_layout", t_lay), ("by_income_size", t_size)):
        write_rows(os.path.join(outdir, f"{nm}.csv"), rows)
    lg, lg_info = logit(free) if free else (None, "no data")
    write_rows(os.path.join(outdir, "logit.csv"), lg or [])
    status_count = {}
    for r in inst:
        s = r.get("status", "").split(" (")[0]
        status_count[s] = status_count.get(s, 0) + 1

    def share_band(rows, band, period=None):
        for r in rows:
            if r["income_band"] == band and (period is None or r.get("period") == period):
                return r["share_by_van_%"]
        return ""
    key = {"run_id": run_id, "instances_in_results": len(inst), "valid_solutions": len(used),
           "status_counts": status_count, "EVs": len(evs), "free_choice_EVs": len(free),
           "stranded_EVs": len(evs) - len(free),
           "share_van_free_choice_%": round(100 * mean([e["van"] for e in free]), 1) if free else "",
           "share_van_by_band_%": {b: share_band(t_inc, b) for b in BAND_NAMES},
           "share_van_by_band_peak_%": {b: share_band(t_per, b, "peak") for b in BAND_NAMES},
           "share_van_by_band_offpeak_%": {b: share_band(t_per, b, "off-peak") for b in BAND_NAMES},
           "mean_theta_FCS_users": round(mean([e["theta"] for e in free if e["type"] == "FCS"]), 3)
           if free else "",
           "mean_theta_van_users": round(mean([e["theta"] for e in free if e["type"] == "MCS"]), 3)
           if free else "",
           "logit": lg_info}
    with open(os.path.join(outdir, "key_numbers.json"), "w") as fh:
        json.dump(key, fh, indent=1)
    made = []
    if free:
        try:
            made = figures(outdir, free, lg_info)
        except Exception as e:
            print(f"analysis: figures not drawn ({e})", flush=True)
    with open(os.path.join(outdir, "insights.md"), "w", encoding="utf-8") as fh:
        fh.write("# Income insights (main_formulation_3)\n\n")
        fh.write(f"Valid solutions: {len(used)} of {len(inst)} instances; {len(evs)} EVs, "
                 f"{len(free)} with a free choice (the {len(evs) - len(free)} stranded EVs must take a van).\n\n")
        fh.write(f"Solver status counts: {status_count}\n\n")
        fh.write("## Free-choice EVs by income band\n\n" + md(t_inc))
        fh.write("\n## By period and income band\n\n" + md(t_per))
        fh.write("\n## By layout, horizon and income band\n\n" + md(t_lay))
        fh.write("\n## By instance size and income band\n\n" + md(t_size))
        fh.write("\n## Stranded vs free-choice EVs\n\n" + md(t_str))
        fh.write("\n## Logistic regression: P(served by a van), free-choice EVs\n\n")
        fh.write(md(lg) if lg else f"_not estimated: {lg_info}_\n")
        if lg:
            fh.write(f"\n{json.dumps({k: v for k, v in lg_info.items() if k != 'coef'})}\n")
        if made:
            fh.write("\n## Figures\n\n" + "".join(f"- {m}.pdf / .png\n" for m in made))
    print(f"analysis written to {os.path.abspath(outdir)}", flush=True)
    return key


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "new_results"),
        sys.argv[2] if len(sys.argv) > 2 else None)
