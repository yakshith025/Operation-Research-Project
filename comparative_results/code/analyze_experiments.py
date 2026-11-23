"""Analysis of the comparative experiments: every variant against the baseline, for the MILP, RL1
(the baseline model applied to the variant) and, if run, RL1 retrained on the variant.

Reads   ../results/<variant>/{milp,rl1,rl1_retrained}/  (runs.csv + plan.csv of run_experiments.py)
        baseline: new_results/ (MILP milp_300s, greedy) and the RL1 results/ (rl1)
        variant inputs ../variants/<variant>/ (rebuilt here if missing; make_variants.py)
Writes  ../analysis/
        summary.md          one section per experiment (priority order): its question, answered with
                            the numbers of every method
        levels.csv          one row per experiment level x method: coverage, Z and its terms, van
                            shares, queues, waits, solver status, change vs the baseline, arc elasticities
        coverage.csv        per level x method: instances, runs, valid plans, optimal, infeasible, no plan
        instance_deltas.csv per instance: Z of the baseline and of the variant (paired, valid in both)
        bands.csv           per level x method x income band: van share, customer cost, time cost, paid
        pressure.csv        per level x method x demand-pressure tercile: queue, van share, cost
        decomposition.csv   E1 / E5 / E6: change of Z = price effect (the baseline plan re-scored with
                            the new constant: the envelope-theorem prediction) + re-optimisation effect
        rl_vs_milp.csv      per level: RL1 (and retrained) Z relative to the MILP on common instances
        E2_capacity.csv, E8_battery.csv, E7_reweighting.csv, E3_status.csv   experiment-specific tables
        figures/*.png
Every comparison is PAIRED: only instances with a valid plan at every level of the factor (for that
method) enter the level means, so a change is never a change of the instance mix. MILP numbers are
also given on the instances solved to optimality at every level ("optimal in all").
Works on partial results (levels or instances not run yet are left out and reported).

Usage (from comparative_results/code; needs numpy, pandas, matplotlib)
    python analyze_experiments.py
    python analyze_experiments.py --results <folder> --out <folder> --only EV-C101   (tests)
"""
import argparse
import json
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd

import common as cm
import make_variants as mv

warnings.filterwarnings("ignore", category=RuntimeWarning)

BANDS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)]     # as the comprehensive report
BAND_NAMES = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1"]
NUM = ["Z_total", "Z1_EV_drive_perceived", "Z2_van_drive", "Z3_dispatch", "Z4_operator_energy",
       "Z5_tariff_perceived", "Z6_time_perceived", "operator_cost_Z2_to_Z4", "customer_cost_Z1_Z5_Z6",
       "EVs_at_FCS", "EVs_by_van", "stranded_EVs", "free_choice_EVs", "free_choice_by_van", "vans_used", "van_km",
       "mean_queue_FCS_min", "max_queue_FCS_min", "mean_van_wait_min", "mean_time_to_charged_min",
       "max_time_to_charged_min", "solve_s", "wall_clock_s", "best_bound", "gap_%"]
PLAN_EV = ["run_id", "instance", "n", "type", "id", "resource_type", "resource", "seq", "start_min", "end_min",
           "queue_min", "van_wait_min", "charge_min", "time_to_charged_min", "pay_Rs", "perceived_money_Rs",
           "time_cost_Rs", "customer_cost_Rs", "stops", "km", "kWh_drawn", "battery_left_kWh"]
PLAN_NUM = PLAN_EV[8:]          # numeric columns (seq stays text: plan_from_rows reads it as given)
# colours (dataviz reference palette, light surface)
METHOD_COLOR = {"MILP": "#2a78d6", "RL1": "#eb6834", "RL1 retrained": "#1baf7a", "greedy": "#8a8984"}
SEQ3 = ["#9ec5f4", "#3987e5", "#184f95"]          # ordered levels: low -> high (sequential blue)
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"


# ====================================================================== loading
class Store:
    """Runs and plans of every (level, method), read once."""

    def __init__(self, EXP, args):
        self.EXP, self.args = EXP, args
        S = EXP["solver"]
        rl1 = cm.rl1_dir(args.project, args.rl1_dir)
        self.base = {"MILP": (cm.project_path(EXP["baseline"]["milp_results"], args.project), EXP["baseline"]["milp_run_id"]),
                     "greedy": (cm.project_path(EXP["baseline"]["milp_results"], args.project), "greedy"),
                     "RL1": (os.path.join(rl1, "results"), EXP["baseline"]["rl1_run_id"]),
                     "RL1 retrained": (os.path.join(rl1, "results"), EXP["baseline"]["rl1_run_id"])}
        self.var = {"MILP": ("milp", S["milp_run_id"]), "greedy": ("milp", "greedy"),
                    "RL1": ("rl1", S["rl1_run_id"]), "RL1 retrained": ("rl1_retrained", S["rl1_retrained_run_id"])}
        self._runs, self._plans = {}, {}

    def where(self, level, method):
        if level == cm.BASE_ID:
            return self.base[method]
        sub, rid = self.var[method]
        return os.path.join(self.args.results, level, sub), rid

    def runs(self, level, method):
        folder, rid = self.where(level, method)
        key = (folder, rid)
        if key not in self._runs:
            p = os.path.join(folder, "runs.csv")
            df = None
            if os.path.exists(p):
                df = pd.read_csv(p, dtype=str, keep_default_na=False)
                df = df[df["run_id"] == rid].copy()
                for c in NUM:
                    if c in df:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                df["n"] = df["n"].astype(float).astype(int)
                df = df.drop_duplicates(["instance", "n"], keep="last").set_index(["instance", "n"])
                if self.args.only:
                    df = df[df.index.get_level_values(0).isin(self.args.only)]
                if df.empty:
                    df = None
            self._runs[key] = df
        return self._runs[key]

    def plan(self, level, method):
        folder, rid = self.where(level, method)
        key = (folder, rid)
        if key not in self._plans:
            p = os.path.join(folder, "plan.csv")
            df = None
            if os.path.exists(p):
                df = pd.read_csv(p, dtype=str, keep_default_na=False, usecols=lambda c: c in PLAN_EV)
                df = df[df["run_id"] == rid].copy()
                for c in PLAN_NUM:
                    if c in df:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                df["n"] = df["n"].astype(float).astype(int)
                df["id"] = df["id"].astype(float).astype(int)
            self._plans[key] = df
        return self._plans[key]


def load_attributes(EXP, args, keys):
    """theta, free choice (not stranded) and request load per EV of the baseline instances."""
    sys.path.insert(0, cm.code_dir(args.project))
    import make_ev_solomon as mes
    import make_income as inc
    import results_io as rio
    from solve_milp_f3 import load_instance
    P3 = inc.load_params(cm.project_path(EXP["baseline"]["f3_params"], args.project))
    data = mes.EvSolomonData(cm.project_path(EXP["baseline"]["dataset_csv"], args.project))
    rows = []
    for nm, n in sorted(keys):
        J = load_instance(nm, n, P3, data, greedy=False)
        for a in rio.ev_attributes(J):
            rows.append({"instance": nm, "n": n, "id": a["id"], "theta": a["theta"],
                         "free": a["stranded"] == "no", "load": a["load_at_request"]})
    return pd.DataFrame(rows)


def theta_of_variant(folder):
    """theta per (instance, EV id) from a variant's income folder."""
    out = []
    inc = os.path.join(folder, "income")
    for f in os.listdir(inc):
        if f.endswith(".csv") and f != "summary.csv":
            df = pd.read_csv(os.path.join(inc, f))
            out.append(pd.DataFrame({"instance": f[:-4], "id": df["EV"].astype(int), "theta_v": df["theta"]}))
    return pd.concat(out, ignore_index=True)


# ====================================================================== statistics
def arc_el(y0, y1, x0, x1):
    """Midpoint arc elasticity (% change of y per % change of x)."""
    try:
        dy = (y1 - y0) / ((y1 + y0) / 2)
        dx = (x1 - x0) / ((x1 + x0) / 2)
        return dy / dx if dx else float("nan")
    except ZeroDivisionError:
        return float("nan")


def band_of(theta):
    return np.digitize(theta, [0.2, 0.4, 0.6, 0.8])


def theta_star(P3, cfg, E=20.0, d=2.0, qF=0.0, qM=10.0, x=4.0, dispatch=0.0):
    """Income threshold of the comprehensive report (Sec. 4.1): Delta_i(theta) = extra cost of serving a
    typical free-choice EV (E kWh, d km from its FCS, van wait qM, van detour x km) by a van rather than
    at an FCS; a van is cheaper above theta* = -Delta(0) / (Delta(1) - Delta(0))."""
    vot, tar, mu = P3["value_of_time_Rs_per_min"], P3["tariff_Rs_per_kWh"], P3["money_sensitivity_mu"]

    def line(e, th):
        return e["theta0"] + (e["theta1"] - e["theta0"]) * th

    def delta(th):
        t = 60 * d / cfg["speed"]
        tauF, tauM = 60 * E / cfg["P_F"], 60 * E / cfg["P_M"]
        c = {a: line(vot[a], th) for a in "DQCV"}
        kappa = (cfg["pi_M"] / cfg["eta"] - cfg["pi_F"]) * E + cfg["c_M"] * x + dispatch
        return (line(mu, th) * ((tar["p_M"] - tar["p_F"]) * E - cfg["c_EV"] * d) + kappa - c["D"] * t
                - c["Q"] * qF - c["C"] * tauF + c["V"] * (qM + tauM))
    a = delta(0.0)
    b = delta(1.0) - a
    return -a / b if b else float("nan")


def base_cfg(EXP, args):
    row = pd.read_csv(os.path.join(cm.project_path(EXP["baseline"]["dataset_csv"], args.project),
                                   f"{EXP['instances']['source']}_config.csv")).iloc[0]
    return {"speed": row["speed_kmph"], "P_F": row["station_power_kW"], "P_M": row["van_power_kW"],
            "pi_M": row["pi_M_Rs_per_kWh"], "pi_F": row["pi_F_Rs_per_kWh"], "eta": row["van_efficiency"],
            "c_M": row["c_M_Rs_per_km"], "c_EV": row["c_EV_Rs_per_km"], "Q": float(row["van_max_evs"]),
            "M": float(row["n_vans"]), "B": float(row["van_battery_kWh"]), "E_min": float(_e_min(EXP, args))}


def _e_min(EXP, args):
    """Smallest energy demand of the dataset (a van with less battery left cannot serve any EV)."""
    d = pd.read_csv(os.path.join(cm.project_path(EXP["baseline"]["dataset_csv"], args.project),
                                 f"{EXP['instances']['source']}_data.csv"))
    return pd.to_numeric(d.loc[d["type"] == "ev", "energy_kWh"]).min()


def level_stats(R, ev, vans, method):
    """Means of one level on the paired instances. R: runs (indexed by key), ev: EV rows with
    attributes, vans: van rows."""
    s = {"instances": len(R), "Z_mean": R["Z_total"].mean(), "Z_sum": R["Z_total"].sum(),
         "operator_mean": R["operator_cost_Z2_to_Z4"].mean(), "customer_mean": R["customer_cost_Z1_Z5_Z6"].mean()}
    for c in ("Z1_EV_drive_perceived", "Z2_van_drive", "Z3_dispatch", "Z4_operator_energy", "Z5_tariff_perceived",
              "Z6_time_perceived"):
        s[c.split("_")[0] + "_mean"] = R[c].mean()
    s["van_share_free_%"] = 100 * R["free_choice_by_van"].sum() / max(1, R["free_choice_EVs"].sum())
    s["van_share_all_%"] = 100 * R["EVs_by_van"].sum() / max(1, (R["EVs_by_van"] + R["EVs_at_FCS"]).sum())
    s["vans_used_mean"] = R["vans_used"].mean()
    s["van_km_mean"] = R["van_km"].mean()
    s["max_queue_mean"] = R["max_queue_FCS_min"].mean()
    fcs = ev[ev["resource_type"] == "FCS"]
    mcs = ev[ev["resource_type"] == "MCS"]
    s["queue_mean_min"] = fcs["queue_min"].mean()
    s["queued_share_%"] = 100 * (fcs["queue_min"] > 0.01).mean() if len(fcs) else float("nan")
    s["van_wait_mean_min"] = mcs["van_wait_min"].mean()
    s["time_to_charged_mean_min"] = ev["time_to_charged_min"].mean()
    fv = ev[ev["free"] & (ev["resource_type"] == "MCS")]
    s["theta_van_free_mean"] = fv["theta"].mean()
    s["paid_mean_Rs"] = ev["pay_Rs"].mean()
    s["tariff_revenue_van_Rs"] = mcs["pay_Rs"].sum() / max(1, len(R))
    if len(vans):
        s["stops_per_van"] = vans["stops"].mean()
        s["battery_left_mean_kWh"] = vans["battery_left_kWh"].mean()
    if method == "MILP":
        s["optimal_%"] = 100 * (R["status"] == "optimal").mean()
        s["gap_mean_%"] = R["gap_%"].mean()
    s["solve_s_mean"] = R["solve_s"].mean()
    return s


# ====================================================================== figures
def _style(ax, ylabel="", xlabel=""):
    ax.set_ylabel(ylabel, color=INK2, fontsize=8)
    ax.set_xlabel(xlabel, color=INK2, fontsize=8)
    ax.tick_params(colors=INK2, labelsize=7, length=0)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)


def fig_overview(plt, L, ex, fac, methods, path):
    """Change of Z and van share across the levels of one factor, one line per method."""
    d = L[(L["experiment"] == ex["id"]) & (L["factor"] == fac["label"])]
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for m in methods:
        g = d[d["method"] == m].sort_values("x")
        if g.empty:
            continue
        kw = dict(color=METHOD_COLOR[m], lw=1.6, marker="o", ms=5, mec="#fcfcfb", mew=1.2, label=m)
        axes[0].plot(g["value_text"], g["dZ_pooled_%"], **kw)
        axes[1].plot(g["value_text"], g["van_share_free_%"], **kw)
    axes[0].axhline(0, color=AXIS, lw=0.8)
    _style(axes[0], "change of Z vs baseline (%)", f"{fac['label']} ({fac['unit']})")
    _style(axes[1], "free-choice EVs served by van (%)", f"{fac['label']} ({fac['unit']})")
    axes[1].legend(fontsize=7, frameon=False)
    fig.suptitle(f"{ex['id']}: {ex['name']}", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_bands(plt, Bd, ex, fac, methods, path, col="van_share_%", ylabel="free-choice EVs served by van (%)"):
    d = Bd[(Bd["experiment"] == ex["id"]) & (Bd["factor"] == fac["label"])]
    ms = [m for m in methods if m in set(d["method"])]
    if d.empty or not ms:
        return
    fig, axes = plt.subplots(1, len(ms), figsize=(3.4 * len(ms), 2.7), squeeze=False)
    for ax, m in zip(axes[0], ms):
        g = d[d["method"] == m]
        lv = g.drop_duplicates("level").sort_values("x")
        cols = SEQ3 if len(lv) == 3 else [SEQ3[0], SEQ3[2]] if len(lv) == 2 else SEQ3[:len(lv)] + ["#0d366b"] * 9
        for c, (_, row) in zip(cols, lv.iterrows()):
            h = g[g["level"] == row["level"]].sort_values("band")
            ax.plot(h["band_name"], h[col], color=c, lw=1.6, marker="o", ms=5, mec="#fcfcfb", mew=1.2,
                    label=row["value_text"] + (" (base)" if row["level"] == cm.BASE_ID else ""))
        _style(ax, ylabel if ax is axes[0][0] else "", "income band (theta)")
        ax.set_title(m, fontsize=8, color=INK)
        ax.legend(fontsize=6.5, frameon=False)
    fig.suptitle(f"{ex['id']}: {ex['name']}", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_pressure(plt, Pd, ex, fac, methods, path):
    d = Pd[(Pd["experiment"] == ex["id"]) & (Pd["factor"] == fac["label"])]
    ms = [m for m in methods if m in set(d["method"])]
    if d.empty or not ms:
        return
    fig, axes = plt.subplots(1, len(ms), figsize=(3.4 * len(ms), 2.7), squeeze=False)
    for ax, m in zip(axes[0], ms):
        g = d[d["method"] == m]
        lv = g.drop_duplicates("level").sort_values("x")
        cols = SEQ3 if len(lv) == 3 else [SEQ3[0], SEQ3[2]]
        w = 0.8 / max(1, len(lv))
        for k, (c, (_, row)) in enumerate(zip(cols, lv.iterrows())):
            h = g[g["level"] == row["level"]].set_index("pressure").reindex(["low", "mid", "high"])
            ax.bar(np.arange(3) + (k - (len(lv) - 1) / 2) * w, h["queue_mean_min"], width=w * 0.92, color=c,
                   label=row["value_text"] + (" (base)" if row["level"] == cm.BASE_ID else ""))
        ax.set_xticks(range(3))
        ax.set_xticklabels(["low", "mid", "high"])
        _style(ax, "mean FCS queue (min)" if ax is axes[0][0] else "", "demand pressure at request")
        ax.set_title(m, fontsize=8, color=INK)
        ax.legend(fontsize=6.5, frameon=False)
    fig.suptitle(f"{ex['id']}: {ex['name']}", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_elasticities(plt, El, methods, path):
    d = El[El["method"].isin(methods)].dropna(subset=["el_Z"])
    if d.empty:
        return
    labels = list(dict.fromkeys(d["level"]))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 0.32 * len(labels) + 1.2), sharey=True)
    ms = [m for m in methods if m in set(d["method"])]
    h = 0.8 / len(ms)
    for k, m in enumerate(ms):
        g = d[d["method"] == m].set_index("level").reindex(labels)
        y = np.arange(len(labels)) + (k - (len(ms) - 1) / 2) * h
        axes[0].barh(y, g["el_Z"], height=h * 0.92, color=METHOD_COLOR[m], label=m)
        axes[1].barh(y, g["el_van_share"], height=h * 0.92, color=METHOD_COLOR[m], label=m)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].invert_yaxis()
    for ax, t in ((axes[0], "arc elasticity of Z"), (axes[1], "arc elasticity of the van share")):
        ax.axvline(0, color=AXIS, lw=0.8)
        _style(ax, "", t)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.grid(axis="y", visible=False)
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ====================================================================== main analysis
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=cm.DEFAULT_RESULTS)
    ap.add_argument("--variants-dir", default=cm.DEFAULT_VARIANTS)
    ap.add_argument("--out", default=cm.DEFAULT_ANALYSIS)
    ap.add_argument("--project", default=cm.PROJECT)
    ap.add_argument("--rl1-dir")
    ap.add_argument("--only", nargs="*", help="only these files (tests)")
    ap.add_argument("--experiments", nargs="*", help="only these experiments")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    args.results, args.out = os.path.abspath(args.results), os.path.abspath(args.out)
    EXP = cm.load_experiments()
    st = Store(EXP, args)
    fig_dir = os.path.join(args.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    sizes = set(EXP["instances"]["sizes"])
    sys.path.insert(0, cm.code_dir(args.project))
    import make_income as inc
    import results_io as rio
    import make_ev_solomon as mes
    from solve_milp_f3 import load_instance

    base_milp = st.runs(cm.BASE_ID, "MILP")
    if base_milp is None:
        raise SystemExit("baseline MILP runs not found (new_results/runs.csv)")
    universe_all = {k for k in base_milp.index if k[1] in sizes}
    print(f"baseline: {len(universe_all)} instances; reading the attributes of every EV ...", flush=True)
    A = load_attributes(EXP, args, universe_all)
    t1, t2 = np.quantile(A.loc[A["free"], "load"], [1 / 3, 2 / 3])
    A["pressure"] = np.where(A["load"] <= t1, "low", np.where(A["load"] <= t2, "mid", "high"))
    cfg = base_cfg(EXP, args)
    P3_base = inc.load_params(cm.project_path(EXP["baseline"]["f3_params"], args.project))
    methods_all = ["MILP", "RL1", "RL1 retrained", "greedy"]
    variants = {v["id"]: v for v in cm.variants(EXP)}

    out = {k: [] for k in ("levels", "coverage", "deltas", "bands", "pressure", "decomp", "rlgap", "E2", "E8",
                           "E7", "E3")}
    exps = sorted(EXP["experiments"], key=lambda e: e["priority"])
    if args.experiments:
        exps = [e for e in exps if e["id"].upper() in {x.upper() for x in args.experiments}]
    folders = {}
    for ex in exps:
        fams = {f.upper() for f in ex.get("families") or []}
        universe = {k for k in universe_all if not fams or mes.family_of(k[0]).upper() in fams}
        for fac in ex["factors"]:
            levels = [(cm.BASE_ID, fac["baseline"])]
            for val in fac["values"]:
                levels.append((f"{ex['id']}_{fac['label']}_{cm.value_tag(fac['kind'], val)}", val))
            for vid, _ in levels[1:]:
                try:
                    folders[vid] = mv.build(variants[vid], EXP, args.variants_dir, args.project, quiet=True)
                except Exception as e:
                    print(f"{vid}: inputs not available ({e})", flush=True)
            for method in methods_all:
                runs = {lv: st.runs(lv, method) for lv, _ in levels}
                avail = [(lv, val) for lv, val in levels if runs[lv] is not None]
                if len(avail) < 2 or avail[0][0] != cm.BASE_ID:
                    continue
                # coverage (all instances of the universe)
                for lv, val in avail:
                    R = runs[lv]
                    R = R[R.index.isin(universe)]
                    stc = R["status"].value_counts()
                    out["coverage"].append({
                        "experiment": ex["id"], "factor": fac["label"], "level": lv,
                        "value": cm.value_text(fac["kind"], val), "method": method, "instances_in_set": len(universe),
                        "runs": len(R), "valid": int((R["valid"] == "yes").sum()), "invalid": int((R["valid"] == "no").sum()),
                        "optimal": int(stc.get("optimal", 0)), "time_limit": int(stc.get("time_limit", 0)),
                        "infeasible": int(stc.get("infeasible", 0)),
                        "no_plan": int(sum(c for s, c in stc.items() if s.endswith("no_plan"))),
                        "error": int(stc.get("error", 0))})
                valid_sets = [{k for k in runs[lv].index[runs[lv]["valid"] == "yes"] if k in universe} for lv, _ in avail]
                paired = set.intersection(*valid_sets)
                opt_all = None
                if method == "MILP":
                    opt_all = set.intersection(*[{k for k in runs[lv].index[runs[lv]["status"] == "optimal"]}
                                                 for lv, _ in avail]) & paired
                if not paired:
                    continue
                keys = sorted(paired)
                stats = {}
                for lv, val in avail:
                    R = runs[lv].loc[keys]
                    P = st.plan(lv, method)
                    P = P[P.set_index(["instance", "n"]).index.isin(paired)] if P is not None else pd.DataFrame(columns=PLAN_EV)
                    ev = P[P["type"] == "ev"].merge(A, on=["instance", "n", "id"], how="left")
                    if fac["kind"] == "theta_beta" and lv != cm.BASE_ID and lv in folders:
                        ev = ev.merge(theta_of_variant(folders[lv]), on=["instance", "id"], how="left")
                        ev["theta"] = ev["theta_v"].fillna(ev["theta"])
                    ev["band"] = band_of(ev["theta"].astype(float))
                    vans = P[P["type"] == "van"]
                    s = level_stats(R, ev, vans, method)
                    if opt_all:
                        s["Z_mean_optimal_in_all"] = runs[lv].loc[sorted(opt_all), "Z_total"].mean()
                        s["instances_optimal_in_all"] = len(opt_all)
                    P3 = P3_base if lv == cm.BASE_ID else inc.load_params(os.path.join(folders[lv], "f3_params.json"))
                    s["theta_star_theory"] = theta_star(P3, cfg)
                    free_ev = ev[ev["free"].astype(bool)]
                    stats[lv] = (val, s, R, ev, vans)
                    for b in range(5):
                        g = free_ev[free_ev["band"] == b]
                        ga = ev[ev["band"] == b]
                        out["bands"].append({
                            "experiment": ex["id"], "factor": fac["label"], "level": lv, "x": cm.numeric(fac["kind"], val),
                            "value_text": cm.value_text(fac["kind"], val), "method": method, "band": b,
                            "band_name": BAND_NAMES[b], "free_EVs": len(g),
                            "van_share_%": 100 * (g["resource_type"] == "MCS").mean() if len(g) else float("nan"),
                            "customer_cost_mean_Rs": ga["customer_cost_Rs"].mean(),
                            "time_cost_mean_Rs": ga["time_cost_Rs"].mean(), "paid_mean_Rs": ga["pay_Rs"].mean(),
                            "time_to_charged_mean_min": ga["time_to_charged_min"].mean()})
                    for pr in ("low", "mid", "high"):
                        g = ev[ev["pressure"] == pr]
                        gf = g[g["resource_type"] == "FCS"]
                        gfree = g[g["free"].astype(bool)]
                        out["pressure"].append({
                            "experiment": ex["id"], "factor": fac["label"], "level": lv, "x": cm.numeric(fac["kind"], val),
                            "value_text": cm.value_text(fac["kind"], val), "method": method, "pressure": pr,
                            "EVs": len(g), "queue_mean_min": gf["queue_min"].mean(),
                            "queue_p90_min": gf["queue_min"].quantile(0.9) if len(gf) else float("nan"),
                            "queued_share_%": 100 * (gf["queue_min"] > 0.01).mean() if len(gf) else float("nan"),
                            "van_share_free_%": 100 * (gfree["resource_type"] == "MCS").mean() if len(gfree) else float("nan"),
                            "customer_cost_mean_Rs": g["customer_cost_Rs"].mean()})
                b_val, b_s, b_R, b_ev, b_vans = stats[cm.BASE_ID]
                x0 = cm.numeric(fac["kind"], b_val)
                for lv, (val, s, R, ev, vans) in stats.items():
                    x1 = cm.numeric(fac["kind"], val)
                    dz = 100 * (R["Z_total"] / b_R["Z_total"] - 1)
                    row = {"experiment": ex["id"], "priority": ex["priority"], "factor": fac["label"], "level": lv,
                           "value_text": cm.value_text(fac["kind"], val), "x": x1, "method": method,
                           "paired_instances": len(keys), **s,
                           "dZ_pooled_%": 100 * (s["Z_sum"] / b_s["Z_sum"] - 1), "dZ_mean_%": dz.mean(),
                           "dZ_median_%": dz.median(), "dZ_min_%": dz.min(), "dZ_max_%": dz.max(),
                           "d_operator_Rs": s["operator_mean"] - b_s["operator_mean"],
                           "d_customer_Rs": s["customer_mean"] - b_s["customer_mean"],
                           "d_van_share_pp": s["van_share_free_%"] - b_s["van_share_free_%"]}
                    if lv != cm.BASE_ID:
                        row["el_Z"] = arc_el(b_s["Z_sum"], s["Z_sum"], x0, x1)
                        row["el_van_share"] = arc_el(b_s["van_share_free_%"], s["van_share_free_%"], x0, x1)
                        if "Z_mean_optimal_in_all" in s:
                            row["dZ_optimal_in_all_%"] = 100 * (s["Z_mean_optimal_in_all"] / b_s["Z_mean_optimal_in_all"] - 1)
                        for k in keys:
                            out["deltas"].append({"experiment": ex["id"], "factor": fac["label"], "level": lv,
                                                  "method": method, "instance": k[0], "n": k[1],
                                                  "class": mes.family_of(k[0]), "Z_base": b_R.loc[k, "Z_total"],
                                                  "Z_variant": R.loc[k, "Z_total"],
                                                  "dZ_%": 100 * (R.loc[k, "Z_total"] / b_R.loc[k, "Z_total"] - 1),
                                                  "status_base": b_R.loc[k, "status"], "status_variant": R.loc[k, "status"],
                                                  "vans_used_base": b_R.loc[k, "vans_used"],
                                                  "vans_used_variant": R.loc[k, "vans_used"]})
                    out["levels"].append(row)

                # ---------------------------------------------------------------- experiment specifics
                if fac["kind"] == "param" and method != "greedy":
                    bplan = st.plan(cm.BASE_ID, method)
                    bplan = bplan[bplan.set_index(["instance", "n"]).index.isin(paired)]
                    groups = {k: g for k, g in bplan.groupby(["instance", "n"])}
                    for lv, (val, s, R, ev, vans) in stats.items():
                        if lv == cm.BASE_ID:
                            continue
                        P3v = inc.load_params(os.path.join(folders[lv], "f3_params.json"))
                        data_v = mes.EvSolomonData(os.path.join(folders[lv], "csv"))
                        zp, bad = [], 0
                        for k in keys:
                            J = load_instance(k[0], k[1], P3v, data_v, greedy=False)
                            m, _, err = rio.evaluate(J, rio.plan_from_rows(groups[k].to_dict("records"), k[1]))
                            bad += bool(err)
                            zp.append(m["Z_exact"])
                        zb, zv, zp = b_R["Z_total"].sum(), R["Z_total"].sum(), float(np.sum(zp))
                        out["decomp"].append({
                            "experiment": ex["id"], "factor": fac["label"], "level": lv,
                            "value_text": cm.value_text(fac["kind"], val), "method": method, "paired_instances": len(keys),
                            "Z_base_mean": zb / len(keys), "Z_base_plan_at_new_constant_mean": zp / len(keys),
                            "Z_resolved_mean": zv / len(keys), "dZ_total_%": 100 * (zv / zb - 1),
                            "price_effect_%": 100 * (zp / zb - 1), "reoptimisation_effect_%": 100 * (zv - zp) / zb,
                            "el_Z_envelope": arc_el(zb, zp, x0, cm.numeric(fac["kind"], val)),
                            "el_Z_resolved": arc_el(zb, zv, x0, cm.numeric(fac["kind"], val)),
                            "van_share_base_%": b_s["van_share_free_%"], "van_share_resolved_%": s["van_share_free_%"],
                            "base_plans_invalid_at_new_constant": bad})
                if ex["id"] == "E2" and method != "greedy":
                    for lv, (val, s, R, ev, vans) in stats.items():
                        base_of = {f_["key"]: float(f_["baseline"]) for f_ in ex["factors"]}
                        cap = float(val) if fac["key"] == "van_max_evs" else base_of.get("van_max_evs", cfg["Q"])
                        M = float(val) if fac["key"] == "n_vans" else base_of.get("n_vans", cfg["M"])
                        vv = vans.copy()
                        vv["class"] = vv["instance"].map(mes.family_of)
                        RR = R.reset_index()
                        RR["class"] = RR["instance"].map(mes.family_of)
                        bR = b_R.reset_index()
                        for cl, g in RR.groupby("class"):
                            gb = bR[bR["instance"].map(mes.family_of) == cl]
                            gv = vv[vv["class"] == cl]
                            dz_rs = g["Z_total"].mean() - gb["Z_total"].mean()
                            step = (float(val) - float(fac["baseline"]))
                            out["E2"].append({
                                "factor": fac["label"], "level": lv, "value_text": cm.value_text(fac["kind"], val),
                                "method": method, "class": cl, "instances": len(g), "Z_mean": g["Z_total"].mean(),
                                "dZ_Rs_per_instance": dz_rs, "dZ_%": 100 * (g["Z_total"].sum() / gb["Z_total"].sum() - 1),
                                "dZ_Rs_per_unit_added": dz_rs / step if step else float("nan"),
                                "vans_used_mean": g["vans_used"].mean(), "vans_available": M,
                                "fleet_use_%": 100 * g["vans_used"].mean() / M,
                                "stops_per_used_van": gv["stops"].mean(),
                                "vans_at_seat_cap_%": 100 * (gv["stops"] >= cap).mean() if len(gv) else float("nan"),
                                "battery_left_mean_kWh": gv["battery_left_kWh"].mean(),
                                "van_share_free_%": 100 * g["free_choice_by_van"].sum() / max(1, g["free_choice_EVs"].sum())})
                if ex["id"] == "E8" and method != "greedy":
                    thr = float(ex.get("decision_threshold_pct", 1.0))
                    for lv, (val, s, R, ev, vans) in stats.items():
                        if lv == cm.BASE_ID:
                            continue
                        dz = 100 * (R["Z_total"] / b_R["Z_total"] - 1)
                        bv = b_vans
                        out["E8"].append({
                            "level": lv, "value_text": cm.value_text(fac["kind"], val), "method": method,
                            "paired_instances": len(keys), "dZ_pooled_%": 100 * (R["Z_total"].sum() / b_R["Z_total"].sum() - 1),
                            "instances_Z_moves_over_0.1%": int((dz.abs() > 0.1).sum()),
                            "instances_Z_moves_over_threshold": int((dz.abs() > thr).sum()), "threshold_%": thr,
                            "base_used_vans": len(bv), "base_mean_kWh_drawn": bv["kWh_drawn"].mean(),
                            "base_vans_battery_nearly_empty_%": 100 * (bv["battery_left_kWh"] < cfg["E_min"] / cfg["eta"]).mean() if len(bv) else float("nan"),
                            "base_vans_full_of_EVs_%": 100 * (bv["stops"] >= cfg["Q"]).mean() if len(bv) else float("nan"),
                            "van_share_base_%": b_s["van_share_free_%"], "van_share_variant_%": s["van_share_free_%"],
                            "verdict": ("Z barely moves: van recharging need not be modelled"
                                        if abs(100 * (R["Z_total"].sum() / b_R["Z_total"].sum() - 1)) < thr
                                        else "Z moves: the battery binds, recharging is worth modelling")})
                if ex["id"] == "E7" and method != "greedy":
                    bf = b_ev[b_ev["free"].astype(bool)]
                    dec = np.minimum((bf["theta"].astype(float) * 10).astype(int), 9)
                    prop = bf.groupby(dec)["resource_type"].apply(lambda s_: (s_ == "MCS").mean()).reindex(range(10)).fillna(0)
                    for lv, (val, s, R, ev, vans) in stats.items():
                        a_, b_ = (val if isinstance(val, (list, tuple)) else fac["baseline"])
                        w = np.array([mv.beta_cdf((d_ + 1) / 10, a_, b_) - mv.beta_cdf(d_ / 10, a_, b_) for d_ in range(10)])
                        est = 100 * float(np.sum(w * prop.values))
                        f_ = ev[ev["free"].astype(bool)]
                        decv = np.minimum((f_["theta"].astype(float) * 10).astype(int), 9)
                        propv = f_.groupby(decv)["resource_type"].apply(lambda s_: (s_ == "MCS").mean()).reindex(range(10))
                        out["E7"].append({
                            "level": lv, "value_text": cm.value_text(fac["kind"], val), "method": method,
                            "paired_instances": len(keys), "mean_theta": f_["theta"].astype(float).mean(),
                            "table25_estimate_%": est, "resolved_van_share_%": s["van_share_free_%"],
                            "resolved_minus_estimate_pp": s["van_share_free_%"] - est,
                            **{f"decile{d_}_van_%": 100 * propv.get(d_, float("nan")) for d_ in range(10)}})
                if ex["id"] == "E3" and method != "greedy":
                    bR = runs[cm.BASE_ID]
                    bR = bR[bR.index.isin(universe)]
                    for lv, val in avail[1:]:
                        R = runs[lv]
                        R = R[R.index.isin(universe)]
                        common_keys = sorted(set(bR.index) & set(R.index))
                        tr = pd.crosstab(bR.loc[common_keys, "status"], R.loc[common_keys, "status"])
                        for sb in tr.index:
                            for sv in tr.columns:
                                if tr.loc[sb, sv]:
                                    out["E3"].append({"level": lv, "value_text": cm.value_text(fac["kind"], val),
                                                      "method": method, "status_base": sb, "status_variant": sv,
                                                      "instances": int(tr.loc[sb, sv])})
            # RL vs MILP per level (instances valid for both)
            for lv, val in levels:
                Rm = st.runs(lv, "MILP")
                for m in ("RL1", "RL1 retrained"):
                    Rr = st.runs(lv, m)
                    if Rm is None or Rr is None or (m == "RL1 retrained" and lv == cm.BASE_ID):
                        continue
                    km = {k for k in Rm.index[Rm["valid"] == "yes"] if k in universe}
                    kr = {k for k in Rr.index[Rr["valid"] == "yes"] if k in universe}
                    both = sorted(km & kr)
                    if not both:
                        continue
                    zm, zr = Rm.loc[both, "Z_total"], Rr.loc[both, "Z_total"]
                    opt = [k for k in both if Rm.loc[k, "status"] == "optimal"]
                    out["rlgap"].append({
                        "experiment": ex["id"], "factor": fac["label"], "level": lv,
                        "value_text": cm.value_text(fac["kind"], val), "method": m,
                        "RL_valid": len(kr), "MILP_valid": len(km), "RL_valid_MILP_without_plan": len(kr - km),
                        "common": len(both), "RL_over_MILP_pooled_%": 100 * (zr.sum() / zm.sum() - 1),
                        "RL_over_MILP_mean_%": (100 * (zr / zm - 1)).mean(),
                        "RL_over_MILP_on_MILP_optima_%": 100 * (Rr.loc[opt, "Z_total"].sum() / Rm.loc[opt, "Z_total"].sum() - 1) if opt else float("nan"),
                        "RL_solve_s_mean": Rr.loc[both, "solve_s"].mean(), "MILP_solve_s_mean": Rm.loc[both, "solve_s"].mean()})

    # ---------------------------------------------------------------- write tables
    tabs = {k: pd.DataFrame(v) for k, v in out.items()}
    names = {"levels": "levels.csv", "coverage": "coverage.csv", "deltas": "instance_deltas.csv", "bands": "bands.csv",
             "pressure": "pressure.csv", "decomp": "decomposition.csv", "rlgap": "rl_vs_milp.csv",
             "E2": "E2_capacity.csv", "E8": "E8_battery.csv", "E7": "E7_reweighting.csv", "E3": "E3_status.csv"}
    if not tabs["rlgap"].empty:
        tabs["rlgap"] = tabs["rlgap"].drop_duplicates(["level", "method"])
    for k, df in tabs.items():
        if not df.empty:
            df.round(4).to_csv(os.path.join(args.out, names[k]), index=False)
    L = tabs["levels"]
    if L.empty:
        print("no experiment has results yet (nothing to compare)")
        return
    El = L[L["level"] != cm.BASE_ID][["experiment", "factor", "level", "value_text", "method", "el_Z", "el_van_share",
                                       "dZ_pooled_%", "d_van_share_pp"]]
    El.round(4).to_csv(os.path.join(args.out, "elasticities.csv"), index=False)

    # ---------------------------------------------------------------- figures
    plot_methods = [m for m in ("MILP", "RL1", "RL1 retrained") if m in set(L["method"])]
    if not args.no_figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for ex in exps:
            for fac in ex["factors"]:
                tag = f"{ex['id']}_{fac['label']}"
                fig_overview(plt, L, ex, fac, plot_methods, os.path.join(fig_dir, f"{tag}_overview.png"))
                fig_bands(plt, tabs["bands"], ex, fac, plot_methods, os.path.join(fig_dir, f"{tag}_van_share_by_income.png"))
                if ex["id"] in ("E1", "E4", "E5", "E6", "E7"):
                    fig_bands(plt, tabs["bands"], ex, fac, plot_methods,
                              os.path.join(fig_dir, f"{tag}_customer_cost_by_income.png"),
                              col="customer_cost_mean_Rs", ylabel="mean perceived customer cost (Rs)")
                if ex["id"] in ("E4", "E2"):
                    fig_pressure(plt, tabs["pressure"], ex, fac, plot_methods, os.path.join(fig_dir, f"{tag}_queue_by_pressure.png"))
        fig_elasticities(plt, El, [m for m in plot_methods if m != "greedy"], os.path.join(fig_dir, "elasticities.png"))

    write_summary(args, EXP, exps, tabs, El)
    print(f"analysis written to {args.out}", flush=True)


# ====================================================================== summary.md
def f1(x, fmt="{:.1f}"):
    try:
        return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else fmt.format(x)
    except (TypeError, ValueError):
        return str(x)


def write_summary(args, EXP, exps, T, El):
    L, C = T["levels"], T["coverage"]
    lines = ["# Comparative experiments: summary", "",
             "Generated by `analyze_experiments.py`. Every variant changes one input; the baseline is the finished run "
             "(MILP `milp_300s` in `new_results/`, RL1 `rl1`). Numbers are **paired**: means over the instances with a "
             "valid plan at every level of the factor for that method. Van share = free-choice EVs (an FCS in range) "
             "served by a van. Arc elasticity = % change of the quantity per % change of the input (midpoint formula). "
             "RL1 = the baseline model applied to the variant (not retrained); RL1 retrained = a model trained on the "
             "variant's inputs (only if that method was run).", ""]
    for ex in exps:
        d = L[L["experiment"] == ex["id"]]
        lines += [f"## {ex['id']} (experiment {ex.get('number')}, priority {ex['priority']}): {ex['name']}", "",
                  f"*Question:* {ex.get('question', '')}", ""]
        if d.empty:
            lines += ["No results yet.", ""]
            continue
        for fac in ex["factors"]:
            g = d[d["factor"] == fac["label"]]
            if g.empty:
                continue
            lines += [f"**{fac['label']} ({fac['unit']}), baseline {cm.value_text(fac['kind'], fac['baseline'])}**", "",
                      "| method | value | paired inst. | mean Z (Rs) | dZ pooled | dZ median | van share (free) | "
                      "vans used | mean FCS queue (min) | mean van wait (min) | el. Z | el. van share |",
                      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for _, r in g[g["method"] != "greedy"].sort_values(["method", "x"]).iterrows():
                lines.append(f"| {r['method']} | {r['value_text']}{' (base)' if r['level'] == cm.BASE_ID else ''} | "
                             f"{r['paired_instances']} | {f1(r['Z_mean'], '{:,.0f}')} | {f1(r['dZ_pooled_%'], '{:+.2f} %')} | "
                             f"{f1(r['dZ_median_%'], '{:+.2f} %')} | {f1(r['van_share_free_%'])} % | {f1(r['vans_used_mean'], '{:.2f}')} | "
                             f"{f1(r['queue_mean_min'])} | {f1(r['van_wait_mean_min'])} | {f1(r.get('el_Z'), '{:.2f}')} | "
                             f"{f1(r.get('el_van_share'), '{:.2f}')} |")
            lines.append("")
            cv = C[(C["experiment"] == ex["id"]) & (C["factor"] == fac["label"]) & (C["method"] == "MILP")]
            if not cv.empty:
                lines.append("MILP status per level (all instances of the set): " + "; ".join(
                    f"{r['value']}: {r['optimal']} optimal, {r['time_limit']} time limit, {r['infeasible']} infeasible, "
                    f"{r['no_plan']} no plan ({r['runs']}/{r['instances_in_set']} solved so far)" for _, r in cv.iterrows()) + ".")
                lines.append("")
            # the answer, worded from the numbers
            for m in ("MILP", "RL1", "RL1 retrained"):
                h = g[g["method"] == m].sort_values("x")
                if len(h) < 2:
                    continue
                parts = [f"{r['value_text']}: Z {f1(r['dZ_pooled_%'], '{:+.2f}')} %, van share {f1(r['van_share_free_%'])} %"
                         for _, r in h.iterrows() if r["level"] != cm.BASE_ID]
                b = h[h["level"] == cm.BASE_ID].iloc[0]
                lines.append(f"- **{m}**: baseline van share {f1(b['van_share_free_%'])} %; " + "; ".join(parts) + ".")
            lines.append("")
        lines += extra_lines(ex, T)
    rg = T["rlgap"]
    if not rg.empty:
        rg = rg.drop_duplicates(["level", "method"])          # the baseline once, not once per experiment
        lines += ["## RL1 against the MILP at every level", "",
                  "| experiment | value | method | common inst. | RL valid (MILP none) | RL / MILP pooled | on MILP optima | "
                  "RL s | MILP s |", "|---|---|---|---|---|---|---|---|---|"]
        for _, r in rg.iterrows():
            lines.append(f"| {'baseline' if r['level'] == cm.BASE_ID else r['experiment']} | "
                         f"{r['value_text'] if r['level'] != cm.BASE_ID else 'all baseline values'} | {r['method']} | "
                         f"{r['common']} | {r['RL_valid']} ({r['RL_valid_MILP_without_plan']}) | "
                         f"{f1(r['RL_over_MILP_pooled_%'], '{:+.2f} %')} | {f1(r['RL_over_MILP_on_MILP_optima_%'], '{:+.2f} %')} | "
                         f"{f1(r['RL_solve_s_mean'], '{:.2f}')} | {f1(r['MILP_solve_s_mean'], '{:.0f}')} |")
        lines.append("")
    if not El.empty:
        e = El[El["method"] == "MILP"].copy()
        if not e.empty:
            e["abs"] = e["el_Z"].abs()
            lines += ["## Ranking of the inputs (MILP, measured by re-solving)", "",
                      "| variant | arc el. of Z | arc el. of van share | dZ pooled | van share change (pp) |", "|---|---|---|---|---|"]
            for _, r in e.sort_values("abs", ascending=False).iterrows():
                lines.append(f"| {r['level']} | {f1(r['el_Z'], '{:.3f}')} | {f1(r['el_van_share'], '{:.2f}')} | "
                             f"{f1(r['dZ_pooled_%'], '{:+.2f} %')} | {f1(r['d_van_share_pp'], '{:+.2f}')} |")
            lines.append("")
    lines += ["Figures: `figures/` (overview = change of Z and van share per level; van share and customer cost by "
              "income band; queue by demand pressure; elasticities). Tables: the CSV files next to this summary.", ""]
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def extra_lines(ex, T):
    out = []
    if ex["factors"][0]["kind"] == "param" and not T["decomp"].empty:
        d = T["decomp"][T["decomp"]["experiment"] == ex["id"]]
        if not d.empty:
            out += ["Price effect vs re-optimisation (the baseline plan re-scored with the new constant = the "
                    "envelope-theorem prediction; the rest is what re-solving saves):", "",
                    "| method | value | dZ total | price effect (same plan) | re-optimisation | el. Z envelope | el. Z re-solved | "
                    "van share base -> re-solved |", "|---|---|---|---|---|---|---|---|"]
            for _, r in d.iterrows():
                out.append(f"| {r['method']} | {r['value_text']} | {f1(r['dZ_total_%'], '{:+.2f} %')} | "
                           f"{f1(r['price_effect_%'], '{:+.2f} %')} | {f1(r['reoptimisation_effect_%'], '{:+.2f} %')} | "
                           f"{f1(r['el_Z_envelope'], '{:.3f}')} | {f1(r['el_Z_resolved'], '{:.3f}')} | "
                           f"{f1(r['van_share_base_%'])} -> {f1(r['van_share_resolved_%'])} % |")
            out.append("")
        L = T["levels"]
        d = L[(L["experiment"] == ex["id"]) & (L["method"] == "MILP")].sort_values("x")
        if not d.empty:
            out.append("Income threshold theta* of the theory (typical EV, Sec. 4.1): " + ", ".join(
                f"{r['value_text']}: {f1(r['theta_star_theory'], '{:.2f}')}" for _, r in d.iterrows())
                + "; mean theta of free-choice van users (MILP): " + ", ".join(
                f"{r['value_text']}: {f1(r['theta_van_free_mean'], '{:.2f}')}" for _, r in d.iterrows()) + ".")
            out.append("")
    if ex["id"] == "E1" and not T["bands"].empty:
        b = T["bands"][(T["bands"]["experiment"] == "E1") & (T["bands"]["method"] == "MILP")]
        if not b.empty:
            out += ["Who gets vans and what customers bear, by income band (MILP):", "",
                    "| p^M | " + " | ".join(f"van % {n}" for n in BAND_NAMES) + " | " +
                    " | ".join(f"cost {n}" for n in BAND_NAMES) + " |", "|---|" + "---|" * 10]
            for lv, g in b.sort_values("x").groupby("x", sort=True):
                g = g.sort_values("band")
                out.append(f"| {g['value_text'].iloc[0]} | " + " | ".join(f1(v) for v in g["van_share_%"]) + " | "
                           + " | ".join(f1(v, '{:.0f}') for v in g["customer_cost_mean_Rs"]) + " |")
            out.append("")
            L = T["levels"]
            d = L[(L["experiment"] == "E1") & (L["method"] == "MILP")].sort_values("x")
            out.append("Operator cost Z2-Z4 per instance (MILP): " + ", ".join(
                f"p^M {r['value_text']}: {f1(r['operator_mean'], '{:,.0f}')} Rs (tariff paid for vans "
                f"{f1(r['tariff_revenue_van_Rs'], '{:,.0f}')} Rs)" for _, r in d.iterrows()) + ".")
            out.append("")
    if ex["id"] == "E2" and not T["E2"].empty:
        d = T["E2"][T["E2"]["method"] == "MILP"]
        if not d.empty:
            out += ["Capacity by Solomon class (MILP): change of Z per instance and per van / seat added, fleet use, "
                    "full vans:", "", "| factor | value | class | inst. | dZ Rs/inst. | Rs per unit added | vans used / avail. | "
                    "stops per van | vans at seat cap |", "|---|---|---|---|---|---|---|---|---|"]
            for _, r in d.sort_values(["factor", "level", "class"]).iterrows():
                out.append(f"| {r['factor']} | {r['value_text']} | {r['class']} | {r['instances']} | "
                           f"{f1(r['dZ_Rs_per_instance'], '{:+,.0f}')} | {f1(r['dZ_Rs_per_unit_added'], '{:+,.0f}')} | "
                           f"{f1(r['vans_used_mean'], '{:.2f}')} / {r['vans_available']:g} | {f1(r['stops_per_used_van'], '{:.2f}')} | "
                           f"{f1(r['vans_at_seat_cap_%'])} % |")
            out.append("")
    if ex["id"] == "E8" and not T["E8"].empty:
        for _, r in T["E8"].iterrows():
            out.append(f"- **{r['method']}**, B = {r['value_text']} kWh: Z {f1(r['dZ_pooled_%'], '{:+.3f}')} % pooled; "
                       f"{r['instances_Z_moves_over_0.1%']} of {r['paired_instances']} instances move by more than 0.1 %, "
                       f"{r['instances_Z_moves_over_threshold']} by more than {r['threshold_%']:g} %. In the baseline "
                       f"{f1(r['base_vans_battery_nearly_empty_%'])} % of the used vans end with less than 11.1 kWh "
                       f"(cannot serve another EV) and {f1(r['base_vans_full_of_EVs_%'])} % are full of EVs (all seats used). "
                       f"Verdict: {r['verdict']}.")
        out.append("")
    if ex["id"] == "E4" and not T["pressure"].empty:
        p = T["pressure"][(T["pressure"]["experiment"] == "E4") & (T["pressure"]["method"] == "MILP")]
        if not p.empty:
            out += ["Queues by demand pressure at the request (terciles of requests in the last mean charging time; MILP):", "",
                    "| guns | pressure | mean queue (min) | 90 % queue | EVs queued | van share (free) | customer cost (Rs) |",
                    "|---|---|---|---|---|---|---|"]
            for _, r in p.sort_values(["x"]).iterrows():
                out.append(f"| {r['value_text']} | {r['pressure']} | {f1(r['queue_mean_min'])} | {f1(r['queue_p90_min'])} | "
                           f"{f1(r['queued_share_%'])} % | {f1(r['van_share_free_%'])} % | {f1(r['customer_cost_mean_Rs'], '{:.0f}')} |")
            out.append("")
    if ex["id"] == "E7" and not T["E7"].empty:
        out += ["Table 25 estimate (baseline van propensity per income decile, reweighted with the new Beta) against the "
                "re-solved van share:", "", "| method | distribution | mean theta | estimate | re-solved | difference |",
                "|---|---|---|---|---|---|"]
        for _, r in T["E7"].iterrows():
            out.append(f"| {r['method']} | {r['value_text']} | {f1(r['mean_theta'], '{:.3f}')} | {f1(r['table25_estimate_%'])} % | "
                       f"{f1(r['resolved_van_share_%'])} % | {f1(r['resolved_minus_estimate_pp'], '{:+.1f}')} pp |")
        out.append("")
    if ex["id"] == "E3" and not T["E3"].empty:
        out += ["Status transitions (baseline -> longer horizon):", "", "| method | T | baseline status | new status | instances |",
                "|---|---|---|---|---|"]
        for _, r in T["E3"].iterrows():
            out.append(f"| {r['method']} | {r['value_text']} | {r['status_base']} | {r['status_variant']} | {r['instances']} |")
        out.append("")
    return out


if __name__ == "__main__":
    main()
