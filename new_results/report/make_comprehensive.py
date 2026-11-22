"""Figures, tables and numbers for comprehensive_report.tex.

Reads the finished run in the parent folder (results.csv, analysis/ev_level.csv, <instance>/vans.csv,
run_settings.json), the dual values written by compute_duals.py (data/*.csv) and the EV-Solomon data.
Writes figures/c_*.pdf, tables/c_*.tex and tables/numbers_c.tex. Run compute_duals.py and
make_figures.py first (the report also uses their outputs).
Usage (from this folder):  python make_comprehensive.py
"""
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "..")
PROJECT = os.path.join(RUN, "..")
sys.path.insert(0, os.path.join(PROJECT, "code"))
import make_ev_solomon as mes  # noqa: E402

FIG = os.path.join(HERE, "figures")
TAB = os.path.join(HERE, "tables")
DATA = os.path.join(HERE, "data")
INSTANCES = os.path.join(PROJECT, "dataset_ev_solomon", "instances")
plt.rcParams.update({"font.size": 9, "font.family": "serif", "mathtext.fontset": "cm",
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
BANDS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)]
BLAB = ["0.0--0.2", "0.2--0.4", "0.4--0.6", "0.6--0.8", "0.8--1.0"]
PLAB = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
FAMS = ["C1", "C2", "R1", "R2", "RC1", "RC2"]
CHARGERS = 29
num = {}


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def mean(v):
    v = [x for x in v if x == x]
    return float(np.mean(v)) if v else float("nan")


def med(v):
    v = [x for x in v if x == x]
    return float(np.median(v)) if v else float("nan")


def f(v, fmt="{:.1f}"):
    try:
        return "--" if v != v else fmt.format(v)
    except (TypeError, ValueError):
        return str(v)


def mpl(s):
    """LaTeX label -> matplotlib mathtext (needs braces after \\mathrm)."""
    import re
    return re.sub(r"\\mathrm\s+(\w+)", r"\\mathrm{\1}", s)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)


def table(name, head, rows, spec):
    def cell(k, h):
        if "|" not in h:
            return h
        return f"\\shortstack[{'l' if k == 0 else 'r'}]{{" + "\\\\".join(h.split("|")) + "}"
    out = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(cell(k, h) for k, h in enumerate(head)) + " \\\\",
           "\\midrule"]
    for r in rows:
        out.append("\\midrule" if r == "MID" else " & ".join(str(c) for c in r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, name + ".tex"), "w").write("\n".join(out) + "\n")


# ====================================================================== data
settings = json.load(open(os.path.join(RUN, "run_settings.json")))
P = settings["f3_params"]
VOT = P["value_of_time_Rs_per_min"]
MU0, MU1 = P["money_sensitivity_mu"]["theta0"], P["money_sensitivity_mu"]["theta1"]
PF_T, PM_T = P["tariff_Rs_per_kWh"]["p_F"], P["tariff_Rs_per_kWh"]["p_M"]
C_EV, C_M, C_F, PI_F, PI_M, ETA = 4.0, 15.0, 1500.0, 8.7, 10.7, 0.9
POW_F, POW_M, SPEED = 22.0, 30.0, 15.6


def line(e, th):
    return e["theta0"] + (e["theta1"] - e["theta0"]) * th


results = list(csv.DictReader(open(os.path.join(RUN, "results.csv"))))
valid = [r for r in results if r.get("valid") == "yes"]
ev = list(csv.DictReader(open(os.path.join(RUN, "analysis", "ev_level.csv"))))
for e in ev:
    for k in ("theta", "mu", "c_D", "c_Q", "c_V", "c_C", "W_min", "E_kWh", "r_min", "load_at_request", "drive_km",
              "drive_min", "queue_min", "van_wait_min", "charge_min", "time_to_charged_min", "pay_Rs",
              "perceived_money_Rs", "time_cost_Rs", "customer_cost_Rs", "nearest_reachable_fcs_km",
              "operator_energy_Rs"):
        e[k] = fnum(e[k])
    e["n"] = int(e["n"])
    e["EV"] = int(e["EV"])
    e["van"] = int(e["van"])
    e["opt"] = e["solver_status"].startswith("optimal")
    e["free"] = e["stranded"] == "no"
    e["pressure"] = e["load_at_request"] / CHARGERS
    e["band"] = next(k for k, (lo, hi) in enumerate(BANDS) if lo <= e["theta"] < hi)
free = [e for e in ev if e["free"]]
T1, T2 = np.quantile([e["pressure"] for e in free], [1 / 3, 2 / 3])
for e in ev:
    e["lvl"] = "low" if e["pressure"] <= T1 else ("mid" if e["pressure"] <= T2 else "high")
EVKEY = {(e["instance"], e["EV"]): e for e in ev}
BY_INST = defaultdict(list)
for e in ev:
    BY_INST[e["instance"]].append(e)
num.update(NValidC=len(valid), NEVC=len(ev), NFreeC=len(free))

# ====================================================================== EDA: instances
zper = [fnum(r["Z_Rs"]) / int(r["EVs"]) for r in valid]
rows = []
spec_metrics = [
    ("objective $Z$ per EV (Rs)", zper, "{:.0f}"),
    ("operator share of $Z$ (\\%)", [100 * fnum(r["operator_cost_Z2_to_Z4"]) / fnum(r["total_Z"]) for r in valid], "{:.1f}"),
    ("vans used", [fnum(r["vans_used"]) for r in valid], "{:.1f}"),
    ("EVs served by a van (\\%)", [100 * fnum(r["EVs_by_van"]) / int(r["EVs"]) for r in valid], "{:.1f}"),
    ("free-choice EVs by van (\\%)", [fnum(r["share_free_choice_by_van_%"]) for r in valid], "{:.1f}"),
    ("mean FCS queue (min)", [fnum(r["mean_queue_FCS_min"]) for r in valid], "{:.2f}"),
    ("mean van wait (min)", [fnum(r["mean_van_wait_min"]) for r in valid], "{:.1f}"),
    ("mean request to charged (min)", [fnum(r["mean_time_to_charged_min"]) for r in valid], "{:.1f}"),
    ("van km", [fnum(r["van_km"]) for r in valid], "{:.1f}"),
    ("solver time (s), all instances", [fnum(r["runtime_s"]) for r in results if r.get("runtime_s")], "{:.0f}"),
    ("gap of time-limit plans (\\%)", [fnum(r["gap_%"]) for r in valid if r["status"].startswith("time limit")], "{:.2f}"),
]
for name, vals, fmt in spec_metrics:
    v = np.array([x for x in vals if x == x])
    rows.append([name, len(v), f(v.mean(), fmt), f(np.median(v), fmt), f(np.quantile(v, 0.1), fmt),
                 f(np.quantile(v, 0.9), fmt), f(v.min(), fmt), f(v.max(), fmt)])
table("c_inst_stats", ["measure (per instance)", "count", "mean", "median", "10th pct", "90th pct", "min", "max"],
      rows, "@{}lrrrrrrr@{}")
num.update(ZperMean=f(mean(zper), "{:.0f}"), ZperMin=f(min(zper), "{:.0f}"), ZperMax=f(max(zper), "{:.0f}"))

ns = sorted({int(r["EVs"]) for r in results})
fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.4))
axes[0].boxplot([[fnum(r["Z_Rs"]) / n for r in valid if int(r["EVs"]) == n] for n in ns], showfliers=False)
axes[0].set(ylabel="Z per EV (Rs)", xlabel="$n$")
axes[1].boxplot([[fnum(r["runtime_s"]) for r in results if int(r["EVs"]) == n and r.get("runtime_s")] for n in ns],
                showfliers=False)
axes[1].set(ylabel="solver time (s)", xlabel="$n$")
axes[2].boxplot([[fnum(r["gap_%"]) for r in valid if int(r["EVs"]) == n and r["status"].startswith("time limit")] or [np.nan]
                 for n in ns], showfliers=False)
axes[2].set(ylabel="gap of time-limit plans (%)", xlabel="$n$")
for ax in axes:
    ax.set_xticks(range(1, len(ns) + 1))
    ax.set_xticklabels([str(n) for n in ns], fontsize=6.5)
save(fig, "c_inst_by_n")

rows = []
for fam in FAMS:
    g = [r for r in valid if r["family"] == fam]
    ge = [e for e in free if e["family"] == fam]
    rows.append([fam, len([r for r in results if r["family"] == fam]), len(g), f(mean([fnum(r["Z_Rs"]) / int(r["EVs"]) for r in g]), "{:.0f}"),
                 f(mean([100 * fnum(r["operator_cost_Z2_to_Z4"]) / fnum(r["total_Z"]) for r in g])),
                 f(100 * mean([e["van"] for e in ge])), f(mean([e["queue_min"] for e in ge if not e["van"]]), "{:.2f}"),
                 f(mean([e["time_to_charged_min"] for e in ge])), f(mean([fnum(r["vans_used"]) for r in g]), "{:.2f}"),
                 f(fnum(g[0]["T_min"]) if g else float("nan"), "{:.0f}")])
table("c_family", ["family", "instances", "valid plans", "Z per EV|(Rs)", "operator|share (\\%)", "free-choice|by van (\\%)",
                   "mean queue|(min)", "request to|charged (min)", "vans|used", "$T$ (min)"], rows, "@{}lrrrrrrrrr@{}")

terms = [("Z1_EV_drive_perceived", "$Z_1$ EV driving"), ("Z2_van_drive", "$Z_2$ van driving"),
         ("Z3_dispatch", "$Z_3$ van dispatch"), ("Z4_operator_energy", "$Z_4$ operator energy"),
         ("Z5_tariff_perceived", "$Z_5$ tariff"), ("Z6_time_perceived", "$Z_6$ time")]
cols = ["#8172B2", "#64B5CD", "#4C72B0", "#55A868", "#DD8452", "#C44E52"]
fig, ax = plt.subplots(figsize=(5.6, 2.8))
bottom = np.zeros(len(FAMS))
for (k, lab), col in zip(terms, cols):
    vals = np.array([100 * sum(fnum(r[k]) for r in valid if r["family"] == fam)
                     / sum(fnum(r["total_Z"]) for r in valid if r["family"] == fam) for fam in FAMS])
    ax.bar(FAMS, vals, bottom=bottom, color=col, label=lab, width=0.65)
    bottom += vals
ax.set(ylabel="share of $Z$ (%)")
ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
save(fig, "c_obj_family")

# ====================================================================== EDA: EVs
rows = []
for name, vals, fmt in [
        ("income $\\theta$", [e["theta"] for e in ev], "{:.2f}"),
        ("energy demand $E_i$ (kWh)", [e["E_kWh"] for e in ev], "{:.1f}"),
        ("nearest reachable FCS (km)", [e["nearest_reachable_fcs_km"] for e in free], "{:.2f}"),
        ("demand pressure", [e["pressure"] for e in free], "{:.2f}"),
        ("drive to FCS (km), FCS users", [e["drive_km"] for e in free if not e["van"]], "{:.2f}"),
        ("FCS queue (min), FCS users", [e["queue_min"] for e in free if not e["van"]], "{:.2f}"),
        ("van wait (min), van users", [e["van_wait_min"] for e in ev if e["van"]], "{:.1f}"),
        ("charging time (min)", [e["charge_min"] for e in ev], "{:.1f}"),
        ("request to charged (min)", [e["time_to_charged_min"] for e in ev], "{:.1f}"),
        ("money paid (Rs)", [e["pay_Rs"] for e in ev], "{:.0f}"),
        ("perceived money (Rs)", [e["perceived_money_Rs"] for e in ev], "{:.0f}"),
        ("perceived time (Rs)", [e["time_cost_Rs"] for e in ev], "{:.0f}"),
        ("perceived total (Rs)", [e["customer_cost_Rs"] for e in ev], "{:.0f}")]:
    v = np.array([x for x in vals if x == x])
    rows.append([name, len(v), f(v.mean(), fmt), f(np.median(v), fmt), f(np.quantile(v, 0.1), fmt),
                 f(np.quantile(v, 0.9), fmt), f(v.max(), fmt)])
table("c_ev_stats", ["variable (per EV)", "count", "mean", "median", "10th pct", "90th pct", "max"], rows, "@{}lrrrrrr@{}")

fig, axes = plt.subplots(2, 3, figsize=(6.9, 4.4))
ax = axes[0, 0]
ax.hist([e["theta"] for e in free if not e["van"]], bins=20, range=(0, 1), density=True, alpha=0.6, color="#4C72B0", label="FCS users")
ax.hist([e["theta"] for e in free if e["van"]], bins=20, range=(0, 1), density=True, alpha=0.6, color="#C44E52", label="van users")
ax.set(xlabel=r"income $\theta$", ylabel="density", title="income of free-choice EVs")
ax.legend(fontsize=7)
ax = axes[0, 1]
ax.hist([e["time_to_charged_min"] for e in ev if not e["van"]], bins=30, range=(0, 200), alpha=0.6, color="#4C72B0", label="FCS")
ax.hist([e["time_to_charged_min"] for e in ev if e["van"]], bins=30, range=(0, 200), alpha=0.6, color="#C44E52", label="van")
ax.set(xlabel="request to charged (min)", ylabel="EVs", title="time until fully charged")
ax.legend(fontsize=7)
ax = axes[0, 2]
qs = [e["queue_min"] for e in free if not e["van"] and e["queue_min"] > 1e-6]
ax.hist(qs, bins=30, color="#DD8452")
ax.set(xlabel="queue (min)", ylabel="EVs", title="FCS queues above zero")
ax = axes[1, 0]
ax.hist([e["van_wait_min"] for e in ev if e["van"]], bins=30, color="#C44E52")
ax.set(xlabel="wait for the van (min)", ylabel="EVs", title="van waits (all van users)")
ax = axes[1, 1]
ax.hist([e["pay_Rs"] for e in ev if not e["van"]], bins=30, alpha=0.6, color="#4C72B0", label="FCS")
ax.hist([e["pay_Rs"] for e in ev if e["van"]], bins=30, alpha=0.6, color="#C44E52", label="van")
ax.set(xlabel="money paid (Rs)", ylabel="EVs", title="money paid")
ax.legend(fontsize=7)
ax = axes[1, 2]
ax.hist([e["customer_cost_Rs"] for e in ev], bins=30, color="#55A868")
ax.set(xlabel="perceived cost (Rs)", ylabel="EVs", title="perceived cost")
save(fig, "c_ev_dists")
num.update(NQueued=len(qs), NFCSfree=len([e for e in free if not e["van"]]))

vars_ = [("theta", r"$\theta$"), ("E_kWh", "$E$"), ("nearest_reachable_fcs_km", "nearest FCS"), ("pressure", "pressure"),
         ("van", "van"), ("queue_min", "queue"), ("time_to_charged_min", "time to charged"), ("pay_Rs", "paid"),
         ("perceived_money_Rs", "perc. money"), ("time_cost_Rs", "perc. time")]


def ranks(x):
    order = np.argsort(x, kind="mergesort")
    rk = np.empty(len(x))
    rk[order] = np.arange(len(x))
    # average ties
    xs = np.asarray(x)[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        rk[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return rk


M = np.array([[e[k] for k, _ in vars_] for e in free], dtype=float)
R = np.column_stack([ranks(M[:, k]) for k in range(M.shape[1])])
C = np.corrcoef(R, rowvar=False)
fig, ax = plt.subplots(figsize=(5.2, 4.3))
im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(vars_)))
ax.set_yticks(range(len(vars_)))
ax.set_xticklabels([l for _, l in vars_], rotation=45, ha="right", fontsize=7)
ax.set_yticklabels([l for _, l in vars_], fontsize=7)
for a in range(len(vars_)):
    for b in range(len(vars_)):
        ax.text(b, a, f"{C[a, b]:.2f}", ha="center", va="center", fontsize=5.5,
                color="white" if abs(C[a, b]) > 0.6 else "black")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
save(fig, "c_corr")
idx = {k: i for i, (k, _) in enumerate(vars_)}
num.update(CorrThetaVan=f(C[idx["theta"], idx["van"]], "{:.2f}"), CorrThetaQueue=f(C[idx["theta"], idx["queue_min"]], "{:.2f}"),
           CorrPressQueue=f(C[idx["pressure"], idx["queue_min"]], "{:.2f}"),
           CorrThetaTime=f(C[idx["theta"], idx["time_to_charged_min"]], "{:.2f}"),
           CorrEPay=f(C[idx["E_kWh"], idx["pay_Rs"]], "{:.2f}"), CorrNearVan=f(C[idx["nearest_reachable_fcs_km"], idx["van"]], "{:.2f}"))

# ====================================================================== A1: coefficient elasticities (envelope theorem)
coefs = []   # (symbol, value, meaning, total quantity over all valid plans)
Ztot = sum(fnum(r["total_Z"]) for r in valid)


def tot(fn, sel=lambda e: True):
    return sum(fn(e) for e in ev if sel(e))


fcs = lambda e: not e["van"]
vanu = lambda e: bool(e["van"])
coefs += [("$c^{\\mathrm{EV}}$", C_EV, "EV driving cost (Rs/km)", tot(lambda e: e["mu"] * e["drive_km"], fcs)),
          ("$c^{\\mathrm M}$", C_M, "van driving cost (Rs/km)", sum(fnum(r["van_km"]) for r in valid)),
          ("$c^{\\mathrm F}_m$", C_F, "van dispatch cost (Rs)", sum(fnum(r["vans_used"]) for r in valid)),
          ("$\\pi^{\\mathrm F}$", PI_F, "grid energy at FCS (Rs/kWh)", tot(lambda e: e["E_kWh"], fcs)),
          ("$\\pi^{\\mathrm M}$", PI_M, "van battery energy (Rs/kWh)", tot(lambda e: e["E_kWh"] / ETA, vanu)),
          ("$p^{\\mathrm F}$", PF_T, "FCS tariff (Rs/kWh)", tot(lambda e: e["mu"] * e["E_kWh"], fcs)),
          ("$p^{\\mathrm M}$", PM_T, "van tariff (Rs/kWh)", tot(lambda e: e["mu"] * e["E_kWh"], vanu)),
          ("$\\mu^{0}$", MU0, "money sensitivity at $\\theta=0$", tot(lambda e: (1 - e["theta"]) * e["pay_Rs"])),
          ("$\\mu^{1}$", MU1, "money sensitivity at $\\theta=1$", tot(lambda e: e["theta"] * e["pay_Rs"]))]
tq = {"D": lambda e: e["drive_min"] if not e["van"] else 0.0, "Q": lambda e: e["queue_min"] if not e["van"] else 0.0,
      "C": lambda e: e["charge_min"] if not e["van"] else 0.0,
      "V": lambda e: (e["van_wait_min"] + e["charge_min"]) if e["van"] else 0.0}
names = {"D": "driving", "Q": "queueing", "C": "FCS charging", "V": "van wait + charging"}
for a in "DQCV":
    coefs.append((f"$c^{{\\mathrm{{{a}}},0}}$", VOT[a]["theta0"], f"value of time, {names[a]}, $\\theta=0$",
                  tot(lambda e, a=a: (1 - e["theta"]) * tq[a](e))))
    coefs.append((f"$c^{{\\mathrm{{{a}}},1}}$", VOT[a]["theta1"], f"value of time, {names[a]}, $\\theta=1$",
                  tot(lambda e, a=a: e["theta"] * tq[a](e))))
rows, el_names, el_vals = [], [], []
nv = len(valid)
for sym, val, meaning, q in coefs:
    el = 100 * val * q / Ztot
    rows.append([sym, f"{val:g}", meaning, f(q / nv, "{:.1f}"), f(el, "{:.2f}"), f(0.1 * val * q / nv, "{:.0f}")])
    el_names.append(sym)
    el_vals.append(el)
table("c_elasticity", ["coefficient", "value", "meaning", "$\\partial Z/\\partial$coef.|per plan", "elasticity|(\\%)",
                       "Rs per plan|for +10\\%"], rows, "@{}lrlrrr@{}")
el = dict(zip(el_names, el_vals))
num.update(ElOperator=f(el["$c^{\\mathrm M}$"] + el["$c^{\\mathrm F}_m$"] + el["$\\pi^{\\mathrm F}$"] + el["$\\pi^{\\mathrm M}$"], "{:.1f}"),
           ElMoney=f(el["$\\mu^{0}$"] + el["$\\mu^{1}$"], "{:.1f}"),
           ElTariffs=f(el["$p^{\\mathrm F}$"] + el["$p^{\\mathrm M}$"] + el["$c^{\\mathrm{EV}}$"], "{:.1f}"),
           ElTime=f(sum(v for k, v in el.items() if k.startswith("$c^{\\mathrm{") and ",0}" in k or ",1}" in k), "{:.1f}"),
           ElPiF=f(el["$\\pi^{\\mathrm F}$"], "{:.1f}"), ElCF=f(el["$c^{\\mathrm F}_m$"], "{:.1f}"),
           ElPF=f(el["$p^{\\mathrm F}$"], "{:.1f}"), ElPM=f(el["$p^{\\mathrm M}$"], "{:.1f}"),
           ElMuZero=f(el["$\\mu^{0}$"], "{:.1f}"), ElMuOne=f(el["$\\mu^{1}$"], "{:.1f}"),
           ElCCOne=f(el["$c^{\\mathrm{C},1}$"], "{:.1f}"), ElCCZero=f(el["$c^{\\mathrm{C},0}$"], "{:.1f}"),
           ElCQOne=f(el["$c^{\\mathrm{Q},1}$"], "{:.2f}"), ElCQZero=f(el["$c^{\\mathrm{Q},0}$"], "{:.2f}"),
           ElCVOne=f(el["$c^{\\mathrm{V},1}$"], "{:.1f}"), ElCDOne=f(el["$c^{\\mathrm{D},1}$"], "{:.1f}"))
order = np.argsort(el_vals)
fig, ax = plt.subplots(figsize=(5.0, 4.2))
ax.barh(range(len(order)), [el_vals[k] for k in order], color="#4C72B0")
ax.set_yticks(range(len(order)))
ax.set_yticklabels([mpl(el_names[k]) for k in order], fontsize=8)
ax.set(xlabel="elasticity of $Z$ (% change of $Z$ per 1% change of the coefficient)")
save(fig, "c_elasticity")

# ====================================================================== A2: shadow prices
dc = list(csv.DictReader(open(os.path.join(DATA, "duals_constraints.csv"))))
di = list(csv.DictReader(open(os.path.join(DATA, "duals_instances.csv"))))
for r in dc:
    r["dual"] = fnum(r["dual"])
kinds = [("arrival", "FCS arrival $S_i\\ge r_i+t_{if}$", "not unique (degenerate, Section 4.2.7); not interpreted"),
         ("release", "van release $S_i\\ge r_i$", "not unique (degenerate); not interpreted"),
         ("gun", "gun order $S_j\\ge S_i+\\tau^{\\mathrm F}_i$", "value of one gun-minute: delay EV $i$ passes to those behind it"),
         ("van", "van order $S_j\\ge S_i+\\tau^{\\mathrm M}_i+t_{ij}$", "value of one van-minute on that leg"),
         ("depart", "first stop $S_a\\ge t_{0a}$", "value of an earlier van departure"),
         ("patience", "queue limit $q^{\\mathrm F}_i\\le W_i$", "value of one more minute of patience"),
         ("deadline", "deadline $C_i\\le T$, van return", "value of one more minute of horizon")]
rows = []
stats = {}
for k, lab, meaning in kinds:
    if k == "deadline":
        v = np.array([r["dual"] for r in dc if r["kind"] in ("deadline_fcs", "deadline_van", "return")])
        nrows = sum(1 for e in ev) + sum(fnum(r["vans_used"]) for r in valid)
    else:
        v = np.array([r["dual"] for r in dc if r["kind"] == k])
        nrows = len(v)
    nz = v[np.abs(v) > 1e-9]
    stats[k] = (nrows, len(nz), nz)
    rows.append([lab, f(nrows, "{:.0f}"), f(100 * len(nz) / nrows if nrows else float("nan")),
                 f(np.mean(np.abs(nz)) if len(nz) else 0.0, "{:.2f}"), f(np.median(np.abs(nz)) if len(nz) else 0.0, "{:.2f}"),
                 f(np.max(np.abs(nz)) if len(nz) else 0.0, "{:.2f}"), meaning])
table("c_duals", ["constraint", "rows", "non-zero|dual (\\%)", "mean|(Rs/min)", "median|(Rs/min)", "max|(Rs/min)", "meaning of the dual"],
      rows, "@{}lrrrrrp{4.6cm}@{}")
gun = [r for r in dc if r["kind"] == "gun"]
vanr = [r for r in dc if r["kind"] == "van"]
num.update(GunRows=len(gun), GunBind=f(100 * stats["gun"][1] / stats["gun"][0]), GunMean=f(np.mean(stats["gun"][2]), "{:.1f}"),
           GunMed=f(np.median(stats["gun"][2]), "{:.1f}"), GunMax=f(np.max(stats["gun"][2]), "{:.1f}"),
           VanBind=f(100 * stats["van"][1] / stats["van"][0]), VanMean=f(np.mean(stats["van"][2]), "{:.1f}"),
           ArrMean=f(np.mean(stats["arrival"][2]), "{:.1f}"), RelMean=f(np.mean(stats["release"][2]), "{:.1f}"),
           PatRows=stats["patience"][0], DeadRows=f(stats["deadline"][0], "{:.0f}"))
# gun duals by follower pressure and by theta of occupier / follower
def band(t):
    return next(k for k, (lo, hi) in enumerate(BANDS) if lo <= t < hi)


for r in gun:
    fj = EVKEY.get((r["instance"], int(r["j"])))
    r["lvl_j"] = fj["lvl"] if fj else "?"
    r["bi"], r["bj"] = band(fnum(r["theta_i"])), band(fnum(r["theta_j"]))
rows = []
for L, nm in (("low", "low"), ("mid", "medium"), ("high", "high")):
    g = [r for r in gun if r["lvl_j"] == L]
    nz = [r["dual"] for r in g if r["dual"] > 1e-9]
    rows.append([nm, len(g), f(100 * len(nz) / len(g) if g else float("nan")), f(mean(nz), "{:.1f}"), f(sum(nz) / len(g) if g else float("nan"), "{:.2f}")])
table("c_gun_pressure", ["demand pressure|of the follower", "gun links", "binding|(\\%)", "mean dual if|binding (Rs/min)",
                         "mean dual|all links (Rs/min)"], rows, "@{}lrrrr@{}")
num.update(GunBindHigh=rows[2][2], GunBindLow=rows[0][2])
share_occ = [100 * mean([r["dual"] > 1e-9 for r in gun if r["bi"] == b]) for b in range(5)]
share_fol = [100 * mean([r["dual"] > 1e-9 for r in gun if r["bj"] == b]) for b in range(5)]
num.update(GunOccBot=f(share_occ[0]), GunOccTop=f(share_occ[4]), GunFolBot=f(share_fol[0]), GunFolTop=f(share_fol[4]))
# mean dual by (occupier band, follower band)
H = np.full((5, 5), np.nan)
for a in range(5):
    for b in range(5):
        g = [r["dual"] for r in gun if r["bi"] == a and r["bj"] == b]
        if len(g) >= 20:
            H[a, b] = 100 * mean([x > 1e-9 for x in g])
fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.5))
nz = stats["gun"][2]
axes[0].hist(nz, bins=30, color="#C44E52")
axes[0].set(xlabel="gun-minute shadow price (Rs/min)", ylabel="binding gun links")
x = np.arange(5)
axes[1].bar(x - 0.2, share_occ, 0.4, color="#4C72B0", label="EV on the gun ($i$)")
axes[1].bar(x + 0.2, share_fol, 0.4, color="#DD8452", label="EV behind it ($j$)")
axes[1].set_xticks(x)
axes[1].set_xticklabels(PLAB, fontsize=6, rotation=30)
axes[1].set(xlabel=r"income band $\theta$", ylabel="gun links binding (%)")
axes[1].legend(fontsize=6.5)
im = axes[2].imshow(H, cmap="Reds", origin="lower")
axes[2].set_xticks(x)
axes[2].set_yticks(x)
axes[2].set_xticklabels(PLAB, fontsize=5.5, rotation=30)
axes[2].set_yticklabels(PLAB, fontsize=5.5)
axes[2].set(xlabel=r"$\theta$ of the EV behind ($j$)", ylabel=r"$\theta$ of the EV on the gun ($i$)")
for a in range(5):
    for b in range(5):
        if H[a, b] == H[a, b]:
            axes[2].text(b, a, f"{H[a, b]:.0f}", ha="center", va="center", fontsize=6)
save(fig, "c_gun_duals")
# arrival duals: own c^Q vs passed-on delay
arr = [r for r in dc if r["kind"] == "arrival"]
own, passed = [], defaultdict(list)
for r in arr:
    e = EVKEY.get((r["instance"], int(r["i"])))
    if e is None:
        continue
    passed[e["band"]].append(r["dual"] - e["c_Q"])
    own.append(e["c_Q"])
num.update(PassedBot=f(mean(passed[0]), "{:.2f}"), PassedTop=f(mean(passed[4]), "{:.2f}"))
# dZ/dP
dpf = np.array([fnum(r["dZ_dPF"]) for r in di])
dpm = np.array([fnum(r["dZ_dPM"]) for r in di])
own_f, own_m = [], []
for r in di:
    g = BY_INST[r["instance"]]
    own_f.append(sum(e["c_C"] * (-e["charge_min"] / POW_F) for e in g if not e["van"]))
    own_m.append(sum(e["c_V"] * (-e["charge_min"] / POW_M) for e in g if e["van"]))
own_f, own_m = np.array(own_f), np.array(own_m)
cong_f = dpf - own_f
num.update(DPFmean=f(-dpf.mean(), "{:.0f}"), DPFmed=f(-np.median(dpf), "{:.0f}"), DPFmax=f(-dpf.min(), "{:.0f}"),
           DPMmean=f(-dpm.mean(), "{:.0f}"), DPMmax=f(-dpm.min(), "{:.0f}"),
           DPFcong=f(100 * cong_f.sum() / dpf.sum(), "{:.1f}"),
           DPFperEV=f(-np.mean([fnum(r["dZ_dPF"]) / int(r["n"]) for r in di]), "{:.1f}"),
           DTnonzero=sum(1 for r in di if abs(fnum(r["dZ_dT"])) > 1e-9))
fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.5))
nsv = sorted({int(r["n"]) for r in di})
axes[0].boxplot([[-fnum(r["dZ_dPF"]) for r in di if int(r["n"]) == n] for n in nsv], showfliers=False)
axes[0].set(xlabel="$n$", ylabel="Rs saved per instance", title="one more kW of FCS gun power")
axes[1].boxplot([[-fnum(r["dZ_dPM"]) for r in di if int(r["n"]) == n] for n in nsv], showfliers=False)
axes[1].set(xlabel="$n$", ylabel="Rs saved per instance", title="one more kW of van power")
for ax in axes:
    ax.set_xticks(range(1, len(nsv) + 1))
    ax.set_xticklabels([str(n) for n in nsv], fontsize=7)
save(fig, "c_power_value")

# ====================================================================== A3: the income threshold (theory)
def delta(th, E=20.0, d=2.0, qF=0.0, qM=10.0, x=4.0, dispatch=0.0, pM=PM_T, pF=PF_T, mu0=MU0, mu1=MU1, vot=VOT,
          piF=PI_F, piM=PI_M, cM=C_M):
    t = 60 * d / SPEED
    tauF, tauM = 60 * E / POW_F, 60 * E / POW_M
    mu = mu0 + (mu1 - mu0) * th
    cD, cQ, cC, cV = (line(vot[a], th) for a in "DQCV")
    kappa = (piM / ETA - piF) * E + cM * x + dispatch
    return mu * ((pM - pF) * E - C_EV * d) + kappa - cD * t - cQ * qF - cC * tauF + cV * (qM + tauM)


def theta_star(**kw):
    a = delta(0.0, **kw)
    b = delta(1.0, **kw) - a
    return -a / b if b != 0 else float("nan")


base = theta_star()
num.update(ThetaStar=f(base, "{:.2f}"), DeltaZero=f(delta(0.0), "{:.0f}"), DeltaOne=f(delta(1.0), "{:.0f}"),
           ThetaStarDispatch=f(theta_star(dispatch=C_F), "{:.2f}"), ThetaStarQueue=f(theta_star(qF=10.0), "{:.2f}"),
           ThetaStarWait=f(theta_star(qM=40.0), "{:.2f}"), ThetaStarSmallE=f(theta_star(E=10.0), "{:.2f}"),
           ThetaStarBigE=f(theta_star(E=40.0), "{:.2f}"), ThetaStarFar=f(theta_star(d=5.0), "{:.2f}"),
           ThetaStarNear=f(theta_star(d=0.5), "{:.2f}"), DeltaDispatchOne=f(delta(1.0, dispatch=C_F), "{:.0f}"),
           KappaTypical=f((PI_M / ETA - PI_F) * 20 + C_M * 4, "{:.0f}"))


def scaled_vot(a, end, k):
    v = json.loads(json.dumps(VOT))
    v[a][end] *= k
    return v


cases = [("van tariff $p^{\\mathrm M}$", lambda k: theta_star(pM=PF_T + (PM_T - PF_T) * k), "premium $p^{\\mathrm M}-p^{\\mathrm F}$"),
         ("$\\mu^{0}$", lambda k: theta_star(mu0=MU0 * k), ""),
         ("$\\mu^{1}$", lambda k: theta_star(mu1=MU1 * k), ""),
         ("$c^{\\mathrm{C},1}$", lambda k: theta_star(vot=scaled_vot("C", "theta1", k)), ""),
         ("$c^{\\mathrm{C},0}$", lambda k: theta_star(vot=scaled_vot("C", "theta0", k)), ""),
         ("$c^{\\mathrm{V},1}$", lambda k: theta_star(vot=scaled_vot("V", "theta1", k)), ""),
         ("$c^{\\mathrm{D},1}$", lambda k: theta_star(vot=scaled_vot("D", "theta1", k)), ""),
         ("$\\pi^{\\mathrm M}$", lambda k: theta_star(piM=PI_M * k), ""),
         ("$c^{\\mathrm M}$", lambda k: theta_star(cM=C_M * k), ""),
         ("van detour $x$", lambda k: theta_star(x=4.0 * k), "4 km"),
         ("van wait $q^{\\mathrm M}$", lambda k: theta_star(qM=10.0 * k), "10 min"),
         ("energy $E$", lambda k: theta_star(E=20.0 * k), "20 kWh"),
         ("distance to FCS $d$", lambda k: theta_star(d=2.0 * k), "2 km")]
rows, tor = [], []
for lab, fn, note in cases:
    lo_, hi_ = fn(0.75), fn(1.25)
    rows.append([lab, f(lo_, "{:.2f}"), f(base, "{:.2f}"), f(hi_, "{:.2f}"), f(abs(hi_ - lo_), "{:.2f}")])
    tor.append((lab, lo_, hi_))
rows.sort(key=lambda r: -float(r[4]))
table("c_tornado", ["parameter ($\\pm25\\%$)", "$\\theta^{*}$ at $-25\\%$", "$\\theta^{*}$ base", "$\\theta^{*}$ at $+25\\%$", "swing"],
      rows, "@{}lrrrr@{}")
tor.sort(key=lambda t: abs(t[2] - t[1]))
fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9))
th = np.linspace(0, 1, 101)
for E_, col in ((10, "#4C72B0"), (20, "#333333"), (40, "#C44E52")):
    axes[0].plot(th, [delta(t, E=E_) for t in th], color=col, lw=1.3, label=f"$E$ = {E_} kWh")
axes[0].plot(th, [delta(t, dispatch=C_F) for t in th], color="#333333", ls="--", lw=1.1, label="20 kWh, van not yet sent")
axes[0].axhline(0, color="grey", lw=0.6)
axes[0].set(xlabel=r"income $\theta$", ylabel=r"$\Delta_i(\theta)$: extra cost of a van (Rs)")
axes[0].legend(fontsize=6.5)
for k_, (lab, lo_, hi_) in enumerate(tor):
    axes[1].plot([min(lo_, hi_), max(lo_, hi_)], [k_, k_], color="#4C72B0", lw=5, solid_capstyle="butt")
    axes[1].plot([lo_], [k_], "o", color="#4C72B0", ms=3)
    axes[1].plot([hi_], [k_], "s", color="#C44E52", ms=3)
axes[1].axvline(base, color="#333333", lw=0.8)
axes[1].set_yticks(range(len(tor)))
axes[1].set_yticklabels([mpl(t[0]) for t in tor], fontsize=6.5)
axes[1].set(xlabel=r"$\theta^{*}$ (circle: $-25\%$, square: $+25\%$)")
save(fig, "c_threshold")
# theory vs data
dec_opt = []
for b in range(10):
    g = [e for e in free if e["opt"] and min(int(e["theta"] * 10), 9) == b]
    dec_opt.append(100 * mean([e["van"] for e in g]))
fig, ax = plt.subplots(figsize=(4.8, 2.7))
ax.plot([b / 10 + 0.05 for b in range(10)], dec_opt, "s-", color="#333333", ms=3.5, lw=1.1, label="optimal plans (data)")
ax.axvline(base, color="#C44E52", ls="--", lw=1.1, label=rf"theory: $\theta^*$ = {base:.2f} (typical EV)")
ax.axvspan(theta_star(E=40.0), theta_star(E=10.0), color="#C44E52", alpha=0.08, label=r"$\theta^*$ for $E$ = 40 ... 10 kWh")
ax.set(xlabel=r"income $\theta$", ylabel="free-choice EVs by van (%)", xlim=(0, 1))
ax.legend(fontsize=7, loc="upper left")
save(fig, "c_threshold_data")

# ====================================================================== A8: vans
vrows, seat_by = [], Counter()
for r in valid:
    p = os.path.join(RUN, r["instance"], "vans.csv")
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        continue
    T = fnum(r["T_min"])
    for v in csv.DictReader(open(p)):
        stops, drawn, back = fnum(v["stops"]), fnum(v["kWh_drawn"]), fnum(v["back_at_depot_min"])
        util = {"seats ($Q_m=6$)": stops / 6, "battery ($B_m=150$ kWh)": drawn / 150, "time ($T$)": back / T}
        vrows.append({"instance": r["instance"], "family": r["family"], "stops": stops, "drawn": drawn,
                      "time": back / T, "km": fnum(v["km"]), "tight": max(util, key=util.get), "util": util})
for e in ev:
    if e["van"]:
        seat_by["stranded" if not e["free"] else "free-choice"] += 1
tc = Counter(v["tight"] for v in vrows)
rows = [["vans dispatched (all valid plans)", len(vrows), "", ""],
        ["stops per van", f(mean([v["stops"] for v in vrows]), "{:.2f}"), f(med([v["stops"] for v in vrows]), "{:.0f}"),
         f"{100 * mean([v['stops'] == 6 for v in vrows]):.1f}\\,\\% at $Q_m=6$"],
        ["battery drawn (kWh)", f(mean([v["drawn"] for v in vrows]), "{:.1f}"), f(med([v["drawn"] for v in vrows]), "{:.1f}"),
         f"{100 * mean([v['drawn'] >= 135 for v in vrows]):.1f}\\,\\% above 90\\,\\% of $B_m$"],
        ["back at depot ($\\%$ of $T$)", f(100 * mean([v["time"] for v in vrows])), f(100 * med([v["time"] for v in vrows])),
         f"{100 * mean([v['time'] >= 0.9 for v in vrows]):.1f}\\,\\% after 90\\,\\% of $T$"],
        ["van km", f(mean([v["km"] for v in vrows])), f(med([v["km"] for v in vrows])), ""]]
table("c_vans", ["measure", "mean", "median", "at the limit"], rows, "@{}lrrl@{}")
rows = [[k, tc[k], f(100 * tc[k] / len(vrows))] for k in ("seats ($Q_m=6$)", "battery ($B_m=150$ kWh)", "time ($T$)")]
table("c_vans_tight", ["tightest van resource", "vans", "share (\\%)"], rows, "@{}lrr@{}")
num.update(VansN=len(vrows), VansFull=f(100 * mean([v["stops"] == 6 for v in vrows])),
           VanSeatsStranded=f(100 * seat_by["stranded"] / (seat_by["stranded"] + seat_by["free-choice"])),
           VansBatt=f(100 * mean([v["drawn"] >= 135 for v in vrows])), VansTime=f(100 * mean([v["time"] >= 0.9 for v in vrows])),
           TightSeats=f(100 * tc["seats ($Q_m=6$)"] / len(vrows)), TightBatt=f(100 * tc["battery ($B_m=150$ kWh)"] / len(vrows)),
           TightTime=f(100 * tc["time ($T$)"] / len(vrows)))
fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.3))
axes[0].hist([v["stops"] for v in vrows], bins=np.arange(0.5, 7.5, 1), color="#4C72B0", rwidth=0.85)
axes[0].set(xlabel="EVs served by one van", ylabel="vans")
axes[1].hist([v["drawn"] for v in vrows], bins=25, range=(0, 150), color="#55A868")
axes[1].set(xlabel="battery drawn (kWh, of 150)")
axes[2].hist([100 * v["time"] for v in vrows], bins=25, range=(0, 100), color="#DD8452")
axes[2].set(xlabel="back at depot (% of $T$)")
save(fig, "c_vans")
by_fam = []
for fam in FAMS:
    g = [v for v in vrows if v["family"] == fam]
    c = Counter(v["tight"] for v in g)
    by_fam.append([fam, len(g), f(mean([v["stops"] for v in g]), "{:.2f}"), f(100 * mean([v["time"] for v in g])),
                   f(100 * c["seats ($Q_m=6$)"] / len(g) if g else float("nan")), f(100 * c["time ($T$)"] / len(g) if g else float("nan"))])
table("c_vans_family", ["family", "vans", "stops|per van", "back at depot|(\\% of $T$)", "tightest:|seats (\\%)", "tightest:|time (\\%)"],
      by_fam, "@{}lrrrrr@{}")

# ====================================================================== A9: equity
rows = []
paid_tot = sum(e["pay_Rs"] for e in free)
felt_tot = sum(e["customer_cost_Rs"] for e in free)
for b in range(5):
    g = [e for e in free if e["band"] == b]
    rows.append([BLAB[b], f(100 * len(g) / len(free)), f(100 * sum(e["pay_Rs"] for e in g) / paid_tot),
                 f(100 * sum(e["customer_cost_Rs"] for e in g) / felt_tot),
                 f(sum(e["customer_cost_Rs"] for e in g) / sum(e["pay_Rs"] for e in g), "{:.2f}"),
                 f(100 * sum(e["time_cost_Rs"] for e in g) / sum(e["customer_cost_Rs"] for e in g)),
                 f(mean([e["pay_Rs"] / e["E_kWh"] for e in g]), "{:.1f}")])
table("c_equity", ["income $\\theta$", "share of|EVs (\\%)", "share of|money paid (\\%)", "share of|perceived cost (\\%)",
                   "perceived per|Rs paid", "time share of|perceived (\\%)", "paid per|kWh (Rs)"], rows, "@{}lrrrrrr@{}")
num.update(PerRsBot=rows[0][4], PerRsTop=rows[4][4], PerKWhBot=rows[0][6], PerKWhTop=rows[4][6],
           TimeShareBot=rows[0][5], TimeShareTop=rows[4][5])

# ====================================================================== A10: computation
rows = []
for n in ns:
    g = [r for r in results if int(r["EVs"]) == n and r.get("binaries")]
    gv = [r for r in valid if int(r["EVs"]) == n and r.get("improvement_vs_greedy_%") not in ("", None)]
    rows.append([n, f(mean([fnum(r["binaries"]) for r in g]), "{:,.0f}"), f(mean([fnum(r["continuous"]) for r in g]), "{:,.0f}"),
                 f(mean([fnum(r["constraints"]) for r in g]), "{:,.0f}"), f(mean([fnum(r["build_s"]) for r in g]), "{:.1f}"),
                 f(med([fnum(r["nodes_explored"]) for r in g]), "{:,.0f}"),
                 f(med([fnum(r["improvement_vs_greedy_%"]) for r in gv]), "{:.1f}"),
                 f(100 * mean([r["greedy_Z_Rs"] not in ("", None) for r in g]))])
table("c_model", ["$n$", "binary|variables", "continuous|variables", "constraints", "build|time (s)", "median|B\\&B nodes",
                  "median gain over|greedy (\\%)", "greedy plan|found (\\%)"], rows, "@{}rrrrrrrr@{}")
gr = Counter(r.get("greedy_rule") or "none" for r in results)
num.update(GreedyCost=gr.get("cost", 0), GreedyFinish=gr.get("finish", 0), GreedyNone=gr.get("none", 0),
           BinHundred=f(mean([fnum(r["binaries"]) for r in results if int(r["EVs"]) == 100 and r.get("binaries")]), "{:,.0f}"),
           ConHundred=f(mean([fnum(r["constraints"]) for r in results if int(r["EVs"]) == 100 and r.get("binaries")]), "{:,.0f}"))

# ====================================================================== A11: infeasible and unsolved instances
inf_rows, imp_count, unsolved_p, solved_p = [], 0, [], defaultdict(list)


def instance_file(name):
    base, n = name.rsplit("_n", 1)
    return os.path.join(INSTANCES, "100", base + ".txt"), int(n)


def data_pressure(D):
    r, tf = D["r"], D["tauF"]
    w = float(np.mean(tf))
    return float(np.mean([np.sum((r >= ri - w) & (r <= ri)) for ri in r])) / CHARGERS


infeasible = [r for r in results if r["status"].startswith("infeasible")]
fam_inf = defaultdict(list)
late = []
for r in infeasible:
    path, n = instance_file(r["instance"])
    D = mes.read_ev_solomon(path, n)
    bad = mes.impossible_evs(D)
    imp_count += bool(bad)
    fam_inf[r["family"]].append(n)
    for i in bad:
        late.append((r["instance"], i + 1, float(D["r"][i]), float(D["params"]["T"]), float(D["tauF"][i])))
for fam in FAMS:
    if fam_inf[fam]:
        inf_rows.append([fam, len(fam_inf[fam]), min(fam_inf[fam]), max(fam_inf[fam])])
table("c_infeasible", ["family", "infeasible instances", "smallest $n$", "largest $n$"], inf_rows, "@{}lrrr@{}")
for r in results:
    if r["status"].startswith(("optimal", "time limit")):
        path, n = instance_file(r["instance"])
        p = data_pressure(mes.read_ev_solomon(path, n))
        if "no solution" in r["status"]:
            unsolved_p.append((r["family"], n, p))
        else:
            solved_p[(r["family"], n)].append(p)
cmp_rows = []
for fam, n, p in sorted(unsolved_p):
    cmp_rows.append((fam, n, p, mean(solved_p.get((fam, n), []))))
num.update(InfProven=imp_count, InfTotal=len(infeasible),
           LateExample=(f"{late[0][0].replace('_', chr(92) + '_')}: EV {late[0][1]} requests at minute {late[0][2]:.0f}, "
                        f"needs at least {late[0][4]:.0f} min to charge, $T={late[0][3]:.0f}$") if late else "--",
           UnsolvedPress=f(mean([p for _, _, p in unsolved_p]), "{:.2f}"),
           SolvedPressSame=f(mean([c[3] for c in cmp_rows]), "{:.2f}"),
           AllPressMean=f(mean([p for v in solved_p.values() for p in v]), "{:.2f}"))

# ====================================================================== A13: other income distributions
dec_all = []
for b in range(10):
    g = [e for e in free if min(int(e["theta"] * 10), 9) == b]
    dec_all.append(mean([e["van"] for e in g]))


def beta_cdf_mass(a, b_, lo, hi, k=2000):
    xs = np.linspace(lo, hi, k)
    xm = (xs[1:] + xs[:-1]) / 2
    dens = xm ** (a - 1) * (1 - xm) ** (b_ - 1)
    return float(np.sum(dens) * (xs[1] - xs[0]))


rows = []
for (a, b_), lab in (((1, 1), "Beta(1,1) uniform"), ((2, 2), "Beta(2,2) (used)"), ((2, 5), "Beta(2,5) mostly low income"),
                     ((5, 2), "Beta(5,2) mostly high income"), ((0.5, 0.5), "Beta(0.5,0.5) polarised")):
    tot_ = beta_cdf_mass(a, b_, 0, 1)
    w = [beta_cdf_mass(a, b_, d / 10, (d + 1) / 10) / tot_ for d in range(10)]
    share = 100 * sum(wi * pi for wi, pi in zip(w, dec_all))
    rows.append([lab, f(a * 1.0 / (a + b_), "{:.2f}"), f(100 * sum(w[6:]), "{:.1f}"), f(share)])
table("c_beta", ["income distribution", "mean $\\theta$", "EVs with $\\theta\\ge0.6$ (\\%)", "expected van share (\\%)"], rows, "@{}lrrr@{}")

# ====================================================================== numbers
with open(os.path.join(TAB, "numbers_c.tex"), "w") as fh:
    fh.write("% generated by make_comprehensive.py\n")
    for k, v in num.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
for k, v in num.items():
    print(f"{k:18s} {v}")
