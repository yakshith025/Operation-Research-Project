"""Figures, tables and numbers of comprehensive_report_milp.tex (formulation 4, baseline MILP run milp_300s).

Reads data/*.csv written by prepare_data_f4.py, ../../../methods/milp/results/meta.json
and common/f4_params.json. Writes figures/*.pdf, tables/*.tex and tables/numbers.tex (+ numbers.json).
Usage (from this folder):  python prepare_data_f4.py   then   python make_milp_report.py
                           then  pdflatex comprehensive_report_milp.tex  (twice)

Definitions
  free-choice EV   an EV with at least one station in range (stranded EVs must take a van)
  income band      theta in [0,0.2), [0.2,0.4), [0.4,0.6), [0.6,0.8), [0.8,1]
  demand pressure  requests released in [r_i - mean FCS charging time, r_i] / (guns + vans) = 29 chargers
                   (f4_data.peak_flags); terciles pooled over all free-choice EVs of valid plans
  optimal          Gurobi proved the plan optimal; time-limit plans are valid but not proven optimal
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import report_common as rc  # noqa: E402
F4J = rc.F4J
import matplotlib.pyplot as plt  # noqa: E402

DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")
TAB = os.path.join(HERE, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)
N = rc.Numbers()
P = json.load(open(os.path.join(rc.F4_COMMON, "f4_params.json")))
META = json.load(open(F4J("baseline_MILP", "results", "meta.json")))["milp_300s"]
SEGS = rc.SEGMENTS
SEGL = {"personal": "personal", "commercial": "commercial"}
F = rc.fmt


def pct(x, d=1):
    return rc.fmt(100 * x, "{:." + str(d) + "f}")


def tab(name, header, rows, align=None, total=None):
    """Bare tabular (caption lives in the .tex)."""
    align = align or ("l" + "r" * (len(header) - 1))
    out = [f"\\begin{{tabular}}{{{align}}}\\toprule", " & ".join(header) + " \\\\ \\midrule"]
    out += [" & ".join(str(x) for x in r) + " \\\\" for r in rows]
    if total:
        out.append("\\midrule " + " & ".join(str(x) for x in total) + " \\\\")
    out.append("\\bottomrule\\end{tabular}")
    with open(os.path.join(TAB, name + ".tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")


def save(fig, name):
    rc.savefig(fig, os.path.join(FIG, name + ".pdf"))


# =============================================================================================== load
inst = pd.read_csv(os.path.join(DATA, "instances.csv"))
ev = pd.read_csv(os.path.join(DATA, "ev_level.csv"))
vans = pd.read_csv(os.path.join(DATA, "vans.csv"))
dc = pd.read_csv(os.path.join(DATA, "duals_constraints.csv"))
di = pd.read_csv(os.path.join(DATA, "duals_instances.csv"))
valid = inst[inst.valid].copy()
ev["band"] = rc.band_of(ev.theta)
ev["decile"] = np.clip(np.floor(ev.theta * 10).astype(int), 0, 9)
ev["comm"] = (ev.segment == "commercial").astype(int)
ev["optimal"] = ev.status == "optimal"
# shared demand-pressure definition of the three reports (../common/infra.py): data only, terciles pooled over
# the free-choice EVs of all 560 baseline instances
import infra  # noqa: E402
PT = infra.pressure_table()
lo_t, hi_t = infra.pressure_cuts()
ev = ev.merge(PT[["instance", "n", "id", "load", "window_min", "pressure", "terc"]].rename(
    columns={"pressure": "pressure_shared"}), on=["instance", "n", "id"], how="left")
assert ev.pressure_shared.notna().all() and (ev.pressure - ev.pressure_shared).abs().max() < 1e-3
ev["ptc"] = ev.terc
free = ev[~ev.stranded].copy()
N["PressLow"] = F(lo_t, "{:.4f}")
N["PressHigh"] = F(hi_t, "{:.4f}")
N["PressLowLoad"] = F(lo_t * 29, "{:.0f}")
N["PressHighLoad"] = F(hi_t * 29, "{:.0f}")
PTF = PT[~PT.stranded.astype(bool)]
N["NPressPool"] = f"{len(PTF):,}".replace(",", "{,}")
N["NPressPoolInst"] = PT.groupby(["instance", "n"]).ngroups
for t_ in ("low", "medium", "high"):
    N["PoolShare" + t_.capitalize()] = pct((PTF.terc == t_).mean())
    N["MilpShare" + t_.capitalize()] = pct((free.ptc == t_).mean())
win = PT.groupby(["instance", "n"]).window_min.first()
N["TauBarMin"] = F(win.min(), "{:.0f}")
N["TauBarMax"] = F(win.max(), "{:.0f}")
N["PeakLabelShare"] = pct((PTF.period == "peak").mean())
N["PeakLabelHighShare"] = pct((PTF[PTF.period == "peak"].terc == "high").mean())

# =============================================================================================== Part I
N["TimeLimit"] = F(META["time_limit_s"], "{:.0f}")
N["Workers"] = META["workers"]
N["Threads"] = META["threads_per_instance"]
N["Cores"] = META["cores"]
N["MipGap"] = F(100 * META["mip_gap"], "{:.2f}")
N["PiC"] = F(P["segment_distribution"]["pi_commercial"], "{:.2f}")
N["NEVsValid"] = f"{len(ev):,}".replace(",", "{,}")
N["CommShareAll"] = pct(ev.comm.mean())
N["CommShareFree"] = pct(free.comm.mean())
N["StrandedShare"] = pct(ev.stranded.mean())
for s in SEGS:
    cap = s.capitalize()
    N[f"MeanTheta{cap}"] = F(ev[ev.segment == s].theta.mean(), "{:.2f}")


def pline(seg, key, sub=None):
    d = P[seg][key] if sub is None else P[seg][key][sub]
    return d["theta0"], d["theta1"]


rows = []
for lab, unit, key, sub in [("money sensitivity $\\mu$", "Rs/Rs", "money_sensitivity_mu", None),
                            ("value of time $c^{\\mathrm{D}}$ (driving to an FCS)", "Rs/min", "value_of_time_Rs_per_min", "D"),
                            ("value of time $c^{\\mathrm{Q}}$ (queueing at an FCS)", "Rs/min", "value_of_time_Rs_per_min", "Q"),
                            ("value of time $c^{\\mathrm{C}}$ (charging at an FCS)", "Rs/min", "value_of_time_Rs_per_min", "C"),
                            ("value of time $c^{\\mathrm{V}}$ (van wait + van charging)", "Rs/min", "value_of_time_Rs_per_min", "V"),
                            ("queue tolerance $W$", "min", "queue_tolerance_W_min", None)]:
    a0, a1 = pline("personal", key, sub)
    b0, b1 = pline("commercial", key, sub)
    rows.append([lab, unit, F(a0, "{:g}"), F(a1, "{:g}"), F(b0, "{:g}"), F(b1, "{:g}")])
tp = P["personal"]["theta_distribution"]
tcm = P["commercial"]["theta_distribution"]
rows.insert(0, ["income $\\theta$ given the segment", "--", f"\\multicolumn{{2}}{{c}}{{Beta({tp['a']:g}, {tp['b']:g})}}",
                f"\\multicolumn{{2}}{{c}}{{Beta({tcm['a']:g}, {tcm['b']:g})}}"])
rows.append(["tariff $p^{\\mathrm{F}}$ / $p^{\\mathrm{M}}$ (both segments)", "Rs/kWh",
             f"\\multicolumn{{4}}{{c}}{{{P['tariff_Rs_per_kWh']['p_F']:g} / {P['tariff_Rs_per_kWh']['p_M']:g}}}"])
with open(os.path.join(TAB, "params.tex"), "w", encoding="utf-8") as fh:
    fh.write("\\begin{tabular}{llrrrr}\\toprule\n & & \\multicolumn{2}{c}{personal} & \\multicolumn{2}{c}{commercial} \\\\\n"
             "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6} parameter & unit & $\\theta=0$ & $\\theta=1$ & $\\theta=0$ & $\\theta=1$ \\\\ \\midrule\n")
    for r in rows:
        fh.write(" & ".join(r) + " \\\\\n")
    fh.write("\\bottomrule\\end{tabular}\n")
N["DeltaZero"] = F(P["commercial"]["value_of_time_Rs_per_min"]["Q"]["theta0"]
                   - P["commercial"]["value_of_time_Rs_per_min"]["D"]["theta0"], "{:g}")
N["DeltaOne"] = F(P["commercial"]["value_of_time_Rs_per_min"]["Q"]["theta1"]
                  - P["commercial"]["value_of_time_Rs_per_min"]["D"]["theta1"], "{:g}")
N["OmegaZero"] = F(P["commercial"]["value_of_time_Rs_per_min"]["D"]["theta0"], "{:g}")
N["OmegaOne"] = F(P["commercial"]["value_of_time_Rs_per_min"]["D"]["theta1"], "{:g}")

# =============================================================================================== Part II
# --- 3.1 outcome by n
stat = inst.groupby("n").status.value_counts().unstack(fill_value=0)
for c in ["optimal", "time_limit", "time_limit_no_plan", "infeasible"]:
    if c not in stat:
        stat[c] = 0
fig, ax = plt.subplots(figsize=(4.6, 2.8))
bottom = np.zeros(len(stat))
labels = {"optimal": "optimal", "time_limit": "time limit, plan", "time_limit_no_plan": "time limit, no plan",
          "infeasible": "infeasible"}
for c in ["optimal", "time_limit", "time_limit_no_plan", "infeasible"]:
    ax.bar(stat.index.astype(str), stat[c], bottom=bottom, color=rc.STATUS_COL[c], label=labels[c], width=0.7)
    bottom += stat[c].values
ax.set_xlabel("EVs per instance $n$")
ax.set_ylabel("instances (of 56)")
ax.legend(ncol=2, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.25))
save(fig, "solver_outcome")
rows = []
for n_, g in inst.groupby("n"):
    tl = g[g.status == "time_limit"]
    rows.append([n_, len(g), (g.status == "optimal").sum(), (g.status == "time_limit").sum(),
                 (g.status == "time_limit_no_plan").sum(), (g.status == "infeasible").sum(),
                 F(tl.gap.median(), "{:.2f}") if len(tl) else "--", F(g.solve_s.mean(), "{:.0f}")])
tlall = inst[inst.status == "time_limit"]
tab("solver", ["$n$", "instances", "optimal", "\\shortstack{time limit\\\\with plan}",
               "\\shortstack{time limit\\\\no plan}", "infeasible", "\\shortstack{median\\\\gap (\\%)}",
               "\\shortstack{mean\\\\time (s)}"], rows,
    total=["all", len(inst), (inst.status == "optimal").sum(), len(tlall), (inst.status == "time_limit_no_plan").sum(),
           (inst.status == "infeasible").sum(), F(tlall.gap.median(), "{:.2f}"), F(inst.solve_s.mean(), "{:.0f}")])
N["NInst"] = len(inst)
N["NOpt"] = int((inst.status == "optimal").sum())
N["NTL"] = len(tlall)
N["NNone"] = int((inst.status == "time_limit_no_plan").sum())
N["NInf"] = int((inst.status == "infeasible").sum())
N["NValid"] = len(valid)
N["GapMed"] = F(tlall.gap.median(), "{:.1f}")
N["GapMean"] = F(tlall.gap.mean(), "{:.1f}")
N["OptAtTen"] = int(stat.loc[10, "optimal"])
N["OptAtHundred"] = int(stat.loc[100, "optimal"])
first_tl_major = int(min(n_ for n_ in stat.index if stat.loc[n_, "time_limit"] > stat.loc[n_, "optimal"]))
N["NFirstTLMajor"] = first_tl_major
N["NFirstNoPlan"] = int(min(n_ for n_ in stat.index if stat.loc[n_, "time_limit_no_plan"] > 0))
inf = inst[inst.status == "infeasible"]
N["InfFamilies"] = " and ".join(sorted(inf.family.unique()))
N["InfMinN"] = int(inf.n.min())
N["SolveMean"] = F(inst.solve_s.mean(), "{:.0f}")

# --- 3.2 instance level
valid["Zper"] = valid.Z / valid.n
valid["opshare"] = 100 * valid.operator_cost_Z2_to_Z4 / valid.Z
valid["vanpct"] = 100 * valid.EVs_by_van / valid.n
valid["freevanpct"] = 100 * valid.free_choice_by_van / valid.free_choice_EVs
rows = []
for lab, s in [("objective $Z$ per EV (Rs)", valid.Zper), ("operator share of $Z$ (\\%)", valid.opshare),
               ("vans used", valid.vans_used), ("EVs served by a van (\\%)", valid.vanpct),
               ("free-choice EVs by van (\\%)", valid.freevanpct), ("mean FCS queue (min)", valid.mean_queue_FCS_min),
               ("mean van wait (min)", valid.mean_van_wait_min.dropna()),
               ("mean request to charged (min)", valid.mean_time_to_charged_min), ("van km", valid.van_km),
               ("solver time (s), all instances", inst.solve_s), ("gap of time-limit plans (\\%)", tlall.gap)]:
    s = pd.Series(s).dropna()
    fm = "{:.0f}" if "per EV" in lab else ("{:.2f}" if "queue" in lab or "gap" in lab else "{:.1f}")
    rows.append([lab, len(s), F(s.mean(), fm), F(s.median(), fm), F(s.quantile(.1), fm), F(s.quantile(.9), fm),
                 F(s.min(), fm), F(s.max(), fm)])
tab("inst_stats", ["measure (per instance)", "count", "mean", "median", "10th pct", "90th pct", "min", "max"], rows)
N["ZperMean"] = F(valid.Zper.mean(), "{:.0f}")
N["ZperMin"] = F(valid.Zper.min(), "{:.0f}")
N["ZperMax"] = F(valid.Zper.max(), "{:.0f}")
N["VansMean"] = F(valid.vans_used.mean(), "{:.1f}")
N["VanPctMean"] = F(valid.vanpct.mean(), "{:.0f}")
N["FreeVanPctMean"] = F(valid.freevanpct.mean(), "{:.1f}")
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
ns = sorted(inst.n.unique())
axs[0].boxplot([valid[valid.n == k].Zper for k in ns], tick_labels=ns, showfliers=False)
axs[0].set_ylabel("Z per EV (Rs)")
axs[1].boxplot([inst[inst.n == k].solve_s for k in ns], tick_labels=ns, showfliers=False)
axs[1].set_ylabel("solver time (s)")
axs[2].boxplot([tlall[tlall.n == k].gap.values if (tlall.n == k).any() else [np.nan] for k in ns], tick_labels=ns,
               showfliers=False)
axs[2].set_ylabel("gap of time-limit plans (%)")
for a in axs:
    a.set_xlabel("$n$")
    a.tick_params(axis="x", labelsize=6)
save(fig, "inst_by_n")

# --- 3.3 families
rows = []
comp_rows = []
for fam in rc.FAMS:
    g = inst[inst.family == fam]
    v = valid[valid.family == fam]
    e = ev[ev.family == fam]
    vv = vans[vans.family == fam]
    rows.append([fam, len(g), len(v), F(v.Zper.mean(), "{:.0f}"), F(v.opshare.mean(), "{:.1f}"),
                 pct(e.comm.mean()), pct(free[(free.family == fam) & (free.segment == "personal")].van.mean()),
                 pct(free[(free.family == fam) & (free.segment == "commercial")].van.mean()),
                 F(v.mean_queue_FCS_min.mean(), "{:.2f}"), F(v.mean_time_to_charged_min.mean(), "{:.1f}"),
                 F(v.vans_used.mean(), "{:.2f}"), int(g["T"].iloc[0])])
    z = v[["Z1_EV_drive_perceived", "Z2_van_drive", "Z3_dispatch", "Z4_operator_energy", "Z5_tariff_perceived",
           "Z6_time_perceived"]].sum()
    comp_rows.append(100 * z.values / z.sum())
tab("family", ["family", "instances", "\\shortstack{valid\\\\plans}", "\\shortstack{Z per EV\\\\(Rs)}",
               "\\shortstack{operator\\\\share (\\%)}", "\\shortstack{commercial\\\\EVs (\\%)}",
               "\\shortstack{van share\\\\personal (\\%)}", "\\shortstack{van share\\\\commercial (\\%)}",
               "\\shortstack{mean queue\\\\(min)}", "\\shortstack{request to\\\\charged (min)}",
               "\\shortstack{vans\\\\used}", "$T$ (min)"], rows)
comp_rows = np.array(comp_rows)
fig, ax = plt.subplots(figsize=(5.4, 2.8))
cols = ["#8172B2", "#64B5CD", "#4C72B0", "#55A868", "#DD8452", "#C44E52"]
labs = ["$Z_1$ EV driving", "$Z_2$ van driving", "$Z_3$ van dispatch", "$Z_4$ operator energy", "$Z_5$ tariff",
        "$Z_6$ time"]
b = np.zeros(6)
for k in range(6):
    ax.bar(rc.FAMS, comp_rows[:, k], bottom=b, color=cols[k], label=labs[k], width=0.6)
    b += comp_rows[:, k]
ax.set_ylabel("share of $Z$ (%)")
ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
save(fig, "obj_family")
vf = valid.groupby("family")
N["ValidROne"] = int((valid.family == "R1").sum())
N["InstROne"] = int((inst.family == "R1").sum())
N["ValidRCOne"] = int((valid.family == "RC1").sum())
N["InstRCOne"] = int((inst.family == "RC1").sum())
N["ZperRCOneMean"] = F(valid[valid.family == "RC1"].Zper.mean(), "{:.0f}")
N["ZperRTwoMean"] = F(valid[valid.family == "R2"].Zper.mean(), "{:.0f}")
N["OpShareMin"] = F(vf.opshare.mean().min(), "{:.0f}")
N["OpShareMax"] = F(vf.opshare.mean().max(), "{:.0f}")

# --- 3.4 segments
rows = []
for s in SEGS:
    e = ev[ev.segment == s]
    fr = free[free.segment == s]
    fs = e[(e.resource_type == "FCS")]
    vu = e[e.van]
    rows.append([s, len(e), pct(len(e) / len(ev)), pct(e.stranded.mean()), F(e.theta.mean(), "{:.2f}"),
                 pct(fr.van.mean()), F(fs.queue.mean(), "{:.2f}"), pct((fs.queue > 1e-6).mean()),
                 F(vu.van_wait.mean(), "{:.1f}"), F(e.downtime.mean(), "{:.1f}"), F(e.pay.mean(), "{:.0f}"),
                 F(e.perc_total.mean(), "{:.0f}")])
    cap = s.capitalize()
    N[f"VanShare{cap}"] = pct(fr.van.mean())
    N[f"Queue{cap}"] = F(fs.queue.mean(), "{:.2f}")
    N[f"Downtime{cap}"] = F(e.downtime.mean(), "{:.1f}")
    N[f"Perc{cap}"] = F(e.perc_total.mean(), "{:.0f}")
    N[f"Pay{cap}"] = F(e.pay.mean(), "{:.0f}")
    N[f"NEV{cap}"] = f"{len(e):,}".replace(",", "{,}")
    N[f"NFree{cap}"] = f"{len(fr):,}".replace(",", "{,}")
    N[f"NFreeVan{cap}"] = int(fr.van.sum())
tab("segments", ["segment", "EVs", "\\shortstack{share\\\\(\\%)}", "\\shortstack{stranded\\\\(\\%)}",
                 "\\shortstack{mean\\\\$\\theta$}", "\\shortstack{free-choice\\\\by van (\\%)}",
                 "\\shortstack{mean FCS\\\\queue (min)}", "\\shortstack{FCS users\\\\queueing (\\%)}",
                 "\\shortstack{mean van\\\\wait (min)}", "\\shortstack{downtime\\\\(min)}",
                 "\\shortstack{paid\\\\(Rs)}", "\\shortstack{perceived\\\\cost (Rs)}"], rows)
vanfree = free[free.van]
N["CommShareOfFreeVan"] = pct(vanfree.comm.mean())
N["PersShareOfFreeVan"] = pct(1 - vanfree.comm.mean())

# --- 3.5 EV-level
rows = []
fcs_users = ev[ev.resource_type == "FCS"]
van_users = ev[ev.van]
for lab, s, fm in [("income $\\theta$", ev.theta, "{:.2f}"), ("energy demand $E_i$ (kWh)", ev.E, "{:.1f}"),
                   ("nearest reachable FCS (km)", free.nearest_fcs_km, "{:.2f}"),
                   ("demand pressure", free.pressure, "{:.2f}"),
                   ("drive to FCS (km), FCS users", fcs_users.drive_km, "{:.2f}"),
                   ("FCS queue (min), FCS users", fcs_users.queue, "{:.2f}"),
                   ("van wait (min), van users", van_users.van_wait, "{:.1f}"),
                   ("charging time (min)", ev.charge_min, "{:.1f}"),
                   ("request to charged (min)", ev.time_to_charged, "{:.1f}"),
                   ("money paid (Rs)", ev.pay, "{:.0f}"), ("perceived money (Rs)", ev.perc_money, "{:.0f}"),
                   ("perceived time (Rs)", ev.perc_time, "{:.0f}"), ("perceived total (Rs)", ev.perc_total, "{:.0f}")]:
    rows.append([lab, f"{len(s)}", F(s.mean(), fm), F(s.median(), fm), F(s.quantile(.1), fm), F(s.quantile(.9), fm),
                 F(s.max(), fm)])
tab("ev_stats", ["variable (per EV)", "count", "mean", "median", "10th pct", "90th pct", "max"], rows)
N["TTCTen"] = F(ev.time_to_charged.quantile(.1), "{:.0f}")
N["TTCNinety"] = F(ev.time_to_charged.quantile(.9), "{:.0f}")
ffcs = free[free.resource_type == "FCS"]
N["NQueued"] = int((ffcs.queue > 1e-6).sum())
N["NFCSfree"] = f"{len(ffcs):,}".replace(",", "{,}")
N["QueueMax"] = F(ffcs.queue.max(), "{:.0f}")
N["VanWaitZeroShare"] = pct((van_users.van_wait < 1e-6).mean(), 0)
fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.4))
a = axs[0, 0]
for s, ls in (("personal", "-"), ("commercial", "--")):
    for res, col in (("FCS", "#4C72B0"), ("MCS", "#C44E52")):
        d = free[(free.segment == s) & (free.resource_type == res)].theta
        if len(d) > 20:
            a.hist(d, bins=20, range=(0, 1), density=True, histtype="step", color=col, ls=ls,
                   label=f"{s}, {'station' if res == 'FCS' else 'van'}")
a.set_title("income of free-choice EVs", fontsize=9)
a.set_xlabel("income $\\theta$")
a.set_ylabel("density")
a.legend(fontsize=6)
a = axs[0, 1]
for res, col in (("FCS", "#4C72B0"), ("MCS", "#C44E52")):
    a.hist(ev[ev.resource_type == res].time_to_charged, bins=40, range=(0, 200), alpha=.6, color=col,
           label="FCS" if res == "FCS" else "van")
a.set_title("time until fully charged", fontsize=9)
a.set_xlabel("request to charged (min)")
a.set_ylabel("EVs")
a.legend(fontsize=7)
a = axs[0, 2]
for s in SEGS:
    a.hist(ffcs[(ffcs.queue > 1e-6) & (ffcs.segment == s)].queue, bins=30, alpha=.6, color=rc.SEG_COL[s], label=s)
a.set_title("FCS queues above zero", fontsize=9)
a.set_xlabel("queue (min)")
a.set_ylabel("EVs")
a.legend(fontsize=7)
a = axs[1, 0]
for s in SEGS:
    a.hist(van_users[van_users.segment == s].van_wait, bins=30, range=(0, van_users.van_wait.max()), alpha=.6,
           color=rc.SEG_COL[s], label=s)
a.set_title("van waits (all van users)", fontsize=9)
a.set_xlabel("wait for the van (min)")
a.set_ylabel("EVs")
a.legend(fontsize=7)
a = axs[1, 1]
for res, col in (("FCS", "#4C72B0"), ("MCS", "#C44E52")):
    a.hist(ev[ev.resource_type == res].pay, bins=40, alpha=.6, color=col, label="FCS" if res == "FCS" else "van")
a.set_title("money paid", fontsize=9)
a.set_xlabel("money paid (Rs)")
a.set_ylabel("EVs")
a.legend(fontsize=7)
a = axs[1, 2]
for s in SEGS:
    a.hist(ev[ev.segment == s].perc_total, bins=40, alpha=.6, color=rc.SEG_COL[s], label=s)
a.set_title("perceived cost", fontsize=9)
a.set_xlabel("perceived cost (Rs)")
a.set_ylabel("EVs")
a.legend(fontsize=7)
save(fig, "ev_dists")

# --- 3.6 correlations (free-choice EVs, Spearman)
cv = pd.DataFrame({"$\\theta$": free.theta, "commercial": free.comm, "$E$": free.E,
                   "nearest FCS": free.nearest_fcs_km, "pressure": free.pressure, "van": free.van.astype(int),
                   "queue": free.queue, "time to charged": free.time_to_charged, "paid": free.pay,
                   "perc. money": free.perc_money, "perc. time": free.perc_time})
C = cv.corr(method="spearman")
fig, ax = plt.subplots(figsize=(4.8, 4.2))
im = ax.imshow(C.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(C)))
ax.set_xticklabels(C.columns, rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(len(C)))
ax.set_yticklabels(C.columns, fontsize=7)
for i in range(len(C)):
    for j in range(len(C)):
        ax.text(j, i, f"{C.values[i, j]:.2f}", ha="center", va="center", fontsize=5,
                color="white" if abs(C.values[i, j]) > .6 else "black")
fig.colorbar(im, ax=ax, fraction=0.046)
ax.spines[:].set_visible(True)
save(fig, "corr")
N["CorrThetaVan"] = F(C.loc["$\\theta$", "van"], "{:.2f}")
N["CorrCommVan"] = F(C.loc["commercial", "van"], "{:.2f}")
N["CorrThetaComm"] = F(C.loc["$\\theta$", "commercial"], "{:.2f}")
N["CorrThetaQueue"] = F(C.loc["$\\theta$", "queue"], "{:.2f}")
N["CorrCommQueue"] = F(C.loc["commercial", "queue"], "{:.2f}")
N["CorrPressQueue"] = F(C.loc["pressure", "queue"], "{:.2f}")
N["CorrEPay"] = F(C.loc["$E$", "paid"], "{:.2f}")
N["CorrThetaPercMoney"] = F(C.loc["$\\theta$", "perc. money"], "{:.2f}")
N["CorrThetaPercTime"] = F(C.loc["$\\theta$", "perc. time"], "{:.2f}")
N["CorrPersVanWithin"] = F(free[free.segment == "personal"][["theta", "van"]].corr(method="spearman").iloc[0, 1], "{:.2f}")

# =============================================================================================== Part III
# --- 4.1 anatomy
zc = ["Z1_EV_drive_perceived", "Z2_van_drive", "Z3_dispatch", "Z4_operator_energy", "Z5_tariff_perceived",
      "Z6_time_perceived"]
zs = valid[zc].sum()
Ztot = valid.Z.sum()
for k, nm in enumerate(["ZOne", "ZTwo", "ZThree", "ZFour", "ZFive", "ZSix"]):
    N[nm] = F(100 * zs.iloc[k] / Ztot, "{:.1f}")
N["ZOperator"] = F(100 * valid.operator_cost_Z2_to_Z4.sum() / Ztot, "{:.1f}")
N["ZCustomer"] = F(100 * valid.customer_cost_Z1_Z5_Z6.sum() / Ztot, "{:.1f}")
N["ZRecMax"] = F((valid.Z_reconstructed - valid.Z).abs().max(), "{:.2f}")
# customer cost share by segment
cust = ev.groupby("segment").perc_total.sum()
N["CustShareComm"] = pct(cust["commercial"] / cust.sum())
N["TimeShareCustComm"] = pct(ev[ev.segment == "commercial"].perc_time.sum() / ev[ev.segment == "commercial"].perc_total.sum())
N["TimeShareCustPers"] = pct(ev[ev.segment == "personal"].perc_time.sum() / ev[ev.segment == "personal"].perc_total.sum())

cfgv = {"c_EV": 4.0, "c_M": 15.0, "c_F": 1500.0, "pi_F": 8.7, "pi_M": 10.7}
cfgcsv = pd.read_csv(os.path.join(rc.F4_DATA, "solomon_config.csv")).iloc[0]
cfgv = {"c_EV": cfgcsv.c_EV_Rs_per_km, "c_M": cfgcsv.c_M_Rs_per_km, "c_F": cfgcsv.c_F_Rs_per_van,
        "pi_F": cfgcsv.pi_F_Rs_per_kWh, "pi_M": cfgcsv.pi_M_Rs_per_kWh}
coef = [("c_EV", "$c^{\\mathrm{EV}}$", cfgv["c_EV"], "EV driving cost (Rs/km)", "op"),
        ("c_M", "$c^{\\mathrm{M}}$", cfgv["c_M"], "van driving cost (Rs/km)", "op"),
        ("c_F", "$c^{\\mathrm{F}}_m$", cfgv["c_F"], "van dispatch cost (Rs)", "op"),
        ("pi_F", "$\\pi^{\\mathrm{F}}$", cfgv["pi_F"], "grid energy at FCS (Rs/kWh)", "op"),
        ("pi_M", "$\\pi^{\\mathrm{M}}$", cfgv["pi_M"], "van battery energy (Rs/kWh)", "op"),
        ("p_F", "$p^{\\mathrm{F}}$", P["tariff_Rs_per_kWh"]["p_F"], "FCS tariff (Rs/kWh)", "tariff"),
        ("p_M", "$p^{\\mathrm{M}}$", P["tariff_Rs_per_kWh"]["p_M"], "van tariff (Rs/kWh)", "tariff")]
SYM = {"personal": "P", "commercial": "C"}
for s in SEGS:
    sy = SYM[s]
    for e_ in ("0", "1"):
        coef.append((f"mu{e_}_{s}", f"$\\mu^{{{sy},{e_}}}$", P[s]["money_sensitivity_mu"]["theta" + e_],
                     f"money sensitivity, {s}, $\\theta={e_}$", "money"))
    acts = ("D", "Q", "C", "V")
    for a in acts:
        for e_ in ("0", "1"):
            coef.append((f"c{a}{e_}_{s}", f"$c^{{\\mathrm{{{a}}},{sy},{e_}}}$",
                         P[s]["value_of_time_Rs_per_min"][a]["theta" + e_],
                         f"value of time {a}, {s}, $\\theta={e_}$", "time"))
el_rows, el_plot = [], []
el = {}
for key, sym, val, mean_, grp in coef:
    g = valid["g_" + key]
    der = g.mean()
    e_ = 100 * val * g.sum() / Ztot
    el[key] = e_
    el_rows.append([sym, F(val, "{:g}"), mean_, F(der, "{:.1f}"), F(e_, "{:.2f}"), F(0.1 * val * der, "{:.0f}")])
    el_plot.append((sym, e_, grp))
tab("elasticity", ["coefficient", "value", "meaning", "\\shortstack{$\\partial Z/\\partial$coef.\\\\per plan}",
                   "\\shortstack{elasticity\\\\(\\%)}", "\\shortstack{Rs per plan\\\\for $+10\\%$}"], el_rows,
    align="llp{5.2cm}rrr")
el_plot.sort(key=lambda x: x[1])
fig, ax = plt.subplots(figsize=(5.2, 5.6))
colg = {"op": "#55A868", "tariff": "#DD8452", "money": "#4C72B0", "time": "#C44E52"}
ax.barh([x[0] for x in el_plot], [x[1] for x in el_plot], color=[colg[x[2]] for x in el_plot])
ax.set_xlabel("elasticity of $Z$ (% change of $Z$ per 1% change of the coefficient)")
ax.tick_params(axis="y", labelsize=7)
from matplotlib.patches import Patch  # noqa: E402
ax.legend(handles=[Patch(color=colg[k], label=l_) for k, l_ in
                   [("op", "operator cost"), ("tariff", "tariff"), ("money", "money sensitivity"),
                    ("time", "value of time")]], fontsize=7, loc="lower right")
save(fig, "elasticity")
top = sorted(el.items(), key=lambda kv: -kv[1])
symd = {k: s for k, s, *_ in coef}
N["ElTopOneSym"] = symd[top[0][0]]
N["ElTopOne"] = F(top[0][1], "{:.1f}")
N["ElTopTwoSym"] = symd[top[1][0]]
N["ElTopTwo"] = F(top[1][1], "{:.1f}")
N["ElTopThreeSym"] = symd[top[2][0]]
N["ElTopThree"] = F(top[2][1], "{:.1f}")
N["ElTopFourSym"] = symd[top[3][0]]
N["ElTopFour"] = F(top[3][1], "{:.1f}")
N["ElPF"] = F(el["p_F"], "{:.1f}")
N["ElPFTen"] = F(el["p_F"] / 10, "{:.1f}")
N["ElPiF"] = F(el["pi_F"], "{:.1f}")
N["ElCF"] = F(el["c_F"], "{:.1f}")
N["ElOperator"] = F(sum(el[k] for k in ("c_M", "c_F", "pi_F", "pi_M")), "{:.1f}")
N["ElMoney"] = F(sum(v for k, v in el.items() if k.startswith("mu")), "{:.1f}")
N["ElTariffs"] = F(el["p_F"] + el["p_M"] + el["c_EV"], "{:.1f}")
N["ElTime"] = F(sum(v for k, v in el.items() if k[0] == "c" and k[1] in "DQCV"), "{:.1f}")
N["ElMoneyComm"] = F(el["mu0_commercial"] + el["mu1_commercial"], "{:.1f}")
N["ElMoneyPers"] = F(el["mu0_personal"] + el["mu1_personal"], "{:.1f}")
N["ElTimeComm"] = F(sum(v for k, v in el.items() if k[0] == "c" and k[1] in "DQCV" and k.endswith("commercial")),
                    "{:.1f}")
N["ElTimePers"] = F(sum(v for k, v in el.items() if k[0] == "c" and k[1] in "DQCV" and k.endswith("personal")),
                    "{:.1f}")
N["ElQueueAll"] = F(sum(v for k, v in el.items() if k.startswith("cQ")), "{:.2f}")
N["ElVanTimePers"] = F(el["cV0_personal"] + el["cV1_personal"], "{:.2f}")
N["ElOmegaComm"] = F(sum(el[f"c{a}{e_}_commercial"] for a in "DCV" for e_ in "01"), "{:.1f}")

# --- 4.2 duals
nz = lambda s: s[s.abs() > 1e-9]  # noqa: E731
rows = []
meaning = {"arrival": "not unique (degenerate); not interpreted",
           "release": "not unique (degenerate); not interpreted",
           "gun": "value of one gun-minute: delay EV $i$ passes to those behind it",
           "van": "value of one van-minute on that leg",
           "depart": "value of an earlier van departure",
           "patience": "value of one more minute of patience",
           "deadline": "value of one more minute of horizon"}
lab = {"arrival": "FCS arrival $S_i\\ge r_i+t_{if}$", "release": "van release $S_i\\ge r_i$",
       "gun": "gun order $S_j\\ge S_i+\\tau^{\\mathrm{F}}_i$", "van": "van order $S_j\\ge S_i+\\tau^{\\mathrm{M}}_i+t_{ij}$",
       "depart": "first stop $S_a\\ge t_{0a}$"}
for k in ["arrival", "release", "gun", "van", "depart"]:
    d = dc[dc.kind == k].dual
    z = nz(d).abs()
    rows.append([lab[k], len(d), pct(len(z) / len(d)), F(z.mean(), "{:.2f}"), F(z.median(), "{:.2f}"),
                 F(z.max(), "{:.2f}"), meaning[k]])
n_pat = int(di.patience_rows.sum())
n_dl = int(di.deadline_rows.sum())
rows.append(["queue limit $q^{\\mathrm{F}}_i\\le W_i$", n_pat, F(0, "{:.1f}"), "0.00", "0.00",
             F(di.max_abs_patience_dual.max(), "{:.2f}"), meaning["patience"]])
rows.append(["deadline $C_i\\le T$, van return", n_dl, F(0, "{:.1f}"), "0.00", "0.00",
             F(di.max_abs_deadline_dual.max(), "{:.2f}"), meaning["deadline"]])
tab("duals", ["constraint", "rows", "\\shortstack{non-zero\\\\dual (\\%)}", "\\shortstack{mean\\\\(Rs/min)}",
              "\\shortstack{median\\\\(Rs/min)}", "\\shortstack{max\\\\(Rs/min)}", "meaning of the dual"], rows,
    align="p{3.6cm}rrrrrp{4.4cm}")
N["NPatience"] = f"{n_pat:,}".replace(",", "{,}")
N["NDeadline"] = f"{n_dl:,}".replace(",", "{,}")
N["LPDiffMax"] = F(di.lp_minus_plan.abs().max(), "{:.2f}")
N["LPDiffMedian"] = F(di.lp_minus_plan.abs().median(), "{:.3f}")
N["NLPPlans"] = len(di)
gun = dc[dc.kind == "gun"].copy()
gun["bind"] = gun.dual.abs() > 1e-9
gun["bi"] = rc.band_of(gun.theta_i)
gun["bj"] = rc.band_of(gun.theta_j)
N["NGunLinks"] = f"{len(gun):,}".replace(",", "{,}")
N["GunBindPct"] = pct(gun.bind.mean())
N["GunMean"] = F(gun[gun.bind].dual.abs().mean(), "{:.1f}")
N["GunMedian"] = F(gun[gun.bind].dual.abs().median(), "{:.1f}")
N["GunMax"] = F(gun[gun.bind].dual.abs().max(), "{:.1f}")
vd = dc[dc.kind == "van"]
N["VanBindPct"] = pct((vd.dual.abs() > 1e-9).mean())
N["VanDualMean"] = F(nz(vd.dual).abs().mean(), "{:.1f}")
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.5))
axs[0].hist(gun[gun.bind].dual.abs(), bins=30, color="#C44E52")
axs[0].set_xlabel("gun-minute shadow price (Rs/min)")
axs[0].set_ylabel("binding gun links")
bi = gun.groupby("bi").bind.mean() * 100
bj = gun.groupby("bj").bind.mean() * 100
x = np.arange(5)
axs[1].bar(x - .2, bi.reindex(range(5)).values, .4, color="#4C72B0", label="EV on the gun ($i$)")
axs[1].bar(x + .2, bj.reindex(range(5)).values, .4, color="#DD8452", label="EV behind it ($j$)")
axs[1].set_xticks(x)
axs[1].set_xticklabels(rc.PLAB, rotation=40, fontsize=6)
axs[1].set_ylabel("gun links binding (%)")
axs[1].set_xlabel("income band $\\theta$")
axs[1].legend(fontsize=6)
segpair = gun.groupby(["seg_i", "seg_j"]).bind.mean().unstack() * 100
segpair = segpair.reindex(index=SEGS, columns=SEGS)
im = axs[2].imshow(segpair.values, cmap="Reds", vmin=0)
axs[2].set_xticks([0, 1])
axs[2].set_xticklabels(["personal", "commercial"], fontsize=7)
axs[2].set_yticks([0, 1])
axs[2].set_yticklabels(["personal", "commercial"], fontsize=7)
axs[2].set_xlabel("EV behind ($j$)", fontsize=8)
axs[2].set_ylabel("EV on the gun ($i$)", fontsize=8)
for i in range(2):
    for j in range(2):
        axs[2].text(j, i, f"{segpair.values[i, j]:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if segpair.values[i, j] > 0.6 * np.nanmax(segpair.values) else "black")
save(fig, "gun_duals")
N["GunBindRichOn"] = pct(gun[gun.bi == 4].bind.mean())
N["GunBindPoorOn"] = pct(gun[gun.bi == 0].bind.mean())
N["GunBindPoorBehind"] = pct(gun[gun.bj == 0].bind.mean())
N["GunBindRichBehind"] = pct(gun[gun.bj == 4].bind.mean())
for si, nmi in (("personal", "Pers"), ("commercial", "Comm")):
    for sj, nmj in (("personal", "Pers"), ("commercial", "Comm")):
        N[f"GunBind{nmi}On{nmj}Behind"] = F(segpair.loc[si, sj], "{:.1f}")
        cnt = gun[(gun.seg_i == si) & (gun.seg_j == sj)]
        N[f"GunN{nmi}On{nmj}Behind"] = len(cnt)
# who pays the gun-minute price: rate c^Q of the EV behind
gun["ptc_j"] = np.where(gun.pressure_j <= lo_t, "low", np.where(gun.pressure_j <= hi_t, "medium", "high"))
rows = []
for t in ("low", "medium", "high"):
    g = gun[gun.ptc_j == t]
    rows.append([t, len(g), pct(g.bind.mean()), F(g[g.bind].dual.abs().mean(), "{:.1f}"),
                 F(g.dual.abs().mean(), "{:.2f}")])
    N["GunBind" + t.capitalize()] = pct(g.bind.mean())
tab("gun_pressure", ["\\shortstack{demand pressure\\\\of the follower}", "gun links", "\\shortstack{binding\\\\(\\%)}",
                     "\\shortstack{mean dual if\\\\binding (Rs/min)}", "\\shortstack{mean dual\\\\all links (Rs/min)}"],
    rows)
# power
di["n"] = di.n.astype(int)
di["vPF"] = -di.dZ_dPF
di["vPM"] = -di.dZ_dPM
fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.5))
axs[0].boxplot([di[di.n == k].vPF for k in ns], tick_labels=ns, showfliers=False)
axs[0].set_title("one more kW of FCS gun power", fontsize=9)
axs[1].boxplot([di[di.n == k].vPM for k in ns], tick_labels=ns, showfliers=False)
axs[1].set_title("one more kW of van power", fontsize=9)
for a in axs:
    a.set_xlabel("$n$")
    a.set_ylabel("Rs saved per instance")
    a.tick_params(axis="x", labelsize=6)
save(fig, "power_value")
N["DPFmean"] = F(di.vPF.mean(), "{:.0f}")
N["DPFmedian"] = F(di.vPF.median(), "{:.0f}")
N["DPFmax"] = F(di.vPF.max(), "{:.0f}")
N["DPFperEV"] = F((di.vPF / di.n).mean(), "{:.1f}")
N["DPFcong"] = pct((-di.dZ_dPF_congestion).sum() / di.vPF.sum())
N["DPMmean"] = F(di.vPM.mean(), "{:.0f}")
N["DPMmax"] = F(di.vPM.max(), "{:.0f}")
N["DPFoverDPM"] = F(di.vPF.mean() / di.vPM.mean(), "{:.0f}")

# --- 4.3 threshold theory per segment (typical EV)
TY = {"E": 20.0, "d": 2.0, "qM": 10.0, "x": 4.0}
spd = 15.6
PF_, PM_, ETA = 22.0, 30.0, float(cfgcsv.van_efficiency)
pF, pM = P["tariff_Rs_per_kWh"]["p_F"], P["tariff_Rs_per_kWh"]["p_M"]


def lin(ends, th):
    return ends["theta0"] + (ends["theta1"] - ends["theta0"]) * th


def delta(seg, th, Pp=None, E=TY["E"], d=TY["d"], qF=0.0, qM=TY["qM"], x=TY["x"], dispatch=False, over=None):
    """Extra cost of the van over the station for one EV (formulation 4, Section 'How segment and income enter')."""
    Pp = Pp or P
    over = over or {}
    cfg_ = dict(cfgv)
    cfg_.update({k: v for k, v in over.items() if k in cfg_})
    pF_ = over.get("p_F", pF)
    pM_ = over.get("p_M", pM)
    t = 60 * d / spd
    tF, tM = 60 * E / PF_, 60 * E / PM_
    Pi = (pM_ - pF_) * E - cfg_["c_EV"] * d
    kap = (cfg_["pi_M"] / ETA - cfg_["pi_F"]) * E + cfg_["c_M"] * x + (cfg_["c_F"] if dispatch else 0.0)
    S = Pp[seg]
    vot = S["value_of_time_Rs_per_min"]
    mu = lin(S["money_sensitivity_mu"], th)
    cD, cQ, cC, cV = (lin(vot[a], th) for a in "DQCV")
    return mu * Pi + kap - (cD * t + cQ * qF + cC * tF - cV * (qM + tM))


def theta_star(seg, **kw):
    d0, d1 = delta(seg, 0.0, **kw), delta(seg, 1.0, **kw)
    if d0 == d1:
        return np.nan
    return d0 / (d0 - d1)


tsP = theta_star("personal")
tsC = theta_star("commercial")
N["TypE"] = F(TY["E"], "{:.0f}")
N["TypD"] = F(TY["d"], "{:.0f}")
N["TypT"] = F(60 * TY["d"] / spd, "{:.1f}")
N["TypQM"] = F(TY["qM"], "{:.0f}")
N["TypX"] = F(TY["x"], "{:.0f}")
N["TypTauF"] = F(60 * TY["E"] / PF_, "{:.1f}")
N["TypTauM"] = F(60 * TY["E"] / PM_, "{:.0f}")
N["TypPi"] = F((pM - pF) * TY["E"] - cfgv["c_EV"] * TY["d"], "{:.0f}")
N["TypKappa"] = F((cfgv["pi_M"] / ETA - cfgv["pi_F"]) * TY["E"] + cfgv["c_M"] * TY["x"], "{:.0f}")
N["TypD"] = F(TY["d"], "{:.0f}")
Dsaved = 60 * TY["d"] / spd + 60 * TY["E"] / PF_ - (TY["qM"] + 60 * TY["E"] / PM_)
N["TypDsaved"] = F(Dsaved, "{:.1f}")
N["ThetaStarP"] = F(tsP, "{:.2f}")
N["DeltaPZero"] = F(delta("personal", 0), "{:.0f}")
N["DeltaPOne"] = F(delta("personal", 1), "{:.0f}")
N["DeltaCZero"] = F(delta("commercial", 0), "{:.0f}")
N["DeltaCOne"] = F(delta("commercial", 1), "{:.0f}")
N["ThetaStarPDispatch"] = F(theta_star("personal", dispatch=True), "{:.2f}")
N["DeltaPOneDispatch"] = F(delta("personal", 1, dispatch=True), "{:.0f}")
N["ThetaStarPQten"] = F(theta_star("personal", qF=10.0), "{:.2f}")
N["ThetaStarPWaitForty"] = F(theta_star("personal", qM=40.0), "{:.2f}")
N["ThetaStarPFar"] = F(theta_star("personal", d=5.0), "{:.2f}")
N["ThetaStarPNear"] = F(theta_star("personal", d=0.5), "{:.2f}")
N["ThetaStarPEForty"] = F(theta_star("personal", E=40.0), "{:.2f}")
N["ThetaStarPETen"] = F(theta_star("personal", E=10.0), "{:.2f}")
# commercial: queue that makes the van the better choice, by income


def q_needed(th, **kw):
    """Smallest FCS queue q with Delta_C(theta, q) <= 0 (linear in q)."""
    a = delta("commercial", th, qF=0.0, **kw)
    b = delta("commercial", th, qF=1.0, **kw) - a
    return -a / b if b < 0 else np.inf


qrows = []
for th in (0.0, 0.25, 0.5, 0.75, 1.0):
    qn = q_needed(th)
    Wc = lin(P["commercial"]["queue_tolerance_W_min"], th)
    qrows.append([F(th, "{:.2f}"), F(delta("commercial", th), "{:.0f}"), F(qn, "{:.1f}"), F(Wc, "{:.1f}"),
                  F(q_needed(th, qM=0.0), "{:.1f}")])
tab("comm_queue", ["$\\theta$", "\\shortstack{$\\Delta^{\\mathrm{C}}$ at no queue\\\\(Rs)}",
                   "\\shortstack{queue that makes\\\\the van better (min)}", "\\shortstack{queue tolerance\\\\$W^{\\mathrm{C}}(\\theta)$ (min)}",
                   "\\shortstack{same, van\\\\without wait (min)}"], qrows)
N["QNeededZero"] = F(q_needed(0.0), "{:.0f}")
N["QNeededOne"] = F(q_needed(1.0), "{:.0f}")
N["QNeededHalf"] = F(q_needed(0.5), "{:.0f}")
N["WCommZero"] = F(lin(P["commercial"]["queue_tolerance_W_min"], 0), "{:.0f}")
N["WCommOne"] = F(lin(P["commercial"]["queue_tolerance_W_min"], 1), "{:.0f}")
N["QNeededOneNoWait"] = F(q_needed(1.0, qM=0.0), "{:.0f}")
# tornado (personal threshold)
tor = []
base_over = {}
for lab_, kind, key in [("$c^{\\mathrm{C},\\mathrm{P},1}$", "vot", ("personal", "C", "theta1")),
                        ("$\\pi^{\\mathrm{M}}$", "cfg", "pi_M"), ("van tariff $p^{\\mathrm{M}}$", "cfg", "p_M"),
                        ("$c^{\\mathrm{V},\\mathrm{P},1}$", "vot", ("personal", "V", "theta1")),
                        ("$\\mu^{\\mathrm{P},0}$", "mu", ("personal", "theta0")),
                        ("distance to FCS $d$", "ty", "d"), ("$c^{\\mathrm{D},\\mathrm{P},1}$", "vot", ("personal", "D", "theta1")),
                        ("$\\mu^{\\mathrm{P},1}$", "mu", ("personal", "theta1")), ("$c^{\\mathrm{M}}$", "cfg", "c_M"),
                        ("van detour $x$", "ty", "x"), ("van wait $q^{\\mathrm{M}}$", "ty", "qM"),
                        ("$c^{\\mathrm{C},\\mathrm{P},0}$", "vot", ("personal", "C", "theta0")), ("energy $E$", "ty", "E")]:
    vals = []
    for fct in (0.75, 1.25):
        Pp = json.loads(json.dumps(P))
        kw = {}
        if kind == "vot":
            s_, a_, e_ = key
            Pp[s_]["value_of_time_Rs_per_min"][a_][e_] *= fct
        elif kind == "mu":
            s_, e_ = key
            Pp[s_]["money_sensitivity_mu"][e_] *= fct
        elif kind == "cfg":
            base = cfgv.get(key, pF if key == "p_F" else pM)
            kw["over"] = {key: base * fct}
        elif kind == "ty":
            kw[key] = TY[key] * fct
        vals.append(theta_star("personal", Pp=Pp, **kw))
    tor.append((lab_, vals[0], vals[1], abs(vals[1] - vals[0])))
tor.sort(key=lambda r: -r[3])
tab("tornado", ["parameter ($\\pm25\\%$)", "$\\theta^{*}$ at $-25\\%$", "$\\theta^{*}$ base", "$\\theta^{*}$ at $+25\\%$",
                "swing"], [[r[0], F(r[1], "{:.2f}"), F(tsP, "{:.2f}"), F(r[2], "{:.2f}"), F(r[3], "{:.2f}")] for r in tor])
N["TorTopSym"] = tor[0][0]
N["TorTopSwing"] = F(tor[0][3], "{:.2f}")
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9))
th = np.linspace(0, 1, 101)
for E_, col in ((10, "#4C72B0"), (20, "#222222"), (40, "#C44E52")):
    axs[0].plot(th, delta("personal", th, E=E_), color=col, label=f"personal, $E$ = {E_} kWh")
for q_, ls in ((0, "--"), (20, ":")):
    axs[0].plot(th, delta("commercial", th, qF=q_), color="#DD8452", ls=ls,
                label=f"commercial, 20 kWh, queue {q_} min")
axs[0].axhline(0, color="grey", lw=.6)
axs[0].set_xlabel("income $\\theta$")
axs[0].set_ylabel("$\\Delta(\\theta)$: extra cost of a van (Rs)")
axs[0].legend(fontsize=6)
for k, r in enumerate(reversed(tor)):
    axs[1].plot([min(r[1], r[2]), max(r[1], r[2])], [k, k], color="#4C72B0", lw=4)
    axs[1].plot(r[1], k, "o", color="#C44E52", ms=3)
    axs[1].plot(r[2], k, "s", color="#C44E52", ms=3)
axs[1].axvline(tsP, color="grey", lw=.6)
axs[1].set_yticks(range(len(tor)))
axs[1].set_yticklabels([r[0] for r in reversed(tor)], fontsize=6)
axs[1].set_xlabel("personal $\\theta^{*}$ (circle: $-25\\%$, square: $+25\\%$)")
save(fig, "threshold")
# theory vs data
optf = free[free.optimal]
fig, ax = plt.subplots(figsize=(5.2, 2.9))
mid = np.arange(10) / 10 + .05
for s, mk in (("personal", "s"), ("commercial", "o")):
    sh = optf[optf.segment == s].groupby("decile").van.mean().reindex(range(10)) * 100
    ax.plot(mid, sh.values, marker=mk, color=rc.SEG_COL[s], label=f"{s}, optimal plans (data)")
ax.axvline(tsP, color="#C44E52", ls="--", lw=1, label=f"theory: personal $\\theta^*$ = {tsP:.2f} (typical EV)")
ax.set_xlabel("income $\\theta$")
ax.set_ylabel("free-choice EVs by van (%)")
ax.set_xlim(0, 1)
ax.legend(fontsize=7)
save(fig, "threshold_data")
pd_ = optf[optf.segment == "personal"].groupby("decile").van.mean() * 100
N["VanPersBelowFour"] = pct(optf[(optf.segment == "personal") & (optf.theta < 0.4)].van.mean())
N["VanPersSixSeven"] = F(pd_.get(6, np.nan), "{:.1f}")
N["VanPersEightNine"] = F(pd_.get(8, np.nan), "{:.1f}")
N["VanPersNineTen"] = F(pd_.get(9, np.nan), "{:.1f}")
N["VanCommOptMax"] = F((optf[optf.segment == "commercial"].groupby("decile").van.mean() * 100).max(), "{:.1f}")
N["VanCommOpt"] = pct(optf[optf.segment == "commercial"].van.mean())
N["NCommVanOpt"] = int(optf[optf.segment == "commercial"].van.sum())

# --- 4.4 choice by segment x decile, logistic regression
rows = []
for dcl in range(10):
    r_ = [f"{dcl / 10:.1f}--{(dcl + 1) / 10:.1f}"]
    for s in SEGS:
        g = free[(free.segment == s) & (free.decile == dcl)]
        k, n_ = int(g.van.sum()), len(g)
        lo, hi = rc.wilson(k, n_)
        go = g[g.optimal]
        r_ += [n_, pct(k / n_) if n_ else "--", f"[{pct(max(lo, 0.0))}, {pct(min(hi, 1.0))}]" if n_ else "--",
               pct(go.van.mean()) if len(go) else "--"]
    rows.append(r_)
with open(os.path.join(TAB, "deciles.tex"), "w", encoding="utf-8") as fh:
    fh.write("\\begin{tabular}{lrrrrrrrr}\\toprule\n & \\multicolumn{4}{c}{personal} & \\multicolumn{4}{c}{commercial} \\\\\n"
             "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9} income $\\theta$ & EVs & \\shortstack{by van\\\\(\\%)} & "
             "\\shortstack{95\\%\\\\interval} & \\shortstack{optimal\\\\plans (\\%)} & EVs & \\shortstack{by van\\\\(\\%)} & "
             "\\shortstack{95\\%\\\\interval} & \\shortstack{optimal\\\\plans (\\%)} \\\\ \\midrule\n")
    for r_ in rows:
        fh.write(" & ".join(str(x) for x in r_) + " \\\\\n")
    fh.write("\\bottomrule\\end{tabular}\n")
fig, ax = plt.subplots(figsize=(5.2, 2.9))
for s, mk in (("personal", "o"), ("commercial", "s")):
    g = free[free.segment == s]
    sh = g.groupby("decile").van.agg(["sum", "count"]).reindex(range(10))
    p_ = 100 * sh["sum"] / sh["count"]
    ci = np.array([rc.wilson(k, n_) if n_ == n_ and n_ > 0 else (np.nan, np.nan)
                   for k, n_ in zip(sh["sum"].fillna(0), sh["count"].fillna(0))]) * 100
    ax.plot(mid, p_, marker=mk, color=rc.SEG_COL[s], label=f"{s}, all valid plans")
    ax.fill_between(mid, ci[:, 0], ci[:, 1], color=rc.SEG_COL[s], alpha=.15)
    so = g[g.optimal].groupby("decile").van.mean().reindex(range(10)) * 100
    ax.plot(mid, so, ls="--", marker=mk, color=rc.SEG_COL[s], alpha=.6, ms=3, label=f"{s}, optimal plans only")
ax.set_xlabel("income $\\theta$ (decile midpoint)")
ax.set_ylabel("free-choice EVs served by a van (%)")
ax.legend(fontsize=7)
save(fig, "van_by_income")
pv = free[free.segment == "personal"]
cvv = free[free.segment == "commercial"]
N["MeanThetaVanPers"] = F(pv[pv.van].theta.mean(), "{:.2f}")
N["MeanThetaFCSPers"] = F(pv[~pv.van].theta.mean(), "{:.2f}")
low_van = free[(free.theta < 0.4) & free.van]
N["NLowVan"] = len(low_van)
N["NLowVanOpt"] = int(low_van.optimal.sum())
N["NLowVanComm"] = int((low_van.segment == "commercial").sum())
N["LowVanWait"] = F(low_van.van_wait.mean(), "{:.0f}")
N["LowVanPressure"] = F(low_van.pressure.mean(), "{:.2f}")
N["FreePressureMean"] = F(free.pressure.mean(), "{:.2f}")
N["RichPersVanWait"] = F(free[(free.theta >= 0.8) & free.van & (free.segment == "personal")].van_wait.mean(), "{:.0f}")
N["CommVanWait"] = F(cvv[cvv.van].van_wait.mean(), "{:.0f}")
N["CommVanPressure"] = F(cvv[cvv.van].pressure.mean(), "{:.2f}")
N["CommFCSPressure"] = F(cvv[~cvv.van].pressure.mean(), "{:.2f}")
N["CommVanHighPressPct"] = pct((cvv[cvv.van].ptc == "high").mean())
N["CommVanTL"] = pct((~cvv[cvv.van].optimal).mean())


def logit(X, y, iters=50):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p_ = 1 / (1 + np.exp(-X @ b))
        Wd = p_ * (1 - p_)
        H = X.T @ (X * Wd[:, None])
        g = X.T @ (y - p_)
        step = np.linalg.solve(H, g)
        b += step
        if np.max(np.abs(step)) < 1e-9:
            break
    p_ = 1 / (1 + np.exp(-X @ b))
    H = X.T @ (X * (p_ * (1 - p_))[:, None])
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return b, se, p_


Xd = pd.DataFrame({"intercept": 1.0, "theta": free.theta, "commercial": free.comm,
                   "theta_x_comm": free.theta * free.comm,
                   "medium": (free.ptc == "medium").astype(float), "high": (free.ptc == "high").astype(float),
                   "comm_x_high": free.comm * (free.ptc == "high"),
                   "long": (free.horizon == "long").astype(float), "E10": free.E / 10,
                   "near": free.nearest_fcs_km})
y = free.van.astype(float).values
b, se, phat = logit(Xd.values, y)
zv = b / se
pv_ = 2 * (1 - stats.norm.cdf(np.abs(zv)))
names = {"intercept": "intercept", "theta": "income $\\theta$", "commercial": "commercial",
         "theta_x_comm": "$\\theta\\times$ commercial", "medium": "medium pressure", "high": "high pressure",
         "comm_x_high": "commercial $\\times$ high pressure", "long": "long horizon (type 2)",
         "E10": "energy $E_i$ (per 10 kWh)", "near": "nearest reachable FCS (km)"}
rows = []
for k, c in enumerate(Xd.columns):
    rows.append([names[c], F(b[k], "{:.3f}"), F(se[k], "{:.3f}"), F(zv[k], "{:.1f}"),
                 "$<0.001$" if pv_[k] < 0.001 else F(pv_[k], "{:.3f}")])
tab("logit", ["term", "coefficient", "std. error", "$z$", "$p$-value"], rows)
X = Xd.values.copy()


def ame(col, delta_=0.1, mask=None):
    X1 = X.copy()
    X1[:, list(Xd.columns).index(col)] += delta_
    if col == "theta":
        X1[:, list(Xd.columns).index("theta_x_comm")] = X1[:, list(Xd.columns).index("theta")] * X[:, 2]
    p1 = 1 / (1 + np.exp(-X1 @ b))
    d = (p1 - phat) * 100
    return d[mask].mean() if mask is not None else d.mean()


N["AMEThetaPers"] = F(ame("theta", mask=(free.comm == 0).values), "{:.1f}")
N["AMEThetaComm"] = F(ame("theta", mask=(free.comm == 1).values), "{:.2f}")
N["LogitAcc"] = pct(((phat > 0.5) == (y > 0.5)).mean())
N["LogitBaseAcc"] = pct(1 - y.mean())
N["LogitN"] = f"{len(y):,}".replace(",", "{,}")
N["LogitPseudoR"] = F(1 - (np.sum(y * np.log(phat) + (1 - y) * np.log(1 - phat)))
                      / (np.sum(y * np.log(y.mean()) + (1 - y) * np.log(1 - y.mean()))), "{:.2f}")

# --- 4.5 queues by segment x band
fu = free[free.resource_type == "FCS"]
qtot = fu.queue.sum()
rows = []
for s in SEGS:
    for k in range(5):
        g = fu[(fu.segment == s) & (fu.band == k)]
        if len(g) == 0:
            continue
        rows.append([s if k == 0 else "", rc.BLAB[k], len(g), pct(len(g) / len(fu)), pct(g.queue.sum() / qtot),
                     pct((g.queue > 1e-6).mean()), F(g.queue.mean(), "{:.2f}"), F(g.queue.max(), "{:.1f}"),
                     F(g.W.max(), "{:.1f}"), F(g.drive_km.mean(), "{:.2f}")])
tab("queues", ["segment", "income $\\theta$", "\\shortstack{FCS\\\\users}", "\\shortstack{share of\\\\users (\\%)}",
               "\\shortstack{share of queue\\\\minutes (\\%)}", "\\shortstack{queueing\\\\(\\%)}",
               "\\shortstack{mean queue\\\\(min)}", "\\shortstack{longest\\\\queue (min)}",
               "\\shortstack{largest\\\\$W_i$ (min)}", "\\shortstack{drive to\\\\FCS (km)}"], rows)
N["QShareBottom"] = pct(fu[fu.theta < 0.4].queue.sum() / qtot)
N["UserShareBottom"] = pct((fu.theta < 0.4).mean())
N["QShareTop"] = pct(fu[fu.theta >= 0.8].queue.sum() / qtot)
N["UserShareTop"] = pct((fu.theta >= 0.8).mean())
N["QShareComm"] = pct(fu[fu.segment == "commercial"].queue.sum() / qtot)
N["UserShareComm"] = pct((fu.segment == "commercial").mean())
N["DriveBottom"] = F(fu[fu.theta < 0.4].drive_km.mean(), "{:.2f}")
N["DriveTop"] = F(fu[fu.theta >= 0.8].drive_km.mean(), "{:.2f}")
N["DriveComm"] = F(fu[fu.segment == "commercial"].drive_km.mean(), "{:.2f}")
N["DrivePers"] = F(fu[fu.segment == "personal"].drive_km.mean(), "{:.2f}")
N["QueueAtLimit"] = int((fu.queue >= fu.W - 1e-3).sum())
N["QueueMaxOverW"] = F(100 * (fu.queue / fu.W).max(), "{:.2f}")
N["NearLimit"] = int((fu.queue >= 0.9 * fu.W).sum())
# who queues at equal income: compare segments in the band 0.2-0.6
mb = fu[(fu.theta >= 0.2) & (fu.theta < 0.6)]
N["QueueMidComm"] = F(mb[mb.segment == "commercial"].queue.mean(), "{:.2f}")
N["QueueMidPers"] = F(mb[mb.segment == "personal"].queue.mean(), "{:.2f}")
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
x = np.arange(5)
for k, s in enumerate(SEGS):
    g = fu[fu.segment == s]
    us = [100 * ((g.band == i).sum()) / len(fu) for i in range(5)]
    qs = [100 * g[g.band == i].queue.sum() / qtot for i in range(5)]
    off = -0.3 + 0.3 * k
    axs[0].bar(x + off - 0.075, us, 0.15, color=rc.SEG_COL[s], alpha=.45, label=f"{s}: share of FCS users")
    axs[0].bar(x + off + 0.075, qs, 0.15, color=rc.SEG_COL[s], label=f"{s}: share of queue minutes")
    axs[1].plot(x, [g[g.band == i].drive_km.mean() for i in range(5)], marker="o", color=rc.SEG_COL[s], label=s)
axs[0].set_xticks(x)
axs[0].set_xticklabels(rc.PLAB, fontsize=7)
axs[0].set_xlabel("income band $\\theta$")
axs[0].set_ylabel("%")
axs[0].legend(fontsize=6)
axs[1].set_xticks(x)
axs[1].set_xticklabels(rc.PLAB, fontsize=7)
axs[1].set_xlabel("income band $\\theta$")
axs[1].set_ylabel("mean drive to the FCS (km)")
axs[1].legend(fontsize=7)
save(fig, "queues_drive")

# --- 4.6 peak / off-peak
rows = []
for t in ("low", "medium", "high"):
    g = free[free.ptc == t]
    gf = g[g.resource_type == "FCS"]
    r_ = [t, len(g), pct((gf.queue > 1e-6).mean()), F(gf.queue.mean(), "{:.2f}"),
          pct(g[g.segment == "personal"].van.mean()), pct(g[g.segment == "commercial"].van.mean())]
    for k in (0, 2, 4):
        gg = g[(g.segment == "personal") & (g.band == k)]
        r_.append(pct(gg.van.mean()) if len(gg) else "--")
    rows.append(r_)
    T_ = t.capitalize()
    N["Queue" + T_] = F(gf.queue.mean(), "{:.2f}")
    N["Queueing" + T_] = pct((gf.queue > 1e-6).mean())
    N["VanPers" + T_] = pct(g[g.segment == "personal"].van.mean())
    N["VanComm" + T_] = pct(g[g.segment == "commercial"].van.mean())
    N["VanRichPers" + T_] = pct(g[(g.segment == "personal") & (g.band == 4)].van.mean())
    N["QueuePoorComm" + T_] = F(gf[(gf.segment == "commercial") & (gf.theta < 0.4)].queue.mean(), "{:.2f}")
    N["QueueRichPers" + T_] = F(gf[(gf.segment == "personal") & (gf.band == 4)].queue.mean(), "{:.2f}")
tab("pressure", ["\\shortstack{demand\\\\pressure}", "EVs", "\\shortstack{queueing\\\\(\\%)}",
                 "\\shortstack{mean queue\\\\(min)}", "\\shortstack{by van (\\%)\\\\personal}",
                 "\\shortstack{by van (\\%)\\\\commercial}", "\\shortstack{personal by van\\\\$\\theta$ 0.0--0.2}",
                 "\\shortstack{personal by van\\\\$\\theta$ 0.4--0.6}", "\\shortstack{personal by van\\\\$\\theta$ 0.8--1.0}"],
    rows)
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.5))
pc = {"low": "#4C72B0", "medium": "#DD8452", "high": "#C44E52"}
for t in ("low", "medium", "high"):
    g = free[free.ptc == t]
    for k_, s in enumerate(SEGS):
        sh = [100 * g[(g.segment == s) & (g.band == i)].van.mean() for i in range(5)]
        axs[k_].plot(x, sh, marker="o", color=pc[t], label=f"{t} pressure")
    gf = g[g.resource_type == "FCS"]
    axs[2].plot(x, [gf[gf.band == i].queue.mean() for i in range(5)], marker="o", color=pc[t], label=f"{t}")
for k_, s in enumerate(SEGS):
    axs[k_].set_title(f"{s}: served by a van (%)", fontsize=8)
axs[2].set_title("mean FCS queue (min), both segments", fontsize=8)
for a in axs:
    a.set_xticks(x)
    a.set_xticklabels(rc.PLAB, rotation=40, fontsize=6)
    a.set_xlabel("income band $\\theta$", fontsize=7)
axs[0].legend(fontsize=6)
save(fig, "pressure")

# --- 4.7 horizon
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
for k_, s in enumerate(SEGS):
    for h, col, mk in (("short", "#C44E52", "o"), ("long", "#4C72B0", "s")):
        g = free[(free.segment == s) & (free.horizon == h)]
        axs[k_].plot(x, [100 * g[g.band == i].van.mean() for i in range(5)], marker=mk, color=col,
                     label=f"{h} horizon ({'C1, R1, RC1' if h == 'short' else 'C2, R2, RC2'})")
    axs[k_].set_title(s, fontsize=9)
    axs[k_].set_xticks(x)
    axs[k_].set_xticklabels(rc.PLAB, fontsize=7)
    axs[k_].set_xlabel("income band $\\theta$")
    axs[k_].set_ylabel("served by a van (%)")
axs[0].legend(fontsize=7)
save(fig, "horizon")
rows = []
for fam in rc.FAMS:
    g = free[free.family == fam]
    gp_ = g[g.segment == "personal"]
    r_ = [fam, "long" if fam.endswith("2") else "short", len(g), F(g.pressure.mean(), "{:.2f}"),
          pct(gp_.van.mean())]
    r_ += [pct(gp_[gp_.band == i].van.mean()) if (gp_.band == i).any() else "--" for i in range(5)]
    r_ += [pct(g[g.segment == "commercial"].van.mean())]
    rows.append(r_)
tab("family_band", ["family", "horizon", "EVs", "\\shortstack{mean\\\\pressure}", "\\shortstack{personal\\\\by van (\\%)}"]
    + [f"\\shortstack{{personal\\\\$\\theta$ {b_}}}" for b_ in rc.BLAB] + ["\\shortstack{commercial\\\\by van (\\%)}"], rows)
for s in SEGS:
    for h in ("short", "long"):
        g = free[(free.segment == s) & (free.horizon == h)]
        N[f"VanTop{s.capitalize()}{h.capitalize()}"] = pct(g[g.band == 4].van.mean()) if (g.band == 4).any() else "--"
        N[f"Press{h.capitalize()}"] = F(free[free.horizon == h].pressure.mean(), "{:.2f}")

# --- 4.8 infrastructure utilisation (shared definitions: ../common/infra.py)
RESD = F4J("baseline_MILP", "results")
plan_v = rc.load_plan(RESD, "milp_300s").merge(valid[["instance", "n"]], on=["instance", "n"])
U = infra.utilisation(plan_v)
ug, us, uv, ui, upr = U["guns"], U["stations"], U["vans"], U["instances"], U["profile"]
ui["horizon"] = np.where(ui.family.str.endswith("2"), "long", "short")
# data-only active demand window per instance: [min r_i, max_i (r_i + tauF_i)]
aw = ev.assign(end=ev.r + ev.tauF).groupby(["instance", "n"]).agg(w0=("r", "min"), w1=("end", "max"))
ui = ui.merge(aw.reset_index(), on=["instance", "n"], how="left")
ui["active_len"] = ui.w1 - ui.w0
busy_tot = ug.groupby(["instance", "n"]).busy_min.sum().rename("busy_tot").reset_index()
ui = ui.merge(busy_tot, on=["instance", "n"])
ui["gun_util_active"] = ui.busy_tot / (ui.G * ui.active_len)
ui["active_share"] = ui.active_len / ui["T"]
# peak concurrency (exact sweep) and share of guns busy at each request
fcs_iv = plan_v[(plan_v.type == "ev") & (plan_v.resource_type == "FCS")]
van_iv = plan_v[(plan_v.type == "ev") & (plan_v.resource_type == "MCS")]
peak_g, peak_v = {}, {}
for key, g in fcs_iv.groupby(["instance", "n"]):
    evts = sorted([(s, 1) for s in g.start_min] + [(e, -1) for e in g.end_min], key=lambda x: (x[0], x[1]))
    c = m_ = 0
    for _, d in evts:
        c += d
        m_ = max(m_, c)
    peak_g[key] = m_
for key, g in van_iv.groupby(["instance", "n"]):
    evts = sorted([(s, 1) for s in g.start_min] + [(e, -1) for e in g.end_min], key=lambda x: (x[0], x[1]))
    c = m_ = 0
    for _, d in evts:
        c += d
        m_ = max(m_, c)
    peak_v[key] = m_
ui["peak_guns"] = [peak_g.get((a, b), 0) for a, b in zip(ui.instance, ui.n)]
ui["peak_vans"] = [peak_v.get((a, b), 0) for a, b in zip(ui.instance, ui.n)]
busy_at = []
fiv = {k: (g.start_min.to_numpy(), g.end_min.to_numpy()) for k, g in fcs_iv.groupby(["instance", "n"])}
for (inst_, n_), g in free.groupby(["instance", "n"]):
    s_, e_ = fiv.get((inst_, n_), (np.array([]), np.array([])))
    r_ = g.r.to_numpy()
    busy_at.append(pd.Series(((s_[None, :] <= r_[:, None]) & (e_[None, :] > r_[:, None])).sum(1) / 24.0,
                             index=g.index))
free["guns_busy_at_r"] = pd.concat(busy_at)

N["UGunMean"] = pct(ui.gun_util.mean())
N["UGunMedian"] = pct(ui.gun_util.median())
N["UGunActiveMean"] = pct(ui.gun_util_active.mean())
N["UGunActiveMedian"] = pct(ui.gun_util_active.median())
N["UActiveShare"] = pct(ui.active_share.mean(), 0)
N["UPerGunMean"] = pct(ug.util.mean())
N["UPerGunNinety"] = pct(ug.util.quantile(.9))
N["UPerGunMax"] = pct(ug.util.max())
N["UGunIdleShare"] = pct((ug.evs == 0).mean())
N["UStationMax"] = pct(us.util.max())
N["UStationNinety"] = pct(us.util.quantile(.9))
N["UStationIdleShare"] = pct((us.evs == 0).mean())
N["UIdleGunsMean"] = F(ui.idle_guns.mean(), "{:.1f}")
N["UIdleGunsHundred"] = F(ui[ui.n == 100].idle_guns.mean(), "{:.1f}")
N["UIdleGunsTen"] = F(ui[ui.n == 10].idle_guns.mean(), "{:.1f}")
N["UPeakGunsMean"] = F(ui.peak_guns.mean(), "{:.1f}")
N["UPeakGunsMax"] = int(ui.peak_guns.max())
N["UPeakGunsFull"] = int((ui.peak_guns >= 24).sum())
N["UPeakGunsHundred"] = F(ui[ui.n == 100].peak_guns.mean(), "{:.1f}")
N["UPeakVansMax"] = int(ui.peak_vans.max())
for h in ("short", "long"):
    g = ui[ui.horizon == h]
    N["UGun" + h.capitalize()] = pct(g.gun_util.mean())
    N["UGunActive" + h.capitalize()] = pct(g.gun_util_active.mean())
    N["UActiveShare" + h.capitalize()] = pct(g.active_share.mean(), 0)
# system
E_all = ui.kWh_total.sum()
N["UEnergyTotal"] = f"{E_all / 1000:,.0f}".replace(",", "{,}")
N["UEnergyPerInst"] = F(ui.kWh_total.mean(), "{:.0f}")
N["UEnergyFCSShare"] = pct(ui.kWh_fcs.sum() / E_all)
N["UEnergyVanShare"] = pct(ui.kWh_van.sum() / E_all)
N["USystemMean"] = pct(ui.system_util.mean())
N["USystemMax"] = pct(ui.system_util.max())
N["UInstalledKW"] = int(24 * 22 + 5 * 30)
N["UGunKWShare"] = pct(24 * 22 / (24 * 22 + 5 * 30))
N["USystemShort"] = pct(ui[ui.horizon == "short"].system_util.mean())
N["USystemLong"] = pct(ui[ui.horizon == "long"].system_util.mean())
# by family / by n
rows = []
for fam in rc.FAMS:
    g = ui[ui.family == fam]
    gg = ug[ug.family == fam]
    rows.append([fam, int(g["T"].iloc[0]), len(g), pct(g.gun_util.mean()), pct(g.active_share.mean(), 0),
                 pct(g.gun_util_active.mean()), pct(gg.util.quantile(.9)), F(g.idle_guns.mean(), "{:.1f}"),
                 F(g.peak_guns.mean(), "{:.1f}"), pct(g.system_util.mean()), pct(g.kWh_van.sum() / g.kWh_total.sum())])
tab("util_family", ["family", "$T$", "plans", "\\shortstack{gun use\\\\over $[0,T]$ (\\%)}",
                    "\\shortstack{active window\\\\(\\% of $T$)}", "\\shortstack{gun use over\\\\active window (\\%)}",
                    "\\shortstack{90th pct\\\\one gun (\\%)}", "\\shortstack{idle guns\\\\(of 24)}",
                    "\\shortstack{peak guns\\\\at once}", "\\shortstack{system\\\\use (\\%)}",
                    "\\shortstack{energy by\\\\van (\\%)}"], rows)
rows = []
for n_ in ns:
    g = ui[ui.n == n_]
    rows.append([n_, len(g), F(g.kWh_total.mean(), "{:.0f}"), pct(g.gun_util.mean()), pct(g.gun_util_active.mean()),
                 F(g.idle_guns.mean(), "{:.1f}"), F(g.peak_guns.mean(), "{:.1f}"), pct(g.fleet_use.mean()),
                 pct(g.system_util.mean())])
tab("util_n", ["$n$", "plans", "\\shortstack{energy\\\\asked (kWh)}", "\\shortstack{gun use\\\\over $[0,T]$ (\\%)}",
               "\\shortstack{gun use over\\\\active window (\\%)}", "\\shortstack{idle guns\\\\(of 24)}",
               "\\shortstack{peak guns\\\\at once}", "\\shortstack{fleet use\\\\(\\% of 5 vans)}",
               "\\shortstack{system\\\\use (\\%)}"], rows)
g100 = ui[ui.n == 100]
N["UGunHundred"] = pct(g100.gun_util.mean())
N["UGunActiveHundred"] = pct(g100.gun_util_active.mean())
N["UGunActiveTen"] = pct(ui[ui.n == 10].gun_util_active.mean())
# pressure: share of guns busy at the request of free-choice EVs
rows = []
for t_ in ("low", "medium", "high"):
    g = free[free.ptc == t_]
    rows.append([t_, len(g), pct(g.guns_busy_at_r.mean()), pct(g.guns_busy_at_r.quantile(.9)),
                 pct((g.guns_busy_at_r >= 0.5).mean())])
    N["UBusyAtR" + t_.capitalize()] = pct(g.guns_busy_at_r.mean())
    N["UBusyHalf" + t_.capitalize()] = pct((g.guns_busy_at_r >= 0.5).mean())
tab("util_pressure", ["\\shortstack{demand pressure\\\\of the EV}", "free-choice EVs",
                      "\\shortstack{guns busy at its\\\\request, mean (\\%)}", "\\shortstack{90th\\\\pct (\\%)}",
                      "\\shortstack{half or more\\\\of guns busy (\\%)}"], rows)
# figures: busy profile per family, distributions
fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.2), sharex=True, sharey=True)
for a, fam in zip(axs.flat, rc.FAMS):
    p_ = upr[upr.family == fam].groupby("t_rel")[["guns_busy", "vans_busy"]].mean()
    a.plot(p_.index, 100 * p_.guns_busy, color="#4C72B0", label="guns charging (% of 24)")
    a.plot(p_.index, 100 * p_.vans_busy, color="#C44E52", label="vans charging (% of 5)")
    a.set_title(f"{fam} ($T$ = {int(ui[ui.family == fam]['T'].iloc[0])} min)", fontsize=8)
    a.grid(alpha=.3)
for a in axs[-1]:
    a.set_xlabel("$t/T$")
for a in axs[:, 0]:
    a.set_ylabel("busy (%)")
axs[0, 0].legend(fontsize=6)
save(fig, "util_profile")
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.4))
axs[0].hist(100 * ug.util, bins=40, color="#4C72B0")
axs[0].set_xlabel("one gun: busy over $[0,T]$ (%)")
axs[0].set_ylabel("guns")
axs[1].hist(100 * ui.gun_util_active, bins=30, color="#55A868")
axs[1].set_xlabel("all guns: busy over active window (%)")
axs[1].set_ylabel("plans")
axs[2].hist(ui.idle_guns, bins=np.arange(-.5, 25.5), color="#DD8452")
axs[2].set_xlabel("idle guns per plan (of 24)")
axs[2].set_ylabel("plans")
save(fig, "util_guns")
N["UProfilePeakBinShort"] = F(upr[upr.family.isin(["R1", "RC1"])].groupby("t_rel").guns_busy.mean().idxmax(), "{:.2f}")
N["UProfilePeakBinLong"] = F(upr[upr.family.isin(["C2", "R2", "RC2"])].groupby("t_rel").guns_busy.mean().idxmax(), "{:.2f}")
N["UProfileMaxShort"] = pct(upr[upr.family.isin(["R1", "RC1"])].groupby("t_rel").guns_busy.mean().max())
N["UProfileMaxLong"] = pct(upr[upr.family.isin(["C2", "R2", "RC2"])].groupby("t_rel").guns_busy.mean().max())
N["UProfileLateLong"] = pct(upr[upr.family.isin(["C2", "R2", "RC2"]) & (upr.t_rel > 0.8)].guns_busy.mean())

# --- 4.9 vans: utilisation and what limits them
vans["stranded_share"] = vans.stranded_stops / vans.stops
eta_ = float(cfgcsv.van_efficiency)
beta_ = float(cfgcsv.van_use_kWh_per_km)
uv["loss"] = uv.kWh_delivered / eta_ - uv.kWh_delivered
uv["drive_kWh"] = beta_ * uv.km
uv["idle_share"] = uv.idle_min / uv.on_road_min
N["UVanEnergyCheck"] = F((uv.kWh_delivered + uv.loss + uv.drive_kWh - uv.kWh_drawn).abs().max(), "{:.2f}")
drawn = uv.kWh_drawn.sum()
N["UVanDelivShare"] = pct(uv.kWh_delivered.sum() / drawn)
N["UVanLossShare"] = pct(uv.loss.sum() / drawn)
N["UVanDriveShare"] = pct(uv.drive_kWh.sum() / drawn)
rows = [["vans dispatched / 5 (fleet use)", pct(ui.fleet_use.mean()), pct(ui.fleet_use.median()),
         f"{pct((ui.fleet_use == 1).mean())}\\% of plans use all 5"],
        ["seats: stops / 6", pct(uv.seat_util.mean()), pct(uv.seat_util.median()),
         f"{pct((uv.stops == int(cfgcsv.van_max_evs)).mean())}\\% of vans full"],
        ["battery: kWh drawn / 150", pct(uv.battery_util.mean()), pct(uv.battery_util.median()),
         f"{pct((uv.battery_util > .9).mean())}\\% above 90\\%"],
        ["on the road / $T$", pct(uv.on_road_util.mean()), pct(uv.on_road_util.median()),
         f"{pct((uv.on_road_util > .9).mean())}\\% above 90\\%"],
        ["charging / $T$ (= power use)", pct(uv.charge_util.mean()), pct(uv.charge_util.median()), ""],
        ["idle share of on-road time", pct(uv.idle_share.mean()), pct(uv.idle_share.median()), ""],
        ["van km", F(uv.km.mean(), "{:.1f}"), F(uv.km.median(), "{:.1f}"), ""]]
tab("vans", ["measure (per dispatched van unless stated)", "mean", "median", "at the limit"], rows, align="lrrl")
rows = []
for fam in rc.FAMS:
    g = uv[uv.family == fam]
    gi = ui[ui.family == fam]
    rows.append([fam, len(g), pct(gi.fleet_use.mean()), pct(g.seat_util.mean()), pct(g.battery_util.mean()),
                 pct(g.on_road_util.mean()), pct(g.charge_util.mean()), pct(g.idle_share.mean())])
tab("van_util_family", ["family", "vans", "\\shortstack{fleet\\\\use (\\%)}", "\\shortstack{seats\\\\(\\%)}",
                        "\\shortstack{battery\\\\(\\%)}", "\\shortstack{on road\\\\(\\% of $T$)}",
                        "\\shortstack{charging\\\\(\\% of $T$)}", "\\shortstack{idle share of\\\\on-road time (\\%)}"],
    rows)
rows = []
for n_ in ns:
    g = uv[uv.n == n_]
    gi = ui[ui.n == n_]
    rows.append([n_, len(g), pct(gi.fleet_use.mean()), pct(g.seat_util.mean()), pct(g.battery_util.mean()),
                 pct(g.on_road_util.mean()), pct(g.charge_util.mean())])
tab("van_util_n", ["$n$", "vans", "\\shortstack{fleet\\\\use (\\%)}", "\\shortstack{seats\\\\(\\%)}",
                   "\\shortstack{battery\\\\(\\%)}", "\\shortstack{on road\\\\(\\% of $T$)}",
                   "\\shortstack{charging\\\\(\\% of $T$)}"], rows)
N["UFleetMean"] = pct(ui.fleet_use.mean())
N["UFleetAll"] = pct((ui.fleet_use == 1).mean())
N["UFleetHundred"] = pct(ui[ui.n == 100].fleet_use.mean())
N["USeatMean"] = pct(uv.seat_util.mean())
N["UBattMean"] = pct(uv.battery_util.mean())
N["UOnRoadMean"] = pct(uv.on_road_util.mean())
N["UChargeMean"] = pct(uv.charge_util.mean())
N["UIdleShareMean"] = pct(uv.idle_share.mean())
for h, fams in (("Short", ["R1", "RC1"]), ("Long", ["C2", "R2", "RC2"])):
    g = uv[uv.family.isin(fams)]
    N["UOnRoad" + h] = pct(g.on_road_util.mean())
    N["UCharge" + h] = pct(g.charge_util.mean())
    N["UIdleShare" + h] = pct(g.idle_share.mean())
    N["USeat" + h] = pct(g.seat_util.mean())
tc = vans.tightest.value_counts()
rows = [[f"seats ($Q_m={int(cfgcsv.van_max_evs)}$)", int(tc.get("seats", 0)), pct(tc.get("seats", 0) / len(vans))],
        [f"battery ($B_m={int(cfgcsv.van_battery_kWh)}$ kWh)", int(tc.get("battery", 0)), pct(tc.get("battery", 0) / len(vans))],
        ["time ($T$)", int(tc.get("time", 0)), pct(tc.get("time", 0) / len(vans))]]
tab("vans_tight", ["tightest van resource", "vans", "share (\\%)"], rows)
rows = []
for fam in rc.FAMS:
    g = vans[vans.family == fam]
    rows.append([fam, len(g), F(g.stops.mean(), "{:.2f}"), F(100 * g.time_use.mean(), "{:.1f}"),
                 pct((g.tightest == "seats").mean()), pct((g.tightest == "time").mean())])
    N["VanStops" + fam.replace("1", "One").replace("2", "Two")] = F(g.stops.mean(), "{:.1f}")
tab("vans_family", ["family", "vans", "\\shortstack{stops\\\\per van}", "\\shortstack{back at depot\\\\(\\% of $T$)}",
                    "\\shortstack{tightest:\\\\seats (\\%)}", "\\shortstack{tightest:\\\\time (\\%)}"], rows)
N["NVans"] = len(vans)
N["VansFull"] = pct((vans.stops == int(cfgcsv.van_max_evs)).mean())
N["TightSeats"] = pct(tc.get("seats", 0) / len(vans))
N["TightTime"] = pct(tc.get("time", 0) / len(vans))
N["TightBatt"] = pct(tc.get("battery", 0) / len(vans))
seats = ev[ev.van]
N["SeatsStranded"] = pct(seats.stranded.mean())
N["SeatsPersFree"] = pct(((~seats.stranded) & (seats.segment == "personal")).mean())
N["SeatsCommFree"] = pct(((~seats.stranded) & (seats.segment == "commercial")).mean())
short_ = vans[vans.family.isin(["R1", "RC1"])]
N["ShortVanStopsMin"] = F(short_.groupby("family").stops.mean().min(), "{:.1f}")
N["ShortVanStopsMax"] = F(short_.groupby("family").stops.mean().max(), "{:.1f}")
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3))
axs[0].hist(vans.stops, bins=np.arange(.5, int(cfgcsv.van_max_evs) + 1.5), color="#4C72B0", rwidth=.85)
axs[0].set_xlabel("EVs served by one van")
axs[0].set_ylabel("vans")
axs[1].hist(vans.kWh_drawn, bins=30, range=(0, float(cfgcsv.van_battery_kWh)), color="#55A868")
axs[1].set_xlabel(f"battery drawn (kWh, of {int(cfgcsv.van_battery_kWh)})")
axs[2].hist(100 * vans.time_use, bins=30, range=(0, 100), color="#DD8452")
axs[2].set_xlabel("back at depot (% of $T$)")
save(fig, "vans")

# --- 4.9 equity per segment
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
for k_, s in enumerate(SEGS):
    g = ev[ev.segment == s]
    pm = [g[g.band == i].perc_money.mean() for i in range(5)]
    pt = [g[g.band == i].perc_time.mean() for i in range(5)]
    paid = [g[g.band == i].pay.mean() for i in range(5)]
    axs[k_].bar(x, pm, .6, color="#4C72B0", label="perceived money ($\\mu_i\\times$ paid)")
    axs[k_].bar(x, pt, .6, bottom=pm, color="#DD8452", label="perceived time")
    axs[k_].hlines(paid, x - .3, x + .3, color="black", lw=2, label="money actually paid")
    axs[k_].set_xticks(x)
    axs[k_].set_xticklabels(rc.PLAB, fontsize=7)
    axs[k_].set_xlabel("income band $\\theta$")
    axs[k_].set_title(s, fontsize=9)
axs[0].set_ylabel("Rs per EV")
axs[0].legend(fontsize=6)
save(fig, "equity")
rows = []
tot_pay, tot_perc = ev.pay.sum(), ev.perc_total.sum()
for s in SEGS:
    for k in range(5):
        g = ev[(ev.segment == s) & (ev.band == k)]
        if len(g) == 0:
            continue
        rows.append([s if k == 0 else "", rc.BLAB[k], pct(len(g) / len(ev)), pct(g.pay.sum() / tot_pay),
                     pct(g.perc_total.sum() / tot_perc), F(g.perc_total.sum() / g.pay.sum(), "{:.2f}"),
                     pct(g.perc_time.sum() / g.perc_total.sum()), F(g.pay.sum() / g.E.sum(), "{:.1f}"),
                     F(g.downtime.mean(), "{:.1f}")])
tab("equity", ["segment", "income $\\theta$", "\\shortstack{share of\\\\EVs (\\%)}", "\\shortstack{share of\\\\money paid (\\%)}",
               "\\shortstack{share of\\\\perceived cost (\\%)}", "\\shortstack{perceived per\\\\Rs paid}",
               "\\shortstack{time share of\\\\perceived (\\%)}", "\\shortstack{paid per\\\\kWh (Rs)}",
               "\\shortstack{request to\\\\charged (min)}"], rows)
for s in SEGS:
    g = ev[ev.segment == s]
    cap = s.capitalize()
    N[f"PercPerRs{cap}"] = F(g.perc_total.sum() / g.pay.sum(), "{:.2f}")
    N[f"TimeShare{cap}"] = pct(g.perc_time.sum() / g.perc_total.sum())
    N[f"PaidPerKWh{cap}"] = F(g.pay.sum() / g.E.sum(), "{:.1f}")
    N[f"PayShare{cap}"] = pct(g.pay.sum() / tot_pay)
    N[f"PercShare{cap}"] = pct(g.perc_total.sum() / tot_perc)
gp_ = ev[ev.segment == "personal"]
N["PaidPerKWhPersTop"] = F(gp_[gp_.band == 4].pay.sum() / gp_[gp_.band == 4].E.sum(), "{:.1f}")
N["PaidPerKWhPersMid"] = F(gp_[gp_.band == 2].pay.sum() / gp_[gp_.band == 2].E.sum(), "{:.1f}")
gc_ = ev[ev.segment == "commercial"]
N["DowntimeCommPoor"] = F(gc_[gc_.theta < 0.2].downtime.mean(), "{:.1f}")
N["DowntimeCommMid"] = F(gc_[(gc_.theta >= 0.4) & (gc_.theta < 0.6)].downtime.mean(), "{:.1f}")
N["TimeSharePersTop"] = pct(gp_[gp_.band == 4].perc_time.sum() / gp_[gp_.band == 4].perc_total.sum())
N["TimeShareCommPoor"] = pct(gc_[gc_.band == 0].perc_time.sum() / gc_[gc_.band == 0].perc_total.sum())

# --- 4.10 size and computation
rows = []
g_gain = valid[valid.greedy_Z.notna()]
for n_ in ns:
    g = inst[inst.n == n_]
    gg = g_gain[g_gain.n == n_]
    rows.append([n_, f"{int(g.binaries.median()):,}", f"{int(g.continuous.median()):,}",
                 f"{int(g.constraints.median()):,}", F(g.build_s.median(), "{:.1f}"),
                 f"{int(g.nodes.median()):,}" if g.nodes.notna().any() else "--",
                 F(100 * ((gg.greedy_Z - gg.Z) / gg.greedy_Z).median(), "{:.1f}"),
                 pct((g.greedy_status == "plan").mean())])
tab("model", ["$n$", "\\shortstack{binary\\\\variables}", "\\shortstack{continuous\\\\variables}", "constraints",
              "\\shortstack{build\\\\time (s)}", "\\shortstack{median\\\\B\\&B nodes}",
              "\\shortstack{median gain over\\\\greedy (\\%)}", "\\shortstack{greedy plan\\\\found (\\%)}"], rows)
big = inst[inst.n == 100]
N["BinHundred"] = f"{int(big.binaries.median()):,}".replace(",", "{,}")
N["ConsHundred"] = f"{int(big.constraints.median()):,}".replace(",", "{,}")
N["GreedyFoundSmall"] = pct((inst[inst.n == 10].greedy_status == "plan").mean())
N["GreedyFoundBig"] = pct((big.greedy_status == "plan").mean())
N["GreedyGainMed"] = F(100 * ((g_gain.greedy_Z - g_gain.Z) / g_gain.greedy_Z).median(), "{:.1f}")
N["GreedyGainBigMed"] = F(100 * ((g_gain[g_gain.n >= 50].greedy_Z - g_gain[g_gain.n >= 50].Z)
                                 / g_gain[g_gain.n >= 50].greedy_Z).median(), "{:.1f}")
ws = inst.warm_start.value_counts()
N["WarmCost"] = int(ws.get("greedy (cost rule)", 0))
N["WarmFinish"] = int(ws.get("greedy (finish rule)", 0))
N["WarmNone"] = int(ws.get("none", 0))
N["NGreedyValid"] = int((inst.greedy_status == "plan").sum())
N["MILPExtraValid"] = len(valid) - int((inst.greedy_status == "plan").sum())
N["ValidNoWarm"] = int((valid.warm_start == "none").sum())

# --- 4.11 infeasible / unsolved
rows = []
for fam, g in inf.groupby("family"):
    rows.append([fam, len(g), int(g.n.min()), int(g.n.max()), int(g.data_infeasible.sum())])
tab("infeasible", ["family", "infeasible instances", "smallest $n$", "largest $n$", "\\shortstack{proven by the\\\\data test}"],
    rows)
N["NInfData"] = int(inf.data_infeasible.sum())
ex = inf.sort_values(["n"]).iloc[0]
N["InfExample"] = f"{ex.instance.replace('_', '')}\\_n{int(ex.n)}"
N["InfExampleEV"] = int(ex.data_infeasible_ev)
N["InfExampleFinish"] = F(ex.data_infeasible_finish, "{:.0f}")
N["InfExampleT"] = F(ex["T"], "{:.0f}")
# example request time and charge
rc.f4_import()
import f4_data as fd  # noqa: E402
ds = fd.Dataset(rc.F4_DATA, stems=("solomon",))
Jx = ds.instance(ex.instance, int(ex.n), analysis=False)
k_ = int(ex.data_infeasible_ev) - 1
N["InfExampleR"] = F(Jx["ev"]["r_min"][k_], "{:.0f}")
N["InfExampleTauM"] = F(Jx["ev"]["tauM_min"][k_], "{:.0f}")
tN = Jx["t_nodes_min"]
van_end = max(Jx["ev"]["r_min"][k_], tN[0][k_ + 1]) + Jx["ev"]["tauM_min"][k_]
N["InfExampleVanEnd"] = F(van_end, "{:.0f}")
N["InfExampleVanBack"] = F(van_end + tN[k_ + 1][0], "{:.0f}")
N["InfExampleTauF"] = F(Jx["ev"]["tauF_min"][k_], "{:.0f}")
un = inst[inst.status == "time_limit_no_plan"]
rows = []
for fam in rc.FAMS:
    g = un[un.family == fam]
    if len(g):
        rows.append([fam, len(g), int(g.n.min()), F(g.mean_pressure.mean(), "{:.2f}"),
                     F(valid[(valid.family == fam) & valid.n.isin(g.n.unique())].mean_pressure.mean(), "{:.2f}")])
tab("unsolved", ["family", "\\shortstack{unsolved\\\\instances}", "smallest $n$",
                 "\\shortstack{mean pressure\\\\(unsolved)}", "\\shortstack{mean pressure, solved\\\\(same family, sizes)}"],
    rows)
un = inst[inst.status == "time_limit_no_plan"]
N["UnsolvedFamilies"] = ", ".join(sorted(un.family.unique()))
N["UnsolvedPressure"] = F(un.mean_pressure.mean(), "{:.2f}")
same = valid[valid.family.isin(un.family.unique()) & valid.n.isin(un.n.unique())]
N["SolvedSamePressure"] = F(same.mean_pressure.mean(), "{:.2f}")
N["UnsolvedMinN"] = int(un.n.min())

# --- 4.12 sensitivity to the income distribution and segment mix (reweighting decile shares)
sh_dec = {s: free[free.segment == s].groupby("decile").van.mean().reindex(range(10)).fillna(0).values for s in SEGS}
edges = np.linspace(0, 1, 11)


def exp_share(seg, a, b_):
    w = np.diff(stats.beta.cdf(edges, a, b_))
    return float(np.sum(w * sh_dec[seg])), float(1 - stats.beta.cdf(0.6, a, b_))


rows = []
for lab_, s, a, b_ in [("personal: Beta(2.5, 1) (used)", "personal", 2.5, 1.0),
                       ("personal: Beta(2, 2) (formulation 3)", "personal", 2.0, 2.0),
                       ("personal: Beta(1, 1) uniform", "personal", 1.0, 1.0),
                       ("personal: Beta(4, 1) richer", "personal", 4.0, 1.0),
                       ("commercial: Beta(3, 5) (used)", "commercial", 3.0, 5.0),
                       ("commercial: Beta(2, 2)", "commercial", 2.0, 2.0),
                       ("commercial: Beta(5, 3) richer", "commercial", 5.0, 3.0)]:
    e_, hi = exp_share(s, a, b_)
    rows.append([lab_, F(a / (a + b_), "{:.2f}"), pct(hi), pct(e_)])
    if "used" in lab_:
        N[f"ExpShare{s.capitalize()}Used"] = pct(e_)
    if "formulation 3" in lab_:
        N["ExpSharePersFThree"] = pct(e_)
    if "Beta(4, 1)" in lab_:
        N["ExpSharePersRich"] = pct(e_)
    if "Beta(5, 3)" in lab_:
        N["ExpShareCommRich"] = pct(e_)
tab("beta", ["income distribution", "mean $\\theta$", "EVs with $\\theta\\ge0.6$ (\\%)", "expected van share (\\%)"], rows,
    align="lrrr")
sp, sc = exp_share("personal", 2.5, 1.0)[0], exp_share("commercial", 3.0, 5.0)[0]
rows = []
for piC in (0.3, 0.5, 0.7, 0.9):
    tot_ = (1 - piC) * sp + piC * sc
    rows.append([F(piC, "{:.1f}") + (" (used)" if piC == 0.7 else ""), pct(tot_), pct((1 - piC) * sp / tot_)])
    N["ExpVanPi" + {0.3: "Three", 0.5: "Five", 0.7: "Seven", 0.9: "Nine"}[piC]] = pct(tot_)
tab("pimix", ["commercial share $\\pi_{\\mathrm{C}}$", "\\shortstack{expected van share,\\\\all free-choice EVs (\\%)}",
              "\\shortstack{personal share of\\\\free-choice van users (\\%)}"], rows, align="lrr")

# robustness: optimal-only figures
N["VanPersOpt"] = pct(optf[optf.segment == "personal"].van.mean())

N.save(os.path.join(TAB, "numbers.tex"))
print(f"{len(N.d)} numbers; figures in {FIG}; tables in {TAB}")
