"""Greyscale figures and number macros for slides.tex (formulation-4 results).

Reads only the finished formulation-4 job (../../formulation_4: baseline_MILP, baseline_RL, the experiment
comparison tables) and the shared utilisation code of ../../formulation_4/report/comprehensive/common.
Writes figures/*.png (300 dpi, greys #333333 / #6e6e6e, hatching, no top/right spines, transparent) and numbers.tex.
Usage: python make_figures.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))                  # repo root (this folder: presentation/slides_2026-11-27)
F4 = os.path.join(REPO, "formulation_4")
sys.path.insert(0, os.path.join(F4, "report", "comprehensive", "common"))
sys.path.insert(0, os.path.join(F4, "common"))
from paths import resolve as F4J  # noqa: E402
import infra  # noqa: E402

FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
DARK, MID, LIGHT = "#333333", "#6e6e6e", "#c8c8c8"
ACCP, ACCA = "#5F4B6B", "#C58A63"            # muted accents of the slide template (accP, accA)
ACCP_L, ACCA_L = "#cfc5d6", "#ecd3c1"        # light tints
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": DARK, "axes.labelcolor": DARK, "xtick.color": DARK, "ytick.color": DARK,
                     "legend.frameon": False, "savefig.transparent": True, "savefig.dpi": 300})
N = {}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


def f1(x):
    return f"{x:.1f}"


# ---------------------------------------------------------------- runs
m_all = pd.read_csv(F4J("baseline_MILP/results/runs.csv"), low_memory=False)
m = m_all[m_all.run_id == "milp_300s"].copy()
r = pd.read_csv(F4J("baseline_RL/results/runs.csv"), low_memory=False)
m["ok"], r["ok"] = m.valid.eq("yes"), r.valid.eq("yes")
st = m.status.value_counts()
N.update(NInst=len(m), NOpt=int(st.get("optimal", 0)), NTL=int(st.get("time_limit", 0)),
         NNoPlan=int(st.get("time_limit_no_plan", 0)), NInf=int(st.get("infeasible", 0)),
         NValidM=int(m.ok.sum()), NValidR=int(r.ok.sum()))
key = ["instance", "n"]
j = r[r.ok].merge(m[m.ok][key + ["Z_total", "status", "solve_s"]], on=key, suffixes=("", "_m"))
j["gap"] = 100 * (j.Z_total / j.Z_total_m - 1)
N.update(NPaired=len(j), GapMean=f"{j.gap.mean():.2f}", GapMed=f"{j.gap.median():.2f}",
         GapOpt=f"{j[j.status_m == 'optimal'].gap.mean():.2f}", RlBetter=int((j.gap < -1e-6).sum()),
         RlOnly=len(set(zip(r[r.ok].instance, r[r.ok].n)) - set(zip(m[m.ok].instance, m[m.ok].n))),
         TimeM=f1(m.solve_s.mean()), TimeR=f"{r.solve_s.mean():.2f}",
         Speed=f"{m.solve_s.mean() / r.solve_s.mean():,.0f}".replace(",", "\\,"))
meta = json.load(open(F4J("baseline_RL/model/train_meta.json")))
N.update(TrainMin=f1(meta["train_s"] / 60), TrainEpochs=meta["epoch"], BestEpoch=meta["best_epoch"],
         Params=f"{meta['parameters']:,}".replace(",", "\\,"))

# fig: outcome by n (MILP stacked) + RL valid
sizes = list(range(10, 101, 10))
cats = [("optimal", "optimal", ACCP, None), ("time_limit", "time limit, plan", ACCA, None),
        ("time_limit_no_plan", "time limit, no plan", "white", "////"), ("infeasible", "infeasible", LIGHT, None)]
fig, ax = plt.subplots(figsize=(4.6, 2.9))
bottom = np.zeros(len(sizes))
for s, lab, c, h in cats:
    v = np.array([((m.n == n) & (m.status == s)).sum() for n in sizes])
    ax.bar(sizes, v, 7, bottom=bottom, color=c, edgecolor=DARK, linewidth=0.5, hatch=h, label=f"MILP: {lab}")
    bottom += v
ax.plot(sizes, [(r[(r.n == n)].ok).sum() for n in sizes], "o-", color="black", ms=3.5, lw=1.2,
        markerfacecolor="white", label="RL: valid plans")
ax.set_xlabel("EVs per instance $n$")
ax.set_ylabel("instances (of 56)")
ax.set_xticks(sizes)
ax.legend(fontsize=7, loc="lower left", ncol=1)
save(fig, "outcome.png")

# fig: RL gap by n
fig, ax = plt.subplots(figsize=(4.6, 2.9))
data = [j[j.n == n].gap.values for n in sizes]
bp = ax.boxplot(data, positions=sizes, widths=6, patch_artist=True, showfliers=False)
for b in bp["boxes"]:
    b.set(facecolor=ACCA_L, edgecolor=ACCA, linewidth=0.8)
for k in ("whiskers", "caps"):
    for x in bp[k]:
        x.set(color=DARK, linewidth=0.7)
for x in bp["medians"]:
    x.set(color=ACCP, linewidth=1.4)
ax.axhline(0, color=MID, lw=0.7, ls="--")
ax.set_xticks(sizes)
ax.set_xlabel("EVs per instance $n$")
ax.set_ylabel("RL cost above MILP (%)")
save(fig, "gap.png")

# ---------------------------------------------------------------- van share by segment x income (paired instances)
paired = set(zip(j.instance, j.n))


def plan_ev(path, run):
    p = pd.read_csv(path, low_memory=False)
    p = p[(p.run_id == run) & (p.type == "ev")]
    return p[[k in paired for k in zip(p.instance, p.n)]]


pm = plan_ev(F4J("baseline_MILP/results/plan.csv"), "milp_300s")
pr = plan_ev(F4J("baseline_RL/results/plan.csv"), "rl_f4")
edges = np.linspace(0, 1, 6)
mids = (edges[:-1] + edges[1:]) / 2
fig, ax = plt.subplots(figsize=(4.6, 2.9))
for p, meth, ls, mk in ((pm, "MILP", "-", "o"), (pr, "RL", "--", "s")):
    fc = p[p.stranded == "no"]
    for seg, col in (("personal", ACCP), ("commercial", ACCA)):
        q = fc[fc.segment == seg]
        b = np.clip((q.theta // 0.2).astype(int), 0, 4)
        share = [100 * (q[b == k].resource_type != "FCS").mean() if (b == k).any() else np.nan for k in range(5)]
        ax.plot(mids, share, ls, marker=mk, color=col, ms=3.5, lw=1.2, markerfacecolor="white" if meth == "RL" else col,
                label=f"{meth}, {seg}")
    for seg in ("personal", "commercial"):
        q = fc[fc.segment == seg]
        N[f"Van{seg.capitalize()}{meth.capitalize()}"] = f1(100 * (q.resource_type != "FCS").mean())
ax.set_xlabel(r"income index $\theta$ (band midpoint)")
ax.set_ylabel("free-choice EVs served by a van (%)")
ax.set_xticks(mids)
ax.set_xticklabels(["0-.2", ".2-.4", ".4-.6", ".6-.8", ".8-1"])
ax.legend(fontsize=7, loc="upper left")
save(fig, "vanshare.png")
same = pm.merge(pr, on=key + ["id"], suffixes=("_m", "_r"))
fcs = same[same.stranded_m == "no"]
N["Agree"] = f1(100 * ((fcs.resource_type_m == "FCS") == (fcs.resource_type_r == "FCS")).mean())

# ---------------------------------------------------------------- utilisation (MILP plans, paired = all valid MILP)
plm = pd.read_csv(F4J("baseline_MILP/results/plan.csv"), low_memory=False)
plm = plm[(plm.run_id == "milp_300s") & np.array([k in paired for k in zip(plm.instance, plm.n)])]
plr = pd.read_csv(F4J("baseline_RL/results/plan.csv"), low_memory=False)
plr = plr[(plr.run_id == "rl_f4") & np.array([k in paired for k in zip(plr.instance, plr.n)])]
U = {"MILP": infra.utilisation(plm), "RL": infra.utilisation(plr)}
for meth, u in U.items():
    I, V = u["instances"], u["vans"]
    t = meth.capitalize()
    N[f"GunUtil{t}"] = f1(100 * I.gun_util.mean())
    N[f"IdleGuns{t}"] = f1(I.idle_guns.mean())
    N[f"Fleet{t}"] = f1(100 * I.fleet_use.mean())
    N[f"Seats{t}"] = f1(100 * V.seat_util.mean())
    N[f"Batt{t}"] = f1(100 * V.battery_util.mean())
    N[f"Road{t}"] = f1(100 * V.on_road_util.mean())
    N[f"Charge{t}"] = f1(100 * V.charge_util.mean())
    N[f"VanIdle{t}"] = f1(100 * (V.idle_min / V.on_road_min).mean())
    N[f"Sys{t}"] = f1(100 * I.system_util.mean())
    N[f"VanEnergy{t}"] = f1(100 * (I.kWh_van / I.kWh_total).mean())
Im = U["MILP"]["instances"]
short = Im.family.isin(["C1", "R1", "RC1"])
N["GunUtilShort"], N["GunUtilLong"] = f1(100 * Im[short].gun_util.mean()), f1(100 * Im[~short].gun_util.mean())
prof = U["MILP"]["profile"]
fig, axs = plt.subplots(1, 2, figsize=(5.2, 2.6), sharey=True)
for ax, (fams, title) in zip(axs, ((["C1", "R1", "RC1"], "short horizon (C1, R1, RC1)"),
                                     (["C2", "R2", "RC2"], "long horizon (C2, R2, RC2)"))):
    q = prof[prof.family.isin(fams)].groupby("t_rel")[["guns_busy", "vans_busy"]].mean()
    ax.fill_between(q.index, 100 * q.guns_busy, color=ACCA, alpha=0.55, step="mid", label="guns charging")
    ax.step(q.index, 100 * q.vans_busy, where="mid", color=ACCP, lw=1.4, label="vans charging")
    ax.set_title(title, fontsize=8, color=DARK)
    ax.set_xlabel("time $t/T$")
axs[0].set_ylabel("share charging (%)")
axs[0].legend(fontsize=7, loc="upper right")
save(fig, "busy.png")

# ---------------------------------------------------------------- experiments: change of Z, MILP vs RL (retrained)
rows = []
for E in ("E5_value_of_charging_time", "E6_money_sensitivity", "E1_van_tariff", "E2_van_fleet_and_seats",
          "E8_van_battery"):
    cm = pd.read_csv(F4J(f"{E}_MILP/results/comparison/comparison.csv"))
    cr = pd.read_csv(F4J(f"{E}_RL/results/comparison/comparison.csv"))
    cr = cr[cr.run_id == "rl_f4_retrained"]
    for _, a in cm.iterrows():
        b = cr[cr.variant == a.variant]
        rows.append((a.variant, a.label, a["mean_dZ_%"], float(b["mean_dZ_%"].iloc[0]) if len(b) else np.nan,
                     a.share_van_personal_ - a.baseline_share_van_personal_ if False else
                     a["share_van_personal_%"] - a["baseline_share_van_personal_%"]))
ex = pd.DataFrame(rows, columns=["variant", "label", "dZ_milp", "dZ_rl", "dvan_pers_milp"])
ex.to_csv(os.path.join(FIG, "experiments.csv"), index=False)
short_lab = {"E5_cC1_personal_7.5": r"personal $c^{C,1}$ 7.5", "E5_cC1_personal_12.5": r"personal $c^{C,1}$ 12.5",
             "E5_omega1_commercial_4.5": r"commercial $\omega^1$ 4.5", "E5_omega1_commercial_7.5": r"commercial $\omega^1$ 7.5",
             "E6_mu0_personal_1.125": r"personal $\mu^0$ 1.125", "E6_mu0_personal_1.875": r"personal $\mu^0$ 1.875",
             "E6_mu0_commercial_1.125": r"commercial $\mu^0$ 1.125", "E6_mu0_commercial_1.875": r"commercial $\mu^0$ 1.875",
             "E1_pM_15": r"van tariff $p^M$ 15", "E1_pM_35": r"van tariff $p^M$ 35", "E2_vans_3": "3 vans",
             "E2_vans_7": "7 vans", "E2_seats_4": "4 seats / van", "E2_seats_8": "8 seats / van",
             "E8_battery_200": "battery 200 kWh"}
ex["short"] = ex.variant.map(short_lab).fillna(ex.variant)
ex = ex.iloc[np.argsort(-ex.dZ_milp.abs().values)]
fig, ax = plt.subplots(figsize=(5.0, 3.4))
y = np.arange(len(ex))
ax.barh(y - 0.2, ex.dZ_milp, 0.38, color=ACCP, label="MILP")
ax.barh(y + 0.2, ex.dZ_rl, 0.38, color=ACCA_L, edgecolor=ACCA, hatch="////", linewidth=0.5, label="RL (retrained)")
ax.set_yticks(y)
ax.set_yticklabels(ex.short, fontsize=7)
ax.invert_yaxis()
ax.axvline(0, color=DARK, lw=0.6)
ax.set_xlabel("mean change of $Z$ vs baseline (%)")
ax.legend(fontsize=7, loc="lower right")
save(fig, "experiments.png")
g = ex.set_index("variant")
for v, name in (("E6_mu0_commercial_1.125", "EsixComLo"), ("E6_mu0_commercial_1.875", "EsixComHi"),
                ("E1_pM_15", "EonePMLo"), ("E1_pM_35", "EonePMHi"), ("E8_battery_200", "Eeight"),
                ("E2_seats_4", "EtwoSeatsLo"), ("E5_cC1_personal_12.5", "EfivePersHi")):
    N[f"dZ{name}"] = f"{g.loc[v, 'dZ_milp']:+.2f}".replace("-", "$-$")
    N[f"dVan{name}"] = f"{g.loc[v, 'dvan_pers_milp']:+.1f}".replace("-", "$-$")

# =====================================================================================================
# RESULTS DEEP-DIVE: income x segment, demand pressure, infrastructure, experiments (slides 16-25)
# All on the paired instances (valid for MILP and RL) unless stated. Conventions of the comprehensive
# reports: free-choice = not stranded; van share over free-choice EVs; queue and drive over station users;
# utilisation = mean of per-plan / per-van shares (infra.py).
# =====================================================================================================
BL = ["0-.2", ".2-.4", ".4-.6", ".6-.8", ".8-1"]
SEGS = ("personal", "commercial")
SEGCOL = {"personal": ACCP, "commercial": ACCA}
PT = infra.pressure_table()
PT["stranded"] = PT.stranded.astype(str).str.lower().isin(["true", "1"])
QC1, QC2 = infra.pressure_cuts()
N["PressCutLo"], N["PressCutHi"] = f"{QC1:.2f}", f"{QC2:.2f}"
N["PressLoadLo"], N["PressLoadHi"] = int(round(QC1 * 29)), int(round(QC2 * 29))
N["PressWinMin"], N["PressWinMax"] = int(PT.window_min.min()), int(PT.window_min.max())
fcpt = PT[~PT.stranded]
N["PeakLabelShare"] = f1(100 * (fcpt.period == "peak").mean())
N["PeakLabelHigh"] = f1(100 * (fcpt[fcpt.period == "peak"].terc == "high").mean())
N["OffpeakHigh"] = f1(100 * ((fcpt.period != "peak") & (fcpt.terc == "high")).mean())


def ev_frame(p):
    q = p.copy()
    q["free"] = q.stranded == "no"
    q["van"] = q.resource_type != "FCS"
    q["band"] = np.clip((q.theta // 0.2).astype(int), 0, 4)
    q["dec"] = np.clip((q.theta // 0.1).astype(int), 0, 9)
    q = q.merge(PT[["instance", "n", "id", "pressure", "terc", "period"]], on=["instance", "n", "id"], how="left")
    return q


EM, ER = ev_frame(pm), ev_frame(pr)
EM = EM.merge(m[["instance", "n", "status"]], on=["instance", "n"], how="left")


def by(q, keys):
    fc, fu = q[q.free], q[q.resource_type == "FCS"]
    g = pd.DataFrame({"van": 100 * fc.groupby(keys).van.mean(), "nfree": fc.groupby(keys).size()})
    g["queue"] = fu.groupby(keys).queue_min.mean()
    g["drive"] = fu.groupby(keys).drive_km.mean()
    g["ttc"] = q.groupby(keys).time_to_charged_min.mean()
    g["pay"] = q.groupby(keys).pay_Rs.mean()
    g["perc"] = q.groupby(keys).customer_cost_Rs.mean()
    return g


SBm, SBr = by(EM, ["segment", "band"]), by(ER, ["segment", "band"])

# ---- figure: van share by income decile, segment, method (+ optimal-only MILP, theory threshold)
dm = (np.arange(10) + 0.5) / 10
fig, ax = plt.subplots(figsize=(4.8, 2.9))
for seg in SEGS:
    c = SEGCOL[seg]
    for q, ls, lab in ((EM, "-", "MILP"), (ER, "--", "RL")):
        f = q[q.free & (q.segment == seg)]
        ax.plot(dm, [100 * f[f.dec == k].van.mean() for k in range(10)], ls, color=c, marker="o", ms=3,
                markerfacecolor=c if lab == "MILP" else "white", lw=1.3, label=f"{seg}, {lab}")
    if seg == "personal":
        f = EM[EM.free & (EM.segment == seg) & (EM.status == "optimal")]
        ax.plot(dm, [100 * f[f.dec == k].van.mean() for k in range(10)], ":", color=c, lw=1.2,
                label="personal, MILP optimal only")
ax.axvline(0.66, color=MID, lw=0.8, ls="-.")
ax.text(0.665, 45, r"$\theta^*_P=0.66$", fontsize=7, color=MID)
ax.set_xlabel(r"income index $\theta$ (decile midpoint)")
ax.set_ylabel("free-choice EVs by van (%)")
ax.legend(fontsize=6.5, loc="upper left")
save(fig, "van_decile.png")
fp = EM[EM.free & (EM.segment == "personal")]
fpo = fp[fp.status == "optimal"]
N["VanPersOptBelowFour"] = f1(100 * fpo[fpo.theta < 0.4].van.mean())
N["VanPersOptTop"] = f1(100 * fpo[fpo.dec == 9].van.mean())
N["VanPersTopR"] = f1(100 * ER[ER.free & (ER.segment == "personal") & (ER.dec == 9)].van.mean())
cv = EM[EM.free & (EM.segment == "commercial") & EM.van]
N["NComVan"] = len(cv)
N["ComVanTL"] = f1(100 * (cv.status != "optimal").mean())
N["ComVanWait"] = f"{cv.van_wait_min.mean():.0f}"
N["ComVanHigh"] = f1(100 * (cv.terc == "high").mean())
N["ComVanThetaR"] = f"{ER[ER.free & (ER.segment == 'commercial') & ER.van].theta.mean():.2f}"


# theory: threshold of a typical EV (E=20 kWh, 2 km to the station, no queue, 10-min van wait, 4 km detour)
def delta(theta, seg, q=0.0):
    E, d = 20.0, 2.0
    t, tauF, tauM, qM, x = 60 * d / 15.6, 60 * E / 22, 60 * E / 30, 10.0, 4.0
    mu = 1.5 - theta
    if seg == "personal":
        cD, cQ, cC, cV = 1.5 + 13.5 * theta, 2.5 + 22.5 * theta, 1 + 9 * theta, 0.5 + 4.5 * theta
    else:
        cD = cC = cV = 1.5 + 4.5 * theta
        cQ = 2 + 5.5 * theta
    kappa = (10.7 / 0.9 - 8.7) * E + 15 * x
    return mu * ((25 - 15) * E - 4 * d) + kappa - cD * t - cQ * q - cC * tauF + cV * (qM + tauM)


d0, d1 = delta(0, "personal"), delta(1, "personal")
N["ThetaStarP"] = f"{d0 / (d0 - d1):.2f}"
N["DeltaComOne"] = f"{delta(1, 'commercial'):.0f}"
N["QueueNeedCom"] = f"{delta(1, 'commercial') / 7.5:.0f}"
N["QueueNeedComZero"] = f"{delta(0, 'commercial') / 2.0:.0f}"

# ---- figure: 2x2 panels by income band: queue, drive, time to charged, perceived cost
fig, axs = plt.subplots(2, 2, figsize=(5.4, 3.5), sharex=True)
xb = np.arange(5)
for ax, col, yl in zip(axs.flat, ("queue", "drive", "ttc", "perc"),
                       ("station queue (min)", "drive to station (km)", "request to charged (min)",
                        "perceived cost per EV (Rs)")):
    for seg in SEGS:
        ax.plot(xb, SBm.loc[seg][col].reindex(range(5)).values, "-o", color=SEGCOL[seg], ms=3, lw=1.3,
                label=f"{seg}, MILP")
        ax.plot(xb, SBr.loc[seg][col].reindex(range(5)).values, "--o", color=SEGCOL[seg], ms=3, lw=1.1,
                markerfacecolor="white", label=f"{seg}, RL")
    ax.set_ylabel(yl, fontsize=7)
    ax.tick_params(labelsize=7)
for ax in axs[1]:
    ax.set_xticks(xb)
    ax.set_xticklabels(BL)
    ax.set_xlabel(r"income band $\theta$", fontsize=7)
axs[0, 0].legend(fontsize=6, loc="upper right")
save(fig, "seg_income.png")


# ---- who queues: share of station users vs share of queue minutes
def qshare(q):
    fu = q[q.resource_type == "FCS"]
    tot_u, tot_q = len(fu), fu.queue_min.sum()
    out = {}
    for seg in SEGS:
        s = fu[fu.segment == seg]
        out[seg] = (100 * len(s) / tot_u, 100 * s.queue_min.sum() / tot_q)
    for b in range(5):
        s = fu[fu.band == b]
        out[b] = (100 * len(s) / tot_u, 100 * s.queue_min.sum() / tot_q)
    s = fu[fu.theta < 0.4]
    out["poor"] = (100 * len(s) / tot_u, 100 * s.queue_min.sum() / tot_q)
    return out, 100 * (fu.queue_min > 1e-6).mean()


QSm, queuedM = qshare(EM)
QSr, queuedR = qshare(ER)
N["QUserComM"], N["QShareComM"] = f1(QSm["commercial"][0]), f1(QSm["commercial"][1])
N["QUserComR"], N["QShareComR"] = f1(QSr["commercial"][0]), f1(QSr["commercial"][1])
N["QUserPoorM"], N["QSharePoorM"] = f1(QSm["poor"][0]), f1(QSm["poor"][1])
N["QUserPoorR"], N["QSharePoorR"] = f1(QSr["poor"][0]), f1(QSr["poor"][1])
N["QueuedM"], N["QueuedR"] = f1(queuedM), f1(queuedR)
rows = []
BLT = ["0.0--0.2", "0.2--0.4", "0.4--0.6", "0.6--0.8", "0.8--1.0"]
for k, lab in [("personal", "personal"), ("commercial", "commercial")] + [(b, f"$\\theta$ {BLT[b]}")
                                                                          for b in range(5)]:
    rows.append(f"{lab} & {QSm[k][0]:.1f} & {QSm[k][1]:.1f} & {QSr[k][0]:.1f} & {QSr[k][1]:.1f}")
N["TabQueueShare"] = " \\\\ ".join(rows) + " \\\\"
# equity table: paid / perceived per EV, segment x band (MILP)
rows = []
for b in range(5):
    r_ = [f"$\\theta$ {['0.0--0.2', '0.2--0.4', '0.4--0.6', '0.6--0.8', '0.8--1.0'][b]}"]
    for seg in SEGS:
        r_ += [f"{SBm.loc[(seg, b), 'pay']:.0f}", f"{SBm.loc[(seg, b), 'perc']:.0f}",
               f"{SBr.loc[(seg, b), 'perc']:.0f}"]
    rows.append(" & ".join(r_))
N["TabEquity"] = " \\\\ ".join(rows) + " \\\\"
for seg, t in (("personal", "Pers"), ("commercial", "Com")):
    s = EM[EM.segment == seg]
    N[f"Pay{t}"] = f"{s.pay_Rs.mean():.0f}"
    N[f"Perc{t}"] = f"{s.customer_cost_Rs.mean():.0f}"
    N[f"TimeShare{t}"] = f1(100 * s.time_cost_Rs.sum() / s.customer_cost_Rs.sum())

# ---- demand pressure
TER = ["low", "medium", "high"]


def by_terc(q):
    fc, fu = q[q.free], q[q.resource_type == "FCS"]
    out = {}
    for t in TER:
        out[t] = {"share": 100 * (fc.terc == t).mean(),
                  "queue": fu[fu.terc == t].queue_min.mean(),
                  "queued": 100 * (fu[fu.terc == t].queue_min > 1e-6).mean()}
        for seg in SEGS:
            out[t][f"van_{seg}"] = 100 * fc[(fc.terc == t) & (fc.segment == seg)].van.mean()
            out[t][f"q_{seg}"] = fu[(fu.terc == t) & (fu.segment == seg)].queue_min.mean()
    return out


PTm, PTr = by_terc(EM), by_terc(ER)
rows = []
for t in TER:
    a, b = PTm[t], PTr[t]
    rows.append(f"{t} & {a['share']:.1f} & {a['van_personal']:.1f} & {b['van_personal']:.1f} & "
                f"{a['van_commercial']:.1f} & {b['van_commercial']:.1f} & {a['q_personal']:.2f} & "
                f"{b['q_personal']:.2f} & {a['q_commercial']:.2f} & {b['q_commercial']:.2f}")
N["TabPress"] = " \\\\ ".join(rows) + " \\\\"
for t, tt in zip(TER, ("Lo", "Med", "Hi")):
    N[f"QueueAll{tt}M"] = f"{PTm[t]['queue']:.2f}"
    N[f"QueueAll{tt}R"] = f"{PTr[t]['queue']:.2f}"
    N[f"Queued{tt}M"] = f1(PTm[t]["queued"])
    N[f"VanPers{tt}M"] = f1(PTm[t]["van_personal"])
    N[f"VanCom{tt}M"] = f1(PTm[t]["van_commercial"])
    N[f"VanCom{tt}R"] = f1(PTr[t]["van_commercial"])


# guns busy at the moment of each EV's request (in that EV's plan)
def busy_at_request(q):
    vals = []
    for (inst, n), g in q.groupby(["instance", "n"]):
        gun = g[g.resource_type == "FCS"]
        s, e = gun.start_min.to_numpy(), gun.end_min.to_numpy()
        req = g.start_min - g.queue_min - g.van_wait_min - np.where(g.resource_type == "FCS", g.drive_min, 0.0)
        busy = ((s[None, :] <= req.to_numpy()[:, None]) & (e[None, :] > req.to_numpy()[:, None])).sum(1) / 24
        vals.append(pd.DataFrame({"terc": g.terc.values, "free": g.free.values, "busy": busy}))
    return pd.concat(vals)


BAm, BAr = busy_at_request(EM), busy_at_request(ER)
for t, tt in zip(TER, ("Lo", "Med", "Hi")):
    N[f"Busy{tt}M"] = f1(100 * BAm[BAm.free & (BAm.terc == t)].busy.mean())
    N[f"Busy{tt}R"] = f1(100 * BAr[BAr.free & (BAr.terc == t)].busy.mean())
    N[f"BusyHalf{tt}M"] = f1(100 * (BAm[BAm.free & (BAm.terc == t)].busy >= 0.5).mean())

# figure: van share by tercile x segment (bars) + queue by income band at high pressure
fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.4))
ax = axs[0]
xt = np.arange(3)
w = 0.2
for k, (seg, meth, P_, h) in enumerate((("personal", "MILP", PTm, None), ("personal", "RL", PTr, "////"),
                                         ("commercial", "MILP", PTm, None), ("commercial", "RL", PTr, "////"))):
    vals = [P_[t][f"van_{seg}"] for t in TER]
    ax.bar(xt + (k - 1.5) * w, vals, w, color=SEGCOL[seg] if h is None else "white", edgecolor=SEGCOL[seg],
           hatch=h, linewidth=0.6, label=f"{seg[:4]}., {meth}")
ax.set_xticks(xt)
ax.set_xticklabels(TER)
ax.set_xlabel("demand pressure", fontsize=7)
ax.set_ylabel("free-choice EVs by van (%)", fontsize=7)
ax.legend(fontsize=5.5, loc="upper right")
ax.tick_params(labelsize=7)
ax = axs[1]
for seg in SEGS:
    for q, ls, mf in ((EM, "-o", None), (ER, "--o", "white")):
        fu = q[(q.resource_type == "FCS") & (q.terc == "high") & (q.segment == seg)]
        ax.plot(xb, [fu[fu.band == b].queue_min.mean() for b in range(5)], ls, color=SEGCOL[seg], ms=3,
                markerfacecolor=mf or SEGCOL[seg], lw=1.2)
ax.set_xticks(xb)
ax.set_xticklabels(BL, fontsize=6.5)
ax.set_xlabel(r"income band (high pressure)", fontsize=7)
ax.set_ylabel("station queue (min)", fontsize=7)
ax.tick_params(labelsize=7)
save(fig, "pressure.png")

# ---- experiments at low vs high pressure (paired with the same method's baseline)
EXPV = [("E1_van_tariff", "E1_pM_15", r"$p^M$ 15", {}), ("E1_van_tariff", "E1_pM_35", r"$p^M$ 35", {}),
        ("E2_van_fleet_and_seats", "E2_vans_3", "3 vans", {"n_vans": 3}),
        ("E2_van_fleet_and_seats", "E2_vans_7", "7 vans", {"n_vans": 7}),
        ("E2_van_fleet_and_seats", "E2_seats_4", "4 seats", {"van_max_evs": 4}),
        ("E2_van_fleet_and_seats", "E2_seats_8", "8 seats", {"van_max_evs": 8}),
        ("E5_value_of_charging_time", "E5_omega1_commercial_4.5", r"taxi $\omega^1$ 4.5", {}),
        ("E5_value_of_charging_time", "E5_omega1_commercial_7.5", r"taxi $\omega^1$ 7.5", {}),
        ("E5_value_of_charging_time", "E5_cC1_personal_7.5", r"pers. $c^{C,1}$ 7.5", {}),
        ("E5_value_of_charging_time", "E5_cC1_personal_12.5", r"pers. $c^{C,1}$ 12.5", {}),
        ("E6_money_sensitivity", "E6_mu0_personal_1.125", r"pers. $\mu^0$ 1.125", {}),
        ("E6_money_sensitivity", "E6_mu0_personal_1.875", r"pers. $\mu^0$ 1.875", {}),
        ("E6_money_sensitivity", "E6_mu0_commercial_1.125", r"taxi $\mu^0$ 1.125", {}),
        ("E6_money_sensitivity", "E6_mu0_commercial_1.875", r"taxi $\mu^0$ 1.875", {}),
        ("E8_van_battery", "E8_battery_200", "battery 200", {"van_battery_kWh": 200})]
base_runs = {"MILP": m[m.ok], "RL": r[r.ok]}
base_plan = {"MILP": pd.read_csv(F4J("baseline_MILP/results/plan.csv"), low_memory=False),
             "RL": pd.read_csv(F4J("baseline_RL/results/plan.csv"), low_memory=False)}
base_plan["MILP"] = base_plan["MILP"][base_plan["MILP"].run_id == "milp_300s"]
UB = {"MILP": infra.utilisation(base_plan["MILP"][base_plan["MILP"].type.isin(["ev", "van"]) &
                                                   np.array([k in set(zip(m[m.ok].instance, m[m.ok].n))
                                                             for k in zip(base_plan["MILP"].instance,
                                                                          base_plan["MILP"].n)])])}


def util_summary(u, keys=None):
    I, V = u["instances"], u["vans"]
    if keys is not None:
        I = I[np.array([k in keys for k in zip(I.instance, I.n)])]
        V = V[np.array([k in keys for k in zip(V.instance, V.n)])]
    return {"gun": 100 * I.gun_util.mean(), "fleet": 100 * I.fleet_use.mean(), "seats": 100 * V.seat_util.mean(),
            "batt": 100 * V.battery_util.mean(), "vanE": 100 * (I.kWh_van / I.kWh_total).mean()}


def sgn(x, d=1):
    s = f"{x:+.{d}f}"
    return s.replace("-", "$-$")


exp_rows_cost, exp_rows_who, exp_rows_press = [], [], []
for E, v, lab, cfg in EXPV:
    cm = pd.read_csv(F4J(f"{E}_MILP/results/comparison/comparison.csv")).set_index("variant").loc[v]
    cr = pd.read_csv(F4J(f"{E}_RL/results/comparison/comparison.csv"))
    zs = cr[(cr.variant == v) & (cr.run_id == "rl_f4_zero_shot")].iloc[0]
    rt = cr[(cr.variant == v) & (cr.run_id == "rl_f4_retrained")].iloc[0]
    exp_rows_cost.append(f"{lab} & {int(cm.valid)} ({int(cm.newly_valid)}/{int(cm.no_longer_valid)}) & "
                         f"{sgn(cm['mean_dZ_%'], 2)} & {sgn(cm['mean_d_operator_cost_%'], 2)} & "
                         f"{sgn(cm['mean_d_customer_cost_%'], 2)} & {sgn(zs['mean_dZ_%'], 2)} & "
                         f"{sgn(rt['mean_dZ_%'], 2)}")
    # utilisation change and pressure split, MILP and RL retrained, paired with baseline
    pres = {}
    for meth, folder, run in (("MILP", f"{E}_MILP", "milp_300s"), ("RL", f"{E}_RL", "rl_f4_retrained")):
        rv = pd.read_csv(F4J(folder, "results", v, "runs.csv"), low_memory=False)
        rv = rv[(rv.run_id == run) & rv.valid.eq("yes")]
        keys = set(zip(rv.instance, rv.n)) & set(zip(base_runs[meth].instance, base_runs[meth].n))
        pv = pd.read_csv(F4J(folder, "results", v, "plan.csv"), low_memory=False)
        pv = pv[(pv.run_id == run) & np.array([k in keys for k in zip(pv.instance, pv.n)])]
        pb = base_plan[meth]
        pb = pb[(pb.run_id == ("milp_300s" if meth == "MILP" else "rl_f4")) &
                np.array([k in keys for k in zip(pb.instance, pb.n)])]
        Ev, Eb = ev_frame(pv[pv.type == "ev"]), ev_frame(pb[pb.type == "ev"])
        d = {}
        for t in ("low", "high"):
            for seg in SEGS:
                a = Ev[Ev.free & (Ev.terc == t) & (Ev.segment == seg)].van.mean()
                b = Eb[Eb.free & (Eb.terc == t) & (Eb.segment == seg)].van.mean()
                d[f"{seg}_{t}"] = 100 * (a - b)
        fa = Ev[(Ev.resource_type == "FCS") & (Ev.terc == "high") & (Ev.segment == "commercial")].queue_min.mean()
        fb = Eb[(Eb.resource_type == "FCS") & (Eb.terc == "high") & (Eb.segment == "commercial")].queue_min.mean()
        d["q_com_high"] = fa - fb
        if meth == "MILP":
            uv = util_summary(infra.utilisation(pv, cfg))
            ub = util_summary(UB["MILP"], keys)
            d["util"] = {k: uv[k] - ub[k] for k in uv}
            d["vanP"] = cm["share_van_personal_%"] - cm["baseline_share_van_personal_%"]
            d["vanC"] = cm["share_van_commercial_%"] - cm["baseline_share_van_commercial_%"]
        else:
            d["vanP"] = rt["share_van_personal_%"] - rt["baseline_share_van_personal_%"]
        pres[meth] = d
    u = pres["MILP"]["util"]
    exp_rows_who.append(f"{lab} & {sgn(pres['MILP']['vanP'])} & {sgn(pres['MILP']['vanC'])} & "
                        f"{sgn(pres['RL']['vanP'])} & {sgn(u['gun'])} & {sgn(u['fleet'])} & {sgn(u['seats'])} & "
                        f"{sgn(u['batt'])} & {sgn(u['vanE'])}")
    if E in ("E1_van_tariff", "E2_van_fleet_and_seats") or "omega" in v:
        a, b = pres["MILP"], pres["RL"]
        exp_rows_press.append(f"{lab} & {sgn(a['personal_low'])} & {sgn(a['personal_high'])} & "
                              f"{sgn(a['commercial_high'])} & {sgn(b['commercial_high'])} & "
                              f"{sgn(a['q_com_high'], 2)}")
    if v == "E5_omega1_commercial_7.5":
        N["OmegaHiComHighM"] = sgn(pres["MILP"]["commercial_high"])
        N["OmegaHiComHighR"] = sgn(pres["RL"]["commercial_high"])
    if v == "E1_pM_15":
        N["PMLoPersLowM"] = sgn(pres["MILP"]["personal_low"])
        N["PMLoPersHighM"] = sgn(pres["MILP"]["personal_high"])
    if v == "E2_vans_7":
        N["VansSevenFleet"] = sgn(pres["MILP"]["util"]["fleet"])
        N["VansSevenComHigh"] = sgn(pres["MILP"]["commercial_high"])
    if v == "E8_battery_200":
        N["EeightBatt"] = sgn(pres["MILP"]["util"]["batt"])
    if v == "E2_seats_8":
        N["SeatsEightSeats"] = sgn(pres["MILP"]["util"]["seats"])
N["TabExpCost"] = " \\\\ ".join(exp_rows_cost) + " \\\\"
N["TabExpWho"] = " \\\\ ".join(exp_rows_who) + " \\\\"
N["TabExpPress"] = " \\\\ ".join(exp_rows_press) + " \\\\"


# ---- infrastructure: by family table, distributions, by n
def peak_guns(plan):
    out = {}
    for (inst, n), g in plan[(plan.type == "ev") & (plan.resource_type == "FCS")].groupby(["instance", "n"]):
        ev = sorted([(s, 1) for s in g.start_min] + [(e, -1) for e in g.end_min], key=lambda x: (x[0], x[1]))
        c = mx = 0
        for _, dlt in ev:
            c += dlt
            mx = max(mx, c)
        out[(inst, n)] = mx
    return out


PGm, PGr = peak_guns(plm), peak_guns(plr)
rows = []
for fam in ["C1", "C2", "R1", "R2", "RC1", "RC2"]:
    vals = []
    for meth, u, pg in (("MILP", U["MILP"], PGm), ("RL", U["RL"], PGr)):
        I, V = u["instances"], u["vans"]
        I, V = I[I.family == fam], V[V.family == fam]
        vals.append((100 * I.gun_util.mean(), I.idle_guns.mean(),
                     np.mean([pg.get(k, 0) for k in zip(I.instance, I.n)]), 100 * I.fleet_use.mean(),
                     100 * V.seat_util.mean(), 100 * V.battery_util.mean(), 100 * (I.kWh_van / I.kWh_total).mean(),
                     int(I["T"].iloc[0])))
    a, b = vals
    rows.append(f"{fam} & {a[7]} & {a[0]:.1f} & {b[0]:.1f} & {a[1]:.1f} & {a[2]:.1f} & {a[3]:.1f} & {b[3]:.1f} & "
                f"{a[4]:.1f} & {b[4]:.1f} & {a[6]:.1f} & {b[6]:.1f}")
N["TabInfraFam"] = " \\\\ ".join(rows) + " \\\\"
N["PeakGunsM"] = f1(np.mean(list(PGm.values())))
N["PeakGunsR"] = f1(np.mean(list(PGr.values())))
N["PeakGunsHundredM"] = f1(np.mean([v_ for (i_, n_), v_ in PGm.items() if n_ == 100]))
N["AllGunsBusyPlans"] = sum(v_ >= 24 for v_ in PGm.values())
Gm, Gr = U["MILP"]["guns"], U["RL"]["guns"]
N["GunsNeverM"] = f1(100 * (Gm.evs == 0).mean())
N["GunMaxM"] = f1(100 * Gm.util.max())
N["GunPninetyM"] = f1(100 * Gm.util.quantile(0.9))
Vm, Vr = U["MILP"]["vans"], U["RL"]["vans"]
for V_, t in ((Vm, "M"), (Vr, "R")):
    N[f"VanChargeShare{t}"] = f1(100 * (V_.charge_min / V_.on_road_min).mean())
    N[f"VanDriveShare{t}"] = f1(100 * (V_.drive_min / V_.on_road_min).mean())
    N[f"VanWaitShare{t}"] = f1(100 * (V_.idle_min / V_.on_road_min).mean())
    N[f"VansFull{t}"] = f1(100 * (V_.stops == V_.Q).mean())
    N[f"NVans{t}"] = len(V_)
N["VanLossShare"] = f1(100 * ((Vm.kWh_drawn - Vm.kWh_delivered - 0.25 * Vm.km).sum()) / Vm.kWh_drawn.sum())
N["VanDelivShare"] = f1(100 * Vm.kWh_delivered.sum() / Vm.kWh_drawn.sum())
N["VanDrvEShare"] = f1(100 * (0.25 * Vm.km).sum() / Vm.kWh_drawn.sum())
# figure: by n (left) and van time split (right)
fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.4))
ax = axs[0]
for meth, u, ls, mf in (("MILP", U["MILP"], "-o", None), ("RL", U["RL"], "--o", "white")):
    I, V = u["instances"], u["vans"]
    gI, gV = I.groupby("n"), V.groupby("n")
    ax.plot(sizes, 100 * gI.gun_util.mean().reindex(sizes), ls, color=ACCA, ms=3, markerfacecolor=mf or ACCA,
            lw=1.2, label=f"gun busy, {meth}")
    ax.plot(sizes, 100 * gI.fleet_use.mean().reindex(sizes), ls, color=ACCP, ms=3, markerfacecolor=mf or ACCP,
            lw=1.2, label=f"vans sent, {meth}")
    ax.plot(sizes, 100 * gV.seat_util.mean().reindex(sizes), ls, color=MID, ms=2.5, markerfacecolor=mf or MID,
            lw=1.0, label=f"van seats, {meth}")
ax.set_xlabel("EVs per instance $n$", fontsize=7)
ax.set_ylabel("%", fontsize=7)
ax.set_xticks([10, 30, 50, 70, 100])
ax.tick_params(labelsize=7)
ax.legend(fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
ax = axs[1]
cats = [("charging", "charge_min", ACCP), ("driving", "drive_min", ACCA), ("waiting", "idle_min", LIGHT)]
for k, (meth, V_) in enumerate((("MILP", Vm), ("RL", Vr))):
    left = 0
    for lab, col, c in cats:
        val = 100 * (V_[col] / V_.on_road_min).mean()
        ax.barh(k, val, left=left, color=c, edgecolor="white", label=lab if k == 0 else None)
        ax.text(left + val / 2, k, f"{val:.0f}", ha="center", va="center", fontsize=6.5,
                color="white" if c != LIGHT else DARK)
        left += val
ax.set_yticks([0, 1])
ax.set_yticklabels(["MILP", "RL"], fontsize=7)
ax.invert_yaxis()
ax.set_xlabel("share of a van's road time (%)", fontsize=7)
ax.tick_params(labelsize=7)
ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3)
save(fig, "infra_n.png")

# =====================================================================================================
# E9 (station capacity) and E10 (charging demand): RL ONLY (no MILP run exists); retrained RL vs RL baseline
# on instances valid in both. Capacities follow the variant (cfg_over); pressure labels stay the baseline's.
# =====================================================================================================
RLV = [("E9_station_capacity", "E9_guns_1", "1 gun / station", {"guns_per_station": 1}),
       ("E9_station_capacity", "E9_guns_4", "4 guns / station", {"guns_per_station": 4}),
       ("E9_station_capacity", "E9_power_11", "11 kW / gun", {"station_power_kW": 11}),
       ("E9_station_capacity", "E9_power_44", "44 kW / gun", {"station_power_kW": 44}),
       ("E10_charging_demand", "E10_demand_0.75", r"energy $\times$0.75", {"energy_scale": 0.75}),
       ("E10_charging_demand", "E10_demand_1.25", r"energy $\times$1.25", {"energy_scale": 1.25})]
rl_base_plan = base_plan["RL"][base_plan["RL"].run_id == "rl_f4"]
rl_keys_all = set(zip(r[r.ok].instance, r[r.ok].n))
UBR = infra.utilisation(rl_base_plan[np.array([k in rl_keys_all for k in zip(rl_base_plan.instance,
                                                                             rl_base_plan.n)])])


def util_full(u, keys=None):
    I, V = u["instances"], u["vans"]
    if keys is not None:
        I = I[np.array([k in keys for k in zip(I.instance, I.n)])]
        V = V[np.array([k in keys for k in zip(V.instance, V.n)])]
    return {"gun": 100 * I.gun_util.mean(), "idle": I.idle_guns.mean(), "G": int(I.G.iloc[0]),
            "fleet": 100 * I.fleet_use.mean(), "seats": 100 * V.seat_util.mean(),
            "vanE": 100 * (I.kWh_van / I.kWh_total).mean(), "kwh": I.kWh_total.mean()}


rows_cost, rows_util, rows_press, fig_rows = [], [], [], []
for E, v, lab, cfg in RLV:
    c = pd.read_csv(F4J(f"{E}_RL/results/comparison/comparison.csv"))
    c = c[(c.variant == v) & (c.run_id == "rl_f4_retrained")].iloc[0]
    rv = pd.read_csv(F4J(f"{E}_RL/results/{v}/runs.csv"), low_memory=False)
    rv = rv[(rv.run_id == "rl_f4_retrained") & rv.valid.eq("yes")]
    keys = set(zip(rv.instance, rv.n)) & rl_keys_all
    pv = pd.read_csv(F4J(f"{E}_RL/results/{v}/plan.csv"), low_memory=False)
    pv = pv[(pv.run_id == "rl_f4_retrained") & np.array([k in keys for k in zip(pv.instance, pv.n)])]
    uv, ub = util_full(infra.utilisation(pv, cfg)), util_full(UBR, keys)
    dvp = c["share_van_personal_%"] - c["baseline_share_van_personal_%"]
    dvc = c["share_van_commercial_%"] - c["baseline_share_van_commercial_%"]
    rows_cost.append(f"{lab} & {int(c.valid)} ({int(c.newly_valid)}/{int(c.no_longer_valid)}) & "
                     f"{sgn(c['mean_dZ_%'], 2)} & {sgn(c['mean_d_operator_cost_%'], 2)} & "
                     f"{sgn(c['mean_d_customer_cost_%'], 2)} & {sgn(dvp)} & {sgn(dvc)}")
    rows_util.append(f"{lab} & {uv['G']} & {uv['gun']:.1f} ({sgn(uv['gun'] - ub['gun'])}) & "
                     f"{uv['idle']:.1f} ({sgn(uv['idle'] - ub['idle'])}) & {sgn(uv['fleet'] - ub['fleet'])} & "
                     f"{sgn(uv['seats'] - ub['seats'])} & {sgn(uv['vanE'] - ub['vanE'])}")
    Ev = ev_frame(pv[pv.type == "ev"])
    pb = rl_base_plan[np.array([k in keys for k in zip(rl_base_plan.instance, rl_base_plan.n)])]
    Eb = ev_frame(pb[pb.type == "ev"])
    d = {}
    for t in ("low", "high"):
        for seg in SEGS:
            d[f"{seg}_{t}"] = 100 * (Ev[Ev.free & (Ev.terc == t) & (Ev.segment == seg)].van.mean()
                                     - Eb[Eb.free & (Eb.terc == t) & (Eb.segment == seg)].van.mean())
    qa = Ev[(Ev.resource_type == "FCS") & (Ev.terc == "high")].queue_min.mean()
    qb = Eb[(Eb.resource_type == "FCS") & (Eb.terc == "high")].queue_min.mean()
    rows_press.append(f"{lab} & {sgn(d['personal_low'])} & {sgn(d['personal_high'])} & "
                      f"{sgn(d['commercial_high'])} & {sgn(qa - qb, 2)}")
    fig_rows.append((lab.replace(r"$\times$", "x"), c["mean_dZ_%"], int(c.no_longer_valid), int(c.newly_valid),
                     uv["gun"] - ub["gun"], uv["fleet"] - ub["fleet"], uv["vanE"] - ub["vanE"]))
    tag = {"E9_guns_1": "GunsOne", "E9_guns_4": "GunsFour", "E9_power_11": "PowerEleven",
           "E9_power_44": "PowerFortyFour", "E10_demand_0.75": "DemandLow", "E10_demand_1.25": "DemandHigh"}[v]
    N[f"RLdZ{tag}"] = sgn(c["mean_dZ_%"], 2)
    N[f"RLLost{tag}"] = int(c.no_longer_valid)
    N[f"RLNew{tag}"] = int(c.newly_valid)
    N[f"RLVanP{tag}"] = sgn(dvp)
    N[f"RLGun{tag}"] = f"{uv['gun']:.1f}"
    N[f"RLGunBase{tag}"] = f"{ub['gun']:.1f}"
    N[f"RLQHigh{tag}"] = f"{qa:.2f}"
    N[f"RLQHighBase{tag}"] = f"{qb:.2f}"
N["TabRLOnlyCost"] = " \\\\ ".join(rows_cost) + " \\\\"
N["TabRLOnlyUtil"] = " \\\\ ".join(rows_util) + " \\\\"
N["TabRLOnlyPress"] = " \\\\ ".join(rows_press) + " \\\\"
# figure: change of Z (with lost / gained instances) and of utilisation
fr = pd.DataFrame(fig_rows, columns=["lab", "dZ", "lost", "new", "dgun", "dfleet", "dvanE"])
fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.4), gridspec_kw={"width_ratios": [1, 1.3]})
ax = axs[0]
yy = np.arange(len(fr))
ax.barh(yy, fr.dZ, color=[ACCA if x > 0 else ACCP for x in fr.dZ], height=0.6)
for k, row in fr.iterrows():
    txt = f"-{row.lost}" if row.lost else (f"+{row.new}" if row.new else "")
    ax.text(row.dZ + (1 if row.dZ >= 0 else -1), k, txt, va="center", ha="left" if row.dZ >= 0 else "right",
            fontsize=6, color=DARK)
ax.set_yticks(yy)
ax.set_yticklabels(fr.lab, fontsize=6.5)
ax.invert_yaxis()
ax.axvline(0, color=DARK, lw=0.6)
ax.set_xlim(fr.dZ.min() - 9, fr.dZ.max() + 9)
ax.set_xlabel(r"$\Delta Z$ (%)  [instances lost / gained]", fontsize=6.5)
ax.tick_params(labelsize=6.5)
ax = axs[1]
w = 0.26
for k2, (col, lab2, c2, h) in enumerate((("dgun", "gun busy", ACCA, None), ("dfleet", "vans sent", ACCP, None),
                                          ("dvanE", "van energy share", ACCP_L, "////"))):
    ax.barh(yy + (k2 - 1) * w, fr[col], w, color=c2, edgecolor=ACCP if h else c2, hatch=h, linewidth=0.5,
            label=lab2)
ax.set_yticks(yy)
ax.set_yticklabels([])
ax.invert_yaxis()
ax.axvline(0, color=DARK, lw=0.6)
ax.set_xlabel("change vs RL baseline (points)", fontsize=6.5)
ax.tick_params(labelsize=6.5)
ax.legend(fontsize=5.5, loc="lower right")
save(fig, "rl_only.png")

# =====================================================================================================
# presentation.tex extras: RL training curve, toy example (exact Delta Z with formulation-4 coefficients)
# =====================================================================================================
TL = pd.read_csv(F4J("baseline_RL/model/train_log.csv"))
TL = TL[TL.epoch >= 1]
fig, ax = plt.subplots(figsize=(4.6, 2.5))
warm = TL[TL.phase.astype(str).str.contains("imitation")]
ax.axvspan(0.5, warm.epoch.max() + 0.5, color=ACCA_L, alpha=0.8, lw=0, label="imitation warm-up")
ax.plot(TL.epoch, TL.val_ratio_rl_over_greedy, color=ACCP, lw=1.2, label="validation: RL / greedy rule")
ax.axhline(1.0, color=MID, lw=0.7, ls="--")
best = TL.loc[TL.val_ratio_rl_over_greedy.idxmin()]
ax.plot([best.epoch], [best.val_ratio_rl_over_greedy], "o", color=ACCA, ms=4, label="best (saved)")
ax.set_ylim(0.95, 1.2)
ax.set_xlabel("epoch (50 batches each)")
ax.set_ylabel("cost ratio")
ax.legend(fontsize=7, loc="upper right")
save(fig, "training.png")
N["TrainBestRatio"] = f"{best.val_ratio_rl_over_greedy:.3f}"
N["TrainBestGain"] = f1(100 * (1 - best.val_ratio_rl_over_greedy))
N["TrainWarmEpochs"] = int(warm.epoch.max())
N["TrainWarmEndRatio"] = f"{TL[TL.epoch == warm.epoch.max()].val_ratio_rl_over_greedy.iloc[0]:.3f}"
N["TrainInitRatio"] = f"{pd.read_csv(F4J('baseline_RL/model/train_log.csv')).val_ratio_rl_over_greedy.iloc[0]:.2f}"

# toy instance: 1 station (2 guns, 22 kW), 1 van (30 kW), 3 EVs
cEV, pF_, pM_, piF, piM, eta_, cM_, cF_, PF_, PM_, spd = 4, 15, 25, 8.7, 10.7, 0.9, 15, 1500, 22, 30, 15.6


def pers(t):
    return dict(mu=1.5 - t, cD=1.5 + 13.5 * t, cQ=2.5 + 22.5 * t, cC=1 + 9 * t, cV=0.5 + 4.5 * t, W=60 - 45 * t)


def comm(t):
    return dict(mu=1.5 - t, cD=1.5 + 4.5 * t, cQ=2 + 5.5 * t, cC=1.5 + 4.5 * t, cV=1.5 + 4.5 * t, W=40 - 25 * t)


def gun_dz(c, E, d, r, free=0.0):
    t = 60 * d / spd
    s = max(r + t, free)
    w_ = s - r - t
    tau = 60 * E / PF_
    return (c["mu"] * (cEV * d + pF_ * E) + piF * E + c["cD"] * t + c["cC"] * tau + c["cQ"] * w_, s, s + tau, w_)


def van_dz(c, E, dgo, dback, dhome, r, free=0.0, unused=True):
    s = max(r, free + 60 * dgo / spd)
    w_ = s - r
    tau = 60 * E / PM_
    return (c["mu"] * pM_ * E + piM * E / eta_ + c["cV"] * (tau + w_) + cM_ * (dgo + dback - dhome)
            + cF_ * unused, s, s + tau, w_)


e1, e2, e3 = pers(0.9), comm(0.3), pers(0.5)
g11 = gun_dz(e1, 20, 2.0, 0)
v1 = van_dz(e1, 20, 3.0, 3.0, 0, 0)
g2a = gun_dz(e2, 30, 3.0, 0, free=g11[2])          # gun 1 after EV 1
g2b = gun_dz(e2, 30, 3.0, 0)                       # gun 2
v2 = van_dz(e2, 30, 4.0, 4.0, 0, 0)
v3 = van_dz(e3, 15, 2.5, 2.5, 0, 10)
N.update(ToyGunOne=f"{g11[0]:.0f}", ToyVanOne=f"{v1[0]:.0f}", ToyStartOne=f"{g11[1]:.1f}", ToyEndOne=f"{g11[2]:.1f}",
         ToyWaitTwo=f"{g2a[3]:.0f}", ToyWTwo=f"{e2['W']:.1f}", ToyGunTwo=f"{g2b[0]:.0f}", ToyVanTwo=f"{v2[0]:.0f}",
         ToyStartTwo=f"{g2b[1]:.1f}", ToyEndTwo=f"{g2b[2]:.1f}", ToyVanThree=f"{v3[0]:.0f}",
         ToyStartThree=f"{v3[1]:.0f}", ToyEndThree=f"{v3[2]:.0f}",
         ToyTotal=f"{g11[0] + g2b[0] + v3[0]:,.0f}".replace(",", "\\,"), ToyDispatch=f"{cF_}")

with open(os.path.join(HERE, "numbers.tex"), "w", encoding="utf-8") as fh:
    for k, v in sorted(N.items()):
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(json.dumps(N, indent=0, default=str))
