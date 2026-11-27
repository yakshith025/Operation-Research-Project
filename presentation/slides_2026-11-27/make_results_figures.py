"""Per-method results (MILP / RL / combined) and the experiment sensitivities for results_slides.tex.

Reads the finished formulation-4 runs (../../formulation_4) and the shared definitions of
../../formulation_4/report/comprehensive/common/infra.py (utilisation, demand pressure).
Both methods are compared on the same instances: the 474 where MILP and RL both have a valid plan.
Writes figures/res_*.png (300 dpi, transparent, accent colours) and results_numbers.tex.
Usage: python make_results_figures.py
"""
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
DARK = "#333333"
ACCP, ACCA = "#5F4B6B", "#C58A63"
ACCP_L, ACCA_L = "#cfc5d6", "#ecd3c1"
ORANGE, PURPLE = "#F07F3C", "#B84FC2"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": DARK, "axes.labelcolor": DARK, "xtick.color": DARK, "ytick.color": DARK,
                     "legend.frameon": False, "savefig.transparent": True, "savefig.dpi": 300})
N = {}
TERC = ["low", "medium", "high"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


def f1(x):
    return "--" if x != x else f"{x:.1f}"


def f2(x):
    return "--" if x != x else f"{x:.2f}"


# ------------------------------------------------------------------ runs and paired instances
m = pd.read_csv(F4J("baseline_MILP/results/runs.csv"), low_memory=False)
m = m[m.run_id == "milp_300s"]
r = pd.read_csv(F4J("baseline_RL/results/runs.csv"), low_memory=False)
ok = lambda d: set(zip(d[d.valid == "yes"].instance, d[d.valid == "yes"].n))  # noqa: E731
paired = ok(m) & ok(r)
status = dict(zip(zip(m.instance, m.n), m.status))
N["NPairedR"] = len(paired)


def load_plan(path, run):
    p = pd.read_csv(path, low_memory=False)
    p = p[p.run_id == run]
    return p[np.array([k in paired for k in zip(p.instance, p.n)])].copy()


PL = {"M": load_plan(F4J("baseline_MILP/results/plan.csv"), "milp_300s"),
      "R": load_plan(F4J("baseline_RL/results/plan.csv"), "rl_f4")}
PT = infra.pressure_table()[["instance", "n", "id", "pressure", "terc"]]
EV = {}
for k, p in PL.items():
    e = p[p.type == "ev"].merge(PT, on=["instance", "n", "id"], how="left")
    e["van"] = e.resource_type != "FCS"
    e["free"] = e.stranded == "no"
    e["band"] = np.clip((e.theta // 0.2).astype(int), 0, 4)
    e["decile"] = np.clip((e.theta // 0.1).astype(int), 0, 9)
    e["opt"] = [status.get(key) == "optimal" for key in zip(e.instance, e.n)]
    # share of the plan's guns charging at the EV's request time
    busy = []
    for (inst, n), g in e.groupby(["instance", "n"], sort=False):
        fcs = g[g.resource_type == "FCS"]
        s, t = fcs.start_min.to_numpy(), fcs.end_min.to_numpy()
        rq = (g.start_min - g.van_wait_min.fillna(0) - g.queue_min.fillna(0) - g.drive_min.fillna(0)).to_numpy()
        busy.append(pd.Series(((s[None, :] <= rq[:, None]) & (t[None, :] > rq[:, None])).sum(1) / 24.0, index=g.index))
    e["busy_at_req"] = pd.concat(busy)
    EV[k] = e
U = {k: infra.utilisation(p) for k, p in PL.items()}

# ------------------------------------------------------------------ per-method numbers
for k in ("M", "R"):
    e, u = EV[k], U[k]
    I, V, G = u["instances"], u["vans"], u["guns"]
    fc = e[e.free]
    st = e[e.resource_type == "FCS"]
    short = I.family.isin(["C1", "R1", "RC1"])
    N[f"Gun{k}"] = f1(100 * I.gun_util.mean())
    N[f"GunShort{k}"] = f1(100 * I[short].gun_util.mean())
    N[f"GunLong{k}"] = f1(100 * I[~short].gun_util.mean())
    N[f"GunMax{k}"] = f1(100 * G.util.max())
    N[f"GunNever{k}"] = f1(100 * (G.evs == 0).mean())
    N[f"IdleGuns{k}"] = f1(I.idle_guns.mean())
    N[f"Fleet{k}"] = f1(100 * I.fleet_use.mean())
    N[f"Seats{k}"] = f1(100 * V.seat_util.mean())
    N[f"Full{k}"] = f1(100 * (V.stops == V.Q).mean())
    N[f"Batt{k}"] = f1(100 * V.battery_util.mean())
    N[f"Road{k}"] = f1(100 * V.on_road_util.mean())
    N[f"VanCharge{k}"] = f1(100 * (V.charge_min / V.on_road_min).mean())
    N[f"VanWait{k}"] = f1(100 * (V.idle_min / V.on_road_min).mean())
    N[f"VanEnergy{k}"] = f1(100 * (I.kWh_van / I.kWh_total).mean())
    N[f"VansShortSeats{k}"] = f1(100 * V[V.family.isin(['R1', 'RC1'])].seat_util.mean())
    N[f"VansLongSeats{k}"] = f1(100 * V[V.family.isin(['C2', 'R2', 'RC2'])].seat_util.mean())
    N[f"VansShortRoad{k}"] = f1(100 * V[V.family.isin(['R1', 'RC1'])].on_road_util.mean())
    N[f"Vans{k}"] = int(len(V))
    # income / segment
    for seg, tag in (("personal", "Pers"), ("commercial", "Com")):
        q = fc[fc.segment == seg]
        N[f"Van{tag}{k}"] = f1(100 * q.van.mean())
    p = fc[fc.segment == "personal"]
    N[f"VanPersBelowFour{k}"] = f1(100 * p[p.theta < 0.4].van.mean())
    N[f"VanPersTop{k}"] = f1(100 * p[p.theta >= 0.9].van.mean())
    N[f"VanPersSixEight{k}"] = f1(100 * p[(p.theta >= 0.6) & (p.theta < 0.8)].van.mean())
    N[f"ThetaVan{k}"] = f2(fc[fc.van & (fc.segment == 'personal')].theta.mean())
    N[f"ThetaFcs{k}"] = f2(fc[~fc.van].theta.mean())
    qtot = st.queue_min.sum()
    N[f"QShareCom{k}"] = f1(100 * st[st.segment == "commercial"].queue_min.sum() / qtot)
    N[f"QUserCom{k}"] = f1(100 * (st.segment == "commercial").mean())
    N[f"QSharePoor{k}"] = f1(100 * st[st.theta < 0.4].queue_min.sum() / qtot)
    N[f"QUserPoor{k}"] = f1(100 * (st.theta < 0.4).mean())
    N[f"DriveRich{k}"] = f2(st[st.theta >= 0.8].drive_km.mean())
    N[f"DrivePoor{k}"] = f2(st[st.theta < 0.2].drive_km.mean())
    ct = e[e.segment == "commercial"]
    N[f"ComVan{k}"] = int((ct.free & ct.van).sum())
    cv = ct[ct.free & ct.van]
    N[f"ComVanTL{k}"] = f1(100 * (~cv.opt).mean()) if len(cv) else "--"
    N[f"ComVanHigh{k}"] = f1(100 * (cv.terc == "high").mean()) if len(cv) else "--"
    N[f"ComVanWait{k}"] = f1(cv.van_wait_min.mean()) if len(cv) else "--"
    # pressure
    for t in TERC:
        a = fc[fc.terc == t]
        s_ = st[st.terc == t]
        T_ = t.capitalize()[:3]
        N[f"VanPers{T_}{k}"] = f1(100 * a[a.segment == "personal"].van.mean())
        N[f"VanCom{T_}{k}"] = f1(100 * a[a.segment == "commercial"].van.mean())
        N[f"Queue{T_}{k}"] = f2(s_.queue_min.mean())
        N[f"QueueCom{T_}{k}"] = f2(s_[s_.segment == "commercial"].queue_min.mean())
        N[f"QueuePers{T_}{k}"] = f2(s_[s_.segment == "personal"].queue_min.mean())
        N[f"Queued{T_}{k}"] = f1(100 * (s_.queue_min > 1e-6).mean())
        N[f"Busy{T_}{k}"] = f1(100 * a.busy_at_req.mean())
        N[f"Share{T_}{k}"] = f1(100 * (fc.terc == t).mean())
    hq = st[st.terc == "high"]
    N[f"QueueHiPoor{k}"] = f2(hq[hq.theta < 0.4].queue_min.mean())
    N[f"QueueHiRich{k}"] = f2(hq[hq.theta >= 0.8].queue_min.mean())
q1, q2 = infra.pressure_cuts()
N["CutLo"], N["CutHi"] = f2(q1), f2(q2)

# ------------------------------------------------------------------ figures per method
LAB = {"M": "MILP", "R": "RL"}
STYLE = {"M": "-", "R": "--"}


def infra_fig(k):
    prof = U[k]["profile"]
    fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.5), sharey=True)
    for ax, (fams, title) in zip(axs, ((["C1", "R1", "RC1"], "short horizon"), (["C2", "R2", "RC2"], "long horizon"))):
        q = prof[prof.family.isin(fams)].groupby("t_rel")[["guns_busy", "vans_busy"]].mean()
        ax.fill_between(q.index, 100 * q.guns_busy, color=ACCA, alpha=0.55, step="mid", label="guns charging")
        ax.step(q.index, 100 * q.vans_busy, where="mid", color=ACCP, lw=1.5, label="vans charging")
        ax.set_title(title, fontsize=8, color=DARK)
        ax.set_xlabel("time $t/T$")
    axs[0].set_ylabel("share charging (%)")
    axs[0].legend(fontsize=7, loc="upper right")
    save(fig, f"res_infra_{k}.png")


def income_fig(k):
    e = EV[k]
    fc = e[e.free]
    st = e[e.resource_type == "FCS"]
    fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.5))
    mids = np.arange(10) / 10 + 0.05
    ax = axs[0]
    for seg, col in (("personal", ACCP), ("commercial", ACCA)):
        q = fc[fc.segment == seg]
        y = [100 * q[q.decile == d].van.mean() if (q.decile == d).sum() >= 20 else np.nan for d in range(10)]
        ax.plot(mids, y, "-o", color=col, ms=3, lw=1.4, label=seg)
    if k == "M":
        q = fc[(fc.segment == "personal") & fc.opt]
        y = [100 * q[q.decile == d].van.mean() if (q.decile == d).sum() >= 20 else np.nan for d in range(10)]
        ax.plot(mids, y, ":", color=ACCP, lw=1.2, label="personal, optimal plans")
        ax.axvline(0.66, color=PURPLE, lw=0.8, ls="--")
        ax.text(0.67, 2, r"$\theta^*$", color=PURPLE, fontsize=8)
    ax.set_xlabel(r"income $\theta$")
    ax.set_ylabel("free-choice EVs by van (%)")
    ax.set_title("who gets a van", fontsize=8, color=DARK)
    ax.legend(fontsize=6.5, loc="upper left")
    ax = axs[1]
    bands = range(5)
    tot_q, tot_u = st.queue_min.sum(), len(st)
    users = [100 * (st.band == b).sum() / tot_u for b in bands]
    queue = [100 * st[st.band == b].queue_min.sum() / tot_q for b in bands]
    x = np.arange(5)
    ax.bar(x - 0.2, users, 0.38, color=ACCP_L, edgecolor=ACCP, lw=0.6, label="share of station users")
    ax.bar(x + 0.2, queue, 0.38, color=ACCA, label="share of queue minutes")
    ax.set_xticks(x)
    ax.set_xticklabels([".0-.2", ".2-.4", ".4-.6", ".6-.8", ".8-1"], fontsize=7)
    ax.set_xlabel(r"income band $\theta$")
    ax.set_ylabel("%")
    ax.set_title("who queues", fontsize=8, color=DARK)
    ax.set_ylim(0, 1.4 * max(max(users), max(queue)))      # headroom so the legend sits above the bars
    ax.legend(fontsize=6.5, loc="upper right")
    save(fig, f"res_income_{k}.png")


def pressure_fig(k):
    e = EV[k]
    fc = e[e.free]
    st = e[e.resource_type == "FCS"]
    x = np.arange(3)
    fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.5))
    ax = axs[0]
    for j, (seg, col) in enumerate((("personal", ACCP), ("commercial", ACCA))):
        y = [100 * fc[(fc.terc == t) & (fc.segment == seg)].van.mean() for t in TERC]
        ax.bar(x + (j - 0.5) * 0.38, y, 0.38, color=col, label=seg)
    ax.set_xticks(x)
    ax.set_xticklabels(TERC)
    ax.set_ylabel("free-choice EVs by van (%)")
    ax.set_title("van share by pressure", fontsize=8, color=DARK)
    ax.legend(fontsize=7)
    ax = axs[1]
    for j, (seg, col) in enumerate((("personal", ACCP), ("commercial", ACCA))):
        y = [st[(st.terc == t) & (st.segment == seg)].queue_min.mean() for t in TERC]
        ax.bar(x + (j - 0.5) * 0.38, y, 0.38, color=col, label=seg)
    ax.set_xticks(x)
    ax.set_xticklabels(TERC)
    ax.set_ylabel("mean station queue (min)")
    ax.set_title("queue by pressure", fontsize=8, color=DARK)
    save(fig, f"res_pressure_{k}.png")


for k in ("M", "R"):
    infra_fig(k)
    income_fig(k)
    pressure_fig(k)

# ------------------------------------------------------------------ combined
j = r[r.valid == "yes"].merge(m[m.valid == "yes"][["instance", "n", "Z_total", "status"]], on=["instance", "n"],
                              suffixes=("", "_m"))
j["gap"] = 100 * (j.Z_total / j.Z_total_m - 1)
sizes = list(range(10, 101, 10))
fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.5))
ax = axs[0]
bp = ax.boxplot([j[j.n == n].gap for n in sizes], positions=sizes, widths=6, patch_artist=True, showfliers=False)
for b in bp["boxes"]:
    b.set(facecolor=ACCA_L, edgecolor=ACCA, linewidth=0.8)
for kk in ("whiskers", "caps"):
    for x_ in bp[kk]:
        x_.set(color=ACCA, linewidth=0.7)
for x_ in bp["medians"]:
    x_.set(color=ACCP, linewidth=1.4)
ax.axhline(0, color=DARK, lw=0.6, ls="--")
ax.set_xticks(sizes)
ax.set_xticklabels(sizes, fontsize=7)
ax.set_xlabel("EVs per instance $n$")
ax.set_ylabel("RL cost above MILP (%)")
ax = axs[1]
tm = m.groupby("n").solve_s.mean().reindex(sizes)
tr = r.groupby("n").solve_s.mean().reindex(sizes)
ax.semilogy(sizes, tm, "-o", color=ACCP, ms=3, label="MILP")
ax.semilogy(sizes, tr, "--s", color=ACCA, ms=3, label="RL")
ax.set_xticks(sizes)
ax.set_xticklabels(sizes, fontsize=7)
ax.set_xlabel("EVs per instance $n$")
ax.set_ylabel("time per instance (s)")
ax.legend(fontsize=7)
save(fig, "res_combined.png")
fm, fr = EV["M"], EV["R"]
both = fm[["instance", "n", "id", "van", "free"]].merge(fr[["instance", "n", "id", "van"]], on=["instance", "n", "id"],
                                                        suffixes=("_m", "_r"))
b2 = both[both.free]
N["Agree"] = f1(100 * (b2.van_m == b2.van_r).mean())
N["BothVan"] = f1(100 * (b2.van_m & b2.van_r).sum() / max(1, b2.van_m.sum()))
N.update(GapMean=f2(j.gap.mean()), GapMed=f2(j.gap.median()), GapOpt=f2(j[j.status_m == "optimal"].gap.mean()),
         GapHundred=f2(j[j.n == 100].gap.mean()), RlBetter=int((j.gap < -1e-6).sum()),
         RlEqual=int((j.gap.abs() <= 1e-6).sum()), TimeM=f1(m.solve_s.mean()), TimeR=f2(r.solve_s.mean()),
         TimeMHundred=f1(tm.loc[100]), TimeRHundred=f2(tr.loc[100]),
         NValidM=int((m.valid == "yes").sum()), NValidR=int((r.valid == "yes").sum()),
         RlOnly=len(ok(r) - ok(m)))

# ------------------------------------------------------------------ experiments and sensitivities
LABELS = {"E1_pM_15": r"van tariff $p^M$ 15", "E1_pM_35": r"van tariff $p^M$ 35",
          "E2_vans_3": "3 vans", "E2_vans_7": "7 vans", "E2_seats_4": "4 seats/van", "E2_seats_8": "8 seats/van",
          "E5_cC1_personal_7.5": r"pers. $c^{C,1}$ 7.5", "E5_cC1_personal_12.5": r"pers. $c^{C,1}$ 12.5",
          "E5_omega1_commercial_4.5": r"taxi $\omega^1$ 4.5", "E5_omega1_commercial_7.5": r"taxi $\omega^1$ 7.5",
          "E6_mu0_personal_1.125": r"pers. $\mu^0$ 1.125", "E6_mu0_personal_1.875": r"pers. $\mu^0$ 1.875",
          "E6_mu0_commercial_1.125": r"taxi $\mu^0$ 1.125", "E6_mu0_commercial_1.875": r"taxi $\mu^0$ 1.875",
          "E8_battery_200": "battery 200 kWh", "E9_guns_1": "1 gun/station", "E9_guns_4": "4 guns/station",
          "E9_power_11": "11 kW/gun", "E9_power_44": "44 kW/gun", "E10_demand_0.75": r"energy $\times$0.75",
          "E10_demand_1.25": r"energy $\times$1.25"}
rows = []
for E in ("E1_van_tariff", "E2_van_fleet_and_seats", "E5_value_of_charging_time", "E6_money_sensitivity",
          "E8_van_battery", "E9_station_capacity", "E10_charging_demand"):
    pm = F4J(f"{E}_MILP/results/comparison/comparison.csv")
    cm = pd.read_csv(pm) if os.path.exists(pm) else None
    cr = pd.read_csv(F4J(f"{E}_RL/results/comparison/comparison.csv"))
    cr = cr[cr.run_id == "rl_f4_retrained"]
    for _, b in cr.iterrows():
        a = cm[cm.variant == b.variant].iloc[0] if cm is not None and (cm.variant == b.variant).any() else None
        rows.append({"variant": b.variant, "exp": E.split("_")[0],
                     "dZ_m": a["mean_dZ_%"] if a is not None else np.nan, "dZ_r": b["mean_dZ_%"],
                     "lost_m": a.no_longer_valid if a is not None else np.nan, "lost_r": b.no_longer_valid,
                     "new_m": a.newly_valid if a is not None else np.nan, "new_r": b.newly_valid,
                     "vp_m": (a["share_van_personal_%"] - a["baseline_share_van_personal_%"]) if a is not None else np.nan,
                     "vp_r": b["share_van_personal_%"] - b["baseline_share_van_personal_%"]})
ex = pd.DataFrame(rows)
ex["label"] = ex.variant.map(LABELS)
ex["key"] = ex[["dZ_m", "dZ_r"]].abs().max(axis=1)
ex = ex.sort_values("key", ascending=False).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(4.0, 4.2))
y = np.arange(len(ex))
ax.barh(y - 0.2, ex.dZ_m.fillna(0), 0.38, color=ACCP, label="MILP")
ax.barh(y + 0.2, ex.dZ_r, 0.38, color=ACCA_L, edgecolor=ACCA, hatch="////", linewidth=0.5, label="RL (retrained)")
ax.set_yticks(y)
ax.set_yticklabels([lab + (" (RL only)" if np.isnan(dm) else "") for lab, dm in zip(ex.label, ex.dZ_m)], fontsize=8)
ax.tick_params(axis="x", labelsize=8)
ax.invert_yaxis()
ax.axvline(0, color=DARK, lw=0.6)
ax.set_xlabel("mean change of $Z$ vs baseline (%)")
ax.legend(fontsize=7, loc="lower right")
save(fig, "res_experiments.png")
ex.to_csv(os.path.join(FIG, "res_experiments.csv"), index=False)
g = ex.set_index("variant")


def sgn(v, d=1):
    return "--" if v != v else (f"{v:+.{d}f}").replace("-", "$-$")


for v, name in (("E6_mu0_commercial_1.125", "TaxiMuLo"), ("E6_mu0_commercial_1.875", "TaxiMuHi"),
                ("E1_pM_15", "TariffLo"), ("E1_pM_35", "TariffHi"), ("E2_vans_3", "VansThree"),
                ("E2_vans_7", "VansSeven"), ("E2_seats_4", "SeatsFour"), ("E8_battery_200", "Battery"),
                ("E5_cC1_personal_12.5", "PersCHi"), ("E5_omega1_commercial_7.5", "OmegaHi"),
                ("E9_guns_1", "GunsOne"), ("E9_guns_4", "GunsFour"), ("E9_power_11", "PowerLo"),
                ("E9_power_44", "PowerHi"), ("E10_demand_0.75", "DemandLo"), ("E10_demand_1.25", "DemandHi")):
    row = g.loc[v]
    N[f"dZm{name}"] = sgn(row.dZ_m, 2)
    N[f"dZr{name}"] = sgn(row.dZ_r, 2)
    N[f"dVPm{name}"] = sgn(row.vp_m)
    N[f"dVPr{name}"] = sgn(row.vp_r)
    N[f"Lostr{name}"] = int(row.lost_r)
    N[f"Newr{name}"] = int(row.new_r)
    N[f"Lostm{name}"] = "--" if row.lost_m != row.lost_m else int(row.lost_m)
    N[f"Newm{name}"] = "--" if row.new_m != row.new_m else int(row.new_m)
both_ = ex.dropna(subset=["dZ_m"])
N["DirAgree"] = int((np.sign(both_.dZ_m) == np.sign(both_.dZ_r)).sum())
N["DirTotal"] = len(both_)
N["DirMaxDiff"] = f2((both_.dZ_m - both_.dZ_r).abs().max())

with open(os.path.join(HERE, "results_numbers.tex"), "w", encoding="utf-8") as fh:
    for k_, v_ in sorted(N.items()):
        fh.write(f"\\newcommand{{\\r{k_}}}{{{v_}}}\n")
print(len(N), "macros")
for k_ in sorted(N):
    print(k_, N[k_])
