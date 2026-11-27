"""Peak / off-peak (demand pressure) part of make_results_report.py; executed inside its namespace.

Per EV (baseline data, 560 instances), as formulation 4's f4_data.peak_flags:
  load_i     = requests released in [r_i - mean FCS charging time of the instance, r_i]
  period_i   = peak if load_i > median load of the instance, else off-peak
  pressure_i = load_i / (guns + vans) = load_i / 29
Pressure terciles (low / medium / high) are pooled over the free-choice EVs of all 560 instances. The labels are
those of the baseline data and are kept for every variant (in E2 the number of vans changes the denominator; keeping
the baseline labels compares the same EVs).
"""
import infra  # noqa: E402  (../common/infra.py: the definition shared by the three reports)

print("peak / off-peak ...")
PK = infra.pressure_table().copy()
PK["key"] = PK["instance"] + "_n" + PK["n"].astype(int).astype(str)
PK["free"] = ~PK["stranded"].astype(bool)
q1, q2 = infra.pressure_cuts()
N["PressLowCut"], N["PressHighCut"] = f"{q1:.2f}", f"{q2:.2f}"
N["PressWindowMean"] = f"{PK.drop_duplicates('key').window_min.mean():.0f}"
N["PressWindowMin"] = f"{PK.window_min.min():.0f}"
N["PressWindowMax"] = f"{PK.window_min.max():.0f}"
N["PressEVs"] = f"{int(PK.free.sum()):,}"
N["PeakWindowNote"] = "mean FCS charging time"
TERC = ["low", "medium", "high"]
TC = {"low": "Low", "medium": "Mid", "high": "High"}


def labelled(p, keys):
    e = ev_rows(p, keys)
    e = e[e.stranded == "no"][["key", "id", "segment", "theta", "resource_type", "queue_min"]]
    return e.merge(PK[["key", "id", "period", "pressure", "terc"]], on=["key", "id"], how="left")


def share(e):
    return 100 * (e.resource_type == "MCS").mean() if len(e) else np.nan


def fqueue(e):
    f_ = e[e.resource_type == "FCS"]
    return f_.queue_min.mean() if len(f_) else np.nan


LM, LR = labelled(bm_p, kB), labelled(br_p, kB)

# (1) baseline: tercile x segment
rows = []
for s in rc.SEGMENTS:
    for t in TERC:
        a, b = LM[(LM.segment == s) & (LM.terc == t)], LR[(LR.segment == s) & (LR.terc == t)]
        qa = a[a.resource_type == "FCS"]
        rows.append([s if t == "low" else "", t, len(a), pc(share(a), 1), pc(share(b), 1), pc(fqueue(a), 2),
                     pc(fqueue(b), 2), pc(100 * (qa.queue_min > 1e-6).mean(), 1)])
        sk = s.capitalize()
        N[f"PVan{sk}{TC[t]}Milp"] = pc(share(a), 1)
        N[f"PVan{sk}{TC[t]}Rl"] = pc(share(b), 1)
        N[f"PQueue{sk}{TC[t]}Milp"] = pc(fqueue(a), 2)
        N[f"PQueue{sk}{TC[t]}Rl"] = pc(fqueue(b), 2)
rc.latex_table(os.path.join(TAB, "pressure_methods.tex"),
               ["segment", "pressure", "\\shortstack{free-choice\\\\EVs}", "\\shortstack{MILP\\\\van (\\%)}",
                "\\shortstack{RL\\\\van (\\%)}", "\\shortstack{MILP station\\\\queue (min)}",
                "\\shortstack{RL station\\\\queue (min)}", "\\shortstack{MILP station users\\\\queueing (\\%)}"], rows,
               caption="Baseline, paired instances: van share and station queue of free-choice EVs by demand-pressure "
               "tercile and segment, MILP against RL.", label="tab:pressm", size="small")
for meth, L in (("Milp", LM), ("Rl", LR)):
    for s in rc.SEGMENTS:
        for per, nm_ in (("peak", "Peak"), ("off-peak", "Off")):
            N[f"P{nm_}Van{s.capitalize()}{meth}"] = pc(share(L[(L.segment == s) & (L.period == per)]), 1)

# tercile x segment x band (MILP and RL)
rows = []
for s in rc.SEGMENTS:
    for t in TERC:
        row = [s if t == "low" else "", t]
        for b in range(5):
            a = LM[(LM.segment == s) & (LM.terc == t) & (rc.band_of(LM.theta) == b)]
            r_ = LR[(LR.segment == s) & (LR.terc == t) & (rc.band_of(LR.theta) == b)]
            row.append(f"{pc(share(a), 1)} / {pc(share(r_), 1)}")
        rows.append(row)
rc.latex_table(os.path.join(TAB, "pressure_band.tex"), ["segment", "pressure"] + [f"$\\theta$ {b}" for b in rc.BLAB],
               rows, align="llccccc",
               caption="Baseline: van share of free-choice EVs (\\%, MILP / RL) by segment, pressure tercile and income "
               "band (paired instances).", label="tab:pressband", size="scriptsize")
rows = []
for s in rc.SEGMENTS:
    for t in TERC:
        row = [s if t == "low" else "", t]
        for b in range(5):
            a = LM[(LM.segment == s) & (LM.terc == t) & (rc.band_of(LM.theta) == b)]
            r_ = LR[(LR.segment == s) & (LR.terc == t) & (rc.band_of(LR.theta) == b)]
            row.append(f"{pc(fqueue(a), 1)} / {pc(fqueue(r_), 1)}")
        rows.append(row)
rc.latex_table(os.path.join(TAB, "pressure_band_queue.tex"),
               ["segment", "pressure"] + [f"$\\theta$ {b}" for b in rc.BLAB], rows, align="llccccc",
               caption="Baseline: mean station queue of free-choice station users (min, MILP / RL) by segment, "
               "pressure tercile and income band.", label="tab:pressbandq", size="scriptsize")
# poorest commercial at high pressure: queue MILP
a = LM[(LM.segment == "commercial") & (LM.terc == "high") & (rc.band_of(LM.theta) == 0)]
N["PQueueComHighPoorMilp"] = pc(fqueue(a), 1)
a = LM[(LM.segment == "personal") & (LM.terc == "high") & (rc.band_of(LM.theta) == 4)]
N["PQueuePersHighRichMilp"] = pc(fqueue(a), 2)

fig, ax = plt.subplots(2, 2, figsize=(7.4, 4.8))
x = np.arange(5) * 0.2 + 0.1
tcol = {"low": "#4C72B0", "medium": "#DD8452", "high": "#C44E52"}
for j, s in enumerate(rc.SEGMENTS):
    for t in TERC:
        for meth, L, ls in (("MILP", LM, "-"), ("RL", LR, "--")):
            ys = [share(L[(L.segment == s) & (L.terc == t) & (rc.band_of(L.theta) == b)]) for b in range(5)]
            ax[0, j].plot(x, ys, ls, marker="o" if meth == "MILP" else "s", ms=3, color=tcol[t],
                          label=f"{t}, {meth}")
            ys = [fqueue(L[(L.segment == s) & (L.terc == t) & (rc.band_of(L.theta) == b)]) for b in range(5)]
            ax[1, j].plot(x, ys, ls, marker="o" if meth == "MILP" else "s", ms=3, color=tcol[t])
    ax[0, j].set_title(f"{s} EVs")
    ax[1, j].set_xlabel("income $\\theta$ (band midpoint)")
ax[0, 0].set_ylabel("served by a van (%)")
ax[1, 0].set_ylabel("mean station queue (min)")
ax[0, 0].legend(fontsize=6.5, ncol=2)
rc.savefig(fig, os.path.join(FIG, "pressure_methods.pdf"))

# (2) experiments: change at low vs high pressure
PX = {}
rows = []
for vid, (code, sgn, lab) in VAR.items():
    for m in ("MILP", "RT"):
        r = RES[(vid, m)]
        kp = r["keys"]
        bp_ = bm_p if m == "MILP" else br_p
        L0, L1 = labelled(bp_, kp), labelled(RUNS[(vid, m)][1], kp)
        d = {}
        for t in TERC:
            for s in rc.SEGMENTS:
                a0, a1 = L0[(L0.segment == s) & (L0.terc == t)], L1[(L1.segment == s) & (L1.terc == t)]
                d[(s, t, "van0")], d[(s, t, "van1")] = share(a0), share(a1)
                d[(s, t, "q0")], d[(s, t, "q1")] = fqueue(a0), fqueue(a1)
        PX[(vid, m)] = d
    d = PX[(vid, "MILP")]
    e = PX[(vid, "RT")]
    row = [lab]
    for t in ("low", "high"):
        row += [pc(d[("personal", t, "van1")] - d[("personal", t, "van0")], 1, True),
                pc(d[("commercial", t, "van1")] - d[("commercial", t, "van0")], 2, True),
                pc(d[("commercial", t, "q1")] - d[("commercial", t, "q0")], 2, True)]
    row += [pc(e[("commercial", "high", "van1")] - e[("commercial", "high", "van0")], 2, True)]
    rows.append(row)
    for t in TERC:
        for s in rc.SEGMENTS:
            sk = s.capitalize()
            N[f"PX{code}Van{sk}{TC[t]}"] = pc(d[(s, t, "van1")] - d[(s, t, "van0")], 2, True)
            N[f"PX{code}VanB{sk}{TC[t]}"] = pc(d[(s, t, "van0")], 2)
            N[f"PX{code}VanV{sk}{TC[t]}"] = pc(d[(s, t, "van1")], 2)
            N[f"PX{code}Q{sk}{TC[t]}"] = pc(d[(s, t, "q1")] - d[(s, t, "q0")], 2, True)
            N[f"PX{code}QB{sk}{TC[t]}"] = pc(d[(s, t, "q0")], 2)
        N[f"PX{code}RtVanCom{TC[t]}"] = pc(e[("commercial", t, "van1")] - e[("commercial", t, "van0")], 2, True)
rc.latex_table(os.path.join(TAB, "pressure_experiments.tex"),
               ["variant", "\\shortstack{low: van\\\\personal}", "\\shortstack{low: van\\\\commercial}",
                "\\shortstack{low: queue\\\\commercial}", "\\shortstack{high: van\\\\personal}",
                "\\shortstack{high: van\\\\commercial}", "\\shortstack{high: queue\\\\commercial}",
                "\\shortstack{RL$_{\\mathrm{rt}}$ high: van\\\\commercial}"], rows, align="lrrrrrrr",
               caption="Every variant at low and at high demand pressure (MILP unless stated; changes against the "
               "baseline on the paired instances): van share of free-choice EVs in percentage points and mean station "
               "queue of commercial station users in minutes.", label="tab:pressexp", size="scriptsize")

# explicit test of the prediction "queues at peak push commercial EVs to vans"
# (a) in the baseline: commercial van share rises with pressure; (b) E5 omega up/down moves the commercial van share
# at high pressure in the same direction, and more than at low pressure.
cv = {t: PX[("E5_omega1_commercial_7.5", "MILP")][("commercial", t, "van0")] for t in TERC}
N["PredBaseHolds"] = "yes" if cv["high"] > cv["low"] else "no"
hi_up = PX[("E5_omega1_commercial_7.5", "MILP")]
hi_dn = PX[("E5_omega1_commercial_4.5", "MILP")]
up = hi_up[("commercial", "high", "van1")] - hi_up[("commercial", "high", "van0")]
dn = hi_dn[("commercial", "high", "van1")] - hi_dn[("commercial", "high", "van0")]
N["PredOmegaUpHigh"], N["PredOmegaDnHigh"] = pc(up, 2, True), pc(dn, 2, True)
N["PredOmegaHolds"] = "yes" if (up > 0 and dn < 0) else "no"
# commercial van users: share at high pressure, and their mean income vs commercial station users
cm = LM[LM.segment == "commercial"]
cvan = cm[cm.resource_type == "MCS"]
N["ComVanUsers"] = len(cvan)
N["ComVanUsersHighShare"] = pc(100 * (cvan.terc == "high").mean(), 0)
N["ComVanUsersTheta"] = pc(cvan.theta.mean(), 2)
N["ComFcsUsersTheta"] = pc(cm[cm.resource_type == "FCS"].theta.mean(), 2)
pm = LM[LM.segment == "personal"]
pvan = pm[pm.resource_type == "MCS"]
N["PersVanUsersHighShare"] = pc(100 * (pvan.terc == "high").mean(), 0)

# figure: E5 omega, E1, E2 seats/vans: commercial and personal van share by tercile
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.2))
sel = ["E5_omega1_commercial_4.5", "E5_omega1_commercial_7.5", "E1_pM_15", "E1_pM_35", "E2_vans_3", "E2_seats_4"]
cmap = plt.get_cmap("tab10")
xt = np.arange(3)
for j, s in enumerate(rc.SEGMENTS):
    for k, v in enumerate(sel):
        d = PX[(v, "MILP")]
        ax[j].plot(xt, [d[(s, t, "van1")] - d[(s, t, "van0")] for t in TERC], marker="o", ms=3, color=cmap(k),
                   label=PLAIN[v])
    ax[j].axhline(0, color="grey", lw=0.6)
    ax[j].set_xticks(xt)
    ax[j].set_xticklabels(TERC)
    ax[j].set_xlabel("demand pressure tercile")
    ax[j].set_title(f"{s} EVs (MILP)")
ax[0].set_ylabel("change of van share (points)")
h_, l_ = ax[0].get_legend_handles_labels()
fig.legend(h_, l_, fontsize=7, loc="lower center", ncol=3)
fig.tight_layout(rect=(0, 0.16, 1, 1))
fig.savefig(os.path.join(FIG, "pressure_experiments.pdf"))
plt.close(fig)

# median-split artefact: requests tied at the start-of-period burst have load == median and count as off-peak
N["ComVanOffPeak"] = int((cvan.period == "off-peak").sum())
N["OffHighShare"] = pc(100 * ((LM.period == "off-peak") & (LM.terc == "high")).mean(), 0)

# E5 omega test restricted to instances optimal in both runs (no time-limit noise)
for vid in ("E5_omega1_commercial_4.5", "E5_omega1_commercial_7.5"):
    vr = RUNS[(vid, "MILP")][0]
    ko = sorted(set(vr.index[vr.status == "optimal"]) & set(bm_r.index[bm_r.status == "optimal"]))
    L0, L1 = labelled(bm_p, ko), labelled(RUNS[(vid, "MILP")][1], ko)
    a0 = L0[(L0.segment == "commercial") & (L0.terc == "high")]
    a1 = L1[(L1.segment == "commercial") & (L1.terc == "high")]
    code = VAR[vid][0]
    N[f"PredOpt{code}N"] = len(a0)
    N[f"PredOpt{code}Base"] = pc(share(a0), 2)
    N[f"PredOpt{code}Var"] = pc(share(a1), 2)
    N[f"PredOpt{code}Kb"] = int((a0.resource_type == "MCS").sum())
    N[f"PredOpt{code}Kv"] = int((a1.resource_type == "MCS").sum())
    N[f"PredOpt{code}Qb"] = pc(fqueue(a0), 2)
    N[f"PredOpt{code}Qv"] = pc(fqueue(a1), 2)

# where do commercial van users come from (MILP status of their instance)
stat = bm_r.status
N["ComVanInOptimal"] = int(cvan.key.map(stat).eq("optimal").sum())
N["ComVanInTL"] = int(cvan.key.map(stat).eq("time_limit").sum())
N["ComVanInOptimalAll"] = int((ev_rows(bm_p)[lambda e: (e.segment == "commercial") & (e.stranded == "no")
                                              & (e.resource_type == "MCS")].key.map(stat) == "optimal").sum())
hq = LM[(LM.segment == "commercial") & (LM.terc == "high") & (LM.resource_type == "FCS")]
N["ComHighQueueOpt"] = pc(hq[hq.key.map(stat) == "optimal"].queue_min.mean(), 2)
N["ComHighQueueTL"] = pc(hq[hq.key.map(stat) == "time_limit"].queue_min.mean(), 2)
N["ComHighNOpt"] = int((hq.key.map(stat) == "optimal").sum())
