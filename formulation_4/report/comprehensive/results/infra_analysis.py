"""Infrastructure utilisation part of make_results_report.py; executed inside its namespace (after peak_analysis).

Uses infra.utilisation (../common/infra.py; definitions in its docstring) on the plans of valid instances of every
run, with the variant's config overrides (n_vans, van_max_evs, van_battery_kWh) so that M, Q and B are right.
Part II: baseline MILP against RL on the paired instances. Part III: every variant against its own baseline on its
paired instances, for the MILP and the retrained RL.
"""
print("infrastructure utilisation ...")


def cfg_over_of(vid):
    for e in EXPS.values():
        for v in e["variants"]:
            if v["id"] == vid:
                return {k.split(".", 1)[1]: val for k, val in v["set"].items() if k.startswith("config.")} or None
    return None


def util_of(run, cfg_over=None):
    r, p = run
    keys = valid_keys(r)
    pv = p[p.key.isin(keys)]
    u = infra.utilisation(pv, cfg_over)
    for k_ in ("guns", "vans", "instances", "profile", "stations"):
        if len(u[k_]):
            u[k_]["key"] = u[k_]["instance"] + "_n" + u[k_]["n"].astype(int).astype(str)
    return u


UB = {"MILP": util_of(BASE["MILP"]), "RL": util_of(BASE["RL"])}
UV = {}
for vid in VAR:
    co = cfg_over_of(vid)
    for m in ("MILP", "RT"):
        UV[(vid, m)] = util_of(RUNS[(vid, m)], co)


def summ(u, keys):
    """Utilisation measures of one run on a set of instance keys (all in %, except counts)."""
    ks = set(keys)
    I = u["instances"][u["instances"].key.isin(ks)]
    g = u["guns"][u["guns"].key.isin(ks)]
    v = u["vans"][u["vans"].key.isin(ks)]
    out = dict(
        inst=len(I),
        gun=100 * I.gun_util.mean(),                                                  # mean of per-instance shares
        gun_pooled=100 * (g.busy_min.sum() / (g["T"].sum())) if len(g) else np.nan,  # pooled (labelled as such)
        idle_guns=I.idle_guns.mean(),
        idle_gun_share=100 * (g.evs == 0).mean() if len(g) else np.nan,
        gun_p90=100 * g.util.quantile(0.9) if len(g) else np.nan,
        gun_max=100 * g.util.max() if len(g) else np.nan,
        van_energy=100 * (I.kWh_van / I.kWh_total).mean(),
        van_energy_pooled=100 * I.kWh_van.sum() / I.kWh_total.sum(),
        fleet=100 * I.fleet_use.mean(),
        vans_used=I.vans_used.mean(),
        system=100 * I.system_util.mean(),
        seat=100 * v.seat_util.mean() if len(v) else np.nan,
        battery=100 * v.battery_util.mean() if len(v) else np.nan,
        onroad=100 * v.on_road_util.mean() if len(v) else np.nan,
        charge=100 * v.charge_util.mean() if len(v) else np.nan,
        power=100 * v.power_util.mean() if len(v) else np.nan,
        idle_van=100 * (v.idle_min / v.on_road_min).mean() if len(v) else np.nan,      # mean over vans
        drive_van=100 * (v.drive_min / v.on_road_min).mean() if len(v) else np.nan,
        charge_van=100 * (v.charge_min / v.on_road_min).mean() if len(v) else np.nan,
        idle_van_pooled=100 * v.idle_min.sum() / v.on_road_min.sum() if len(v) else np.nan,
        full=100 * (v.stops >= v.Q).mean() if len(v) else np.nan,
        b90=100 * (v.battery_util > 0.9).mean() if len(v) else np.nan,
        kwh_total=I.kWh_total.mean(),
        drawn_van=v.kWh_drawn.mean() if len(v) else np.nan,
    )
    return out


MEAS = [("gun", "gun utilisation, mean over instances (\\% of $G\\,T$)", 1),
        ("gun_pooled", "gun utilisation, pooled over instances (\\%)", 1),
        ("gun_p90", "90th percentile of gun utilisation (\\%)", 1),
        ("gun_max", "busiest gun (\\%)", 1),
        ("idle_guns", "idle guns per instance (of 24)", 1),
        ("idle_gun_share", "guns never used (\\%)", 1),
        ("van_energy", "energy delivered by vans (\\% of the instance's kWh)", 1),
        ("van_energy_pooled", "energy delivered by vans, pooled (\\% of all kWh)", 1),
        ("system", "installed-capacity utilisation (\\%)", 1),
        ("vans_used", "vans dispatched per instance", 2),
        ("fleet", "fleet use (dispatched / $M$, \\%)", 1),
        ("seat", "seat use of dispatched vans (\\%)", 1),
        ("full", "vans with every seat taken (\\%)", 1),
        ("battery", "battery drawn (\\% of $B$)", 1),
        ("b90", "vans drawing $>$90\\,\\% of $B$ (\\%)", 1),
        ("onroad", "time on the road (\\% of $T$)", 1),
        ("charge", "time charging EVs (\\% of $T$)", 1),
        ("power", "van power capacity used (\\% of $P^{\\mathrm M}T$)", 1),
        ("charge_van", "on-road time spent charging (\\%)", 1),
        ("drive_van", "on-road time spent driving (\\%)", 1),
        ("idle_van", "on-road time idle or waiting (\\%)", 1),
        ("idle_van_pooled", "on-road time idle or waiting, pooled (\\%)", 1)]
MNAME = {"gun": "Gun", "gun_pooled": "GunPooled", "van_energy_pooled": "VanEnergyPooled",
         "idle_van_pooled": "IdleVanPooled", "gun_p90": "GunPninety", "gun_max": "GunMax", "idle_guns": "IdleGuns",
         "idle_gun_share": "IdleGunShare", "van_energy": "VanEnergy", "system": "System", "vans_used": "VansUsed",
         "fleet": "Fleet", "seat": "Seat", "full": "Full", "battery": "Battery", "b90": "Bninety", "onroad": "OnRoad",
         "charge": "Charge", "power": "Power", "charge_van": "ChargeVan", "drive_van": "DriveVan",
         "idle_van": "IdleVan"}

# ---------------------------------------------------------------- Part II: baseline MILP vs RL on paired instances
SM, SR = summ(UB["MILP"], kB), summ(UB["RL"], kB)
rows = []
for k_, lab, d in MEAS:
    rows.append([lab, pc(SM[k_], d), pc(SR[k_], d), pc(SR[k_] - SM[k_], d, True)])
    N[f"U{MNAME[k_]}Milp"] = pc(SM[k_], d)
    N[f"U{MNAME[k_]}Rl"] = pc(SR[k_], d)
rc.latex_table(os.path.join(TAB, "util_methods.tex"), ["measure", "MILP", "RL", "RL $-$ MILP"], rows, align="lrrr",
               caption=f"Infrastructure utilisation on the {len(kB)} paired baseline instances (24 guns of 22\\,kW, "
               "5 vans of 30\\,kW, 6 seats, 150\\,kWh, over $[0,T]$). Means of per-instance (guns, energy) and "
               "per-van shares; rows marked pooled divide sums over instances.", label="tab:utilm", size="small")
N["UKwhPerInst"] = f"{SM['kwh_total']:,.0f}"

rows = []
for f in rc.FAMS:
    kf = [k for k in kB if bm_r.loc[k, "class"] == f]
    a, b = summ(UB["MILP"], kf), summ(UB["RL"], kf)
    rows.append([f, len(kf), f"{pc(a['gun'], 1)} / {pc(b['gun'], 1)}", f"{pc(a['idle_guns'], 1)} / {pc(b['idle_guns'], 1)}",
                 f"{pc(a['van_energy'], 1)} / {pc(b['van_energy'], 1)}", f"{pc(a['fleet'], 0)} / {pc(b['fleet'], 0)}",
                 f"{pc(a['seat'], 0)} / {pc(b['seat'], 0)}", f"{pc(a['battery'], 0)} / {pc(b['battery'], 0)}",
                 f"{pc(a['onroad'], 0)} / {pc(b['onroad'], 0)}", f"{pc(a['system'], 1)} / {pc(b['system'], 1)}"])
    N[f"UGunFam{f.replace('1', 'one').replace('2', 'two')}"] = pc(a["gun"], 1)
    N[f"UOnRoadFam{f.replace('1', 'one').replace('2', 'two')}"] = pc(a["onroad"], 0)
    N[f"USeatFam{f.replace('1', 'one').replace('2', 'two')}"] = pc(a["seat"], 0)
rc.latex_table(os.path.join(TAB, "util_family.tex"),
               ["family", "inst.", "\\shortstack{gun time\\\\(\\%)}", "\\shortstack{idle guns\\\\(of 24)}",
                "\\shortstack{van share\\\\of kWh (\\%)}", "\\shortstack{fleet\\\\use (\\%)}", "\\shortstack{seats\\\\(\\%)}",
                "\\shortstack{battery\\\\(\\%)}", "\\shortstack{on the road\\\\(\\% of $T$)}",
                "\\shortstack{installed\\\\capacity (\\%)}"], rows, align="lrcccccccc",
               caption="Utilisation by family, MILP / RL (paired baseline instances; means of per-instance and "
               "per-van shares).", label="tab:utilfam",
               size="small")

# by n (MILP)
rows = []
for n in NS:
    kn = [k for k in kB if bm_r.loc[k, "n"] == n]
    a, b = summ(UB["MILP"], kn), summ(UB["RL"], kn)
    rows.append([n, len(kn), pc(a["gun"], 1), pc(b["gun"], 1), pc(a["idle_guns"], 1), pc(a["van_energy"], 1),
                 pc(b["van_energy"], 1), pc(a["fleet"], 0), pc(a["system"], 1)])
rc.latex_table(os.path.join(TAB, "util_n.tex"),
               ["$n$", "inst.", "\\shortstack{gun time\\\\MILP (\\%)}", "\\shortstack{gun time\\\\RL (\\%)}",
                "\\shortstack{idle guns\\\\MILP}", "\\shortstack{van kWh\\\\MILP (\\%)}", "\\shortstack{van kWh\\\\RL (\\%)}",
                "\\shortstack{fleet use\\\\MILP (\\%)}", "\\shortstack{installed cap.\\\\MILP (\\%)}"], rows,
               caption="Utilisation by instance size (paired baseline instances; means of per-instance shares).",
               label="tab:utiln", size="small")
a100 = summ(UB["MILP"], [k for k in kB if bm_r.loc[k, "n"] == 100])
N["UGunMilpNhundred"] = pc(a100["gun"], 1)
N["UIdleGunsMilpNhundred"] = pc(a100["idle_guns"], 1)
N["USystemMilpNhundred"] = pc(a100["system"], 1)
a10 = summ(UB["MILP"], [k for k in kB if bm_r.loc[k, "n"] == 10])
N["UIdleGunsMilpNten"] = pc(a10["idle_guns"], 1)

# busy profile
fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.6))
for meth, u, ls in (("MILP", UB["MILP"], "-"), ("RL", UB["RL"], "--")):
    pr = u["profile"][u["profile"].key.isin(set(kB))]
    g_ = pr.groupby("t_rel")[["guns_busy", "vans_busy"]].mean() * 100
    ax[0].plot(g_.index, g_.guns_busy, ls, color=MCOL["MILP" if meth == "MILP" else "RL"], label=meth)
    ax[1].plot(g_.index, g_.vans_busy, ls, color=MCOL["MILP" if meth == "MILP" else "RL"], label=meth)
    if meth == "MILP":
        N["UPeakGunsBusy"] = pc(g_.guns_busy.max(), 1)
        N["UPeakGunsAt"] = pc(g_.guns_busy.idxmax(), 3)
        N["ULateGunsBusy"] = pc(g_[g_.index > 0.75].guns_busy.mean(), 1)
        N["UPeakVansBusy"] = pc(g_.vans_busy.max(), 1)
for fam_set, ls, lab in ((["C1", "R1", "RC1"], "-", "short horizon"), (["C2", "R2", "RC2"], ":", "long horizon")):
    pr = UB["MILP"]["profile"]
    pr = pr[pr.key.isin(set(kB)) & pr.family.isin(fam_set)]
    g_ = pr.groupby("t_rel").guns_busy.mean() * 100
    ax[2].plot(g_.index, g_, ls, color="black", label=lab)
ax[0].set_title("guns charging (% of 24)")
ax[1].set_title("vans charging (% of M)")
ax[2].set_title("guns charging, MILP")
for a_ in ax:
    a_.set_xlabel("$t/T$")
    a_.legend(fontsize=7)
ax[0].set_ylabel("share busy (%)")
rc.savefig(fig, os.path.join(FIG, "util_profile.pdf"))

# distributions
fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.9), gridspec_kw={"width_ratios": [1, 1.3, 1]})
gm = UB["MILP"]["guns"][UB["MILP"]["guns"].key.isin(set(kB))]
gr = UB["RL"]["guns"][UB["RL"]["guns"].key.isin(set(kB))]
ax[0].hist([100 * gm.util, 100 * gr.util], bins=25, color=[MCOL["MILP"], MCOL["RL"]], label=["MILP", "RL"])
ax[0].set_xlabel("gun utilisation (% of T)")
ax[0].set_ylabel("guns")
ax[0].legend(fontsize=7)
vm = UB["MILP"]["vans"][UB["MILP"]["vans"].key.isin(set(kB))]
vr_ = UB["RL"]["vans"][UB["RL"]["vans"].key.isin(set(kB))]
ax[1].boxplot([100 * vm.seat_util, 100 * vr_.seat_util, 100 * vm.battery_util, 100 * vr_.battery_util,
               100 * vm.on_road_util, 100 * vr_.on_road_util], showfliers=False)
ax[1].set_xticks(range(1, 7))
ax[1].set_xticklabels(["seats\nMILP", "seats\nRL", "batt.\nMILP", "batt.\nRL", "road\nMILP", "road\nRL"],
                      fontsize=6.5)
ax[1].set_ylabel("% of capacity")
tot = vm[["charge_min", "drive_min", "idle_min"]].sum()
tot2 = vr_[["charge_min", "drive_min", "idle_min"]].sum()
bottom = np.zeros(2)
for c_, colr, lab in (("charge_min", "#55A868", "charging"), ("drive_min", "#4C72B0", "driving"),
                      ("idle_min", "#BBBBBB", "idle / waiting")):
    vals = np.array([(vm[c_] / vm.on_road_min).mean(), (vr_[c_] / vr_.on_road_min).mean()]) * 100  # per-van mean
    ax[2].bar([0, 1], vals, bottom=bottom, color=colr, label=lab)
    bottom += vals
ax[2].set_xticks([0, 1])
ax[2].set_xticklabels(["MILP", "RL"])
ax[2].set_ylabel("% of van on-road time")
ax[2].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, columnspacing=0.6,
             handlelength=1.0)
rc.savefig(fig, os.path.join(FIG, "util_dists.pdf"))

# ---------------------------------------------------------------- Part III: per variant
rows = []
UX = {}
for vid, (code, sgn, lab) in VAR.items():
    for j, m in enumerate(("MILP", "RT")):
        kp = RES[(vid, m)]["keys"]
        b = summ(UB["MILP" if m == "MILP" else "RL"], kp)
        v = summ(UV[(vid, m)], kp)
        UX[(vid, m)] = (b, v)
        rows.append([lab if j == 0 else "", MLAB[m], pc(v["gun"] - b["gun"], 2, True),
                     pc(v["van_energy"] - b["van_energy"], 2, True),
                     f"{pc(b['fleet'], 0)}$\\to${pc(v['fleet'], 0)}", f"{pc(b['seat'], 0)}$\\to${pc(v['seat'], 0)}",
                     f"{pc(b['battery'], 0)}$\\to${pc(v['battery'], 0)}", f"{pc(b['onroad'], 0)}$\\to${pc(v['onroad'], 0)}",
                     f"{pc(b['power'], 1)}$\\to${pc(v['power'], 1)}", pc(v["system"] - b["system"], 2, True)])
        mk = "Milp" if m == "MILP" else "Rt"
        for k_ in ("gun", "van_energy", "fleet", "seat", "battery", "onroad", "power", "system", "vans_used",
                   "idle_guns", "b90", "full"):
            N[f"UX{code}{MNAME[k_]}{mk}"] = pc(v[k_], 2 if k_ == "vans_used" else 1)
            N[f"UXB{code}{MNAME[k_]}{mk}"] = pc(b[k_], 2 if k_ == "vans_used" else 1)
            N[f"UXD{code}{MNAME[k_]}{mk}"] = pc(v[k_] - b[k_], 2, True)
rc.latex_table(os.path.join(TAB, "util_experiments.tex"),
               ["variant", "method", "\\shortstack{$\\Delta$ gun time\\\\(pts)}", "\\shortstack{$\\Delta$ van share\\\\of kWh (pts)}",
                "\\shortstack{fleet use\\\\(\\% of $M$)}", "\\shortstack{seats\\\\(\\% of $Q$)}", "\\shortstack{battery\\\\(\\% of $B$)}",
                "\\shortstack{on the road\\\\(\\% of $T$)}", "\\shortstack{van power\\\\used (\\%)}",
                "\\shortstack{$\\Delta$ installed\\\\capacity (pts)}"],
               rows, align="llrrcccccr",
               caption="Utilisation of every variant against its own baseline on its paired instances (MILP and "
               "retrained RL; capacities $M$, $Q$, $B$ of the variant; arrows: baseline $\\to$ variant).",
               label="tab:utilexp", size="scriptsize")

# figure: change of van energy share and of gun time, MILP vs RL
fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.0))
labs = [PLAIN[v] for v in VAR]
y = np.arange(len(VAR))[::-1]
for i_, (key_, ttl) in enumerate((("van_energy", "change of van share of energy (points)"),
                                  ("gun", "change of gun time used (points)"))):
    for m, off in (("MILP", 0.18), ("RT", -0.18)):
        vals = [UX[(v, m)][1][key_] - UX[(v, m)][0][key_] for v in VAR]
        ax[i_].barh(y + off, vals, 0.34, color=MCOL[m], label=MLAB[m])
    ax[i_].axvline(0, color="grey", lw=0.6)
    ax[i_].set_xlabel(ttl, fontsize=8)
    ax[i_].set_yticks(y)
    ax[i_].set_yticklabels(labs if i_ == 0 else [], fontsize=7)
ax[1].legend(fontsize=7)
rc.savefig(fig, os.path.join(FIG, "util_experiments.pdf"))

# E2: is the extra capacity used?  van capacity absorbed per added van / seat
for vid in ("E2_vans_7", "E2_seats_8", "E2_vans_3", "E2_seats_4", "E8_battery_200", "E1_pM_15", "E1_pM_35"):
    b, v = UX[(vid, "MILP")]
    code = VAR[vid][0]
    N[f"UXKwh{code}"] = pc(v["van_energy"] - b["van_energy"], 1, True)
