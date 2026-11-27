"""E9 (station capacity) and E10 (charging demand): RL-only experiments; executed inside make_results_report.py
after infra_analysis.py. There is no MILP run for them: no gap to the MILP and no direction check.

For every variant and both RL runs (zero-shot, retrained) against the baseline RL run on paired instances:
dZ, operator / customer cost, validity changes, van share by segment and income, demand pressure (baseline
labels kept: the variants change guns / power / energy, hence the chargers and the window), utilisation with the
variant's capacities (infra.utilisation with cfg_over = guns_per_station, station_power_kW or energy_scale).
Also an RL tornado over all 10 experiments (marker for the RL-only factors).
"""
print("E9 / E10 (RL only) ...")
ORDER_RL = ["E9", "E10"]
VAR_RL = {
    "E9_guns_1": ("EnineGunsLo", +1, "1 gun per station"),
    "E9_guns_4": ("EnineGunsHi", -1, "4 guns per station"),
    "E9_power_11": ("EninePowLo", +1, "11 kW per gun"),
    "E9_power_44": ("EninePowHi", -1, "44 kW per gun"),
    "E10_demand_0.75": ("EtenDemLo", -1, "energy per EV $\\times0.75$"),
    "E10_demand_1.25": ("EtenDemHi", +1, "energy per EV $\\times1.25$"),
}
PLAIN.update({k: v[2].replace("$\\times", "x").replace("$", "") for k, v in VAR_RL.items()})
for eid in ORDER_RL:
    e = EXPS[eid]
    for v in e["variants"]:
        rd = F4J(f"{e['folder']}_RL", "results", v["id"])
        rr = rc.load_runs(rd)
        rr["key"] = key(rr)
        rp = rc.load_plan(rd)
        rp["key"] = key(rp)
        for m in ("ZS", "RT"):
            if (rr.run_id == RUNID[m]).any():
                RUNS[(v["id"], m)] = (rr[rr.run_id == RUNID[m]].set_index("key", drop=False), rp[rp.run_id == RUNID[m]])
                RES[(v["id"], m)] = paired(RUNS[(v["id"], m)], BASE["RL"])
RLM = [m for m in ("ZS", "RT") if all((vid, m) in RES for vid in VAR_RL)]
N["RlOnlyHasZs"] = "yes" if "ZS" in RLM else "no"
N["RlOnlyRuns"] = " and ".join(MLAB[m] for m in RLM)

N["NVariantsRl"] = len(VAR_RL)
# set-up table rows
rows = []
for eid in ORDER_RL:
    e = EXPS[eid]
    for i, v in enumerate(e["variants"]):
        code, sgn, lab = VAR_RL[v["id"]]
        bv = e["baseline_values"][v["factor"]]
        ch = 100 * (v["value"] - bv) / bv
        rows.append([eid if i == 0 else "", lab, f"{bv:g}", f"{v['value']:g}", pc(ch, 0, True),
                     "rises" if sgn > 0 else "falls"])
rc.latex_table(os.path.join(TAB, "variants_rl.tex"),
               ["exp.", "variant", "baseline", "value", "change (\\%)", "$Z$ expected to"], rows, align="lllrrl",
               caption="The six RL-only variants (no MILP run). The last column is what a lower cost bound requires "
               "of an exact optimum; RL gives no guarantee, so it is only an expectation here.", label="tab:variantsrl",
               size="small")

# per-experiment table (RL runs only)
for eid in ORDER_RL:
    rows = []
    for v in EXPS[eid]["variants"]:
        vid = v["id"]
        lab = VAR_RL[vid][2]
        for j, m in enumerate(RLM):
            r = RES[(vid, m)]
            rows.append([lab if j == 0 else "", MLAB[m], r["valid"], r["paired"], f"+{r['newly']}/$-${r['lost']}",
                         pc(r["dz_mean"], 2, True), pc(r["dz_median"], 2, True), pc(r["dop_mean"], 2, True),
                         pc(r["dcu_mean"], 2, True),
                         f"{pc(r['van_personal_base'], 1)}$\\to${pc(r['van_personal'], 1)}",
                         f"{pc(r['van_commercial_base'], 2)}$\\to${pc(r['van_commercial'], 2)}",
                         f"{pc(r['vans_base'], 2)}$\\to${pc(r['vans'], 2)}",
                         f"{pc(r['queue_commercial_base'], 2)}$\\to${pc(r['queue_commercial'], 2)}"])
    rc.latex_table(os.path.join(TAB, f"exp_{eid}.tex"),
                   ["variant", "run", "valid", "paired", "\\shortstack{newly valid\\\\/ lost}",
                    "\\shortstack{mean\\\\$\\Delta Z$ (\\%)}", "\\shortstack{median\\\\$\\Delta Z$ (\\%)}",
                    "\\shortstack{operator\\\\(\\%)}", "\\shortstack{customers\\\\(\\%)}",
                    "\\shortstack{van share\\\\personal (\\%)}", "\\shortstack{van share\\\\commercial (\\%)}",
                    "vans used", "\\shortstack{queue\\\\commercial (min)}"],
                   rows, align="llrrrrrrrrrrr",
                   caption=f"{eid} (RL only): every variant against the baseline RL run (paired instances; van shares, "
                   "vans used and the mean station queue of commercial EVs: baseline $\\to$ variant).",
                   label=f"tab:exp{eid}", size="scriptsize")

# macros
for vid, (code, sgn, lab) in VAR_RL.items():
    for m in RLM:
        r = RES[(vid, m)]
        mk = {"ZS": "Zs", "RT": "Rt"}[m]
        N[f"dZ{code}{mk}"] = pc(r["dz_mean"], 2, True)
        N[f"dZmed{code}{mk}"] = pc(r["dz_median"], 2, True)
        N[f"dOp{code}{mk}"] = pc(r["dop_mean"], 2, True)
        N[f"dCu{code}{mk}"] = pc(r["dcu_mean"], 2, True)
        N[f"VanP{code}{mk}"] = pc(r["van_personal"], 1)
        N[f"VanPB{code}{mk}"] = pc(r["van_personal_base"], 1)
        N[f"VanC{code}{mk}"] = pc(r["van_commercial"], 2)
        N[f"VanCB{code}{mk}"] = pc(r["van_commercial_base"], 2)
        N[f"Valid{code}{mk}"] = r["valid"]
        N[f"New{code}{mk}"] = r["newly"]
        N[f"Lost{code}{mk}"] = r["lost"]
        N[f"Vans{code}{mk}"] = pc(r["vans"], 2)
        N[f"VansB{code}{mk}"] = pc(r["vans_base"], 2)
        N[f"QCom{code}{mk}"] = pc(r["queue_commercial"], 2)
        N[f"QComB{code}{mk}"] = pc(r["queue_commercial_base"], 2)
        N[f"QPers{code}{mk}"] = pc(r["queue_personal"], 2)
        N[f"QPersB{code}{mk}"] = pc(r["queue_personal_base"], 2)
        N[f"CcPers{code}{mk}"] = pc(100 * (r["ccost_personal"] / r["ccost_personal_base"] - 1), 1, True)
        N[f"CcCom{code}{mk}"] = pc(100 * (r["ccost_commercial"] / r["ccost_commercial_base"] - 1), 1, True)
        N[f"Down{code}{mk}"] = pc(r["down_commercial"], 1)
        N[f"DownB{code}{mk}"] = pc(r["down_commercial_base"], 1)

# lost instances by family (retrained)
rows = []
for vid, (code, sgn, lab) in VAR_RL.items():
    vr = RUNS[(vid, "RT")][0]
    kv, kb = valid_keys(vr), valid_keys(br_r)
    dz = RES[(vid, "RT")]["dz_series"]
    fam = br_r.loc[dz.index, "class"]
    row = [lab]
    for f in rc.FAMS:
        fk = set(br_r.index[br_r["class"] == f])
        row.append(f"{pc(dz[fam == f].mean(), 2, True)} ({len((kv - kb) & fk)}/{len((kb - kv) & fk)})")
        N[f"LostFam{code}{f.replace('1', 'one').replace('2', 'two')}"] = len((kb - kv) & fk)
    rows.append(row)
rc.latex_table(os.path.join(TAB, "rl_family.tex"), ["variant"] + rc.FAMS, rows, align="lrrrrrr",
               caption="E9 and E10, retrained RL: mean $\\Delta Z$ (\\%) by family on paired instances; in brackets the "
               "instances newly valid / no longer valid.", label="tab:rlfam", size="small")
lost_n = {}
for vid in VAR_RL:
    vr = RUNS[(vid, "RT")][0]
    lost = sorted(valid_keys(br_r) - valid_keys(vr))
    lost_n[vid] = br_r.loc[lost, "n"].astype(int).value_counts().sort_index() if lost else pd.Series(dtype=int)
N["LostGunsLoMinN"] = int(lost_n["E9_guns_1"].index.min()) if len(lost_n["E9_guns_1"]) else 0

# van share by band (retrained), one table per experiment
for eid in ORDER_RL:
    vids = [v["id"] for v in EXPS[eid]["variants"]]
    keys_all = sorted(set.intersection(*[set(RES[(v, "RT")]["keys"]) for v in vids]))
    rows = []
    for s in rc.SEGMENTS:
        for b in range(5):
            row = [s if b == 0 else "", rc.BLAB[b], pc(van_share(br_p, keys_all, s, b)[2], 1)]
            row += [pc(van_share(RUNS[(v, "RT")][1], keys_all, s, b)[2], 1) for v in vids]
            rows.append(row)
    rc.latex_table(os.path.join(TAB, f"band_{eid}.tex"),
                   ["segment", "income $\\theta$", "baseline RL"] + [VAR_RL[v][2] for v in vids], rows,
                   caption=f"{eid}, retrained RL: van share of free-choice EVs (\\%) by segment and income band, on the "
                   f"{len(keys_all)} instances valid in the baseline and every variant.", label=f"tab:band{eid}",
                   size="small")
    N[f"BandKeys{ {'E9': 'Nine', 'E10': 'Ten'}[eid] }"] = len(keys_all)

# demand pressure (baseline labels kept), retrained RL
rows = []
for vid, (code, sgn, lab) in VAR_RL.items():
    kp = RES[(vid, "RT")]["keys"]
    L0, L1 = labelled(br_p, kp), labelled(RUNS[(vid, "RT")][1], kp)
    row = [lab]
    for t in ("low", "high"):
        for s in rc.SEGMENTS:
            a0, a1 = L0[(L0.segment == s) & (L0.terc == t)], L1[(L1.segment == s) & (L1.terc == t)]
            row.append(pc(share(a1) - share(a0), 1, True))
            N[f"PX{code}Van{s.capitalize()}{TC[t]}"] = pc(share(a1) - share(a0), 2, True)
        c0, c1 = L0[(L0.segment == "commercial") & (L0.terc == t)], L1[(L1.segment == "commercial") & (L1.terc == t)]
        row.append(f"{pc(fqueue(c0), 2)}$\\to${pc(fqueue(c1), 2)}")
        N[f"PX{code}QB{TC[t]}"] = pc(fqueue(c0), 2)
        N[f"PX{code}Q{TC[t]}"] = pc(fqueue(c1), 2)
    rows.append(row)
rc.latex_table(os.path.join(TAB, "pressure_rl.tex"),
               ["variant", "\\shortstack{low: van\\\\personal}", "\\shortstack{low: van\\\\commercial}",
                "\\shortstack{low: queue\\\\commercial (min)}", "\\shortstack{high: van\\\\personal}",
                "\\shortstack{high: van\\\\commercial}", "\\shortstack{high: queue\\\\commercial (min)}"], rows,
               align="lrrcrrc", caption="E9 and E10, retrained RL, at low and high demand pressure (baseline pressure "
               "labels): change of the van share of free-choice EVs (points) and mean station queue of commercial "
               "station users, baseline $\\to$ variant.", label="tab:pressrl", size="scriptsize")

# utilisation with the variant's capacities
UXR = {}
rows = []
for vid, (code, sgn, lab) in VAR_RL.items():
    co = cfg_over_of(vid)
    for j, m in enumerate(RLM):
        u = util_of(RUNS[(vid, m)], co)
        kp = RES[(vid, m)]["keys"]
        b, v = summ(UB["RL"], kp), summ(u, kp)
        UXR[(vid, m)] = (b, v)
        mk = {"ZS": "Zs", "RT": "Rt"}[m]
        for k_ in ("gun", "idle_guns", "van_energy", "fleet", "seat", "battery", "onroad", "power", "system",
                   "vans_used", "kwh_total"):
            N[f"UX{code}{MNAME.get(k_, 'KwhTotal')}{mk}"] = pc(v[k_], 2 if k_ == "vans_used" else (0 if k_ == "kwh_total" else 1))
            N[f"UXB{code}{MNAME.get(k_, 'KwhTotal')}{mk}"] = pc(b[k_], 2 if k_ == "vans_used" else (0 if k_ == "kwh_total" else 1))
        rows.append([lab if j == 0 else "", MLAB[m], f"{pc(b['gun'], 1)}$\\to${pc(v['gun'], 1)}",
                     f"{pc(b['idle_guns'], 1)}$\\to${pc(v['idle_guns'], 1)}",
                     f"{pc(b['van_energy'], 1)}$\\to${pc(v['van_energy'], 1)}",
                     f"{pc(b['fleet'], 0)}$\\to${pc(v['fleet'], 0)}", f"{pc(b['seat'], 0)}$\\to${pc(v['seat'], 0)}",
                     f"{pc(b['battery'], 0)}$\\to${pc(v['battery'], 0)}",
                     f"{pc(b['system'], 1)}$\\to${pc(v['system'], 1)}"])
rc.latex_table(os.path.join(TAB, "util_rl.tex"),
               ["variant", "run", "\\shortstack{gun\\\\util. (\\%)}", "\\shortstack{idle guns\\\\per inst.}",
                "\\shortstack{van share\\\\of kWh (\\%)}", "\\shortstack{fleet\\\\use (\\%)}", "\\shortstack{seats\\\\(\\%)}",
                "\\shortstack{battery\\\\(\\%)}", "\\shortstack{installed\\\\capacity (\\%)}"],
               rows, align="llccccccc",
               caption="E9 and E10: utilisation, baseline RL $\\to$ variant, with the variant's capacities (guns per "
               "station, kW per gun, energy per EV); means of per-instance and per-van shares, paired instances.",
               label="tab:utilrl", size="scriptsize")

# zero-shot vs retrained

train = []
for eid in ORDER_RL:
    for v in EXPS[eid]["variants"]:
        mp = F4J(f"{EXPS[eid]['folder']}_RL", "model", v["id"], "train_meta.json")
        if os.path.exists(mp):
            t = json.load(open(mp))
            train.append([VAR_RL[v["id"]][2], f"{t['train_s'] / 60:.0f}", t["best_epoch"], f"{t['best_val']:.3f}",
                          pc(RES[(v['id'], 'RT')]['dz_mean'], 2, True), RES[(v['id'], 'RT')]['valid']])
rc.latex_table(os.path.join(TAB, "retraining_rl.tex"),
               ["model trained on", "minutes (GPU)", "best epoch", "\\shortstack{validation cost\\\\RL / greedy}",
                "\\shortstack{mean $\\Delta Z$\\\\(\\%)}", "valid plans"], train, align="lrrrrr",
               caption="E9 and E10: the six retrained models (no zero-shot run exists for these experiments).",
               label="tab:retrainrl",
               size="small")

# RL tornado over all experiments (retrained RL); RL-only factors marked
FACT_RL = [("guns per station", "E9_guns_1", "E9_guns_4", -50, 100, True),
           ("kW per gun", "E9_power_11", "E9_power_44", -50, 100, True),
           ("energy per EV", "E10_demand_0.75", "E10_demand_1.25", -25, 25, True)]
ALLF = [(lab, lo, hi, -ch, ch, False) for lab, lo, hi, ch in FACT if lo] + \
       [(lab, None, hi, 0, ch, False) for lab, lo, hi, ch in FACT if not lo] + FACT_RL
CHG = {v: (100 * (vv["value"] - EXPS[e]["baseline_values"][vv["factor"]]) / EXPS[e]["baseline_values"][vv["factor"]])
       for e in EXPS for vv in EXPS[e]["variants"] for v in [vv["id"]]}
rows = []
TOR = []
for lab, lo, hi, _, _, rlonly in ALLF:
    zl = RES[(lo, "RT")]["dz_mean"] if lo else 0.0
    zh = RES[(hi, "RT")]["dz_mean"]
    cl = CHG[lo] if lo else 0.0
    el = (zh - zl) / (CHG[hi] - cl)
    ml = RES[(lo, "MILP")]["dz_mean"] if (lo and not rlonly) else np.nan
    mh = RES[(hi, "MILP")]["dz_mean"] if not rlonly else np.nan
    TOR.append((lab, lo, hi, zl, zh, abs(zh - zl), el, ml, mh, rlonly))
TOR.sort(key=lambda r: -max(abs(r[3]) if r[1] else 0, abs(r[4])))
for lab, lo, hi, zl, zh, sw, el, ml, mh, rlonly in TOR:
    lab_ = lab + ("$^{\\dagger}$" if rlonly else "")
    rows.append([lab_, f"{pc(CHG[lo], 0, True)} / {pc(CHG[hi], 0, True)}" if lo else f"-- / {pc(CHG[hi], 0, True)}",
                 pc(zl, 2, True) if lo else "--", pc(zh, 2, True), pc(sw, 2), pc(el, 3, True),
                 "RL only" if rlonly else f"{pc(ml, 2, True) if lo else '--'} / {pc(mh, 2, True)}"])
rc.latex_table(os.path.join(TAB, "tornado_rl.tex"),
               ["factor", "\\shortstack{change of the\\\\factor, low / high (\\%)}", "\\shortstack{RL$_{\\mathrm{rt}}$\\\\low (\\%)}",
                "\\shortstack{RL$_{\\mathrm{rt}}$\\\\high (\\%)}", "\\shortstack{swing\\\\(points)}", "\\shortstack{elasticity\\\\of $Z$}",
                "\\shortstack{MILP low / high\\\\(\\%)}"], rows, align="lcrrrrc",
               caption="All eleven factors of the ten experiments, measured with retrained RL (the only method run on "
               "every experiment), sorted by their largest effect on $Z$. $^{\\dagger}$: RL-only experiment, no MILP "
               "run. Elasticity: \\% change of $Z$ per \\% change of the factor between the low and high variant.",
               label="tab:tornadorl", size="scriptsize")
fig, ax = plt.subplots(figsize=(6.6, 3.6))
yy = np.arange(len(TOR))[::-1]
for i, (lab, lo, hi, zl, zh, sw, el, ml, mh, rlonly) in enumerate(TOR):
    hatch = "///" if rlonly else None
    col = "#C44E52" if rlonly else MCOL["RT"]
    if lo:
        ax.barh(yy[i], zl, 0.6, color=col, alpha=0.45, hatch=hatch, edgecolor="white")
    ax.barh(yy[i], zh, 0.6, color=col, alpha=0.9, hatch=hatch, edgecolor="white")
ax.axvline(0, color="grey", lw=0.6)
ax.set_yticks(yy)
ax.set_yticklabels([r[0].replace(r"\mathrm M", r"\mathrm{M}").replace(r"\mathrm C", r"\mathrm{C}")
                    + (" (RL only)" if r[9] else "") for r in TOR], fontsize=8)
ax.set_xlabel("mean change of $Z$, retrained RL, paired instances (%)")
from matplotlib.patches import Patch  # noqa: E402
ax.legend(handles=[Patch(facecolor=MCOL["RT"], label="experiments with MILP and RL"),
                   Patch(facecolor="#C44E52", hatch="///", edgecolor="white", label="RL-only experiments (E9, E10)")],
          fontsize=7, loc="lower right")
rc.savefig(fig, os.path.join(FIG, "tornado_rl.pdf"))
N["TopFactorRl"] = TOR[0][0]
N["TopSwingRl"] = pc(TOR[0][5], 1)
rank = [r[0] for r in TOR]
N["RankGunsRl"] = rank.index("guns per station") + 1
N["RankPowerRl"] = rank.index("kW per gun") + 1
N["RankEnergyRl"] = rank.index("energy per EV") + 1
N["NFactorsAll"] = len(TOR)
for lab, lo, hi, zl, zh, sw, el, ml, mh, rlonly in TOR:
    if rlonly:
        nm = {"guns per station": "Guns", "kW per gun": "Power", "energy per EV": "Energy"}[lab]
        N[f"Swing{nm}Rl"] = pc(sw, 2)
        N[f"Elast{nm}Rl"] = pc(el, 3, True)

# cross-check with rubric_results (only a check, written to data/)
rub = F4J("results_summary", "rubric_results")
chk = []
for fn in ("sensitivity_station_capacity.csv", "sensitivity_charging_demand.csv"):
    pth = os.path.join(rub, fn)
    if os.path.exists(pth):
        df_ = pd.read_csv(pth)
        df_.to_csv(os.path.join(DAT, f"rubric_{fn}"), index=False)
        chk.append(fn)
pd.DataFrame([{"variant": v, "run": m, "dz_mean": RES[(v, m)]["dz_mean"], "valid": RES[(v, m)]["valid"],
               "lost": RES[(v, m)]["lost"], "newly": RES[(v, m)]["newly"]}
              for v in VAR_RL for m in RLM]).to_csv(os.path.join(DAT, "rl_only_measures.csv"), index=False)
