"""Figures, tables and numbers of comprehensive_report_results.tex (formulation 4, the whole server job).

Reads only the results of formulation_4 (nothing is solved again):
  methods/milp/results (run milp_300s), methods/rl/results (run rl_f4), methods/rl/model/train_meta.json,
  experiments/<E>_<name>/milp/results/<variant>/ (run milp_300s), experiments/<E>_<name>/rl/results/<variant>/ (runs rl_f4_zero_shot and
  rl_f4_retrained), experiments/<E>_<name>/rl/model/<variant>/train_meta.json, common/experiments.json, data/solomon_config.csv.
Writes figures/*.pdf, tables/*.tex, tables/numbers.tex (+ numbers.json) and data/*.csv in this folder.

Definitions
  paired instances  an instance (file, n) with a valid plan in both runs being compared
  dZ %              100 (Z_variant - Z_baseline) / Z_baseline on one paired instance; reported as mean / median
  van share         share of free-choice EVs (station in range) served by a van, pooled over the paired instances
  gap RL to MILP    100 (Z_RL - Z_MILP) / Z_MILP on instances valid for both methods (same inputs)
  direction check   on instances optimal in both the baseline and the variant MILP run, the exact optimum must
                    rise when a cost coefficient rises or a capacity falls, and fall in the opposite case;
                    a move the other way larger than the MIP gap tolerance (0.01 %) counts as a violation
Usage (from this folder):  python make_results_report.py   then   pdflatex comprehensive_report_results.tex (twice)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import report_common as rc  # noqa: E402
F4J = rc.F4J
import matplotlib.pyplot as plt  # noqa: E402

FIG = os.path.join(HERE, "figures")
TAB = os.path.join(HERE, "tables")
DAT = os.path.join(HERE, "data")
for d in (FIG, TAB, DAT):
    os.makedirs(d, exist_ok=True)
F4 = rc.F4
EXPJ = json.load(open(F4J("common", "experiments.json"), encoding="utf-8"))
EXPS = {e["id"]: e for e in EXPJ["experiments"]}
ORDER = ["E5", "E6", "E1", "E2", "E8"]
METHODS = ["MILP", "ZS", "RT"]                      # MILP, RL zero-shot, RL retrained
MLAB = {"MILP": "MILP", "ZS": "RL zero-shot", "RT": "RL retrained"}
MCOL = {"MILP": "#4C72B0", "ZS": "#DD8452", "RT": "#55A868", "RL": "#DD8452"}
RUNID = {"MILP": "milp_300s", "ZS": "rl_f4_zero_shot", "RT": "rl_f4_retrained"}
# letter-only codes of the variants (LaTeX macro names) and the expected sign of dZ (+1: Z must not fall)
VAR = {
    "E5_cC1_personal_7.5": ("EfivePersLo", -1, "personal $c^{\\mathrm C,1}=7.5$"),
    "E5_cC1_personal_12.5": ("EfivePersHi", +1, "personal $c^{\\mathrm C,1}=12.5$"),
    "E5_omega1_commercial_4.5": ("EfiveComLo", -1, "commercial $\\omega^{1}=4.5$"),
    "E5_omega1_commercial_7.5": ("EfiveComHi", +1, "commercial $\\omega^{1}=7.5$"),
    "E6_mu0_personal_1.125": ("EsixPersLo", -1, "personal $\\mu^{0}=1.125$"),
    "E6_mu0_personal_1.875": ("EsixPersHi", +1, "personal $\\mu^{0}=1.875$"),
    "E6_mu0_commercial_1.125": ("EsixComLo", -1, "commercial $\\mu^{0}=1.125$"),
    "E6_mu0_commercial_1.875": ("EsixComHi", +1, "commercial $\\mu^{0}=1.875$"),
    "E1_pM_15": ("EonePMLo", -1, "$p^{\\mathrm M}=15$"),
    "E1_pM_35": ("EonePMHi", +1, "$p^{\\mathrm M}=35$"),
    "E2_vans_3": ("EtwoVansLo", +1, "3 vans"),
    "E2_vans_7": ("EtwoVansHi", -1, "7 vans"),
    "E2_seats_4": ("EtwoSeatsLo", +1, "4 seats per van"),
    "E2_seats_8": ("EtwoSeatsHi", -1, "8 seats per van"),
    "E8_battery_200": ("EeightBatHi", -1, "battery 200 kWh"),
}
PLAIN = {v: lab.replace("\\mathrm C", "\\mathrm{C}").replace("\\mathrm M", "\\mathrm{M}")
         for v, (_, _, lab) in VAR.items()}
CFG = pd.read_csv(os.path.join(rc.F4_DATA, "solomon_config.csv")).set_index("instance")
N = rc.Numbers()
NS = list(range(10, 101, 10))


def pc(x, d=1, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "$-$") if sign or x < 0 else s


def key(df):
    return df["instance"] + "_n" + df["n"].astype(int).astype(str)


def load(results_dir, run_id):
    r = rc.load_runs(results_dir, run_id)
    r["key"] = key(r)
    p = rc.load_plan(results_dir, run_id)
    p["key"] = key(p)
    return r.set_index("key", drop=False), p


# ------------------------------------------------------------------------------------------------ loading
print("loading baselines ...")
BASE = {"MILP": load(F4J("baseline_MILP", "results"), "milp_300s"),
        "RL": load(F4J("baseline_RL", "results"), "rl_f4")}
RUNS = {}
for eid in ORDER:
    e = EXPS[eid]
    for v in e["variants"]:
        vid = v["id"]
        print("loading", vid)
        md = F4J(f"{e['folder']}_MILP", "results", vid)
        rd = F4J(f"{e['folder']}_RL", "results", vid)
        RUNS[(vid, "MILP")] = load(md, "milp_300s")
        rr = rc.load_runs(rd)
        rr["key"] = key(rr)
        rp = rc.load_plan(rd)
        rp["key"] = key(rp)
        for m in ("ZS", "RT"):
            RUNS[(vid, m)] = (rr[rr.run_id == RUNID[m]].set_index("key", drop=False),
                              rp[rp.run_id == RUNID[m]])


def basefor(m):
    return BASE["MILP"] if m == "MILP" else BASE["RL"]


def valid_keys(r):
    return set(r.index[r.valid_plan])


def ev_rows(p, keys=None):
    e = p[p["type"] == "ev"]
    if keys is not None:
        e = e[e.key.isin(keys)]
    return e


def van_share(p, keys, seg=None, band=None):
    e = ev_rows(p, keys)
    e = e[e.stranded == "no"]
    if seg is not None:
        e = e[e.segment == seg]
    if band is not None:
        e = e[rc.band_of(e.theta) == band]
    k = int((e.resource_type == "MCS").sum())
    return k, len(e), (100.0 * k / len(e) if len(e) else np.nan)


def paired(var, base):
    """Measures of a variant run against its baseline run (same method family)."""
    vr, vp = var
    br, bp = base
    kv, kb = valid_keys(vr), valid_keys(br)
    kp = sorted(kv & kb)
    z1, z0 = vr.loc[kp, "Z_total"], br.loc[kp, "Z_total"]
    dz = 100 * (z1 - z0) / z0
    dop = 100 * (vr.loc[kp, "operator_cost_Z2_to_Z4"] - br.loc[kp, "operator_cost_Z2_to_Z4"]) / br.loc[kp, "operator_cost_Z2_to_Z4"]
    dcu = 100 * (vr.loc[kp, "customer_cost_Z1_Z5_Z6"] - br.loc[kp, "customer_cost_Z1_Z5_Z6"]) / br.loc[kp, "customer_cost_Z1_Z5_Z6"]
    out = {"valid": len(kv), "base_valid": len(kb), "paired": len(kp), "newly": len(kv - kb), "lost": len(kb - kv),
           "dz_mean": dz.mean(), "dz_median": dz.median(), "dz_rs": (z1 - z0).mean(),
           "dop_mean": dop.mean(), "dcu_mean": dcu.mean(), "dz_series": dz, "keys": kp}
    for s in rc.SEGMENTS:
        out[f"van_{s}"] = van_share(vp, kp, s)[2]
        out[f"van_{s}_base"] = van_share(bp, kp, s)[2]
        for col, nm in (("mean_queue_{}_min", "queue"), ("mean_pay_{}_Rs", "pay"),
                        ("mean_customer_cost_{}_Rs", "ccost"), ("mean_downtime_{}_min", "down")):
            out[f"{nm}_{s}"] = vr.loc[kp, col.format(s)].mean()
            out[f"{nm}_{s}_base"] = br.loc[kp, col.format(s)].mean()
    out["vans"] = vr.loc[kp, "vans_used"].mean()
    out["vans_base"] = br.loc[kp, "vans_used"].mean()
    out["seats"] = vr.loc[kp, "van_seats_used_%"].mean()
    out["seats_base"] = br.loc[kp, "van_seats_used_%"].mean()
    out["solve"] = vr["solve_s"].mean()
    return out


# ================================================================================================ PART I
N["NInst"] = 560
N["NVariants"] = len(VAR)
N["NRuns"] = len(VAR) * 3 + 2
bm_r, bm_p = BASE["MILP"]
br_r, br_p = BASE["RL"]
meta_rl = json.load(open(F4J("baseline_RL", "model", "train_meta.json")))
N["RlTrainMin"] = f"{meta_rl['train_s'] / 60:.0f}"
N["RlParams"] = f"{meta_rl['parameters']:,}"
N["RlBestEpoch"] = meta_rl["best_epoch"]
prog = open(F4J("job_progress.log"), encoding="utf-8").read().splitlines()
if "/mnt/" not in prog[0]:
    # the file now holds a later local job (E9/E10 bookkeeping); take the server job's log from git history:
    # the newest committed version whose JOB START line is the lab server's
    import subprocess  # noqa: E402
    rel = "formulation_4/job_progress.log"
    revs = subprocess.run(["git", "-C", rc.REPO, "log", "--format=%H", "--", rel], capture_output=True,
                          text=True).stdout.split()
    for h in revs:
        txt = subprocess.run(["git", "-C", rc.REPO, "show", f"{h}:{rel}"], capture_output=True, text=True,
                             encoding="utf-8").stdout
        if txt.startswith(prog[0][:0]) and "/mnt/" in txt.splitlines()[0]:
            prog = txt.splitlines()
            N["JobLogRev"] = h[:7]
            break
N["JobStart"] = prog[0][:16]
N["JobEnd"] = [l for l in prog if "JOB FINISHED" in l][0][:16]
t0 = pd.Timestamp(prog[0][:19])
t1 = pd.Timestamp([l for l in prog if "JOB FINISHED" in l][0][:19])
N["JobHours"] = f"{(t1 - t0).total_seconds() / 3600:.0f}"
N["JobNonZero"] = 0 if "0 step(s) with a non-zero" in "\n".join(prog) else "?"

# variant table (Part I)
rows = []
for eid in ORDER:
    e = EXPS[eid]
    for i, v in enumerate(e["variants"]):
        code, sgn, lab = VAR[v["id"]]
        bv = e["baseline_values"][v["factor"]]
        ch = 100 * (v["value"] - bv) / bv
        rows.append([eid if i == 0 else "", lab, f"{bv:g}", f"{v['value']:g}", pc(ch, 0, True),
                     "rises" if sgn > 0 else "falls"])
rc.latex_table(os.path.join(TAB, "variants.tex"),
               ["exp.", "variant", "baseline", "value", "change (\\%)", "optimal $Z$ must"], rows,
               align="lllrrl", caption="The 15 variants. Each changes one input; the last column is what the exact "
               "optimum must do (a cost coefficient up or a capacity down cannot lower it).", label="tab:variants",
               size="small")

# ================================================================================================ PART II
print("part II ...")
st = bm_r.status.value_counts()
for s, nm in (("optimal", "MilpOptimal"), ("time_limit", "MilpTL"), ("time_limit_no_plan", "MilpTLnoPlan"),
              ("infeasible", "MilpInfeasible")):
    N[nm] = int(st.get(s, 0))
kM, kR = valid_keys(bm_r), valid_keys(br_r)
kB = sorted(kM & kR)
N["MilpValid"], N["RlValid"], N["RlInvalid"] = len(kM), len(kR), 560 - len(kR)
N["BothValid"], N["RlOnly"], N["MilpOnly"] = len(kB), len(kR - kM), len(kM - kR)
N["NeitherValid"] = 560 - len(kM | kR)
rl_only_status = bm_r.loc[sorted(kR - kM), "status"].value_counts()
N["RlOnlyFromTLnoPlan"] = int(rl_only_status.get("time_limit_no_plan", 0))
N["RlOnlyFromInfeasible"] = int(rl_only_status.get("infeasible", 0))
inv = sorted(set(br_r.index) - kR)
inv_status = bm_r.loc[inv, "status"].value_counts()
N["RlInvalidMilpInfeasible"] = int(inv_status.get("infeasible", 0))
N["RlInvalidMilpNoPlan"] = int(inv_status.get("time_limit_no_plan", 0))
N["RlInvalidMilpValid"] = int(inv_status.get("optimal", 0) + inv_status.get("time_limit", 0))
fam_inv = bm_r.loc[inv, "class"].value_counts()
N["RlInvalidShort"] = int(fam_inv.get("R1", 0) + fam_inv.get("RC1", 0))

# status by n
rows = []
for n in NS:
    sub = bm_r[bm_r.n == n]
    ks = set(sub.index)
    s = sub.status.value_counts()
    rows.append([n, int(s.get("optimal", 0)), int(s.get("time_limit", 0)), int(s.get("time_limit_no_plan", 0)),
                 int(s.get("infeasible", 0)), len(kR & ks), len(ks - kR), len((kR - kM) & ks)])
tot = ["all"] + [sum(r[i] for r in rows) for i in range(1, 8)]
rc.latex_table(os.path.join(TAB, "status_by_n.tex"),
               ["$n$", "optimal", "\\shortstack{time limit\\\\with plan}", "\\shortstack{time limit\\\\no plan}",
                "infeasible", "\\shortstack{RL\\\\valid}", "\\shortstack{RL\\\\invalid}", "\\shortstack{only RL\\\\valid}"],
               rows, caption="How the 56 instances of each size ended, MILP (left) and RL (right).",
               label="tab:statusn", total_row=tot, size="small")

fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1), sharey=True)
bot = np.zeros(len(NS))
for s, lab in (("optimal", "optimal"), ("time_limit", "time limit, plan"), ("time_limit_no_plan", "time limit, no plan"),
               ("infeasible", "infeasible")):
    v = np.array([int((bm_r[bm_r.n == n].status == s).sum()) for n in NS])
    ax[0].bar(NS, v, 7, bottom=bot, color=rc.STATUS_COL[s], label=lab)
    bot += v
ax[0].set_title("MILP (Gurobi, 300 s)")
bot = np.zeros(len(NS))
for s, lab in (("plan", "valid plan"), ("invalid_plan", "invalid plan")):
    v = np.array([int((br_r[br_r.n == n].status == s).sum()) for n in NS])
    ax[1].bar(NS, v, 7, bottom=bot, color=rc.STATUS_COL[s], label=lab)
    bot += v
ax[1].set_title("RL (attention policy)")
for a in ax:
    a.set_xlabel("EVs per instance $n$")
    a.set_xticks(NS)
    a.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
ax[0].set_ylabel("instances (of 56)")
rc.savefig(fig, os.path.join(FIG, "status_by_n.pdf"))

# gap RL vs MILP
gap = 100 * (br_r.loc[kB, "Z_total"] - bm_r.loc[kB, "Z_total"]) / bm_r.loc[kB, "Z_total"]
gdf = pd.DataFrame({"gap": gap, "n": bm_r.loc[kB, "n"], "cls": bm_r.loc[kB, "class"], "status": bm_r.loc[kB, "status"]})
gdf.to_csv(os.path.join(DAT, "gap_rl_vs_milp_baseline.csv"))
N["GapMean"], N["GapMedian"] = f"{gap.mean():.2f}", f"{gap.median():.2f}"
N["GapPninety"] = f"{gap.quantile(0.9):.2f}"
N["GapMax"] = f"{gap.max():.1f}"
N["GapOpt"] = f"{gdf[gdf.status == 'optimal'].gap.mean():.2f}"
N["GapTL"] = f"{gdf[gdf.status == 'time_limit'].gap.mean():.2f}"
N["RlBetter"] = int((gap < -1e-6).sum())
N["RlBetterTL"] = int(((gap < -1e-6) & (gdf.status == "time_limit")).sum())
N["RlEqual"] = int((gap.abs() <= 1e-6).sum())
N["RlWithinOne"] = f"{100 * (gap <= 1).mean():.0f}"
rows = []
for n in NS:
    g = gdf[gdf.n == n].gap
    rows.append([n, len(g), pc(g.mean(), 2), pc(g.median(), 2), pc(g.quantile(0.9), 2), pc(g.max(), 1),
                 int((g < -1e-6).sum())])
rows.append(["all", len(gdf), pc(gap.mean(), 2), pc(gap.median(), 2), pc(gap.quantile(0.9), 2), pc(gap.max(), 1),
             int((gap < -1e-6).sum())])
rc.latex_table(os.path.join(TAB, "gap_by_n.tex"),
               ["$n$", "paired", "mean (\\%)", "median (\\%)", "90th pct (\\%)", "max (\\%)", "RL cheaper"], rows,
               caption="Gap of RL to the MILP on the instances valid for both, by size.", label="tab:gapn", size="small")
rows = []
for f in rc.FAMS:
    g = gdf[gdf.cls == f]
    rows.append([f, len(g), pc(g.gap.mean(), 2), pc(g.gap.median(), 2),
                 pc(g[g.status == "optimal"].gap.mean(), 2), pc(g[g.status == "time_limit"].gap.mean(), 2),
                 int((bm_r["class"] == f).sum() - len(set(bm_r.index[bm_r["class"] == f]) - kR)),
                 len(set(bm_r.index[bm_r["class"] == f]) & (kR - kM))])
rc.latex_table(os.path.join(TAB, "gap_by_family.tex"),
               ["family", "paired", "mean (\\%)", "median (\\%)", "\\shortstack{mean, MILP\\\\optimal (\\%)}",
                "\\shortstack{mean, MILP\\\\time limit (\\%)}", "RL valid", "only RL valid"], rows,
               caption="Gap of RL to the MILP by Solomon family, split by how the MILP ended.", label="tab:gapfam",
               size="small")
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.6))
ax[0].boxplot([gdf[gdf.n == n].gap for n in NS], positions=NS, widths=6, showfliers=False)
ax[0].axhline(0, color="grey", lw=0.6)
ax[0].set_xlabel("$n$")
ax[0].set_ylabel("gap RL to MILP (%)")
ax[0].set_xticks(NS)
ax[0].set_xticklabels(NS)
for i, s in enumerate(["optimal", "time_limit"]):
    sub = gdf[gdf.status == s]
    ax[1].boxplot([sub[sub.cls == f].gap for f in rc.FAMS], positions=np.arange(6) + (i - 0.5) * 0.35, widths=0.3,
                  showfliers=False, patch_artist=True,
                  boxprops=dict(facecolor=rc.STATUS_COL[s], alpha=0.6), medianprops=dict(color="black"))
ax[1].axhline(0, color="grey", lw=0.6)
ax[1].set_xticks(range(6))
ax[1].set_xticklabels(rc.FAMS)
ax[1].set_ylabel("gap RL to MILP (%)")
ax[1].plot([], [], color=rc.STATUS_COL["optimal"], lw=6, alpha=0.6, label="MILP optimal")
ax[1].plot([], [], color=rc.STATUS_COL["time_limit"], lw=6, alpha=0.6, label="MILP time limit")
ax[1].legend(fontsize=7)
rc.savefig(fig, os.path.join(FIG, "gap_baseline.pdf"))

# decision agreement on paired instances
cols = ["key", "id", "segment", "theta", "stranded", "resource_type", "resource", "queue_min", "pay_Rs",
        "customer_cost_Rs", "downtime_min"]
em = ev_rows(bm_p, kB)[cols]
er = ev_rows(br_p, kB)[cols]
mm = em.merge(er, on=["key", "id"], suffixes=("_m", "_r"))
fc = mm[mm.stranded_m == "no"]
N["AgreeAll"] = f"{100 * (mm.resource_type_m == mm.resource_type_r).mean():.1f}"
N["AgreeFree"] = f"{100 * (fc.resource_type_m == fc.resource_type_r).mean():.1f}"
both_f = fc[(fc.resource_type_m == "FCS") & (fc.resource_type_r == "FCS")]
st_m = both_f.resource_m.str.split("-").str[0]
st_r = both_f.resource_r.str.split("-").str[0]
N["AgreeStation"] = f"{100 * (st_m == st_r).mean():.1f}"
N["AgreeStrandedVan"] = f"{100 * ((mm[mm.stranded_m == 'yes'].resource_type_r == 'MCS').mean()):.0f}"
rows = []
for s in rc.SEGMENTS + ["all"]:
    sub = fc if s == "all" else fc[fc.segment_m == s]
    t = pd.crosstab(sub.resource_type_m, sub.resource_type_r).reindex(index=["FCS", "MCS"], columns=["FCS", "MCS"],
                                                                         fill_value=0)
    rows.append([s, len(sub), int(t.loc["FCS", "FCS"]), int(t.loc["FCS", "MCS"]), int(t.loc["MCS", "FCS"]),
                 int(t.loc["MCS", "MCS"]), pc(100 * (sub.resource_type_m == sub.resource_type_r).mean(), 1)])
    if s != "all":
        N[f"Agree{s.capitalize()}"] = pc(100 * (sub.resource_type_m == sub.resource_type_r).mean(), 1)
        N[f"MvanRfcs{s.capitalize()}"] = int(t.loc["MCS", "FCS"])
        N[f"MfcsRvan{s.capitalize()}"] = int(t.loc["FCS", "MCS"])
rc.latex_table(os.path.join(TAB, "agreement.tex"),
               ["segment", "\\shortstack{free-choice\\\\EVs}", "\\shortstack{both\\\\station}",
                "\\shortstack{MILP station\\\\RL van}", "\\shortstack{MILP van\\\\RL station}", "\\shortstack{both\\\\van}",
                "same (\\%)"], rows,
               caption="Resource chosen for the same EV by the two methods (free-choice EVs of the paired instances).",
               label="tab:agree", size="small")

# van share by segment x band, both methods
rows = []
vs = {}
for s in rc.SEGMENTS:
    for meth, p in (("MILP", bm_p), ("RL", br_p)):
        for b in range(5):
            k, n_, sh = van_share(p, kB, s, b)
            vs[(s, meth, b)] = (k, n_, sh)
    for meth, p in (("MILP", bm_p), ("RL", br_p)):
        k, n_, sh = van_share(p, kB, s)
        N[f"Van{s.capitalize()}{meth.capitalize()}"] = pc(sh, 1)
for s in rc.SEGMENTS:
    for b in range(5):
        km_, nm_, shm = vs[(s, "MILP", b)]
        kr_, _, shr = vs[(s, "RL", b)]
        lo, hi = rc.wilson(km_, nm_)
        rows.append([s if b == 0 else "", rc.BLAB[b], nm_, pc(shm, 1), f"[{max(0.0, 100 * lo):.1f}, {100 * hi:.1f}]" if nm_ else "--",
                     pc(shr, 1)])
rc.latex_table(os.path.join(TAB, "vanshare_methods.tex"),
               ["segment", "income $\\theta$", "free-choice EVs", "MILP van (\\%)", "95\\,\\% interval", "RL van (\\%)"],
               rows, caption="Van share of free-choice EVs by segment and income band on the paired instances, "
               "MILP against RL (Wilson interval for the MILP share).", label="tab:vsm", size="small")
N["VanPersTopMilp"] = pc(vs[("personal", "MILP", 4)][2], 1)
N["VanPersTopRl"] = pc(vs[("personal", "RL", 4)][2], 1)
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.6), sharey=False)
x = np.arange(5) * 0.2 + 0.1
for i, s in enumerate(rc.SEGMENTS):
    for meth, ls in (("MILP", "-"), ("RL", "--")):
        ax[i].plot(x, [vs[(s, meth, b)][2] for b in range(5)], ls, marker="o", ms=3,
                   color=MCOL["MILP" if meth == "MILP" else "RL"], label=meth)
    ax[i].set_title(f"{s} EVs")
    ax[i].set_xlabel("income $\\theta$ (band midpoint)")
    ax[i].legend(fontsize=7)
ax[0].set_ylabel("free-choice EVs by van (%)")
rc.savefig(fig, os.path.join(FIG, "vanshare_methods.pdf"))

# per segment measures + objective composition + time
def seg_means(p, keys):
    e = ev_rows(p, keys)
    out = {}
    for s in rc.SEGMENTS:
        es = e[e.segment == s]
        f_ = es[es.resource_type == "FCS"]
        out[s] = dict(queue=f_.queue_min.mean(), queuers=100 * (f_.queue_min > 1e-6).mean(), down=es.downtime_min.mean(),
                      pay=es.pay_Rs.mean(), cc=es.customer_cost_Rs.mean(), ttc=es.time_to_charged_min.mean())
    return out


sm, sr = seg_means(bm_p, kB), seg_means(br_p, kB)
rows = []
for nm_, lab, d in (("queue", "mean station queue (min), station users", 2), ("queuers", "station users queueing (\\%)", 1),
                    ("down", "mean request to charged, downtime (min)", 1), ("pay", "mean money paid (Rs)", 0),
                    ("cc", "mean perceived cost (Rs)", 0)):
    rows.append([lab, pc(sm["personal"][nm_], d), pc(sr["personal"][nm_], d), pc(sm["commercial"][nm_], d),
                 pc(sr["commercial"][nm_], d)])
rc.latex_table(os.path.join(TAB, "segment_methods.tex"),
               ["measure (per EV)", "\\shortstack{personal\\\\MILP}", "\\shortstack{personal\\\\RL}",
                "\\shortstack{commercial\\\\MILP}", "\\shortstack{commercial\\\\RL}"], rows,
               align="lrrrr", caption="Service per segment on the paired instances.", label="tab:segm", size="small")
N["QueueComMilp"], N["QueueComRl"] = pc(sm["commercial"]["queue"], 2), pc(sr["commercial"]["queue"], 2)
N["QueuePersMilp"], N["QueuePersRl"] = pc(sm["personal"]["queue"], 2), pc(sr["personal"]["queue"], 2)

comp = ["Z1_EV_drive_perceived", "Z2_van_drive", "Z3_dispatch", "Z4_operator_energy", "Z5_tariff_perceived",
        "Z6_time_perceived"]
rows = []
for c, lab in zip(comp, ["$Z_1$ EV driving", "$Z_2$ van driving", "$Z_3$ van dispatch", "$Z_4$ operator energy",
                         "$Z_5$ tariff", "$Z_6$ time"]):
    a = bm_r.loc[kB, c].sum() / bm_r.loc[kB, "Z_total"].sum() * 100
    b = br_r.loc[kB, c].sum() / br_r.loc[kB, "Z_total"].sum() * 100
    d = (br_r.loc[kB, c].sum() - bm_r.loc[kB, c].sum()) / bm_r.loc[kB, "Z_total"].sum() * 100
    rows.append([lab, pc(a, 1), pc(b, 1), pc(d, 2, True)])
rows.append(["vans used per instance", pc(bm_r.loc[kB, "vans_used"].mean(), 2), pc(br_r.loc[kB, "vans_used"].mean(), 2), ""])
rows.append(["van seats used (\\%)", pc(bm_r.loc[kB, "van_seats_used_%"].mean(), 1),
             pc(br_r.loc[kB, "van_seats_used_%"].mean(), 1), ""])
rows.append(["van km per instance", pc(bm_r.loc[kB, "van_km"].mean(), 1), pc(br_r.loc[kB, "van_km"].mean(), 1), ""])
rc.latex_table(os.path.join(TAB, "composition_methods.tex"),
               ["term", "MILP share of $Z$ (\\%)", "RL share of $Z$ (\\%)", "RL $-$ MILP (\\% of MILP $Z$)"], rows,
               align="lrrr", caption="Composition of the objective and van use on the paired instances.",
               label="tab:compm", size="small")
N["VansMilp"] = pc(bm_r.loc[kB, "vans_used"].mean(), 2)
N["VansRl"] = pc(br_r.loc[kB, "vans_used"].mean(), 2)
dz3 = (br_r.loc[kB, "Z3_dispatch"].sum() - bm_r.loc[kB, "Z3_dispatch"].sum()) / bm_r.loc[kB, "Z_total"].sum() * 100
dz6 = (br_r.loc[kB, "Z6_time_perceived"].sum() - bm_r.loc[kB, "Z6_time_perceived"].sum()) / bm_r.loc[kB, "Z_total"].sum() * 100
N["DZthreeRl"], N["DZsixRl"] = pc(dz3, 2, True), pc(dz6, 2, True)

N["MilpSolveMean"] = f"{bm_r.solve_s.mean():.0f}"
N["RlSolveMean"] = f"{br_r.solve_s.mean():.2f}"
N["SpeedUp"] = f"{bm_r.solve_s.mean() / br_r.solve_s.mean():,.0f}"
N["MilpCPUh"] = f"{bm_r.solve_s.sum() * 4 / 3600:.0f}"
fig, ax = plt.subplots(figsize=(3.8, 2.5))
ax.plot(NS, [bm_r[bm_r.n == n].solve_s.mean() for n in NS], "o-", ms=3, color=MCOL["MILP"], label="MILP (4 threads)")
ax.plot(NS, [br_r[br_r.n == n].solve_s.mean() for n in NS], "s--", ms=3, color=MCOL["RL"], label="RL (520 rollouts)")
ax.set_yscale("log")
ax.set_xlabel("$n$")
ax.set_ylabel("mean time per instance (s)")
ax.legend(fontsize=7)
rc.savefig(fig, os.path.join(FIG, "time_methods.pdf"))

# ================================================================================================ PART III
print("part III ...")
RES = {}
for vid in VAR:
    for m in METHODS:
        RES[(vid, m)] = paired(RUNS[(vid, m)], basefor(m))

# direction check on optimal pairs (MILP)
DIR = {}
for vid, (code, sgn, lab) in VAR.items():
    vr = RUNS[(vid, "MILP")][0]
    ko = sorted(set(vr.index[vr.status == "optimal"]) & set(bm_r.index[bm_r.status == "optimal"]))
    dz = 100 * (vr.loc[ko, "Z_total"] - bm_r.loc[ko, "Z_total"]) / bm_r.loc[ko, "Z_total"]
    bad = int((sgn * dz < -0.01).sum())
    DIR[vid] = dict(n=len(ko), mean=dz.mean(), bad=bad, zero=int((dz.abs() <= 0.01).sum()))
N["DirPairs"] = f"{sum(d['n'] for d in DIR.values()):,}"
N["DirBad"] = sum(d["bad"] for d in DIR.values())

# RL gap to MILP of the same variant; sign agreement; instance-level correlation of dZ
GAPV = {}
for vid in VAR:
    vm = RUNS[(vid, "MILP")][0]
    for m in ("ZS", "RT"):
        vr = RUNS[(vid, m)][0]
        kk = sorted(valid_keys(vm) & valid_keys(vr))
        g = 100 * (vr.loc[kk, "Z_total"] - vm.loc[kk, "Z_total"]) / vm.loc[kk, "Z_total"]
        GAPV[(vid, m)] = (len(kk), g.mean(), g.median())
    a = RES[(vid, "MILP")]["dz_series"]
    b = RES[(vid, "RT")]["dz_series"]
    kk = sorted(set(a.index) & set(b.index))
    GAPV[(vid, "corr")] = np.corrcoef(a.loc[kk], b.loc[kk])[0, 1] if len(kk) > 2 else np.nan

N["GapVarMin"] = pc(min(GAPV[(v, m)][1] for v in VAR for m in ("ZS", "RT")), 2)
N["GapVarMax"] = pc(max(GAPV[(v, m)][1] for v in VAR for m in ("ZS", "RT")), 2)
agree_sign = sum(np.sign(RES[(v, "MILP")]["dz_mean"]) == np.sign(RES[(v, "RT")]["dz_mean"]) for v in VAR)
agree_sign_zs = sum(np.sign(RES[(v, "MILP")]["dz_mean"]) == np.sign(RES[(v, "ZS")]["dz_mean"]) for v in VAR)
N["SignAgreeRt"], N["SignAgreeZs"] = agree_sign, agree_sign_zs
diffs_rt = [abs(RES[(v, "MILP")]["dz_mean"] - RES[(v, "RT")]["dz_mean"]) for v in VAR]
diffs_zs = [abs(RES[(v, "MILP")]["dz_mean"] - RES[(v, "ZS")]["dz_mean"]) for v in VAR]
N["EffDiffRtMean"], N["EffDiffRtMax"] = f"{np.mean(diffs_rt):.2f}", f"{np.max(diffs_rt):.2f}"
N["EffDiffZsMean"], N["EffDiffZsMax"] = f"{np.mean(diffs_zs):.2f}", f"{np.max(diffs_zs):.2f}"

# per-variant macros
for vid, (code, sgn, lab) in VAR.items():
    for m in METHODS:
        r = RES[(vid, m)]
        mk = {"MILP": "Milp", "ZS": "Zs", "RT": "Rt"}[m]
        N[f"dZ{code}{mk}"] = pc(r["dz_mean"], 2, True)
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
        N[f"Seats{code}{mk}"] = pc(r["seats"], 1)
        N[f"CcPers{code}{mk}"] = pc(100 * (r["ccost_personal"] / r["ccost_personal_base"] - 1), 1, True)
        N[f"CcCom{code}{mk}"] = pc(100 * (r["ccost_commercial"] / r["ccost_commercial_base"] - 1), 1, True)
        N[f"SeatsB{code}{mk}"] = pc(r["seats_base"], 1)
    N[f"DirN{code}"] = DIR[vid]["n"]
    N[f"DirMean{code}"] = pc(DIR[vid]["mean"], 2, True)
    N[f"GapZs{code}"] = pc(GAPV[(vid, "ZS")][1], 2)
    N[f"GapRt{code}"] = pc(GAPV[(vid, "RT")][1], 2)
    N[f"Corr{code}"] = pc(GAPV[(vid, "corr")], 2)

# full table of all variants x methods
def exp_table(eid, path, label):
    rows = []
    for v in EXPS[eid]["variants"]:
        vid = v["id"]
        code, sgn, lab = VAR[vid]
        for j, m in enumerate(METHODS):
            r = RES[(vid, m)]
            rows.append([lab if j == 0 else "", MLAB[m], r["valid"], r["paired"], f"+{r['newly']}/$-${r['lost']}",
                         pc(r["dz_mean"], 2, True), pc(r["dz_median"], 2, True), pc(r["dop_mean"], 2, True),
                         pc(r["dcu_mean"], 2, True),
                         f"{pc(r['van_personal_base'], 1)}$\\to${pc(r['van_personal'], 1)}",
                         f"{pc(r['van_commercial_base'], 2)}$\\to${pc(r['van_commercial'], 2)}",
                         f"{pc(r['vans_base'], 2)}$\\to${pc(r['vans'], 2)}"])
    rc.latex_table(path, ["variant", "method", "valid", "paired", "\\shortstack{newly valid\\\\/ lost}",
                          "\\shortstack{mean\\\\$\\Delta Z$ (\\%)}", "\\shortstack{median\\\\$\\Delta Z$ (\\%)}",
                          "\\shortstack{operator\\\\(\\%)}", "\\shortstack{customers\\\\(\\%)}",
                          "\\shortstack{van share\\\\personal (\\%)}", "\\shortstack{van share\\\\commercial (\\%)}",
                          "vans used"],
                   rows, align="llrrrrrrrrrr",
                   caption=f"{eid}: every variant against its own baseline, for the three runs (paired instances; "
                   "van shares and vans used: baseline $\\to$ variant on the same paired instances).",
                   label=label, size="scriptsize")


for eid in ORDER:
    exp_table(eid, os.path.join(TAB, f"exp_{eid}.tex"), f"tab:exp{eid}")

# direction + RL accuracy table per experiment
rows = []
for vid, (code, sgn, lab) in VAR.items():
    d = DIR[vid]
    rows.append([lab, "$+$" if sgn > 0 else "$-$", d["n"], pc(d["mean"], 2, True), d["zero"], d["bad"],
                 pc(RES[(vid, "RT")]["dz_mean"] - RES[(vid, "MILP")]["dz_mean"], 2, True),
                 pc(GAPV[(vid, "ZS")][1], 2), pc(GAPV[(vid, "RT")][1], 2), pc(GAPV[(vid, "corr")], 2)])
rc.latex_table(os.path.join(TAB, "direction.tex"),
               ["variant", "\\shortstack{expected\\\\sign}", "\\shortstack{optimal\\\\pairs}",
                "\\shortstack{mean $\\Delta Z$\\\\optimal (\\%)}", "\\shortstack{unchanged\\\\($\\le$0.01\\,\\%)}",
                "violations", "\\shortstack{RL$_{\\mathrm{rt}}$ effect\\\\$-$ MILP effect}",
                "\\shortstack{gap to MILP\\\\zero-shot (\\%)}", "\\shortstack{gap to MILP\\\\retrained (\\%)}",
                "\\shortstack{corr.\\ of\\\\$\\Delta Z$}"],
               rows, align="lrrrrrrrrr",
               caption="Checks of every variant. Left: the exact optimum on instances optimal in both MILP runs "
               "(direction check). Right: RL against the MILP of the same variant; last column: correlation over "
               "instances of the MILP and the retrained-RL change of $Z$.", label="tab:direction", size="scriptsize")

# van share by band per experiment (MILP), personal and commercial
def band_shares(p, keys, s):
    return [van_share(p, keys, s, b)[2] for b in range(5)]


for eid in ORDER:
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
    vids = [v["id"] for v in EXPS[eid]["variants"]]
    keys_all = sorted(set.intersection(*[set(RES[(v, "MILP")]["keys"]) for v in vids]))
    x = np.arange(5) * 0.2 + 0.1
    cmap = plt.get_cmap("viridis", len(vids) + 1)
    for i, s in enumerate(rc.SEGMENTS):
        ax[i].plot(x, band_shares(bm_p, keys_all, s), "k-", lw=2, marker="o", ms=3, label="baseline")
        for j, v in enumerate(vids):
            ax[i].plot(x, band_shares(RUNS[(v, "MILP")][1], keys_all, s), "--", marker="s", ms=3, color=cmap(j),
                       label=PLAIN[v])
        ax[i].set_title(f"{s} EVs (MILP)")
        ax[i].set_xlabel("income $\\theta$ (band midpoint)")
    ax[0].set_ylabel("free-choice EVs by van (%)")
    h_, l_ = ax[0].get_legend_handles_labels()
    fig.legend(h_, l_, fontsize=7, loc="lower center", ncol=len(vids) + 1)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(FIG, f"vanshare_{eid}.pdf"))
    plt.close(fig)
    # table by band (MILP and RL retrained)
    rows = []
    for s in rc.SEGMENTS:
        for b in range(5):
            row = [s if b == 0 else "", rc.BLAB[b], pc(van_share(bm_p, keys_all, s, b)[2], 1)]
            row += [pc(van_share(RUNS[(v, "MILP")][1], keys_all, s, b)[2], 1) for v in vids]
            rows.append(row)
    rc.latex_table(os.path.join(TAB, f"band_{eid}.tex"),
                   ["segment", "income $\\theta$", "baseline"] + [VAR[v][2] for v in vids], rows,
                   caption=f"{eid}, MILP: van share of free-choice EVs (\\%) by segment and income band, on the "
                   f"{len(keys_all)} instances valid in the baseline and every variant.", label=f"tab:band{eid}",
                   size="small")
    N[f"BandKeys{ {'E5': 'Five', 'E6': 'Six', 'E1': 'One', 'E2': 'Two', 'E8': 'Eight'}[eid] }"] = len(keys_all)

# equity: customer cost per EV by segment x band, baseline vs variant (MILP, paired)
def cc_band(p, keys, s, b):
    e = ev_rows(p, keys)
    e = e[(e.segment == s) & (rc.band_of(e.theta) == b)]
    return e.customer_cost_Rs.mean(), e.pay_Rs.mean(), e.queue_min[e.resource_type == "FCS"].mean()


for eid in ["E1", "E6", "E5"]:
    vids = [v["id"] for v in EXPS[eid]["variants"]]
    rows = []
    for s in rc.SEGMENTS:
        for b in range(5):
            row = [s if b == 0 else "", rc.BLAB[b]]
            for v in vids:
                kp = RES[(v, "MILP")]["keys"]
                c0, p0, _ = cc_band(bm_p, kp, s, b)
                c1, p1, _ = cc_band(RUNS[(v, "MILP")][1], kp, s, b)
                row += [pc(100 * (c1 - c0) / c0, 1, True), pc(p1 - p0, 0, True)]
            rows.append(row)
    hdr = ["segment", "income $\\theta$"]
    for v in vids:
        hdr += [f"\\shortstack{{{VAR[v][2]}\\\\perceived (\\%)}}", "\\shortstack{paid\\\\(Rs)}"]
    rc.latex_table(os.path.join(TAB, f"equity_{eid}.tex"), hdr, rows, align="ll" + "rr" * len(vids),
                   caption=f"{eid}, MILP: change of the perceived cost (\\%) and of the money paid (Rs) per EV, "
                   "by segment and income band (all EVs of the paired instances, stranded EVs included: they always pay the van "
                   "tariff).", label=f"tab:eq{eid}", size="scriptsize")

# E1 specific numbers: equity
for v in ["E1_pM_15", "E1_pM_35"]:
    code = VAR[v][0]
    kp = RES[(v, "MILP")]["keys"]
    c0, _, _ = cc_band(bm_p, kp, "personal", 4)
    c1, _, _ = cc_band(RUNS[(v, "MILP")][1], kp, "personal", 4)
    N[f"CcPersTop{code}"] = pc(100 * (c1 - c0) / c0, 1, True)
    c0, _, _ = cc_band(bm_p, kp, "commercial", 0)
    c1, _, _ = cc_band(RUNS[(v, "MILP")][1], kp, "commercial", 0)
    N[f"CcComLow{code}"] = pc(100 * (c1 - c0) / c0, 1, True)
    N[f"VanPersTop{code}"] = pc(van_share(RUNS[(v, "MILP")][1], kp, "personal", 4)[2], 1)
    N[f"VanPersTopB{code}"] = pc(van_share(bm_p, kp, "personal", 4)[2], 1)

# E2 specific: by family
rows = []
for v in ["E2_vans_3", "E2_vans_7", "E2_seats_4", "E2_seats_8"]:
    r = RES[(v, "MILP")]
    vr = RUNS[(v, "MILP")][0]
    dz = r["dz_series"]
    fam = bm_r.loc[dz.index, "class"]
    kv, kb = valid_keys(vr), valid_keys(bm_r)
    row = [VAR[v][2]]
    for f in rc.FAMS:
        fk = set(bm_r.index[bm_r["class"] == f])
        row.append(f"{pc(dz[fam == f].mean(), 2, True)} ({len((kv - kb) & fk)}/{len((kb - kv) & fk)})")
    rows.append(row)
rc.latex_table(os.path.join(TAB, "e2_family.tex"), ["variant"] + rc.FAMS, rows, align="lrrrrrr",
               caption="E2, MILP: mean $\\Delta Z$ (\\%) by family on paired instances; in brackets the instances "
               "newly valid / no longer valid.", label="tab:e2fam", size="small")


# van resources: tightest (baseline, E2, E8)
def van_limits(p, keys, Q, B):
    v = p[(p["type"] == "van") & p.key.isin(keys)].copy()
    v["inst"] = v.key.str.split("_n").str[0]
    v["T"] = v.inst.map(CFG["T_min"])
    v["seat"] = v.stops / Q
    v["batt"] = v.kWh_drawn / B
    v["time"] = v.return_min / v["T"]
    t = v[["seat", "batt", "time"]].idxmax(axis=1).value_counts(normalize=True) * 100
    return dict(vans=len(v), seat=t.get("seat", 0), batt=t.get("batt", 0), time=t.get("time", 0),
                full=100 * (v.stops >= Q).mean(), b90=100 * (v.batt > 0.9).mean(), kwh=v.kWh_drawn.mean(),
                bmax=v.kWh_drawn.max())


rows = []
LIM = {}
for lab, p, keys, Q, B in [("baseline (5 vans, 6 seats, 150 kWh)", bm_p, sorted(kM), 6, 150),
                           ("3 vans", RUNS[("E2_vans_3", "MILP")][1], RES[("E2_vans_3", "MILP")]["keys"], 6, 150),
                           ("7 vans", RUNS[("E2_vans_7", "MILP")][1], RES[("E2_vans_7", "MILP")]["keys"], 6, 150),
                           ("4 seats", RUNS[("E2_seats_4", "MILP")][1], RES[("E2_seats_4", "MILP")]["keys"], 4, 150),
                           ("8 seats", RUNS[("E2_seats_8", "MILP")][1], RES[("E2_seats_8", "MILP")]["keys"], 8, 150),
                           ("battery 200 kWh", RUNS[("E8_battery_200", "MILP")][1],
                            RES[("E8_battery_200", "MILP")]["keys"], 6, 200)]:
    d = van_limits(p, keys, Q, B)
    LIM[lab] = d
    rows.append([lab, d["vans"], pc(d["full"], 1), pc(d["b90"], 1), pc(d["kwh"], 1), pc(d["bmax"], 1),
                 pc(d["seat"], 1), pc(d["batt"], 1), pc(d["time"], 1)])
rc.latex_table(os.path.join(TAB, "van_limits.tex"),
               ["run (MILP)", "vans", "\\shortstack{full\\\\(\\%)}", "\\shortstack{battery\\\\$>$90\\,\\% (\\%)}",
                "\\shortstack{mean kWh\\\\drawn}", "\\shortstack{max kWh\\\\drawn}", "\\shortstack{tightest:\\\\seats (\\%)}",
                "\\shortstack{tightest:\\\\battery (\\%)}", "\\shortstack{tightest:\\\\time (\\%)}"], rows,
               align="lrrrrrrrr", caption="Which van resource is closest to its limit (every dispatched van of the "
               "valid MILP plans; tightest = largest used share of seats, battery and time $T$).", label="tab:vanlim",
               size="small")
b = LIM["baseline (5 vans, 6 seats, 150 kWh)"]
N["VanBattTightBase"] = pc(b["batt"], 1)
N["VanSeatTightBase"] = pc(b["seat"], 1)
N["VanTimeTightBase"] = pc(b["time"], 1)
N["VanBninetyBase"] = pc(b["b90"], 1)
N["VanBmaxBase"] = pc(b["bmax"], 0)
N["VanBninetyEeight"] = pc(LIM["battery 200 kWh"]["b90"], 1)
N["VanBattTightEeight"] = pc(LIM["battery 200 kWh"]["batt"], 1)
e8 = RES[("E8_battery_200", "MILP")]
N["EeightDecision"] = "below" if abs(e8["dz_mean"]) < EXPS["E8"]["decision_threshold_pct"] else "above"
N["EeightAbs"] = f"{abs(e8['dz_mean']):.2f}"
N["EeightOptChanged"] = len(DIR["E8_battery_200"]) and (DIR["E8_battery_200"]["n"] - DIR["E8_battery_200"]["zero"])

# synthesis: tornado (MILP and RL retrained), elasticities
FACT = [("personal $c^{\\mathrm C,1}$", "E5_cC1_personal_7.5", "E5_cC1_personal_12.5", 25),
        ("commercial $\\omega^{1}$", "E5_omega1_commercial_4.5", "E5_omega1_commercial_7.5", 25),
        ("personal $\\mu^{0}$", "E6_mu0_personal_1.125", "E6_mu0_personal_1.875", 25),
        ("commercial $\\mu^{0}$", "E6_mu0_commercial_1.125", "E6_mu0_commercial_1.875", 25),
        ("van tariff $p^{\\mathrm M}$", "E1_pM_15", "E1_pM_35", 40),
        ("vans $|M|$", "E2_vans_3", "E2_vans_7", 40),
        ("seats $Q$", "E2_seats_4", "E2_seats_8", 33.3),
        ("battery $B$", None, "E8_battery_200", 33.3)]
FACT.sort(key=lambda f: -max(abs(RES[(v, "MILP")]["dz_mean"]) for v in f[1:3] if v))
rows = []
for lab, lo, hi, ch in FACT:
    def g(v, m, k="dz_mean"):
        return RES[(v, m)][k] if v else np.nan
    swing = abs(g(hi, "MILP") - (g(lo, "MILP") if lo else 0))
    el = (g(hi, "MILP") - (g(lo, "MILP") if lo else 0)) / (2 * ch if lo else ch)
    rows.append([lab, pc(g(lo, "MILP"), 2, True), pc(g(hi, "MILP"), 2, True), pc(g(lo, "RT"), 2, True),
                 pc(g(hi, "RT"), 2, True), pc(swing, 2), pc(el, 3, True),
                 pc(g(lo, "MILP", "van_personal") - g(lo, "MILP", "van_personal_base") if lo else np.nan, 1, True),
                 pc(g(hi, "MILP", "van_personal") - g(hi, "MILP", "van_personal_base"), 1, True)])
rc.latex_table(os.path.join(TAB, "tornado.tex"),
               ["factor", "\\shortstack{MILP\\\\low (\\%)}", "\\shortstack{MILP\\\\high (\\%)}",
                "\\shortstack{RL$_{\\mathrm{rt}}$\\\\low (\\%)}", "\\shortstack{RL$_{\\mathrm{rt}}$\\\\high (\\%)}",
                "\\shortstack{swing\\\\(points)}", "\\shortstack{elasticity\\\\of $Z$}",
                "\\shortstack{$\\Delta$ van share\\\\personal, low}", "\\shortstack{$\\Delta$ van share\\\\personal, high}"],
               rows, align="lrrrrrrrr",
               caption="All factors sorted by their largest effect on $Z$ (mean $\\Delta Z$ over paired instances). "
               "Elasticity: \\% change of $Z$ per \\% change of the factor (between the low and high variant; for the "
               "battery the high variant alone). Van-share changes in percentage points (MILP).", label="tab:tornado",
               size="scriptsize")
fig, ax = plt.subplots(figsize=(6.4, 3.2))
yy = np.arange(len(FACT))[::-1]
for i, (lab, lo, hi, ch) in enumerate(FACT):
    for v, sh in ((lo, -0.17), (hi, 0.17)):
        if v is None:
            continue
        ax.barh(yy[i] + sh * 0.0 + 0.17, RES[(v, "MILP")]["dz_mean"], 0.3, color=MCOL["MILP"],
                alpha=0.9 if v == hi else 0.45)
        ax.barh(yy[i] - 0.17, RES[(v, "RT")]["dz_mean"], 0.3, color=MCOL["RT"], alpha=0.9 if v == hi else 0.45)
ax.axvline(0, color="grey", lw=0.6)
ax.set_yticks(yy)
ax.set_yticklabels([f[0].replace(r"\mathrm M", r"\mathrm{M}").replace(r"\mathrm C", r"\mathrm{C}") for f in FACT])
ax.set_xlabel("mean change of $Z$ on paired instances (%)")
ax.barh([], [], color=MCOL["MILP"], label="MILP (dark: higher value, light: lower)")
ax.barh([], [], color=MCOL["RT"], label="RL retrained")
ax.legend(fontsize=7, loc="lower right")
rc.savefig(fig, os.path.join(FIG, "tornado.pdf"))
N["TopFactor"] = FACT[0][0]
N["TopSwing"] = pc(abs(RES[(FACT[0][2], "MILP")]["dz_mean"] - RES[(FACT[0][1], "MILP")]["dz_mean"]), 1)

# MILP vs RL effect scatter (all variants)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
for m, mk in (("ZS", "o"), ("RT", "s")):
    xs = [RES[(v, "MILP")]["dz_mean"] for v in VAR]
    ys = [RES[(v, m)]["dz_mean"] for v in VAR]
    ax[0].scatter(xs, ys, marker=mk, s=16, color=MCOL[m], label=MLAB[m])
lim = max(abs(np.array([RES[(v, m)]["dz_mean"] for v in VAR for m in METHODS]))) * 1.1
ax[0].plot([-lim, lim], [-lim, lim], color="grey", lw=0.6, ls=":")
ax[0].set_xlabel("MILP: mean $\\Delta Z$ (%)")
ax[0].set_ylabel("RL: mean $\\Delta Z$ (%)")
ax[0].legend(fontsize=7)
xs = [RES[(v, "MILP")]["van_personal"] - RES[(v, "MILP")]["van_personal_base"] for v in VAR]
for m, mk in (("ZS", "o"), ("RT", "s")):
    ys = [RES[(v, m)]["van_personal"] - RES[(v, m)]["van_personal_base"] for v in VAR]
    ax[1].scatter(xs, ys, marker=mk, s=16, color=MCOL[m], label=MLAB[m])
lim = max(abs(np.array(xs))) * 1.1
ax[1].plot([-lim, lim], [-lim, lim], color="grey", lw=0.6, ls=":")
ax[1].set_xlabel("MILP: change of personal van share (pts)")
ax[1].set_ylabel("RL: change (pts)")
rc.savefig(fig, os.path.join(FIG, "effects_methods.pdf"))
xs = np.array([RES[(v, "MILP")]["van_personal"] - RES[(v, "MILP")]["van_personal_base"] for v in VAR])
ys = np.array([RES[(v, "RT")]["van_personal"] - RES[(v, "RT")]["van_personal_base"] for v in VAR])
N["VanEffSignAgree"] = int(sum(np.sign(xs[np.abs(xs) > 0.5]) == np.sign(ys[np.abs(xs) > 0.5])))
N["VanEffBig"] = int(sum(np.abs(xs) > 0.5))
N["VanEffRatio"] = f"{np.sum(np.abs(ys)) / np.sum(np.abs(xs)):.2f}"

# zero-shot vs retrained
zs_gap = np.mean([GAPV[(v, "ZS")][1] for v in VAR])
rt_gap = np.mean([GAPV[(v, "RT")][1] for v in VAR])
N["GapZsAll"], N["GapRtAll"] = f"{zs_gap:.2f}", f"{rt_gap:.2f}"
N["RtBetterCount"] = int(sum(GAPV[(v, "RT")][1] < GAPV[(v, "ZS")][1] for v in VAR))
train = []
for eid in ORDER:
    for v in EXPS[eid]["variants"]:
        mp = F4J(f"{EXPS[eid]['folder']}_RL", "model", v["id"], "train_meta.json")
        if os.path.exists(mp):
            t = json.load(open(mp))
            train.append(dict(variant=v["id"], min=t["train_s"] / 60, best=t["best_val"], epoch=t["best_epoch"],
                              stop=t.get("epoch")))
tr = pd.DataFrame(train)
tr.to_csv(os.path.join(DAT, "retraining.csv"), index=False)
N["RetrainMinMean"] = f"{tr['min'].mean():.0f}"
N["RetrainMinTotal"] = f"{tr['min'].sum() / 60:.1f}"
N["RetrainBestMin"], N["RetrainBestMax"] = f"{tr.best.min():.3f}", f"{tr.best.max():.3f}"
N["BaseBestVal"] = f"{meta_rl['best_val']:.3f}"
rows = [[VAR[r.variant][2], f"{r['min']:.0f}", r.epoch, f"{r.best:.3f}", pc(GAPV[(r.variant, 'ZS')][1], 2),
         pc(GAPV[(r.variant, 'RT')][1], 2)] for _, r in tr.iterrows()]
rows.append(["baseline model", f"{meta_rl['train_s'] / 60:.0f}", meta_rl["best_epoch"], f"{meta_rl['best_val']:.3f}",
             "", ""])
rc.latex_table(os.path.join(TAB, "retraining.tex"),
               ["model trained on", "minutes (GPU)", "best epoch", "\\shortstack{validation cost\\\\RL / greedy}",
                "\\shortstack{gap to MILP\\\\zero-shot (\\%)}", "\\shortstack{gap to MILP\\\\retrained (\\%)}"],
               rows, align="lrrrrr", caption="The 15 retrained RL models and the gap of both RL runs to the MILP of "
               "the same variant.", label="tab:retrain", size="small")

# summary data table
out = []
for (vid, m), r in RES.items():
    out.append({k: v for k, v in r.items() if k not in ("dz_series", "keys")} | {"variant": vid, "method": m})
pd.DataFrame(out).to_csv(os.path.join(DAT, "variant_measures.csv"), index=False)
pd.DataFrame([{"variant": v} | d for v, d in DIR.items()]).to_csv(os.path.join(DAT, "direction_check.csv"), index=False)

exec(open(os.path.join(HERE, "peak_analysis.py"), encoding="utf-8").read())
exec(open(os.path.join(HERE, "infra_analysis.py"), encoding="utf-8").read())
exec(open(os.path.join(HERE, "rl_only_analysis.py"), encoding="utf-8").read())

# fit every table to the text width (shrink only)
import glob  # noqa: E402
for fn in glob.glob(os.path.join(TAB, "*.tex")):
    if os.path.basename(fn).startswith("numbers"):
        continue
    t_ = open(fn, encoding="utf-8").read()
    if "adjustbox" not in t_:
        t_ = t_.replace(r"\begin{tabular}", r"\begin{adjustbox}{max width=\textwidth}\begin{tabular}")
        t_ = t_.replace(r"\end{tabular}", r"\end{tabular}\end{adjustbox}")
        open(fn, "w", encoding="utf-8").write(t_)
N.save(os.path.join(TAB, "numbers.tex"))
print("done:", len(N.d), "numbers")
