"""Figures, tables and numbers of comprehensive_report_rl.tex (formulation-4 RL baseline).

Reads (nothing is solved here):
  ../../../methods/rl/results     runs.csv, plan.csv, meta.json     (run rl_f4)
  ../../../methods/rl/model       train_log.csv, train_meta.json
  ../../../methods/milp/results   runs.csv, plan.csv               (milp_300s, greedy)
  ../../../data, common/f4_params.json   instances (pressure, distances, energy)
  data/duals_*.csv                                        from compute_duals_rl_f4.py (run it first)
Writes figures/r_*.pdf, tables/r_*.tex, tables/numbers.tex (+ numbers.json).
Usage (from this folder):  python compute_duals_rl_f4.py ; python make_rl_report.py ; pdflatex x2

Definitions
  free-choice EV    an EV with at least one station in range (stranded EVs must take a van)
  income band       theta in [0,0.2), ..., [0.8,1]; deciles of width 0.1 where stated
  demand pressure   ../common/infra.py (one definition for the three reports): window = mean 60 E_j / 22 kW of
                    the instance's EVs; load_i = EVs j with r_i - window <= r_j <= r_i (i included);
                    pressure_i = load_i / (24 guns + 5 vans); terciles = infra.pressure_cuts(), pooled over the
                    free-choice EVs of all 560 baseline instances (data only, no plan)
  utilisation       infra.utilisation() on the plan rows of the valid instances (definitions in infra.py)
  gap               100 (Z_RL - Z_other) / Z_other over instances where both plans are valid
"""
import json
import os
import re
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import report_common as rc  # noqa: E402
F4J = rc.F4J
import infra  # noqa: E402

rc.f4_import()
import f4_data as fd  # noqa: E402

FIG = os.path.join(HERE, "figures")
TAB = os.path.join(HERE, "tables")
DATA = os.path.join(HERE, "data")
RL_DIR = F4J("baseline_RL", "results")
RL_MODEL = F4J("baseline_RL", "model")
MILP_DIR = F4J("baseline_MILP", "results")
N = rc.Numbers()
C_RL, C_M, C_G = "#C44E52", "#4C72B0", "#8C8C8C"
SEGN = {0: "personal", 1: "commercial"}


def fit_width(path):
    """Shrink a tabular to the text width if (and only if) it is wider (adjustbox max width)."""
    with open(path, encoding="utf-8") as fh:
        s = fh.read()
    s = s.replace("\\begin{tabular}", "\\begin{adjustbox}{max width=\\textwidth}\\begin{tabular}", 1)
    s = s.replace("\\end{tabular}", "\\end{tabular}\\end{adjustbox}", 1)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(s)


def tab(name, header, rows, **kw):
    rc.latex_table(os.path.join(TAB, name + ".tex"), header, rows, **kw)
    fit_width(os.path.join(TAB, name + ".tex"))


def tabular_only(name, header, rows, align=None, total_row=None):
    """Bare tabular (for \\resizebox or tables whose caption is in the .tex file)."""
    align = align or ("l" + "r" * (len(header) - 1))
    out = [f"\\begin{{tabular}}{{{align}}}\\toprule", " & ".join(header) + " \\\\ \\midrule"]
    out += [" & ".join(str(x) for x in r) + " \\\\" for r in rows]
    if total_row:
        out.append("\\midrule " + " & ".join(str(x) for x in total_row) + " \\\\")
    out.append("\\bottomrule\\end{tabular}")
    with open(os.path.join(TAB, name + ".tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    fit_width(os.path.join(TAB, name + ".tex"))


f1 = lambda v: rc.fmt(v, "{:.1f}")   # noqa: E731
f2 = lambda v: rc.fmt(v, "{:.2f}")   # noqa: E731
f0 = lambda v: rc.fmt(v, "{:.0f}")   # noqa: E731
f3 = lambda v: rc.fmt(v, "{:.3f}")   # noqa: E731


def mathtext(c):
    """LaTeX '\\mathrm F' / '\\mathrm{D,P,0}' -> matplotlib mathtext with braces."""
    c = c.replace("\\mathrm{", "\\mathrm{")
    return re.sub(r"\\mathrm ([A-Za-z])", lambda m_: "\\mathrm{" + m_.group(1) + "}", c)


def ci(v):
    return f"{v:,.0f}".replace(",", "\\,")


# ====================================================================== loading
def load():
    runs = pd.read_csv(os.path.join(RL_DIR, "runs.csv"))
    runs["valid_plan"] = runs.valid.eq("yes")
    mall = pd.read_csv(os.path.join(MILP_DIR, "runs.csv"), low_memory=False)
    mall["valid_plan"] = mall.valid.eq("yes")
    milp = mall[mall.run_id == "milp_300s"].copy()
    greedy = mall[mall.run_id == "greedy"].copy()
    plan_r = pd.read_csv(os.path.join(RL_DIR, "plan.csv"), low_memory=False)
    plan_m = pd.read_csv(os.path.join(MILP_DIR, "plan.csv"), low_memory=False)
    plan_m = plan_m[plan_m.run_id == "milp_300s"]
    meta = json.load(open(os.path.join(RL_DIR, "meta.json")))["rl_f4"]
    tmeta = json.load(open(os.path.join(RL_MODEL, "train_meta.json")))
    tlog = pd.read_csv(os.path.join(RL_MODEL, "train_log.csv"))
    return runs, milp, greedy, plan_r, plan_m, meta, tmeta, tlog


def instance_info(names):
    """Per (instance, n): EV attributes from the formulation-4 data (pressure, E, nearest reachable FCS, ...)."""
    cache = os.path.join(DATA, "ev_info.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache), json.load(open(cache.replace(".csv", "_inst.json")))
    ds = fd.Dataset(rc.F4_DATA, stems=("solomon",))
    rows, inst = [], {}
    for name, n in names:
        J = ds.instance(name, int(n), analysis=True)
        ev = J["ev"]
        inst[f"{name}|{n}"] = {"T": J["params"]["T_min"], "family": J["family"]}
        for i in range(J["n_ev"]):
            reach = fd.reachable(J, i)
            rows.append({"instance": name, "n": int(n), "id": i + 1, "E": ev["E_kWh"][i],
                         "r": ev["r_min"][i], "pressure": ev["pressure"][i],
                         "nearest_fcs_km": min(J["d_ev_fcs_km"][i][f] for f in reach) if reach else np.nan,
                         "mu": ev["mu"][i], "cQ": ev["cQ"][i], "W": ev["W"][i],
                         "lone_fcs_ok": any(ev["r_min"][i] + J["t_ev_fcs_min"][i][f] + ev["tauF_min"][i]
                                            <= J["params"]["T_min"] + 1e-6 for f in reach),
                         "lone_van_ok": max(ev["r_min"][i], J["t_nodes_min"][0][i + 1]) + ev["tauM_min"][i]
                         + J["t_nodes_min"][i + 1][0] <= J["params"]["T_min"] + 1e-6})
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    json.dump(inst, open(cache.replace(".csv", "_inst.json"), "w"))
    return df, inst


def ev_table(plan, runs, info):
    ev = plan[plan.type == "ev"].copy()
    ok = runs[runs.valid_plan][["instance", "n", "class", "status"]]
    ev = ev.merge(ok, on=["instance", "n"])
    ev = ev.merge(info, on=["instance", "n", "id"], how="left")
    ev["van"] = ev.resource_type.eq("MCS")
    ev["free"] = ev.stranded.eq("no")
    ev["seg"] = ev.segment.map({"personal": 0, "commercial": 1})
    ev["band"] = rc.band_of(ev.theta)
    ev["decile"] = np.clip(np.floor(ev.theta * 10).astype(int), 0, 9)
    ev["long"] = ev["class"].str.endswith("2")
    return ev


# ====================================================================== Part I tables
def part1(meta, tmeta):
    P = meta.get("f4_params") or tmeta["f4_params"]
    rows = []
    lab = {"mu": ("money sensitivity $\\mu$", "Rs/Rs"), "D": ("value of time $c^{\\mathrm D}$ (driving)", "Rs/min"),
           "Q": ("value of time $c^{\\mathrm Q}$ (queueing)", "Rs/min"),
           "C": ("value of time $c^{\\mathrm C}$ (FCS charging)", "Rs/min"),
           "V": ("value of time $c^{\\mathrm V}$ (van wait + charging)", "Rs/min"),
           "W": ("queue tolerance $W$", "min")}
    for key in ("mu", "D", "Q", "C", "V", "W"):
        vals = []
        for s in ("personal", "commercial"):
            S = P[s]
            line = (S["money_sensitivity_mu"] if key == "mu" else S["queue_tolerance_W_min"] if key == "W"
                    else S["value_of_time_Rs_per_min"][key])
            vals += [f"{line['theta0']:g}", f"{line['theta1']:g}"]
        rows.append([lab[key][0], lab[key][1]] + vals)
    td = P["personal"]["theta_distribution"], P["commercial"]["theta_distribution"]
    rows.insert(0, ["income $\\theta$", "--", f"\\multicolumn{{2}}{{c}}{{Beta({td[0]['a']:g}, {td[0]['b']:g})}}",
                    f"\\multicolumn{{2}}{{c}}{{Beta({td[1]['a']:g}, {td[1]['b']:g})}}"])
    tabular_only("r_params", ["parameter", "unit", "pers. $\\theta=0$", "pers. $\\theta=1$",
                              "comm. $\\theta=0$", "comm. $\\theta=1$"], rows, align="llrrrr")
    N["PiC"] = f"{P['segment_distribution']['pi_commercial']:.2f}"
    N["TariffF"] = f"{P['tariff_Rs_per_kWh']['p_F']:g}"
    N["TariffM"] = f"{P['tariff_Rs_per_kWh']['p_M']:g}"

    tp = meta["training_params"]
    g, m, t, inf = tp["generator"], tp["model"], tp["training"], tp["inference"]
    rows = [
        ["Network", "embedding size $d$ / heads / encoder layers", f"{m['d']} / {m['heads']} / {m['layers']}"],
        ["", "feed-forward width / logit clip $C$ / dynamic MLP width", f"{m['ff']} / {m['clip']:g} / {m['dyn_hidden']}"],
        ["", "trainable parameters", ci(tmeta["parameters"])],
        ["Generator", "templates (file size, class)", f"{len(tmeta['templates'])} (100 EVs; C1 \\dots RC2)"],
        ["", "clusters / cluster spread (share of area)", f"{g['clusters'][0]}--{g['clusters'][1]} / "
                                                         f"{g['cluster_sigma_share'][0]}--{g['cluster_sigma_share'][1]}"],
        ["", "share of requests at $t=0$ / Beta shape of the rest", f"{g['request_zero_share'][0]}--{g['request_zero_share'][1]}"
                                                                  f" / $a\\in[{g['request_beta'][0]},{g['request_beta'][1]}]$"],
        ["", "latest request (share of $T$) / jitter of $T$", f"{g['request_max_share'][0]}--{g['request_max_share'][1]}"
                                                            f" / $\\pm{g['T_jitter']*100:.0f}$\\%"],
        ["Training", "imitation warm-up batches / learning rate", f"{t['imitation_batches']} / {t['lr']:g}"],
        ["", "REINFORCE learning rate / decay per epoch", f"{t['lr_reinforce']:g} / {t['lr_decay_per_epoch']:g}"],
        ["", "instances per batch $\\times$ rollouts per instance", f"{t['batch_instances']} $\\times$ {t['samples']}"],
        ["", "batches per epoch / instance size $n$", f"{t['iters_per_epoch']} / {t['n_min']}--{t['n_max']}"],
        ["", "penalty per unplaced EV (cost scales)", f"{t['drop_penalty']:g}"],
        ["", "stop rule", f"$\\ge${t['min_epochs']} epochs, patience {t['patience_epochs']}, max {t['max_epochs']}, {t['hours']:g} h"],
        ["", "validation", f"{t['val_instances_per_size']} instances at $n\\in\\{{{','.join(map(str, t['val_sizes']))}\\}}$"],
        ["Inference", "map symmetries $\\times$ (greedy + sampled rollouts)", f"{inf['augment']} $\\times$ (1 + {inf['samples']})"
                                                                         f" = {inf['augment'] * (1 + inf['samples'])}"],
        ["", "plan kept", "fewest unplaced EVs, then lowest $Z$"],
        ["Hardware", "device / CPU threads / torch", f"{tmeta['device'].upper()} GPU / {tmeta['threads']} / {tmeta['torch'].split('+')[0]}"],
    ]
    tabular_only("r_settings", ["", "setting", "value"], rows, align="llr")
    N["NParams"] = ci(tmeta["parameters"])
    N["Rollouts"] = str(inf["augment"] * (1 + inf["samples"]))
    N["Augment"] = str(inf["augment"])
    N["Samples"] = str(inf["samples"])
    N["ImitBatches"] = str(t["imitation_batches"])
    N["ImitEpochs"] = str(t["imitation_batches"] // t["iters_per_epoch"])
    N["MinEpochs"] = str(t["min_epochs"])
    N["Patience"] = str(t["patience_epochs"])
    N["BatchInst"] = str(t["batch_instances"])
    N["TrainSamples"] = str(t["samples"])
    N["DropPenalty"] = f"{t['drop_penalty']:g}"


# ====================================================================== training
def training(tlog, tmeta, meta):
    tl = tlog[tlog.epoch > 0].copy()
    byn = tl.val_ratio_by_n.apply(json.loads)
    fig, ax = plt.subplots(1, 3, figsize=(9.0, 3.0))
    imit = tl[tl.phase == "imitation"].epoch.max()
    for a in ax:
        a.axvspan(0, imit, color="#EEEEEE", lw=0)
    ax[0].plot(tl.epoch, tl.val_ratio_rl_over_greedy, color="k", lw=1.4, label="all sizes")
    for k, c in zip(("20", "50", "100"), ("#55A868", "#DD8452", "#8172B2")):
        ax[0].plot(tl.epoch, byn.apply(lambda d: d[k]), color=c, lw=0.8, label=f"$n={k}$")
    ax[0].axhline(1, color="grey", ls=":", lw=0.8)
    ax[0].axvline(tmeta["best_epoch"], color=C_RL, ls="--", lw=0.8)
    ax[0].set_ylim(0.9, 1.25)
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("validation cost RL / greedy rule")
    ax[0].legend(fontsize=7)
    ax[1].plot(tl.epoch, tl["val_unplaced_rl_%"], color=C_RL, lw=1, label="RL (validation)")
    ax[1].axhline(tl["val_unplaced_greedy_%"].iloc[-1], color=C_G, ls="--", lw=1, label="greedy rule")
    ax[1].plot(tl.epoch, tl["train_unplaced_%"], color="#DD8452", lw=0.6, alpha=0.8, label="RL (training rollouts)")
    ax[1].set_ylim(0, 10)
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("unplaced EVs (%)")
    ax[1].legend(fontsize=7)
    ax[2].plot(tl.epoch, tl["train_cost_spread_%"], color="#4C72B0", lw=0.8)
    ax[2].set_xlabel("epoch")
    ax[2].set_ylabel("cost spread of the 8 rollouts (%)")
    rc.savefig(fig, os.path.join(FIG, "r_training.pdf"))

    best = tl[tl.epoch == tmeta["best_epoch"]].iloc[0]
    bb = json.loads(best.val_ratio_by_n)
    first_rf = tl[tl.phase == "reinforce"].iloc[0]
    N["BestEpoch"] = str(tmeta["best_epoch"])
    N["LastEpoch"] = str(tmeta["epoch"])
    N["TrainMinTotal"] = f"{tmeta['train_s'] / 60:.1f}"
    N["TrainMinBest"] = f"{meta['train_s'] / 60:.1f}"
    N["ValBest"] = f"{tmeta['best_val']:.3f}"
    N["ValBestPct"] = f"{(1 - tmeta['best_val']) * 100:.1f}"
    N["ValImitEnd"] = f"{tl[tl.phase == 'imitation'].val_ratio_rl_over_greedy.iloc[-1]:.3f}"
    N["ValStart"] = f"{tlog[tlog.epoch == 0].val_ratio_rl_over_greedy.iloc[0]:.2f}"
    N["ValFirstRf"] = f"{first_rf.val_ratio_rl_over_greedy:.3f}"
    N["ValBestTwenty"] = f"{bb['20']:.3f}"
    N["ValBestFifty"] = f"{bb['50']:.3f}"
    N["ValBestHundred"] = f"{bb['100']:.3f}"
    N["ValUnplRl"] = f"{best["val_unplaced_rl_%"]:.2f}"
    N["ValUnplGr"] = f"{best["val_unplaced_greedy_%"]:.2f}"
    N["ValUnplStart"] = f"{tlog[tlog.epoch == 0]["val_unplaced_rl_%"].iloc[0]:.1f}"
    N["FailedBatches"] = str(int(tl.failed_batches.fillna(0).sum()))
    N["SpreadEnd"] = f"{tl['train_cost_spread_%'].iloc[-5:].mean():.1f}"


# ====================================================================== outcomes / feasibility
def outcomes(runs, milp, greedy, info, inst):
    x = runs.merge(milp[["instance", "n", "status", "valid_plan", "Z_total", "solve_s", "gap_%"]],
                   on=["instance", "n"], suffixes=("", "_m"))
    x = x.merge(greedy[["instance", "n", "valid_plan", "Z_total"]].rename(
        columns={"valid_plan": "valid_plan_g", "Z_total": "Z_total_g"}), on=["instance", "n"])
    # figure: outcome by n
    ns = sorted(x.n.unique())
    v = [((x.n == n) & x.valid_plan).sum() for n in ns]
    inf_ = [((x.n == n) & ~x.valid_plan & (x.status_m == "infeasible")).sum() for n in ns]
    oth = [((x.n == n) & ~x.valid_plan & (x.status_m != "infeasible")).sum() for n in ns]
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    ax.bar(ns, v, width=6, color="#55A868", label="valid plan")
    ax.bar(ns, inf_, width=6, bottom=v, color="#C44E52", label="invalid; MILP proved infeasible")
    ax.bar(ns, oth, width=6, bottom=np.add(v, inf_), color="#8172B2", label="invalid; MILP found no plan in 300 s")
    ax.set_xticks(ns)
    ax.set_xlabel("EVs per instance $n$")
    ax.set_ylabel("instances (of 56)")
    ax.set_ylim(0, 75)
    ax.legend(fontsize=7, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    rc.savefig(fig, os.path.join(FIG, "r_outcome.pdf"))

    rows = []
    for n in ns:
        s = x[x.n == n]
        rows.append([n, len(s), s.valid_plan.sum(), (~s.valid_plan).sum(), s.valid_plan_m.sum(), s.valid_plan_g.sum(),
                     ((s.valid_plan) & (~s.valid_plan_m)).sum(), f3(s.solve_s.mean())])
    rows_t = ["all", len(x), x.valid_plan.sum(), (~x.valid_plan).sum(), x.valid_plan_m.sum(), x.valid_plan_g.sum(),
              ((x.valid_plan) & (~x.valid_plan_m)).sum(), f3(x.solve_s.mean())]
    tab("r_outcome", ["$n$", "instances", "RL valid", "RL invalid", "MILP plan", "greedy plan", "RL only",
                      "RL time (s)"], rows, total_row=rows_t,
        caption="RL outcome per instance size, next to the MILP (300\\,s) and the greedy rule. RL only = RL has a "
                "valid plan, the MILP none.", label="tab:outcome")
    ct = pd.crosstab(x.status, x.status_m)
    order = ["optimal", "time_limit", "time_limit_no_plan", "infeasible"]
    lab = {"optimal": "optimal", "time_limit": "time limit, plan", "time_limit_no_plan": "time limit, no plan",
           "infeasible": "infeasible"}
    rows = [["valid plan" if s == "plan" else "invalid plan"] + [ct.loc[s, o] if o in ct.columns else 0
                                                                for o in order] + [ct.loc[s].sum()]
            for s in ("plan", "invalid_plan")]
    tab("r_crosstab", ["RL $\\backslash$ MILP"] + [lab[o] for o in order] + ["total"], rows,
        caption="RL outcome against the MILP outcome of the same instance.", label="tab:crosstab")
    N["NInst"] = str(len(x))
    N["NValid"] = str(int(x.valid_plan.sum()))
    N["NInvalid"] = str(int((~x.valid_plan).sum()))
    N["NValidM"] = str(int(x.valid_plan_m.sum()))
    N["NValidG"] = str(int(x.valid_plan_g.sum()))
    N["NOptM"] = str(int((x.status_m == "optimal").sum()))
    N["NTlM"] = str(int((x.status_m == "time_limit").sum()))
    N["NNoPlanM"] = str(int((x.status_m == "time_limit_no_plan").sum()))
    N["NInfM"] = str(int((x.status_m == "infeasible").sum()))
    N["RLonly"] = str(int((x.valid_plan & ~x.valid_plan_m).sum()))
    N["MILPonly"] = str(int((~x.valid_plan & x.valid_plan_m).sum()))
    N["InvalidInf"] = str(int((~x.valid_plan & (x.status_m == "infeasible")).sum()))
    N["InvalidNoPlan"] = str(int((~x.valid_plan & (x.status_m == "time_limit_no_plan")).sum()))
    N["RLonlyGr"] = str(int((x.valid_plan & ~x.valid_plan_g).sum()))

    # invalid plans: violation categories, dropped EVs, lone-EV check
    bad = x[~x.valid_plan].copy()
    cats = {"EV finishes after $T$": r"finishes after T", "van back at the depot after $T$": r"van back after T",
            "queue longer than $W_i$": r"queues .* > W", "van battery exceeded": r"battery exceeded"}
    rows = []
    for k, pat in cats.items():
        has = bad.violation_detail.str.contains(pat, regex=True)
        rows.append([k, has.sum(), bad[has].violation_detail.str.count(pat).sum()])
    tab("r_violations", ["violated rule", "plans", "violations"], rows, align="lrr",
        caption="Rules broken by the invalid RL plans (a plan can break several).", label="tab:violations")
    bad["dropped"] = bad.message.str.extract(r"\(\[([\d, ]+)\]\)")[0]
    bad["n_dropped"] = bad.dropped.fillna("").apply(lambda s: len([t for t in s.split(",") if t.strip()]))
    lone = info.set_index(["instance", "n", "id"])
    def lone_impossible(row):
        ids = [int(t) for t in str(row.dropped).split(",") if t.strip()] if isinstance(row.dropped, str) else []
        return any(not (lone.loc[(row.instance, row.n, i)].lone_fcs_ok or lone.loc[(row.instance, row.n, i)].lone_van_ok)
                   for i in ids)
    bad["data_infeasible"] = bad.apply(lone_impossible, axis=1)
    fam = bad.groupby("class").agg(plans=("n", "size"), nmin=("n", "min"), nmax=("n", "max"),
                                   milp_inf=("status_m", lambda s: (s == "infeasible").sum()),
                                   milp_noplan=("status_m", lambda s: (s == "time_limit_no_plan").sum()),
                                   dropped=("n_dropped", "mean"), lone=("data_infeasible", "sum"))
    rows = [[c, int(r.plans), int(r.nmin), int(r.nmax), int(r.milp_inf), int(r.milp_noplan), f1(r.dropped),
             int(r.lone)] for c, r in fam.iterrows()]
    tab("r_invalid", ["family", "invalid", "smallest $n$", "largest $n$", "MILP infeasible", "MILP no plan",
                      "unplaced EVs (mean)", "one EV alone impossible"], rows,
        caption="Invalid RL plans by family. Last column: instances where an EV the policy could not place cannot be "
                "fully charged by $T$ even alone (free gun at once, or a van leaving for it at once).",
        label="tab:invalid")
    N["InvLone"] = str(int(bad.data_infeasible.sum()))
    N["InvLoneNoPlan"] = str(int((bad.data_infeasible & (bad.status_m == "time_limit_no_plan")).sum()))
    N["InvOneEV"] = str(int((bad.n_dropped == 1).sum()))
    N["InvDroppedMax"] = str(int(bad.n_dropped.max()))
    N["InvFamR"] = str(int((bad["class"] == "R1").sum()))
    N["InvFamRC"] = str(int((bad["class"] == "RC1").sum()))
    N["InvFamOther"] = str(int((~bad["class"].isin(["R1", "RC1"])).sum()))
    N["InvSmallestN"] = str(int(bad.n.min()))
    N["InvHundred"] = str(int((bad.n == 100).sum()))
    for k, pat, nm in (("after", r"finishes after T", "InvAfterT"), ("van", r"van back after T", "InvVanBack"),
                       ("q", r"queues .* > W", "InvQueue")):
        N[nm] = str(int(bad.violation_detail.str.contains(pat, regex=True).sum()))
    # RL-only instances
    only = x[x.valid_plan & ~x.valid_plan_m]
    N["RLonlyMeanN"] = f"{only.n.mean():.0f}"
    N["RLonlyFams"] = ", ".join(f"{c} {k}" for c, k in only["class"].value_counts().sort_index().items())
    return x


# ====================================================================== instance overview / families
def overview(x, ev_r, plan_r):
    v = x[x.valid_plan].copy()
    v["Zev"] = v.Z_total / v.n
    v["opshare"] = 100 * v.operator_cost_Z2_to_Z4 / v.Z_total
    v["vanshare"] = 100 * v.EVs_by_van / v.n
    rows = []
    def add(lbl, s, spec=f1):
        s = s.dropna()
        rows.append([lbl, len(s), spec(s.mean()), spec(s.median()), spec(s.quantile(.1)), spec(s.quantile(.9)),
                     spec(s.min()), spec(s.max())])
    add("objective $Z$ per EV (Rs)", v.Zev, f0)
    add("operator share of $Z$ (\\%)", v.opshare)
    add("vans used", v.vans_used)
    add("EVs served by a van (\\%)", v.vanshare)
    add("free-choice EVs by van (\\%)", v["share_free_choice_by_van_%"])
    add("mean FCS queue (min)", v.mean_queue_FCS_min, f2)
    add("mean van wait (min)", v.mean_van_wait_min)
    add("mean request to charged (min)", v.mean_time_to_charged_min)
    add("van km", v.van_km)
    add("RL time (s), all instances", x.solve_s, f3)
    tab("r_inst_stats", ["measure (per instance)", "count", "mean", "median", "10th pct", "90th pct", "min", "max"],
        rows, align="lrrrrrrr", size="small",
        caption="Distribution of instance-level results over the valid RL plans.", label="tab:inststats")
    N["ZevMean"] = f0(v.Zev.mean())
    N["ZevMin"] = f0(v.Zev.min())
    N["ZevMax"] = f0(v.Zev.max())
    N["OpShare"] = f1(v.opshare.mean())
    N["VansMean"] = f1(v.vans_used.mean())
    N["VanShareEV"] = f1(v.vanshare.mean())
    N["FreeVanInst"] = f1(v["share_free_choice_by_van_%"].mean())

    fig, ax = plt.subplots(1, 3, figsize=(10, 2.9))
    ns = sorted(v.n.unique())
    ax[0].boxplot([v[v.n == n].Zev for n in ns], tick_labels=ns, showfliers=False)
    ax[0].set_ylabel("Z per EV (Rs)")
    ax[1].boxplot([x[x.n == n].solve_s for n in ns], tick_labels=ns, showfliers=False)
    ax[1].set_ylabel("RL time per instance (s)")
    ax[2].boxplot([v[v.n == n].vans_used for n in ns], tick_labels=ns, showfliers=False)
    ax[2].set_ylabel("vans used")
    for a in ax:
        a.set_xlabel("$n$")
        a.tick_params(axis="x", labelsize=7)
    rc.savefig(fig, os.path.join(FIG, "r_inst_by_n.pdf"))

    rows = []
    for fam in rc.FAMS:
        s, a = v[v["class"] == fam], x[x["class"] == fam]
        rows.append([fam, len(a), len(s), f0(s.Zev.mean()), f1(s.opshare.mean()),
                     f1(s["share_free_choice_by_van_%"].mean()), f2(s.mean_queue_FCS_min.mean()),
                     f1(s.mean_time_to_charged_min.mean()), f2(s.vans_used.mean()), f0(s.T_min.iloc[0])
                     if "T_min" in s else ""])
    T = {"C1": 1236, "C2": 3390, "R1": 230, "R2": 1000, "RC1": 240, "RC2": 960}
    for r in rows:
        r[-1] = T[r[0]]
    tab("r_family", ["family", "instances", "valid", "$Z$/EV (Rs)", "operator (\\%)", "free-choice by van (\\%)",
                     "queue (min)", "to charged (min)", "vans", "$T$ (min)"], rows, size="small",
        caption="RL results by Solomon family (valid plans).", label="tab:family")
    terms = ["Z1_EV_drive_perceived", "Z2_van_drive", "Z3_dispatch", "Z4_operator_energy", "Z5_tariff_perceived",
             "Z6_time_perceived"]
    labs = ["$Z_1$ EV driving", "$Z_2$ van driving", "$Z_3$ van dispatch", "$Z_4$ operator energy", "$Z_5$ tariff",
            "$Z_6$ time"]
    cols = ["#8172B2", "#64B5CD", "#4C72B0", "#55A868", "#DD8452", "#C44E52"]
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    bottom = np.zeros(6)
    for t, l, c in zip(terms, labs, cols):
        vals = np.array([100 * v[v["class"] == f][t].sum() / v[v["class"] == f].Z_total.sum() for f in rc.FAMS])
        ax.bar(rc.FAMS, vals, bottom=bottom, color=c, label=l, width=0.6)
        bottom += vals
    ax.set_ylabel("share of $Z$ (%)")
    ax.legend(fontsize=6.5, bbox_to_anchor=(1.01, 0.5), loc="center left")
    rc.savefig(fig, os.path.join(FIG, "r_obj_family.pdf"))
    fv = v.groupby("class").size()
    N["ValidRone"] = str(int(fv.get("R1", 0)))
    N["ValidRCone"] = str(int(fv.get("RC1", 0)))
    return v


def ev_distributions(ev):
    fc = ev[ev.free]
    rows = []
    def add(lbl, s, spec=f1):
        s = s.dropna()
        rows.append([lbl, ci(len(s)), spec(s.mean()), spec(s.median()), spec(s.quantile(.1)), spec(s.quantile(.9)),
                     spec(s.max())])
    add("income $\\theta$", ev.theta, f2)
    add("energy demand $E_i$ (kWh)", ev.E)
    add("nearest reachable FCS (km)", fc.nearest_fcs_km, f2)
    add("demand pressure", fc.pressure, f2)
    fcs = ev[~ev.van]
    van = ev[ev.van]
    add("drive to FCS (km), FCS users", fcs.drive_km, f2)
    add("FCS queue (min), FCS users", fcs.queue_min, f2)
    add("van wait (min), van users", van.van_wait_min)
    add("charging time (min)", ev.charge_min)
    add("request to charged (min)", ev.time_to_charged_min)
    add("money paid (Rs)", ev.pay_Rs, f0)
    add("perceived money (Rs)", ev.perceived_money_Rs, f0)
    add("perceived time (Rs)", ev.time_cost_Rs, f0)
    add("perceived total (Rs)", ev.customer_cost_Rs, f0)
    tab("r_ev_stats", ["variable (per EV)", "count", "mean", "median", "10th pct", "90th pct", "max"], rows, size="small",
        caption="EV-level variables over all EVs of the valid RL plans (unless stated).", label="tab:evstats")

    fig, ax = plt.subplots(2, 3, figsize=(10, 5.2))
    a = ax[0, 0]
    for s, c in rc.SEG_COL.items():
        g = fc[fc.segment == s]
        a.hist(g[g.van].theta, bins=20, range=(0, 1), density=True, alpha=0.6, color=c, label=f"{s}, van")
    a.hist(fc[~fc.van].theta, bins=20, range=(0, 1), density=True, histtype="step", color="k", label="all FCS users")
    a.set_title("income of free-choice van users")
    a.set_xlabel("income $\\theta$")
    a.legend(fontsize=6.5)
    a = ax[0, 1]
    a.hist(ev[~ev.van].time_to_charged_min, bins=40, range=(0, 250), alpha=0.6, color=C_M, label="FCS")
    a.hist(ev[ev.van].time_to_charged_min, bins=40, range=(0, 250), alpha=0.6, color=C_RL, label="van")
    a.set_title("time until fully charged")
    a.set_xlabel("request to charged (min)")
    a.legend(fontsize=7)
    a = ax[0, 2]
    q = ev[(~ev.van) & (ev.queue_min > 1e-6)]
    for s, c in rc.SEG_COL.items():
        a.hist(q[q.segment == s].queue_min, bins=30, range=(0, 60), alpha=0.6, color=c, label=s)
    a.set_title("FCS queues above zero")
    a.set_xlabel("queue (min)")
    a.legend(fontsize=7)
    a = ax[1, 0]
    a.hist(van.van_wait_min, bins=40, color=C_RL)
    a.set_title("van waits (all van users)")
    a.set_xlabel("wait for the van (min)")
    a = ax[1, 1]
    for s, c in rc.SEG_COL.items():
        a.hist(ev[ev.segment == s].pay_Rs, bins=40, alpha=0.6, color=c, label=s)
    a.set_title("money paid")
    a.set_xlabel("money paid (Rs)")
    a.legend(fontsize=7)
    a = ax[1, 2]
    for s, c in rc.SEG_COL.items():
        a.hist(ev[ev.segment == s].customer_cost_Rs, bins=40, alpha=0.6, color=c, label=s)
    a.set_title("perceived cost")
    a.set_xlabel("perceived cost (Rs)")
    a.legend(fontsize=7)
    for a in ax[:, 0]:
        a.set_ylabel("EVs / density")
    rc.savefig(fig, os.path.join(FIG, "r_ev_dists.pdf"))
    fv = fc[fc.van]
    N["CommShareVanFree"] = f1(100 * (fv.segment == "commercial").mean())
    N["ThetaVanComm"] = f2(fv[fv.segment == "commercial"].theta.mean())
    N["QueueersFcs"] = ci(int(((~ev.van) & ev.free & (ev.queue_min > 1e-6)).sum()))
    N["FcsUsersFree"] = ci(int(((~ev.van) & ev.free).sum()))
    N["VanWaitZero"] = f1(100 * (van.van_wait_min <= 1e-6).mean())
    N["ToChargedLo"] = f0(ev.time_to_charged_min.quantile(.1))
    N["ToChargedHi"] = f0(ev.time_to_charged_min.quantile(.9))

    cols = {"$\\theta$": fc.theta, "commercial": fc.seg, "$E$": fc.E, "nearest FCS": fc.nearest_fcs_km,
            "pressure": fc.pressure, "van": fc.van.astype(float), "queue": fc.queue_min.fillna(0),
            "to charged": fc.time_to_charged_min, "paid": fc.pay_Rs, "perc. money": fc.perceived_money_Rs,
            "perc. time": fc.time_cost_Rs}
    df = pd.DataFrame(cols)
    rho = df.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(rho.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(rho)))
    ax.set_yticks(range(len(rho)))
    ax.set_xticklabels(rho.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(rho.columns, fontsize=7)
    for i in range(len(rho)):
        for j in range(len(rho)):
            ax.text(j, i, f"{rho.values[i, j]:.2f}", ha="center", va="center", fontsize=5)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.savefig(os.path.join(FIG, "r_corr.pdf"), bbox_inches="tight")
    plt.close(fig)
    for a, b, nm in (("$\\theta$", "van", "RhoThetaVan"), ("commercial", "van", "RhoSegVan"),
                     ("pressure", "queue", "RhoPressQueue"), ("$E$", "paid", "RhoEPaid"),
                     ("$\\theta$", "queue", "RhoThetaQueue"), ("nearest FCS", "van", "RhoDistVan")):
        N[nm] = f"{rho.loc[a, b]:.2f}"


# ====================================================================== quality
def quality(x, ev_r, plan_m, info):
    p = x[x.valid_plan & x.valid_plan_m].copy()
    p["gap"] = 100 * (p.Z_total - p.Z_total_m) / p.Z_total_m
    g = x[x.valid_plan & x.valid_plan_g].copy()
    g["gapg"] = 100 * (g.Z_total - g.Z_total_g) / g.Z_total_g
    ns = sorted(x.n.unique())
    rows = []
    for n in ns:
        s, so, st, sg = p[p.n == n], p[(p.n == n) & (p.status_m == "optimal")], p[(p.n == n) & (p.status_m == "time_limit")], g[g.n == n]
        rows.append([n, len(s), f2(s.gap.mean()), f2(s.gap.median()), f2(so.gap.mean()), f2(st.gap.mean()),
                     (s.gap < -1e-6).sum(), (s.gap.abs() <= 0.01).sum(), f2(sg.gapg.mean())])
    so, st = p[p.status_m == "optimal"], p[p.status_m == "time_limit"]
    tot = ["all", len(p), f2(p.gap.mean()), f2(p.gap.median()), f2(so.gap.mean()), f2(st.gap.mean()),
           (p.gap < -1e-6).sum(), (p.gap.abs() <= 0.01).sum(), f2(g.gapg.mean())]
    tab("r_quality_n", ["$n$", "paired", "gap mean", "gap median", "vs optimal", "vs time-limit", "RL cheaper",
                        "equal ($\\le$0.01\\%)", "vs greedy"], rows, total_row=tot, size="small",
        caption="Gap of RL to the MILP (\\%) by instance size over instances where both are valid; vs optimal / "
                "vs time-limit = only MILP plans of that status; RL cheaper = negative gap; last column: gap to the "
                "greedy rule.", label="tab:qualityn")
    rows = []
    for fam in rc.FAMS:
        s, sg = p[p["class"] == fam], g[g["class"] == fam]
        rows.append([fam, len(s), f2(s.gap.mean()), f2(s.gap.median()), f2(s[s.status_m == "optimal"].gap.mean()),
                     (s.gap < -1e-6).sum(), len(sg), f2(sg.gapg.mean()), f1(100 * (sg.gapg < 0).mean())])
    tab("r_quality_class", ["family", "paired", "gap mean", "gap median", "vs optimal", "RL cheaper",
                            "paired greedy", "vs greedy", "beats greedy (\\%)"], rows, size="small",
        caption="Gap of RL to the MILP and to the greedy rule by family (\\%).", label="tab:qualityclass")

    fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.9))
    ax[0].plot(ns, [p[p.n == n].gap.mean() for n in ns], "o-", color="k", label="all paired")
    ax[0].plot(ns, [p[(p.n == n) & (p.status_m == "optimal")].gap.mean() for n in ns], "s--", color="#55A868",
               label="MILP optimal")
    ax[0].plot(ns, [p[(p.n == n) & (p.status_m == "time_limit")].gap.mean() for n in ns], "^--", color="#DD8452",
               label="MILP time limit")
    ax[0].axhline(0, color="grey", lw=0.6)
    ax[0].set_xlabel("$n$")
    ax[0].set_ylabel("mean gap to MILP (%)")
    ax[0].legend(fontsize=7)
    ax[1].hist(p.gap.clip(-3, 6), bins=45, color=C_RL)
    ax[1].axvline(0, color="k", lw=0.6)
    ax[1].set_xlabel("gap to MILP (%), clipped to [-3, 6]")
    ax[1].set_ylabel("instances")
    ax[2].plot(ns, [g[g.n == n].gapg.mean() for n in ns], "o-", color=C_G)
    ax[2].axhline(0, color="grey", lw=0.6)
    ax[2].set_xlabel("$n$")
    ax[2].set_ylabel("mean gap to greedy rule (%)")
    rc.savefig(fig, os.path.join(FIG, "r_gap.pdf"))

    N["NPaired"] = str(len(p))
    N["GapMean"] = f2(p.gap.mean())
    N["GapMed"] = f2(p.gap.median())
    N["GapP"] = f2(p.gap.quantile(.9))
    N["GapMax"] = f2(p.gap.max())
    N["GapOpt"] = f2(so.gap.mean())
    N["GapTl"] = f2(st.gap.mean())
    N["NPairedOpt"] = str(len(so))
    N["RLbetter"] = str(int((p.gap < -1e-6).sum()))
    N["RLbetterTl"] = str(int((st.gap < -1e-6).sum()))
    N["RLequal"] = str(int((p.gap.abs() <= 0.01).sum()))
    N["RLequalOpt"] = str(int((so.gap.abs() <= 0.01).sum()))
    N["RLbelowOpt"] = str(int((so.gap < -1e-6).sum()))
    N["GapTen"] = f2(p[p.n == 10].gap.mean())
    N["GapHundred"] = f2(p[p.n == 100].gap.mean())
    gn = {n: p[p.n == n].gap.mean() for n in ns}
    nmax = max(gn, key=gn.get)
    N["GapMaxN"] = str(nmax)
    N["GapMaxNVal"] = f2(gn[nmax])
    fg = {f: p[p["class"] == f].gap.mean() for f in rc.FAMS}
    N["GapBestFam"] = min(fg, key=fg.get)
    N["GapBestFamVal"] = f2(min(fg.values()))
    N["GapWorstFam"] = max(fg, key=fg.get)
    N["GapWorstFamVal"] = f2(max(fg.values()))
    N["GreedyN"] = str(len(g))
    N["GreedyBeat"] = str(int((g.gapg < -1e-6).sum()))
    N["GreedyGap"] = f2(g.gapg.mean())
    N["GreedyGapAbs"] = f2(-g.gapg.mean())
    N["GreedyWorse"] = str(int((g.gapg > 1e-6).sum()))
    return p


# ====================================================================== computation
def computation(x, meta):
    ns = sorted(x.n.unique())
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    ax.plot(ns, [x[x.n == n].solve_s.mean() for n in ns], "o-", color=C_RL, label="RL (inference)")
    ax.plot(ns, [x[x.n == n].solve_s_m.mean() for n in ns], "s-", color=C_M, label="MILP (Gurobi, 300 s limit)")
    ax.set_yscale("log")
    ax.set_xlabel("$n$")
    ax.set_ylabel("mean seconds per instance")
    ax.legend(fontsize=7)
    rc.savefig(fig, os.path.join(FIG, "r_time.pdf"))
    rl, mi = x.solve_s.mean(), x.solve_s_m.mean()
    N["RLtime"] = f3(rl)
    N["RLtimeMax"] = f3(x.solve_s.max())
    N["MILPtime"] = f1(mi)
    N["Speedup"] = ci(round(mi / rl, -1))
    N["RLtotalS"] = f1(x.solve_s.sum())
    N["MILPtotalH"] = f1(x.solve_s_m.sum() / 3600)
    N["BreakEven"] = f0(meta["train_s"] / (mi - rl))
    N["RLtimeHundred"] = f3(x[x.n == 100].solve_s.mean())
    N["MILPtimeHundred"] = f0(x[x.n == 100].solve_s_m.mean())


# ====================================================================== objective anatomy and elasticities
TERMS = ["Z1_EV_drive_perceived", "Z2_van_drive", "Z3_dispatch", "Z4_operator_energy", "Z5_tariff_perceived",
         "Z6_time_perceived"]
TLAB = ["$Z_1$ EV driving", "$Z_2$ van driving", "$Z_3$ van dispatch", "$Z_4$ operator energy",
        "$Z_5$ tariff (perceived)", "$Z_6$ time (perceived)"]


def anatomy(p, ev_r, ev_m, plan_r, plan_m, meta):
    Zm = p.Z_total_m.sum()
    mm = p[["instance", "n"]].merge(
        pd.read_csv(os.path.join(MILP_DIR, "runs.csv"), low_memory=False).query("run_id=='milp_300s'"),
        on=["instance", "n"])
    rows = []
    diffs = []
    for t, l in zip(TERMS, TLAB):
        r_, m_ = p[t].sum(), mm[t].sum()
        diffs.append(100 * (r_ - m_) / Zm)
        rows.append([l, f1(100 * r_ / p.Z_total.sum()), f1(100 * m_ / Zm), f2(100 * (r_ - m_) / Zm)])
    rows.append(["total $Z$", "100.0", "100.0", f2(100 * (p.Z_total.sum() - Zm) / Zm)])
    tab("r_terms", ["term", "RL share of $Z$ (\\%)", "MILP share (\\%)", "RL $-$ MILP (\\% of MILP $Z$)"], rows,
        caption="Composition of $Z$ over the paired instances and where the RL's extra cost comes from.",
        label="tab:terms")
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    ax.barh(TLAB[::-1], diffs[::-1], color=[C_RL if d > 0 else C_M for d in diffs[::-1]])
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("RL minus MILP, % of the MILP's Z (paired instances)")
    rc.savefig(fig, os.path.join(FIG, "r_terms.pdf"))
    for k, nm in enumerate(["DOne", "DTwo", "DThree", "DFour", "DFive", "DSix"]):
        N["Term" + nm] = f2(diffs[k])
    N["OpShareRL"] = f1(100 * p[["Z2_van_drive", "Z3_dispatch", "Z4_operator_energy"]].sum().sum() / p.Z_total.sum())
    N["TariffShareRL"] = f1(100 * p.Z5_tariff_perceived.sum() / p.Z_total.sum())
    N["TimeShareRL"] = f1(100 * p.Z6_time_perceived.sum() / p.Z_total.sum())

    # elasticities, pooled over the valid plans of each method
    P = meta["f4_params"] if "f4_params" in meta else None
    tm = json.load(open(os.path.join(RL_MODEL, "train_meta.json")))
    P = P or tm["f4_params"]
    cfg = pd.read_csv(os.path.join(rc.F4_DATA, "solomon_config.csv")).iloc[0]

    def quantities(ev, plan, runs_valid):
        q = {}
        fcs, van = ev[~ev.van], ev[ev.van]
        q["c^{\\mathrm{EV}}"] = (cfg.c_EV_Rs_per_km, (ev.mu * ev.drive_km).sum())
        vr = plan[(plan.type == "van")].merge(runs_valid, on=["instance", "n"])
        q["c^{\\mathrm M}"] = (cfg.c_M_Rs_per_km, vr.km.sum())
        q["c^{\\mathrm F}_m"] = (cfg.c_F_Rs_per_van, len(vr))
        q["\\pi^{\\mathrm F}"] = (cfg.pi_F_Rs_per_kWh, fcs.E.sum())
        q["\\pi^{\\mathrm M}"] = (cfg.pi_M_Rs_per_kWh, (van.E / cfg.van_efficiency).sum())
        q["p^{\\mathrm F}"] = (P["tariff_Rs_per_kWh"]["p_F"], (fcs.mu * fcs.E).sum())
        q["p^{\\mathrm M}"] = (P["tariff_Rs_per_kWh"]["p_M"], (van.mu * van.E).sum())
        for s, sl in (("personal", "P"), ("commercial", "C")):
            e = ev[ev.segment == s]
            S = P[s]
            for end, w in (("0", 1 - e.theta), ("1", e.theta)):
                q[f"\\mu^{{\\mathrm{{{sl},{end}}}}}"] = (S["money_sensitivity_mu"]["theta" + end], (w * e.pay_Rs).sum())
                ef, evn = e[~e.van], e[e.van]
                wf, wv = (1 - ef.theta) if end == "0" else ef.theta, (1 - evn.theta) if end == "0" else evn.theta
                vt = S["value_of_time_Rs_per_min"]
                q[f"c^{{\\mathrm{{D,{sl},{end}}}}}"] = (vt["D"]["theta" + end], (wf * ef.drive_min).sum())
                q[f"c^{{\\mathrm{{Q,{sl},{end}}}}}"] = (vt["Q"]["theta" + end], (wf * ef.queue_min).sum())
                q[f"c^{{\\mathrm{{C,{sl},{end}}}}}"] = (vt["C"]["theta" + end], (wf * ef.charge_min).sum())
                q[f"c^{{\\mathrm{{V,{sl},{end}}}}}"] = (vt["V"]["theta" + end], (wv * (evn.van_wait_min + evn.charge_min)).sum())
        return q

    rv = p[["instance", "n"]]
    qr = quantities(ev_r.merge(rv, on=["instance", "n"]), plan_r, rv)
    qm = quantities(ev_m.merge(rv, on=["instance", "n"]), plan_m, rv)
    Zr, Zm2 = p.Z_total.sum(), Zm
    el = pd.DataFrame([{"coef": k, "value": qr[k][0], "el_rl": 100 * qr[k][0] * qr[k][1] / Zr,
                        "el_m": 100 * qm[k][0] * qm[k][1] / Zm2} for k in qr])
    N["ElSumRL"] = f1(el.el_rl.sum())
    N["MoneyShareRL"] = f1(100 * (p.Z1_EV_drive_perceived.sum() + p.Z5_tariff_perceived.sum()) / Zr)
    N["ElSumM"] = f1(el.el_m.sum())
    el = el.sort_values("el_rl", ascending=False)
    rows = [[f"${r.coef}$", f"{r.value:g}", f2(r.el_rl), f2(r.el_m), f2(r.el_rl - r.el_m)] for r in el.itertuples()]
    tab("r_elasticity", ["coefficient", "value", "elasticity RL (\\%)", "elasticity MILP (\\%)", "difference"], rows,
        size="small", caption="Elasticity of $Z$ (share of $Z$ carried by each coefficient, envelope theorem) in the "
                              "RL plans and the MILP plans of the paired instances. Superscripts: activity, segment "
                              "(P personal, C commercial), end point ($\\theta=0$ or 1).", label="tab:elasticity")
    top = el.head(12)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    y = np.arange(len(top))[::-1]
    ax.barh(y + 0.2, top.el_rl, height=0.4, color=C_RL, label="RL")
    ax.barh(y - 0.2, top.el_m, height=0.4, color=C_M, label="MILP")
    ax.set_yticks(y)
    ax.set_yticklabels(["$" + mathtext(c) + "$" for c in top.coef], fontsize=8)
    ax.set_xlabel("elasticity of Z (%)")
    ax.legend(fontsize=7)
    rc.savefig(fig, os.path.join(FIG, "r_elasticity.pdf"))
    t3 = el.head(4)
    N["ElTopOne"] = f"${t3.iloc[0].coef}$"
    N["ElTopOneV"] = f1(t3.iloc[0].el_rl)
    N["ElTopTwo"] = f"${t3.iloc[1].coef}$"
    N["ElTopTwoV"] = f1(t3.iloc[1].el_rl)
    N["ElTopThree"] = f"${t3.iloc[2].coef}$"
    N["ElTopThreeV"] = f1(t3.iloc[2].el_rl)
    N["ElTopFour"] = f"${t3.iloc[3].coef}$"
    N["ElTopFourV"] = f1(t3.iloc[3].el_rl)
    dmax = el.iloc[(el.el_rl - el.el_m).abs().argmax()]
    N["ElDiffCoef"] = f"${dmax.coef}$"
    N["ElDiffRL"] = f2(dmax.el_rl)
    N["ElDiffM"] = f2(dmax.el_m)


# ====================================================================== agreement and income
def agreement(p, ev_r, ev_m):
    key = ["instance", "n", "id"]
    a = ev_r[ev_r.free].merge(p[["instance", "n"]], on=["instance", "n"])[key + ["van", "resource", "segment", "theta"]]
    b = ev_m[ev_m.free][key + ["van", "resource"]]
    j = a.merge(b, on=key, suffixes=("_r", "_m"))
    j["st_r"] = j.resource_r.str.split("-").str[0]
    j["st_m"] = j.resource_m.str.split("-").str[0]
    rows = []
    for s in ("personal", "commercial", "all"):
        g = j if s == "all" else j[j.segment == s]
        rows.append([s, ci(len(g)), ci(int((~g.van_r & ~g.van_m).sum())), ci(int((~g.van_r & g.van_m).sum())),
                     ci(int((g.van_r & ~g.van_m).sum())), ci(int((g.van_r & g.van_m).sum())),
                     f1(100 * (g.van_r == g.van_m).mean()),
                     f1(100 * ((~g.van_r & ~g.van_m & (g.st_r == g.st_m)).sum() / max(1, (~g.van_r & ~g.van_m).sum())))])
    tab("r_confusion", ["segment", "EVs", "both FCS", "RL FCS, MILP van", "RL van, MILP FCS", "both van",
                        "same type (\\%)", "same station (\\%)"], rows, size="small",
        caption="Decisions of RL and MILP for the same free-choice EV (paired instances). Same station: share of EVs "
                "both send to a station that go to the same station.", label="tab:confusion")
    N["AgreeType"] = f1(100 * (j.van_r == j.van_m).mean())
    N["AgreeStation"] = f1(100 * ((~j.van_r & ~j.van_m & (j.st_r == j.st_m)).sum() / (~j.van_r & ~j.van_m).sum()))
    N["MilpVanRlFcs"] = ci(int((~j.van_r & j.van_m).sum()))
    N["RlVanMilpFcs"] = ci(int((j.van_r & ~j.van_m).sum()))
    jp = j[j.segment == "personal"]
    N["AgreePers"] = f1(100 * (jp.van_r == jp.van_m).mean())
    jm = j[j.van_m & (j.segment == "personal")]
    N["MilpVanPersThetaMean"] = f2(jm.theta.mean())
    N["MilpVanPersKeptRl"] = f1(100 * jm.van_r.mean())


def typical_threshold(P, cfg):
    """Delta(theta) of the typical EV of the MILP report (E 20 kWh, 2 km / 7.7 min to the station, no queue,
    10 min van wait, 4 km van detour), per segment."""
    E, d, t, qM, x = 20.0, 2.0, 7.7, 10.0, 4.0
    tF, tM = 60 * E / 22.0, 60 * E / 30.0
    pF, pM = P["tariff_Rs_per_kWh"]["p_F"], P["tariff_Rs_per_kWh"]["p_M"]
    Pi = (pM - pF) * E - cfg.c_EV_Rs_per_km * d
    kappa = (cfg.pi_M_Rs_per_kWh / cfg.van_efficiency - cfg.pi_F_Rs_per_kWh) * E + cfg.c_M_Rs_per_km * x
    out = {}
    th = np.linspace(0, 1, 101)
    for s in ("personal", "commercial"):
        S = P[s]
        L = lambda ln: ln["theta0"] + (ln["theta1"] - ln["theta0"]) * th  # noqa: E731
        vt = S["value_of_time_Rs_per_min"]
        delta = L(S["money_sensitivity_mu"]) * Pi + kappa - L(vt["D"]) * t - L(vt["C"]) * tF + L(vt["V"]) * (qM + tM)
        star = np.nan
        if delta[0] > 0 > delta[-1]:
            star = float(np.interp(0, -delta, th))
        out[s] = (th, delta, star, delta + cfg.c_F_Rs_per_van)
    return out


def income(ev_r, ev_m, meta):
    tm = json.load(open(os.path.join(RL_MODEL, "train_meta.json")))
    P = tm["f4_params"]
    cfg = pd.read_csv(os.path.join(rc.F4_DATA, "solomon_config.csv")).iloc[0]
    thr = typical_threshold(P, cfg)
    N["ThetaStarP"] = f2(thr["personal"][2])
    N["DeltaZeroP"] = f0(thr["personal"][1][0])
    N["DeltaOneP"] = f0(thr["personal"][1][-1])
    N["DeltaZeroC"] = f0(thr["commercial"][1][0])
    N["DeltaOneC"] = f0(thr["commercial"][1][-1])
    N["DeltaOnePDispatch"] = f0(thr["personal"][3][-1])

    fr, fm = ev_r[ev_r.free], ev_m[ev_m.free]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
    th, dP, sP, dPd = thr["personal"]
    ax[0].plot(th, dP, color=rc.SEG_COL["personal"], label="personal")
    ax[0].plot(th, thr["commercial"][1], color=rc.SEG_COL["commercial"], label="commercial")
    ax[0].plot(th, dPd, color=rc.SEG_COL["personal"], ls="--", lw=0.8, label="personal, van not yet sent")
    ax[0].axhline(0, color="grey", lw=0.6)
    ax[0].set_xlabel("income $\\theta$")
    ax[0].set_ylabel("$\\Delta(\\theta)$: extra cost of a van (Rs)")
    ax[0].legend(fontsize=6.5)
    mids = (np.arange(10) + 0.5) / 10
    rows = []
    for k, s in enumerate(("personal", "commercial")):
        a = ax[k + 1]
        for lab, f, c, mk in (("RL", fr, C_RL, "o"), ("MILP", fm, C_M, "s")):
            g = f[f.segment == s]
            sh, lo, hi = [], [], []
            for d in range(10):
                gg = g[g.decile == d]
                kk, nn = int(gg.van.sum()), len(gg)
                sh.append(100 * kk / nn if nn else np.nan)
                w = rc.wilson(kk, nn)
                lo.append(100 * w[0])
                hi.append(100 * w[1])
            a.plot(mids, sh, marker=mk, color=c, label=lab, ms=3)
            a.fill_between(mids, lo, hi, color=c, alpha=0.15, lw=0)
        if s == "personal" and not np.isnan(sP):
            a.axvline(sP, color="k", ls="--", lw=0.8, label=f"theory $\\theta^*={sP:.2f}$")
        a.set_title(f"{s} free-choice EVs")
        a.set_xlabel("income $\\theta$ (decile midpoint)")
        a.set_ylabel("served by a van (%)")
        a.legend(fontsize=7)
    rc.savefig(fig, os.path.join(FIG, "r_van_income.pdf"))

    for s in ("personal", "commercial"):
        for b in range(5):
            gr = fr[(fr.segment == s) & (fr.band == b)]
            gm = fm[(fm.segment == s) & (fm.band == b)]
            w = rc.wilson(int(gr.van.sum()), len(gr))
            rows.append([s if b == 0 else "", rc.BLAB[b], ci(len(gr)), f1(100 * gr.van.mean()),
                         f"[{max(0.0, 100 * w[0]):.1f}, {100 * w[1]:.1f}]", ci(len(gm)), f1(100 * gm.van.mean())])
    tab("r_vanband", ["segment", "income $\\theta$", "EVs (RL)", "by van RL (\\%)", "95\\% interval", "EVs (MILP)",
                      "by van MILP (\\%)"], rows, align="llrrrrr", size="small",
        caption="Van share of free-choice EVs by segment and income band; 95\\,\\% Wilson interval for RL; all valid "
                "plans of each method.", label="tab:vanband")
    for s, sl in (("personal", "Pers"), ("commercial", "Comm")):
        N["VanR" + sl] = f1(100 * fr[fr.segment == s].van.mean())
        N["VanM" + sl] = f1(100 * fm[fm.segment == s].van.mean())
        N["VanTopR" + sl] = f1(100 * fr[(fr.segment == s) & (fr.band == 4)].van.mean())
        N["VanTopM" + sl] = f1(100 * fm[(fm.segment == s) & (fm.band == 4)].van.mean())
        N["VanLowR" + sl] = f1(100 * fr[(fr.segment == s) & (fr.band <= 1)].van.mean())
    N["VanRAll"] = f1(100 * fr.van.mean())
    N["VanMAll"] = f1(100 * fm.van.mean())
    N["ThetaVanR"] = f2(fr[fr.van & (fr.segment == "personal")].theta.mean())
    N["ThetaFcsR"] = f2(fr[~fr.van & (fr.segment == "personal")].theta.mean())
    # where does the personal van share of RL first exceed 5 %?
    g = fr[fr.segment == "personal"].groupby("decile").van.mean()
    above = g[g > 0.05]
    N["StepR"] = f"{above.index.min() / 10:.1f}--{above.index.min() / 10 + 0.1:.1f}" if len(above) else "--"
    g = fm[fm.segment == "personal"].groupby("decile").van.mean()
    above = g[g > 0.05]
    N["StepM"] = f"{above.index.min() / 10:.1f}--{above.index.min() / 10 + 0.1:.1f}" if len(above) else "--"

    # logistic regression (RL, free-choice EVs)
    terc = np.array(infra.pressure_cuts())
    N["PressLo"] = f"{terc[0]:.4f}"
    N["PressHi"] = f"{terc[1]:.4f}"
    df = fr.dropna(subset=["pressure", "nearest_fcs_km"]).copy()
    hi = (df.pressure > terc[1]).astype(float)
    X = np.column_stack([np.ones(len(df)), df.theta, df.seg, df.theta * df.seg, hi, df.theta * hi,
                         df.E / 10, df.nearest_fcs_km])
    names = ["intercept", "income $\\theta$", "commercial", "$\\theta\\times$ commercial", "high pressure",
             "$\\theta\\times$ high pressure", "energy $E_i$ (per 10 kWh)", "nearest reachable FCS (km)"]
    y = df.van.astype(float).values
    beta = np.zeros(X.shape[1])
    for _ in range(50):
        eta = X @ beta
        pr = 1 / (1 + np.exp(-eta))
        Wt = pr * (1 - pr)
        H = X.T @ (X * Wt[:, None]) + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - pr))
        beta += step
        if np.abs(step).max() < 1e-9:
            break
    cov = np.linalg.inv(H)
    se = np.sqrt(np.diag(cov))
    z = beta / se
    pv = 2 * stats.norm.sf(np.abs(z))
    rows = [[nm, f"{b:.3f}", f"{s:.3f}", f"{zz:.1f}", "$<0.001$" if p_ < 0.001 else f"{p_:.3f}"]
            for nm, b, s, zz, p_ in zip(names, beta, se, z, pv)]
    tab("r_logit", ["term", "coefficient", "std. error", "$z$", "$p$-value"], rows, align="lrrrr", size="small",
        caption=f"Logistic regression of $P$(served by a van) over the {ci(len(df))} free-choice EVs of the valid RL "
                "plans.", label="tab:logit")
    pr = 1 / (1 + np.exp(-(X @ beta)))
    for seg, nm in ((0, "AmePers"), (1, "AmeComm")):
        msk = df.seg.values == seg
        dth = beta[1] + beta[3] * seg + beta[5] * hi.values[msk]
        v_ = 10 * np.mean(pr[msk] * (1 - pr[msk]) * dth)   # percentage points per +0.1 of theta
        N[nm] = f1(0.0 if abs(v_) < 0.05 else v_)
    N["LogitAcc"] = f1(100 * np.mean((pr > 0.5) == (y > 0.5)))
    rk = stats.rankdata(pr)
    n1 = y.sum()
    N["LogitAUC"] = f2((rk[y > 0.5].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))
    N["LogitBaseRate"] = f1(100 * (1 - y.mean()))
    return terc


# ====================================================================== duals
def duals():
    dc = pd.read_csv(os.path.join(DATA, "duals_constraints.csv"))
    di = pd.read_csv(os.path.join(DATA, "duals_instances.csv"))
    names = {"arrival": ("FCS arrival $S_i\\ge r_i+t_{if}$", "not unique (degenerate); not interpreted"),
             "release": ("van release $S_i\\ge r_i$", "not unique (degenerate); not interpreted"),
             "gun": ("gun order $S_j\\ge S_i+\\tau^{\\mathrm F}_i$", "value of one gun-minute"),
             "van": ("van order $S_j\\ge S_i+\\tau^{\\mathrm M}_i+t_{ij}$", "value of one van-minute"),
             "depart": ("first stop $S_a\\ge t_{0a}$", "value of an earlier van departure"),
             "patience": ("queue limit $q^{\\mathrm F}_i\\le W_i$", "one more minute of patience"),
             "deadline": ("deadline $C_i\\le T$, van return", "one more minute of horizon")}
    rows = []
    dc["kind2"] = dc.kind.replace({"deadline_fcs": "deadline", "deadline_van": "deadline", "return": "deadline"})
    for k, (lab, mean_) in names.items():
        row = [lab]
        for meth in ("RL", "MILP"):
            s = dc[(dc.kind2 == k) & (dc.method == meth)].dual.abs()
            nz = s[s > 1e-6]
            row += [ci(len(s)), f1(100 * len(nz) / max(1, len(s))), f2(nz.mean()) if len(nz) else "0.00"]
        rows.append(row + [mean_])
    tab("r_duals", ["constraint", "rows RL", "non-zero (\\%)", "mean if $\\neq0$", "rows MILP", "non-zero (\\%)",
                    "mean if $\\neq0$", "meaning"], rows, align="lrrrrrrp{3.2cm}", size="footnotesize",
        caption="Shadow prices of the fixed-plan LPs (absolute values, Rs/min) of every valid RL and MILP plan.",
        label="tab:duals")
    g = dc[dc.kind == "gun"].copy()
    g["bind"] = g.dual.abs() > 1e-6
    for meth, sfx in (("RL", "R"), ("MILP", "M")):
        s = g[g.method == meth]
        N["GunLinks" + sfx] = ci(len(s))
        N["GunBind" + sfx] = f1(100 * s.bind.mean())
        N["GunPrice" + sfx] = f1(s[s.bind].dual.abs().mean())
        N["GunPriceMed" + sfx] = f1(s[s.bind].dual.abs().median())
        v = dc[(dc.kind == "van") & (dc.method == meth)]
        N["VanBind" + sfx] = f1(100 * (v.dual.abs() > 1e-6).mean())
        N["VanPrice" + sfx] = f2(v[v.dual.abs() > 1e-6].dual.abs().mean())
        pt = dc[(dc.kind == "patience") & (dc.method == meth)]
        N["PatNZ" + sfx] = str(int((pt.dual.abs() > 1e-6).sum()))
        dl = dc[(dc.kind2 == "deadline") & (dc.method == meth)]
        N["DeadNZ" + sfx] = str(int((dl.dual.abs() > 1e-6).sum()))
        d = di[di.method == meth]
        N["PowerF" + sfx] = f0(-d.dZ_dPF.mean())
        N["PowerFMed" + sfx] = f0(-d.dZ_dPF.median())
        N["PowerM" + sfx] = f0(-d.dZ_dPM.mean())
        N["PowerCong" + sfx] = f1(100 * d.congestion_part_PF.sum() / d.dZ_dPF.sum())
        N["LPPlans" + sfx] = str(len(d))
        N["LPmaxdiff" + sfx] = f"{d.lp_minus_plan.abs().max():.1e}"
    # gun links by pressure of the follower (shared cut points of ../common/infra.py)
    terc = np.array(infra.pressure_cuts())
    rows = []
    for meth in ("RL", "MILP"):
        s = g[g.method == meth]
        for lab, msk in (("low", s.pressure_j <= terc[0]), ("medium", (s.pressure_j > terc[0]) & (s.pressure_j <= terc[1])),
                         ("high", s.pressure_j > terc[1])):
            t = s[msk]
            rows.append([meth if lab == "low" else "", lab, ci(len(t)), f1(100 * t.bind.mean()),
                         f1(t[t.bind].dual.abs().mean()), f2(t.dual.abs().mean())])
    tab("r_gun_pressure", ["method", "pressure of the EV behind", "gun links", "binding (\\%)", "mean dual if binding",
                           "mean dual, all links"], rows, align="llrrrr", size="small",
        caption="Gun links (EV $i$ then EV $j$ on one gun) by the demand-pressure tercile of the EV behind "
                "(shared cut points, Section~\\ref{sec:defs}).", label="tab:gunpressure")
    s = g[g.method == "RL"]
    N["GunBindHighR"] = f1(100 * s[s.pressure_j > terc[1]].bind.mean())
    N["GunBindLowR"] = f1(100 * s[s.pressure_j <= terc[0]].bind.mean())

    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
    for meth, c in (("RL", C_RL), ("MILP", C_M)):
        s = g[(g.method == meth) & g.bind]
        ax[0].hist(s.dual.abs(), bins=40, range=(0, 60), color=c, alpha=0.55, label=meth)
    ax[0].set_xlabel("gun-minute shadow price (Rs/min)")
    ax[0].set_ylabel("binding gun links")
    ax[0].legend(fontsize=7)
    s = g[g.method == "RL"].copy()
    s["bi"] = rc.band_of(s.theta_i)
    s["bj"] = rc.band_of(s.theta_j.astype(float))
    x = np.arange(5)
    ax[1].bar(x - 0.2, [100 * s[s.bi == b].bind.mean() for b in range(5)], 0.4, color=C_M, label="EV on the gun ($i$)")
    ax[1].bar(x + 0.2, [100 * s[s.bj == b].bind.mean() for b in range(5)], 0.4, color="#DD8452",
              label="EV behind it ($j$)")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(rc.PLAB, rotation=30, fontsize=7)
    ax[1].set_xlabel("income band $\\theta$")
    ax[1].set_ylabel("RL gun links binding (%)")
    ax[1].legend(fontsize=7)
    segs = [("P", 0), ("C", 1)]
    M = np.array([[100 * s[(s.segment_i == a) & (s.segment_j.astype(float) == b)].bind.mean() for b in (0, 1)]
                  for a in (0, 1)])
    im = ax[2].imshow(M, cmap="Reds", vmin=0)
    for i in range(2):
        for j in range(2):
            ax[2].text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=8,
                       color="white" if M[i, j] > 0.6 * np.nanmax(M) else "black")
    ax[2].set_xticks([0, 1])
    ax[2].set_yticks([0, 1])
    ax[2].set_xticklabels(["personal", "commercial"])
    ax[2].set_yticklabels(["personal", "commercial"])
    ax[2].set_xlabel("segment of the EV behind ($j$)")
    ax[2].set_ylabel("segment of the EV on the gun ($i$)")
    ax[2].set_title("RL gun links binding (%)", fontsize=8)
    rc.savefig(fig, os.path.join(FIG, "r_gun_duals.pdf"))
    N["BindRichOnGun"] = f1(100 * s[s.bi == 4].bind.mean())
    N["BindPoorOnGun"] = f1(100 * s[s.bi == 0].bind.mean())
    N["BindPoorBehind"] = f1(100 * s[s.bj == 0].bind.mean())
    N["BindRichBehind"] = f1(100 * s[s.bj == 4].bind.mean())
    N["BindPC"] = f1(M[0, 1])
    N["BindCP"] = f1(M[1, 0])
    N["BindCC"] = f1(M[1, 1])
    N["BindPP"] = f1(M[0, 0])

    fig, ax = plt.subplots(1, 2, figsize=(8.5, 2.8))
    ns = sorted(di.n.unique())
    for k, (col, ttl) in enumerate((("dZ_dPF", "one more kW of FCS gun power"), ("dZ_dPM", "one more kW of van power"))):
        for meth, c, off in (("RL", C_RL, -1.5), ("MILP", C_M, 1.5)):
            d = di[di.method == meth]
            ax[k].plot(ns, [-d[d.n == n][col].mean() for n in ns], "o-", color=c, label=meth, ms=3)
        ax[k].set_title(ttl, fontsize=9)
        ax[k].set_xlabel("$n$")
        ax[k].set_ylabel("Rs saved per instance (mean)")
        ax[k].legend(fontsize=7)
    rc.savefig(fig, os.path.join(FIG, "r_power.pdf"))


# ====================================================================== queues, pressure, horizon
def queues(ev_r, ev_m, terc):
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
    rows = []
    for k, s in enumerate(("personal", "commercial")):
        a = ax[k]
        f = ev_r[(~ev_r.van) & (ev_r.segment == s)]
        users = [100 * (f.band == b).mean() for b in range(5)]
        qmin = [100 * f[f.band == b].queue_min.sum() / max(1e-9, f.queue_min.sum()) for b in range(5)]
        x = np.arange(5)
        a.bar(x - 0.2, users, 0.4, color="#9DB4D6", label="share of FCS users")
        a.bar(x + 0.2, qmin, 0.4, color=C_RL, label="share of queue minutes")
        a.set_xticks(x)
        a.set_xticklabels(rc.PLAB, rotation=30, fontsize=7)
        a.set_title(f"RL, {s} FCS users", fontsize=9)
        a.set_ylabel("%")
        a.legend(fontsize=6.5)
        for b in range(5):
            g = f[f.band == b]
            gm = ev_m[(~ev_m.van) & (ev_m.segment == s) & (ev_m.band == b)]
            rows.append([s if b == 0 else "", rc.BLAB[b], ci(len(g)), f1(users[b]), f1(qmin[b]),
                         f1(100 * (g.queue_min > 1e-6).mean()), f2(g.queue_min.mean()), f1(g.queue_min.max()),
                         f2(gm.queue_min.mean()), f2(g.drive_km.mean())])
    for s, c in rc.SEG_COL.items():
        f = ev_r[(~ev_r.van) & (ev_r.segment == s)]
        ax[2].plot(range(5), [f[f.band == b].drive_km.mean() for b in range(5)], "o-", color=c, label=s)
    ax[2].set_xticks(range(5))
    ax[2].set_xticklabels(rc.PLAB, rotation=30, fontsize=7)
    ax[2].set_ylabel("mean drive to the FCS (km)")
    ax[2].legend(fontsize=7)
    rc.savefig(fig, os.path.join(FIG, "r_queues.pdf"))
    tab("r_queues", ["segment", "income $\\theta$", "FCS users", "users (\\%)", "queue min. (\\%)", "queueing (\\%)",
                     "mean queue", "longest", "MILP mean queue", "drive (km)"], rows, align="llrrrrrrrr",
        size="footnotesize", caption="Station users of the RL plans by segment and income band (queues in minutes; "
                                     "shares within the segment).", label="tab:queues")
    for s, sl in (("personal", "Pers"), ("commercial", "Comm")):
        f = ev_r[(~ev_r.van) & (ev_r.segment == s)]
        fm = ev_m[(~ev_m.van) & (ev_m.segment == s)]
        N["QueueR" + sl] = f2(f.queue_min.mean())
        N["QueueM" + sl] = f2(fm.queue_min.mean())
        N["PoorQShare" + sl] = f1(100 * f[f.band <= 1].queue_min.sum() / f.queue_min.sum())
        N["PoorUShare" + sl] = f1(100 * (f.band <= 1).mean())
    f = ev_r[~ev_r.van]
    N["QueueersR"] = f1(100 * (f.queue_min > 1e-6).mean())
    N["QueueersM"] = f1(100 * (ev_m[~ev_m.van].queue_min > 1e-6).mean())
    N["CommQShare"] = f1(100 * f[f.segment == "commercial"].queue_min.sum() / f.queue_min.sum())
    N["CommUShare"] = f1(100 * (f.segment == "commercial").mean())
    N["PatienceHitR"] = str(int((f.queue_min > f.W - 1e-3).sum()))

    # pressure
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 2.9))
    rows = []
    for lab, lo, hi_, c in (("low", -1, terc[0], "#4C72B0"), ("medium", terc[0], terc[1], "#DD8452"),
                            ("high", terc[1], 99, "#C44E52")):
        for meth, e in (("RL", ev_r), ("MILP", ev_m)):
            f = e[e.free & (e.pressure > lo) & (e.pressure <= hi_)]
            fq = f[~f.van]
            r = [lab if meth == "RL" else "", meth, ci(len(f)), f1(100 * (fq.queue_min > 1e-6).mean()),
                 f2(fq.queue_min.mean())]
            for s in ("personal", "commercial"):
                r.append(f1(100 * f[f.segment == s].van.mean()))
            rows.append(r)
            if meth == "RL":
                ax[0].plot(range(5), [100 * f[(f.segment == "personal") & (f.band == b)].van.mean() for b in range(5)],
                           "o-", color=c, label=f"{lab} pressure", ms=3)
                ax[1].plot(range(5), [fq[fq.band == b].queue_min.mean() for b in range(5)], "o-", color=c,
                           label=f"{lab}", ms=3)
                N["QueueRPress" + lab.capitalize()] = f2(fq.queue_min.mean())
                N["VanCommPress" + lab.capitalize()] = f1(100 * f[f.segment == "commercial"].van.mean())
                N["VanPersPress" + lab.capitalize()] = f1(100 * f[f.segment == "personal"].van.mean())
    for a in ax:
        a.set_xticks(range(5))
        a.set_xticklabels(rc.PLAB, rotation=30, fontsize=7)
        a.set_xlabel("income band $\\theta$")
        a.legend(fontsize=7)
    ax[0].set_ylabel("RL: personal EVs by van (%)")
    ax[1].set_ylabel("RL: mean FCS queue (min)")
    rc.savefig(fig, os.path.join(FIG, "r_pressure.pdf"))
    tab("r_pressure", ["pressure", "method", "free-choice EVs", "queueing (\\%)", "mean queue (min)",
                       "personal by van (\\%)", "commercial by van (\\%)"], rows, align="llrrrrr", size="small",
        caption="Free-choice EVs of the valid plans by demand-pressure tercile (shared cut points), RL against MILP.",
        label="tab:pressure")

    # horizon
    rows = []
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    for lng, lab, c in ((False, "short horizon (C1, R1, RC1)", "#C44E52"), (True, "long horizon (C2, R2, RC2)", "#4C72B0")):
        for meth, e, ls in (("RL", ev_r, "-"), ("MILP", ev_m, "--")):
            f = e[e.free & (e.long == lng) & (e.segment == "personal")]
            ax.plot(range(5), [100 * f[f.band == b].van.mean() for b in range(5)], marker="o", ls=ls, color=c, ms=3,
                    label=f"{meth}, {lab}")
    ax.set_xticks(range(5))
    ax.set_xticklabels(rc.PLAB, fontsize=7)
    ax.set_xlabel("income band $\\theta$")
    ax.set_ylabel("personal EVs by van (%)")
    ax.legend(fontsize=6)
    rc.savefig(fig, os.path.join(FIG, "r_horizon.pdf"))
    for fam in rc.FAMS:
        f = ev_r[ev_r.free & (ev_r["class"] == fam)]
        fm = ev_m[ev_m.free & (ev_m["class"] == fam)]
        rows.append([fam, "long" if fam.endswith("2") else "short", ci(len(f)), f2(f.pressure.mean()),
                     f1(100 * f[f.segment == "personal"].van.mean()), f1(100 * fm[fm.segment == "personal"].van.mean()),
                     f1(100 * f[f.segment == "commercial"].van.mean()), f1(100 * fm[fm.segment == "commercial"].van.mean())])
    tab("r_families", ["family", "horizon", "free-choice EVs", "mean pressure", "pers. by van RL", "pers. MILP",
                       "comm. by van RL", "comm. MILP"], rows, align="llrrrrrr", size="small",
        caption="Van share (\\%) of free-choice EVs by family and segment, RL against MILP.", label="tab:families")
    for lng, nm in ((False, "Short"), (True, "Long")):
        f = ev_r[ev_r.free & (ev_r.long == lng) & (ev_r.segment == "personal")]
        N["VanTop" + nm] = f1(100 * f[f.band == 4].van.mean())
        N["Press" + nm] = f2(ev_r[ev_r.free & (ev_r.long == lng)].pressure.mean())


# ====================================================================== vans
def vans(plan_r, plan_m, runs, milp, ev_r, ev_m, pairs):
    """Van use on the paired instances (both methods valid), so that counts are comparable."""
    out = {}
    T = {"C1": 1236, "C2": 3390, "R1": 230, "R2": 1000, "RC1": 240, "RC2": 960}
    for meth, plan, rr, ev in (("RL", plan_r, runs, ev_r), ("MILP", plan_m, milp, ev_m)):
        v = plan[plan.type == "van"].merge(rr[rr.valid_plan][["instance", "n", "class"]], on=["instance", "n"])
        v = v.merge(pairs, on=["instance", "n"])
        v["T"] = v["class"].map(T)
        v["seats"] = v.stops / 6
        v["batt"] = v.kWh_drawn / 150
        v["time"] = v.return_min / v["T"]
        v["tight"] = v[["seats", "batt", "time"]].idxmax(axis=1)
        e = ev[ev.van].merge(pairs, on=["instance", "n"])
        out[meth] = (v, e)
    rows = []
    for lab, fn in (("vans dispatched (paired instances)", lambda v, e: ci(len(v))),
                    ("stops per van (mean)", lambda v, e: f2(v.stops.mean())),
                    ("vans full (6 EVs, \\%)", lambda v, e: f1(100 * (v.stops == 6).mean())),
                    ("battery above 90\\% of $B$ (\\%)", lambda v, e: f1(100 * (v.batt > 0.9).mean())),
                    ("back after 90\\% of $T$ (\\%)", lambda v, e: f1(100 * (v.time > 0.9).mean())),
                    ("seats taken by stranded EVs (\\%)", lambda v, e: f1(100 * e.stranded.eq("yes").mean())),
                    ("vans serving only stranded EVs (\\%)", lambda v, e: f1(100 * (
                        e.groupby(["instance", "n", "resource"]).stranded.apply(lambda s: s.eq("yes").all()).mean()))),
                    ("tightest: seats (\\%)", lambda v, e: f1(100 * (v.tight == "seats").mean())),
                    ("tightest: battery (\\%)", lambda v, e: f1(100 * (v.tight == "batt").mean())),
                    ("tightest: time (\\%)", lambda v, e: f1(100 * (v.tight == "time").mean()))):
        rows.append([lab, fn(*out["RL"]), fn(*out["MILP"])])
    tab("r_vans", ["measure", "RL", "MILP"], rows, align="lrr",
        caption="What limits the dispatched vans, on the paired instances (both methods valid).", label="tab:vans")
    v, e = out["RL"]
    vm, em = out["MILP"]
    N["VansR"] = ci(len(v))
    N["VansM"] = ci(len(vm))
    N["VansFullR"] = f1(100 * (v.stops == 6).mean())
    N["VansFullM"] = f1(100 * (vm.stops == 6).mean())
    N["StopsR"] = f2(v.stops.mean())
    N["StopsM"] = f2(vm.stops.mean())
    N["StrandedSeatsR"] = f1(100 * e.stranded.eq("yes").mean())
    N["StrandedSeatsM"] = f1(100 * em.stranded.eq("yes").mean())
    N["OnlyStrandedR"] = f1(100 * e.groupby(["instance", "n", "resource"]).stranded.apply(lambda s: s.eq("yes").all()).mean())
    N["OnlyStrandedM"] = f1(100 * em.groupby(["instance", "n", "resource"]).stranded.apply(lambda s: s.eq("yes").all()).mean())
    N["TightSeatsR"] = f1(100 * (v.tight == "seats").mean())
    N["TightTimeR"] = f1(100 * (v.tight == "time").mean())
    N["TightBattR"] = f1(100 * (v.tight == "batt").mean())
    rows = []
    for fam in rc.FAMS:
        a, b = v[v["class"] == fam], vm[vm["class"] == fam]
        rows.append([fam, len(a), f2(a.stops.mean()), f1(100 * a.time.mean()), f1(100 * (a.tight == "seats").mean()),
                     f1(100 * (a.tight == "time").mean()), len(b), f2(b.stops.mean())])
    tab("r_vans_family", ["family", "vans RL", "stops/van", "back (\\% of $T$)", "tightest seats (\\%)",
                          "tightest time (\\%)", "vans MILP", "stops/van MILP"], rows, size="small",
        caption="RL van use by family, with the MILP's van count and stops for comparison.", label="tab:vansfamily")
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.8))
    bins = np.arange(0.5, 7.5, 1)
    ax[0].hist([v.stops, vm.stops], bins=bins, color=[C_RL, C_M], label=["RL", "MILP"], density=True)
    ax[0].set_xlabel("EVs served by one van")
    ax[0].set_ylabel("share of vans")
    ax[0].legend(fontsize=7)
    ax[1].hist([v.kWh_drawn, vm.kWh_drawn], bins=30, range=(0, 150), color=[C_RL, C_M], density=True)
    ax[1].set_xlabel("battery drawn (kWh, of 150)")
    ax[2].hist([100 * v.time, 100 * vm.time], bins=30, range=(0, 100), color=[C_RL, C_M], density=True)
    ax[2].set_xlabel("back at depot (% of T)")
    rc.savefig(fig, os.path.join(FIG, "r_vans.pdf"))


# ====================================================================== equity and segment mix
def equity(ev_r, ev_m):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.0), sharey=True)
    rows = []
    for k, s in enumerate(("personal", "commercial")):
        e = ev_r[ev_r.segment == s]
        pm = [e[e.band == b].perceived_money_Rs.mean() for b in range(5)]
        pt = [e[e.band == b].time_cost_Rs.mean() for b in range(5)]
        pay = [e[e.band == b].pay_Rs.mean() for b in range(5)]
        x = np.arange(5)
        ax[k].bar(x, pm, 0.6, color="#4C72B0", label="perceived money ($\\mu_i\\times$ paid)")
        ax[k].bar(x, pt, 0.6, bottom=pm, color="#DD8452", label="perceived time")
        ax[k].hlines(pay, x - 0.3, x + 0.3, color="k", lw=2, label="money actually paid")
        ax[k].set_xticks(x)
        ax[k].set_xticklabels(rc.PLAB, fontsize=7)
        ax[k].set_title(f"RL, {s}", fontsize=9)
        ax[k].set_xlabel("income band $\\theta$")
        tot_pay, tot_perc = ev_r.pay_Rs.sum(), ev_r.customer_cost_Rs.sum()
        for b in range(5):
            g = e[e.band == b]
            gm = ev_m[(ev_m.segment == s) & (ev_m.band == b)]
            rows.append([s if b == 0 else "", rc.BLAB[b], f1(100 * len(g) / len(ev_r)), f1(100 * g.pay_Rs.sum() / tot_pay),
                         f1(100 * g.customer_cost_Rs.sum() / tot_perc), f2(g.customer_cost_Rs.sum() / g.pay_Rs.sum()),
                         f1(100 * g.time_cost_Rs.sum() / g.customer_cost_Rs.sum()),
                         f1(g.pay_Rs.sum() / g.E.sum()), f0(g.customer_cost_Rs.mean()), f0(gm.customer_cost_Rs.mean())])
    ax[0].set_ylabel("Rs per EV")
    ax[0].legend(fontsize=6.5)
    rc.savefig(fig, os.path.join(FIG, "r_equity.pdf"))
    tab("r_equity", ["segment", "income $\\theta$", "EVs (\\%)", "money paid (\\%)", "perceived (\\%)",
                     "perceived per Rs", "time share (\\%)", "paid per kWh", "perceived/EV RL", "MILP"], rows,
        align="llrrrrrrrr", size="footnotesize",
        caption="Money paid and perceived cost in the RL plans by segment and income band (shares of all EVs of the "
                "valid RL plans); last column: perceived cost per EV of the same group in the MILP plans.",
        label="tab:equity")
    for s, sl in (("personal", "Pers"), ("commercial", "Comm")):
        e = ev_r[ev_r.segment == s]
        N["PerRsPoor" + sl] = f2(e[e.band == 0].customer_cost_Rs.sum() / max(1e-9, e[e.band == 0].pay_Rs.sum()))
        N["PerRsRich" + sl] = f2(e[e.band == 4].customer_cost_Rs.sum() / max(1e-9, e[e.band == 4].pay_Rs.sum()))
        N["PaidKwh" + sl] = f1(e.pay_Rs.sum() / e.E.sum())
        N["PercEv" + sl] = f0(e.customer_cost_Rs.mean())
        N["PercEvM" + sl] = f0(ev_m[ev_m.segment == s].customer_cost_Rs.mean())


def segment_mix(ev_r, ev_m):
    """Expected van share of free-choice EVs when the commercial share pi_C changes (reweighting the segments,
    first-order: the plans would adapt)."""
    rows = []
    sh = {}
    for meth, e in (("RL", ev_r), ("MILP", ev_m)):
        f = e[e.free]
        sh[meth] = {s: f[f.segment == s].van.mean() for s in ("personal", "commercial")}
    for pi in (0.3, 0.5, 0.7, 0.9):
        rows.append([f"{pi:.1f}" + (" (used)" if pi == 0.7 else ""),
                     f1(100 * (pi * sh["RL"]["commercial"] + (1 - pi) * sh["RL"]["personal"])),
                     f1(100 * (pi * sh["MILP"]["commercial"] + (1 - pi) * sh["MILP"]["personal"]))])
    tab("r_mix", ["commercial share $\\pi_{\\mathrm C}$", "expected van share RL (\\%)", "MILP (\\%)"], rows, align="lrr",
        caption="Expected van share of free-choice EVs for other segment mixes (reweighting of the observed segment "
                "shares; a first-order estimate).", label="tab:mix")
    N["MixRLThree"] = f1(100 * (0.3 * sh["RL"]["commercial"] + 0.7 * sh["RL"]["personal"]))
    N["MixRLNine"] = f1(100 * (0.9 * sh["RL"]["commercial"] + 0.1 * sh["RL"]["personal"]))
    N["MixMThree"] = f1(100 * (0.3 * sh["MILP"]["commercial"] + 0.7 * sh["MILP"]["personal"]))
    N["MixMNine"] = f1(100 * (0.9 * sh["MILP"]["commercial"] + 0.1 * sh["MILP"]["personal"]))


# ====================================================================== shared pressure definition: facts
def pressure_facts(ev_r):
    pt = infra.pressure_table()
    q1, q2 = infra.pressure_cuts()
    fc = pt[~pt.stranded.astype(bool)]
    N["PressNFree"] = ci(len(fc))
    N["PressWindowMin"] = f0(pt.window_min.min())
    N["PressWindowMax"] = f0(pt.window_min.max())
    N["PressMedian"] = f"{fc.pressure.median():.3f}"
    N["PressMax"] = f2(fc.pressure.max())
    for t_, nm in (("low", "Low"), ("medium", "Med"), ("high", "High")):
        N["PressShare" + nm] = f1(100 * (fc.terc == t_).mean())
    pk = fc[fc.period == "peak"]
    N["PeakLabelShare"] = f1(100 * (fc.period == "peak").mean())
    N["PeakLabelHigh"] = f1(100 * (pk.terc == "high").mean())
    N["PeakLabelLow"] = f1(100 * (pk.terc == "low").mean())
    fr = ev_r[ev_r.free]
    N["PressHighShareRL"] = f1(100 * (fr.pressure > q2).mean())


# ====================================================================== infrastructure utilisation
def max_concurrency(plan, kind):
    """Largest number of guns (kind FCS) or vans charging at the same moment, per instance (exact sweep)."""
    ev = plan[(plan.type == "ev") & ((plan.resource_type == "FCS") == (kind == "FCS"))]
    out = {}
    for key, g in ev.groupby(["instance", "n"]):
        t = np.r_[g.start_min.to_numpy(), g.end_min.to_numpy()]
        d = np.r_[np.ones(len(g)), -np.ones(len(g))]
        o = np.lexsort((d, t))                 # ends before starts at the same minute
        out[key] = int(np.cumsum(d[o]).max()) if len(g) else 0
    return out


def utilisation_section(runs, milp, plan_r, plan_m, p):
    vr = runs[runs.valid_plan][["instance", "n"]]
    pr_all = plan_r.merge(vr, on=["instance", "n"])
    pair = p[["instance", "n"]]
    pr_p = plan_r.merge(pair, on=["instance", "n"])
    pm_p = plan_m.merge(pair, on=["instance", "n"])
    U = {"RL": infra.utilisation(pr_all), "RLp": infra.utilisation(pr_p), "M": infra.utilisation(pm_p)}
    for k, pl in (("RL", pr_all), ("RLp", pr_p), ("M", pm_p)):
        cg = max_concurrency(pl, "FCS")
        cv = max_concurrency(pl, "van")
        ins = U[k]["instances"]
        ins["peak_guns"] = [cg.get((a, b), 0) for a, b in zip(ins.instance, ins.n)]
        ins["peak_vans"] = [cv.get((a, b), 0) for a, b in zip(ins.instance, ins.n)]
        v = U[k]["vans"]
        v["eta"] = 0.9
        v["loss_kWh"] = v.kWh_delivered / 0.9 - v.kWh_delivered
        v["drive_kWh"] = v.kWh_drawn - v.kWh_delivered / 0.9
        v["drive_util"] = v.drive_min / v["T"]
        v["idle_util"] = v.idle_min / v["T"]
    # ---- system and stations table: RL (all valid), RL paired, MILP paired
    def col(k):
        ins, g, st, v = U[k]["instances"], U[k]["guns"], U[k]["stations"], U[k]["vans"]
        return [ci(len(ins)), f0(ins.kWh_total.mean()), f1(100 * ins.kWh_fcs.sum() / ins.kWh_total.sum()),
                f2(100 * ins.gun_util.mean()), f1(ins.idle_guns.mean()), f1(100 * (g.evs == 0).mean()),
                f1(100 * g.util.max()), f1(100 * st.util.quantile(.9)), f1(ins.peak_guns.mean()),
                str(int(ins.peak_guns.max())), f2(100 * ins.system_util.mean()), f1(100 * ins.fleet_use.mean()),
                f2(ins.vans_used.mean()), f1(ins.peak_vans.mean())]
    labels = ["instances (valid plans)", "energy asked per instance (kWh)", "charged at stations (\\% of kWh)",
              "gun time = energy utilisation over $[0,T]$ (\\%)", "idle guns per instance (of 24)",
              "guns without any EV (\\% of guns)", "busiest gun (\\% of $T$)", "station utilisation, 90th pct (\\%)",
              "peak concurrent guns (mean)", "peak concurrent guns (max)", "installed-capacity utilisation (\\%)",
              "van fleet use (dispatched / 5, \\%)", "vans per instance", "peak concurrent vans (mean)"]
    cols = [col("RL"), col("RLp"), col("M")]
    rows = [[lab] + [c[i] for c in cols] for i, lab in enumerate(labels)]
    tab("r_util_system", ["measure", "RL, all valid", "RL, paired", "MILP, paired"], rows, align="lrrr",
        caption="Infrastructure utilisation (\\code{infra.utilisation}): stations, guns and the system. Paired = "
                "instances where both RL and MILP have a valid plan.", label="tab:utilsystem")
    # ---- vans table
    def vcol(k):
        v = U[k]["vans"]
        drawn = v.kWh_drawn.sum()
        return [ci(len(v)), f1(100 * v.seat_util.mean()), f1(100 * v.battery_util.mean()),
                f1(100 * v.on_road_util.mean()), f1(100 * v.charge_util.mean()), f1(100 * v.drive_util.mean()),
                f1(100 * v.idle_util.mean()), f1(100 * (v.idle_min / v.on_road_min.clip(lower=1e-9)).mean()),
                f1(v.kWh_delivered.mean()), f1(100 * v.kWh_delivered.sum() / drawn), f1(100 * v.loss_kWh.sum() / drawn),
                f1(100 * v.drive_kWh.sum() / drawn)]
    labels = ["vans dispatched", "seats used (\\% of 6)", "battery drawn (\\% of 150 kWh)", "on the road (\\% of $T$)",
              "charging = power-capacity use (\\% of $T$)", "driving (\\% of $T$)", "idle on the road (\\% of $T$)",
              "idle (\\% of on-road time)", "kWh delivered per van", "energy drawn: delivered (\\%)",
              "energy drawn: transfer loss (\\%)", "energy drawn: driving (\\%)"]
    cols = [vcol("RL"), vcol("RLp"), vcol("M")]
    rows = [[lab] + [c[i] for c in cols] for i, lab in enumerate(labels)]
    tab("r_util_vans", ["measure (per dispatched van)", "RL, all valid", "RL, paired", "MILP, paired"], rows,
        align="lrrr", caption="Use of the dispatched vans (\\code{infra.utilisation}); means over vans.",
        label="tab:utilvans")
    # ---- by family
    rows = []
    for fam in rc.FAMS:
        a, b = U["RLp"]["instances"], U["M"]["instances"]
        a, b = a[a.family == fam], b[b.family == fam]
        va, vb = U["RLp"]["vans"], U["M"]["vans"]
        va, vb = va[va.family == fam], vb[vb.family == fam]
        rows.append([fam, len(a), f2(100 * a.gun_util.mean()), f2(100 * b.gun_util.mean()), f1(a.idle_guns.mean()),
                     f1(b.idle_guns.mean()), f1(100 * a.fleet_use.mean()), f1(100 * b.fleet_use.mean()),
                     f1(100 * va.seat_util.mean()), f1(100 * vb.seat_util.mean()),
                     f1(100 * va.on_road_util.mean()), f1(100 * vb.on_road_util.mean())])
    tab("r_util_family", ["family", "paired", "gun util RL", "MILP", "idle guns RL", "MILP", "fleet use RL", "MILP",
                          "seats RL", "MILP", "on road RL", "MILP"], rows, size="small",
        caption="Utilisation by family on the paired instances (\\%; idle guns of 24).", label="tab:utilfamily")
    # ---- figure: by n, profile, station distribution
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
    ns = sorted(U["RLp"]["instances"].n.unique())
    for k, c, lab, mk in (("RLp", C_RL, "RL", "o"), ("M", C_M, "MILP", "s")):
        ins = U[k]["instances"]
        ax[0].plot(ns, [100 * ins[ins.n == n].gun_util.mean() for n in ns], marker=mk, color=c, label=f"{lab}: guns", ms=3)
        ax[0].plot(ns, [100 * ins[ins.n == n].fleet_use.mean() for n in ns], marker=mk, color=c, ls="--",
                   label=f"{lab}: van fleet", ms=3)
        pf = U[k]["profile"].groupby("t_rel")[["guns_busy", "vans_busy"]].mean()
        ax[1].plot(pf.index, 100 * pf.guns_busy, color=c, label=f"{lab}: guns charging")
        ax[1].plot(pf.index, 100 * pf.vans_busy, color=c, ls="--", label=f"{lab}: vans charging")
        st = U[k]["stations"]
        ax[2].hist(100 * st.util, bins=30, range=(0, 30), color=c, alpha=0.55, density=True, label=lab)
    ax[0].set_xlabel("$n$")
    ax[0].set_ylabel("utilisation (%)")
    ax[0].legend(fontsize=6.5)
    ax[1].set_xlabel("t / T")
    ax[1].set_ylabel("share busy (%), mean over instances")
    ax[1].legend(fontsize=6.5)
    ax[2].set_xlabel("station time utilisation (%), clipped at 30")
    ax[2].set_ylabel("density (stations)")
    ax[2].legend(fontsize=7)
    rc.savefig(fig, os.path.join(FIG, "r_util.pdf"))
    # ---- numbers
    a, b = U["RLp"], U["M"]
    ia, ib = a["instances"], b["instances"]
    va, vb = a["vans"], b["vans"]
    N["UtilPaired"] = str(len(ia))
    N["UtilKwhInst"] = f0(ia.kWh_total.mean())
    N["UtilFcsShareR"] = f1(100 * ia.kWh_fcs.sum() / ia.kWh_total.sum())
    N["UtilFcsShareM"] = f1(100 * ib.kWh_fcs.sum() / ib.kWh_total.sum())
    N["UtilGunR"] = f2(100 * ia.gun_util.mean())
    N["UtilGunM"] = f2(100 * ib.gun_util.mean())
    N["UtilGunMaxR"] = f1(100 * a["guns"].util.max())
    N["UtilIdleGunsR"] = f1(ia.idle_guns.mean())
    N["UtilIdleGunsM"] = f1(ib.idle_guns.mean())
    N["UtilSysR"] = f2(100 * ia.system_util.mean())
    N["UtilSysM"] = f2(100 * ib.system_util.mean())
    N["UtilFleetR"] = f1(100 * ia.fleet_use.mean())
    N["UtilFleetM"] = f1(100 * ib.fleet_use.mean())
    N["UtilPeakGunsR"] = f1(ia.peak_guns.mean())
    N["UtilPeakGunsM"] = f1(ib.peak_guns.mean())
    N["UtilPeakGunsMaxR"] = str(int(ia.peak_guns.max()))
    N["UtilPeakGunsMaxM"] = str(int(ib.peak_guns.max()))
    N["UtilGunHundredR"] = f1(100 * ia[ia.n == 100].gun_util.mean())
    N["UtilGunHundredM"] = f1(100 * ib[ib.n == 100].gun_util.mean())
    fam_gun = ia.groupby("family").gun_util.mean()
    N["UtilGunTopFam"] = fam_gun.idxmax()
    N["UtilGunTopFamV"] = f1(100 * fam_gun.max())
    N["UtilGunLowFam"] = fam_gun.idxmin()
    N["UtilGunLowFamV"] = f2(100 * fam_gun.min())
    pfa = a["profile"].groupby("t_rel").guns_busy.mean()
    N["UtilProfPeakR"] = f1(100 * pfa.max())
    N["UtilProfPeakAt"] = f"{pfa.idxmax():.2f}"
    pfb = b["profile"].groupby("t_rel").guns_busy.mean()
    N["UtilProfPeakM"] = f1(100 * pfb.max())
    N["UtilSeatR"] = f1(100 * va.seat_util.mean())
    N["UtilSeatM"] = f1(100 * vb.seat_util.mean())
    N["UtilBattR"] = f1(100 * va.battery_util.mean())
    N["UtilBattM"] = f1(100 * vb.battery_util.mean())
    N["UtilRoadR"] = f1(100 * va.on_road_util.mean())
    N["UtilRoadM"] = f1(100 * vb.on_road_util.mean())
    N["UtilChargeR"] = f1(100 * va.charge_util.mean())
    N["UtilChargeM"] = f1(100 * vb.charge_util.mean())
    N["UtilIdleR"] = f1(100 * (va.idle_min / va.on_road_min.clip(lower=1e-9)).mean())
    N["UtilIdleM"] = f1(100 * (vb.idle_min / vb.on_road_min.clip(lower=1e-9)).mean())
    N["UtilVanKwhR"] = f1(va.kWh_delivered.mean())
    N["UtilVanKwhM"] = f1(vb.kWh_delivered.mean())
    N["UtilDriveShareR"] = f1(100 * va.drive_kWh.sum() / va.kWh_drawn.sum())
    N["UtilDriveShareM"] = f1(100 * vb.drive_kWh.sum() / vb.kWh_drawn.sum())
    N["UtilLossShareR"] = f1(100 * va.loss_kWh.sum() / va.kWh_drawn.sum())
    N["UtilVansR"] = ci(len(va))
    N["UtilVansM"] = ci(len(vb))
    N["UtilVanKwhTotR"] = ci(round(va.kWh_delivered.sum()))
    N["UtilVanKwhTotM"] = ci(round(vb.kWh_delivered.sum()))
    N["UtilDepartNeg"] = str(int((U["RL"]["vans"].depart_min < -1e-6).sum()))


# ====================================================================== main
def main():
    for d in (FIG, TAB, DATA):
        os.makedirs(d, exist_ok=True)
    runs, milp, greedy, plan_r, plan_m, meta, tmeta, tlog = load()
    info, inst = instance_info(runs[["instance", "n"]].drop_duplicates().itertuples(index=False))
    part1(meta, tmeta)
    training(tlog, tmeta, meta)
    x = outcomes(runs, milp, greedy, info, inst)
    ev_r = ev_table(plan_r, runs, info)
    ev_m = ev_table(plan_m, milp, info)
    overview(x, ev_r, plan_r)
    ev_distributions(ev_r)
    p = quality(x, ev_r, plan_m, info)
    computation(x, meta)
    anatomy(p, ev_r, ev_m, plan_r, plan_m, meta)
    agreement(p, ev_r, ev_m)
    terc = income(ev_r, ev_m, meta)
    duals()
    queues(ev_r, ev_m, terc)
    vans(plan_r, plan_m, runs, milp, ev_r, ev_m, p[["instance", "n"]])
    pressure_facts(ev_r)
    utilisation_section(runs, milp, plan_r, plan_m, p)
    equity(ev_r, ev_m)
    segment_mix(ev_r, ev_m)
    N["NEVs"] = ci(len(ev_r))
    N["NFree"] = ci(int(ev_r.free.sum()))
    N["NFreePers"] = ci(int((ev_r.free & (ev_r.segment == "personal")).sum()))
    N["NFreeComm"] = ci(int((ev_r.free & (ev_r.segment == "commercial")).sum()))
    N["CommShareEV"] = f1(100 * (ev_r.segment == "commercial").mean())
    N.save(os.path.join(TAB, "numbers.tex"))
    print(f"{len(N.d)} numbers, figures in {FIG}, tables in {TAB}")


if __name__ == "__main__":
    main()
