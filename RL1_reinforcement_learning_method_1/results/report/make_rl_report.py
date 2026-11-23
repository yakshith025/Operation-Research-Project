"""Figures, tables and numbers for comprehensive_report_rl1.tex (RL1 results, compared with the MILP).

Reads (results_io layout):
  ../runs.csv, ../plan.csv                  RL1 (run_id rl1)
  ../../../new_results/runs.csv, plan.csv   MILP (run_id milp_300s) and greedy (run_id greedy)
  ../../model/train_log.csv, train_meta.json, ../meta.json
  data/duals_*.csv                          from compute_duals_rl.py
and rebuilds every EV's inputs (theta, income coefficients, stranded, nearest station, demand
pressure) from dataset_ev_solomon/csv with results_io.ev_attributes.
Writes figures/r_*.pdf, tables/r_*.tex and tables/numbers_r.tex.
Usage (from this folder):  python compute_duals_rl.py ; python make_rl_report.py
"""
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RLDIR = os.path.join(HERE, "..")
RLROOT = os.path.join(HERE, "..", "..")
PROJECT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MILPDIR = os.path.join(PROJECT, "new_results")
sys.path.insert(0, os.path.join(PROJECT, "code"))
import make_ev_solomon as mes  # noqa: E402
import make_income as inc       # noqa: E402
import results_io as rio        # noqa: E402
import solve_milp_f3 as f3      # noqa: E402

FIG, TAB, DAT = (os.path.join(HERE, d) for d in ("figures", "tables", "data"))
for d in (FIG, TAB, DAT):
    os.makedirs(d, exist_ok=True)
plt.rcParams.update({"font.size": 9, "font.family": "serif", "mathtext.fontset": "cm",
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
P3 = inc.load_params()
DATA = mes.EvSolomonData(os.path.join(PROJECT, "dataset_ev_solomon", "csv"))
BANDS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)]
BLAB = ["0.0--0.2", "0.2--0.4", "0.4--0.6", "0.6--0.8", "0.8--1.0"]
PLAB = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
FAMS = ["C1", "C2", "R1", "R2", "RC1", "RC2"]
NS = list(range(10, 101, 10))
CHARGERS = 29
C_EV, C_M, C_F, PI_F, PI_M, ETA = 4.0, 15.0, 1500.0, 8.7, 10.7, 0.9
POW_F, POW_M = 22.0, 30.0
COLR, COLM, COLG = "#C44E52", "#4C72B0", "#8C8C8C"
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


# ====================================================================== load
def read_runs(folder, run_id):
    return {(r["instance"], int(r["n"])): r for r in csv.DictReader(open(os.path.join(folder, "runs.csv")))
            if r["run_id"] == run_id}


def read_plan(folder, run_id):
    out = defaultdict(lambda: {"ev": {}, "van": []})
    for r in csv.DictReader(open(os.path.join(folder, "plan.csv"))):
        if r["run_id"] != run_id:
            continue
        k = (r["instance"], int(r["n"]))
        if r["type"] == "ev":
            out[k]["ev"][int(r["id"])] = r
        elif r["type"] == "van":
            out[k]["van"].append(r)
    return out


RUNS = {"RL": read_runs(RLDIR, "rl1"), "MILP": read_runs(MILPDIR, "milp_300s"), "GREEDY": read_runs(MILPDIR, "greedy")}
PLANS = {"RL": read_plan(RLDIR, "rl1"), "MILP": read_plan(MILPDIR, "milp_300s")}
KEYS = sorted(RUNS["RL"], key=lambda k: (k[1], k[0]))
ATTR = {}
JS = {}
for k in KEYS:
    J = f3.load_instance(k[0], k[1], P3, DATA, greedy=False)
    JS[k] = {"T": J["params"]["T_min"]}
    ATTR[k] = {a["id"]: a for a in rio.ev_attributes(J)}
NUMF = ["drive_km", "drive_min", "queue_min", "van_wait_min", "charge_min", "time_to_charged_min", "pay_Rs",
        "perceived_money_Rs", "time_cost_Rs", "customer_cost_Rs", "operator_energy_Rs", "start_min", "end_min"]
ATTF = ["theta", "mu", "c_D", "c_Q", "c_V", "c_C", "W_min", "nearest_reachable_fcs_km", "E_kWh", "r_min", "load_at_request"]


def records(m):
    out = []
    for k in KEYS:
        run = RUNS[m].get(k)
        if not run or run["valid"] != "yes":
            continue
        for i, p in PLANS[m][k]["ev"].items():
            a = ATTR[k][i]
            e = {"instance": k[0], "n": k[1], "cls": run["class"], "status": run["status"], "EV": i,
                 "resource": p["resource"]}
            for c in NUMF:
                e[c] = fnum(p[c])
            for c in ATTF:
                e[c] = fnum(a[c])
            e["van"] = 1 if str(p["resource"]).startswith("van") else 0
            e["free"] = a["stranded"] == "no"
            e["pressure"] = e["load_at_request"] / CHARGERS
            e["band"] = next(b for b, (lo, hi) in enumerate(BANDS) if lo <= e["theta"] < hi)
            e["opt"] = run["status"] == "optimal"
            e["h2"] = run["class"].endswith("2")
            out.append(e)
    return out


EV = {"RL": records("RL"), "MILP": records("MILP")}
FREE = {m: [e for e in EV[m] if e["free"]] for m in EV}
T1, T2 = np.quantile([e["pressure"] for e in FREE["MILP"]], [1 / 3, 2 / 3])
for m in EV:
    for e in EV[m]:
        e["lvl"] = "low" if e["pressure"] <= T1 else ("mid" if e["pressure"] <= T2 else "high")
VALID = {m: [k for k in KEYS if RUNS[m].get(k, {}).get("valid") == "yes"] for m in ("RL", "MILP", "GREEDY")}
BOTH = [k for k in KEYS if k in set(VALID["RL"]) and k in set(VALID["MILP"])]
num.update(PressLowR=f(T1, "{:.2f}"), PressHighR=f(T2, "{:.2f}"))


def share(g):
    return 100 * float(np.mean([e["van"] for e in g])) if g else float("nan")


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


# ====================================================================== 1. outcome
def rl_status(k):
    return "valid plan" if RUNS["RL"][k]["valid"] == "yes" else "invalid plan"


MSTAT = {"optimal": "optimal", "time_limit": "time limit, plan", "time_limit_no_plan": "time limit, no plan",
         "infeasible": "infeasible"}
rows = []
for n in NS:
    ks = [k for k in KEYS if k[1] == n]
    rv = sum(RUNS["RL"][k]["valid"] == "yes" for k in ks)
    mv = sum(RUNS["MILP"][k]["valid"] == "yes" for k in ks)
    gv = sum(RUNS["GREEDY"].get(k, {}).get("valid") == "yes" for k in ks)
    rows.append([n, len(ks), rv, len(ks) - rv, mv, gv, f(mean([fnum(RUNS["RL"][k]["solve_s"]) for k in ks]), "{:.2f}"),
                 f(mean([fnum(RUNS["MILP"][k]["solve_s"]) for k in ks]), "{:.1f}")])
tv = {m: len(VALID[m]) for m in VALID}
rows += ["MID", ["all", len(KEYS), tv["RL"], len(KEYS) - tv["RL"], tv["MILP"], tv["GREEDY"],
                 f(mean([fnum(RUNS["RL"][k]["solve_s"]) for k in KEYS]), "{:.2f}"),
                 f(mean([fnum(RUNS["MILP"][k]["solve_s"]) for k in KEYS]), "{:.1f}")]]
table("r_outcome", ["$n$", "instances", "RL1|valid", "RL1|invalid", "MILP|valid", "greedy|valid", "RL1 time|(s)", "MILP time|(s)"],
      rows, "@{}rrrrrrrr@{}")
ct = Counter((MSTAT[RUNS["MILP"][k]["status"]], rl_status(k)) for k in KEYS)
rows = [[ms, ct[(ms, "valid plan")], ct[(ms, "invalid plan")], ct[(ms, "valid plan")] + ct[(ms, "invalid plan")]]
        for ms in ("optimal", "time limit, plan", "time limit, no plan", "infeasible")]
table("r_crosstab", ["MILP outcome", "RL1 valid plan", "RL1 invalid plan", "instances"], rows, "@{}lrrr@{}")
num.update(NInstR=len(KEYS), NValidR=tv["RL"], NInvalidR=len(KEYS) - tv["RL"], NValidM=tv["MILP"], NValidG=tv["GREEDY"],
           NBoth=len(BOTH), RLonly=ct[("time limit, no plan", "valid plan")], RLinvNoPlan=ct[("time limit, no plan", "invalid plan")],
           RLinvInf=ct[("infeasible", "invalid plan")], NOptM=sum(RUNS["MILP"][k]["status"] == "optimal" for k in KEYS),
           RLtime=f(mean([fnum(RUNS["RL"][k]["solve_s"]) for k in KEYS]), "{:.2f}"),
           RLtimeMax=f(max(fnum(RUNS["RL"][k]["solve_s"]) for k in KEYS), "{:.2f}"),
           MILPtime=f(mean([fnum(RUNS["MILP"][k]["solve_s"]) for k in KEYS]), "{:.0f}"))
fig, ax = plt.subplots(figsize=(5.2, 2.6))
x = np.arange(len(NS))
w = 0.38
for off, m, col, lab in ((-w / 2, "RL", COLR, "RL1 valid plans"), (w / 2, "MILP", COLM, "MILP valid plans")):
    ax.bar(x + off, [sum(RUNS[m][k]["valid"] == "yes" for k in KEYS if k[1] == n) for n in NS], w, color=col, label=lab)
ax.axhline(56, color="grey", lw=0.6)
ax.set_xticks(x)
ax.set_xticklabels([str(n) for n in NS])
ax.set(xlabel="EVs per instance $n$", ylabel="instances (of 56)")
ax.set_ylim(0, 72)
ax.legend(fontsize=7.5, loc="upper right", ncol=2)
save(fig, "r_outcome")

# ====================================================================== 2. training
tl = list(csv.DictReader(open(os.path.join(RLROOT, "model", "train_log.csv"))))
tmeta = json.load(open(os.path.join(RLROOT, "model", "train_meta.json")))
rmeta = json.load(open(os.path.join(RLDIR, "meta.json")))["rl1"]
tp = rmeta["training_params"]["training"]
ep = [int(r["epoch"]) for r in tl]
vr = [fnum(r["val_ratio_rl_over_greedy"]) for r in tl]
byn = [json.loads(r["val_ratio_by_n"]) if r["val_ratio_by_n"] else {} for r in tl]
imit_last = max([int(r["epoch"]) for r in tl if r["phase"] == "imitation"] or [0])
best_ep = int(rmeta["training_state"]["best_epoch"])
best_val = float(rmeta["training_state"]["best_val"])
fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.7))
ax = axes[0]
ax.axvspan(0.5, imit_last + 0.5, color="#DDDDDD", alpha=0.7, lw=0, label="imitation warm-up")
ax.plot(ep[1:], vr[1:], color="#333333", lw=1.1, label="all (n = 20, 50, 100)")
for nn, col in (("20", "#55A868"), ("50", "#DD8452"), ("100", COLR)):
    ax.plot(ep[1:], [b.get(nn, np.nan) for b in byn[1:]], color=col, lw=0.8, alpha=0.8, label=f"n = {nn}")
ax.axhline(1.0, color="grey", lw=0.6, ls=":")
ax.axvline(best_ep, color=COLM, lw=0.9, ls="--", label=f"best epoch {best_ep}")
ax.set(xlabel="epoch", ylabel="validation cost RL1 / greedy rule")
ax.set_ylim(0.9, 1.05)
ax.legend(fontsize=6.3, loc="upper right", ncol=2)
ax = axes[1]
ax.plot(ep, [fnum(r["val_unplaced_rl_%"]) for r in tl], color=COLR, lw=1.0, label="RL1 (validation)")
ax.plot(ep, [fnum(r["val_unplaced_greedy_%"]) for r in tl], color=COLG, lw=1.0, ls="--", label="greedy rule (validation)")
ax.plot(ep, [fnum(r["train_unplaced_%"]) for r in tl], color="#8172B2", lw=0.8, alpha=0.7, label="RL1 sampled (training)")
ax.set(xlabel="epoch", ylabel="unplaced EVs (%)")
ax.set_ylim(bottom=0)
ax.legend(fontsize=6.5)
save(fig, "r_training")
num.update(Epochs=max(ep), ImitEpochs=imit_last, BestEpoch=best_ep, BestVal=f(best_val, "{:.3f}"),
           BestValPct=f(100 * (1 - best_val), "{:.1f}"), StartVal=f(vr[0], "{:.2f}"), AfterImit=f(vr[imit_last], "{:.3f}"),
           TrainMin=f(fnum(tl[-1]["train_min"]), "{:.0f}"), TrainBestS=f(rmeta["training_state"]["train_s"] / 60, "{:.0f}"),
           Iters=rmeta["training_state"]["iters"], ValHundred=f(byn[ep.index(best_ep)].get("100", np.nan), "{:.3f}"),
           ValTwenty=f(byn[ep.index(best_ep)].get("20", np.nan), "{:.3f}"), ValFifty=f(byn[ep.index(best_ep)].get("50", np.nan), "{:.3f}"),
           Device=tmeta.get("device", "--"), Rollouts=rmeta["augment"] * (1 + rmeta["samples"]),
           Augment=rmeta["augment"], Samples=rmeta["samples"], StopReason=str(tmeta.get("stop_reason", "early stop")).replace("_", " "))
g = rmeta["training_params"]["generator"]
mc = rmeta["model_cfg"]
rows = [["encoder", f"{mc['layers']} Transformer layers, width $d={mc['d']}$, {mc['heads']} heads, feed-forward {mc['ff']}"],
        ["decoder", f"two-step pointer (EV, then gun/van), logit clip $C={mc['clip']:g}$, MLP of dynamic features ({mc['dyn_hidden']} hidden)"],
        ["warm-up", f"{tp['imitation_batches']} imitation batches of the greedy cost rule (teacher forcing), learning rate {tp['lr']:g}"],
        ["REINFORCE", f"POMO shared baseline, {tp['batch_instances']} instances $\\times$ {tp['samples']} rollouts per batch, {tp['iters_per_epoch']} batches per epoch"],
        ["learning rate", f"{tp['lr_reinforce']:g} after the warm-up, decay {tp['lr_decay_per_epoch']:g} per epoch, gradient clip {tp['max_grad_norm']:g}"],
        ["training sizes", f"$n$ drawn from {tp['n_min']}--{tp['n_max']} per batch; unplaced-EV penalty {tp['drop_penalty']:g}$\\times$ scale"],
        ["stopping", f"at least {tp['min_epochs']} epochs, then stop after {tp['patience_epochs']} epochs without a better validation; at most {tp['max_epochs']} epochs or {tp['hours']:g} h"],
        ["validation", f"greedy rollout vs greedy cost rule on {tp['val_instances_per_size']} fixed instances each of $n$ = {', '.join(map(str, tp['val_sizes']))}"],
        ["training data", f"random instances by the EV-Solomon rules: {g['clusters'][0]}--{g['clusters'][1]} clusters (C), 15--30\\,\\% requests at 0, Beta-shaped request times, $T$ jittered $\\pm${100 * g['T_jitter']:.0f}\\,\\%; the 116 test files are never seen"],
        ["inference", f"{rmeta['augment']} map symmetries $\\times$ (1 greedy + {rmeta['samples']} sampled) = {rmeta['augment'] * (1 + rmeta['samples'])} rollouts; keep the plan with fewest unplaced EVs, then lowest $Z$"],
        ["hardware", f"{tmeta.get('device', 'GPU')}, torch {tmeta.get('torch', '')}; training {fnum(tl[-1]['train_min']):.0f} min, {rmeta['training_state']['iters']} batches to the best model"]]
table("r_settings", ["item", "setting"], rows, "@{}lp{12.4cm}@{}")

# ====================================================================== 3. quality vs MILP and greedy
def gap(a, b):
    return 100 * (a - b) / b


gaps = {k: gap(fnum(RUNS["RL"][k]["Z_total"]), fnum(RUNS["MILP"][k]["Z_total"])) for k in BOTH}
gopt = {k: v for k, v in gaps.items() if RUNS["MILP"][k]["status"] == "optimal"}
gtl = {k: v for k, v in gaps.items() if RUNS["MILP"][k]["status"] == "time_limit"}
gk = [k for k in KEYS if RUNS["RL"][k]["valid"] == "yes" and RUNS["GREEDY"].get(k, {}).get("valid") == "yes"]
ggap = {k: gap(fnum(RUNS["RL"][k]["Z_total"]), fnum(RUNS["GREEDY"][k]["Z_total"])) for k in gk}
gv = np.array(list(gaps.values()))
num.update(GapMean=f(gv.mean(), "{:.2f}"), GapMed=f(np.median(gv), "{:.2f}"), GapMax=f(gv.max(), "{:.2f}"),
           GapMin=f(gv.min(), "{:.2f}"), GapOptMean=f(mean(list(gopt.values())), "{:.2f}"),
           GapTLMean=f(mean(list(gtl.values())), "{:.2f}"), NGapOpt=len(gopt), NGapTL=len(gtl),
           RLbetter=sum(v < -1e-6 for v in gaps.values()), RLequal=sum(abs(v) <= 1e-6 for v in gaps.values()),
           RLwithinOne=f(100 * mean([v <= 1.0 for v in gaps.values()]), "{:.0f}"),
           RLwithinTwo=f(100 * mean([v <= 2.0 for v in gaps.values()]), "{:.0f}"),
           OptEqual=sum(abs(v) <= 1e-6 for v in gopt.values()),
           GreedyBeat=sum(v < -1e-6 for v in ggap.values()), GreedyN=len(ggap), GreedyGap=f(mean(list(ggap.values())), "{:.2f}"),
           GreedyGapAbs=f(abs(mean(list(ggap.values()))), "{:.2f}"),
           GreedyWorse=sum(v > 1e-6 for v in ggap.values()))
better_list = sorted([k for k, v in gaps.items() if v < -1e-6], key=lambda k: gaps[k])
num["BetterList"] = ", ".join(f"{k[0].replace('EV-', '')} $n$={k[1]} ({gaps[k]:.2f}\\,\\%)" for k in better_list[:6])


def qrow(label, ks):
    g_ = [gaps[k] for k in ks if k in gaps]
    go = [gaps[k] for k in ks if k in gopt]
    gg = [ggap[k] for k in ks if k in ggap]
    return [label, len(ks), sum(RUNS["RL"][k]["valid"] == "yes" for k in ks), sum(RUNS["MILP"][k]["valid"] == "yes" for k in ks),
            sum(RUNS["RL"][k]["valid"] == "yes" and RUNS["MILP"][k]["valid"] != "yes" for k in ks),
            f(mean(g_), "{:.2f}"), f(med(g_), "{:.2f}"), f(max(g_) if g_ else float("nan"), "{:.2f}"),
            f(mean(go), "{:.2f}"), f(mean(gg), "{:.2f}")]


head = ["group", "inst.", "RL1|valid", "MILP|valid", "RL1 only|valid", "gap to MILP|mean (\\%)", "gap to MILP|median (\\%)",
        "gap to MILP|max (\\%)", "gap to|optimum (\\%)", "gap to|greedy (\\%)"]
table("r_quality_n", head, [qrow(f"$n={n}$", [k for k in KEYS if k[1] == n]) for n in NS] + ["MID", qrow("all", KEYS)],
      "@{}lrrrrrrrrr@{}")
table("r_quality_class", head, [qrow(c, [k for k in KEYS if RUNS["RL"][k]["class"] == c]) for c in FAMS] + ["MID", qrow("all", KEYS)],
      "@{}lrrrrrrrrr@{}")
fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.5))
axes[0].hist([list(gopt.values()), list(gtl.values())], bins=30, stacked=True, color=["#55A868", "#DD8452"],
             label=["MILP proven optimal", "MILP time-limit plan"])
axes[0].axvline(0, color="grey", lw=0.6)
axes[0].set(xlabel="gap RL1 vs MILP (%)", ylabel="instances")
axes[0].legend(fontsize=6.3)
axes[1].boxplot([[gaps[k] for k in BOTH if k[1] == n] for n in NS], showfliers=False)
axes[1].axhline(0, color="grey", lw=0.6)
axes[1].set_xticks(range(1, len(NS) + 1))
axes[1].set_xticklabels([str(n) for n in NS], fontsize=6.5)
axes[1].set(xlabel="$n$", ylabel="gap RL1 vs MILP (%)")
axes[2].boxplot([[ggap[k] for k in gk if k[1] == n] for n in NS], showfliers=False)
axes[2].axhline(0, color="grey", lw=0.6)
axes[2].set_xticks(range(1, len(NS) + 1))
axes[2].set_xticklabels([str(n) for n in NS], fontsize=6.5)
axes[2].set(xlabel="$n$", ylabel="gap RL1 vs greedy rule (%)")
save(fig, "r_gap")
# time
fig, ax = plt.subplots(figsize=(4.8, 2.7))
for m, col, lab in (("RL", COLR, "RL1 (inference, GPU)"), ("MILP", COLM, "MILP (Gurobi, 4 threads)")):
    ax.plot(NS, [med([fnum(RUNS[m][k]["solve_s"]) for k in KEYS if k[1] == n]) for n in NS], "o-", color=col, ms=3.5, lw=1.2, label=lab)
ax.set_yscale("log")
ax.set(xlabel="EVs per instance $n$", ylabel="median seconds per instance")
ax.legend(fontsize=7.5)
save(fig, "r_time")
train_s = rmeta["training_state"]["train_s"]
total_rl = sum(fnum(RUNS["RL"][k]["solve_s"]) for k in KEYS)
total_m = sum(fnum(RUNS["MILP"][k]["solve_s"]) for k in KEYS)
num.update(RLtotalS=f(total_rl, "{:.0f}"), MILPtotalH=f(total_m / 3600, "{:.1f}"), TrainS=f(train_s, "{:.0f}"),
           BreakEven=f(train_s / mean([fnum(RUNS["MILP"][k]["solve_s"]) for k in KEYS]), "{:.0f}"),
           Speedup=f(mean([fnum(RUNS["MILP"][k]["solve_s"]) for k in KEYS]) / mean([fnum(RUNS["RL"][k]["solve_s"]) for k in KEYS]), "{:,.0f}"))
# RL valid where MILP has none
only = [k for k in KEYS if RUNS["RL"][k]["valid"] == "yes" and RUNS["MILP"][k]["valid"] != "yes"]
rows = [[k[0].replace("EV-", ""), k[1], RUNS["RL"][k]["class"], f(fnum(RUNS["RL"][k]["Z_total"]) / k[1], "{:.0f}"),
         f(fnum(RUNS["RL"][k]["mean_queue_FCS_min"]), "{:.2f}"), RUNS["RL"][k]["vans_used"]] for k in only]
table("r_rlonly", ["file", "$n$", "family", "$Z$ per EV|(Rs)", "mean queue|(min)", "vans|used"], rows, "@{}lrlrrr@{}")

# ====================================================================== 4. instance-level and EV-level EDA (RL1)
vr_ = [RUNS["RL"][k] for k in VALID["RL"]]
zper = [fnum(r["Z_total"]) / int(r["n"]) for r in vr_]
rows = []
for name, fn, fmt in [
        ("objective $Z$ per EV (Rs)", lambda r: fnum(r["Z_total"]) / int(r["n"]), "{:.0f}"),
        ("operator share of $Z$ (\\%)", lambda r: 100 * fnum(r["operator_cost_Z2_to_Z4"]) / fnum(r["Z_total"]), "{:.1f}"),
        ("vans used", lambda r: fnum(r["vans_used"]), "{:.1f}"),
        ("EVs served by a van (\\%)", lambda r: 100 * fnum(r["EVs_by_van"]) / int(r["n"]), "{:.1f}"),
        ("free-choice EVs by van (\\%)", lambda r: fnum(r["share_free_choice_by_van_%"]), "{:.1f}"),
        ("mean FCS queue (min)", lambda r: fnum(r["mean_queue_FCS_min"]), "{:.2f}"),
        ("mean van wait (min)", lambda r: fnum(r["mean_van_wait_min"]), "{:.1f}"),
        ("mean request to charged (min)", lambda r: fnum(r["mean_time_to_charged_min"]), "{:.1f}"),
        ("van km", lambda r: fnum(r["van_km"]), "{:.1f}"),
        ("inference time (s)", lambda r: fnum(r["solve_s"]), "{:.2f}")]:
    v = np.array([x for x in (fn(r) for r in vr_) if x == x])
    vm = np.array([x for x in (fn(RUNS["MILP"][k]) for k in VALID["MILP"]) if x == x]) if "inference" not in name else np.array([np.nan])
    rows.append([name, len(v), f(v.mean(), fmt), f(np.median(v), fmt), f(np.quantile(v, 0.1), fmt), f(np.quantile(v, 0.9), fmt),
                 f(v.min(), fmt), f(v.max(), fmt), f(np.nanmean(vm), fmt)])
table("r_inst_stats", ["measure (per instance)", "count", "mean", "median", "10th pct", "90th pct", "min", "max", "MILP|mean"],
      rows, "@{}lrrrrrrrr@{}")
num.update(ZperMeanR=f(mean(zper), "{:.0f}"), ZperMinR=f(min(zper), "{:.0f}"), ZperMaxR=f(max(zper), "{:.0f}"))
fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.4))
axes[0].boxplot([[fnum(RUNS["RL"][k]["Z_total"]) / n for k in VALID["RL"] if k[1] == n] for n in NS], showfliers=False)
axes[0].set(ylabel="Z per EV (Rs)", xlabel="$n$")
axes[1].boxplot([[fnum(RUNS["RL"][k]["solve_s"]) for k in KEYS if k[1] == n] for n in NS], showfliers=False)
axes[1].set(ylabel="inference time (s)", xlabel="$n$")
for ax in axes:
    ax.set_xticks(range(1, len(NS) + 1))
    ax.set_xticklabels([str(n) for n in NS], fontsize=6.5)
save(fig, "r_inst_by_n")
rows = []
for fam in FAMS:
    gk_ = [k for k in VALID["RL"] if RUNS["RL"][k]["class"] == fam]
    ge = [e for e in FREE["RL"] if e["cls"] == fam]
    rows.append([fam, sum(1 for k in KEYS if RUNS["RL"][k]["class"] == fam), len(gk_),
                 sum(1 for k in VALID["MILP"] if RUNS["MILP"][k]["class"] == fam),
                 f(mean([fnum(RUNS["RL"][k]["Z_total"]) / k[1] for k in gk_]), "{:.0f}"),
                 f(mean([100 * fnum(RUNS["RL"][k]["operator_cost_Z2_to_Z4"]) / fnum(RUNS["RL"][k]["Z_total"]) for k in gk_])),
                 f(share(ge)), f(mean([e["queue_min"] for e in ge if not e["van"]]), "{:.2f}"),
                 f(mean([e["time_to_charged_min"] for e in ge])), f(mean([fnum(RUNS["RL"][k]["vans_used"]) for k in gk_]), "{:.2f}"),
                 f(mean([gaps[k] for k in BOTH if RUNS["RL"][k]["class"] == fam]), "{:.2f}")])
table("r_family", ["family", "inst.", "RL1|valid", "MILP|valid", "Z per EV|(Rs)", "operator|share (\\%)", "free-choice|by van (\\%)",
                   "mean queue|(min)", "request to|charged (min)", "vans|used", "gap to|MILP (\\%)"], rows, "@{}lrrrrrrrrrr@{}")
terms = [("Z1_EV_drive_perceived", "$Z_1$ EV driving"), ("Z2_van_drive", "$Z_2$ van driving"), ("Z3_dispatch", "$Z_3$ van dispatch"),
         ("Z4_operator_energy", "$Z_4$ operator energy"), ("Z5_tariff_perceived", "$Z_5$ tariff"), ("Z6_time_perceived", "$Z_6$ time")]
shares = {m: [100 * sum(fnum(RUNS[m][k][t]) for k in BOTH) / sum(fnum(RUNS[m][k]["Z_total"]) for k in BOTH) for t, _ in terms]
          for m in ("RL", "MILP")}
fig, ax = plt.subplots(figsize=(5.4, 2.6))
x = np.arange(len(terms))
ax.bar(x - 0.2, shares["MILP"], 0.4, color=COLM, label="MILP")
ax.bar(x + 0.2, shares["RL"], 0.4, color=COLR, label="RL1")
ax.set_xticks(x)
ax.set_xticklabels([mpl(lab) for _, lab in terms], fontsize=7, rotation=15)
ax.set(ylabel="share of $Z$ (%), same instances")
ax.legend(fontsize=7.5)
save(fig, "r_obj_compare")
sumZ = {m: sum(fnum(RUNS[m][k]["Z_total"]) for k in BOTH) for m in ("RL", "MILP")}
rows = []
for (t, lab), sr, sm in zip(terms, shares["RL"], shares["MILP"]):
    tr, tm = sum(fnum(RUNS["RL"][k][t]) for k in BOTH), sum(fnum(RUNS["MILP"][k][t]) for k in BOTH)
    rows.append([lab, f(tm / len(BOTH), "{:.0f}"), f(tr / len(BOTH), "{:.0f}"), f(100 * (tr - tm) / sumZ["MILP"], "{:+.2f}"), f(sm), f(sr)])
rows += ["MID", ["$Z$", f(sumZ["MILP"] / len(BOTH), "{:.0f}"), f(sumZ["RL"] / len(BOTH), "{:.0f}"),
                 f(100 * (sumZ["RL"] - sumZ["MILP"]) / sumZ["MILP"], "{:+.2f}"), "100.0", "100.0"]]
table("r_terms", ["term", "MILP|Rs per plan", "RL1|Rs per plan", "RL1 $-$ MILP|(\\% of $Z$)", "MILP|share (\\%)", "RL1|share (\\%)"],
      rows, "@{}lrrrrr@{}")
num["ExtraTotal"] = f(100 * (sumZ["RL"] - sumZ["MILP"]) / sumZ["MILP"], "{:.2f}")
diffs = {lab: 100 * (sum(fnum(RUNS["RL"][k][t]) for k in BOTH) - sum(fnum(RUNS["MILP"][k][t]) for k in BOTH)) / sumZ["MILP"]
         for t, lab in terms}
num.update(ExtraTime=f(diffs["$Z_6$ time"], "{:+.2f}"), ExtraTariff=f(diffs["$Z_5$ tariff"], "{:+.2f}"),
           ExtraEnergy=f(diffs["$Z_4$ operator energy"], "{:+.2f}"), ExtraDispatch=f(diffs["$Z_3$ van dispatch"], "{:+.2f}"),
           ExtraVanKm=f(diffs["$Z_2$ van driving"], "{:+.2f}"), ExtraDrive=f(diffs["$Z_1$ EV driving"], "{:+.2f}"))

# EV-level distributions
fr = FREE["RL"]
evr = EV["RL"]
rows = []
for name, vals, valsm, fmt in [
        ("income $\\theta$", [e["theta"] for e in evr], [e["theta"] for e in EV["MILP"]], "{:.2f}"),
        ("drive to FCS (km), FCS users", [e["drive_km"] for e in fr if not e["van"]], [e["drive_km"] for e in FREE["MILP"] if not e["van"]], "{:.2f}"),
        ("FCS queue (min), FCS users", [e["queue_min"] for e in fr if not e["van"]], [e["queue_min"] for e in FREE["MILP"] if not e["van"]], "{:.2f}"),
        ("van wait (min), van users", [e["van_wait_min"] for e in evr if e["van"]], [e["van_wait_min"] for e in EV["MILP"] if e["van"]], "{:.1f}"),
        ("charging time (min)", [e["charge_min"] for e in evr], [e["charge_min"] for e in EV["MILP"]], "{:.1f}"),
        ("request to charged (min)", [e["time_to_charged_min"] for e in evr], [e["time_to_charged_min"] for e in EV["MILP"]], "{:.1f}"),
        ("money paid (Rs)", [e["pay_Rs"] for e in evr], [e["pay_Rs"] for e in EV["MILP"]], "{:.0f}"),
        ("perceived money (Rs)", [e["perceived_money_Rs"] for e in evr], [e["perceived_money_Rs"] for e in EV["MILP"]], "{:.0f}"),
        ("perceived time (Rs)", [e["time_cost_Rs"] for e in evr], [e["time_cost_Rs"] for e in EV["MILP"]], "{:.0f}"),
        ("perceived total (Rs)", [e["customer_cost_Rs"] for e in evr], [e["customer_cost_Rs"] for e in EV["MILP"]], "{:.0f}")]:
    v = np.array([x for x in vals if x == x])
    rows.append([name, len(v), f(v.mean(), fmt), f(np.median(v), fmt), f(np.quantile(v, 0.1), fmt), f(np.quantile(v, 0.9), fmt),
                 f(v.max(), fmt), f(mean(valsm), fmt)])
table("r_ev_stats", ["variable (per EV)", "count", "mean", "median", "10th pct", "90th pct", "max", "MILP|mean"], rows, "@{}lrrrrrrr@{}")
fig, axes = plt.subplots(2, 3, figsize=(6.9, 4.4))
ax = axes[0, 0]
ax.hist([e["theta"] for e in fr if not e["van"]], bins=20, range=(0, 1), density=True, alpha=0.6, color=COLM, label="FCS users")
ax.hist([e["theta"] for e in fr if e["van"]], bins=20, range=(0, 1), density=True, alpha=0.6, color=COLR, label="van users")
ax.set(xlabel=r"income $\theta$", ylabel="density", title="income of free-choice EVs")
ax.legend(fontsize=7)
ax = axes[0, 1]
ax.hist([e["time_to_charged_min"] for e in evr if not e["van"]], bins=30, range=(0, 200), alpha=0.6, color=COLM, label="FCS")
ax.hist([e["time_to_charged_min"] for e in evr if e["van"]], bins=30, range=(0, 200), alpha=0.6, color=COLR, label="van")
ax.set(xlabel="request to charged (min)", ylabel="EVs", title="time until fully charged")
ax.legend(fontsize=7)
ax = axes[0, 2]
qs = [e["queue_min"] for e in fr if not e["van"] and e["queue_min"] > 1e-6]
ax.hist(qs, bins=30, color="#DD8452")
ax.set(xlabel="queue (min)", ylabel="EVs", title="FCS queues above zero")
ax = axes[1, 0]
ax.hist([e["van_wait_min"] for e in evr if e["van"]], bins=30, color=COLR)
ax.set(xlabel="wait for the van (min)", ylabel="EVs", title="van waits")
ax = axes[1, 1]
ax.hist([e["pay_Rs"] for e in evr if not e["van"]], bins=30, alpha=0.6, color=COLM, label="FCS")
ax.hist([e["pay_Rs"] for e in evr if e["van"]], bins=30, alpha=0.6, color=COLR, label="van")
ax.set(xlabel="money paid (Rs)", ylabel="EVs", title="money paid")
ax.legend(fontsize=7)
ax = axes[1, 2]
ax.hist([e["customer_cost_Rs"] for e in evr], bins=30, color="#55A868")
ax.set(xlabel="perceived cost (Rs)", ylabel="EVs", title="perceived cost")
save(fig, "r_ev_dists")
num.update(NQueuedR=len(qs), NFCSfreeR=len([e for e in fr if not e["van"]]), NEVR=len(evr), NFreeR=len(fr),
           NStrandedR=len(evr) - len(fr))


def ranks(x):
    order = np.argsort(x, kind="mergesort")
    rk = np.empty(len(x))
    xs = np.asarray(x)[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        rk[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return rk


vars_ = [("theta", r"$\theta$"), ("E_kWh", "$E$"), ("nearest_reachable_fcs_km", "nearest FCS"), ("pressure", "pressure"),
         ("van", "van"), ("queue_min", "queue"), ("time_to_charged_min", "time to charged"), ("pay_Rs", "paid"),
         ("perceived_money_Rs", "perc. money"), ("time_cost_Rs", "perc. time")]
Mx = np.array([[e[k] for k, _ in vars_] for e in fr], dtype=float)
Cr = np.corrcoef(np.column_stack([ranks(Mx[:, k]) for k in range(Mx.shape[1])]), rowvar=False)
Mm = np.array([[e[k] for k, _ in vars_] for e in FREE["MILP"]], dtype=float)
Cm = np.corrcoef(np.column_stack([ranks(Mm[:, k]) for k in range(Mm.shape[1])]), rowvar=False)
fig, ax = plt.subplots(figsize=(5.2, 4.3))
im = ax.imshow(Cr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(vars_)))
ax.set_yticks(range(len(vars_)))
ax.set_xticklabels([l for _, l in vars_], rotation=45, ha="right", fontsize=7)
ax.set_yticklabels([l for _, l in vars_], fontsize=7)
for a in range(len(vars_)):
    for b in range(len(vars_)):
        ax.text(b, a, f"{Cr[a, b]:.2f}", ha="center", va="center", fontsize=5.5, color="white" if abs(Cr[a, b]) > 0.6 else "black")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
save(fig, "r_corr")
ix = {k: i for i, (k, _) in enumerate(vars_)}
num.update(CorrThetaVanR=f(Cr[ix["theta"], ix["van"]], "{:.2f}"), CorrThetaVanM=f(Cm[ix["theta"], ix["van"]], "{:.2f}"),
           CorrThetaQueueR=f(Cr[ix["theta"], ix["queue_min"]], "{:.2f}"), CorrPressQueueR=f(Cr[ix["pressure"], ix["queue_min"]], "{:.2f}"),
           CorrMaxDiff=f(np.max(np.abs(Cr - Cm)), "{:.2f}"))

# ====================================================================== 5. decision agreement with the MILP
agree = []
for k in BOTH:
    pr, pm = PLANS["RL"][k]["ev"], PLANS["MILP"][k]["ev"]
    for i in pr:
        a = ATTR[k][i]
        rr, rm = pr[i]["resource"], pm[i]["resource"]
        tr = "van" if rr.startswith("van") else "FCS"
        tm = "van" if rm.startswith("van") else "FCS"
        sr = rr.split("-")[0] if tr == "FCS" else "van"
        sm = rm.split("-")[0] if tm == "FCS" else "van"
        agree.append({"opt": RUNS["MILP"][k]["status"] == "optimal", "free": a["stranded"] == "no", "theta": fnum(a["theta"]),
                      "tr": tr, "tm": tm, "same_type": tr == tm, "same_station": (sr == sm) if tm == "FCS" and tr == "FCS" else None})
fa = [a for a in agree if a["free"]]
cm = Counter((a["tm"], a["tr"]) for a in fa)
table("r_confusion", ["MILP resource $\\downarrow$ / RL1 resource $\\rightarrow$", "FCS", "van", "agreement (\\%)"],
      [["FCS", cm[("FCS", "FCS")], cm[("FCS", "van")], f(100 * cm[("FCS", "FCS")] / max(1, cm[("FCS", "FCS")] + cm[("FCS", "van")]))],
       ["van", cm[("van", "FCS")], cm[("van", "van")], f(100 * cm[("van", "van")] / max(1, cm[("van", "FCS")] + cm[("van", "van")]))]],
      "@{}lrrr@{}")
ss = [a["same_station"] for a in fa if a["same_station"] is not None]
num.update(AgreeType=f(100 * mean([a["same_type"] for a in fa])), AgreeTypeOpt=f(100 * mean([a["same_type"] for a in fa if a["opt"]])),
           AgreeStation=f(100 * mean(ss)), AgreeVanVan=f(100 * cm[("van", "van")] / max(1, cm[("van", "FCS")] + cm[("van", "van")])),
           AgreeFcsFcs=f(100 * cm[("FCS", "FCS")] / max(1, cm[("FCS", "FCS")] + cm[("FCS", "van")])),
           RLvanMILPfcs=cm[("FCS", "van")], RLfcsMILPvan=cm[("van", "FCS")])
ag_band = [100 * mean([a["same_type"] for a in fa if lo <= a["theta"] < hi]) for lo, hi in BANDS]

# ====================================================================== 6. income and the choice
dec = {}
for m in ("RL", "MILP"):
    dec[m] = []
    for b in range(10):
        g_ = [e for e in FREE[m] if min(int(e["theta"] * 10), 9) == b]
        dec[m].append((len(g_), sum(e["van"] for e in g_)))
dec_opt = []
for b in range(10):
    g_ = [e for e in FREE["MILP"] if e["opt"] and min(int(e["theta"] * 10), 9) == b]
    dec_opt.append(100 * mean([e["van"] for e in g_]))
rows = []
for b in range(10):
    nr, kr = dec["RL"][b]
    nm_, km_ = dec["MILP"][b]
    lo, hi = wilson(kr, nr)
    rows.append([f"{b / 10:.1f}--{(b + 1) / 10:.1f}", nr, f(100 * kr / nr if nr else float("nan")), f"[{f(lo)}, {f(hi)}]",
                 f(100 * km_ / nm_ if nm_ else float("nan")), f(dec_opt[b])])
table("r_deciles", ["income $\\theta$", "EVs|(RL1)", "RL1 by|van (\\%)", "RL1 95\\,\\%|interval", "MILP by|van (\\%)",
                    "MILP optimal|by van (\\%)"], rows, "@{}lrrrrr@{}")
THETA_STAR = 0.66
fig, ax = plt.subplots(figsize=(4.9, 2.9))
mids = [b / 10 + 0.05 for b in range(10)]
ax.plot(mids, [100 * k_ / n_ for n_, k_ in dec["RL"]], "o-", color=COLR, ms=3.5, lw=1.3, label="RL1 (all valid plans)")
ax.plot(mids, [100 * k_ / n_ for n_, k_ in dec["MILP"]], "s--", color=COLM, ms=3.2, lw=1.1, label="MILP (all valid plans)")
ax.plot(mids, dec_opt, "^:", color="#333333", ms=3.2, lw=1.0, label="MILP proven-optimal plans")
ax.axvline(THETA_STAR, color="grey", ls="--", lw=0.9, label=r"theory: $\theta^*$ = 0.66")
ax.set(xlabel=r"income $\theta$ (decile midpoint)", ylabel="free-choice EVs by van (%)", xlim=(0, 1))
ax.set_ylim(bottom=0)
ax.legend(fontsize=7, loc="upper left")
save(fig, "r_van_by_income")
num.update(VanAllR=f(share(FREE["RL"])), VanAllM=f(share(FREE["MILP"])),
           VanLowR=f(share([e for e in FREE["RL"] if e["theta"] < 0.4])), VanLowM=f(share([e for e in FREE["MILP"] if e["theta"] < 0.4])),
           VanTopR=f(share([e for e in FREE["RL"] if e["band"] == 4])), VanTopM=f(share([e for e in FREE["MILP"] if e["band"] == 4])),
           VanSixEightR=f(share([e for e in FREE["RL"] if 0.6 <= e["theta"] < 0.8])),
           ThetaVanR=f(mean([e["theta"] for e in FREE["RL"] if e["van"]]), "{:.2f}"),
           ThetaFCSR=f(mean([e["theta"] for e in FREE["RL"] if not e["van"]]), "{:.2f}"))
# first decile whose RL van share exceeds 5 %
step = next((b for b in range(10) if dec["RL"][b][0] and 100 * dec["RL"][b][1] / dec["RL"][b][0] > 5), None)
num["StepR"] = f"{step / 10:.1f}--{(step + 1) / 10:.1f}" if step is not None else "--"
fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.6))
x = np.arange(5)
axes[0].bar(x - 0.2, [share([e for e in FREE["MILP"] if e["band"] == b]) for b in range(5)], 0.4, color=COLM, label="MILP")
axes[0].bar(x + 0.2, [share([e for e in FREE["RL"] if e["band"] == b]) for b in range(5)], 0.4, color=COLR, label="RL1")
axes[0].set(ylabel="served by a van (%)", xlabel=r"income band $\theta$")
axes[0].legend(fontsize=7.5)
axes[1].bar(x, ag_band, 0.6, color="#55A868")
axes[1].set_ylim(max(0, 10 * math.floor((min(ag_band) - 5) / 10)), 100)
axes[1].set(ylabel="same resource type as MILP (%)", xlabel=r"income band $\theta$")
for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels(PLAB, fontsize=7)
save(fig, "r_agreement")
num.update(AgreeBandMin=f(min(ag_band)), AgreeBandMax=f(max(ag_band)))


def logit(evs):
    rows_ = [e for e in evs if e["nearest_reachable_fcs_km"] == e["nearest_reachable_fcs_km"]]
    X = np.array([[1.0, e["theta"], e["lvl"] == "mid", e["lvl"] == "high", e["theta"] * (e["lvl"] == "high"), e["h2"],
                   e["E_kWh"] / 10, e["nearest_reachable_fcs_km"]] for e in rows_], dtype=float)
    y = np.array([e["van"] for e in rows_], dtype=float)
    lam = 1e-3 * np.eye(X.shape[1])
    lam[0, 0] = 0
    b = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -35, 35)))
        step_ = np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]) + lam, X.T @ (y - p) - lam @ b)
        b += step_
        if np.max(np.abs(step_)) < 1e-9:
            break
    p = 1 / (1 + np.exp(-np.clip(X @ b, -35, 35)))
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]) + lam)))
    ame = (b[1] + b[4] * X[:, 3]) * p * (1 - p)
    hi_ = X[:, 3] == 1
    return b, se, {"AME": 100 * 0.1 * float(np.mean(ame)), "AMEhigh": 100 * 0.1 * float(np.mean(ame[hi_])),
                   "AMEnot": 100 * 0.1 * float(np.mean(ame[~hi_])), "acc": 100 * float(np.mean((p >= 0.5) == (y == 1))), "n": len(rows_)}


bR, seR, infR = logit(FREE["RL"])
bM, seM, infM = logit(FREE["MILP"])
names = ["intercept", "income $\\theta$", "medium pressure", "high pressure", "$\\theta\\times$ high pressure", "long horizon (type 2)",
         "energy $E_i$ (per 10 kWh)", "nearest reachable FCS (km)"]
rows = []
for k_, nm in enumerate(names):
    z = bR[k_] / seR[k_]
    pv = math.erfc(abs(z) / math.sqrt(2))
    rows.append([nm, f(bR[k_], "{:.3f}"), f(seR[k_], "{:.3f}"), f(z, "{:.1f}"), "$<0.001$" if pv < 0.001 else f(pv, "{:.3f}"), f(bM[k_], "{:.3f}")])
table("r_logit", ["term", "RL1 coef.", "std.\\ error", "$z$", "$p$-value", "MILP coef."], rows, "@{}lrrrrr@{}")
num.update(AMER=f(infR["AME"]), AMEM=f(infM["AME"]), AMEhighR=f(infR["AMEhigh"]), AMEnotR=f(infR["AMEnot"]),
           AccR=f(infR["acc"]), LogitNR=infR["n"])

# ====================================================================== 7. queues, pressure, horizon
def queue_rows(fe):
    fc = [e for e in fe if not e["van"]]
    qt = sum(e["queue_min"] for e in fc)
    out = []
    for b in range(5):
        g_ = [e for e in fc if e["band"] == b]
        out.append({"users": 100 * len(g_) / len(fc), "minutes": 100 * sum(e["queue_min"] for e in g_) / qt if qt else float("nan"),
                    "queueing": 100 * mean([e["queue_min"] > 1e-6 for e in g_]), "mean": mean([e["queue_min"] for e in g_]),
                    "drive": mean([e["drive_km"] for e in g_]), "n": len(g_)})
    return out


QR, QM = queue_rows(FREE["RL"]), queue_rows(FREE["MILP"])
table("r_queues", ["income $\\theta$", "RL1 FCS|users", "share of|users (\\%)", "share of queue|minutes (\\%)", "queueing|(\\%)",
                   "mean queue|(min)", "drive to|FCS (km)", "MILP share of|queue min. (\\%)", "MILP drive|(km)"],
      [[BLAB[b], QR[b]["n"], f(QR[b]["users"]), f(QR[b]["minutes"]), f(QR[b]["queueing"]), f(QR[b]["mean"], "{:.2f}"),
        f(QR[b]["drive"], "{:.2f}"), f(QM[b]["minutes"]), f(QM[b]["drive"], "{:.2f}")] for b in range(5)], "@{}lrrrrrrrr@{}")
num.update(QShareBottomR=f(QR[0]["minutes"] + QR[1]["minutes"]), UShareBottomR=f(QR[0]["users"] + QR[1]["users"]),
           QShareTopR=f(QR[4]["minutes"]), QShareBottomM=f(QM[0]["minutes"] + QM[1]["minutes"]), QShareTopM=f(QM[4]["minutes"]),
           DriveBotR=f(QR[0]["drive"], "{:.2f}"), DriveTopR=f(QR[4]["drive"], "{:.2f}"))
fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.6))
x = np.arange(5)
axes[0].bar(x - 0.27, [q["users"] for q in QR], 0.27, color="#9DB4D3", label="share of FCS users (RL1)")
axes[0].bar(x, [q["minutes"] for q in QR], 0.27, color=COLR, label="share of queue minutes (RL1)")
axes[0].bar(x + 0.27, [q["minutes"] for q in QM], 0.27, color=COLM, label="share of queue minutes (MILP)")
axes[0].set(ylabel="%", xlabel=r"income band $\theta$")
axes[0].set_ylim(0, 1.35 * max(max(q["users"] for q in QR), max(q["minutes"] for q in QR + QM)))
axes[0].legend(fontsize=6.5, loc="upper right")
axes[1].plot(x, [q["drive"] for q in QM], "s--", color=COLM, label="MILP")
axes[1].plot(x, [q["drive"] for q in QR], "o-", color=COLR, label="RL1")
axes[1].set(ylabel="mean drive to the FCS (km)", xlabel=r"income band $\theta$")
axes[1].set_ylim(bottom=0)
axes[1].legend(fontsize=7.5)
for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels(PLAB, fontsize=7)
save(fig, "r_queues")
rows = []
for L, nm in (("low", "low"), ("mid", "medium"), ("high", "high")):
    g_ = [e for e in FREE["RL"] if e["lvl"] == L]
    gm_ = [e for e in FREE["MILP"] if e["lvl"] == L]
    fc = [e for e in g_ if not e["van"]]
    rows.append([nm, len(g_), f(100 * mean([e["queue_min"] > 1e-6 for e in fc])), f(mean([e["queue_min"] for e in fc]), "{:.2f}"),
                 f(share(g_))] + [f(share([e for e in g_ if e["band"] == b])) for b in range(5)] +
                [f(mean([e["queue_min"] for e in gm_ if not e["van"]]), "{:.2f}"), f(share([e for e in gm_ if e["band"] == 4]))])
table("r_pressure", ["demand|pressure", "EVs", "queueing|(\\%)", "mean queue|(min)", "by van|(\\%)"] +
      [f"by van|$\\theta$ {b}" for b in BLAB] + ["MILP mean|queue (min)", "MILP top|band van (\\%)"], rows, "@{}lrrrrrrrrrrr@{}")
for L, tag in (("low", "Low"), ("mid", "Mid"), ("high", "High")):
    g_ = [e for e in FREE["RL"] if e["lvl"] == L]
    fc = [e for e in g_ if not e["van"]]
    num[f"QR{tag}"] = f(mean([e["queue_min"] for e in fc]), "{:.2f}")
    num[f"TopR{tag}"] = f(share([e for e in g_ if e["band"] == 4]))
    num[f"LowQR{tag}"] = f(mean([e["queue_min"] for e in fc if e["band"] <= 1]), "{:.2f}")
    num[f"TopQR{tag}"] = f(mean([e["queue_min"] for e in fc if e["band"] == 4]), "{:.2f}")
fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.7))
cols = {"low": "#4C72B0", "mid": "#DD8452", "high": "#C44E52"}
for L, nm in (("low", "low pressure"), ("mid", "medium"), ("high", "high pressure")):
    g_ = [e for e in FREE["RL"] if e["lvl"] == L]
    gm_ = [e for e in FREE["MILP"] if e["lvl"] == L]
    axes[0].plot(x, [share([e for e in g_ if e["band"] == b]) for b in range(5)], "o-", color=cols[L], ms=4, lw=1.3, label=f"RL1, {nm}")
    axes[0].plot(x, [share([e for e in gm_ if e["band"] == b]) for b in range(5)], ":", color=cols[L], lw=1.0)
    axes[1].plot(x, [mean([e["queue_min"] for e in g_ if e["band"] == b and not e["van"]]) for b in range(5)], "o-", color=cols[L], ms=4, lw=1.3, label=f"RL1, {nm}")
    axes[1].plot(x, [mean([e["queue_min"] for e in gm_ if e["band"] == b and not e["van"]]) for b in range(5)], ":", color=cols[L], lw=1.0)
axes[0].set_ylabel("served by a van (%)")
axes[1].set_ylabel("mean FCS queue (min)")
for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels(PLAB, fontsize=7)
    ax.set_xlabel(r"income band $\theta$")
    ax.set_ylim(bottom=0)
axes[0].legend(fontsize=6.5, loc="upper left", title="dotted: MILP", title_fontsize=6.5)
save(fig, "r_pressure")
rows = []
for fam in FAMS:
    g_ = [e for e in FREE["RL"] if e["cls"] == fam]
    rows.append([fam, "short" if fam.endswith("1") else "long", len(g_), f(mean([e["pressure"] for e in g_]), "{:.2f}"), f(share(g_))] +
                [f(share([e for e in g_ if e["band"] == b])) for b in range(5)] +
                [f(share([e for e in FREE["MILP"] if e["cls"] == fam and e["band"] == 4]))])
table("r_families", ["family", "horizon", "EVs", "mean|pressure", "by van|(\\%)"] + [f"by van|$\\theta$ {b}" for b in BLAB] +
      ["MILP|top band"], rows, "@{}llrrrrrrrrr@{}")
fig, ax = plt.subplots(figsize=(4.9, 2.7))
for h2, lab, col, mk in ((False, "short horizon (type 1)", COLR, "o"), (True, "long horizon (type 2)", COLM, "s")):
    g_ = [e for e in FREE["RL"] if e["h2"] == h2]
    gm_ = [e for e in FREE["MILP"] if e["h2"] == h2]
    ax.plot(x, [share([e for e in g_ if e["band"] == b]) for b in range(5)], marker=mk, color=col, ms=4, lw=1.3, label=f"RL1, {lab}")
    ax.plot(x, [share([e for e in gm_ if e["band"] == b]) for b in range(5)], ls=":", color=col, lw=1.0, label=f"MILP, {lab}")
ax.set_xticks(x)
ax.set_xticklabels(PLAB, fontsize=7)
ax.set(xlabel=r"income band $\theta$", ylabel="served by a van (%)")
ax.set_ylim(bottom=0)
ax.legend(fontsize=6.5, loc="upper left")
save(fig, "r_horizon")
for h2, tag in ((False, "Short"), (True, "Long")):
    num[f"Top{tag}R"] = f(share([e for e in FREE["RL"] if e["h2"] == h2 and e["band"] == 4]))

# ====================================================================== 8. vans
def van_stats(m):
    vs = []
    for k in VALID[m]:
        T = JS[k]["T"]
        for v in PLANS[m][k]["van"]:
            stops, drawn, back = fnum(v["stops"]), fnum(v["kWh_drawn"]), fnum(v["return_min"])
            util = {"seats ($Q_m=6$)": stops / 6, "battery ($B_m=150$ kWh)": drawn / 150, "time ($T$)": back / T}
            vs.append({"cls": RUNS[m][k]["class"], "stops": stops, "drawn": drawn, "time": back / T, "km": fnum(v["km"]),
                       "tight": max(util, key=util.get)})
    seats = Counter("stranded" if not e["free"] else "free" for e in EV[m] if e["van"])
    return vs, seats


VR, SR = van_stats("RL")
VM, SM = van_stats("MILP")
rows = [["vans dispatched (all valid plans)", len(VR), "", "", len(VM)],
        ["stops per van", f(mean([v["stops"] for v in VR]), "{:.2f}"), f(med([v["stops"] for v in VR]), "{:.0f}"),
         f"{100 * mean([v['stops'] == 6 for v in VR]):.1f}\\,\\% at $Q_m=6$", f(mean([v["stops"] for v in VM]), "{:.2f}")],
        ["battery drawn (kWh)", f(mean([v["drawn"] for v in VR]), "{:.1f}"), f(med([v["drawn"] for v in VR]), "{:.1f}"),
         f"{100 * mean([v['drawn'] >= 135 for v in VR]):.1f}\\,\\% above 90\\,\\% of $B_m$", f(mean([v["drawn"] for v in VM]), "{:.1f}")],
        ["back at depot (\\% of $T$)", f(100 * mean([v["time"] for v in VR])), f(100 * med([v["time"] for v in VR])),
         f"{100 * mean([v['time'] >= 0.9 for v in VR]):.1f}\\,\\% after 90\\,\\% of $T$", f(100 * mean([v["time"] for v in VM]))],
        ["van km", f(mean([v["km"] for v in VR])), f(med([v["km"] for v in VR])), "", f(mean([v["km"] for v in VM]))]]
table("r_vans", ["measure", "RL1 mean", "RL1 median", "RL1 at the limit", "MILP mean"], rows, "@{}lrrlr@{}")
tc = Counter(v["tight"] for v in VR)
tcm = Counter(v["tight"] for v in VM)
table("r_vans_tight", ["tightest van resource", "RL1 vans", "RL1 share (\\%)", "MILP share (\\%)"],
      [[k, tc[k], f(100 * tc[k] / len(VR)), f(100 * tcm[k] / len(VM))] for k in ("seats ($Q_m=6$)", "battery ($B_m=150$ kWh)", "time ($T$)")],
      "@{}lrrr@{}")
num.update(VansNR=len(VR), VansNM=len(VM), VansFullR=f(100 * mean([v["stops"] == 6 for v in VR])),
           SeatsStrandedR=f(100 * SR["stranded"] / (SR["stranded"] + SR["free"])),
           SeatsStrandedM=f(100 * SM["stranded"] / (SM["stranded"] + SM["free"])),
           TightSeatsR=f(100 * tc["seats ($Q_m=6$)"] / len(VR)), TightTimeR=f(100 * tc["time ($T$)"] / len(VR)),
           TightBattR=f(100 * tc["battery ($B_m=150$ kWh)"] / len(VR)),
           VansPerPlanR=f(len(VR) / len(VALID["RL"]), "{:.2f}"), VansPerPlanM=f(len(VM) / len(VALID["MILP"]), "{:.2f}"))
rows = []
for fam in FAMS:
    g_ = [v for v in VR if v["cls"] == fam]
    c_ = Counter(v["tight"] for v in g_)
    rows.append([fam, len(g_), f(mean([v["stops"] for v in g_]), "{:.2f}"), f(100 * mean([v["time"] for v in g_])),
                 f(100 * c_["seats ($Q_m=6$)"] / len(g_) if g_ else float("nan")), f(100 * c_["time ($T$)"] / len(g_) if g_ else float("nan"))])
table("r_vans_family", ["family", "RL1 vans", "stops|per van", "back at depot|(\\% of $T$)", "tightest:|seats (\\%)", "tightest:|time (\\%)"],
      rows, "@{}lrrrrr@{}")
fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.3))
axes[0].hist([[v["stops"] for v in VM], [v["stops"] for v in VR]], bins=np.arange(0.5, 7.5, 1), color=[COLM, COLR], label=["MILP", "RL1"])
axes[0].set(xlabel="EVs served by one van", ylabel="vans")
axes[0].legend(fontsize=7)
axes[1].hist([[v["drawn"] for v in VM], [v["drawn"] for v in VR]], bins=15, range=(0, 150), color=[COLM, COLR])
axes[1].set(xlabel="battery drawn (kWh)")
axes[2].hist([[100 * v["time"] for v in VM], [100 * v["time"] for v in VR]], bins=15, range=(0, 100), color=[COLM, COLR])
axes[2].set(xlabel="back at depot (% of $T$)")
save(fig, "r_vans")

# ====================================================================== 9. equity
rows = []
paid_t = sum(e["pay_Rs"] for e in FREE["RL"])
felt_t = sum(e["customer_cost_Rs"] for e in FREE["RL"])
for b in range(5):
    g_ = [e for e in FREE["RL"] if e["band"] == b]
    gm_ = [e for e in FREE["MILP"] if e["band"] == b]
    rows.append([BLAB[b], f(100 * len(g_) / len(FREE["RL"])), f(100 * sum(e["pay_Rs"] for e in g_) / paid_t),
                 f(100 * sum(e["customer_cost_Rs"] for e in g_) / felt_t), f(sum(e["customer_cost_Rs"] for e in g_) / sum(e["pay_Rs"] for e in g_), "{:.2f}"),
                 f(mean([e["customer_cost_Rs"] for e in g_]), "{:.0f}"), f(mean([e["customer_cost_Rs"] for e in gm_]), "{:.0f}"),
                 f(mean([e["pay_Rs"] / e["E_kWh"] for e in g_]), "{:.1f}")])
table("r_equity", ["income $\\theta$", "share of|EVs (\\%)", "share of|money paid (\\%)", "share of|perceived (\\%)", "perceived per|Rs paid",
                   "RL1 perceived|per EV (Rs)", "MILP perceived|per EV (Rs)", "paid per|kWh (Rs)"], rows, "@{}lrrrrrrr@{}")
num.update(PerRsBotR=rows[0][4], PerRsTopR=rows[4][4])
fig, ax = plt.subplots(figsize=(4.9, 2.7))
money = [mean([e["perceived_money_Rs"] for e in FREE["RL"] if e["band"] == b]) for b in range(5)]
tcost = [mean([e["time_cost_Rs"] for e in FREE["RL"] if e["band"] == b]) for b in range(5)]
paid = [mean([e["pay_Rs"] for e in FREE["RL"] if e["band"] == b]) for b in range(5)]
feltm = [mean([e["customer_cost_Rs"] for e in FREE["MILP"] if e["band"] == b]) for b in range(5)]
ax.bar(x, money, 0.6, color=COLM, label=r"perceived money ($\mu_i\times$ paid)")
ax.bar(x, tcost, 0.6, bottom=money, color="#DD8452", label="perceived time")
ax.plot(x, paid, "k_", ms=20, mew=1.6, label="money paid")
ax.plot(x, feltm, "D", color="#333333", ms=3.5, label="MILP perceived total")
ax.set_xticks(x)
ax.set_xticklabels(PLAB, fontsize=7)
ax.set(xlabel=r"income band $\theta$", ylabel="Rs per EV (RL1)")
ax.set_ylim(0, 1.5 * max(m_ + t_ for m_, t_ in zip(money, tcost)))
ax.legend(fontsize=6.5, loc="upper left", ncol=2)
save(fig, "r_costs")

# ====================================================================== 10. elasticities (envelope theorem), RL1 vs MILP
def elasticities(m, keys):
    keyset = set(keys)
    evs = [e for e in EV[m] if (e["instance"], e["n"]) in keyset]
    Z = sum(fnum(RUNS[m][k]["Z_total"]) for k in keys)
    fcs = [e for e in evs if not e["van"]]
    van = [e for e in evs if e["van"]]
    vot = P3["value_of_time_Rs_per_min"]
    q = {"$c^{\\mathrm{EV}}$": (C_EV, sum(e["mu"] * e["drive_km"] for e in fcs)),
         "$c^{\\mathrm M}$": (C_M, sum(fnum(RUNS[m][k]["van_km"]) for k in keys)),
         "$c^{\\mathrm F}_m$": (C_F, sum(fnum(RUNS[m][k]["vans_used"]) for k in keys)),
         "$\\pi^{\\mathrm F}$": (PI_F, sum(e["E_kWh"] for e in fcs)),
         "$\\pi^{\\mathrm M}$": (PI_M, sum(e["E_kWh"] / ETA for e in van)),
         "$p^{\\mathrm F}$": (P3["tariff_Rs_per_kWh"]["p_F"], sum(e["mu"] * e["E_kWh"] for e in fcs)),
         "$p^{\\mathrm M}$": (P3["tariff_Rs_per_kWh"]["p_M"], sum(e["mu"] * e["E_kWh"] for e in van)),
         "$\\mu^{0}$": (P3["money_sensitivity_mu"]["theta0"], sum((1 - e["theta"]) * e["pay_Rs"] for e in evs)),
         "$\\mu^{1}$": (P3["money_sensitivity_mu"]["theta1"], sum(e["theta"] * e["pay_Rs"] for e in evs))}
    tq = {"D": lambda e: e["drive_min"] if not e["van"] else 0.0, "Q": lambda e: e["queue_min"] if not e["van"] else 0.0,
          "C": lambda e: e["charge_min"] if not e["van"] else 0.0, "V": lambda e: (e["van_wait_min"] + e["charge_min"]) if e["van"] else 0.0}
    for a in "DQCV":
        q[f"$c^{{\\mathrm{{{a}}},0}}$"] = (vot[a]["theta0"], sum((1 - e["theta"]) * tq[a](e) for e in evs))
        q[f"$c^{{\\mathrm{{{a}}},1}}$"] = (vot[a]["theta1"], sum(e["theta"] * tq[a](e) for e in evs))
    return {k_: 100 * c_ * qq / Z for k_, (c_, qq) in q.items()}, {k_: c_ for k_, (c_, qq) in q.items()}


ER, coefv = elasticities("RL", BOTH)
EM, _ = elasticities("MILP", BOTH)
order = sorted(ER, key=lambda k_: -EM[k_])
table("r_elasticity", ["coefficient", "value", "MILP elasticity (\\%)", "RL1 elasticity (\\%)", "difference"],
      [[k_, f"{coefv[k_]:g}", f(EM[k_], "{:.2f}"), f(ER[k_], "{:.2f}"), f(ER[k_] - EM[k_], "{:+.2f}")] for k_ in order], "@{}lrrrr@{}")
fig, ax = plt.subplots(figsize=(5.0, 4.2))
y = np.arange(len(order))[::-1]
ax.barh(y + 0.2, [EM[k_] for k_ in order], 0.4, color=COLM, label="MILP")
ax.barh(y - 0.2, [ER[k_] for k_ in order], 0.4, color=COLR, label="RL1")
ax.set_yticks(y)
ax.set_yticklabels([mpl(k_) for k_ in order], fontsize=8)
ax.set(xlabel="elasticity of $Z$ (%)")
ax.legend(fontsize=7.5)
save(fig, "r_elasticity")
dmax = max(order, key=lambda k_: abs(ER[k_] - EM[k_]))
num.update(ElDiffMax=f(abs(ER[dmax] - EM[dmax]), "{:.2f}"), ElDiffMaxName=dmax, ElPFR=f(ER["$p^{\\mathrm F}$"]), ElMuZeroR=f(ER["$\\mu^{0}$"]),
           ElCCOneR=f(ER["$c^{\\mathrm{C},1}$"]), ElPiFR=f(ER["$\\pi^{\\mathrm F}$"]), ElCFR=f(ER["$c^{\\mathrm F}_m$"]),
           ElCFM=f(EM["$c^{\\mathrm F}_m$"]), ElPMR=f(ER["$p^{\\mathrm M}$"]), ElPMM=f(EM["$p^{\\mathrm M}$"]))

# ====================================================================== 11. shadow prices
def duals(rid):
    c = list(csv.DictReader(open(os.path.join(DAT, f"duals_{rid}_constraints.csv"))))
    i = list(csv.DictReader(open(os.path.join(DAT, f"duals_{rid}_instances.csv"))))
    for r in c:
        r["dual"] = fnum(r["dual"])
    return c, i


DCR, DIR = duals("rl1")
DCM, DIM = duals("milp_300s")
kinds = [("gun", "gun order $S_j\\ge S_i+\\tau^{\\mathrm F}_i$", "value of one gun-minute"),
         ("van", "van order $S_j\\ge S_i+\\tau^{\\mathrm M}_i+t_{ij}$", "value of one van-minute"),
         ("depart", "first stop $S_a\\ge t_{0a}$", "value of an earlier departure"),
         ("arrival", "FCS arrival $S_i\\ge r_i+t_{if}$", "not unique (degenerate)"),
         ("release", "van release $S_i\\ge r_i$", "not unique (degenerate)"),
         ("patience", "queue limit $q^{\\mathrm F}_i\\le W_i$", "value of patience"),
         ("deadline", "deadline $C_i\\le T$, van return", "value of horizon")]


def dstat(c, k_, nplans_ev, nvans):
    if k_ == "deadline":
        v = np.array([r["dual"] for r in c if r["kind"] in ("deadline_fcs", "deadline_van", "return")])
        rows_n = nplans_ev + nvans
    else:
        v = np.array([r["dual"] for r in c if r["kind"] == k_])
        rows_n = len(v)
    nz = v[np.abs(v) > 1e-9]
    return rows_n, nz


rows = []
st = {}
for k_, lab, meaning in kinds:
    nr, nzr = dstat(DCR, k_, len(EV["RL"]), len(VR))
    nm_, nzm = dstat(DCM, k_, len(EV["MILP"]), len(VM))
    st[k_] = (nr, nzr, nm_, nzm)
    rows.append([lab, nr, f(100 * len(nzr) / nr if nr else float("nan")), f(np.mean(np.abs(nzr)) if len(nzr) else 0.0, "{:.2f}"),
                 f(np.max(np.abs(nzr)) if len(nzr) else 0.0, "{:.2f}"), f(100 * len(nzm) / nm_ if nm_ else float("nan")),
                 f(np.mean(np.abs(nzm)) if len(nzm) else 0.0, "{:.2f}"), meaning])
table("r_duals", ["constraint", "RL1 rows", "RL1 non-|zero (\\%)", "RL1 mean|(Rs/min)", "RL1 max|(Rs/min)", "MILP non-|zero (\\%)",
                  "MILP mean|(Rs/min)", "meaning"], rows, "@{}lrrrrrrp{3.0cm}@{}")
num.update(GunBindR=f(100 * len(st["gun"][1]) / st["gun"][0]), GunMeanR=f(np.mean(st["gun"][1]), "{:.1f}"),
           GunBindM=f(100 * len(st["gun"][3]) / st["gun"][2]), GunMeanM=f(np.mean(st["gun"][3]), "{:.1f}"),
           VanBindR=f(100 * len(st["van"][1]) / st["van"][0]), VanMeanR=f(np.mean(st["van"][1]), "{:.1f}"),
           PatNZR=len(st["patience"][1]), DeadNZR=len(st["deadline"][1]))
RLE = {(e["instance"], e["n"], e["EV"]): e for e in EV["RL"]}
for r in DCR:
    if r["kind"] == "gun":
        e_ = RLE.get((r["instance"], int(r["n"]), int(r["j"])))
        r["lvl_j"] = e_["lvl"] if e_ else "?"
        r["bi"] = next(b for b, (lo, hi) in enumerate(BANDS) if lo <= fnum(r["theta_i"]) < hi)
        r["bj"] = next(b for b, (lo, hi) in enumerate(BANDS) if lo <= fnum(r["theta_j"]) < hi)
gunR = [r for r in DCR if r["kind"] == "gun"]
rows = []
for L, nm in (("low", "low"), ("mid", "medium"), ("high", "high")):
    g_ = [r for r in gunR if r["lvl_j"] == L]
    nz = [r["dual"] for r in g_ if r["dual"] > 1e-9]
    rows.append([nm, len(g_), f(100 * len(nz) / len(g_) if g_ else float("nan")), f(mean(nz), "{:.1f}"), f(sum(nz) / len(g_) if g_ else float("nan"), "{:.2f}")])
table("r_gun_pressure", ["demand pressure|of the follower", "gun links", "binding|(\\%)", "mean dual if|binding (Rs/min)", "mean dual|all links (Rs/min)"],
      rows, "@{}lrrrr@{}")
num.update(GunBindHighR=rows[2][2], GunBindLowR=rows[0][2])
occ = [100 * mean([r["dual"] > 1e-9 for r in gunR if r["bi"] == b]) for b in range(5)]
fol = [100 * mean([r["dual"] > 1e-9 for r in gunR if r["bj"] == b]) for b in range(5)]
num.update(GunOccBotR=f(occ[0]), GunOccTopR=f(occ[4]), GunFolBotR=f(fol[0]), GunFolTopR=f(fol[4]))
H = np.full((5, 5), np.nan)
for a in range(5):
    for b in range(5):
        g_ = [r["dual"] for r in gunR if r["bi"] == a and r["bj"] == b]
        if len(g_) >= 20:
            H[a, b] = 100 * mean([v > 1e-9 for v in g_])
fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.5))
axes[0].hist([[v for v in st["gun"][3]], [v for v in st["gun"][1]]], bins=25, color=[COLM, COLR], label=["MILP", "RL1"])
axes[0].set(xlabel="gun-minute shadow price (Rs/min)", ylabel="binding gun links")
axes[0].legend(fontsize=7)
axes[1].bar(x - 0.2, occ, 0.4, color=COLM, label="EV on the gun ($i$)")
axes[1].bar(x + 0.2, fol, 0.4, color="#DD8452", label="EV behind it ($j$)")
axes[1].set_xticks(x)
axes[1].set_xticklabels(PLAB, fontsize=6, rotation=30)
axes[1].set(xlabel=r"income band $\theta$ (RL1)", ylabel="gun links binding (%)")
axes[1].legend(fontsize=6.5)
axes[2].imshow(H, cmap="Reds", origin="lower")
axes[2].set_xticks(x)
axes[2].set_yticks(x)
axes[2].set_xticklabels(PLAB, fontsize=5.5, rotation=30)
axes[2].set_yticklabels(PLAB, fontsize=5.5)
axes[2].set(xlabel=r"$\theta$ of the EV behind ($j$)", ylabel=r"$\theta$ of the EV on the gun ($i$)")
for a in range(5):
    for b in range(5):
        if H[a, b] == H[a, b]:
            axes[2].text(b, a, f"{H[a, b]:.0f}", ha="center", va="center", fontsize=6)
save(fig, "r_gun_duals")
dpf_r = {(r["instance"], int(r["n"])): -fnum(r["dZ_dPF"]) for r in DIR}
dpf_m = {(r["instance"], int(r["n"])): -fnum(r["dZ_dPF"]) for r in DIM}
dpm_r = {(r["instance"], int(r["n"])): -fnum(r["dZ_dPM"]) for r in DIR}
cong_r = sum(-fnum(r["dZ_dPF"]) + fnum(r["dZ_dPF_own"]) for r in DIR) / sum(-fnum(r["dZ_dPF"]) for r in DIR)
num.update(DPFmeanR=f(mean(list(dpf_r.values())), "{:.0f}"), DPFmeanM=f(mean(list(dpf_m.values())), "{:.0f}"),
           DPFpairR=f(mean([dpf_r[k] for k in BOTH if k in dpf_r]), "{:.0f}"), DPFpairM=f(mean([dpf_m[k] for k in BOTH if k in dpf_m]), "{:.0f}"),
           DPMmeanR=f(mean(list(dpm_r.values())), "{:.0f}"), DPFcongR=f(100 * cong_r, "{:.1f}"),
           DTnonzeroR=sum(1 for r in DIR if abs(fnum(r["dZ_dT"])) > 1e-9))
fig, ax = plt.subplots(figsize=(4.9, 2.6))
ax.plot(NS, [mean([dpf_m[k] for k in BOTH if k[1] == n and k in dpf_m]) for n in NS], "s--", color=COLM, label="MILP")
ax.plot(NS, [mean([dpf_r[k] for k in BOTH if k[1] == n and k in dpf_r]) for n in NS], "o-", color=COLR, label="RL1")
ax.set(xlabel="EVs per instance $n$", ylabel="Rs saved per instance", title="one more kW of FCS gun power (same instances)")
ax.legend(fontsize=7.5)
save(fig, "r_power")

# ====================================================================== 12. invalid plans
inv = [k for k in KEYS if RUNS["RL"][k]["valid"] != "yes"]
vt = Counter()
for k in inv:
    for part in str(RUNS["RL"][k]["violation_detail"]).split(";"):
        part = part.strip()
        if not part:
            continue
        kind_ = ("van back after T" if "van back after T" in part else "EV finishes after T" if "after T" in part else
                 "queue above W" if "queues" in part else "battery" if "battery" in part else "Q exceeded" if "Q exceeded" in part else part[:40])
        vt[kind_] += 1
rows = [[k_, v_] for k_, v_ in vt.most_common()]
table("r_violations", ["broken limit (first three listed per plan)", "count"], rows, "@{}lr@{}")
rows = []
for fam in ("R1", "RC1"):
    ks = [k for k in inv if RUNS["RL"][k]["class"] == fam]
    rows.append([fam, len(ks), sum(RUNS["MILP"][k]["status"] == "infeasible" for k in ks),
                 sum(RUNS["MILP"][k]["status"] == "time_limit_no_plan" for k in ks), f(mean([fnum(RUNS["RL"][k]["violations"]) for k in ks]), "{:.1f}"),
                 min(k[1] for k in ks) if ks else "--"])
table("r_invalid", ["family", "RL1 invalid", "MILP: infeasible", "MILP: no plan", "broken limits|per plan", "smallest $n$"], rows, "@{}lrrrrr@{}")
num.update(InvR=len(inv), InvViolMean=f(mean([fnum(RUNS["RL"][k]["violations"]) for k in inv]), "{:.1f}"),
           InvViolMax=max(int(fnum(RUNS["RL"][k]["violations"])) for k in inv) if inv else 0)

# ====================================================================== 13. income distribution reweighting
dec_r = [k_ / n_ if n_ else 0.0 for n_, k_ in dec["RL"]]
dec_m = [k_ / n_ if n_ else 0.0 for n_, k_ in dec["MILP"]]


def bmass(a, b_, lo, hi, kk=2000):
    xs = np.linspace(lo, hi, kk)
    xm = (xs[1:] + xs[:-1]) / 2
    return float(np.sum(xm ** (a - 1) * (1 - xm) ** (b_ - 1)) * (xs[1] - xs[0]))


rows = []
for (a, b_), lab in (((1, 1), "Beta(1,1) uniform"), ((2, 2), "Beta(2,2) (used)"), ((2, 5), "Beta(2,5) mostly low income"),
                     ((5, 2), "Beta(5,2) mostly high income"), ((0.5, 0.5), "Beta(0.5,0.5) polarised")):
    t_ = bmass(a, b_, 0, 1)
    w_ = [bmass(a, b_, d / 10, (d + 1) / 10) / t_ for d in range(10)]
    rows.append([lab, f(a / (a + b_), "{:.2f}"), f(100 * sum(wi * pi for wi, pi in zip(w_, dec_r))),
                 f(100 * sum(wi * pi for wi, pi in zip(w_, dec_m)))])
table("r_beta", ["income distribution", "mean $\\theta$", "RL1 expected|van share (\\%)", "MILP expected|van share (\\%)"], rows, "@{}lrrr@{}")

# MILP counterparts used in the text
gmh = [e for e in FREE["MILP"] if e["lvl"] == "high"]
gml = [e for e in FREE["MILP"] if e["lvl"] == "low"]
num.update(QMHigh=f(mean([e["queue_min"] for e in gmh if not e["van"]]), "{:.2f}"),
           LowQMHigh=f(mean([e["queue_min"] for e in gmh if not e["van"] and e["band"] <= 1]), "{:.2f}"),
           TopMLow=f(share([e for e in gml if e["band"] == 4])), TopMHigh=f(share([e for e in gmh if e["band"] == 4])),
           VansFullM=f(100 * mean([v["stops"] == 6 for v in VM])), TightSeatsM=f(100 * tcm["seats ($Q_m=6$)"] / len(VM)),
           TightTimeM=f(100 * tcm["time ($T$)"] / len(VM)), DriveBotM=f(QM[0]["drive"], "{:.2f}"), DriveTopM=f(QM[4]["drive"], "{:.2f}"),
           ZperMeanM=f(mean([fnum(RUNS["MILP"][k]["Z_total"]) / k[1] for k in VALID["MILP"]]), "{:.0f}"),
           TopShortM=f(share([e for e in FREE["MILP"] if not e["h2"] and e["band"] == 4])),
           TopLongM=f(share([e for e in FREE["MILP"] if e["h2"] and e["band"] == 4])),
           NQueuedM=sum(1 for e in FREE["MILP"] if not e["van"] and e["queue_min"] > 1e-6))

# income parameter table (same values as the MILP run; copied so this folder compiles on its own)
import shutil  # noqa: E402
src = os.path.join(MILPDIR, "report", "tables", "params.tex")
if os.path.exists(src):
    shutil.copy(src, os.path.join(TAB, "r_params.tex"))

# ====================================================================== numbers
with open(os.path.join(TAB, "numbers_r.tex"), "w") as fh:
    fh.write("% generated by make_rl_report.py\n")
    for k_, v_ in num.items():
        fh.write(f"\\newcommand{{\\{k_}}}{{{v_}}}\n")
for k_, v_ in num.items():
    print(f"{k_:16s} {v_}")
