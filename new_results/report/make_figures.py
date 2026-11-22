"""Figures, tables and headline numbers for report.tex (formulation 3 insights).

Reads the finished run in the parent folder (../results.csv, ../analysis/ev_level.csv,
../run_settings.json) and writes figures/*.pdf, tables/*.tex and tables/numbers.tex here.
Usage (from this folder):  python make_figures.py      then  pdflatex report.tex  (twice)

Definitions used throughout
  free-choice EV   an EV with at least one station in range (stranded EVs must take a van)
  income band      theta in [0,0.2), [0.2,0.4), [0.4,0.6), [0.6,0.8), [0.8,1]
  demand pressure  requests released in [r_i - mean FCS charging time, r_i] divided by the number
                   of chargers (24 guns + 5 vans); split into terciles over all free-choice EVs
  optimal          Gurobi proved the plan optimal; time-limit plans are valid but not proven optimal
"""
import csv
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "..")
FIG = os.path.join(HERE, "figures")
TAB = os.path.join(HERE, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)
plt.rcParams.update({"font.size": 9, "font.family": "serif", "mathtext.fontset": "cm",
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
BANDS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)]
BLAB = ["0.0--0.2", "0.2--0.4", "0.4--0.6", "0.6--0.8", "0.8--1.0"]
PLAB = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
CHARGERS = 24 + 5
COL = {"low": "#4C72B0", "mid": "#DD8452", "high": "#C44E52"}


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# ------------------------------------------------------------------ data
results = list(csv.DictReader(open(os.path.join(RUN, "results.csv"))))
ev = list(csv.DictReader(open(os.path.join(RUN, "analysis", "ev_level.csv"))))
settings = json.load(open(os.path.join(RUN, "run_settings.json")))
for e in ev:
    for k in ("theta", "queue_min", "van_wait_min", "load_at_request", "E_kWh", "pay_Rs", "W_min",
              "perceived_money_Rs", "time_cost_Rs", "customer_cost_Rs", "nearest_reachable_fcs_km",
              "drive_km", "time_to_charged_min"):
        e[k] = fnum(e[k])
    e["n"] = int(e["n"])
    e["opt"] = e["solver_status"].startswith("optimal")
    e["van"] = int(e["van"])
    e["pressure"] = e["load_at_request"] / CHARGERS
    e["band"] = next(k for k, (lo, hi) in enumerate(BANDS) if lo <= e["theta"] < hi)
    e["h2"] = e["horizon"].startswith("type 2")
free = [e for e in ev if e["stranded"] == "no"]
T1, T2 = np.quantile([e["pressure"] for e in free], [1 / 3, 2 / 3])
for e in free:
    e["lvl"] = "low" if e["pressure"] <= T1 else ("mid" if e["pressure"] <= T2 else "high")


def share(g):
    return 100 * float(np.mean([e["van"] for e in g])) if g else float("nan")


def mean(vals):
    vals = [v for v in vals if v == v]
    return float(np.mean(vals)) if vals else float("nan")


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def f(v, fmt="{:.1f}"):
    return "--" if v != v else fmt.format(v)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)


def table(name, head, rows, spec):
    """LaTeX table; a '|' in a header cell breaks it into two lines."""
    def cell(k, h):
        if "|" not in h:
            return h
        return f"\\shortstack[{'l' if k == 0 else 'r'}]{{" + "\\\\".join(h.split("|")) + "}"
    head = [cell(k, h) for k, h in enumerate(head)]
    out = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(head) + " \\\\", "\\midrule"]
    for r in rows:
        out.append("\\midrule" if r == "MID" else " & ".join(str(c) for c in r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, name + ".tex"), "w").write("\n".join(out) + "\n")


num = {}

# ------------------------------------------------------------------ 1. solver outcome
by_n = {}
for r in results:
    n = int(r["EVs"])
    s = r["status"]
    d = by_n.setdefault(n, {"opt": 0, "tl": 0, "none": 0, "inf": 0, "gap": [], "rt": []})
    if s.startswith("optimal"):
        d["opt"] += 1
    elif s.startswith("time limit") and "no solution" in s:
        d["none"] += 1
    elif s.startswith("time limit"):
        d["tl"] += 1
        d["gap"].append(fnum(r["gap_%"]))
    elif s.startswith("infeasible"):
        d["inf"] += 1
    if r.get("runtime_s"):
        d["rt"].append(fnum(r["runtime_s"]))
rows = []
for n in sorted(by_n):
    d = by_n[n]
    rows.append([n, d["opt"] + d["tl"] + d["none"] + d["inf"], d["opt"], d["tl"], d["none"], d["inf"],
                 f(float(np.median(d["gap"])) if d["gap"] else float("nan"), "{:.2f}"), f(mean(d["rt"]), "{:.0f}")])
tot = {k: sum(by_n[n][k] for n in by_n) for k in ("opt", "tl", "none", "inf")}
allgap = [g for n in by_n for g in by_n[n]["gap"]]
rows.append("MID")
rows.append(["all", sum(tot.values()), tot["opt"], tot["tl"], tot["none"], tot["inf"],
             f(float(np.median(allgap)), "{:.2f}"), f(mean([x for n in by_n for x in by_n[n]["rt"]]), "{:.0f}")])
table("solver", ["$n$", "instances", "optimal", "time limit|with plan", "time limit|no plan", "infeasible",
                 "median|gap (\\%)", "mean|time (s)"], rows, "@{}rrrrrrrr@{}")
num.update(NInst=len(results), NOpt=tot["opt"], NTL=tot["tl"], NNone=tot["none"], NInf=tot["inf"],
           NValid=sum(1 for r in results if r.get("valid") == "yes"),
           GapMed=f(float(np.median(allgap)), "{:.1f}"), GapMax=f(max(allgap), "{:.1f}"),
           NEV=len(ev), NFree=len(free), NStranded=len(ev) - len(free),
           TimeLimit=f(settings.get("time_limit_s", float("nan")), "{:.0f}"),
           Workers=settings.get("workers", "--"), Threads=settings.get("threads_per_instance", "--"))
old = {}
try:
    old = {r["instance"]: r["status"] for r in csv.DictReader(open(os.path.join(RUN, "..", "results.csv")))}
except OSError:
    pass
new_inf = {r["instance"] for r in results if r["status"].startswith("infeasible")}
old_inf = {k for k, v in old.items() if v.startswith("infeasible")}
num["InfSame"] = "the same" if old and new_inf == old_inf else "not the same"

fig, ax = plt.subplots(figsize=(4.8, 2.6))
ns = sorted(by_n)
bottom = np.zeros(len(ns))
for key, lab, col in (("opt", "optimal", "#55A868"), ("tl", "time limit, plan", "#DD8452"),
                      ("none", "time limit, no plan", "#8172B2"), ("inf", "infeasible", "#C44E52")):
    vals = np.array([by_n[n][key] for n in ns])
    ax.bar([str(n) for n in ns], vals, bottom=bottom, color=col, label=lab, width=0.7)
    bottom += vals
ax.set(xlabel="EVs per instance $n$", ylabel="instances (of 56)")
ax.legend(fontsize=7, ncol=2, loc="upper right", bbox_to_anchor=(1, 1.25))
save(fig, "solver_outcome")

# ------------------------------------------------------------------ 2. income -> van, deciles
rows, mids, s_all, s_opt, lo_ci, hi_ci = [], [], [], [], [], []
for b in range(10):
    g = [e for e in free if min(int(e["theta"] * 10), 9) == b]
    go = [e for e in g if e["opt"]]
    k = sum(e["van"] for e in g)
    lo, hi = wilson(k, len(g))
    mids.append(b / 10 + 0.05)
    s_all.append(share(g))
    s_opt.append(share(go))
    lo_ci.append(lo)
    hi_ci.append(hi)
    rows.append([f"{b / 10:.1f}--{(b + 1) / 10:.1f}", len(g), f(share(g)), f"[{f(lo)}, {f(hi)}]", len(go), f(share(go))])
table("deciles", ["income $\\theta$", "EVs", "by van|(\\%)", "95\\,\\%|interval", "EVs|(optimal plans)",
                  "by van (\\%)|(optimal plans)"], rows, "@{}lrrrrr@{}")
fig, ax = plt.subplots(figsize=(4.8, 2.9))
ax.fill_between(mids, lo_ci, hi_ci, color="#4C72B0", alpha=0.18, lw=0)
ax.plot(mids, s_all, "o-", color="#4C72B0", ms=4, lw=1.3, label="all valid plans (95% interval shaded)")
ax.plot(mids, s_opt, "s--", color="#333333", ms=3.5, lw=1.1, label="proven-optimal plans only")
ax.set(xlabel=r"income $\theta$ (decile midpoint)", ylabel="free-choice EVs served by a van (%)",
       xlim=(0, 1), xticks=np.linspace(0, 1, 6))
ax.set_ylim(bottom=0)
ax.legend(fontsize=7.5, loc="upper left")
save(fig, "van_by_income")
num.update(VanAll=f(share(free)), VanLowOpt=f(share([e for e in free if e["opt"] and e["theta"] < 0.4])),
           VanSixSeven=f(s_opt[6]), VanSevenEight=f(s_opt[7]), VanEightNine=f(s_opt[8]), VanNineTen=f(s_opt[9]),
           VanTopAll=f(share([e for e in free if e["band"] == 4])),
           VanBotAll=f(share([e for e in free if e["band"] == 0])),
           LowVanOpt=sum(e["van"] for e in free if e["opt"] and e["theta"] < 0.4),
           LowVanTL=sum(e["van"] for e in free if not e["opt"] and e["theta"] < 0.4),
           ThetaVan=f(mean([e["theta"] for e in free if e["van"]]), "{:.2f}"),
           ThetaFCS=f(mean([e["theta"] for e in free if not e["van"]]), "{:.2f}"))

# ------------------------------------------------------------------ 3. demand pressure
rows = []
for L, name in (("low", "low"), ("mid", "medium"), ("high", "high")):
    g = [e for e in free if e["lvl"] == L]
    fc = [e for e in g if not e["van"]]
    rows.append([name, len(g), f(100 * mean([e["queue_min"] > 1e-6 for e in fc])), f(mean([e["queue_min"] for e in fc]), "{:.2f}"),
                 f(share(g))] + [f(share([e for e in g if e["band"] == b])) for b in range(5)])
table("pressure", ["demand|pressure", "EVs", "queueing|(\\%)", "mean queue|(min)", "by van|(\\%)"]
      + [f"by van|$\\theta$ {b}" for b in BLAB], rows, "@{}lrrrrrrrrr@{}")
num.update(PressLow=f(T1, "{:.2f}"), PressHigh=f(T2, "{:.2f}"))
for L in ("low", "mid", "high"):
    g = [e for e in free if e["lvl"] == L]
    fc = [e for e in g if not e["van"]]
    tag = {"low": "Low", "mid": "Mid", "high": "High"}[L]
    num[f"Q{tag}"] = f(mean([e["queue_min"] for e in fc]), "{:.2f}")
    num[f"Queuing{tag}"] = f(100 * mean([e["queue_min"] > 1e-6 for e in fc]))
    num[f"Top{tag}"] = f(share([e for e in g if e["band"] == 4]))
    num[f"LowQ{tag}"] = f(mean([e["queue_min"] for e in fc if e["band"] <= 1]), "{:.2f}")
    num[f"TopQ{tag}"] = f(mean([e["queue_min"] for e in fc if e["band"] == 4]), "{:.2f}")
fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.7))
x = np.arange(5)
for L, name in (("low", "low pressure"), ("mid", "medium"), ("high", "high pressure")):
    g = [e for e in free if e["lvl"] == L]
    axes[0].plot(x, [share([e for e in g if e["band"] == b]) for b in range(5)], "o-", color=COL[L], ms=4, lw=1.3, label=name)
    axes[1].plot(x, [mean([e["queue_min"] for e in g if e["band"] == b and not e["van"]]) for b in range(5)],
                 "o-", color=COL[L], ms=4, lw=1.3, label=name)
for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels(PLAB, fontsize=7.5)
    ax.set_xlabel(r"income band $\theta$")
    ax.set_ylim(bottom=0)
axes[0].set_ylabel("served by a van (%)")
axes[1].set_ylabel("mean FCS queue (min)")
axes[0].legend(fontsize=7.5, loc="upper left")
save(fig, "pressure")

# ------------------------------------------------------------------ 4. who queues, who drives
fcs = [e for e in free if not e["van"]]
qtot = sum(e["queue_min"] for e in fcs)
rows, users, minutes = [], [], []
for b in range(5):
    g = [e for e in fcs if e["band"] == b]
    users.append(100 * len(g) / len(fcs))
    minutes.append(100 * sum(e["queue_min"] for e in g) / qtot if qtot else float("nan"))
    rows.append([BLAB[b], len(g), f(users[-1]), f(minutes[-1]), f(100 * mean([e["queue_min"] > 1e-6 for e in g])),
                 f(mean([e["queue_min"] for e in g]), "{:.2f}"), f(max(e["queue_min"] for e in g)),
                 f(max(e["W_min"] for e in g)), f(mean([e["drive_km"] for e in g]), "{:.2f}")])
table("queues", ["income $\\theta$", "FCS|users", "share of|users (\\%)", "share of queue|minutes (\\%)",
                 "queueing|(\\%)", "mean queue|(min)", "longest|queue (min)", "largest|$W_i$ (min)",
                 "drive to|FCS (km)"], rows, "@{}lrrrrrrrr@{}")
num.update(QShareBottom=f(minutes[0] + minutes[1]), UShareBottom=f(users[0] + users[1]),
           QShareTop=f(minutes[4]), UShareTop=f(users[4]),
           DriveBot=f(mean([e["drive_km"] for e in fcs if e["band"] == 0]), "{:.2f}"),
           DriveTop=f(mean([e["drive_km"] for e in fcs if e["band"] == 4]), "{:.2f}"))
fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.6))
w = 0.38
axes[0].bar(x - w / 2, users, w, color="#9DB4D3", label="share of FCS users")
axes[0].bar(x + w / 2, minutes, w, color="#C44E52", label="share of all queue minutes")
axes[0].set(ylabel="%", xlabel=r"income band $\theta$")
axes[0].set_ylim(0, 1.3 * max(max(users), max(minutes)))
axes[0].legend(fontsize=7.5, loc="upper right")
axes[1].bar(x, [mean([e["drive_km"] for e in fcs if e["band"] == b]) for b in range(5)], 0.6, color="#4C72B0")
axes[1].set(ylabel="mean drive to the FCS (km)", xlabel=r"income band $\theta$")
for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels(PLAB, fontsize=7.5)
save(fig, "queues_drive")

# ------------------------------------------------------------------ 5. horizon and layout
rows = []
for fam in ("C1", "C2", "R1", "R2", "RC1", "RC2"):
    g = [e for e in free if e["family"] == fam]
    rows.append([fam, "short" if fam.endswith("1") else "long", len(g), f(mean([e["pressure"] for e in g]), "{:.2f}"),
                 f(share(g))] + [f(share([e for e in g if e["band"] == b])) for b in range(5)])
table("families", ["family", "horizon", "EVs", "mean|pressure", "by van|(\\%)"]
      + [f"by van|$\\theta$ {b}" for b in BLAB], rows, "@{}llrrrrrrrr@{}")
fig, ax = plt.subplots(figsize=(4.8, 2.7))
for h2, lab, col, mk in ((False, "short horizon (type 1: C1, R1, RC1)", "#C44E52", "o"),
                         (True, "long horizon (type 2: C2, R2, RC2)", "#4C72B0", "s")):
    g = [e for e in free if e["h2"] == h2]
    ax.plot(x, [share([e for e in g if e["band"] == b]) for b in range(5)], marker=mk, color=col, ms=4, lw=1.3, label=lab)
ax.set_xticks(x)
ax.set_xticklabels(PLAB, fontsize=7.5)
ax.set(xlabel=r"income band $\theta$", ylabel="served by a van (%)")
ax.set_ylim(bottom=0)
ax.legend(fontsize=7.5, loc="upper left")
save(fig, "horizon")
for h2, tag in ((False, "Short"), (True, "Long")):
    g = [e for e in free if e["h2"] == h2]
    num[f"Top{tag}"] = f(share([e for e in g if e["band"] == 4]))
    num[f"Press{tag}"] = f(mean([e["pressure"] for e in g]), "{:.2f}")

# ------------------------------------------------------------------ 6. what customers pay and feel
rows = []
for b in range(5):
    g = [e for e in free if e["band"] == b]
    rows.append([BLAB[b], f(mean([e["pay_Rs"] for e in g]), "{:.0f}"), f(mean([e["perceived_money_Rs"] for e in g]), "{:.0f}"),
                 f(mean([e["time_cost_Rs"] for e in g]), "{:.0f}"), f(mean([e["customer_cost_Rs"] for e in g]), "{:.0f}"),
                 f(mean([e["time_to_charged_min"] for e in g]), "{:.1f}")])
table("costs", ["income $\\theta$", "paid|(Rs)", "perceived|money (Rs)", "perceived|time (Rs)",
                "perceived|total (Rs)", "request to|charged (min)"], rows, "@{}lrrrrr@{}")
fig, ax = plt.subplots(figsize=(4.8, 2.7))
money = [mean([e["perceived_money_Rs"] for e in free if e["band"] == b]) for b in range(5)]
tcost = [mean([e["time_cost_Rs"] for e in free if e["band"] == b]) for b in range(5)]
paid = [mean([e["pay_Rs"] for e in free if e["band"] == b]) for b in range(5)]
ax.bar(x, money, 0.6, color="#4C72B0", label=r"perceived money ($\mu_i\times$ paid)")
ax.bar(x, tcost, 0.6, bottom=money, color="#DD8452", label="perceived time")
ax.plot(x, paid, "k_", ms=20, mew=1.6, label="money actually paid")
ax.set_xticks(x)
ax.set_xticklabels(PLAB, fontsize=7.5)
ax.set(xlabel=r"income band $\theta$", ylabel="Rs per EV")
ax.set_ylim(0, 1.45 * max(m + t for m, t in zip(money, tcost)))
ax.legend(fontsize=7, loc="upper left", ncol=2)
save(fig, "costs")
num.update(PaidBot=f(paid[0], "{:.0f}"), PaidTop=f(paid[4], "{:.0f}"),
           FeltBot=f(money[0] + tcost[0], "{:.0f}"), FeltTop=f(money[4] + tcost[4], "{:.0f}"),
           MoneyBot=f(money[0], "{:.0f}"), MoneyTop=f(money[4], "{:.0f}"),
           TimeBot=f(tcost[0], "{:.0f}"), TimeTop=f(tcost[4], "{:.0f}"))
valid = [r for r in results if r.get("valid") == "yes"]
Z = sum(fnum(r["total_Z"]) for r in valid)
for k, tag in (("Z1_EV_drive_perceived", "ZOne"), ("Z2_van_drive", "ZTwo"), ("Z3_dispatch", "ZThree"),
               ("Z4_operator_energy", "ZFour"), ("Z5_tariff_perceived", "ZFive"), ("Z6_time_perceived", "ZSix")):
    num[tag] = f(100 * sum(fnum(r[k]) for r in valid) / Z)
num["ZOperator"] = f(100 * sum(fnum(r["operator_cost_Z2_to_Z4"]) for r in valid) / Z)
num["ZCustomer"] = f(100 * sum(fnum(r["customer_cost_Z1_Z5_Z6"]) for r in valid) / Z)
num["VanWaitBot"] = f(mean([e["van_wait_min"] for e in free if e["van"] and e["band"] <= 1]), "{:.0f}")
num["VanWaitTop"] = f(mean([e["van_wait_min"] for e in free if e["van"] and e["band"] == 4]), "{:.0f}")

# ------------------------------------------------------------------ 7. logistic regression
names = ["intercept", "income $\\theta$", "medium pressure", "high pressure", "$\\theta\\times$ high pressure",
         "long horizon (type 2)", "energy $E_i$ (per 10 kWh)", "nearest reachable FCS (km)"]
X = np.array([[1.0, e["theta"], e["lvl"] == "mid", e["lvl"] == "high", e["theta"] * (e["lvl"] == "high"),
               e["h2"], e["E_kWh"] / 10, e["nearest_reachable_fcs_km"]] for e in free], dtype=float)
y = np.array([e["van"] for e in free], dtype=float)
lam = 1e-3 * np.eye(X.shape[1])
lam[0, 0] = 0
b = np.zeros(X.shape[1])
for _ in range(100):
    p = 1 / (1 + np.exp(-np.clip(X @ b, -35, 35)))
    step = np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]) + lam, X.T @ (y - p) - lam @ b)
    b += step
    if np.max(np.abs(step)) < 1e-9:
        break
p = 1 / (1 + np.exp(-np.clip(X @ b, -35, 35)))
se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]) + lam)))
rows = []
for k, nm in enumerate(names):
    z = b[k] / se[k]
    pv = math.erfc(abs(z) / math.sqrt(2))
    rows.append([nm, f(b[k], "{:.3f}"), f(se[k], "{:.3f}"), f(z, "{:.1f}"), "$<0.001$" if pv < 0.001 else f(pv, "{:.3f}")])
table("logit", ["term", "coefficient", "std.\\ error", "$z$", "$p$-value"], rows, "@{}lrrrr@{}")
hi_ = X[:, 3] == 1
ame = (b[1] + b[4] * X[:, 3]) * p * (1 - p)
num.update(AME=f(100 * 0.1 * float(np.mean(ame)), "{:.1f}"),
           AMEHigh=f(100 * 0.1 * float(np.mean(ame[hi_])), "{:.1f}"),
           AMENotHigh=f(100 * 0.1 * float(np.mean(ame[~hi_])), "{:.1f}"),
           LogitAcc=f(100 * float(np.mean((p >= 0.5) == (y == 1)))), LogitN=len(free))

# ------------------------------------------------------------------ parameters of the run
P = settings.get("f3_params", {})
vot = P.get("value_of_time_Rs_per_min", {})
rows = [["income $\\theta_i$", "--", f"Beta({P['theta_distribution']['a']:g}, {P['theta_distribution']['b']:g})", ""],
        ["money sensitivity $\\mu$", "Rs/Rs", f"{P['money_sensitivity_mu']['theta0']:g}", f"{P['money_sensitivity_mu']['theta1']:g}"]]
for a, lab in (("D", "driving to an FCS"), ("Q", "queueing at an FCS"), ("C", "charging at an FCS"),
               ("V", "van wait + van charging")):
    rows.append([f"value of time $c^{{\\mathrm{{{a}}}}}$ ({lab})", "Rs/min", f"{vot[a]['theta0']:g}", f"{vot[a]['theta1']:g}"])
rows.append(["queue tolerance $W$", "min", f"{P['queue_tolerance_W_min']['theta0']:g}", f"{P['queue_tolerance_W_min']['theta1']:g}"])
rows.append(["tariff $p^{\\mathrm F}$ / $p^{\\mathrm M}$", "Rs/kWh", f"{P['tariff_Rs_per_kWh']['p_F']:g}", f"{P['tariff_Rs_per_kWh']['p_M']:g}"])
table("params", ["parameter", "unit", "at $\\theta=0$", "at $\\theta=1$"], rows, "@{}llrr@{}")

with open(os.path.join(TAB, "numbers.tex"), "w") as fh:
    fh.write("% generated by make_figures.py\n")
    for k, v in num.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
for k, v in num.items():
    print(f"{k:14s} {v}")
