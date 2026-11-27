"""Tables, figures and numbers of the formulation-4 comprehensive report (comprehensive_report_f4.tex).

Reads only (nothing outside report/ is written):
  ../data/solomon_data.csv, ../data/solomon_config.csv           the dataset (56 Solomon files)
  ../common/f4_params.json, ../common/experiments.json            customer parameters, experiments
  ../baseline_MILP/results/{runs.csv, plan.csv, analysis/milp_300s/*}
  ../baseline_RL/results/{runs.csv, analysis/rl_f4/*}, ../baseline_RL/model/train_log.csv, train_meta.json
  ../<E>_<name>_MILP/results/<variant>/runs.csv, results/comparison/*.csv
  ../<E>_<name>_RL/results/<variant>/runs.csv, results/comparison/*.csv, model/<variant>/train_meta.json
Writes: figures/*.pdf, tables/*.tex, numbers.tex (LaTeX macros), numbers.json (the same values, to read).

Usage (from 2026-11-22_formulation_4/report):  python make_report_f4.py
"""
import csv
import json
import math
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
F4 = os.path.abspath(os.path.join(HERE, ".."))
FIG, TAB = os.path.join(HERE, "figures"), os.path.join(HERE, "tables")
CLASSES = ["C1", "C2", "R1", "R2", "RC1", "RC2"]
SIZES = list(range(10, 101, 10))
# E1,E2,E5,E6,E8 ran MILP + RL (zero-shot + retrained). E9,E10 are RL-only, retrained-only and are
# handled by station_capacity()/charging_demand() so the shared experiment machinery stays unchanged.
FULL_EXPERIMENTS = ("E1", "E2", "E5", "E6", "E8")

# validated reference palette (light): blue, orange, aqua, yellow
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e4e3df", "#ffffff"
SEG_COL = {"personal": BLUE, "commercial": ORANGE}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight", "axes.titlesize": 9, "axes.titlecolor": INK,
})
NUM = {}


def num(key, value, fmt=None):
    """Store a number for numbers.tex (macro \\f<key>) and numbers.json."""
    NUM[key] = value if fmt is None else (fmt.format(value) if value is not None else "--")
    return value


def read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fnum(x, default=None):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def tex(s):
    return (str(s).replace("\\", "/").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")
            .replace("#", "\\#").replace("^", "\\^{}"))


def wrap_header(h):
    """Long header text on two lines (split at the space nearest the middle)."""
    if len(h) <= 12 or " " not in h or "\\multicolumn" in h:
        return h
    spaces = [i for i, ch in enumerate(h) if ch == " " and h.count("$", 0, i) % 2 == 0]
    if not spaces:
        return h
    i = min(spaces, key=lambda j: abs(j - len(h) / 2))
    return "\\shortstack{" + h[:i] + "\\\\" + h[i + 1:] + "}"


def write_table(name, header, rows, colspec, caption=None, size="\\small", note=None):
    if len(header) > 5:
        header = [wrap_header(h) for h in header]
    lines = [size, "\\begin{adjustbox}{max width=\\linewidth}", "\\begin{tabular}{" + colspec + "}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        if r == "MID":
            lines.append("\\midrule")
        else:
            lines.append(" & ".join(str(c) for c in r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}"]
    if note:
        lines.append("\\par\\smallskip{\\footnotesize " + note + "}")
    with open(os.path.join(TAB, name + ".tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def f(x, d=1, plus=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    s = f"{x:+.{d}f}" if plus else f"{x:.{d}f}"
    return s.replace("-", "$-$") if not plus else s.replace("-", "$-$").replace("+", "$+$")


# ====================================================================== the dataset
def dataset():
    cfg = {r["instance"]: r for r in read(os.path.join(F4, "data", "solomon_config.csv"))}
    rows = read(os.path.join(F4, "data", "solomon_data.csv"))
    by_inst = {}
    for r in rows:
        by_inst.setdefault(r["instance"], []).append(r)
    per_class = {c: {"files": 0} for c in CLASSES}
    ev_unique = {}                      # class -> list of EV dicts of the first file (shared within class)
    req_rel = {c: [] for c in CLASSES}
    for name, rs in by_inst.items():
        c = cfg[name]["class"]
        T = float(cfg[name]["T_min"])
        k = float(cfg[name]["detour"]) * float(cfg[name]["km_per_unit"])
        st_xy = np.array([[float(r["x"]), float(r["y"])] for r in rs if r["type"] == "station"])
        dep = [r for r in rs if r["type"] == "depot"][0]
        evs = [r for r in rs if r["type"] == "ev"]
        per_class[c]["files"] += 1
        per_class[c]["T"] = T
        per_class[c]["guns"] = int(cfg[name]["total_guns"])
        per_class[c]["vans"] = int(cfg[name]["n_vans"])
        for r in evs:
            req_rel[c].append(float(r["request_min"]) / T)
        if c not in ev_unique:
            lst = []
            for r in evs:
                xy = np.array([float(r["x"]), float(r["y"])])
                d_near = k * np.linalg.norm(st_xy - xy, axis=1).min()
                d_dep = k * np.linalg.norm(xy - np.array([float(dep["x"]), float(dep["y"])]))
                lst.append({"seg": r["segment"], "theta": float(r["theta"]), "E": float(r["energy_kWh"]),
                            "R": float(r["range_km"]), "stranded": d_near > float(r["range_km"]) + 1e-9,
                            "d_near": d_near, "d_dep": d_dep})
            ev_unique[c] = lst
    trows = []
    for c in CLASSES:
        evs = ev_unique[c]
        com = [e for e in evs if e["seg"] == "commercial"]
        per = [e for e in evs if e["seg"] == "personal"]
        rr = req_rel[c]
        trows.append([c, per_class[c]["files"], f"{per_class[c]['T']:.0f}",
                      f(100 * sum(x == 0 for x in rr) / len(rr), 0),
                      f(100 * st.median(rr), 0),
                      f(st.mean(e["E"] for e in evs), 1),
                      f(100 * sum(e["stranded"] for e in evs) / len(evs), 0),
                      f(st.mean(e["d_near"] for e in evs), 1),
                      f(100 * len(com) / len(evs), 0),
                      f(st.mean(e["theta"] for e in per), 2), f(st.mean(e["theta"] for e in com), 2)])
    write_table("data_classes",
                ["class", "files", "$T$ (min)", "requests at 0 (\\%)", "median $r/T$ (\\%)", "mean $E$ (kWh)",
                 "stranded (\\%)", "km to nearest FCS", "commercial (\\%)", "mean $\\theta$ pers.",
                 "mean $\\theta$ comm."],
                trows, "@{}lrrrrrrrrrr@{}", size="\\footnotesize")
    allev = [e for c in CLASSES for e in ev_unique[c]]
    num("comShareData", 100 * sum(e["seg"] == "commercial" for e in allev) / len(allev), "{:.0f}")
    num("thetaPersData", st.mean(e["theta"] for e in allev if e["seg"] == "personal"), "{:.2f}")
    num("thetaComData", st.mean(e["theta"] for e in allev if e["seg"] == "commercial"), "{:.2f}")
    num("strandedData", 100 * sum(e["stranded"] for e in allev) / len(allev), "{:.0f}")

    # figure: theta per segment (EVs of the 6 classes, 100 each) with the calibrated Beta densities
    P = json.load(open(os.path.join(F4, "common", "f4_params.json"), encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    bins = np.linspace(0, 1, 21)
    xs = np.linspace(0.001, 0.999, 300)
    for seg in ("personal", "commercial"):
        th = [e["theta"] for e in allev if e["seg"] == seg]
        ax.hist(th, bins=bins, density=True, color=SEG_COL[seg], alpha=0.35, edgecolor=SURF, linewidth=1.0)
        a, b = P[seg]["theta_distribution"]["a"], P[seg]["theta_distribution"]["b"]
        pdf = xs ** (a - 1) * (1 - xs) ** (b - 1) / (math.gamma(a) * math.gamma(b) / math.gamma(a + b))
        ax.plot(xs, pdf, color=SEG_COL[seg], lw=1.8, label=f"{seg}: {len(th)} EVs, Beta({a:g}, {b:g})")
    ax.set_xlabel("income index $\\theta$ (0 = poorest, 1 = richest)")
    ax.set_ylabel("density")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.5)
    fig.savefig(os.path.join(FIG, "data_theta.pdf"))
    plt.close(fig)

    # figure: request times relative to T per class
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ax.boxplot([req_rel[c] for c in CLASSES], tick_labels=CLASSES, widths=0.5, showfliers=False,
               medianprops={"color": ORANGE, "lw": 1.6}, boxprops={"color": BLUE},
               whiskerprops={"color": BLUE}, capprops={"color": BLUE})
    ax.set_ylabel("request time / $T$")
    ax.set_ylim(-0.02, 1.0)
    fig.savefig(os.path.join(FIG, "data_requests.pdf"))
    plt.close(fig)

    # figure: sensitivity lines per segment
    th = np.array([0.0, 1.0])
    panels = [("money sensitivity $\\mu$\n(Rs felt per Rs paid)", lambda s: P[s]["money_sensitivity_mu"]),
              ("charging at an FCS\n$c^{C}$ (Rs/min)", lambda s: P[s]["value_of_time_Rs_per_min"]["C"]),
              ("van wait and charging\n$c^{V}$ (Rs/min)", lambda s: P[s]["value_of_time_Rs_per_min"]["V"]),
              ("longest accepted queue\n$W$ (min)", lambda s: P[s]["queue_tolerance_W_min"])]
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.3))
    for ax, (title, get) in zip(axs, panels):
        for seg in ("personal", "commercial"):
            d = get(seg)
            ax.plot(th, [d["theta0"], d["theta1"]], color=SEG_COL[seg], lw=1.8, label=seg,
                    ls="-" if seg == "personal" else (0, (4, 2)))
        ax.set_title(title, fontsize=7.5)
        ax.set_xlabel("$\\theta$")
        ax.set_xticks([0, 0.5, 1])
    axs[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout(w_pad=0.8)
    fig.savefig(os.path.join(FIG, "data_lines.pdf"))
    plt.close(fig)

    # parameter table
    prow = []
    for label, key, sub in [("money sensitivity $\\mu$", "money_sensitivity_mu", None),
                            ("drive to FCS $c^{D}$ (Rs/min)", "value_of_time_Rs_per_min", "D"),
                            ("queue at FCS $c^{Q}$ (Rs/min)", "value_of_time_Rs_per_min", "Q"),
                            ("charge at FCS $c^{C}$ (Rs/min)", "value_of_time_Rs_per_min", "C"),
                            ("wait for / charged by van $c^{V}$ (Rs/min)", "value_of_time_Rs_per_min", "V"),
                            ("longest queue $W$ (min)", "queue_tolerance_W_min", None)]:
        cells = [label]
        for seg in ("personal", "commercial"):
            d = P[seg][key] if sub is None else P[seg][key][sub]
            cells += [f"{d['theta0']:g}", f"{d['theta1']:g}"]
        prow.append(cells)
    write_table("params", ["quantity", "personal $\\theta=0$", "personal $\\theta=1$", "commercial $\\theta=0$",
                           "commercial $\\theta=1$"], prow, "@{}lrrrr@{}")
    num("pF", P["tariff_Rs_per_kWh"]["p_F"], "{:g}")
    num("pM", P["tariff_Rs_per_kWh"]["p_M"], "{:g}")


# ====================================================================== experiments table
def label_tex(s):
    """Plain-text labels of experiments.json -> LaTeX with the symbols in math mode."""
    s = s.replace("%", "\\%")
    for a, b in (("c^{C,1}", "$c^{C,1}$"), ("omega^1", "$\\omega^1$"), ("mu^0", "$\\mu^0$"), ("p^M", "$p^M$"),
                 ("p^F", "$p^F$"), ("theta = 1", "$\\theta=1$"), ("(-25", "($-$25"), ("(+25", "($+$25")):
        s = s.replace(a, b)
    return s


def experiments_table():
    E = json.load(open(os.path.join(F4, "common", "experiments.json"), encoding="utf-8"))["experiments"]
    E = [e for e in E if e["id"] in FULL_EXPERIMENTS]
    rows = []
    for e in E:
        base = label_tex(", ".join(f"{k} = {v:g}" for k, v in e["baseline_values"].items()))
        for j, v in enumerate(e["variants"]):
            rows.append([e["id"] if j == 0 else "", tex(v["id"]).replace("\\_", "\\_\\allowbreak "),
                         label_tex(v["label"]), base if j == 0 else ""])
        rows.append("MID")
    rows = rows[:-1]
    write_table("experiments", ["exp.", "variant id", "what is changed", "baseline value"], rows,
                "@{}l>{\\raggedright\\arraybackslash}p{4.3cm}>{\\raggedright\\arraybackslash}p{5.6cm}"
                ">{\\raggedright\\arraybackslash}p{3.6cm}@{}", size="\\footnotesize")
    return E


# ====================================================================== baselines
def runs_of(path, run_id):
    return [r for r in read(path) if r["run_id"] == run_id]


def key(r):
    return (r["instance"], int(r["n"]))


def baseline_milp():
    rs = runs_of(os.path.join(F4, "baseline_MILP", "results", "runs.csv"), "milp_300s")
    gr = runs_of(os.path.join(F4, "baseline_MILP", "results", "runs.csv"), "greedy")
    stat = {}
    for r in rs:
        stat[r["status"]] = stat.get(r["status"], 0) + 1
    num("milpOptimal", stat.get("optimal", 0))
    num("milpTL", stat.get("time_limit", 0))
    num("milpTLnoPlan", stat.get("time_limit_no_plan", 0))
    num("milpInfeasible", stat.get("infeasible", 0))
    valid = [r for r in rs if r["valid"] == "yes"]
    num("milpValid", len(valid))
    gaps = [fnum(r["gap_%"]) for r in rs if r["status"] == "time_limit" and fnum(r["gap_%"]) is not None]
    num("milpGapMedian", st.median(gaps), "{:.1f}")
    num("milpGapMean", st.mean(gaps), "{:.1f}")
    num("milpSolveMean", st.mean(fnum(r["solve_s"], 0) for r in rs), "{:.0f}")
    num("greedyValid", sum(r["valid"] == "yes" for r in gr))

    # status by n (stacked bars)
    cats = [("optimal", BLUE, "optimal"), ("time_limit", AQUA, "time limit, plan found"),
            ("time_limit_no_plan", YELLOW, "time limit, no plan"), ("infeasible", ORANGE, "infeasible (proven)")]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    bottom = np.zeros(len(SIZES))
    for st_, col, lab in cats:
        h = np.array([sum(1 for r in rs if int(r["n"]) == n and r["status"] == st_) for n in SIZES])
        ax.bar(SIZES, h, bottom=bottom, width=7, color=col, edgecolor=SURF, linewidth=1.0, label=lab)
        bottom += h
    ax.set_xticks(SIZES)
    ax.set_xlabel("number of EVs $n$")
    ax.set_ylabel("instances (56 per $n$)")
    ax.legend(ncol=4, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.grid(axis="x", visible=False)
    fig.savefig(os.path.join(FIG, "milp_status_by_n.pdf"))
    plt.close(fig)

    # by class table
    trows = []
    for c in CLASSES:
        cr = [r for r in rs if r["class"] == c]
        cv = [r for r in cr if r["valid"] == "yes"]
        g = [fnum(r["gap_%"]) for r in cr if r["status"] == "time_limit" and fnum(r["gap_%"]) is not None]
        trows.append([c, len(cr), sum(r["status"] == "optimal" for r in cr), sum(r["status"] == "time_limit" for r in cr),
                      sum(r["status"] == "time_limit_no_plan" for r in cr), sum(r["status"] == "infeasible" for r in cr),
                      len(cv), f(st.median(g), 1) if g else "--", f(st.mean(fnum(r["solve_s"], 0) for r in cr), 0),
                      f(st.mean(fnum(r["Z_total"]) for r in cv), 0)])
    write_table("milp_by_class", ["class", "instances", "optimal", "time limit", "no plan", "infeasible", "valid",
                                  "median gap (\\%)", "mean time (s)", "mean $Z$ (Rs)"], trows, "@{}lrrrrrrrrr@{}")

    # objective anatomy
    terms = [("Z1_EV_drive_perceived", "$Z_1$ EV driving (perceived)"), ("Z2_van_drive", "$Z_2$ van driving"),
             ("Z3_dispatch", "$Z_3$ van dispatch"), ("Z4_operator_energy", "$Z_4$ operator energy"),
             ("Z5_tariff_perceived", "$Z_5$ tariff (perceived)"), ("Z6_time_perceived", "$Z_6$ time (perceived)")]
    tot = sum(fnum(r["Z_total"]) for r in valid)
    arows = []
    for col, lab in terms:
        s = sum(fnum(r[col], 0) for r in valid)
        arows.append([lab, f(s / len(valid), 0), f(100 * s / tot, 1)])
        num("share" + col.split("_")[0], 100 * s / tot, "{:.1f}")
    op = sum(fnum(r["operator_cost_Z2_to_Z4"], 0) for r in valid)
    num("shareOperator", 100 * op / tot, "{:.1f}")
    arows.append("MID")
    arows.append(["$Z$ (all six)", f(tot / len(valid), 0), "100.0"])
    write_table("milp_anatomy", ["term", "mean per instance (Rs)", "share of $Z$ (\\%)"], arows, "@{}lrr@{}")
    num("milpMeanZ", tot / len(valid), "{:,.0f}")
    return {key(r): r for r in rs}, {key(r): r for r in gr}


def baseline_rl(milp):
    rs = runs_of(os.path.join(F4, "baseline_RL", "results", "runs.csv"), "rl_f4")
    rl = {key(r): r for r in rs}
    num("rlValid", sum(r["valid"] == "yes" for r in rs))
    num("rlInvalid", sum(r["valid"] != "yes" for r in rs))
    both = [k for k in rl if rl[k]["valid"] == "yes" and milp.get(k, {}).get("valid") == "yes"]
    gap = {k: 100 * (fnum(rl[k]["Z_total"]) - fnum(milp[k]["Z_total"])) / fnum(milp[k]["Z_total"]) for k in both}
    num("rlPaired", len(both))
    num("rlGapMean", st.mean(gap.values()), "{:.2f}")
    num("rlGapMedian", st.median(gap.values()), "{:.2f}")
    opt = [k for k in both if milp[k]["status"] == "optimal"]
    num("rlGapOpt", st.mean(gap[k] for k in opt), "{:.2f}")
    num("rlOnly", sum(1 for k in rl if rl[k]["valid"] == "yes" and milp.get(k, {}).get("valid") != "yes"))
    num("milpOnly", sum(1 for k in rl if rl[k]["valid"] != "yes" and milp.get(k, {}).get("valid") == "yes"))
    num("rlBetter", sum(1 for k in both if gap[k] < -1e-6))
    num("rlSolveMean", st.mean(fnum(r["solve_s"], 0) for r in rs), "{:.2f}")
    rl_only_status = {}
    for k in rl:
        if rl[k]["valid"] == "yes" and milp.get(k, {}).get("valid") != "yes":
            s = milp[k]["status"]
            rl_only_status[s] = rl_only_status.get(s, 0) + 1
    num("rlOnlyFromNoPlan", rl_only_status.get("time_limit_no_plan", 0))
    num("rlOnlyFromInfeasible", rl_only_status.get("infeasible", 0))

    # by class and by n
    def grp(label, sel):
        ks = [k for k in rl if sel(k)]
        g = [gap[k] for k in ks if k in gap]
        return [label, len(ks), sum(milp[k]["valid"] == "yes" for k in ks), sum(rl[k]["valid"] == "yes" for k in ks),
                f(st.mean(g), 2) if g else "--", f(st.median(g), 2) if g else "--",
                f(st.mean(fnum(milp[k]["solve_s"], 0) for k in ks), 0), f(st.mean(fnum(rl[k]["solve_s"], 0) for k in ks), 2)]
    rows = [grp(c, lambda k, c=c: milp[k]["class"] == c) for c in CLASSES] + ["MID"]
    rows += [grp("all", lambda k: True)]
    write_table("rl_by_class", ["class", "instances", "MILP valid", "RL valid", "gap mean (\\%)", "gap median (\\%)",
                                "MILP s", "RL s"], rows, "@{}lrrrrrrr@{}")
    gn = [st.mean([gap[k] for k in gap if k[1] == n]) for n in SIZES]
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    bars = ax.bar(SIZES, gn, width=6, color=BLUE, edgecolor=SURF)
    for b, m in zip(bars, gn):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.03, f"{m:.2f}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(SIZES)
    ax.set_xlabel("number of EVs $n$")
    ax.set_ylabel("mean gap to MILP (%)")
    ax.set_ylim(0, max(gn) * 1.25)
    ax.grid(axis="x", visible=False)
    fig.savefig(os.path.join(FIG, "rl_gap_by_n.pdf"))
    plt.close(fig)

    # training curve
    log = read(os.path.join(F4, "baseline_RL", "model", "train_log.csv"))
    meta = json.load(open(os.path.join(F4, "baseline_RL", "model", "train_meta.json"), encoding="utf-8"))
    num("rlParams", meta["parameters"], "{:,}")
    num("rlTrainMin", meta["train_s"] / 60, "{:.0f}")
    num("rlBestEpoch", meta["best_epoch"])
    num("rlStopEpoch", meta["epoch"])
    num("rlBestVal", meta["best_val"], "{:.3f}")
    num("rlUntrained", fnum(log[0]["val_ratio_rl_over_greedy"]), "{:.2f}")
    num("rlAfterWarm1", fnum(log[1]["val_ratio_rl_over_greedy"]), "{:.3f}")
    ep = np.array([int(r["epoch"]) for r in log])
    byn = [json.loads(r["val_ratio_by_n"]) for r in log]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    for n, col in (("20", BLUE), ("50", ORANGE), ("100", AQUA)):
        y = np.array([b[n] for b in byn])
        ax.plot(ep[1:], y[1:], color=col, lw=1.5, label=f"n = {n}")
        ax.annotate(f"n = {n}", (ep[-1], y[-1]), xytext=(5, 0), textcoords="offset points", va="center",
                    fontsize=7.5, color=INK2)
    ax.axhline(1.0, color=INK2, lw=0.9, ls="--")
    ax.set_ylim(0.9, 1.25)
    for x, t, yy, ha in ((10, "warm-up ends", 1.245, "left"), (meta["best_epoch"], f"best epoch {meta['best_epoch']}", 1.245, "right"),
                         (meta["epoch"], f"stop {meta['epoch']}", 1.205, "right")):
        ax.axvline(x, color=MUTED, lw=0.8, ls=":")
        ax.text(x + (2 if ha == "left" else -2), yy, t, fontsize=7, color=INK2, va="top", ha=ha)
    ax.set_xlabel("epoch (50 batches each)")
    ax.set_ylabel("validation cost / greedy cost")
    ax.legend(loc="lower left", ncol=3, fontsize=7.5)
    fig.savefig(os.path.join(FIG, "rl_training.pdf"))
    plt.close(fig)
    return rl, gap


def segment_tables():
    a_m = {r["segment"]: r for r in read(os.path.join(F4, "baseline_MILP", "results", "analysis", "milp_300s", "by_segment.csv"))}
    a_r = {r["segment"]: r for r in read(os.path.join(F4, "baseline_RL", "results", "analysis", "rl_f4", "by_segment.csv"))}
    cols = [("EVs", "EVs in valid plans", 0), ("free_choice", "free-choice EVs", 0),
            ("share_van_free_choice_%", "van share of free-choice EVs (\\%)", 1), ("mean_theta", "mean $\\theta$", 2),
            ("mean_queue_FCS_min", "mean queue at FCS (min)", 2), ("mean_van_wait_min", "mean wait for a van (min)", 1),
            ("mean_drive_min", "mean drive to FCS (min)", 1), ("mean_downtime_min", "mean time until charged (min)", 1),
            ("mean_pay_Rs", "mean money paid (Rs)", 0), ("mean_time_cost_Rs", "mean time cost (Rs)", 0),
            ("mean_customer_cost_Rs", "mean perceived cost (Rs)", 0)]
    rows = []
    for c, lab, d in cols:
        rows.append([lab] + [f(fnum(src[s][c]), d) for s in ("personal", "commercial") for src in (a_m, a_r)])
    write_table("segments", ["", "personal MILP", "personal RL", "commercial MILP", "commercial RL"], rows,
                "@{}lrrrr@{}")
    num("vanPersMilp", fnum(a_m["personal"]["share_van_free_choice_%"]), "{:.1f}")
    num("vanComMilp", fnum(a_m["commercial"]["share_van_free_choice_%"]), "{:.1f}")
    num("vanPersRl", fnum(a_r["personal"]["share_van_free_choice_%"]), "{:.1f}")
    num("vanComRl", fnum(a_r["commercial"]["share_van_free_choice_%"]), "{:.1f}")
    num("queueComMilp", fnum(a_m["commercial"]["mean_queue_FCS_min"]), "{:.2f}")
    num("queuePersMilp", fnum(a_m["personal"]["mean_queue_FCS_min"]), "{:.2f}")
    thr = {r["segment"]: r for r in read(os.path.join(F4, "baseline_MILP", "results", "analysis", "milp_300s", "threshold.csv"))}
    num("thetaStarPers", fnum(thr["personal"]["theta_star"]), "{:.2f}")

    # van share by income band: MILP vs RL, two panels
    bi_m = read(os.path.join(F4, "baseline_MILP", "results", "analysis", "milp_300s", "by_segment_income.csv"))
    bi_r = read(os.path.join(F4, "baseline_RL", "results", "analysis", "rl_f4", "by_segment_income.csv"))
    bands = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.5), sharey=True)
    x = np.arange(len(bands))
    for ax, seg in zip(axs, ("personal", "commercial")):
        for j, (src, lab, col) in enumerate(((bi_m, "MILP", BLUE), (bi_r, "RL", ORANGE))):
            v = []
            for b in bands:
                row = [r for r in src if r["segment"] == seg and r["group"] == b]
                v.append(fnum(row[0]["share_van_free_choice_%"], 0) if row else 0)
            bars = ax.bar(x + (j - 0.5) * 0.38, v, width=0.36, color=col, edgecolor=SURF, label=lab)
            for bb, vv in zip(bars, v):
                if vv >= 1:
                    ax.text(bb.get_x() + bb.get_width() / 2, vv + 0.6, f"{vv:.0f}", ha="center", fontsize=6.5, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels(bands, fontsize=7)
        ax.set_title(seg)
        ax.set_xlabel("income band $\\theta$")
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("van share of free-choice EVs (%)")
    axs[0].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "van_share_income.pdf"))
    plt.close(fig)
    # income-band table for MILP
    rows = []
    for seg in ("personal", "commercial"):
        for b in bands:
            rm = [r for r in bi_m if r["segment"] == seg and r["group"] == b][0]
            rr = [r for r in bi_r if r["segment"] == seg and r["group"] == b][0]
            rows.append([seg if b == bands[0] else "", b, rm["EVs"], f(fnum(rm["share_van_free_choice_%"]), 1),
                         f(fnum(rr["share_van_free_choice_%"]), 1), f(fnum(rm["mean_queue_FCS_min"]), 2),
                         f(fnum(rm["mean_perceived_money_Rs"]), 0), f(fnum(rm["mean_time_cost_Rs"]), 0),
                         f(fnum(rm["mean_customer_cost_Rs"]), 0)])
        if seg == "personal":
            rows.append("MID")
    write_table("income_bands", ["segment", "$\\theta$ band", "EVs", "van \\% MILP", "van \\% RL", "queue (min)",
                                 "money felt (Rs)", "time cost (Rs)", "perceived cost (Rs)"], rows,
                "@{}llrrrrrrr@{}", size="\\footnotesize")


# ====================================================================== experiments
def experiments(E):
    out = []
    for e in E:
        mf = os.path.join(F4, e["folder"] + "_MILP", "results", "comparison", "comparison.csv")
        rf = os.path.join(F4, e["folder"] + "_RL", "results", "comparison", "comparison.csv")
        cm = {r["variant"]: r for r in read(mf)}
        cr = {}
        for r in read(rf):
            cr[(r["variant"], r["run_id"])] = r
        for v in e["variants"]:
            m = cm[v["id"]]
            zs, rt = cr[(v["id"], "rl_f4_zero_shot")], cr[(v["id"], "rl_f4_retrained")]
            tm = json.load(open(os.path.join(F4, e["folder"] + "_RL", "model", v["id"], "train_meta.json"), encoding="utf-8"))
            out.append({"E": e["id"], "folder": e["folder"], "id": v["id"], "label": v["label"], "milp": m, "zs": zs,
                        "rt": rt, "train_min": tm["train_s"] / 60, "best_epoch": tm["best_epoch"]})
    return out


def experiment_tables(X):
    def row(x, r, name):
        return [name, r["valid"], r["paired"], r["newly_valid"], r["no_longer_valid"], f(fnum(r["mean_dZ_%"]), 2, True),
                f(fnum(r["mean_d_operator_cost_%"]), 2, True), f(fnum(r["mean_d_customer_cost_%"]), 2, True),
                f"{f(fnum(r['baseline_share_van_personal_%']), 1)} $\\to$ {f(fnum(r['share_van_personal_%']), 1)}",
                f"{f(fnum(r['baseline_share_van_commercial_%']), 1)} $\\to$ {f(fnum(r['share_van_commercial_%']), 1)}",
                f(fnum(r["vans_used"]), 2), f(fnum(r.get("mean_gap_to_milp_%")), 2) if name != "MILP" else "--"]
    header = ["method", "valid", "paired", "new", "lost", "$\\Delta Z$ \\%", "$\\Delta$ oper. \\%", "$\\Delta$ cust. \\%",
              "van \\% personal", "van \\% comm.", "vans used", "gap to MILP \\%"]
    for Eid in sorted({x["E"] for x in X}):
        rows = []
        for x in [x for x in X if x["E"] == Eid]:
            rows.append(["\\multicolumn{12}{@{}l}{\\textbf{" + tex(x["id"]) + "}: " + label_tex(x["label"]) + "}"])
            rows.append(row(x, x["milp"], "MILP"))
            rows.append(row(x, x["zs"], "RL zero-shot"))
            rows.append(row(x, x["rt"], "RL retrained"))
        write_table(f"exp_{Eid}", header, rows, "@{}lrrrrrrrrrrr@{}", size="\\scriptsize")

    # tornado: mean dZ per variant, MILP bars + RL markers
    order = sorted(X, key=lambda x: abs(fnum(x["milp"]["mean_dZ_%"])))
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    y = np.arange(len(order))
    ax.barh(y, [fnum(x["milp"]["mean_dZ_%"]) for x in order], height=0.6, color=BLUE, edgecolor=SURF, label="MILP")
    ax.scatter([fnum(x["zs"]["mean_dZ_%"]) for x in order], y, marker="o", s=22, color=ORANGE, zorder=3,
               label="RL zero-shot", edgecolor=SURF, linewidth=0.8)
    ax.scatter([fnum(x["rt"]["mean_dZ_%"]) for x in order], y, marker="D", s=20, color=AQUA, zorder=3,
               label="RL retrained", edgecolor=SURF, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([x["id"] for x in order], fontsize=7)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("mean change of $Z$ against the baseline (%, paired instances)")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.savefig(os.path.join(FIG, "exp_tornado.pdf"))
    plt.close(fig)

    # personal van share per variant: MILP, RL zs, RL rt (with baseline lines)
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    xs = np.arange(len(X))
    for j, (k, lab, col) in enumerate((("milp", "MILP", BLUE), ("zs", "RL zero-shot", ORANGE), ("rt", "RL retrained", AQUA))):
        ax.bar(xs + (j - 1) * 0.27, [fnum(x[k]["share_van_personal_%"]) for x in X], width=0.25, color=col,
               edgecolor=SURF, label=lab)
    ax.axhline(fnum(X[0]["milp"]["baseline_share_van_personal_%"]), color=BLUE, lw=0.9, ls="--")
    ax.axhline(fnum(X[0]["zs"]["baseline_share_van_personal_%"]), color=ORANGE, lw=0.9, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([x["id"] for x in X], rotation=55, ha="right", fontsize=6.5)
    ax.set_ylabel("van share, personal free-choice EVs (%)")
    ax.legend(fontsize=7.5, ncol=3, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.savefig(os.path.join(FIG, "exp_vanshare.pdf"))
    plt.close(fig)

    # dZ agreement MILP vs RL
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    mz = [fnum(x["milp"]["mean_dZ_%"]) for x in X]
    ax.scatter(mz, [fnum(x["zs"]["mean_dZ_%"]) for x in X], color=ORANGE, s=24, label="RL zero-shot",
               edgecolor=SURF, linewidth=0.8, zorder=3)
    ax.scatter(mz, [fnum(x["rt"]["mean_dZ_%"]) for x in X], color=AQUA, marker="D", s=20, label="RL retrained",
               edgecolor=SURF, linewidth=0.8, zorder=3)
    lim = max(abs(v) for v in mz) * 1.15
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.8, ls="--")
    ax.text(lim * 0.35, lim * 0.62, "RL = MILP", color=INK2, fontsize=7.5, rotation=38)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("MILP: mean change of $Z$ (%)")
    ax.set_ylabel("RL: mean change of $Z$ (%)")
    ax.legend(fontsize=7.5, loc="upper left")
    fig.savefig(os.path.join(FIG, "exp_agreement.pdf"))
    plt.close(fig)
    zs_err = [abs(fnum(x["zs"]["mean_dZ_%"]) - fnum(x["milp"]["mean_dZ_%"])) for x in X]
    rt_err = [abs(fnum(x["rt"]["mean_dZ_%"]) - fnum(x["milp"]["mean_dZ_%"])) for x in X]
    num("agreeZsMean", st.mean(zs_err), "{:.2f}")
    num("agreeRtMean", st.mean(rt_err), "{:.2f}")
    num("agreeSign", sum(1 for x in X if np.sign(fnum(x["zs"]["mean_dZ_%"])) == np.sign(fnum(x["milp"]["mean_dZ_%"]))
                         and np.sign(fnum(x["rt"]["mean_dZ_%"])) == np.sign(fnum(x["milp"]["mean_dZ_%"]))))
    num("nVariants", len(X))

    # valid plans per variant (feasibility), MILP vs RL retrained
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    ax.bar(xs - 0.2, [int(x["milp"]["valid"]) for x in X], width=0.38, color=BLUE, edgecolor=SURF, label="MILP")
    ax.bar(xs + 0.2, [int(x["rt"]["valid"]) for x in X], width=0.38, color=AQUA, edgecolor=SURF, label="RL retrained")
    ax.axhline(int(X[0]["milp"]["baseline_valid"]), color=BLUE, lw=0.9, ls="--")
    ax.axhline(int(X[0]["rt"]["baseline_valid"]), color=AQUA, lw=0.9, ls="--")
    ax.set_ylim(400, 520)
    ax.set_xticks(xs)
    ax.set_xticklabels([x["id"] for x in X], rotation=55, ha="right", fontsize=6.5)
    ax.set_ylabel("instances with a valid plan")
    ax.legend(fontsize=7.5, ncol=2, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.savefig(os.path.join(FIG, "exp_valid.pdf"))
    plt.close(fig)

    # summary table: one row per variant
    rows = []
    for x in X:
        rows.append([tex(x["id"]), f(fnum(x["milp"]["mean_dZ_%"]), 2, True), f(fnum(x["zs"]["mean_dZ_%"]), 2, True),
                     f(fnum(x["rt"]["mean_dZ_%"]), 2, True), x["milp"]["valid"], x["rt"]["valid"],
                     f(fnum(x["zs"]["mean_gap_to_milp_%"]), 2), f(fnum(x["rt"]["mean_gap_to_milp_%"]), 2),
                     f(x["train_min"], 0), x["best_epoch"]])
    write_table("exp_summary", ["variant", "$\\Delta Z$ MILP", "$\\Delta Z$ RL-ZS", "$\\Delta Z$ RL-RT", "valid MILP",
                                "valid RL-RT", "gap RL-ZS", "gap RL-RT", "train (min)", "best epoch"], rows,
                "@{}lrrrrrrrrr@{}", size="\\footnotesize")
    for x in X:
        k = x["id"].replace("_", "").replace(".", "p").replace("0", "o").replace("1", "i").replace("2", "ii").replace("3", "iii").replace("4", "iv").replace("5", "v").replace("6", "vi").replace("7", "vii").replace("8", "viii").replace("9", "ix")
        num("dZ" + k, fnum(x["milp"]["mean_dZ_%"]), "{:+.2f}")


def by_income_figure(X):
    """Personal van share by income band, MILP, for E1 and E5-personal variants vs baseline."""
    sets = [("E1_van_tariff_MILP", ["E1_pM_15", "E1_pM_35"], "E1: van tariff $p^M$"),
            ("E5_value_of_charging_time_MILP", ["E5_cC1_personal_7.5", "E5_cC1_personal_12.5"],
             "E5: personal value of charging time")]
    bands = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.5), sharey=True)
    for ax, (folder, vars_, title) in zip(axs, sets):
        rows = read(os.path.join(F4, folder, "results", "comparison", "comparison_by_income.csv"))
        series = [("baseline (milp_300s)", "baseline", BLUE)] + [(f"{v} (milp_300s)", v, c) for v, c in zip(vars_, (ORANGE, AQUA))]
        x = np.arange(len(bands))
        for j, (run, lab, col) in enumerate(series):
            vals = []
            for b in bands:
                r = [r for r in rows if r["run"] == run and r["segment"] == "personal" and r["income_band"] == b]
                vals.append(fnum(r[0]["share_van_free_choice_%"], 0) if r else 0)
            ax.bar(x + (j - 1) * 0.27, vals, width=0.25, color=col, edgecolor=SURF, label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels(bands, fontsize=7)
        ax.set_title(title)
        ax.set_xlabel("income band $\\theta$ (personal EVs)")
        ax.legend(fontsize=6.5, loc="upper left")
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("van share (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "exp_income.pdf"))
    plt.close(fig)


def direction_check():
    def opt(path):
        return {(r["instance"], r["n"]): float(r["Z_total"]) for r in read(path)
                if r["run_id"] == "milp_300s" and r["status"] == "optimal"}
    base = opt(os.path.join(F4, "baseline_MILP", "results", "runs.csv"))
    cases = [("E1_van_tariff", "E1_pM_15", -1), ("E1_van_tariff", "E1_pM_35", 1),
             ("E5_value_of_charging_time", "E5_cC1_personal_7.5", -1), ("E5_value_of_charging_time", "E5_cC1_personal_12.5", 1),
             ("E5_value_of_charging_time", "E5_omega1_commercial_4.5", -1), ("E5_value_of_charging_time", "E5_omega1_commercial_7.5", 1),
             ("E6_money_sensitivity", "E6_mu0_personal_1.125", -1), ("E6_money_sensitivity", "E6_mu0_personal_1.875", 1),
             ("E6_money_sensitivity", "E6_mu0_commercial_1.125", -1), ("E6_money_sensitivity", "E6_mu0_commercial_1.875", 1),
             ("E2_van_fleet_and_seats", "E2_vans_3", 1), ("E2_van_fleet_and_seats", "E2_vans_7", -1),
             ("E2_van_fleet_and_seats", "E2_seats_4", 1), ("E2_van_fleet_and_seats", "E2_seats_8", -1),
             ("E8_van_battery", "E8_battery_200", -1)]
    rows, bad_total = [], 0
    for folder, var, sign in cases:
        v = opt(os.path.join(F4, folder + "_MILP", "results", var, "runs.csv"))
        both = set(v) & set(base)
        bad = sum(1 for k in both if sign * (v[k] - base[k]) < -1e-3 * max(1.0, abs(base[k])))
        bad_total += bad
        mean = st.mean((v[k] - base[k]) / base[k] * 100 for k in both)
        rows.append([tex(var), "must not fall" if sign > 0 else "must not rise", len(both), f(mean, 3, True), bad])
    write_table("direction", ["variant", "$Z$ at the optimum", "optimal in both", "mean $\\Delta Z$ (\\%)",
                              "wrong direction"], rows, "@{}lrrrr@{}", size="\\footnotesize")
    num("directionBad", bad_total)


# ===================== E9 station capacity and E10 charging demand (RL only, retrained models) ============
def _retrained_comp(folder):
    """{variant: comparison row} for the retrained RL run of an RL-only experiment (robust if only one run)."""
    all_rows = read(os.path.join(F4, folder, "results", "comparison", "comparison.csv"))
    rt = [r for r in all_rows if r.get("run_id") == "rl_f4_retrained"]
    return {r["variant"]: r for r in (rt if rt else all_rows)}


def _baseline_rl_Z():
    """{(instance, n): Z} for the valid baseline-RL plans (the reference the experiments are measured against)."""
    rs = runs_of(os.path.join(F4, "baseline_RL", "results", "runs.csv"), "rl_f4")
    return {key(r): fnum(r["Z_total"]) for r in rs if r["valid"] == "yes"}


def _dZ_by_n(folder, variant, baseZ):
    """mean paired Delta Z (%) against the baseline RL plan, per instance size n."""
    out = {n: [] for n in SIZES}
    for r in read(os.path.join(F4, folder, "results", variant, "runs.csv")):
        b = baseZ.get(key(r))
        if r["valid"] == "yes" and b:
            out[int(r["n"])].append(100 * (fnum(r["Z_total"]) - b) / b)
    return [st.mean(out[n]) if out[n] else float("nan") for n in SIZES]


def _mean_energy():
    return st.mean(fnum(r["energy_kWh"]) for r in read(os.path.join(F4, "data", "solomon_data.csv"))
                   if r["type"] == "ev")


def station_capacity():
    """E9: guns per station and power per gun (RL, retrained)."""
    comp = _retrained_comp("E9_station_capacity_RL")
    order = ["E9_guns_1", "E9_guns_4", "E9_power_11", "E9_power_44"]
    shortlab = {"E9_guns_1": "1 gun\n(half)", "E9_guns_4": "4 guns\n(double)",
                "E9_power_11": "11 kW\n(half)", "E9_power_44": "44 kW\n(double)"}
    rows = []
    for v in order:
        r = comp[v]
        rows.append([tex(v), label_tex(r["label"]), f(fnum(r["mean_dZ_%"]), 2, True),
                     f(fnum(r["mean_d_operator_cost_%"]), 2, True), f(fnum(r["mean_d_customer_cost_%"]), 2, True),
                     f"{f(fnum(r['baseline_mean_queue_commercial_min']), 2)} $\\to$ {f(fnum(r['mean_queue_commercial_min']), 2)}",
                     f"{f(fnum(r['baseline_share_van_personal_%']), 1)} $\\to$ {f(fnum(r['share_van_personal_%']), 1)}",
                     r["newly_valid"], r["no_longer_valid"], r["paired"]])
    write_table("exp_E9",
                ["variant", "what changes", "$\\Delta Z$ \\%", "$\\Delta$ oper. \\%", "$\\Delta$ cust. \\%",
                 "comm. queue (min)", "van \\% pers.", "gained", "lost", "paired"],
                rows, "@{}llrrrllrrr@{}", size="\\scriptsize")
    km = {"E9_guns_1": "eNineGunsHalf", "E9_guns_4": "eNineGunsDbl",
          "E9_power_11": "eNinePowHalf", "E9_power_44": "eNinePowDbl"}
    for v, kk in km.items():
        num(kk + "Dz", fnum(comp[v]["mean_dZ_%"]), "{:+.1f}")
        num(kk + "Lost", comp[v]["no_longer_valid"])
        num(kk + "Gain", comp[v]["newly_valid"])
    meanE = _mean_energy()
    num("meanEnergy", meanE, "{:.0f}")
    for P, kk in ((11, "eNineTauHalf"), (22, "eNineTauBase"), (44, "eNineTauDbl")):
        num(kk, 60 * meanE / P, "{:.0f}")
    num("eNinePowDblDzAbs", abs(fnum(comp["E9_power_44"]["mean_dZ_%"])), "{:.1f}")
    num("eNineGunsDblDzAbs", abs(fnum(comp["E9_guns_4"]["mean_dZ_%"])), "{:.1f}")
    num("eNinePowHalfVan", fnum(comp["E9_power_11"]["share_van_personal_%"]), "{:.0f}")
    num("eNineGunsHalfQ", fnum(comp["E9_guns_1"]["mean_queue_commercial_min"]), "{:.1f}")
    num("eNineGunsHalfQBase", fnum(comp["E9_guns_1"]["baseline_mean_queue_commercial_min"]), "{:.1f}")

    # figure 1: effect on cost (bars) + feasibility change
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.5, 1]})
    x = np.arange(len(order))
    ax = axs[0]
    for j, (col_, c, lab) in enumerate((("mean_dZ_%", BLUE, "total $Z$"),
                                        ("mean_d_operator_cost_%", AQUA, "operator"),
                                        ("mean_d_customer_cost_%", ORANGE, "customer"))):
        ax.bar(x + (j - 1) * 0.27, [fnum(comp[v][col_]) for v in order], width=0.25, color=c,
               edgecolor=SURF, label=lab)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([shortlab[v] for v in order], fontsize=7)
    ax.set_ylabel("change vs baseline (%)")
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    ax.grid(axis="x", visible=False)
    ax = axs[1]
    ax.bar(x - 0.19, [int(comp[v]["newly_valid"]) for v in order], width=0.36, color=AQUA, edgecolor=SURF, label="gained")
    ax.bar(x + 0.19, [int(comp[v]["no_longer_valid"]) for v in order], width=0.36, color=ORANGE, edgecolor=SURF, label="lost")
    ax.set_xticks(x); ax.set_xticklabels([shortlab[v] for v in order], fontsize=7)
    ax.set_ylabel("instances")
    ax.legend(fontsize=7, loc="upper center")
    ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.3)
    fig.savefig(os.path.join(FIG, "exp_e9.pdf"))
    plt.close(fig)

    # figure 2: mechanism -- charging time vs power, and dZ vs the charging-time stretch
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axs[0]
    Pg = np.linspace(9, 46, 200)
    ax.plot(Pg, 60 * meanE / Pg, color=BLUE, lw=1.8)
    for Pv, c in ((11, ORANGE), (22, INK2), (44, AQUA)):
        ax.scatter([Pv], [60 * meanE / Pv], color=c, s=30, zorder=3, edgecolor=SURF)
        ax.annotate(f"{Pv} kW: {60*meanE/Pv:.0f} min", (Pv, 60*meanE/Pv), xytext=(5, 5),
                    textcoords="offset points", fontsize=6.5, color=INK2)
    ax.set_xlabel("power per gun $P^F$ (kW)")
    ax.set_ylabel("mean charging time (min)")
    ax.set_title(r"$\tau^F = 60\,\bar E/P^F \;\propto\; 1/P^F$", fontsize=8)
    ax = axs[1]
    stretch = [22 / 11, 1.0, 22 / 44]
    dzp = [fnum(comp["E9_power_11"]["mean_dZ_%"]), 0.0, fnum(comp["E9_power_44"]["mean_dZ_%"])]
    o = np.argsort(stretch)
    ax.plot(np.array(stretch)[o], np.array(dzp)[o], color=BLUE, lw=1.4, marker="o",
            markerfacecolor=ORANGE, markeredgecolor=SURF, markersize=6)
    for xv, yv, t in zip(stretch, dzp, ["11 kW", "22 kW (base)", "44 kW"]):
        ax.annotate(t, (xv, yv), xytext=(4, 7 if t == "44 kW" else -10), textcoords="offset points",
                    fontsize=6.5, color=INK2)
    ax.axhline(0, color=INK2, lw=0.8); ax.axvline(1, color=MUTED, lw=0.7, ls=":")
    ax.set_xlabel("charging-time stretch $22/P^F$")
    ax.set_ylabel("change of $Z$ (%)")
    ax.set_title("cost rises with charging time", fontsize=8)
    fig.tight_layout(w_pad=1.3)
    fig.savefig(os.path.join(FIG, "exp_e9_power.pdf"))
    plt.close(fig)


def charging_demand():
    """E10: energy demand per EV scaled by s (RL, retrained)."""
    comp = _retrained_comp("E10_charging_demand_RL")
    lo, hi = "E10_demand_0.75", "E10_demand_1.25"
    s = [0.75, 1.0, 1.25]
    dz = [fnum(comp[lo]["mean_dZ_%"]), 0.0, fnum(comp[hi]["mean_dZ_%"])]
    dop = [fnum(comp[lo]["mean_d_operator_cost_%"]), 0.0, fnum(comp[hi]["mean_d_operator_cost_%"])]
    dcu = [fnum(comp[lo]["mean_d_customer_cost_%"]), 0.0, fnum(comp[hi]["mean_d_customer_cost_%"])]
    slope = (dz[2] - dz[0]) / (s[2] - s[0])
    num("eTenDownDz", dz[0], "{:+.1f}"); num("eTenUpDz", dz[2], "{:+.1f}")
    num("eTenSlope", slope, "{:.0f}"); num("eTenElastic", slope / 100, "{:.2f}")
    num("eTenUpLost", comp[hi]["no_longer_valid"]); num("eTenDownGain", comp[lo]["newly_valid"])
    num("eTenDownDzAbs", abs(dz[0]), "{:.1f}"); num("eTenNonElastic", 100 - slope, "{:.0f}")
    rows = []
    for v in (lo, hi):
        r = comp[v]
        rows.append([tex(v), label_tex(r["label"]), f(fnum(r["mean_dZ_%"]), 2, True),
                     f(fnum(r["mean_d_operator_cost_%"]), 2, True), f(fnum(r["mean_d_customer_cost_%"]), 2, True),
                     r["newly_valid"], r["no_longer_valid"], r["paired"]])
    write_table("exp_E10",
                ["variant", "what changes", "$\\Delta Z$ \\%", "$\\Delta$ oper. \\%", "$\\Delta$ cust. \\%",
                 "gained", "lost", "paired"], rows, "@{}llrrrrrr@{}", size="\\footnotesize")

    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axs[0]
    ax.plot(s, [slope * (ss - 1) for ss in s], color=MUTED, lw=1.0, ls="--", label=f"affine fit ({slope:.0f}\\%/unit)")
    ax.plot(s, dz, color=BLUE, lw=1.6, marker="o", markerfacecolor=ORANGE, markeredgecolor=SURF, markersize=7, label="total $Z$")
    ax.plot(s, dop, color=AQUA, lw=1.2, marker="s", markersize=4, label="operator")
    ax.plot(s, dcu, color=ORANGE, lw=1.2, marker="^", markersize=4, label="customer")
    ax.axhline(0, color=INK2, lw=0.8); ax.axvline(1, color=MUTED, lw=0.7, ls=":")
    ax.set_xlabel("energy scale $s$ (per-EV demand $\\times\\, s$)")
    ax.set_ylabel("change of $Z$ (%)")
    ax.set_title("cost is nearly linear in demand", fontsize=8)
    ax.legend(fontsize=6.5, loc="upper left"); ax.set_xticks(s)
    ax = axs[1]
    x = np.arange(2)
    ax.bar(x - 0.19, [int(comp[v]["newly_valid"]) for v in (lo, hi)], width=0.36, color=AQUA, edgecolor=SURF, label="gained")
    ax.bar(x + 0.19, [int(comp[v]["no_longer_valid"]) for v in (lo, hi)], width=0.36, color=ORANGE, edgecolor=SURF, label="lost")
    ax.set_xticks(x); ax.set_xticklabels(["$s=0.75$", "$s=1.25$"], fontsize=7.5)
    ax.set_ylabel("instances"); ax.legend(fontsize=7, loc="upper center")
    ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.3)
    fig.savefig(os.path.join(FIG, "exp_e10.pdf"))
    plt.close(fig)


def write_numbers():
    with open(os.path.join(HERE, "numbers.json"), "w", encoding="utf-8") as fh:
        json.dump(NUM, fh, indent=1)
    with open(os.path.join(HERE, "numbers.tex"), "w", encoding="utf-8") as fh:
        for k, v in NUM.items():
            words = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
            name = "f" + "".join(ch if ch.isalpha() else (words[int(ch)] if ch.isdigit() else "") for ch in k)
            fh.write(f"\\newcommand{{\\{name}}}{{{v}}}\n")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    dataset()
    E = experiments_table()
    milp, greedy = baseline_milp()
    baseline_rl(milp)
    segment_tables()
    X = experiments(E)
    experiment_tables(X)
    by_income_figure(X)
    direction_check()
    station_capacity()
    charging_demand()
    write_numbers()
    print(json.dumps(NUM, indent=1))
