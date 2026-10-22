"""Solve main_formulation (continuous-time MILP, eqs (1)-(20)) with Gurobi on the
instances made by make_dataset.py, validate every solution and write result tables.

Usage examples
  python solve_milp.py ../dataset/instances                         # all 39 instances
  python solve_milp.py ../dataset/instances --resume                # continue an interrupted run
  python solve_milp.py ../dataset/instances --only C101_n10
  python solve_milp.py ../dataset/instances --sizes 10 25 --time-limit 300
  python solve_milp.py ../dataset/instances --sizes 10 --gpu

Every instance is solved with the complete main_formulation, eqs (1)-(20). Every solution is
checked against all constraints. Instances run from small to large; summary.csv / costs.csv
are rewritten after every instance.

Outputs (in --out, default ../results):
  summary.csv, costs.csv                        one row per instance
  <instance>/evs.csv, vans.csv, guns.csv        detailed tables per instance
  <instance>/gurobi.log, solution.json
  results.xlsx                                  all tables in one workbook
  report.md                                     all tables as Markdown

GPU: Gurobi's GPU support is the PDHG algorithm for *linear* programs. It cannot run
branch-and-bound. With --gpu the script solves the LP relaxation (root bound) with PDHG on
the GPU (Method=6, PDHGGPU=1) and reports it; the MILP itself is solved on the CPU.
This only uses the GPU on a GPU build of Gurobi (Linux + NVIDIA); otherwise Gurobi
says "Not a GPU build" and runs PDHG on the CPU. The log line is reported in the summary.
"""
import argparse
import csv
import gc
import glob
import json
import os
import sys
import time

import gurobipy as gp
from gurobipy import GRB

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_plan import validate  # noqa: E402


# --------------------------------------------------------------------- model
def build_model(J, env):
    """main_formulation, eqs (1)-(20)."""
    p, ev = J["params"], J["ev"]
    n, F, Mn = J["n_ev"], J["n_fcs"], J["n_vans"]
    T, c = p["T_min"], p["costs_Rs"]
    E, r, R, w = ev["E_kWh"], ev["r_min"], ev["R_km"], ev["w"]
    tauF, tauM = ev["tauF_min"], ev["tauM_min"]
    dEF, tEF = J["d_ev_fcs_km"], J["t_ev_fcs_min"]
    dN, tN = J["d_nodes_km"], J["t_nodes_min"]
    eta, beta, B, Q = p["eta"], p["beta_kWh_per_km"], p["B_kWh"], p["Q"]

    I = range(1, n + 1)                     # EVs are nodes 1..n
    END = n + 1                             # end depot (same place as 0)
    def d(a, b):
        return dN[0 if a == END else a][0 if b == END else b]
    def t(a, b):
        return tN[0 if a == END else a][0 if b == END else b]
    K = p["guns_per_fcs"]
    guns = [(f, k) for f in range(F) for k in range(K)]
    Fi = {i: [f for f in range(F) if dEF[i - 1][f] <= R[i - 1]] for i in I}      # F_i
    Gi = {i: [(f, k) for f in Fi[i] for k in range(K)] for i in I}               # G_i
    Mv = range(Mn)
    A = [(i, j) for i in [0, *I] for j in [*I, END] if i != j and (i, j) != (0, END)]
    out_arcs = {a: [] for a in [0, *I]}
    in_arcs = {b: [] for b in [*I, END]}
    for (a, b) in A:
        out_arcs[a].append(b)
        in_arcs[b].append(a)
    users = {g: [i for i in I if g in Gi[i]] for g in guns}                     # EVs per gun
    Pg = {g: [(a, b) for idx, a in enumerate(users[g]) for b in users[g][idx + 1:]]
          for g in guns}                                                         # P_g
    pairs = sorted({pr for g in guns for pr in Pg[g]})
    Gplus = [g for g in guns if users[g]]

    m = gp.Model(J["name"], env=env)
    y = m.addVars([(i, g) for i in I for g in Gi[i]], vtype=GRB.BINARY, name="y")
    z = m.addVars(I, Mv, vtype=GRB.BINARY, name="z")
    x = m.addVars([(i, j, k) for (i, j) in A for k in Mv], vtype=GRB.BINARY, name="x")
    u = m.addVars(Mv, vtype=GRB.BINARY, name="u")
    o = m.addVars(pairs, vtype=GRB.BINARY, name="o")
    S = m.addVars(I, lb=0.0, name="S")

    Y = {(i, f): gp.quicksum(y[i, g] for g in Gi[i] if g[0] == f) for i in I for f in Fi[i]}
    Cexp = {i: S[i] + tauF[i - 1] * gp.quicksum(y[i, g] for g in Gi[i])
            + tauM[i - 1] * gp.quicksum(z[i, k] for k in Mv) for i in I}

    # (1) objective
    Z1 = c["c_EV"] * gp.quicksum(dEF[i - 1][f] * Y[i, f] for i in I for f in Fi[i])
    Z2 = c["c_M"] * gp.quicksum(d(i, j) * x[i, j, k] for (i, j) in A for k in Mv)
    Z3 = c["c_F"] * u.sum()
    Z4 = (c["pi_F"] * gp.quicksum(E[i - 1] * y[i, g] for i in I for g in Gi[i])
          + c["pi_M"] * gp.quicksum(E[i - 1] / eta * z[i, k] for i in I for k in Mv))
    Z5 = c["c_W"] * gp.quicksum(w[i - 1] * (Cexp[i] - r[i - 1]) for i in I)
    m.setObjective(Z1 + Z2 + Z3 + Z4 + Z5, GRB.MINIMIZE)

    # (2) assignment
    m.addConstrs((gp.quicksum(y[i, g] for g in Gi[i]) + z.sum(i, "*") == 1 for i in I), "assign")
    # (3) FCS arrival
    m.addConstrs((S[i] >= r[i - 1] + gp.quicksum(tEF[i - 1][f] * Y[i, f] for f in Fi[i])
                  for i in I), "fcs_arrival")
    # (4)-(5) disjunctive gun sequencing, M^F_i = T + tauF_i
    for g in guns:
        for (i, j) in Pg[g]:
            MFi, MFj = T + tauF[i - 1], T + tauF[j - 1]
            m.addConstr(S[j] >= S[i] + tauF[i - 1] - MFi * (3 - y[i, g] - y[j, g] - o[i, j]),
                        f"disj1[{i},{j},{g[0]},{g[1]}]")
            m.addConstr(S[i] >= S[j] + tauF[j - 1] - MFj * (2 - y[i, g] - y[j, g] + o[i, j]),
                        f"disj2[{i},{j},{g[0]},{g[1]}]")
    # (6) gun load (valid inequality), only for guns in G+
    for g in Gplus:
        first = min(r[i - 1] + tEF[i - 1][g[0]] for i in users[g])
        m.addConstr(gp.quicksum(tauF[i - 1] * y[i, g] for i in users[g]) <= T - first,
                    f"gunload[{g[0]},{g[1]}]")
    # (7) symmetry between the guns of a station (written for consecutive guns)
    for f in range(F):
        for k in range(K - 1):
            m.addConstr(gp.quicksum(y[i, (f, k)] for i in users[(f, k)])
                        >= gp.quicksum(y[i, (f, k + 1)] for i in users[(f, k + 1)]),
                        f"gunsym[{f},{k}]")
    # (8)-(11) routing (sums over the arcs that exist in A)
    m.addConstrs((gp.quicksum(x[0, j, k] for j in out_arcs[0]) == u[k] for k in Mv), "depot_out")
    m.addConstrs((gp.quicksum(x[i, END, k] for i in in_arcs[END]) == u[k] for k in Mv), "depot_in")
    m.addConstrs((gp.quicksum(x[i, j, k] for i in in_arcs[j]) == z[j, k]
                  for j in I for k in Mv), "flow_in")
    m.addConstrs((gp.quicksum(x[j, l, k] for l in out_arcs[j]) == z[j, k]
                  for j in I for k in Mv), "flow_out")
    # (12) capacity, (13) battery
    m.addConstrs((z.sum("*", k) <= Q * u[k] for k in Mv), "van_cap")
    m.addConstrs((gp.quicksum(E[i - 1] / eta * z[i, k] for i in I)
                  + beta * gp.quicksum(d(i, j) * x[i, j, k] for (i, j) in A) <= B * u[k]
                  for k in Mv), "battery")
    # (14) first stop, (15) sequencing, (16) return by T
    m.addConstrs((S[j] >= t(0, j) * x[0, j, k] for j in I for k in Mv), "depart")
    ev_arcs = [(i, j) for (i, j) in A if i != 0 and j != END]
    for k in Mv:
        for (i, j) in ev_arcs:                     # (15) only for arcs that exist
            MM = T + tauM[i - 1] + t(i, j)
            m.addConstr(S[j] >= S[i] + tauM[i - 1] + t(i, j) - MM * (1 - x[i, j, k]),
                        f"seq[{i},{j},{k}]")
        for i in I:
            m.addConstr(S[i] + tauM[i - 1] + t(i, END)
                        <= T + (tauM[i - 1] + t(i, END)) * (1 - x[i, END, k]), f"return[{i},{k}]")
    # (17) release, (18) horizon
    m.addConstrs((S[i] >= r[i - 1] for i in I), "release")
    m.addConstrs((Cexp[i] <= T for i in I), "horizon")
    # (19) van symmetry
    m.addConstrs((u[k] >= u[k + 1] for k in range(Mn - 1)), "vansym")

    data = dict(I=I, END=END, guns=guns, Gi=Gi, Mv=Mv, A=A, pairs=pairs,
                y=y, z=z, x=x, u=u, o=o, S=S, Cexp=Cexp, Z=(Z1, Z2, Z3, Z4, Z5))
    return m, data


# --------------------------------------------------------------- warm start
def warm_start(J, data):
    """Load the greedy baseline plan (feasible) as the MIP start."""
    g = J["greedy_baseline"]
    I, Mv, A, END = data["I"], data["Mv"], data["A"], data["END"]
    y, z, x, u, o, S = (data[k] for k in "y z x u o S".split())
    for v in [*y.values(), *z.values(), *x.values(), *u.values(), *o.values()]:
        v.Start = 0
    gun_of = {}
    van_of = {}
    # relabel the guns of each station so gun 1 carries the most EVs (constraint (7))
    count = {}
    for res in g["resource"]:
        if res.startswith("FCS"):
            key = (int(res[3:res.index("-")]) - 1, int(res.split("gun")[1]) - 1)
            count[key] = count.get(key, 0) + 1
    relabel = {}
    for f in range(J["n_fcs"]):
        ks = sorted(range(J["params"]["guns_per_fcs"]), key=lambda k: -count.get((f, k), 0))
        relabel.update({(f, old): (f, new) for new, old in enumerate(ks)})
    for i in I:
        res = g["resource"][i - 1]
        if res.startswith("FCS"):
            gun = relabel[(int(res[3:res.index("-")]) - 1, int(res.split("gun")[1]) - 1)]
            y[i, gun].Start = 1
            gun_of[i] = gun
        S[i].Start = g["S"][i - 1]
    # greedy van numbers: routes are listed in van order among used vans
    used = 0
    for route in g["van_routes"]:
        k = used
        used += 1
        u[k].Start = 1
        nodes = [0, *route[1:-1], END]
        for a, b in zip(nodes, nodes[1:]):
            x[a, b, k].Start = 1
        for node in route[1:-1]:
            z[node, k].Start = 1
            van_of[node] = k
    for (i, j) in data["pairs"]:
        if i in gun_of and gun_of.get(j) == gun_of[i]:
            o[i, j].Start = 1 if g["S"][i - 1] <= g["S"][j - 1] else 0


# ------------------------------------------------------------ extract plan
def extract(J, data):
    I, Mv, END = data["I"], data["Mv"], data["END"]
    y, z, x, S, Cexp = data["y"], data["z"], data["x"], data["S"], data["Cexp"]
    res, Sv, Cv = [], [], []
    for i in I:
        gs = [g for g in data["Gi"][i] if y[i, g].X > 0.5]
        ks = [k for k in Mv if z[i, k].X > 0.5]
        res.append(f"FCS{gs[0][0] + 1}-gun{gs[0][1] + 1}" if gs else f"van{ks[0] + 1}")
        Sv.append(S[i].X)
        Cv.append(Cexp[i].getValue())
    routes, route_van = [], []
    for k in Mv:
        succ = {a: b for (a, b) in data["A"] if x[a, b, k].X > 0.5}
        if 0 not in succ:
            continue
        route, node = [0], succ[0]
        while node != END:
            route.append(node)
            node = succ[node]
        route.append(0)
        routes.append(route)
        route_van.append(k + 1)
    return {"resource": res, "S": Sv, "C": Cv, "van_routes": routes, "route_van": route_van}


# ------------------------------------------------------------- GPU (PDHG)
def gpu_root_lp(model, threads):
    """Solve the LP relaxation with PDHG on the GPU (if this is a GPU build)."""
    lp = model.relax()
    lp.Params.Method = 6
    lp.Params.PDHGGPU = 1
    lp.Params.Threads = threads
    lp.Params.OutputFlag = 0
    lp.Params.LogFile = ""
    msgs = []
    def cb(mm, where):
        if where == GRB.Callback.MESSAGE:
            msgs.append(mm.cbGet(GRB.Callback.MSG_STRING))
    t0 = time.time()
    lp.optimize(cb)
    used_gpu = not any("Not a GPU build" in s for s in msgs)
    bound = lp.ObjVal if lp.Status == GRB.OPTIMAL else float("nan")
    return bound, time.time() - t0, used_gpu


# ------------------------------------------------------------ result tables
def tables(J, plan, Zparts):
    p, ev = J["params"], J["ev"]
    n, T = J["n_ev"], p["T_min"]
    dEF, dN = J["d_ev_fcs_km"], J["d_nodes_km"]
    ev_rows = []
    for i in range(n):
        res = plan["resource"][i]
        kind = "FCS" if res.startswith("FCS") else "MCS"
        drive = dEF[i][int(res[3:res.index("-")]) - 1] if kind == "FCS" else 0.0
        wait = plan["C"][i] - ev["r_min"][i]
        ev_rows.append({
            "EV": i + 1, "served_by": res, "type": kind,
            "reachable_fcs": sum(1 for f in range(J["n_fcs"]) if dEF[i][f] <= ev["R_km"][i]),
            "E_kWh": round(ev["E_kWh"][i], 2), "w": round(ev["w"][i], 3),
            "r_min": round(ev["r_min"][i], 1), "start_min": round(plan["S"][i], 1),
            "end_min": round(plan["C"][i], 1), "time_to_charged_min": round(wait, 1),
            "EV_drive_km": round(drive, 2),
            "inconvenience_Rs": round(p["costs_Rs"]["c_W"] * ev["w"][i] * wait, 2)})
    van_rows = []
    for route, k in zip(plan["van_routes"], plan["route_van"]):
        stops = route[1:-1]
        km = sum(dN[a][b] for a, b in zip(route, route[1:]))
        delivered = sum(ev["E_kWh"][s - 1] for s in stops)
        drawn = delivered / p["eta"] + p["beta_kWh_per_km"] * km
        last = stops[-1]
        back = plan["C"][last - 1] + J["t_nodes_min"][last][0]
        van_rows.append({
            "van": k, "route": " -> ".join("D" if s == 0 else f"EV{s}" for s in route),
            "stops": len(stops), "km": round(km, 2), "kWh_delivered": round(delivered, 2),
            "kWh_drawn": round(drawn, 2), "battery_left_kWh": round(p["B_kWh"] - drawn, 2),
            "back_at_depot_min": round(back, 1), "slack_to_T_min": round(T - back, 1)})
    gun_rows = []
    for f in range(J["n_fcs"]):
        for k in range(p["guns_per_fcs"]):
            name = f"FCS{f + 1}-gun{k + 1}"
            evs = sorted((plan["S"][i], i + 1) for i in range(n) if plan["resource"][i] == name)
            busy = sum(ev["tauF_min"][i - 1] for _, i in evs)
            gun_rows.append({"gun": name, "EVs": len(evs),
                             "sequence": " -> ".join(f"EV{i}" for _, i in evs) or "-",
                             "busy_min": round(busy, 1),
                             "utilisation_%": round(100 * busy / T, 1)})
    return ev_rows, van_rows, gun_rows


def all_keys(rows):
    """Union of the keys of all rows, in first-seen order (error rows have fewer keys)."""
    keys = []
    for r_ in rows:
        keys += [k for k in r_ if k not in keys]
    return keys


def write_csv(path, rows):
    if not rows:
        open(path, "w").close()
        return
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=all_keys(rows), restval="")
        wr.writeheader()
        wr.writerows(rows)


def md_table(rows):
    if not rows:
        return "_(none)_\n"
    keys = all_keys(rows)
    out = "| " + " | ".join(keys) + " |\n|" + "---|" * len(keys) + "\n"
    for r_ in rows:
        out += "| " + " | ".join(str(r_.get(k, "")) for k in keys) + " |\n"
    return out


def print_table(title, rows, keys=None):
    if not rows:
        return
    keys = keys or all_keys(rows)
    width = {k: max(len(k), *(len(str(r_.get(k, ""))) for r_ in rows)) for k in keys}
    print(f"\n{title}")
    print("  ".join(k.ljust(width[k]) for k in keys))
    print("  ".join("-" * width[k] for k in keys))
    for r_ in rows:
        print("  ".join(str(r_.get(k, "")).ljust(width[k]) for k in keys))


def write_xlsx(path, sheets):
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
    except ImportError:
        return False
    wb = Workbook()
    wb.remove(wb.active)
    for name, rows in sheets:
        ws = wb.create_sheet(name[:31])
        if not rows:
            continue
        keys = all_keys(rows)
        ws.append(keys)
        for cell in ws[1]:
            cell.font = Font(bold=True)
        for r_ in rows:
            ws.append([r_.get(k, "") for k in keys])
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = min(
                60, 2 + max(len(str(cl.value)) for cl in col))
        ws.freeze_panes = "A2"
    wb.save(path)
    return True


# ---------------------------------------------------------------------- main
def solve_one(path, args, env):
    J = json.load(open(path))
    name = J["name"]
    outdir = os.path.join(args.out, name)
    os.makedirs(outdir, exist_ok=True)
    print(f"{name}: building model ...", flush=True)
    t_build = time.time()
    m, data = build_model(J, env)
    t_build = time.time() - t_build
    m.Params.TimeLimit = args.time_limit
    m.Params.MIPGap = args.mip_gap
    m.Params.Threads = args.threads
    m.Params.MIPFocus = args.mip_focus
    if args.node_file_dir:                         # spill branch-and-bound nodes to disk
        m.Params.NodefileStart = 0.5
        m.Params.NodefileDir = args.node_file_dir
    logfile = os.path.join(outdir, "gurobi.log")
    if os.path.exists(logfile):
        os.remove(logfile)                         # Gurobi appends otherwise
    m.Params.LogFile = logfile
    m.Params.OutputFlag = 1                        # always write gurobi.log
    m.Params.LogToConsole = 1 if args.verbose else 0
    m.update()

    gpu_info = {"gpu_root_LP_bound": "", "gpu_LP_time_s": "", "gpu_used": ""}
    if args.gpu:
        bound, secs, used = gpu_root_lp(m, args.threads)
        gpu_info = {"gpu_root_LP_bound": round(bound, 2), "gpu_LP_time_s": round(secs, 2),
                    "gpu_used": "yes" if used else "no (not a GPU build, PDHG ran on CPU)"}

    if not args.no_warm_start:
        warm_start(J, data)
    m.optimize()

    greedy = J["greedy_baseline"]["Z_Rs"]
    row = {"instance": name, "EVs": J["n_ev"], "stations": J["n_fcs"],
           "guns": J["n_fcs"] * J["params"]["guns_per_fcs"], "vans": J["n_vans"],
           "graph_nodes": 1 + J["n_ev"] + J["n_fcs"],   # depot + EVs + stations
           "binaries": m.NumBinVars, "continuous": m.NumVars - m.NumBinVars,
           "constraints": m.NumConstrs, "build_s": round(t_build, 2)}
    status = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time limit",
              GRB.INFEASIBLE: "infeasible"}.get(m.Status, f"status {m.Status}")
    if m.SolCount == 0:
        row.update({"status": status, "Z_Rs": "", "best_bound_Rs": "", "gap_%": "",
                    "runtime_s": round(m.Runtime, 2), "nodes_explored": int(m.NodeCount),
                    "greedy_Z_Rs": greedy, "improvement_vs_greedy_%": "",
                    "valid": "", **gpu_info})
        print(f"{name}: {status}, no solution")
        m.dispose()
        return row, None, J

    row.update({"status": status, "gurobi_obj_Rs": round(m.ObjVal, 2),
                "best_bound_Rs": round(m.ObjBound, 2), "runtime_s": round(m.Runtime, 2),
                "nodes_explored": int(m.NodeCount), "greedy_Z_Rs": greedy, **gpu_info})
    plan = extract(J, data)
    m.dispose()                                    # free solver memory before the next instance
    del data
    gc.collect()
    return finish(J, plan, row, outdir, args)


def repair_times(J, plan):
    """Keep the MILP's decisions (resource, gun order, van routes) and recompute every
    start time as early as the constraints allow. This removes the tiny violations
    (about 1e-4 min) that big-M constraints leave within Gurobi's tolerances; since
    Z grows with every completion time, it never makes Z worse."""
    ev = J["ev"]
    r, tF, tM = ev["r_min"], ev["tauF_min"], ev["tauM_min"]
    tEF, tN = J["t_ev_fcs_min"], J["t_nodes_min"]
    n = J["n_ev"]
    S, C = list(plan["S"]), list(plan["C"])
    by_gun = {}
    for i in range(n):
        res = plan["resource"][i]
        if res.startswith("FCS"):
            by_gun.setdefault(res, []).append(i)
    for res, evs in by_gun.items():
        f = int(res[3:res.index("-")]) - 1
        free = 0.0
        for i in sorted(evs, key=lambda q: plan["S"][q]):
            S[i] = max(r[i] + tEF[i][f], free)
            C[i] = S[i] + tF[i]
            free = C[i]
    for route in plan["van_routes"]:
        t, pos = 0.0, 0
        for k in route[1:-1]:
            i = k - 1
            S[i] = max(r[i], t + tN[pos][k])
            C[i] = S[i] + tM[i]
            t, pos = C[i], k
    return {**plan, "S": S, "C": C}


def finish(J, raw_plan, row, outdir, args):
    """Repair times, validate, build tables and write all per-instance files."""
    name = J["name"]
    plan = repair_times(J, raw_plan)
    err, Zchk, parts = validate(J, plan)
    plan["Z_Rs"] = Zchk
    greedy = row["greedy_Z_Rs"]
    bound = row["best_bound_Rs"]
    ev_rows, van_rows, gun_rows = tables(J, plan, parts)
    row = {**{k: v for k, v in row.items() if k not in ("Z_Rs", "gap_%", "valid",
                                                          "improvement_vs_greedy_%")},
           "Z_Rs": round(Zchk, 2),
           "gap_%": round(100 * (Zchk - bound) / Zchk, 3),
           "improvement_vs_greedy_%": round(100 * (greedy - Zchk) / greedy, 2),
           "valid": "yes" if not err else "NO: " + "; ".join(err[:3])}
    order = ["instance", "EVs", "stations", "guns", "vans", "graph_nodes",
             "binaries", "continuous",
             "constraints", "build_s", "status", "Z_Rs", "best_bound_Rs", "gap_%",
             "gurobi_obj_Rs", "runtime_s", "nodes_explored", "greedy_Z_Rs",
             "improvement_vs_greedy_%", "valid"]
    if args.gpu or row.get("gpu_used"):
        order += ["gpu_root_LP_bound", "gpu_LP_time_s", "gpu_used"]
    row = {k: row.get(k, "") for k in order}
    status = row["status"]
    cost = {"instance": name, "Z1_EV_drive": round(parts["Z1"], 2),
            "Z2_van_drive": round(parts["Z2"], 2), "Z3_dispatch": round(parts["Z3"], 2),
            "Z4_energy": round(parts["Z4"], 2), "Z5_inconvenience": round(parts["Z5"], 2),
            "operating_Z1_to_Z4": round(sum(parts[k] for k in ("Z1", "Z2", "Z3", "Z4")), 2),
            "total_Z": round(Zchk, 2),
            "EVs_at_FCS": sum(1 for e in ev_rows if e["type"] == "FCS"),
            "EVs_by_van": sum(1 for e in ev_rows if e["type"] == "MCS"),
            "vans_used": len(van_rows),
            "mean_time_to_charged_min": round(sum(e["time_to_charged_min"] for e in ev_rows)
                                              / len(ev_rows), 1),
            "max_time_to_charged_min": max(e["time_to_charged_min"] for e in ev_rows),
            "van_km": round(sum(v["km"] for v in van_rows), 2)}
    write_csv(os.path.join(outdir, "evs.csv"), ev_rows)
    write_csv(os.path.join(outdir, "vans.csv"), van_rows)
    write_csv(os.path.join(outdir, "guns.csv"), gun_rows)
    with open(os.path.join(outdir, "solution.json"), "w") as fh:
        json.dump({"summary": row, "costs": cost, **plan}, fh, indent=1)
    print(f"{name}: {status}, Z = {Zchk:,.2f} Rs, gap {row['gap_%']:.2f} %, "
          f"{row['runtime_s']:.1f} s, greedy {greedy:,.2f}, valid: {row['valid']}")
    if args.details:
        print_table(f"[{name}] EV assignment", ev_rows)
        print_table(f"[{name}] Van routes", van_rows)
        print_table(f"[{name}] Gun schedules", gun_rows)
    return row, (cost, ev_rows, van_rows, gun_rows), J


def reuse_one(path, args):
    """Rebuild the result tables from a previous run's solution.json (no solving)."""
    J = json.load(open(path))
    outdir = os.path.join(args.out, J["name"])
    sol = json.load(open(os.path.join(outdir, "solution.json")))
    row = dict(sol["summary"])
    if "gurobi_obj_Rs" not in row:              # files from the first version of this script
        row["gurobi_obj_Rs"] = row.get("Z_Rs")
    # results from older versions of this script: rename / add the node columns
    for old in ("nodes", "bnb_nodes"):
        if old in row and "nodes_explored" not in row:
            row["nodes_explored"] = row.pop(old)
    row.setdefault("graph_nodes", 1 + J["n_ev"] + J["n_fcs"])
    raw = {k: sol[k] for k in ("resource", "S", "C", "van_routes", "route_van")}
    return finish(J, raw, row, outdir, args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("instances", help="folder with <instance>/instance.json")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
    ap.add_argument("--only", nargs="*", help="instance names, e.g. C101_n10")
    ap.add_argument("--sizes", nargs="*", type=int, help="only these n (e.g. 10 25)")
    ap.add_argument("--time-limit", type=float, default=300.0, help="seconds per instance")
    ap.add_argument("--mip-gap", type=float, default=1e-4)
    ap.add_argument("--threads", type=int, default=0, help="0 = all cores")
    ap.add_argument("--gpu", action="store_true", help="also solve the root LP with PDHG on the GPU")
    ap.add_argument("--no-warm-start", action="store_true", help="do not start from the greedy plan")
    ap.add_argument("--details", action="store_true", help="print per-EV / van / gun tables")
    ap.add_argument("--verbose", action="store_true", help="show the Gurobi log")
    ap.add_argument("--reuse", action="store_true",
                    help="do not solve; rebuild tables from solution.json files in --out")
    ap.add_argument("--resume", action="store_true",
                    help="skip instances that already have a solution.json in --out")
    ap.add_argument("--mip-focus", type=int, default=0, help="Gurobi MIPFocus (1 = find good solutions)")
    ap.add_argument("--node-file-dir", default="", help="folder for Gurobi node files (saves RAM)")
    args = ap.parse_args()

    def size_of(q):
        return int(os.path.basename(os.path.dirname(q)).split("_n")[1])
    paths = sorted(glob.glob(os.path.join(args.instances, "*", "instance.json")),
                   key=lambda q: (size_of(q), os.path.basename(os.path.dirname(q))))
    if args.only:
        paths = [q for q in paths if os.path.basename(os.path.dirname(q)) in args.only]
    if args.sizes:
        paths = [q for q in paths if size_of(q) in args.sizes]
    os.makedirs(args.out, exist_ok=True)
    print(f"{len(paths)} instance(s) to process", flush=True)

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    summary, costs, sheets = [], [], []
    for path in paths:
        name = os.path.basename(os.path.dirname(path))
        done = os.path.exists(os.path.join(args.out, name, "solution.json"))
        if args.reuse and not done:
            print(f"{name}: no solution.json yet, skipped", flush=True)
            continue
        t0 = time.time()
        try:
            if args.reuse or (args.resume and done):
                row, detail, J = reuse_one(path, args)
            else:
                row, detail, J = solve_one(path, args, env)
        except (gp.GurobiError, MemoryError) as e:    # keep going with the next instance
            gc.collect()
            row, detail = {"instance": name, "EVs": size_of(path), "status": f"error: {e}"}, None
            print(f"{name}: ERROR {e}", flush=True)
        print(f"   ({time.time() - t0:.0f} s wall clock)", flush=True)
        summary.append(row)
        if detail:
            cost, ev_rows, van_rows, gun_rows = detail
            costs.append(cost)
            sheets += [(f"{name}_EVs", ev_rows), (f"{name}_vans", van_rows),
                       (f"{name}_guns", gun_rows)]
        # write the overall tables after every instance, so a long run keeps partial results
        write_csv(os.path.join(args.out, "summary.csv"), summary)
        write_csv(os.path.join(args.out, "costs.csv"), costs)

    show = ["instance", "EVs", "vans", "binaries", "status", "Z_Rs", "best_bound_Rs",
            "gap_%", "runtime_s", "greedy_Z_Rs", "improvement_vs_greedy_%", "valid"]
    if args.gpu:
        show += ["gpu_root_LP_bound", "gpu_used"]
    print_table("SUMMARY", summary, show)
    print_table("COST BREAKDOWN (Rs)", costs)
    write_csv(os.path.join(args.out, "summary.csv"), summary)
    write_csv(os.path.join(args.out, "costs.csv"), costs)
    with open(os.path.join(args.out, "report.md"), "w") as fh:
        fh.write("# MILP results (main_formulation, Gurobi)\n\n## Summary\n\n" + md_table(summary)
                 + "\n## Cost breakdown (Rs)\n\n" + md_table(costs))
        for name, rows in sheets:
            fh.write(f"\n## {name}\n\n" + md_table(rows))
    ok = write_xlsx(os.path.join(args.out, "results.xlsx"),
                    [("summary", summary), ("costs", costs), *sheets])
    print(f"\nResults written to {os.path.abspath(args.out)}"
          f" (summary.csv, costs.csv, report.md{', results.xlsx' if ok else ''}, per-instance folders)")


if __name__ == "__main__":
    main()
