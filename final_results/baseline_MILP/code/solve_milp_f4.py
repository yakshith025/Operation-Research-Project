"""Solve formulation 4 (formulation 3 + use segment: personal / commercial) with Gurobi on the
formulation-4 dataset (../../data), validate every solution, write the result tables and, at the end,
the segment x income analysis (../../common/analyze_f4.py).

Formulation 4 (../../formulation_4.pdf) has the variables and constraints (1)-(23) of formulation 3.
The use segment s_i and the income index theta_i of EV i are data; they set the per-EV coefficients
mu_i, c^D_i, c^Q_i, c^C_i, c^V_i and W_i (../../common/f4_params.py). The model is therefore a linear
MILP of exactly the same size as in formulation 3.

An instance = one file of the dataset + a size n (its first n EVs, as in Solomon).
Default instances: the 56 Solomon files at n = 10, 20, ..., 100 (560 instances, as the formulation-3 run).

Usage examples (from baseline_MILP/code)
  python solve_milp_f4.py --workers 8 --threads 4                  # 560 Solomon instances, 8 at a time
  python solve_milp_f4.py --file-size 100 200                      # also the 60 Gehring-Homberger files
  python solve_milp_f4.py --only EV-R101 --sizes 10 --details
  python solve_milp_f4.py --resume ...                             # continue an interrupted run
  python solve_milp_f4.py --reuse                                  # re-score the plans in plan.csv
  python solve_milp_f4.py --set tariff_Rs_per_kWh.p_M=35 --variant E1_pM_35 --out <folder>
                                                                   # one experiment variant (what the
                                                                   # experiment folders call; no data copy)
Every instance goes to Gurobi with the complete model; nothing is skipped or decided before the solver.
Gurobi's verdict (optimal / time_limit / infeasible / ...) is what is reported. The run never stops
because of one instance: any error becomes that run's status (message column, traceback in the logs)
and the next instance starts. With --workers W, W instances are solved at the same time, each in its
own process. Results are appended to the tables after every instance.

Outputs (all in --out, default ../results; schema in ../../common/results_io_f4.py, shared with RL):
  runs.csv     one row per run (run_id, method, seed, instance, n): variant, status, valid, Z1-Z6,
               service numbers overall and per segment, times, and for the MILP its bound, gap, nodes and
               model size. Every instance gives two runs: method MILP (run_id --run-id, default
               milp_<TL>s) and method greedy (run_id greedy, the plan that warm-starts the MILP).
  plan.csv     one row per decision: type ev (segment, theta, resource, position, start / end, waits,
               money) and type van (stops, km, energy, return time), keyed like runs.csv
  meta.json    settings of each run_id (incl. the parameters and the --set overrides)
  logs.zip     Gurobi log, console output and error traceback of every instance
  analysis/    segment x income insights (analyze_f4.py), unless --no-analysis
"""
import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
F4_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(F4_DIR, "common"))
import f4_params as fp             # noqa: E402  (parameters, --set overrides, per-EV coefficients)
import f4_data as fd               # noqa: E402  (reads data/*.csv)
import results_io_f4 as rio        # noqa: E402  (output schema: runs.csv + plan.csv)
from check_plan_f4 import validate  # noqa: E402

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:                 # reported per instance, the run still finishes
    gp, GRB = None, None


# ====================================================================== instance
def load_instance(name, n, ds, greedy=True):
    """Instance dict of the first n EVs of file `name` (f4_data.Dataset.instance) plus, unless
    greedy=False, the greedy plan that warm-starts the MILP."""
    J = ds.instance(name, n)
    J["greedy_baseline"], J["greedy_s"] = None, ""
    if greedy:
        t0 = time.time()
        J["greedy_baseline"] = greedy_f4(J)
        J["greedy_s"] = round(time.time() - t0, 3)
    return J


# ====================================================================== greedy plan
def greedy_f4(J):
    """Feasible plan for formulation 4 (warm start and baseline), or None.
    EVs are placed one by one (stranded EVs first, then by request time, ties: higher queue cost c^Q
    first); each takes the option with the lowest increase of Z ('cost' rule). If that fails, the
    earliest-finish rule is tried."""
    for rule in ("cost", "finish"):
        plan = _greedy(J, rule)
        if plan is not None:
            err, Z, _, _ = validate(J, plan)
            if not err:
                plan["Z_Rs"] = round(Z, 2)
                plan["rule"] = rule
                return plan
    return None


def _greedy(J, rule):
    p, ev = J["params"], J["ev"]
    c, tar = p["costs_Rs"], p["tariff_Rs_per_kWh"]
    n, F, M, T = J["n_ev"], J["n_fcs"], J["n_vans"], p["T_min"]
    r, E, R, W = ev["r_min"], ev["E_kWh"], ev["R_km"], ev["W"]
    tF, tM = ev["tauF_min"], ev["tauM_min"]
    dEF, tEF, dN, tN = J["d_ev_fcs_km"], J["t_ev_fcs_min"], J["d_nodes_km"], J["t_nodes_min"]
    reach = [[f for f in range(F) if dEF[i][f] <= R[i]] for i in range(n)]
    gun_free = {(f, g): 0.0 for f in range(F) for g in range(p["guns"][f])}
    van = [{"pos": 0, "free": 0.0, "batt": p["B_kWh"], "cnt": 0, "stops": []} for _ in range(M)]
    S, C, res = [0.0] * n, [0.0] * n, [""] * n
    for i in sorted(range(n), key=lambda q: (len(reach[q]) > 0, r[q], -ev["cQ"][q])):
        best = None
        for f in reach[i]:
            arrive = r[i] + tEF[i][f]
            for g in range(p["guns"][f]):
                s = max(arrive, gun_free[(f, g)])
                if s - arrive > W[i] + 1e-9 or s + tF[i] > T + 1e-9:
                    continue
                cost = (ev["mu"][i] * (c["c_EV"] * dEF[i][f] + tar["p_F"] * E[i]) + c["pi_F"] * E[i]
                        + ev["cD"][i] * tEF[i][f] + ev["cQ"][i] * (s - arrive) + ev["cC"][i] * tF[i])
                key = cost if rule == "cost" else s + tF[i]
                if best is None or key < best[0]:
                    best = (key, ("gun", f, g), s, s + tF[i])
        node = i + 1
        for m, v in enumerate(van):
            s = max(r[i], v["free"] + tN[v["pos"]][node])
            need = E[i] / p["eta"] + p["beta_kWh_per_km"] * (dN[v["pos"]][node] + dN[node][0])
            if v["cnt"] >= p["Q"] or need > v["batt"] + 1e-9 or s + tM[i] + tN[node][0] > T + 1e-9:
                continue
            extra_km = dN[v["pos"]][node] + dN[node][0] - dN[v["pos"]][0]
            cost = (ev["mu"][i] * tar["p_M"] * E[i] + c["pi_M"] * E[i] / p["eta"] + c["c_M"] * extra_km
                    + (c["c_F"] if v["cnt"] == 0 else 0.0)
                    + ev["cV"][i] * (s - r[i] + tM[i]))
            key = cost if rule == "cost" else s + tM[i]
            if best is None or key < best[0]:
                best = (key, ("van", m), s, s + tM[i])
        if best is None:
            return None
        _, ch, S[i], C[i] = best
        if ch[0] == "gun":
            gun_free[(ch[1], ch[2])] = C[i]
            res[i] = f"FCS{ch[1] + 1}-gun{ch[2] + 1}"
        else:
            v = van[ch[1]]
            v["batt"] -= E[i] / p["eta"] + p["beta_kWh_per_km"] * dN[v["pos"]][node]
            v["pos"], v["free"], v["cnt"] = node, C[i], v["cnt"] + 1
            v["stops"].append(node)
    # vans in use get the numbers 1, 2, ... (van symmetry constraint)
    routes, route_van, k = [], [], 0
    for v in van:
        if v["stops"]:
            k += 1
            for node in v["stops"]:
                res[node - 1] = f"van{k}"
            routes.append([0, *v["stops"], 0])
            route_van.append(k)
    return {"resource": res, "S": S, "C": C, "van_routes": routes, "route_van": route_van}


# ====================================================================== model
def build_model(J, env):
    """Formulation 4, equations (1)-(23): the structure of formulation 3, coefficients set by s_i and theta_i."""
    p, ev = J["params"], J["ev"]
    n, F, Mn = J["n_ev"], J["n_fcs"], J["n_vans"]
    T, c, tar = p["T_min"], p["costs_Rs"], p["tariff_Rs_per_kWh"]
    E, r, R = ev["E_kWh"], ev["r_min"], ev["R_km"]
    mu, cD, cQ, cV, cC, W = (ev[k] for k in ("mu", "cD", "cQ", "cV", "cC", "W"))
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

    guns = [(f, k) for f in range(F) for k in range(p["guns"][f])]
    Fi = {i: [f for f in range(F) if dEF[i - 1][f] <= R[i - 1]] for i in I}      # F_i
    Gi = {i: [g for g in guns if g[0] in Fi[i]] for i in I}                      # G_i
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
    qF = m.addVars(I, lb=0.0, name="qF")
    qM = m.addVars(I, lb=0.0, name="qM")

    Y = {(i, f): gp.quicksum(y[i, g] for g in Gi[i] if g[0] == f) for i in I for f in Fi[i]}
    yi = {i: gp.quicksum(y[i, g] for g in Gi[i]) for i in I}
    zi = {i: gp.quicksum(z[i, k] for k in Mv) for i in I}
    drive = {i: gp.quicksum(tEF[i - 1][f] * Y[i, f] for f in Fi[i]) for i in I}
    Cexp = {i: S[i] + tauF[i - 1] * yi[i] + tauM[i - 1] * zi[i] for i in I}

    # (1) objective: operator cost Z2-Z4 + perceived customer cost Z1, Z5, Z6 (mu_i, c^a_i from s_i, theta_i)
    Z1 = gp.quicksum(mu[i - 1] * c["c_EV"] * dEF[i - 1][f] * Y[i, f] for i in I for f in Fi[i])
    Z2 = c["c_M"] * gp.quicksum(d(i, j) * x[i, j, k] for (i, j) in A for k in Mv)
    Z3 = c["c_F"] * u.sum()
    Z4 = (c["pi_F"] * gp.quicksum(E[i - 1] * yi[i] for i in I)
          + c["pi_M"] * gp.quicksum(E[i - 1] / eta * zi[i] for i in I))
    Z5 = gp.quicksum(mu[i - 1] * E[i - 1] * (tar["p_F"] * yi[i] + tar["p_M"] * zi[i]) for i in I)
    Z6 = gp.quicksum(cD[i - 1] * drive[i] + cQ[i - 1] * qF[i] + cC[i - 1] * tauF[i - 1] * yi[i]
                     + cV[i - 1] * (qM[i] + tauM[i - 1] * zi[i]) for i in I)
    m.setObjective(Z1 + Z2 + Z3 + Z4 + Z5 + Z6, GRB.MINIMIZE)

    # (2) assignment
    m.addConstrs((yi[i] + zi[i] == 1 for i in I), "assign")
    # (3) FCS arrival
    m.addConstrs((S[i] >= r[i - 1] + drive[i] for i in I), "fcs_arrival")
    # (4) waiting time split, (5) FCS queue <= W_i (only FCS users), (6) van wait (only MCS users)
    m.addConstrs((qF[i] + qM[i] == S[i] - r[i - 1] - drive[i] for i in I), "wait")
    m.addConstrs((qF[i] <= W[i - 1] * yi[i] for i in I), "queue_limit")
    m.addConstrs((qM[i] <= T * zi[i] for i in I), "van_wait")
    # (7)-(8) disjunctive gun sequencing, M^F_i = T + tauF_i
    for g in guns:
        for (i, j) in Pg[g]:
            MFi, MFj = T + tauF[i - 1], T + tauF[j - 1]
            m.addConstr(S[j] >= S[i] + tauF[i - 1] - MFi * (3 - y[i, g] - y[j, g] - o[i, j]),
                        f"disj1[{i},{j},{g[0]},{g[1]}]")
            m.addConstr(S[i] >= S[j] + tauF[j - 1] - MFj * (2 - y[i, g] - y[j, g] + o[i, j]),
                        f"disj2[{i},{j},{g[0]},{g[1]}]")
    # (9) gun load (valid inequality), only for guns in G+
    for g in Gplus:
        first = min(r[i - 1] + tEF[i - 1][g[0]] for i in users[g])
        m.addConstr(gp.quicksum(tauF[i - 1] * y[i, g] for i in users[g]) <= T - first,
                    f"gunload[{g[0]},{g[1]}]")
    # (10) symmetry between the guns of a station (consecutive guns)
    for f in range(F):
        for k in range(p["guns"][f] - 1):
            m.addConstr(gp.quicksum(y[i, (f, k)] for i in users[(f, k)])
                        >= gp.quicksum(y[i, (f, k + 1)] for i in users[(f, k + 1)]),
                        f"gunsym[{f},{k}]")
    # (11)-(14) routing
    m.addConstrs((gp.quicksum(x[0, j, k] for j in out_arcs[0]) == u[k] for k in Mv), "depot_out")
    m.addConstrs((gp.quicksum(x[i, END, k] for i in in_arcs[END]) == u[k] for k in Mv), "depot_in")
    m.addConstrs((gp.quicksum(x[i, j, k] for i in in_arcs[j]) == z[j, k]
                  for j in I for k in Mv), "flow_in")
    m.addConstrs((gp.quicksum(x[j, l_, k] for l_ in out_arcs[j]) == z[j, k]
                  for j in I for k in Mv), "flow_out")
    # (15) capacity, (16) battery
    m.addConstrs((z.sum("*", k) <= Q * u[k] for k in Mv), "van_cap")
    m.addConstrs((gp.quicksum(E[i - 1] / eta * z[i, k] for i in I)
                  + beta * gp.quicksum(d(i, j) * x[i, j, k] for (i, j) in A) <= B * u[k]
                  for k in Mv), "battery")
    # (17) first stop, (18) sequencing, (19) return by T
    m.addConstrs((S[j] >= t(0, j) * x[0, j, k] for j in I for k in Mv), "depart")
    ev_arcs = [(i, j) for (i, j) in A if i != 0 and j != END]
    for k in Mv:
        for (i, j) in ev_arcs:
            MM = T + tauM[i - 1] + t(i, j)
            m.addConstr(S[j] >= S[i] + tauM[i - 1] + t(i, j) - MM * (1 - x[i, j, k]),
                        f"seq[{i},{j},{k}]")
        for i in I:
            m.addConstr(S[i] + tauM[i - 1] + t(i, END)
                        <= T + (tauM[i - 1] + t(i, END)) * (1 - x[i, END, k]), f"return[{i},{k}]")
    # (20) release, (21) horizon
    m.addConstrs((S[i] >= r[i - 1] for i in I), "release")
    m.addConstrs((Cexp[i] <= T for i in I), "horizon")
    # (22) van symmetry
    m.addConstrs((u[k] >= u[k + 1] for k in range(Mn - 1)), "vansym")

    data = dict(I=I, END=END, guns=guns, Gi=Gi, Fi=Fi, Mv=Mv, A=A, pairs=pairs,
                y=y, z=z, x=x, u=u, o=o, S=S, qF=qF, qM=qM, Cexp=Cexp)
    return m, data


# ====================================================================== warm start
def warm_start(J, data):
    """Give Gurobi the greedy plan as MIP start (Gurobi checks it; a bad start is ignored)."""
    g = J["greedy_baseline"]
    I, END = data["I"], data["END"]
    y, z, x, u, o, S, qF, qM = (data[k] for k in "y z x u o S qF qM".split())
    for v in [*y.values(), *z.values(), *x.values(), *u.values(), *o.values()]:
        v.Start = 0
    count = {}
    for res in g["resource"]:
        if res.startswith("FCS"):
            key = (int(res[3:res.index("-")]) - 1, int(res.split("gun")[1]) - 1)
            count[key] = count.get(key, 0) + 1
    relabel = {}                    # gun 1 of a station carries the most EVs (gun symmetry)
    for f in range(J["n_fcs"]):
        ks = sorted(range(J["params"]["guns"][f]), key=lambda k: -count.get((f, k), 0))
        relabel.update({(f, old): (f, new) for new, old in enumerate(ks)})
    gun_of = {}
    for i in I:
        res = g["resource"][i - 1]
        Si = g["S"][i - 1]
        S[i].Start = Si
        if res.startswith("FCS"):
            f = int(res[3:res.index("-")]) - 1
            gun = relabel[(f, int(res.split("gun")[1]) - 1)]
            y[i, gun].Start = 1
            gun_of[i] = gun
            qF[i].Start = max(0.0, Si - J["ev"]["r_min"][i - 1] - J["t_ev_fcs_min"][i - 1][f])
            qM[i].Start = 0.0
        else:
            qF[i].Start = 0.0
            qM[i].Start = max(0.0, Si - J["ev"]["r_min"][i - 1])
    for route, k1 in zip(g["van_routes"], g["route_van"]):
        k = k1 - 1
        u[k].Start = 1
        nodes = [0, *route[1:-1], END]
        for a, b in zip(nodes, nodes[1:]):
            x[a, b, k].Start = 1
        for node in route[1:-1]:
            z[node, k].Start = 1
    for (i, j) in data["pairs"]:
        if i in gun_of and gun_of.get(j) == gun_of[i]:
            o[i, j].Start = 1 if g["S"][i - 1] <= g["S"][j - 1] else 0


# ====================================================================== plan
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
        route, node, guard = [0], succ[0], 0
        while node != END and guard <= J["n_ev"]:
            route.append(node)
            node = succ.get(node, END)
            guard += 1
        route.append(0)
        routes.append(route)
        route_van.append(k + 1)
    return {"resource": res, "S": Sv, "C": Cv, "van_routes": routes, "route_van": route_van}


def repair_times(J, plan):
    """Keep the MILP's decisions (resource, gun order, van routes) and recompute every start
    time as early as the constraints allow. This removes the tiny violations big-M constraints
    leave within Gurobi's tolerances. Every waiting term of Z grows with the start time and a
    queue only gets shorter, so Z never gets worse and every constraint stays satisfied."""
    ev = J["ev"]
    r, tF, tM = ev["r_min"], ev["tauF_min"], ev["tauM_min"]
    tEF, tN = J["t_ev_fcs_min"], J["t_nodes_min"]
    S, C = list(plan["S"]), list(plan["C"])
    by_gun = {}
    for i in range(J["n_ev"]):
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


# ====================================================================== console tables
def all_keys(rows):
    keys = []
    for r_ in rows:
        keys += [k for k in r_ if k not in keys]
    return keys


def print_table(title, rows, keys=None):
    if not rows:
        return
    keys = [k for k in (keys or all_keys(rows)) if any(k in r_ for r_ in rows)]
    width = {k: max(len(k), *(len(str(r_.get(k, ""))) for r_ in rows)) for k in keys}
    print(f"\n{title}")
    print("  ".join(k.ljust(width[k]) for k in keys))
    print("  ".join("-" * width[k] for k in keys))
    for r_ in rows:
        print("  ".join(str(r_.get(k, "")).ljust(width[k]) for k in keys))


# ====================================================================== solve
STATUS_NAMES = {"OPTIMAL": "optimal", "TIME_LIMIT": "time_limit", "INFEASIBLE": "infeasible",
                "INF_OR_UNBD": "infeasible", "INTERRUPTED": "interrupted", "MEM_LIMIT": "memory_limit",
                "NODE_LIMIT": "node_limit", "SOLUTION_LIMIT": "solution_limit",
                "NUMERIC": "numeric_trouble", "WORK_LIMIT": "work_limit"}


VARIANT = {"id": "baseline"}       # set in main() from --variant


def run_keys(J, run_id, method, seed):
    return {"run_id": run_id, "method": method, "seed": seed, "instance": J["file"],
            "source": J["source"], "class": J["family"], "n": J["n_ev"], "variant": VARIANT["id"]}


def greedy_run(J):
    """The greedy plan (warm start of the MILP) as a run of its own: method greedy, run_id greedy."""
    keys = run_keys(J, "greedy", "greedy", 0)
    g = J["greedy_baseline"]
    if g is None:
        return rio.run_row(keys, "no_plan", solve_s=J["greedy_s"],
                           message="no feasible plan by the cost or the earliest-finish rule"), []
    plan = {k: g[k] for k in ("resource", "S", "C", "van_routes", "route_van")}
    metrics, rows, _ = rio.evaluate(J, plan)
    return (rio.run_row(keys, "plan", metrics, solve_s=J["greedy_s"], message=f"rule: {g['rule']}"),
            rio.with_key(keys, rows))


def solve_one(J, args, env):
    """Build and solve the complete model of instance J. Returns (solver fields, plan or None):
    the plan has its start times repaired (repair_times) and is scored later by results_io."""
    name = J["name"]
    if gp is None:
        raise RuntimeError("gurobipy is not installed (pip install gurobipy)")
    print(f"{name}: building model ...", flush=True)
    t_build = time.time()
    m, data = build_model(J, env)
    t_build = time.time() - t_build
    m.Params.TimeLimit = args.time_limit
    m.Params.MIPGap = args.mip_gap
    m.Params.Threads = args.threads
    m.Params.MIPFocus = args.mip_focus
    m.Params.Seed = args.seed
    if args.mem_limit_gb > 0:                      # Gurobi stops cleanly instead of running out of RAM
        m.Params.SoftMemLimit = args.mem_limit_gb
    if args.node_file_dir:
        m.Params.NodefileStart = 0.5
        m.Params.NodefileDir = args.node_file_dir
    logs = rio.paths(args.out)["logs"]
    os.makedirs(logs, exist_ok=True)
    logfile = os.path.join(logs, rio.log_stem(args.run_id, J["file"], J["n_ev"]) + ".gurobi.log")
    if os.path.exists(logfile):
        os.remove(logfile)
    m.Params.LogFile = logfile
    m.Params.OutputFlag = 1
    m.Params.LogToConsole = 1 if args.verbose else 0
    m.update()

    info = {}

    g = J["greedy_baseline"]
    warm = "none"
    if not args.no_warm_start and g:
        warm_start(J, data)
        warm = f"greedy ({g['rule']} rule)"
    m.optimize()

    codes = {getattr(GRB.Status, k): v for k, v in STATUS_NAMES.items() if hasattr(GRB.Status, k)}
    status = codes.get(m.Status, f"status_{m.Status}")
    info.update({"binaries": m.NumBinVars, "continuous": m.NumVars - m.NumBinVars,
                 "constraints": m.NumConstrs, "build_s": round(t_build, 2),
                 "solve_s": round(m.Runtime, 2), "nodes": int(m.NodeCount), "warm_start": warm})
    if m.SolCount == 0:
        info["status"] = status if status == "infeasible" else status + "_no_plan"
        m.dispose()
        return info, None
    info.update({"status": status, "solver_obj": round(m.ObjVal, 2), "best_bound": round(m.ObjBound, 2)})
    plan = extract(J, data)
    m.dispose()
    del data
    gc.collect()
    return info, repair_times(J, plan)


def milp_run(J, info, plan, args):
    """runs.csv row and plan.csv rows of the MILP (scored by the shared checker)."""
    keys = run_keys(J, args.run_id, "MILP", args.seed)
    info = dict(info)
    status = info.pop("status")
    if plan is None:
        return rio.run_row(keys, status, **info), []
    metrics, rows, _ = rio.evaluate(J, plan)
    Z, bound = metrics["Z_exact"], info.get("best_bound", "")
    gap = max(0.0, round(100 * (Z - bound) / Z, 3)) if bound not in ("", None) and Z else ""
    return rio.run_row(keys, status, metrics, **info, **{"gap_%": gap}), rio.with_key(keys, rows)


def print_details(J, plan_rows):
    evs = [r for r in plan_rows if r["type"] == "ev"]
    print_table(f"[{J['name']}] EV assignment", evs,
                ["id", "segment", "theta", "stranded", "resource", "seq", "start_min", "queue_min",
                 "van_wait_min", "downtime_min", "pay_Rs", "customer_cost_Rs"])
    print_table(f"[{J['name']}] Vans", [r for r in plan_rows if r["type"] == "van"],
                ["id", "stops", "km", "kWh_delivered", "kWh_drawn", "battery_left_kWh", "return_min"])


def rescore(tasks, args):
    """--reuse: score the plans stored in plan.csv again (every method) for the selected
    instances, e.g. after a change of the checker or of f4_params.json. Nothing is solved."""
    out = args.out
    where = {(base_of(f_), str(n_)): f_ for f_, n_ in tasks}
    targets = [r for r in rio.read_runs(out)
               if (r["instance"], rio.key_of(r)[4]) in where and r.get("valid") in ("yes", "no")]
    keys = {rio.key_of(r) for r in targets}
    rows_by_key = {}
    for r in rio.iter_plan(out, keys):
        rows_by_key.setdefault(rio.key_of(r), []).append(r)
    cache, new_runs, new_plan = {}, [], []
    for r in targets:
        k = rio.key_of(r)
        try:
            if (k[3], k[4]) not in cache:
                cache[(k[3], k[4])] = load_instance(where[(k[3], k[4])], int(k[4]), args.data,
                                                    greedy=False)
            J = cache[(k[3], k[4])]
            metrics, rows, _ = rio.evaluate(J, rio.plan_from_rows(rows_by_key.get(k, []), J["n_ev"]))
        except Exception as e:
            print(f"{k}: not re-scored ({type(e).__name__}: {e})", flush=True)
            continue
        row = {**r, **{c: v for c, v in metrics.items() if c in rio.RUN_COLUMNS}}
        bound, Z = r.get("best_bound", ""), metrics["Z_exact"]
        if bound not in ("", None) and Z:
            row["gap_%"] = max(0.0, round(100 * (Z - float(bound)) / Z, 3))
        new_runs.append(row)
        new_plan += rio.with_key(r, rows)
        print(f"{k[3]}_n{k[4]} {k[0]}: Z = {metrics['Z_total']}, valid: {metrics['valid']}", flush=True)
    rio.replace_runs(out, new_runs, new_plan)
    print(f"{len(new_runs)} run(s) re-scored", flush=True)


# ====================================================================== running instances
DEFAULT_OUT = os.path.join(HERE, "..", "results")


def base_of(ref):
    """File name of an instance reference: 'EV-C101' or a text-file path '.../100/EV-C101.txt'."""
    base = os.path.basename(ref)
    return base[:-4] if base.lower().endswith(".txt") else base


def milp_key(args, path, n):
    return (args.run_id, "MILP", str(args.seed), base_of(path), str(n))


def start_env(max_wait):
    """Gurobi environment. If the licence has no free session (a WLS licence limits concurrent
    sessions), wait and try again instead of failing the instance."""
    t0, delay = time.time(), 10
    while True:
        try:
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.start()
            return env
        except gp.GurobiError as e:
            if time.time() - t0 > max_wait:
                raise
            print(f"Gurobi licence not available yet ({e}); trying again in {delay} s", flush=True)
            time.sleep(delay)
            delay = min(60, 2 * delay)


def error_keys(args, path, n, J=None):
    if J is not None:
        return run_keys(J, args.run_id, "MILP", args.seed)
    base = base_of(path)
    source, cls = args.data.source.get(base, ""), args.data.config.get(base, {}).get("class", "")
    return {"run_id": args.run_id, "method": "MILP", "seed": args.seed, "instance": base,
            "source": source, "class": cls, "n": n, "variant": VARIANT["id"]}


def run_one(path, n, args, env=None):
    """Solve one instance (greedy plan + MILP) and write its runs and plan rows as one part file
    (results_io.write_part). Never raises: any failure becomes the MILP run's status 'error'
    with the message in its row and the traceback in logs/<run>.error.log."""
    stem = rio.log_stem(args.run_id, base_of(path), n)
    logs = rio.paths(args.out)["logs"]
    os.makedirs(logs, exist_ok=True)
    t0 = time.time()
    J, g_row, g_plan, m_plan = None, None, [], []
    try:
        J = load_instance(path, n, args.data)
        try:
            g_row, g_plan = greedy_run(J)
        except Exception as e:
            g_row = rio.run_row(run_keys(J, "greedy", "greedy", 0), "error",
                                message=f"{type(e).__name__}: {e}")
        own_env = env is None and gp is not None
        if own_env:
            env = start_env(args.licence_wait)
        try:
            info, plan = solve_one(J, args, env)
        finally:
            if own_env:
                env.dispose()
        m_row, m_plan = milp_run(J, info, plan, args)
    except Exception as e:
        gc.collect()
        m_row = rio.run_row(error_keys(args, path, n, J), "error", message=f"{type(e).__name__}: {e}")
        print(f"{stem}: ERROR {type(e).__name__}: {e}", flush=True)
        try:
            with open(os.path.join(logs, stem + ".error.log"), "w") as fh:
                fh.write(traceback.format_exc())
        except OSError:
            pass
    m_row["wall_clock_s"] = round(time.time() - t0, 1)
    if J is not None and m_row["Z_total"] != "":
        g = g_row["Z_total"] if g_row and g_row["Z_total"] != "" else "none"
        print(f"{J['name']}: {m_row['status']}, Z = {m_row['Z_total']:,.2f} Rs, gap {m_row['gap_%']} %, "
              f"{m_row['solve_s']} s, greedy Z {g}, vans for free-choice EVs: "
              f"{m_row['free_choice_by_van']}/{m_row['free_choice_EVs']}, valid: {m_row['valid']}", flush=True)
        if args.details:
            print_details(J, m_plan)
    try:
        rio.write_part(args.out, [m_row] + ([g_row] if g_row else []), m_plan + g_plan)
    except OSError as e:
        print(f"{stem}: results not written ({e})", flush=True)
    return m_row


def collect_tasks(args):
    """(file name, n) pairs: every file of the formulation-4 dataset that passes the filters, at every
    requested size. Default sizes: 10, 20, ..., up to the number of EVs in the file."""
    files = args.data.names()
    tasks, too_big = [], []
    for f in files:
        n_file = args.data.n_evs(f)
        if args.only and f not in args.only:
            continue
        if args.families and args.data.family(f).upper() not in [x.upper() for x in args.families]:
            continue
        if args.file_size and n_file not in args.file_size:
            continue
        sizes = args.sizes or list(range(10, n_file + 1, 10))
        for n in sizes:
            (tasks if 1 <= n <= n_file else too_big).append((f, n))
    if too_big:
        f, n = too_big[0]
        print(f"{len(too_big)} requested size(s) are larger than their file (it has fewer EVs) and do "
              f"not exist, e.g. {base_of(f)}_n{n}", flush=True)
    tasks.sort(key=lambda t_: (t_[1], base_of(t_[0])))      # small instances first
    return files, tasks


def child_cmd(path, n, args):
    """Command that solves one instance in its own process (used by --workers > 1)."""
    cmd = [sys.executable, os.path.abspath(__file__), args.data.path,
           "--one", base_of(path), str(n), "--out", os.path.abspath(args.out),
           "--params", os.path.abspath(args.params), "--variant", VARIANT["id"],
           "--time-limit", str(args.time_limit),
           "--mip-gap", str(args.mip_gap), "--threads", str(args.threads),
           "--mip-focus", str(args.mip_focus), "--mem-limit-gb", str(args.mem_limit_gb),
           "--licence-wait", str(args.licence_wait), "--run-id", args.run_id, "--seed", str(args.seed)]
    if args.no_warm_start:
        cmd.append("--no-warm-start")
    for s in args.set or []:
        cmd += ["--set", s]
    if args.details:
        cmd.append("--details")
    if args.node_file_dir:
        cmd += ["--node-file-dir", args.node_file_dir]
    return cmd


def tail(path, k=5):
    try:
        with open(path, errors="replace") as fh:
            return " | ".join(l_.strip() for l_ in fh.readlines()[-k:] if l_.strip())
    except OSError:
        return ""


def short(row):
    text = str(row.get("status", ""))
    if row.get("Z_total", "") != "":
        text += f", Z = {row['Z_total']} Rs, gap {row.get('gap_%', '')} %, valid: {row.get('valid', '')}"
    elif row.get("message"):
        text += f" ({row['message']})"
    return text


# ====================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", nargs="?", default=fd.DEFAULT_DATA, metavar="DATA",
                    help="formulation-4 dataset folder (default ../../data)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output folder (default ../results)")
    ap.add_argument("--params", default=fp.DEFAULT_PARAMS, help="customer parameters (../../common/f4_params.json)")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="override a parameter (dotted key of f4_params.json) or a config column "
                         "(config.n_vans, config.van_max_evs, config.van_battery_kWh, ...); repeatable")
    ap.add_argument("--variant", default="baseline", help="name of the experiment variant (column variant)")
    ap.add_argument("--run-id", default=None,
                    help="name of this MILP run in runs.csv (default milp_<time limit>s)")
    ap.add_argument("--seed", type=int, default=0, help="Gurobi Seed (column seed of runs.csv)")
    ap.add_argument("--only", nargs="*", help="file names, e.g. EV-R101 EV-C1_2_1")
    ap.add_argument("--families", nargs="*", help="only these families, e.g. C1 R2 RC1")
    ap.add_argument("--file-size", nargs="*", type=int, default=[100],
                    help="only files with this many EVs: 100 = Solomon (default), 200 = Gehring-Homberger")
    ap.add_argument("--sizes", nargs="*", type=int,
                    help="number of EVs (first n of each file); default 10, 20, ..., file size")
    ap.add_argument("--time-limit", type=float, default=300.0, help="seconds per instance")
    ap.add_argument("--mip-gap", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=1,
                    help="instances solved at the same time, each in its own process")
    ap.add_argument("--threads", type=int, default=0,
                    help="Gurobi threads per instance; 0 = all cores divided by --workers")
    ap.add_argument("--mem-limit-gb", type=float, default=0.0,
                    help="Gurobi SoftMemLimit per instance in GB (0 = none); Gurobi then stops cleanly")
    ap.add_argument("--licence-wait", type=float, default=3600.0,
                    help="seconds to keep retrying when no Gurobi licence session is free")
    ap.add_argument("--no-warm-start", action="store_true", help="do not start from the greedy plan")
    ap.add_argument("--details", action="store_true", help="print per-EV and van tables")
    ap.add_argument("--verbose", action="store_true", help="show the Gurobi log")
    ap.add_argument("--reuse", action="store_true",
                    help="do not solve; score the plans already in plan.csv again")
    ap.add_argument("--resume", action="store_true",
                    help="skip instances whose MILP run (same run id and seed) is already in runs.csv")
    ap.add_argument("--fresh", action="store_true",
                    help="first remove every run of this run id from the tables (other runs stay)")
    ap.add_argument("--mip-focus", type=int, default=0, help="Gurobi MIPFocus (1 = find good solutions)")
    ap.add_argument("--node-file-dir", default="", help="folder for Gurobi node files (saves RAM)")
    ap.add_argument("--no-analysis", action="store_true", help="skip analyze_f4 at the end")
    ap.add_argument("--one", nargs=2, metavar=("FILE", "N"), help=argparse.SUPPRESS)  # worker mode
    args = ap.parse_args()
    args.run_id = args.run_id or f"milp_{args.time_limit:g}s"
    VARIANT["id"] = args.variant
    P4, cfg_over = fp.load_params(args.params, args.set, quiet=bool(args.one))
    cores = os.cpu_count() or 1
    args.workers = max(1, args.workers)
    if args.threads <= 0 and args.workers > 1:
        args.threads = max(1, cores // args.workers)
    args.data = fd.Dataset(args.data_dir, P4, cfg_over)

    if args.one:                                   # worker: one instance, then exit
        run_one(args.one[0], int(args.one[1]), args)
        return

    for msg in args.data.problems:
        print(f"WARNING (dataset): {msg}", flush=True)
    files, tasks = collect_tasks(args)
    if not files:
        print(f"no dataset files found in {args.data.path} (run ../../data/make_f4_dataset.py)")
        return
    print(f"input: {len(files)} file(s) from {args.data.path}; variant {args.variant}"
          + (f", overrides {P4['_overrides']}" if P4["_overrides"] else ""), flush=True)
    out = args.out
    P = rio.paths(out)
    os.makedirs(P["logs"], exist_ok=True)

    if args.reuse:
        rio.prepare(out)
        rescore(tasks, args)
    else:
        rio.update_meta(out, args.run_id, {
            "method": "MILP", "argv": sys.argv, "variant": args.variant, "overrides": P4["_overrides"],
            "time_limit_s": args.time_limit, "mip_gap": args.mip_gap,
            "seed": args.seed, "workers": args.workers, "threads_per_instance": args.threads,
            "cores": cores, "warm_start": not args.no_warm_start,
            "data": args.data.path, "params": os.path.abspath(args.params),
            "f4_params": P4, "started": time.strftime("%Y-%m-%d %H:%M:%S")})
        rio.update_meta(out, "greedy", {
            "method": "greedy", "variant": args.variant,
            "rule": "cost rule, else earliest-finish rule (solve_milp_f4.greedy_f4); the plan that warm-starts "
                    "the MILP"})
        rio.merge_parts(out)                       # results left by an interrupted run
        existing = {rio.key_of(r) for r in rio.read_runs(out)}
        if args.fresh:
            existing = {k for k in existing if k[0] != args.run_id}
        todo = [(p_, n_) for p_, n_ in tasks if not (args.resume and milp_key(args, p_, n_) in existing)]
        redo = {(base_of(p_), str(n_)) for p_, n_ in todo}

        def drop(k, _row):                        # runs that are solved again in this run
            if args.fresh and k[0] == args.run_id:
                return True
            mine = (k[0] == args.run_id and k[1] == "MILP" and k[2] == str(args.seed)) or k[0] == "greedy"
            return mine and (k[3], k[4]) in redo
        rio.prepare(out, drop)
        print(f"{len(tasks)} instance(s), {len(tasks) - len(todo)} already in runs.csv, {len(todo)} to solve; "
              f"run id {args.run_id}; {args.workers} at a time with {args.threads or 'all'} Gurobi thread(s) "
              f"each ({cores} cores); output in {os.path.abspath(out)}", flush=True)
        solve_all(todo, args)
        left = len(rio.merge_parts(out))
        if os.path.isdir(P["parts"]) and os.listdir(P["parts"]):
            print(f"WARNING: some results could not be appended (runs.csv / plan.csv locked?); they wait in "
                  f"{P['parts']} and are added by the next run (any options)", flush=True)
        elif left:
            print(f"{left} late result(s) appended", flush=True)
        rio.update_meta(out, args.run_id, {"finished": time.strftime("%Y-%m-%d %H:%M:%S")})
        mine = [r_ for r_ in rio.read_runs(out) if r_["run_id"] == args.run_id]
        counts = {}
        for r_ in mine:
            counts[r_["status"]] = counts.get(r_["status"], 0) + 1
        print(f"\nstatus of the {len(mine)} run(s) of {args.run_id} in runs.csv: {counts}", flush=True)
    rio.pack_logs(out)
    if not args.no_analysis:
        try:
            import analyze_f4
            analyze_f4.run(out, args.run_id)
        except Exception as e:
            print(f"analysis not written: {type(e).__name__}: {e}", flush=True)
            try:
                with open(os.path.join(out, "analysis_error.log"), "w") as fh:
                    fh.write(traceback.format_exc())
            except OSError:
                pass
    print(f"\nResults written to {os.path.abspath(out)}", flush=True)


def solve_all(todo, args):
    """Solve the (file, n) tasks, one at a time or --workers at a time in their own processes, and
    append each finished instance to runs.csv / plan.csv."""
    out = args.out
    P = rio.paths(out)
    total, k = len(todo), 0
    t_start = time.time()
    running = {}

    def part_of(path, n):
        return os.path.abspath(os.path.join(P["parts"], rio.part_name(milp_key(args, path, n)) + ".json"))

    def collect(path, n, t0, fallback):
        nonlocal k
        part = part_of(path, n)
        if not os.path.exists(part):            # the worker died before writing its results
            try:
                rio.write_part(out, [rio.run_row(error_keys(args, path, n), "error", message=fallback,
                                                 wall_clock_s=round(time.time() - t0, 1))], [])
            except OSError as e:
                print(f"{base_of(path)}_n{n}: error row not written ({e})", flush=True)
        merged = rio.merge_parts(out, only={part})
        row = next((r_ for r_ in merged if r_["method"] == "MILP"), None)
        if row is None:                         # could not be appended now: read it from the part
            try:
                with open(part) as fh:
                    row = json.load(fh)["runs"][0]
            except (OSError, ValueError, IndexError, KeyError):
                row = {"status": "unknown"}
        k += 1
        print(f"[{k}/{total}] {base_of(path)}_n{n}: {short(row)} ({time.time() - t0:.0f} s; "
              f"{(time.time() - t_start) / 60:.0f} min so far)", flush=True)

    try:
        if args.workers <= 1:
            env = None
            if gp is not None:
                try:
                    env = start_env(args.licence_wait)
                except Exception as e:             # each instance then tries on its own
                    print(f"Gurobi could not start ({e}); every instance will try again", flush=True)
            for path, n in todo:
                t0 = time.time()
                run_one(path, n, args, env)
                collect(path, n, t0, "no result written")
            if env is not None:
                env.dispose()
        else:
            queue = list(todo)
            while queue or running:
                while queue and len(running) < args.workers:
                    path, n = queue.pop(0)
                    stale = part_of(path, n)
                    if os.path.exists(stale):
                        os.remove(stale)
                    logfile = os.path.join(P["logs"], rio.log_stem(args.run_id, base_of(path), n) + ".log")
                    try:
                        fh = open(logfile, "w")
                        proc = subprocess.Popen(child_cmd(path, n, args), stdout=fh,
                                                stderr=subprocess.STDOUT, cwd=HERE)
                        running[proc] = (path, n, fh, logfile, time.time())
                    except OSError as e:
                        collect(path, n, time.time(), f"could not start a worker: {e}")
                finished = [p_ for p_ in running if p_.poll() is not None]
                if not finished:
                    time.sleep(1.0)
                    continue
                for proc in finished:
                    path, n, fh, logfile, t0 = running.pop(proc)
                    fh.close()
                    collect(path, n, t0, f"worker exit code {proc.returncode}: {tail(logfile)}")
    except KeyboardInterrupt:
        print("\nInterrupted: stopping the workers; the instances finished so far are in the tables",
              flush=True)
        for proc in list(running):
            try:
                proc.terminate()
            except OSError:
                pass
        for proc, (_, _, fh, _, _) in list(running.items()):
            try:
                proc.wait(timeout=30)
                fh.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
