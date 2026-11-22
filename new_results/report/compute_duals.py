"""Shadow prices (dual values) of every valid plan of the formulation-3 run.

A MILP has no duals. The standard way to get them is to fix all discrete decisions of the plan
(who is served where, the order on every gun, every van route) and solve the remaining linear
program in the continuous variables: start times S_i, FCS queue q^F_i and van wait q^M_i.
Everything else in Z is a constant once the decisions are fixed. The duals of that LP are the
shadow prices *for the given plan*: how much Z would change if a right-hand side moved by one unit
while the decisions stay as they are.

Timing LP for one plan (minimise sum_i c^Q_i q^F_i + c^V_i q^M_i):
  FCS user i at station f:  arrival   S_i >= r_i + t_if
                            wait      q^F_i = S_i - r_i - t_if
                            patience  q^F_i <= W_i
                            deadline  S_i <= T - tau^F_i
  consecutive EVs i -> j on a gun:   gun       S_j >= S_i + tau^F_i
  van user i:               release   S_i >= r_i
                            wait      q^M_i = S_i - r_i
                            deadline  S_i <= T - tau^M_i
  van route 0 -> a -> ... -> b -> 0: depart    S_a >= t_0a
                                     van       S_j >= S_i + tau^M_i + t_ij  (consecutive stops)
                                     return    S_b <= T - tau^M_b - t_b0

Reads ../results.csv, ../<instance>/solution.json and the EV-Solomon data; writes
duals_constraints.csv (one row per constraint with a dual) and duals_instances.csv
(per instance: LP check, dZ/dT, dZ/dP^F, dZ/dP^M, totals).
Usage (from this folder):  python compute_duals.py
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "..")
PROJECT = os.path.join(RUN, "..")
sys.path.insert(0, os.path.join(PROJECT, "code"))
import gurobipy as gp                    # noqa: E402
from gurobipy import GRB                 # noqa: E402
import solve_milp_f3 as f3               # noqa: E402

INSTANCES = os.path.join(PROJECT, "dataset_ev_solomon", "instances")
P3 = json.load(open(os.path.join(RUN, "run_settings.json")))["f3_params"]


def file_of(instance):
    base, n = instance.rsplit("_n", 1)
    for size in ("100", "200"):
        p = os.path.join(INSTANCES, size, base + ".txt")
        if os.path.exists(p):
            return p, int(n)
    raise FileNotFoundError(instance)


def timing_lp(J, plan, env):
    ev = J["ev"]
    n, T = J["n_ev"], J["params"]["T_min"]
    r, tF, tM, W = ev["r_min"], ev["tauF_min"], ev["tauM_min"], ev["W"]
    cQ, cV = ev["cQ"], ev["cV"]
    tEF, tN = J["t_ev_fcs_min"], J["t_nodes_min"]
    res = plan["resource"]
    m = gp.Model(env=env)
    S = m.addVars(n, lb=0.0, name="S")
    q = m.addVars(n, lb=0.0, name="q")
    cons = []                                        # (constraint, kind, i, j)
    fcs_of = {}
    for i in range(n):
        if res[i].startswith("FCS"):
            f = int(res[i][3:res[i].index("-")]) - 1
            fcs_of[i] = f
            cons.append((m.addConstr(S[i] >= r[i] + tEF[i][f]), "arrival", i, None))
            m.addConstr(q[i] == S[i] - r[i] - tEF[i][f])
            cons.append((m.addConstr(q[i] <= W[i]), "patience", i, None))
            cons.append((m.addConstr(S[i] <= T - tF[i]), "deadline_fcs", i, None))
        else:
            cons.append((m.addConstr(S[i] >= r[i]), "release", i, None))
            m.addConstr(q[i] == S[i] - r[i])
            cons.append((m.addConstr(S[i] <= T - tM[i]), "deadline_van", i, None))
    guns = {}
    for i in fcs_of:
        guns.setdefault(res[i], []).append(i)
    for g, evs in guns.items():
        evs.sort(key=lambda k: plan["S"][k])
        for a, b in zip(evs, evs[1:]):
            cons.append((m.addConstr(S[b] >= S[a] + tF[a]), "gun", a, b))
    for route in plan["van_routes"]:
        stops = [k - 1 for k in route[1:-1]]
        cons.append((m.addConstr(S[stops[0]] >= tN[0][stops[0] + 1]), "depart", stops[0], None))
        for a, b in zip(stops, stops[1:]):
            cons.append((m.addConstr(S[b] >= S[a] + tM[a] + tN[a + 1][b + 1]), "van", a, b))
        last = stops[-1]
        cons.append((m.addConstr(S[last] <= T - tM[last] - tN[last + 1][0]), "return", last, None))
    m.setObjective(gp.quicksum((cQ[i] if i in fcs_of else cV[i]) * q[i] for i in range(n)), GRB.MINIMIZE)
    m.Params.OutputFlag = 0
    m.optimize()
    return m, cons, fcs_of


def main():
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    results = list(csv.DictReader(open(os.path.join(RUN, "results.csv"))))
    rows_c, rows_i = [], []
    PF, PM = 22.0, 30.0
    for k, r in enumerate(results):
        if r.get("valid") != "yes":
            continue
        name = r["instance"]
        path, n = file_of(name)
        J = f3.load_instance(path, n, P3, INSTANCES)
        sol = json.load(open(os.path.join(RUN, name, "solution.json")))
        plan = {k_: sol[k_] for k_ in ("resource", "S", "C", "van_routes", "route_van")}
        m, cons, fcs_of = timing_lp(J, plan, env)
        if m.Status != GRB.OPTIMAL:
            rows_i.append({"instance": name, "lp_status": m.Status})
            m.dispose()
            continue
        ev = J["ev"]
        wait_plan = sum((ev["cQ"][i] * max(0.0, plan["S"][i] - ev["r_min"][i] - J["t_ev_fcs_min"][i][fcs_of[i]])
                         if i in fcs_of else ev["cV"][i] * max(0.0, plan["S"][i] - ev["r_min"][i]))
                        for i in range(n))
        dT = 0.0
        dPF = sum(ev["cC"][i] * (-ev["tauF_min"][i] / PF) for i in fcs_of)       # own charging-time term
        dPM = sum(ev["cV"][i] * (-ev["tauM_min"][i] / PM) for i in range(n) if i not in fcs_of)
        for c, kind, i, j in cons:
            pi = c.Pi
            if kind in ("deadline_fcs", "deadline_van", "return"):
                dT += pi                                        # dZ/dT = sum of duals of the <= T rows
            # dZ/dP via tau: d tau^F_i / d P^F = -tau^F_i / P^F; tau enters gun (+tau, >=) and deadline (-tau, <=)
            if kind == "gun":
                dPF += pi * (-ev["tauF_min"][i] / PF)
            elif kind == "deadline_fcs":
                dPF += (-pi) * (-ev["tauF_min"][i] / PF)
            elif kind == "van":
                dPM += pi * (-ev["tauM_min"][i] / PM)
            elif kind == "deadline_van":
                dPM += (-pi) * (-ev["tauM_min"][i] / PM)
            elif kind == "return":
                dPM += (-pi) * (-ev["tauM_min"][i] / PM)
            if kind not in ("deadline_fcs", "deadline_van", "return") or abs(pi) > 1e-9:
                rows_c.append({"instance": name, "family": r["family"], "n": n, "status": r["status"].split(" (")[0],
                               "kind": kind, "i": i + 1, "j": (j + 1) if j is not None else "",
                               "theta_i": round(ev["theta"][i], 4),
                               "theta_j": round(ev["theta"][j], 4) if j is not None else "",
                               "dual": round(pi, 6)})
        rows_i.append({"instance": name, "family": r["family"], "n": n, "status": r["status"].split(" (")[0],
                       "lp_obj": round(m.ObjVal, 4), "plan_wait_cost": round(wait_plan, 4),
                       "lp_minus_plan": round(m.ObjVal - wait_plan, 6),
                       "dZ_dT": round(dT, 6), "dZ_dPF": round(dPF, 6), "dZ_dPM": round(dPM, 6),
                       "Z": r["Z_Rs"], "guns_used": len({plan["resource"][i] for i in fcs_of}),
                       "vans_used": len(plan["van_routes"])})
        m.dispose()
        if len(rows_i) % 50 == 0:
            print(f"{len(rows_i)} plans done", flush=True)
    for nm, rows in (("duals_constraints.csv", rows_c), ("duals_instances.csv", rows_i)):
        keys = []
        for row in rows:
            keys += [k_ for k_ in row if k_ not in keys]
        with open(os.path.join(HERE, "data", nm), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, restval="")
            w.writeheader()
            w.writerows(rows)
    bad = [x for x in rows_i if "lp_minus_plan" in x and abs(x["lp_minus_plan"]) > 0.05]
    print(f"{len(rows_i)} plans; {len(rows_c)} constraint duals; LP = plan waiting cost in all but {len(bad)} plans")
    env.dispose()


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    main()
