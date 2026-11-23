"""Shadow prices (fixed-plan LP duals) of every valid plan of one method, read from a results
folder in the results_io layout (runs.csv + plan.csv). Same LP as new_results/report/compute_duals.py:
all discrete decisions of the plan are fixed and only the start times / waits are re-optimised.

Usage (from this folder):
  python compute_duals_rl.py                       # RL1 plans (../ , run_id rl1) and MILP plans
                                                   # (../../../new_results, run_id milp_300s)
Writes data/duals_<run_id>_constraints.csv and data/duals_<run_id>_instances.csv.
"""
import csv
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(PROJECT, "code"))
import gurobipy as gp                    # noqa: E402
from gurobipy import GRB                 # noqa: E402
import make_ev_solomon as mes            # noqa: E402
import make_income as inc                # noqa: E402
import results_io as rio                 # noqa: E402
import solve_milp_f3 as f3               # noqa: E402

P3 = inc.load_params()
DATA = mes.EvSolomonData(os.path.join(PROJECT, "dataset_ev_solomon", "csv"))
SOURCES = {"rl1": os.path.join(HERE, ".."), "milp_300s": os.path.join(PROJECT, "new_results")}
POW_F, POW_M = 22.0, 30.0


def plans_of(folder, run_id):
    runs = {(r["instance"], int(r["n"])): r for r in csv.DictReader(open(os.path.join(folder, "runs.csv")))
            if r["run_id"] == run_id}
    rows = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(folder, "plan.csv"))):
        if r["run_id"] == run_id:
            rows[(r["instance"], int(r["n"]))].append(r)
    return runs, rows


def timing_lp(J, plan, env):
    ev = J["ev"]
    n, T = J["n_ev"], J["params"]["T_min"]
    r, tF, tM, W = ev["r_min"], ev["tauF_min"], ev["tauM_min"], ev["W"]
    cQ, cV = ev["cQ"], ev["cV"]
    tEF, tN = J["t_ev_fcs_min"], J["t_nodes_min"]
    res = plan["resource"]
    m = gp.Model(env=env)
    S = m.addVars(n, lb=0.0)
    q = m.addVars(n, lb=0.0)
    cons, fcs_of = [], {}
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
    guns = defaultdict(list)
    for i in fcs_of:
        guns[res[i]].append(i)
    for g, evs in guns.items():
        evs.sort(key=lambda k: plan["S"][k])
        for a, b in zip(evs, evs[1:]):
            cons.append((m.addConstr(S[b] >= S[a] + tF[a]), "gun", a, b))
    for route in plan["van_routes"]:
        stops = [k - 1 for k in route[1:-1]]
        if not stops:
            continue
        cons.append((m.addConstr(S[stops[0]] >= tN[0][stops[0] + 1]), "depart", stops[0], None))
        for a, b in zip(stops, stops[1:]):
            cons.append((m.addConstr(S[b] >= S[a] + tM[a] + tN[a + 1][b + 1]), "van", a, b))
        last = stops[-1]
        cons.append((m.addConstr(S[last] <= T - tM[last] - tN[last + 1][0]), "return", last, None))
    m.setObjective(gp.quicksum((cQ[i] if i in fcs_of else cV[i]) * q[i] for i in range(n)), GRB.MINIMIZE)
    m.Params.OutputFlag = 0
    m.optimize()
    return m, cons, fcs_of


def run(run_id, env):
    runs, rows = plans_of(SOURCES[run_id], run_id)
    out_c, out_i = [], []
    for key, row in sorted(runs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if row["valid"] != "yes":
            continue
        name, n = key
        J = f3.load_instance(name, n, P3, DATA, greedy=False)
        plan = rio.plan_from_rows(rows[key], n)
        m, cons, fcs_of = timing_lp(J, plan, env)
        if m.Status != GRB.OPTIMAL:
            out_i.append({"instance": name, "n": n, "lp_status": m.Status})
            m.dispose()
            continue
        ev = J["ev"]
        wait = sum((ev["cQ"][i] * max(0.0, plan["S"][i] - ev["r_min"][i] - J["t_ev_fcs_min"][i][fcs_of[i]])
                    if i in fcs_of else ev["cV"][i] * max(0.0, plan["S"][i] - ev["r_min"][i])) for i in range(n))
        dT = 0.0
        dPF = sum(ev["cC"][i] * (-ev["tauF_min"][i] / POW_F) for i in fcs_of)
        dPM = sum(ev["cV"][i] * (-ev["tauM_min"][i] / POW_M) for i in range(n) if i not in fcs_of)
        own_f, own_m = dPF, dPM
        for c, kind, i, j in cons:
            pi = c.Pi
            if kind in ("deadline_fcs", "deadline_van", "return"):
                dT += pi
            if kind == "gun":
                dPF += pi * (-ev["tauF_min"][i] / POW_F)
            elif kind == "deadline_fcs":
                dPF += (-pi) * (-ev["tauF_min"][i] / POW_F)
            elif kind == "van":
                dPM += pi * (-ev["tauM_min"][i] / POW_M)
            elif kind in ("deadline_van", "return"):
                dPM += (-pi) * (-ev["tauM_min"][i] / POW_M)
            if kind not in ("deadline_fcs", "deadline_van", "return") or abs(pi) > 1e-9:
                out_c.append({"instance": name, "class": row["class"], "n": n, "kind": kind, "i": i + 1,
                              "j": (j + 1) if j is not None else "", "theta_i": round(ev["theta"][i], 4),
                              "theta_j": round(ev["theta"][j], 4) if j is not None else "", "dual": round(pi, 6)})
        out_i.append({"instance": name, "class": row["class"], "n": n, "status": row["status"],
                      "lp_obj": round(m.ObjVal, 4), "plan_wait_cost": round(wait, 4),
                      "lp_minus_plan": round(m.ObjVal - wait, 6), "dZ_dT": round(dT, 6),
                      "dZ_dPF": round(dPF, 6), "dZ_dPF_own": round(own_f, 6), "dZ_dPM": round(dPM, 6),
                      "dZ_dPM_own": round(own_m, 6), "Z": row["Z_total"]})
        m.dispose()
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    for nm, rows_ in ((f"duals_{run_id}_constraints.csv", out_c), (f"duals_{run_id}_instances.csv", out_i)):
        keys = []
        for x in rows_:
            keys += [k for k in x if k not in keys]
        with open(os.path.join(HERE, "data", nm), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, restval="")
            w.writeheader()
            w.writerows(rows_)
    bad = [x for x in out_i if "lp_minus_plan" in x and abs(x["lp_minus_plan"]) > 0.05]
    print(f"{run_id}: {len(out_i)} plans, {len(out_c)} constraint duals, LP = plan waiting cost in all but {len(bad)}")


if __name__ == "__main__":
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    for rid in sys.argv[1:] or ["rl1", "milp_300s"]:
        run(rid, env)
    env.dispose()
