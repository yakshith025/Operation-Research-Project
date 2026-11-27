"""Shadow prices (dual values) of every valid formulation-4 plan of the RL baseline (rl_f4) and, for comparison,
of the MILP baseline (milp_300s).

A plan has no duals of its own: the decisions are discrete. As in the formulation-3 reports, every yes/no decision
of the plan is fixed (who is served where, the order on every gun, every van route) and the remaining linear
program in the start times S_i and the waits q_i is solved again:

  min  sum_{i at a gun} c^Q_i q_i + sum_{i on a van} c^V_i q_i
  gun user i at station f:   S_i >= r_i + t_if          (arrival)      q_i = S_i - r_i - t_if
                             q_i <= W_i                  (patience)     S_i <= T - tau^F_i (deadline)
  i then j on the same gun:  S_j >= S_i + tau^F_i        (gun order)
  van user i:                S_i >= r_i                  (release)      q_i = S_i - r_i
                             S_i <= T - tau^M_i          (deadline)
  van route 0 -> a .. b -> 0: S_a >= t_0a (depart),  S_j >= S_i + tau^M_i + t_ij (van order),
                              S_b <= T - tau^M_b - t_b0 (return)

Everything else in Z is a constant once the decisions are fixed. The LP optimum must equal the plan's own waiting
cost (check). Its duals are the shadow prices of the plan. dZ/dP^F and dZ/dP^M (one more kW at every gun / van)
follow from the chain rule through tau = 60 E / P (customer's own charging-time term + the gun / van order and
deadline rows).

Reads ../../../{baseline_RL,baseline_MILP}/results (runs.csv, plan.csv) and the
formulation-4 data. Writes data/duals_constraints.csv and data/duals_instances.csv (column method = RL / MILP).
Usage (from this folder):  python compute_duals_rl_f4.py
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import report_common as rc  # noqa: E402
F4J = rc.F4J

rc.f4_import()
import f4_data as fd  # noqa: E402
import gurobipy as gp  # noqa: E402
from gurobipy import GRB  # noqa: E402

RUNS = {"RL": (F4J("baseline_RL", "results"), "rl_f4"),
        "MILP": (F4J("baseline_MILP", "results"), "milp_300s")}
PF, PM = 22.0, 30.0


def plan_of(rows):
    """resource per EV (0-based index), S per EV, guns {gun: [EVs in order]}, van routes [[EVs in order]]."""
    ev = rows[rows.type == "ev"].copy()
    ev["i"] = ev["id"].astype(int) - 1
    res = dict(zip(ev.i, ev.resource))
    S = dict(zip(ev.i, ev.start_min.astype(float)))
    guns, vans = {}, {}
    for _, r in ev.sort_values(["resource", "seq", "start_min"]).iterrows():
        (guns if r.resource_type == "FCS" else vans).setdefault(r.resource, []).append(r.i)
    return res, S, guns, list(vans.values())


def timing_lp(J, res, guns, routes, env):
    ev = J["ev"]
    n, T = J["n_ev"], J["params"]["T_min"]
    r, tF, tM, W, cQ, cV = ev["r_min"], ev["tauF_min"], ev["tauM_min"], ev["W"], ev["cQ"], ev["cV"]
    tEF, tN = J["t_ev_fcs_min"], J["t_nodes_min"]
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
    for evs in guns.values():
        for a, b in zip(evs, evs[1:]):
            cons.append((m.addConstr(S[b] >= S[a] + tF[a]), "gun", a, b))
    for stops in routes:
        cons.append((m.addConstr(S[stops[0]] >= tN[0][stops[0] + 1]), "depart", stops[0], None))
        for a, b in zip(stops, stops[1:]):
            cons.append((m.addConstr(S[b] >= S[a] + tM[a] + tN[a + 1][b + 1]), "van", a, b))
        last = stops[-1]
        cons.append((m.addConstr(S[last] <= T - tM[last] - tN[last + 1][0]), "return", last, None))
    m.setObjective(gp.quicksum((cQ[i] if i in fcs_of else cV[i]) * q[i] for i in range(n)), GRB.MINIMIZE)
    m.optimize()
    return m, cons, fcs_of


def main():
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    ds = fd.Dataset(rc.F4_DATA, stems=("solomon",))
    rows_c, rows_i = [], []
    for method, (rdir, run_id) in RUNS.items():
        runs = rc.load_runs(rdir, run_id)
        runs = runs[runs.valid_plan]
        plan = rc.load_plan(rdir, run_id)
        groups = dict(tuple(plan.groupby(["instance", "n"])))
        for _, run in runs.iterrows():
            name, n = run.instance, int(run.n)
            J = ds.instance(name, n, analysis=True)
            ev = J["ev"]
            res, S_plan, guns, routes = plan_of(groups[(name, n)])
            m, cons, fcs_of = timing_lp(J, res, guns, routes, env)
            if m.Status != GRB.OPTIMAL:
                rows_i.append({"method": method, "instance": name, "n": n, "lp_status": m.Status})
                m.dispose()
                continue
            wait_plan = sum(ev["cQ"][i] * max(0.0, S_plan[i] - ev["r_min"][i] - J["t_ev_fcs_min"][i][fcs_of[i]])
                            if i in fcs_of else ev["cV"][i] * max(0.0, S_plan[i] - ev["r_min"][i])
                            for i in range(n))
            dT = 0.0
            dPF = sum(ev["cC"][i] * (-ev["tauF_min"][i] / PF) for i in fcs_of)
            dPM = sum(ev["cV"][i] * (-ev["tauM_min"][i] / PM) for i in range(n) if i not in fcs_of)
            congestion = 0.0
            for c, kind, i, j in cons:
                pi = c.Pi
                if kind in ("deadline_fcs", "deadline_van", "return"):
                    dT += pi
                if kind == "gun":
                    dPF += pi * (-ev["tauF_min"][i] / PF)
                    congestion += pi * (-ev["tauF_min"][i] / PF)
                elif kind == "deadline_fcs":
                    dPF += (-pi) * (-ev["tauF_min"][i] / PF)
                elif kind == "van":
                    dPM += pi * (-ev["tauM_min"][i] / PM)
                elif kind in ("deadline_van", "return"):
                    dPM += (-pi) * (-ev["tauM_min"][i] / PM)
                if kind not in ("deadline_fcs", "deadline_van", "return") or abs(pi) > 1e-9:
                    rows_c.append({"method": method, "instance": name, "family": J["family"], "n": n,
                                   "status": run.status, "kind": kind, "i": i + 1,
                                   "j": (j + 1) if j is not None else "",
                                   "theta_i": round(ev["theta"][i], 4), "segment_i": int(ev["segment"][i]),
                                   "theta_j": round(ev["theta"][j], 4) if j is not None else "",
                                   "segment_j": int(ev["segment"][j]) if j is not None else "",
                                   "pressure_j": ev["pressure"][j] if j is not None else ev["pressure"][i],
                                   "dual": round(pi, 6)})
                elif kind in ("deadline_fcs", "deadline_van", "return"):
                    rows_c.append({"method": method, "instance": name, "family": J["family"], "n": n,
                                   "status": run.status, "kind": kind, "i": i + 1, "j": "",
                                   "theta_i": round(ev["theta"][i], 4), "segment_i": int(ev["segment"][i]),
                                   "theta_j": "", "segment_j": "", "pressure_j": ev["pressure"][i], "dual": 0.0})
            rows_i.append({"method": method, "instance": name, "family": J["family"], "n": n,
                           "status": run.status, "lp_status": m.Status, "lp_obj": round(m.ObjVal, 4),
                           "plan_wait_cost": round(wait_plan, 4), "lp_minus_plan": round(m.ObjVal - wait_plan, 6),
                           "dZ_dT": round(dT, 6), "dZ_dPF": round(dPF, 6), "dZ_dPM": round(dPM, 6),
                           "congestion_part_PF": round(congestion, 6), "Z": run.Z_total,
                           "guns_used": len(guns), "vans_used": len(routes)})
            m.dispose()
        print(f"{method}: {sum(1 for x in rows_i if x['method'] == method)} plans", flush=True)
    pd.DataFrame(rows_c).to_csv(os.path.join(HERE, "data", "duals_constraints.csv"), index=False)
    di = pd.DataFrame(rows_i)
    di.to_csv(os.path.join(HERE, "data", "duals_instances.csv"), index=False)
    bad = (di.lp_minus_plan.abs() > 0.05).sum()
    print(f"{len(di)} plans, {len(rows_c)} constraint rows; LP differs from the plan's waiting cost in {bad} plans")
    env.dispose()


if __name__ == "__main__":
    main()
