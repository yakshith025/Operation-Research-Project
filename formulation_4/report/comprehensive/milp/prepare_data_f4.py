"""Data of the MILP comprehensive report (formulation 4, baseline run milp_300s).

One pass over the 560 instances of ../../../methods/milp/results:
  data/instances.csv   one row per instance: runs.csv columns of milp_300s and of the greedy run, a data-only
                       infeasibility test (some EV cannot be charged by T even with a free gun or a van at once)
                       and the envelope quantities g_k(x*) of every cost coefficient (Z = sum_k c_k g_k)
  data/ev_level.csv    one row per EV of every valid plan: plan.csv columns + instance data (E, r, tau, mu,
                       c^a, W, demand pressure, nearest reachable FCS, family, n, horizon type)
  data/vans.csv        one row per dispatched van: stops, km, battery drawn, return time, tightest resource
  data/duals_constraints.csv, data/duals_instances.csv
                       shadow prices of the fixed-plan LP of every valid plan (see compute_duals of the
                       formulation-3 report; same LP, formulation-4 coefficients):
                         min sum_FCS c^Q_i q^F_i + sum_van c^V_i q^M_i
                         FCS user: S_i >= r_i + t_if (arrival), q^F_i = S_i - r_i - t_if, q^F_i <= W_i (patience),
                                   S_i <= T - tau^F_i (deadline)
                         i then j on a gun: S_j >= S_i + tau^F_i (gun)
                         van user: S_i >= r_i (release), q^M_i = S_i - r_i, S_i <= T - tau^M_i (deadline)
                         van route 0 -> a -> ... -> b -> 0: S_a >= t_0a (depart), S_j >= S_i + tau^M_i + t_ij (van),
                                   S_b <= T - tau^M_b - t_b0 (return)
Usage (from this folder):  python prepare_data_f4.py          (about 2 minutes; needs gurobipy)
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import report_common as rc  # noqa: E402
F4J = rc.F4J

rc.f4_import()
import f4_data as fd  # noqa: E402
import f4_params as fp  # noqa: E402
import gurobipy as gp  # noqa: E402
from gurobipy import GRB  # noqa: E402

RES = F4J("baseline_MILP", "results")
DATA = os.path.join(HERE, "data")
RUN = "milp_300s"
SEG = fp.SEGMENTS                     # ['personal', 'commercial'] (index = segment code)


def timing_lp(J, evs, routes, env):
    """Fixed-plan LP. evs: list of plan rows (dicts, index = EV index); routes: {van: [ev index in order]}."""
    ev = J["ev"]
    n, T = J["n_ev"], J["params"]["T_min"]
    r, tF, tM, W = ev["r_min"], ev["tauF_min"], ev["tauM_min"], ev["W"]
    cQ, cV = ev["cQ"], ev["cV"]
    tN = J["t_nodes_min"]
    m = gp.Model(env=env)
    S = m.addVars(n, lb=0.0)
    q = m.addVars(n, lb=0.0)
    cons, fcs = [], {}
    for i in range(n):
        e = evs[i]
        if e["resource_type"] == "FCS":
            t = float(e["drive_min"])
            fcs[i] = t
            cons.append((m.addConstr(S[i] >= r[i] + t), "arrival", i, None))
            m.addConstr(q[i] == S[i] - r[i] - t)
            cons.append((m.addConstr(q[i] <= W[i]), "patience", i, None))
            cons.append((m.addConstr(S[i] <= T - tF[i]), "deadline_fcs", i, None))
        else:
            cons.append((m.addConstr(S[i] >= r[i]), "release", i, None))
            m.addConstr(q[i] == S[i] - r[i])
            cons.append((m.addConstr(S[i] <= T - tM[i]), "deadline_van", i, None))
    guns = {}
    for i in fcs:
        guns.setdefault(evs[i]["resource"], []).append(i)
    for g, lst in guns.items():
        lst.sort(key=lambda k: (float(evs[k]["start_min"]), float(evs[k]["seq"])))
        for a, b in zip(lst, lst[1:]):
            cons.append((m.addConstr(S[b] >= S[a] + tF[a]), "gun", a, b))
    for van, stops in routes.items():
        cons.append((m.addConstr(S[stops[0]] >= tN[0][stops[0] + 1]), "depart", stops[0], None))
        for a, b in zip(stops, stops[1:]):
            cons.append((m.addConstr(S[b] >= S[a] + tM[a] + tN[a + 1][b + 1]), "van", a, b))
        last = stops[-1]
        cons.append((m.addConstr(S[last] <= T - tM[last] - tN[last + 1][0]), "return", last, None))
    m.setObjective(gp.quicksum((cQ[i] if i in fcs else cV[i]) * q[i] for i in range(n)), GRB.MINIMIZE)
    m.optimize()
    return m, cons, fcs


def data_infeasible(J):
    """True if some EV cannot finish by T even with a free gun at once or a van leaving the depot at once."""
    ev, T = J["ev"], J["params"]["T_min"]
    tEF, tN = J["t_ev_fcs_min"], J["t_nodes_min"]
    worst = None
    for i in range(J["n_ev"]):
        best = np.inf
        for f in fd.reachable(J, i):
            best = min(best, ev["r_min"][i] + tEF[i][f] + ev["tauF_min"][i])
        van_end = max(ev["r_min"][i], tN[0][i + 1]) + ev["tauM_min"][i]
        if van_end + tN[i + 1][0] <= T:
            best = min(best, van_end)
        if best > T + 1e-6:
            return True, i + 1, best
        worst = best if worst is None else max(worst, best)
    return False, None, worst


def main():
    os.makedirs(DATA, exist_ok=True)
    runs = pd.read_csv(os.path.join(RES, "runs.csv"), low_memory=False)
    plan = pd.read_csv(os.path.join(RES, "plan.csv"), low_memory=False)
    milp = runs[runs.run_id == RUN].set_index(["instance", "n"])
    greedy = runs[runs.run_id == "greedy"].set_index(["instance", "n"])
    pl = plan[plan.run_id == RUN]
    pl_ev = {k: g for k, g in pl[pl.type == "ev"].groupby(["instance", "n"])}
    pl_van = {k: g for k, g in pl[pl.type == "van"].groupby(["instance", "n"])}

    P = fp.load_params(quiet=True)[0]
    ds = fd.Dataset(rc.F4_DATA, P=P, stems=("solomon",))
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    inst_rows, ev_rows, van_rows, dual_rows, dual_inst = [], [], [], [], []
    pressures = []
    for (name, n), row in milp.iterrows():
        J = ds.instance(name, int(n))
        cfg = J["params"]
        T = cfg["T_min"]
        ev = J["ev"]
        g = greedy.loc[(name, n)]
        infe, infe_ev, infe_t = data_infeasible(J)
        ir = {"instance": name, "n": int(n), "family": row["class"], "T": T,
              "horizon": "long" if row["class"].endswith("2") else "short",
              "status": row["status"], "valid": row["valid"] == "yes", "Z": row["Z_total"],
              "gap": row["gap_%"], "solve_s": row["solve_s"], "nodes": row["nodes"],
              "binaries": row["binaries"], "continuous": row["continuous"], "constraints": row["constraints"],
              "build_s": row["build_s"], "warm_start": row["warm_start"],
              "greedy_status": g["status"], "greedy_Z": g["Z_total"] if g["valid"] == "yes" else np.nan,
              "data_infeasible": infe, "data_infeasible_ev": infe_ev, "data_infeasible_finish": infe_t,
              "mean_pressure": float(np.mean(ev["pressure"]))}
        for c in ["Z1_EV_drive_perceived", "Z2_van_drive", "Z3_dispatch", "Z4_operator_energy",
                  "Z5_tariff_perceived", "Z6_time_perceived", "operator_cost_Z2_to_Z4", "customer_cost_Z1_Z5_Z6",
                  "vans_used", "van_km", "EVs_by_van", "free_choice_EVs", "free_choice_by_van",
                  "mean_queue_FCS_min", "mean_van_wait_min", "mean_time_to_charged_min"]:
            ir[c] = row[c]
        if not ir["valid"]:
            inst_rows.append(ir)
            continue
        evs_df = pl_ev[(name, n)].sort_values("id")
        assert len(evs_df) == n, (name, n)
        evs = evs_df.to_dict("records")
        routes = {}
        for e in evs:
            if e["resource_type"] == "MCS":
                routes.setdefault(e["resource"], []).append((float(e["seq"]), int(e["id"]) - 1))
        routes = {v: [i for _, i in sorted(s)] for v, s in routes.items()}

        # ---- envelope quantities g_k: Z = sum_k c_k g_k
        th = np.array(ev["theta"])
        seg = np.array(ev["segment"])
        E = np.array(ev["E_kWh"])
        mu = np.array(ev["mu"])
        isF = np.array([e["resource_type"] == "FCS" for e in evs])
        dkm = np.array([float(e["drive_km"]) for e in evs])
        dmin = np.array([float(e["drive_min"]) for e in evs])
        qF = np.array([float(e["queue_min"]) for e in evs])
        qM = np.array([float(e["van_wait_min"]) for e in evs])
        ch = np.array([float(e["charge_min"]) for e in evs])
        eta = cfg["eta"]
        q = {"c_EV": float(np.sum(mu * dkm)), "c_M": float(row["van_km"]), "c_F": float(row["vans_used"]),
             "pi_F": float(np.sum(E[isF])), "pi_M": float(np.sum(E[~isF] / eta)),
             "p_F": float(np.sum(mu[isF] * E[isF])), "p_M": float(np.sum(mu[~isF] * E[~isF]))}
        pay = np.array([float(e["pay_Rs"]) for e in evs])
        mins = {"D": dmin, "Q": np.where(isF, qF, 0.0), "C": np.where(isF, ch, 0.0),
                "V": np.where(isF, 0.0, qM + ch)}
        for s, sname in enumerate(SEG):
            w0, w1 = (1 - th) * (seg == s), th * (seg == s)
            q[f"mu0_{sname}"] = float(np.sum(w0 * pay))
            q[f"mu1_{sname}"] = float(np.sum(w1 * pay))
            for a, mm in mins.items():
                q[f"c{a}0_{sname}"] = float(np.sum(w0 * mm))
                q[f"c{a}1_{sname}"] = float(np.sum(w1 * mm))
        Zrec = (P["tariff_Rs_per_kWh"]["p_F"] * q["p_F"] + P["tariff_Rs_per_kWh"]["p_M"] * q["p_M"]
                + cfg["costs_Rs"]["c_EV"] * q["c_EV"] + cfg["costs_Rs"]["c_M"] * q["c_M"]
                + cfg["costs_Rs"]["c_F"] * q["c_F"] + cfg["costs_Rs"]["pi_F"] * q["pi_F"]
                + cfg["costs_Rs"]["pi_M"] * q["pi_M"])
        for sname in SEG:
            S_ = P[sname]
            Zrec += 0  # money part of mu already inside p and c_EV terms; time part below
            for a in "DQCV":
                Zrec += (S_["value_of_time_Rs_per_min"][a]["theta0"] * q[f"c{a}0_{sname}"]
                         + S_["value_of_time_Rs_per_min"][a]["theta1"] * q[f"c{a}1_{sname}"])
        ir["Z_reconstructed"] = Zrec
        for k_, v in q.items():
            ir["g_" + k_] = v

        # ---- EV level
        tEF = J["t_ev_fcs_min"]
        dEF = J["d_ev_fcs_km"]
        for i, e in enumerate(evs):
            reach = fd.reachable(J, i)
            nearest = min((dEF[i][f] for f in reach), default=np.nan)
            if e["resource_type"] == "FCS":
                fidx = int(str(e["resource"])[3:str(e["resource"]).index("-")]) - 1
                assert abs(dEF[i][fidx] - float(e["drive_km"])) < 0.02, (name, n, i, e["resource"])
            pressures.append(ev["pressure"][i])
            ev_rows.append({
                "instance": name, "n": int(n), "family": row["class"], "horizon": ir["horizon"], "T": T,
                "status": row["status"], "id": i + 1, "segment": SEG[seg[i]], "theta": th[i],
                "stranded": e["stranded"] == "yes", "resource_type": e["resource_type"],
                "resource": e["resource"], "van": e["resource_type"] == "MCS",
                "E": E[i], "r": ev["r_min"][i], "R": ev["R_km"][i], "tauF": ev["tauF_min"][i],
                "tauM": ev["tauM_min"][i], "mu": mu[i], "cD": ev["cD"][i], "cQ": ev["cQ"][i],
                "cC": ev["cC"][i], "cV": ev["cV"][i], "W": ev["W"][i], "pressure": ev["pressure"][i],
                "period": ev["period"][i], "nearest_fcs_km": nearest, "n_reachable_fcs": len(reach),
                "start": float(e["start_min"]), "drive_km": dkm[i], "drive_min": dmin[i], "queue": qF[i],
                "van_wait": qM[i], "charge_min": ch[i], "downtime": float(e["downtime_min"]),
                "time_to_charged": float(e["time_to_charged_min"]), "pay": pay[i],
                "perc_money": float(e["perceived_money_Rs"]), "perc_time": float(e["time_cost_Rs"]),
                "perc_total": float(e["customer_cost_Rs"]), "op_energy": float(e["operator_energy_Rs"])})

        # ---- vans
        B, Q = cfg["B_kWh"], cfg["Q"]
        for _, v in pl_van.get((name, n), pd.DataFrame()).iterrows():
            vid = f"van{int(v['id'])}"
            stops_ = routes.get(vid, [])
            nstr = sum(1 for i in stops_ if evs[i]["stranded"] == "yes")
            use = {"seats": v["stops"] / Q, "battery": v["kWh_drawn"] / B, "time": v["return_min"] / T}
            van_rows.append({"instance": name, "n": int(n), "family": row["class"], "status": row["status"],
                             "van": vid, "stops": int(v["stops"]), "stranded_stops": nstr, "km": v["km"],
                             "kWh_drawn": v["kWh_drawn"], "kWh_delivered": v["kWh_delivered"],
                             "return_min": v["return_min"], "T": T, "seat_use": use["seats"],
                             "battery_use": use["battery"], "time_use": use["time"],
                             "tightest": max(use, key=use.get)})

        # ---- fixed-plan LP and shadow prices
        m, cons, fcs = timing_lp(J, evs, routes, env)
        if m.Status != GRB.OPTIMAL:
            dual_inst.append({"instance": name, "n": int(n), "lp_status": m.Status})
            m.dispose()
            continue
        wait_plan = float(np.sum(np.where(isF, np.array(ev["cQ"]) * qF, np.array(ev["cV"]) * qM)))
        PF, PM = 22.0, 30.0
        dPF = sum(ev["cC"][i] * (-ev["tauF_min"][i] / PF) for i in fcs)
        dPM = sum(ev["cV"][i] * (-ev["tauM_min"][i] / PM) for i in range(n) if i not in fcs)
        dPF_cong = 0.0
        dT = 0.0
        for c, kind, i, j in cons:
            pi = c.Pi
            if kind in ("deadline_fcs", "deadline_van", "return"):
                dT += pi
            if kind == "gun":
                dPF += pi * (-ev["tauF_min"][i] / PF)
                dPF_cong += pi * (-ev["tauF_min"][i] / PF)
            elif kind == "deadline_fcs":
                dPF += (-pi) * (-ev["tauF_min"][i] / PF)
            elif kind == "van":
                dPM += pi * (-ev["tauM_min"][i] / PM)
            elif kind in ("deadline_van", "return"):
                dPM += (-pi) * (-ev["tauM_min"][i] / PM)
            if kind not in ("deadline_fcs", "deadline_van", "return", "patience") or abs(pi) > 1e-9:
                dual_rows.append({"instance": name, "n": int(n), "family": row["class"], "status": row["status"],
                                  "kind": kind, "i": i + 1, "j": (j + 1) if j is not None else "",
                                  "theta_i": th[i], "seg_i": SEG[seg[i]],
                                  "theta_j": th[j] if j is not None else np.nan,
                                  "seg_j": SEG[seg[j]] if j is not None else "",
                                  "pressure_j": ev["pressure"][j] if j is not None else np.nan,
                                  "dual": pi})
        n_pat = sum(1 for c in cons if c[1] == "patience")
        n_dl = sum(1 for c in cons if c[1] in ("deadline_fcs", "deadline_van", "return"))
        dual_inst.append({"instance": name, "n": int(n), "family": row["class"], "status": row["status"],
                          "lp_obj": m.ObjVal, "plan_wait_cost": wait_plan, "lp_minus_plan": m.ObjVal - wait_plan,
                          "dZ_dT": dT, "dZ_dPF": dPF, "dZ_dPF_congestion": dPF_cong, "dZ_dPM": dPM,
                          "Z": row["Z_total"], "patience_rows": n_pat, "deadline_rows": n_dl,
                          "max_abs_patience_dual": max([abs(c.Pi) for c, k_, _, _ in cons if k_ == "patience"],
                                                       default=0.0),
                          "max_abs_deadline_dual": max([abs(c.Pi) for c, k_, _, _ in cons
                                                        if k_ in ("deadline_fcs", "deadline_van", "return")],
                                                       default=0.0)})
        m.dispose()
        inst_rows.append(ir)
        if len(dual_inst) % 100 == 0:
            print(f"{len(dual_inst)} plans done", flush=True)

    for nm, rows in (("instances.csv", inst_rows), ("ev_level.csv", ev_rows), ("vans.csv", van_rows),
                     ("duals_constraints.csv", dual_rows), ("duals_instances.csv", dual_inst)):
        pd.DataFrame(rows).to_csv(os.path.join(DATA, nm), index=False)
    di = pd.DataFrame(dual_inst)
    inst = pd.DataFrame(inst_rows)
    v = inst[inst.valid]
    print(f"{len(inst)} instances, {len(v)} valid; Z reconstruction max |diff| "
          f"{(v.Z_reconstructed - v.Z).abs().max():.3f}; LP = plan waiting cost max |diff| "
          f"{di.lp_minus_plan.abs().max():.4f}; EV rows {len(ev_rows)}, vans {len(van_rows)}")
    env.dispose()


if __name__ == "__main__":
    main()
