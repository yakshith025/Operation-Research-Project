"""Check a plan against every constraint of main_formulation_3 and recompute its objective
Z = Z1 + ... + Z6 from scratch (independently of the MILP).

A plan is a dict with: resource (list of "FCS<f>-gun<g>" or "van<m>", 1-based), S and C
(start / completion minutes per EV), van_routes (node lists, 0 = depot), optionally Z_Rs.
J is the instance dict built by solve_milp_f3.load_instance (per-EV income coefficients in
J["ev"]: theta, mu, cD, cQ, cV, cC, W).
"""
import numpy as np

EPS = 1e-4   # minutes / kWh tolerance (solver feasibility tolerance is ~1e-6 relative)


def fcs_index(res):
    return int(res[3:res.index("-")]) - 1


def ev_breakdown(J, plan):
    """Per EV: resource type, driving, queue / van wait, charging, money paid, perceived
    money (mu * paid), perceived time cost and the operator's energy cost."""
    p, ev = J["params"], J["ev"]
    c, tar = p["costs_Rs"], p["tariff_Rs_per_kWh"]
    out = []
    for i in range(J["n_ev"]):
        res = plan["resource"][i]
        E, r, S = ev["E_kWh"][i], ev["r_min"][i], plan["S"][i]
        if res.startswith("FCS"):
            f = fcs_index(res)
            dkm, dmin = J["d_ev_fcs_km"][i][f], J["t_ev_fcs_min"][i][f]
            qF, qM, charge = max(0.0, S - r - dmin), 0.0, ev["tauF_min"][i]
            pay = c["c_EV"] * dkm + tar["p_F"] * E
            energy = c["pi_F"] * E
            kind = "FCS"
        else:
            f, dkm, dmin = None, 0.0, 0.0
            qF, qM, charge = 0.0, max(0.0, S - r), ev["tauM_min"][i]
            pay = tar["p_M"] * E
            energy = c["pi_M"] * E / p["eta"]
            kind = "MCS"
        mu = ev["mu"][i]
        # FCS user: driving c^D, queue c^Q, charging c^C (driver present);
        # MCS user: wait for the van and charging both at c^V (car parked, driver free)
        if kind == "FCS":
            time_cost = ev["cD"][i] * dmin + ev["cQ"][i] * qF + ev["cC"][i] * charge
        else:
            time_cost = ev["cV"][i] * (qM + charge)
        out.append({"kind": kind, "fcs": f, "drive_km": dkm, "drive_min": dmin,
                    "queue_min": qF, "van_wait_min": qM, "charge_min": charge,
                    "pay_Rs": pay, "perceived_money_Rs": mu * pay,
                    "drive_money_perceived_Rs": mu * c["c_EV"] * dkm,
                    "tariff_perceived_Rs": mu * (tar["p_F"] if kind == "FCS" else tar["p_M"]) * E,
                    "time_cost_Rs": time_cost, "customer_cost_Rs": mu * pay + time_cost,
                    "operator_energy_Rs": energy})
    return out


def validate(J, plan):
    """Return (list of violations, recomputed Z, dict Z1..Z6, per-EV breakdown)."""
    p, ev = J["params"], J["ev"]
    n, T = J["n_ev"], p["T_min"]
    r, E, R = map(np.array, (ev["r_min"], ev["E_kWh"], ev["R_km"]))
    W = np.array(ev["W"])
    tF, tM = np.array(ev["tauF_min"]), np.array(ev["tauM_min"])
    dEF, tEF = np.array(J["d_ev_fcs_km"]), np.array(J["t_ev_fcs_min"])
    dN, tN = np.array(J["d_nodes_km"]), np.array(J["t_nodes_min"])
    S, C, res = np.array(plan["S"]), np.array(plan["C"]), plan["resource"]
    err = []
    gun_jobs, on_van = {}, {}
    for i in range(n):                                   # (2) one resource each
        if res[i].startswith("FCS"):
            f = fcs_index(res[i])
            if dEF[i, f] > R[i] + EPS:
                err.append(f"EV{i+1} cannot reach FCS{f+1}")
            arrive = r[i] + tEF[i, f]
            if S[i] < arrive - EPS:                      # (3)
                err.append(f"EV{i+1} starts before arriving")
            if S[i] - arrive > W[i] + EPS:               # (5) queue limit W_i
                err.append(f"EV{i+1} queues {S[i] - arrive:.2f} > W = {W[i]:.2f} min")
            if abs(C[i] - S[i] - tF[i]) > 1e-3:
                err.append(f"EV{i+1} wrong FCS duration")
            gun_jobs.setdefault(res[i], []).append((S[i], C[i], i))
        else:
            on_van[i + 1] = res[i]
        if S[i] < r[i] - EPS:                            # release
            err.append(f"EV{i+1} starts before its request")
        if C[i] > T + EPS:                               # horizon
            err.append(f"EV{i+1} finishes after T")
    for gun, jobs in gun_jobs.items():                   # no overlap on a gun
        jobs.sort()
        for a, b in zip(jobs, jobs[1:]):
            if b[0] < a[1] - EPS:
                err.append(f"overlap on {gun}: EV{a[2]+1} and EV{b[2]+1}")
    seen = []
    for route in plan["van_routes"]:                     # routing, capacity, battery, timing
        stops = route[1:-1]
        seen += stops
        if route[0] != 0 or route[-1] != 0:
            err.append(f"route {route} does not start and end at the depot")
        if len(stops) > p["Q"]:
            err.append(f"Q exceeded on route {route}")
        km = sum(dN[a, b] for a, b in zip(route, route[1:]))
        energy = sum(E[k - 1] / p["eta"] for k in stops) + p["beta_kWh_per_km"] * km
        if energy > p["B_kWh"] + EPS:
            err.append(f"battery exceeded on route {route}")
        t, pos = 0.0, 0
        for k in stops:
            i = k - 1
            if S[i] < max(r[i], t + tN[pos, k]) - EPS:
                err.append(f"van starts EV{k} too early")
            if abs(C[i] - S[i] - tM[i]) > 1e-3:
                err.append(f"EV{k} wrong van duration")
            t, pos = C[i], k
        if t + tN[pos, 0] > T + EPS:
            err.append(f"van back after T on route {route}")
    if sorted(seen) != sorted(on_van):
        err.append("van EVs not visited exactly once")
    vans_named = [res[k - 1] for rt in plan["van_routes"] for k in rt[1:-1]]
    if len(set(vans_named)) != len(plan["van_routes"]):
        err.append("van labels do not match the routes")

    detail = ev_breakdown(J, plan)
    c = p["costs_Rs"]
    Z1 = sum(d["drive_money_perceived_Rs"] for d in detail)
    Z2 = c["c_M"] * sum(sum(dN[a, b] for a, b in zip(rt, rt[1:])) for rt in plan["van_routes"])
    Z3 = c["c_F"] * len(plan["van_routes"])
    Z4 = sum(d["operator_energy_Rs"] for d in detail)
    Z5 = sum(d["tariff_perceived_Rs"] for d in detail)
    Z6 = sum(d["time_cost_Rs"] for d in detail)
    Z = Z1 + Z2 + Z3 + Z4 + Z5 + Z6
    if "Z_Rs" in plan and plan["Z_Rs"] not in ("", None) and \
            abs(Z - plan["Z_Rs"]) > max(0.05, 1e-6 * abs(Z)):
        err.append(f"objective mismatch {Z:.2f} vs {plan['Z_Rs']}")
    parts = {"Z1": Z1, "Z2": Z2, "Z3": Z3, "Z4": Z4, "Z5": Z5, "Z6": Z6}
    return err, Z, parts, detail
