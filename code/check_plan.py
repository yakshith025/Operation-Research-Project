"""Check a plan (greedy baseline or MILP solution) against the constraints of
main_formulation (minutes, km, kWh) and recompute its objective from scratch.

A plan is a dict with:  resource (list of "FCS<f>-gun<g>" or "van<m>", 1-based),
S and C (start / completion minutes per EV), van_routes (lists of nodes, 0 = depot),
Z_Rs (claimed objective).

Usage:  python check_plan.py <instances_dir>     (checks every greedy baseline)
"""
import glob
import json
import os
import sys

import numpy as np

EPS = 1e-4   # minutes / kWh tolerance (solver feasibility tolerance is ~1e-6 relative)


def fcs_index(res):
    return int(res[3:res.index("-")]) - 1


def validate(J, plan):
    """Return (list of violations, recomputed Z, dict of Z1..Z5)."""
    p, ev = J["params"], J["ev"]
    n, T = J["n_ev"], p["T_min"]
    r, E, R, w = map(np.array, (ev["r_min"], ev["E_kWh"], ev["R_km"], ev["w"]))
    tF, tM = np.array(ev["tauF_min"]), np.array(ev["tauM_min"])
    dEF, tEF = np.array(J["d_ev_fcs_km"]), np.array(J["t_ev_fcs_min"])
    dN, tN = np.array(J["d_nodes_km"]), np.array(J["t_nodes_min"])
    S, C, res = np.array(plan["S"]), np.array(plan["C"]), plan["resource"]
    err = []
    gun_jobs = {}
    on_van = {}
    for i in range(n):                                   # (2) one resource each
        if res[i].startswith("FCS"):
            f = fcs_index(res[i])
            if dEF[i, f] > R[i] + EPS:
                err.append(f"EV{i+1} cannot reach FCS{f+1}")
            if S[i] < r[i] + tEF[i, f] - EPS:            # (3)
                err.append(f"EV{i+1} starts before arriving")
            if abs(C[i] - S[i] - tF[i]) > 1e-3:
                err.append(f"EV{i+1} wrong FCS duration")
            gun_jobs.setdefault(res[i], []).append((S[i], C[i], i))
        else:
            on_van[i + 1] = res[i]
        if S[i] < r[i] - EPS:                            # (17)
            err.append(f"EV{i+1} starts before its request")
        if C[i] > T + EPS:                               # (18)
            err.append(f"EV{i+1} finishes after T")
    for gun, jobs in gun_jobs.items():                   # (4)-(5) no overlap on a gun
        jobs.sort()
        for a, b in zip(jobs, jobs[1:]):
            if b[0] < a[1] - EPS:
                err.append(f"overlap on {gun}: EV{a[2]+1} and EV{b[2]+1}")
    seen = []
    for route in plan["van_routes"]:                     # (7)-(16)
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
            if S[i] < max(r[i], t + tN[pos, k]) - EPS:   # (14), (15)
                err.append(f"van starts EV{k} too early")
            if abs(C[i] - S[i] - tM[i]) > 1e-3:
                err.append(f"EV{k} wrong van duration")
            t, pos = C[i], k
        if t + tN[pos, 0] > T + EPS:                     # (16)
            err.append(f"van back after T on route {route}")
    if sorted(seen) != sorted(on_van):                   # (9)-(10)
        err.append("van EVs not visited exactly once")
    c = p["costs_Rs"]
    Z1 = c["c_EV"] * sum(dEF[i, fcs_index(res[i])] for i in range(n) if res[i].startswith("FCS"))
    Z2 = c["c_M"] * sum(sum(dN[a, b] for a, b in zip(rt, rt[1:])) for rt in plan["van_routes"])
    Z3 = c["c_F"] * len(plan["van_routes"])
    Z4 = (c["pi_F"] * sum(E[i] for i in range(n) if res[i].startswith("FCS"))
          + c["pi_M"] * sum(E[k - 1] / p["eta"] for k in on_van))
    Z5 = c["c_W"] * float(np.sum(w * (C - r)))
    Z = Z1 + Z2 + Z3 + Z4 + Z5
    if "Z_Rs" in plan and abs(Z - plan["Z_Rs"]) > max(0.05, 1e-6 * abs(Z)):
        err.append(f"objective mismatch {Z:.2f} vs {plan['Z_Rs']}")
    return err, Z, {"Z1": Z1, "Z2": Z2, "Z3": Z3, "Z4": Z4, "Z5": Z5}


def check(path):
    J = json.load(open(path))
    err, Z, _ = validate(J, J["greedy_baseline"])
    return J["name"], err, Z


if __name__ == "__main__":
    bad = 0
    for path in sorted(glob.glob(os.path.join(sys.argv[1], "*", "instance.json"))):
        name, err, Z = check(path)
        bad += bool(err)
        print(f"{name:12s} {'OK ' if not err else 'FAIL'} Z={Z:10.2f} {'; '.join(err[:3])}")
    print("all plans valid" if not bad else f"{bad} instance(s) with problems")
