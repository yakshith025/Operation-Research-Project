"""Build EV-charging instances for main_formulation from Solomon VRPTW files.

Usage:  python make_dataset.py <solomon_raw_dir> <output_dir>

Each instance gets its own folder with CSV files, one JSON with everything,
and a greedy feasibility check (a feasible plan that obeys every constraint
of main_formulation, with its objective value as a baseline).
All times are in minutes, distances in km, energy in kWh, money in Rs.
"""
import csv
import json
import math
import os
import sys

import numpy as np

# ---------------------------------------------------------------- settings
INSTANCES = ["c101", "r101", "rc101"]          # clustered, random, mixed
SIZES = [10, 25, 50, 100]                      # first n Solomon customers
LARGE_SIZES = list(range(200, 1001, 100))      # synthesised beyond Solomon's 100 customers
N_FCS = {10: 2, 25: 3, 50: 6, 100: 12}         # stations (2 guns each)
N_VANS = {10: 1, 25: 2, 50: 3, 100: 5}
# large sizes keep the n = 100 ratios: 12 stations (24 guns) and 5 vans per 100 EVs
N_FCS.update({n: 12 * n // 100 for n in LARGE_SIZES})
N_VANS.update({n: 5 * n // 100 for n in LARGE_SIZES})
JITTER = 3.0      # grid units (0.6 km): position noise for synthesised EVs (n > 100)

LAMBDA = 0.2      # km per Solomon grid unit (100 x 100 grid -> 20 x 20 km city)
RHO = 1.3         # detour factor: road distance = RHO * straight line
T = 300.0         # planning period, minutes (17:00-22:00 evening peak)
R_MAX_RELEASE = 120.0   # release times scaled into [0, 120] min
E_MIN, E_MAX = 10.0, 40.0   # energy demand range, kWh
RANGE_MIN, RANGE_MAX = 5.0, 40.0   # remaining range, km
STRANDED_SHARE = 0.15   # share of EVs that cannot reach any station (van-only)
KAPPA = 2.0
GUNS = 2
P_F, P_M = 22.0, 30.0   # kW
V = 25.0                # km/h
BETA, ETA = 0.25, 0.9
B_VAN, Q_VAN = 150.0, 6
COSTS = {"c_EV": 4.0, "c_M": 15.0, "c_F": 1500.0, "pi_F": 8.7, "pi_M": 10.7,
         "c_W": 200.0 / 60.0}
SEED_BASE = 2026


def read_solomon(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 7 and all(p.lstrip("-").isdigit() for p in parts):
                rows.append([int(p) for p in parts])
    arr = np.array(rows, dtype=float)
    assert arr[0, 0] == 0 and len(arr) == 101, path
    return arr  # cust, x, y, demand, ready, due, service


def kmeans(points, k, rng, iters=100):
    # k-means++ initialisation, then Lloyd iterations
    centers = [points[rng.integers(len(points))]]
    for _ in range(1, k):
        d2 = np.min([np.sum((points - c) ** 2, axis=1) for c in centers], axis=0)
        centers.append(points[rng.choice(len(points), p=d2 / d2.sum())])
    centers = np.array(centers)
    for _ in range(iters):
        lab = np.argmin(((points[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
        new = np.array([points[lab == j].mean(0) if np.any(lab == j) else centers[j]
                        for j in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    return centers


def greedy_plan(inst):
    """Feasible plan by a simple rule; returns (feasible, Z, details)."""
    n, F, M = inst["n"], inst["n_fcs"], inst["n_vans"]
    r, E, w = inst["r"], inst["E"], inst["w"]
    tF, tM = inst["tauF"], inst["tauM"]
    dEF, tEF = inst["d_ev_fcs"], inst["t_ev_fcs"]
    dN, tN = inst["d_nodes"], inst["t_nodes"]
    reach = dEF <= inst["R"][:, None]
    gun_free = np.zeros((F, GUNS))
    van = [{"pos": 0, "free": 0.0, "batt": B_VAN, "count": 0, "km": 0.0, "route": []}
           for _ in range(M)]
    S = np.full(n, np.nan)
    C = np.full(n, np.nan)
    res = [None] * n
    # van-only EVs first, then the rest by release time (ties: higher priority)
    order = sorted(range(n), key=lambda i: (reach[i].any(), r[i], -w[i]))
    for i in order:
        best = None
        for f in np.where(reach[i])[0]:
            for g in range(GUNS):
                s = max(r[i] + tEF[i, f], gun_free[f, g])
                if s + tF[i] <= T + 1e-9 and (best is None or s + tF[i] < best[0]):
                    best = (s + tF[i], ("gun", f, g), s)
        for m, vm in enumerate(van):
            node = i + 1
            s = max(r[i], vm["free"] + tN[vm["pos"], node])
            c = s + tM[i]
            need = E[i] / ETA + BETA * (dN[vm["pos"], node] + dN[node, 0])
            if (vm["count"] < Q_VAN and need <= vm["batt"] + 1e-9
                    and c + tN[node, 0] <= T + 1e-9
                    and (best is None or c < best[0])):
                best = (c, ("van", m), s)
        if best is None:
            return False, None, {"failed_ev": i + 1}
        c, choice, s = best
        S[i], C[i] = s, c
        if choice[0] == "gun":
            _, f, g = choice
            gun_free[f, g] = c
            res[i] = f"FCS{f + 1}-gun{g + 1}"
        else:
            vm = van[choice[1]]
            node = i + 1
            vm["batt"] -= E[i] / ETA + BETA * dN[vm["pos"], node]
            vm["km"] += dN[vm["pos"], node]
            vm["pos"], vm["free"] = node, c
            vm["count"] += 1
            vm["route"].append(node)
            res[i] = f"van{choice[1] + 1}"
    for vm in van:
        if vm["count"]:
            vm["km"] += dN[vm["pos"], 0]
            vm["batt"] -= BETA * dN[vm["pos"], 0]
    k = COSTS
    fcs = [i for i in range(n) if res[i].startswith("FCS")]
    vanev = [i for i in range(n) if res[i].startswith("van")]
    Z1 = k["c_EV"] * sum(dEF[i, int(res[i][3:res[i].index("-")]) - 1] for i in fcs)
    Z2 = k["c_M"] * sum(vm["km"] for vm in van)
    Z3 = k["c_F"] * sum(1 for vm in van if vm["count"])
    Z4 = k["pi_F"] * sum(E[i] for i in fcs) + k["pi_M"] * sum(E[i] / ETA for i in vanev)
    Z5 = k["c_W"] * float(np.sum(w * (C - r)))
    Z = Z1 + Z2 + Z3 + Z4 + Z5
    return True, Z, {"Z1": Z1, "Z2": Z2, "Z3": Z3, "Z4": Z4, "Z5": Z5,
                     "resource": res, "S": S.tolist(), "C": C.tolist(),
                     "van_routes": [[0] + vm["route"] + [0] for vm in van if vm["count"]],
                     "van_battery_left": [round(vm["batt"], 2) for vm in van]}


def write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        wr.writerows(rows)


def build(raw, name, n, out_root):
    rng = np.random.default_rng(SEED_BASE + 1000 * INSTANCES.index(name) + n)
    depot = raw[0]
    if n <= 100:
        cust = raw[1:n + 1]
    else:
        # all 100 Solomon customers, plus n - 100 synthesised EVs: each copies a randomly
        # chosen Solomon customer (demand, ready time) at a jittered position
        syn = np.random.default_rng(SEED_BASE + 1000 * INSTANCES.index(name) + n + 7)
        extra = raw[1:][syn.integers(0, 100, n - 100)].copy()
        extra[:, 1:3] = np.clip(extra[:, 1:3] + syn.normal(0, JITTER, (n - 100, 2)), 0, 100)
        cust = np.vstack([raw[1:], extra])
    xy = cust[:, 1:3] * LAMBDA                  # km
    dep_xy = depot[1:3] * LAMBDA
    # energy: Solomon demand (1..50) mapped linearly to [E_MIN, E_MAX] kWh
    E = E_MIN + (E_MAX - E_MIN) * (np.clip(cust[:, 3], 1, 50) - 1) / 49.0
    # release: Solomon READY TIME scaled from [0, depot due date] to [0, R_MAX_RELEASE]
    r = cust[:, 4] / depot[5] * R_MAX_RELEASE
    F = N_FCS[n]
    fcs_xy = kmeans(xy, F, rng)
    tauF = 60 * E / P_F
    tauM = 60 * E / P_M
    dEF = RHO * np.linalg.norm(xy[:, None, :] - fcs_xy[None], axis=2)
    # remaining range: most EVs U[RANGE_MIN, RANGE_MAX] km; a share of "stranded" EVs
    # gets a range below the distance to its nearest station, so only a van can serve it
    R = rng.uniform(RANGE_MIN, RANGE_MAX, n)
    stranded = rng.choice(n, size=int(round(STRANDED_SHARE * n)), replace=False)
    R[stranded] = rng.uniform(0.2, 0.8, len(stranded)) * dEF[stranded].min(1)
    w = 1 + KAPPA * (1 - R / R.max())
    tEF = 60 * dEF / V
    nodes = np.vstack([dep_xy, xy])            # 0 = depot, 1..n = EVs
    dN = RHO * np.linalg.norm(nodes[:, None, :] - nodes[None], axis=2)
    tN = 60 * dN / V
    # round once, so the stored files are exactly the data the plan was built from
    E, r, R, tauF, tauM = (np.round(a, 4) for a in (E, r, R, tauF, tauM))
    w = np.round(w, 5)
    dEF, tEF, dN, tN = (np.round(a, 4) for a in (dEF, tEF, dN, tN))
    inst = {"name": f"{name.upper()}_n{n}", "n": n, "n_fcs": F, "n_vans": N_VANS[n],
            "r": r, "E": E, "R": R, "w": w, "tauF": tauF, "tauM": tauM,
            "d_ev_fcs": dEF, "t_ev_fcs": tEF, "d_nodes": dN, "t_nodes": tN}
    ok, Z, det = greedy_plan(inst)
    # if the greedy rule fails, add vans until it finds a feasible plan
    while not ok:
        inst["n_vans"] += 1
        ok, Z, det = greedy_plan(inst)
    M = inst["n_vans"]

    d = os.path.join(out_root, inst["name"])
    os.makedirs(d, exist_ok=True)
    nreach = (dEF <= R[:, None]).sum(1)
    write_csv(os.path.join(d, "evs.csv"),
              ["ev", "solomon_customer", "x_km", "y_km", "E_kWh", "r_min", "R_km", "w",
               "tauF_min", "tauM_min", "reachable_fcs"],
              [[i + 1, int(cust[i, 0]), round(xy[i, 0], 4), round(xy[i, 1], 4),
                E[i], r[i], R[i], w[i],
                tauF[i], tauM[i], int(nreach[i])] for i in range(n)])
    write_csv(os.path.join(d, "stations.csv"), ["fcs", "x_km", "y_km", "guns", "P_kW"],
              [[f + 1, round(fcs_xy[f, 0], 3), round(fcs_xy[f, 1], 3), GUNS, P_F]
               for f in range(F)])
    write_csv(os.path.join(d, "vans.csv"), ["van", "B_kWh", "Q", "P_kW", "fixed_cost_Rs"],
              [[m + 1, B_VAN, Q_VAN, P_M, COSTS["c_F"]] for m in range(M)])
    write_csv(os.path.join(d, "depot.csv"), ["x_km", "y_km"],
              [[round(dep_xy[0], 3), round(dep_xy[1], 3)]])
    fh = ["ev"] + [f"fcs{f + 1}" for f in range(F)]
    write_csv(os.path.join(d, "dist_ev_fcs_km.csv"), fh,
              [[i + 1] + [round(x, 4) for x in dEF[i]] for i in range(n)])
    write_csv(os.path.join(d, "time_ev_fcs_min.csv"), fh,
              [[i + 1] + [round(x, 4) for x in tEF[i]] for i in range(n)])
    nh = ["node"] + list(range(n + 1))
    write_csv(os.path.join(d, "dist_nodes_km.csv"), nh,
              [[a] + [round(x, 4) for x in dN[a]] for a in range(n + 1)])
    write_csv(os.path.join(d, "time_nodes_min.csv"), nh,
              [[a] + [round(x, 4) for x in tN[a]] for a in range(n + 1)])
    params = {"T_min": T, "v_kmh": V, "beta_kWh_per_km": BETA, "eta": ETA,
              "P_F_kW": P_F, "P_M_kW": P_M, "guns_per_fcs": GUNS,
              "B_kWh": B_VAN, "Q": Q_VAN, "kappa": KAPPA, "R_max_km": float(R.max()),
              "costs_Rs": COSTS, "lambda_km_per_unit": LAMBDA, "rho_detour": RHO,
              "source": f"Solomon {name.upper()} (first {n} customers)",
              "seed": int(SEED_BASE + 1000 * INSTANCES.index(name) + n),
              "node_numbering": "0 = depot (start and end), 1..n = EVs; n+1 in the model is node 0 again"}
    full = {"name": inst["name"], "n_ev": n, "n_fcs": F, "n_vans": M, "params": params,
            "depot_xy_km": dep_xy.round(4).tolist(), "fcs_xy_km": fcs_xy.round(4).tolist(),
            "ev": {"xy_km": xy.round(4).tolist(), "E_kWh": E.round(4).tolist(),
                   "r_min": r.round(4).tolist(), "R_km": R.round(4).tolist(),
                   "w": w.round(5).tolist(), "tauF_min": tauF.round(4).tolist(),
                   "tauM_min": tauM.round(4).tolist()},
            "d_ev_fcs_km": dEF.round(4).tolist(), "t_ev_fcs_min": tEF.round(4).tolist(),
            "d_nodes_km": dN.round(4).tolist(), "t_nodes_min": tN.round(4).tolist(),
            "greedy_baseline": {"feasible": ok, "Z_Rs": round(Z, 2),
                                **{k: (round(v, 2) if isinstance(v, float) else v)
                                   for k, v in det.items()}}}
    with open(os.path.join(d, "instance.json"), "w") as fh_:
        json.dump(full, fh_, indent=1 if n <= 100 else None)   # compact for large files
    van_only = int((nreach == 0).sum())
    return {"instance": inst["name"], "n_ev": n, "n_fcs": F, "guns": F * GUNS,
            "n_vans": M, "van_only_evs": van_only,
            "total_E_kWh": round(float(E.sum()), 1),
            "fcs_work_min_if_all_fcs": round(float(tauF.sum()), 1),
            "greedy_feasible": ok, "greedy_Z_Rs": round(float(Z), 2),
            "greedy_evs_on_vans": sum(1 for x in det["resource"] if x.startswith("van")),
            "greedy_vans_used": len(det["van_routes"])}


def main(raw_dir, out_root, sizes):
    os.makedirs(out_root, exist_ok=True)
    built = []
    for name in INSTANCES:
        raw = read_solomon(os.path.join(raw_dir, f"{name}.txt"))
        for n in sizes:
            built.append(build(raw, name, n, out_root))
            print(built[-1])
    # merge into summary.csv (keep rows of sizes not rebuilt now)
    path = os.path.join(out_root, "summary.csv")
    rows = {}
    if os.path.exists(path):
        with open(path) as fh:
            rows = {r["instance"]: r for r in csv.DictReader(fh)}
    rows.update({s["instance"]: s for s in built})
    keys = list(built[0])
    order = sorted(rows, key=lambda k: (INSTANCES.index(k.split("_n")[0].lower()),
                                        int(k.split("_n")[1])))
    write_csv(path, keys, [[rows[k][c] for c in keys] for k in order])


if __name__ == "__main__":
    # python make_dataset.py <raw_dir> <out_dir> [n ...]   (default: all sizes)
    chosen = [int(a) for a in sys.argv[3:]] or SIZES + LARGE_SIZES
    main(sys.argv[1], sys.argv[2], chosen)
