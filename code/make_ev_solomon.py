"""EV-Solomon: an EV-charging dataset for main_formulation, built on Solomon's VRPTW set.

One output file per source instance, in a Solomon-like text format:
  - 56 Solomon instances (C1, C2, R1, R2, RC1, RC2), depot + 100 EVs  -> instances/100/
  - 60 Gehring & Homberger instances (same classes), depot + 200 EVs  -> instances/200/
As in Solomon, smaller versions are the first n EVs of a file (e.g. 25 / 50 of a 100 file).

Usage
  python make_ev_solomon.py <out_dir> <source_dir> [<source_dir> ...]
  (every *.txt / *.TXT file in the source folders is converted)

Units: coordinates in Solomon grid units (km = grid x KM_PER_UNIT), road distance =
DETOUR x straight line, times in minutes, energy in kWh, money in Rs.

Within a Solomon class (e.g. R101-R112) customer positions and demands are identical and
only the time windows differ. EV-Solomon keeps that structure: positions, energy, stations,
ranges and priorities are the same for every instance of a class (seeded by class); only the
request times (from Solomon READY TIME) differ.
"""
import csv
import glob
import json
import math
import os
import re
import sys

import numpy as np

# ---------------------------------------------------------------- settings
KM_PER_UNIT = 0.2         # 100 x 100 grid -> 20 x 20 km city
DETOUR = 1.3              # road distance = DETOUR * straight line
T = 300.0                 # planning period, minutes (17:00-22:00 evening peak)
SPEED = 25.0              # km/h, all vehicles
R_MAX_RELEASE = 120.0     # request times scaled into [0, 120] min
E_MIN, E_MAX = 10.0, 40.0 # energy demand range, kWh (from Solomon DEMAND 1..50)
RANGE_MIN, RANGE_MAX = 5.0, 40.0
STRANDED_SHARE = 0.15     # EVs whose range is below the distance to their nearest station
KAPPA = 2.0               # priority w_i = 1 + KAPPA (1 - R_i / R_max)
GUNS, P_F = 2, 22.0
B_VAN, Q_VAN, P_M, BETA, ETA = 150.0, 6, 30.0, 0.25, 0.9
FLEET = {100: (12, 5), 200: (24, 12)}       # EVs -> (stations, vans). 200: 12 vans, not 10, because the
#                                           28x28 km city makes van trips longer (10 vans: 12 of 60 infeasible)
CHECK = {100: (25, 50, 100), 200: (50, 100, 200)}   # prefix sizes checked for feasibility
COSTS = {"C_EV": 4.0, "C_M": 15.0, "C_F": 1500.0, "PI_F": 8.7, "PI_M": 10.7,
         "C_W": round(200.0 / 60.0, 6)}
SEED_BASE = 2026


# ---------------------------------------------------------------- Solomon input
def read_solomon(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 7 and all(p.lstrip("-").isdigit() for p in parts):
                rows.append([int(p) for p in parts])
    arr = np.array(rows, dtype=float)
    assert arr[0, 0] == 0 and len(arr) - 1 in FLEET, f"{path}: unexpected size {len(arr) - 1}"
    return arr  # cust, x, y, demand, ready, due, service


def solomon_class(name):
    """'r105' -> 'R1', 'rc208' -> 'RC2', 'c101' -> 'C1', 'c1_2_3' -> 'C1' (Homberger)."""
    m = re.match(r"([A-Z]+)(\d)", name.upper())
    return m.group(1) + m.group(2)


def kmeans(points, k, rng, iters=200):
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


# ---------------------------------------------------------------- build one instance
def build(raw, name):
    cls = solomon_class(name)
    depot, cust = raw[0], raw[1:]
    N = len(cust)
    n_fcs, n_vans = FLEET[N]
    rng = np.random.default_rng(SEED_BASE + sum(ord(ch) for ch in cls) * 101 + N)   # per class+size
    xy = cust[:, 1:3]
    fcs = np.round(kmeans(xy, n_fcs, rng), 2)                       # grid units
    E = np.round(E_MIN + (E_MAX - E_MIN) * (np.clip(cust[:, 3], 1, 50) - 1) / 49.0, 2)
    r = np.round(cust[:, 4] / depot[5] * R_MAX_RELEASE, 2)          # differs within a class
    d_near = DETOUR * KM_PER_UNIT * np.linalg.norm(xy[:, None] - fcs[None], axis=2).min(1)
    R = rng.uniform(RANGE_MIN, RANGE_MAX, N)
    stranded = rng.choice(N, size=int(round(STRANDED_SHARE * N)), replace=False)
    R[stranded] = rng.uniform(0.2, 0.8, len(stranded)) * d_near[stranded]
    R = np.round(R, 2)
    return {"name": "EV-" + name.upper(), "solomon": name.upper(), "class": cls, "N": N,
            "n_vans": n_vans, "depot": depot[1:3], "xy": xy, "E": E, "r": r, "R": R, "fcs": fcs}


def write_ev_solomon(inst, path):
    """Solomon-style text file."""
    L = [inst["name"], "",
         "PLANNING", "HORIZON_T(min)  SPEED(km/h)  KM_PER_UNIT  DETOUR",
         f"  {T:<15g}{SPEED:<13g}{KM_PER_UNIT:<13g}{DETOUR:g}", "",
         "COSTS", "C_EV(Rs/km)  C_M(Rs/km)  C_F(Rs/van)  PI_F(Rs/kWh)  PI_M(Rs/kWh)  C_W(Rs/min)  KAPPA",
         f"  {COSTS['C_EV']:<11g}{COSTS['C_M']:<12g}{COSTS['C_F']:<13g}{COSTS['PI_F']:<14g}"
         f"{COSTS['PI_M']:<14g}{COSTS['C_W']:<13g}{KAPPA:g}", "",
         "MOBILE CHARGING VANS",
         "NUMBER  BATTERY(kWh)  MAX_EVS  POWER(kW)  USE(kWh/km)  EFFICIENCY",
         f"  {inst['n_vans']:<6d}{B_VAN:<14g}{Q_VAN:<9d}{P_M:<11g}{BETA:<13g}{ETA:g}", "",
         "FIXED CHARGING STATIONS",
         "STN NO.  XCOORD.  YCOORD.  GUNS  POWER(kW)"]
    for f, (x, y) in enumerate(inst["fcs"], start=1):
        L.append(f"  {f:<7d}{x:<9.2f}{y:<9.2f}{GUNS:<6d}{P_F:g}")
    L += ["", "EV REQUESTS",
          "EV NO.  XCOORD.  YCOORD.  ENERGY(kWh)  REQUEST(min)  RANGE(km)",
          f"  {0:<6d}{inst['depot'][0]:<9g}{inst['depot'][1]:<9g}{0:<13g}{0:<14g}{0:g}"]
    for i in range(inst["N"]):
        L.append(f"  {i + 1:<6d}{inst['xy'][i, 0]:<9g}{inst['xy'][i, 1]:<9g}"
                 f"{inst['E'][i]:<13.2f}{inst['r'][i]:<14.2f}{inst['R'][i]:.2f}")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


# ---------------------------------------------------------------- read back (for solvers)
def read_ev_solomon(path, n=None):
    """Parse an EV-Solomon file; keep the depot and the first n EVs. Returns a dict with
    all model inputs, including derived tau, w, F_i, distance and time matrices."""
    sec, P, out = None, {}, {"stations": [], "evs": []}
    header_next = False
    for line in open(path):
        s = line.strip()
        if not s:
            continue
        if s in ("PLANNING", "COSTS", "MOBILE CHARGING VANS", "FIXED CHARGING STATIONS",
                 "EV REQUESTS"):
            sec, header_next = s, True
            continue
        if sec is None:
            out["name"] = s
            continue
        if header_next:
            header_next = False
            continue
        vals = [float(v) for v in s.split()]
        if sec == "PLANNING":
            P.update(dict(zip(["T", "speed", "km_per_unit", "detour"], vals)))
        elif sec == "COSTS":
            P.update(dict(zip(["c_EV", "c_M", "c_F", "pi_F", "pi_M", "c_W", "kappa"], vals)))
        elif sec == "MOBILE CHARGING VANS":
            P.update(dict(zip(["n_vans", "B", "Q", "P_M", "beta", "eta"], vals)))
        elif sec == "FIXED CHARGING STATIONS":
            out["stations"].append(vals)
        else:
            out["evs"].append(vals)
    dep = np.array(out["evs"][0][1:3])
    n = n or len(out["evs"]) - 1                  # default: all EVs in the file
    ev = np.array(out["evs"][1:n + 1])
    st = np.array(out["stations"])
    k = P["detour"] * P["km_per_unit"]
    xy, fxy = ev[:, 1:3], st[:, 1:3]
    d_ev_fcs = k * np.linalg.norm(xy[:, None] - fxy[None], axis=2)
    nodes = np.vstack([dep, xy])
    d_nodes = k * np.linalg.norm(nodes[:, None] - nodes[None], axis=2)
    E, r, R = ev[:, 3], ev[:, 4], ev[:, 5]
    return {"name": f"{out['name']}_n{n}", "n": n, "n_fcs": len(st), "params": P,
            "guns": st[:, 3].astype(int), "P_F": st[:, 4],
            "E": E, "r": r, "R": R, "w": 1 + P["kappa"] * (1 - R / R.max()),
            "tauF": 60 * E / st[0, 4], "tauM": 60 * E / P["P_M"],
            "d_ev_fcs": d_ev_fcs, "t_ev_fcs": 60 * d_ev_fcs / P["speed"],
            "d_nodes": d_nodes, "t_nodes": 60 * d_nodes / P["speed"]}


# ---------------------------------------------------------------- feasibility check
def greedy_feasible(D):
    """Simple rule obeying every constraint of main_formulation (incl. C_i <= T).
    Returns (feasible, Z of the plan)."""
    P = D["params"]
    n, F, M, T_ = D["n"], D["n_fcs"], int(P["n_vans"]), P["T"]
    r, E, w, R = D["r"], D["E"], D["w"], D["R"]
    tF, tM, dEF, tEF, dN, tN = D["tauF"], D["tauM"], D["d_ev_fcs"], D["t_ev_fcs"], D["d_nodes"], D["t_nodes"]
    reach = dEF <= R[:, None]
    gun_free = [np.zeros(g) for g in D["guns"]]
    van = [{"pos": 0, "free": 0.0, "batt": P["B"], "cnt": 0, "km": 0.0} for _ in range(M)]
    C = np.zeros(n)
    Z = 0.0
    for i in sorted(range(n), key=lambda q: (reach[q].any(), r[q], -w[q])):
        best = None
        for f in np.where(reach[i])[0]:
            for g in range(D["guns"][f]):
                s = max(r[i] + tEF[i, f], gun_free[f][g])
                if s + tF[i] <= T_ + 1e-9 and (best is None or s + tF[i] < best[0]):
                    best = (s + tF[i], ("gun", f, g))
        for m, v in enumerate(van):
            node = i + 1
            s = max(r[i], v["free"] + tN[v["pos"], node])
            need = E[i] / P["eta"] + P["beta"] * (dN[v["pos"], node] + dN[node, 0])
            if (v["cnt"] < P["Q"] and need <= v["batt"] + 1e-9
                    and s + tM[i] + tN[node, 0] <= T_ + 1e-9
                    and (best is None or s + tM[i] < best[0])):
                best = (s + tM[i], ("van", m))
        if best is None:
            return False, None
        C[i], ch = best
        if ch[0] == "gun":
            gun_free[ch[1]][ch[2]] = C[i]
            Z += P["c_EV"] * dEF[i, ch[1]] + P["pi_F"] * E[i]
        else:
            v, node = van[ch[1]], i + 1
            v["batt"] -= E[i] / P["eta"] + P["beta"] * dN[v["pos"], node]
            v["km"] += dN[v["pos"], node]
            v["pos"], v["free"], v["cnt"] = node, C[i], v["cnt"] + 1
            Z += P["pi_M"] * E[i] / P["eta"]
    for v in van:
        if v["cnt"]:
            v["km"] += dN[v["pos"], 0]
            Z += P["c_M"] * v["km"] + P["c_F"]
    Z += P["c_W"] * float(np.sum(w * (C - r)))
    return True, Z


# ---------------------------------------------------------------- main
def main(out, sources):
    rows = []
    files = []
    for src in sources:
        found = {os.path.normcase(f): f for pat in ("*.txt", "*.TXT")
                 for f in glob.glob(os.path.join(src, pat))}
        files += sorted(found.values())
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0].lower()
        inst = build(read_solomon(f), name)
        folder = os.path.join(out, "instances", str(inst["N"]))
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{inst['name']}.txt")
        write_ev_solomon(inst, path)
        row = {"instance": inst["name"], "source": inst["solomon"], "class": inst["class"],
               "EVs": inst["N"], "stations": FLEET[inst["N"]][0], "vans": inst["n_vans"]}
        for k, n in enumerate(CHECK[inst["N"]]):
            D = read_ev_solomon(path, n)                 # check what a solver will read
            ok, Z = greedy_feasible(D)
            row[f"check{k + 1}_n"] = n
            row[f"check{k + 1}_van_only"] = int((D["d_ev_fcs"] > D["R"][:, None]).all(1).sum())
            row[f"check{k + 1}_feasible"] = ok
            row[f"check{k + 1}_greedy_Z"] = round(float(Z), 2) if ok else ""
        rows.append(row)
        print(row["instance"], [(row[f"check{k}_n"], row[f"check{k}_feasible"]) for k in (1, 2, 3)],
              flush=True)
    with open(os.path.join(out, "summary.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    bad = [r["instance"] for r in rows if not all(r[f"check{k}_feasible"] for k in (1, 2, 3))]
    print(f"{len(rows)} instances written; infeasible checks: {bad or 'none'}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
