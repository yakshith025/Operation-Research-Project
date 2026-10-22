# EV charging dataset (built from Solomon VRPTW instances)

Instances for `main_formulation.pdf`: EV → charger (FCS / MCS) assignment + mobile charger routing.
All times are in **minutes**, distances in **km**, energy in **kWh**, money in **Rs**.

Rebuild everything with:

```
python code/make_dataset.py dataset/solomon_raw dataset/instances
python code/check_plan.py dataset/instances
```

## Source

`solomon_raw/` holds three of Solomon's 1987 VRPTW instances (100 customers each), downloaded from SINTEF
(https://www.sintef.no/projectweb/top/vrptw/100-customers/, file `solomon-100.zip`):

| File | Type | Why |
|---|---|---|
| `c101.txt` | clustered customers | demand concentrated in a few areas |
| `r101.txt` | random customers | demand spread over the city |
| `rc101.txt` | mixed | both |

Each instance is cut to its first n = 10, 25, 50 and 100 customers (the usual Solomon convention),
giving **12 instances**: `C101_n10 … RC101_n100`.

## How Solomon data becomes EV data

| Model input | Taken from | Rule |
|---|---|---|
| Depot location | Solomon customer 0 | coordinates × λ |
| EV location | Solomon XCOORD, YCOORD | coordinates × λ, λ = 0.2 km per grid unit (100 × 100 grid → 20 × 20 km city) |
| E_i (kWh) | Solomon DEMAND (1–50) | linear map to [10, 40] kWh: E = 10 + 30·(DEMAND − 1)/49 |
| r_i (min) | Solomon READY TIME | scaled from [0, depot due date] to [0, 120] min |
| Solomon DUE DATE, SERVICE TIME | not used | the formulation has no due dates; charging time comes from E_i |
| FCS locations | **synthesised** | k-means centres of the n EV locations (seeded); 2, 3, 6, 12 stations for n = 10, 25, 50, 100 |
| Guns, power | problem statement | 2 guns per station, 22 kW each |
| R_i (km) | **synthesised** | U[5, 40] km; **15 % of EVs are "stranded"**: R_i = U[0.2, 0.8] × distance to their nearest station, so only a van can serve them |
| w_i | formula | w_i = 1 + κ(1 − R_i/R_max), κ = 2, R_max = largest R_i in the instance |
| τ^F_i, τ^M_i (min) | formula | 60·E_i/22 and 60·E_i/30 |
| Distances | formula | road distance = ρ × straight line, ρ = 1.3 (detour factor) |
| Travel times (min) | formula | 60 · distance / v, v = 25 km/h |

Fixed parameters (all instances): T = 300 min (17:00–22:00 evening peak), P^M = 30 kW, B_m = 150 kWh,
Q_m = 6, β = 0.25 kWh/km, η = 0.9, c^EV = 4 Rs/km, c^M = 15 Rs/km, c^F = 1500 Rs per van,
π^F = 8.7 Rs/kWh, π^M = 10.7 Rs/kWh, c^W = 200/60 = 3.333 Rs/min.
Vans: 1, 2, 3, 5 for n = 10, 25, 50, 100. Every value is in `instance.json → params`.

## Files in each instance folder

| File | Content |
|---|---|
| `evs.csv` | one row per EV: id, Solomon customer no., x_km, y_km, E_kWh, r_min, R_km, w, tauF_min, tauM_min, number of reachable stations (0 = van-only) |
| `stations.csv` | fcs id, x_km, y_km, guns, P_kW |
| `vans.csv` | van id, B_kWh, Q, P_kW, fixed cost |
| `depot.csv` | depot x_km, y_km |
| `dist_ev_fcs_km.csv`, `time_ev_fcs_min.csv` | d_if and t_if (rows = EVs, columns = stations) |
| `dist_nodes_km.csv`, `time_nodes_min.csv` | d_ij and t_ij between routing nodes: **0 = depot**, 1..n = EVs. Node n+1 of the model is the depot again, so use row/column 0 for it. |
| `instance.json` | everything above in one file, plus the greedy baseline plan |

## Feasibility and baseline

Every instance is **feasible** for `main_formulation` (including the hard deadline C_i ≤ T).
This is shown by a simple greedy plan stored in `instance.json → greedy_baseline`:
van-only EVs first, then the others by request time, each to the gun or van that finishes it earliest.
`code/check_plan.py` re-checks every plan against the constraints (range, arrival, no overlap on a gun,
van timing, battery, Q_m, return by T, every van EV visited once) and recomputes Z from scratch.

The greedy Z is an **upper bound**: the MILP optimum must be equal or lower. Use it to check a solver and as the
baseline for the RL comparison. See `summary.csv` for all instances.

| Instance | EVs | Stations (guns) | Vans | Van-only EVs | Greedy Z (Rs) |
|---|---|---|---|---|---|
| C101_n10 | 10 | 2 (4) | 1 | 2 | 6,471.81 |
| C101_n25 | 25 | 3 (6) | 2 | 4 | 20,875.66 |
| C101_n50 | 50 | 6 (12) | 3 | 8 | 44,859.80 |
| C101_n100 | 100 | 12 (24) | 5 | 15 | 97,187.08 |
| R101_n10 | 10 | 2 (4) | 1 | 2 | 7,368.03 |
| R101_n25 | 25 | 3 (6) | 2 | 4 | 22,824.16 |
| R101_n50 | 50 | 6 (12) | 3 | 8 | 46,271.87 |
| R101_n100 | 100 | 12 (24) | 5 | 15 | 91,889.16 |
| RC101_n10 | 10 | 2 (4) | 1 | 2 | 9,061.00 |
| RC101_n25 | 25 | 3 (6) | 2 | 4 | 26,503.83 |
| RC101_n50 | 50 | 6 (12) | 3 | 8 | 55,674.83 |
| RC101_n100 | 100 | 12 (24) | 5 | 15 | 99,490.69 |

## Large instances: 200 to 1000 EVs

Added 15 Oct 2026: n = 200, 300, …, 1000 for each of C101, R101, RC101 (27 instances, same file layout).

- **EVs:** all 100 Solomon customers, plus n − 100 synthesised EVs. Each synthesised EV copies a randomly
  chosen Solomon customer (its demand and ready time, so the same E_i and r_i rules apply) and is placed at that
  customer's position plus Gaussian noise (σ = 3 grid units = 0.6 km), clipped to the grid. Seeded. In `evs.csv`,
  EVs 101..n are the synthesised ones; `solomon_customer` gives the customer they were copied from.
- **Fleet scaled with the n = 100 ratios:** 12 stations (24 guns) and 5 vans per 100 EVs.
- Everything else (city size, T, ranges, 15 % stranded EVs, costs) is unchanged.
- `instance.json` is written without indentation for these sizes (n = 1000 is about 35 MB per instance).

| EVs | Stations | Guns | Vans | Van-only EVs |
|---|---|---|---|---|
| 200 | 24 | 48 | 10 | 30 |
| 300 | 36 | 72 | 15 | 45 |
| 400 | 48 | 96 | 20 | 59–60 |
| 500 | 60 | 120 | 25 | 74–75 |
| 600 | 72 | 144 | 30 | 89–90 |
| 700 | 84 | 168 | 35 | 105 |
| 800 | 96 | 192 | 40 | 119–120 |
| 900 | 108 | 216 | 45 | 135 |
| 1000 | 120 | 240 | 50 | 149 |

All 27 have a greedy plan that passes `check_plan.py` (greedy Z in `summary.csv`), so they are feasible.
The MILP has about n²·|M| route variables (≈ 50 million for n = 1000), so these sizes are meant for heuristics / RL;
an exact MILP is not practical beyond roughly n = 50–100.

Generate only some sizes with: `python code/make_dataset.py dataset/solomon_raw dataset/instances 200 300`.

## Changing the dataset

All choices are constants at the top of `code/make_dataset.py` (sizes, number of stations and vans,
λ, ρ, T, energy and range ranges, stranded share, costs). The random parts are seeded, so a rebuild gives the
same files. For the Analysis section, vary `N_VANS`, `N_FCS` or `E_MIN/E_MAX` and rebuild.
