# EV-Solomon dataset

EV-charging instances for `main_formulation.pdf` (EV → FCS / MCS assignment + mobile charger routing),
built one-to-one on the classic VRPTW benchmarks:

| Folder | Source | Instances | EVs per file | Stations (guns) | Vans |
|---|---|---|---|---|---|
| `instances/100/` | Solomon (1987), `source/solomon_100/` | 56 (C1 9, C2 8, R1 12, R2 11, RC1 8, RC2 8) | 100 | 12 (24) | 5 |
| `instances/200/` | Gehring & Homberger (1999), `source/homberger_200/` | 60 (10 per class) | 200 | 24 (48) | 12 |

As in Solomon, smaller instances are the **first n EVs** of a file (25 or 50 of a 100 file; 50 or 100 of a 200 file).
Rebuild: `python code/make_ev_solomon.py dataset_ev_solomon dataset_ev_solomon/source/solomon_100 dataset_ev_solomon/source/homberger_200`.

## File format (one text file per instance, Solomon style)

```
EV-R101

PLANNING
HORIZON_T(min)  SPEED(km/h)  KM_PER_UNIT  DETOUR
  300            25           0.2          1.3

COSTS
C_EV(Rs/km)  C_M(Rs/km)  C_F(Rs/van)  PI_F(Rs/kWh)  PI_M(Rs/kWh)  C_W(Rs/min)  KAPPA
  4          15          1500         8.7           10.7          3.33333      2

MOBILE CHARGING VANS
NUMBER  BATTERY(kWh)  MAX_EVS  POWER(kW)  USE(kWh/km)  EFFICIENCY
  5     150           6        30         0.25         0.9

FIXED CHARGING STATIONS
STN NO.  XCOORD.  YCOORD.  GUNS  POWER(kW)
  1      25.79    25.71    2     22
  ...

EV REQUESTS
EV NO.  XCOORD.  YCOORD.  ENERGY(kWh)  REQUEST(min)  RANGE(km)
  0     35       35       0            0             0          <- depot (start and end of every van)
  1     41       49       ...
```

| Block / column | Model symbol | Meaning |
|---|---|---|
| HORIZON_T | T | planning period (min); every EV fully charged and every van back by T |
| SPEED, KM_PER_UNIT, DETOUR | v, — | road distance (km) = DETOUR × KM_PER_UNIT × straight-line grid distance; time (min) = 60·distance/SPEED |
| C_EV, C_M, C_F, PI_F, PI_M, C_W | c^EV, c^M, c^F_m, π^F, π^M, c^W | costs (C_W per minute) |
| KAPPA | κ | priority w_i = 1 + κ(1 − R_i / R_max), R_max = largest range in the instance (or prefix) |
| NUMBER, BATTERY, MAX_EVS, POWER, USE, EFFICIENCY | \|M\|, B_m, Q_m, P^M, β, η | van fleet |
| STN NO., XCOORD., YCOORD., GUNS, POWER | f, location, \|G_f\|, P^F | fixed charging stations |
| EV NO., XCOORD., YCOORD. | i, location | row 0 = depot, rows 1..N = EVs |
| ENERGY | E_i | kWh; τ^F_i = 60·E_i/P^F, τ^M_i = 60·E_i/P^M (min) |
| REQUEST | r_i | minutes |
| RANGE | R_i | km; F_i = stations with road distance ≤ R_i |

`code/make_ev_solomon.py` has `read_ev_solomon(path, n)`, which parses a file, keeps the first n EVs and returns all
model inputs (τ, w, F_i data, distance and time matrices) ready for a solver.

## How Solomon / Homberger data becomes EV data

| Source column | EV-Solomon | Rule |
|---|---|---|
| depot (customer 0) | depot | unchanged |
| XCOORD, YCOORD | EV location | unchanged (grid units). Solomon grid 100×100 → 20×20 km city; Homberger 140×140 → 28×28 km (same EV density) |
| DEMAND (1–50) | ENERGY | linear map to 10–40 kWh: E = 10 + 30·(DEMAND − 1)/49 |
| READY TIME | REQUEST | scaled from [0, depot due date] to [0, 120] min |
| DUE DATE, SERVICE TIME | — | not used (no deadlines in the model; charging time comes from E/P) |
| VEHICLE NUMBER, CAPACITY | — | replaced by the van block |
| — | stations | **synthesised**: k-means centres of the EV locations, 2 guns × 22 kW |
| — | RANGE | **synthesised**: U[5, 40] km; **15 % of EVs are "stranded"** (range = 20–80 % of the distance to their nearest station), so only a van can serve them |

**Class structure is kept.** In Solomon and Homberger all instances of a class share locations and demands and differ only
in time windows. Here too: within a class (and size), locations, energy, stations, ranges and priorities are identical and
**only the request times differ** (checked for all 12 class groups). Random parts are seeded per class and size, so a rebuild
gives identical files.

**Fleet size.** 12 stations and 5 vans per 100 EVs. For 200 EVs: 24 stations but **12 vans** (not 10), because the larger
28×28 km city with one central depot makes van trips longer; with 10 vans the simple feasibility check failed on 12 of the 60
files, with 12 vans on none.

## Feasibility

Every file is feasible for `main_formulation` (including the hard deadline C_i ≤ T) at three sizes: n = 25, 50, 100 for
100-EV files and n = 50, 100, 200 for 200-EV files. This is shown by a simple greedy plan (van-only EVs first, then by request
time, each to the gun or van that finishes it earliest) that obeys every constraint. Its cost is in `summary.csv`
(`checkK_greedy_Z`); a MILP optimum must be equal or lower. `checkK_van_only` = EVs that can reach no station.

| EVs | Classes | Van-only EVs | Greedy Z for the full file (Rs) |
|---|---|---|---|
| 100 | C1 / C2 / R1 / R2 / RC1 / RC2 | 15 | 83,546 – 107,429 |
| 200 | C1 / C2 / R1 / R2 / RC1 / RC2 | 30 | 194,383 – 233,494 |
