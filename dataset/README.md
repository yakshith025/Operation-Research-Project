# EV-Solomon dataset

EV-charging instances for `main_formulation_3.pdf` (EV → FCS / MCS assignment + mobile charger routing),
built one-to-one on the classic VRPTW benchmarks:

| Folder | Source | Instances | EVs per file | Stations (guns) | Vans |
|---|---|---|---|---|---|
| `instances/100/` | Solomon (1987), `source/solomon_100/` | 56 (C1 9, C2 8, R1 12, R2 11, RC1 8, RC2 8) | 100 | 12 (24) | 5 |
| `instances/200/` | Gehring & Homberger (1999), `source/homberger_200/` | 60 (10 per class) | 200 | 24 (48) | 12 |

As in Solomon, smaller instances are the **first n EVs** of a file (25 or 50 of a 100 file; 50 or 100 of a 200 file).
Rebuild: `python code/make_ev_solomon.py dataset dataset/source/solomon_100 dataset/source/homberger_200`.

## File format (one text file per instance, Solomon style)

```
EV-R101

PLANNING
HORIZON_T(min)  SPEED(km/h)  KM_PER_UNIT  DETOUR
  230            15.6         0.2          1.3

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
| HORIZON_T | T | planning period (min) = Solomon's depot DUE DATE (differs by family); every EV fully charged and every van back by T |
| SPEED, KM_PER_UNIT, DETOUR | v, — | road distance (km) = DETOUR × KM_PER_UNIT × straight-line grid distance; time (min) = 60·distance/SPEED, which equals the straight-line grid distance (Solomon's rule: 1 grid unit = 1 minute; SPEED = 15.6 km/h) |
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

## CSV copies (`csv/`)

The same data as CSV, two files per source. Built by `python code/export_csv.py`, which copies the values exactly as written in the text files:

| File | Rows | Content |
|---|---|---|
| `solomon_data.csv` | 6328 (56 × (1 depot + 12 stations + 100 EVs)) | `instance, type, id, x, y, energy_kWh, request_min, range_km, guns, power_kW`; `type` = depot / station / ev |
| `solomon_config.csv` | 56 (one per file) | `instance, source, class, n_evs`, T, speed, km per unit, detour, all costs, kappa, van block (`n_vans`, battery, max EVs, power, use, efficiency), `n_stations, guns_per_station, total_guns, station_power_kW` |
| `gehring_homberger_data.csv` | 13500 (60 × (1 + 24 + 200)) | same columns |
| `gehring_homberger_config.csv` | 60 | same columns |

The energy/request/range columns are filled only for EV rows, and guns/power only for station rows. To get the first n EVs of a file, use `type == "ev"` and `id <= n`.

## How Solomon / Homberger data becomes EV data

| Source column | EV-Solomon | Rule |
|---|---|---|
| depot (customer 0) | depot | unchanged |
| XCOORD, YCOORD | EV location | unchanged (grid units). Solomon grid 100×100 → 20×20 km city; Homberger 140×140 → 28×28 km (same EV density) |
| DEMAND (1–50) | ENERGY | linear map to 10–40 kWh: E = 10 + 30·(DEMAND − 1)/49 |
| READY TIME | REQUEST | **unchanged** (minutes) |
| depot DUE DATE | HORIZON_T | **unchanged** (minutes): type-1 families have short days, type-2 long days, exactly as in Solomon |
| customer DUE DATE, SERVICE TIME | — | not used (no deadlines in the model; charging time comes from E/P) |
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

## Timing is exactly Solomon's (nothing extended)

Time unit = 1 minute. The planning period T is the source file's depot due date and every request time is the
source file's ready time, unchanged. Travel time follows Solomon's rule (1 grid unit of distance = 1 minute).

| Family | T in 100-EV files (Solomon) | T in 200-EV files (Homberger) |
|---|---|---|
| C1 | 1236 | 1351 |
| C2 | 3390 | 3598 |
| R1 | 230 | 634 |
| R2 | 1000 | 2535 |
| RC1 | 240 | 634 |
| RC2 | 960 | 2535 |

Vans and stations are the same for every family (5 vans / 12 stations per 100 EVs; 12 vans / 24 stations for 200).

## Feasibility (reported, not fixed)

Each file is checked at three sizes (n = 25, 50, 100 for 100-EV files; 50, 100, 200 for 200-EV files):

- **yes**: a simple plan that obeys every constraint of `main_formulation` was found (so the instance is feasible);
- **no (proven)**: at least one EV cannot be fully charged by T even in the best case (a free gun at a reachable
  station, or a van driving straight from the depot to it and back). The EV numbers are in `checkK_impossible_EVs`;
- **unknown**: the simple plan failed but no EV is provably impossible; only an exact solver can decide.

| Family | 25 EVs | 50 EVs | 100 EVs |
|---|---|---|---|
| C1, C2, R2, RC2 | all yes | all yes | all yes |
| R1 (T = 230) | 12 yes | 10 yes, 2 no | 0 yes, 7 no, 5 unknown |
| RC1 (T = 240) | 7 yes, 1 no | 5 yes, 1 no, 2 unknown | 0 yes, 4 no, 4 unknown |

200-EV files: C1, C2, R2, RC2 all yes at 50 / 100 / 200; R1 (T = 634): 10 yes at 50, 2 yes + 8 no at 100 and 200;
RC1 (T = 634): 9 yes + 1 no at every size.

The short type-1 days of R1 and RC1 (230 / 240 min) are too short for some late or far-away EVs to be charged
(charging alone takes 20–109 min), so those instances are infeasible under the hard rule C_i ≤ T. This is kept on
purpose, exactly as in Solomon. `checkK_greedy_Z` gives the cost of the simple plan where one was found; a MILP
optimum must be equal or lower.
