# Formulation 4: EV charging with customer segment and income

Assign each EV request to a **fixed station gun** (FCS) or a **mobile charging van** (MCS) and route the vans, minimising
operator cost plus the customers' perceived cost. Each EV carries a use **segment** (personal / commercial) and an
**income index** θ, which set its money sensitivity, value of time and queue tolerance.
The model is in [`formulation_4.pdf`](formulation_4.pdf).

## Folder

```
formulation_4/
├── formulation_4.tex / .pdf     the MILP: variables, objective Z1–Z6, constraints (2)–(23)
├── data/                        the dataset every solver reads (Solomon 56 × 100 EVs, Gehring-Homberger 60 × 200)
│                                + segment and θ per EV; make_f4_dataset.py builds it from ../dataset/csv
├── common/                      shared code: paths.py (folder layout), parameters, data reader, plan checker,
│                                result I/O, analysis, experiments.json, run_experiment.py, compare_experiment.py
├── methods/
│   ├── milp/                    Gurobi, complete model (code/solve_milp_f4.py, results/)
│   └── rl/                      attention policy + REINFORCE (code/, model/, results/, report/RL_code_explained.pdf)
├── experiments/                 one factor at a time; each has milp/ and/or rl/ (code/, results/, rl: model/)
│   ├── E1_van_tariff/  E2_van_fleet_and_seats/  E5_value_of_charging_time/
│   ├── E6_money_sensitivity/  E8_van_battery/
│   └── E9_station_capacity/  E10_charging_demand/            (RL only)
├── results_summary/             baseline MILP vs RL tables, rubric_results/ (performance, quality, sensitivities)
├── report/                      comprehensive_report_f4.pdf (team report) and comprehensive/ (MILP, RL, results reports)
├── analysis/evening_peak/       request times reshaped to add an evening peak (dataset + profiles; not solved yet)
├── job.sh, run_solomon.sh       run everything on the server (see below); README_solomon.md explains run_solomon.sh
└── job_progress.log             one line per step of the last full run
```

Every result folder has the same format: `runs.csv` (one row per instance), `plan.csv` (one row per EV),
`meta.json`, `logs.zip`, `analysis/<run id>/`. Experiments add `results/<variant>/` and `results/comparison/`.
Every plan is scored by the same independent checker (`common/check_plan_f4.py`).

## Experiments (`common/experiments.json`)

| # | factor | variants (baseline) |
|---|---|---|
| E1 | van tariff p^M | 15 / 35 Rs/kWh (25) |
| E2 | van fleet and seats | 3 / 7 vans (5); 4 / 8 seats (6) |
| E5 | value of charging time | personal c^{C,1} 7.5 / 12.5 (10); commercial ω^1 4.5 / 7.5 (6) |
| E6 | money sensitivity μ^0 | 1.125 / 1.875 per segment (1.5) |
| E8 | van battery | 200 kWh (150) |
| E9 | station capacity (RL) | 1 / 4 guns per station (2); 11 / 44 kW (22) |
| E10 | charging demand (RL) | energy per EV × 0.75 / 1.25 |

A variant is a set of `--set key=value` overrides applied in memory; no data is copied.

## Headline results (560 Solomon instances, n = 10–100)

- **MILP** (300 s): 277 proven optimal, 197 time limit with a plan, 35 time limit without a plan, 51 infeasible.
- **RL**: 500 valid plans, 0.85 % above the MILP on average (median 0.76 %), about 1,170× faster.
- RL is cheaper than the MILP on 28 instances, all of them MILP time-limit runs; it is never below Gurobi's
  proven lower bound and never beats a proven optimum.

## Running

```
python methods/milp/code/solve_milp_f4.py --only EV-C101 --sizes 10 --out <folder>   # one MILP instance
python methods/rl/code/solve_rl_f4.py --only EV-C101 --sizes 10 --out <folder>       # one RL instance
python experiments/E1_van_tariff/milp/code/run_E1_MILP.py --dry-run                  # show an experiment's commands
python common/make_rubric_tables.py                                                  # -> results_summary/rubric_results
```

The full job on the server (Python with numpy, gurobipy, torch; matplotlib optional):

```
DRY_RUN=1 bash job.sh                  # checks + list of steps, nothing solved
nohup bash job.sh > job.log 2>&1 &     # phase 1 MILP (8 × 4 threads), phase 2 RL, phase 3 comparisons
tail -f job.log                        # job_progress.log: one line per step; each folder keeps its own job.log
```

Reruns continue where they stopped (solved instances are skipped, a finished training is not repeated).
Settings are environment variables (`TL`, `WORKERS`, `THREADS`, `RL_DEVICE`, `EXPERIMENTS`, `SKIP_MILP=1`, ...);
see the header of `job.sh`.
