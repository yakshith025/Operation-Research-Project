# EV Charging Network Optimisation (CP 320 course project)

Plan public EV charging for one day: each request goes either to a **fixed station gun** or to a **mobile charging
van** that drives to the parked car, and the vans are routed. The cost counts the operator (van driving, dispatch,
energy) and each customer (driving, tariff, waiting), weighted by the customer's segment and income.
It is solved exactly with **Gurobi (MILP)** and approximately with a learned **attention policy (RL)**, then compared
on 560 benchmark instances.

## Where to start

| You want | Go to |
|---|---|
| The problem statement | [`docs/spec_problem_statement.pdf`](docs/spec_problem_statement.pdf) |
| The model | [`formulation_4/formulation_4.pdf`](formulation_4/formulation_4.pdf) |
| The results in one place | [`formulation_4/report/`](formulation_4/report/) and [`formulation_4/results_summary/`](formulation_4/results_summary/) |
| The RL algorithm explained | [`docs/rl_algorithm/`](docs/rl_algorithm/) (pseudocode, system model, flowcharts) |
| The slides | [`presentation/`](presentation/) |
| Code and how to run it | [`formulation_4/README.md`](formulation_4/README.md) |

## Layout

```
.
├── docs/                problem statement; rl_algorithm/ (pseudocode, diagrams, maths, code traceability)
├── dataset/             EV-Solomon instances built on Solomon (100 EVs) and Gehring-Homberger (200 EVs)
├── formulation_4/       the project: data, MILP and RL code, experiments E1–E10, results, reports
│   ├── methods/         milp/, rl/          baseline solvers with their results
│   ├── experiments/     E1 … E10            one factor at a time, MILP and/or RL
│   ├── results_summary/                     MILP vs RL tables, rubric tables
│   └── report/                              comprehensive reports (PDF)
└── presentation/        Beamer deck (presentation.tex) and slides_2026-11-27/
```

## Key results

- MILP (300 s per instance): 277 of 560 proven optimal; the rest hit the time limit or are infeasible.
- RL: 500 valid plans, on average 0.85 % costlier than the MILP and about 1,170× faster.
- Vans go mostly to high-income personal drivers; commercial drivers (taxis) queue at stations.

## Requirements

Python 3 with numpy, pandas, matplotlib, gurobipy (licence needed for the MILP) and torch (RL).
LaTeX for the reports and slides.

Team: Yakshith Maddineni, Kartikeya Gaur (IISc).
