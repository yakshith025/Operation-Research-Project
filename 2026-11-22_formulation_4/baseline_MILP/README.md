# Baseline MILP of formulation 4 (Gurobi)

`code/solve_milp_f4.py` builds the complete model of `../formulation_4.pdf` (equations (1)–(23)) for every
instance and solves it with Gurobi. It is the formulation-3 solver (`../../code/solve_milp_f3.py`) with:

- input = `../data` (the formulation-4 dataset: segment and θ on every EV row), read by `../common/f4_data.py`;
- per-EV coefficients μ_i, c^D_i, c^Q_i, c^C_i, c^V_i, W_i from segment **and** income (`../common/f4_params.py`;
  commercial: c^D = c^C = c^V = downtime rate);
- `--set key=value` overrides and `--variant <id>`, used by the experiment folders (nothing is copied);
- result tables with per-segment columns (`../common/results_io_f4.py`) and the segment × income analysis
  (`../common/analyze_f4.py`) at the end;
- greedy warm start ordered by stranded, request time, then higher queue cost c^Q (formulation 3: higher θ).

Rules kept from formulation 3: every instance goes to Gurobi with the complete model (no feasibility pre-check,
no reduced model, no heuristic replacing the solver's plan); one failing instance never stops the run; results are
appended after every instance; `--resume` continues an interrupted run. Start times of the solver's plan are
recomputed as early as its decisions allow (removes big-M rounding noise, never raises Z), then the plan is scored
by the independent checker `../common/check_plan_f4.py`.

## Run (server, from `code/`)

```
python solve_milp_f4.py --workers 8 --threads 4               # 56 Solomon files x n = 10..100 = 560 instances, 300 s each
python solve_milp_f4.py --workers 8 --threads 4 --resume      # continue after an interruption
python solve_milp_f4.py --only EV-R101 --sizes 10 --details   # one small instance with the plan printed
python solve_milp_f4.py --file-size 100 200 ...               # also the 60 Gehring-Homberger files
```
Defaults: time limit 300 s, MIP gap 0.01 %, Gurobi seed 0, run id `milp_300s` (the experiments compare against this
run id), output `../results`.

## Output (`results/`)

| file | content |
|---|---|
| `runs.csv` | one row per run: method MILP (`milp_300s`) and method greedy (its warm start) per instance; status, valid, Z1–Z6, operator / customer cost, van shares overall and per segment, queues, downtime and money per segment, gap, bound, model size |
| `plan.csv` | one row per EV (segment, θ, stranded, resource, start / end, queue, van wait, downtime, money, perceived cost) and per van (stops, km, energy, return) |
| `meta.json` | settings, parameters, overrides |
| `logs.zip` | Gurobi log, console output, tracebacks |
| `analysis/milp_300s/` | `summary.md`, per segment, per segment × income band / class / size, income threshold θ* per segment, figures |

Tested on the laptop (22 Nov, small instances only, outside this folder): EV-C101 and EV-R101 at n = 10 and 20
optimal with valid plans; `--workers 2` with `--set config.n_vans=3 --set tariff_Rs_per_kWh.p_M=35` reaches the
worker processes (1,825 binaries with 3 vans vs 2,707 with 5). `results/` is empty until the server run.
