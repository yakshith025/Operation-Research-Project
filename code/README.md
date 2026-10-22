# Code

| File | What it does |
|---|---|
| `make_ev_solomon.py` | Builds the **EV-Solomon dataset** (`../dataset_ev_solomon`) from the Solomon and Gehring & Homberger files, and contains `read_ev_solomon(file, n)`, the reader used by the solver. |
| `solve_milp.py` | Gurobi model of `main_formulation`, equations (1)–(20), one-to-one. Reads EV-Solomon files, solves each one at the chosen sizes, validates every solution and writes result tables. |
| `check_plan.py` | Checks a plan against every constraint of `main_formulation` and recomputes Z from scratch. Used for the greedy plans and for every MILP solution. |
| `make_dataset.py` | Older generator for `../dataset` (superseded by EV-Solomon). |

## Requirements

Python 3.10+, `numpy`, `gurobipy` (13.x, with a licence: the free pip licence is too small for n ≥ 25),
`openpyxl` (optional, for `results.xlsx`). Install with `pip install -r requirements.txt`.

## Running the MILP

An **instance** is one EV-Solomon file plus a size n: its **first n EVs** (Solomon's convention). By default every
100-EV file (Solomon) is solved at n = 25, 50, 100 and every 200-EV file (Homberger) at n = 50, 100, 200.

```bash
cd code
python solve_milp.py                                         # every file at its default sizes (348 instances)
python solve_milp.py --file-size 100 --sizes 25              # the 56 Solomon files, first 25 EVs
python solve_milp.py --only EV-R101 EV-C101 --sizes 10 25 50  # chosen files and sizes (any n works)
python solve_milp.py --families C1 R1 --sizes 25 --details    # families, print all tables
python solve_milp.py --resume                                # continue after an interruption
```

On a server, run it in the background so it survives logout, for example:

```bash
nohup python solve_milp.py --time-limit 600 --threads 16 > run.log 2>&1 &
```

Instances are solved from small to large. `summary.csv` and `costs.csv` are rewritten after every instance, so
partial results are kept; `report.md` and `results.xlsx` are written at the end (or re-create them any time with
`--reuse`). If one instance fails (e.g. out of memory), it is recorded as `error` in the summary and the run continues.

Every instance is solved with the complete `main_formulation`, equations (1)–(20).

**Infeasible instances.** The dataset keeps Solomon's exact timing, so some R1 / RC1 instances cannot serve every EV
by T. If the data already proves this (an EV that cannot be charged by T even in the best case), the instance is
reported as `infeasible (proven: EVs …)` without building the model; `--solve-infeasible` runs Gurobi on them anyway.
Instances that are not provably infeasible are always solved, and Gurobi decides (status `optimal`, `time limit`
or `infeasible`). The greedy plan is used as a warm start when it exists (`greedy_Z_Rs` empty = none found).

| Option | Meaning |
|---|---|
| `instances` (positional) | EV-Solomon instances folder (default `../dataset_ev_solomon/instances`) |
| `--only NAME ...` | only these files (e.g. `EV-C101 EV-R1_2_1`) |
| `--families F ...` | only these families (`C1 C2 R1 R2 RC1 RC2`) |
| `--file-size 100/200` | only Solomon (100-EV) or Homberger (200-EV) files |
| `--sizes N ...` | number of EVs to solve (first n of each file) |
| `--time-limit S` | seconds per instance (default 300) |
| `--mip-gap G` | relative gap to stop at (default 1e-4 = 0.01 %) |
| `--threads K` | CPU threads (default 0 = all cores) |
| `--solve-infeasible` | also run Gurobi on instances the data proves infeasible |
| `--gpu` | also solve the LP relaxation with PDHG on the GPU (see below) |
| `--no-warm-start` | do not give Gurobi the greedy plan as a starting solution |
| `--resume` | skip instances that already have `solution.json` in `--out` |
| `--reuse` | do not solve at all; rebuild all tables from existing `solution.json` files |
| `--mip-focus F` | Gurobi MIPFocus (0 default, 1 = good solutions first) |
| `--node-file-dir DIR` | write Gurobi node files to DIR to save RAM |
| `--details` | print per-EV, per-van and per-gun tables |
| `--out DIR` | output folder (default `../results_ev_solomon`) |

## Output (in `../results_ev_solomon`)

| File | Content |
|---|---|
| `summary.csv` | one row per instance: family, EVs, T, stations, guns, vans, graph nodes, model size, status, Z, best bound, gap, runtime, nodes explored, greedy Z, improvement over greedy, validity |
| `costs.csv` | Z1 … Z5, operating cost (Z1–Z4), EVs at FCS / by van, vans used, mean and max time until charged, van km |
| `<instance>/evs.csv` | per EV: resource (gun or van), start, end, time until charged, km driven, inconvenience cost |
| `<instance>/vans.csv` | per van: route, stops, km, kWh delivered and drawn, battery left, return time, slack to T |
| `<instance>/guns.csv` | per gun: EVs in order, busy minutes, utilisation |
| `<instance>/gurobi.log`, `solution.json` | full solver log and solution |
| `report.md`, `results.xlsx` | all of the above in one Markdown file and one Excel workbook |

Every solution is re-checked by `check_plan.validate` (column `valid`) and its Z is recomputed independently.

## About the GPU

Gurobi's GPU support is the **PDHG** algorithm for **linear** programs. Branch-and-bound for a MILP always runs
on the CPU. With `--gpu`, the script solves the LP relaxation of the model with PDHG (`Method=6`, `PDHGGPU=1`)
and reports its bound (`gpu_root_LP_bound`) and whether the GPU was really used (`gpu_used`).
The GPU is only used by a **GPU build** of Gurobi (Linux with an NVIDIA data-centre GPU); the standard
`pip install gurobipy` is not a GPU build, so Gurobi prints "Not a GPU build" and runs PDHG on the CPU.
