# Code

| File | What it does |
|---|---|
| `make_dataset.py` | Builds the 12 instances in `../dataset/instances` from the Solomon files (see `../dataset/README.md`). |
| `check_plan.py` | Checks a plan against every constraint of `main_formulation` and recomputes Z from scratch. Used for the greedy baselines and for every MILP solution. |
| `solve_milp.py` | Gurobi model of `main_formulation`, equations (1)–(20), one-to-one. Solves the instances, validates each solution, writes result tables. |

## Requirements

Python 3.10+, `numpy`, `gurobipy` (13.x, with a licence: the free pip licence is too small for n ≥ 25), `openpyxl` (optional, for `results.xlsx`).

## Running the MILP

```bash
cd code
python solve_milp.py ../dataset/instances                                # all 39 instances (10 ... 1000 EVs)
python solve_milp.py ../dataset/instances --resume                       # continue after an interruption
python solve_milp.py ../dataset/instances --sizes 10 --details          # small instances, print all tables
python solve_milp.py ../dataset/instances --only C101_n25 --verbose      # one instance, show Gurobi log
```

On a server, run it in the background so it survives logout, for example:

```bash
nohup python solve_milp.py ../dataset/instances --time-limit 600 --threads 16 > run.log 2>&1 &
```

Instances are solved from small to large. `results/summary.csv` and `costs.csv` are rewritten after every
instance, so partial results are kept; `report.md` and `results.xlsx` are written at the end (or re-create
them any time with `--reuse`). If one instance fails (e.g. out of memory), it is recorded as `error` in the
summary and the run continues.

Every instance is solved with the complete `main_formulation`, equations (1)–(20).

Useful options for big runs: `--time-limit` (per instance), `--threads`, `--mip-focus 1` (concentrate on good
solutions rather than the bound), `--node-file-dir /tmp/grb` (spill the search tree to disk to save RAM).

| Option | Meaning |
|---|---|
| `--only NAME ...` | only these instances (e.g. `C101_n10 R101_n25`) |
| `--sizes N ...` | only these sizes (10, 25, 50, 100) |
| `--time-limit S` | seconds per instance (default 300) |
| `--mip-gap G` | relative gap to stop at (default 1e-4 = 0.01 %) |
| `--threads K` | CPU threads (default 0 = all cores) |
| `--gpu` | also solve the LP relaxation with PDHG on the GPU (see below) |
| `--no-warm-start` | do not give Gurobi the greedy plan as a starting solution |
| `--resume` | skip instances that already have `solution.json` in `--out` |
| `--reuse` | do not solve at all; rebuild all tables from existing `solution.json` files |
| `--mip-focus F` | Gurobi MIPFocus (0 default, 1 = good solutions first) |
| `--node-file-dir DIR` | write Gurobi node files to DIR to save RAM |
| `--details` | print per-EV, per-van and per-gun tables |
| `--out DIR` | output folder (default `../results`) |

## Output (in `../results`)

| File | Content |
|---|---|
| `summary.csv` | one row per instance: size, model size, status, Z, best bound, gap, runtime, nodes, greedy Z, improvement over greedy, validity, GPU columns |
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
The GPU is only used by a **GPU build** of Gurobi (Linux with an NVIDIA data-centre GPU); the standard Windows
`pip install gurobipy` is not a GPU build, so Gurobi prints "Not a GPU build" and runs PDHG on the CPU.
For models of this size, PDHG on a GPU would not make the MILP faster; it is mainly useful for very large LPs.
