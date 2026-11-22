# Code

| File | What it does |
|---|---|
| `make_ev_solomon.py` | Builds the **EV-Solomon dataset** (`../dataset_ev_solomon`) from the Solomon and Gehring & Homberger files. Also the readers: `EvSolomonData(folder)` lists the files and reads one (first n EVs) from the **CSV form** (`dataset_ev_solomon/csv`, the solver's input) or from the text files; `read_ev_solomon(file, n)` reads one text file. Both give the same instance. |
| `export_csv.py` | Writes the same data as CSV: `../dataset_ev_solomon/csv/` (data + config file per source). This is what the solver and RL read; re-run it after rebuilding the text files. |
| `solve_milp_f3.py` | Gurobi model of **`main_formulation_3`** (income θ ∈ [0, 1] per EV), equations (1)–(23). Reads `../dataset_ev_solomon/csv` (or the text files, see *Input*). Every instance goes to Gurobi (nothing skipped beforehand); `--workers` solves several at once; validates every solution, writes everything to `../new_results` with the income analysis. See *Formulation 3* below. |
| `make_income.py` | Income data for formulation 3: θ per EV in `../dataset_ev_solomon/income/<FILE>.csv` (Beta draw), and the sensitivities μ(θ), c^a(θ), W(θ). Called by the solver automatically. |
| `f3_params.json` | All income constants of formulation 3 (θ distribution, μ, values of time, queue tolerance, tariffs). Placeholder values: change them here, no code change needed. |
| `check_plan_f3.py` | Checks a plan against every constraint of `main_formulation_3` (incl. the queue limit W_i) and recomputes Z1 … Z6. |
| `analyze_f3.py` | Income insights from a run: share served by a van per income band, peak vs off-peak, waits, perceived cost, logistic regression, figures. |
| `results_io.py` | **Output schema shared by every method** (MILP, greedy, RL): `runs.csv` + `plan.csv` + `meta.json` + `logs.zip`; `evaluate(J, plan)` scores any plan with the same checker. See *Output schema* below. |
| `convert_results_f3.py` | Converts a results folder of the old layout (one folder per instance) to the shared schema, without solving. |
| `run_f3.sh` | Server script: 56 Solomon files at n = 10…100, 300 s, 8 workers × 4 threads (32 cores), all output and the report in `../new_results`. |

## Requirements

Python 3.10+, `numpy`, `gurobipy` (13.x, with a licence: the free pip licence is too small for n ≥ 25),
`openpyxl` (optional, for `results.xlsx`). Install with `pip install -r requirements.txt`.

## About the GPU

Gurobi's GPU support is the **PDHG** algorithm for **linear** programs. Branch-and-bound for a MILP always runs
on the CPU. With `--gpu`, the script solves the LP relaxation of the model with PDHG (`Method=6`, `PDHGGPU=1`)
and reports its bound (`gpu_root_LP_bound`) and whether the GPU was really used (`gpu_used`).
The GPU is only used by a **GPU build** of Gurobi (Linux with an NVIDIA data-centre GPU); the standard
`pip install gurobipy` is not a GPU build, so Gurobi prints "Not a GPU build" and runs PDHG on the CPU.

## Formulation 3: income θ ∈ [0, 1] (`solve_milp_f3.py`)

`main_formulation_3.pdf` replaces the three customer classes of formulation 2 with a continuous
income index θ_i per EV (0 = lowest, 1 = highest income). θ sets each customer's money sensitivity
μ(θ) (falls with income), values of time c^D, c^Q, c^C, c^V (rise with income) and queue tolerance
W(θ) (falls with income). The objective is the operator's cost plus the customers' perceived cost,
so who is served at an FCS and who by a van follows from income, costs and capacity.

```bash
cd code
bash run_f3.sh                                             # server: 56 Solomon files, n = 10..100, 300 s, 8 x 4 threads
python solve_milp_f3.py --file-size 100 --workers 8 --threads 4   # the same without the report step
python solve_milp_f3.py --only EV-R101 --sizes 25 --details        # one instance, printed in full
python solve_milp_f3.py --resume ...                               # continue after an interruption
python solve_milp_f3.py --reuse                                    # re-score every stored plan (after a checker change)
python solve_milp_f3.py ../dataset_ev_solomon/instances --only EV-R101 --sizes 25   # read the text files instead
python analyze_f3.py ../new_results [RUN_ID]               # re-make the income analysis only
cd ../report_f3 && python prepare_report_f3.py ../new_results       # rebuild the report in new_results/report
```

**Input.** By default the solver reads the CSV form of the dataset, `../dataset_ev_solomon/csv/`:
`solomon_config.csv` + `solomon_data.csv` (56 files × 100 EVs) and `gehring_homberger_config.csv` +
`gehring_homberger_data.csv` (60 files × 200 EVs). A "file" (`--only`, `instance` in the results) is one
value of the CSV column `instance`, e.g. `EV-C101`; an instance is its first n EVs. The config row gives the
scalars (T, speed, costs, vans), the data rows the depot, stations and EVs; distances and times are derived
exactly as from the text files (`make_ev_solomon.model_inputs`). Giving `../dataset_ev_solomon/instances`
reads the text files instead: all 116 files at every size n give identical instances, greedy plans and
task lists in both forms (checked 22 Nov 2026). θ comes from `../dataset_ev_solomon/income/` in both cases.
A file with incomplete rows becomes an `error` run with the reason; the other files still run.

**Every instance is solved by Gurobi.** Nothing is filtered or decided before the solver: no
feasibility pre-check, no replacement of the solver's plan. The status in `results.csv` is
Gurobi's verdict (optimal / time limit / infeasible / ...), and `valid` is the independent check of
the plan Gurobi returned. The greedy plan is only offered as a MIP start (`--no-warm-start` turns
that off); it is also stored as a run of its own (method `greedy`) for comparison. The run never
stops because of one instance: an error becomes that run's status `error` (message column,
traceback in `logs.zip`), and the next instance starts. With `--workers W` each instance runs in
its own process, so a crash (e.g. out of memory) only affects that instance; when the Gurobi
licence has no free session the worker waits and retries (`--licence-wait`, default 1 h). Each
finished instance is appended to `runs.csv` / `plan.csv` at once; if a table is locked (open in
Excel) the result waits in `_parts/` and is added by the next merge or the next run. Re-running with
`--resume` skips instances whose run (same run id and seed) is already in `runs.csv`.

| Option | Meaning |
|---|---|
| `DATA` (first argument) | dataset folder: `../dataset_ev_solomon/csv` (default) or `../dataset_ev_solomon/instances` (text files) |
| `--file-size 100/200`, `--families`, `--only`, `--sizes` | which instances (default sizes 10, 20, …, file size) |
| `--time-limit S`, `--mip-gap G`, `--mip-focus F` | Gurobi settings per instance (default 300 s, 1e-4) |
| `--workers W` | instances solved at the same time, each in its own process (default 1) |
| `--threads K` | Gurobi threads per instance (default: cores / workers) |
| `--mem-limit-gb X` | Gurobi SoftMemLimit per instance: stops cleanly at X GB instead of being killed |
| `--out DIR` | output folder (default `../new_results`) |
| `--run-id ID`, `--seed S` | name of this MILP run in the tables (default `milp_<time limit>s`) and Gurobi seed |
| `--resume`, `--reuse`, `--fresh` | continue / re-score the plans in `plan.csv` / first remove every run of this run id |
| `--params FILE` | income constants (default `f3_params.json`) |
| `--no-warm-start` | do not give Gurobi the greedy plan as a starting solution |
| `--gpu` | also solve the LP relaxation with PDHG on the GPU (see *About the GPU*) |
| `--node-file-dir DIR` | write Gurobi node files to DIR to save RAM |
| `--no-analysis`, `--details`, `--verbose` | skip the income analysis / print per-EV and van tables / show the Gurobi log |

The first run writes `../dataset_ev_solomon/income/<FILE>.csv` (θ per EV; the same EV number has
the same θ in every file of a class). Changing `theta_distribution` in `f3_params.json` rewrites them.

**Output (all in `../new_results`)**: `runs.csv`, `plan.csv`, `meta.json`, `logs.zip` (see *Output
schema*), `analysis/` (income tables, `ev_level.csv`, `insights.md`, `key_numbers.json`, figures)
and `report/` (the report, see below).

### Output schema (`results_io.py`, the same for MILP, greedy and RL)

Four files, whatever the number of methods or instances. The keys are those of the input CSVs
(`dataset_ev_solomon/csv/`): `instance` = file name (e.g. `EV-C101`), `n` = its first n EVs, and in
`plan.csv` `id` = EV or van number.

| File | One row per | Columns |
|---|---|---|
| `runs.csv` | run = (`run_id`, `method`, `seed`, `instance`, `n`) | `source`, `class`; `status`, `valid`, `violations`, `violation_detail`; `Z_total`, Z1–Z6, operator and customer cost; EVs at FCS / by van, stranded, free-choice EVs by van and their shares (all, peak, off-peak), mean θ of each group, vans used, van km, queue / van wait / time to charged; `solve_s`, `train_s` (RL), `wall_clock_s`; solver only: `best_bound`, `gap_%`, `solver_obj`, `nodes`, model size, `build_s`, `warm_start`, GPU bound; `message` |
| `plan.csv` | decision, `type` = `ev` or `van` | run key, `type`, `id`; ev: `resource_type` (FCS / MCS), `resource` (`FCS6-gun1` / `van2`), `seq` (position on the gun / in the van route), `start_min`, `end_min` (full precision), drive, queue, van wait, charge, time to charged, money paid, perceived money, time cost, customer cost, operator energy; van: `stops`, `km`, kWh delivered / drawn, battery left, `return_min` |
| `meta.json` | run id | settings (time limit, threads, seed, paths, `f3_params`, start / finish time) |
| `logs.zip` | instance | `<run_id>__<instance>_n<n>.gurobi.log`, `.log` (console), `.error.log` |

Status values: `optimal`, `time_limit`, `infeasible`, `<status>_no_plan` (e.g. `time_limit_no_plan`),
`error` (MILP); `plan` / `no_plan` (greedy, RL). `valid` is blank when there is no plan.

Rules: a method only produces a plan, and `results_io.evaluate(J, plan)` scores it with
`check_plan_f3.validate`, so every number means the same for every method. Invalid plans are kept
(`valid = no`). Inputs (θ, E, r, range, period) are not copied into the output;
`results_io.ev_attributes(J)` rebuilds them (the analysis joins them on `instance, n, id`). A van
route is its EV rows with `resource = van<k>` sorted by `seq`; `results_io.plan_from_rows` rebuilds
the plan from `plan.csv`, and it re-scores to exactly the stored Z.

An RL method writes into the same folder: per instance, `keys = {run_id, method, seed, instance,
source, class, n}`, `metrics, rows, _ = results_io.evaluate(J, plan)`, then
`results_io.write_part(out, [results_io.run_row(keys, "plan", metrics, solve_s=..., train_s=...)],
results_io.with_key(keys, rows))` and `results_io.merge_parts(out)`.

**Peak / off-peak** (analysis only, not in the model): load_i = requests released in
[r_i − mean τ^F, r_i]; EV i is in the peak if its load is above the instance median.

**Report** (built in `<results_dir>/report/`; `../report_f3` keeps only the template and the script):
`prepare_report_f3.py [results_dir]` copies the template and the formulation body from
`main_formulation_3.tex`, draws the income figures, writes every results table **and the insight
paragraphs** (`tables/insight_*.tex`, worded from the run's own numbers, e.g. van share per income
band, peak vs off-peak, short vs long horizon, logistic regression), then compiles `report_f3.pdf`.
Nothing in the report text is fixed in advance, so it is correct for whatever run it is built from.
