# Comparative results: parameter experiments with the MILP and RL1

Each experiment changes **one input at a time** and solves the 560 Solomon instances again with the two
finished methods: the formulation-3 MILP (`code/solve_milp_f3.py`, the same settings as the baseline run `milp_300s`)
and RL1 (`RL1_reinforcement_learning_method_1`, the trained model `model_best.pt`, epoch 153). The baseline value of
every factor is **not solved again**: the analysis takes it from `new_results/` (MILP) and the RL1 `results/`.
The solvers themselves are not changed; an experiment only changes their input files.

## The experiments (in run order)

| priority | experiment | baseline → variants | variant ids | question |
|---|---|---|---|---|
| 1 | **E5** value of charging time at θ = 1, c^{C,1} | 10 → 7.5, 12.5 Rs/min (±25 %) | `E5_cC1_7.5`, `E5_cC1_12.5` | How sensitive is the van share to the most influential assumed value? |
| 1 | **E6** money sensitivity at θ = 0, μ⁰ | 1.5 → 1.125, 1.875 (±25 %) | `E6_mu0_1.125`, `E6_mu0_1.875` | The same, for the coefficient with the largest elasticity on Z |
| 2 | **E1** van tariff p^M | 25 → 15, 35 Rs/kWh | `E1_pM_15`, `E1_pM_35` | Who gets vans, and at what cost to the operator and equity? |
| 2 | **E2** van fleet and seats, one at a time | vans 5 → 3, 7; seats 6 → 4, 8 | `E2_vans_3`, `E2_vans_7`, `E2_seats_4`, `E2_seats_8` | Where does another van or seat pay off, and where is capacity idle? |
| 2 | **E8** van battery upper bound B | 150 → 200 kWh | `E8_battery_200` | Is recharging worth modelling? If Z barely moves, drop it |
| 3 | **E4** guns per station | 2 → 1, 3 | `E4_guns_1`, `E4_guns_3` | Do guns fix peak queues, and who benefits? |
| 4 | **E7** income distribution | Beta(2,2) → Beta(2,5), Beta(5,2) | `E7_beta_2-5`, `E7_beta_5-2` | Does the van share follow the Table 25 estimate once re-solved? |
| 5 (optional) | **E3** horizon T, R1 and RC1 only | Solomon's T → ×1.25, ×1.5 | `E3_T_x1.25`, `E3_T_x1.5` | How much does the deadline cost, and how many infeasible instances become feasible? |

E3 is not run by default: the classes with a long horizon (C2, R2, RC2) already show the horizon effect in the
baseline. Run it with `--experiments E3` (only the 20 R1 / RC1 files = 200 instances per variant).

Every value is in `code/experiments.json` (change a value or add one there, no code change).

### How each change is made (`code/make_variants.py`)

A variant is a changed copy of the baseline inputs in `variants/<variant id>/`: `csv/solomon_config.csv` +
`csv/solomon_data.csv`, `income/<FILE>.csv` (θ per EV) and `f3_params.json`. Only the named input differs;
`python make_variants.py --all --check` reads every file through the solvers' own reader and confirms that
(all 17 variants pass).

- E5, E6, E1: one constant of `f3_params.json` (`value_of_time_Rs_per_min.C.theta1`, `money_sensitivity_mu.theta0`, `tariff_Rs_per_kWh.p_M`). All parameter checks of `make_income.check_params` still hold (p^M = 15 = p^F is allowed).
- E2, E8: one column of the config file for all 56 files (`n_vans`, `van_max_evs`, `van_battery_kWh`).
- E4: the guns of every station (data rows of type station).
- E7: θ is **re-mapped by quantiles**, θ_new = F⁻¹_{Beta(a,b)}(F_{Beta(2,2)}(θ)). Every EV keeps its income rank (the richest EV stays the richest), so the result shows the effect of the distribution's shape, not of a new random draw.
- E3: T_min × factor for the R1 and RC1 files.

## How to run (server)

The runner calls the two solvers as they are; it never stops on one failure and every step resumes, so the
same command continues an interrupted run.

1. Upload the code (Windows cmd, from the laptop):
   ```
   ssh romit@10.72.19.181 "mkdir -p /mnt/DataDrive_2/Yakshith/or_project/project/comparative_results"
   scp -r "E:\AUG-2026\02_CP320_OR_for_Mobility\project\comparative_results\code" romit@10.72.19.181:/mnt/DataDrive_2/Yakshith/or_project/project/comparative_results/
   ```
   It expects, next to `comparative_results/`: `code/`, `dataset_ev_solomon/`, `new_results/runs.csv` (only for the time
   projection) and the RL1 folder with `code/` and `model/model_best.pt`, either as
   `RL1_reinforcement_learning_method_1/` or as `rl1/RL1_reinforcement_learning_method_1/` (the server layout of
   22 Nov; found automatically, or give `--rl1-dir`).
2. On the server (`conda activate yakshith`, `cd .../project/comparative_results/code`):
   ```
   python run_experiments.py --dry-run          # selected variants, commands, projected time; nothing solved
   nohup bash run_experiments.sh 1 > run_p1.log 2>&1 &        # priority 1: E5, E6 (about 12 h)
   nohup bash run_experiments.sh "2 3 4" > run_p234.log 2>&1 &  # afterwards: the rest
   ```
   `run_experiments.sh` runs `run_experiments.py --priority ...` (MILP then RL1 per variant) and then the analysis.
   Other options: `--experiments E2`, `--variants E2_vans_7`, `--methods milp`, `--sizes 10 20 30 40 50`,
   `--time-limit`, `--workers`, `--threads` (defaults = the baseline run: 300 s, 8 × 4).
3. Optional: **RL1 retrained per variant** (a new model trained on the variant's inputs, same recipe as the baseline
   training): `--methods rl1_retrain`. It uses the GPU, so it can run in a second terminal while the MILP runs
   (then give the MILP `--workers 7` and RL1 `--rl-threads 4`). Without it, RL1 means *the baseline model applied to the
   changed inputs* (its features read the costs, fleet and guns of each instance, so it does react to the change;
   the analysis shows how close it stays to the MILP).
4. Copy the results back and run the analysis on the laptop (it has pandas and matplotlib):
   ```
   scp -r romit@10.72.19.181:/mnt/DataDrive_2/Yakshith/or_project/project/comparative_results/results "E:\AUG-2026\02_CP320_OR_for_Mobility\project\comparative_results\"
   python analyze_experiments.py
   ```
   The analysis also works on partial results (variants or instances not finished yet are left out and reported).

## Projected run time

From the measured times of the baseline runs (`code/project_timing.py`; full table in `timing_projection.md`).
MILP: each instance takes its baseline time, scaled by the model-size ratio (binaries + big-M rows, counted
exactly; only vans and guns change the size), capped at 300 s, on 8 workers × 4 threads. Worst case = every instance
runs into the 300 s limit. RL1: 0.12 s per instance on the RTX 4090 plus loading; retraining ≈ 39 min per variant
(the baseline training stopped early at epoch 210), worst case 500 epochs.

| priority | experiments | variants | instances | MILP expected | MILP worst case | RL1 | RL1 retrained (optional) |
|---|---|---|---|---|---|---|---|
| 1 | E5, E6 | 4 | 2,240 | **12.2 h** | 23.6 h | 8 min | 2.7 h |
| 2 | E1, E2, E8 | 7 | 3,920 | **21.3 h** | 41.4 h | 15 min | 4.8 h |
| 3 | E4 | 2 | 1,120 | **6.1 h** | 11.8 h | 4 min | 1.4 h |
| 4 | E7 | 2 | 1,120 | **6.1 h** | 11.8 h | 4 min | 1.4 h |
| 1–4 (default) | | 15 | 8,400 | **45.7 h** (≈ 1.9 days) | 88.7 h | 32 min | 10.2 h |
| 5 (optional) | E3 | 2 | 400 | 3.4 h | 4.2 h | 3 min | 1.3 h |

Each full variant (560 instances) costs about 3.0 h of MILP, the same as the baseline run (2.97 h). Instances
n ≤ 50 only (`SIZES="10 20 30 40 50"`): 3.7 h for priority 1 and 13.9 h for priorities 1–4; more of them are
solved to optimality, so the comparisons are cleaner, but the large instances (where queues and vans matter most) are
missing.

## Output

```
comparative_results/
├── code/                 experiments.json, common.py, make_variants.py, run_experiments.py (+ .sh),
│                         project_timing.py, analyze_experiments.py
├── variants/<id>/        inputs of each variant (written by the runner)
├── results/<id>/milp/    runs.csv, plan.csv, meta.json, logs.zip  (run ids milp_300s and greedy)
├── results/<id>/rl1/     the same for RL1 (run id rl1)
├── results/<id>/rl1_retrained/   (only with --methods rl1_retrain; model in models/<id>/)
├── results/progress.csv  one row per step: start, end, minutes, exit code, status counts (actual timing)
├── results/_runner_logs/ console output of every step
├── analysis/             summary.md + tables (CSV) + figures/ (analyze_experiments.py)
└── timing_projection.md / .csv
```

Each method writes into its own folder, so two methods can run at the same time without touching the same table.
The tables have the schema of `code/results_io.py` (the same as `new_results/`).

### What the analysis computes (`analysis/summary.md`)

All comparisons are **paired**: a level's means use only the instances with a valid plan at every level of the
factor (for that method), so a change never comes from a different instance mix; MILP numbers are also given on the
instances solved to optimality at every level.

- Per level and method: Z and Z1–Z6, operator and customer cost, change of Z (pooled, mean, median), van share of
  free-choice EVs, vans used, van km, FCS queue, van wait, MILP status / gap, and arc elasticities of Z and of the van share.
- E5, E6, E1: change of Z split into the **price effect** (the baseline plan re-scored with the new constant, i.e. the
  envelope-theorem prediction of the comprehensive report) and the **re-optimisation effect** (what re-solving saves);
  the theory income threshold θ* per level.
- E1: van share and customer cost per income band, operator cost and van tariff paid.
- E2: per Solomon class, change of Z per van / seat added, fleet use, stops per van, vans at the seat limit.
- E8: how many instances move by more than 0.1 % and by more than the 1 % threshold, how often a baseline van ends
  nearly empty, and the verdict (drop recharging or keep it).
- E4: queue (mean, 90 %), share queued, van share and customer cost per demand-pressure tercile.
- E7: the Table 25 estimate (baseline van propensity per θ decile reweighted with the new Beta) against the re-solved
  van share.
- E3: status transitions (e.g. infeasible → optimal) and the change of Z on instances feasible in both.
- RL1 against the MILP at every level (does RL1 stay near the MILP when the inputs move?).
