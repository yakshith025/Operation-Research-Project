# Formulation 4 — Solomon run (26 Nov 2026)

Runs the **complete** formulation-4 model on the **Solomon** dataset and produces the master tables the
course rubric asks for. One Gurobi MILP baseline; everything else is the RL method.

> The Gehring-Homberger (GH) dataset is **not** used (dropped at the user's request). The code still supports
> it — `DATASETS="solomon gh" bash run_solomon.sh` would add it — but it is **off by default**.

## What is new vs the 22 Nov run

| | 22 Nov run (`job.sh`) | this run (`run_solomon.sh`) |
|---|---|---|
| Dataset | Solomon (56 files, n=10..100) | Solomon (same) |
| MILP | baseline + 5 MILP experiments | **baseline only** (one MILP) |
| RL | baseline + 5 RL experiments | baseline + **7 RL experiments** |
| Experiments | E1 E2 E5 E6 E8 | E1 E2 E5 E6 E8 **+ E9 station capacity + E10 charging demand** |
| Output | per-experiment comparisons | the above **+ `rubric_results/` master tables/CSVs** |

### The two new experiments (rubric Analysis factors)

- **E9 — station capacity** (`E9_station_capacity`): guns per station `config.guns_per_station` (baseline 2 →
  1 / 4) and power per gun `config.station_power_kW` (baseline 22 → 11 / 44 kW).
- **E10 — charging demand** (`E10_charging_demand`): energy per EV `config.energy_scale` (baseline 1.0 →
  0.75 / 1.25). The demand **level** (number of EVs) is the n=10..100 sweep itself, reported in
  `charging_demand_by_n.csv`.

Both are **RL only** and apply in memory with `--set` (no data copy), exactly like E1/E2/E5/E6/E8. The code
hooks are in `common/f4_data.py` (`raw()`) and, for retraining, `baseline_RL/code/rl_f4_generator.py`.

## How the rubric maps to the output (`rubric_results/`, written by `common/make_rubric_tables.py`)

| Rubric item | File(s) |
|---|---|
| **Formulation / MILP** (variables, objective Z1–Z6, constraints 2–23) | `../formulation_4.pdf` |
| **Solution method** — Gurobi + RL | `baseline_MILP/`, `baseline_RL/` |
| **Experimental evaluation** — dataset | `data/` (Solomon, with segment & income) |
| **Experimental evaluation** — solution quality | `solution_quality.csv`, `solution_quality_overall.csv`, `milp_vs_rl.csv` |
| **Experimental evaluation** — computational performance / speed | `performance.csv`, `milp_vs_rl.csv` (RL speed-up) |
| **Analysis** — number of mobile chargers | `sensitivity_mobile_chargers.csv` (E2) |
| **Analysis** — station capacities | `sensitivity_station_capacity.csv` (E9) |
| **Analysis** — charging demand | `sensitivity_charging_demand.csv` (E10) + `charging_demand_by_n.csv` |
| **Analysis** — where mobile chargers significantly improve the network | `mobile_charger_impact.csv` |
| everything, with headline numbers | `summary.md`, `sensitivity_summary.csv` |

Every number is computed from the run tables (`runs.csv`). The script is pure standard library and is robust
to partial results, so you can run it on the laptop after copying `*/results/` back:
`python common/make_rubric_tables.py`.

## Run it on the server

```
conda activate yakshith
cd /mnt/DataDrive_2/Yakshith/or_project/project/2026-11-22_formulation_4
nohup bash run_solomon.sh > run_solomon.log 2>&1 &
tail -f run_solomon.log             # follow; cat job_progress.log = one line per step
kill $(cat run_solomon.pid)         # stop (finished instances stay in the tables; resumes next time)
```

Phases: (1) MILP baseline; (2) RL — train one model, solve the 560 instances, then the 7 experiments;
(3) the rubric tables + a status summary (`job_summary.txt`).

### RL modes (`RL_MODE`)

For every experiment variant the RL can be run two ways:

- **`zero_shot`** — apply the **already-trained baseline model** to the variant's changed inputs, with no new
  training. The network reads the new coefficients / costs / fleet and produces a plan in a fraction of a
  second. Measures the effect of the change on a fixed policy. **Fast.**
- **`retrain`** — train a **new** model specifically on the variant (same recipe as the baseline), then solve
  with it. Measures the effect when the policy is re-optimised for the new setting. **Slow** (one training per
  variant, ~tens of minutes each).
- **`both`** (default) — do zero-shot **and** retrain for each variant, so you can compare them.

The 22 Nov analysis found retrained ≈ zero-shot quality, so `RL_MODE=zero_shot bash run_solomon.sh` gives
every sensitivity at a fraction of the time.

### Other switches (environment variables)

```
RL_MODE=zero_shot bash run_solomon.sh           # no retraining: fast, still gives every sensitivity
SKIP_MILP=1 bash run_solomon.sh                 # RL only
EXPERIMENTS="E2 E9 E10" bash run_solomon.sh     # only some experiments
DRY_RUN=1 bash run_solomon.sh                   # print the steps, run nothing
SIZES="10 20" ONLY="EV-C101" bash run_solomon.sh   # tiny smoke test
```

Defaults: `TL=300 WORKERS=8 THREADS=4` (MILP, 32 cores), `RL_MODE=both`, `RL_FILE_SIZES="100" RL_N_MAX=100`,
`RL_HOURS=8` per model.

## Notes

- **Clean, consistent run:** upload only code + data (see the chat for the `scp` commands). Do **not** upload the
  old `baseline_RL/model`, `baseline_RL/results` or `E*_RL/results` if you want every RL number from a single
  fresh model; the old `baseline_MILP/results` are model-independent and safe to reuse (the job `--resume`s them).
- The complete model is used for every instance size (no reduction), per the standing project rule.
- Git is left to the user.
