# Formulation 4 (22 Nov 2026): use segment (personal / commercial) + income index

Formulation 3 plus one more customer attribute per EV, the **use segment** `s_i` ∈ {personal, commercial}
(the extension proposed in Section 4.14 of the formulation-3 comprehensive report). A personal driver's time
depends on where the driver is: the van charges the parked car with the driver free. A commercial car (taxi,
ride-hailing, fleet) loses earnings for every minute out of service, so driving, charging and van time are all
priced at one downtime rate, and a van only helps it by cutting downtime. The variables and constraints
(1)–(23) are those of formulation 3; only the per-EV coefficients change, so the MILP is the same size.

- `formulation_4.tex / .pdf`: the formulation, the theory of how segment and income drive the choice
  (thresholds per segment), the calibration of the two attributes, and the experiment design.
- Calibrated attributes (`data/calibration.md`): `s_i ~ Bernoulli(π_C = 0.70)` (commercial share among *public
  charging requests*, from Vahan registrations and charging habits), `θ | personal ~ Beta(2.5, 1)` (car ownership
  by urban income quintile, HCES), `θ | commercial ~ Beta(3, 5)` (cab-driver earnings surveys).

## Folder

```
2026-11-22_formulation_4/
├── README.md                     this file
├── job.sh                        runs everything on the server: MILP (32 cores) first, then RL, then comparisons
├── formulation_4.tex / .pdf      the formulation (13 pages)
├── data/                         THE complete dataset of formulation 4 (all code reads only this)
│   ├── solomon_config.csv, solomon_data.csv                       56 files x 100 EVs
│   ├── gehring_homberger_config.csv, gehring_homberger_data.csv   60 files x 200 EVs
│   │       (= dataset_ev_solomon/csv + columns segment, theta on every EV row)
│   ├── make_f4_dataset.py        builds it from ../dataset_ev_solomon/csv (--check verifies it)
│   ├── calibration.py / .md / .json, calibration_densities.png    where pi_C and the Betas come from
│   └── summary.csv, dataset_settings.json
├── common/                       shared by every MILP and RL folder
│   ├── f4_params.json            customer parameters per segment (placeholders except the calibrated distributions)
│   ├── f4_params.py              loads them, applies --set overrides, computes mu, c^D, c^Q, c^C, c^V, W per EV
│   ├── f4_data.py                data/*.csv -> instance dicts (config overrides applied in memory)
│   ├── check_plan_f4.py          independent checker of constraints (2)-(23) and Z (every method is scored by it)
│   ├── results_io_f4.py          runs.csv / plan.csv / meta.json / logs.zip, with per-segment columns
│   ├── analyze_f4.py             segment x income analysis of one run (called at the end of every solve)
│   ├── experiments.json          the experiments E5, E6, E1, E2, E8 and their variants
│   ├── run_experiment.py         runs the variants with the MILP or the RL (the experiment folders call it)
│   └── compare_experiment.py     variants vs baseline (paired instances), per segment and income band
├── report/                       comprehensive_report_f4.tex/.pdf (32 p., 25 Nov 2026): data, experiments, MILP and RL
│                                 results of the whole job in simple language; make_report_f4.py -> figures/, tables/,
│                                 numbers.tex (reads only the results); sections/*.tex
├── baseline_MILP/                Gurobi, the complete model of formulation_4.pdf
│   ├── code/solve_milp_f4.py
│   └── results/
├── baseline_RL/                  attention policy + REINFORCE (RL1's method, segment-aware inputs)
│   ├── code/  rl_f4_env.py rl_f4_model.py rl_f4_generator.py train_rl_f4.py solve_rl_f4.py test_rl_f4.py rl_f4_params.json
│   ├── model/
│   └── results/
├── E5_value_of_charging_time_MILP/   E5_value_of_charging_time_RL/     priority 1
├── E6_money_sensitivity_MILP/        E6_money_sensitivity_RL/          priority 1
├── E1_van_tariff_MILP/               E1_van_tariff_RL/                 priority 2
├── E2_van_fleet_and_seats_MILP/      E2_van_fleet_and_seats_RL/        priority 2
└── E8_van_battery_MILP/              E8_van_battery_RL/                priority 2
     each: README.md, code/run_<E>_<METHOD>.py, code/analyze_<E>_<METHOD>.py, results/ (RL also model/)
```

No data is copied anywhere else: an experiment variant is a set of `--set key=value` overrides applied when the
solver reads `data/` (e.g. `--set config.n_vans=3`, `--set tariff_Rs_per_kWh.p_M=35`). Results of every method
and experiment have the same schema (`common/results_io_f4.py`), so any two folders can be compared.

## Experiments (`common/experiments.json`)

| # | factor | variants | question |
|---|---|---|---|
| E5 | value of charging time at θ = 1 | personal c^{C,1} 7.5 / 12.5 (10); commercial downtime rate ω^1 4.5 / 7.5 (6) | How sensitive is each segment's van share to the most influential value? |
| E6 | money sensitivity at θ = 0 | personal μ^0 1.125 / 1.875 (1.5); commercial μ^0 1.125 / 1.875 (1.5) | Same, for the largest elasticity on Z |
| E1 | van tariff p^M | 15 (= p^F), 35 (25) | Who gets vans, at what cost to the operator and equity? |
| E2 | van fleet and seats | 3 / 7 vans (5); 4 / 8 seats (6), one at a time | Where does another van or seat pay off? |
| E8 | van battery B | 200 kWh (150) | Is recharging worth modelling? (drop it if Z moves < 1 %) |

Each experiment has a MILP folder and an RL folder. MILP: complete model, baseline settings (300 s, 8 × 4
threads). RL: `rl_f4_zero_shot` (baseline model on the changed inputs) and `rl_f4_retrained` (a model trained on
the variant). The baseline is not solved again: the comparison reads `baseline_MILP/results` and
`baseline_RL/results`.

## Running everything on the server: `job.sh`

One job runs every program in this order and writes each result into the `results/` folder of its own folder, in
the common format (`runs.csv`, `plan.csv`, `meta.json`, `logs.zip`, `analysis/<run id>/`; experiments:
`results/<variant>/...` and `results/comparison/`):

1. **Phase 1, MILP on all 32 cores** (8 instances at a time × 4 Gurobi threads, 300 s each): `baseline_MILP`, then
   E5, E6, E1, E2, E8 (`<E>_..._MILP`). 16 runs × 560 instances: about 48 h if instances take as long as in
   formulation 3, at most about 93 h.
2. **Phase 2, RL**, only after every MILP run: `baseline_RL` (self-checks, training, solving), then E5, E6, E1, E2,
   E8 (`<E>_..._RL`: zero-shot + retrained model per variant). About 13 h on the RTX 4090.
3. **Phase 3**: every experiment compared with its baseline (`results/comparison/`), status of every run in
   `job_summary.txt`.

Upload (Windows cmd, from the laptop); the folder is self-contained (Python with numpy, gurobipy, torch;
matplotlib optional):
```
scp -r "E:\AUG-2026\02_CP320_OR_for_Mobility\project\2026-11-22_formulation_4" romit@10.72.19.181:/mnt/DataDrive_2/Yakshith/or_project/project/
```
On the server:
```
conda activate yakshith
cd /mnt/DataDrive_2/Yakshith/or_project/project/2026-11-22_formulation_4
DRY_RUN=1 bash job.sh                  # optional: preflight checks + the list of steps, nothing solved
nohup bash job.sh > job.log 2>&1 &     # the whole job
tail -f job.log                        # follow it; job_progress.log has one line per step (start, end, exit code)
tail -f E5_value_of_charging_time_MILP/job.log    # one experiment's own log
kill $(cat job.pid)                    # stop everything; finished instances stay in the tables
```
Logs: `job.log` (everything), `job_progress.log` (one line per step), and a `job.log` inside every folder that
runs something (`baseline_MILP/`, `baseline_RL/`, each `<E>_..._MILP/` and `<E>_..._RL/`) with the full output of
its own steps, each step framed by a START header and an END line with its exit code. Reruns append.
The job first checks numpy, the Gurobi licence (a 3000-binary model), torch and the data, and stops at once if one
fails, before anything is solved. A failing step never stops the job (it is counted and logged). Starting
`job.sh` again continues where it stopped: solved instances are skipped, a finished RL training is not repeated.
Settings can be changed with environment variables (`TL`, `WORKERS`, `THREADS`, `RL_THREADS`, `RL_DEVICE`,
`RL_HOURS`, `EXPERIMENTS`, `SKIP_MILP=1`, `SKIP_RL=1`); see the header of `job.sh`.

Each piece can also be run on its own: `baseline_MILP/code/solve_milp_f4.py`, `baseline_RL/code/train_rl_f4.py` +
`solve_rl_f4.py`, `<E>_..._<METHOD>/code/run_<E>_<METHOD>.py` (`--dry-run` shows the commands) and
`analyze_<E>_<METHOD>.py` (comparison only, also on partial results).

Every solver run ends with `results/analysis/<run id>/summary.md` (segment × income tables, income threshold per
segment); every experiment ends with `results/comparison/comparison.md`.
