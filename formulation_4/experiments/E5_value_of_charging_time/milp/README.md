# E5: Value of charging time at theta = 1 (MILP (Gurobi))

**Question.** How sensitive is the van share (per segment) to the most influential assumed value?

Personal: c^{C,1} (value of a minute of FCS charging at the top income), the most influential value in formulation 3. Commercial: the downtime rate at theta = 1 (D = C = V, and Q with it), the commercial counterpart; it is scaled as a whole so that D = C = V still holds. +-25 %.

Part of formulation 4 (`../../../formulation_4.pdf`). Priority 1 (E5 and E6 first, then E1, E2, E8).

## Variants (one change each; everything else = baseline)

| variant | change | `--set` overrides |
|---|---|---|
| `E5_cC1_personal_7.5` | personal c^{C,1} = 7.5 (-25 %) | `personal.value_of_time_Rs_per_min.C.theta1=7.5` |
| `E5_cC1_personal_12.5` | personal c^{C,1} = 12.5 (+25 %) | `personal.value_of_time_Rs_per_min.C.theta1=12.5` |
| `E5_omega1_commercial_4.5` | commercial downtime rate at theta = 1: 4.5 (-25 %) | `commercial.value_of_time_Rs_per_min.D.theta1=4.5`, `commercial.value_of_time_Rs_per_min.C.theta1=4.5`, `commercial.value_of_time_Rs_per_min.V.theta1=4.5`, `commercial.value_of_time_Rs_per_min.Q.theta1=5.625` |
| `E5_omega1_commercial_7.5` | commercial downtime rate at theta = 1: 7.5 (+25 %) | `commercial.value_of_time_Rs_per_min.D.theta1=7.5`, `commercial.value_of_time_Rs_per_min.C.theta1=7.5`, `commercial.value_of_time_Rs_per_min.V.theta1=7.5`, `commercial.value_of_time_Rs_per_min.Q.theta1=9.375` |
Baseline value(s): personal c^{C,1} = 10.0, commercial omega^1 = 6.0. The baseline itself is not solved again: `../../../methods/milp/results (run milp_300s)` is the reference.
The variants are defined once in `../../../common/experiments.json` (shared with `experiments/E5_value_of_charging_time/rl`).

## Folder

```
experiments/E5_value_of_charging_time/milp/
├── README.md
├── code/
│   ├── run_E5_MILP.py       runs every variant (calls ../../../../common/run_experiment.py --experiment E5 --method milp)
│   └── analyze_E5_MILP.py   compares the variants with the baseline (../../../../common/compare_experiment.py)
└── results/
    ├── <variant>/            runs.csv, plan.csv, meta.json, logs.zip, analysis/<run id>/ (format of ../../../common/results_io_f4.py)
    ├── comparison/           comparison.csv, comparison_by_income.csv, comparison.md, fig_van_share.png, fig_dZ.png
    ├── progress.csv          one row per step (variant, start, end, exit code, runs written)
    └── _runner_logs/         console output of every step
```

## How it runs

Each variant = `../../../methods/milp/code/solve_milp_f4.py` on the 560 Solomon instances with the baseline settings (300 s per instance, 8 instances at a time x 4 Gurobi threads = 32 cores, seed 0, greedy warm start), with the variant's `--set` overrides. Every instance goes to Gurobi with the complete model; nothing is skipped.

Commands (server, `conda activate yakshith`, from `experiments/E5_value_of_charging_time/milp/code`):

```
python run_E5_MILP.py --dry-run      # see the commands first
python run_E5_MILP.py                # all variants, resumes if interrupted
python analyze_E5_MILP.py            # comparison only (also on partial results)
```

Projected time: about 3 h per variant on the 32-core server if instances take as long as in the formulation-3 run (560 x ~150 s / 8 workers); at most 5.8 h per variant if every instance runs into the 300 s limit. 4 variant(s): about 12 h (at most 23 h).

## Output: what to look at

- `results/comparison/comparison.md`: per variant the mean change of Z, operator and customer cost, the van share of
  each segment (variant vs baseline on the same instances), vans and seats used, queues, downtime, money paid.
- `results/comparison/comparison_by_income.csv`: van share per segment x income band, variant vs baseline.
- `results/<variant>/analysis/<run id>/summary.md`: the full segment x income analysis of that variant.
