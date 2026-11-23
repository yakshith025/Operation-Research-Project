# E5: Value of charging time at theta = 1 (RL (attention policy))

**Question.** How sensitive is the van share (per segment) to the most influential assumed value?

Personal: c^{C,1} (value of a minute of FCS charging at the top income), the most influential value in formulation 3. Commercial: the downtime rate at theta = 1 (D = C = V, and Q with it), the commercial counterpart; it is scaled as a whole so that D = C = V still holds. +-25 %.

Part of formulation 4 (`../formulation_4.pdf`). Priority 1 (E5 and E6 first, then E1, E2, E8).

## Variants (one change each; everything else = baseline)

| variant | change | `--set` overrides |
|---|---|---|
| `E5_cC1_personal_7.5` | personal c^{C,1} = 7.5 (-25 %) | `personal.value_of_time_Rs_per_min.C.theta1=7.5` |
| `E5_cC1_personal_12.5` | personal c^{C,1} = 12.5 (+25 %) | `personal.value_of_time_Rs_per_min.C.theta1=12.5` |
| `E5_omega1_commercial_4.5` | commercial downtime rate at theta = 1: 4.5 (-25 %) | `commercial.value_of_time_Rs_per_min.D.theta1=4.5`, `commercial.value_of_time_Rs_per_min.C.theta1=4.5`, `commercial.value_of_time_Rs_per_min.V.theta1=4.5`, `commercial.value_of_time_Rs_per_min.Q.theta1=5.625` |
| `E5_omega1_commercial_7.5` | commercial downtime rate at theta = 1: 7.5 (+25 %) | `commercial.value_of_time_Rs_per_min.D.theta1=7.5`, `commercial.value_of_time_Rs_per_min.C.theta1=7.5`, `commercial.value_of_time_Rs_per_min.V.theta1=7.5`, `commercial.value_of_time_Rs_per_min.Q.theta1=9.375` |
Baseline value(s): personal c^{C,1} = 10.0, commercial omega^1 = 6.0. The baseline itself is not solved again: `../baseline_RL/results (run rl_f4)` is the reference.
The variants are defined once in `../common/experiments.json` (shared with `E5_value_of_charging_time_MILP`).

## Folder

```
E5_value_of_charging_time_RL/
├── README.md
├── code/
│   ├── run_E5_RL.py       runs every variant (calls ../../common/run_experiment.py --experiment E5 --method rl)
│   └── analyze_E5_RL.py   compares the variants with the baseline (../../common/compare_experiment.py)
├── model/<variant>/          retrained models (model_best.pt, train_log.csv, train_meta.json)
└── results/
    ├── <variant>/            runs.csv, plan.csv, meta.json, logs.zip, analysis/<run id>/ (format of ../common/results_io_f4.py)
    ├── comparison/           comparison.csv, comparison_by_income.csv, comparison.md, fig_van_share.png, fig_dZ.png
    ├── progress.csv          one row per step (variant, start, end, exit code, runs written)
    └── _runner_logs/         console output of every step
```

## How it runs

Each variant gives two runs: `rl_f4_zero_shot` = the baseline model (`../baseline_RL/model/model_best.pt`) on the variant's inputs (its features read the costs, fleet and coefficients of each instance, so it reacts to the change without retraining), and `rl_f4_retrained` = a new model trained on random instances with the variant's overrides (`../baseline_RL/code/train_rl_f4.py --set ...`, same recipe as the baseline), saved in `model/<variant>/` and solved. Train `baseline_RL` first (zero-shot needs its model).

Commands (server, `conda activate yakshith`, from `E5_value_of_charging_time_RL/code`):

```
python run_E5_RL.py --dry-run      # see the commands first
python run_E5_RL.py                # all variants, resumes if interrupted
python analyze_E5_RL.py            # comparison only (also on partial results)
```

Projected time: zero-shot: a few minutes per variant (about 0.12 s per instance on the RTX 4090). Retraining: about 40 min per variant on the RTX 4090 (RL1: about 10 s per epoch, early stop near epoch 200-250), so 4 variant(s): about 2.8 h. It uses the GPU, so it can run while the MILP experiments use the CPU cores.

## Output: what to look at

- `results/comparison/comparison.md`: per variant the mean change of Z, operator and customer cost, the van share of
  each segment (variant vs baseline on the same instances), vans and seats used, queues, downtime, money paid, and the gap to the MILP plans of the same variant (if the MILP folder has them).
- `results/comparison/comparison_by_income.csv`: van share per segment x income band, variant vs baseline.
- `results/<variant>/analysis/<run id>/summary.md`: the full segment x income analysis of that variant.
