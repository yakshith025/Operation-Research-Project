# E8: Van battery upper bound B (RL (attention policy))

**Question.** Is recharging worth modelling? If Z barely moves, drop it.

B = 200 kWh for every van. If the mean change of Z on paired instances stays below decision_threshold_pct, a larger battery (or recharging) changes little and van recharging need not be modelled.

Part of formulation 4 (`../../../formulation_4.pdf`). Priority 2 (E5 and E6 first, then E1, E2, E8).

## Variants (one change each; everything else = baseline)

| variant | change | `--set` overrides |
|---|---|---|
| `E8_battery_200` | B = 200 kWh | `config.van_battery_kWh=200` |
Baseline value(s): battery = 150. The baseline itself is not solved again: `../../../methods/rl/results (run rl_f4)` is the reference.
The variants are defined once in `../../../common/experiments.json` (shared with `experiments/E8_van_battery/milp`).

## Folder

```
experiments/E8_van_battery/rl/
├── README.md
├── code/
│   ├── run_E8_RL.py       runs every variant (calls ../../../../common/run_experiment.py --experiment E8 --method rl)
│   └── analyze_E8_RL.py   compares the variants with the baseline (../../../../common/compare_experiment.py)
├── model/<variant>/          retrained models (model_best.pt, train_log.csv, train_meta.json)
└── results/
    ├── <variant>/            runs.csv, plan.csv, meta.json, logs.zip, analysis/<run id>/ (format of ../../../common/results_io_f4.py)
    ├── comparison/           comparison.csv, comparison_by_income.csv, comparison.md, fig_van_share.png, fig_dZ.png
    ├── progress.csv          one row per step (variant, start, end, exit code, runs written)
    └── _runner_logs/         console output of every step
```

## How it runs

Each variant gives two runs: `rl_f4_zero_shot` = the baseline model (`../../../methods/rl/model/model_best.pt`) on the variant's inputs (its features read the costs, fleet and coefficients of each instance, so it reacts to the change without retraining), and `rl_f4_retrained` = a new model trained on random instances with the variant's overrides (`../../../methods/rl/code/train_rl_f4.py --set ...`, same recipe as the baseline), saved in `model/<variant>/` and solved. Train `methods/rl` first (zero-shot needs its model).

Commands (server, `conda activate yakshith`, from `experiments/E8_van_battery/rl/code`):

```
python run_E8_RL.py --dry-run      # see the commands first
python run_E8_RL.py                # all variants, resumes if interrupted
python analyze_E8_RL.py            # comparison only (also on partial results)
```

Projected time: zero-shot: a few minutes per variant (about 0.12 s per instance on the RTX 4090). Retraining: about 40 min per variant on the RTX 4090 (RL1: about 10 s per epoch, early stop near epoch 200-250), so 1 variant(s): about 0.7 h. It uses the GPU, so it can run while the MILP experiments use the CPU cores.

## Output: what to look at

- `results/comparison/comparison.md`: per variant the mean change of Z, operator and customer cost, the van share of
  each segment (variant vs baseline on the same instances), vans and seats used, queues, downtime, money paid, and the gap to the MILP plans of the same variant (if the MILP folder has them).
- `results/comparison/comparison_by_income.csv`: van share per segment x income band, variant vs baseline.
- `results/<variant>/analysis/<run id>/summary.md`: the full segment x income analysis of that variant.
