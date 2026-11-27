# E6: Money sensitivity at theta = 0, mu^0 (MILP (Gurobi))

**Question.** How sensitive are Z and the van share to the coefficient with the largest elasticity on Z?

mu^0 of each segment separately, +-25 % (mu^1 = 0.5 stays, so mu stays non-increasing).

Part of formulation 4 (`../../../formulation_4.pdf`). Priority 1 (E5 and E6 first, then E1, E2, E8).

## Variants (one change each; everything else = baseline)

| variant | change | `--set` overrides |
|---|---|---|
| `E6_mu0_personal_1.125` | personal mu^0 = 1.125 (-25 %) | `personal.money_sensitivity_mu.theta0=1.125` |
| `E6_mu0_personal_1.875` | personal mu^0 = 1.875 (+25 %) | `personal.money_sensitivity_mu.theta0=1.875` |
| `E6_mu0_commercial_1.125` | commercial mu^0 = 1.125 (-25 %) | `commercial.money_sensitivity_mu.theta0=1.125` |
| `E6_mu0_commercial_1.875` | commercial mu^0 = 1.875 (+25 %) | `commercial.money_sensitivity_mu.theta0=1.875` |
Baseline value(s): personal mu^0 = 1.5, commercial mu^0 = 1.5. The baseline itself is not solved again: `../../../methods/milp/results (run milp_300s)` is the reference.
The variants are defined once in `../../../common/experiments.json` (shared with `experiments/E6_money_sensitivity/rl`).

## Folder

```
experiments/E6_money_sensitivity/milp/
├── README.md
├── code/
│   ├── run_E6_MILP.py       runs every variant (calls ../../../../common/run_experiment.py --experiment E6 --method milp)
│   └── analyze_E6_MILP.py   compares the variants with the baseline (../../../../common/compare_experiment.py)
└── results/
    ├── <variant>/            runs.csv, plan.csv, meta.json, logs.zip, analysis/<run id>/ (format of ../../../common/results_io_f4.py)
    ├── comparison/           comparison.csv, comparison_by_income.csv, comparison.md, fig_van_share.png, fig_dZ.png
    ├── progress.csv          one row per step (variant, start, end, exit code, runs written)
    └── _runner_logs/         console output of every step
```

## How it runs

Each variant = `../../../methods/milp/code/solve_milp_f4.py` on the 560 Solomon instances with the baseline settings (300 s per instance, 8 instances at a time x 4 Gurobi threads = 32 cores, seed 0, greedy warm start), with the variant's `--set` overrides. Every instance goes to Gurobi with the complete model; nothing is skipped.

Commands (server, `conda activate yakshith`, from `experiments/E6_money_sensitivity/milp/code`):

```
python run_E6_MILP.py --dry-run      # see the commands first
python run_E6_MILP.py                # all variants, resumes if interrupted
python analyze_E6_MILP.py            # comparison only (also on partial results)
```

Projected time: about 3 h per variant on the 32-core server if instances take as long as in the formulation-3 run (560 x ~150 s / 8 workers); at most 5.8 h per variant if every instance runs into the 300 s limit. 4 variant(s): about 12 h (at most 23 h).

## Output: what to look at

- `results/comparison/comparison.md`: per variant the mean change of Z, operator and customer cost, the van share of
  each segment (variant vs baseline on the same instances), vans and seats used, queues, downtime, money paid.
- `results/comparison/comparison_by_income.csv`: van share per segment x income band, variant vs baseline.
- `results/<variant>/analysis/<run id>/summary.md`: the full segment x income analysis of that variant.
