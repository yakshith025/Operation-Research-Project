# E1: Van tariff p^M (MILP (Gurobi))

**Question.** Who gets vans (per segment and income), and at what cost to the operator and to equity?

p^M = 15 equals the FCS tariff p^F (no van premium); 35 doubles the premium.

Part of formulation 4 (`../formulation_4.pdf`). Priority 2 (E5 and E6 first, then E1, E2, E8).

## Variants (one change each; everything else = baseline)

| variant | change | `--set` overrides |
|---|---|---|
| `E1_pM_15` | p^M = 15 Rs/kWh (= p^F) | `tariff_Rs_per_kWh.p_M=15.0` |
| `E1_pM_35` | p^M = 35 Rs/kWh | `tariff_Rs_per_kWh.p_M=35.0` |
Baseline value(s): p^M = 25.0. The baseline itself is not solved again: `../baseline_MILP/results (run milp_300s)` is the reference.
The variants are defined once in `../common/experiments.json` (shared with `E1_van_tariff_RL`).

## Folder

```
E1_van_tariff_MILP/
├── README.md
├── code/
│   ├── run_E1_MILP.py       runs every variant (calls ../../common/run_experiment.py --experiment E1 --method milp)
│   └── analyze_E1_MILP.py   compares the variants with the baseline (../../common/compare_experiment.py)
└── results/
    ├── <variant>/            runs.csv, plan.csv, meta.json, logs.zip, analysis/<run id>/ (format of ../common/results_io_f4.py)
    ├── comparison/           comparison.csv, comparison_by_income.csv, comparison.md, fig_van_share.png, fig_dZ.png
    ├── progress.csv          one row per step (variant, start, end, exit code, runs written)
    └── _runner_logs/         console output of every step
```

## How it runs

Each variant = `../baseline_MILP/code/solve_milp_f4.py` on the 560 Solomon instances with the baseline settings (300 s per instance, 8 instances at a time x 4 Gurobi threads = 32 cores, seed 0, greedy warm start), with the variant's `--set` overrides. Every instance goes to Gurobi with the complete model; nothing is skipped.

Commands (server, `conda activate yakshith`, from `E1_van_tariff_MILP/code`):

```
python run_E1_MILP.py --dry-run      # see the commands first
python run_E1_MILP.py                # all variants, resumes if interrupted
python analyze_E1_MILP.py            # comparison only (also on partial results)
```

Projected time: about 3 h per variant on the 32-core server if instances take as long as in the formulation-3 run (560 x ~150 s / 8 workers); at most 5.8 h per variant if every instance runs into the 300 s limit. 2 variant(s): about 6 h (at most 12 h).

## Output: what to look at

- `results/comparison/comparison.md`: per variant the mean change of Z, operator and customer cost, the van share of
  each segment (variant vs baseline on the same instances), vans and seats used, queues, downtime, money paid.
- `results/comparison/comparison_by_income.csv`: van share per segment x income band, variant vs baseline.
- `results/<variant>/analysis/<run id>/summary.md`: the full segment x income analysis of that variant.
