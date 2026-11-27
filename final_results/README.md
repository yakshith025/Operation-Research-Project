# Final results (formulation 4)

A copy (26 Nov 2026) of the parts of `../2026-11-22_formulation_4/` chosen for the final results:
the formulation, the Gurobi baseline, and the RL baseline with all RL experiments. Every file is an exact
copy (530 files, checked by size and checksum); the original folder is unchanged.

```
final_results/
├── formulation_4.tex / .pdf          the MILP formulation (variables, objective Z1-Z6, constraints (2)-(23))
├── data/                             the dataset: 56 Solomon + 60 Gehring-Homberger files, with segment and theta
├── common/                           shared code: parameters, data reader, plan checker, result tables, analysis
├── baseline_MILP/                    Gurobi, complete model, 560 instances (300 s each)
│   ├── code/solve_milp_f4.py
│   └── results/                      runs.csv, plan.csv, meta.json, logs.zip, analysis/
├── baseline_RL/                      RL baseline: attention policy + REINFORCE
│   ├── code/                         simulator, network, generator, training, solving, tests
│   ├── model/                        trained model (best epoch 180)
│   ├── results/                      560 instances solved
│   └── report/RL_code_explained.pdf  line-by-line explanation of the RL code
├── E1_van_tariff_RL/                 RL experiments: for every variant a zero-shot run (baseline model)
├── E2_van_fleet_and_seats_RL/        and a retrained model + run; results/<variant>/, model/<variant>/,
├── E5_value_of_charging_time_RL/     results/comparison/
├── E6_money_sensitivity_RL/
└── E8_van_battery_RL/
```

| | baseline | variants |
|---|---|---|
| E5 value of charging time | personal c^{C,1} = 10, commercial downtime rate = 6 | 7.5 / 12.5 and 4.5 / 7.5 |
| E6 money sensitivity | mu^0 = 1.5 (both segments) | 1.125 / 1.875 per segment |
| E1 van tariff | p^M = 25 Rs/kWh | 15 / 35 |
| E2 van fleet and seats | 5 vans, 6 seats | 3 / 7 vans, 4 / 8 seats |
| E8 van battery | 150 kWh | 200 kWh |

Not copied (still in `../2026-11-22_formulation_4/`): the MILP experiment folders (`E*_MILP`), the
comprehensive report (`report/`), `job.sh` and the job logs. The RL comparison files
(`E*_RL/results/comparison/`) contain a "gap to MILP" column that was computed on the server with those
MILP experiment results; re-running the RL comparisons here would leave that column empty.

The folder layout is the same as the original, so the code runs from here: `baseline_RL/code` and
`baseline_MILP/code` read `../../common` and `../../data`, i.e. the copies in this folder.
