# RL2 results

This folder is empty until `code/solve_rl2.py` runs (on the server: `bash run_rl2.sh`). The files follow the schema of `code/results_io.py`, the same as `new_results/` and RL1's `results/`:

| File | Content |
|---|---|
| `runs.csv` | Two rows per instance: run id `rl2_sampling` (method `RL2`, sampling as in RL1) and `rl2_eas` (method `RL2-EAS`, active search from that plan). Columns: status `plan` / `invalid_plan` / `error`, valid, violations, Z_total, Z1–Z6, service numbers, `solve_s` (for `rl2_eas`: sampling + active search), `train_s`, message (rollouts, active-search iterations and improvement, unplaced EVs) |
| `plan.csv` | The plan of every run: one row per EV (type `ev`: resource, seq, start/end, drive, queue, van wait, charge, money) and per van (type `van`: stops, km, energy, battery left, return minute) |
| `meta.json` | Settings of each run id: model file, training state, inference and active-search settings, f3 parameters |
| `logs.zip` | Console output and error tracebacks |
| `comparison.csv`, `comparison_summary.md` | Written by `code/compare_rl2.py`: RL2-EAS vs RL2 vs RL1 vs greedy, with the MILP (`milp_300s`) as reference, per instance and by class and n |
