# RL1 results

This folder is empty until `code/solve_rl1.py` runs (on the server: `bash run_rl1.sh`). The files follow the schema of `code/results_io.py`, the same as `new_results/`:

| File | Content |
|---|---|
| `runs.csv` | One row per instance (run_id `rl1`, method `RL1`, seed, instance, n): status `plan` / `invalid_plan` / `error`, valid, violations, Z_total, Z1–Z6, service numbers, `solve_s` (inference), `train_s` (training of the model used), message |
| `plan.csv` | One row per EV (type `ev`: resource, seq, start/end, drive, queue, van wait, charge, money) and per van (type `van`: stops, km, energy, battery left, return minute) |
| `meta.json` | Settings of the run: model file, training state, inference settings, f3 parameters |
| `logs.zip` | Console output and error tracebacks |
| `comparison.csv`, `comparison_summary.md` | Written by `code/compare_rl1.py`: RL1 vs MILP (`milp_300s`) vs greedy, per instance and by class and n |
