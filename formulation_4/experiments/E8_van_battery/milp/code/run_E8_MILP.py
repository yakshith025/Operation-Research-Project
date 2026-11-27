"""E8 (Van battery upper bound B) with the formulation-4 MILP (Gurobi): runs every variant of E8.

Variants (E8 in ../../../../common/experiments.json; one change each, applied in memory with --set, no data copy):
  E8_battery_200               B = 200 kWh
For each variant: ../../../../methods/milp/code/solve_milp_f4.py with the baseline settings (300 s, 8 workers x 4 threads, seed 0, greedy warm start) on the 560 Solomon instances
-> ../results/<variant>/ (run ids milp_300s and greedy).
The baseline is not solved again: ../../../../methods/milp/results (run milp_300s) is the reference.
Everything resumes (instances already in a runs.csv are skipped), never stops on one failure
(../results/progress.csv, ../results/_runner_logs/), and writes the comparison with the baseline at the end
(../results/comparison/).

Usage (from this code/ folder; the user runs it on the server)
  python run_E8_MILP.py --dry-run                 # the commands, nothing solved
  python run_E8_MILP.py                           # every variant of E8
  python run_E8_MILP.py --variants E8_battery_200
  python run_E8_MILP.py --time-limit 300 --workers 8 --threads 4   (the defaults)
Any option of ../../../../common/run_experiment.py can be added (--sizes, --only, --families, ...).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "common"))
sys.path.insert(0, COMMON)
import run_experiment  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--experiment", "E8", "--method", "milp", *sys.argv[1:]]
    run_experiment.main()
