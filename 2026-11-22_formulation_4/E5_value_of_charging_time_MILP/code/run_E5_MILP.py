"""E5 (Value of charging time at theta = 1) with the formulation-4 MILP (Gurobi): runs every variant of E5.

Variants (E5 in ../../common/experiments.json; one change each, applied in memory with --set, no data copy):
  E5_cC1_personal_7.5          personal c^{C,1} = 7.5 (-25 %)
  E5_cC1_personal_12.5         personal c^{C,1} = 12.5 (+25 %)
  E5_omega1_commercial_4.5     commercial downtime rate at theta = 1: 4.5 (-25 %)
  E5_omega1_commercial_7.5     commercial downtime rate at theta = 1: 7.5 (+25 %)
For each variant: ../../baseline_MILP/code/solve_milp_f4.py with the baseline settings (300 s, 8 workers x 4 threads, seed 0, greedy warm start) on the 560 Solomon instances
-> ../results/<variant>/ (run ids milp_300s and greedy).
The baseline is not solved again: ../../baseline_MILP/results (run milp_300s) is the reference.
Everything resumes (instances already in a runs.csv are skipped), never stops on one failure
(../results/progress.csv, ../results/_runner_logs/), and writes the comparison with the baseline at the end
(../results/comparison/).

Usage (from this code/ folder; the user runs it on the server)
  python run_E5_MILP.py --dry-run                 # the commands, nothing solved
  python run_E5_MILP.py                           # every variant of E5
  python run_E5_MILP.py --variants E5_cC1_personal_7.5
  python run_E5_MILP.py --time-limit 300 --workers 8 --threads 4   (the defaults)
Any option of ../../common/run_experiment.py can be added (--sizes, --only, --families, ...).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.abspath(os.path.join(HERE, "..", "..", "common"))
sys.path.insert(0, COMMON)
import run_experiment  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--experiment", "E5", "--method", "milp", *sys.argv[1:]]
    run_experiment.main()
