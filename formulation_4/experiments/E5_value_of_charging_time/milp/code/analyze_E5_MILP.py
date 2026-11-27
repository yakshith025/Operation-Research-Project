"""E5 (Value of charging time at theta = 1), MILP (Gurobi): compare every variant with the baseline (no solving).

Reads ../results/<variant>/runs.csv + plan.csv and ../../../../methods/milp/results (run milp_300s); writes ../results/comparison/
(comparison.csv, comparison_by_income.csv, comparison.md, figures). run_E5_MILP.py calls this at the
end; run it again any time (also on partial results, e.g. on the laptop after copying results back).

Usage (from this code/ folder):  python analyze_E5_MILP.py
Per variant and run id, the segment x income analysis is in ../results/<variant>/analysis/<run id>/.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "common"))
sys.path.insert(0, COMMON)
import compare_experiment  # noqa: E402

if __name__ == "__main__":
    compare_experiment.run("E5", "milp")
