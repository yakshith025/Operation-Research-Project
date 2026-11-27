"""E1 (Van tariff p^M), RL (attention policy): compare every variant with the baseline (no solving).

Reads ../results/<variant>/runs.csv + plan.csv and ../../../../methods/rl/results (run rl_f4); writes ../results/comparison/
(comparison.csv, comparison_by_income.csv, comparison.md, figures) incl. the gap of every RL plan to the MILP plan of the same variant when ../milp/results has it. run_E1_RL.py calls this at the
end; run it again any time (also on partial results, e.g. on the laptop after copying results back).

Usage (from this code/ folder):  python analyze_E1_RL.py
Per variant and run id, the segment x income analysis is in ../results/<variant>/analysis/<run id>/.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "common"))
sys.path.insert(0, COMMON)
import compare_experiment  # noqa: E402

if __name__ == "__main__":
    compare_experiment.run("E1", "rl")
