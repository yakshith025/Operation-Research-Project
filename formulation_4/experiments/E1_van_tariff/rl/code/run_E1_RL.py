"""E1 (Van tariff p^M) with the formulation-4 RL (attention policy): runs every variant of E1.

Variants (E1 in ../../../../common/experiments.json; one change each, applied in memory with --set, no data copy):
  E1_pM_15                     p^M = 15 Rs/kWh (= p^F)
  E1_pM_35                     p^M = 35 Rs/kWh
For each variant, two RL runs on the 560 Solomon instances -> ../results/<variant>/:
  rl_f4_zero_shot   the baseline model (../../../../methods/rl/model/model_best.pt) applied to the variant's inputs
  rl_f4_retrained   a new model trained on the variant (same recipe as the baseline training) in
                    ../model/<variant>/, then solved with it
The baseline is not solved again: ../../../../methods/rl/results (run rl_f4) is the reference.
Everything resumes (instances already in a runs.csv are skipped; a finished training of the same variant is not repeated), never stops on one failure
(../results/progress.csv, ../results/_runner_logs/), and writes the comparison with the baseline at the end
(../results/comparison/).

Usage (from this code/ folder; the user runs it on the server)
  python run_E1_RL.py --dry-run                 # the commands, nothing solved
  python run_E1_RL.py                           # every variant of E1
  python run_E1_RL.py --variants E1_pM_15
  python run_E1_RL.py --rl-mode zero_shot            # only the baseline model (minutes)
  python run_E1_RL.py --rl-mode retrain --hours 8     # only the retrained models
Any option of ../../../../common/run_experiment.py can be added (--sizes, --only, --families, ...).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "common"))
sys.path.insert(0, COMMON)
import run_experiment  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--experiment", "E1", "--method", "rl", *sys.argv[1:]]
    run_experiment.main()
