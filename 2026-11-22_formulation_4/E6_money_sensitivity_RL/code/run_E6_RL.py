"""E6 (Money sensitivity at theta = 0, mu^0) with the formulation-4 RL (attention policy): runs every variant of E6.

Variants (E6 in ../../common/experiments.json; one change each, applied in memory with --set, no data copy):
  E6_mu0_personal_1.125        personal mu^0 = 1.125 (-25 %)
  E6_mu0_personal_1.875        personal mu^0 = 1.875 (+25 %)
  E6_mu0_commercial_1.125      commercial mu^0 = 1.125 (-25 %)
  E6_mu0_commercial_1.875      commercial mu^0 = 1.875 (+25 %)
For each variant, two RL runs on the 560 Solomon instances -> ../results/<variant>/:
  rl_f4_zero_shot   the baseline model (../../baseline_RL/model/model_best.pt) applied to the variant's inputs
  rl_f4_retrained   a new model trained on the variant (same recipe as the baseline training) in
                    ../model/<variant>/, then solved with it
The baseline is not solved again: ../../baseline_RL/results (run rl_f4) is the reference.
Everything resumes (instances already in a runs.csv are skipped; a finished training of the same variant is not repeated), never stops on one failure
(../results/progress.csv, ../results/_runner_logs/), and writes the comparison with the baseline at the end
(../results/comparison/).

Usage (from this code/ folder; the user runs it on the server)
  python run_E6_RL.py --dry-run                 # the commands, nothing solved
  python run_E6_RL.py                           # every variant of E6
  python run_E6_RL.py --variants E6_mu0_personal_1.125
  python run_E6_RL.py --rl-mode zero_shot            # only the baseline model (minutes)
  python run_E6_RL.py --rl-mode retrain --hours 8     # only the retrained models
Any option of ../../common/run_experiment.py can be added (--sizes, --only, --families, ...).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.abspath(os.path.join(HERE, "..", "..", "common"))
sys.path.insert(0, COMMON)
import run_experiment  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--experiment", "E6", "--method", "rl", *sys.argv[1:]]
    run_experiment.main()
