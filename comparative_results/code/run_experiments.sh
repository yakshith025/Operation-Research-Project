#!/usr/bin/env bash
# Comparative experiments on the lab server: MILP + RL1 for every variant of the selected priorities,
# then the analysis. Run from comparative_results/code in the conda env (conda activate yakshith).
#
#   bash run_experiments.sh                 # priorities 1-4: E5 E6 | E1 E2 E8 | E4 | E7  (MILP + RL1)
#   bash run_experiments.sh 1               # only priority 1 (E5, E6)
#   bash run_experiments.sh "1 2"           # priorities 1 and 2
#   bash run_experiments.sh 5               # the optional horizon experiment E3
#   METHODS="rl1_retrain" bash run_experiments.sh 1     # retrain RL1 per variant (GPU; e.g. a 2nd terminal)
#   SIZES="10 20 30 40 50" bash run_experiments.sh 1    # smaller instances only
# Long runs: nohup bash run_experiments.sh > run_experiments.log 2>&1 &
# The same command continues an interrupted run (finished instances are skipped).
set -u
PRI=${1:-1 2 3 4}
METHODS=${METHODS:-milp rl1}
SIZES=${SIZES:-10 20 30 40 50 60 70 80 90 100}
echo "$(date '+%F %T')  comparative experiments: priorities ${PRI}, methods ${METHODS}, sizes ${SIZES}"
python run_experiments.py --priority $PRI --methods $METHODS --sizes $SIZES
python analyze_experiments.py || echo "analysis not written (needs pandas and matplotlib: pip install pandas matplotlib)"
echo "$(date '+%F %T')  finished: results in $(cd .. && pwd)/results, analysis in $(cd .. && pwd)/analysis"
