#!/usr/bin/env bash
# RL1 on the lab server: self-checks, training, solving the EV-Solomon instances, comparison with the MILP.
# Everything goes into the RL1 folder:
#   ../model/     model_best.pt, model_last.pt, train_log.csv, train_meta.json
#   ../results/   runs.csv, plan.csv, meta.json, logs.zip (schema of code/results_io.py, same as the MILP)
#                 comparison.csv, comparison_summary.md (RL1 vs MILP vs greedy, if ../../new_results exists)
#
# Usage (run from RL1_reinforcement_learning_method_1/code):
#   bash run_rl1.sh                # train (all cores), then the 560 Solomon instances
#   bash run_rl1.sh 8 32           # time cap 8 h, 32 torch threads
# Training (code/rl1_params.json): the first 200 epochs always run; after that it stops when 10 epochs
# in a row bring no better validation result (early stop), at 500 epochs in total, or at the time cap
# HOURS, whichever comes first.
# Which instances (optional environment variables, as run_f3.sh):
#   FILE_SIZE="100"                # 56 Solomon files (default); "200" = 60 Homberger files; "100 200" = both
#   SIZES="10 20 30 40 50 60 70 80 90 100"
# Every run trains a new model from epoch 1. The previous run's model files are first moved to
# ../model/previous_runs/<date-time>/ (kept, not deleted). Solving then uses the new model for every instance.
set -u
HOURS=${1:-8}
TH=${2:-0}
FILE_SIZE=${FILE_SIZE:-100}
SIZES=${SIZES:-10 20 30 40 50 60 70 80 90 100}
echo "$(date '+%F %T')  RL1: training at least 200 and at most 500 epochs (early stop after 10 without improvement) / ${HOURS} h cap, threads ${TH} (0 = all), files with ${FILE_SIZE} EVs, sizes: ${SIZES}"

python test_rl1.py --quick || echo "WARNING: a self-check failed (see above); continuing"
python train_rl1.py --hours "$HOURS" --threads "$TH"
python solve_rl1.py --file-size $FILE_SIZE --sizes $SIZES --threads "$TH" --resume
python compare_rl1.py || echo "comparison not written (MILP results missing?)"
echo "$(date '+%F %T')  finished: results in $(cd ../results && pwd)"
