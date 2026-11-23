#!/usr/bin/env bash
# RL2 on the lab server: self-checks, training, solving the EV-Solomon instances (sampling + active
# search), comparison with RL1, the MILP and greedy. RL2 reads the same data as RL1
# (../../dataset_ev_solomon/csv) and writes only into the RL2 folder:
#   ../model/     model_best.pt, model_last.pt, train_log.csv, train_meta.json, previous_runs/
#   ../results/   runs.csv, plan.csv, meta.json, logs.zip (schema of code/results_io.py), run ids
#                 rl2_sampling and rl2_eas; comparison.csv, comparison_summary.md
#
# Usage (run from RL2_reinforcement_learning_method_2/code):
#   bash run_rl2.sh                # train (all cores / GPU), then the 560 Solomon instances
#   bash run_rl2.sh 8 32           # time cap 8 h, 32 torch threads
# Training (code/rl2_params.json): the first 200 epochs always run; after that it stops when 10 epochs
# in a row bring no better validation result (early stop), at 500 epochs in total, or at the time cap
# HOURS, whichever comes first. Every run trains a new model from epoch 1 (the previous run's model
# files are moved to ../model/previous_runs/<date-time>/).
# Which instances (optional environment variables):
#   FILE_SIZE="100"                # 56 Solomon files (default); "200" = 60 Homberger files; "100 200" = both
#   SIZES="10 20 30 40 50 60 70 80 90 100"
set -u
HOURS=${1:-8}
TH=${2:-0}
FILE_SIZE=${FILE_SIZE:-100}
SIZES=${SIZES:-10 20 30 40 50 60 70 80 90 100}
echo "$(date '+%F %T')  RL2: training at least 200 and at most 500 epochs (early stop after 10 without improvement) / ${HOURS} h cap, threads ${TH} (0 = all), files with ${FILE_SIZE} EVs, sizes: ${SIZES}"

python test_rl2.py --quick || echo "WARNING: a self-check failed (see above); continuing"
python train_rl2.py --hours "$HOURS" --threads "$TH"
python solve_rl2.py --file-size $FILE_SIZE --sizes $SIZES --threads "$TH" --resume
python compare_rl2.py || echo "comparison not written (RL1 or MILP results missing?)"
echo "$(date '+%F %T')  finished: results in $(cd ../results && pwd)"
