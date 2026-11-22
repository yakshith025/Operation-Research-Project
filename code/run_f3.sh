#!/usr/bin/env bash
# main_formulation_3 on the lab server. Every instance is solved by Gurobi with the complete
# model; everything is written to ../new_results (schema in results_io.py, shared with RL):
#   runs.csv     one row per run (MILP and its greedy warm start): status, Z1-Z6, gap, times, ...
#   plan.csv     one row per EV and per van of every plan
#   meta.json    settings of each run id
#   logs.zip     Gurobi log, console output and error traceback of every instance
#   analysis/    income tables, logistic regression, figures, insights.md
#   report/      report_f3.tex + tables + figures (+ report_f3.pdf if pdflatex is installed)
#
# Usage (run from code/):  bash run_f3.sh [TIME_LIMIT_S] [WORKERS] [THREADS_PER_WORKER]
#   bash run_f3.sh                # 300 s per instance, 8 instances at a time x 4 Gurobi threads = 32 cores
#   bash run_f3.sh 300 16 2       # 16 at a time x 2 threads = 32 cores
# Input: ../dataset_ev_solomon/csv (solomon_* and gehring_homberger_* data + config CSVs) and
# ../dataset_ev_solomon/income/; both must be on the server.
# Which instances (optional environment variables):
#   FILE_SIZE="100"               # 56 Solomon files (default); "200" = 60 Homberger files; "100 200" = both
#   SIZES="10 20 30 40 50 60 70 80 90 100"   # first n EVs of each file (default)
# Running the same command again continues where it stopped (instances already in runs.csv
# under the same run id milp_<TIME_LIMIT>s are kept).
set -u
TL=${1:-300}
W=${2:-8}
TH=${3:-4}
FILE_SIZE=${FILE_SIZE:-100}
SIZES=${SIZES:-10 20 30 40 50 60 70 80 90 100}
OUT=../new_results
mkdir -p "$OUT"
echo "$(date '+%F %T')  time limit ${TL}s, ${W} instances at a time x ${TH} threads, files with ${FILE_SIZE} EVs, sizes: ${SIZES}"

python solve_milp_f3.py --file-size $FILE_SIZE --sizes $SIZES --time-limit "$TL" \
       --workers "$W" --threads "$TH" --out "$OUT" --resume

(cd ../report_f3 && python prepare_report_f3.py ../new_results) || echo "report not built (see the messages above)"
echo "$(date '+%F %T')  finished: everything is in $(cd "$OUT" && pwd)"
