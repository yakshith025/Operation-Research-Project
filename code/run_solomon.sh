#!/usr/bin/env bash
# MILP (main_formulation, eqs (1)-(20)) on the 56 EV-Solomon files (100-EV files from Solomon),
# each solved for its first n EVs, n = 5, 10, 15, ..., 100  ->  56 x 20 = 1120 instances.
# Results: ../results_Solomon/<family>/  (one sub-folder per family: summary.csv, costs.csv,
# report.md, results.xlsx and one folder per instance) and a log per family.
#
# Usage (run from code/):  bash run_solomon.sh [TIME_LIMIT_S] [THREADS_PER_JOB] [PARALLEL_JOBS]
#   bash run_solomon.sh                 # 60 s per instance, 8 threads per job, 6 families in parallel
#   bash run_solomon.sh 300 8 6         # 300 s per instance
#   bash run_solomon.sh 60 16 1         # one job after the other (all families in one queue)
# Re-running the same command continues where it stopped (--resume).
# Needs PARALLEL_JOBS x THREADS_PER_JOB CPU cores (check with: nproc).
set -u
TL=${1:-60}
TH=${2:-8}
PAR=${3:-6}
SIZES=${SIZES:-$(seq 5 5 100)}
OUT=../results_Solomon
mkdir -p "$OUT"
echo "time limit ${TL}s, ${TH} threads per job, ${PAR} parallel job(s), sizes: $(echo $SIZES)"

if [ "$PAR" -le 1 ]; then
  python solve_milp.py --file-size 100 --sizes $SIZES --time-limit "$TL" --threads "$TH" \
         --out "$OUT/all" --resume > "$OUT/run_all.log" 2>&1
else
  running=0
  for fam in C1 C2 R1 R2 RC1 RC2; do
    python solve_milp.py --file-size 100 --families "$fam" --sizes $SIZES --time-limit "$TL" \
           --threads "$TH" --out "$OUT/$fam" --resume > "$OUT/run_$fam.log" 2>&1 &
    running=$((running + 1))
    if [ "$running" -ge "$PAR" ]; then wait -n; running=$((running - 1)); fi
  done
  wait
fi
echo "finished: results in $OUT"
