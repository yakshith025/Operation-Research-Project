#!/usr/bin/env bash
# =====================================================================================================
# Formulation 4 -- Solomon only, ONE MILP baseline + everything else RL
# (asked 26 Nov 2026; the Gehring-Homberger dataset is dropped at the user's request -- Solomon only).
#
#   dataset:    solomon = 56 files x 100 EVs  (n = 10,20,...,100)   --file-size 100
#               (GH is OFF by default; it can be re-enabled with DATASETS="solomon gh" if ever wanted)
#
#   PHASE 1  MILP  -- the ONLY MILP runs: the baseline (NO MILP experiments)
#               baseline_MILP   run id milp_300s   (complete model, 300 s per instance)
#   PHASE 2  RL   -- one model trained on the 100-EV instances, then used everywhere
#               baseline_RL: train, then solve the 560 instances
#               experiments E5 E6 E1 E2 E8 E9 E10, RL only:
#                 zero-shot (the baseline model on the variant inputs) + a model retrained on the variant
#   PHASE 3  comparisons, the rubric tables (rubric_results/) and a status summary
#
# Experiments (common/experiments.json):
#   E1 van tariff            E2 van fleet & seats (= mobile chargers)   E5 value of charging time
#   E6 money sensitivity     E8 van battery
#   E9 station capacity (NEW: guns per station + power per gun)
#   E10 charging demand (NEW: energy per EV x0.75 / x1.25)
#
# Every result goes to the results/ of its own folder, in the common schema (common/results_io_f4.py):
#   runs.csv, plan.csv, meta.json, logs.zip, analysis/<run id>/   (experiments: results/<variant>/...).
#
# Run it (server, in the conda env):
#   conda activate yakshith
#   cd /mnt/DataDrive_2/Yakshith/or_project/project/2026-11-22_formulation_4
#   nohup bash run_solomon.sh > run_solomon.log 2>&1 &
#   tail -f run_solomon.log                       # follow it; cat job_progress.log = one line per step
#   kill $(cat run_solomon.pid)                   # stop everything (finished instances stay in the tables)
#
# It resumes: finished instances are skipped (--resume); a finished training is not repeated.
# Fast path (no retraining; still gives every sensitivity): RL_MODE=zero_shot bash run_solomon.sh
#
# Settings (environment variables; defaults = the baseline settings):
#   DATASETS="solomon"                    which datasets to run (set "solomon gh" to also run GH)
#   SOLOMON_SIZES="10 20 ... 100"         the n grid of Solomon
#   GH_SIZES="20 40 ... 200"              the n grid of GH (only used if GH is turned on)
#   TL=300 WORKERS=8 THREADS=4            MILP time limit (s), instances at a time, Gurobi threads each
#   MILP_GH_MAX_N=200                     largest n the MILP baseline runs on GH (only if GH is on)
#   RL_MODE=both                          RL per variant: both | zero_shot | retrain
#   RL_THREADS=0 RL_DEVICE=auto RL_HOURS=8   torch CPU threads (0=all), device, training cap per model
#   RL_FILE_SIZES="100" RL_N_MAX=100      train the RL on these file sizes / up to this n
#   EXPERIMENTS="E5 E6 E1 E2 E8 E9 E10"
#   FRESH_RL=0                            1 = retrain the baseline model even if a finished one exists
#   SKIP_MILP=1 / SKIP_RL=1 / SKIP_TABLES=1 / DRY_RUN=1
#   SIZES="10 20" ONLY="EV-C101"          small tests (SIZES overrides the per-dataset grid)
# =====================================================================================================
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 1
F4="$(pwd)"
PY=${PY:-python}
TL=${TL:-300}
WORKERS=${WORKERS:-8}
THREADS=${THREADS:-4}
MILP_GH_MAX_N=${MILP_GH_MAX_N:-200}
RL_MODE=${RL_MODE:-both}
RL_THREADS=${RL_THREADS:-0}
RL_DEVICE=${RL_DEVICE:-auto}
RL_HOURS=${RL_HOURS:-8}
RL_FILE_SIZES=${RL_FILE_SIZES:-"100"}
RL_N_MAX=${RL_N_MAX:-100}
RL_TRAIN_EXTRA=${RL_TRAIN_EXTRA:-}
RL_SOLVE_EXTRA=${RL_SOLVE_EXTRA:-}
EXPERIMENTS=${EXPERIMENTS:-"E5 E6 E1 E2 E8 E9 E10"}
DATASETS=${DATASETS:-"solomon"}
SOLOMON_SIZES=${SOLOMON_SIZES:-"10 20 30 40 50 60 70 80 90 100"}
GH_SIZES=${GH_SIZES:-"20 40 60 80 100 120 140 160 180 200"}
SIZES=${SIZES:-}
ONLY=${ONLY:-}
FRESH_RL=${FRESH_RL:-0}
SKIP_MILP=${SKIP_MILP:-0}
SKIP_RL=${SKIP_RL:-0}
SKIP_TABLES=${SKIP_TABLES:-0}
DRY_RUN=${DRY_RUN:-0}
export PYTHONUNBUFFERED=1
PROGRESS="$F4/job_progress.log"
SUMMARY="$F4/job_summary.txt"
FAILED=0

ONLYSEL=()
if [ -n "$ONLY" ]; then ONLYSEL+=(--only $ONLY); fi

ts() { date '+%F %T'; }
note() { echo "$(ts)  $*" | tee -a "$PROGRESS"; }

dataset_fs() { case "$1" in solomon) echo 100;; gh) echo 200;; *) echo 0;; esac; }
dataset_sizes() {
    if [ -n "$SIZES" ]; then echo "$SIZES"; return; fi
    case "$1" in solomon) echo "$SOLOMON_SIZES";; gh) echo "$GH_SIZES";; esac
}
milp_sizes() {                              # MILP baseline sizes for dataset $1 (GH capped at MILP_GH_MAX_N)
    local ds=$1 s out=""
    for s in $(dataset_sizes "$ds"); do
        if [ "$ds" = gh ] && [ "$s" -gt "$MILP_GH_MAX_N" ]; then continue; fi
        out="$out $s"
    done
    echo "$out"
}

# step "<name>" <folder relative to F4> <command ...>: run in that folder, log start / end / exit code /
# minutes; a failing step is counted and the job goes on. Output goes to the main log AND to that
# folder's own job.log (baseline_MILP/job.log, baseline_RL/job.log, common/job.log).
step() {
    local name=$1 dir=$2
    shift 2
    local log="$F4/${dir%%/*}/job.log"
    note "START  $name   (in $dir: $*; log ${dir%%/*}/job.log)"
    if [ "$DRY_RUN" = 1 ]; then return 0; fi
    local t0 rc pid
    t0=$(date +%s)
    {
        echo "=================================================================================================="
        echo "$(ts)  START  $name"
        echo "folder: $dir"; echo "command: $*"
        echo "=================================================================================================="
    } >> "$log"
    ( set -o pipefail; ( cd "$F4/$dir" && "$@" ) 2>&1 | tee -a "$log" ) &
    pid=$!
    wait "$pid"
    rc=$?
    if [ "$rc" -ne 0 ]; then FAILED=$((FAILED + 1)); fi
    echo "$(ts)  END    $name   exit code $rc, $(( ($(date +%s) - t0) / 60 )) min" >> "$log"
    note "END    $name   exit code $rc, $(( ($(date +%s) - t0) / 60 )) min"
    return 0
}

on_stop() {
    note "STOPPED (signal): stopping every running solver; finished instances stay in the tables"
    rm -f "$F4/run_solomon.pid"
    trap - INT TERM
    kill 0
    exit 130
}
trap on_stop INT TERM
echo $$ > "$F4/run_solomon.pid"

# ------------------------------------------------------------------------------- RL baseline model guard
BL_MODEL="$F4/baseline_RL/model/model_best.pt"
BL_MARK="$F4/baseline_RL/model/.trained_${RL_FILE_SIZES// /_}"   # e.g. .trained_100

need_baseline_train() {                     # train unless a FINISHED baseline model already exists (ours OR uploaded)
    [ "$FRESH_RL" = 1 ] && return 0
    [ -s "$BL_MODEL" ] || return 0          # no model -> train
    [ -f "$BL_MARK" ] && return 1           # our marker -> reuse
    # no marker (e.g. the existing 22 Nov baseline, or an uploaded one): reuse it if its train_meta says
    # the training finished, so running only the new experiments does NOT retrain the baseline.
    if $PY -c "
import json, sys
try:
    d = json.load(open('baseline_RL/model/train_meta.json'))
except Exception:
    sys.exit(1)
r = str(d.get('stop_reason', ''))
sys.exit(0 if r and not r.startswith(('interrupted', 'stopped')) else 1)"; then
        note "RL baseline: reusing the existing finished model (baseline_RL/model); FRESH_RL=1 to retrain it"
        return 1
    fi
    return 0
}
mark_baseline_trained() {                   # write the marker only if the training finished cleanly
    [ "$DRY_RUN" = 1 ] && return 0
    if $PY -c "
import json, os, sys
try:
    d = json.load(open('baseline_RL/model/train_meta.json'))
except Exception:
    sys.exit(1)
r = str(d.get('stop_reason', ''))
sys.exit(0 if r and not r.startswith(('interrupted', 'stopped')) and os.path.exists('baseline_RL/model/model_best.pt') else 1)"; then
        touch "$BL_MARK"
        note "baseline model marked trained on file sizes '$RL_FILE_SIZES'"
    fi
}

# --------------------------------------------------------------------------------------------- preflight
preflight() {
    local ok=1
    note "JOB START  ($F4)"
    note "settings: DATASETS='$DATASETS' TL=$TL WORKERS=$WORKERS THREADS=$THREADS MILP_GH_MAX_N=$MILP_GH_MAX_N RL_MODE=$RL_MODE RL_FILE_SIZES='$RL_FILE_SIZES' RL_N_MAX=$RL_N_MAX EXPERIMENTS='$EXPERIMENTS' SIZES='$SIZES' ONLY='$ONLY' SKIP_MILP=$SKIP_MILP SKIP_RL=$SKIP_RL"
    echo "python: $(command -v "$PY") ($($PY --version 2>&1)); cores: $(nproc 2>/dev/null || echo '?')"
    echo "GRB_LICENSE_FILE=${GRB_LICENSE_FILE:-<not set>}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi: no GPU listed"
    local need=("solomon_config.csv" "solomon_data.csv")
    case " $DATASETS " in *" gh "*) need+=("gehring_homberger_config.csv" "gehring_homberger_data.csv");; esac
    for f in "${need[@]}"; do
        if [ ! -s "$F4/data/$f" ]; then echo "MISSING data/$f"; ok=0; fi
    done
    $PY -c "import numpy; print('numpy', numpy.__version__)" || { echo "numpy missing: pip install numpy"; ok=0; }
    $PY -c "import matplotlib" 2>/dev/null || echo "note: matplotlib missing (no figures; tables are written anyway)"
    if [ "$SKIP_MILP" != 1 ]; then
        $PY -c "
import gurobipy as gp
m = gp.Model(); m.Params.OutputFlag = 0
x = m.addVars(3000, vtype='B'); m.addConstr(x.sum() >= 1500); m.setObjective(x.sum()); m.optimize()
print('gurobi', gp.gurobi.version(), 'licence OK, status', m.Status)" \
            || { echo "Gurobi does not work (licence? GRB_LICENSE_FILE?): fix it before the MILP runs"; ok=0; }
    fi
    if [ "$SKIP_RL" != 1 ]; then
        $PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
            || { echo "torch missing: the RL phase cannot run"; ok=0; }
    fi
    if [ "$ok" != 1 ]; then
        note "PREFLIGHT FAILED: nothing was run (fix the problem above and start again)"
        rm -f "$F4/run_solomon.pid"
        exit 1
    fi
    note "preflight OK"
}

preflight

# ------------------------------------------------------------------------------------------------ phase 1
if [ "$SKIP_MILP" != 1 ]; then
    note "===== PHASE 1: MILP baseline on [$DATASETS] ($WORKERS instances at a time x $THREADS threads, $TL s each)"
    for ds in $DATASETS; do
        fs=$(dataset_fs "$ds"); szs=$(milp_sizes "$ds")
        if [ "$fs" = 0 ]; then note "unknown dataset '$ds': skipped"; continue; fi
        if [ -z "$szs" ]; then note "MILP baseline $ds: no sizes to run (MILP_GH_MAX_N=$MILP_GH_MAX_N); skipped"; continue; fi
        step "MILP baseline ($ds, file-size $fs, n=$szs)" baseline_MILP/code \
            $PY solve_milp_f4.py --file-size "$fs" --sizes $szs \
            --time-limit "$TL" --workers "$WORKERS" --threads "$THREADS" --run-id milp_300s --resume \
            ${ONLYSEL[@]+"${ONLYSEL[@]}"}
    done
    note "===== PHASE 1 finished"
fi

# ------------------------------------------------------------------------------------------------ phase 2
if [ "$SKIP_RL" != 1 ]; then
    note "===== PHASE 2: RL (device $RL_DEVICE, torch threads $RL_THREADS, mode $RL_MODE)"
    step "RL self-checks" baseline_RL/code $PY test_rl_f4.py --quick --device "$RL_DEVICE"
    if need_baseline_train; then
        step "RL baseline training (file sizes $RL_FILE_SIZES, n<=$RL_N_MAX)" baseline_RL/code \
            $PY train_rl_f4.py --file-sizes $RL_FILE_SIZES --n-max "$RL_N_MAX" \
            --threads "$RL_THREADS" --device "$RL_DEVICE" --hours "$RL_HOURS" $RL_TRAIN_EXTRA
        mark_baseline_trained
    else
        note "RL baseline training: a finished model on file sizes '$RL_FILE_SIZES' exists; not repeated (FRESH_RL=1 to force)"
    fi
    for ds in $DATASETS; do
        fs=$(dataset_fs "$ds"); szs=$(dataset_sizes "$ds")
        if [ "$fs" = 0 ]; then continue; fi
        step "RL baseline solve ($ds, file-size $fs)" baseline_RL/code \
            $PY solve_rl_f4.py --file-size "$fs" --sizes $szs \
            --threads "$RL_THREADS" --device "$RL_DEVICE" --resume ${ONLYSEL[@]+"${ONLYSEL[@]}"} $RL_SOLVE_EXTRA
    done
    # Experiments (RL only): zero-shot (baseline model on the variant inputs) + a model retrained on the variant.
    for ds in $DATASETS; do
        fs=$(dataset_fs "$ds"); szs=$(dataset_sizes "$ds")
        if [ "$fs" = 0 ]; then continue; fi
        step "RL experiments ($ds): $EXPERIMENTS" common \
            $PY run_experiment.py --experiment $EXPERIMENTS --method rl --rl-mode "$RL_MODE" \
            --file-size "$fs" --sizes $szs --device "$RL_DEVICE" --rl-threads "$RL_THREADS" --hours "$RL_HOURS" \
            --rl-train-extra "--file-sizes $RL_FILE_SIZES --n-max $RL_N_MAX $RL_TRAIN_EXTRA" \
            --rl-solve-extra "$RL_SOLVE_EXTRA" ${ONLYSEL[@]+"${ONLYSEL[@]}"}
    done
    note "===== PHASE 2 finished"
fi

# ------------------------------------------------------------------------------------------------ phase 3
note "===== PHASE 3: rubric tables and summary"
if [ "$SKIP_TABLES" != 1 ]; then
    step "rubric tables" common $PY make_rubric_tables.py --out "$F4/rubric_results"
fi
if [ "$DRY_RUN" != 1 ]; then
    $PY -c "
import csv, glob, os
root = os.getcwd()
lines = []
for path in sorted(glob.glob(os.path.join(root, '*', 'results', 'runs.csv')) + glob.glob(os.path.join(root, '*', 'results', '*', 'runs.csv'))):
    counts = {}
    with open(path, newline='') as fh:
        for r in csv.DictReader(fh):
            c = counts.setdefault((r['run_id'], r.get('source', '')), {})
            c[r['status']] = c.get(r['status'], 0) + 1
    rel = os.path.relpath(os.path.dirname(path), root)
    for (run_id, src), c in sorted(counts.items()):
        lines.append(f'{rel:<52} {run_id:<17} {src:<18} {sum(c.values()):>4} runs  ' + ', '.join(f'{k} {v}' for k, v in sorted(c.items())))
text = 'results folder                                       run id            source               runs  status\n' + '\n'.join(lines)
open(os.path.join(root, 'job_summary.txt'), 'w').write(text + '\n')
print(text)" || echo "summary not written"
fi
note "JOB FINISHED: $FAILED step(s) with a non-zero exit code (see job_progress.log); summary in job_summary.txt"
rm -f "$F4/run_solomon.pid"
