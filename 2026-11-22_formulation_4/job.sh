#!/usr/bin/env bash
# =====================================================================================================
# Formulation 4: every run in one job, in this order
#
#   PHASE 1  MILP (all 32 cores: 8 instances at a time x 4 Gurobi threads, 300 s per instance)
#     1. baseline_MILP                                   560 Solomon instances (n = 10..100)
#     2. E5_value_of_charging_time_MILP                  4 variants x 560
#     3. E6_money_sensitivity_MILP                       4 variants x 560
#     4. E1_van_tariff_MILP                              2 variants x 560
#     5. E2_van_fleet_and_seats_MILP                     4 variants x 560
#     6. E8_van_battery_MILP                             1 variant  x 560
#   PHASE 2  RL (starts only after every MILP run has finished; GPU + all cores)
#     7. baseline_RL: self-checks, training (model/), solving the 560 instances (results/)
#     8. E5, E6, E1, E2, E8 _RL: per variant the baseline model on the changed inputs (rl_f4_zero_shot) and a
#        model retrained on the variant (model/<variant>/, rl_f4_retrained)
#   PHASE 3  comparisons of every experiment with its baseline (results/comparison/), and a status summary
#
# Every result goes into the results/ folder of its own folder, in the common format (common/results_io_f4.py):
#   runs.csv, plan.csv, meta.json, logs.zip, analysis/<run id>/      (experiments: results/<variant>/...)
#
# Run it (on the server, in the conda env):
#   conda activate yakshith
#   cd /mnt/DataDrive_2/Yakshith/or_project/project/2026-11-22_formulation_4
#   nohup bash job.sh > job.log 2>&1 &
#   tail -f job.log                      # follow it;  cat job_progress.log = one line per step
#   tail -f E5_value_of_charging_time_MILP/job.log   # the log of one experiment (every folder has its own job.log)
#   kill $(cat job.pid)                  # stop everything (finished instances stay in the tables)
#
# Logs: job.log (everything, from nohup), job_progress.log (one line per step), and in every folder that runs
# something its own job.log with the full output of its steps (baseline_MILP/job.log, baseline_RL/job.log,
# <E>_..._MILP/job.log, <E>_..._RL/job.log; the comparison of phase 3 is appended there too). Reruns append.
# Running it again continues where it stopped: solved instances are skipped (--resume), a finished RL
# training is not repeated.
#
# Projected time on the 32-core server (instances as long as in the formulation-3 run, ~150 s on average):
#   MILP  16 runs x 560 instances: about 3 h each, about 48 h in all (at most ~93 h if every instance hits 300 s)
#   RL    baseline ~40 min training + ~5 min solving; 15 variants x (~40 min training + ~10 min solving): ~13 h
#
# Settings (environment variables; the defaults are the baseline settings):
#   TL=300 WORKERS=8 THREADS=4           MILP time limit per instance, instances at a time, Gurobi threads each
#   RL_THREADS=0 RL_DEVICE=auto          torch CPU threads (0 = all cores) and device (auto = GPU if present)
#   RL_HOURS=8                           cap on every RL training
#   EXPERIMENTS="E5 E6 E1 E2 E8"         experiments, in this order
#   SKIP_MILP=1 / SKIP_RL=1              leave out a phase
#   DRY_RUN=1                            print the steps, run nothing
#   SIZES="10 20" ONLY="EV-C101" RL_TRAIN_EXTRA="..." RL_SOLVE_EXTRA="..."   only for small tests
# =====================================================================================================
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 1
F4="$(pwd)"
PY=${PY:-python}
TL=${TL:-300}
WORKERS=${WORKERS:-8}
THREADS=${THREADS:-4}
RL_THREADS=${RL_THREADS:-0}
RL_DEVICE=${RL_DEVICE:-auto}
RL_HOURS=${RL_HOURS:-8}
EXPERIMENTS=${EXPERIMENTS:-"E5 E6 E1 E2 E8"}
SIZES=${SIZES:-}
ONLY=${ONLY:-}
RL_TRAIN_EXTRA=${RL_TRAIN_EXTRA:-}
RL_SOLVE_EXTRA=${RL_SOLVE_EXTRA:-}
SKIP_MILP=${SKIP_MILP:-0}
SKIP_RL=${SKIP_RL:-0}
DRY_RUN=${DRY_RUN:-0}
export PYTHONUNBUFFERED=1
PROGRESS="$F4/job_progress.log"
SUMMARY="$F4/job_summary.txt"
FAILED=0

declare -A FOLDER=([E5]=E5_value_of_charging_time [E6]=E6_money_sensitivity [E1]=E1_van_tariff
                   [E2]=E2_van_fleet_and_seats [E8]=E8_van_battery)
SEL=()                                   # instance selection (empty = the 560 Solomon instances)
if [ -n "$SIZES" ]; then SEL+=(--sizes $SIZES); fi
if [ -n "$ONLY" ]; then SEL+=(--only $ONLY); fi

ts() { date '+%F %T'; }

note() { echo "$(ts)  $*" | tee -a "$PROGRESS"; }

# step "<name>" <folder relative to this one> <command ...>: runs the command in that folder, logs start, end,
# exit code and minutes; a failing step is counted and the job goes on with the next one.
# The step's whole output goes to the main log (stdout = job.log) AND to the job.log of its own top folder
# (e.g. E5_value_of_charging_time_MILP/job.log, baseline_RL/job.log), appended with a header and a footer.
step() {
    local name=$1 dir=$2
    shift 2
    local log="$F4/${dir%%/*}/job.log"
    note "START  $name   (in $dir: $*; log ${dir%%/*}/job.log)"
    if [ "$DRY_RUN" = 1 ]; then
        return 0
    fi
    local t0 rc pid
    t0=$(date +%s)
    {
        echo "=================================================================================================="
        echo "$(ts)  START  $name"
        echo "folder: $dir"
        echo "command: $*"
        echo "=================================================================================================="
    } >> "$log"
    # in the background + wait, so that a stop signal is handled at once (bash defers traps while a
    # foreground command runs); pipefail: the exit code is the command's, not tee's
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
    rm -f "$F4/job.pid"
    trap - INT TERM
    kill 0
    exit 130
}
trap on_stop INT TERM
echo $$ > "$F4/job.pid"

# ------------------------------------------------------------------------------------------------ preflight
preflight() {
    local ok=1
    note "JOB START  ($F4)"
    note "settings: TL=$TL WORKERS=$WORKERS THREADS=$THREADS RL_THREADS=$RL_THREADS RL_DEVICE=$RL_DEVICE RL_HOURS=$RL_HOURS EXPERIMENTS='$EXPERIMENTS' SIZES='$SIZES' ONLY='$ONLY' SKIP_MILP=$SKIP_MILP SKIP_RL=$SKIP_RL"
    echo "python: $(command -v "$PY") ($($PY --version 2>&1)); cores: $(nproc 2>/dev/null || echo '?')"
    echo "GRB_LICENSE_FILE=${GRB_LICENSE_FILE:-<not set>}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi: no GPU listed"
    for f in solomon_config.csv solomon_data.csv; do
        if [ ! -s "$F4/data/$f" ]; then echo "MISSING data/$f"; ok=0; fi
    done
    if [ -d "$F4/../dataset_ev_solomon/csv" ]; then
        ( cd "$F4/data" && $PY make_f4_dataset.py --check ) || echo "WARNING: data check reported a problem (see above)"
    fi
    $PY -c "import numpy; print('numpy', numpy.__version__)" || { echo "numpy missing: pip install numpy"; ok=0; }
    $PY -c "import matplotlib" 2>/dev/null || echo "note: matplotlib missing (no figures; tables are written anyway)"
    if [ "$SKIP_MILP" != 1 ]; then
        # a 3000-binary model needs the full academic licence (the pip licence stops at 2000)
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
        rm -f "$F4/job.pid"
        exit 1
    fi
    note "preflight OK"
}

rl_trained() {    # true if baseline_RL/model holds a finished training
    $PY -c "
import json, os, sys
d = os.path.join(os.getcwd(), 'baseline_RL', 'model')   # the job runs in the formulation-4 folder
try:
    m = json.load(open(os.path.join(d, 'train_meta.json')))
except Exception:
    sys.exit(1)
r = str(m.get('stop_reason', ''))
sys.exit(0 if r and not r.startswith(('interrupted', 'stopped')) and os.path.exists(os.path.join(d, 'model_best.pt')) else 1)"
}

preflight

# ------------------------------------------------------------------------------------------------ phase 1
if [ "$SKIP_MILP" != 1 ]; then
    note "===== PHASE 1: MILP ($WORKERS instances at a time x $THREADS threads, $TL s each)"
    step "MILP baseline" baseline_MILP/code \
        $PY solve_milp_f4.py --time-limit "$TL" --workers "$WORKERS" --threads "$THREADS" --run-id milp_300s \
        --resume ${SEL[@]+"${SEL[@]}"}
    for E in $EXPERIMENTS; do
        f=${FOLDER[$E]:-}
        if [ -z "$f" ]; then note "unknown experiment $E: skipped"; continue; fi
        step "MILP $E" "${f}_MILP/code" \
            $PY "run_${E}_MILP.py" --time-limit "$TL" --workers "$WORKERS" --threads "$THREADS" ${SEL[@]+"${SEL[@]}"}
    done
    note "===== PHASE 1 finished"
fi

# ------------------------------------------------------------------------------------------------ phase 2
if [ "$SKIP_RL" != 1 ]; then
    note "===== PHASE 2: RL (device $RL_DEVICE, torch threads $RL_THREADS)"
    step "RL self-checks" baseline_RL/code $PY test_rl_f4.py --quick --device "$RL_DEVICE"
    if [ "$DRY_RUN" != 1 ] && rl_trained; then
        note "RL baseline training: already finished (baseline_RL/model/train_meta.json); not repeated"
    else
        step "RL baseline training" baseline_RL/code \
            $PY train_rl_f4.py --threads "$RL_THREADS" --device "$RL_DEVICE" --hours "$RL_HOURS" $RL_TRAIN_EXTRA
    fi
    step "RL baseline solve" baseline_RL/code \
        $PY solve_rl_f4.py --threads "$RL_THREADS" --device "$RL_DEVICE" --resume ${SEL[@]+"${SEL[@]}"} $RL_SOLVE_EXTRA
    for E in $EXPERIMENTS; do
        f=${FOLDER[$E]:-}
        if [ -z "$f" ]; then continue; fi
        step "RL $E" "${f}_RL/code" \
            $PY "run_${E}_RL.py" --rl-threads "$RL_THREADS" --device "$RL_DEVICE" --hours "$RL_HOURS" \
            --rl-train-extra="$RL_TRAIN_EXTRA" --rl-solve-extra="$RL_SOLVE_EXTRA" ${SEL[@]+"${SEL[@]}"}
    done
    note "===== PHASE 2 finished"
fi

# ------------------------------------------------------------------------------------------------ phase 3
note "===== PHASE 3: comparisons and summary"
for E in $EXPERIMENTS; do
    f=${FOLDER[$E]:-}
    if [ -z "$f" ]; then continue; fi
    for M in MILP RL; do
        if [ "$M" = MILP ] && [ "$SKIP_MILP" = 1 ]; then continue; fi
        if [ "$M" = RL ] && [ "$SKIP_RL" = 1 ]; then continue; fi
        step "compare $E $M" "${f}_${M}/code" $PY "analyze_${E}_${M}.py"
    done
done
if [ "$DRY_RUN" != 1 ]; then
    $PY -c "
import csv, glob, os
root = os.getcwd()   # the job runs in the formulation-4 folder
lines = []
for path in sorted(glob.glob(os.path.join(root, '*', 'results', 'runs.csv')) + glob.glob(os.path.join(root, '*', 'results', '*', 'runs.csv'))):
    counts = {}
    with open(path, newline='') as fh:
        for r in csv.DictReader(fh):
            c = counts.setdefault(r['run_id'], {})
            c[r['status']] = c.get(r['status'], 0) + 1
    rel = os.path.relpath(os.path.dirname(path), root)
    for run_id, c in counts.items():
        lines.append(f'{rel:<60} {run_id:<18} {sum(c.values()):>4} runs  ' + ', '.join(f'{k} {v}' for k, v in sorted(c.items())))
text = 'results folder                                               run id             runs  status\n' + '\n'.join(lines)
open(os.path.join(root, 'job_summary.txt'), 'w').write(text + '\n')
print(text)" || echo "summary not written"
fi
note "JOB FINISHED: $FAILED step(s) with a non-zero exit code (see job_progress.log); summary in job_summary.txt"
rm -f "$F4/job.pid"
