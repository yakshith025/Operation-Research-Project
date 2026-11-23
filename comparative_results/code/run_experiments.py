"""Run the comparative experiments (experiments.json) with the formulation-3 MILP and RL1.

For every selected variant, in priority order (1 = first):
  1. its inputs are written to ../variants/<variant>/ (make_variants.py; skipped when up to date)
  2. milp         code/solve_milp_f3.py on those inputs, the same settings as the baseline run
                  (300 s, 8 workers x 4 threads, seed 0, greedy warm start)  -> ../results/<variant>/milp/
                  (run ids milp_300s and greedy, the MILP's warm start)
  3. rl1          RL1's trained model (model_best.pt of the baseline training) on those inputs, the same
                  inference settings as the baseline run                      -> ../results/<variant>/rl1/
  4. rl1_retrain  (only if asked) a NEW RL1 model trained on the variant's inputs (same recipe as the
                  baseline training, ~40 min on the RTX 4090) -> ../models/<variant>/, then solved
                                                                              -> ../results/<variant>/rl1_retrained/
The baseline itself is not solved again: the analysis reads new_results/ (MILP) and the RL1 results/.

Never stops on one failure: a step that fails is logged (../results/progress.csv, ../results/_runner_logs/)
and the next step starts. Every step resumes: instances already in a runs.csv are skipped, a finished
retraining is not repeated, so the same command continues an interrupted run.

Usage (from comparative_results/code; the user runs it on the server)
  python run_experiments.py --dry-run                   # what would run + projected time, nothing solved
  python run_experiments.py                             # priorities 1-4 (E5, E6 | E1, E2, E8 | E4 | E7)
  python run_experiments.py --priority 1                # only E5 and E6
  python run_experiments.py --experiments E3            # the optional horizon experiment
  python run_experiments.py --methods rl1_retrain       # retrain RL1 per variant (e.g. in a 2nd terminal)
  python run_experiments.py --variants E2_vans_7 --methods milp
Test runs (small): --only EV-C101 EV-R101 --sizes 10 20 --time-limit 20 --workers 2 --threads 2
                   --results <folder> --rl1-extra "--samples 4 --augment 2"
"""
import argparse
import csv
import os
import shlex
import subprocess
import sys
import time

import common as cm
import make_variants as mv

PROGRESS_COLS = ["variant", "experiment", "priority", "method", "started", "finished", "minutes",
                 "return_code", "instances", "runs_in_table", "status_counts", "command"]


def stream(cmd, log_path, cwd):
    """Run cmd, show its output and append it to log_path. Returns the exit code."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(shlex.quote(c) for c in cmd)}\n")
        log.flush()
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            return proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise


def status_counts(out, run_id):
    path = os.path.join(out, "runs.csv")
    counts, total = {}, 0
    if os.path.exists(path):
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("run_id") == run_id:
                    counts[r["status"]] = counts.get(r["status"], 0) + 1
                    total += 1
    return total, counts


def log_progress(results, row):
    path = os.path.join(results, "progress.csv")
    new = not os.path.exists(path)
    try:
        with open(path, "a", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=PROGRESS_COLS, extrasaction="ignore")
            if new:
                wr.writeheader()
            wr.writerow(row)
    except OSError as e:
        print(f"WARNING: progress.csv not written ({e})", flush=True)


def instance_args(v, args, EXP):
    a = ["--file-size", str(EXP["instances"]["file_size"]), "--sizes", *map(str, args.sizes)]
    if v["families"]:
        a += ["--families", *v["families"]]
    if args.only:
        a += ["--only", *args.only]
    return a


def milp_cmd(v, folder, out, args, EXP):
    cmd = [sys.executable, os.path.join(args.code, "solve_milp_f3.py"), os.path.join(folder, "csv"),
           "--params", os.path.join(folder, "f3_params.json"), "--out", out,
           *instance_args(v, args, EXP),
           "--time-limit", f"{args.time_limit:g}", "--workers", str(args.workers), "--threads", str(args.threads),
           "--run-id", EXP["solver"]["milp_run_id"], "--resume", "--no-analysis"]
    if args.mem_limit_gb > 0:
        cmd += ["--mem-limit-gb", f"{args.mem_limit_gb:g}"]
    return cmd + shlex.split(args.milp_extra)


def rl1_solve_cmd(v, folder, out, model, run_id, args, EXP):
    cmd = [sys.executable, os.path.join(args.rl1, "code", "solve_rl1.py"), "--csv", os.path.join(folder, "csv"),
           "--f3-params", os.path.join(folder, "f3_params.json"), "--model", model, "--out", out,
           "--run-id", run_id, *instance_args(v, args, EXP), "--resume", "--device", args.rl_device,
           "--threads", str(args.rl_threads)]
    return cmd + shlex.split(args.rl1_extra)


def rl1_train_cmd(folder, model_dir, args):
    cmd = [sys.executable, os.path.join(args.rl1, "code", "train_rl1.py"), "--csv", os.path.join(folder, "csv"),
           "--f3-params", os.path.join(folder, "f3_params.json"), "--model-dir", model_dir,
           "--hours", f"{args.retrain_hours:g}", "--device", args.rl_device, "--threads", str(args.rl_threads)]
    return cmd + shlex.split(args.train_extra)


def training_finished(model_dir):
    import json
    try:
        meta = json.load(open(os.path.join(model_dir, "train_meta.json")))
    except (OSError, ValueError):
        return False
    return bool(meta.get("finished")) and os.path.exists(os.path.join(model_dir, "model_best.pt"))


def n_tasks(v, args, EXP):
    """Number of instances of a variant (for the progress table)."""
    try:
        import project_timing as pt
        sys.path.insert(0, cm.code_dir(args.project))
        import make_ev_solomon as mes
        data = mes.EvSolomonData(cm.project_path(EXP["baseline"]["dataset_csv"], args.project))
        return len(pt.tasks_of(v, EXP, data, args.sizes, args.only))
    except Exception:
        return ""


def run_step(v, method, cmd, out, run_id, args, EXP, cwd):
    t0 = time.time()
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'=' * 100}\n{started}  {v['id']} ({v['name']}): {method}\n{'=' * 100}", flush=True)
    log = os.path.join(args.results, "_runner_logs", f"{v['id']}__{method}.log")
    try:
        code = stream(cmd, log, cwd)
    except KeyboardInterrupt:
        raise
    except Exception as e:                       # e.g. the interpreter could not start
        print(f"{v['id']} {method}: could not run ({type(e).__name__}: {e})", flush=True)
        code = f"error: {e}"
    total, counts = status_counts(out, run_id) if run_id else ("", {})
    row = {"variant": v["id"], "experiment": v["experiment"], "priority": v["priority"], "method": method,
           "started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
           "minutes": round((time.time() - t0) / 60, 1), "return_code": code, "instances": n_tasks(v, args, EXP),
           "runs_in_table": total, "status_counts": " ".join(f"{k}={c}" for k, c in sorted(counts.items())),
           "command": " ".join(shlex.quote(c) for c in cmd)}
    log_progress(args.results, row)
    print(f"{v['id']} {method}: exit {code}, {row['minutes']} min, runs in table: {total} "
          f"({row['status_counts'] or 'none'})", flush=True)
    return row


def main():
    EXP = cm.load_experiments()
    S = EXP["solver"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--priority", nargs="*", type=int, help=f"default {EXP['default_priorities']}")
    ap.add_argument("--experiments", nargs="*", help="e.g. E5 E6 (overrides --priority)")
    ap.add_argument("--variants", nargs="*", help="variant ids, e.g. E2_vans_7 (overrides the others)")
    ap.add_argument("--methods", nargs="*", default=["milp", "rl1"], choices=["milp", "rl1", "rl1_retrain"],
                    help="what to run per variant, in this order (default: milp rl1)")
    ap.add_argument("--sizes", nargs="*", type=int, default=EXP["instances"]["sizes"])
    ap.add_argument("--only", nargs="*", help="only these files (tests), e.g. EV-C101")
    ap.add_argument("--time-limit", type=float, default=S["milp_time_limit_s"])
    ap.add_argument("--workers", type=int, default=S["milp_workers"])
    ap.add_argument("--threads", type=int, default=S["milp_threads"])
    ap.add_argument("--mem-limit-gb", type=float, default=0.0, help="Gurobi SoftMemLimit per instance (0 = none)")
    ap.add_argument("--rl-device", default="auto")
    ap.add_argument("--rl-threads", type=int, default=0, help="torch CPU threads (0 = all cores)")
    ap.add_argument("--retrain-hours", type=float, default=8.0, help="time cap of one retraining")
    ap.add_argument("--milp-extra", default="", help="extra options for solve_milp_f3.py (one string)")
    ap.add_argument("--rl1-extra", default="", help="extra options for solve_rl1.py, e.g. \"--samples 8\"")
    ap.add_argument("--train-extra", default="", help="extra options for train_rl1.py")
    ap.add_argument("--project", default=cm.PROJECT, help="project folder (code/, dataset_ev_solomon/, ...)")
    ap.add_argument("--rl1-dir", help="RL1 folder (default: <project>/RL1_reinforcement_learning_method_1)")
    ap.add_argument("--rl1-model", help="trained RL1 model (default: <RL1>/model/model_best.pt)")
    ap.add_argument("--variants-dir", default=cm.DEFAULT_VARIANTS)
    ap.add_argument("--results", default=cm.DEFAULT_RESULTS)
    ap.add_argument("--models-dir", default=cm.DEFAULT_MODELS)
    ap.add_argument("--rebuild-variants", action="store_true", help="write the variant inputs again")
    ap.add_argument("--dry-run", action="store_true", help="show the plan and the projected time only")
    args = ap.parse_args()
    args.project = os.path.abspath(args.project)
    args.code = cm.code_dir(args.project)
    args.rl1 = cm.rl1_dir(args.project, args.rl1_dir)
    model = os.path.abspath(args.rl1_model or os.path.join(args.rl1, "model", "model_best.pt"))
    args.results = os.path.abspath(args.results)
    sel = cm.select(EXP, args.priority, args.experiments, args.variants)
    if not sel:
        raise SystemExit("no variant selected")

    print(f"{len(sel)} variant(s), methods {args.methods}, sizes {args.sizes}"
          f"{', files ' + ' '.join(args.only) if args.only else ''}; MILP {args.time_limit:g} s, "
          f"{args.workers} workers x {args.threads} threads; results in {args.results}", flush=True)
    problems = []
    if "milp" in args.methods and not os.path.exists(os.path.join(args.code, "solve_milp_f3.py")):
        problems.append(f"solve_milp_f3.py not found in {args.code}")
    if {"rl1", "rl1_retrain"} & set(args.methods) and not os.path.exists(os.path.join(args.rl1, "code", "solve_rl1.py")):
        problems.append(f"solve_rl1.py not found in {args.rl1}/code (give --rl1-dir)")
    if "rl1" in args.methods and not os.path.exists(model):
        problems.append(f"RL1 model not found: {model} (give --rl1-model)")
    for p_ in problems:
        print(f"WARNING: {p_}; that method's steps will fail and be logged", flush=True)
    try:
        import project_timing as pt
        rows = pt.project(sel, EXP, args.sizes, args.only, args.workers, args.time_limit, args.project, args.rl1_dir)
        md, cum = pt.table(rows, args.methods)
        print("\nProjected time (from the baseline runs' measured times):\n" + md, flush=True)
        total_h = (cum["milp"] if "milp" in args.methods else 0) + (cum["rl"] / 60 if "rl1" in args.methods else 0) \
            + (cum["rt"] / 60 if "rl1_retrain" in args.methods else 0)
        print(f"projected total: {total_h:.1f} h (MILP worst case {cum['worst']:.1f} h)\n", flush=True)
    except Exception as e:
        print(f"(no time projection: {type(e).__name__}: {e})", flush=True)

    for v in sel:
        print(f"  {v['priority']}  {v['id']:<16} {v['name']} = {cm.value_text(v['kind'], v['value'])} "
              f"(baseline {cm.value_text(v['kind'], v['baseline'])})", flush=True)
    if args.dry_run:
        v = sel[0]
        folder = os.path.join(args.variants_dir, v["id"])
        print("\ncommands of the first variant:")
        print("  " + " ".join(milp_cmd(v, folder, os.path.join(args.results, v["id"], "milp"), args, EXP)))
        print("  " + " ".join(rl1_solve_cmd(v, folder, os.path.join(args.results, v["id"], "rl1"), model,
                                            S["rl1_run_id"], args, EXP)))
        return

    os.makedirs(args.results, exist_ok=True)
    t_all = time.time()
    try:
        for k, v in enumerate(sel, start=1):
            print(f"\n##### [{k}/{len(sel)}] {v['id']}  ({(time.time() - t_all) / 3600:.1f} h so far)", flush=True)
            try:
                folder = mv.build(v, EXP, args.variants_dir, args.project, force=args.rebuild_variants)
            except Exception as e:
                print(f"{v['id']}: variant inputs not written ({type(e).__name__}: {e}); skipped", flush=True)
                log_progress(args.results, {"variant": v["id"], "experiment": v["experiment"],
                                            "priority": v["priority"], "method": "inputs",
                                            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                                            "return_code": f"error: {e}"})
                continue
            for method in args.methods:
                out = os.path.join(args.results, v["id"], cm.METHOD_DIRS[method])
                if method == "milp":
                    run_step(v, method, milp_cmd(v, folder, out, args, EXP), out, S["milp_run_id"], args, EXP,
                             args.code)
                elif method == "rl1":
                    run_step(v, method, rl1_solve_cmd(v, folder, out, model, S["rl1_run_id"], args, EXP), out,
                             S["rl1_run_id"], args, EXP, os.path.join(args.rl1, "code"))
                else:
                    mdir = os.path.join(args.models_dir, v["id"])
                    if training_finished(mdir):
                        print(f"{v['id']}: retrained model already finished ({mdir}); not trained again", flush=True)
                    else:
                        run_step(v, "rl1_train", rl1_train_cmd(folder, mdir, args), out, None, args, EXP,
                                 os.path.join(args.rl1, "code"))
                    best = os.path.join(mdir, "model_best.pt")
                    if not os.path.exists(best) and os.path.exists(os.path.join(mdir, "model_last.pt")):
                        best = os.path.join(mdir, "model_last.pt")
                    run_step(v, method, rl1_solve_cmd(v, folder, out, best, S["rl1_retrained_run_id"], args, EXP),
                             out, S["rl1_retrained_run_id"], args, EXP, os.path.join(args.rl1, "code"))
    except KeyboardInterrupt:
        print("\ninterrupted: finished instances are in the tables; the same command continues", flush=True)
        sys.exit(130)
    print(f"\nall steps done in {(time.time() - t_all) / 3600:.2f} h; progress table: "
          f"{os.path.join(args.results, 'progress.csv')}\nnext: python analyze_experiments.py", flush=True)


if __name__ == "__main__":
    main()
