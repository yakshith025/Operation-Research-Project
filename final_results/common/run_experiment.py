"""Run the variants of one or more experiments (experiments.json) with the formulation-4 MILP or RL.

The experiment folders call this (code/run_<E>_MILP.py, code/run_<E>_RL.py); it can also be run directly.
For every selected variant, in priority order:
  milp  baseline_MILP/code/solve_milp_f4.py --set <overrides> --variant <id>, the same settings as the
        baseline run (300 s, 8 workers x 4 threads, Gurobi seed 0, greedy warm start)
                                              -> <experiment>_MILP/results/<variant>/  (run ids milp_300s, greedy)
  rl    zero_shot: the baseline model (baseline_RL/model/model_best.pt) on the variant's inputs
                                              -> <experiment>_RL/results/<variant>/  (run id rl_f4_zero_shot)
        retrain:   a NEW model trained on the variant (baseline_RL/code/train_rl_f4.py --set ..., the same recipe
                   as the baseline training)  -> <experiment>_RL/model/<variant>/
                   then solved with it        -> <experiment>_RL/results/<variant>/  (run id rl_f4_retrained)
Nothing is copied: the solvers read ../data and apply the overrides in memory. The baseline is not solved
again (the analysis reads baseline_MILP/results and baseline_RL/results).

Never stops on one failure: a step that fails is logged (<experiment>_<METHOD>/results/progress.csv and
results/_runner_logs/) and the next one starts. Every step resumes: instances already in a runs.csv are
skipped; a finished retraining with the same overrides is not repeated. So the same command continues an
interrupted run. At the end the comparison with the baseline is written (compare_experiment.py).

Usage (from common/, or through the wrappers in the experiment folders)
  python run_experiment.py --experiment E5 --method milp
  python run_experiment.py --experiment E5 E6 --method rl --rl-mode both
  python run_experiment.py --priority 1 --method milp --dry-run           # what would run, nothing solved
  python run_experiment.py --experiment E2 --variants E2_vans_7 --method milp --time-limit 300 --workers 8 --threads 4
Small test: --only EV-C101 EV-R101 --sizes 10 20 --time-limit 20 --workers 2 --threads 2 --results-root <folder>
            (RL: --rl-train-extra "--max-epochs 2 --min-epochs 1 --iters-per-epoch 2 --imitation-batches 2")
"""
import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
F4_DIR = os.path.dirname(HERE)
EXPERIMENTS = os.path.join(HERE, "experiments.json")
MILP_SOLVER = os.path.join(F4_DIR, "baseline_MILP", "code", "solve_milp_f4.py")
RL_DIR = os.path.join(F4_DIR, "baseline_RL", "code")
RL_SOLVER = os.path.join(RL_DIR, "solve_rl_f4.py")
RL_TRAIN = os.path.join(RL_DIR, "train_rl_f4.py")
PROGRESS_COLS = ["experiment", "variant", "method", "step", "started", "finished", "minutes", "return_code",
                 "runs_in_table", "status_counts", "command"]


def load_experiments(path=EXPERIMENTS):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def folder_of(exp, method, root=None):
    """<root>/<folder>_MILP or <folder>_RL (root = the formulation-4 folder unless --results-root)."""
    return os.path.join(root or F4_DIR, f"{exp['folder']}_{'MILP' if method == 'milp' else 'RL'}")


def set_args(variant):
    out = []
    for k, v in variant["set"].items():
        out += ["--set", f"{k}={v}"]
    return out


def stream(cmd, log_path, cwd):
    """Run cmd, show its output and append it to log_path. Returns the exit code."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(shlex.quote(c) for c in cmd)}\n")
        log.flush()
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        try:
            proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                    text=True, encoding="utf-8", errors="replace", bufsize=1)
        except OSError as e:
            log.write(f"could not start: {e}\n")
            print(f"could not start {cmd[1]}: {e}", flush=True)
            return -1
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            return proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except Exception:
                pass
            raise


def runs_summary(results_dir, run_id):
    path = os.path.join(results_dir, "runs.csv")
    if not os.path.exists(path):
        return 0, {}
    counts = {}
    try:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("run_id") == run_id:
                    counts[r["status"]] = counts.get(r["status"], 0) + 1
    except OSError:
        return 0, {}
    return sum(counts.values()), counts


def log_progress(results_root, row):
    path = os.path.join(results_root, "progress.csv")
    new = not os.path.exists(path)
    try:
        with open(path, "a", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=PROGRESS_COLS)
            if new:
                wr.writeheader()
            wr.writerow(row)
    except OSError as e:
        print(f"WARNING: progress.csv not written ({e})", flush=True)


def run_step(exp, var, method, step, cmd, results_root, out_dir, run_id, args):
    print(f"\n{'=' * 100}\n{exp['id']} / {var['id']} / {step}: {var['label']}\n{'=' * 100}", flush=True)
    if args.dry_run:
        print(" ".join(shlex.quote(c) for c in cmd), flush=True)
        return 0
    t0 = time.time()
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    log = os.path.join(results_root, "_runner_logs", f"{var['id']}__{step}.log")
    try:
        rc = stream(cmd, log, os.path.dirname(cmd[1]))
    except KeyboardInterrupt:
        raise
    except Exception as e:                       # never stop the experiment because of one step
        print(f"step failed: {type(e).__name__}: {e}", flush=True)
        rc = -1
    n, counts = runs_summary(out_dir, run_id) if run_id else (0, {})
    log_progress(results_root, {"experiment": exp["id"], "variant": var["id"], "method": method, "step": step,
                                "started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "minutes": round((time.time() - t0) / 60, 1), "return_code": rc,
                                "runs_in_table": n, "status_counts": json.dumps(counts),
                                "command": " ".join(shlex.quote(c) for c in cmd)})
    if rc != 0:
        print(f"{exp['id']} / {var['id']} / {step}: exit code {rc} (logged; continuing)", flush=True)
    return rc


def instance_args(cfg, args):
    a = ["--file-size", *[str(s) for s in (args.file_size or cfg["instances"]["file_size"])],
         "--sizes", *[str(s) for s in (args.sizes or cfg["instances"]["sizes"])]]
    if args.only:
        a += ["--only", *args.only]
    if args.families:
        a += ["--families", *args.families]
    return a


def retrain_done(model_dir, variant):
    """A finished training of this variant (same overrides) is not repeated."""
    try:
        meta = json.load(open(os.path.join(model_dir, "train_meta.json")))
    except (OSError, ValueError):
        return False
    if not meta.get("stop_reason") or str(meta.get("stop_reason", "")).startswith(("interrupted", "stopped")):
        return False
    want = {k: v for k, v in variant["set"].items()}
    have = meta.get("overrides") or {}
    same = all(str(have.get(k)) == str(v) or (isinstance(v, (int, float)) and _num(have.get(k)) == float(v))
               for k, v in want.items()) and len(have) == len(want)
    return same and os.path.exists(os.path.join(model_dir, "model_best.pt"))


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    cfg = load_experiments()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", nargs="*", help="experiment ids (E5 E6 E1 E2 E8); default: by --priority")
    ap.add_argument("--priority", nargs="*", type=int, help="run these priorities (default: all, in order)")
    ap.add_argument("--variants", nargs="*", help="only these variant ids")
    ap.add_argument("--method", choices=["milp", "rl"], required=True)
    ap.add_argument("--results-root", default=None,
                    help="where the <experiment>_<METHOD> folders are (default: the formulation-4 folder)")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, solve nothing")
    ap.add_argument("--no-compare", action="store_true", help="skip compare_experiment at the end")
    ap.add_argument("--baseline-dir", default=None,
                    help="baseline results folder for the comparison (default: baseline_MILP/results or baseline_RL/results)")
    # instance selection (default: experiments.json = the baseline instances)
    ap.add_argument("--file-size", nargs="*", type=int)
    ap.add_argument("--sizes", nargs="*", type=int)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--families", nargs="*")
    # MILP
    ap.add_argument("--time-limit", type=float, default=cfg["solver"]["milp_time_limit_s"])
    ap.add_argument("--workers", type=int, default=cfg["solver"]["milp_workers"])
    ap.add_argument("--threads", type=int, default=cfg["solver"]["milp_threads"])
    ap.add_argument("--milp-extra", default="", help="more options for solve_milp_f4.py (one string)")
    # RL
    ap.add_argument("--rl-mode", choices=["both", "zero_shot", "retrain"], default="both")
    ap.add_argument("--rl-model", default=os.path.join(F4_DIR, cfg["baseline"]["rl_model"]),
                    help="baseline model for zero_shot")
    ap.add_argument("--hours", type=float, default=None, help="training time cap per variant (retrain)")
    ap.add_argument("--rl-threads", type=int, default=0, help="torch CPU threads (0 = all)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--rl-train-extra", default="", help="more options for train_rl_f4.py (one string)")
    ap.add_argument("--rl-solve-extra", default="", help="more options for solve_rl_f4.py (one string)")
    args = ap.parse_args()

    exps = cfg["experiments"]
    if args.experiment:
        want = [e.upper() for e in args.experiment]
        exps = [e for e in exps if e["id"].upper() in want]
    elif args.priority:
        exps = [e for e in exps if e["priority"] in args.priority]
    exps = sorted(exps, key=lambda e: (e["priority"], cfg["experiments"].index(e)))
    if not exps:
        print("no experiment selected")
        sys.exit(1)
    py = sys.executable
    S = cfg["solver"]
    summary = []
    for exp in exps:
        root = folder_of(exp, args.method, args.results_root)
        results_root = os.path.join(root, "results")
        os.makedirs(results_root, exist_ok=True)
        variants = [v for v in exp["variants"] if not args.variants or v["id"] in args.variants]
        print(f"\n### {exp['id']} {exp['name']} ({args.method.upper()}): {len(variants)} variant(s) -> {results_root}",
              flush=True)
        for var in variants:
            out = os.path.join(results_root, var["id"])
            if args.method == "milp":
                cmd = [py, MILP_SOLVER, "--out", out, "--variant", var["id"], *set_args(var),
                       *instance_args(cfg, args), "--time-limit", str(args.time_limit), "--workers", str(args.workers),
                       "--threads", str(args.threads), "--run-id", S["milp_run_id"], "--resume",
                       *shlex.split(args.milp_extra)]
                rc = run_step(exp, var, "milp", "milp", cmd, results_root, out, S["milp_run_id"], args)
                summary.append((exp["id"], var["id"], "milp", rc))
                continue
            dev = ["--device", args.device, "--threads", str(args.rl_threads)]
            if args.rl_mode in ("both", "zero_shot"):
                cmd = [py, RL_SOLVER, "--model", args.rl_model, "--out", out, "--variant", var["id"],
                       "--run-id", S["rl_zero_shot_run_id"], *set_args(var), *instance_args(cfg, args), *dev,
                       "--resume", *shlex.split(args.rl_solve_extra)]
                if not os.path.exists(args.rl_model) and not args.dry_run:
                    print(f"zero_shot skipped: no baseline model at {args.rl_model} (train baseline_RL first)", flush=True)
                    summary.append((exp["id"], var["id"], "rl zero_shot", "no model"))
                else:
                    rc = run_step(exp, var, "rl", "zero_shot", cmd, results_root, out, S["rl_zero_shot_run_id"], args)
                    summary.append((exp["id"], var["id"], "rl zero_shot", rc))
            if args.rl_mode in ("both", "retrain"):
                mdir = os.path.join(root, "model", var["id"])
                if retrain_done(mdir, var):
                    print(f"{var['id']}: model already trained for this variant ({mdir}); not repeated", flush=True)
                else:
                    cmd = [py, RL_TRAIN, "--model-dir", mdir, *set_args(var), *dev, *shlex.split(args.rl_train_extra)]
                    if args.hours is not None:
                        cmd += ["--hours", str(args.hours)]
                    rc = run_step(exp, var, "rl", "train", cmd, results_root, None, None, args)
                    summary.append((exp["id"], var["id"], "rl train", rc))
                model = os.path.join(mdir, "model_best.pt")
                cmd = [py, RL_SOLVER, "--model", model, "--out", out, "--variant", var["id"],
                       "--run-id", S["rl_retrained_run_id"], *set_args(var), *instance_args(cfg, args), *dev,
                       "--resume", *shlex.split(args.rl_solve_extra)]
                rc = run_step(exp, var, "rl", "solve_retrained", cmd, results_root, out, S["rl_retrained_run_id"], args)
                summary.append((exp["id"], var["id"], "rl retrained", rc))
        if not args.no_compare and not args.dry_run:
            try:
                import compare_experiment
                compare_experiment.run(exp["id"], args.method, results_root_base=args.results_root,
                                       baseline_dir=args.baseline_dir)
            except Exception as e:
                print(f"comparison of {exp['id']} not written: {type(e).__name__}: {e}", flush=True)
    print("\nsummary (exit code 0 = finished):")
    for row in summary:
        print("  " + " / ".join(str(x) for x in row))


if __name__ == "__main__":
    main()
