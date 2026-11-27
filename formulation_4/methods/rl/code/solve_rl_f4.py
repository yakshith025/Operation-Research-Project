"""Solve the formulation-4 instances with the trained RL policy and write the results in the shared
format of ../../common/results_io_f4.py (the same tables as the MILP and the greedy runs).

An instance = one file of the formulation-4 dataset (../../data) and its first n EVs.
Default: the 56 Solomon files (100 EVs) at n = 10, 20, ..., 100 = 560 instances, as the MILP run.
Per instance the policy builds `augment` x (1 greedy + `samples` sampled) plans (augment = symmetric
copies of the map: rotations / reflections, the instance itself does not change). The plan with the
fewest unplaced EVs, then the lowest Z, is kept and scored by check_plan_f4 (results_io_f4.evaluate),
exactly like a MILP plan. Nothing else touches the plan: no repair, no heuristic replacement.
If some EVs cannot be placed within the limits, they are appended to the resource that finishes
them earliest; the plan is stored with status invalid_plan and its violations (valid = no).

Usage (from methods/rl/code)
  python solve_rl_f4.py                             # 56 Solomon files x n = 10..100
  python solve_rl_f4.py --file-size 100 200         # also the 60 Gehring-Homberger files (n = 10..200)
  python solve_rl_f4.py --only EV-R101 --sizes 50 --details
  python solve_rl_f4.py --resume                    # skip instances already in runs.csv
  python solve_rl_f4.py --set config.n_vans=3 --variant E2_vans_3 --model <model> --out <folder> --run-id <id>
                                                    # an experiment variant (the experiment folders call this)
The run never stops because of one instance (status error + traceback in logs/), and results are
appended after every instance. With --resume, a changed model (new training) re-solves everything.

Outputs in --out (default ../results), schema of ../../common/results_io_f4.py:
  runs.csv    one row per run (run_id, method RL-F4, seed, instance, n): variant, status (plan / invalid_plan /
              error), valid, Z1-Z6, service numbers, solve_s (inference), train_s (training of the
              model), message (rollouts, unplaced EVs)
  plan.csv    one row per EV (resource, position, start / end, waits, money) and per van
  meta.json   settings of the run id (model, training summary, inference settings)
  logs.zip    console output and error tracebacks
  analysis/   segment x income insights (../../common/analyze_f4.py), unless --no-analysis
"""
import argparse
import os
import sys
import time
import traceback
import zlib

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RL_DIR = os.path.abspath(os.path.join(HERE, ".."))
F4_DIR = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if os.path.join(F4_DIR, "common") not in sys.path:
    sys.path.insert(0, os.path.join(F4_DIR, "common"))
import f4_params as fp             # noqa: E402  (parameters, --set overrides)
import f4_data as fd               # noqa: E402  (the formulation-4 dataset, data/*.csv)
import results_io_f4 as rio        # noqa: E402
from rl_f4_env import Batch, Env, plan_of  # noqa: E402
from rl_f4_model import Policy, load_policy, rollout  # noqa: E402
from train_rl_f4 import DEFAULT_MODEL_DIR, DEFAULT_RL_PARAMS, load_rl_params, pick_device  # noqa: E402

DEFAULT_OUT = os.path.join(RL_DIR, "results")
METHOD = "RL-F4"
VARIANT = {"id": "baseline"}       # set in main() from --variant


class Tee:
    """Console output also written to logs/<run_id>.console.log (packed into logs.zip)."""

    def __init__(self, path):
        self.fh = open(path, "a")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        try:
            self.fh.write(s)
        except (OSError, ValueError):
            pass

    def flush(self):
        self.out.flush()
        try:
            self.fh.flush()
        except (OSError, ValueError):
            pass


@torch.no_grad()
def solve_instance(policy, J, samples, augment, chunk, device, seed):
    """Best plan of augment x (1 + samples) rollouts. Returns (plan, dropped EVs, simulator Z, rollouts)."""
    torch.manual_seed(seed)
    batch = Batch([J], device)
    A = max(1, min(8, int(augment)))
    per = 1 + max(0, int(samples))
    total = A * per
    env_u = Env(batch.repeat(A))
    emb_u = policy.encode(*env_u.static_features(torch.arange(A, device=device)))
    aug_of = torch.arange(A, device=device).repeat_interleave(per)
    greedy = (torch.arange(total, device=device) % per) == 0
    best = None
    step = max(per, int(chunk) // per * per)
    for lo in range(0, total, step):
        idx = torch.arange(lo, min(total, lo + step), device=device)
        env = Env(batch.repeat(len(idx)))
        rollout(policy, env, Policy.select(emb_u, aug_of[idx]), greedy=greedy[idx])
        key = env.dropped.sum(1).double() * 1e12 + env.cost
        row = int(torch.argmin(key))
        if best is None or float(key[row]) < best[0]:
            plan, dropped = plan_of(env, row, J)
            best = (float(key[row]), plan, dropped, float(env.cost[row]))
    return best[1], best[2], best[3], total


def run_keys(J, run_id, seed):
    return {"run_id": run_id, "method": METHOD, "seed": seed, "instance": J["file"], "source": J["source"],
            "class": J["family"], "n": J["n_ev"], "variant": VARIANT["id"]}


def run_one(ds, name, n, policy, args, train_s, logs):
    t0 = time.time()
    J, plan_rows = None, []
    keys = {"run_id": args.run_id, "method": METHOD, "seed": args.seed, "instance": name,
            "source": ds.source.get(name, ""), "class": ds.config.get(name, {}).get("class", ""), "n": n,
            "variant": VARIANT["id"]}
    try:
        J = ds.instance(name, n)
        keys = run_keys(J, args.run_id, args.seed)
        t1 = time.time()
        seed = args.seed * 1000003 + zlib.crc32(f"{name}_{n}".encode())
        plan, dropped, envZ, total = solve_instance(policy, J, args.samples, args.augment, args.chunk,
                                                    args.device, seed % (2 ** 31))
        solve_s = time.time() - t1
        metrics, rows, err = rio.evaluate(J, plan)
        msg = f"best of {total} rollouts ({args.augment} map symmetries x (1 greedy + {args.samples} sampled))"
        if dropped:
            msg += (f"; {len(dropped)} EV(s) could not be placed within the limits ({dropped[:8]}"
                    f"{' ...' if len(dropped) > 8 else ''}); appended to the earliest-finishing resource")
        elif abs(envZ - metrics["Z_exact"]) > 1e-6 * max(1.0, abs(envZ)):
            msg += f"; WARNING simulator Z {envZ:.4f} differs from the checker"
        status = "plan" if not err else "invalid_plan"
        row = rio.run_row(keys, status, metrics, solve_s=round(solve_s, 3), train_s=train_s, message=msg)
        plan_rows = rio.with_key(keys, rows)
        if args.details:
            for r in rows:
                if r["type"] == "ev":
                    print(f"  EV{r['id']:>3} {r['resource']:<14} start {r['start_min']:8.1f} end {r['end_min']:8.1f} "
                          f"queue {r['queue_min']:6.1f} van wait {r['van_wait_min']:6.1f} cost {r['customer_cost_Rs']}")
                else:
                    print(f"  van{r['id']}: {r['stops']} stops, {r['km']} km, battery left {r['battery_left_kWh']} kWh, "
                          f"back {r['return_min']}")
    except Exception as e:
        row = rio.run_row(keys, "error", train_s=train_s, message=f"{type(e).__name__}: {e}")
        try:
            with open(os.path.join(logs, rio.log_stem(args.run_id, name, n) + ".error.log"), "w") as fh:
                fh.write(traceback.format_exc())
        except OSError:
            pass
    row["wall_clock_s"] = round(time.time() - t0, 2)
    return row, plan_rows


def collect_tasks(ds, args):
    tasks, too_big = [], 0
    for name in ds.files():
        size = ds.n_evs(name)
        if args.only and name not in args.only:
            continue
        if args.families and ds.family(name).upper() not in [f.upper() for f in args.families]:
            continue
        if args.file_size and size not in args.file_size:
            continue
        for n in (args.sizes or range(10, size + 1, 10)):
            if 1 <= n <= size:
                tasks.append((name, n))
            else:
                too_big += 1
    if too_big:
        print(f"{too_big} requested size(s) are larger than their file and do not exist", flush=True)
    tasks.sort(key=lambda t: (t[1], t[0]))
    return tasks


def main():
    RP = load_rl_params(DEFAULT_RL_PARAMS)
    INF = RP["inference"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=fd.DEFAULT_DATA, help="formulation-4 dataset folder (../../data)")
    ap.add_argument("--params", default=fp.DEFAULT_PARAMS, help="../../common/f4_params.json")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="override a parameter or a config column (as solve_milp_f4.py); repeatable")
    ap.add_argument("--variant", default="baseline", help="name of the experiment variant (column variant)")
    ap.add_argument("--model", default=os.path.join(DEFAULT_MODEL_DIR, "model_best.pt"))
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--run-id", default="rl_f4", help="name of this run in runs.csv")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (column seed)")
    ap.add_argument("--only", nargs="*", help="file names, e.g. EV-R101 EV-C1_2_1")
    ap.add_argument("--families", nargs="*", help="e.g. C1 R2 RC1")
    ap.add_argument("--file-size", nargs="*", type=int, default=[100],
                    help="files with this many EVs: 100 = Solomon (default), 200 = Gehring-Homberger")
    ap.add_argument("--sizes", nargs="*", type=int, help="first n EVs; default 10, 20, ..., file size")
    ap.add_argument("--samples", type=int, default=INF["samples"], help="sampled rollouts per symmetry")
    ap.add_argument("--augment", type=int, default=INF["augment"], help="map symmetries used (1..8)")
    ap.add_argument("--chunk", type=int, default=INF["chunk"], help="rollouts simulated together")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = all cores)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", action="store_true", help="skip instances already in runs.csv (same model)")
    ap.add_argument("--fresh", action="store_true", help="first remove every run of this run id")
    ap.add_argument("--details", action="store_true", help="print the plan of every instance")
    ap.add_argument("--no-analysis", action="store_true", help="skip analyze_f4 at the end")
    args = ap.parse_args()
    VARIANT["id"] = args.variant
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    args.device = pick_device(args.device)
    out = args.out
    P = rio.paths(out)
    os.makedirs(P["logs"], exist_ok=True)
    sys.stdout = Tee(os.path.join(P["logs"], f"{args.run_id}.console.log"))

    if not os.path.exists(args.model):
        last = os.path.join(os.path.dirname(args.model), "model_last.pt")
        if os.path.exists(last):
            args.model = last
        else:
            print(f"no trained model at {os.path.abspath(args.model)}: run train_rl_f4.py first", flush=True)
            sys.exit(1)
    policy, ck = load_policy(args.model, args.device)
    st = ck.get("state", {})
    train_s = round(float(st.get("train_s", 0.0)), 1)
    model_id = f"{os.path.abspath(args.model)}|{ck.get('saved', '')}|epoch {st.get('epoch')}"

    P4, cfg_over = fp.load_params(args.params, args.set)
    ds = fd.Dataset(args.data, P4, cfg_over)
    if ds.missing:
        print(f"note: files of {ds.missing} not found in {ds.path}", flush=True)
    tasks = collect_tasks(ds, args)
    if not tasks:
        print(f"no instances selected (data folder {ds.path})", flush=True)
        sys.exit(1)

    old = rio.read_meta(out).get(args.run_id, {})
    if args.resume and old.get("model_id") not in (None, model_id):
        print(f"the model changed since the last {args.run_id} run ({old.get('model_id')} -> {model_id}): "
              f"every instance is solved again", flush=True)
        args.fresh, args.resume = True, False
    rio.merge_parts(out)
    existing = {rio.key_of(r) for r in rio.read_runs(out)}
    if args.fresh:
        existing = {k for k in existing if k[0] != args.run_id}

    def key(name, n):
        return (args.run_id, METHOD, str(args.seed), name, str(n))
    todo = [(nm, n) for nm, n in tasks if not (args.resume and key(nm, n) in existing)]
    redo = {(nm, str(n)) for nm, n in todo}

    def drop(k, _row):
        if args.fresh and k[0] == args.run_id:
            return True
        return k[0] == args.run_id and k[1] == METHOD and k[2] == str(args.seed) and (k[3], k[4]) in redo
    rio.prepare(out, drop)
    rio.update_meta(out, args.run_id, {
        "method": METHOD, "argv": sys.argv, "variant": args.variant, "overrides": P4["_overrides"],
        "model": os.path.abspath(args.model), "model_id": model_id,
        "model_cfg": ck.get("model_cfg"), "training_state": st, "training_params": ck.get("rl_params"),
        "model_trained_with_overrides": (ck.get("f4_params") or {}).get("_overrides"),
        "train_s": train_s, "samples": args.samples, "augment": args.augment, "seed": args.seed,
        "device": args.device, "threads": torch.get_num_threads(), "data": ds.path, "f4_params": P4,
        "selection": "fewest unplaced EVs, then lowest Z; scored by check_plan_f4 (results_io_f4.evaluate)",
        "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    print(f"RL (formulation 4), variant {args.variant}: {len(tasks)} instance(s), {len(tasks) - len(todo)} already in runs.csv, {len(todo)} to solve; "
          f"model {args.model} (epoch {st.get('epoch')}, {train_s / 3600:.2f} h training); "
          f"{args.augment} x (1 + {args.samples}) rollouts each on {args.device} "
          f"({torch.get_num_threads()} threads); output {os.path.abspath(out)}", flush=True)

    t_start = time.time()
    try:
        for k, (name, n) in enumerate(todo, start=1):
            row, plan_rows = run_one(ds, name, n, policy, args, train_s, P["logs"])
            try:
                rio.write_part(out, [row], plan_rows)
                rio.merge_parts(out)
            except OSError as e:
                print(f"{name}_n{n}: results not written ({e})", flush=True)
            z = f"Z = {row['Z_total']:,.2f} Rs, valid: {row['valid']}" if row["Z_total"] != "" else row["message"]
            print(f"[{k}/{len(todo)}] {name}_n{n}: {row['status']}, {z}, {row['solve_s']} s "
                  f"({(time.time() - t_start) / 60:.1f} min so far)", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted: the instances finished so far are in the tables", flush=True)
    rio.merge_parts(out)
    rio.update_meta(out, args.run_id, {"finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    mine = [r for r in rio.read_runs(out) if r["run_id"] == args.run_id]
    counts = {}
    for r in mine:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"\nstatus of the {len(mine)} run(s) of {args.run_id} in runs.csv: {counts}", flush=True)
    print(f"Results written to {os.path.abspath(out)}", flush=True)
    tee = sys.stdout
    sys.stdout = tee.out
    tee.fh.close()
    rio.pack_logs(out)
    if not args.no_analysis:
        try:
            import analyze_f4
            analyze_f4.run(out, args.run_id)
        except Exception as e:
            print(f"analysis not written: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
