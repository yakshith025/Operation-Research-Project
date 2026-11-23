"""Solve the EV-Solomon instances with the trained RL2 policy and write the results in the shared
format of code/results_io.py (the same tables as the MILP, greedy and RL1 runs).

An instance = one file of the CSV dataset (dataset_ev_solomon/csv, the same data RL1 reads) and its
first n EVs. Default: the 56 Solomon files (100 EVs) at n = 10, 20, ..., 100 = 560 instances.
Every instance gives two runs:
  run id <prefix>_sampling  (method RL2)      as RL1: `augment` symmetric copies of the map x
                                              (1 greedy + `samples` sampled) plans; the best is kept
  run id <prefix>_eas       (method RL2-EAS)  Efficient Active Search (rl2_eas.py) starting from that
                                              plan: the instance embeddings are tuned on this instance
                                              for up to `seconds` s / `iterations` iterations
<prefix> = --run-id-prefix (default rl2). "Best" = fewest unplaced EVs, then lowest Z. Both plans are
scored by check_plan_f3 (results_io.evaluate), exactly like a MILP plan; nothing else touches them.
If some EVs cannot be placed within the limits, they are appended to the resource that finishes them
earliest; the plan is stored with status invalid_plan and its violations (valid = no).

Usage (from RL2_reinforcement_learning_method_2/code)
  python solve_rl2.py                               # 56 Solomon files x n = 10..100, both runs
  python solve_rl2.py --no-eas                      # only the sampling run
  python solve_rl2.py --only EV-R101 --sizes 50 --details
  python solve_rl2.py --model ../../RL1_reinforcement_learning_method_1/model/model_best.pt \
                      --run-id-prefix rl1model      # ablation: active search on RL1's trained model
  python solve_rl2.py --resume                      # skip instances already in runs.csv (same model)
The run never stops because of one instance (status error + traceback in logs/), and results are
appended after every instance.

Outputs in --out (default ../results), schema of code/results_io.py:
  runs.csv    one row per run: status (plan / invalid_plan / error), valid, Z1-Z6, service numbers,
              solve_s (sampling, or sampling + active search), train_s (training of the model),
              message (rollouts, active-search iterations and improvement, unplaced EVs)
  plan.csv    one row per EV (resource, position, start / end, waits, money) and per van
  meta.json   settings of each run id (model, training summary, inference and search settings)
  logs.zip    console output and error tracebacks
"""
import argparse
import os
import sys
import time
import traceback
import zlib

import torch

from rl2_data import DEFAULT_CSV, DEFAULT_PARAMS_F3, RL2_DIR, CsvDataset
import make_income as inc
import results_io as rio
from rl2_eas import active_search, plan_actions
from rl2_env import Batch, Env, plan_of
from rl2_model import Policy, load_policy, rollout
from train_rl2 import DEFAULT_MODEL_DIR, DEFAULT_RL2_PARAMS, load_rl2_params, pick_device

DEFAULT_OUT = os.path.join(RL2_DIR, "results")
M_SAMPLING, M_EAS = "RL2", "RL2-EAS"


class Tee:
    """Console output also written to logs/<prefix>.console.log (packed into logs.zip)."""

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
def sampling_search(policy, batch, J, samples, augment, chunk, device):
    """Step 1 (as RL1): best of augment x (1 greedy + samples sampled) rollouts. Returns the best as a
    dict (key, actions, plan, dropped, cost), the instance embeddings of each symmetry, and the
    number of rollouts."""
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
        if best is None or float(key[row]) < best["key"]:
            plan, dropped = plan_of(env, row, J)
            best = {"key": float(key[row]), "actions": plan_actions(env, row), "plan": plan,
                    "dropped": dropped, "cost": float(env.cost[row])}
    return best, emb_u, total


def run_keys(J, run_id, method, seed):
    return {"run_id": run_id, "method": method, "seed": seed, "instance": J["file"], "source": J["source"],
            "class": J["family"], "n": J["n_ev"]}


def scored(J, keys, best, solve_s, train_s, msg):
    """runs.csv row + plan.csv rows of one plan, scored by the shared checker."""
    metrics, rows, err = rio.evaluate(J, best["plan"])
    if best["dropped"]:
        d = best["dropped"]
        msg += (f"; {len(d)} EV(s) could not be placed within the limits ({d[:8]}{' ...' if len(d) > 8 else ''}); "
                f"appended to the earliest-finishing resource")
    elif abs(best["cost"] - metrics["Z_exact"]) > 1e-6 * max(1.0, abs(best["cost"])):
        msg += f"; WARNING simulator Z {best['cost']:.4f} differs from the checker"
    status = "plan" if not err else "invalid_plan"
    row = rio.run_row(keys, status, metrics, solve_s=round(solve_s, 3), train_s=train_s, message=msg)
    return row, rio.with_key(keys, rows)


def run_one(ds, name, n, policy, args, train_s, logs, drop_penalty):
    """Both runs of one instance. Never raises: a failure becomes status error."""
    t0 = time.time()
    rid_s, rid_e = f"{args.run_id_prefix}_sampling", f"{args.run_id_prefix}_eas"
    base = {"seed": args.seed, "instance": name, "source": ds.source.get(name, ""),
            "class": ds.config.get(name, {}).get("class", ""), "n": n}
    rows, plans = [], []
    J = None
    try:
        J = ds.instance(name, n)
        seed = (args.seed * 1000003 + zlib.crc32(f"{name}_{n}".encode())) % (2 ** 31)
        torch.manual_seed(seed)
        batch = Batch([J], args.device)
        t1 = time.time()
        start, emb_u, total = sampling_search(policy, batch, J, args.samples, args.augment, args.chunk, args.device)
        t_samp = time.time() - t1
        msg = f"best of {total} rollouts ({args.augment} map symmetries x (1 greedy + {args.samples} sampled))"
        r, p = scored(J, run_keys(J, rid_s, M_SAMPLING, args.seed), start, t_samp, train_s, msg)
        r["wall_clock_s"] = round(time.time() - t0, 2)
        rows.append(r)
        plans += p
        if not args.no_eas:
            t2 = time.time()
            best, iters, secs = active_search(policy, J, batch, emb_u, start, args.eas, drop_penalty,
                                              args.device, seed + 1)
            t_eas = time.time() - t2
            change = 100.0 * (best["cost"] - start["cost"]) / start["cost"] if start["cost"] else 0.0
            fewer = len(start["dropped"]) - len(best["dropped"])
            msg = (f"active search (EAS-Emb): {iters} iterations in {secs:.1f} s x {args.augment} symmetries x "
                   f"{args.eas['samples']} samples, from the sampling plan Z {start['cost']:,.2f}"
                   + (f" ({len(start['dropped'])} unplaced)" if start["dropped"] else "")
                   + f" to {best['cost']:,.2f} ({change:+.2f} %"
                   + (f", {fewer} more EVs placed" if fewer > 0 else "")
                   + f"; best found in iteration {best['iteration']})")
            r, p = scored(J, run_keys(J, rid_e, M_EAS, args.seed), best, t_samp + t_eas, train_s, msg)
            r["wall_clock_s"] = round(time.time() - t0, 2)
            rows.append(r)
            plans += p
            if args.details:
                print(f"  {msg}")
    except Exception as e:
        done = {r_["run_id"] for r_ in rows}
        for rid, meth in ((rid_s, M_SAMPLING), (rid_e, M_EAS)):
            if rid in done or (rid == rid_e and args.no_eas):
                continue
            keys = run_keys(J, rid, meth, args.seed) if J is not None else {**base, "run_id": rid, "method": meth}
            r = rio.run_row(keys, "error", train_s=train_s, message=f"{type(e).__name__}: {e}")
            r["wall_clock_s"] = round(time.time() - t0, 2)
            rows.append(r)
        try:
            with open(os.path.join(logs, rio.log_stem(args.run_id_prefix, name, n) + ".error.log"), "w") as fh:
                fh.write(traceback.format_exc())
        except OSError:
            pass
    return rows, plans


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


def short(row):
    if row.get("Z_total", "") != "":
        return f"{row['status']}, Z = {row['Z_total']:,.2f}, valid: {row['valid']}, {row['solve_s']} s"
    return f"{row['status']} ({row.get('message', '')})"


def main():
    RP = load_rl2_params(DEFAULT_RL2_PARAMS)
    INF, EAS = RP["inference"], RP["active_search"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=DEFAULT_CSV, help="dataset_ev_solomon/csv folder (shared with RL1)")
    ap.add_argument("--f3-params", default=DEFAULT_PARAMS_F3, help="code/f3_params.json")
    ap.add_argument("--model", default=os.path.join(DEFAULT_MODEL_DIR, "model_best.pt"))
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--run-id-prefix", default="rl2", help="run ids <prefix>_sampling and <prefix>_eas")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (column seed)")
    ap.add_argument("--only", nargs="*", help="file names, e.g. EV-R101 EV-C1_2_1")
    ap.add_argument("--families", nargs="*", help="e.g. C1 R2 RC1")
    ap.add_argument("--file-size", nargs="*", type=int, default=[100],
                    help="files with this many EVs: 100 = Solomon (default), 200 = Homberger")
    ap.add_argument("--sizes", nargs="*", type=int, help="first n EVs; default 10, 20, ..., file size")
    ap.add_argument("--samples", type=int, default=INF["samples"], help="step 1: sampled rollouts per symmetry")
    ap.add_argument("--augment", type=int, default=INF["augment"], help="map symmetries used (1..8)")
    ap.add_argument("--chunk", type=int, default=INF["chunk"], help="step 1: rollouts simulated together")
    ap.add_argument("--no-eas", action="store_true", help="skip step 2 (active search)")
    ap.add_argument("--eas-iterations", type=int, default=EAS["iterations"])
    ap.add_argument("--eas-seconds", type=float, default=EAS["seconds"], help="time limit of step 2 per instance")
    ap.add_argument("--eas-samples", type=int, default=EAS["samples"], help="step 2: samples per symmetry per iteration")
    ap.add_argument("--eas-lr", type=float, default=EAS["lr"])
    ap.add_argument("--eas-imitation", type=float, default=EAS["imitation_weight"])
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = all cores)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", action="store_true", help="skip instances already in runs.csv (same model)")
    ap.add_argument("--fresh", action="store_true", help="first remove every run of these run ids")
    ap.add_argument("--details", action="store_true", help="print the active-search summary per instance")
    args = ap.parse_args()
    args.eas = {"iterations": args.eas_iterations, "seconds": args.eas_seconds, "samples": args.eas_samples,
                "lr": args.eas_lr, "imitation_weight": args.eas_imitation}
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    args.device = pick_device(args.device)
    out = args.out
    P = rio.paths(out)
    os.makedirs(P["logs"], exist_ok=True)
    sys.stdout = Tee(os.path.join(P["logs"], f"{args.run_id_prefix}.console.log"))

    if not os.path.exists(args.model):
        last = os.path.join(os.path.dirname(args.model), "model_last.pt")
        if os.path.exists(last):
            args.model = last
        else:
            print(f"no trained model at {os.path.abspath(args.model)}: run train_rl2.py first", flush=True)
            sys.exit(1)
    policy, ck = load_policy(args.model, args.device)
    st = ck.get("state", {})
    train_s = round(float(st.get("train_s", 0.0)), 1)
    drop_penalty = float((ck.get("rl2_params") or ck.get("rl1_params") or RP)["training"]["drop_penalty"])
    model_id = f"{os.path.abspath(args.model)}|{ck.get('saved', '')}|epoch {st.get('epoch')}"

    P3 = inc.load_params(args.f3_params)
    ds = CsvDataset(args.csv, P3)
    if ds.missing:
        print(f"note: CSV files of {ds.missing} not found in {ds.csv_dir}", flush=True)
    tasks = collect_tasks(ds, args)
    if not tasks:
        print(f"no instances selected (CSV folder {ds.csv_dir})", flush=True)
        sys.exit(1)

    rid_s, rid_e = f"{args.run_id_prefix}_sampling", f"{args.run_id_prefix}_eas"
    mine = [(rid_s, M_SAMPLING)] + ([] if args.no_eas else [(rid_e, M_EAS)])
    meta = rio.read_meta(out)
    if args.resume and any(meta.get(r, {}).get("model_id") not in (None, model_id) for r, _ in mine):
        print(f"the model changed since the last run of these run ids: every instance is solved again", flush=True)
        args.fresh, args.resume = True, False
    rio.merge_parts(out)
    existing = {rio.key_of(r) for r in rio.read_runs(out)}
    ids = {r for r, _ in mine}
    if args.fresh:
        existing = {k for k in existing if k[0] not in ids}

    def done(name, n):
        return all((r, m, str(args.seed), name, str(n)) in existing for r, m in mine)
    todo = [(nm, n) for nm, n in tasks if not (args.resume and done(nm, n))]
    redo = {(nm, str(n)) for nm, n in todo}

    def drop(k, _row):
        if args.fresh and k[0] in ids:
            return True
        return k[0] in ids and k[2] == str(args.seed) and (k[3], k[4]) in redo
    rio.prepare(out, drop)
    common = {"argv": sys.argv, "model": os.path.abspath(args.model), "model_id": model_id,
              "model_cfg": ck.get("model_cfg"), "training_state": st, "train_s": train_s,
              "samples": args.samples, "augment": args.augment, "seed": args.seed, "device": args.device,
              "threads": torch.get_num_threads(), "csv": ds.csv_dir, "f3_params": P3, "drop_penalty": drop_penalty,
              "selection": "fewest unplaced EVs, then lowest Z; scored by check_plan_f3 (results_io.evaluate)",
              "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    rio.update_meta(out, rid_s, {"method": M_SAMPLING, **common})
    if not args.no_eas:
        rio.update_meta(out, rid_e, {"method": M_EAS, "active_search": args.eas,
                                     "start": f"best plan of run {rid_s}", **common})
    print(f"RL2: {len(tasks)} instance(s), {len(tasks) - len(todo)} already in runs.csv, {len(todo)} to solve; "
          f"model {args.model} (epoch {st.get('epoch')}, {train_s / 3600:.2f} h training); step 1: "
          f"{args.augment} x (1 + {args.samples}) rollouts"
          + ("" if args.no_eas else f"; step 2: active search up to {args.eas['iterations']} iterations / "
                                    f"{args.eas['seconds']} s") +
          f"; {args.device} ({torch.get_num_threads()} threads); output {os.path.abspath(out)}", flush=True)

    t_start = time.time()
    try:
        for k, (name, n) in enumerate(todo, start=1):
            rows, plans = run_one(ds, name, n, policy, args, train_s, P["logs"], drop_penalty)
            try:
                rio.write_part(out, rows, plans)
                rio.merge_parts(out)
            except OSError as e:
                print(f"{name}_n{n}: results not written ({e})", flush=True)
            parts = " | ".join(f"{r['method']}: {short(r)}" for r in rows)
            print(f"[{k}/{len(todo)}] {name}_n{n}: {parts} ({(time.time() - t_start) / 60:.1f} min so far)",
                  flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted: the instances finished so far are in the tables", flush=True)
    rio.merge_parts(out)
    for rid, _ in mine:
        rio.update_meta(out, rid, {"finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    for rid, _ in mine:
        counts = {}
        for r in rio.read_runs(out):
            if r["run_id"] == rid:
                counts[r["status"]] = counts.get(r["status"], 0) + 1
        print(f"status of {rid} in runs.csv: {counts}", flush=True)
    print(f"Results written to {os.path.abspath(out)}", flush=True)
    tee = sys.stdout
    sys.stdout = tee.out
    tee.fh.close()
    rio.pack_logs(out)


if __name__ == "__main__":
    main()
