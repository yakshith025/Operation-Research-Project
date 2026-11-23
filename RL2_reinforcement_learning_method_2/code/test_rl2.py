"""Self-checks of RL2 (no training, a few seconds to minutes). Run before training:
    python test_rl2.py            # every Solomon file at n = 10, 50, 100 (+ Homberger at 200)
    python test_rl2.py --quick    # a handful of instances
1. CSV loader = text-file loader: rl2_data builds exactly the instance of solve_milp_f3.load_instance.
2. Simulator = greedy of solve_milp_f3: the greedy cost rule run inside the simulator gives the same
   plan cost as solve_milp_f3.greedy_f3, and the simulator's Z equals check_plan_f3's Z.
3. Random policy: plans of an untrained network with no unplaced EV pass check_plan_f3 with no
   violation and the same Z as the simulator; plans with unplaced EVs are complete and flagged.
4. Generator: random training instances of every class run through the simulator and the checker.
5. One training step: finite loss and gradients.
6. Active search (RL2): replaying the best plan's actions rebuilds the same plan; a few iterations run,
   never return a worse plan, leave the starting embeddings untouched, and the plan passes the checker.
Prints PASS / FAIL per check; the exit code is 1 if a check failed.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

from rl2_data import DEFAULT_CSV, DEFAULT_DATASET, CsvDataset
import make_income as inc
from check_plan_f3 import validate
from solve_milp_f3 import greedy_f3, load_instance
from rl2_env import Batch, Env, GreedyRule, heuristic_rollout, plan_of
from rl2_generator import Generator
from rl2_model import Policy, rollout
from train_rl2 import imitation_step, load_rl2_params, pick_device, train_step
from rl2_eas import ReplayTeacher, active_search
from solve_rl2 import sampling_search


def compare(a, b, path=""):
    """Differences between two nested structures (numbers to 1e-9)."""
    out = []
    if isinstance(b, dict):
        for k in b:
            if k in ("greedy_baseline", "greedy_s"):
                continue
            if k not in a:
                out.append(f"{path}/{k} missing")
            else:
                out += compare(a[k], b[k], f"{path}/{k}")
    elif isinstance(b, (list, tuple, np.ndarray)) and not isinstance(b, str):
        A, Bv = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
        if A.shape != Bv.shape:
            out.append(f"{path} shape {A.shape} vs {Bv.shape}")
        else:
            try:
                if not np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=0, atol=1e-9):
                    out.append(f"{path} values differ")
            except (TypeError, ValueError):
                if list(A.ravel()) != list(Bv.ravel()):
                    out.append(f"{path} differs")
    elif isinstance(b, float) or isinstance(a, float):
        if abs(float(a) - float(b)) > 1e-9:
            out.append(f"{path} {a} vs {b}")
    elif a != b:
        out.append(f"{path} {a!r} vs {b!r}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="auto", help="auto (GPU if torch sees one) / cpu / cuda")
    args = ap.parse_args()
    dev = pick_device(args.device)
    print(f"device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""), flush=True)
    torch.manual_seed(0)
    P3 = inc.load_params()
    RP = load_rl2_params()
    ds = CsvDataset(args.csv, P3)
    files = ds.files()
    if args.quick:
        files = [f for f in files if f in ("EV-C101", "EV-R101", "EV-RC201", "EV-R1_2_1")] or files[:3]
    sizes = {100: (10, 50, 100), 200: (200,)}
    tasks = [(f, n) for f in files for n in sizes.get(ds.n_evs(f), (ds.n_evs(f),)) if n <= ds.n_evs(f)]
    if args.quick:
        tasks = [(f, n) for f, n in tasks if n in (50, 200)]
    results = {}

    # 1 ------------------------------------------------------------------
    inst_dir = os.path.join(DEFAULT_DATASET, "instances")
    bad, done = [], 0
    for f, n in tasks:
        txt = glob.glob(os.path.join(inst_dir, "*", f + ".txt"))
        if not txt:
            continue
        diff = compare(ds.instance(f, n), load_instance(txt[0], n, P3, inst_dir, greedy=False))
        done += 1
        if diff:
            bad.append(f"{f}_n{n}: {diff[:3]}")
    results["1 CSV loader = text loader"] = (not bad, f"{done} instances compared" if done else
                                             "skipped (no text files found)", bad[:5])

    # 2 ------------------------------------------------------------------
    bad, same, skipped = [], 0, 0
    for f, n in tasks:
        J = ds.instance(f, n)
        g = greedy_f3(J)
        env = heuristic_rollout(Env(Batch([J], dev)))
        plan, dropped = plan_of(env, 0, J)
        err, Z, _, _ = validate(J, plan)
        envZ = float(env.cost[0])
        if not dropped:
            if err:
                bad.append(f"{f}_n{n}: simulator plan invalid: {err[:2]}")
            if abs(Z - envZ) > 1e-6 * max(1.0, abs(Z)):
                bad.append(f"{f}_n{n}: simulator Z {envZ:.4f} vs checker {Z:.4f}")
        if g is not None and g.get("rule") == "cost":
            if dropped:
                bad.append(f"{f}_n{n}: greedy_f3 found a plan, simulator dropped {dropped[:5]}")
            elif abs(envZ - g["Z_Rs"]) > 0.011:
                bad.append(f"{f}_n{n}: simulator greedy Z {envZ:.2f} vs greedy_f3 {g['Z_Rs']:.2f}")
            else:
                same += 1
        elif g is None and not dropped:
            bad.append(f"{f}_n{n}: greedy_f3 failed but the simulator placed every EV")
        else:
            skipped += 1
    results["2 simulator = greedy_f3 and checker"] = (not bad, f"{same} identical greedy plans, {skipped} "
                                                      f"where the cost rule fails in both", bad[:5])

    # 3 ------------------------------------------------------------------
    policy = Policy(**RP["model"]).to(dev)
    bad, valid, flagged, forced = [], 0, 0, 0
    with torch.no_grad():
        for f, n in tasks[: (4 if args.quick else 12)]:
            J = ds.instance(f, n)
            batch = Batch([J], dev)
            env = Env(batch.repeat(8))
            emb = policy.encode(*env.static_features(torch.arange(8, device=dev)))
            rollout(policy, env, emb)
            for row in range(env.B):
                plan, dropped = plan_of(env, row, J)
                err, Z, _, _ = validate(J, plan)
                if dropped:
                    flagged += 1
                    if not err:
                        bad.append(f"{f}_n{n} row {row}: dropped EVs but no violation reported")
                    continue
                valid += 1
                if err:
                    bad.append(f"{f}_n{n} row {row}: {err[:2]}")
                if abs(Z - float(env.cost[row])) > 1e-6 * max(1.0, abs(Z)):
                    bad.append(f"{f}_n{n} row {row}: Z {float(env.cost[row]):.4f} vs checker {Z:.4f}")
            # the policy's code path driven by the greedy rule must give the greedy plan
            env_t, env_h = Env(batch), heuristic_rollout(Env(batch))
            rollout(policy, env_t, policy.encode(*env_t.static_features()), teacher=GreedyRule(env_t))
            if not torch.equal(env_t.res, env_h.res) or abs(float(env_t.cost[0] - env_h.cost[0])) > 1e-6:
                bad.append(f"{f}_n{n}: teacher-forced rollout differs from the greedy rule")
            else:
                forced += 1
    results["3 policy plans pass the checker"] = (not bad, f"{valid} complete random-policy plans valid, "
                                                  f"{flagged} with unplaced EVs flagged, {forced} teacher-forced "
                                                  f"rollouts = greedy rule", bad[:5])

    # 4 ------------------------------------------------------------------
    bad, cnt = [], 0
    gen = Generator(ds, RP["generator"], P3, seed=5, file_sizes=(100, 200))
    for key in gen.keys:
        J = gen.sample(30 if args.quick else 60, key)
        env = heuristic_rollout(Env(Batch([J], dev)))
        plan, dropped = plan_of(env, 0, J)
        err, Z, _, _ = validate(J, plan)
        cnt += 1
        if not dropped and (err or abs(Z - float(env.cost[0])) > 1e-6 * max(1.0, Z)):
            bad.append(f"{key}: {err[:2]} Z {Z:.2f} vs {float(env.cost[0]):.2f}")
    results["4 generator instances"] = (not bad, f"{cnt} random instances ({len(gen.keys)} templates)", bad[:5])

    # 5 ------------------------------------------------------------------
    TR = dict(RP["training"], n_min=10, n_max=15, batch_instances=4, samples=4)
    gen1 = Generator(ds, RP["generator"], P3, seed=6, file_sizes=(100,))
    optim = torch.optim.Adam(policy.parameters(), lr=1e-4)
    try:
        im = imitation_step(policy, optim, gen1, TR, dev, np.random.default_rng(0))
        st = train_step(policy, optim, gen1, TR, dev, np.random.default_rng(0))
        finite = all(torch.isfinite(p).all() for p in policy.parameters()) and np.isfinite(im["nll"])
        results["5 training step"] = (finite, f"imitation loss {im['nll']:.3f}; REINFORCE: unplaced "
                                              f"{st['unplaced']:.1f} %, spread {st['spread']:.2f} %", [])
    except Exception as e:
        results["5 training step"] = (False, f"{type(e).__name__}: {e}", [])

    # 6 ------------------------------------------------------------------ active search (RL2)
    bad, info = [], ""
    try:
        f6 = next(f for f, n in tasks)
        J = ds.instance(f6, min(30, ds.n_evs(f6)))
        batch = Batch([J], dev)
        start, emb_u, total = sampling_search(policy, batch, J, samples=4, augment=2, chunk=1040, device=dev)
        # the replayed action sequence must rebuild exactly the same plan
        env_r = Env(batch)
        rollout(policy, env_r, Policy.select(emb_u, torch.arange(1, device=dev)),
                teacher=ReplayTeacher(*start["actions"], dev))
        if abs(float(env_r.cost[0]) - start["cost"]) > 1e-6 or int(env_r.dropped.sum()) != len(start["dropped"]):
            bad.append(f"replay gives Z {float(env_r.cost[0]):.2f}, the sampled plan has {start['cost']:.2f}")
        cfg = {"iterations": 3, "seconds": 600, "samples": 4, "lr": 0.004, "imitation_weight": 0.01}
        before = {k: v.detach().clone() for k, v in emb_u.items()}
        best, iters, secs = active_search(policy, J, batch, emb_u, start, cfg, 10.0, dev, 1)
        if best["key"] > start["key"] + 1e-6:
            bad.append("active search returned a worse plan than its start")
        if iters != 3:
            bad.append(f"{iters} iterations instead of 3")
        if any(not torch.equal(before[k], emb_u[k]) for k in emb_u):
            bad.append("the starting embeddings were changed in place")
        err, Z, _, _ = validate(J, best["plan"])
        if not best["dropped"] and (err or abs(Z - best["cost"]) > 1e-6 * max(1.0, Z)):
            bad.append(f"active-search plan: {err[:2]} Z {Z:.2f} vs {best['cost']:.2f}")
        info = (f"{f6}_n{J['n_ev']}: sampling Z {start['cost']:,.2f}, after {iters} active-search iterations "
                f"{best['cost']:,.2f} ({secs:.1f} s); replay of the incumbent = same plan")
    except Exception as e:
        bad.append(f"{type(e).__name__}: {e}")
    results["6 active search"] = (not bad, info, bad[:5])

    ok_all = True
    print()
    for name, (ok, info, detail) in results.items():
        ok_all &= ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {info}")
        for d in detail:
            print(f"      {d}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
