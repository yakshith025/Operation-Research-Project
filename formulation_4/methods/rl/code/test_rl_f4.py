"""Self-checks of the formulation-4 RL (no training; seconds to a few minutes). Run before training:
    python test_rl_f4.py            # every Solomon file at n = 10, 50, 100 (+ Gehring-Homberger at 200)
    python test_rl_f4.py --quick    # a handful of instances
1. Coefficients: every EV's mu, c^D, c^Q, c^C, c^V, W in the instance dict equals f4_params' line of its
   segment at its theta; commercial EVs have c^D = c^C = c^V.
2. Simulator = greedy of the MILP: the greedy cost rule run inside the simulator gives the same plan cost
   as solve_milp_f4.greedy_f4, and the simulator's Z equals check_plan_f4's Z.
3. Random policy: plans of an untrained network with no unplaced EV pass check_plan_f4 with the same Z as
   the simulator; plans with unplaced EVs are complete and flagged; teacher forcing = greedy rule.
4. Generator: random training instances of every class run through the simulator and the checker; the
   commercial share of the generated EVs is close to pi_commercial.
5. Overrides: --set config.n_vans / van_max_evs / van_battery_kWh and a parameter reach the instance,
   the simulator and the generator.
6. One training step: finite loss and gradients.
Prints PASS / FAIL per check; the exit code is 1 if a check failed.
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
F4_DIR = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(F4_DIR, "common"))
sys.path.insert(0, os.path.join(F4_DIR, "methods", "milp", "code"))
import f4_data as fd  # noqa: E402
import f4_params as fp  # noqa: E402
from check_plan_f4 import validate  # noqa: E402
from solve_milp_f4 import greedy_f4  # noqa: E402  (gurobipy is not needed for the greedy plan)
from rl_f4_env import Batch, Env, GreedyRule, heuristic_rollout, plan_of  # noqa: E402
from rl_f4_generator import Generator  # noqa: E402
from rl_f4_model import Policy, rollout  # noqa: E402
from train_rl_f4 import imitation_step, load_rl_params, pick_device, train_step  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=fd.DEFAULT_DATA)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="auto", help="auto (GPU if torch sees one) / cpu / cuda")
    args = ap.parse_args()
    dev = pick_device(args.device)
    print(f"device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""), flush=True)
    torch.manual_seed(0)
    P, _ = fp.load_params()
    RP = load_rl_params()
    ds = fd.Dataset(args.data, P)
    files = ds.files()
    if args.quick:
        files = [f for f in files if f in ("EV-C101", "EV-R101", "EV-RC201", "EV-R1_2_1")] or files[:3]
    sizes = {100: (10, 50, 100), 200: (200,)}
    tasks = [(f, n) for f in files for n in sizes.get(ds.n_evs(f), (ds.n_evs(f),)) if n <= ds.n_evs(f)]
    if args.quick:
        tasks = [(f, n) for f, n in tasks if n in (50, 200)]
    results = {}

    # 1 ------------------------------------------------------------------
    bad, cnt = [], 0
    for f, n in tasks:
        J = ds.instance(f, n)
        ev = J["ev"]
        for i in range(n):
            seg = fp.SEGMENTS[int(ev["segment"][i])]
            S = P[seg]
            th = ev["theta"][i]
            want = {"mu": fp.line(S["money_sensitivity_mu"], th), "W": max(0.0, fp.line(S["queue_tolerance_W_min"], th)),
                    **{"c" + a: max(0.0, fp.line(S["value_of_time_Rs_per_min"][a], th)) for a in "DQVC"}}
            for k, v in want.items():
                if abs(float(ev[k][i]) - float(v)) > 1e-9:
                    bad.append(f"{f}_n{n} EV{i + 1} {k} {ev[k][i]} vs {v}")
            if seg == "commercial" and not (abs(ev["cD"][i] - ev["cC"][i]) < 1e-9 and abs(ev["cV"][i] - ev["cC"][i]) < 1e-9):
                bad.append(f"{f}_n{n} EV{i + 1}: commercial with c^D, c^C, c^V not equal")
            cnt += 1
    results["1 per-EV coefficients"] = (not bad, f"{cnt} EVs checked", bad[:5])

    # 2 ------------------------------------------------------------------
    bad, same, skipped = [], 0, 0
    for f, n in tasks:
        J = ds.instance(f, n)
        g = greedy_f4(J)
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
                bad.append(f"{f}_n{n}: greedy_f4 found a plan, simulator dropped {dropped[:5]}")
            elif abs(envZ - g["Z_Rs"]) > 0.011:
                bad.append(f"{f}_n{n}: simulator greedy Z {envZ:.2f} vs greedy_f4 {g['Z_Rs']:.2f}")
            else:
                same += 1
        elif g is None and not dropped:
            bad.append(f"{f}_n{n}: greedy_f4 failed but the simulator placed every EV")
        else:
            skipped += 1
    results["2 simulator = greedy_f4 and checker"] = (not bad, f"{same} identical greedy plans, {skipped} "
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
    bad, cnt, segs = [], 0, []
    gen = Generator(ds, RP["generator"], P, seed=5, file_sizes=(100, 200))
    for key in gen.keys:
        for _ in range(2 if args.quick else 4):
            J = gen.sample(30 if args.quick else 60, key)
            segs += list(np.asarray(J["ev"]["segment"]))
            env = heuristic_rollout(Env(Batch([J], dev)))
            plan, dropped = plan_of(env, 0, J)
            err, Z, _, _ = validate(J, plan)
            cnt += 1
            if not dropped and (err or abs(Z - float(env.cost[0])) > 1e-6 * max(1.0, Z)):
                bad.append(f"{key}: {err[:2]} Z {Z:.2f} vs {float(env.cost[0]):.2f}")
    share = float(np.mean(segs))
    pc = P["segment_distribution"]["pi_commercial"]
    if abs(share - pc) > 0.06:
        bad.append(f"commercial share of generated EVs {share:.3f} vs pi_commercial {pc}")
    results["4 generator instances"] = (not bad, f"{cnt} random instances ({len(gen.keys)} templates), commercial "
                                                 f"share {share:.3f} (pi_C = {pc})", bad[:5])

    # 5 ------------------------------------------------------------------
    bad = []
    sets = ["config.n_vans=3", "config.van_max_evs=8", "config.van_battery_kWh=200",
            "personal.value_of_time_Rs_per_min.C.theta1=12.5"]
    P2, cfg2 = fp.load_params(sets=sets, quiet=True)
    ds2 = fd.Dataset(args.data, P2, cfg2)
    f0 = tasks[0][0]
    J2 = ds2.instance(f0, 20)
    if (J2["n_vans"], J2["params"]["Q"], J2["params"]["B_kWh"]) != (3, 8, 200.0):
        bad.append(f"config overrides not in the instance: {(J2['n_vans'], J2['params']['Q'], J2['params']['B_kWh'])}")
    pers = [i for i in range(20) if J2["ev"]["segment"][i] == 0]
    if pers:
        i = pers[0]
        want = 1.0 + (12.5 - 1.0) * J2["ev"]["theta"][i]
        if abs(J2["ev"]["cC"][i] - want) > 1e-9:
            bad.append(f"parameter override not in the instance: cC {J2['ev']['cC'][i]} vs {want}")
    env2 = Env(Batch([J2], dev))
    if env2.M != 3:
        bad.append(f"simulator has {env2.M} vans, expected 3")
    gen2 = Generator(ds2, RP["generator"], P2, seed=1, file_sizes=(100,))
    Jg = gen2.sample(20)
    if (Jg["n_vans"], Jg["params"]["Q"], Jg["params"]["B_kWh"]) != (3, 8, 200.0):
        bad.append("generator templates do not carry the config overrides")
    results["5 overrides reach instance, simulator, generator"] = (not bad, f"{len(sets)} overrides on {f0}", bad[:5])

    # 6 ------------------------------------------------------------------
    TR = dict(RP["training"], n_min=10, n_max=15, batch_instances=4, samples=4)
    gen1 = Generator(ds, RP["generator"], P, seed=6, file_sizes=(100,))
    optim = torch.optim.Adam(policy.parameters(), lr=1e-4)
    try:
        im = imitation_step(policy, optim, gen1, TR, dev, np.random.default_rng(0))
        st = train_step(policy, optim, gen1, TR, dev, np.random.default_rng(0))
        finite = all(torch.isfinite(p).all() for p in policy.parameters()) and np.isfinite(im["nll"])
        results["6 training step"] = (finite, f"imitation loss {im['nll']:.3f}; REINFORCE: unplaced "
                                              f"{st['unplaced']:.1f} %, spread {st['spread']:.2f} %", [])
    except Exception as e:
        results["6 training step"] = (False, f"{type(e).__name__}: {e}", [])

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
