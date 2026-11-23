"""Train the RL1 policy (attention model, REINFORCE with a shared baseline) for main_formulation_3.

Training uses random instances made with the EV-Solomon rules (rl1_generator.py); the EV-Solomon
files themselves are only used for testing (solve_rl1.py).
Warm-up (first imitation_batches batches, 0 = none): the policy is taught the actions of the greedy
cost rule of solve_milp_f3 (teacher forcing, negative log-likelihood), so REINFORCE starts from a
sensible policy instead of random weights. Then REINFORCE. Each batch: batch_instances random
instances with the same n (n drawn from [n_min, n_max]), each rolled out `samples` times by the
current policy (random map symmetry per instance). For rollout j of instance b
    cost_bj = Z_bj + drop_penalty x scale_b x (EVs that could not be placed)
    advantage_bj = (cost_bj - mean_j cost_bj) / mean_j cost_bj           (shared baseline, POMO)
    loss = mean_bj advantage_bj x log pi(rollout_bj)                      (REINFORCE, Williams 1992)
After every epoch the greedy rollout of the policy is compared with the greedy cost rule of
solve_milp_f3 (same simulator) on fixed validation instances; the best epoch is kept.

Training stops at whichever comes first (all in rl1_params.json):
  max_epochs        total epochs, warm-up included (default 500)
  patience_epochs   early stop: this many REINFORCE epochs in a row without a better validation
                    result (default 10; 0 = off). The count starts only after min_epochs (default
                    200): the first min_epochs epochs always run, so the earliest early stop is at
                    epoch min_epochs + patience_epochs.
  hours             time budget (a safety cap)

Usage (from RL1_reinforcement_learning_method_1/code)
  python train_rl1.py                         # new training from epoch 1, settings of rl1_params.json
  python train_rl1.py --max-epochs 500 --min-epochs 200 --patience 10 --threads 32
  python train_rl1.py --resume                # only if asked: continue model/model_last.pt
Every run without --resume starts at epoch 1 with new weights; the previous run's files (models, log,
meta) are first moved to model/previous_runs/<date-time>/, so nothing is deleted.
The run never stops because of one bad batch (it is logged in train_errors.log and skipped);
Ctrl-C saves the model first.

Outputs in --model-dir (default ../model)
  model_best.pt    network of the best validation epoch (solve_rl1.py uses this one)
  model_last.pt    network + optimiser of the last epoch (for an explicit --resume)
  previous_runs/   files of earlier runs, one folder per run (moved there when a new run starts)
  train_log.csv    one row per epoch: batches, training cost spread, unplaced EVs, validation ratio
                   RL / greedy rule (overall and per n), learning rate, minutes
  train_meta.json  settings, data templates, torch version, total training seconds
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np
import torch

from rl1_data import DEFAULT_CSV, DEFAULT_PARAMS_F3, RL1_DIR, CsvDataset
import make_income as inc
from rl1_env import Batch, Env, GreedyRule, heuristic_rollout
from rl1_generator import Generator
from rl1_model import Policy, rollout

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RL1_PARAMS = os.path.join(HERE, "rl1_params.json")
DEFAULT_MODEL_DIR = os.path.join(RL1_DIR, "model")
LOG_COLUMNS = ["epoch", "phase", "iters", "batches_this_epoch", "failed_batches", "imitation_nll",
               "train_cost_spread_%", "train_unplaced_%", "val_ratio_rl_over_greedy", "val_unplaced_rl_%",
               "val_unplaced_greedy_%", "val_ratio_by_n", "epochs_without_improvement", "lr", "train_min",
               "time"]


def strip_about(d):
    if isinstance(d, dict):
        return {k: strip_about(v) for k, v in d.items() if not k.startswith("_")}
    return d


def load_rl1_params(path=DEFAULT_RL1_PARAMS):
    with open(path, encoding="utf-8") as fh:
        return strip_about(json.load(fh))


def pick_device(name):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def make_validation(dataset, RP, P3, device):
    TR = RP["training"]
    sizes = RP["generator"]["file_sizes"]
    gen = Generator(dataset, RP["generator"], P3, seed=TR["val_seed"], file_sizes=sizes)
    val = []
    for size in sizes:
        for n in TR["val_sizes"]:
            if n > size:
                continue
            batch = Batch(gen.batch(n, TR["val_instances_per_size"], file_size=size), device)
            env = heuristic_rollout(Env(batch))
            val.append({"n": n, "size": size, "batch": batch,
                        "ref": env.penalised_cost(TR["drop_penalty"]).cpu(),
                        "ref_drops": int(env.dropped.sum()), "evs": env.B * env.n})
    return val


@torch.no_grad()
def validate(policy, val, drop_penalty):
    policy.eval()
    ratios, by_n, drops, ref_drops, evs = [], {}, 0, 0, 0
    for v in val:
        env = Env(v["batch"])
        emb = policy.encode(*env.static_features())
        rollout(policy, env, emb, greedy=torch.ones(env.B, dtype=torch.bool))
        ratio = (env.penalised_cost(drop_penalty).cpu() / v["ref"]).numpy()
        ratios += ratio.tolist()
        by_n.setdefault(v["n"], []).extend(ratio.tolist())
        drops += int(env.dropped.sum())
        ref_drops += v["ref_drops"]
        evs += v["evs"]
    policy.train()
    return {"ratio": float(np.mean(ratios)),
            "by_n": {n: round(float(np.mean(r)), 4) for n, r in sorted(by_n.items())},
            "unplaced": 100.0 * drops / max(1, evs), "ref_unplaced": 100.0 * ref_drops / max(1, evs)}


def imitation_step(policy, optim, gen, TR, device, rng):
    """Warm-up batch: the policy is taught the actions of the greedy cost rule (teacher forcing,
    loss = mean negative log-likelihood per decision). REINFORCE then improves on that start."""
    n = int(rng.integers(TR["n_min"], TR["n_max"] + 1))
    Bi = int(TR["batch_instances"])
    batch = Batch(gen.batch(n, Bi), device)
    env = Env(batch)
    aug = torch.as_tensor(rng.integers(0, 8, Bi), device=device)
    emb = policy.encode(*env.static_features(aug))
    logp = rollout(policy, env, emb, teacher=GreedyRule(env))
    loss = -(logp / n).mean()
    optim.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), TR["max_grad_norm"])
    optim.step()
    return {"nll": loss.item(), "unplaced": 100.0 * float(env.dropped.double().mean())}


def train_step(policy, optim, gen, TR, device, rng):
    n = int(rng.integers(TR["n_min"], TR["n_max"] + 1))
    S, Bi = int(TR["samples"]), int(TR["batch_instances"])
    batch = Batch(gen.batch(n, Bi), device)
    env1 = Env(batch)
    aug = torch.as_tensor(rng.integers(0, 8, Bi), device=device)
    emb = policy.encode(*env1.static_features(aug))
    emb = Policy.select(emb, torch.arange(Bi, device=device).repeat_interleave(S))
    env = Env(batch.repeat(S))
    logp = rollout(policy, env, emb)
    cost = env.penalised_cost(TR["drop_penalty"]).view(Bi, S)
    base = cost.mean(1, keepdim=True)
    adv = ((cost - base) / base.abs().clamp(min=1e-9)).reshape(-1).float().to(device)
    loss = (adv * logp).mean()
    optim.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), TR["max_grad_norm"])
    optim.step()
    spread = float((cost.std(1) / base.squeeze(1).abs().clamp(min=1e-9)).mean())
    return {"spread": 100.0 * spread, "unplaced": 100.0 * float(env.dropped.double().mean())}


def save(path, policy, optim, state, RP, args):
    tmp = path + ".tmp"
    torch.save({"model": policy.state_dict(), "model_cfg": policy.cfg, "optim": optim.state_dict(),
                "state": dict(state), "rl1_params": RP, "args": vars(args),
                "saved": time.strftime("%Y-%m-%d %H:%M:%S")}, tmp)
    os.replace(tmp, path)


def archive_previous(model_dir, paths):
    """New training: move the previous run's files to <model_dir>/previous_runs/<date-time>/ (kept, not
    deleted), so the new run starts with an empty train_log.csv and its own model_best.pt."""
    old = [p for p in paths.values() if os.path.exists(p)]
    if not old:
        return None
    dest = os.path.join(model_dir, "previous_runs", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(dest, exist_ok=True)
    for p in old:
        shutil.move(p, os.path.join(dest, os.path.basename(p)))
    return dest


def append_log(path, row):
    new = not os.path.exists(path)
    try:
        with open(path, "a", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=LOG_COLUMNS, extrasaction="ignore")
            if new:
                wr.writeheader()
            wr.writerow(row)
    except OSError as e:
        print(f"WARNING: train_log.csv not written ({e})", flush=True)


def write_meta(path, meta):
    try:
        with open(path + ".tmp", "w") as fh:
            json.dump(meta, fh, indent=1, default=str)
        os.replace(path + ".tmp", path)
    except OSError as e:
        print(f"WARNING: train_meta.json not written ({e})", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=DEFAULT_CSV, help="dataset_ev_solomon/csv folder")
    ap.add_argument("--f3-params", default=DEFAULT_PARAMS_F3, help="code/f3_params.json")
    ap.add_argument("--rl1-params", default=DEFAULT_RL1_PARAMS)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--hours", type=float, help="training time cap")
    ap.add_argument("--imitation-batches", type=int, help="warm-up batches imitating the greedy rule (0 = none)")
    ap.add_argument("--max-epochs", type=int, help="stop after this many epochs in total")
    ap.add_argument("--min-epochs", type=int,
                    help="epochs that always run; the early-stop count starts after them")
    ap.add_argument("--patience", type=int,
                    help="stop after this many REINFORCE epochs without a better validation (0 = off)")
    ap.add_argument("--min-improvement", type=float,
                    help="a validation ratio counts as better only if it beats the best by more than this")
    ap.add_argument("--iters-per-epoch", type=int)
    ap.add_argument("--batch-instances", type=int)
    ap.add_argument("--samples", type=int, help="rollouts per instance (shared baseline)")
    ap.add_argument("--n-min", type=int)
    ap.add_argument("--n-max", type=int)
    ap.add_argument("--lr", type=float, help="learning rate (imitation, or REINFORCE without warm-up)")
    ap.add_argument("--lr-reinforce", type=float, help="learning rate of REINFORCE after the warm-up")
    ap.add_argument("--file-sizes", nargs="*", type=int, help="train on templates of these file sizes (100, 200)")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = all cores)")
    ap.add_argument("--device", default="auto", help="auto / cpu / cuda")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", action="store_true",
                    help="continue from model_last.pt instead of starting at epoch 1 (run_rl1.sh does not use it)")
    args = ap.parse_args()

    RP = load_rl1_params(args.rl1_params)
    TR = RP["training"]
    for k in ("hours", "imitation_batches", "max_epochs", "iters_per_epoch", "batch_instances", "samples",
              "n_min", "n_max", "lr", "lr_reinforce"):
        if getattr(args, k) is not None:
            TR[k] = getattr(args, k)
    if args.patience is not None:
        TR["patience_epochs"] = args.patience
    if args.min_improvement is not None:
        TR["min_improvement"] = args.min_improvement
    if args.min_epochs is not None:
        TR["min_epochs"] = args.min_epochs
    patience = int(TR.get("patience_epochs", 0))
    min_epochs = int(TR.get("min_epochs", 0))
    min_imp = float(TR.get("min_improvement", 0.0))
    if args.file_sizes:
        RP["generator"]["file_sizes"] = args.file_sizes
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.model_dir, exist_ok=True)
    paths = {k: os.path.join(args.model_dir, f) for k, f in
             (("last", "model_last.pt"), ("best", "model_best.pt"), ("log", "train_log.csv"),
              ("meta", "train_meta.json"), ("err", "train_errors.log"))}
    resuming = args.resume and os.path.exists(paths["last"])
    if not resuming:                                   # new training from epoch 1
        moved = archive_previous(args.model_dir, paths)
        if moved:
            print(f"new training from epoch 1; the previous run's files were moved to {moved}", flush=True)

    P3 = inc.load_params(args.f3_params)
    dataset = CsvDataset(args.csv, P3)
    if not dataset.files():
        print(f"no CSV dataset found in {os.path.abspath(args.csv)}", flush=True)
        sys.exit(1)
    gen = Generator(dataset, RP["generator"], P3, seed=args.seed, file_sizes=RP["generator"]["file_sizes"])
    policy = Policy(**RP["model"]).to(device)
    optim = torch.optim.Adam(policy.parameters(), lr=TR["lr"])
    # stale = REINFORCE epochs in a row without a better validation (early-stop counter)
    state = {"epoch": 0, "iters": 0, "train_s": 0.0, "best_val": float("inf"), "best_epoch": None, "stale": 0}
    if resuming:
        ck = torch.load(paths["last"], map_location=device, weights_only=False)
        policy = Policy(**ck["model_cfg"]).to(device)
        policy.load_state_dict(ck["model"])
        optim = torch.optim.Adam(policy.parameters(), lr=TR["lr"])
        optim.load_state_dict(ck["optim"])
        state.update(ck["state"])
        print(f"resumed {paths['last']}: epoch {state['epoch']}, {state['iters']} batches, "
              f"{state['train_s'] / 3600:.2f} h trained", flush=True)
    budget = 3600.0 * float(TR["hours"])
    meta = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv, "device": device,
            "threads": torch.get_num_threads(), "torch": torch.__version__, "rl1_params": RP, "f3_params": P3,
            "csv": os.path.abspath(args.csv), "templates": [list(k) for k in gen.keys],
            "parameters": sum(p.numel() for p in policy.parameters())}
    write_meta(paths["meta"], {**meta, **state})
    print(f"RL1 training on {device} ({torch.get_num_threads()} threads), {meta['parameters']:,} parameters, "
          f"templates {gen.keys}; stops at {TR['max_epochs']} epochs, after "
          f"{patience or 'no'} REINFORCE epochs without improvement (counted after epoch {min_epochs}), "
          f"or after {TR['hours']} h ({state['train_s'] / 3600:.2f} h used)", flush=True)
    t_val = time.time()
    val = make_validation(dataset, RP, P3, device)
    print(f"validation: {sum(v['batch'].B for v in val)} fixed instances, n = {TR['val_sizes']} "
          f"(greedy cost rule as reference, built in {time.time() - t_val:.0f} s)", flush=True)
    if state["epoch"] == 0 and state["iters"] == 0:
        v = validate(policy, val, TR["drop_penalty"])
        append_log(paths["log"], {"epoch": 0, "iters": 0, "val_ratio_rl_over_greedy": round(v["ratio"], 4),
                                  "val_unplaced_rl_%": round(v["unplaced"], 2),
                                  "val_unplaced_greedy_%": round(v["ref_unplaced"], 2),
                                  "val_ratio_by_n": json.dumps(v["by_n"]), "lr": TR["lr"], "train_min": 0,
                                  "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        print(f"untrained policy: RL / greedy rule = {v['ratio']:.4f}", flush=True)

    def early_stop():
        return patience > 0 and state["epoch"] >= min_epochs + patience and state["stale"] >= patience

    fails_in_row = 0
    interrupted = False
    try:
        while state["train_s"] < budget and state["epoch"] < TR["max_epochs"] and not early_stop():
            warm = int(TR["imitation_batches"]) > 0
            base_lr = TR["lr_reinforce"] if warm and state["iters"] >= int(TR["imitation_batches"]) else TR["lr"]
            lr = base_lr * TR["lr_decay_per_epoch"] ** state["epoch"]
            for grp in optim.param_groups:
                grp["lr"] = lr
            t0 = time.time()
            stats, failed = [], 0
            for _ in range(int(TR["iters_per_epoch"])):
                try:
                    if state["iters"] < int(TR["imitation_batches"]):
                        stats.append({**imitation_step(policy, optim, gen, TR, device, rng), "phase": "imitation"})
                    else:
                        stats.append({**train_step(policy, optim, gen, TR, device, rng), "phase": "reinforce"})
                    state["iters"] += 1
                    fails_in_row = 0
                except Exception as e:                       # a bad batch is skipped, never fatal
                    failed += 1
                    fails_in_row += 1
                    print(f"batch failed ({type(e).__name__}: {e}); skipped", flush=True)
                    with open(paths["err"], "a") as fh:
                        fh.write(time.strftime("%F %T ") + traceback.format_exc() + "\n")
                    if fails_in_row >= 20:
                        raise RuntimeError("20 batches in a row failed; see train_errors.log") from e
                if state["train_s"] + time.time() - t0 >= budget:
                    break
            v = validate(policy, val, TR["drop_penalty"])
            state["train_s"] += time.time() - t0
            state["epoch"] += 1
            phases = sorted({s["phase"] for s in stats})
            rl = [s for s in stats if s["phase"] == "reinforce"]
            im = [s for s in stats if s["phase"] == "imitation"]
            better = v["ratio"] < state["best_val"] - min_imp
            if better:
                state["best_val"], state["best_epoch"] = v["ratio"], state["epoch"]
                state["stale"] = 0
            elif rl and state["epoch"] > min_epochs:     # count only REINFORCE epochs after min_epochs
                state["stale"] += 1
            else:                                        # still inside the epochs that always run
                state["stale"] = 0
            save(paths["last"], policy, optim, state, RP, args)
            if better:
                save(paths["best"], policy, optim, state, RP, args)
            row = {"epoch": state["epoch"], "phase": "+".join(phases), "iters": state["iters"],
                   "batches_this_epoch": len(stats), "failed_batches": failed,
                   "imitation_nll": round(float(np.mean([s["nll"] for s in im])), 4) if im else "",
                   "train_cost_spread_%": round(float(np.mean([s["spread"] for s in rl])), 3) if rl else "",
                   "train_unplaced_%": round(float(np.mean([s["unplaced"] for s in rl])), 3) if rl else "",
                   "val_ratio_rl_over_greedy": round(v["ratio"], 4), "val_unplaced_rl_%": round(v["unplaced"], 2),
                   "val_unplaced_greedy_%": round(v["ref_unplaced"], 2), "val_ratio_by_n": json.dumps(v["by_n"]),
                   "epochs_without_improvement": state["stale"],
                   "lr": round(lr, 8), "train_min": round(state["train_s"] / 60, 1),
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            append_log(paths["log"], row)
            write_meta(paths["meta"], {**meta, **state})
            what = (f"imitation loss {row['imitation_nll']}" if im and not rl else
                    f"unplaced in training {row['train_unplaced_%']} %")
            if not patience:
                waiting = ""
            elif state["epoch"] <= min_epochs:
                waiting = f", early stop counted after epoch {min_epochs}"
            else:
                waiting = f", no improvement for {state['stale']}/{patience} epochs"
            print(f"epoch {state['epoch']} [{row['phase']}]: {len(stats)} batches ({failed} failed), {what}, "
                  f"validation RL / greedy rule {v['ratio']:.4f} {v['by_n']} "
                  f"(best {state['best_val']:.4f} at epoch {state['best_epoch']}{waiting}), "
                  f"unplaced RL {v['unplaced']:.2f} % vs greedy {v['ref_unplaced']:.2f} %, "
                  f"epoch {state['epoch']} of {TR['max_epochs']}, {state['train_s'] / 60:.0f} min", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted: saving model_last.pt", flush=True)
        save(paths["last"], policy, optim, state, RP, args)
    if not os.path.exists(paths["best"]):
        save(paths["best"], policy, optim, state, RP, args)
    if interrupted:
        reason = "interrupted (Ctrl-C)"
    elif early_stop():
        reason = (f"early stop: no better validation in the last {patience} REINFORCE epochs after epoch "
                  f"{min_epochs} (best at epoch {state['best_epoch']})")
    elif state["epoch"] >= TR["max_epochs"]:
        reason = f"reached max_epochs = {TR['max_epochs']}"
    elif state["train_s"] >= budget:
        reason = f"time budget of {TR['hours']} h used"
    else:
        reason = "stopped (too many failed batches, see train_errors.log)"
    write_meta(paths["meta"], {**meta, **state, "stop_reason": reason,
                               "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    print(f"training finished ({reason}): {state['epoch']} epochs, {state['iters']} batches, "
          f"{state['train_s'] / 3600:.2f} h; best validation RL / greedy rule {state['best_val']:.4f} "
          f"(epoch {state['best_epoch']}); models in {os.path.abspath(args.model_dir)}", flush=True)


if __name__ == "__main__":
    main()
