"""Efficient Active Search (EAS-Emb) for one test instance (Hottung, Kwon, Tierney, ICLR 2022).

The trained policy stays frozen. Only the embeddings of this one instance (the encoder output for
each map symmetry) become trainable parameters, and they are tuned on this instance alone:

  repeat for up to `iterations` iterations or `seconds` seconds:
      sample `samples` plans per symmetry with the current embeddings
      cost  = Z + drop_penalty x scale x (unplaced EVs)                        (as in training)
      loss  = REINFORCE with the shared baseline (mean cost of the plans of the same symmetry)
            + imitation_weight x NLL of the best plan found so far (teacher forcing: the incumbent
              is replayed step by step and its log-probability is raised)
      one Adam step on the embeddings
      keep the best plan seen so far (fewest unplaced EVs, then lowest Z)

The best plan is only ever replaced by a better one, so the result is never worse than the start
(the best plan of the sampling step). The imitation term keeps the search close to the incumbent
while REINFORCE explores around it.
"""
import time

import torch

from rl2_env import Batch, Env, plan_of
from rl2_model import Policy, rollout


def plan_actions(env, row):
    """Action sequence (EV indices, resource indices) that built row `row` of env, in step order.
    Replaying it in a fresh simulator gives exactly the same plan (the simulator is deterministic)."""
    order, res = env.order[row], env.res[row]
    placed = torch.nonzero(order >= 0).squeeze(1)
    seq = placed[torch.argsort(order[placed])]
    return seq.tolist(), res[seq].tolist()


class ReplayTeacher:
    """Teacher for rl2_model.rollout that replays a fixed action sequence in every row."""

    def __init__(self, i_seq, k_seq, device):
        self.i = torch.tensor(list(i_seq) + [0], dtype=torch.long, device=device)
        self.k = torch.tensor(list(k_seq) + [0], dtype=torch.long, device=device)

    def __call__(self, env, opt):
        t = min(env.steps, len(self.i) - 1)
        return self.i[t].expand(env.B).clone(), self.k[t].expand(env.B).clone()


def active_search(policy, J, batch, emb0, start, cfg, drop_penalty, device, seed):
    """Tune the instance embeddings emb0 (one row per map symmetry) and return
    (best, iterations, seconds). `start` / `best` = dict(key, actions, plan, dropped, cost, iteration)."""
    torch.manual_seed(seed)
    A = emb0["ev"].shape[0]
    S = max(1, int(cfg["samples"]))
    n = int(J["n_ev"])
    params = {k: torch.nn.Parameter(v.detach().clone()) for k, v in emb0.items()}
    optim = torch.optim.Adam(list(params.values()), lr=float(cfg["lr"]))
    idx = torch.arange(A, device=device).repeat_interleave(S)
    rows = torch.arange(A, device=device)
    lam = float(cfg["imitation_weight"])
    best = dict(start, iteration=0)
    frozen = [p.requires_grad for p in policy.parameters()]
    for p in policy.parameters():
        p.requires_grad_(False)
    t0, it = time.time(), 0
    try:
        while it < int(cfg["iterations"]) and time.time() - t0 < float(cfg["seconds"]):
            env = Env(batch.repeat(A * S))
            logp = rollout(policy, env, Policy.select(params, idx))
            cost = env.penalised_cost(drop_penalty).view(A, S)
            base = cost.mean(1, keepdim=True)
            adv = ((cost - base) / base.abs().clamp(min=1e-9)).reshape(-1).float().to(logp.device)
            loss = (adv * logp).mean()
            if lam > 0 and best["actions"][0]:
                env_t = Env(batch.repeat(A))
                lp_t = rollout(policy, env_t, Policy.select(params, rows),
                               teacher=ReplayTeacher(*best["actions"], device))
                loss = loss + lam * (-(lp_t / n).mean())
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            it += 1
            with torch.no_grad():
                key = env.dropped.sum(1).double() * 1e12 + env.cost
                row = int(torch.argmin(key))
                if float(key[row]) < best["key"] - 1e-9:
                    plan, dropped = plan_of(env, row, J)
                    best = {"key": float(key[row]), "actions": plan_actions(env, row), "plan": plan,
                            "dropped": dropped, "cost": float(env.cost[row]), "iteration": it}
    finally:
        for p, g in zip(policy.parameters(), frozen):
            p.requires_grad_(g)
    return best, it, time.time() - t0
