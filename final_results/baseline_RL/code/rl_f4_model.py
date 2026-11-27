"""RL policy for formulation 4 (the network of RL1): attention encoder + two-step pointer decoder
(choose an EV, then its resource). The EV embedding also reads the segment, mu and c^Q tauF (rl_f4_env).

Encoder (Kool et al. 2019, "Attention, learn to solve routing problems!")
  every node (EVs, stations, depot) gets a linear embedding of its static features; a stack of
  Transformer layers (multi-head self-attention, no positional encoding, so the order of the nodes
  does not matter and one network handles any n, any number of stations and vans).

Decoder, at every step of the construction (rl_f4_env.Env)
  1. EV choice: context = mean embedding of the EVs not yet placed + depot embedding + global state;
     a multi-head glimpse over the open EVs gives a query; score of EV i =
         C tanh( q . W h_i / sqrt(d) + MLP(phi_i) )
     phi_i = dynamic features of EV i (options left, cheapest option, regret, earliest finish, ...).
     Infeasible EVs are masked.
  2. Resource choice for that EV (vehicle selection of Li et al. 2021 for heterogeneous fleets, here
     the "fleet" = G guns that stay put + M vans that move): key of resource k = W (station
     embedding) for a gun or W (embedding of the van's current node) for a van, + a linear map of the
     resource's state (free time, battery, EVs served, ...); score of k =
         C tanh( q_i . key_k / sqrt(d) + MLP(pair features of (i, k)) )
     Infeasible resources are masked.
  log pi(action) = log pi(i) + log pi(k | i).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from rl_f4_env import N_DEP_FEAT, N_EV_FEAT, N_GLOBAL, N_PAIR, N_PHI, N_RES, N_ST_FEAT


class MLP(nn.Module):
    def __init__(self, n_in, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Policy(nn.Module):
    def __init__(self, d=128, heads=8, layers=3, ff=512, clip=10.0, dyn_hidden=32):
        super().__init__()
        self.cfg = {"d": d, "heads": heads, "layers": layers, "ff": ff, "clip": clip, "dyn_hidden": dyn_hidden}
        self.d, self.h, self.clip = d, heads, clip
        self.ev_in = nn.Linear(N_EV_FEAT, d)
        self.st_in = nn.Linear(N_ST_FEAT, d)
        self.dep_in = nn.Linear(N_DEP_FEAT, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.ctx = nn.Linear(2 * d + N_GLOBAL, d)
        self.gq = nn.Linear(d, d, bias=False)
        self.gk = nn.Linear(d, d, bias=False)
        self.gv = nn.Linear(d, d, bias=False)
        self.go = nn.Linear(d, d, bias=False)
        self.ev_key = nn.Linear(d, d, bias=False)
        self.ev_dyn = MLP(N_PHI, dyn_hidden)
        self.res_q = nn.Linear(2 * d, d)
        self.res_key = nn.Linear(d, d, bias=False)
        self.gun_dyn = nn.Linear(N_RES, d)
        self.van_dyn = nn.Linear(N_RES, d)
        self.pair_dyn = MLP(N_PAIR, dyn_hidden)

    # ------------------------------------------------------------------ encoder
    def encode(self, ev, st, dep):
        x = torch.cat([self.ev_in(ev), self.st_in(st), self.dep_in(dep)], 1)
        x = self.norm(self.encoder(x))
        n, F = ev.shape[1], st.shape[1]
        return {"ev": x[:, :n], "st": x[:, n:n + F], "dep": x[:, n + F]}

    @staticmethod
    def select(emb, idx):
        """Rows idx of every embedding (e.g. repeat each instance for its rollouts)."""
        return {k: v[idx] for k, v in emb.items()}

    def precompute(self, emb):
        """Projections that do not change during an episode (computed once per rollout batch)."""
        B, n, d = emb["ev"].shape
        dk = d // self.h
        nodes = torch.cat([emb["dep"][:, None], emb["ev"]], 1)
        return {"K": self.gk(emb["ev"]).view(B, n, self.h, dk).transpose(1, 2),
                "V": self.gv(emb["ev"]).view(B, n, self.h, dk).transpose(1, 2),
                "evk": self.ev_key(emb["ev"]),
                "st_key": self.res_key(emb["st"]),          # key part of a gun = its station
                "node_key": self.res_key(nodes)}             # key part of a van = its current node

    # ------------------------------------------------------------------ decoder
    def ev_logits(self, emb, pre, env, phi, mask):
        B, n, d = emb["ev"].shape
        dk = d // self.h
        live = (~env.done).float()
        mean = (live[:, None, :] @ emb["ev"]).squeeze(1) / live.sum(1, keepdim=True).clamp(min=1.0)
        q = self.ctx(torch.cat([mean, emb["dep"], env.global_features()], -1))
        att = (self.gq(q).view(B, self.h, 1, dk) @ pre["K"].transpose(-1, -2)) / math.sqrt(dk)
        att = att.masked_fill(~mask[:, None, None, :], float("-inf"))
        g = self.go((torch.softmax(att, -1) @ pre["V"]).reshape(B, d))
        u = (pre["evk"] @ g[..., None]).squeeze(-1) / math.sqrt(d) + self.ev_dyn(phi)
        return (self.clip * torch.tanh(u)).masked_fill(~mask, float("-inf")), g

    def res_logits(self, emb, pre, env, i, g, pair, mask):
        d = self.d
        h_i = emb["ev"][env.ar, i]
        q = self.res_q(torch.cat([h_i, g], -1))
        rf = env.resource_features().to(q.device)
        G = env.G
        # key of resource k = W_key (base embedding) + linear map of its state (computed in key space)
        gun_key = pre["st_key"][:, env.b.gun_station]
        # clone: the index is kept for backprop and env.step changes van_pos in place
        van_key = torch.gather(pre["node_key"], 1, env.van_pos.clone()[..., None].expand(-1, -1, d))
        key = torch.cat([gun_key + self.gun_dyn(rf[:, :G]), van_key + self.van_dyn(rf[:, G:])], 1)
        u = (key @ q[..., None]).squeeze(-1) / math.sqrt(d) + self.pair_dyn(pair)
        return (self.clip * torch.tanh(u)).masked_fill(~mask, float("-inf"))


def _choose(logp, greedy):
    if bool(greedy.all()):
        return logp.argmax(1)
    sampled = torch.multinomial(logp.exp(), 1).squeeze(1)
    return torch.where(greedy, logp.argmax(1), sampled)


def rollout(policy, env, emb, greedy=None, teacher=None):
    """Build one plan per row of env. greedy (bool, per row): take the most likely action instead of
    sampling. teacher (callable (env, opt) -> (i, k)): take the teacher's actions instead and return
    their log-probability under the policy (imitation warm-up). Returns the summed log-probability of
    the chosen actions (B,); the plan and its cost are in env (env.cost, env.dropped, env.res / S / C)."""
    pre = policy.precompute(emb)
    B = env.B
    dev = emb["ev"].device
    logp = torch.zeros(B, device=dev)
    greedy = torch.zeros(B, dtype=torch.bool, device=dev) if greedy is None else greedy.to(dev)
    while True:
        opt = env.options()
        env.drop_dead(opt)
        active = ~env.finished()
        if not active.any():
            break
        ev_mask = opt["feas"].any(2)
        ev_mask[~active, 0] = True                        # finished rows: dummy action, not counted
        forced = teacher(env, opt) if teacher is not None else None
        u, g = policy.ev_logits(emb, pre, env, env.ev_features(opt).to(dev), ev_mask.to(dev))
        lp_ev = Fn.log_softmax(u, -1)
        i = forced[0].to(dev) if forced is not None else _choose(lp_ev, greedy)
        i = torch.where(active.to(dev), i, torch.zeros_like(i))
        pair, k_mask = env.pair_features(opt, i.to(env.ar.device))
        k_mask = k_mask.clone()
        k_mask[~active, 0] = True
        v = policy.res_logits(emb, pre, env, i, g, pair.to(dev), k_mask.to(dev))
        lp_res = Fn.log_softmax(v, -1)
        k = forced[1].to(dev) if forced is not None else _choose(lp_res, greedy)
        k = torch.where(active.to(dev), k, torch.zeros_like(k))
        step_lp = lp_ev.gather(1, i[:, None]).squeeze(1) + lp_res.gather(1, k[:, None]).squeeze(1)
        logp = logp + torch.where(active.to(dev), step_lp, torch.zeros_like(step_lp))
        env.step(i, k, active, opt)
    return logp


def load_policy(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    policy = Policy(**ck["model_cfg"]).to(device)
    policy.load_state_dict(ck["model"])
    policy.eval()
    return policy, ck
