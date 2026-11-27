"""RL simulator of formulation 4 (RL1's simulator of formulation 3 with the use segment in the inputs):
a batched (PyTorch) construction of a plan, one decision at a time. Formulation 4 has the constraints of
formulation 3; the segment only changes the per-EV coefficients (mu, c^D, c^Q, c^C, c^V, W), which the
simulator reads per EV, so the masks and the reward below are exact for formulation 4.


Markov decision process
  state   which EVs are placed; per gun the minute it becomes free; per van its position (depot or
          last EV), the minute it is free, battery left and EVs served so far
  action  a pair (EV i, resource k): k is one of the G guns or one of the M vans. EV i is appended
          to the end of k's sequence (gun queue or van route)
  timing  not learned: every session starts as early as possible,
              gun  S_i = max(r_i + t_if, free_g)            (EV drives off at its request time)
              van  S_i = max(r_i, free_m + t_(pos, i))
          With assignments and sequences fixed this is the optimal timing, because every time cost
          of formulation 4 grows with a later start (queue, van wait) and nothing rewards waiting.
  masks   (only feasible pairs can be chosen)
              gun: station within range R_i, queue S_i - r_i - t_if <= W_i, S_i + tauF_i <= T
              van: fewer than Q EVs so far, battery left >= E_i/eta + beta (d_(pos,i) + d_(i,0)),
                   S_i + tauM_i + t_(i,0) <= T (back at the depot by T)
          symmetry (lossless): of two guns of one station that are free at the same minute only the
          first is offered; of the unused vans (all identical, at the depot) only the first.
  reward  minus the increase of Z = Z1 + ... + Z6 caused by the action (check_plan_f4's terms):
              gun: mu (c_EV d_if + p_F E) + pi_F E + c^D t_if + c^Q queue + c^C tauF
              van: mu p_M E + pi_M E / eta + c^V (wait + tauM) + c_M (d_(pos,i) + d_(i,0) - d_(pos,0))
                   + c_F if the van was unused
          The van km term already contains the way back to the depot, so the increases add up to Z.
  end     when every EV is placed or can no longer be placed. An EV with no feasible pair stays so
          for the rest of the episode (gun queues only grow, van batteries only shrink, triangle
          inequality), so it is dropped at once and counted; the training cost adds a penalty per
          dropped EV and the plan is reported as invalid.
All times / costs are float64 so that feasibility agrees with check_plan_f4 (tolerance 1e-4).
New network inputs of formulation 4 (static EV features 17-19): commercial (0/1), money sensitivity mu,
and the cost of queueing one charging duration c^Q tauF (relative to the instance's cost scale).
"""
import numpy as np
import torch

TOL = 1e-9
DT = torch.float64
INF = float("inf")

N_EV_FEAT, N_ST_FEAT, N_DEP_FEAT = 19, 4, 4      # static node features (RL1: 16 EV features)
N_GLOBAL, N_PHI, N_RES, N_PAIR = 4, 7, 6, 6      # dynamic features

EV_KEYS = ("E_kWh", "r_min", "R_km", "tauF_min", "tauM_min", "theta", "segment", "mu", "cD", "cQ", "cV",
           "cC", "W")


class Batch:
    """Static data of B instances with the same n, stations, guns and vans (tensors, batch first)."""

    def __init__(self, Js, device="cpu"):
        J0 = Js[0]
        n, F, M = int(J0["n_ev"]), int(J0["n_fcs"]), int(J0["n_vans"])
        guns = [int(g) for g in J0["params"]["guns"]]
        for J in Js:
            if (int(J["n_ev"]), int(J["n_fcs"]), int(J["n_vans"]), [int(g) for g in J["params"]["guns"]]) \
                    != (n, F, M, guns):
                raise ValueError("instances of one batch need the same n, stations, guns and vans")
        self.n, self.F, self.M, self.guns, self.device = n, F, M, guns, device
        self.gun_station = torch.tensor([f for f in range(F) for _ in range(guns[f])], device=device)
        self.gun_local = torch.tensor([g for f in range(F) for g in range(guns[f])], device=device)
        self.G = len(self.gun_station)
        self.K = self.G + M

        def stack(get):
            return torch.as_tensor(np.stack([np.asarray(get(J), dtype=float) for J in Js]), dtype=DT,
                                   device=device)

        t = {k: stack(lambda J, k=k: np.asarray(J["ev"][k], dtype=float)[:n]) for k in EV_KEYS}
        t["dEF"] = stack(lambda J: J["d_ev_fcs_km"])
        t["tEF"] = stack(lambda J: J["t_ev_fcs_min"])
        t["dN"] = stack(lambda J: J["d_nodes_km"])
        t["tN"] = stack(lambda J: J["t_nodes_min"])
        par = {"T": lambda J: J["params"]["T_min"], "eta": lambda J: J["params"]["eta"],
               "beta": lambda J: J["params"]["beta_kWh_per_km"], "Bk": lambda J: J["params"]["B_kWh"],
               "Q": lambda J: J["params"]["Q"]}
        for k in ("c_EV", "c_M", "c_F", "pi_F", "pi_M"):
            par[k] = lambda J, k=k: J["params"]["costs_Rs"][k]
        for k in ("p_F", "p_M"):
            par[k] = lambda J, k=k: J["params"]["tariff_Rs_per_kWh"][k]
        for k, get in par.items():
            t[k] = stack(get)
        t["xy_dep"] = stack(lambda J: J["xy"]["depot"])
        t["xy_fcs"] = stack(lambda J: J["xy"]["fcs"])
        t["xy_ev"] = stack(lambda J: J["xy"]["ev"])
        # derived static quantities
        gs = self.gun_station
        dEG, tEG = t["dEF"][:, :, gs], t["tEF"][:, :, gs]                     # (B, n, G)
        E, mu = t["E_kWh"], t["mu"]
        t["reach_g"] = dEG <= t["R_km"][..., None]
        t["arrive_g"] = t["r_min"][..., None] + tEG
        t["gun_cost0"] = (mu[..., None] * (t["c_EV"][:, None, None] * dEG + (t["p_F"][:, None] * E)[..., None])
                          + (t["pi_F"][:, None] * E)[..., None] + t["cD"][..., None] * tEG
                          + (t["cC"] * t["tauF_min"])[..., None])
        t["van_cost0"] = (mu * t["p_M"][:, None] * E + t["pi_M"][:, None] * E / t["eta"][:, None]
                          + t["cV"] * t["tauM_min"])
        t["d_back"], t["t_back"] = t["dN"][:, 1:, 0], t["tN"][:, 1:, 0]
        self.t = t

    @property
    def B(self):
        return self.t["T"].shape[0]

    def repeat(self, k):
        """Each instance repeated k times in a row (rollouts of one instance side by side)."""
        out = object.__new__(Batch)
        out.__dict__.update({a: v for a, v in self.__dict__.items() if a != "t"})
        out.t = {a: v.repeat_interleave(k, 0) for a, v in self.t.items()}
        return out


def dihedral(xy, aug):
    """One of the 8 symmetries of the square (aug in 0..7) applied to coordinates relative to the
    depot. Distances do not change, so the instance is the same; only the network input differs."""
    a = aug.view(-1, *([1] * (xy.dim() - 2)))
    x, y = xy[..., 0], xy[..., 1]
    swap = (a % 2) == 1
    xs, ys = torch.where(swap, y, x), torch.where(swap, x, y)
    xs = torch.where(((a // 2) % 2) == 1, -xs, xs)
    ys = torch.where((a // 4) == 1, -ys, ys)
    return torch.stack([xs, ys], -1)


class Env:
    def __init__(self, batch):
        self.b = b = batch
        t = b.t
        B, n, G, M = b.B, b.n, b.G, b.M
        dev = b.device
        self.B, self.n, self.G, self.M, self.K = B, n, G, M, b.K
        self.ar = torch.arange(B, device=dev)
        self.done = torch.zeros(B, n, dtype=torch.bool, device=dev)
        self.dropped = torch.zeros(B, n, dtype=torch.bool, device=dev)
        self.gun_free = torch.zeros(B, G, dtype=DT, device=dev)
        self.van_pos = torch.zeros(B, M, dtype=torch.long, device=dev)
        self.van_free = torch.zeros(B, M, dtype=DT, device=dev)
        self.van_batt = t["Bk"][:, None].repeat(1, M)
        self.van_cnt = torch.zeros(B, M, dtype=DT, device=dev)
        self.cost = torch.zeros(B, dtype=DT, device=dev)
        self.res = torch.full((B, n), -1, dtype=torch.long, device=dev)
        self.S = torch.zeros(B, n, dtype=DT, device=dev)
        self.C = torch.zeros(B, n, dtype=DT, device=dev)
        self.order = torch.full((B, n), -1, dtype=torch.long, device=dev)
        self.steps = 0
        idx = torch.arange(G, device=dev)
        gs = b.gun_station
        self.same_earlier = (gs[:, None] == gs[None, :]) & (idx[None, :] < idx[:, None])   # [g, g'<g]
        # cost scale of the instance: mean over EVs of the cheapest first option (normalises features
        # and the training cost; falls back to the van cost if nothing can be placed)
        opt = self.options()
        best = opt["dz"].masked_fill(~opt["feas"], INF).min(2).values
        ok = torch.isfinite(best)
        mean_best = torch.where(ok, best, torch.zeros_like(best)).sum(1) / ok.sum(1).clamp(min=1)
        fallback = (t["van_cost0"] + (t["c_F"] / t["Q"])[:, None]).mean(1)
        self.cost_scale = torch.where(ok.any(1), mean_best, fallback).clamp(min=1.0)

    # ------------------------------------------------------------------ feasible pairs
    def options(self):
        """Every (EV, resource) pair: feasible?, start, finish, wait and increase of Z. Shapes (B, n, K)."""
        b, t = self.b, self.b.t
        n = self.n
        T = t["T"][:, None, None]
        # guns
        start_g = torch.maximum(t["arrive_g"], self.gun_free[:, None, :])
        wait_g = start_g - t["arrive_g"]
        fin_g = start_g + t["tauF_min"][..., None]
        dup = ((self.gun_free[:, :, None] - self.gun_free[:, None, :]).abs() <= 1e-12) & self.same_earlier[None]
        feas_g = (t["reach_g"] & (wait_g <= t["W"][..., None] + TOL) & (fin_g <= T + TOL)
                  & ~dup.any(2)[:, None, :])
        dz_g = t["gun_cost0"] + t["cQ"][..., None] * wait_g
        # vans
        pos = self.van_pos
        idx = pos[:, :, None].expand(-1, -1, n)
        d_go = torch.gather(t["dN"][:, :, 1:], 1, idx).transpose(1, 2)            # (B, n, M)
        t_go = torch.gather(t["tN"][:, :, 1:], 1, idx).transpose(1, 2)
        d_home = torch.gather(t["dN"][:, :, 0], 1, pos)                             # (B, M)
        start_v = torch.maximum(t["r_min"][..., None], self.van_free[:, None, :] + t_go)
        wait_v = start_v - t["r_min"][..., None]
        fin_v = start_v + t["tauM_min"][..., None]
        need = (t["E_kWh"] / t["eta"][:, None])[..., None] + t["beta"][:, None, None] * (d_go + t["d_back"][..., None])
        unused = self.van_cnt == 0
        first_unused = unused & (torch.cumsum(unused.long(), 1) == 1)
        feas_v = (((self.van_cnt < t["Q"][:, None]) & (~unused | first_unused))[:, None, :]
                  & (need <= self.van_batt[:, None, :] + TOL)
                  & (fin_v + t["t_back"][..., None] <= T + TOL))
        dz_v = (t["van_cost0"][..., None]
                + t["c_M"][:, None, None] * (d_go + t["d_back"][..., None] - d_home[:, None, :])
                + t["c_F"][:, None, None] * unused[:, None, :].to(DT)
                + t["cV"][..., None] * wait_v)
        feas = torch.cat([feas_g, feas_v], 2) & ~self.done[..., None]
        # full (B, n, K) tensors only where every pair is needed; the rest is read per chosen EV
        return {"feas": feas, "finish": torch.cat([fin_g, fin_v], 2), "dz": torch.cat([dz_g, dz_v], 2),
                "start_g": start_g, "start_v": start_v, "wait_g": wait_g, "wait_v": wait_v, "need": need}

    def at(self, opt, name, i):
        """Values of the gun part and the van part of `name` for EV i[b] of every row: (B, K)."""
        return torch.cat([opt[name + "_g"][self.ar, i], opt[name + "_v"][self.ar, i]], 1)

    def drop_dead(self, opt):
        """EVs without any feasible pair are dropped (they never get one again)."""
        dead = ~self.done & ~opt["feas"].any(2)
        if dead.any():
            self.done |= dead
            self.dropped |= dead
        return dead

    def step(self, i, k, active, opt):
        """Append EV i[b] to resource k[b] in every row b with active[b]."""
        b, t = self.b, self.b.t
        G = self.G
        bi, ii, kk = self.ar[active], i[active], k[active]
        if len(bi) == 0:
            return
        st = torch.where(kk < G, opt["start_g"][bi, ii, kk.clamp(max=G - 1)],
                         opt["start_v"][bi, ii, (kk - G).clamp(min=0)])
        fn = opt["finish"][bi, ii, kk]
        self.cost[bi] = self.cost[bi] + opt["dz"][bi, ii, kk]
        self.res[bi, ii] = kk
        self.S[bi, ii], self.C[bi, ii] = st, fn
        self.order[bi, ii] = self.steps
        self.done[bi, ii] = True
        g = kk < G
        if g.any():
            self.gun_free[bi[g], kk[g]] = fn[g]
        v = ~g
        if v.any():
            bv, iv, mv = bi[v], ii[v], kk[v] - G
            d_go = t["dN"][bv, self.van_pos[bv, mv], iv + 1]
            self.van_batt[bv, mv] = self.van_batt[bv, mv] - (t["E_kWh"][bv, iv] / t["eta"][bv] + t["beta"][bv] * d_go)
            self.van_pos[bv, mv] = iv + 1
            self.van_free[bv, mv] = fn[v]
            self.van_cnt[bv, mv] = self.van_cnt[bv, mv] + 1
        self.steps += 1

    def finished(self):
        return self.done.all(1)

    def penalised_cost(self, drop_penalty):
        """Training / selection cost: Z + drop_penalty * cost_scale per dropped EV."""
        return self.cost + drop_penalty * self.cost_scale * self.dropped.sum(1).to(DT)

    # ------------------------------------------------------------------ network inputs
    def static_features(self, aug=None):
        """Node features (float32): EVs (B, n, N_EV_FEAT), stations (B, F, N_ST_FEAT), depot (B, 1, N_DEP_FEAT)."""
        b, t = self.b, self.b.t
        T = t["T"][:, None]
        rel = t["xy_ev"] - t["xy_dep"][:, None]
        relf = t["xy_fcs"] - t["xy_dep"][:, None]
        scale = torch.maximum(rel.abs().amax((1, 2)), relf.abs().amax((1, 2))).clamp(min=1.0)[:, None, None]
        rel, relf = rel / scale, relf / scale
        if aug is not None:
            rel, relf = dihedral(rel, aug), dihedral(relf, aug)
        reach_f = t["dEF"] <= t["R_km"][..., None]
        nreach = reach_f.sum(2).to(DT)
        near_t = t["tEF"].masked_fill(~reach_f, INF).min(2).values
        near_t = torch.where(torch.isfinite(near_t), near_t, torch.zeros_like(near_t))
        gmin = t["gun_cost0"].masked_fill(~t["reach_g"], INF).min(2).values
        gmin = torch.where(torch.isfinite(gmin), gmin, torch.zeros_like(gmin))
        sc = self.cost_scale[:, None]
        city = t["dEF"].amax((1, 2)).clamp(min=1e-6)[:, None]
        ev = torch.stack([
            rel[..., 0], rel[..., 1], t["r_min"] / T, t["tauF_min"] / T, t["tauM_min"] / T, t["W"] / T,
            t["theta"], t["E_kWh"] / t["Bk"][:, None], (nreach == 0).to(DT), nreach / b.F,
            (near_t / T).clamp(0, 2), (t["t_back"] / T).clamp(0, 2), (t["R_km"] / city).clamp(0, 3),
            (gmin / sc).clamp(0, 10), (t["van_cost0"] / sc).clamp(0, 10),
            ((T - t["r_min"] - t["tauM_min"] - t["t_back"]) / T).clamp(-1, 1),
            t["segment"], t["mu"] / 2.0, (t["cQ"] * t["tauF_min"] / sc).clamp(0, 10)], -1)
        guns = torch.tensor(b.guns, dtype=DT, device=b.device)[None].expand(self.B, -1)
        st = torch.stack([relf[..., 0], relf[..., 1], guns / 2.0, reach_f.to(DT).mean(1)], -1)
        dep = torch.stack([t["Q"] * self.M / self.n, torch.full_like(t["T"], self.G / self.n),
                           torch.log(t["T"].clamp(min=1.0)) / 8.0, torch.full_like(t["T"], self.n / 100.0)], -1)
        return ev.float(), st.float(), dep[:, None, :].float()

    def global_features(self):
        t = self.b.t
        T = t["T"][:, None]
        return torch.stack([self.done.to(DT).mean(1), (self.gun_free / T).mean(1),
                            (self.van_cnt > 0).to(DT).mean(1), (self.van_batt / t["Bk"][:, None]).mean(1)],
                           -1).float()

    def ev_features(self, opt):
        """Dynamic per-EV features (B, n, N_PHI): options left, cheapest option, regret (second
        cheapest - cheapest), earliest finish, cheapest gun vs cheapest van, request relative to now,
        place among the open requests."""
        t = self.b.t
        feas, G = opt["feas"], self.G
        T = t["T"][:, None]
        sc = self.cost_scale[:, None]
        dz = opt["dz"].masked_fill(~feas, INF)
        best, arg = dz.min(2)
        second = dz.scatter(2, arg[..., None], INF).min(2).values
        has = torch.isfinite(best)
        zero = torch.zeros_like(best)
        f_best = torch.where(has, (best / sc).clamp(0, 10), zero)
        f_regret = torch.where(has, ((second - best) / sc).nan_to_num(5.0, 5.0, 5.0).clamp(0, 5), zero)
        fin = opt["finish"].masked_fill(~feas, INF).min(2).values
        f_fin = torch.where(has, (fin / T).clamp(0, 2), zero)
        diff = (dz[..., :G].min(2).values - dz[..., G:].min(2).values) / sc
        f_diff = torch.where(has, diff.nan_to_num(0.0, 5.0, -5.0).clamp(-5, 5), zero)
        r = t["r_min"]
        open_r = r.masked_fill(self.done, INF)
        now = open_r.min(1).values
        now = torch.where(torch.isfinite(now), now, torch.zeros_like(now))[:, None]
        f_rel = ((r - now) / T).clamp(-1, 1)
        # position in the queue of open requests (earliest request first, ties: higher queue cost c^Q
        # first), 0 = next in line, 1 = last
        key = open_r - 1e-6 * t["cQ"]
        rank = torch.argsort(torch.argsort(key, dim=1), dim=1).to(DT)
        n_open = (~self.done).sum(1, keepdim=True).to(DT)
        f_rank = torch.where(self.done, torch.ones_like(rank), rank / (n_open - 1).clamp(min=1.0))
        return torch.stack([feas.sum(2).to(DT) / self.K, f_best, f_regret, f_fin, f_diff, f_rel, f_rank],
                           -1).float()

    def resource_features(self):
        """Dynamic per-resource features (B, K, N_RES)."""
        t = self.b.t
        T = t["T"][:, None]
        zg = torch.zeros_like(self.gun_free)
        gun = torch.stack([self.gun_free / T, zg, zg, zg, (self.gun_free > 0).to(DT), zg], -1)
        home = torch.gather(t["tN"][:, :, 0], 1, self.van_pos)
        van = torch.stack([self.van_free / T, torch.ones_like(self.van_free), self.van_batt / t["Bk"][:, None],
                           self.van_cnt / t["Q"][:, None], (self.van_cnt > 0).to(DT), home / T], -1)
        return torch.cat([gun, van], 1).float()

    def pair_features(self, opt, i):
        """Features of every resource for the chosen EV i (B, K, N_PAIR) and the feasibility mask (B, K)."""
        t = self.b.t
        ar, G = self.ar, self.G
        T = t["T"][:, None]
        sc = self.cost_scale[:, None]
        feas = opt["feas"][ar, i]
        dz = opt["dz"][ar, i]
        best = dz.masked_fill(~feas, INF).min(1).values[:, None]
        best = torch.where(torch.isfinite(best), best, torch.zeros_like(best))
        is_van = torch.zeros_like(dz)
        is_van[:, G:] = 1.0
        batt_after = torch.cat([torch.zeros(self.B, G, dtype=DT, device=dz.device),
                                self.van_batt - opt["need"][ar, i]], 1)
        f = torch.stack([(dz / sc).clamp(0, 10), ((dz - best) / sc).clamp(0, 5),
                         (self.at(opt, "wait", i) / T).clamp(0, 2), (opt["finish"][ar, i] / T).clamp(0, 2),
                         (batt_after / t["Bk"][:, None]).clamp(-1, 1), is_van], -1)
        return torch.where(feas[..., None], f, torch.zeros_like(f)).float(), feas


# ====================================================================== baseline policy
class GreedyRule:
    """The 'cost' rule of solve_milp_f4.greedy_f4 as a policy of the simulator: EVs in the order
    stranded first, then request time, then higher queue cost c^Q; each takes its cheapest feasible resource.
    Reference in the validation, and teacher of the imitation warm-up."""

    def __init__(self, env):
        t = env.b.t
        stranded = (~t["reach_g"].any(2)).long()
        order = torch.argsort(-t["cQ"], dim=1, stable=True)
        order = torch.gather(order, 1, torch.argsort(torch.gather(t["r_min"], 1, order), dim=1, stable=True))
        order = torch.gather(order, 1, torch.argsort(1 - torch.gather(stranded, 1, order), dim=1, stable=True))
        self.rank = torch.empty_like(order)
        self.rank.scatter_(1, order, torch.arange(env.n, device=order.device).expand(env.B, -1).contiguous())

    def __call__(self, env, opt):
        i = self.rank.masked_fill(env.done, env.n + 1).argmin(1)
        dz = opt["dz"][env.ar, i].masked_fill(~opt["feas"][env.ar, i], INF)
        return i, dz.argmin(1)


def heuristic_rollout(env):
    """Run the greedy cost rule in the simulator (an EV without options is dropped instead of
    aborting, so every instance gets a cost)."""
    rule = GreedyRule(env)
    while True:
        opt = env.options()
        env.drop_dead(opt)
        active = ~env.finished()
        if not active.any():
            break
        i, k = rule(env, opt)
        env.step(i, k, active, opt)
    return env


# ====================================================================== plan out of the simulator
def plan_of(env, row, J):
    """Plan dict of one row (input of results_io.evaluate). EVs that could not be placed within the
    limits (dropped) are appended afterwards to the resource that finishes them earliest, ignoring
    the violated limit, so that the plan is complete; check_plan_f4 then lists the violations.
    Returns (plan, list of dropped EV numbers)."""
    b = env.b
    n, G, M = env.n, env.G, env.M
    res = env.res[row].tolist()
    S, C = env.S[row].tolist(), env.C[row].tolist()
    order = env.order[row].tolist()
    gun_station, gun_local = b.gun_station.tolist(), b.gun_local.tolist()
    dropped = [i for i in range(n) if res[i] < 0]
    if dropped:
        ev = J["ev"]
        r, R, tF, tM = ev["r_min"], ev["R_km"], ev["tauF_min"], ev["tauM_min"]
        dEF, tEF, tN = J["d_ev_fcs_km"], J["t_ev_fcs_min"], J["t_nodes_min"]
        gun_free = env.gun_free[row].tolist()
        van_pos, van_free = env.van_pos[row].tolist(), env.van_free[row].tolist()
        step = max(order) + 1 if order else 0
        for i in sorted(dropped, key=lambda q: (r[q], q)):
            best = None
            for g in range(G):
                f = gun_station[g]
                if dEF[i][f] > R[i]:
                    continue
                s = max(r[i] + tEF[i][f], gun_free[g])
                if best is None or s + tF[i] < best[0]:
                    best = (s + tF[i], g, s)
            for m in range(M):
                s = max(r[i], van_free[m] + tN[van_pos[m]][i + 1])
                if best is None or s + tM[i] < best[0]:
                    best = (s + tM[i], G + m, s)
            fin, k, s = best
            res[i], S[i], C[i], order[i] = k, s, fin, step
            step += 1
            if k < G:
                gun_free[k] = fin
            else:
                van_pos[k - G], van_free[k - G] = i + 1, fin
    resource = [""] * n
    for i in range(n):
        if res[i] < G:
            resource[i] = f"FCS{gun_station[res[i]] + 1}-gun{gun_local[res[i]] + 1}"
    routes, route_van, label = [], [], 0
    for m in range(M):
        stops = sorted((i for i in range(n) if res[i] == G + m), key=lambda q: order[q])
        if not stops:
            continue
        label += 1                                    # vans in use are numbered 1, 2, ... (as the MILP)
        for i in stops:
            resource[i] = f"van{label}"
        routes.append([0] + [i + 1 for i in stops] + [0])
        route_van.append(label)
    plan = {"resource": resource, "S": [float(x) for x in S], "C": [float(x) for x in C],
            "van_routes": routes, "route_van": route_van}
    return plan, [i + 1 for i in dropped]
