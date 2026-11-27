"""Trace one complete episode of the trained formulation-4 RL policy, step by step, for the code report.

Reads (nothing outside this folder is written):
  ../../data/*.csv, ../../common/f4_params.json      the dataset and the customer parameters
  ../model/model_best.pt                              the trained baseline policy
Writes: tables/trace_*.tex, trace.json (all numbers of the trace)

What it does: picks a small Solomon instance, runs ONE greedy rollout (map symmetry 0, the most likely
action at every step) exactly as rl_f4_model.rollout does, but records at every step: the open EVs and
their probabilities, the chosen EV, every resource with its mask reason, start, wait, finish and
increase of Z, the chosen resource, and the state update. At the end the simulator's Z is compared with
the independent checker (check_plan_f4) term by term.

Usage (from methods/rl/report):  python trace_episode.py [--instance EV-R201 --n 8]
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
F4 = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(F4, "common"))
sys.path.insert(0, os.path.join(F4, "methods", "rl", "code"))
import f4_data as fd            # noqa: E402
import f4_params as fp          # noqa: E402
from check_plan_f4 import validate  # noqa: E402
from rl_f4_env import Batch, Env, plan_of  # noqa: E402
from rl_f4_model import load_policy  # noqa: E402

TAB = os.path.join(HERE, "tables")


def f(x, d=1):
    s = f"{x:.{d}f}"
    return s.replace("-", "$-$")


def res_name(k, env):
    G = env.G
    if k < G:
        return f"FCS{int(env.b.gun_station[k]) + 1}-g{int(env.b.gun_local[k]) + 1}"
    return f"van{k - G + 1}"


def mask_reason(env, opt, i, k):
    """Why pair (i, k) is masked (or 'ok')."""
    t, G = env.b.t, env.G
    T = float(t["T"][0])
    if k < G:
        if not bool(t["reach_g"][0, i, k]):
            return "out of range"
        wait = float(opt["wait_g"][0, i, k])
        fin = float(opt["finish"][0, i, k])
        if wait > float(t["W"][0, i]) + 1e-9:
            return f"queue {wait:.0f} $>$ W"
        if fin > T + 1e-9:
            return "after T"
        if not bool(opt["feas"][0, i, k]):
            return "same as gun 1"
        return "ok"
    m = k - G
    if not bool(opt["feas"][0, i, k]):
        cnt = float(env.van_cnt[0, m])
        if cnt >= float(t["Q"][0]):
            return "van full"
        if cnt == 0 and not bool(opt["feas"][0, i, G:][[mm for mm in range(env.M) if float(env.van_cnt[0, mm]) == 0][0]]):
            pass
        unused = [mm for mm in range(env.M) if float(env.van_cnt[0, mm]) == 0]
        if cnt == 0 and unused and m != unused[0]:
            return "same as unused van"
        if float(opt["need"][0, i, m]) > float(env.van_batt[0, m]) + 1e-9:
            return "battery"
        return "back after T"
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="EV-RC201")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(TAB, exist_ok=True)
    P, cfg = fp.load_params(quiet=True)
    ds = fd.Dataset(fd.DEFAULT_DATA, P, cfg)
    J = ds.instance(args.instance, args.n)
    policy, ck = load_policy(os.path.join(F4, "methods", "rl", "model", "model_best.pt"), "cpu")
    env = Env(Batch([J], "cpu"))
    t = env.b.t
    emb = policy.encode(*env.static_features(torch.zeros(1, dtype=torch.long)))
    pre = policy.precompute(emb)
    out = {"instance": args.instance, "n": args.n, "T": float(t["T"][0]), "G": env.G, "M": env.M,
           "cost_scale": float(env.cost_scale[0]), "evs": [], "steps": []}
    ev = J["ev"]
    for i in range(args.n):
        reach = [fi + 1 for fi in fd.reachable(J, i)]
        out["evs"].append({"i": i + 1, "seg": "C" if ev["segment"][i] == 1 else "P", "theta": ev["theta"][i],
                           "r": ev["r_min"][i], "E": ev["E_kWh"][i], "R": ev["R_km"][i], "W": ev["W"][i],
                           "mu": ev["mu"][i], "cQ": ev["cQ"][i], "cC": ev["cC"][i], "cV": ev["cV"][i],
                           "tauF": ev["tauF_min"][i], "tauM": ev["tauM_min"][i], "stations": reach})
    with torch.no_grad():
        while True:
            opt = env.options()
            dead = env.drop_dead(opt)
            if bool(env.finished()[0]):
                break
            ev_mask = opt["feas"].any(2)
            phi = env.ev_features(opt)
            u, g = policy.ev_logits(emb, pre, env, phi, ev_mask)
            lp = Fn.log_softmax(u, -1)[0]
            i = int(lp.argmax())
            pair, kmask = env.pair_features(opt, torch.tensor([i]))
            v = policy.res_logits(emb, pre, env, torch.tensor([i]), g, pair, kmask)
            lpk = Fn.log_softmax(v, -1)[0]
            k = int(lpk.argmax())
            # all resources of the chosen EV (one line per station's gun 1 + every van)
            rows = []
            for kk in range(env.K):
                if kk < env.G and int(env.b.gun_local[kk]) != 0 and mask_reason(env, opt, i, kk) == "same as gun 1":
                    continue
                reason = mask_reason(env, opt, i, kk)
                if kk < env.G:
                    start, wait = float(opt["start_g"][0, i, kk]), float(opt["wait_g"][0, i, kk])
                else:
                    start, wait = float(opt["start_v"][0, i, kk - env.G]), float(opt["wait_v"][0, i, kk - env.G])
                rows.append({"k": kk, "name": res_name(kk, env), "reason": reason, "start": start, "wait": wait,
                             "finish": float(opt["finish"][0, i, kk]), "dz": float(opt["dz"][0, i, kk]),
                             "p": float(lpk[kk].exp()) if reason == "ok" else 0.0})
            open_evs = [(j + 1, float(lp[j].exp())) for j in range(env.n) if bool(ev_mask[0, j])]
            before = {"gun_free": env.gun_free[0].tolist(), "van_pos": env.van_pos[0].tolist(),
                      "van_free": env.van_free[0].tolist(), "van_batt": env.van_batt[0].tolist(),
                      "van_cnt": env.van_cnt[0].tolist()}
            dz = float(opt["dz"][0, i, k])
            env.step(torch.tensor([i]), torch.tensor([k]), torch.tensor([True]), opt)
            out["steps"].append({"step": len(out["steps"]) + 1, "dropped_before": [int(x) + 1 for x in torch.nonzero(dead[0]).flatten()],
                                 "open": open_evs, "ev": i + 1, "p_ev": float(lp[i].exp()),
                                 "res": res_name(k, env), "k": k, "p_res": float(lpk[k].exp()), "dz": dz,
                                 "S": float(env.S[0, i]), "C": float(env.C[0, i]), "cost_after": float(env.cost[0]),
                                 "rows": rows, "before": before,
                                 "after": {"gun_free": env.gun_free[0].tolist(), "van_pos": env.van_pos[0].tolist(),
                                           "van_free": env.van_free[0].tolist(), "van_batt": env.van_batt[0].tolist(),
                                           "van_cnt": env.van_cnt[0].tolist()}})
    plan, dropped = plan_of(env, 0, J)
    err, Z, parts, detail = validate(J, plan)
    out.update({"sim_Z": float(env.cost[0]), "check_Z": Z, "parts": parts, "violations": err, "dropped": dropped,
                "plan": plan})
    # van km telescoping for every route
    dN = J["d_nodes_km"]
    tele = []
    for rt in plan["van_routes"]:
        inc, pos = [], 0
        for node in rt[1:-1]:
            inc.append(dN[pos][node] + dN[node][0] - dN[pos][0])
            pos = node
        tele.append({"route": rt, "increments": inc, "sum": sum(inc),
                     "route_km": sum(dN[a][b] for a, b in zip(rt, rt[1:]))})
    out["telescoping"] = tele
    with open(os.path.join(HERE, "trace.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    # ---------------- LaTeX tables
    seg_word = {"P": "personal", "C": "commercial"}
    rows = []
    for e in out["evs"]:
        rows.append(f"{e['i']} & {seg_word[e['seg']]} & {f(e['theta'], 2)} & {f(e['r'], 0)} & {f(e['E'], 1)} & "
                    f"{f(e['R'], 1)} & {','.join(map(str, e['stations'])) or 'none (stranded)'} & {f(e['W'], 0)} & "
                    f"{f(e['mu'], 2)} & {f(e['cQ'], 2)} & {f(e['cC'], 2)} & {f(e['cV'], 2)} & {f(e['tauF'], 0)} & "
                    f"{f(e['tauM'], 0)} \\\\")
    with open(os.path.join(TAB, "trace_evs.tex"), "w", encoding="utf-8") as fh:
        fh.write("\\footnotesize\\begin{adjustbox}{max width=\\linewidth}\\begin{tabular}{@{}rlrrrrlrrrrrrr@{}}\\toprule\n"
                 "EV & segment & $\\theta$ & $r_i$ & $E_i$ & $R_i$ & reachable FCS & $W_i$ & $\\mu_i$ & $c^Q_i$ & $c^C_i$ & "
                 "$c^V_i$ & $\\tau^F_i$ & $\\tau^M_i$ \\\\\\midrule\n" + "\n".join(rows) +
                 "\n\\bottomrule\\end{tabular}\\end{adjustbox}\n")
    srows = []
    for s in out["steps"]:
        top = sorted(s["open"], key=lambda x: -x[1])[:3]
        top_s = ", ".join(f"{j}: {p:.2f}" for j, p in top)
        srows.append(f"{s['step']} & {len(s['open'])} & {top_s} & EV{s['ev']} ({s['p_ev']:.2f}) & {s['res']} ({s['p_res']:.2f}) & "
                     f"{f(s['S'], 1)} & {f(s['C'], 1)} & {f(s['dz'], 1)} & {f(s['cost_after'], 1)} \\\\")
    with open(os.path.join(TAB, "trace_steps.tex"), "w", encoding="utf-8") as fh:
        fh.write("\\footnotesize\\begin{adjustbox}{max width=\\linewidth}\\begin{tabular}{@{}rrlllrrrr@{}}\\toprule\n"
                 "step & open & top EV probabilities & chosen EV ($p$) & chosen resource ($p$) & $S_i$ & $C_i$ & "
                 "$\\Delta Z$ (Rs) & cost so far \\\\\\midrule\n" + "\n".join(srows) +
                 "\n\\bottomrule\\end{tabular}\\end{adjustbox}\n")
    for snum in (1, 3):
        sx = out["steps"][snum - 1]
        rrows = []
        for r in sx["rows"]:
            ok = r["reason"] == "ok"
            rrows.append(f"{r['name']} & {r['reason']} & {f(r['start'], 1) if ok else '--'} & "
                         f"{f(r['wait'], 1) if ok else '--'} & {f(r['finish'], 1) if ok else '--'} & "
                         f"{f(r['dz'], 1) if ok else '--'} & {r['p']:.3f} \\\\")
        with open(os.path.join(TAB, f"trace_step{snum}.tex"), "w", encoding="utf-8") as fh:
            fh.write("\\footnotesize\\begin{adjustbox}{max width=\\linewidth}\\begin{tabular}{@{}llrrrrr@{}}\\toprule\n"
                     "resource & mask & start $S$ & wait & finish $C$ & $\\Delta Z$ (Rs) & $p(k\\mid i)$ \\\\\\midrule\n"
                     + "\n".join(rrows) + "\n\\bottomrule\\end{tabular}\\end{adjustbox}\n")
    prow = [f"$Z_{j}$ & {f(parts[f'Z{j}'], 2)} \\\\" for j in range(1, 7)]
    with open(os.path.join(TAB, "trace_parts.tex"), "w", encoding="utf-8") as fh:
        fh.write("\\small\\begin{tabular}{@{}lr@{}}\\toprule term & checker (Rs) \\\\\\midrule\n" + "\n".join(prow) +
                 f"\n\\midrule $Z$ (checker) & {f(Z, 2)} \\\\\n sum of $\\Delta Z$ (simulator) & {f(out['sim_Z'], 2)} \\\\"
                 "\n\\bottomrule\\end{tabular}\n")
    print(json.dumps({k: out[k] for k in ("instance", "n", "T", "sim_Z", "check_Z", "violations", "dropped", "telescoping")},
                     indent=1))
    for s in out["steps"]:
        print(s["step"], "EV", s["ev"], round(s["p_ev"], 3), s["res"], round(s["p_res"], 3), "dz", round(s["dz"], 2),
              "S", round(s["S"], 1), "open", len(s["open"]), "dropped", s["dropped_before"])


if __name__ == "__main__":
    main()
