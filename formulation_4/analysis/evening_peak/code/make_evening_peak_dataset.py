"""Reshape the EV-Solomon request times so that demand has an evening peak.

Only `request_min` changes; positions, energies, ranges, stations, vans, costs, segments and incomes are those of
the formulation-4 dataset (formulation_4/data/solomon_*.csv). The result has the same format, so every
formulation-4 solver can read it:
    python methods/milp/code/solve_milp_f4.py analysis/evening_peak/data ...
    python methods/rl/code/solve_rl_f4.py --data analysis/evening_peak/data ...
(both from inside formulation_4)

Method (per file, 100 EVs, seeded, deterministic):
 1. Target profile over the planning period, u = t / T in [0, 1]:
        g(u) = (1 - w) * Uniform(0, 1) + w * Normal(mu, sigma) truncated to [0, 1]
    default w = 0.4, mu = 0.75, sigma = 0.08: a flat daytime base and a peak at three quarters of the day.
 2. The EVs keep Solomon's order: EVs are ranked by their original request time (ties, e.g. the many
    requests at t = 0, broken at random) and EV of rank k gets the quantile u_k = G^-1((k - 0.5) / 100).
    The first-n instances are then a spread-out sample of the profile (Solomon's ids are not sorted by time).
 3. Feasibility: the model needs every EV to finish charging by T (constraint 21), so an EV's request is
    placed inside its own feasible window:  r_i = round(u_k * L_i),
        L_i = max(0, T - slack * need_i),
        need_i = min( drive to a reachable station + station charging time,  van charging time + van drive
                      from the EV back to the depot ),
    slack = 1.25 by default (room for queues). For long periods (C2, R2, RC2, C1) L_i is close to T and the
    peak sits near 0.75 T; in the short R1 / RC1 periods (230-240 min) a full charge takes up to ~110 min, so
    the peak lands earlier in clock time.

Writes <out>/solomon_config.csv (copied), <out>/solomon_data.csv, <out>/request_mapping.csv (old and new time
per EV, L_i), <out>/settings.json and <out>/README.md.
Usage: python make_evening_peak_dataset.py [--w 0.4] [--mu 0.75] [--sigma 0.08] [--slack 1.25] [--seed 2026]
                                            [--out ../data] [--force]
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # formulation_4/analysis/evening_peak
REPO = ROOT.parent.parent.parent
F4 = REPO / "formulation_4"
SRC = F4 / "data"


def target_quantiles(p, w, mu, sigma, grid=200001):
    u = np.linspace(0, 1, grid)
    norm_part = np.trapezoid(np.exp(-0.5 * ((u - mu) / sigma) ** 2), u)
    dens = (1 - w) + w * np.exp(-0.5 * ((u - mu) / sigma) ** 2) / norm_part
    cdf = np.concatenate([[0], np.cumsum((dens[1:] + dens[:-1]) / 2 * np.diff(u))])
    cdf /= cdf[-1]
    return np.interp(p, cdf, u)


def latest_feasible(evs, stations, depot, cfg, slack):
    """L_i = T - slack * need_i (minutes), need_i = fastest way to finish charging (see module doc)."""
    k = cfg["detour"] * cfg["km_per_unit"]
    to_min = 60.0 / cfg["speed_kmph"]
    E = evs["energy_kWh"].to_numpy()
    R = evs["range_km"].to_numpy()
    xy = evs[["x", "y"]].to_numpy()
    sxy = stations[["x", "y"]].to_numpy()
    spow = stations["power_kW"].to_numpy()
    d_es = k * np.linalg.norm(xy[:, None, :] - sxy[None, :, :], axis=2)            # km
    fcs = np.where(d_es <= R[:, None], d_es * to_min + 60 * E[:, None] / spow[None, :], np.inf).min(1)
    d_e0 = k * np.linalg.norm(xy - depot, axis=1)
    van = 60 * E / cfg["van_power_kW"] + d_e0 * to_min
    need = np.minimum(fcs, van)
    return np.maximum(0.0, cfg["T_min"] - slack * need), need


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=float, default=0.4, help="weight of the evening peak")
    ap.add_argument("--mu", type=float, default=0.75, help="peak position as a fraction of the feasible window")
    ap.add_argument("--sigma", type=float, default=0.08, help="peak width (fraction of the window)")
    ap.add_argument("--slack", type=float, default=1.25, help="multiplier on the fastest charging path")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(ROOT / "data"))
    ap.add_argument("--force", action="store_true", help="overwrite an existing output folder")
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists() and any(out.iterdir()) and not a.force:
        raise SystemExit(f"{out} exists; use --force to overwrite")
    out.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(SRC / "solomon_data.csv", dtype={"segment": str})
    cfg_all = pd.read_csv(SRC / "solomon_config.csv").set_index("instance")
    new = data.copy()
    mapping = []
    for f in cfg_all.index:
        cfg = cfg_all.loc[f]
        rows = data[data.instance == f]
        evs = rows[rows.type == "ev"]
        st = rows[rows.type == "station"]
        depot = rows[rows.type == "depot"][["x", "y"]].to_numpy()[0]
        L, need = latest_feasible(evs, st, depot, cfg, a.slack)
        seed = int(hashlib.sha256(f"{a.seed}-{f}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        r_old = evs["request_min"].to_numpy()
        order = np.lexsort((rng.random(len(r_old)), r_old))          # by old time, random tie-break
        rank = np.empty(len(r_old), int)
        rank[order] = np.arange(len(r_old))
        u = target_quantiles((rank + 0.5) / len(r_old), a.w, a.mu, a.sigma)
        r_new = np.floor(u * L).astype(int)                         # floor keeps r_new <= L_i
        new.loc[evs.index, "request_min"] = r_new
        for j, idx in enumerate(evs.index):
            mapping.append({"instance": f, "class": cfg["class"], "id": int(evs.loc[idx, "id"]),
                            "T_min": int(cfg["T_min"]), "request_old": int(r_old[j]),
                            "request_new": int(r_new[j]), "rank": int(rank[j]), "u": round(float(u[j]), 4),
                            "need_min": round(float(need[j]), 1), "latest_min": round(float(L[j]), 1),
                            "old_after_latest": bool(r_old[j] > L[j])})

    # keep integer formatting of the source (blank for depot / station rows)
    new["request_min"] = new["request_min"].astype("Int64")
    new.to_csv(out / "solomon_data.csv", index=False)
    shutil.copy(SRC / "solomon_config.csv", out / "solomon_config.csv")
    mp = pd.DataFrame(mapping)
    mp.to_csv(out / "request_mapping.csv", index=False)

    settings = {"source": str(SRC.relative_to(REPO)).replace("\\", "/"), "changed_column": "request_min",
                "profile": {"type": "(1-w) Uniform(0,1) + w TruncNormal(mu, sigma) on u = t / L_i",
                            "w": a.w, "mu": a.mu, "sigma": a.sigma},
                "latest_feasible": "L_i = max(0, T - slack * need_i)", "slack": a.slack, "seed": a.seed,
                "order": "rank of original request time, random tie-break (seed per file)",
                "evs_with_old_request_after_L": int(mp.old_after_latest.sum()),
                "evs_with_L_zero": int((mp.latest_min == 0).sum())}
    (out / "settings.json").write_text(json.dumps(settings, indent=2))

    by_cls = mp.groupby("class").agg(T=("T_min", "first"), mean_need=("need_min", "mean"),
                                     mean_L_over_T=("latest_min", lambda s: (s / mp.loc[s.index, "T_min"]).mean()),
                                     old_after_L=("old_after_latest", "sum")).round(3)
    readme = f"""# Formulation-4 data with an evening demand peak

Same as `formulation_4/data/solomon_*.csv` except the column `request_min` of EV rows (Solomon 100-EV files only; the
Gehring-Homberger files are not included). Built by `../code/make_evening_peak_dataset.py`.

- Profile: (1 - w) uniform + w truncated normal, w = {a.w}, peak at {a.mu} of each EV's feasible window,
  sigma = {a.sigma}.
- Order of the EVs over the day kept from Solomon (rank of the original request time; ties at random).
- Every request leaves time to finish charging by T: r_i <= L_i = T - {a.slack} x (fastest charging path).
  EVs with L_i = 0: {settings['evs_with_L_zero']}.
- Original data for comparison: {settings['evs_with_old_request_after_L']} of {len(mp)} EVs had a request
  after their L_i.

Per class (n = 100):

{by_cls.reset_index().to_markdown(index=False)}

Run any formulation-4 solver on it by pointing it at this folder, e.g. from `formulation_4`:
`python methods/milp/code/solve_milp_f4.py analysis/evening_peak/data --out <new results folder>`
(use a new results folder: the default one holds the baseline results on the original data).
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    print(readme)


if __name__ == "__main__":
    main()
