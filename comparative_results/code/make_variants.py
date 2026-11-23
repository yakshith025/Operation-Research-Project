"""Inputs of every experiment variant: a changed copy of the baseline inputs.

For a variant (e.g. E2_vans_3) the folder comparative_results/variants/<variant id>/ holds
    csv/<source>_config.csv, csv/<source>_data.csv   the dataset (CSV form) with the change applied
    income/<FILE>.csv, income/theta_settings.json      theta per EV (copied, or re-mapped for E7)
    f3_params.json                                     income constants with the change applied
    variant.json                                       what was changed, and a fingerprint of the base
Both solvers read exactly these files (solve_milp_f3.py DATA --params ..., solve_rl1.py --csv ...
--f3-params ...); the income folder sits next to csv/ as in dataset_ev_solomon/.

Kinds of change (experiments.json, factor "kind")
    param       one constant of f3_params.json (dotted key, e.g. tariff_Rs_per_kWh.p_M)
    config      one column of <source>_config.csv for every file (n_vans, van_max_evs, van_battery_kWh)
    guns        guns of every station (data rows of type station; config guns_per_station / total_guns)
    horizon     T_min times the factor, only for the files of the listed families (E3: R1, RC1)
    theta_beta  income distribution Beta(a, b); theta is re-mapped by quantiles,
                theta_new = F_{a,b}^-1( F_{a0,b0}(theta_base) ), so each EV keeps its income rank
Everything else is copied unchanged. A variant whose folder is complete and whose base inputs did
not change is not written again (use --force to rebuild).

Usage (from comparative_results/code)
    python make_variants.py                    # variants of the default priorities (1-4)
    python make_variants.py --all              # every variant, incl. E3 (priority 5)
    python make_variants.py --experiments E7 --force
    python make_variants.py --check            # build, then verify each variant against the base
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time

import common as cm


# ====================================================================== Beta distribution (stdlib only)
def _betacf(a, b, x):
    """Continued fraction of the regularised incomplete beta function (Numerical Recipes)."""
    tiny, eps = 1e-300, 3e-15
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def beta_cdf(x, a, b):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    bt = math.exp(lbt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def beta_ppf(u, a, b):
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if beta_cdf(mid, a, b) < u:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13:
            break
    return 0.5 * (lo + hi)


def remap_theta(theta, a0, b0, a1, b1):
    """Quantile mapping Beta(a0, b0) -> Beta(a1, b1), rounded like the income files (4 decimals)."""
    return [round(beta_ppf(beta_cdf(t, a0, b0), a1, b1), 4) for t in theta]


# ====================================================================== helpers
def fmt(x):
    x = float(x)
    return str(int(x)) if x.is_integer() else f"{x:.10g}"


def read_csv(path):
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        return rd.fieldnames, list(rd)


def write_csv(path, cols, rows):
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)


def set_dotted(P, key, value):
    node = P
    parts = key.split(".")
    for p in parts[:-1]:
        if p not in node:
            raise KeyError(f"f3_params has no {key}")
        node = node[p]
    if parts[-1] not in node:
        raise KeyError(f"f3_params has no {key}")
    node[parts[-1]] = value


def get_dotted(P, key):
    node = P
    for p in key.split("."):
        node = node[p]
    return node


def base_paths(EXP, project=cm.PROJECT):
    csv_dir = cm.project_path(EXP["baseline"]["dataset_csv"], project)
    src = EXP["instances"]["source"]
    return {"csv": csv_dir, "config": os.path.join(csv_dir, f"{src}_config.csv"),
            "data": os.path.join(csv_dir, f"{src}_data.csv"),
            "income": os.path.join(os.path.dirname(csv_dir), "income"),
            "params": cm.project_path(EXP["baseline"]["f3_params"], project), "source": src}


def fingerprint(B, names):
    h = hashlib.sha256()
    for p in (B["config"], B["data"], B["params"]):
        with open(p, "rb") as fh:
            h.update(fh.read())
    for nm in sorted(names):
        p = os.path.join(B["income"], f"{nm}.csv")
        if os.path.exists(p):
            with open(p, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()[:16]


def read_theta(path):
    with open(path, newline="") as fh:
        return [float(r["theta"]) for r in csv.DictReader(fh)]


# ====================================================================== build one variant
def build(v, EXP, out_root=cm.DEFAULT_VARIANTS, project=cm.PROJECT, force=False, quiet=False):
    """Write the inputs of variant v (a dict of common.variants) and return its folder."""
    B = base_paths(EXP, project)
    for k in ("config", "data", "params"):
        if not os.path.exists(B[k]):
            raise FileNotFoundError(f"baseline input missing: {B[k]}")
    cfg_cols, cfg_rows = read_csv(B["config"])
    names = [r["instance"] for r in cfg_rows]
    folder = os.path.join(out_root, v["id"])
    spec = {"id": v["id"], "experiment": v["experiment"], "name": v["name"], "kind": v["kind"],
            "key": v["key"], "value": v["value"], "baseline": v["baseline"], "families": v["families"],
            "source": B["source"], "base_fingerprint": fingerprint(B, names)}
    marker = os.path.join(folder, "variant.json")
    if not force and os.path.exists(marker):
        try:
            old = json.load(open(marker))
            if {k: old.get(k) for k in spec} == spec:
                return folder
        except (OSError, ValueError):
            pass
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    csv_out, inc_out = os.path.join(folder, "csv"), os.path.join(folder, "income")
    os.makedirs(csv_out)
    os.makedirs(inc_out)
    data_cols, data_rows = read_csv(B["data"])
    with open(B["params"], encoding="utf-8") as fh:
        P = json.load(fh)
    P_new = copy.deepcopy(P)
    kind, key, val = v["kind"], v["key"], v["value"]
    changed = []
    if kind == "param":
        old = get_dotted(P, key)
        set_dotted(P_new, key, float(val))
        changed.append(f"f3_params {key}: {old:g} -> {float(val):g}")
    elif kind == "config":
        if key not in cfg_cols:
            raise KeyError(f"{B['config']} has no column {key}")
        olds = sorted({r[key] for r in cfg_rows})
        for r in cfg_rows:
            r[key] = fmt(val)
        changed.append(f"config {key}: {', '.join(olds)} -> {fmt(val)} (all {len(cfg_rows)} files)")
    elif kind == "guns":
        olds = sorted({r["guns"] for r in data_rows if r["type"] == "station"})
        for r in data_rows:
            if r["type"] == "station":
                r["guns"] = fmt(val)
        for r in cfg_rows:
            if "guns_per_station" in r:
                r["guns_per_station"] = fmt(val)
            if "total_guns" in r and r.get("n_stations"):
                r["total_guns"] = fmt(int(float(r["n_stations"])) * int(val))
        changed.append(f"guns per station: {', '.join(olds)} -> {fmt(val)} (every station of every file)")
    elif kind == "horizon":
        fams = {f.upper() for f in (v["families"] or [])}
        hit = 0
        for r in cfg_rows:
            if not fams or r["class"].upper() in fams:
                r["T_min"] = fmt(round(float(r["T_min"]) * float(val), 6))
                hit += 1
        changed.append(f"T_min x {float(val):g} for {hit} files ({', '.join(sorted(fams)) or 'all'})")
    elif kind == "theta_beta":
        a0, b0 = P["theta_distribution"]["a"], P["theta_distribution"]["b"]
        P_new["theta_distribution"]["a"], P_new["theta_distribution"]["b"] = float(val[0]), float(val[1])
        changed.append(f"theta ~ Beta({a0:g},{b0:g}) -> Beta({val[0]:g},{val[1]:g}) by quantile mapping "
                       f"(same income rank per EV)")
    else:
        raise ValueError(f"unknown kind {kind}")

    write_csv(os.path.join(csv_out, f"{B['source']}_config.csv"), cfg_cols, cfg_rows)
    write_csv(os.path.join(csv_out, f"{B['source']}_data.csv"), data_cols, data_rows)
    with open(os.path.join(folder, "f3_params.json"), "w", encoding="utf-8") as fh:
        json.dump(P_new, fh, indent=2)

    # income: theta per EV, next to csv/ (where make_income looks for it)
    sys.path.insert(0, cm.code_dir(project))
    import make_income as inc                           # noqa: E402  (project code)
    n_of = {r["instance"]: int(float(r["n_evs"])) for r in cfg_rows}
    for nm in names:
        src = os.path.join(B["income"], f"{nm}.csv")
        theta = read_theta(src) if os.path.exists(src) else []
        if len(theta) < n_of[nm]:                       # no base file: the base draw (as make_income)
            theta = [float(t) for t in inc.draw_theta(nm, n_of[nm], P)]
        if kind == "theta_beta":
            theta = remap_theta(theta, P["theta_distribution"]["a"], P["theta_distribution"]["b"],
                                float(val[0]), float(val[1]))
        with open(os.path.join(inc_out, f"{nm}.csv"), "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["EV", "theta"])
            wr.writerows([i + 1, f"{t:.4f}"] for i, t in enumerate(theta))
    with open(os.path.join(inc_out, "theta_settings.json"), "w") as fh:   # same stamp -> never re-drawn
        json.dump({k: P_new["theta_distribution"][k] for k in ("a", "b", "seed_base")}, fh, indent=1)
    for w in inc.check_params(copy.deepcopy(P_new)):
        print(f"WARNING ({v['id']} f3_params): {w}", flush=True)

    spec.update({"changed": changed, "base_inputs": {k: B[k] for k in ("config", "data", "income", "params")},
                 "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    with open(marker, "w") as fh:
        json.dump(spec, fh, indent=1)
    if not quiet:
        print(f"{v['id']}: {'; '.join(changed)}  -> {folder}", flush=True)
    return folder


# ====================================================================== verification
def check(v, folder, EXP, project=cm.PROJECT, files=None):
    """Read every file of the variant and of the base at n = file size through the solvers' own
    reader (make_ev_solomon.EvSolomonData + make_income) and confirm that exactly the intended input
    changed. Returns a list of problems (empty = fine)."""
    import numpy as np
    sys.path.insert(0, cm.code_dir(project))
    import make_ev_solomon as mes
    import make_income as inc
    B = base_paths(EXP, project)
    base, var = mes.EvSolomonData(B["csv"]), mes.EvSolomonData(os.path.join(folder, "csv"))
    P0, P1 = inc.load_params(B["params"]), inc.load_params(os.path.join(folder, "f3_params.json"))
    fams = {f.upper() for f in (v["families"] or [])}
    bad = []
    base_names = [nm for nm in base.names() if base.source(nm) == B["source"]]
    names = files or base_names
    if sorted(var.names()) != sorted(base_names):
        bad.append(f"the variant lists other files than the base's {B['source']} files")
    for nm in names:
        D0, D1 = base.read(nm, base.n_evs(nm)), var.read(nm, var.n_evs(nm))
        th0 = inc.read_income(base.path, nm, D0["n"], P0)
        th1 = inc.read_income(var.path, nm, D1["n"], P1)
        expect = dict(D0["params"])
        guns0 = D0["guns"]
        if v["kind"] == "config":
            ck = {"n_vans": "n_vans", "van_max_evs": "Q", "van_battery_kWh": "B"}[v["key"]]
            expect[ck] = float(v["value"])
        if v["kind"] == "horizon" and (not fams or base.family(nm).upper() in fams):
            expect["T"] = round(D0["params"]["T"] * float(v["value"]), 6)
        if v["kind"] == "guns":
            guns0 = np.full_like(D0["guns"], int(v["value"]))
        for k_, x0 in expect.items():
            if abs(float(D1["params"][k_]) - float(x0)) > 1e-6:
                bad.append(f"{nm}: {k_} = {D1['params'][k_]}, expected {x0}")
        if not np.array_equal(D1["guns"], guns0):
            bad.append(f"{nm}: guns {sorted(set(D1['guns']))}")
        for k_ in ("E", "r", "R", "tauF", "tauM", "d_ev_fcs", "d_nodes", "P_F"):
            if not np.allclose(D0[k_], D1[k_]):
                bad.append(f"{nm}: {k_} changed")
        if v["kind"] == "theta_beta":
            # quantile mapping is increasing: sorted by the base theta, the new theta never falls
            # (up to the 4-decimal rounding of the income files)
            if not np.all(np.diff(th1[np.argsort(th0, kind="stable")]) >= -1e-4):
                bad.append(f"{nm}: theta ranks changed")
        elif not np.array_equal(th0, th1):
            bad.append(f"{nm}: theta changed")
    p0 = copy.deepcopy(P0)
    if v["kind"] == "param":
        set_dotted(p0, v["key"], float(v["value"]))
    if v["kind"] == "theta_beta":
        p0["theta_distribution"]["a"], p0["theta_distribution"]["b"] = float(v["value"][0]), float(v["value"][1])
    if json.dumps(p0, sort_keys=True) != json.dumps(P1, sort_keys=True):
        bad.append("f3_params differ from the base in more than the intended constant")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--priority", nargs="*", type=int)
    ap.add_argument("--experiments", nargs="*")
    ap.add_argument("--variants", nargs="*", help="variant ids")
    ap.add_argument("--all", action="store_true", help="every variant (all priorities)")
    ap.add_argument("--out", default=cm.DEFAULT_VARIANTS)
    ap.add_argument("--project", default=cm.PROJECT)
    ap.add_argument("--force", action="store_true", help="rebuild even if the folder is up to date")
    ap.add_argument("--check", action="store_true", help="verify every variant against the base inputs")
    args = ap.parse_args()
    EXP = cm.load_experiments()
    pr = sorted({e["priority"] for e in EXP["experiments"]}) if args.all else args.priority
    sel = cm.select(EXP, pr, args.experiments, args.variants)
    for v in sel:
        folder = build(v, EXP, args.out, args.project, force=args.force)
        if args.check:
            bad = check(v, folder, EXP, args.project)
            print(f"  check {v['id']}: {'OK' if not bad else f'{len(bad)} problem(s): {bad[:5]}'}", flush=True)
    print(f"{len(sel)} variant(s) in {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
