"""Income attribute of main_formulation_3: one income index theta_i in [0, 1] per EV
(0 = lowest income, 1 = highest) and the sensitivities it sets.

The theta values are data, written once per EV-Solomon file to
    ../dataset_ev_solomon/income/<FILE>.csv        (columns: EV, theta)
and drawn from Beta(a, b) (settings in f3_params.json). Within a Solomon class (e.g. R101-R112)
positions, demands and ranges are identical, so the same EV number also gets the same theta in
every file of that class: files of one class then differ only in their request times.

The sensitivities are computed from theta at solve time (so changing a constant in
f3_params.json never needs new data):
    mu(theta)  money sensitivity (Rs perceived per Rs paid), non-increasing
    c^a(theta) value of time (Rs/min), non-decreasing, for a = D driving to an FCS,
               Q queueing at an FCS, C charging at an FCS, V car parked for a van (driver free)
    W(theta)   queue tolerance at an FCS (min), non-increasing

Usage (from code/)
    python make_income.py                  # write the income files for every EV-Solomon file
    python make_income.py --force          # rewrite them (e.g. after changing theta_distribution)
The solver (solve_milp_f3.py) calls ensure_income() itself, so this script is optional.
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_ev_solomon as mes  # noqa: E402

DEFAULT_PARAMS = os.path.join(HERE, "f3_params.json")
DEFAULT_INSTANCES = os.path.join(HERE, "..", "dataset_ev_solomon", "instances")


# ------------------------------------------------------------------ parameters
def load_params(path=None):
    """Read f3_params.json and return it with every number as float. Problems with the values
    (wrong direction, Q <= D, ...) are printed as warnings, never raised: the run goes on."""
    path = path or DEFAULT_PARAMS
    with open(path, encoding="utf-8") as fh:
        P = json.load(fh)
    for w in check_params(P):
        print(f"WARNING (f3_params): {w}", flush=True)
    return P


def check_params(P):
    warn = []
    mu = P["money_sensitivity_mu"]
    if not (mu["theta0"] >= mu["theta1"] > 0):
        warn.append("money sensitivity should satisfy theta0 >= theta1 > 0")
    vot = P["value_of_time_Rs_per_min"]
    for a in "DQVC":
        if not (vot[a]["theta1"] >= vot[a]["theta0"] >= 0):
            warn.append(f"value of time {a} should satisfy theta1 >= theta0 >= 0")
    for end in ("theta0", "theta1"):
        if not vot["Q"][end] > vot["D"][end]:
            warn.append(f"queueing should cost more than driving at {end}")
        if not vot["V"][end] <= vot["C"][end]:
            warn.append(f"driver-free van time (V) should cost at most FCS charging time (C) at {end}")
    if vot["V"]["theta1"] - vot["V"]["theta0"] > vot["C"]["theta1"] - vot["C"]["theta0"]:
        warn.append("V should not rise faster with income than C")
    W = P["queue_tolerance_W_min"]
    if not (W["theta0"] >= W["theta1"] >= 0):
        warn.append("queue tolerance should satisfy theta0 >= theta1 >= 0")
    tar = P["tariff_Rs_per_kWh"]
    if not tar["p_M"] >= tar["p_F"]:
        warn.append("van tariff p_M should be at least the FCS tariff p_F")
    d = P["theta_distribution"]
    if not (d["a"] > 0 and d["b"] > 0):
        warn.append("Beta(a, b) needs a > 0 and b > 0; using a = b = 1 (uniform)")
        d["a"], d["b"] = 1.0, 1.0
    return warn


def line(ends, theta):
    """Straight line from ends['theta0'] (theta = 0) to ends['theta1'] (theta = 1)."""
    return ends["theta0"] + (ends["theta1"] - ends["theta0"]) * np.asarray(theta, dtype=float)


def sensitivities(theta, P):
    """Per-EV coefficients of main_formulation_3 from the income indices theta (array)."""
    theta = np.clip(np.asarray(theta, dtype=float), 0.0, 1.0)
    vot = P["value_of_time_Rs_per_min"]
    out = {"theta": theta,
           "mu": line(P["money_sensitivity_mu"], theta),
           "W": np.maximum(0.0, line(P["queue_tolerance_W_min"], theta))}
    for a in "DQVC":
        out["c" + a] = np.maximum(0.0, line(vot[a], theta))
    return out


# ------------------------------------------------------------------ data
def class_seed(base, N, P):
    """Seed per Solomon class and file size: same theta for the same EV within a class."""
    cls = mes.solomon_class(base[3:] if base.upper().startswith("EV-") else base)
    return int(P["theta_distribution"]["seed_base"]) + sum(ord(ch) for ch in cls) * 101 + int(N)


def draw_theta(base, N, P):
    d = P["theta_distribution"]
    rng = np.random.default_rng(class_seed(base, N, P))
    return np.round(rng.beta(d["a"], d["b"], int(N)), 4)


def income_dir(instances_dir):
    return os.path.join(os.path.dirname(os.path.abspath(instances_dir.rstrip("/\\"))), "income")


def income_file(instances_dir, base):
    return os.path.join(income_dir(instances_dir), f"{base}.csv")


def n_evs_in_file(path):
    """Number of EVs in an EV-Solomon file (rows after the depot in EV REQUESTS)."""
    lines = open(path).read().splitlines()
    k = lines.index("EV REQUESTS")
    return sum(1 for s in lines[k + 2:] if s.strip()) - 1


def ensure_income(files, instances_dir, P, force=False, quiet=False):
    """Write <income>/<FILE>.csv for every EV-Solomon file that has none (or all of them with
    force=True, or when the theta settings changed since the files were written)."""
    folder = income_dir(instances_dir)
    os.makedirs(folder, exist_ok=True)
    stamp_path = os.path.join(folder, "theta_settings.json")
    stamp = {k: P["theta_distribution"][k] for k in ("a", "b", "seed_base")}
    try:
        old = json.load(open(stamp_path))
    except (OSError, ValueError):
        old = None
    if old is not None and old != stamp:
        if not quiet:
            print(f"theta settings changed ({old} -> {stamp}): rewriting the income files", flush=True)
        force = True
    written = 0
    for path in files:
        base = os.path.splitext(os.path.basename(path))[0]
        out = income_file(instances_dir, base)
        if os.path.exists(out) and not force:
            continue
        N = n_evs_in_file(path)
        theta = draw_theta(base, N, P)
        with open(out, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["EV", "theta"])
            wr.writerows([i + 1, f"{t:.4f}"] for i, t in enumerate(theta))
        written += 1
    with open(stamp_path, "w") as fh:
        json.dump(stamp, fh, indent=1)
    if written and not quiet:
        print(f"income files written: {written} (in {os.path.abspath(folder)})", flush=True)
    return written


def read_income(instances_dir, base, n, P):
    """theta of the first n EVs of a file; regenerates the file if it is missing or short."""
    path = income_file(instances_dir, base)
    theta = []
    try:
        with open(path) as fh:
            theta = [float(r["theta"]) for r in csv.DictReader(fh)]
    except (OSError, KeyError, ValueError):
        theta = []
    if len(theta) < n:
        theta = list(draw_theta(base, max(n, len(theta)), P))
    return np.clip(np.array(theta[:n], dtype=float), 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("instances", nargs="?", default=DEFAULT_INSTANCES)
    ap.add_argument("--params", default=DEFAULT_PARAMS)
    ap.add_argument("--force", action="store_true", help="rewrite existing income files")
    args = ap.parse_args()
    P = load_params(args.params)
    files = sorted(glob.glob(os.path.join(args.instances, "*", "EV-*.txt")))
    ensure_income(files, args.instances, P, force=args.force)
    rows = []
    for path in files:
        base = os.path.splitext(os.path.basename(path))[0]
        th = read_income(args.instances, base, n_evs_in_file(path), P)
        rows.append({"file": base, "EVs": len(th), "theta_mean": round(float(th.mean()), 3),
                     "theta_min": round(float(th.min()), 3), "theta_max": round(float(th.max()), 3),
                     **{f"share_{lo:.1f}_{lo + 0.2:.1f}": round(float(np.mean(
                         (th >= lo) & ((th < lo + 0.2) if lo < 0.8 else (th <= 1.0)))), 3)
                        for lo in (0.0, 0.2, 0.4, 0.6, 0.8)}})
    out = os.path.join(income_dir(args.instances), "summary.csv")
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"{len(rows)} files; summary in {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
