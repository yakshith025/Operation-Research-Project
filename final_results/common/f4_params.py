"""Customer-side parameters of formulation 4 and the per-EV coefficients they set.

Every EV i has two attributes (data, in data/<source>_data.csv):
    segment s_i in {personal, commercial}     (stored as the words; 0 = personal, 1 = commercial in arrays)
    theta_i in [0, 1]                          income index (0 = lowest, 1 = highest)
and the coefficients of the objective are straight lines in theta with end points per segment:
    mu_i   = mu^{s_i}(theta_i)                  money sensitivity (Rs perceived per Rs paid), non-increasing
    c^a_i  = c^{a, s_i}(theta_i), a in D Q C V  value of time (Rs/min), non-decreasing
    W_i    = W^{s_i}(theta_i)                   queue tolerance at an FCS (min), non-increasing
Commercial: D = C = V (downtime rate omega), Q >= D.

Overrides (experiments): load_params(path, sets) applies "dotted.key=value" strings, e.g.
    personal.value_of_time_Rs_per_min.C.theta1=12.5     tariff_Rs_per_kWh.p_M=35
Keys starting with "config." are not parameters: they change a column of <source>_config.csv for every
file (config.n_vans=3, config.van_max_evs=8, config.van_battery_kWh=200) and are returned separately.
Problems with the values are printed as warnings, never raised (the run goes on).
"""
import copy
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
F4_DIR = os.path.dirname(HERE)
DEFAULT_PARAMS = os.path.join(HERE, "f4_params.json")
SEGMENTS = ("personal", "commercial")          # index 0, 1
ACTIVITIES = "DQVC"
CONFIG_KEYS = {"n_vans", "van_max_evs", "van_battery_kWh", "van_power_kW", "van_use_kWh_per_km",
               "van_efficiency", "c_EV_Rs_per_km", "c_M_Rs_per_km", "c_F_Rs_per_van", "pi_F_Rs_per_kWh",
               "pi_M_Rs_per_kWh", "T_min"}


def strip_about(d):
    if isinstance(d, dict):
        return {k: strip_about(v) for k, v in d.items() if not str(k).startswith("_")}
    return d


def parse_value(text):
    text = str(text).strip()
    try:
        return json.loads(text)
    except ValueError:
        return text


def split_sets(sets):
    """'key=value' strings -> (parameter overrides {dotted key: value}, config overrides {column: value})."""
    par, cfg = {}, {}
    for s in sets or []:
        if "=" not in s:
            print(f"WARNING: --set {s!r} has no '=' and is ignored", flush=True)
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        if k.startswith("config."):
            col = k[len("config."):]
            if col not in CONFIG_KEYS:
                print(f"WARNING: config override {col!r} is not a known config column; applied anyway", flush=True)
            cfg[col] = parse_value(v)
        else:
            par[k] = parse_value(v)
    return par, cfg


def set_dotted(d, key, value):
    parts = key.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            raise KeyError(f"unknown parameter {key!r} (no {p!r})")
        cur = cur[p]
    if parts[-1] not in cur:
        raise KeyError(f"unknown parameter {key!r}")
    cur[parts[-1]] = value


def get_dotted(d, key):
    cur = d
    for p in key.split("."):
        cur = cur[p]
    return cur


def load_params(path=None, sets=None, quiet=False):
    """f4_params.json (without the _about notes) with the overrides applied.
    Returns (params, config_overrides). Unknown keys in sets are reported and skipped."""
    path = path or DEFAULT_PARAMS
    with open(path, encoding="utf-8") as fh:
        P = strip_about(json.load(fh))
    par, cfg = split_sets(sets)
    for k, v in par.items():
        try:
            set_dotted(P, k, float(v) if isinstance(v, (int, float)) else v)
        except KeyError as e:
            print(f"WARNING: {e}; override skipped", flush=True)
    P["_overrides"] = {**par, **{f"config.{k}": v for k, v in cfg.items()}}
    if not quiet:
        for w in check_params(P):
            print(f"WARNING (f4_params): {w}", flush=True)
    return P, cfg


def check_params(P):
    warn = []
    pc = P["segment_distribution"]["pi_commercial"]
    if not 0.0 <= pc <= 1.0:
        warn.append(f"pi_commercial = {pc} is not a probability")
    for seg in SEGMENTS:
        S = P[seg]
        mu = S["money_sensitivity_mu"]
        if not (mu["theta0"] >= mu["theta1"] > 0):
            warn.append(f"{seg}: money sensitivity should satisfy theta0 >= theta1 > 0")
        vot = S["value_of_time_Rs_per_min"]
        for a in ACTIVITIES:
            if not (vot[a]["theta1"] >= vot[a]["theta0"] >= 0):
                warn.append(f"{seg}: value of time {a} should satisfy theta1 >= theta0 >= 0")
        for end in ("theta0", "theta1"):
            if seg == "personal" and not vot["Q"][end] > vot["D"][end]:
                warn.append(f"{seg}: queueing should cost more than driving at {end}")
            if seg == "commercial" and not vot["Q"][end] >= vot["D"][end]:
                warn.append(f"{seg}: queueing should cost at least the downtime rate at {end}")
            if not vot["V"][end] <= vot["C"][end] + 1e-12:
                warn.append(f"{seg}: van time (V) should cost at most FCS charging time (C) at {end}")
            if seg == "commercial" and not (abs(vot["D"][end] - vot["C"][end]) < 1e-9
                                            and abs(vot["V"][end] - vot["C"][end]) < 1e-9):
                warn.append(f"commercial: D, C and V should all equal the downtime rate at {end} "
                            f"(D={vot['D'][end]}, C={vot['C'][end]}, V={vot['V'][end]})")
        if vot["V"]["theta1"] - vot["V"]["theta0"] > vot["C"]["theta1"] - vot["C"]["theta0"] + 1e-12:
            warn.append(f"{seg}: V should not rise faster with income than C")
        W = S["queue_tolerance_W_min"]
        if not (W["theta0"] >= W["theta1"] >= 0):
            warn.append(f"{seg}: queue tolerance should satisfy theta0 >= theta1 >= 0")
        d = S["theta_distribution"]
        if not (d["a"] > 0 and d["b"] > 0):
            warn.append(f"{seg}: Beta(a, b) needs a > 0 and b > 0")
    tar = P["tariff_Rs_per_kWh"]
    if not tar["p_M"] >= tar["p_F"]:
        warn.append("van tariff p_M should be at least the FCS tariff p_F")
    return warn


def line(ends, theta):
    return ends["theta0"] + (ends["theta1"] - ends["theta0"]) * np.asarray(theta, dtype=float)


def segment_index(segment):
    """'personal'/'commercial' (or P/C, 0/1) -> 0/1 array."""
    out = []
    for s in np.atleast_1d(segment):
        t = str(s).strip().lower()
        out.append(1 if t in ("commercial", "c", "1", "1.0") else 0)
    return np.array(out, dtype=int)


def sensitivities(theta, segment, P):
    """Per-EV coefficients of formulation 4. theta: array in [0, 1]; segment: 0/1 array (or words)."""
    theta = np.clip(np.asarray(theta, dtype=float), 0.0, 1.0)
    seg = segment_index(segment) if not np.issubdtype(np.asarray(segment).dtype, np.number) \
        else np.asarray(segment, dtype=int)
    out = {"theta": theta, "segment": seg}
    pick = {}
    for k, name in enumerate(SEGMENTS):
        S = P[name]
        vot = S["value_of_time_Rs_per_min"]
        pick[k] = {"mu": line(S["money_sensitivity_mu"], theta),
                   "W": np.maximum(0.0, line(S["queue_tolerance_W_min"], theta)),
                   **{"c" + a: np.maximum(0.0, line(vot[a], theta)) for a in ACTIVITIES}}
    for key in ("mu", "W", "cD", "cQ", "cV", "cC"):
        out[key] = np.where(seg == 1, pick[1][key], pick[0][key])
    return out


def public_params(P):
    """Parameters without internal keys (for meta.json)."""
    return {k: copy.deepcopy(v) for k, v in P.items()}
