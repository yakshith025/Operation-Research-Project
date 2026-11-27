"""Formulation-4 dataset (data/*.csv) -> instance dicts used by the MILP, the RL simulator, the checker
and the result tables.

data/ holds the complete dataset of formulation 4, two files per source (data/make_f4_dataset.py):
    solomon_config.csv, solomon_data.csv                       56 files of 100 EVs (12 stations, 5 vans)
    gehring_homberger_config.csv, gehring_homberger_data.csv   60 files of 200 EVs (24 stations, 12 vans)
data.csv   : instance, type (depot / station / ev), id, x, y, energy_kWh, request_min, range_km, guns,
             power_kW, segment (personal / commercial), theta     (segment and theta on EV rows only)
config.csv : one row per file (T, speed, km per grid unit, detour, costs, vans, battery, seats, stations)

An instance = one file (column `instance`) and its first n EVs (Solomon's convention).
Distances: road km = detour x km_per_unit x straight-line grid distance; minutes = 60 km / speed;
charging minutes = 60 E / power (as in make_ev_solomon of formulation 3, so the network is unchanged).
Config overrides (experiments, e.g. n_vans = 3) are applied to every file when the dataset is read;
nothing is written to disk.
"""
import csv
import os

import numpy as np

import f4_params as fp

HERE = os.path.dirname(os.path.abspath(__file__))
F4_DIR = os.path.dirname(HERE)
DEFAULT_DATA = os.path.join(F4_DIR, "data")
STEMS = {100: "solomon", 200: "gehring_homberger"}       # file size -> CSV stem = `source` in runs.csv


def config_scalars(row):
    """Scalar settings of one file (a row of *_config.csv) as numbers."""
    f = lambda k: float(row[k])       # noqa: E731
    return {"T": f("T_min"), "speed": f("speed_kmph"), "km_per_unit": f("km_per_unit"),
            "detour": f("detour"), "c_EV": f("c_EV_Rs_per_km"), "c_M": f("c_M_Rs_per_km"),
            "c_F": f("c_F_Rs_per_van"), "pi_F": f("pi_F_Rs_per_kWh"), "pi_M": f("pi_M_Rs_per_kWh"),
            "n_vans": int(round(f("n_vans"))), "B": f("van_battery_kWh"), "Q": int(round(f("van_max_evs"))),
            "P_M": f("van_power_kW"), "beta": f("van_use_kWh_per_km"), "eta": f("van_efficiency")}


def peak_flags(r, tauF, n_chargers):
    """Analysis only (not used by the model). load_i = requests released in [r_i - window, r_i], window =
    mean FCS charging time of the instance; period = peak if load_i > median load; pressure_i =
    load_i / number of chargers (guns + vans)."""
    r = np.asarray(r, dtype=float)
    window = float(np.mean(tauF)) if len(tauF) else 0.0
    load = np.array([int(np.sum((r >= ri - window) & (r <= ri))) for ri in r])
    med = float(np.median(load)) if len(load) else 0.0
    period = ["peak" if l_ > med else "off-peak" for l_ in load]
    pressure = (load / max(1, n_chargers)).round(4)
    return load.tolist(), period, pressure.tolist(), window


def make_instance(file, n, source, family, cfg, dep_xy, fcs_xy, guns, station_power, ev_xy,
                  E, r, R, theta, segment, P, analysis=True, as_lists=True):
    """Instance dict (the layout of formulation 3's load_instance, plus ev['segment'] and 'xy').
    analysis=False skips the peak labels; as_lists=False keeps numpy arrays (RL training)."""
    dep_xy = np.asarray(dep_xy, dtype=float).reshape(2)
    fcs_xy = np.asarray(fcs_xy, dtype=float).reshape(-1, 2)
    ev_xy = np.asarray(ev_xy, dtype=float).reshape(-1, 2)
    E, r, R = (np.asarray(a, dtype=float) for a in (E, r, R))
    k = cfg["detour"] * cfg["km_per_unit"]
    d_ev_fcs = k * np.linalg.norm(ev_xy[:, None] - fcs_xy[None], axis=2)
    nodes = np.vstack([dep_xy, ev_xy])
    d_nodes = k * np.linalg.norm(nodes[:, None] - nodes[None], axis=2)
    tauF, tauM = 60 * E / station_power, 60 * E / cfg["P_M"]
    sens = fp.sensitivities(theta, segment, P)
    conv = (lambda a: np.asarray(a).tolist()) if as_lists else (lambda a: np.asarray(a))
    ev = {"E_kWh": conv(E), "r_min": conv(r), "R_km": conv(R), "tauF_min": conv(tauF),
          "tauM_min": conv(tauM), "theta": conv(sens["theta"]), "segment": conv(sens["segment"]),
          "mu": conv(sens["mu"]), "cD": conv(sens["cD"]), "cQ": conv(sens["cQ"]), "cV": conv(sens["cV"]),
          "cC": conv(sens["cC"]), "W": conv(sens["W"])}
    window = 0.0
    if analysis:
        load, period, pressure, window = peak_flags(r, tauF, int(sum(guns)) + int(cfg["n_vans"]))
        ev["load"], ev["period"], ev["pressure"] = load, period, pressure
    J = {"name": f"{file}_n{n}", "file": file, "family": family, "source": source,
         "n_ev": int(n), "n_fcs": len(fcs_xy), "n_vans": int(cfg["n_vans"]),
         "params": {"T_min": float(cfg["T"]), "guns_per_fcs": max(int(g) for g in guns),
                    "guns": [int(g) for g in guns], "eta": cfg["eta"],
                    "beta_kWh_per_km": cfg["beta"], "B_kWh": cfg["B"], "Q": int(cfg["Q"]),
                    "costs_Rs": {"c_EV": cfg["c_EV"], "c_M": cfg["c_M"], "c_F": cfg["c_F"],
                                 "pi_F": cfg["pi_F"], "pi_M": cfg["pi_M"]},
                    "tariff_Rs_per_kWh": {k_: float(v) for k_, v in P["tariff_Rs_per_kWh"].items()
                                          if not str(k_).startswith("_")},
                    "peak_window_min": round(window, 2)},
         "ev": ev,
         "d_ev_fcs_km": conv(d_ev_fcs), "t_ev_fcs_min": conv(60 * d_ev_fcs / cfg["speed"]),
         "d_nodes_km": conv(d_nodes), "t_nodes_min": conv(60 * d_nodes / cfg["speed"]),
         "xy": {"depot": dep_xy.tolist(), "fcs": fcs_xy.tolist(), "ev": ev_xy.tolist()},
         "greedy_baseline": None, "greedy_s": ""}
    return J


class Dataset:
    """Every file of the formulation-4 dataset. instance(name, n) gives the instance dict of the first n EVs.
    P = parameters (f4_params.load_params), cfg_over = config overrides {column: value} for every file."""

    def __init__(self, data_dir=DEFAULT_DATA, P=None, cfg_over=None, stems=tuple(STEMS.values())):
        self.path = os.path.abspath(data_dir)
        self.P = P if P is not None else fp.load_params(quiet=True)[0]
        self.cfg_over = dict(cfg_over or {})
        self.config, self.rows, self.source = {}, {}, {}
        self.missing, self.problems = [], []
        for stem in stems:
            cpath = os.path.join(self.path, f"{stem}_config.csv")
            dpath = os.path.join(self.path, f"{stem}_data.csv")
            if not (os.path.exists(cpath) and os.path.exists(dpath)):
                self.missing.append(stem)
                continue
            with open(cpath, newline="") as fh:
                for row in csv.DictReader(fh):
                    for col, val in self.cfg_over.items():
                        row[col] = str(val)
                    self.config[row["instance"]] = row
                    self.source[row["instance"]] = stem
            with open(dpath, newline="") as fh:
                for row in csv.DictReader(fh):
                    self.rows.setdefault(row["instance"], []).append(row)
        for name in list(self.config):
            if name not in self.rows:
                self.problems.append(f"{name} has a config row but no data rows; skipped")
                del self.config[name]

    # ------------------------------------------------------------------ listing
    def files(self):
        return list(self.config)

    def names(self):
        return self.files()

    def n_evs(self, name):
        return int(float(self.config[name]["n_evs"]))

    def family(self, name):
        return self.config[name]["class"]

    def cfg(self, name):
        return config_scalars(self.config[name])

    # ------------------------------------------------------------------ one file
    def raw(self, name):
        """Depot, stations and all EVs of a file as arrays (also used by the RL training generator)."""
        rows = self.rows[name]
        dep = [r for r in rows if r["type"] == "depot"]
        if len(dep) != 1:
            raise ValueError(f"{name}: {len(dep)} depot rows (need 1)")
        st = sorted((r for r in rows if r["type"] == "station"), key=lambda r: int(r["id"]))
        ev = sorted((r for r in rows if r["type"] == "ev"), key=lambda r: int(r["id"]))
        if not st or not ev:
            raise ValueError(f"{name}: no station or no EV rows")
        if any((r.get("segment") or "") == "" or (r.get("theta") or "") == "" for r in ev):
            raise ValueError(f"{name}: EV rows without segment / theta; run data/make_f4_dataset.py")
        guns = [int(float(r["guns"])) for r in st]
        if "guns_per_station" in self.cfg_over:                    # experiment E9: guns per station (station capacity)
            guns = [int(self.cfg_over["guns_per_station"])] * len(st)
        station_power = float(st[0]["power_kW"])
        if "station_power_kW" in self.cfg_over:                    # experiment E9: FCS charging power (station capacity)
            station_power = float(self.cfg_over["station_power_kW"])
        E = np.array([float(r["energy_kWh"]) for r in ev])
        if "energy_scale" in self.cfg_over:                        # experiment E10: charging demand per EV (energy x factor)
            E = np.round(E * float(self.cfg_over["energy_scale"]), 4)
        return {"depot": np.array([float(dep[0]["x"]), float(dep[0]["y"])]),
                "fcs": np.array([[float(r["x"]), float(r["y"])] for r in st]),
                "guns": guns, "station_power": station_power,
                "ev_xy": np.array([[float(r["x"]), float(r["y"])] for r in ev]),
                "E": E,
                "r": np.array([float(r["request_min"]) for r in ev]),
                "R": np.array([float(r["range_km"]) for r in ev]),
                "theta": np.array([float(r["theta"]) for r in ev]),
                "segment": fp.segment_index([r["segment"] for r in ev])}

    def instance(self, name, n, analysis=True):
        cfg = self.cfg(name)
        raw = self.raw(name)
        if not 1 <= n <= len(raw["E"]):
            raise ValueError(f"{name} has {len(raw['E'])} EVs, asked for the first {n}")
        return make_instance(name, n, self.source[name], self.family(name), cfg, raw["depot"], raw["fcs"],
                             raw["guns"], raw["station_power"], raw["ev_xy"][:n], raw["E"][:n], raw["r"][:n],
                             raw["R"][:n], raw["theta"][:n], raw["segment"][:n], self.P,
                             analysis=analysis, as_lists=True)


def reachable(J, i):
    return [f for f in range(J["n_fcs"]) if J["d_ev_fcs_km"][i][f] <= J["ev"]["R_km"][i]]
