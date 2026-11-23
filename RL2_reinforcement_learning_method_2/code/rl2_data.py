"""RL2 data: the EV-Solomon CSV dataset -> instance dicts.

The dict has the layout of solve_milp_f3.load_instance (plus the coordinates under "xy", which the
policy network reads), so check_plan_f3 / results_io score an RL plan exactly like a MILP plan.

Input (written by code/export_csv.py; one pair of files per source dataset)
  dataset_ev_solomon/csv/solomon_data.csv           + solomon_config.csv            (56 files, 100 EVs)
  dataset_ev_solomon/csv/gehring_homberger_data.csv + gehring_homberger_config.csv  (60 files, 200 EVs)
  dataset_ev_solomon/income/<FILE>.csv               theta per EV (formulation 3)
  code/f3_params.json                                income sensitivities and tariffs
data.csv   : instance, type (depot / station / ev), id, x, y, energy_kWh, request_min, range_km, guns, power_kW
config.csv : one row per file (T, speed, km per grid unit, detour, costs, vans, stations)

An instance = one file (column `instance`) and its first n EVs, the same as for the MILP.
Distances and times follow make_ev_solomon.read_ev_solomon: road km = detour x km_per_unit x
straight-line grid distance, minutes = 60 km / speed, charging minutes = 60 E / power.
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RL2_DIR = os.path.abspath(os.path.join(HERE, ".."))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))
PROJECT_CODE = os.path.join(PROJECT, "code")
if PROJECT_CODE not in sys.path:
    sys.path.insert(0, PROJECT_CODE)

import make_income as inc          # noqa: E402  (theta and its sensitivities, f3_params.json)
from solve_milp_f3 import peak_flags  # noqa: E402  (same peak / off-peak label as the MILP runs)

DEFAULT_DATASET = os.path.join(PROJECT, "dataset_ev_solomon")
DEFAULT_CSV = os.path.join(DEFAULT_DATASET, "csv")
DEFAULT_PARAMS_F3 = inc.DEFAULT_PARAMS
STEMS = {100: "solomon", 200: "gehring_homberger"}       # file size -> CSV stem = `source` in runs.csv


def config_scalars(row):
    """Scalar settings of one file (a row of *_config.csv) as numbers."""
    f = lambda k: float(row[k])       # noqa: E731
    return {"T": f("T_min"), "speed": f("speed_kmph"), "km_per_unit": f("km_per_unit"),
            "detour": f("detour"), "c_EV": f("c_EV_Rs_per_km"), "c_M": f("c_M_Rs_per_km"),
            "c_F": f("c_F_Rs_per_van"), "pi_F": f("pi_F_Rs_per_kWh"), "pi_M": f("pi_M_Rs_per_kWh"),
            "n_vans": int(f("n_vans")), "B": f("van_battery_kWh"), "Q": int(f("van_max_evs")),
            "P_M": f("van_power_kW"), "beta": f("van_use_kWh_per_km"), "eta": f("van_efficiency")}


def make_instance(file, n, source, family, cfg, dep_xy, fcs_xy, guns, station_power, ev_xy,
                  E, r, R, theta, P3, analysis=True, as_lists=True):
    """Instance dict (layout of solve_milp_f3.load_instance) from raw data.
    analysis=False skips the peak labels (only needed for the result tables);
    as_lists=False keeps numpy arrays (faster, used for generated training instances)."""
    dep_xy = np.asarray(dep_xy, dtype=float).reshape(2)
    fcs_xy = np.asarray(fcs_xy, dtype=float).reshape(-1, 2)
    ev_xy = np.asarray(ev_xy, dtype=float).reshape(-1, 2)
    E, r, R = (np.asarray(a, dtype=float) for a in (E, r, R))
    k = cfg["detour"] * cfg["km_per_unit"]
    d_ev_fcs = k * np.linalg.norm(ev_xy[:, None] - fcs_xy[None], axis=2)
    nodes = np.vstack([dep_xy, ev_xy])
    d_nodes = k * np.linalg.norm(nodes[:, None] - nodes[None], axis=2)
    tauF, tauM = 60 * E / station_power, 60 * E / cfg["P_M"]
    sens = inc.sensitivities(theta, P3)
    conv = (lambda a: np.asarray(a).tolist()) if as_lists else (lambda a: np.asarray(a))
    ev = {"E_kWh": conv(E), "r_min": conv(r), "R_km": conv(R), "tauF_min": conv(tauF),
          "tauM_min": conv(tauM), "theta": conv(sens["theta"]), "mu": conv(sens["mu"]),
          "cD": conv(sens["cD"]), "cQ": conv(sens["cQ"]), "cV": conv(sens["cV"]),
          "cC": conv(sens["cC"]), "W": conv(sens["W"])}
    window = 0.0
    if analysis:
        load, period, window = peak_flags(r, tauF)
        ev["load"], ev["period"] = load, period
    J = {"name": f"{file}_n{n}", "file": file, "family": family, "source": source,
         "n_ev": int(n), "n_fcs": len(fcs_xy), "n_vans": int(cfg["n_vans"]),
         "params": {"T_min": float(cfg["T"]), "guns_per_fcs": max(int(g) for g in guns),
                    "guns": [int(g) for g in guns], "eta": cfg["eta"],
                    "beta_kWh_per_km": cfg["beta"], "B_kWh": cfg["B"], "Q": int(cfg["Q"]),
                    "costs_Rs": {"c_EV": cfg["c_EV"], "c_M": cfg["c_M"], "c_F": cfg["c_F"],
                                 "pi_F": cfg["pi_F"], "pi_M": cfg["pi_M"]},
                    "tariff_Rs_per_kWh": {k_: float(v) for k_, v in P3["tariff_Rs_per_kWh"].items()
                                          if not k_.startswith("_")},
                    "peak_window_min": round(window, 2)},
         "ev": ev,
         "d_ev_fcs_km": conv(d_ev_fcs), "t_ev_fcs_min": conv(60 * d_ev_fcs / cfg["speed"]),
         "d_nodes_km": conv(d_nodes), "t_nodes_min": conv(60 * d_nodes / cfg["speed"]),
         "xy": {"depot": dep_xy.tolist(), "fcs": fcs_xy.tolist(), "ev": ev_xy.tolist()},
         "greedy_baseline": None, "greedy_s": ""}
    return J


class CsvDataset:
    """All files of the CSV dataset. instance(name, n) gives the instance dict of the first n EVs."""

    def __init__(self, csv_dir=DEFAULT_CSV, P3=None, stems=tuple(STEMS.values())):
        self.csv_dir = os.path.abspath(csv_dir)
        # income files live in <dataset>/income; make_income locates them from <dataset>/instances
        self.instances_dir = os.path.join(os.path.dirname(self.csv_dir), "instances")
        self.P3 = P3 if P3 is not None else inc.load_params(DEFAULT_PARAMS_F3)
        self.config, self.rows, self.source = {}, {}, {}
        self.missing = []
        for stem in stems:
            cpath = os.path.join(self.csv_dir, f"{stem}_config.csv")
            dpath = os.path.join(self.csv_dir, f"{stem}_data.csv")
            if not (os.path.exists(cpath) and os.path.exists(dpath)):
                self.missing.append(stem)
                continue
            with open(cpath, newline="") as fh:
                for row in csv.DictReader(fh):
                    self.config[row["instance"]] = row
                    self.source[row["instance"]] = stem
            with open(dpath, newline="") as fh:
                for row in csv.DictReader(fh):
                    self.rows.setdefault(row["instance"], []).append(row)

    def files(self):
        return list(self.config)

    def n_evs(self, name):
        return int(self.config[name]["n_evs"])

    def family(self, name):
        return self.config[name]["class"]

    def raw(self, name):
        """Depot, stations and all EVs of a file as arrays (used by the training generator)."""
        rows = self.rows[name]
        dep = next(r for r in rows if r["type"] == "depot")
        st = sorted((r for r in rows if r["type"] == "station"), key=lambda r: int(r["id"]))
        ev = sorted((r for r in rows if r["type"] == "ev"), key=lambda r: int(r["id"]))
        return {"depot": np.array([float(dep["x"]), float(dep["y"])]),
                "fcs": np.array([[float(r["x"]), float(r["y"])] for r in st]),
                "guns": [int(float(r["guns"])) for r in st],
                "station_power": float(st[0]["power_kW"]),
                "ev_xy": np.array([[float(r["x"]), float(r["y"])] for r in ev]),
                "E": np.array([float(r["energy_kWh"]) for r in ev]),
                "r": np.array([float(r["request_min"]) for r in ev]),
                "R": np.array([float(r["range_km"]) for r in ev])}

    def instance(self, name, n, analysis=True):
        cfg = config_scalars(self.config[name])
        raw = self.raw(name)
        if not 1 <= n <= len(raw["E"]):
            raise ValueError(f"{name} has {len(raw['E'])} EVs, asked for the first {n}")
        theta = inc.read_income(self.instances_dir, name, n, self.P3)
        return make_instance(name, n, self.source[name], self.family(name), cfg, raw["depot"],
                             raw["fcs"], raw["guns"], raw["station_power"], raw["ev_xy"][:n],
                             raw["E"][:n], raw["r"][:n], raw["R"][:n], theta, self.P3,
                             analysis=analysis, as_lists=True)
