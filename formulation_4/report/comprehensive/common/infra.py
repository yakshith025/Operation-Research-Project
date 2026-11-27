"""Shared definitions for the three reports: demand pressure and infrastructure utilisation.

Demand pressure (one definition for every report)
  The code of formulation 4 (f4_data.peak_flags, called by f4_data.make_instance) gives, for EV i of an
  instance (a file and its first n EVs):
      window   = mean over the instance's EVs of the station charging time tauF_j = 60 E_j / 22 kW   (minutes)
      load_i   = number of EVs j of the instance (i itself included) with r_i - window <= r_j <= r_i
      pressure_i = load_i / (guns + vans) = load_i / (24 + 5)
  i.e. "how many requests arrived during one average charging time up to my request, per charger".
  It uses only the data (request times, energies, fleet), never a plan, so an EV has the same pressure in
  every method and every variant. Terciles: cut points q1, q2 = 1/3 and 2/3 quantiles of pressure over the
  free-choice EVs (at least one station in range) of ALL 560 baseline instances (56 Solomon files x
  n = 10..100); low <= q1 < medium <= q2 < high. pressure_table() returns them.

Infrastructure utilisation (from a plan.csv and the instance data)
  Guns (G = 24, P_F = 22 kW, available during [0, T]):
      busy_g = sum of the station charging minutes tauF of the EVs on gun g
      time utilisation  u_g = busy_g / T;   energy delivered = sum E;  energy capacity = P_F T / 60
      (constant power: energy utilisation = time utilisation)
  Station f: mean of its guns.  FCS system: sum busy / (G T).
  Vans (M = 5 in the baseline; B = 150 kWh, Q = 6 seats, P_M = 30 kW):
      fleet use       vans dispatched / M
      seats           stops / Q
      battery         kWh drawn / B   (drawn = delivered / eta + beta * km)
      charging time   sum tauM of its EVs / T;  driving time = 60 km / speed
      on the road     (return - departure) / T, departure = start of first stop - drive time depot -> first EV
      power capacity  kWh delivered / (P_M T / 60)
  System: total energy asked (sum E_i); share charged at stations / by vans; installed power
      24 x 22 + M x 30 kW over [0, T]; overall utilisation = energy delivered / (installed kW x T / 60).
  Busy profile: share of guns (vans) charging at time t, on 20 bins of t / T.

Aggregation convention (all three reports): a headline utilisation is the MEAN OF PER-INSTANCE (per-van)
SHARES, e.g. mean over plans of gun_util, mean over dispatched vans of idle_min / on_road_min. A pooled ratio
(sum of busy minutes / sum of G x T) weights long horizons far more (baseline MILP: 7.2 % pooled against
12.8 % mean for guns; van idle 59.7 % pooled against 38.2 % mean) and must be labelled "pooled" if shown.
"""
import os
import sys

import numpy as np
import pandas as pd

import report_common as rc

rc.f4_import()
import f4_data as fd  # noqa: E402
import f4_params as fp  # noqa: E402

_DS = {}


def dataset(cfg_over=None):
    key = tuple(sorted((cfg_over or {}).items()))
    if key not in _DS:
        P, _ = fp.load_params(quiet=True)
        _DS[key] = fd.Dataset(rc.F4_DATA, P, dict(cfg_over or {}), stems=("solomon",))
    return _DS[key]


# ------------------------------------------------------------------------------------------ pressure
def pressure_table(cache=os.path.join(rc.ROOT, "common", "pressure_labels.csv")):
    """Per EV of the 560 baseline instances: instance, n, id (1-based), stranded, load, window, pressure,
    period (f4 label: peak if load > instance median), terc (pooled terciles over free-choice EVs)."""
    if os.path.exists(cache):
        return pd.read_csv(cache)
    ds = dataset()
    rows = []
    for f in ds.files():
        for n in range(10, 101, 10):
            J = ds.instance(f, n)
            ev = J["ev"]
            reach = (np.asarray(J["d_ev_fcs_km"]) <= np.asarray(ev["R_km"])[:, None]).any(1)
            for i in range(n):
                rows.append((f, n, i + 1, not bool(reach[i]), ev["load"][i], J["params"]["peak_window_min"],
                             ev["pressure"][i], ev["period"][i]))
    df = pd.DataFrame(rows, columns=["instance", "n", "id", "stranded", "load", "window_min", "pressure", "period"])
    q1, q2 = df.loc[~df.stranded, "pressure"].quantile([1 / 3, 2 / 3])
    df["terc"] = np.where(df.pressure <= q1, "low", np.where(df.pressure <= q2, "medium", "high"))
    df.attrs["cuts"] = (q1, q2)
    df.to_csv(cache, index=False)
    return df


def pressure_cuts():
    df = pressure_table()
    q1, q2 = df.loc[~df.stranded.astype(bool), "pressure"].quantile([1 / 3, 2 / 3])
    return float(q1), float(q2)


# ------------------------------------------------------------------------------------------ utilisation
def utilisation(plan, cfg_over=None, bins=20):
    """plan: rows of plan.csv of ONE run_id and only valid instances (caller filters).
    Returns dict of DataFrames: guns, stations, vans, instances, profile."""
    ds = dataset(cfg_over)
    guns_rows, van_rows, inst_rows, prof_rows = [], [], [], []
    edges = np.linspace(0, 1, bins + 1)
    for (inst, n), grp in plan.groupby(["instance", "n"], sort=False):
        n = int(n)
        cfg = ds.cfg(inst)
        raw = ds.raw(inst)
        T, speed = cfg["T"], cfg["speed"]
        PF = float(np.asarray(raw["station_power"]).ravel()[0])
        guns = [int(g) for g in raw["guns"]]
        G, M = sum(guns), int(cfg["n_vans"])
        J = ds.instance(inst, n, analysis=False)
        tN = np.asarray(J["t_nodes_min"])
        ev = grp[grp.type == "ev"].copy()
        vans = grp[grp.type == "van"]
        fam = ds.family(inst)
        # guns
        gun_ev = ev[ev.resource_type == "FCS"]
        busy = gun_ev.groupby("resource").charge_min.sum()
        kwh = (gun_ev.charge_min * PF / 60).groupby(gun_ev.resource).sum()
        cnt = gun_ev.groupby("resource").size()
        for f, gf in enumerate(guns):
            for g in range(gf):
                name = f"FCS{f + 1}-gun{g + 1}"
                b = float(busy.get(name, 0.0))
                guns_rows.append((inst, n, fam, T, f + 1, g + 1, int(cnt.get(name, 0)), b, b / T,
                                  float(kwh.get(name, 0.0)), PF * T / 60))
        # vans
        van_ev = ev[ev.resource_type != "FCS"]
        for _, v in vans.iterrows():
            mine = van_ev[van_ev.resource == f"van{int(v.id)}"].sort_values("seq")
            if mine.empty:
                continue
            first = int(mine.iloc[0].id)
            depart = float(mine.iloc[0].start_min) - float(tN[0, first])
            charge = float(mine.charge_min.sum())
            drive = 60.0 * float(v.km) / speed
            on_road = float(v.return_min) - depart
            van_rows.append((inst, n, fam, T, int(v.id), int(v.stops), int(cfg["Q"]), float(v.kWh_delivered),
                             float(v.kWh_drawn), float(cfg["B"]), float(v.km), charge, drive, on_road,
                             max(0.0, on_road - charge - drive), depart, float(v.return_min),
                             float(v.kWh_delivered) / (cfg["P_M"] * T / 60)))
        # instance level
        E_tot = float((ev.charge_min * np.where(ev.resource_type == "FCS", PF, cfg["P_M"]) / 60).sum())
        E_fcs = float((gun_ev.charge_min * PF / 60).sum())
        cap_kw = G * PF + M * cfg["P_M"]
        nv = int(len(vans))
        inst_rows.append((inst, n, fam, T, G, M, nv, E_tot, E_fcs, E_tot - E_fcs,
                          float(busy.sum()) / (G * T), int((cnt.reindex([f"FCS{f + 1}-gun{g + 1}" for f, gf in enumerate(guns) for g in range(gf)]).fillna(0) == 0).sum()),
                          E_tot / (cap_kw * T / 60), nv / M if M else np.nan))
        # busy profile (share of guns / vans charging), sampled at bin midpoints of t/T
        mids = (edges[:-1] + edges[1:]) / 2 * T
        s, e = gun_ev.start_min.to_numpy(), gun_ev.end_min.to_numpy()
        gb = ((s[None, :] <= mids[:, None]) & (e[None, :] > mids[:, None])).sum(1) / G
        s2, e2 = van_ev.start_min.to_numpy(), van_ev.end_min.to_numpy()
        vb = ((s2[None, :] <= mids[:, None]) & (e2[None, :] > mids[:, None])).sum(1) / max(M, 1)
        for b_, (x, y) in enumerate(zip(gb, vb)):
            prof_rows.append((inst, n, fam, b_, (edges[b_] + edges[b_ + 1]) / 2, x, y))
    out = {
        "guns": pd.DataFrame(guns_rows, columns=["instance", "n", "family", "T", "station", "gun", "evs", "busy_min",
                                                 "util", "kWh", "kWh_capacity"]),
        "vans": pd.DataFrame(van_rows, columns=["instance", "n", "family", "T", "van", "stops", "Q", "kWh_delivered",
                                                "kWh_drawn", "B", "km", "charge_min", "drive_min", "on_road_min",
                                                "idle_min", "depart_min", "return_min", "power_util"]),
        "instances": pd.DataFrame(inst_rows, columns=["instance", "n", "family", "T", "G", "M", "vans_used",
                                                      "kWh_total", "kWh_fcs", "kWh_van", "gun_util", "idle_guns",
                                                      "system_util", "fleet_use"]),
        "profile": pd.DataFrame(prof_rows, columns=["instance", "n", "family", "bin", "t_rel", "guns_busy",
                                                    "vans_busy"]),
    }
    v = out["vans"]
    if len(v):
        v["seat_util"] = v.stops / v.Q
        v["battery_util"] = v.kWh_drawn / v.B
        v["on_road_util"] = v.on_road_min / v["T"]
        v["charge_util"] = v.charge_min / v["T"]
    g = out["guns"]
    out["stations"] = g.groupby(["instance", "n", "family", "station"], as_index=False).agg(
        guns=("gun", "size"), evs=("evs", "sum"), util=("util", "mean"), kWh=("kWh", "sum"),
        kWh_capacity=("kWh_capacity", "sum"))
    return out
