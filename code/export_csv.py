"""Export the EV-Solomon text files to two CSV files per source dataset.

    python code/export_csv.py [dataset_dir]        (default: dataset_ev_solomon next to code/)

Writes into <dataset_dir>/csv/:
    solomon_data.csv              solomon_config.csv              (instances/100, 56 files)
    gehring_homberger_data.csv    gehring_homberger_config.csv    (instances/200, 60 files)

data.csv   = every location row of every file, one row each:
             instance, type (depot / station / ev), id, x, y, energy_kWh, request_min, range_km, guns, power_kW
             (energy/request/range only for EVs, guns/power only for stations)
config.csv = one row per file with every scalar setting (planning period T, speed, costs, vans, station counts).

Values are copied as written in the text files (no rounding or re-formatting).
"""
import csv
import glob
import os
import re
import sys

SECTIONS = ("PLANNING", "COSTS", "MOBILE CHARGING VANS", "FIXED CHARGING STATIONS", "EV REQUESTS")
PLANNING = ["T_min", "speed_kmph", "km_per_unit", "detour"]
COSTS = ["c_EV_Rs_per_km", "c_M_Rs_per_km", "c_F_Rs_per_van", "pi_F_Rs_per_kWh", "pi_M_Rs_per_kWh",
         "c_W_Rs_per_min", "kappa"]
VANS = ["n_vans", "van_battery_kWh", "van_max_evs", "van_power_kW", "van_use_kWh_per_km", "van_efficiency"]
CONFIG_COLS = (["instance", "source", "class", "n_evs"] + PLANNING + COSTS + VANS
               + ["n_stations", "guns_per_station", "total_guns", "station_power_kW"])
DATA_COLS = ["instance", "type", "id", "x", "y", "energy_kWh", "request_min", "range_km", "guns", "power_kW"]


def parse(path):
    """Return (name, scalars dict, station rows, ev rows) with the values as strings."""
    name, sec, header_next = None, None, False
    scal, stations, evs = {}, [], []
    for line in open(path):
        s = line.strip()
        if not s:
            continue
        if s in SECTIONS:
            sec, header_next = s, True
            continue
        if sec is None:
            name = s
            continue
        if header_next:                      # column-title line under each section name
            header_next = False
            continue
        vals = s.split()
        if sec == "PLANNING":
            scal.update(zip(PLANNING, vals))
        elif sec == "COSTS":
            scal.update(zip(COSTS, vals))
        elif sec == "MOBILE CHARGING VANS":
            scal.update(zip(VANS, vals))
        elif sec == "FIXED CHARGING STATIONS":
            stations.append(vals)
        else:
            evs.append(vals)
    return name, scal, stations, evs


def natural_key(path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", os.path.basename(path))]


def export(files, data_path, config_path):
    n_rows = 0
    with open(data_path, "w", newline="") as fd, open(config_path, "w", newline="") as fc:
        wd, wc = csv.writer(fd), csv.writer(fc)
        wd.writerow(DATA_COLS)
        wc.writerow(CONFIG_COLS)
        for path in files:
            name, scal, stations, evs = parse(path)
            depot, evs = evs[0], evs[1:]
            guns = {st[3] for st in stations}
            power = {st[4] for st in stations}
            assert len(power) == 1, f"{name}: stations have different power {power}"
            source = name[3:] if name.startswith("EV-") else name
            cfg = dict(scal, instance=name, source=source, n_evs=len(evs),
                       **{"class": re.match(r"[A-Z]+\d", source).group(0),
                          "n_stations": len(stations),
                          "guns_per_station": guns.pop() if len(guns) == 1 else "varies",
                          "total_guns": sum(int(st[3]) for st in stations),
                          "station_power_kW": power.pop()})
            wc.writerow([cfg[c] for c in CONFIG_COLS])
            wd.writerow([name, "depot", depot[0], depot[1], depot[2], "", "", "", "", ""])
            for st in stations:
                wd.writerow([name, "station", st[0], st[1], st[2], "", "", "", st[3], st[4]])
            for ev in evs:
                wd.writerow([name, "ev", ev[0], ev[1], ev[2], ev[3], ev[4], ev[5], "", ""])
            n_rows += 1 + len(stations) + len(evs)
    print(f"{os.path.basename(data_path)}: {n_rows} rows | {os.path.basename(config_path)}: {len(files)} rows")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "dataset_ev_solomon")
    out = os.path.join(root, "csv")
    os.makedirs(out, exist_ok=True)
    for sub, stem in (("100", "solomon"), ("200", "gehring_homberger")):
        files = sorted(glob.glob(os.path.join(root, "instances", sub, "*.txt")), key=natural_key)
        export(files, os.path.join(out, f"{stem}_data.csv"), os.path.join(out, f"{stem}_config.csv"))


if __name__ == "__main__":
    main()
