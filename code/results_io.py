"""Output schema shared by every method (MILP, greedy, RL, ...) for main_formulation_3.

A results folder holds four things, whatever the number of methods or instances:

  runs.csv    one row per run = (run_id, method, seed, instance, n): status, validity, Z1-Z6,
              service numbers, compute time, and solver-only columns (bound, gap, nodes, ...)
              that stay blank for methods without them.
  plan.csv    one row per decision, with a `type` column like the input data CSV:
                type = ev   EV <id>: resource (FCS<f>-gun<g> or van<k>), seq (position on the gun /
                            in the van route), start and end minute, and its outcomes
                            (driving, queue, van wait, charging, money paid, perceived costs)
                type = van  van <k>: stops, km, energy delivered / drawn, battery left, return minute
              Every row carries the run key (run_id, method, seed, instance, n).
  meta.json   settings of every run_id (time limit, threads, parameters, start / end time, ...)
  logs.zip    solver and console logs (logs/ while a run is going, packed at the end)

Rules
- instance = EV-Solomon file name (e.g. EV-C101) and n = number of EVs (its first n), the same keys
  as dataset_ev_solomon/csv/*_data.csv, so results join the input on (instance, id).
- A method only produces a plan; evaluate() scores every plan with the same checker
  (check_plan_f3.validate), so Z and every metric mean the same for MILP, greedy and RL.
- Invalid plans are kept (valid = no, violations = count, violation_detail = first three).
- Inputs (theta, E, r, range, ...) are not copied into the output; ev_attributes(J) rebuilds them.

How a method writes its results: build one run row with run_row(...) and its plan rows with
evaluate(...); a worker process writes them with write_part(); the parent process appends them
to the two tables with merge_parts(). A part file stays in _parts/ until it is merged (for example
while plan.csv is open in Excel), so nothing is lost and the next merge picks it up.
"""
import csv
import glob
import json
import os
import re
import time
import zipfile

import numpy as np

from check_plan_f3 import validate

KEY = ["run_id", "method", "seed", "instance", "n"]

RUN_COLUMNS = [
    # keys
    "run_id", "method", "seed", "instance", "source", "class", "n",
    # outcome
    "status", "valid", "violations", "violation_detail",
    # cost (same checker for every method)
    "Z_total", "Z1_EV_drive_perceived", "Z2_van_drive", "Z3_dispatch", "Z4_operator_energy",
    "Z5_tariff_perceived", "Z6_time_perceived", "operator_cost_Z2_to_Z4", "customer_cost_Z1_Z5_Z6",
    # service
    "EVs_at_FCS", "EVs_by_van", "stranded_EVs", "free_choice_EVs", "free_choice_by_van",
    "share_free_choice_by_van_%", "share_van_peak_%", "share_van_offpeak_%",
    "mean_theta_all", "mean_theta_FCS", "mean_theta_van_free_choice",
    "vans_used", "van_km", "mean_queue_FCS_min", "max_queue_FCS_min", "mean_van_wait_min",
    "mean_time_to_charged_min", "max_time_to_charged_min",
    # compute
    "solve_s", "train_s", "wall_clock_s",
    # solver only (blank for heuristics / RL)
    "best_bound", "gap_%", "solver_obj", "nodes", "binaries", "continuous", "constraints",
    "build_s", "warm_start", "gpu_root_LP_bound", "gpu_LP_s",
    # free text (error message, greedy rule, ...)
    "message"]

PLAN_COLUMNS = [
    "run_id", "method", "seed", "instance", "n", "type", "id",
    # type = ev
    "resource_type", "resource", "seq", "start_min", "end_min",
    "drive_km", "drive_min", "queue_min", "van_wait_min", "charge_min", "time_to_charged_min",
    "pay_Rs", "perceived_money_Rs", "time_cost_Rs", "customer_cost_Rs", "operator_energy_Rs",
    # type = van
    "stops", "km", "kWh_delivered", "kWh_drawn", "battery_left_kWh", "return_min"]

SOURCE_OF_DIR = {"100": "solomon", "200": "gehring_homberger"}


# ====================================================================== keys
def _int_str(v):
    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        return str(v)


def key_of(row):
    """(run_id, method, seed, instance, n) as strings, the same for dict rows read from CSV and
    rows built in memory."""
    return (str(row.get("run_id", "")), str(row.get("method", "")), _int_str(row.get("seed", "")),
            str(row.get("instance", "")), _int_str(row.get("n", "")))


def part_name(key):
    run_id, method, seed, instance, n = key
    raw = f"{run_id}__{method}__s{seed}__{instance}_n{n}"
    return re.sub(r"[^A-Za-z0-9_.-]", "-", raw)


def log_stem(run_id, instance, n):
    return re.sub(r"[^A-Za-z0-9_.-]", "-", f"{run_id}__{instance}_n{n}")


# ====================================================================== scoring
def _mean(vals, digits):
    vals = [v for v in vals if v != ""]
    return round(float(np.mean(vals)), digits) if vals else ""


def reachable(J, i):
    return [f for f in range(J["n_fcs"]) if J["d_ev_fcs_km"][i][f] <= J["ev"]["R_km"][i]]


def evaluate(J, plan):
    """Score a plan with the shared checker.

    plan = {resource: ["FCS<f>-gun<g>" | "van<k>"] per EV, S, C: start / end minute per EV,
            van_routes: [[0, node, ..., 0], ...], route_van: van number of each route}
    Returns (metrics, plan_rows, violations): metrics fills the outcome / cost / service columns
    of runs.csv, plan_rows are the plan.csv rows without the run key (add it with with_key)."""
    err, Z, parts, detail = validate(J, plan)
    p, ev = J["params"], J["ev"]
    n, T = J["n_ev"], p["T_min"]
    dN, tN = J["d_nodes_km"], J["t_nodes_min"]
    seq = {}
    by_gun = {}
    for i in range(n):
        if plan["resource"][i].startswith("FCS"):
            by_gun.setdefault(plan["resource"][i], []).append(i)
    for evs in by_gun.values():
        for k, i in enumerate(sorted(evs, key=lambda q: (plan["S"][q], q)), start=1):
            seq[i] = k
    for route in plan["van_routes"]:
        for k, node in enumerate(route[1:-1], start=1):
            seq[node - 1] = k
    ev_rows = []
    for i in range(n):
        d = detail[i]
        ev_rows.append({
            "type": "ev", "id": i + 1, "resource_type": d["kind"], "resource": plan["resource"][i],
            "seq": seq.get(i, ""), "start_min": float(plan["S"][i]),    # full precision: the plan
            "end_min": float(plan["C"][i]),                                 # re-scores to the same Z
            "drive_km": round(d["drive_km"], 3), "drive_min": round(d["drive_min"], 2),
            "queue_min": round(d["queue_min"], 2), "van_wait_min": round(d["van_wait_min"], 2),
            "charge_min": round(d["charge_min"], 2),
            "time_to_charged_min": round(plan["C"][i] - ev["r_min"][i], 2),
            "pay_Rs": round(d["pay_Rs"], 2), "perceived_money_Rs": round(d["perceived_money_Rs"], 2),
            "time_cost_Rs": round(d["time_cost_Rs"], 2),
            "customer_cost_Rs": round(d["customer_cost_Rs"], 2),
            "operator_energy_Rs": round(d["operator_energy_Rs"], 2)})
    van_rows = []
    for route, k in zip(plan["van_routes"], plan["route_van"]):
        stops = route[1:-1]
        km = sum(dN[a][b] for a, b in zip(route, route[1:]))
        delivered = sum(ev["E_kWh"][s - 1] for s in stops)
        drawn = delivered / p["eta"] + p["beta_kWh_per_km"] * km
        back = plan["C"][stops[-1] - 1] + tN[stops[-1]][0] if stops else 0.0
        van_rows.append({"type": "van", "id": k, "stops": len(stops), "km": round(km, 2),
                         "kWh_delivered": round(delivered, 2), "kWh_drawn": round(drawn, 2),
                         "battery_left_kWh": round(p["B_kWh"] - drawn, 2),
                         "return_min": round(back, 2)})

    free = [i for i in range(n) if reachable(J, i)]
    fcs = [e for e in ev_rows if e["resource_type"] == "FCS"]
    mcs = [e for e in ev_rows if e["resource_type"] == "MCS"]
    free_mcs = [i for i in free if ev_rows[i]["resource_type"] == "MCS"]
    theta = ev["theta"]
    metrics = {
        "valid": "yes" if not err else "no", "violations": len(err),
        "violation_detail": "; ".join(err[:3]),
        "Z_total": round(Z, 2), "Z_exact": Z,     # Z_exact: unrounded, for gaps (not a column)
        "Z1_EV_drive_perceived": round(parts["Z1"], 2), "Z2_van_drive": round(parts["Z2"], 2),
        "Z3_dispatch": round(parts["Z3"], 2), "Z4_operator_energy": round(parts["Z4"], 2),
        "Z5_tariff_perceived": round(parts["Z5"], 2), "Z6_time_perceived": round(parts["Z6"], 2),
        "operator_cost_Z2_to_Z4": round(parts["Z2"] + parts["Z3"] + parts["Z4"], 2),
        "customer_cost_Z1_Z5_Z6": round(parts["Z1"] + parts["Z5"] + parts["Z6"], 2),
        "EVs_at_FCS": len(fcs), "EVs_by_van": len(mcs), "stranded_EVs": n - len(free),
        "free_choice_EVs": len(free), "free_choice_by_van": len(free_mcs),
        "share_free_choice_by_van_%": round(100 * len(free_mcs) / len(free), 1) if free else "",
        "mean_theta_all": _mean([round(t, 4) for t in theta[:n]], 3),
        "mean_theta_FCS": _mean([round(theta[e["id"] - 1], 4) for e in fcs], 3),
        "mean_theta_van_free_choice": _mean([round(theta[i], 4) for i in free_mcs], 3),
        "vans_used": len(van_rows), "van_km": round(sum(v["km"] for v in van_rows), 2),
        "mean_queue_FCS_min": _mean([e["queue_min"] for e in fcs], 2),
        "max_queue_FCS_min": max([e["queue_min"] for e in fcs], default=""),
        "mean_van_wait_min": _mean([e["van_wait_min"] for e in mcs], 2),
        "mean_time_to_charged_min": _mean([e["time_to_charged_min"] for e in ev_rows], 1),
        "max_time_to_charged_min": max([e["time_to_charged_min"] for e in ev_rows], default="")}
    for per in ("peak", "off-peak"):
        grp = [i for i in free if ev["period"][i] == per]
        metrics[f"share_van_{per.replace('-', '')}_%"] = (
            round(100 * sum(ev_rows[i]["resource_type"] == "MCS" for i in grp) / len(grp), 1)
            if grp else "")
    return metrics, ev_rows + van_rows, err


def ev_attributes(J):
    """Per-EV inputs and derived attributes (income coefficients, stranded, nearest FCS, period)
    for the analysis. Not stored in plan.csv: rebuilt from the dataset when needed."""
    ev, dEF = J["ev"], J["d_ev_fcs_km"]
    out = []
    for i in range(J["n_ev"]):
        reach = reachable(J, i)
        out.append({
            "id": i + 1, "theta": round(ev["theta"][i], 4), "mu": round(ev["mu"][i], 4),
            "c_D": round(ev["cD"][i], 3), "c_Q": round(ev["cQ"][i], 3),
            "c_V": round(ev["cV"][i], 3), "c_C": round(ev["cC"][i], 3),
            "W_min": round(ev["W"][i], 2), "reachable_fcs": len(reach),
            "stranded": "yes" if not reach else "no",
            "nearest_reachable_fcs_km": round(min(dEF[i][f] for f in reach), 3) if reach else "",
            "E_kWh": round(ev["E_kWh"][i], 2), "r_min": round(ev["r_min"][i], 1),
            "load_at_request": ev["load"][i], "period": ev["period"][i]})
    return out


def plan_from_rows(rows, n):
    """Rebuild a plan dict (input of evaluate / validate) from the plan.csv rows of one run."""
    resource, S, C = [""] * n, [0.0] * n, [0.0] * n
    vans = {}
    for r in rows:
        if r.get("type") == "van":
            vans.setdefault(int(float(r["id"])), [])
    for r in rows:
        if r.get("type") != "ev":
            continue
        i = int(float(r["id"])) - 1
        if not 0 <= i < n:
            continue
        resource[i] = r["resource"]
        S[i], C[i] = float(r["start_min"]), float(r["end_min"])
        if str(r["resource"]).startswith("van"):
            vans.setdefault(int(r["resource"][3:]), []).append((float(r["seq"] or 0), i + 1))
    route_van = sorted(vans)
    routes = [[0] + [node for _, node in sorted(vans[k])] + [0] for k in route_van]
    return {"resource": resource, "S": S, "C": C, "van_routes": routes, "route_van": route_van}


# ====================================================================== rows
def run_row(keys, status, metrics=None, **extra):
    """A runs.csv row with every column (blank where nothing is known). Unknown fields are
    dropped so the table keeps one fixed schema; put method-specific settings in meta.json."""
    row = {c: "" for c in RUN_COLUMNS}
    for src in (keys, {"status": status}, metrics or {}, extra):
        row.update({k: v for k, v in src.items() if k in row and v is not None})
    return row


def with_key(keys, plan_rows):
    k = {c: keys[c] for c in KEY}
    return [{**k, **r} for r in plan_rows]


# ====================================================================== files
def paths(out):
    return {"runs": os.path.join(out, "runs.csv"), "plan": os.path.join(out, "plan.csv"),
            "parts": os.path.join(out, "_parts"), "logs": os.path.join(out, "logs"),
            "meta": os.path.join(out, "meta.json"), "zip": os.path.join(out, "logs.zip")}


def read_runs(out):
    path = paths(out)["runs"]
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _header(path):
    try:
        with open(path, newline="") as fh:
            return next(csv.reader(fh), None)
    except OSError:
        return None


def iter_plan(out, keys=None, types=None):
    """Yield plan.csv rows (dicts), optionally only for the given run keys / row types."""
    path = paths(out)["plan"]
    if not os.path.exists(path):
        return
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if types is not None and r.get("type") not in types:
                continue
            if keys is not None and key_of(r) not in keys:
                continue
            yield r


def _replace(tmp, path, tries=10):
    """os.replace with retries (on Windows the target can be locked by Excel or a scanner)."""
    for k in range(tries):
        try:
            os.replace(tmp, path)
            return True
        except OSError:
            time.sleep(0.5 * (k + 1))
    return False


def write_table(path, columns, rows):
    """Write a whole table via a temporary file. Returns False (and warns) if the file stays
    locked; never raises."""
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore", restval="")
            wr.writeheader()
            wr.writerows(rows)
    except OSError as e:
        print(f"WARNING: could not write {tmp}: {e}", flush=True)
        return False
    if _replace(tmp, path):
        return True
    print(f"WARNING: {path} is locked (open in Excel?); the new version is in {tmp}", flush=True)
    return False


def rewrite_plan(out, keep, extra_rows=()):
    """Stream plan.csv into a new file: keep the rows whose run key passes keep(key) (the last copy
    of duplicated rows), then add extra_rows. Memory stays small for any table size."""
    path = paths(out)["plan"]
    last = {}
    if os.path.exists(path):
        with open(path, newline="") as fh:
            for k, r in enumerate(csv.DictReader(fh)):
                last[(key_of(r), r.get("type"), r.get("id"))] = k
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", newline="") as fo:
            wr = csv.DictWriter(fo, fieldnames=PLAN_COLUMNS, extrasaction="ignore", restval="")
            wr.writeheader()
            if os.path.exists(path):
                with open(path, newline="") as fh:
                    for k, r in enumerate(csv.DictReader(fh)):
                        kk = key_of(r)
                        if last.get((kk, r.get("type"), r.get("id"))) == k and keep(kk):
                            wr.writerow(r)
            wr.writerows(extra_rows)
    except OSError as e:
        print(f"WARNING: could not write {tmp}: {e}", flush=True)
        return False
    if _replace(tmp, path):
        return True
    print(f"WARNING: {path} is locked (open in Excel?); the new version is in {tmp}", flush=True)
    return False


def _append(path, columns, rows):
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore", restval="")
        if new:
            wr.writeheader()
        wr.writerows(rows)


def write_part(out, runs, plan):
    """Results of one task (any number of runs, e.g. MILP + greedy) as one JSON file in _parts/.
    Written by the process that solved the task; merged into the tables by the parent."""
    d = paths(out)["parts"]
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, part_name(key_of(runs[0])) + ".json")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"runs": runs, "plan": plan}, fh)
    os.replace(tmp, path)
    return path


def merge_parts(out, only=None):
    """Append waiting part files to plan.csv and runs.csv (plan first, so a run row only exists
    once its plan is stored) and delete them. Returns the merged run rows. A part that cannot be
    appended (file locked) stays for the next call."""
    d = paths(out)["parts"]
    files = sorted(glob.glob(os.path.join(d, "*.json")), key=os.path.getmtime)
    if only is not None:
        files = [f for f in files if os.path.abspath(f) in only]
    merged = []
    for f in files:
        try:
            with open(f) as fh:
                part = json.load(fh)
        except (OSError, ValueError):
            continue                      # half-written by a killed worker: ignored, never merged
        try:
            if part.get("plan"):
                _append(paths(out)["plan"], PLAN_COLUMNS, part["plan"])
            _append(paths(out)["runs"], RUN_COLUMNS, part["runs"])
        except OSError as e:
            print(f"WARNING: results not appended yet ({e}); kept in {f}, merged later", flush=True)
            break
        merged += part["runs"]
        try:
            os.remove(f)
        except OSError:
            pass
    try:
        if os.path.isdir(d) and not os.listdir(d):
            os.rmdir(d)
    except OSError:
        pass
    return merged


def prepare(out, drop=None):
    """Bring runs.csv / plan.csv to the current schema before a run: merge leftover parts, keep
    the last copy of any duplicated run, drop runs for which drop(key, row) is true (they are
    solved again) and plan rows without a run row. Returns the runs that stay."""
    os.makedirs(out, exist_ok=True)
    merge_parts(out)
    P = paths(out)
    runs = read_runs(out)
    by_key = {}
    for r in runs:
        by_key[key_of(r)] = r
    kept = {k: r for k, r in by_key.items() if not (drop and drop(k, r))}
    plan_header = _header(P["plan"])
    if (len(kept) != len(runs) or (runs and _header(P["runs"]) != RUN_COLUMNS)
            or (plan_header is not None and plan_header != PLAN_COLUMNS) or _plan_has_orphans(out, kept)):
        if runs or os.path.exists(P["runs"]):
            write_table(P["runs"], RUN_COLUMNS, list(kept.values()))
        if plan_header is not None:
            rewrite_plan(out, lambda k: k in kept)
    return list(kept.values())


def _plan_has_orphans(out, kept):
    seen = set()
    for r in iter_plan(out):
        k = key_of(r)
        if k not in kept:
            return True
        t = (k, r.get("type"), r.get("id"))
        if t in seen:
            return True
        seen.add(t)
    return False


def replace_runs(out, new_runs, new_plan):
    """Replace whole runs (their run row and all their plan rows) in both tables."""
    keys = {key_of(r) for r in new_runs}
    runs = [r for r in read_runs(out) if key_of(r) not in keys] + list(new_runs)
    ok = write_table(paths(out)["runs"], RUN_COLUMNS, runs)
    return rewrite_plan(out, lambda k: k not in keys, new_plan) and ok


# ====================================================================== meta and logs
def update_meta(out, run_id, info):
    path = paths(out)["meta"]
    try:
        with open(path) as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        meta = {}
    meta[run_id] = {**meta.get(run_id, {}), **info}
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as fh:
            json.dump(meta, fh, indent=1)
        _replace(tmp, path)
    except OSError as e:
        print(f"WARNING: meta.json not written: {e}", flush=True)


def read_meta(out):
    try:
        with open(paths(out)["meta"]) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def pack_logs(out, src=None):
    """Move every file of logs/ (or of src) into logs.zip (a file already in the zip is replaced)."""
    P = paths(out)
    src = src or P["logs"]
    files = sorted(f for f in glob.glob(os.path.join(src, "*")) if os.path.isfile(f))
    if not files:
        try:
            os.rmdir(src)                 # an empty logs/ folder is not kept
        except OSError:
            pass
        return 0
    names = {os.path.basename(f) for f in files}
    tmp = P["zip"] + ".tmp"
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
            if os.path.exists(P["zip"]):
                with zipfile.ZipFile(P["zip"]) as zin:
                    for item in zin.infolist():
                        if item.filename not in names:
                            zout.writestr(item, zin.read(item.filename))
            for f in files:
                zout.write(f, os.path.basename(f))
        if not _replace(tmp, P["zip"]):
            print(f"WARNING: logs.zip locked; logs stay in {src}", flush=True)
            return 0
    except (OSError, zipfile.BadZipFile) as e:
        print(f"WARNING: logs not packed ({e}); they stay in {src}", flush=True)
        return 0
    for f in files:
        try:
            os.remove(f)
        except OSError:
            pass
    try:
        os.rmdir(src)
    except OSError:
        pass
    return len(files)
