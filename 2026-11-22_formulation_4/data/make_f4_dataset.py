"""Build the complete formulation-4 dataset in this folder (data/).

Source: the EV-Solomon dataset of formulation 3 in CSV form (../../dataset_ev_solomon/csv): depot,
stations, EV positions, energy, request times, ranges, fleet and costs. Those columns are copied
unchanged. New per EV: the use segment and the income index, drawn from the distributions of
../common/f4_params.json (calibrated in calibration.md):
    s_i ~ Bernoulli(pi_commercial)                       commercial or personal
    theta_i | s_i ~ Beta(a_s, b_s)                       per segment
Seeds: one per Solomon class and file size (seed_base + class + size), with three independent streams
(segment, theta if personal, theta if commercial). So
  - files of one class share positions, energies, ranges AND attributes; they differ only in request times
    (as in formulation 3);
  - changing pi_commercial only re-labels EVs, the theta draws of each segment stay the same (common
    random numbers across settings).

Writes (two files per source, the same columns as the source plus `segment`, `theta` on the EV rows):
    solomon_config.csv, solomon_data.csv                         56 files x 100 EVs
    gehring_homberger_config.csv, gehring_homberger_data.csv     60 files x 200 EVs
    dataset_settings.json    distributions, seeds, source fingerprint, date
    summary.csv              per file: EVs, commercial share, mean theta per segment, first-10 / first-50 shares

Usage (from data/)
    python make_f4_dataset.py              # write the dataset (refuses to overwrite a different one)
    python make_f4_dataset.py --force      # rewrite (e.g. after changing pi_commercial or a Beta)
    python make_f4_dataset.py --check      # verify: network columns identical to the source, attributes valid
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
F4_DIR = os.path.dirname(HERE)
PROJECT = os.path.dirname(F4_DIR)
sys.path.insert(0, os.path.join(F4_DIR, "common"))
import f4_params as fp  # noqa: E402

SOURCE_CSV = os.path.join(PROJECT, "dataset_ev_solomon", "csv")
STEMS = ("solomon", "gehring_homberger")
SEG_NAMES = fp.SEGMENTS


def fingerprint(paths):
    h = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:16]


def class_seed(cls, N, base):
    return int(base) + sum(ord(ch) for ch in cls) * 101 + int(N)


def draw_attributes(cls, N, P):
    """(segment 0/1 array, theta array) for the N EVs of a file of class cls."""
    sd = P["segment_distribution"]
    ss = np.random.SeedSequence(class_seed(cls, N, sd["seed_base"]))
    r_seg, r_p, r_c = (np.random.default_rng(s) for s in ss.spawn(3))
    seg = (r_seg.random(N) < float(sd["pi_commercial"])).astype(int)
    dP, dC = P["personal"]["theta_distribution"], P["commercial"]["theta_distribution"]
    th_p = r_p.beta(dP["a"], dP["b"], N)
    th_c = r_c.beta(dC["a"], dC["b"], N)
    theta = np.round(np.where(seg == 1, th_c, th_p), 4)
    return seg, theta


def settings_of(P, src_fp):
    return {"segment_distribution": P["segment_distribution"],
            "personal_theta": P["personal"]["theta_distribution"],
            "commercial_theta": P["commercial"]["theta_distribution"],
            "source": os.path.relpath(SOURCE_CSV, HERE), "source_fingerprint": src_fp}


def build(P, force=False):
    files = []
    for stem in STEMS:
        files += [os.path.join(SOURCE_CSV, f"{stem}_config.csv"), os.path.join(SOURCE_CSV, f"{stem}_data.csv")]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        print(f"source files missing: {missing}")
        sys.exit(1)
    src_fp = fingerprint(files)
    stamp_path = os.path.join(HERE, "dataset_settings.json")
    new = settings_of(P, src_fp)
    if os.path.exists(stamp_path) and not force:
        try:
            old = json.load(open(stamp_path))
            old = {k: old[k] for k in new}
        except (OSError, ValueError, KeyError):
            old = None
        if old == new:
            print("the dataset in data/ is up to date (same distributions and source); nothing written")
            return
        print("data/ holds a dataset made with other settings or another source; run with --force to rewrite "
              "it (every result made with the old data then refers to a different dataset)")
        sys.exit(1)
    summary = []
    for stem in STEMS:
        with open(os.path.join(SOURCE_CSV, f"{stem}_config.csv"), newline="") as fh:
            cfg_rows = list(csv.DictReader(fh))
            cfg_cols = list(cfg_rows[0].keys())
        with open(os.path.join(SOURCE_CSV, f"{stem}_data.csv"), newline="") as fh:
            rd = csv.DictReader(fh)
            data_cols = list(rd.fieldnames)
            data_rows = list(rd)
        cls_of = {r["instance"]: r["class"] for r in cfg_rows}
        n_of = {r["instance"]: int(float(r["n_evs"])) for r in cfg_rows}
        attrs = {name: draw_attributes(cls_of[name], n_of[name], P) for name in cls_of}
        out_rows = []
        for r in data_rows:
            row = dict(r)
            row["segment"], row["theta"] = "", ""
            if r["type"] == "ev":
                seg, theta = attrs[r["instance"]]
                i = int(r["id"]) - 1
                row["segment"] = SEG_NAMES[seg[i]]
                row["theta"] = f"{theta[i]:.4f}"
            out_rows.append(row)
        with open(os.path.join(HERE, f"{stem}_config.csv"), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=cfg_cols)
            wr.writeheader()
            wr.writerows(cfg_rows)
        with open(os.path.join(HERE, f"{stem}_data.csv"), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=data_cols + ["segment", "theta"])
            wr.writeheader()
            wr.writerows(out_rows)
        for name, (seg, theta) in attrs.items():
            row = {"instance": name, "source": stem, "class": cls_of[name], "EVs": len(seg),
                   "commercial_share": round(float(seg.mean()), 3),
                   "mean_theta_personal": round(float(theta[seg == 0].mean()), 3) if (seg == 0).any() else "",
                   "mean_theta_commercial": round(float(theta[seg == 1].mean()), 3) if (seg == 1).any() else "",
                   "commercial_share_first10": round(float(seg[:10].mean()), 3),
                   "commercial_share_first50": round(float(seg[:50].mean()), 3)}
            summary.append(row)
        print(f"{stem}: {len(cfg_rows)} files, {sum(n_of.values())} EVs written")
    with open(os.path.join(HERE, "summary.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(summary[0]))
        wr.writeheader()
        wr.writerows(summary)
    with open(stamp_path, "w") as fh:
        json.dump({**new, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "note": "segment and theta drawn by make_f4_dataset.py; every other column copied from the source"},
                  fh, indent=1)
    allseg = [r["commercial_share"] * r["EVs"] for r in summary]
    print(f"commercial share over all EVs: {sum(allseg) / sum(r['EVs'] for r in summary):.3f}; "
          f"settings in {stamp_path}")


def check(P):
    """Network columns identical to the source; segment / theta present and valid; draws reproducible."""
    problems = 0
    for stem in STEMS:
        src = {}
        with open(os.path.join(SOURCE_CSV, f"{stem}_data.csv"), newline="") as fh:
            for r in csv.DictReader(fh):
                src[(r["instance"], r["type"], r["id"])] = r
        with open(os.path.join(HERE, f"{stem}_data.csv"), newline="") as fh:
            mine = list(csv.DictReader(fh))
        with open(os.path.join(HERE, f"{stem}_config.csv"), newline="") as fh:
            cfg = {r["instance"]: r for r in csv.DictReader(fh)}
        with open(os.path.join(SOURCE_CSV, f"{stem}_config.csv"), newline="") as fh:
            cfg_src = {r["instance"]: r for r in csv.DictReader(fh)}
        if cfg != cfg_src:
            print(f"{stem}: config differs from the source")
            problems += 1
        if len(mine) != len(src):
            print(f"{stem}: {len(mine)} rows vs {len(src)} in the source")
            problems += 1
        by_file = {}
        for r in mine:
            s = src.get((r["instance"], r["type"], r["id"]))
            if s is None or any(r[c] != s[c] for c in s):
                problems += 1
                if problems < 5:
                    print(f"{stem}: row {r['instance']} {r['type']} {r['id']} differs from the source")
            if r["type"] == "ev":
                by_file.setdefault(r["instance"], []).append(r)
        for name, rows in by_file.items():
            rows.sort(key=lambda r: int(r["id"]))
            seg, theta = draw_attributes(cfg[name]["class"], len(rows), P)
            got_seg = [r["segment"] for r in rows]
            got_th = [float(r["theta"]) for r in rows]
            if got_seg != [SEG_NAMES[s] for s in seg] or np.max(np.abs(np.array(got_th) - theta)) > 1e-9:
                problems += 1
                print(f"{stem}: {name} attributes are not the draws of the current f4_params.json")
            if not all(0.0 <= t <= 1.0 for t in got_th):
                problems += 1
                print(f"{stem}: {name} theta outside [0, 1]")
        print(f"{stem}: {len(by_file)} files checked")
    print("check passed" if problems == 0 else f"check found {problems} problem(s)")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", default=fp.DEFAULT_PARAMS, help="common/f4_params.json")
    ap.add_argument("--force", action="store_true", help="rewrite the dataset")
    ap.add_argument("--check", action="store_true", help="verify the dataset in data/")
    args = ap.parse_args()
    P, _ = fp.load_params(args.params)
    if args.check:
        sys.exit(1 if check(P) else 0)
    build(P, force=args.force)


if __name__ == "__main__":
    main()
