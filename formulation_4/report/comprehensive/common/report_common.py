"""Shared helpers of the three formulation-4 comprehensive reports (milp/, rl/, results/).

Paths, the plotting style of the formulation-3 comprehensive report, loaders for the common result format of
formulation 4 (runs.csv / plan.csv, see ../../../common/results_io_f4.py), a LaTeX table
writer and a numbers.tex writer (\\newcommand macros so the text never holds a hand-typed number).
Import with:  sys.path.insert(0, os.path.join(HERE, "..", "common")); import report_common as rc
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                   # formulation_4/report/comprehensive
F4 = os.path.dirname(os.path.dirname(ROOT))                    # formulation_4
REPO = os.path.dirname(F4)
F4_DATA = os.path.join(F4, "data")
F4_COMMON = os.path.join(F4, "common")
sys.path.insert(0, F4_COMMON)
from paths import resolve as F4J  # noqa: E402  old folder names -> methods/, experiments/
EXPERIMENTS = ["E5_value_of_charging_time", "E6_money_sensitivity", "E1_van_tariff",
               "E2_van_fleet_and_seats", "E8_van_battery"]

FAMS = ["C1", "C2", "R1", "R2", "RC1", "RC2"]
SEGMENTS = ["personal", "commercial"]
SEG_COL = {"personal": "#4C72B0", "commercial": "#DD8452"}
BANDS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)]
BLAB = ["0.0--0.2", "0.2--0.4", "0.4--0.6", "0.6--0.8", "0.8--1.0"]      # LaTeX
PLAB = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]           # plots
STATUS_COL = {"optimal": "#55A868", "time_limit": "#DD8452", "time_limit_no_plan": "#8172B2",
              "infeasible": "#C44E52", "plan": "#55A868", "invalid_plan": "#C44E52", "no_plan": "#8172B2"}

plt.rcParams.update({"font.size": 9, "font.family": "serif", "mathtext.fontset": "cm",
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def f4_import():
    """Make the formulation-4 modules importable (f4_data, f4_params, check_plan_f4, ...)."""
    if F4_COMMON not in sys.path:
        sys.path.insert(0, F4_COMMON)


def load_runs(results_dir, run_id=None):
    df = pd.read_csv(os.path.join(results_dir, "runs.csv"), low_memory=False)
    if run_id is not None:
        df = df[df.run_id == run_id].copy()
    df["valid_plan"] = df["valid"].astype(str).str.lower().eq("yes")
    return df


def load_plan(results_dir, run_id=None):
    df = pd.read_csv(os.path.join(results_dir, "plan.csv"), low_memory=False)
    if run_id is not None:
        df = df[df.run_id == run_id].copy()
    return df


def band_of(theta):
    """Index 0..4 of the income band of theta (array or scalar)."""
    return np.clip(np.floor(np.asarray(theta, float) / 0.2).astype(int), 0, 4)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


class Numbers:
    """Collects \\newcommand macros; save() writes tables/numbers.tex."""

    def __init__(self):
        self.d = {}

    def __setitem__(self, name, value):
        assert name.isalpha(), f"macro names must be letters only: {name}"
        self.d[name] = value

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for k, v in sorted(self.d.items()):
                fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
        with open(path.replace(".tex", ".json"), "w", encoding="utf-8") as fh:
            json.dump(self.d, fh, indent=1, default=str)


def fmt(v, spec="{:.1f}"):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "--"
        return spec.format(v)
    except (TypeError, ValueError):
        return str(v)


def latex_table(path, header, rows, align=None, caption=None, label=None, total_row=None, size=None):
    """Booktabs table. header: list of column titles (may contain \\\\ via makecell-free \\shortstack)."""
    align = align or ("l" + "r" * (len(header) - 1))
    out = []
    out.append("\\begin{table}[H]\\centering")
    if size:
        out.append(f"\\{size}")
    if caption:
        out.append(f"\\caption{{{caption}}}" + (f"\\label{{{label}}}" if label else ""))
    out.append(f"\\begin{{tabular}}{{{align}}}\\toprule")
    out.append(" & ".join(header) + " \\\\ \\midrule")
    for r in rows:
        out.append(" & ".join(str(x) for x in r) + " \\\\")
    if total_row:
        out.append("\\midrule " + " & ".join(str(x) for x in total_row) + " \\\\")
    out.append("\\bottomrule\\end{tabular}\\end{table}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")


def savefig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
