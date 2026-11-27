"""Folder layout of formulation 4: the one place that knows where things live.

formulation_4/
  data/  common/  report/
  methods/milp/   methods/rl/                       baselines (code/, results/, rl: model/)
  experiments/<folder>/milp   experiments/<folder>/rl  (folder = "E1_van_tariff", ... as in experiments.json)
  results_summary/                                  MILP-vs-RL tables, rubric_results/
"""
import os

COMMON_DIR = os.path.dirname(os.path.abspath(__file__))
F4_DIR = os.path.dirname(COMMON_DIR)
PROJECT_DIR = os.path.dirname(F4_DIR)
DATA_DIR = os.path.join(F4_DIR, "data")
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")          # EV-Solomon source dataset
MILP_DIR = os.path.join(F4_DIR, "methods", "milp")
RL_DIR = os.path.join(F4_DIR, "methods", "rl")
EXPERIMENTS_DIR = os.path.join(F4_DIR, "experiments")
SUMMARY_DIR = os.path.join(F4_DIR, "results_summary")


def method_dir(method, root=None):
    """methods/milp or methods/rl (method: 'milp' / 'rl', any case)."""
    return os.path.join(root or F4_DIR, "methods", method.lower())


def exp_dir(folder, method, root=None):
    """experiments/<folder>/<milp|rl>; folder like 'E1_van_tariff' (a trailing _MILP/_RL is accepted)."""
    m = (method or "").lower()
    for suf in ("_MILP", "_RL"):
        if folder.endswith(suf):
            folder, m = folder[: -len(suf)], suf[1:].lower()
    return os.path.join(root or F4_DIR, "experiments", folder, m)


def locate(name, root=None):
    """Old flat folder name -> new folder: baseline_MILP, baseline_RL, <E>_..._MILP, <E>_..._RL; else <F4>/name."""
    if name == "baseline_MILP":
        return method_dir("milp", root)
    if name == "baseline_RL":
        return method_dir("rl", root)
    if name.endswith(("_MILP", "_RL")):
        return exp_dir(name, None, root)
    return os.path.join(root or F4_DIR, name)


def resolve(*parts, root=None):
    """Like os.path.join(<F4>, *parts) but the first component is mapped by locate() ('baseline_RL/results/x')."""
    rel = os.path.join(*parts).replace("\\", "/")
    first, _, rest = rel.partition("/")
    return os.path.join(locate(first, root), rest) if rest else locate(first, root)


def f4_join(name, *rest, root=None):
    """os.path.join(<F4>/name, *rest) with name mapped by locate()."""
    return os.path.join(locate(name, root), *rest)
