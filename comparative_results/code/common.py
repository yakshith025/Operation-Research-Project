"""Shared helpers of the comparative experiments: paths, the experiment list (experiments.json) and
the variants it expands to. Standard library only, so the runner works wherever the solvers work.

A variant = one experiment factor at one non-baseline value, e.g. E5_cC1_7.5 (c^{C,1} = 7.5).
Its inputs live in comparative_results/variants/<variant id>/ (make_variants.py) and its results in
comparative_results/results/<variant id>/<method>/ (one results_io folder per method, so two methods
can run at the same time without writing into the same tables).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = os.path.abspath(os.path.join(HERE, ".."))           # comparative_results/
PROJECT = os.path.abspath(os.path.join(COMP, ".."))        # project/
EXPERIMENTS_JSON = os.path.join(HERE, "experiments.json")
DEFAULT_VARIANTS = os.path.join(COMP, "variants")
DEFAULT_RESULTS = os.path.join(COMP, "results")
DEFAULT_MODELS = os.path.join(COMP, "models")
DEFAULT_ANALYSIS = os.path.join(COMP, "analysis")

# method -> folder name under results/<variant>/ ; run id key in experiments.json "solver"
METHOD_DIRS = {"milp": "milp", "rl1": "rl1", "rl1_retrain": "rl1_retrained"}
BASE_ID = "base"


def load_experiments(path=EXPERIMENTS_JSON):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def project_path(rel, project=PROJECT):
    return rel if os.path.isabs(rel) else os.path.abspath(os.path.join(project, rel))


def code_dir(project=PROJECT):
    return os.path.join(project, "code")


def rl1_dir(project=PROJECT, given=None):
    """RL1 folder: the given one, else <project>/RL1_reinforcement_learning_method_1, else the
    server layout <project>/rl1/RL1_reinforcement_learning_method_1."""
    if given:
        return os.path.abspath(given)
    for cand in (os.path.join(project, "RL1_reinforcement_learning_method_1"),
                 os.path.join(project, "rl1", "RL1_reinforcement_learning_method_1")):
        if os.path.isdir(os.path.join(cand, "code")):
            return cand
    return os.path.join(project, "RL1_reinforcement_learning_method_1")


def value_tag(kind, value):
    """Text of a value in a variant id: 7.5 -> '7.5', 3 -> '3', [2, 5] -> '2-5', T factor 1.25 -> 'x1.25'."""
    if isinstance(value, (list, tuple)):
        return "-".join(value_tag("", v) for v in value)
    if kind == "horizon":
        return "x" + value_tag("", value)
    v = float(value)
    return str(int(v)) if v.is_integer() else f"{v:g}"


def value_text(kind, value):
    """Human-readable value, e.g. 'Beta(2,5)', 'x1.25 T', '7.5'."""
    if kind == "theta_beta":
        return f"Beta({value[0]:g},{value[1]:g})"
    if kind == "horizon":
        return f"T x {float(value):g}"
    return f"{float(value):g}"


def numeric(kind, value):
    """A number for elasticities and plots (theta_beta: the mean a / (a + b))."""
    if kind == "theta_beta":
        return value[0] / (value[0] + value[1])
    return float(value)


def variants(EXP=None):
    """Every (non-baseline) variant of every experiment, in experiment-file order:
    dicts with id, experiment, number, priority, name, question, factor (its dict), kind, key,
    label, value, baseline, families (None = all)."""
    EXP = EXP or load_experiments()
    out = []
    for ex in EXP["experiments"]:
        for fac in ex["factors"]:
            for val in fac["values"]:
                out.append({"id": f"{ex['id']}_{fac['label']}_{value_tag(fac['kind'], val)}",
                            "experiment": ex["id"], "number": ex.get("number"), "priority": ex["priority"],
                            "name": ex["name"], "question": ex.get("question", ""), "factor": fac,
                            "kind": fac["kind"], "key": fac["key"], "label": fac["label"],
                            "value": val, "baseline": fac["baseline"], "families": ex.get("families")})
    return out


def select(EXP, priorities=None, experiments=None, ids=None):
    """Variants to run, ordered by priority (then file order). experiments / ids override priorities."""
    allv = variants(EXP)
    if ids:
        want = set(ids)
        sel = [v for v in allv if v["id"] in want]
        missing = want - {v["id"] for v in sel}
        if missing:
            raise SystemExit(f"unknown variant id(s): {sorted(missing)}; known: {[v['id'] for v in allv]}")
    elif experiments:
        want = {e.upper() for e in experiments}
        sel = [v for v in allv if v["experiment"].upper() in want]
    else:
        pr = set(priorities or EXP.get("default_priorities", [1, 2, 3, 4]))
        sel = [v for v in allv if v["priority"] in pr]
    order = {v["id"]: k for k, v in enumerate(allv)}
    return sorted(sel, key=lambda v: (v["priority"], order[v["id"]]))


def experiment(EXP, exp_id):
    return next(e for e in EXP["experiments"] if e["id"] == exp_id)
