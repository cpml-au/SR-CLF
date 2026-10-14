"""config.yaml -> environment variables.

The code reads every numerical setting from an environment variable when it starts.  This module turns the
blocks of an example's config.yaml into those variables:

    key in config.yaml  ->  the table KEYMAP below  ->  environment variable  ->  the code

It is used in two places:

  * the job (tools/cluster.sh), before the workers start, so that every worker gets the variables:
        eval "$(python -m src.settings --export examples/<name>/config.yaml --backend cpu)"
  * the runner (src/CPU/run.py), before it imports the pipeline:
        settings.apply("examples/<name>/config.yaml")

Rules:

  * A variable that is already set wins over config.yaml.  That is how one setting is changed for one job
    (sbatch --export=ALL,NAME=value).  Such variables are named in a "settings: overrides" line.
  * A key that is missing from config.yaml, or set to null, gives no variable: the code then uses its own
    default (src/CPU/config.py, src/CPU/gp_engine.py).
  * A key that is not in the table is an error.

The `gp:` block of config.yaml is not read here; flex reads it.  The `template_cma:` block has no variables:
the template job scripts read it with `python -m src.settings --get template_cma.budget <config>`.

    python -m src.settings --table [config.yaml]     the table as markdown (docs/SETTINGS_CPU.md)
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

# --- how a value is written into a variable -----------------------------------------------------
# Every value reaches the code as text, so the formatter decides the exact characters.


def TEXT(v):
    """Verbatim for strings; canonical Python repr for numbers; 1/0 for booleans."""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, str):
        return v
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    raise TypeError(f"cannot render {v!r} as an environment value")


def FLAG(v):
    """A switch: true/false (or 1/0) -> "1"/"0"."""
    if isinstance(v, str):
        return "1" if v.strip().lower() in ("1", "true", "on", "yes") else "0"
    return "1" if v else "0"


def CSV(v):
    """A list -> comma-separated, each element rendered by TEXT."""
    if isinstance(v, str):
        return v
    return ",".join(TEXT(x) for x in v)


def SEMI(v):
    """A list of expressions -> ';'-separated (flex expressions contain commas), each element by TEXT."""
    if isinstance(v, str):
        return v
    return ";".join(TEXT(x) for x in v)


def ONOFF_LIST(v):
    """A non-empty list of things to do -> "1", an empty list (or null) -> "0"."""
    return "1" if v else "0"


# --- the table ---------------------------------------------------------------------------------
# (key in config.yaml, environment variable or None, formatter, meaning)

KEYMAP = [
    # ---- system: which system, and the box of states the search works in
    ("system.name", "SYMCLF_CPU_SYSTEM", TEXT,
     "the system file: examples/<example>/systems/<name>.py (f, G, the box)"),
    ("system.box", "SYMCLF_BOX", CSV,
     "half-width of the state box on each axis"),
    ("system.template", "SYMCLF_CPU_TEMPLATE", TEXT,
     "the quadratic seed template: full = all n(n+1)/2 monomials | are = only those of the Riccati solution"),
    ("system.template_x0", "SYMCLF_CPU_TEMPLATE_X0", TEXT,
     "start value of every coefficient of the template"),

    # ---- backend: how the work is spread over the cores
    ("backend.threads.cpu", "NUMBA_NUM_THREADS", TEXT,
     "numba threads per worker (one core per worker)"),
    ("backend.check_batch.cpu", "SYMCLF_CPU_S1_BATCH", TEXT,
     "trees handed to a worker in one call"),
    ("backend.driver_cpus", "SYMCLF_CPU_DRIVER_CPUS", TEXT,
     "cores kept for the driver; the others each run one worker"),

    # ---- grid: the training grid of the grid stage
    ("grid.points_per_axis", "SYMCLF_CPU_GRID_POINTS", TEXT,
     "points per axis of the training grid (odd, so that the origin is a grid point)"),

    # ---- verification: the check of every candidate (c_v*, kappa*)
    ("verification.final", "SYMCLF_CPU_FINAL_AUDIT", ONOFF_LIST,
     "what is run once on the best tree at the end of a run: [audit] = src/CPU/audit.py, [] = nothing"),
    ("verification.u_max", "SYMCLF_CHECK_RHO", TEXT,
     "the input bound rho in kappa* = -max (a - rho |b|) / W"),
    ("verification.b_norm", "SYMCLF_CPU_B_NORM", TEXT,
     "the norm of b over the inputs: l1 (each input bounded by rho) | l2 (the input vector bounded by rho)"),
    ("verification.pd_eps", "SYMCLF_CHECK_PD_EPS", TEXT,
     "positive definite iff c_v* > this"),
    ("verification.kappa_min", "SYMCLF_CHECK_KAPPA_MIN", TEXT,
     "numerically verified iff kappa* > this (and c_v* > pd_eps)"),
    ("verification.b_rel_tol", "SYMCLF_CHECK_B_REL_TOL", TEXT,
     "a search end point counts as on b = 0 when rho |b| <= this x |a|"),
    ("verification.project_ascent", "SYMCLF_CHECK_PROJECT_ASCENT", FLAG,
     "project the end points of the search on b = 0 onto b = 0 before that test"),
    ("verification.origin_seeds", "SYMCLF_CHECK_ORIGIN_SEEDS", FLAG,
     "add points along the worst direction at the origin (from the linearisation) to the search"),
    ("verification.kappa_norm", "SYMCLF_CHECK_KAPPA_NORM", TEXT,
     "what kappa* divides by: w = W = V - V(0) | xqx = x'Qx with the system's Q"),
    ("verification.lattice.mesh", "SYMCLF_CPU_LATTICE_MESH", TEXT,
     "scan points of the check: mesh points per axis"),
    ("verification.lattice.lines", "SYMCLF_CPU_LATTICE_LINES", TEXT,
     "scan points of the check: scan lines per remaining axis"),
    ("verification.lattice.samples", "SYMCLF_CPU_LATTICE_SAMPLES", TEXT,
     "scan points of the check: samples along each scan line"),
    ("verification.lattice.random_lines", "SYMCLF_CPU_LATTICE_RANDOM_LINES", TEXT,
     "scan points of the check: number of random lines"),
    ("verification.lattice.random_points", "SYMCLF_CPU_LATTICE_RANDOM_POINTS", TEXT,
     "scan points of the check: points per random line"),
    ("verification.lattice.seed", "SYMCLF_CPU_LATTICE_SEED", TEXT,
     "scan points of the check: seed of the random lines"),

    # ---- fitness: the rule that turns c_v*, kappa* and the region into one number (src/CPU/rule.py)
    ("fitness.region_weight", "SYMCLF_FIT_ROA_WEIGHT", TEXT,
     "region term: weight"),
    ("fitness.region_target", "SYMCLF_FIT_ROA_C_TARGET", TEXT,
     "region term: target of (smallest W on the faces of the training grid) / (median |W|)"),
    ("fitness.hinge.cv_weight", "SYMCLF_CHECK_PEN_CV", TEXT,
     "hinge on c_v*: weight"),
    ("fitness.hinge.cv_target", "SYMCLF_CHECK_PEN_CV_TARGET", TEXT,
     "hinge on c_v*: target (no charge at or above it)"),
    ("fitness.hinge.kappa_weight", "SYMCLF_CHECK_PEN_KAPPA", TEXT,
     "hinge on kappa*: weight"),
    ("fitness.hinge.kappa_target", "SYMCLF_CHECK_PEN_KAPPA_TARGET", TEXT,
     "hinge on kappa*: target (no charge at or above it)"),
    ("fitness.hinge.volume_weight", "SYMCLF_CHECK_PEN_VOL", TEXT,
     "hinge on the sublevel fraction: weight"),
    ("fitness.hinge.volume_target", "SYMCLF_CHECK_PEN_VOL_TARGET", TEXT,
     "hinge on the sublevel fraction: target.  The fraction is the share of the training grid points "
     "with W below its smallest value on the faces"),
    ("fitness.non_pd_kappa_cap", "SYMCLF_CHECK_GATE_KAPPA_CAP", TEXT,
     "a candidate that is not positive definite is charged kappa* = -this"),
    ("fitness.length_reg", "SYMCLF_LENGTH_REG", TEXT,
     "charge per node of the tree"),
    ("fitness.nested_penalty", "SYMCLF_FIT_NESTED_PENALTY", TEXT,
     "charge for a sin or cos inside a sin or cos, for an exp inside an exp, for an aq inside an aq "
     "(once for each of the three)"),
    ("fitness.non_pd_weight", "SYMCLF_FIT_NON_PD_WEIGHT", TEXT,
     "extra charge for a tree that is not positive definite: this x the share of the check's scan "
     "points that fail the c_v* test (0 = off)"),

    # ---- stages: what happens to a tree (src/CPU/gp_engine.py)
    ("stages.gate.max_length", "SYMCLF_MAX_TREE_LENGTH", TEXT,
     "a tree with this many nodes or more is not evaluated (fitness 1e8)"),
    ("stages.tune.mode", "SYMCLF_TUNE_MODE", TEXT,
     "how the constants are tuned: cma_all = option (b), CMA-ES on every live structure | "
     "cma_vip = option (c), the seed template first, then a draw from the best, then the rest"),
    ("stages.cma.popsize", "SYMCLF_CMA_ELITE_POPSIZE", TEXT,
     "CMA-ES: evaluations per iteration"),
    ("stages.cma.seed", "SYMCLF_CMA_ELITE_SEED", TEXT,
     "CMA-ES: seed of every search"),
    ("stages.cma.max_restarts", "SYMCLF_CMA_MAX_RESTARTS", TEXT,
     "CMA-ES: a search that stops by itself restarts from its best point, at most this many times"),
    ("stages.cma_all.total_evals", "SYMCLF_CMA_ALL_TOTAL_EVALS", TEXT,
     "option (b): the most CMA evaluations of one generation, all structures together"),
    ("stages.cma_all.max_evals", "SYMCLF_CMA_ALL_MAX_EVALS", TEXT,
     "option (b): evaluations of the first rank per generation"),
    ("stages.cma_all.min_evals", "SYMCLF_CMA_ALL_MIN_EVALS", TEXT,
     "option (b): evaluations of the last rank per generation (linear in between)"),
    ("stages.cma_all.cap_verified", "SYMCLF_CMA_ALL_CAP_VERIFIED", TEXT,
     "options (b) and (c): lifetime evaluations of a structure once it is numerically verified"),
    ("stages.cma_all.cap_other", "SYMCLF_CMA_ALL_CAP_OTHER", TEXT,
     "options (b) and (c): lifetime evaluations of a structure that is not numerically verified"),
    ("stages.cma_vip.seed_first", "SYMCLF_CMA_VIP_SEED_FIRST", TEXT,
     "option (c), the seed template: evaluations of each island copy in generation 0"),
    ("stages.cma_vip.seed_again", "SYMCLF_CMA_VIP_SEED_AGAIN", TEXT,
     "option (c), the seed template: evaluations each time it appears again"),
    ("stages.cma_vip.seed_cap", "SYMCLF_CMA_VIP_SEED_CAP", TEXT,
     "option (c), the seed template: lifetime evaluations"),
    ("stages.cma_vip.vip_pool", "SYMCLF_CMA_VIP_POOL", TEXT,
     "option (c), the draw: it is made among this many best structures (the seed excluded)"),
    ("stages.cma_vip.vip_count", "SYMCLF_CMA_VIP_COUNT", TEXT,
     "option (c), the draw: structures drawn per generation"),
    ("stages.cma_vip.vip_first", "SYMCLF_CMA_VIP_FIRST", TEXT,
     "option (c), a drawn structure: evaluations the first time"),
    ("stages.cma_vip.vip_again", "SYMCLF_CMA_VIP_AGAIN", TEXT,
     "option (c), a drawn structure: evaluations each later time"),
    ("stages.cma_vip.vip_cap", "SYMCLF_CMA_VIP_CAP", TEXT,
     "option (c), a drawn structure: lifetime evaluations"),
    ("stages.cma_vip.std_total", "SYMCLF_CMA_VIP_STD_TOTAL", TEXT,
     "option (c), the other structures: evaluations per generation, all together"),
    ("stages.cma_vip.std_max", "SYMCLF_CMA_VIP_STD_MAX", TEXT,
     "option (c), the other structures: evaluations of their first rank"),
    ("stages.cma_vip.std_min", "SYMCLF_CMA_VIP_STD_MIN", TEXT,
     "option (c), the other structures: evaluations of their last rank"),
    ("stages.cma_vip.total_max", "SYMCLF_CMA_VIP_TOTAL_MAX", TEXT,
     "option (c): the most CMA evaluations of one generation, everything together"),
    ("stages.cma_vip.draw_seed", "SYMCLF_CMA_VIP_DRAW_SEED", TEXT,
     "option (c), the draw: its random seed"),

    # ---- gp_runtime: the seed trees and the migration between islands
    ("gp_runtime.seeds", "SYMCLF_CPU_SEEDS", FLAG,
     "put seed trees into the first population (true) or start from random trees only (false)"),
    ("gp_runtime.seed_trees", "SYMCLF_CPU_SEED_TREES", SEMI,
     "the seed trees, as a list of expressions; without it: x1^2 + ... + xn^2 + x1 x2, and the "
     "quadratic template"),
    ("gp_runtime.seed_islands", "FLEX_SEED_ISLANDS", TEXT,
     "how many islands get the seed trees"),
    ("gp_runtime.migration_fix.cpu", "FLEX_MIGRATION_FIX", FLAG,
     "migrate every gp.multi_island.migration.freq generations; an immigrant replaces a tree by its "
     "position (docs/flex_patch)"),

    # ---- template_cma: the single-template CMA run (src/CPU/cma_run.py, slurm/template_<name>.slurm)
    ("template_cma.budget", None, TEXT, "template run: fitness evaluations per seed (cma_run --budget)"),
    ("template_cma.popsize", None, TEXT, "template run: evaluations per CMA iteration (cma_run --popsize)"),
    ("template_cma.sigma0", None, TEXT, "template run: first step size (cma_run --sigma0)"),
    ("template_cma.seeds", None, CSV, "template run: the seeds, one run per seed (cma_run --seed)"),
    ("template_cma.workers", None, TEXT, "template run: worker processes per seed (cma_run --workers)"),
    ("template_cma.threads", None, TEXT, "template run: numba threads per worker (cma_run --threads)"),
]

#: The top-level blocks of config.yaml this module reads.  `gp` is flex's and is left alone.
OWNED_BLOCKS = ("system", "backend", "grid", "verification", "fitness", "stages", "gp_runtime", "template_cma")

_BY_KEY = {key: (name, fmt, meaning) for key, name, fmt, meaning in KEYMAP}


def _only_cpu(backend):
    if backend not in (None, "cpu"):
        raise ValueError(f"this repository holds the CPU pipeline only; backend must be cpu, got {backend!r}")


# --- loading -----------------------------------------------------------------------------------

def _flatten(node, prefix=""):
    """Dotted leaf keys of a nested mapping.  Lists and scalars are leaves."""
    out = {}
    for name, value in node.items():
        key = f"{prefix}{name}"
        if isinstance(value, dict):
            out.update(_flatten(value, key + "."))
        else:
            out[key] = value
    return out


def load(path):
    """Read a config file and check its keys.  Returns the flat ``{dotted key: value}`` mapping of the
    blocks this module reads (not the ``gp`` block)."""
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: the top level must be a mapping of blocks")

    flat = {}
    for block, body in raw.items():
        if block == "gp":
            continue
        if block not in OWNED_BLOCKS:
            raise ValueError(
                f"{path}: unknown block {block!r}.  Valid blocks: "
                + ", ".join(OWNED_BLOCKS + ("gp",))
            )
        if not isinstance(body, dict):
            raise ValueError(f"{path}: block {block!r} must be a mapping")
        flat.update(_flatten(body, block + "."))

    unknown = sorted(k for k in flat if k not in _BY_KEY)
    if unknown:
        raise ValueError(
            f"{path}: unknown setting(s): " + ", ".join(unknown) + "\nValid keys:\n  "
            + "\n  ".join(k for k, *_ in KEYMAP)
        )
    return flat


# --- translation --------------------------------------------------------------------------------

def to_env(cfg, backend=None):
    """``{dotted key: value}`` -> ``{VARIABLE: text}``.  A key whose value is ``None`` and a key without a
    variable give nothing, so the code keeps its own default."""
    _only_cpu(backend)
    env = {}
    for key, name, fmt, _meaning in KEYMAP:
        if key not in cfg:
            continue
        value = cfg[key]
        if value is None or name is None:
            continue
        env[name] = fmt(value)
    return env


def apply(path, backend=None, environ=None, verbose=True):
    """Set the environment from ``path``.  Call this BEFORE importing the pipeline.

    A variable that is already set keeps its value; those are printed as ``overrides:`` and returned as the
    second element.  Returns ``(applied, overrides)``.
    """
    environ = os.environ if environ is None else environ
    wanted = to_env(load(path), backend)

    applied, overrides = {}, {}
    for name in sorted(wanted):
        current = environ.get(name)
        if current is not None and current != "":
            overrides[name] = current
        else:
            environ[name] = wanted[name]
            applied[name] = wanted[name]
    if verbose:
        print(f"settings: {path} [cpu] -> {len(applied)} variables", flush=True)
        if overrides:
            print(
                "settings: overrides (already in the environment, kept): "
                + " ".join(f"{k}={v}" for k, v in sorted(overrides.items())),
                flush=True,
            )
    return applied, overrides


# --- rendering ----------------------------------------------------------------------------------

def _shell_quote(value):
    return "'" + value.replace("'", "'\"'\"'") + "'"


def export_lines(path, backend=None, environ=None):
    """``export NAME='value'`` lines for ``eval "$(...)"``; returns ``(lines, overrides)``."""
    environ = os.environ if environ is None else environ
    wanted = to_env(load(path), backend)

    lines, overrides = [], {}
    for name in sorted(wanted):
        current = environ.get(name)
        if current is not None and current != "":
            overrides[name] = current
            continue
        lines.append(f"export {name}={_shell_quote(wanted[name])}")
    return lines, overrides


def _md(text):
    """Escape what would break a markdown table cell."""
    return text.replace("|", "\\|").replace("<", "&lt;")


def table(path=None, backend=None):
    """KEYMAP as a markdown table: key | variable | value (when a config is given) | meaning."""
    _only_cpu(backend)
    cfg = load(path) if path else {}
    head = "| key | environment variable | value | meaning |\n|---|---|---|---|"
    rows = []
    for key, name, fmt, meaning in KEYMAP:
        if path and key in cfg:
            value = cfg[key]
            shown = "(unset)" if value is None else "`" + _md(fmt(value)) + "`"
        elif path:
            shown = "(absent)"
        else:
            shown = ""
        rows.append(f"| `{key}` | {name or '-'} | {shown} | {_md(meaning)} |")
    return head + "\n" + "\n".join(rows)


def get(path, key):
    """The value of one setting of ``path`` as text; "" when it is null or absent.  For the settings a job
    script reads itself (the template_cma block)."""
    if key not in _BY_KEY:
        raise ValueError(f"unknown setting {key!r}")
    value = load(path).get(key)
    return "" if value is None else _BY_KEY[key][1](value)


# --- command line -------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m src.settings",
        description="Translate an example's config.yaml into the environment variables the code reads.",
    )
    parser.add_argument("--export", metavar="CONFIG",
                        help="print `export NAME='value'` lines for eval \"$(...)\"")
    parser.add_argument("--table", nargs="?", const="", metavar="CONFIG",
                        help="print the key / variable / value / meaning table as markdown")
    parser.add_argument("--backend", choices=("cpu",), help="accepted for the job scripts; only cpu exists")
    parser.add_argument("--get", nargs=2, metavar=("KEY", "CONFIG"),
                        help="print one setting's value as text (empty when null)")
    args = parser.parse_args(argv)

    if args.get:
        print(get(args.get[1], args.get[0]))
        return 0

    if args.export:
        lines, overrides = export_lines(args.export, args.backend)
        print("\n".join(lines))
        if overrides:
            print(
                "settings: overrides (already in the environment, kept): "
                + " ".join(f"{k}={v}" for k, v in sorted(overrides.items())),
                file=sys.stderr,
            )
        return 0
    if args.table is not None:
        print(table(args.table or None, args.backend))
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
