#!/usr/bin/env python
"""CMA-ES on the quadratic template alone (no genetic programming), with the pipeline's own fitness.

The recipe:

    pycma fmin2, BIPOP restarts, popsize 10, sigma0 1.0, per-coordinate stds 0.3 max(|x0|, 0.1),
    fixed seed, budget in EVALUATIONS, keep-best, one jsonl line per evaluation.

The objective is src/CPU/rule.py::fitness_fixed_constants -- the training-grid stage
(src/CPU/lattice.py), the check (src/CPU/deterministic_search.py), the check cost and the
complexity penalty: the same number a GP run gives this candidate.  No other objective, and no
start from a Riccati solution (x0 = ones by default).

    fitness = grid_score + check_cost + complexity_penalty

Parallel: the population of one CMA iteration is evaluated by a pool of worker processes
(spawn -- forking after numba's threading layer has started deadlocks), each worker with its own
numba thread team.  workers * threads should be the node's core count.

Settings read here: system.name and system.template (the defaults of --system / --template, written
back as SYMCLF_CPU_SYSTEM / SYMCLF_CPU_TEMPLATE for the workers), system.template_x0, and every key the
fitness reads (src/CPU/config.py).

    python -m src.CPU.cma_run --budget 3000 --workers 10 --threads 19 --seed 1 --out runs/quad6d_s1.jsonl

Without --out the log goes to <example>/runs/cma_<tag>.jsonl, <example> being the folder whose systems/ holds
the module (src/CPU/__init__.py::example_dir).  The examples' template job scripts
(examples/<name>/slurm/template_<name>.slurm) take --budget/--popsize/--sigma0/--workers/--threads and the
seeds from the template_cma block of config.yaml.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))   # src/CPU -> src -> repository root
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=int, default=3000, help="fitness evaluations")
    ap.add_argument("--popsize", type=int, default=10, help="CMA population")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sigma0", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=0, help="worker processes (0 = popsize)")
    ap.add_argument("--threads", type=int, default=0, help="numba threads per worker (0 = auto)")
    ap.add_argument("--system", default=os.environ.get("SYMCLF_CPU_SYSTEM", "quad6d"))
    ap.add_argument("--template", default=os.environ.get("SYMCLF_CPU_TEMPLATE", "full"),
                    choices=("full", "are"))
    ap.add_argument("--x0", default="ones", help="'ones', a comma list, or a .json/.jsonl best file")
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="")
    return ap.parse_args()


ARGS = _parse()
# numba reads these at import time, and the spawned workers inherit os.environ
os.environ["SYMCLF_CPU_SYSTEM"] = ARGS.system
os.environ["SYMCLF_CPU_TEMPLATE"] = ARGS.template
if ARGS.threads > 0:
    os.environ["NUMBA_NUM_THREADS"] = str(ARGS.threads)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# `python -m src.CPU.cma_run` imports the src.CPU package BEFORE --system is parsed, so the numba
# cache directory it pinned belongs to the DEFAULT system -- and the spawned workers inherit it.
# Re-pin it now that SYMCLF_CPU_SYSTEM and the example path are final, before anything compiles.
from src.CPU import apply_numba_cache_dir as _apply_numba_cache_dir  # noqa: E402
from src.CPU import example_dir as _example_dir  # noqa: E402

# unchanged loader contract (systems.<name>): the example folder of the selected system on sys.path
# (examples/<system>, or the folder whose systems/ holds <system>.py: src/CPU/__init__.py::example_dir)
_EXAMPLE = _example_dir(ARGS.system)
if _EXAMPLE and _EXAMPLE not in sys.path:
    sys.path.insert(0, _EXAMPLE)

_apply_numba_cache_dir()

import numpy as np  # noqa: E402

from src.CPU import config  # noqa: E402
from src.CPU.rule import complexity_penalty, fitness_fixed_constants  # noqa: E402
from src.CPU.templates import default_template  # noqa: E402

_STATE = {}


def _worker_init(environ):
    os.environ.update(environ)
    import src.CPU.evalnb  # noqa: F401  (compile the kernels once per worker)
    from src.CPU.rule import fitness_fixed_constants as _f
    _STATE["fitness"] = _f


def _worker_eval(item):
    expression, constants, extra = item
    try:
        out = _STATE["fitness"](expression, np.asarray(constants, float), extra=extra)
    except Exception as exc:                      # a failure costs CHECK_FAILED_PENALTY, never fatal
        return {"fitness": float(config.CHECK_FAILED_PENALTY), "error": f"{type(exc).__name__}: {exc}",
                "numerically_verified": False, "dead": False, "check": None, "grid": None}
    d4 = out.get("check") or {}
    return {"fitness": out["fitness"], "numerically_verified": out["numerically_verified"], "dead": out["dead"],
            "kappa": d4.get("kappa"), "cv": d4.get("cv"), "check_cost": out["check_cost"],
            "grid_score": out["grid_score"],
            "volume": (out.get("grid") or {}).get("sublevel_fraction")}


class BudgetExhausted(Exception):
    pass


class Objective:
    """The pipeline's fitness of the template at constants c; keep-best, jsonl log, budget guard."""

    def __init__(self, expression, budget, out_path, pool=None):
        self.expr = expression
        self.extra = complexity_penalty(expression)
        self.budget = int(budget)
        self.pool = pool
        self.n = 0
        self.best = np.inf
        self.best_c = None
        self.verified_at = None
        self.n_verified = 0
        self.t0 = time.time()
        self.fh = open(out_path, "a")

    def log(self, record):
        self.fh.write(json.dumps(record, default=float) + "\n")
        self.fh.flush()

    def batch(self, candidates):
        if self.n >= self.budget:
            raise BudgetExhausted()
        items = [(self.expr, np.asarray(c, float), self.extra) for c in candidates]
        results = self.pool.map(_worker_eval, items) if self.pool else [_worker_eval(i) for i in items]
        values = []
        for c, r in zip(candidates, results):
            self.n += 1
            f = float(r["fitness"])
            if r.get("numerically_verified"):
                self.n_verified += 1
                if self.verified_at is None:
                    self.verified_at = self.n
            if np.isfinite(f) and f < self.best:
                self.best, self.best_c = f, [float(v) for v in c]
            self.log({"eval": self.n, "fitness": f, "numerically_verified": bool(r.get("numerically_verified")),
                      "kappa": r.get("kappa"), "cv": r.get("cv"), "volume": r.get("volume"),
                      "dead": bool(r.get("dead")), "error": r.get("error"),
                      "c": [float(v) for v in c], "best": self.best, "t": time.time() - self.t0})
            values.append(f if np.isfinite(f) else 1.0e12)
        return values

    def __call__(self, c):
        return self.batch([c])[0]


def _stds(x0):
    return 0.3 * np.maximum(np.abs(np.asarray(x0, float)), 0.1)


def run_cma(obj, x0, seed, budget, popsize, sigma0):
    import cma

    opts = {"CMA_stds": _stds(x0).tolist(), "popsize": int(popsize), "seed": int(seed),
            "verbose": -9, "verb_disp": 0, "verb_log": 0,
            "termination_callback": [lambda es: obj.n >= budget]}
    try:
        cma.fmin2(None, list(map(float, x0)), float(sigma0), opts, restarts=9, bipop=True,
                  parallel_objective=obj.batch)
    except BudgetExhausted:
        pass


def _x0_from(spec, default_x0):
    if spec == "ones":
        return np.asarray(default_x0, float)
    if os.path.exists(spec):
        best = None
        with open(spec) as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("final") and record.get("best_c"):
                    best = record["best_c"]
                elif record.get("c") is not None and (best is None or record.get("fitness", np.inf) < best[0]):
                    pass
        if best is None:
            raise ValueError(f"no 'final' record with best_c in {spec}")
        return np.asarray(best, float)
    return np.asarray([float(v) for v in spec.split(",")], float)


def main():
    from multiprocessing import get_context

    expression, x0_default, kind = default_template()
    x0 = _x0_from(ARGS.x0, x0_default)
    tag = ARGS.tag or f"{ARGS.system}_{kind}_s{ARGS.seed}"
    out_path = ARGS.out or os.path.join(_EXAMPLE or HERE, "runs", f"cma_{tag}.jsonl")   # the example's runs/
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    workers = ARGS.workers if ARGS.workers > 0 else int(ARGS.popsize)
    threads = int(os.environ.get("NUMBA_NUM_THREADS", "0") or 0)

    header = {
        "header": True, "system": ARGS.system, "template": kind, "n_coefficients": int(len(x0)),
        "expression": expression, "x0": [float(v) for v in x0], "budget": ARGS.budget,
        "popsize": ARGS.popsize, "seed": ARGS.seed, "sigma0": ARGS.sigma0, "workers": workers,
        "numba_threads": threads, "host": socket.gethostname(), "box": list(config.box()),
        "grid_points": config.grid_points(), "lattice": config.lattice_kwargs(),
        "rho": config.RHO, "b_norm": config.B_NORM, "pd_eps": config.PD_EPS,
        "kappa_min": config.KAPPA_MIN,
        "hinges": {"cv": [config.PEN_CV, config.PEN_CV_TARGET],
                   "kappa": [config.PEN_KAPPA, config.PEN_KAPPA_TARGET],
                   "vol": [config.PEN_VOL, config.PEN_VOL_TARGET]},
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(json.dumps(header), flush=True)

    ctx = get_context("spawn")
    pool = ctx.Pool(workers, initializer=_worker_init, initargs=(dict(os.environ),)) if workers > 1 else None
    obj = Objective(expression, ARGS.budget, out_path, pool)
    obj.log(header)
    t0 = time.time()
    f0 = obj([float(v) for v in x0])
    print(f"x0 fitness {f0:.6f}  ({time.time() - t0:.1f} s incl. worker warm-up)", flush=True)
    t0 = time.time()
    run_cma(obj, x0, ARGS.seed, ARGS.budget, ARGS.popsize, ARGS.sigma0)
    seconds = time.time() - t0
    final = {"final": True, "evals": obj.n, "best": obj.best, "best_c": obj.best_c,
             "numerically_verified_at": obj.verified_at, "n_numerically_verified": obj.n_verified,
             "seconds": seconds, "per_eval_s": seconds / max(obj.n, 1)}
    obj.log(final)
    print(json.dumps(final), flush=True)
    print(f"done: {obj.n} evals, best {obj.best:.6f}, numerically_verified_at {obj.verified_at}, "
          f"{seconds / max(obj.n, 1):.2f} s/eval, log {out_path}", flush=True)
    if pool is not None:
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
