#!/usr/bin/env python
"""Independent audit of one candidate: c_v* and kappa* recomputed with sympy and scipy.

It is a second, separately written computation next to the check (src/CPU/deterministic_search.py).
It shares only the settings (src/CPU/config.py), the scan points and Sobol starts
(src/CPU/geometry.py) and the template with the pipeline: V, a, b and W are built with sympy and
lambdified, the searches are scipy's own minimize (SLSQP + Nelder-Mead), and the reported worst point
is re-evaluated in 60-digit mpmath.  Agreement with the check to ~1e-10 is what makes the fitness
numbers trustworthy; a disagreement is a bug in one of the two.

    c_v*   = min over the lattice and the polishes of min(W/|x|^2, x.grad W/(2|x|^2))
    kappa* = -max over {W > 0} of (a - rho*||b|| + gamma1)/W      (lattice, SLSQP ascent on
             {b = 0}, Nelder-Mead rate polish; the origin ray's ratio is reported, not folded in)

This is a numerical search: "no violation found" is not a proof.
Tolerances: numerically verified iff c_v* > PD_EPS + PD_RATE_TOL (1e-4 + 1e-6) and
kappa* > KAPPA_MIN (0.0); thin-|b| seeds = the lowest 1 % of ||b|| on the scan points; mpmath 60 digits.
Settings read here: system.name and system.template (the defaults of --system / --template; this module
writes SYMCLF_CPU_SYSTEM and SYMCLF_CPU_TEMPLATE back into os.environ WHEN IMPORTED), system.box,
verification.u_max, verification.b_norm, verification.pd_eps, verification.kappa_min,
verification.lattice.*; verification.final runs it on the best tree at the end of a run
(SYMCLF_CPU_FINAL_AUDIT).  Environment only: SYMCLF_CLF_GAMMA1, SYMCLF_CHECK_PD_RATE_TOL.

    python -m src.CPU.audit --best runs/cma_quad6d_full_s1.jsonl        (from the repository root)
    python -m src.CPU.audit --constants 1,1,...  [--system quad6d] [--template full]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))   # src/CPU -> src -> repository root
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", default="", help="jsonl log: audit its best_c")
    ap.add_argument("--constants", default="", help="comma-separated coefficients")
    ap.add_argument("--expression", default="", help="explicit expression (default: the template)")
    ap.add_argument("--system", default=os.environ.get("SYMCLF_CPU_SYSTEM", "quad6d"))
    ap.add_argument("--template", default=os.environ.get("SYMCLF_CPU_TEMPLATE", "full"))
    ap.add_argument("--seeds", type=int, default=40, help="lattice seeds per stage")
    ap.add_argument("--json", default="", help="write the audit to this file")
    return ap.parse_args()


ARGS = _args()
os.environ["SYMCLF_CPU_SYSTEM"] = ARGS.system
os.environ["SYMCLF_CPU_TEMPLATE"] = ARGS.template

# unchanged loader contract (systems.<name>): the example folder of the selected system on sys.path
# (examples/<system>, or the folder whose systems/ holds <system>.py: src/CPU/__init__.py::example_dir)
from src.CPU import apply_numba_cache_dir as _apply_numba_cache_dir  # noqa: E402
from src.CPU import example_dir as _example_dir  # noqa: E402

_EXAMPLE = _example_dir(ARGS.system)
if _EXAMPLE and _EXAMPLE not in sys.path:
    sys.path.insert(0, _EXAMPLE)
# `python -m src.CPU.audit` imports the src.CPU package before --system is parsed: re-pin the numba cache
# directory now that the system is final (as cma_run.py does), before anything compiles
_apply_numba_cache_dir()

import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from src.CPU import config  # noqa: E402
from src.CPU.geometry import lattice_coords, sobol_cached  # noqa: E402
from src.CPU.templates import default_template  # noqa: E402


def symbolic_fields(expression, constants):
    """V, a, b (m), W and their gradients, lambdified from sympy -- an independent evaluator."""
    sys_mod = config.system()
    n, m = sys_mod.N_STATES, sys_mod.N_INPUTS
    xs = list(sp.symbols(f"x1:{n + 1}"))
    import re

    text = str(expression)
    values = list(np.asarray(constants, float).reshape(-1))
    # substitute the constants into the standalone `a` terminals, left to right
    holes = list(re.finditer(r"\ba\b", text))
    if len(holes) != len(values):
        raise ValueError(f"expression has {len(holes)} placeholders, {len(values)} constants")
    pieces, last = [], 0
    for value, hole in zip(values, holes):
        pieces.append(text[last:hole.start()])
        pieces.append(repr(float(value)))
        last = hole.end()
    pieces.append(text[last:])
    text = "".join(pieces)
    local = {"add": lambda p, q: p + q, "sub": lambda p, q: p - q, "mul": lambda p, q: p * q,
             "neg": lambda p: -p, "sin": sp.sin, "exp": sp.exp,
             "aq": lambda p, q: p / sp.sqrt(1 + q ** 2)}
    local.update({f"x{i + 1}": xs[i] for i in range(n)})
    V = sp.sympify(sp.parse_expr(text, local_dict=local, evaluate=True))
    gradV = sp.Matrix([sp.diff(V, s) for s in xs])
    f = sp.Matrix(sys_mod.fSR(*xs))
    G = sp.Matrix(sys_mod.GSR(*xs))
    a = (gradV.T * f)[0]
    b = [(gradV.T * G[:, k])[0] for k in range(m)]
    W = V - V.subs({s: 0 for s in xs})
    fns = {
        "V": sp.lambdify(xs, V, "numpy"), "W": sp.lambdify(xs, W, "numpy"),
        "a": sp.lambdify(xs, a, "numpy"), "b": [sp.lambdify(xs, bk, "numpy") for bk in b],
        "gW": sp.lambdify(xs, sp.Matrix([sp.diff(W, s) for s in xs]), "numpy"),
        "ga": sp.lambdify(xs, sp.Matrix([sp.diff(a, s) for s in xs]), "numpy"),
        "gb": [sp.lambdify(xs, sp.Matrix([sp.diff(bk, s) for s in xs]), "numpy") for bk in b],
        "radial": sp.lambdify(xs, sum(xs[i] * sp.diff(W, xs[i]) for i in range(n)), "numpy"),
        "hessV": sp.lambdify(xs, sp.hessian(V, xs), "numpy"),
    }
    return fns, xs, V, a, b, W, n, m


def audit(expression, constants, n_seeds=40):
    fns, xs, V, a_sym, b_sym, W_sym, n, m = symbolic_fields(expression, constants)
    bounds = config.bounds()
    lo, hi = bounds[:, 0], bounds[:, 1]
    rho, l1 = config.RHO, config.B_NORM == "l1"
    coords = lattice_coords(bounds, **config.lattice_kwargs())
    cols = [coords[:, i] for i in range(n)]

    def _vec(fn):
        out = np.asarray(fn(*cols), dtype=float)
        return out if out.shape == (coords.shape[0],) else np.full(coords.shape[0], float(out))

    W = _vec(fns["W"])
    A = _vec(fns["a"])
    B = np.stack([_vec(fn) for fn in fns["b"]], axis=1)
    Bn = np.sum(np.abs(B), axis=1) if l1 else np.linalg.norm(B, axis=1)
    R = _vec(fns["radial"])
    r2 = np.sum(coords * coords, axis=1)
    keep = r2 > 0.0

    def W_of(z):
        return float(fns["W"](*z))

    def a_of(z):
        return float(fns["a"](*z))

    def b_of(z):
        return np.array([float(fn(*z)) for fn in fns["b"]])

    def ratio(z):
        w = W_of(z)
        if not (np.isfinite(w) and w > 0):
            return -np.inf
        bb = b_of(z)
        nb = float(np.sum(np.abs(bb)) if l1 else np.linalg.norm(bb))
        v = (a_of(z) - rho * nb + config.GAMMA1) / w
        return v if np.isfinite(v) else -np.inf

    # --- c_v*: min W/|x|^2 and the radial ratio, from the best lattice seeds ------------------
    with np.errstate(all="ignore"):
        q = np.where(keep, W / np.maximum(r2, 1e-300), np.inf)
        qr = np.where(keep, R / (2.0 * np.maximum(r2, 1e-300)), np.inf)
    cv, cv_x = float(np.min(q)), coords[int(np.argmin(q))]
    cvr, cvr_x = float(np.min(qr)), coords[int(np.argmin(qr))]
    box = [(float(l), float(h)) for l, h in zip(lo, hi)]
    for seed in np.vstack([coords[np.argsort(q)[:n_seeds]], sobol_cached(bounds, 64)]):
        for method in ("SLSQP", "Nelder-Mead"):
            try:
                res = minimize(lambda z: (W_of(z) / max(float(z @ z), 1e-300)) if float(z @ z) > 0 else np.inf,
                               np.clip(seed, lo, hi), method=method, bounds=box)
                z = np.clip(res.x, lo, hi)
                if float(z @ z) > 0 and np.isfinite(res.fun) and res.fun < cv:
                    cv, cv_x = float(res.fun), z
            except Exception:
                continue
    for seed in coords[np.argsort(qr)[:n_seeds]]:
        try:
            res = minimize(lambda z: float(fns["radial"](*z)) / (2.0 * max(float(z @ z), 1e-300)),
                           np.clip(seed, lo, hi), method="Nelder-Mead", bounds=box)
            z = np.clip(res.x, lo, hi)
            if float(z @ z) > 0 and np.isfinite(res.fun) and res.fun < cvr:
                cvr, cvr_x = float(res.fun), z
        except Exception:
            continue
    c_v = min(cv, cvr)
    pd_valid = bool(np.isfinite(c_v) and c_v > config.PD_EPS + config.PD_RATE_TOL)

    # --- kappa*: manifold ascent (m equality constraints) + rate stage ------------------------
    with np.errstate(all="ignore"):
        rate = np.where(keep & (W > 0), (A - rho * Bn) / np.where(W > 0, W, 1.0), -np.inf)
        score = np.where(keep & (W > 0), A / np.where(W > 0, W, 1.0), -np.inf)
    best, best_x = float(np.max(rate)), coords[int(np.argmax(rate))]
    cons = [{"type": "eq", "fun": (lambda z, k=k: float(fns["b"][k](*z))),
             "jac": (lambda z, k=k: np.asarray(fns["gb"][k](*z), float).reshape(-1))} for k in range(m)]
    thin = np.argsort(np.where(Bn <= np.quantile(Bn[np.isfinite(Bn)], 0.01), score, -np.inf))[::-1][:n_seeds]
    for seed in np.vstack([coords[thin], coords[np.argsort(score)[::-1][:n_seeds]],
                           sobol_cached(bounds, 64)]):
        try:
            res = minimize(lambda z: -(a_of(z) / W_of(z)) if W_of(z) > 0 else 1e12,
                           np.clip(seed, lo, hi), method="SLSQP", bounds=box, constraints=cons,
                           options={"maxiter": 100, "ftol": 1e-12})
            z = np.clip(res.x, lo, hi)
            v = ratio(z)
            if v > best:
                best, best_x = v, z
        except Exception:
            continue
    for seed in coords[np.argsort(rate)[::-1][:n_seeds]]:
        try:
            res = minimize(lambda z: -ratio(z), np.clip(seed, lo, hi), method="Nelder-Mead",
                           bounds=box, options={"maxiter": 2000, "fatol": 1e-14, "xatol": 1e-10})
            z = np.clip(res.x, lo, hi)
            v = ratio(z)
            if v > best:
                best, best_x = v, z
        except Exception:
            continue

    # the origin ray, from the symbolic Hessian
    import scipy.linalg as sl

    zero = np.zeros(n)
    P = np.asarray(fns["hessV"](*zero), float) / 2.0
    Bmat = np.array([[float(sp.Matrix(config.system().GSR(*xs))[i, k].subs({s: 0 for s in xs}))
                      for k in range(m)] for i in range(n)])
    Amat = np.array([[float(sp.diff(sp.Matrix(config.system().fSR(*xs))[i], xs[j]).subs({s: 0 for s in xs}))
                      for j in range(n)] for i in range(n)])
    origin_ratio = float("nan")
    Nsp = np.eye(n) if m == 0 else sl.null_space((Bmat.T @ P).reshape(m, n))   # no input: the whole space
    if Nsp.size:
        try:
            lam = sl.eigh(Nsp.T @ (P @ Amat + Amat.T @ P) @ Nsp, Nsp.T @ P @ Nsp, eigvals_only=True)
            origin_ratio = float(lam[-1])
        except Exception:
            pass

    kappa = -float(best)
    # 60-digit re-evaluation of the reported worst point
    mp_ratio = None
    try:
        import mpmath as mp

        mp.mp.dps = 60
        sub = {s: mp.mpf(float(v)) for s, v in zip(xs, best_x)}
        wm = sp.lambdify(xs, W_sym, "mpmath")(*[sub[s] for s in xs])
        am = sp.lambdify(xs, a_sym, "mpmath")(*[sub[s] for s in xs])
        bm = [sp.lambdify(xs, bk, "mpmath")(*[sub[s] for s in xs]) for bk in b_sym]
        nb = sum(abs(v) for v in bm) if l1 else mp.sqrt(sum(v * v for v in bm))
        mp_ratio = float((am - mp.mpf(rho) * nb) / wm)
    except Exception:
        pass
    return {
        "cv": c_v, "cv_w": cv, "cv_rad": cvr, "pd_valid": pd_valid,
        "cv_point": [float(v) for v in (cv_x if cv <= cvr else cvr_x)],
        "kappa": kappa, "worst_point": [float(v) for v in best_x],
        "worst_r": float(np.linalg.norm(best_x)), "origin_ratio": origin_ratio,
        "mp60_ratio_at_worst": mp_ratio, "numerically_verified": bool(pd_valid and kappa > config.KAPPA_MIN),
        "n_lattice": int(coords.shape[0]), "b_norm": config.B_NORM, "rho": rho,
    }


def main():
    expression, x0, _ = default_template()
    if ARGS.expression:
        expression = ARGS.expression
    if ARGS.constants:
        constants = [float(v) for v in ARGS.constants.split(",")]
    elif ARGS.best:
        best = None
        with open(ARGS.best) as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("final") and record.get("best_c"):
                    best = record["best_c"]
                if record.get("header") and record.get("expression"):
                    expression = record["expression"]
        if best is None:
            raise SystemExit(f"no final best_c in {ARGS.best}")
        constants = best
    else:
        constants = list(x0)

    from src.CPU.deterministic_search import deterministic_search
    from src.CPU.rule import fitness_fixed_constants

    fast = deterministic_search(expression, constants)
    ref = audit(expression, constants, ARGS.seeds)
    full = fitness_fixed_constants(expression, constants)
    out = {"constants": [float(v) for v in constants], "fitness": full["fitness"],
           "numerically_verified_fitness_path": full["numerically_verified"], "check": {k: fast.get(k) for k in
           ("cv", "kappa", "pd_valid", "worst_point", "worst_r", "origin_ratio")},
           "audit": ref,
           "diff": {"cv": fast["cv"] - ref["cv"],
                    "kappa": (fast["kappa"] - ref["kappa"]) if fast["kappa"] is not None else None}}
    print(json.dumps(out, indent=2, default=float))
    if ARGS.json:
        with open(ARGS.json, "w") as fh:
            json.dump(out, fh, indent=2, default=float)


if __name__ == "__main__":
    main()
