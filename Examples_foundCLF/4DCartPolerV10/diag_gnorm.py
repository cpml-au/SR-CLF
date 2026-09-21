"""Diagnostics (NOT priced) of every generation champion of a V9 run: the |x|^2-referenced numbers.

    python diag_gnorm.py <job>_best_per_generation.jsonl        (a CPU node: ~10-30 s per distinct champion)

2026-09-18 (the user's order, from the LQR-vs-CLF study ~/symclf_tmp/lqr_roa): on the 41^4 box grid (origin
excluded), with the project's controller (src/SymVVdot_Calculations.py, Q = I, R = 1e-4):
    Vdot = -sqrt(a^2 + |x|^2 * 1e4 * b^2)            a = grad V . f, b = grad V . G
    gamma      = min(-Vdot / |x|^2)
    lambda_max = max(W / |x|^2),  c_v_grid = min(W / |x|^2),  W = V - V(0)
    g_norm     = gamma / lambda_max                   (scale-free; the ARE + LQR has 0.0159)
    kappa_x    = min(-(a - rho|b|) / |x|^2)           (DET4 with W -> |x|^2; the ARE has 1.000)
next to the run's own numbers from the record (fitness, det4 c_v*, kappa*, certified). The box is SYMCLF_V2_BOX.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT]

from sympy import symbols, lambdify, Matrix  # noqa: E402
from SystemDynamicsSR import fSR, GSR  # noqa: E402
from src.SymFunctions import substitute_paramsCoef, DeapSimplifier  # noqa: E402

H = np.array([float(v) for v in os.environ.get("SYMCLF_V2_BOX", "0.25,0.25,0.0494,0.1155").split(",")])
RHO = float(os.environ.get("SYMCLF_DET4_RHO", "1000"))
R_INV = 1.0e4


def grid():
    axes = [np.round(np.linspace(-h, h, 41), 12) for h in H]
    X = np.stack([m.ravel() for m in np.meshgrid(*axes, indexing="ij")])
    return X[:, np.linalg.norm(X, axis=0) > 1e-9]


def numbers(expr, consts, X, r2):
    xs = symbols("x1 x2 x3 x4")
    V = DeapSimplifier(substitute_paramsCoef(expr, consts), should_print=False)
    g = Matrix([V.diff(s) for s in xs])
    Vf = lambdify(xs, V, "numpy")
    af = lambdify(xs, (g.T * fSR(*xs))[0], "numpy")
    bf = lambdify(xs, (g.T * GSR(*xs))[0, 1], "numpy")
    W = np.asarray(Vf(*X), float) - float(Vf(0, 0, 0, 0))
    a = np.asarray(af(*X), float) * np.ones(X.shape[1])
    b = np.asarray(bf(*X), float) * np.ones(X.shape[1])
    vdot = -np.sqrt(a * a + r2 * R_INV * b * b)
    gamma = float(np.min(-vdot / r2))
    lam = float(np.max(W / r2))
    return {"gamma": gamma, "lambda_max": lam, "g_norm": gamma / lam if lam > 0 else float("nan"),
            "kappa_x": float(np.min(-(a - RHO * np.abs(b)) / r2)), "c_v_grid": float(np.min(W / r2))}


def main():
    path = sys.argv[1]
    recs = [json.loads(l) for l in open(path) if l.strip()]
    X = grid()
    r2 = np.sum(X * X, axis=0)
    print(f"{os.path.basename(path)} | box {H.tolist()} | rho {RHO} | {len(recs)} generations | ARE reference: "
          "g_norm 0.0159, kappa_x 1.000, gamma 0.996, kappa* 0.724, c_v* 0.044")
    print(f"{'gen':>4s} {'fitness':>10s} {'cert':>5s} {'kappa*':>8s} {'c_v*':>8s} | {'g_norm':>9s} {'gamma':>9s} "
          f"{'lam_max':>8s} {'kappa_x':>9s} {'c_v_grid':>9s} | expression")
    cache = {}
    for r in recs:
        key = (r["expression"], tuple(r.get("constants", [])))
        if key not in cache:
            try:
                cache[key] = numbers(r["expression"], list(r.get("constants", [])), X, r2)
            except Exception as exc:  # a champion the grid cannot evaluate: report, do not stop
                cache[key] = {"error": f"{type(exc).__name__}: {exc}"}
        d = cache[key]
        bd = r.get("breakdown") or {}
        d4 = bd.get("det4") or {}
        head = (f"{r.get('generation', '?'):>4} {r.get('fitness', float('nan')):10.4g} {str(bd.get('det4_certified')):>5s} "
                f"{d4.get('kappa', float('nan')):8.4g} {d4.get('cv', float('nan')):8.4g} | ")
        if "error" in d:
            print(head + d["error"])
        else:
            print(head + f"{d['g_norm']:9.3e} {d['gamma']:9.4g} {d['lambda_max']:8.4g} {d['kappa_x']:9.4g} "
                         f"{d['c_v_grid']:9.4g} | {r['expression'][:60]}", flush=True)


if __name__ == "__main__":
    main()
