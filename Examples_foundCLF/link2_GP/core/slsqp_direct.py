"""scipy's SLSQP C core driven directly (vendored from examples/DET4Fast4D_V8/slsqp_direct.py).

Unchanged except that it already accepts any number of equality/inequality constraints, which is
what the multi-input manifold b_1 = ... = b_m = 0 needs.
"""
import numpy as np
from scipy.optimize._slsqplib import slsqp as _slsqp_core


def slsqp_direct(fun, jac, x0, bounds, cons_eq=(), cons_ineq=(), maxiter=60, ftol=1e-14):
    """fun(x) -> float, jac(x) -> (n,); cons_* = tuples of (fun, jac) with scalar constraints."""
    lb = np.asarray([b[0] for b in bounds], dtype=float)
    ub = np.asarray([b[1] for b in bounds], dtype=float)
    x = np.clip(np.asarray(x0, dtype=float).reshape(-1), lb, ub)
    n = x.shape[0]
    meq = len(cons_eq)
    mieq = len(cons_ineq)
    m = meq + mieq
    xl = lb.copy()
    xu = ub.copy()
    xl[~np.isfinite(lb)] = np.nan
    xu[~np.isfinite(ub)] = np.nan
    acc = float(ftol)
    state = {
        "acc": acc, "alpha": 0.0, "f0": 0.0, "gs": 0.0, "h1": 0.0, "h2": 0.0, "h3": 0.0,
        "h4": 0.0, "t": 0.0, "t0": 0.0, "tol": 10.0 * acc, "exact": 0, "inconsistent": 0,
        "reset": 0, "iter": 0, "itermax": int(maxiter), "line": 0, "m": m, "meq": meq,
        "mode": 0, "n": n,
    }
    indices = np.zeros([max(m + 2 * n + 2, 1)], dtype=np.int32)
    buffer_size = (
        n * (n + 1) // 2 + 3 * m * n - (m + 5 * n + 7) * meq + 9 * m + 8 * n * n + 35 * n
        + meq * meq + 28
    )
    if mieq == 0:
        buffer_size += 2 * n * (n + 1)
    buffer = np.zeros(max(buffer_size, 1), dtype=np.float64)
    mult = np.zeros([max(1, m + 2 * n + 2)], dtype=np.float64)
    C = np.zeros([max(1, m), n], dtype=np.float64, order="F")
    d = np.zeros([max(1, m)], dtype=np.float64)
    cons = tuple(cons_eq) + tuple(cons_ineq)

    def clipped(z):
        if (z < lb).any() or (z > ub).any():
            return np.clip(z, lb, ub)
        return z

    def eval_d(z):
        for i, (cf, _) in enumerate(cons):
            d[i] = cf(z)

    def eval_C(z):
        for i, (_, cj) in enumerate(cons):
            C[i, :] = cj(z)

    xc = clipped(x)
    fx = float(fun(xc))
    g = np.asarray(jac(xc), dtype=float).reshape(-1)
    eval_C(x)
    eval_d(x)
    while True:
        _slsqp_core(state, fx, g, C, d, x, mult, xl, xu, buffer, indices)
        mode = state["mode"]
        if mode == 1:
            xc = clipped(x)
            fx = float(fun(xc))
            eval_d(x)
        if mode == -1:
            xc = clipped(x)
            g = np.asarray(jac(xc), dtype=float).reshape(-1)
            eval_C(x)
        if abs(mode) != 1:
            break
    return x.copy(), fx, state["mode"], state["iter"]
