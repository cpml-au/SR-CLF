"""Solve {b(x) = 0} directly and check the Artstein condition a(x) < 0 on it.

For unbounded input the CLF condition is exactly

    b(x) = 0  =>  a(x) < 0        for all x != 0,

with a = grad(V).f and b = grad(V).g (g = active column of G). Sampling
falsifiers see this manifold only through the barrier a - rho*|b|, whose
violation funnel has width ~ a/rho, so degenerate candidates (flat
valleys with a = b = 0 on a whole plane) can hide from them. This module
finds the manifold directly: it scans axis-aligned lines, brackets sign
changes of b, refines each root by vectorized bisection, keeps exact
zeros of b at scan nodes (flat valleys), and evaluates the strict margin

    a + gamma1  >  margin_tol   ->  violation

at every root away from the origin.

All V evaluations go through the candidate's callable; a vectorized
(numpy lambdified) callable is used as-is, a scalar-only callable is
looped over as a fallback.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BManifoldResult:
    n_roots: int
    n_violations: int
    margin_max: float
    violation_points: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    status: str = "ok"


def _eval_v_batch(individual, consts, X):
    """V at points X of shape (n_states, M) -> (M,)."""
    M = X.shape[1]
    try:
        out = np.asarray(individual(*X, consts), dtype=float)
        if out.shape == () or out.size == 1:
            return np.full(M, float(out.reshape(-1)[0]) if out.size else np.nan)
        return out.reshape(M)
    except Exception:
        vals = np.empty(M)
        for j in range(M):
            try:
                vals[j] = float(np.asarray(individual(*X[:, j], consts)).reshape(-1)[0])
            except Exception:
                vals[j] = np.nan
        return vals


def _fd_grad_batch(individual, consts, X, step):
    """Central-difference gradient at X (n_states, M) -> (n_states, M)."""
    n, _ = X.shape
    grad = np.empty_like(X)
    for i in range(n):
        h = step * np.maximum(1.0, np.abs(X[i]))
        Xp = X.copy()
        Xm = X.copy()
        Xp[i] = X[i] + h
        Xm[i] = X[i] - h
        grad[i] = (
            _eval_v_batch(individual, consts, Xp)
            - _eval_v_batch(individual, consts, Xm)
        ) / (2.0 * h)
    return grad


def _b_batch(individual, consts, G_col, X, step):
    grad = _fd_grad_batch(individual, consts, X, step)
    g = np.asarray(G_col(*X), dtype=float).reshape(X.shape)
    return np.sum(grad * g, axis=0)


def check_b_manifold(
    individual,
    consts,
    f,
    G_col,
    bounds,
    decay_rate=0.0012,
    gamma1=0.0,
    fd_step=1e-5,
    scan_axes=(2, 3),
    scan_points=401,
    grid_points_per_axis=7,
    margin_tol=0.0,
    origin_tol=1e-9,
    refine_iters=30,
    zero_tol=1e-12,
):
    """Find b = 0 roots and flag the constant-margin Artstein condition.

    f:     vectorized drift, f(x1..xn) -> (n_states, M)
    G_col: vectorized active input column, G_col(x1..xn) -> (n_states, M)

    ``decay_rate`` is retained for API compatibility but is deliberately not
    used in this Artstein check.  Use ``gamma1`` for the strict margin on the
    punctured verification domain.
    """
    bounds = np.asarray(bounds, dtype=float)
    n = bounds.shape[0]

    try:
        root_list = []
        for axis in scan_axes:
            others = [i for i in range(n) if i != axis]
            lines = [
                np.linspace(bounds[i, 0], bounds[i, 1], grid_points_per_axis)
                for i in others
            ]
            mesh = np.meshgrid(*lines, indexing="ij")
            combos = np.stack([m.ravel() for m in mesh])  # (n-1, C)
            C = combos.shape[1]
            s = np.linspace(bounds[axis, 0], bounds[axis, 1], scan_points)

            X = np.empty((n, C * scan_points))
            for k, i in enumerate(others):
                X[i] = np.repeat(combos[k], scan_points)
            X[axis] = np.tile(s, C)

            bv = _b_batch(individual, consts, G_col, X, fd_step)
            bv = bv.reshape(C, scan_points)

            # exact zeros at scan nodes (flat valleys / degenerate planes)
            zc, zs = np.nonzero(np.abs(bv) <= zero_tol)
            if zc.size:
                Z = np.empty((n, zc.size))
                for k, i in enumerate(others):
                    Z[i] = combos[k, zc]
                Z[axis] = s[zs]
                root_list.append(Z)

            # bracketed sign changes, refined by vectorized bisection
            sg = np.sign(bv)
            ci, si = np.nonzero(sg[:, :-1] * sg[:, 1:] < 0)
            if ci.size:
                lo = s[si].astype(float)
                hi = s[si + 1].astype(float)
                b_lo = bv[ci, si]
                fixed = combos[:, ci]  # (n-1, K)
                Xm = np.empty((n, ci.size))
                for k, i in enumerate(others):
                    Xm[i] = fixed[k]
                for _ in range(refine_iters):
                    mid = 0.5 * (lo + hi)
                    Xm[axis] = mid
                    bm = _b_batch(individual, consts, G_col, Xm, fd_step)
                    left = np.sign(bm) * np.sign(b_lo) > 0
                    lo = np.where(left, mid, lo)
                    b_lo = np.where(left, bm, b_lo)
                    hi = np.where(left, hi, mid)
                Xm[axis] = 0.5 * (lo + hi)
                root_list.append(Xm.copy())

        if not root_list:
            return BManifoldResult(0, 0, np.nan, np.empty((0, n)), "ok")

        R = np.concatenate(root_list, axis=1)  # (n, K)
        grad = _fd_grad_batch(individual, consts, R, fd_step)
        fv = np.asarray(f(*R), dtype=float).reshape(R.shape)
        a = np.sum(grad * fv, axis=0)
        # margin = a + decay_rate * np.sum(R**2, axis=0)  # intentionally off
        margin = a + gamma1

        nonorigin = np.sqrt(np.sum(R**2, axis=0)) > origin_tol
        viol = nonorigin & np.isfinite(margin) & (margin > margin_tol)

        finite = margin[nonorigin & np.isfinite(margin)]
        margin_max = float(finite.max()) if finite.size else np.nan

        return BManifoldResult(
            n_roots=int(nonorigin.sum()),
            n_violations=int(viol.sum()),
            margin_max=margin_max,
            violation_points=R[:, viol].T,
            status="ok",
        )
    except Exception as exc:
        return BManifoldResult(
            0, 0, np.nan, np.empty((0, n)), f"error: {type(exc).__name__}: {exc}"
        )
