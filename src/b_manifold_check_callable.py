"""Callable-exact Artstein falsifier: complex-step gradients, no sympy.

Same architecture as ``src/b_manifold_check_exact.py`` (axis-aligned line
scans + fixed-seed random-direction lines + vectorized bisection on {b=0} +
punctured-domain SLSQP polish), but a = grad(V).f and b = grad(V).g are
computed by COMPLEX-STEP differentiation of the candidate's *numpy callable*:

    dV/dx_i = Im( V(x + i*h*e_i) ) / h,   h = 1e-30

which is exact to machine precision for analytic callables (the active GP
primitive set add/sub/mul/aq/neg/sin/exp is analytic) and needs no symbolic
processing at all. Benchmarked against the sympy-exact check on real
candidates: margins agree to 6-7 significant digits, runtime is independent
of tree size (no diff/lambdify build), so the sympy big-tree tail vanishes.
It also works directly on ``individual`` -- no ``ind2MSE`` needed, removing
the callable/expression mismatch hazard entirely.

Safety net: complex-step silently mis-differentiates non-analytic primitives
(abs, where-guards, prot_log). A startup self-check compares the CS gradient
against central finite differences at a few random points; on disagreement
the module returns status "cs_incompatible" so the caller can fall back to
the sympy-exact or FD check.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from src.b_manifold_check import BManifoldResult


def _v_batch(individual, consts, X):
    """V at X (M, n) -> (M,), real or complex."""
    out = np.asarray(individual(*X.T, consts))
    if out.shape == () or out.size == 1:
        return np.full(X.shape[0], complex(out.reshape(-1)[0])
                       if np.iscomplexobj(out) else float(out.reshape(-1)[0]))
    return out.reshape(X.shape[0])


def cs_grad_batch(individual, consts, X, h=1e-30):
    """Complex-step gradient at X (M, n) -> (M, n), machine precision."""
    M, n = X.shape
    G = np.empty((M, n))
    Z = X.astype(complex)
    for i in range(n):
        Zi = Z.copy()
        Zi[:, i] += 1j * h
        with np.errstate(all="ignore"):
            G[:, i] = np.imag(_v_batch(individual, consts, Zi)) / h
    return G


def _fd_grad_batch(individual, consts, X, step=1e-5):
    M, n = X.shape
    G = np.empty((M, n))
    for i in range(n):
        hh = step * np.maximum(1.0, np.abs(X[:, i]))
        Xp = X.copy(); Xm = X.copy()
        Xp[:, i] += hh; Xm[:, i] -= hh
        with np.errstate(all="ignore"):
            G[:, i] = (
                np.real(_v_batch(individual, consts, Xp))
                - np.real(_v_batch(individual, consts, Xm))
            ) / (2.0 * hh)
    return G


def _cs_selfcheck(individual, consts, bounds, tol, n_probe=8, seed=1):
    """CS vs FD agreement probe: guards against non-analytic primitives."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_probe, bounds.shape[0]))
    g_cs = cs_grad_batch(individual, consts, X)
    g_fd = _fd_grad_batch(individual, consts, X)
    scale = 1.0 + np.max(np.abs(g_fd))
    if not np.all(np.isfinite(g_cs)):
        return False
    return bool(np.max(np.abs(g_cs - g_fd)) / scale < tol)


def check_b_manifold_callable(
    individual,
    consts,
    f,
    G_col,
    bounds,
    gamma1=0.0,
    input_index=None,          # unused; G_col already selects the column
    scan_axes=(0, 1, 2, 3),
    scan_points=801,
    grid_points_per_axis=9,
    margin_tol=1e-12,
    origin_tol=1.1e-3,
    refine_iters=60,
    zero_tol=1e-14,
    random_lines=200,
    random_line_points=401,
    rng_seed=0,
    polish_top_k=40,
    polish_maxiter=60,
    polish_b_tol=1e-8,
    cs_selfcheck_tol=1e-5,
):
    """Artstein manifold falsifier on the candidate's numpy callable.

    f:     vectorized drift, f(x1..xn) -> (n, M)     (as in check_b_manifold)
    G_col: vectorized active input column -> (n, M)  (as in check_b_manifold)
    Flags roots (outside the origin ball) with a + gamma1 > margin_tol.
    """
    bounds = np.asarray(bounds, dtype=float)
    n = bounds.shape[0]

    try:
        if not _cs_selfcheck(individual, consts, bounds, cs_selfcheck_tol):
            return BManifoldResult(
                0, 0, np.nan, np.empty((0, n)), "cs_incompatible"
            )

        def ab(X):  # X (M, n) -> a (M,), b (M,)
            G = cs_grad_batch(individual, consts, X)
            fv = np.asarray(f(*X.T), dtype=float).reshape(n, X.shape[0]).T
            gv = np.asarray(G_col(*X.T), dtype=float).reshape(n, X.shape[0]).T
            return np.sum(G * fv, axis=1), np.sum(G * gv, axis=1)

        def b_only(X):
            return ab(X)[1]

        roots = []

        # --- stage 1: axis-aligned line scans ---
        for axis in scan_axes:
            others = [i for i in range(n) if i != axis]
            lines = [
                np.linspace(bounds[i, 0], bounds[i, 1], grid_points_per_axis)
                for i in others
            ]
            mesh = np.meshgrid(*lines, indexing="ij")
            combos = np.stack([m.ravel() for m in mesh], axis=1)  # (C, n-1)
            C = combos.shape[0]
            s = np.linspace(bounds[axis, 0], bounds[axis, 1], scan_points)
            X = np.empty((C * scan_points, n))
            for k, i in enumerate(others):
                X[:, i] = np.repeat(combos[:, k], scan_points)
            X[:, axis] = np.tile(s, C)
            bv = b_only(X).reshape(C, scan_points)

            zc, zs = np.nonzero(np.abs(bv) <= zero_tol)
            if zc.size:
                Z = np.empty((zc.size, n))
                for k, i in enumerate(others):
                    Z[:, i] = combos[zc, k]
                Z[:, axis] = s[zs]
                roots.append(Z)

            sg = np.sign(bv)
            ci, si = np.nonzero(sg[:, :-1] * sg[:, 1:] < 0)
            if ci.size:
                lo = s[si].astype(float)
                hi = s[si + 1].astype(float)
                blo = bv[ci, si]
                P = np.empty((ci.size, n))
                for k, i in enumerate(others):
                    P[:, i] = combos[ci, k]
                for _ in range(refine_iters):
                    mid = 0.5 * (lo + hi)
                    P[:, axis] = mid
                    bm = b_only(P)
                    left = np.sign(bm) * np.sign(blo) > 0
                    lo = np.where(left, mid, lo)
                    blo = np.where(left, bm, blo)
                    hi = np.where(left, hi, mid)
                P[:, axis] = 0.5 * (lo + hi)
                roots.append(P.copy())

        # --- stage 2: fixed-seed random-direction line scans ---
        if random_lines > 0:
            rng = np.random.default_rng(rng_seed)
            P0 = rng.uniform(bounds[:, 0], bounds[:, 1], (random_lines, n))
            D = rng.normal(size=(random_lines, n))
            D /= np.linalg.norm(D, axis=1, keepdims=True)
            with np.errstate(all="ignore"):
                tA = (bounds[:, 0][None] - P0) / D
                tB = (bounds[:, 1][None] - P0) / D
            t0 = np.where(np.abs(D) > 1e-12, np.minimum(tA, tB), -np.inf).max(1)
            t1 = np.where(np.abs(D) > 1e-12, np.maximum(tA, tB), np.inf).min(1)
            frac = np.linspace(0.0, 1.0, random_line_points)
            T = t0[:, None] + (t1 - t0)[:, None] * frac[None]
            X = (P0[:, None, :] + T[..., None] * D[:, None, :]).reshape(-1, n)
            bv = b_only(X).reshape(random_lines, random_line_points)
            sg = np.sign(bv)
            li, ti = np.nonzero(sg[:, :-1] * sg[:, 1:] < 0)
            if li.size:
                lo = T[li, ti].copy()
                hi = T[li, ti + 1].copy()
                blo = bv[li, ti]
                Pl, Dl = P0[li], D[li]
                for _ in range(refine_iters):
                    mid = 0.5 * (lo + hi)
                    bm = b_only(Pl + mid[:, None] * Dl)
                    left = np.sign(bm) * np.sign(blo) > 0
                    lo = np.where(left, mid, lo)
                    blo = np.where(left, bm, blo)
                    hi = np.where(left, hi, mid)
                roots.append(Pl + (0.5 * (lo + hi))[:, None] * Dl)

        if not roots:
            return BManifoldResult(0, 0, np.nan, np.empty((0, n)), "ok")

        R = np.concatenate(roots, axis=0)  # (K, n)

        # --- stage 3: punctured-domain SLSQP polish from the top-K roots ---
        if polish_top_k > 0 and len(R) > 0:
            a_seed, _ = ab(R)
            a_seed = np.where(np.isfinite(a_seed), a_seed, -np.inf)
            order = np.argsort(a_seed)[-int(polish_top_k):]
            r0sq = (origin_tol * (1.0 + 1e-6)) ** 2

            def a_scalar(z):
                return float(ab(z.reshape(1, n))[0][0])

            def b_scalar(z):
                return float(ab(z.reshape(1, n))[1][0])

            polished = []
            for idx in order:
                z0 = np.clip(R[idx], bounds[:, 0], bounds[:, 1])
                try:
                    res = minimize(
                        lambda z: -a_scalar(z),
                        z0,
                        constraints=[
                            {"type": "eq", "fun": b_scalar},
                            {"type": "ineq",
                             "fun": lambda z: float(z @ z) - r0sq},
                        ],
                        bounds=[tuple(bounds[i]) for i in range(n)],
                        method="SLSQP",
                        options={"maxiter": int(polish_maxiter),
                                 "ftol": 1e-12},
                    )
                    z = np.clip(res.x, bounds[:, 0], bounds[:, 1])
                    if (np.all(np.isfinite(z))
                            and abs(b_scalar(z)) <= polish_b_tol):
                        polished.append(z)
                except Exception:
                    continue
            if polished:
                R = np.concatenate([R, np.array(polished)], axis=0)

        a_all, _ = ab(R)
        margin = a_all + gamma1
        nonorigin = np.linalg.norm(R, axis=1) > origin_tol
        viol = nonorigin & np.isfinite(margin) & (margin > margin_tol)
        finite = margin[nonorigin & np.isfinite(margin)]
        margin_max = float(finite.max()) if finite.size else np.nan

        return BManifoldResult(
            n_roots=int(nonorigin.sum()),
            n_violations=int(viol.sum()),
            margin_max=margin_max,
            violation_points=R[viol],
            status="ok",
        )
    except Exception as exc:
        return BManifoldResult(
            0, 0, np.nan, np.empty((0, n)),
            f"error: {type(exc).__name__}: {exc}"
        )
