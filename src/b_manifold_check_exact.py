"""Exact-gradient (symbolic) version of the b=0 manifold check.

Same idea as ``src/b_manifold_check.py`` — solve {b(x)=0} and require
a + gamma1 <= 0 at every retained root — but a and b are built from the
candidate's *sympy expression* with exact derivatives instead of finite
differences on a callable, so there is no fd_step error.

Three stages:

1. **Axis-aligned line scans** (as before): bracket sign changes of b along
   axis-parallel lines over a combo grid, refine by vectorized bisection,
   keep exact zeros at scan nodes (flat valleys).
2. **Random-direction line scans** (coverage fix): a 3-D branch of {b=0} can
   be parallel to every axis-aligned scan line (the "corner monster" failure
   mode), but a generic line intersects it transversally. A fixed-seed set of
   random lines through random box points closes that blind spot
   deterministically (same lines every call).
3. **Root-seeded constrained ascent** (accuracy fix): the scan roots are exact
   points ON the manifold; SLSQP then maximizes a(x) subject to b(x)=0 inside
   the box from the top-K roots, with exact jacobians (one extra sympy diff
   layer). Multi-start local ascent from globally-scattered roots ~ global
   optimization of the margin on the manifold — no sampling luck involved.
"""

from sympy import Matrix, diff, lambdify, symbols

import numpy as np
from scipy.optimize import minimize

from src.b_manifold_check import BManifoldResult


def check_b_manifold_exact(
    V_expr,
    fSR,
    GSR,
    bounds,
    decay_rate=0.0012,
    gamma1=0.0,
    input_index=1,
    scan_axes=(2, 3),
    scan_points=801,
    grid_points_per_axis=9,
    margin_tol=0.0,
    origin_tol=1e-9,
    refine_iters=60,
    zero_tol=1e-14,
    random_lines=200,
    random_line_points=401,
    rng_seed=0,
    polish_top_k=40,
    polish_maxiter=60,
    polish_b_tol=1e-8,
):
    """Exact-gradient manifold check on a sympy candidate expression.

    fSR / GSR: the symbolic dynamics used elsewhere (return sympy Matrix).
    ``decay_rate`` is retained for API compatibility only. ``random_lines=0``
    and/or ``polish_top_k=0`` disable the respective upgrade stages.
    """
    bounds = np.asarray(bounds, dtype=float)
    n = bounds.shape[0]

    try:
        x_syms = symbols(f"x1:{n + 1}")
        grad = Matrix([diff(V_expr, s) for s in x_syms])
        f_vec = fSR(*x_syms)
        G_mat = GSR(*x_syms)
        a_expr = (grad.T * f_vec)[0]
        b_expr = sum(grad[i] * G_mat[i, input_index] for i in range(n))
        # The state-dependent decay strengthening is intentionally not part of
        # the Artstein check. ``decay_rate`` remains for API compatibility.
        margin_expr = a_expr + gamma1

        b_fn = lambdify(x_syms, b_expr, "numpy")
        m_fn = lambdify(x_syms, margin_expr, "numpy")

        def b_batch(X):
            with np.errstate(all="ignore"):
                out = np.asarray(b_fn(*X), dtype=float)
            if out.shape != X.shape[1:]:
                out = np.broadcast_to(out, X.shape[1:]).copy()
            return out

        root_list = []

        # --- stage 1: axis-aligned line scans -------------------------------
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
            bv = b_batch(X).reshape(C, scan_points)

            zc, zs = np.nonzero(np.abs(bv) <= zero_tol)
            if zc.size:
                Z = np.empty((n, zc.size))
                for k, i in enumerate(others):
                    Z[i] = combos[k, zc]
                Z[axis] = s[zs]
                root_list.append(Z)

            sg = np.sign(bv)
            ci, si = np.nonzero(sg[:, :-1] * sg[:, 1:] < 0)
            if ci.size:
                lo = s[si].astype(float)
                hi = s[si + 1].astype(float)
                b_lo = bv[ci, si]
                Xm = np.empty((n, ci.size))
                for k, i in enumerate(others):
                    Xm[i] = combos[k, ci]
                for _ in range(refine_iters):
                    mid = 0.5 * (lo + hi)
                    Xm[axis] = mid
                    bm = b_batch(Xm)
                    left = np.sign(bm) * np.sign(b_lo) > 0
                    lo = np.where(left, mid, lo)
                    b_lo = np.where(left, bm, b_lo)
                    hi = np.where(left, hi, mid)
                Xm[axis] = 0.5 * (lo + hi)
                root_list.append(Xm.copy())

        # --- stage 2: random-direction line scans (fixed seed) --------------
        if random_lines > 0:
            rng = np.random.default_rng(rng_seed)
            P0 = rng.uniform(bounds[:, 0], bounds[:, 1], size=(random_lines, n))
            D = rng.normal(size=(random_lines, n))
            D /= np.linalg.norm(D, axis=1, keepdims=True)

            # per-line parameter interval [t0, t1] staying inside the box
            with np.errstate(divide="ignore", invalid="ignore"):
                tA = (bounds[:, 0][None, :] - P0) / D
                tB = (bounds[:, 1][None, :] - P0) / D
            t_low = np.where(np.abs(D) > 1e-12, np.minimum(tA, tB), -np.inf)
            t_high = np.where(np.abs(D) > 1e-12, np.maximum(tA, tB), np.inf)
            t0 = t_low.max(axis=1)
            t1 = t_high.min(axis=1)

            frac = np.linspace(0.0, 1.0, random_line_points)
            T = t0[:, None] + (t1 - t0)[:, None] * frac[None, :]  # (L, P)
            X = (P0[:, None, :] + T[..., None] * D[:, None, :])  # (L, P, n)
            bv = b_batch(X.reshape(-1, n).T).reshape(random_lines,
                                                     random_line_points)

            sg = np.sign(bv)
            li, ti = np.nonzero(sg[:, :-1] * sg[:, 1:] < 0)
            if li.size:
                lo = T[li, ti].astype(float)
                hi = T[li, ti + 1].astype(float)
                b_lo = bv[li, ti]
                P = P0[li]
                Dd = D[li]
                for _ in range(refine_iters):
                    mid = 0.5 * (lo + hi)
                    Xm = P + mid[:, None] * Dd
                    bm = b_batch(Xm.T)
                    left = np.sign(bm) * np.sign(b_lo) > 0
                    lo = np.where(left, mid, lo)
                    b_lo = np.where(left, bm, b_lo)
                    hi = np.where(left, hi, mid)
                root_list.append((P + (0.5 * (lo + hi))[:, None] * Dd).T)

        if not root_list:
            return BManifoldResult(0, 0, np.nan, np.empty((0, n)), "ok")

        R = np.concatenate(root_list, axis=1)  # (n, K)

        # --- stage 3: root-seeded constrained ascent (SLSQP polish) ---------
        if polish_top_k > 0 and R.shape[1] > 0:
            a_fn = lambdify(x_syms, a_expr, "numpy")
            ga_fn = lambdify(x_syms, [diff(a_expr, s) for s in x_syms], "numpy")
            gb_fn = lambdify(x_syms, [diff(b_expr, s) for s in x_syms], "numpy")

            with np.errstate(all="ignore"):
                m_seed = np.asarray(m_fn(*R), dtype=float)
            if m_seed.shape != (R.shape[1],):
                m_seed = np.broadcast_to(m_seed, (R.shape[1],)).copy()
            m_seed = np.where(np.isfinite(m_seed), m_seed, -np.inf)
            order = np.argsort(m_seed)[-int(polish_top_k):]

            box = [tuple(bounds[i]) for i in range(n)]
            # the ascent must respect the punctured domain: without the ball
            # constraint every VALID candidate drains to the origin (sup of a
            # on {b=0} is 0 there) and the polish teaches nothing
            r0_sq = (origin_tol * (1.0 + 1e-6)) ** 2
            polished = []
            for idx in order:
                x0 = np.clip(R[:, idx], bounds[:, 0], bounds[:, 1])
                try:
                    res = minimize(
                        lambda z: -float(a_fn(*z)),
                        x0,
                        jac=lambda z: -np.asarray(ga_fn(*z), dtype=float).ravel(),
                        constraints=[
                            {
                                "type": "eq",
                                "fun": lambda z: float(b_fn(*z)),
                                "jac": lambda z: np.asarray(
                                    gb_fn(*z), dtype=float
                                ).ravel(),
                            },
                            {
                                "type": "ineq",
                                "fun": lambda z: float(z @ z) - r0_sq,
                                "jac": lambda z: 2.0 * z,
                            },
                        ],
                        bounds=box,
                        method="SLSQP",
                        options={"maxiter": int(polish_maxiter),
                                 "ftol": 1e-12},
                    )
                    z = np.clip(res.x, bounds[:, 0], bounds[:, 1])
                    if (
                        np.all(np.isfinite(z))
                        and abs(float(b_fn(*z))) <= polish_b_tol
                    ):
                        polished.append(z)
                except Exception:
                    continue
            if polished:
                R = np.concatenate([R, np.array(polished).T], axis=1)

        with np.errstate(all="ignore"):
            margin = np.asarray(m_fn(*R), dtype=float)
        if margin.shape != (R.shape[1],):
            margin = np.broadcast_to(margin, (R.shape[1],)).copy()

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
