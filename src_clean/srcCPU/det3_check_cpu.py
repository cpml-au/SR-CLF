"""DET3 on the CPU: the EXPONENTIAL bounded-input Artstein condition, searched deterministically.

The CPU counterpart of examples/VerifierBenchmark/run_det3_gpu.py (built on run_det2.py's
functions); kappa = 0 reproduces DET2, the deterministic DE2 barrier check.

THE CONDITION (conditions.py, condition 3)

    vdot(x) := a(x) - rho*|b(x)| + gamma1          W(x) := V(x) - V(0)
    for all x in box, ||x|| > ORIGIN_TOL :
        vdot <= -kappa*W  (non-strict)   /   vdot < -kappa*W  (strict)
    margin  m(x) := a(x) - rho*|b(x)| + kappa*W(x) + gamma1
        non-strict VALID iff max m <= 0            strict VALID iff max m <= -MARGIN_TOL

THE SEARCH -- lattice + polishes, all deterministic, no DE

  1. LATTICE. scan_geometry with the 21^n mesh: the same 2 610 397 points (n = 4) the PD check
     and, up to the random-line seed, the exact Artstein scan use.  a, b and V are evaluated
     ONCE per candidate; m for every (rho, kappa) is arithmetic on those arrays.
  2. SEEDS per (rho, kappa): the top-K lattice points by m (K = 40).
  3. MANIFOLD ASCENT per kappa: maximise a + kappa*W subject to b = 0 (SLSQP, exact jacobians)
     from a rho-independent seed set -- top-K by a + kappa*W among the 1% thinnest |b|, top-K
     overall, 64 scrambled Sobol starts; converged points with |b| <= B_TOL are barrier points.
  4. THE EXACT ARTSTEIN CHECK, folded in: its violation points are scored as candidates for m.
  5. POLISH per (rho, kappa) from every seed, keeping the best:
       (a) piecewise-smooth SLSQP on the half-space s*b >= 0 (s = sign b(seed)) -- the barrier
           has a kink on b = 0 where its max usually sits, and that boundary IS the manifold;
       (b) Nelder-Mead on m itself, the origin ball and the box as hard walls.
     margin_max = max over lattice U polished U manifold U exact-check points, outside the ball.

PD is check_positive_definite_cpu, once per candidate, shared by every (rho, kappa).

The host/CPU stages (lattice_stage, manifold_seed_set, manifold_ascent, polish,
det3_from_lattice) are shared with src/srcGPU/det3_check_gpu.py, which only replaces the
lattice evaluation of a, b, V and the exact/PD checks by their GPU versions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from src.srcCPU import conditions as C
from src.srcCPU.exact_check_cpu import (
    BManifoldResult,
    PDResult,
    check_b_manifold_exact,
    check_positive_definite_cpu,
    eval_lattice,
    scan_geometry,
)
from src.srcCPU.symbolic import (
    ab_expressions,
    polish_callables,
    state_symbols,
    sympy_expression,
)

#: the benchmark lattice (scan_axes=None -> every axis)
LATTICE = dict(
    scan_points=801,
    grid_points_per_axis=9,
    random_lines=200,
    random_line_points=401,
    rng_seed=0,
    mesh_points_per_axis=21,
)
POLISH_TOP_K = 40
POLISH_MAXITER = 60
SOBOL_SEEDS = 64


@dataclass
class DET3Row:
    rho: float
    kappa: float
    margin_max: float | None
    lattice_margin_max: float | None
    worst_abs_b: float | None
    worst_W: float | None
    n_lattice_violations: int
    n_polish_seeds: int
    n_manifold_points: int
    artstein_valid: bool  # strict, DEFINITE: max m <= -MARGIN_TOL
    artstein_valid_nonstrict: bool  # max m <= 0
    manifold_s: float = 0.0
    polish_s: float = 0.0


@dataclass
class DET3Result:
    rows: dict = field(default_factory=dict)  # (rho, kappa) -> DET3Row
    exact: BManifoldResult | None = None  # the exact Artstein check, folded in
    pd: PDResult | None = None
    pd_valid: bool | None = None
    status: str = "ok"
    build_s: float = 0.0
    scan_s: float = 0.0
    exact_s: float = 0.0
    pd_s: float = 0.0

    def row(self, rho, kappa):
        return self.rows[(float(rho), float(kappa))]

    @property
    def exact_margin_max(self):
        if self.exact is None or self.exact.margin_max != self.exact.margin_max:
            return None
        return float(self.exact.margin_max)


# --------------------------------------------------------------------------
# host stage: seeds from the lattice arrays
# --------------------------------------------------------------------------


def lattice_stage(
    A,
    B,
    W,
    keep,
    coords,
    rho,
    kappa,
    gamma1=C.DEFAULT_GAMMA1,
    top_k=POLISH_TOP_K,
    margin_tol=C.MARGIN_TOL,
):
    """m on the lattice for one (rho, kappa): top-K seeds, lattice max, lattice violations."""
    with np.errstate(all="ignore"):
        m = A - rho * np.abs(B) + kappa * W + gamma1
    m = np.where(keep & np.isfinite(m), m, -np.inf)
    n_viol = int(np.sum(m > -margin_tol))
    order = np.argsort(m)[::-1][: int(top_k)]
    order = order[np.isfinite(m[order])]
    return coords[order], (float(m[order[0]]) if order.size else -np.inf), n_viol


def manifold_seed_set(
    A, B, W, keep, coords, kappa, bounds, top_k=POLISH_TOP_K, sobol_seeds=SOBOL_SEEDS
):
    """rho-independent seeds for the manifold ascent: top-K by a + kappa*W among the 1%
    thinnest |b|, top-K overall, plus scrambled Sobol starts (fixed seed)."""
    from scipy.stats import qmc

    bounds = np.asarray(bounds, dtype=float)
    n = bounds.shape[0]
    score = np.where(
        keep & np.isfinite(A) & np.isfinite(B) & np.isfinite(W), A + kappa * W, -np.inf
    )
    absb = np.where(np.isfinite(B), np.abs(B), np.inf)
    thin = (
        (absb <= np.quantile(absb[np.isfinite(absb)], 0.01))
        if np.isfinite(absb).any()
        else np.zeros_like(keep)
    )
    f1 = np.argsort(np.where(thin, score, -np.inf))[::-1][: int(top_k)]
    f2 = np.argsort(score)[::-1][: int(top_k)]
    seeds = [
        coords[i] for i in np.unique(np.concatenate([f1, f2])) if np.isfinite(score[i])
    ]
    if int(sobol_seeds) > 0:
        seeds += list(
            qmc.scale(
                qmc.Sobol(d=n, scramble=True, seed=0).random(int(sobol_seeds)),
                bounds[:, 0],
                bounds[:, 1],
            )
        )
    return np.asarray(seeds, dtype=float).reshape(-1, n)


# --------------------------------------------------------------------------
# CPU stage: SciPy on the sympy-lambdified callables (polish_callables)
# --------------------------------------------------------------------------


def _ball(origin_tol):
    r0_sq = (origin_tol * (1.0 + 1e-6)) ** 2
    return r0_sq, {
        "type": "ineq",
        "fun": lambda z: float(z @ z) - r0_sq,
        "jac": lambda z: 2.0 * z,
    }


def manifold_ascent(
    fns,
    kappa,
    seeds,
    bounds,
    origin_tol=C.ORIGIN_TOL,
    b_tol=C.B_TOL,
    maxiter=POLISH_MAXITER,
):
    """max a + kappa*W  s.t. b = 0, from every seed; keeps points with |b| <= b_tol outside
    the origin ball."""
    bounds = np.asarray(bounds, dtype=float)
    n = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    box = list(zip(lo, hi))
    r0_sq, ball = _ball(origin_tol)
    a, ga, b, gb, V, gV, v0 = (fns[k] for k in ("a", "ga", "b", "gb", "V", "gV", "v0"))
    out = []
    for z0 in np.asarray(seeds, dtype=float).reshape(-1, n):
        z0 = np.clip(z0, lo, hi)
        try:
            res = minimize(
                lambda z: -(a(z) + kappa * (V(z) - v0)),
                z0,
                jac=lambda z: -(ga(z) + kappa * gV(z)),
                constraints=[{"type": "eq", "fun": b, "jac": gb}, ball],
                bounds=box,
                method="SLSQP",
                options={"maxiter": int(maxiter), "ftol": 1e-12},
            )
            z = np.clip(res.x, lo, hi)
            if np.all(np.isfinite(z)) and float(z @ z) > r0_sq and abs(b(z)) <= b_tol:
                out.append(z)
        except Exception:
            continue
    return np.asarray(out) if out else np.empty((0, n))


def polish(
    fns,
    seeds,
    lattice_best,
    rho,
    kappa,
    gamma1,
    extra_points,
    bounds,
    origin_tol=C.ORIGIN_TOL,
    maxiter=POLISH_MAXITER,
):
    """Per (rho, kappa): score the extra points, then piecewise-smooth SLSQP and Nelder-Mead
    on m from every seed; returns the max and where it sits."""
    bounds = np.asarray(bounds, dtype=float)
    n = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    box = list(zip(lo, hi))
    r0_sq, ball = _ball(origin_tol)
    a, ga, b, gb, V, gV, v0 = (fns[k] for k in ("a", "ga", "b", "gb", "V", "gV", "v0"))
    seeds = np.asarray(seeds, dtype=float).reshape(-1, n)
    best, best_x = float(lattice_best), (seeds[0] if seeds.shape[0] else None)

    def m_of(z):
        v = a(z) - rho * abs(b(z)) + kappa * (V(z) - v0) + gamma1
        return v if np.isfinite(v) else -np.inf

    def m_wall(z):
        z = np.asarray(z, dtype=float)
        if float(z @ z) <= r0_sq or np.any(z < lo) or np.any(z > hi):
            return np.inf
        v = m_of(z)
        return -v if np.isfinite(v) else np.inf

    def consider(z):
        nonlocal best, best_x
        z = np.clip(np.asarray(z, dtype=float), lo, hi)
        if not np.all(np.isfinite(z)) or float(z @ z) <= r0_sq:
            return
        v = m_of(z)
        if np.isfinite(v) and v > best:
            best, best_x = float(v), z

    for z in np.asarray(extra_points, dtype=float).reshape(-1, n):
        consider(z)
    for z0 in seeds:
        z0 = np.clip(z0, lo, hi)
        s = 1.0 if b(z0) >= 0.0 else -1.0
        try:  # (a) piecewise-smooth SLSQP on the half-space s*b >= 0
            res = minimize(
                lambda z: -(a(z) - rho * s * b(z) + kappa * (V(z) - v0)),
                z0,
                jac=lambda z: -(ga(z) - rho * s * gb(z) + kappa * gV(z)),
                constraints=[
                    {
                        "type": "ineq",
                        "fun": lambda z: s * b(z),
                        "jac": lambda z: s * gb(z),
                    },
                    ball,
                ],
                bounds=box,
                method="SLSQP",
                options={"maxiter": int(maxiter), "ftol": 1e-12},
            )
            consider(res.x)
        except Exception:
            pass
        try:  # (b) Nelder-Mead on m itself
            res = minimize(
                m_wall,
                z0,
                method="Nelder-Mead",
                options={"maxiter": 400, "xatol": 1e-10, "fatol": 1e-14},
            )
            consider(res.x)
        except Exception:
            pass
    wb = abs(b(best_x)) if best_x is not None else None
    ww = (V(best_x) - v0) if best_x is not None else None
    return {
        "margin_max": None if not np.isfinite(best) else best,
        "lattice_margin_max": (
            None if not np.isfinite(lattice_best) else float(lattice_best)
        ),
        "worst_abs_b": None if wb is None or not np.isfinite(wb) else wb,
        "worst_W": None if ww is None or not np.isfinite(ww) else ww,
        "n_polish_seeds": int(seeds.shape[0]),
    }


def det3_from_lattice(
    fns,
    A,
    B,
    W,
    coords,
    keep,
    bounds,
    rhos=C.DEFAULT_RHOS,
    kappas=C.DEFAULT_KAPPAS,
    gamma1=C.DEFAULT_GAMMA1,
    extra_points=None,
    origin_tol=C.ORIGIN_TOL,
    b_tol=C.B_TOL,
    margin_tol=C.MARGIN_TOL,
    top_k=POLISH_TOP_K,
    maxiter=POLISH_MAXITER,
    sobol_seeds=SOBOL_SEEDS,
):
    """Everything after the lattice arrays: seeds, manifold ascent per kappa, polish per
    (rho, kappa).  ``extra_points`` = the exact check's violation points (or None).
    Returns {(rho, kappa): DET3Row}."""
    n = np.asarray(bounds).shape[0]
    extra = (
        np.empty((0, n))
        if extra_points is None
        else np.asarray(extra_points, float).reshape(-1, n)
    )
    manifold, n_m, man_s = {}, {}, {}
    for kappa in kappas:
        t0 = time.perf_counter()
        pts = manifold_ascent(
            fns,
            kappa,
            manifold_seed_set(A, B, W, keep, coords, kappa, bounds, top_k, sobol_seeds),
            bounds,
            origin_tol,
            b_tol,
            maxiter,
        )
        manifold[kappa] = np.vstack([pts, extra]) if extra.size else pts
        n_m[kappa], man_s[kappa] = int(pts.shape[0]), time.perf_counter() - t0
    rows = {}
    for rho in rhos:
        for kappa in kappas:
            t0 = time.perf_counter()
            seeds, lattice_best, n_viol = lattice_stage(
                A, B, W, keep, coords, rho, kappa, gamma1, top_k, margin_tol
            )
            r = polish(
                fns,
                seeds,
                lattice_best,
                rho,
                kappa,
                gamma1,
                manifold[kappa],
                bounds,
                origin_tol,
                maxiter,
            )
            m = r["margin_max"]
            rows[(float(rho), float(kappa))] = DET3Row(
                rho=float(rho),
                kappa=float(kappa),
                margin_max=m,
                lattice_margin_max=r["lattice_margin_max"],
                worst_abs_b=r["worst_abs_b"],
                worst_W=r["worst_W"],
                n_lattice_violations=n_viol,
                n_polish_seeds=r["n_polish_seeds"],
                n_manifold_points=n_m[kappa],
                artstein_valid=(
                    C.artstein_valid(m, margin_tol) if m is not None else False
                ),
                artstein_valid_nonstrict=(
                    C.nonstrict_valid(m) if m is not None else False
                ),
                manifold_s=man_s[kappa],
                polish_s=time.perf_counter() - t0,
            )
    return rows


# --------------------------------------------------------------------------
# the CPU check
# --------------------------------------------------------------------------


def check_det3_cpu(
    expression,
    constants=None,
    *,
    fSR,
    GSR,
    bounds,
    rhos=C.DEFAULT_RHOS,
    kappas=C.DEFAULT_KAPPAS,
    gamma1=C.DEFAULT_GAMMA1,
    input_index=1,
    with_pd=True,
    with_exact=True,
    lattice=None,
    origin_tol=C.ORIGIN_TOL,
    b_tol=C.B_TOL,
    margin_tol=C.MARGIN_TOL,
    top_k=POLISH_TOP_K,
    maxiter=POLISH_MAXITER,
    sobol_seeds=SOBOL_SEEDS,
    eval_chunk=524288,
):
    """DET3 for one candidate (GP prefix string + constants, or a sympy V) at every
    (rho, kappa).  kappa = 0 in ``kappas`` gives the DET2 row."""
    bounds = np.asarray(bounds, dtype=float)
    n = bounds.shape[0]
    result = DET3Result()
    try:
        t0 = time.perf_counter()
        x_syms = state_symbols(n)
        V = sympy_expression(expression, constants)
        a, b = ab_expressions(V, fSR, GSR, x_syms, input_index)
        fns = polish_callables(V, a, b, x_syms)
        result.build_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        geometry = scan_geometry(bounds, **{**LATTICE, **(lattice or {})})
        coords = geometry["coordinates"]
        keep = np.einsum("ij,ij->i", coords, coords) > origin_tol**2
        A = eval_lattice(fns["a_raw"], coords, eval_chunk)
        B = eval_lattice(fns["b_raw"], coords, eval_chunk)
        W = eval_lattice(fns["V_raw"], coords, eval_chunk) - fns["v0"]
        result.scan_s = time.perf_counter() - t0

        extra = None
        if with_exact:
            t0 = time.perf_counter()
            result.exact = check_b_manifold_exact(
                V,
                fSR,
                GSR,
                bounds,
                gamma1=gamma1,
                input_index=input_index,
                margin_tol=-margin_tol,
                origin_tol=origin_tol,
                polish_b_tol=b_tol,
                polish_top_k=top_k,
                polish_maxiter=maxiter,
            )
            extra = result.exact.violation_points
            result.exact_s = time.perf_counter() - t0
        if with_pd:
            t0 = time.perf_counter()
            result.pd = check_positive_definite_cpu(
                V,
                bounds,
                origin_tol=origin_tol,
                polish_top_k=top_k,
                polish_maxiter=maxiter,
                **{**LATTICE, **(lattice or {})},
            )
            result.pd_valid = (
                C.pd_valid(result.pd.min_margin) if result.pd.status == "ok" else False
            )
            result.pd_s = time.perf_counter() - t0

        result.rows = det3_from_lattice(
            fns,
            A,
            B,
            W,
            coords,
            keep,
            bounds,
            rhos,
            kappas,
            gamma1,
            extra,
            origin_tol,
            b_tol,
            margin_tol,
            top_k,
            maxiter,
            sobol_seeds,
        )
    except Exception as exc:
        result.status = f"error: {type(exc).__name__}: {str(exc)[:200]}"
    return result
