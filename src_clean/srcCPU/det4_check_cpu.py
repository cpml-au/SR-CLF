"""DET4 on the CPU: the NORMALISED (scale-free) certificate check.

Port of examples/VerifierBenchmark/run_det4.py (numerics unchanged; n states, bounds and the
exclusion radius passed explicitly instead of module globals).

WHY -- the absolute margins are properties of ORIGIN_TOL, not of V

Near the origin V is quadratic and f is linear, so a = grad(V).f and W = V - V(0) both scale
like ||x||^2.  The Artstein maximum of a on the manifold and the PD minimum of
W - PD_EPS*||x||^2 are therefore both attained ON the exclusion sphere ||x|| = ORIGIN_TOL and
are ~c*ORIGIN_TOL^2 ~ 1e-7 for the certified cart-pole CLFs: a verdict decided on ~1e-12 of the
box volume, at a number the tolerance 1e-9 turns into a coin flip.  DET4 encloses the RATIOS,
which are O(0.01..20) and scale-free:

    c_v*(x)   = W(x) / ||x||^2                            normalised positive definiteness
    kappa*(x) = -(a(x) - rho*|b(x)| + gamma1) / W(x)      certified exponential rate

    c_v*   = min over the punctured box     VALID iff c_v* > PD_EPS + PD_RATE_TOL
    kappa* = min over the punctured box     VALID iff kappa* > KAPPA_MIN   (gated on PD)

kappa* is the largest kappa for which DET3 (vdot <= -kappa*W) holds, found directly instead of
by a ladder.  It is only defined where W > 0, so DET4 is GATED ON PD: if c_v* fails, kappa* is
None and the candidate fails.  Verdicts at every KAPPA_LADDER value are read from the same
kappa*.

THE SEARCH -- DET2's, with the ratios as objectives

  lattice   scan_geometry with the 21^n mesh (2 610 397 points for n = 4); V, a, b once.
  PD rate   seeds = bottom-K lattice points by W/r^2  U  bottom-K by the absolute margin
            W - PD_EPS*r^2 (the PD check's own seeds)  U  Sobol starts; SLSQP (exact gradient
            of the ratio, ball as constraint) + Nelder-Mead (ball and box as walls), min kept.
            The ratio's own seeds sit in the box corners and miss off-lattice valleys the
            absolute seeds find; the union cannot search less than the absolute check.
  rate      per rho: seeds = top-K by m = (a - rho|b| + gamma1)/W  U  top-K by the absolute
            barrier (DET2's seeds)  U  Sobol; piecewise-smooth SLSQP on
            (a - rho*s*b + gamma1)/W under s*b >= 0, Nelder-Mead on m; plus a rho-independent
            manifold ascent of a/W subject to b = 0 (top-K thin-|b| by a/W, top-K overall,
            Sobol) whose points are scored as candidates.  kappa* = -max m.

The exact Artstein check is NOT folded in here (DET4 has no absolute margin to bound).
The stage helpers are shared with src/srcGPU/det4_check_gpu.py, which only replaces the lattice
evaluation of a, b, V.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from src.srcCPU import conditions as C
from src.srcCPU.det3_check_cpu import (
    LATTICE,
    POLISH_MAXITER,
    POLISH_TOP_K,
    SOBOL_SEEDS,
    _ball,
)
from src.srcCPU.exact_check_cpu import eval_lattice, scan_geometry
from src.srcCPU.symbolic import (
    ab_expressions,
    polish_callables,
    state_symbols,
    sympy_expression,
)

#: scale-free tolerances (run_det4.py)
PD_RATE_TOL = 1.0e-6  # c_v* must exceed PD_EPS by this
KAPPA_MIN = 1.0e-3  # headline threshold on kappa*
KAPPA_LADDER = (1e-4, 1e-3, 1e-2, 1e-1)
DEFAULT_RHOS = (1000.0,)


@dataclass
class DET4Row:
    rho: float
    kappa_star: float | None  # min over the punctured box of -(a - rho|b| + gamma1)/W
    lattice_kappa: float | None
    worst_abs_b: float | None  # |b| at the binding point of kappa*
    worst_r: float | None  # ||x|| at the binding point
    worst_W: float | None
    n_lattice_nonpos_W: int | None
    n_polish_seeds: int
    n_manifold_points: int
    artstein_valid: bool  # pd_valid and kappa* > kappa_min
    valid_at: dict  # {kappa_min: pd_valid and kappa* > kappa_min} for KAPPA_LADDER
    kappa_min: float
    status: str = "ok"
    polish_s: float = 0.0
    worst_point: (
        tuple
    ) = ()  # the binding point of kappa* (2026-09-09, for Danskin refines)


@dataclass
class DET4Result:
    rows: dict = field(default_factory=dict)  # rho -> DET4Row
    pd_rate: float | None = None  # c_v* = min(pd_rate_w, pd_rate_radial)
    pd_rate_w: float | None = None  # min W/||x||^2
    pd_rate_radial: float | None = (
        None  # min (x.gradV)/(2||x||^2)  (2026-09-10, flat V)
    )
    pd_radial_point: np.ndarray | None = None
    pd_valid: bool = False
    pd_worst_r: float | None = None  # ||x|| at the binding point of c_v*
    lattice_pd_rate: float | None = None
    pd_point: np.ndarray | None = None
    status: str = "ok"
    build_s: float = 0.0
    scan_s: float = 0.0
    pd_s: float = 0.0
    manifold_s: float = 0.0

    def row(self, rho):
        return self.rows[float(rho)]


def sobol_points(bounds, count):
    """Scrambled Sobol starts, fixed seed (the same set every call)."""
    from scipy.stats import qmc

    bounds = np.asarray(bounds, dtype=float)
    if int(count) <= 0:
        return np.empty((0, bounds.shape[0]))
    sampler = qmc.Sobol(d=bounds.shape[0], scramble=True, seed=0)
    return qmc.scale(sampler.random(int(count)), bounds[:, 0], bounds[:, 1])


# --------------------------------------------------------------------------
# host stage: seeds from the lattice arrays
# --------------------------------------------------------------------------


def pd_rate_stage(
    W,
    r2,
    keep,
    coords,
    bounds,
    pd_eps=C.PD_EPS,
    top_k=POLISH_TOP_K,
    sobol_seeds=SOBOL_SEEDS,
):
    """Seeds = bottom-K by the RATIO W/r^2  U  bottom-K by the ABSOLUTE margin
    W - pd_eps*r^2  U  Sobol starts.  Returns (seeds, lattice min of the ratio)."""
    with np.errstate(all="ignore"):
        q = W / r2
        g = W - pd_eps * r2
    q = np.where(keep & np.isfinite(q), q, np.inf)
    g = np.where(keep & np.isfinite(g), g, np.inf)
    o1 = np.argsort(q)[: int(top_k)]
    o1 = o1[np.isfinite(q[o1])]
    o2 = np.argsort(g)[: int(top_k)]
    o2 = o2[np.isfinite(g[o2])]
    order = np.concatenate([o1, o2[~np.isin(o2, o1)]])
    sob = sobol_points(bounds, sobol_seeds)
    seeds = np.vstack([coords[order], sob]) if order.size else sob
    return seeds, (float(q[o1[0]]) if o1.size else np.inf)


def rate_stage(
    A,
    B,
    W,
    keep,
    coords,
    bounds,
    rho,
    gamma1=C.DEFAULT_GAMMA1,
    top_k=POLISH_TOP_K,
    sobol_seeds=SOBOL_SEEDS,
):
    """Seeds = top-K by the RATIO m = (a - rho|b| + gamma1)/W  U  top-K by the ABSOLUTE
    barrier (DET2's seeds)  U  Sobol.  Returns (seeds, lattice max of m, #lattice W <= 0).
    """
    with np.errstate(all="ignore"):
        bar = A - rho * np.abs(B) + gamma1
        m = bar / W
    ok = keep & np.isfinite(m) & (W > 0)
    m = np.where(ok, m, -np.inf)
    bar = np.where(keep & np.isfinite(bar), bar, -np.inf)
    n_nonpos_W = int(np.sum(keep & np.isfinite(W) & (W <= 0)))
    o1 = np.argsort(m)[::-1][: int(top_k)]
    o1 = o1[np.isfinite(m[o1])]
    o2 = np.argsort(bar)[::-1][: int(top_k)]
    o2 = o2[np.isfinite(bar[o2])]
    order = np.concatenate([o1, o2[~np.isin(o2, o1)]])
    sob = sobol_points(bounds, sobol_seeds)
    seeds = np.vstack([coords[order], sob]) if order.size else sob
    return seeds, (float(m[o1[0]]) if o1.size else -np.inf), n_nonpos_W


def manifold_seed_set_rate(
    A, B, W, keep, coords, bounds, top_k=POLISH_TOP_K, sobol_seeds=SOBOL_SEEDS
):
    """rho-independent seeds for the a/W manifold ascent: top-K by a/W among the 1% thinnest
    |b| (W > 0), top-K overall, plus Sobol."""
    n = np.asarray(bounds).shape[0]
    with np.errstate(all="ignore"):
        score = np.where(
            keep & np.isfinite(A) & np.isfinite(B) & np.isfinite(W) & (W > 0),
            A / W,
            -np.inf,
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
    seeds += list(sobol_points(bounds, sobol_seeds))
    return np.asarray(seeds, dtype=float).reshape(-1, n)


# --------------------------------------------------------------------------
# CPU stage: SciPy polishes on the ratios (polish_callables)
# --------------------------------------------------------------------------


def _walls(bounds, origin_tol, fns=None, sublevel=None):
    """Box + origin ball as hard walls; with ``sublevel`` also W = V - V(0) <= sublevel
    (the check restricted to the certified set Omega_c)."""
    bounds = np.asarray(bounds, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    r0_sq, ball = _ball(origin_tol)
    cap = (
        None
        if sublevel is None or fns is None
        else (float(sublevel), fns["V"], fns["v0"])
    )

    def outside(z):
        z = np.asarray(z, dtype=float)
        if float(z @ z) <= r0_sq or np.any(z < lo) or np.any(z > hi):
            return True
        return cap is not None and (cap[1](z) - cap[2]) > cap[0]

    return lo, hi, list(zip(lo, hi)), r0_sq, ball, outside


def pd_rate_polish(
    fns, seeds, lattice_min, bounds, origin_tol=C.ORIGIN_TOL, maxiter=POLISH_MAXITER
):
    """min of W/||x||^2 over the punctured box: SLSQP (exact gradient of the ratio) and
    Nelder-Mead from every seed, lower value kept.  Returns (c_v*, its point)."""
    n = np.asarray(bounds).shape[0]
    lo, hi, box, r0_sq, ball, outside = _walls(bounds, origin_tol)
    V, gV, v0 = fns["V"], fns["gV"], fns["v0"]
    seeds = np.asarray(seeds, dtype=float).reshape(-1, n)
    best, best_x = float(lattice_min), (seeds[0] if seeds.shape[0] else None)

    def q(z):
        r2 = float(z @ z)
        v = (V(z) - v0) / r2
        return v if np.isfinite(v) else np.inf

    def dq(z):
        r2 = float(z @ z)
        return gV(z) / r2 - 2.0 * (V(z) - v0) * z / (r2 * r2)

    def consider(z):
        nonlocal best, best_x
        z = np.clip(np.asarray(z, dtype=float), lo, hi)
        if not np.all(np.isfinite(z)) or float(z @ z) <= r0_sq:
            return
        v = q(z)
        if np.isfinite(v) and v < best:
            best, best_x = float(v), z

    for z0 in seeds:
        z0 = np.clip(z0, lo, hi)
        try:
            res = minimize(
                q,
                z0,
                jac=dq,
                constraints=[ball],
                bounds=box,
                method="SLSQP",
                options={"maxiter": int(maxiter), "ftol": 1e-14},
            )
            consider(res.x)
        except Exception:
            pass
        try:
            res = minimize(
                lambda z: np.inf if outside(z) else q(z),
                z0,
                method="Nelder-Mead",
                options={"maxiter": 400, "xatol": 1e-10, "fatol": 1e-16},
            )
            consider(res.x)
        except Exception:
            pass
    return best, best_x


def radial_pd_stage(fns, coords, r2, keep, top_k=POLISH_TOP_K, eval_chunk=524288):
    """Lattice values of the RADIAL PD ratio (x.gradV)/(2|x|^2) and its lowest-value seeds.
    2026-09-10: equal to W/|x|^2 for quadratics, zero for flat V (a constant with a 1e-23
    crease passed min W/|x|^2 = 190 in job 126166 because no search can land in the crease,
    while gradV ~ 0 is visible everywhere).  Needs fns["gV_raw"]."""
    from src.srcCPU.exact_check_cpu import eval_lattice

    n = coords.shape[1]
    radial = np.zeros(coords.shape[0])
    for i in range(n):
        radial += coords[:, i] * eval_lattice(fns["gV_raw"][i], coords, eval_chunk)
    radial = radial / (2.0 * np.where(r2 > 0, r2, np.inf))
    q = np.where(keep & np.isfinite(radial), radial, np.inf)
    order = np.argsort(q)[: int(top_k)]
    order = order[np.isfinite(q[order])]
    return coords[order], (float(q[order[0]]) if order.size else np.inf)


def radial_pd_polish(fns, seeds, lattice_min, bounds, origin_tol=C.ORIGIN_TOL):
    """Nelder-Mead on (x.gradV)/(2|x|^2) from every seed (no Hessian needed), lower kept."""
    n = np.asarray(bounds).shape[0]
    lo, hi, _, r0_sq, _, outside = _walls(bounds, origin_tol)
    gV = fns["gV"]
    seeds = np.asarray(seeds, dtype=float).reshape(-1, n)
    best, best_x = float(lattice_min), (seeds[0] if seeds.shape[0] else None)

    def q(z):
        r2 = float(z @ z)
        v = float(z @ gV(z)) / (2.0 * r2)
        return v if np.isfinite(v) else np.inf

    for z0 in seeds:
        z0 = np.clip(z0, lo, hi)
        try:
            res = minimize(
                lambda z: np.inf if outside(z) else q(z),
                z0,
                method="Nelder-Mead",
                options={"maxiter": 400, "xatol": 1e-10, "fatol": 1e-16},
            )
            z = np.clip(np.asarray(res.x, dtype=float), lo, hi)
            if np.all(np.isfinite(z)) and float(z @ z) > r0_sq:
                v = q(z)
                if np.isfinite(v) and v < best:
                    best, best_x = float(v), z
        except Exception:
            pass
    return best, best_x


def project_onto_manifold(fns, z, lo, hi, iters=60):
    """2026-09-13 (V7): Newton steps along grad b until b(z) = 0 (relative 1e-15), clipped."""
    b, gb, a = fns["b"], fns["gb"], fns["a"]
    z = np.asarray(z, dtype=float).copy()
    for _ in range(iters):
        bz, g = float(b(z)), np.asarray(gb(z), dtype=float)
        gg = float(g @ g)
        if not np.isfinite(bz) or gg == 0.0 or abs(bz) <= 1e-15 * max(1.0, abs(float(a(z)))):
            break
        z = np.clip(z - bz * g / gg, lo, hi)
    return z


def manifold_ascent_rate(
    fns, seeds, bounds, origin_tol=C.ORIGIN_TOL, b_tol=C.B_TOL, maxiter=POLISH_MAXITER,
    project=False, gate_rho=None, b_rel_tol=1e-6,
):
    """max a/W s.t. b = 0 from every seed; keeps points with |b| <= b_tol and W > 0 outside
    the origin ball.  ``project=True`` (2026-09-13, V7): the endpoint is Newton-projected onto
    b = 0 before the |b| test, so a point the optimiser found a hair off the manifold is kept
    (default False = the historical behaviour).  With ``project`` and ``gate_rho`` the gate is
    normalised, gate_rho |b| <= b_rel_tol |a| (2026-09-13, the user's call), instead of the absolute
    |b| <= b_tol."""
    n = np.asarray(bounds).shape[0]
    lo, hi, box, r0_sq, ball, _ = _walls(bounds, origin_tol)
    a, ga, b, gb, V, gV, v0 = (fns[k] for k in ("a", "ga", "b", "gb", "V", "gV", "v0"))

    def obj(z):
        W = V(z) - v0
        return 1e12 if not (np.isfinite(W) and W > 0) else -a(z) / W

    def jac(z):
        W = V(z) - v0
        if not (np.isfinite(W) and W > 0):
            return np.zeros(n)
        return -(ga(z) / W - a(z) * gV(z) / (W * W))

    out = []
    for z0 in np.asarray(seeds, dtype=float).reshape(-1, n):
        z0 = np.clip(z0, lo, hi)
        try:
            res = minimize(
                obj,
                z0,
                jac=jac,
                constraints=[{"type": "eq", "fun": b, "jac": gb}, ball],
                bounds=box,
                method="SLSQP",
                options={"maxiter": int(maxiter), "ftol": 1e-14},
            )
            z = np.clip(res.x, lo, hi)
            if project and np.all(np.isfinite(z)):
                z = project_onto_manifold(fns, z, lo, hi)
            if not (np.all(np.isfinite(z)) and float(z @ z) > r0_sq and V(z) - v0 > 0):
                continue
            if project and gate_rho is not None:
                on_manifold = float(gate_rho) * abs(b(z)) <= float(b_rel_tol) * abs(a(z))
            else:
                on_manifold = abs(b(z)) <= b_tol
            if on_manifold:
                out.append(z)
        except Exception:
            continue
    return np.asarray(out) if out else np.empty((0, n))


def origin_ray_points(V, fns, fSR, GSR, x_syms, input_index, bounds, radii=(1e-3, 1e-2, 3e-2, 1e-1)):
    """2026-09-13 (V7): the origin's worst manifold direction from the linearisation (the max
    generalised eigenvalue of (N'(PA+A'P)N, N'PN), N = null(B'P), P = Hessian(V)(0)/2) and its
    +-r v ray points projected onto b = 0.  Returns (that eigenvalue = the exact DET4 ratio at
    the origin, points)."""
    import scipy.linalg as sl
    import sympy as sp
    n = len(x_syms)
    zero = {s: 0 for s in x_syms}
    P = np.array(sp.hessian(V, x_syms).subs(zero), dtype=float) / 2.0
    f = sp.Matrix(fSR(*x_syms))
    g = sp.Matrix([GSR(*x_syms)[i, input_index] for i in range(n)])
    A = np.array(f.jacobian(list(x_syms)).subs(zero), dtype=float)
    B = np.array(g.subs(zero), dtype=float).reshape(n)
    N = np.linalg.svd((B @ P).reshape(1, n))[2][1:].T
    try:
        lam, U = sl.eigh(N.T @ (P @ A + A.T @ P) @ N, N.T @ P @ N)
    except Exception:
        return float("nan"), np.empty((0, n))
    v = N @ U[:, -1]
    nv = np.linalg.norm(v)
    if not np.isfinite(nv) or nv == 0.0:
        return float(lam[-1]), np.empty((0, n))
    v = v / nv
    bounds = np.asarray(bounds, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    pts = [project_onto_manifold(fns, np.clip(s * r * v, lo, hi), lo, hi)
           for s in (1.0, -1.0) for r in radii]
    return float(lam[-1]), np.asarray(pts, dtype=float).reshape(-1, n)


def rate_polish(
    fns,
    seeds,
    lattice_max,
    rho,
    gamma1,
    extra_points,
    bounds,
    origin_tol=C.ORIGIN_TOL,
    maxiter=POLISH_MAXITER,
    sublevel=None,
):
    """max of m = (a - rho|b| + gamma1)/W over the punctured box -> kappa* = -max m.
    With ``sublevel`` the max is over {W <= sublevel} only (polished points that leave
    the set are discarded)."""
    n = np.asarray(bounds).shape[0]
    lo, hi, box, r0_sq, ball, outside = _walls(bounds, origin_tol, fns, sublevel)
    a, ga, b, gb, V, gV, v0 = (fns[k] for k in ("a", "ga", "b", "gb", "V", "gV", "v0"))
    seeds = np.asarray(seeds, dtype=float).reshape(-1, n)
    best, best_x = float(lattice_max), (seeds[0] if seeds.shape[0] else None)

    def m_of(z):
        W = V(z) - v0
        if not (np.isfinite(W) and W > 0):
            return -np.inf
        v = (a(z) - rho * abs(b(z)) + gamma1) / W
        return v if np.isfinite(v) else -np.inf

    def consider(z):
        nonlocal best, best_x
        z = np.clip(np.asarray(z, dtype=float), lo, hi)
        if not np.all(np.isfinite(z)) or outside(z):
            return
        v = m_of(z)
        if np.isfinite(v) and v > best:
            best, best_x = float(v), z

    for z in np.asarray(extra_points, dtype=float).reshape(-1, n):
        consider(z)
    for z0 in seeds:
        z0 = np.clip(z0, lo, hi)
        s = 1.0 if b(z0) >= 0.0 else -1.0

        def obj(z, s=s):
            W = V(z) - v0
            if not (np.isfinite(W) and W > 0):
                return 1e12
            return -(a(z) - rho * s * b(z) + gamma1) / W

        def jac(z, s=s):
            W = V(z) - v0
            if not (np.isfinite(W) and W > 0):
                return np.zeros(n)
            num = a(z) - rho * s * b(z) + gamma1
            return -((ga(z) - rho * s * gb(z)) / W - num * gV(z) / (W * W))

        try:  # (a) piecewise-smooth SLSQP on the half-space s*b >= 0
            res = minimize(
                obj,
                z0,
                jac=jac,
                constraints=[
                    {
                        "type": "ineq",
                        "fun": lambda z, s=s: s * b(z),
                        "jac": lambda z, s=s: s * gb(z),
                    },
                    ball,
                ],
                bounds=box,
                method="SLSQP",
                options={"maxiter": int(maxiter), "ftol": 1e-14},
            )
            consider(res.x)
        except Exception:
            pass
        try:  # (b) Nelder-Mead on m itself
            res = minimize(
                lambda z: np.inf if outside(z) else -m_of(z),
                z0,
                method="Nelder-Mead",
                options={"maxiter": 400, "xatol": 1e-10, "fatol": 1e-16},
            )
            consider(res.x)
        except Exception:
            pass
    if best_x is None or not np.isfinite(best):
        return {
            "kappa_star": None,
            "worst_abs_b": None,
            "worst_r": None,
            "worst_W": None,
            "n_polish_seeds": int(seeds.shape[0]),
        }
    return {
        "kappa_star": -best,
        "worst_abs_b": abs(b(best_x)),
        "worst_r": float(np.sqrt(best_x @ best_x)),
        "worst_W": V(best_x) - v0,
        "worst_point": tuple(float(v) for v in np.asarray(best_x).reshape(-1)),
        "n_polish_seeds": int(seeds.shape[0]),
    }


def det4_verdict(
    pd_rate,
    kappa_star,
    pd_eps=C.PD_EPS,
    pd_rate_tol=PD_RATE_TOL,
    kappa_min=KAPPA_MIN,
    kappa_ladder=KAPPA_LADDER,
):
    """(pd_valid, artstein_valid, valid_at) -- the scale-free verdicts from the two ratios."""
    pd_valid = (
        pd_rate is not None and np.isfinite(pd_rate) and pd_rate > pd_eps + pd_rate_tol
    )
    ks = kappa_star
    valid_at = {
        float(k): bool(pd_valid and ks is not None and ks > k) for k in kappa_ladder
    }
    return (
        bool(pd_valid),
        bool(pd_valid and ks is not None and ks > kappa_min),
        valid_at,
    )


def det4_from_lattice(
    fns,
    A,
    B,
    W,
    coords,
    keep,
    bounds,
    rhos=DEFAULT_RHOS,
    gamma1=C.DEFAULT_GAMMA1,
    pd_eps=C.PD_EPS,
    pd_rate_tol=PD_RATE_TOL,
    kappa_min=KAPPA_MIN,
    kappa_ladder=KAPPA_LADDER,
    origin_tol=C.ORIGIN_TOL,
    b_tol=C.B_TOL,
    top_k=POLISH_TOP_K,
    maxiter=POLISH_MAXITER,
    sobol_seeds=SOBOL_SEEDS,
    result=None,
    sublevel=None,
    gate_on_pd=True,
    project_ascent=False,
    origin_seeds=False,
    origin_args=None,
):
    """Everything after the lattice arrays: the PD rate, the manifold ascent (if PD holds)
    and the rate polish per rho.  Fills and returns ``result`` (a DET4Result).
    ``sublevel``: restrict the rate search to {W <= sublevel} (the certified set).
    ``gate_on_pd=False``: kappa* is computed even when PD fails, over the W > 0 region (the
    ratio is defined there); None only when no lattice point has W > 0 (constant V)."""
    result = DET4Result() if result is None else result
    n = np.asarray(bounds).shape[0]
    r2 = np.einsum("ij,ij->i", coords, coords)
    keep_rate = (
        keep if sublevel is None else (keep & np.isfinite(W) & (W <= float(sublevel)))
    )

    t0 = time.perf_counter()
    seeds, lmin = pd_rate_stage(W, r2, keep, coords, bounds, pd_eps, top_k, sobol_seeds)
    pd_rate, pd_x = pd_rate_polish(fns, seeds, lmin, bounds, origin_tol, maxiter)
    result.pd_rate_w = None if not np.isfinite(pd_rate) else float(pd_rate)
    if (
        fns.get("gV_raw") is not None
    ):  # 2026-09-10 radial PD, the smaller of the two counts
        rseeds, rmin = radial_pd_stage(fns, coords, r2, keep, top_k)
        rrate, rx = radial_pd_polish(fns, rseeds, rmin, bounds, origin_tol)
        result.pd_rate_radial = None if not np.isfinite(rrate) else float(rrate)
        result.pd_radial_point = rx
        if np.isfinite(rrate) and rrate < pd_rate:
            pd_rate = rrate
    result.pd_s = time.perf_counter() - t0
    result.pd_rate = None if not np.isfinite(pd_rate) else float(pd_rate)
    result.lattice_pd_rate = None if not np.isfinite(lmin) else float(lmin)
    result.pd_point = pd_x
    result.pd_worst_r = None if pd_x is None else float(np.sqrt(pd_x @ pd_x))
    result.pd_valid = bool(np.isfinite(pd_rate) and pd_rate > pd_eps + pd_rate_tol)

    t0 = time.perf_counter()
    mpts = np.empty((0, n))
    rate_on = result.pd_valid or not gate_on_pd
    if rate_on:
        mpts = manifold_ascent_rate(
            fns,
            manifold_seed_set_rate(A, B, W, keep, coords, bounds, top_k, sobol_seeds),
            bounds,
            origin_tol,
            b_tol,
            maxiter,
            project=project_ascent,
            gate_rho=(max(float(r) for r in rhos) if project_ascent else None),
        )
        if origin_seeds and origin_args is not None:
            V_, fSR_, GSR_, x_syms_, input_index_ = origin_args
            result.origin_ratio, opts = origin_ray_points(V_, fns, fSR_, GSR_, x_syms_, input_index_, bounds)
            if opts.shape[0]:
                mpts = np.vstack([mpts, opts]) if mpts.shape[0] else opts
    result.manifold_s = time.perf_counter() - t0

    for rho in rhos:
        t0 = time.perf_counter()
        if rate_on:
            sd, lmax, n_neg = rate_stage(
                A, B, W, keep_rate, coords, bounds, rho, gamma1, top_k, sobol_seeds
            )
            r = rate_polish(
                fns, sd, lmax, rho, gamma1, mpts, bounds, origin_tol, maxiter, sublevel
            )
            lattice_kappa = None if not np.isfinite(lmax) else -float(lmax)
            status = "ok" if result.pd_valid else "ok (PD failed; kappa* over W > 0)"
        else:
            r = {
                "kappa_star": None,
                "worst_abs_b": None,
                "worst_r": None,
                "worst_W": None,
                "n_polish_seeds": 0,
            }
            lattice_kappa, n_neg, status = (
                None,
                None,
                "ok (PD failed; kappa* undefined)",
            )
        pd_valid, valid, valid_at = det4_verdict(
            result.pd_rate,
            r["kappa_star"],
            pd_eps,
            pd_rate_tol,
            kappa_min,
            kappa_ladder,
        )
        result.rows[float(rho)] = DET4Row(
            rho=float(rho),
            kappa_star=r["kappa_star"],
            lattice_kappa=lattice_kappa,
            worst_abs_b=r["worst_abs_b"],
            worst_r=r["worst_r"],
            worst_W=r["worst_W"],
            worst_point=tuple(r.get("worst_point", ())),
            n_lattice_nonpos_W=n_neg,
            n_polish_seeds=r["n_polish_seeds"],
            n_manifold_points=int(mpts.shape[0]),
            artstein_valid=valid,
            valid_at=valid_at,
            kappa_min=float(kappa_min),
            status=status,
            polish_s=time.perf_counter() - t0,
        )
    return result


# --------------------------------------------------------------------------
# the CPU check
# --------------------------------------------------------------------------


def check_det4_cpu(
    expression,
    constants=None,
    *,
    fSR,
    GSR,
    bounds,
    rhos=DEFAULT_RHOS,
    gamma1=C.DEFAULT_GAMMA1,
    input_index=1,
    lattice=None,
    pd_eps=C.PD_EPS,
    pd_rate_tol=PD_RATE_TOL,
    kappa_min=KAPPA_MIN,
    kappa_ladder=KAPPA_LADDER,
    origin_tol=C.ORIGIN_TOL,
    b_tol=C.B_TOL,
    top_k=POLISH_TOP_K,
    maxiter=POLISH_MAXITER,
    sobol_seeds=SOBOL_SEEDS,
    eval_chunk=524288,
    sublevel=None,
    gate_on_pd=True,
    project_ascent=False,
    origin_seeds=False,
):
    """DET4 for one candidate (GP prefix string + constants, or a sympy V): c_v* once, then
    kappa* at every rho.  The exclusion radius is a knob here (the ratios are scale-free);
    ``sublevel`` restricts the rate search to the certified set {W <= sublevel};
    ``gate_on_pd=False`` reports kappa* over the W > 0 region even when PD fails.
    ``project_ascent`` / ``origin_seeds`` (2026-09-13, V7): see manifold_ascent_rate and
    origin_ray_points; both default False (the historical behaviour, byte-identical).
    """
    bounds = np.asarray(bounds, dtype=float)
    n = bounds.shape[0]
    result = DET4Result()
    try:
        t0 = time.perf_counter()
        x_syms = state_symbols(n)
        V = sympy_expression(expression, constants)
        a, b = ab_expressions(V, fSR, GSR, x_syms, input_index)
        fns = polish_callables(V, a, b, x_syms)
        result.build_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        coords = scan_geometry(bounds, **{**LATTICE, **(lattice or {})})["coordinates"]
        keep = np.einsum("ij,ij->i", coords, coords) > origin_tol**2
        A = eval_lattice(fns["a_raw"], coords, eval_chunk)
        B = eval_lattice(fns["b_raw"], coords, eval_chunk)
        W = eval_lattice(fns["V_raw"], coords, eval_chunk) - fns["v0"]
        result.scan_s = time.perf_counter() - t0

        det4_from_lattice(
            fns,
            A,
            B,
            W,
            coords,
            keep,
            bounds,
            rhos,
            gamma1,
            pd_eps,
            pd_rate_tol,
            kappa_min,
            kappa_ladder,
            origin_tol,
            b_tol,
            top_k,
            maxiter,
            sobol_seeds,
            result,
            sublevel,
            gate_on_pd,
            project_ascent=project_ascent,
            origin_seeds=origin_seeds,
            origin_args=(V, fSR, GSR, x_syms, input_index),
        )
    except Exception as exc:
        result.status = f"error: {type(exc).__name__}: {str(exc)[:200]}"
    return result
