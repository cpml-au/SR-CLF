"""CPU exact check: the reference Artstein manifold check + native-NumPy positive definiteness.

Ports, numerics unchanged, of symclf-main_rnv/src/b_manifold_check_exact.py (the referee every
GPU version had to reproduce) and examples/VerifierBenchmark/cpu_pd.py.  The benchmark's forced
tolerances are the DEFAULTS here: every axis scanned, origin radius 1.1e-3, manifold tolerance
1e-10 and the DEFINITE margin predicate (margin_tol = -1e-9).  The historical defaults
(scan_axes=(2, 3), origin 1e-9, b_tol 1e-8, margin_tol 0) are NOT the defaults.

ARTSTEIN -- three deterministic stages on a sympy V (exact derivatives, no fd_step):
  1. axis-aligned line scans: 801 samples on 9^(n-1) lines per axis; sign changes of b are
     bracketed and refined by 60 vectorised bisections; exact zeros at scan nodes are kept;
  2. 200 fixed-seed random-direction lines (a branch of {b = 0} can be parallel to every axis
     line; a generic line cuts it transversally);
  3. SLSQP ascent of a subject to b = 0 from the 40 best roots, exact jacobians, the origin
     ball as an inequality constraint; a polished point is kept when |b| <= polish_b_tol.
  margin_max = max(a + gamma1) over every retained root outside the origin ball.

PD -- V on the 2 610 397-point lattice (21^4 mesh + the axis scans + the random lines), then
the 40 worst values of W - pd_eps*||x||^2 are polished by SLSQP AND Nelder-Mead from every seed
(SLSQP stalls on a semidefinite valley where grad vanishes along the flat direction).

The stage helpers (scan_geometry, manifold_roots_numpy, polish_manifold_roots,
score_manifold_roots, eval_lattice, pd_margins, pd_polish) are what src/srcGPU reuses: the GPU
replaces only the lattice evaluations and the bisection, never the polish or the verdict.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from scipy.optimize import minimize

from src.srcCPU import conditions as C
from src.srcCPU.symbolic import ab_expressions, state_symbols


@dataclass
class BManifoldResult:
    n_roots: int
    n_violations: int
    margin_max: float
    violation_points: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    status: str = "ok"


@dataclass
class PDResult:
    n_violations: int
    min_margin: float  # min over the box of W - pd_eps*||x||^2
    violation_points: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    status: str = "ok"
    positive_definite: bool = True
    min_point: tuple = ()
    metrics: dict = field(default_factory=dict)


@dataclass
class ExactResult:
    """Both definite conditions for one candidate; verdicts recomputed from the raw margins."""

    artstein: BManifoldResult
    pd: PDResult | None
    artstein_valid: bool
    artstein_valid_semidefinite: bool
    pd_valid: bool | None
    valid: bool
    artstein_s: float = 0.0
    pd_s: float = 0.0

    @property
    def margin_max(self):
        return self.artstein.margin_max

    @property
    def pd_min_margin(self):
        return None if self.pd is None else self.pd.min_margin


def verdicts(margin_max, pd_min_margin, margin_tol=C.MARGIN_TOL, pd_tol=C.PD_TOL):
    """(artstein_valid, artstein_valid_semidefinite, pd_valid, valid) from raw margins."""
    m = None if margin_max is None else float(margin_max)
    a_valid = C.artstein_valid(m, margin_tol) if m is not None else False
    a_semi = C.nonstrict_valid(m) if m is not None else False
    p_valid = None if pd_min_margin is None else C.pd_valid(pd_min_margin, pd_tol)
    return a_valid, a_semi, p_valid, bool(a_valid and (p_valid is None or p_valid))


# --------------------------------------------------------------------------
# the lattice (identical points, identical order, to the GPU _scan_geometry)
# --------------------------------------------------------------------------


def scan_geometry(
    bounds,
    scan_axes=None,
    scan_points=801,
    grid_points_per_axis=9,
    random_lines=200,
    random_line_points=401,
    rng_seed=0,
    mesh_points_per_axis=0,
):
    """Mesh (optional), axis-aligned scan lines and fixed-seed random lines, as one point
    array plus the per-field structure the root extraction needs.

    Returns {"mesh": (axes_lin, g, slice) | None,
             "axes": [(axis, others, combos (C, n-1), samples (P,), slice)],
             "random": (origins, directions, parameters (L, P), slice) | None,
             "coordinates": (N, n)}.
    """
    bounds = np.asarray(bounds, dtype=float)
    n = bounds.shape[0]
    scan_axes = (
        tuple(range(n)) if scan_axes is None else tuple(int(a) for a in scan_axes)
    )
    geometry = {"mesh": None, "axes": [], "random": None}
    chunks = []
    offset = 0

    if mesh_points_per_axis and int(mesh_points_per_axis) > 1:
        g = int(mesh_points_per_axis)
        axes_lin = [np.linspace(bounds[i, 0], bounds[i, 1], g) for i in range(n)]
        mesh = np.meshgrid(*axes_lin, indexing="ij")
        points = np.stack([m.ravel() for m in mesh], axis=1)
        geometry["mesh"] = (axes_lin, g, slice(offset, offset + points.shape[0]))
        chunks.append(points)
        offset += points.shape[0]

    for axis in scan_axes:
        others = [i for i in range(n) if i != axis]
        lines = [
            np.linspace(bounds[i, 0], bounds[i, 1], grid_points_per_axis)
            for i in others
        ]
        mesh = np.meshgrid(*lines, indexing="ij")
        combos = np.stack([m.ravel() for m in mesh], axis=1)  # (C, n-1)
        line_count = combos.shape[0]
        samples = np.linspace(bounds[axis, 0], bounds[axis, 1], scan_points)
        points = np.empty((line_count * scan_points, n))
        for k, coordinate in enumerate(others):
            points[:, coordinate] = np.repeat(combos[:, k], scan_points)
        points[:, axis] = np.tile(samples, line_count)
        geometry["axes"].append(
            (axis, others, combos, samples, slice(offset, offset + points.shape[0]))
        )
        chunks.append(points)
        offset += points.shape[0]

    if random_lines > 0:
        rng = np.random.default_rng(rng_seed)
        origins = rng.uniform(bounds[:, 0], bounds[:, 1], (random_lines, n))
        directions = rng.normal(size=(random_lines, n))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            ta = (bounds[:, 0][None, :] - origins) / directions
            tb = (bounds[:, 1][None, :] - origins) / directions
        t0 = np.where(np.abs(directions) > 1.0e-12, np.minimum(ta, tb), -np.inf).max(1)
        t1 = np.where(np.abs(directions) > 1.0e-12, np.maximum(ta, tb), np.inf).min(1)
        fraction = np.linspace(0.0, 1.0, random_line_points)
        parameters = t0[:, None] + (t1 - t0)[:, None] * fraction[None, :]
        points = (
            origins[:, None, :] + parameters[..., None] * directions[:, None, :]
        ).reshape(-1, n)
        geometry["random"] = (
            origins,
            directions,
            parameters,
            slice(offset, offset + points.shape[0]),
        )
        chunks.append(points)
        offset += points.shape[0]

    geometry["coordinates"] = (
        np.concatenate(chunks, axis=0) if chunks else np.empty((0, n))
    )
    return geometry


def scan_coordinates(bounds, **kwargs):
    """The (N, n) lattice only (defaults: all axes, mesh_points_per_axis=21 as cpu_pd)."""
    kwargs.setdefault("mesh_points_per_axis", 21)
    return scan_geometry(bounds, **kwargs)["coordinates"]


def eval_lattice(fn, coords, chunk=524288):
    """A lambdified scalar on (N, n) coords -> (N,), chunked, NaN-safe, constants broadcast."""
    out = np.empty(coords.shape[0], dtype=float)
    n = coords.shape[1]
    step = int(chunk)
    for lo in range(0, coords.shape[0], step):
        hi = lo + step
        block = coords[lo:hi]
        with np.errstate(all="ignore"):
            v = np.asarray(fn(*[block[:, i] for i in range(n)]), dtype=float)
        if v.ndim == 0 or v.size == 1:
            v = np.full(block.shape[0], float(np.reshape(v, -1)[0]))
        out[lo:hi] = v.reshape(-1)
    return out


# --------------------------------------------------------------------------
# ARTSTEIN: reference stages
# --------------------------------------------------------------------------


def reference_callables(V_expr, fSR, GSR, x_syms, input_index=1, gamma1=0.0):
    """The reference's sympy setup: b, margin, a and the exact jacobians (numpy lambdified)."""
    a_expr, b_expr = ab_expressions(V_expr, fSR, GSR, x_syms, input_index)
    margin_expr = a_expr + gamma1
    return {
        "a_expr": a_expr,
        "b_expr": b_expr,
        "b_fn": sp.lambdify(x_syms, b_expr, "numpy"),
        "m_fn": sp.lambdify(x_syms, margin_expr, "numpy"),
        "a_fn": sp.lambdify(x_syms, a_expr, "numpy"),
        "ga_fn": sp.lambdify(x_syms, [sp.diff(a_expr, s) for s in x_syms], "numpy"),
        "gb_fn": sp.lambdify(x_syms, [sp.diff(b_expr, s) for s in x_syms], "numpy"),
    }


def batch_evaluator(fn):
    """fn(*X) on X (n, M) -> (M,), NaN-safe, scalar results broadcast."""

    def b_batch(X):
        with np.errstate(all="ignore"):
            out = np.asarray(fn(*X), dtype=float)
        if out.shape != X.shape[1:]:
            out = np.broadcast_to(out, X.shape[1:]).copy()
        return out

    return b_batch


def manifold_roots_numpy(b_batch, geometry, refine_iters=60, zero_tol=1e-14):
    """Stages 1-2 of the reference: exact zeros + bisected sign changes of b on the axis
    scan lines and the random lines of ``geometry``.  Returns roots as (n, K)."""
    coords = geometry["coordinates"]
    n = coords.shape[1]
    root_list = []

    for axis, others, combos, s, sl in geometry["axes"]:
        scan_points = s.shape[0]
        Cc = combos.shape[0]
        # contiguous rows as the reference builds X (same numpy ufunc path, ULP-identical)
        bv = b_batch(np.ascontiguousarray(coords[sl].T)).reshape(Cc, scan_points)

        zc, zs = np.nonzero(np.abs(bv) <= zero_tol)
        if zc.size:
            Z = np.empty((n, zc.size))
            for k, i in enumerate(others):
                Z[i] = combos[zc, k]
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
                Xm[i] = combos[ci, k]
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

    if geometry["random"] is not None:
        P0, D, T, sl = geometry["random"]
        L, P = T.shape
        bv = b_batch(coords[sl].T).reshape(L, P)
        sg = np.sign(bv)
        li, ti = np.nonzero(sg[:, :-1] * sg[:, 1:] < 0)
        if li.size:
            lo = T[li, ti].astype(float)
            hi = T[li, ti + 1].astype(float)
            b_lo = bv[li, ti]
            Pl = P0[li]
            Dd = D[li]
            for _ in range(refine_iters):
                mid = 0.5 * (lo + hi)
                Xm = Pl + mid[:, None] * Dd
                bm = b_batch(Xm.T)
                left = np.sign(bm) * np.sign(b_lo) > 0
                lo = np.where(left, mid, lo)
                b_lo = np.where(left, bm, b_lo)
                hi = np.where(left, hi, mid)
            root_list.append((Pl + (0.5 * (lo + hi))[:, None] * Dd).T)

    if not root_list:
        return np.empty((n, 0))
    return np.concatenate(root_list, axis=1)


def polish_manifold_roots(
    R,
    fns,
    bounds,
    origin_tol=C.ORIGIN_TOL,
    polish_top_k=40,
    polish_maxiter=60,
    polish_b_tol=C.B_TOL,
):
    """Stage 3 of the reference: SLSQP ascent of a subject to b = 0 from the top-K roots.
    ``fns`` is reference_callables(...).  Returns R with the accepted polished points appended.
    """
    bounds = np.asarray(bounds, dtype=float)
    n = bounds.shape[0]
    if polish_top_k <= 0 or R.shape[1] == 0:
        return R
    a_fn, ga_fn, b_fn, gb_fn, m_fn = (
        fns[k] for k in ("a_fn", "ga_fn", "b_fn", "gb_fn", "m_fn")
    )

    with np.errstate(all="ignore"):
        m_seed = np.asarray(m_fn(*R), dtype=float)
    if m_seed.shape != (R.shape[1],):
        m_seed = np.broadcast_to(m_seed, (R.shape[1],)).copy()
    m_seed = np.where(np.isfinite(m_seed), m_seed, -np.inf)
    top_k = int(polish_top_k)
    order = np.argsort(m_seed)[-top_k:]

    box = [tuple(bounds[i]) for i in range(n)]
    # the ascent must respect the punctured domain: without the ball constraint every
    # VALID candidate drains to the origin (sup of a on {b=0} is 0 there)
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
                        "jac": lambda z: np.asarray(gb_fn(*z), dtype=float).ravel(),
                    },
                    {
                        "type": "ineq",
                        "fun": lambda z: float(z @ z) - r0_sq,
                        "jac": lambda z: 2.0 * z,
                    },
                ],
                bounds=box,
                method="SLSQP",
                options={"maxiter": int(polish_maxiter), "ftol": 1e-12},
            )
            z = np.clip(res.x, bounds[:, 0], bounds[:, 1])
            if np.all(np.isfinite(z)) and abs(float(b_fn(*z))) <= polish_b_tol:
                polished.append(z)
        except Exception:
            continue
    if polished:
        R = np.concatenate([R, np.array(polished).T], axis=1)
    return R


def score_manifold_roots(R, m_fn, origin_tol=C.ORIGIN_TOL, margin_tol=-C.MARGIN_TOL):
    """The reference verdict: margin = m_fn at every root, violation iff margin > margin_tol."""
    with np.errstate(all="ignore"):
        margin = np.asarray(m_fn(*R), dtype=float)
    if margin.shape != (R.shape[1],):
        margin = np.broadcast_to(margin, (R.shape[1],)).copy()

    nonorigin = np.sqrt(np.sum(R**2, axis=0)) > origin_tol
    viol = nonorigin & np.isfinite(margin) & (margin > margin_tol)
    finite = margin[nonorigin & np.isfinite(margin)]
    margin_max = float(finite.max()) if finite.size else np.nan

    result = BManifoldResult(
        n_roots=int(nonorigin.sum()),
        n_violations=int(viol.sum()),
        margin_max=margin_max,
        violation_points=R[:, viol].T,
        status="ok",
    )
    # mean of the positive part of the margin over every non-origin root (dense signal)
    result.margin_mean_pos = (
        float(np.maximum(finite, 0.0).mean()) if finite.size else np.nan
    )
    return result


def check_b_manifold_exact(
    V_expr,
    fSR,
    GSR,
    bounds,
    gamma1=C.DEFAULT_GAMMA1,
    input_index=1,
    scan_axes=None,
    scan_points=801,
    grid_points_per_axis=9,
    margin_tol=-C.MARGIN_TOL,
    origin_tol=C.ORIGIN_TOL,
    refine_iters=60,
    zero_tol=1e-14,
    random_lines=200,
    random_line_points=401,
    rng_seed=0,
    polish_top_k=40,
    polish_maxiter=60,
    polish_b_tol=C.B_TOL,
    decay_rate=0.0,
):
    """Exact-gradient manifold check on a sympy candidate expression (the CPU referee).

    Defaults are the benchmark's forced conditions (scan_axes=None means every axis).
    ``decay_rate`` is accepted for API compatibility and ignored.
    """
    bounds = np.asarray(bounds, dtype=float)
    n = bounds.shape[0]
    started = time.perf_counter()
    try:
        x_syms = state_symbols(n)
        fns = reference_callables(V_expr, fSR, GSR, x_syms, input_index, gamma1)
        geometry = scan_geometry(
            bounds,
            scan_axes,
            scan_points,
            grid_points_per_axis,
            random_lines,
            random_line_points,
            rng_seed,
            0,
        )
        R = manifold_roots_numpy(
            batch_evaluator(fns["b_fn"]), geometry, refine_iters, zero_tol
        )
        if R.shape[1] == 0:
            return BManifoldResult(0, 0, np.nan, np.empty((0, n)), "ok")
        R = polish_manifold_roots(
            R, fns, bounds, origin_tol, polish_top_k, polish_maxiter, polish_b_tol
        )
        result = score_manifold_roots(R, fns["m_fn"], origin_tol, margin_tol)
        result.metrics = {
            "engine": "cpu_numpy",
            "total_s": time.perf_counter() - started,
            "scan_points": int(geometry["coordinates"].shape[0]),
        }
        return result
    except Exception as exc:
        return BManifoldResult(
            0, 0, np.nan, np.empty((0, n)), f"error: {type(exc).__name__}: {exc}"
        )


def min_w_violator(
    fns,
    seeds,
    bounds,
    gamma1=C.DEFAULT_GAMMA1,
    origin_tol=C.ORIGIN_TOL,
    margin_tol=C.MARGIN_TOL,
    b_tol=C.B_TOL,
    top_k=40,
    maxiter=60,
):
    """The lowest-W point of the violating part of the manifold: minimise W = V - V(0)
    subject to b = 0 and margin >= -margin_tol, SLSQP from the ``top_k`` lowest-W seeds
    (the exact check's violation_points).  ``fns`` is symbolic.polish_callables(...).

    This is the number the certified level c* = min(boundary W, min W over violators) needs;
    the exact check's own violation points sit one lattice step from the origin (measured
    median r 0.063 vs 0.0011, W 3700x too high on half the batch children), because its
    polish maximises a and never descends in W.  Returns (W, point) or (inf, None)."""
    bounds = np.asarray(bounds, float)
    n = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    box = list(zip(lo, hi))
    r0_sq = (origin_tol * (1.0 + 1e-6)) ** 2
    a, ga, b, gb, V, gV, v0 = (fns[k] for k in ("a", "ga", "b", "gb", "V", "gV", "v0"))
    seeds = np.asarray(seeds, float).reshape(-1, n)
    if seeds.shape[0] == 0:
        return np.inf, None
    Wseed = np.array([V(z) - v0 for z in seeds])
    order = np.argsort(Wseed)[: int(top_k)]
    best, best_x = np.inf, None
    for i in order:
        z0 = np.clip(seeds[i], lo, hi)
        try:
            res = minimize(
                lambda z: V(z) - v0,
                z0,
                jac=gV,
                constraints=[
                    {"type": "eq", "fun": b, "jac": gb},
                    {
                        "type": "ineq",
                        "fun": lambda z: a(z) + gamma1 + margin_tol,
                        "jac": ga,
                    },
                    {
                        "type": "ineq",
                        "fun": lambda z: float(z @ z) - r0_sq,
                        "jac": lambda z: 2.0 * z,
                    },
                ],
                bounds=box,
                method="SLSQP",
                options={"maxiter": int(maxiter), "ftol": 1e-14},
            )
            z = np.clip(res.x, lo, hi)
            if (
                np.all(np.isfinite(z))
                and float(z @ z) > r0_sq
                and abs(b(z)) <= b_tol
                and a(z) + gamma1 > -margin_tol
            ):
                w = V(z) - v0
                if np.isfinite(w) and w < best:
                    best, best_x = float(w), z
        except Exception:
            continue
    return best, best_x


# --------------------------------------------------------------------------
# POSITIVE DEFINITENESS: native NumPy twin of the GPU5 check
# --------------------------------------------------------------------------


def pd_margins(
    values,
    coords,
    v0,
    pd_eps=C.PD_EPS,
    origin_tol=C.ORIGIN_TOL,
    pd_reference_matrix=None,
):
    """g = V - V(0) - pd_eps*Q(x) on the lattice; +inf inside the origin ball / non-finite."""
    r2 = np.einsum("ij,ij->i", coords, coords)
    if pd_reference_matrix is None:
        quad = r2
    else:
        n = coords.shape[1]
        P = np.asarray(pd_reference_matrix, dtype=float).reshape(n, n)
        P = 0.5 * (P + P.T)
        quad = np.einsum("ij,jk,ik->i", coords, P, coords)
    with np.errstate(all="ignore"):
        g = np.asarray(values, dtype=float) - v0 - pd_eps * quad
    keep = r2 > origin_tol * origin_tol
    return np.where(keep & np.isfinite(g), g, np.inf)


def pd_objective(V_fn, grad_fn, v0, pd_eps=C.PD_EPS, pd_reference_matrix=None):
    """z -> (g(z), grad g(z)) with exact gradients, for the polish."""
    P = None
    if pd_reference_matrix is not None:
        P = np.asarray(pd_reference_matrix, dtype=float)
        P = 0.5 * (P + P.T)

    def g_and_grad(z):
        z = np.asarray(z, dtype=float)
        with np.errstate(all="ignore"):
            value = float(V_fn(*z))
            gradient = np.asarray(grad_fn(*z), dtype=float).reshape(-1)
        if P is None:
            penalty, dpenalty = float(z @ z), 2.0 * z
        else:
            Pz = P @ z
            penalty, dpenalty = float(z @ Pz), 2.0 * Pz
        return value - v0 - pd_eps * penalty, gradient - pd_eps * dpenalty

    return g_and_grad


def pd_polish(
    g,
    coords,
    g_and_grad,
    bounds,
    origin_tol=C.ORIGIN_TOL,
    polish_top_k=40,
    polish_maxiter=60,
    metrics=None,
):
    """Stages 2-3 of the PD check: the top-K worst lattice points are polished by SLSQP and
    Nelder-Mead (both, keeping the better).  Returns a PDResult."""
    bounds = np.asarray(bounds, dtype=float)
    n = bounds.shape[0]
    metrics = {} if metrics is None else metrics
    started = time.perf_counter()

    top_k = int(polish_top_k)
    order = np.argsort(g)[:top_k]
    order = order[np.isfinite(g[order])]
    metrics["polish_starts"] = int(order.size)
    best = float(g[order[0]]) if order.size else float("inf")
    best_at = coords[order[0]] if order.size else np.zeros(n)

    box = [tuple(bounds[i]) for i in range(n)]
    r0_sq = (origin_tol * (1.0 + 1e-6)) ** 2

    def g_barrier(z):
        # Nelder-Mead cannot take the ball as a constraint, so bar it here.
        z = np.asarray(z, dtype=float)
        if float(z @ z) <= r0_sq:
            return np.inf
        val = g_and_grad(z)[0]
        return val if np.isfinite(val) else np.inf

    violations = []
    for index in order:
        z0 = np.clip(coords[index], bounds[:, 0], bounds[:, 1])
        # SLSQP stalls on a semidefinite valley where grad(g) vanishes along the flat
        # direction; Nelder-Mead finds the true minimum there. Both, from every seed.
        for method in ("SLSQP", "Nelder-Mead"):
            try:
                if method == "SLSQP":
                    res = minimize(
                        lambda z: g_and_grad(z)[0],
                        z0,
                        jac=lambda z: g_and_grad(z)[1],
                        constraints=[
                            {
                                "type": "ineq",
                                "fun": lambda z: float(z @ z) - r0_sq,
                                "jac": lambda z: 2.0 * z,
                            }
                        ],
                        bounds=box,
                        method="SLSQP",
                        options={"maxiter": int(polish_maxiter), "ftol": 1e-12},
                    )
                else:
                    res = minimize(
                        g_barrier,
                        z0,
                        bounds=box,
                        method="Nelder-Mead",
                        options={"maxiter": 400, "xatol": 1e-10, "fatol": 1e-14},
                    )
                z = np.clip(res.x, bounds[:, 0], bounds[:, 1])
                if not np.all(np.isfinite(z)) or float(z @ z) <= r0_sq:
                    continue
                value = g_and_grad(z)[0]
                if np.isfinite(value):
                    if value < best:
                        best, best_at = float(value), z
                    if value <= 0.0:
                        violations.append(z)
            except Exception:  # one bad seed is not fatal
                continue
    metrics["polish_s"] = time.perf_counter() - started

    points = np.asarray(violations) if violations else np.empty((0, n))
    return PDResult(
        n_violations=int(points.shape[0]) + int(best <= 0.0 and not violations),
        min_margin=float(best),
        violation_points=points,
        status="ok",
        positive_definite=bool(best > 0.0),
        min_point=tuple(float(v) for v in np.asarray(best_at).reshape(-1)),
        metrics=metrics,
    )


def lambdify_V(V_expr, n):
    """(V_fn, grad_fn, v0) numpy callables of a sympy V in n states."""
    x_syms = state_symbols(n)
    V_fn = sp.lambdify(x_syms, V_expr, "numpy")
    grad_fn = sp.lambdify(x_syms, [sp.diff(V_expr, s) for s in x_syms], "numpy")
    with np.errstate(all="ignore"):
        v0 = float(V_fn(*np.zeros(n)))
    return V_fn, grad_fn, v0


def check_positive_definite_cpu(
    V_expr,
    bounds,
    pd_eps=C.PD_EPS,
    origin_tol=C.ORIGIN_TOL,
    scan_axes=None,
    scan_points=801,
    grid_points_per_axis=9,
    random_lines=200,
    random_line_points=401,
    rng_seed=0,
    mesh_points_per_axis=21,
    polish_top_k=40,
    polish_maxiter=60,
    pd_reference_matrix=None,
    eval_chunk=524288,
):
    """Minimise W - pd_eps*Q(x) over the punctured box (Q = ||x||^2, or x'Px when
    ``pd_reference_matrix`` is given).  Negative => not a Lyapunov function at all."""
    started = time.perf_counter()
    bounds = np.asarray(bounds, dtype=float)
    n = bounds.shape[0]
    metrics = {
        "engine": "cpu_numpy",
        "scan_s": 0.0,
        "polish_s": 0.0,
        "total_s": 0.0,
        "scan_points": 0,
        "polish_starts": 0,
    }
    try:
        V_fn, grad_fn, v0 = lambdify_V(V_expr, n)
        coords = scan_geometry(
            bounds,
            scan_axes,
            scan_points,
            grid_points_per_axis,
            random_lines,
            random_line_points,
            rng_seed,
            mesh_points_per_axis,
        )["coordinates"]
        metrics["scan_points"] = int(coords.shape[0])
        t0 = time.perf_counter()
        values = eval_lattice(V_fn, coords, eval_chunk)
        metrics["scan_s"] = time.perf_counter() - t0
        g = pd_margins(values, coords, v0, pd_eps, origin_tol, pd_reference_matrix)
        result = pd_polish(
            g,
            coords,
            pd_objective(V_fn, grad_fn, v0, pd_eps, pd_reference_matrix),
            bounds,
            origin_tol,
            polish_top_k,
            polish_maxiter,
            metrics,
        )
        metrics["total_s"] = time.perf_counter() - started
        return result
    except Exception as exc:
        metrics["total_s"] = time.perf_counter() - started
        return PDResult(
            0,
            float("nan"),
            np.empty((0, n)),
            f"error: {type(exc).__name__}: {exc}",
            positive_definite=False,
            metrics=metrics,
        )


# --------------------------------------------------------------------------
# both conditions, one call
# --------------------------------------------------------------------------


def verify_exact_cpu(
    V_expr,
    fSR,
    GSR,
    bounds,
    gamma1=C.DEFAULT_GAMMA1,
    input_index=1,
    with_pd=True,
    artstein_kwargs=None,
    pd_kwargs=None,
):
    """The CPU exact verdict on both definite conditions (what run_cpu.py recorded per row)."""
    t0 = time.perf_counter()
    art = check_b_manifold_exact(
        V_expr,
        fSR,
        GSR,
        bounds,
        gamma1=gamma1,
        input_index=input_index,
        **(artstein_kwargs or {}),
    )
    artstein_s = time.perf_counter() - t0
    pd, pd_s = None, 0.0
    if with_pd:
        t1 = time.perf_counter()
        pd = check_positive_definite_cpu(V_expr, bounds, **(pd_kwargs or {}))
        pd_s = time.perf_counter() - t1
    a_valid, a_semi, p_valid, valid = verdicts(
        art.margin_max, None if pd is None else pd.min_margin
    )
    return ExactResult(art, pd, a_valid, a_semi, p_valid, valid, artstein_s, pd_s)
