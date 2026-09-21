"""DET4 for an (n states, m inputs) control-affine system -- the check that makes the fitness.

Same algorithm as examples/DET4Fast4D_V8/det4_fast.py (itself the referee's det4 in numba):

  lattice scan  -> W, a, b (vector), x.gradV on the referee's scan_geometry lattice
  PD stage      -> c_v* = min over the box of  min(W/|x|^2, (x.gradV)/(2|x|^2))
                   (lattice seeds -> SLSQP + Nelder-Mead polishes)
  manifold      -> max a/W on {b = 0}, SLSQP from the thin-|b| seeds, each endpoint
                   Newton/Gauss-Newton projected onto {b = 0} (V7's PROJECT_ASCENT) and
                   accepted on the normalised gate rho |b| <= B_REL_TOL |a|
  origin ray    -> the origin's worst manifold direction from the linearisation, projected
  rate stage    -> max (a - rho||b|| + gamma1)/W by SLSQP (sign-split) and Nelder-Mead
  kappa*        = -max over all considered points of (a - rho||b|| + gamma1)/W

The only change for m > 1: b, grad b and the manifold constraint are vectors, so
  * the ascent has m equality constraints b_k = 0 (scipy's SLSQP core takes them as is),
  * the projection is the Gauss-Newton min-norm step  x <- x - J^T (J J^T)^-1 b,  J = db/dx,
  * the rate SLSQP splits on the sign of every component (s_k b_k >= 0, m inequalities) so that
    rho * sum_k |b_k| stays smooth inside a run,
  * null(B'P) in the origin ray is the null space of an (m, n) matrix.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.linalg as _sl
from numba import njit, prange

from core import config, evalnb
from core.encode import program_arrays
from core.evalnb import _N, _M, _abvr_point, _dual2, bnorm, lattice_eval, point_full
from core.geometry import lattice_coords, sobol_cached
from core.slsqp_direct import slsqp_direct

# SLSQP over seeds in threads: measured NO gain (the scipy driver's callbacks are Python-level,
# so the GIL serialises them) -- opt-in only, default off.
_SLSQP_THREADS = max(1, int(os.environ.get("SYMCLF6D_SLSQP_THREADS", "1") or 1))
_POOL = ThreadPoolExecutor(max_workers=_SLSQP_THREADS) if _SLSQP_THREADS > 1 else None
# the seed stage's six argsorts over the whole lattice DO release the GIL (numpy), as in V8
_SEED_POOL = ThreadPoolExecutor(max_workers=8)


def _map_seeds(fn, seeds):
    """One independent SLSQP run per seed; the numba callbacks release the GIL, so threads
    overlap. Results keep the seed order, so the maximum over seeds is unchanged."""
    seeds = list(seeds)
    if _POOL is None or len(seeds) < 2:
        return [fn(z) for z in seeds]
    return list(_POOL.map(fn, seeds))

OBJ_PD = 0
OBJ_RADIAL = 1
OBJ_RATE = 2

_RHO = config.RHO
_GAMMA1 = config.GAMMA1
_L1 = config.B_NORM == "l1"


# ---------------------------------------------------------------------------------------------
# seeds: the referee's pd_rate_stage, rate_stage and manifold_seed_set_rate (numpy, verbatim
# apart from the vector-b norm)
# ---------------------------------------------------------------------------------------------
def seed_stages(A, Bn, W, keep, r2, coords, bounds, rho, gamma1, pd_eps, top_k, sobol_seeds):
    """Bn = the norm of b per lattice row (sum |b_k| or ||b||_2)."""
    top_k = int(top_k)
    with np.errstate(all="ignore"):
        q = W / r2
        g = W - pd_eps * r2
        bar = A - rho * Bn + gamma1
        m = bar / W
        score = np.where(keep & np.isfinite(A) & np.isfinite(Bn) & np.isfinite(W) & (W > 0), A / W, -np.inf)
    q = np.where(keep & np.isfinite(q), q, np.inf)
    g = np.where(keep & np.isfinite(g), g, np.inf)
    ok = keep & np.isfinite(m) & (W > 0)
    m = np.where(ok, m, -np.inf)
    bar = np.where(keep & np.isfinite(bar), bar, -np.inf)
    n_nonpos_W = int(np.sum(keep & np.isfinite(W) & (W <= 0)))
    absb = np.where(np.isfinite(Bn), np.abs(Bn), np.inf)
    fin = np.isfinite(absb)
    jobs = {
        "q": _SEED_POOL.submit(np.argsort, q),
        "g": _SEED_POOL.submit(np.argsort, g),
        "m": _SEED_POOL.submit(np.argsort, m),
        "bar": _SEED_POOL.submit(np.argsort, bar),
        "score": _SEED_POOL.submit(np.argsort, score),
        "quant": _SEED_POOL.submit(lambda: np.quantile(absb[fin], 0.01) if fin.any() else None),
    }
    sob = sobol_cached(bounds, sobol_seeds)
    # pd_rate_stage
    o1 = jobs["q"].result()[:top_k]
    o1 = o1[np.isfinite(q[o1])]
    o2 = jobs["g"].result()[:top_k]
    o2 = o2[np.isfinite(g[o2])]
    order = np.concatenate([o1, o2[~np.isin(o2, o1)]])
    pd_seeds = np.vstack([coords[order], sob]) if order.size else sob
    lattice_min = float(q[o1[0]]) if o1.size else np.inf
    # rate_stage
    r1 = jobs["m"].result()[::-1][:top_k]
    r1 = r1[np.isfinite(m[r1])]
    r2o = jobs["bar"].result()[::-1][:top_k]
    r2o = r2o[np.isfinite(bar[r2o])]
    order = np.concatenate([r1, r2o[~np.isin(r2o, r1)]])
    rate_seeds = np.vstack([coords[order], sob]) if order.size else sob
    lattice_max = float(m[r1[0]]) if r1.size else -np.inf
    # manifold_seed_set_rate
    qv = jobs["quant"].result()
    thin = (absb <= qv) if qv is not None else np.zeros_like(keep)
    f1 = np.argsort(np.where(thin, score, -np.inf))[::-1][:top_k]
    f2 = jobs["score"].result()[::-1][:top_k]
    mseeds = [coords[i] for i in np.unique(np.concatenate([f1, f2])) if np.isfinite(score[i])]
    mseeds += list(sob)
    mseeds = np.asarray(mseeds, dtype=float).reshape(-1, coords.shape[1])
    return (pd_seeds, lattice_min), (rate_seeds, lattice_max, n_nonpos_W), mseeds


# ---------------------------------------------------------------------------------------------
# projection onto {b = 0}: Gauss-Newton min-norm steps, the m = 1 case is V7's Newton step
# ---------------------------------------------------------------------------------------------
def project_to_manifold(prog, x, lo, hi, iters=60):
    z = np.clip(np.asarray(x, float).copy(), lo, hi)
    for _ in range(int(iters)):
        v, a, b, ga, gb, g = point_full(*prog, z)
        if not np.all(np.isfinite(b)):
            break
        scale = max(1.0, abs(a))
        if np.max(np.abs(b)) <= 1e-15 * scale:
            break
        J = np.asarray(gb, float).reshape(_M, _N)
        JJt = J @ J.T
        try:
            step = J.T @ np.linalg.solve(JJt + 1e-300 * np.eye(_M), b)
        except np.linalg.LinAlgError:
            step = J.T @ np.linalg.lstsq(JJt, b, rcond=None)[0]
        if not np.all(np.isfinite(step)):
            break
        z = np.clip(z - step, lo, hi)
    return z


def origin_ray_points(prog, lo, hi):
    """(the origin's exact DET4 ratio, the projected ray points): P = Hess(V)(0)/2, A = Jf(0),
    B = G(0); N = null(B'P); the largest generalised eigenvalue of (N'(PA+A'P)N, N'PN)."""
    zero = np.zeros(_N)
    H = np.asarray(_dual2(*prog, zero)[2], dtype=float)
    fld = np.asarray(evalnb._FIELDS(zero, np.empty(evalnb._K_FIELDS)), dtype=float)
    Bv = np.empty((_N, _M))
    for k in range(_M):
        Bv[:, k] = fld[evalnb._OFF_G + k * _N: evalnb._OFF_G + (k + 1) * _N]
    A = fld[evalnb._OFF_JF: evalnb._OFF_JF + _N * _N].reshape(_N, _N)
    P = H / 2.0
    S = (Bv.T @ P).reshape(_M, _N)
    Nsp = _sl.null_space(S)
    if Nsp.size == 0:
        return float("nan"), np.empty((0, _N))
    Q, Pr = Nsp.T @ (P @ A + A.T @ P) @ Nsp, Nsp.T @ P @ Nsp
    try:
        lam, U = _sl.eigh(Q, Pr)
    except Exception:
        return float("nan"), np.empty((0, _N))
    v = Nsp @ U[:, -1]
    nv = np.linalg.norm(v)
    if not np.isfinite(nv) or nv == 0.0:
        return float(lam[-1]), np.empty((0, _N))
    v = v / nv
    pts = [project_to_manifold(prog, np.clip(s * r * v, lo, hi), lo, hi, 60)
           for s in (1.0, -1.0) for r in config.ORIGIN_RADII]
    return float(lam[-1]), np.asarray(pts, dtype=float).reshape(-1, _N)


# ---------------------------------------------------------------------------------------------
# Nelder-Mead (scipy 1.17 _minimize_neldermead, ported; n + 1 vertices, stable insertion sort)
# ---------------------------------------------------------------------------------------------
@njit(cache=False, fastmath=False)
def _outside(z, lo, hi, r0_sq):
    r2 = 0.0
    for i in range(_N):
        r2 += z[i] * z[i]
    if r2 <= r0_sq:
        return True
    for i in range(_N):
        if z[i] < lo[i] or z[i] > hi[i]:
            return True
    return False


@njit(cache=False, fastmath=False)
def _nm_objective(obj, z, opcodes, operands, literals, n_ops, parameters, v0, rho, gamma1,
                  lo, hi, r0_sq):
    if _outside(z, lo, hi, r0_sq):
        return np.inf
    v, a, b, rad, g = _abvr_point(opcodes, operands, literals, n_ops, parameters, z)
    r2 = 0.0
    for i in range(_N):
        r2 += z[i] * z[i]
    if obj == OBJ_PD:
        q = (v - v0) / r2
        return q if np.isfinite(q) else np.inf
    if obj == OBJ_RADIAL:
        q = rad / (2.0 * r2)
        return q if np.isfinite(q) else np.inf
    W = v - v0
    if not (np.isfinite(W) and W > 0.0):
        return np.inf
    m = (a - rho * bnorm(b) + gamma1) / W
    if not np.isfinite(m):
        return np.inf
    return -m


@njit(cache=False, fastmath=False)
def _sort_simplex(sim, fsim):
    """Stable insertion sort of the N + 1 vertices (the 3-D/4-D ports' _sort4 / _sort5)."""
    for i in range(1, _N + 1):
        vf = fsim[i]
        row = sim[i].copy()
        j = i - 1
        while j >= 0 and vf < fsim[j]:
            fsim[j + 1] = fsim[j]
            for c in range(_N):
                sim[j + 1, c] = sim[j, c]
            j -= 1
        fsim[j + 1] = vf
        for c in range(_N):
            sim[j + 1, c] = row[c]


@njit(cache=False, fastmath=False)
def nelder_mead(obj, x0, opcodes, operands, literals, n_ops, parameters, v0, rho, gamma1,
                lo, hi, r0_sq, maxiter, xatol, fatol):
    rho_ = 1.0
    chi = 2.0
    psi = 0.5
    sigma = 0.5
    nonzdelt = 0.05
    zdelt = 0.00025
    N = _N
    sim = np.empty((N + 1, N))
    fsim = np.full(N + 1, np.inf)
    for i in range(N):
        sim[0, i] = x0[i]
    for k in range(N):
        for i in range(N):
            sim[k + 1, i] = x0[i]
        if sim[k + 1, k] != 0.0:
            sim[k + 1, k] = (1.0 + nonzdelt) * sim[k + 1, k]
        else:
            sim[k + 1, k] = zdelt
    for k in range(N + 1):
        fsim[k] = _nm_objective(obj, sim[k].copy(), opcodes, operands, literals, n_ops,
                                parameters, v0, rho, gamma1, lo, hi, r0_sq)
    _sort_simplex(sim, fsim)
    iterations = 1
    xbar = np.empty(N)
    xr = np.empty(N)
    xe = np.empty(N)
    xc = np.empty(N)
    xcc = np.empty(N)
    while iterations < maxiter:
        dx = 0.0
        df = 0.0
        for k in range(1, N + 1):
            for i in range(N):
                t = abs(sim[k, i] - sim[0, i])
                if t > dx or t != t:
                    dx = t if t == t else np.nan
            t = abs(fsim[0] - fsim[k])
            if t > df or t != t:
                df = t if t == t else np.nan
        if dx <= xatol and df <= fatol:
            break
        for i in range(N):
            acc = sim[0, i]
            for k in range(1, N):
                acc = acc + sim[k, i]
            xbar[i] = acc / N
        for i in range(N):
            xr[i] = (1.0 + rho_) * xbar[i] - rho_ * sim[N, i]
        fxr = _nm_objective(obj, xr, opcodes, operands, literals, n_ops, parameters, v0, rho,
                            gamma1, lo, hi, r0_sq)
        doshrink = False
        if fxr < fsim[0]:
            for i in range(N):
                xe[i] = (1.0 + rho_ * chi) * xbar[i] - rho_ * chi * sim[N, i]
            fxe = _nm_objective(obj, xe, opcodes, operands, literals, n_ops, parameters, v0,
                                rho, gamma1, lo, hi, r0_sq)
            if fxe < fxr:
                for i in range(N):
                    sim[N, i] = xe[i]
                fsim[N] = fxe
            else:
                for i in range(N):
                    sim[N, i] = xr[i]
                fsim[N] = fxr
        else:
            if fxr < fsim[N - 1]:
                for i in range(N):
                    sim[N, i] = xr[i]
                fsim[N] = fxr
            else:
                if fxr < fsim[N]:
                    for i in range(N):
                        xc[i] = (1.0 + psi * rho_) * xbar[i] - psi * rho_ * sim[N, i]
                    fxc = _nm_objective(obj, xc, opcodes, operands, literals, n_ops, parameters,
                                        v0, rho, gamma1, lo, hi, r0_sq)
                    if fxc <= fxr:
                        for i in range(N):
                            sim[N, i] = xc[i]
                        fsim[N] = fxc
                    else:
                        doshrink = True
                else:
                    for i in range(N):
                        xcc[i] = (1.0 - psi) * xbar[i] + psi * sim[N, i]
                    fxcc = _nm_objective(obj, xcc, opcodes, operands, literals, n_ops,
                                         parameters, v0, rho, gamma1, lo, hi, r0_sq)
                    if fxcc < fsim[N]:
                        for i in range(N):
                            sim[N, i] = xcc[i]
                        fsim[N] = fxcc
                    else:
                        doshrink = True
                if doshrink:
                    for j in range(1, N + 1):
                        for i in range(N):
                            sim[j, i] = sim[0, i] + sigma * (sim[j, i] - sim[0, i])
                        fsim[j] = _nm_objective(obj, sim[j].copy(), opcodes, operands, literals,
                                                n_ops, parameters, v0, rho, gamma1, lo, hi, r0_sq)
        iterations += 1
        _sort_simplex(sim, fsim)
    return sim[0].copy(), fsim[0]


@njit(cache=False, fastmath=False, parallel=True)
def nelder_mead_many(obj, seeds, opcodes, operands, literals, n_ops, parameters, v0, rho, gamma1,
                     lo, hi, r0_sq, maxiter, xatol, fatol):
    K = seeds.shape[0]
    xs = np.empty((K, _N))
    fs = np.empty(K)
    for k in prange(K):
        x0 = np.empty(_N)
        for i in range(_N):
            x0[i] = min(max(seeds[k, i], lo[i]), hi[i])
        x, f = nelder_mead(obj, x0, opcodes, operands, literals, n_ops, parameters, v0, rho,
                           gamma1, lo, hi, r0_sq, maxiter, xatol, fatol)
        xs[k] = x
        fs[k] = f
    return xs, fs


# ---------------------------------------------------------------------------------------------
# SLSQP stages (scipy's C core; the callbacks are the numba point evaluator)
# ---------------------------------------------------------------------------------------------
class _Cache:
    __slots__ = ("prog", "z", "out")

    def __init__(self, prog):
        self.prog = prog
        self.z = None
        self.out = None

    def __call__(self, z):
        z = np.asarray(z, dtype=float).reshape(_N)
        if self.z is None or not np.array_equal(self.z, z):
            self.out = point_full(*self.prog, z)
            self.z = z.copy()
        return self.out


def _slsqp_pd(prog, v0, seeds, bounds, r0_sq, maxiter):
    """min W/r^2 s.t. the origin ball and the box."""
    lo, hi = bounds[:, 0], bounds[:, 1]
    box = [tuple(bounds[i]) for i in range(_N)]
    ball = (lambda z: float(z @ z) - r0_sq, lambda z: 2.0 * np.asarray(z, float))

    def one(z0):
        cache = _Cache(prog)

        def q(z):
            r2 = float(z @ z)
            v = (cache(z)[0] - v0) / r2
            return v if np.isfinite(v) else np.inf

        def dq(z):
            r2 = float(z @ z)
            c = cache(z)
            return c[5] / r2 - 2.0 * (c[0] - v0) * np.asarray(z, float) / (r2 * r2)

        try:
            x, _, _, _ = slsqp_direct(q, dq, np.clip(z0, lo, hi), box, (), (ball,), int(maxiter), 1e-14)
            return x
        except Exception:
            return np.full(_N, np.nan)

    out = _map_seeds(one, seeds)
    return np.asarray(out).reshape(-1, _N)


def _slsqp_ascent(prog, v0, seeds, bounds, r0_sq, maxiter):
    """max a/W s.t. b_k = 0 for every input k, the ball and the box."""
    lo, hi = bounds[:, 0], bounds[:, 1]
    box = [tuple(bounds[i]) for i in range(_N)]
    ball = (lambda z: float(z @ z) - r0_sq, lambda z: 2.0 * np.asarray(z, float))

    def one(z0):
        cache = _Cache(prog)

        def obj(z):
            c = cache(z)
            W = c[0] - v0
            return 1e12 if not (np.isfinite(W) and W > 0) else -c[1] / W

        def jac(z):
            c = cache(z)
            W = c[0] - v0
            if not (np.isfinite(W) and W > 0):
                return np.zeros(_N)
            return -(c[3] / W - c[1] * c[5] / (W * W))

        eqs = tuple(
            (lambda z, k=k: float(cache(z)[2][k]), lambda z, k=k: np.asarray(cache(z)[4][k], float))
            for k in range(_M)
        )
        try:
            x, _, _, _ = slsqp_direct(obj, jac, np.clip(z0, lo, hi), box, eqs, (ball,), int(maxiter), 1e-14)
            return np.clip(x, lo, hi)
        except Exception:
            return None

    out = [z for z in _map_seeds(one, seeds) if z is not None]
    return np.asarray(out).reshape(-1, _N)


def _slsqp_rate(prog, v0, seeds, bounds, r0_sq, rho, gamma1, maxiter):
    """max (a - rho sum_k s_k b_k + gamma1)/W on the sign cell s_k b_k >= 0 of each seed.

    For m = 1 this is the 4-D rate polish; for m > 1 the l1 norm is smooth inside one sign cell,
    which is exactly what the sign split buys. Under B_NORM = l2 the objective is already smooth
    away from b = 0, so the constraints are dropped and the norm is used directly.
    """
    lo, hi = bounds[:, 0], bounds[:, 1]
    box = [tuple(bounds[i]) for i in range(_N)]
    ball = (lambda z: float(z @ z) - r0_sq, lambda z: 2.0 * np.asarray(z, float))
    def one(z0):
        cache = _Cache(prog)
        z0 = np.clip(z0, lo, hi)
        b0 = np.asarray(cache(z0)[2], float)
        s = np.where(b0 >= 0.0, 1.0, -1.0)

        if _L1:
            def obj(z, s=s):
                c = cache(z)
                W = c[0] - v0
                if not (np.isfinite(W) and W > 0):
                    return 1e12
                return -(c[1] - rho * float(s @ np.asarray(c[2], float)) + gamma1) / W

            def jac(z, s=s):
                c = cache(z)
                W = c[0] - v0
                if not (np.isfinite(W) and W > 0):
                    return np.zeros(_N)
                num = c[1] - rho * float(s @ np.asarray(c[2], float)) + gamma1
                dnum = c[3] - rho * (s @ np.asarray(c[4], float).reshape(_M, _N))
                return -(dnum / W - num * c[5] / (W * W))

            cons = tuple(
                (lambda z, k=k, s=s: float(s[k] * cache(z)[2][k]),
                 lambda z, k=k, s=s: s[k] * np.asarray(cache(z)[4][k], float))
                for k in range(_M)
            ) + (ball,)
        else:
            def obj(z):
                c = cache(z)
                W = c[0] - v0
                if not (np.isfinite(W) and W > 0):
                    return 1e12
                nb = float(np.linalg.norm(np.asarray(c[2], float)))
                return -(c[1] - rho * nb + gamma1) / W

            def jac(z):
                c = cache(z)
                W = c[0] - v0
                if not (np.isfinite(W) and W > 0):
                    return np.zeros(_N)
                bb = np.asarray(c[2], float)
                nb = float(np.linalg.norm(bb))
                dnb = (bb / nb) @ np.asarray(c[4], float).reshape(_M, _N) if nb > 0 else np.zeros(_N)
                num = c[1] - rho * nb + gamma1
                return -((c[3] - rho * dnb) / W - num * c[5] / (W * W))

            cons = (ball,)
        try:
            x, _, _, _ = slsqp_direct(obj, jac, z0, box, (), cons, int(maxiter), 1e-14)
            return x
        except Exception:
            return np.full(_N, np.nan)

    out = _map_seeds(one, seeds)
    return np.asarray(out).reshape(-1, _N)


# ---------------------------------------------------------------------------------------------
# the check
# ---------------------------------------------------------------------------------------------
def det4(expression, constants, bounds=None, rho=None, gamma1=None, pd_eps=None,
         pd_rate_tol=None, top_k=None, maxiter=None, sobol_seeds=None, origin_tol=0.0,
         lattice=None, gate_on_pd=None):
    """The DET4 numbers of one candidate; the dict keys of det4_fast.det4_fast."""
    rho = config.RHO if rho is None else float(rho)
    gamma1 = config.GAMMA1 if gamma1 is None else float(gamma1)
    pd_eps = config.PD_EPS if pd_eps is None else float(pd_eps)
    pd_rate_tol = config.PD_RATE_TOL if pd_rate_tol is None else float(pd_rate_tol)
    top_k = config.POLISH_TOP_K if top_k is None else int(top_k)
    maxiter = config.POLISH_MAXITER if maxiter is None else int(maxiter)
    sobol_seeds = config.SOBOL_SEEDS if sobol_seeds is None else int(sobol_seeds)
    gate_on_pd = (config.GATE_KAPPA_CAP > 0.0) if gate_on_pd is None else bool(gate_on_pd)
    bounds = config.bounds() if bounds is None else np.asarray(bounds, float)
    lattice = config.lattice_kwargs() if lattice is None else lattice

    t = {}
    t0 = time.perf_counter()
    lo, hi = np.ascontiguousarray(bounds[:, 0]), np.ascontiguousarray(bounds[:, 1])
    r0_sq = (origin_tol * (1.0 + 1e-6)) ** 2
    prog = program_arrays(expression, constants, _N)
    coords = lattice_coords(bounds, **lattice)
    t["build_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    V, A, B, R = lattice_eval(*prog, coords)
    v0 = 0.0                      # the evaluators return W = V - V(0) directly
    W = V - v0
    Bn = np.sum(np.abs(B), axis=1) if _L1 else np.linalg.norm(B, axis=1)
    r2 = np.einsum("ij,ij->i", coords, coords)
    keep = r2 > origin_tol ** 2
    t["scan_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    (seeds, lmin), (sd, lmax, n_neg), mseeds = seed_stages(
        A, Bn, W, keep, r2, coords, bounds, rho, gamma1, pd_eps, top_k, sobol_seeds
    )
    seeds = np.asarray(seeds, float).reshape(-1, _N)
    sd = np.asarray(sd, float).reshape(-1, _N)
    mseeds = np.asarray(mseeds, float).reshape(-1, _N)
    t["seeds_s"] = time.perf_counter() - t0

    # --- PD: W/|x|^2 (SLSQP + NM) and the radial ratio (NM) ---------------------------------
    t0 = time.perf_counter()
    cand_s_pd = (_slsqp_pd(prog, v0, seeds, bounds, r0_sq, maxiter)
                 if seeds.shape[0] else np.empty((0, _N)))
    best, best_x = float(lmin), (seeds[0] if seeds.shape[0] else None)
    cand_n, _ = nelder_mead_many(OBJ_PD, seeds, *prog, v0, rho, gamma1, lo, hi, r0_sq, 400,
                                 1e-10, 1e-16)
    for k in range(seeds.shape[0]):
        for z in (cand_s_pd[k], cand_n[k]):
            z = np.clip(np.asarray(z, float), lo, hi)
            if not np.all(np.isfinite(z)) or float(z @ z) <= r0_sq:
                continue
            q = (point_full(*prog, z)[0] - v0) / float(z @ z)
            if np.isfinite(q) and q < best:
                best, best_x = float(q), z
    cv_w, pd_point = best, best_x
    with np.errstate(all="ignore"):
        radial = R / (2.0 * np.where(r2 > 0.0, r2, np.inf))
    q_r = np.where(keep & np.isfinite(radial), radial, np.inf)
    order = np.argsort(q_r)[: int(top_k)]
    order = order[np.isfinite(q_r[order])]
    rseeds = coords[order]
    rbest = float(q_r[order[0]]) if order.size else np.inf
    rbest_x = coords[order[0]] if order.size else None
    if rseeds.shape[0]:
        cand_r, _ = nelder_mead_many(OBJ_RADIAL, rseeds, *prog, v0, rho, gamma1, lo, hi, r0_sq,
                                     400, 1e-10, 1e-16)
        for z in cand_r:
            z = np.clip(np.asarray(z, float), lo, hi)
            if np.all(np.isfinite(z)) and float(z @ z) > r0_sq:
                c = point_full(*prog, z)
                v = float(z @ c[5]) / (2.0 * float(z @ z))
                if np.isfinite(v) and v < rbest:
                    rbest, rbest_x = float(v), z
    cv = min(cv_w, rbest) if np.isfinite(rbest) else cv_w
    pd_valid = bool(np.isfinite(cv) and cv > pd_eps + pd_rate_tol)
    t["pd_s"] = time.perf_counter() - t0

    # --- the rate stages (skipped for a non-PD candidate when the gate is on) ---------------
    rate_on = bool(pd_valid or not gate_on_pd)
    if not rate_on:
        sd, mseeds = sd[:0], mseeds[:0]
    t0 = time.perf_counter()
    mpts_raw = (_slsqp_ascent(prog, v0, mseeds, bounds, r0_sq, maxiter)
                if mseeds.shape[0] else np.empty((0, _N)))
    accepted = []
    n_raw_accepted = 0
    for x in mpts_raw:
        z = np.clip(x, lo, hi)
        if not np.all(np.isfinite(z)) or float(z @ z) <= r0_sq:
            continue
        c = point_full(*prog, z)
        if np.max(np.abs(np.asarray(c[2], float))) <= config.B_TOL and c[0] - v0 > 0:
            n_raw_accepted += 1
        if config.PROJECT_ASCENT:
            z = project_to_manifold(prog, z, lo, hi, 60)
            if float(z @ z) <= r0_sq:
                continue
            c = point_full(*prog, z)
            nb = float(np.sum(np.abs(np.asarray(c[2], float))) if _L1
                       else np.linalg.norm(np.asarray(c[2], float)))
            on_manifold = rho * nb <= config.B_REL_TOL * abs(c[1])      # normalised gate
        else:
            on_manifold = float(np.max(np.abs(np.asarray(c[2], float)))) <= config.B_TOL
        if on_manifold and c[0] - v0 > 0:
            accepted.append(z)
    mpts = np.asarray(accepted).reshape(-1, _N)
    cand_s_rate = (_slsqp_rate(prog, v0, sd, bounds, r0_sq, rho, gamma1, maxiter)
                   if sd.shape[0] else np.empty((0, _N)))
    t["slsqp_s"] = time.perf_counter() - t0
    origin_ratio, n_origin = float("nan"), 0
    if config.ORIGIN_SEEDS and rate_on:
        t0 = time.perf_counter()
        origin_ratio, opts = origin_ray_points(prog, lo, hi)
        if opts.shape[0]:
            mpts = np.vstack([mpts, opts]) if mpts.shape[0] else opts
            n_origin = int(opts.shape[0])
        t["origin_s"] = time.perf_counter() - t0

    # --- rate: (a - rho||b|| + gamma1)/W over every considered point -------------------------
    t0 = time.perf_counter()
    cand_n, _ = nelder_mead_many(OBJ_RATE, sd, *prog, v0, rho, gamma1, lo, hi, r0_sq, 400,
                                 1e-10, 1e-16)
    t["manifold_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    best, best_x = float(lmax), (sd[0] if sd.shape[0] else None)

    def m_of(z):
        c = point_full(*prog, z)
        Wz = c[0] - v0
        if not (np.isfinite(Wz) and Wz > 0):
            return -np.inf
        bb = np.asarray(c[2], float)
        nb = float(np.sum(np.abs(bb)) if _L1 else np.linalg.norm(bb))
        v = (c[1] - rho * nb + gamma1) / Wz
        return v if np.isfinite(v) else -np.inf

    def consider(z):
        nonlocal best, best_x
        z = np.clip(np.asarray(z, dtype=float), lo, hi)
        if not np.all(np.isfinite(z)) or _outside(z, lo, hi, r0_sq):
            return
        v = m_of(z)
        if np.isfinite(v) and v > best:
            best, best_x = float(v), z

    for z in mpts:
        consider(z)
    for k in range(sd.shape[0]):
        consider(cand_s_rate[k])
        consider(cand_n[k])
    t["polish_s"] = time.perf_counter() - t0
    kappa = None if best_x is None or not np.isfinite(best) else -float(best)
    out = {
        "status": "ok", "cv": float(cv), "cv_w": float(cv_w),
        "cv_rad": None if not np.isfinite(rbest) else float(rbest), "pd_valid": pd_valid,
        "pd_point": None if pd_point is None else [float(v) for v in pd_point],
        "pd_radial_point": None if rbest_x is None else [float(v) for v in rbest_x],
        "kappa": kappa, "lattice_kappa": None if not np.isfinite(lmax) else -float(lmax),
        "worst_point": None if best_x is None else [float(v) for v in best_x],
        "n_polish_seeds": int(sd.shape[0]), "n_manifold_points": int(mpts.shape[0]),
        "n_lattice_nonpos_W": int(n_neg), "rate_on": rate_on,
        "n_ascent_raw_accepted": int(n_raw_accepted), "project_ascent": config.PROJECT_ASCENT,
        "origin_seeds": config.ORIGIN_SEEDS, "origin_ratio": origin_ratio,
        "n_origin_points": n_origin, "b_norm": config.B_NORM, "n_inputs": _M,
    }
    if best_x is not None and kappa is not None:
        c = point_full(*prog, np.asarray(best_x, float))
        bb = np.asarray(c[2], float)
        out["worst_abs_b"] = float(np.sum(np.abs(bb)) if _L1 else np.linalg.norm(bb))
        out["worst_r"] = float(np.sqrt(np.dot(best_x, best_x)))
        out["worst_W"] = float(c[0] - v0)
    out.update({k: float(v) for k, v in t.items()})
    out["total_s"] = float(sum(t.values()))
    return out
