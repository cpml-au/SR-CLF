"""GPU exact check: scans + bisection on the GPU, the reference SciPy polish and verdict on
the CPU -- the GPU3 split (measured: the 2.6M-point scan is ~0.07 s on an A100, the 40
four-dimensional SLSQP polishes are milliseconds on a host core, and a GPU SQP polish was both
slower and less accurate than the reference).

Two engines, selected per call:

  engine="jax"     (default; any plant, any dimension)
      sympy V -> JAX.  Every Float of V is abstracted to a parameter so kernels are compiled
      ONCE per tree template and constants are runtime data; a and b are exact autodiff
      derivatives (float64).  Stages: b on the axis/random-line lattice (chunked), root
      extraction on the host (bit-identical brackets to the CPU reference), one fori_loop
      bisection kernel for every bracket, then polish_manifold_roots / score_manifold_roots
      from src.srcCPU.exact_check_cpu on the sympy callables -- the reference verbatim.
      PD: V on the PD lattice on the GPU, pd_polish (SLSQP + Nelder-Mead) on the CPU.

  engine="pallas"  (4-D cart-pole only; needs the GP prefix string + constants)
      the vendored srcGPU5_9 bytecode engine (src/srcGPU/gpu5_9): GPU3 exact check
      (check_b_manifold_exact_gpu3_cached), GPU5 PD (check_positive_definite_gpu5), the
      GPU4 batched exact check (verify_exact_pallas_batch, bit-identical to GPU3, ~10x
      faster per candidate) and the cheap screen (cheap_screen_pallas).  No per-tree XLA
      compile.  A sympy V is converted with symbolic.to_prefix.

Defaults are the benchmark's forced conditions (conditions.py): every axis scanned, origin
radius 1.1e-3, manifold tolerance 1e-10, DEFINITE margin predicate.  Verdicts are recomputed
from the raw margins (exact_check_cpu.verdicts), never taken from a checker's own count.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict

import numpy as np
import sympy as sp

import src.srcGPU  # noqa: F401  (float64 + XLA env before JAX)

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from src.srcCPU import conditions as C  # noqa: E402
from src.srcCPU.exact_check_cpu import (  # noqa: E402
    BManifoldResult,
    ExactResult,
    PDResult,
    lambdify_V,
    pd_margins,
    pd_objective,
    pd_polish,
    polish_manifold_roots,
    reference_callables,
    scan_geometry,
    score_manifold_roots,
    verdicts,
)
from src.srcCPU.symbolic import state_symbols, sympy_expression, to_prefix  # noqa: E402

#: points per device launch for the lattice evaluations (memory bound: ~8 B x nodes x chunk)
SCAN_CHUNK = int(os.environ.get("SYMCLF_GPU_SCAN_CHUNK", "262144"))
_PAD_MIN = 256


# --------------------------------------------------------------------------
# engine="jax": compile-once-per-template kernels for a sympy V
# --------------------------------------------------------------------------


def abstract_floats(expr):
    """Replace every Float atom of ``expr`` with a parameter symbol -> (template, params, values).
    Two trees with the same shape but different constants share one set of compiled kernels.
    """
    floats = []
    for atom in sp.preorder_traversal(expr):
        if isinstance(atom, sp.Float) and atom not in floats:
            floats.append(atom)
    params = [sp.Symbol(f"_p{i}") for i in range(len(floats))]
    template = expr.xreplace(dict(zip(floats, params)))
    values = np.asarray([float(f) for f in floats], dtype=np.float64)
    return template, params, values


_DYN_CACHE: dict = {}


def _dynamics_jax(fSR, GSR, n, input_index):
    """JAX-compiled drift components and active G column (built once per dynamics)."""
    key = (id(fSR), id(GSR), int(n), int(input_index))
    if key not in _DYN_CACHE:
        xs = state_symbols(n)
        f_list = list(fSR(*xs))
        g_list = [GSR(*xs)[i, input_index] for i in range(n)]
        _DYN_CACHE[key] = (
            sp.lambdify(xs, f_list, modules="jax"),
            sp.lambdify(xs, g_list, modules="jax"),
        )
    return _DYN_CACHE[key]


class JaxCandidate:
    """Jitted kernels for one tree template; the constants vector ``c`` is a runtime argument.

    V_batch(X, c) -> (M,);  ab_batch(X, c) -> ((M,), (M,));  b_batch(X, c) -> (M,);
    bisect_line(P0, D, lo, hi, blo, c, iters) -> roots (K, n)   (fixed-count bisection, the
    reference update rule: left = sign(b_mid)*sign(b_lo) > 0).
    """

    def __init__(self, template, params, n, fj=None, gj=None):
        self.n = int(n)
        self.n_params = len(params)
        xs = state_symbols(n)
        Vj = sp.lambdify(tuple(xs) + tuple(params), template, modules="jax")
        k = len(params)

        def V_s(x, c):
            return jnp.asarray(
                Vj(*[x[i] for i in range(n)], *[c[j] for j in range(k)]),
                dtype=jnp.float64,
            )

        self.V_batch = jax.jit(jax.vmap(V_s, in_axes=(0, None)))
        self.ab_batch = self.b_batch = self.bisect_line = None
        if fj is None:
            return

        def ab_s(x, c):
            g = jax.grad(V_s, argnums=0)(x, c)
            comps = [x[i] for i in range(n)]
            fv = jnp.stack([jnp.asarray(v, dtype=jnp.float64) for v in fj(*comps)])
            gv = jnp.stack([jnp.asarray(v, dtype=jnp.float64) for v in gj(*comps)])
            return g @ fv, g @ gv

        self.ab_batch = jax.jit(jax.vmap(ab_s, in_axes=(0, None)))
        self.b_batch = jax.jit(jax.vmap(lambda x, c: ab_s(x, c)[1], in_axes=(0, None)))

        def bisect_line(P0, D, lo, hi, blo, c, iters):
            def body(_, state):
                lo_, hi_, blo_ = state
                mid = 0.5 * (lo_ + hi_)
                bm = jax.vmap(lambda x: ab_s(x, c)[1])(P0 + mid[:, None] * D)
                left = jnp.sign(bm) * jnp.sign(blo_) > 0
                return (
                    jnp.where(left, mid, lo_),
                    jnp.where(left, hi_, mid),
                    jnp.where(left, bm, blo_),
                )

            lo, hi, blo = jax.lax.fori_loop(0, iters, body, (lo, hi, blo))
            return P0 + (0.5 * (lo + hi))[:, None] * D

        self.bisect_line = jax.jit(bisect_line, static_argnums=6)


_BUNDLE_CACHE: OrderedDict = OrderedDict()
_BUNDLE_CACHE_MAX = 128


def get_jax_candidate(V_expr, fSR=None, GSR=None, n=None, input_index=1):
    """(JaxCandidate, constants) for a sympy V, cached by tree template.  Without fSR/GSR
    only V_batch is built (enough for the PD check)."""
    n = len(V_expr.free_symbols) if n is None else int(n)
    template, params, values = abstract_floats(V_expr)
    key = (sp.srepr(template), id(fSR), id(GSR), int(input_index), n)
    if key in _BUNDLE_CACHE:
        _BUNDLE_CACHE.move_to_end(key)
        return _BUNDLE_CACHE[key], values
    fj, gj = (None, None) if fSR is None else _dynamics_jax(fSR, GSR, n, input_index)
    bundle = JaxCandidate(template, params, n, fj, gj)
    _BUNDLE_CACHE[key] = bundle
    while len(_BUNDLE_CACHE) > _BUNDLE_CACHE_MAX:
        _BUNDLE_CACHE.popitem(last=False)
    return bundle, values


def _pad_size(m):
    size = _PAD_MIN
    while size < m:
        size *= 2
    return size


def _pad_rows(arr, size):
    pad = size - arr.shape[0]
    if pad <= 0:
        return arr
    return np.concatenate([arr, np.repeat(arr[:1], pad, axis=0)], axis=0)


def eval_on_device(fn, X, c, chunk=SCAN_CHUNK):
    """fn(X_block, c) over (N, n) host coordinates in fixed-shape chunks (power-of-two padding,
    so each shape compiles once); returns host arrays (a tuple if fn returns a tuple).
    """
    X = np.asarray(X, dtype=float)
    N = X.shape[0]
    step = min(int(chunk), _pad_size(N))
    parts = []
    for start in range(0, N, step):
        stop = start + step
        block = X[start:stop]
        m = block.shape[0]
        out = jax.device_get(fn(jnp.asarray(_pad_rows(block, step)), c))
        if isinstance(out, (tuple, list)):
            parts.append(tuple(np.asarray(o)[:m] for o in out))
        else:
            parts.append(np.asarray(out)[:m])
    if isinstance(parts[0], tuple):
        return tuple(
            np.concatenate([p[i] for p in parts]) for i in range(len(parts[0]))
        )
    return np.concatenate(parts)


def manifold_roots_gpu(
    bundle, c, geometry, refine_iters=60, zero_tol=1e-14, chunk=SCAN_CHUNK, metrics=None
):
    """Stages 1-2 with b on the GPU: exact zeros + bisected sign changes on the axis scan
    lines and the random lines of ``geometry`` (brackets bit-identical to the CPU
    reference; every bracket bisected in ONE padded kernel call).  Returns (n, K)."""
    metrics = {} if metrics is None else metrics
    coords = geometry["coordinates"]
    n = coords.shape[1]
    t0 = time.perf_counter()
    bfield = eval_on_device(bundle.b_batch, coords, c, chunk)
    metrics["scan_s"] = time.perf_counter() - t0

    zero_points = []
    O, Dd, LO, HI, BLO = [], [], [], [], []
    for axis, others, combos, s, sl in geometry["axes"]:
        bv = bfield[sl].reshape(combos.shape[0], s.shape[0])
        zc, zs = np.nonzero(np.abs(bv) <= zero_tol)
        if zc.size:
            Z = np.empty((zc.size, n))
            for k, i in enumerate(others):
                Z[:, i] = combos[zc, k]
            Z[:, axis] = s[zs]
            zero_points.append(Z)
        sg = np.sign(bv)
        ci, si = np.nonzero(sg[:, :-1] * sg[:, 1:] < 0)
        if ci.size:
            origins = np.zeros((ci.size, n))
            for k, i in enumerate(others):
                origins[:, i] = combos[ci, k]
            directions = np.zeros((ci.size, n))
            directions[:, axis] = 1.0
            O.append(origins)
            Dd.append(directions)
            LO.append(s[si].astype(float))
            HI.append(s[si + 1].astype(float))
            BLO.append(bv[ci, si])
    if geometry["random"] is not None:
        P0, D, T, sl = geometry["random"]
        bv = bfield[sl].reshape(T.shape)
        sg = np.sign(bv)
        li, ti = np.nonzero(sg[:, :-1] * sg[:, 1:] < 0)
        if li.size:
            O.append(P0[li])
            Dd.append(D[li])
            LO.append(T[li, ti].astype(float))
            HI.append(T[li, ti + 1].astype(float))
            BLO.append(bv[li, ti])

    roots = list(zero_points)
    metrics["brackets"] = int(sum(o.shape[0] for o in O))
    if O:
        t0 = time.perf_counter()
        origins = np.concatenate(O, axis=0)
        m = origins.shape[0]
        size = _pad_size(m)
        bisected = bundle.bisect_line(
            jnp.asarray(_pad_rows(origins, size)),
            jnp.asarray(_pad_rows(np.concatenate(Dd, axis=0), size)),
            jnp.asarray(_pad_rows(np.concatenate(LO), size)),
            jnp.asarray(_pad_rows(np.concatenate(HI), size)),
            jnp.asarray(_pad_rows(np.concatenate(BLO), size)),
            c,
            int(refine_iters),
        )
        roots.append(np.asarray(jax.device_get(bisected))[:m])
        metrics["bisection_s"] = time.perf_counter() - t0
    if not roots:
        return np.empty((n, 0))
    return np.concatenate(roots, axis=0).T


def check_b_manifold_exact_gpu(
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
    scan_chunk=SCAN_CHUNK,
    decay_rate=0.0,
):
    """Reference-accuracy manifold check with the scans and the bisection on the GPU
    (engine="jax").  Same knobs and defaults as exact_check_cpu.check_b_manifold_exact.
    """
    bounds = np.asarray(bounds, dtype=float)
    n = bounds.shape[0]
    started = time.perf_counter()
    metrics = {
        "engine": "gpu_jax",
        "build_s": 0.0,
        "scan_s": 0.0,
        "bisection_s": 0.0,
        "sympy_s": 0.0,
        "polish_s": 0.0,
        "total_s": 0.0,
        "scan_points": 0,
        "brackets": 0,
        "roots_before_polish": 0,
    }
    try:
        t0 = time.perf_counter()
        bundle, c = get_jax_candidate(V_expr, fSR, GSR, n, input_index)
        metrics["build_s"] = time.perf_counter() - t0
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
        metrics["scan_points"] = int(geometry["coordinates"].shape[0])
        R = manifold_roots_gpu(
            bundle,
            jnp.asarray(c),
            geometry,
            refine_iters,
            zero_tol,
            scan_chunk,
            metrics,
        )
        metrics["roots_before_polish"] = int(R.shape[1])
        if R.shape[1] == 0:
            metrics["total_s"] = time.perf_counter() - started
            result = BManifoldResult(0, 0, np.nan, np.empty((0, n)), "ok")
            result.metrics = metrics
            return result
        t0 = time.perf_counter()
        fns = reference_callables(
            V_expr, fSR, GSR, state_symbols(n), input_index, gamma1
        )
        metrics["sympy_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        R = polish_manifold_roots(
            R, fns, bounds, origin_tol, polish_top_k, polish_maxiter, polish_b_tol
        )
        metrics["polish_s"] = time.perf_counter() - t0
        result = score_manifold_roots(R, fns["m_fn"], origin_tol, margin_tol)
        metrics["total_s"] = time.perf_counter() - started
        result.metrics = metrics
        return result
    except Exception as exc:
        metrics["total_s"] = time.perf_counter() - started
        result = BManifoldResult(
            0, 0, np.nan, np.empty((0, n)), f"error: {type(exc).__name__}: {exc}"
        )
        result.metrics = metrics
        return result


def check_positive_definite_gpu(
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
    scan_chunk=SCAN_CHUNK,
    fSR=None,
    GSR=None,
    input_index=1,
):
    """PD with V on the lattice from the GPU (engine="jax"), polish on the CPU."""
    started = time.perf_counter()
    bounds = np.asarray(bounds, dtype=float)
    n = bounds.shape[0]
    metrics = {
        "engine": "gpu_jax",
        "scan_s": 0.0,
        "polish_s": 0.0,
        "total_s": 0.0,
        "scan_points": 0,
        "polish_starts": 0,
    }
    try:
        V_fn, grad_fn, v0 = lambdify_V(V_expr, n)
        bundle, c = get_jax_candidate(V_expr, fSR, GSR, n, input_index)
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
        values = eval_on_device(bundle.V_batch, coords, jnp.asarray(c), scan_chunk)
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
# engine="pallas": the vendored srcGPU5_9 engine (4-D cart-pole)
# --------------------------------------------------------------------------


def runtime_candidate(expression, constants=None):
    """RuntimeExactCandidate from a GP prefix string + constants (a sympy V is converted)."""
    from src.srcGPU.gpu5_9.runtime_exact_candidate import RuntimeExactCandidate

    if isinstance(expression, sp.Basic):
        expression, constants = to_prefix(expression)
    return RuntimeExactCandidate(
        expression=str(expression),
        constants=tuple(
            np.asarray(
                [] if constants is None else constants, dtype=np.float64
            ).reshape(-1)
        ),
    )


def _pallas_artstein_kwargs(bounds, gamma1, input_index, kw):
    out = dict(
        bounds=[tuple(b) for b in np.asarray(bounds, float)],
        gamma1=gamma1,
        input_index=input_index,
        margin_tol=-C.MARGIN_TOL,
        origin_tol=C.ORIGIN_TOL,
        scan_axes=tuple(range(len(bounds))),
        polish_top_k=40,
        polish_maxiter=60,
        polish_b_tol=C.B_TOL,
        mesh_points_per_axis=0,
    )
    out.update(kw)
    return out


def _pallas_pd(cand, bounds, **kw):
    from src.srcGPU.gpu5_9.pd_check_gpu5_9 import check_positive_definite_gpu5

    args = dict(
        pd_eps=C.PD_EPS,
        origin_tol=C.ORIGIN_TOL,
        scan_axes=tuple(range(len(bounds))),
        scan_points=801,
        grid_points_per_axis=9,
        random_lines=200,
        random_line_points=401,
        rng_seed=0,
        mesh_points_per_axis=21,
        polish_top_k=40,
        polish_maxiter=60,
        pd_reference_matrix=None,
    )
    args.update(kw)
    pd = check_positive_definite_gpu5(
        cand, [tuple(b) for b in np.asarray(bounds, float)], **args
    )
    return PDResult(
        pd.n_violations,
        pd.min_margin,
        pd.violation_points,
        pd.status,
        pd.positive_definite,
        pd.min_point,
        dict(pd.gpu5_metrics, engine="gpu5_pallas"),
    )


def verify_exact_gpu(
    expression,
    constants=None,
    *,
    fSR,
    GSR,
    bounds,
    gamma1=C.DEFAULT_GAMMA1,
    input_index=1,
    with_artstein=True,
    with_pd=True,
    engine="jax",
    artstein_kwargs=None,
    pd_kwargs=None,
):
    """Both definite conditions for one candidate (GP prefix string + constants, or a sympy V)
    with the chosen engine.  Same result type as exact_check_cpu.verify_exact_cpu."""
    bounds = np.asarray(bounds, dtype=float)
    n = bounds.shape[0]
    art = BManifoldResult(0, 0, np.nan, np.empty((0, n)), "skipped")
    pd, artstein_s, pd_s = None, 0.0, 0.0
    if engine == "jax":
        V = sympy_expression(expression, constants)
        if with_artstein:
            t0 = time.perf_counter()
            art = check_b_manifold_exact_gpu(
                V,
                fSR,
                GSR,
                bounds,
                gamma1=gamma1,
                input_index=input_index,
                **(artstein_kwargs or {}),
            )
            artstein_s = time.perf_counter() - t0
        if with_pd:
            t0 = time.perf_counter()
            pd = check_positive_definite_gpu(
                V,
                bounds,
                fSR=fSR,
                GSR=GSR,
                input_index=input_index,
                **(pd_kwargs or {}),
            )
            pd_s = time.perf_counter() - t0
    elif engine == "pallas":
        from src.srcGPU.gpu5_9.b_manifold_check_gpu3 import (
            check_b_manifold_exact_gpu3_cached,
        )

        cand = runtime_candidate(expression, constants)
        if with_artstein:
            t0 = time.perf_counter()
            art = check_b_manifold_exact_gpu3_cached(
                cand,
                fSR,
                GSR,
                **_pallas_artstein_kwargs(
                    bounds, gamma1, input_index, artstein_kwargs or {}
                ),
            )
            artstein_s = time.perf_counter() - t0
        if with_pd:
            t0 = time.perf_counter()
            pd = _pallas_pd(cand, bounds, **(pd_kwargs or {}))
            pd_s = time.perf_counter() - t0
    else:
        raise ValueError(f"unknown engine {engine!r} (jax | pallas)")
    a_valid, a_semi, p_valid, valid = verdicts(
        art.margin_max, None if pd is None else pd.min_margin
    )
    return ExactResult(art, pd, a_valid, a_semi, p_valid, valid, artstein_s, pd_s)


def verify_exact_pallas_batch(
    candidates,
    *,
    fSR,
    GSR,
    bounds,
    gamma1=C.DEFAULT_GAMMA1,
    input_index=1,
    with_pd=True,
    artstein_kwargs=None,
    pd_kwargs=None,
):
    """GPU4: one batched exact GPU call for many candidates [(expression, constants), ...]
    (bit-identical to GPU3, the launch-bound scan/bisection amortised); any candidate the
    batch path fails on falls back to GPU3.  PD per candidate.  Returns [ExactResult].
    """
    from src.srcGPU.gpu5_9.b_manifold_batch_gpu4 import (
        check_b_manifold_exact_gpu4_batch,
    )
    from src.srcGPU.gpu5_9.b_manifold_check_gpu3 import (
        check_b_manifold_exact_gpu3_cached,
    )

    bounds = np.asarray(bounds, dtype=float)
    cands = [runtime_candidate(e, c) for e, c in candidates]
    kwargs = _pallas_artstein_kwargs(bounds, gamma1, input_index, artstein_kwargs or {})
    t0 = time.perf_counter()
    try:
        results = check_b_manifold_exact_gpu4_batch(cands, fSR, GSR, **kwargs)
    except Exception:
        results = [None] * len(cands)
    batch_s = (time.perf_counter() - t0) / max(1, len(cands))
    out = []
    for cand, art in zip(cands, results):
        artstein_s = batch_s
        if art is None or art.status != "ok":
            t1 = time.perf_counter()
            art = check_b_manifold_exact_gpu3_cached(cand, fSR, GSR, **kwargs)
            artstein_s += time.perf_counter() - t1
        pd, pd_s = None, 0.0
        if with_pd:
            t1 = time.perf_counter()
            pd = _pallas_pd(cand, bounds, **(pd_kwargs or {}))
            pd_s = time.perf_counter() - t1
        a_valid, a_semi, p_valid, valid = verdicts(
            art.margin_max, None if pd is None else pd.min_margin
        )
        out.append(
            ExactResult(art, pd, a_valid, a_semi, p_valid, valid, artstein_s, pd_s)
        )
    return out


def cheap_screen_pallas(
    candidates,
    *,
    bounds,
    gamma1=C.DEFAULT_GAMMA1,
    margin_tol=-C.MARGIN_TOL,
    origin_tol=C.ORIGIN_TOL,
    b_tol=C.B_TOL,
    **options,
):
    """The cheap fully-GPU Artstein screen (224 chords/axis x 225 samples + 112 oblique, 30
    bisection iterations, direct root scoring, NO polish) for many candidates in one launch.
    Understates the margin ~1.4x vs the exact check; a screen, never a verdict.  Returns
    [BManifoldResult] with margin_max = a_max + gamma1."""
    from src.srcGPU.gpu5_9.artstein_gpu5_9 import check_artstein_gpu_many

    cheap = dict(
        fusion_width=32,
        lines_per_axis=224,
        line_samples=225,
        oblique_lines=112,
        max_brackets=6144,
        bisection_iterations=30,
        starts=0,
        iterations=0,
        initial_projection_steps=0,
        projection_steps=0,
        step_size=0.0,
        report_points=64,
    )
    cheap.update(options)
    cands = [runtime_candidate(e, c) for e, c in candidates]
    return check_artstein_gpu_many(
        cands,
        bounds=np.asarray(bounds, float),
        gamma1=gamma1,
        margin_tol=margin_tol,
        origin_tol=origin_tol,
        b_tol=b_tol,
        **cheap,
    )
