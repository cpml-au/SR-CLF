"""GPU3 exact b=0 manifold check: GPU scans + the reference SLSQP polish.

Motivation (measured on job 83224): the GPU2 exact check spends ~0.07 s in the
2.61M-point scan, ~0.4 s in fused bisection, and ~6.4 s in the batched GPU
SQP/KKT polish. That polish is a 60-iteration sequential loop of tiny
40-point kernel launches — it keeps the A100 nearly idle — and its damped-BFGS
SQP is *not* the reference optimizer, so its stationary points can differ
from ``src.b_manifold_check_exact``. Meanwhile the reference's own stage-3
polish (SciPy SLSQP, 40 starts, exact sympy jacobians) costs only tens of
milliseconds on a host core because each problem is four-dimensional.

GPU3 therefore splits the work by what each processor is good at:

* stages 0-2 (the 2.61M-point b field, root masks, and the fused
  line-bisection batch) run unchanged on the GPU through the *unmodified*
  ``srcGPU2`` runtime engine (read-only imports; srcGPU2 files untouched);
* stage 3 (polish) and the final margin evaluation are the reference
  implementation *verbatim*: the same sympy ``diff``/``lambdify`` callables
  and the same ``scipy.optimize.minimize(..., method="SLSQP")`` calls with
  identical constraints, bounds, tolerances, and acceptance tests as
  ``src/b_manifold_check_exact.py``. The sympy build runs in a worker thread
  concurrently with the GPU scan, so its latency is hidden behind device work.

Result: identical decision logic to the CPU reference (roots differ only at
float64 bisection round-off), at roughly one second per candidate instead of
seven, with the GPU doing exactly the part that needs a GPU.

The GPU2 additive mesh-bracket harvest (stage 0) is still available but
defaults to OFF (``mesh_points_per_axis=0``) so the root population matches
the reference exactly. Set 21 to reproduce the GPU2 extra coverage.

Any failure of the GPU fast path falls back to the reference checker itself,
so a result is never less accurate than ``check_b_manifold_exact``.
"""

from collections import OrderedDict
import time

import numpy as np
from scipy.optimize import minimize
from sympy import Matrix, diff, lambdify, symbols

import srcGPU5_7  # noqa: F401  (configure float64 before importing JAX)

import jax
import jax.numpy as jnp

# srcGPU2 only sets the JAX_ENABLE_X64 env var, which is a no-op if anything
# imported jax first; float64 arrays then silently become float32 and the
# Pallas kernels fail with "Invalid dtype for swap". Force it explicitly.
jax.config.update("jax_enable_x64", True)

from src.b_manifold_check import BManifoldResult
from src.b_manifold_check_exact import check_b_manifold_exact
from src.SymFunctions import DeapSimplifier, substitute_paramsCoef
from srcGPU5_7.b_manifold_check_gpu import (
    _ab_padded,
    _bisect_line_padded_device,
    _device_nonzero_many,
    _scan_geometry,
)
from srcGPU5_7.jax_candidate import get_bundle
from srcGPU5_7.runtime_exact_candidate import (
    RuntimeExactCandidate,
    runtime_candidate_bundle,
)
from srcGPU5_7.cpu_polish import make_cpu_polish_callables

def candidate_sympy_expression(V_expr):
    """Rebuild the exact sympy expression the CPU reference would receive.

    Uses the same ``substitute_paramsCoef`` + ``DeapSimplifier`` conversion as
    ``examples/4DCartPoler/Evaluate._sympy_expression``.
    """
    if isinstance(V_expr, RuntimeExactCandidate):
        expr_str = substitute_paramsCoef(
            str(V_expr.expression), list(V_expr.constants)
        )
        return DeapSimplifier(expr_str, should_print=False)
    return V_expr


def _build_reference_callables(V_sym, fSR, GSR, n, input_index, gamma1):
    """The reference's sympy setup, verbatim (both scan and polish layers)."""
    x_syms = symbols(f"x1:{n + 1}")
    grad = Matrix([diff(V_sym, s) for s in x_syms])
    f_vec = fSR(*x_syms)
    G_mat = GSR(*x_syms)
    a_expr = (grad.T * f_vec)[0]
    b_expr = sum(grad[i] * G_mat[i, input_index] for i in range(n))
    margin_expr = a_expr + gamma1
    return {
        "b_fn": lambdify(x_syms, b_expr, "numpy"),
        "m_fn": lambdify(x_syms, margin_expr, "numpy"),
        "a_fn": lambdify(x_syms, a_expr, "numpy"),
        "ga_fn": lambdify(x_syms, [diff(a_expr, s) for s in x_syms], "numpy"),
        "gb_fn": lambdify(x_syms, [diff(b_expr, s) for s in x_syms], "numpy"),
    }


def _attach_gpu3_metrics(result, metrics):
    """Attach profiling metadata under both the GPU3 and GPU2 attribute
    names, so the unchanged stage logging/report tooling keeps working."""
    result.gpu3_metrics = dict(metrics)
    result.gpu2_metrics = result.gpu3_metrics
    return result


def check_b_manifold_exact_gpu3(
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
    mesh_points_per_axis=0,
):
    """Reference-accuracy manifold check with the scans on the GPU.

    Same knobs and defaults as ``check_b_manifold_exact``;
    ``mesh_points_per_axis`` optionally adds the GPU2 grid-mesh bracket
    harvest (0 = reference-strict, the default). ``decay_rate`` is retained
    for API compatibility only.
    """
    bounds = np.asarray(bounds, dtype=float)
    n = bounds.shape[0]
    total_started = time.perf_counter()
    metrics = {
        "engine": "gpu3",
        "geometry_s": 0.0,
        "sympy_build_s": 0.0,
        "sympy_wait_s": 0.0,
        "scan_and_masks_s": 0.0,
        "bisection_s": 0.0,
        "seed_score_s": 0.0,
        "polish_s": 0.0,
        "final_score_s": 0.0,
        "total_s": 0.0,
        "scan_points": 0,
        "brackets": 0,
        "roots_before_polish": 0,
        "polish_started_count": 0,
        "polished_accepted": 0,
        "polish_nfev": 0,
        "margin_point": None,
        "margin_point_a": np.nan,
        "margin_point_abs_b": np.nan,
        "program_nodes": int(getattr(V_expr, "program_nodes", 0)),
    }

    # ---- reference sympy layer -------------------------------------------
    # Built on the MAIN thread, issued right after the async scan dispatch so
    # the in-flight GPU b-field computation overlaps it (JAX ops are async
    # until the first device_get). The earlier ThreadPoolExecutor version
    # overlapped more of the pipeline but its lambdify (an `exec`, GIL-heavy)
    # contended with the host-side mask compaction, inflating scan_and_masks_s
    # ~23x on the A100 (measured 0.07s -> 1.6s). Single-threaded async overlap
    # removes that contention while keeping sympy hidden behind the scan.
    def build_callables():
        started = time.perf_counter()
        if isinstance(V_expr, RuntimeExactCandidate):
            # numba bytecode dual-number a/b/grad callables for the point-wise
            # SLSQP polish: no per-check sympy diff/lambdify. Vectorized
            # seed/final scoring uses the GPU bundle instead (below).
            fns = make_cpu_polish_callables(
                V_expr.expression, V_expr.constants,
                fSR, GSR, gamma1, input_index,
            )
        else:
            V_sym = candidate_sympy_expression(V_expr)
            fns = _build_reference_callables(
                V_sym, fSR, GSR, n, input_index, gamma1
            )
        fns["build_s"] = time.perf_counter() - started
        return fns

    try:
        if isinstance(V_expr, RuntimeExactCandidate):
            bundle, const_values = runtime_candidate_bundle(V_expr)
        else:
            bundle, const_values = get_bundle(
                V_expr, fSR, GSR, input_index, n
            )
        metrics["program_nodes"] = int(
            getattr(bundle, "program_n_ops", metrics["program_nodes"])
        )
        c = jnp.asarray(const_values)
        geometry_started = time.perf_counter()
        geometry = _scan_geometry(
            bounds,
            scan_axes,
            scan_points,
            grid_points_per_axis,
            random_lines,
            random_line_points,
            rng_seed,
            mesh_points_per_axis,
        )
        metrics["geometry_s"] = time.perf_counter() - geometry_started
        metrics["scan_points"] = int(geometry["coordinates"].shape[0])

        # ---- stages 0-2 on the GPU: b field, root masks, fused bisection ---
        scan_started = time.perf_counter()
        scan_values = bundle.b_batch(geometry["coordinates"], c)
        # Build the reference sympy callables now, while the just-dispatched
        # b-field kernel runs asynchronously on the GPU. No second thread ->
        # no GIL contention with the host-side mask compaction below.
        build_started = time.perf_counter()
        fns = build_callables()
        metrics["sympy_build_s"] = float(fns["build_s"])
        metrics["sympy_wait_s"] = time.perf_counter() - build_started
        root_list = []
        mask_keys = []
        masks = []

        def register_mask(key, mask):
            mask_keys.append(key)
            masks.append(mask)

        bm = None
        if geometry["mesh"] is not None:
            axes_lin, g, mesh_slice = geometry["mesh"]
            bm = scan_values[mesh_slice].reshape((g,) * n)
            register_mask(("mesh_zero",), jnp.abs(bm) <= zero_tol)
            sg = jnp.sign(bm)
            for axis in range(n):
                lo_sl = tuple(
                    slice(None, -1) if i == axis else slice(None)
                    for i in range(n)
                )
                hi_sl = tuple(
                    slice(1, None) if i == axis else slice(None)
                    for i in range(n)
                )
                register_mask(
                    ("mesh_change", axis), sg[lo_sl] * sg[hi_sl] < 0
                )

        axis_fields = []
        for field_index, (axis, others, combos, s, scan_slice) in enumerate(
            geometry["axes"]
        ):
            C = combos.shape[0]
            bv = scan_values[scan_slice].reshape(C, scan_points)
            sg = jnp.sign(bv)
            axis_fields.append((axis, others, combos, s, bv))
            register_mask(("axis_zero", field_index), jnp.abs(bv) <= zero_tol)
            register_mask(
                ("axis_change", field_index), sg[:, :-1] * sg[:, 1:] < 0
            )

        random_field = None
        if geometry["random"] is not None:
            P0, D, T, random_slice = geometry["random"]
            bv = scan_values[random_slice].reshape(
                random_lines, random_line_points
            )
            sg = jnp.sign(bv)
            random_field = (P0, D, T, bv)
            register_mask(("random_change",), sg[:, :-1] * sg[:, 1:] < 0)

        compacted = dict(zip(mask_keys, _device_nonzero_many(masks)))
        metrics["scan_and_masks_s"] = time.perf_counter() - scan_started

        bracket_origins = []
        bracket_directions = []
        bracket_lower = []
        bracket_upper = []
        bracket_lower_values = []

        def register_brackets(origins, directions, lower, upper, lower_values):
            count = int(origins.shape[0])
            if count == 0:
                return
            bracket_origins.append(np.asarray(origins, dtype=float))
            bracket_directions.append(np.asarray(directions, dtype=float))
            bracket_lower.append(np.asarray(lower, dtype=float))
            bracket_upper.append(np.asarray(upper, dtype=float))
            bracket_lower_values.append(jnp.asarray(lower_values))
            metrics["brackets"] += count

        # stage 0 (optional, additive): grid-mesh bracket harvest
        if bm is not None:
            zidx = compacted[("mesh_zero",)]
            if zidx is not None:
                root_list.append(
                    np.stack(
                        [axes_lin[i][zidx[i]] for i in range(n)], axis=1
                    )
                )
            for axis in range(n):
                idx = compacted[("mesh_change", axis)]
                if idx is None:
                    continue
                points = np.stack(
                    [axes_lin[i][idx[i]] for i in range(n)], axis=1
                )
                device_idx = tuple(jnp.asarray(item) for item in idx)
                origins = points.copy()
                origins[:, axis] = 0.0
                directions = np.zeros_like(points)
                directions[:, axis] = 1.0
                register_brackets(
                    origins,
                    directions,
                    axes_lin[axis][idx[axis]].astype(float),
                    axes_lin[axis][idx[axis] + 1].astype(float),
                    bm[device_idx],
                )

        # stage 1: axis-aligned line scans
        for field_index, (axis, others, combos, s, bv) in enumerate(
            axis_fields
        ):
            zero_indices = compacted[("axis_zero", field_index)]
            if zero_indices is not None:
                zc, zs = zero_indices
                zero_points = np.empty((zc.size, n))
                for k, coordinate in enumerate(others):
                    zero_points[:, coordinate] = combos[zc, k]
                zero_points[:, axis] = s[zs]
                root_list.append(zero_points)

            change_indices = compacted[("axis_change", field_index)]
            if change_indices is None:
                continue
            ci, si = change_indices
            points = np.empty((ci.size, n))
            for k, coordinate in enumerate(others):
                points[:, coordinate] = combos[ci, k]
            origins = points.copy()
            origins[:, axis] = 0.0
            directions = np.zeros_like(points)
            directions[:, axis] = 1.0
            register_brackets(
                origins,
                directions,
                s[si].astype(float),
                s[si + 1].astype(float),
                bv[(jnp.asarray(ci), jnp.asarray(si))],
            )

        # stage 2: fixed-seed random-direction line scans
        if random_field is not None:
            P0, D, T, bv = random_field
            change_indices = compacted[("random_change",)]
            if change_indices is not None:
                li, ti = change_indices
                register_brackets(
                    P0[li],
                    D[li],
                    T[li, ti].astype(float),
                    T[li, ti + 1].astype(float),
                    bv[(jnp.asarray(li), jnp.asarray(ti))],
                )

        if bracket_origins:
            bisection_started = time.perf_counter()
            bisected = _bisect_line_padded_device(
                bundle,
                c,
                np.concatenate(bracket_origins, axis=0),
                np.concatenate(bracket_directions, axis=0),
                np.concatenate(bracket_lower, axis=0),
                np.concatenate(bracket_upper, axis=0),
                jnp.concatenate(bracket_lower_values, axis=0),
                refine_iters,
            )
            root_list.append(np.asarray(jax.device_get(bisected)))
            metrics["bisection_s"] = time.perf_counter() - bisection_started

        if not root_list:
            metrics["total_s"] = time.perf_counter() - total_started
            empty = _attach_gpu3_metrics(
                BManifoldResult(0, 0, np.nan, np.empty((0, n)), "ok"),
                metrics,
            )
            empty.margin_mean_pos = np.nan
            return empty

        R = np.concatenate(root_list, axis=0).T  # (n, K): reference layout
        metrics["roots_before_polish"] = int(R.shape[1])

        # ---- point-wise polish callables (numba for runtime, sympy else) ---
        b_fn = fns["b_fn"]
        a_fn = fns["a_fn"]
        ga_fn = fns["ga_fn"]
        gb_fn = fns["gb_fn"]

        # Vectorized a,b over a full (n, K) root array via the GPU bundle
        # (same exact-derivative engine as the scan; matches sympy to ~1e-14,
        # far below margin_tol). Replaces the per-check sympy m_fn/b_fn.
        def ab_vec(root_columns):
            a_vals, b_vals = _ab_padded(bundle, c, root_columns.T)
            return np.asarray(a_vals, dtype=float), np.asarray(
                b_vals, dtype=float
            )

        # ---- stage 3: reference SLSQP polish, verbatim ---------------------
        # (src/b_manifold_check_exact.py lines "stage 3"; only the timing
        # instrumentation and nfev accounting are additions.)
        if polish_top_k > 0 and R.shape[1] > 0:
            seed_started = time.perf_counter()
            with np.errstate(all="ignore"):
                a_seed, _ = ab_vec(R)
            m_seed = a_seed + gamma1
            m_seed = np.where(np.isfinite(m_seed), m_seed, -np.inf)
            order = np.argsort(m_seed)[-int(polish_top_k):]
            metrics["seed_score_s"] = time.perf_counter() - seed_started
            metrics["polish_started_count"] = int(order.size)

            polish_started = time.perf_counter()
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
                        jac=lambda z: -np.asarray(
                            ga_fn(*z), dtype=float
                        ).ravel(),
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
                    metrics["polish_nfev"] += int(getattr(res, "nfev", 0))
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
            metrics["polished_accepted"] = len(polished)
            metrics["polish_s"] = time.perf_counter() - polish_started

        # ---- final margins: reference logic, GPU-bundle a,b ----------------
        final_started = time.perf_counter()
        with np.errstate(all="ignore"):
            a_all, b_all = ab_vec(R)
        margin = a_all + gamma1

        nonorigin = np.sqrt(np.sum(R**2, axis=0)) > origin_tol
        viol = nonorigin & np.isfinite(margin) & (margin > margin_tol)
        finite = margin[nonorigin & np.isfinite(margin)]
        margin_max = float(finite.max()) if finite.size else np.nan
        # GPU5_2 (purely additive -- nothing in GPU3/4/5/5_1 reads this):
        # mean of the POSITIVE part of the margin over every non-origin root.
        # margin_max is a single point out of n_roots, so a change that fixes
        # most of the b=0 manifold does not move it; this measures how much of
        # the manifold violates, which is a dense signal for the GP.
        margin_mean_pos = (
            float(np.maximum(finite, 0.0).mean()) if finite.size else np.nan
        )
        metrics["margin_mean_pos"] = margin_mean_pos

        keep = nonorigin & np.isfinite(margin)
        if np.any(keep):
            worst_index = int(np.argmax(np.where(keep, margin, -np.inf)))
            worst = R[:, worst_index]
            metrics["margin_point"] = worst.tolist()
            metrics["margin_point_a"] = float(margin[worst_index]) - float(
                gamma1
            )
            with np.errstate(all="ignore"):
                metrics["margin_point_abs_b"] = float(abs(b_all[worst_index]))
        metrics["final_score_s"] = time.perf_counter() - final_started
        metrics["total_s"] = time.perf_counter() - total_started

        result = _attach_gpu3_metrics(
            BManifoldResult(
                n_roots=int(nonorigin.sum()),
                n_violations=int(viol.sum()),
                margin_max=margin_max,
                violation_points=R[:, viol].T,
                status="ok",
            ),
            metrics,
        )
        result.margin_mean_pos = margin_mean_pos
        return result
    except Exception as exc:
        # Fail open to the reference itself: the answer can be slower here,
        # but it can never be less accurate than check_b_manifold_exact.
        metrics["engine"] = "cpu_reference_fallback"
        metrics["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        fallback_started = time.perf_counter()
        try:
            V_sym = candidate_sympy_expression(V_expr)
        except Exception as sym_exc:
            metrics["total_s"] = time.perf_counter() - total_started
            return _attach_gpu3_metrics(
                BManifoldResult(
                    0, 0, np.nan, np.empty((0, n)),
                    f"error: {type(sym_exc).__name__}: {sym_exc}",
                ),
                metrics,
            )
        result = check_b_manifold_exact(
            V_sym,
            fSR,
            GSR,
            bounds,
            decay_rate=decay_rate,
            gamma1=gamma1,
            input_index=input_index,
            scan_axes=scan_axes,
            scan_points=scan_points,
            grid_points_per_axis=grid_points_per_axis,
            margin_tol=margin_tol,
            origin_tol=origin_tol,
            refine_iters=refine_iters,
            zero_tol=zero_tol,
            random_lines=random_lines,
            random_line_points=random_line_points,
            rng_seed=rng_seed,
            polish_top_k=polish_top_k,
            polish_maxiter=polish_maxiter,
            polish_b_tol=polish_b_tol,
        )
        metrics["polish_s"] = time.perf_counter() - fallback_started
        metrics["total_s"] = time.perf_counter() - total_started
        return _attach_gpu3_metrics(result, metrics)


# --------------------------------------------------------------------------
# result memoization: identical candidates verified once per process
# (same policy as srcGPU5_7.b_manifold_check_gpu)
# --------------------------------------------------------------------------
_RESULT_CACHE: OrderedDict = OrderedDict()
_RESULT_CACHE_MAX = 8192
_CACHE_STATS = {"hits": 0, "misses": 0}


def _cache_key(tag, V_expr, bounds, kwargs):
    if isinstance(V_expr, RuntimeExactCandidate):
        expr_key = ("runtime", V_expr.expression, V_expr.constants)
    else:
        expr_key = ("sympy", str(V_expr))
    return (
        tag,
        expr_key,
        np.asarray(bounds, dtype=float).tobytes(),
        tuple(sorted((str(k), repr(v)) for k, v in kwargs.items())),
    )


def check_b_manifold_exact_gpu3_cached(V_expr, fSR, GSR, bounds, **kwargs):
    key = _cache_key("gpu3", V_expr, bounds, kwargs)
    if key in _RESULT_CACHE:
        _CACHE_STATS["hits"] += 1
        _RESULT_CACHE.move_to_end(key)
        return _RESULT_CACHE[key]
    _CACHE_STATS["misses"] += 1
    result = check_b_manifold_exact_gpu3(V_expr, fSR, GSR, bounds, **kwargs)
    if result.status == "ok":
        _RESULT_CACHE[key] = result
        while len(_RESULT_CACHE) > _RESULT_CACHE_MAX:
            _RESULT_CACHE.popitem(last=False)
    return result


def cache_stats():
    return dict(_CACHE_STATS)
