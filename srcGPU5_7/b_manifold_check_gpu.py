"""JAX/GPU port of the exact-gradient b=0 manifold check.

Drop-in replacement for ``src.b_manifold_check_exact.check_b_manifold_exact``
(same core signature/defaults and BManifoldResult) with unchanged scan
architecture and resolution:

* stage 1: axis-aligned line scans, identical lattice (801 pts, 9/axis),
  identical bisection with a 60-iteration cap and an exact floating-point
  stagnation exit (the exit only fires when another midpoint is impossible);
* stage 2: 200 fixed-seed random-direction lines, identical;
* stage 3: fixed-shape GPU SQP/KKT polish from the top-40 roots on the
  punctured domain with exact JAX gradients, damped BFGS, parallel line
  search, and batched Newton restoration to b=0. All starts advance together;
  no SciPy/CPU optimizer is involved;
* NEW stage 0 (additive coverage only): b is also evaluated on the same
  21-per-axis mesh the grid stage uses, and its sign changes are harvested
  as extra bisection brackets. The dedicated scan lattice is NOT shrunk.

Differences from the CPU exact module are in the execution engine and polish
algorithm. GP actor candidates use analytic float64 forward derivatives and
fixed-width runtime bytecode, so structural mutations do not trigger per-tree
XLA compilation (``srcGPU5_7.runtime_exact_candidate``). Legacy SymPy callers
retain the per-template JAX autodiff path (``srcGPU5_7.jax_candidate``).
The GPU SQP and Fortran SLSQP iterates are not bitwise identical; both use 40
starts, 60 iterations, float64 derivatives, the same bounds, and the same
final b-feasibility tolerance. Every unpolished scan root is retained.

``check_b_manifold_exact_gpu_cached`` adds result memoization keyed on the
exact expression string: identical candidates (clones, elites) are verified
once per process.
"""

from collections import OrderedDict
import time

import jax
import jax.numpy as jnp
import numpy as np

from src.b_manifold_check import BManifoldResult
from srcGPU5_7.jax_candidate import get_bundle
from srcGPU5_7.runtime_exact_candidate import (
    RuntimeExactCandidate,
    runtime_candidate_bundle,
)

_PAD_MIN = 256
_SCAN_GEOMETRY_CACHE: OrderedDict = OrderedDict()
_SCAN_GEOMETRY_CACHE_MAX = 4


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


def _ab_padded(bundle, c, R):
    m = R.shape[0]
    size = _pad_size(m)
    a, b = bundle.ab_batch(jnp.asarray(_pad_rows(R, size)), c)
    return np.asarray(a)[:m], np.asarray(b)[:m]


def _device_nonzero_many(masks):
    """Compact all scan masks with two collective device barriers."""
    if not masks:
        return []
    counts = np.asarray(
        jax.device_get(jnp.stack([jnp.count_nonzero(mask) for mask in masks])),
        dtype=np.int64,
    )
    pending = []
    positions = []
    results = [None] * len(masks)
    for position, (mask, count) in enumerate(zip(masks, counts)):
        count = int(count)
        if count == 0:
            continue
        size = _pad_size(count)
        pending.append(jnp.nonzero(mask, size=size, fill_value=0))
        positions.append((position, count))
    if pending:
        compacted = jax.device_get(pending)
        for (position, count), indices in zip(positions, compacted):
            results[position] = tuple(
                np.asarray(index)[:count] for index in indices
            )
    return results


def _bisect_axis_padded_device(bundle, c, P, axis, lo, hi, blo, iters, n):
    """Dispatch a padded axis bisection without synchronizing the host."""
    m = P.shape[0]
    size = _pad_size(m)
    onehot = np.zeros(n)
    onehot[axis] = 1.0
    roots = bundle.bisect_axis(
        jnp.asarray(_pad_rows(P, size)),
        jnp.asarray(onehot),
        jnp.asarray(_pad_rows(lo, size)),
        jnp.asarray(_pad_rows(hi, size)),
        jnp.asarray(_pad_rows(np.asarray(blo), size)) if isinstance(blo, np.ndarray)
        else jnp.concatenate(
            [blo, jnp.repeat(blo[:1], size - m, axis=0)], axis=0
        ),
        c,
        int(iters),
    )
    return roots[:m]


def _bisect_line_padded_device(bundle, c, P0, D, lo, hi, blo, iters):
    """Dispatch a padded arbitrary-line bisection without host sync."""
    m = P0.shape[0]
    size = _pad_size(m)
    blo_padded = (
        jnp.asarray(_pad_rows(np.asarray(blo), size))
        if isinstance(blo, np.ndarray)
        else jnp.concatenate(
            [blo, jnp.repeat(blo[:1], size - m, axis=0)], axis=0
        )
    )
    roots = bundle.bisect_line(
        jnp.asarray(_pad_rows(P0, size)),
        jnp.asarray(_pad_rows(D, size)),
        jnp.asarray(_pad_rows(lo, size)),
        jnp.asarray(_pad_rows(hi, size)),
        blo_padded,
        c,
        int(iters),
    )
    return roots[:m]


def _scan_geometry(
    bounds,
    scan_axes,
    scan_points,
    grid_points_per_axis,
    random_lines,
    random_line_points,
    rng_seed,
    mesh_points_per_axis,
):
    """Build immutable scan coordinates once per persistent GPU actor."""
    key = (
        tuple(np.asarray(bounds).shape),
        np.asarray(bounds, dtype=np.float64).tobytes(),
        tuple(int(axis) for axis in scan_axes),
        int(scan_points),
        int(grid_points_per_axis),
        int(random_lines),
        int(random_line_points),
        int(rng_seed),
        int(mesh_points_per_axis or 0),
    )
    if key in _SCAN_GEOMETRY_CACHE:
        _SCAN_GEOMETRY_CACHE.move_to_end(key)
        return _SCAN_GEOMETRY_CACHE[key]

    n = bounds.shape[0]
    geometry = {"mesh": None, "axes": [], "random": None}
    if mesh_points_per_axis and int(mesh_points_per_axis) > 1:
        axes_lin = [
            np.linspace(bounds[i, 0], bounds[i, 1], int(mesh_points_per_axis))
            for i in range(n)
        ]
        mesh = np.meshgrid(*axes_lin, indexing="ij")
        points = np.stack([item.ravel() for item in mesh], axis=1)
        geometry["mesh"] = (
            axes_lin,
            int(mesh_points_per_axis),
            jnp.asarray(points),
        )

    for axis in scan_axes:
        axis = int(axis)
        others = [i for i in range(n) if i != axis]
        lines = [
            np.linspace(bounds[i, 0], bounds[i, 1], grid_points_per_axis)
            for i in others
        ]
        mesh = np.meshgrid(*lines, indexing="ij")
        combos = np.stack([item.ravel() for item in mesh], axis=1)
        line_count = combos.shape[0]
        samples = np.linspace(bounds[axis, 0], bounds[axis, 1], scan_points)
        points = np.empty((line_count * scan_points, n))
        for k, coordinate in enumerate(others):
            points[:, coordinate] = np.repeat(combos[:, k], scan_points)
        points[:, axis] = np.tile(samples, line_count)
        geometry["axes"].append(
            (axis, others, combos, samples, jnp.asarray(points))
        )

    if random_lines > 0:
        rng = np.random.default_rng(rng_seed)
        origins = rng.uniform(bounds[:, 0], bounds[:, 1], (random_lines, n))
        directions = rng.normal(size=(random_lines, n))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            ta = (bounds[:, 0][None, :] - origins) / directions
            tb = (bounds[:, 1][None, :] - origins) / directions
        t0 = np.where(
            np.abs(directions) > 1.0e-12, np.minimum(ta, tb), -np.inf
        ).max(1)
        t1 = np.where(
            np.abs(directions) > 1.0e-12, np.maximum(ta, tb), np.inf
        ).min(1)
        fraction = np.linspace(0.0, 1.0, random_line_points)
        parameters = t0[:, None] + (t1 - t0)[:, None] * fraction[None, :]
        points = (
            origins[:, None, :]
            + parameters[..., None] * directions[:, None, :]
        ).reshape(-1, n)
        geometry["random"] = (
            origins,
            directions,
            parameters,
            jnp.asarray(points),
        )

    # One large pointwise launch saturates the A100 better and avoids six
    # separate candidate-kernel launches. Slices preserve the exact original
    # point order and scan resolution.
    coordinate_chunks = []
    offset = 0
    if geometry["mesh"] is not None:
        axes_lin, g, points = geometry["mesh"]
        stop = offset + points.shape[0]
        coordinate_chunks.append(points)
        geometry["mesh"] = (axes_lin, g, slice(offset, stop))
        offset = stop
    fused_axes = []
    for axis, others, combos, samples, points in geometry["axes"]:
        stop = offset + points.shape[0]
        coordinate_chunks.append(points)
        fused_axes.append(
            (axis, others, combos, samples, slice(offset, stop))
        )
        offset = stop
    geometry["axes"] = fused_axes
    if geometry["random"] is not None:
        origins, directions, parameters, points = geometry["random"]
        stop = offset + points.shape[0]
        coordinate_chunks.append(points)
        geometry["random"] = (
            origins,
            directions,
            parameters,
            slice(offset, stop),
        )
    geometry["coordinates"] = (
        jnp.concatenate(coordinate_chunks, axis=0)
        if coordinate_chunks
        else jnp.empty((0, n), dtype=jnp.float64)
    )

    _SCAN_GEOMETRY_CACHE[key] = geometry
    while len(_SCAN_GEOMETRY_CACHE) > _SCAN_GEOMETRY_CACHE_MAX:
        _SCAN_GEOMETRY_CACHE.popitem(last=False)
    return geometry


def _attach_gpu2_metrics(result, metrics):
    """Attach profiling metadata without changing the shared result API."""
    result.gpu2_metrics = dict(metrics)
    return result


def check_b_manifold_exact_gpu(
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
    polish_step_size=0.25,
    polish_projection_steps=6,
    polish_line_search_steps=8,
    mesh_points_per_axis=21,
):
    """Exact-architecture manifold check on JAX. Same knobs and defaults as
    ``check_b_manifold_exact``; ``mesh_points_per_axis`` adds the stage-0
    grid-mesh bracket harvest (set 0 to disable). ``decay_rate`` retained
    for API compatibility only.
    """
    bounds = np.asarray(bounds, dtype=float)
    n = bounds.shape[0]
    total_started = time.perf_counter()
    metrics = {
        "geometry_s": 0.0,
        "scan_and_masks_s": 0.0,
        "bisection_s": 0.0,
        "seed_score_s": 0.0,
        "polish_s": 0.0,
        "final_score_s": 0.0,
        "total_s": 0.0,
        "scan_points": 0,
        "brackets": 0,
        "roots_before_polish": 0,
        "polished_accepted": 0,
        "final_feasible_roots": 0,
        "final_b_rejected": 0,
        "final_bounds_rejected": 0,
        "final_max_abs_b": np.nan,
        "margin_point": None,
        "margin_point_a": np.nan,
        "margin_point_abs_b": np.nan,
        "program_nodes": int(getattr(V_expr, "program_nodes", 0)),
    }

    try:
        if isinstance(V_expr, RuntimeExactCandidate):
            bundle, const_values = runtime_candidate_bundle(V_expr)
        else:
            # Backward-compatible path for notebook callers that provide a
            # SymPy expression. The GP actor path above avoids per-tree JIT.
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
        scan_started = time.perf_counter()
        scan_values = bundle.b_batch(geometry["coordinates"], c)
        root_list = []
        mask_keys = []
        masks = []

        def register_mask(key, mask):
            mask_keys.append(key)
            masks.append(mask)

        # All root masks are formed on-device from the one fused b field.
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
            register_mask(
                ("random_change",), sg[:, :-1] * sg[:, 1:] < 0
                )

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

        # --- stage 0: additive grid-mesh bracket harvest -------------------
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

        # --- stage 1: axis-aligned line scans ------------------------------
        for field_index, (axis, others, combos, s, bv) in enumerate(axis_fields):
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

        # --- stage 2: fixed-seed random-direction line scans ---------------
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
            return _attach_gpu2_metrics(
                BManifoldResult(0, 0, np.nan, np.empty((0, n)), "ok"),
                metrics,
            )

        R = np.concatenate(root_list, axis=0)  # (K, n)
        metrics["roots_before_polish"] = int(R.shape[0])

        # --- stage 3: fully batched GPU polish on the punctured manifold ----
        if polish_top_k > 0 and R.shape[0] > 0:
            seed_started = time.perf_counter()
            with np.errstate(all="ignore"):
                a_seed, _ = _ab_padded(bundle, c, R)
            m_seed = np.where(np.isfinite(a_seed), a_seed + gamma1, -np.inf)
            top_k = int(polish_top_k)
            real_k = min(top_k, R.shape[0])
            order = np.argsort(m_seed)[-real_k:]
            seeds = np.clip(R[order], bounds[:, 0], bounds[:, 1])
            if real_k < top_k:
                seeds = np.concatenate(
                    [seeds, np.repeat(seeds[:1], top_k - real_k, axis=0)],
                    axis=0,
                )
            metrics["seed_score_s"] = time.perf_counter() - seed_started
            r0_sq = (origin_tol * (1.0 + 1e-6)) ** 2
            polish_started = time.perf_counter()
            polished, accepted, _ = bundle.polish_batch(
                jnp.asarray(seeds),
                c,
                jnp.asarray(bounds[:, 0]),
                jnp.asarray(bounds[:, 1]),
                jnp.asarray(r0_sq),
                jnp.asarray(polish_b_tol),
                jnp.asarray(polish_step_size),
                int(polish_maxiter),
                int(polish_projection_steps),
                int(polish_line_search_steps),
            )
            polished = np.asarray(polished)[:real_k]
            accepted = np.asarray(accepted, dtype=bool)[:real_k]
            metrics["polish_s"] = time.perf_counter() - polish_started
            metrics["polished_accepted"] = int(accepted.sum())
            if np.any(accepted):
                R = np.concatenate([R, polished[accepted]], axis=0)

        final_started = time.perf_counter()
        with np.errstate(all="ignore"):
            a_all, b_all = _ab_padded(bundle, c, R)
        margin = a_all + gamma1
        nonorigin = np.linalg.norm(R, axis=1) > origin_tol
        finite_ab = np.isfinite(margin) & np.isfinite(b_all)
        bound_scale = max(1.0, float(np.max(np.abs(bounds))))
        bound_slack = 64.0 * np.finfo(float).eps * bound_scale
        in_bounds = np.all(
            (R >= bounds[:, 0] - bound_slack)
            & (R <= bounds[:, 1] + bound_slack),
            axis=1,
        )
        on_manifold = np.abs(b_all) <= polish_b_tol
        feasible = nonorigin & finite_ab & in_bounds & on_manifold
        viol = feasible & (margin > margin_tol)
        finite = margin[feasible]
        margin_max = float(finite.max()) if finite.size else np.nan
        metrics["final_feasible_roots"] = int(feasible.sum())
        metrics["final_b_rejected"] = int(
            (nonorigin & finite_ab & in_bounds & ~on_manifold).sum()
        )
        metrics["final_bounds_rejected"] = int(
            (nonorigin & finite_ab & ~in_bounds).sum()
        )
        if np.any(nonorigin & finite_ab):
            metrics["final_max_abs_b"] = float(
                np.max(np.abs(b_all[nonorigin & finite_ab]))
            )
        if np.any(feasible):
            worst_index = int(np.argmax(np.where(feasible, margin, -np.inf)))
            metrics["margin_point"] = R[worst_index].tolist()
            metrics["margin_point_a"] = float(a_all[worst_index])
            metrics["margin_point_abs_b"] = float(abs(b_all[worst_index]))
        metrics["final_score_s"] = time.perf_counter() - final_started
        metrics["total_s"] = time.perf_counter() - total_started

        return _attach_gpu2_metrics(
            BManifoldResult(
                n_roots=int(feasible.sum()),
                n_violations=int(viol.sum()),
                margin_max=margin_max,
                violation_points=R[viol],
                status="ok",
            ),
            metrics,
        )
    except Exception as exc:
        metrics["total_s"] = time.perf_counter() - total_started
        return _attach_gpu2_metrics(
            BManifoldResult(
                0, 0, np.nan, np.empty((0, n)),
                f"error: {type(exc).__name__}: {exc}"
            ),
            metrics,
        )


# --------------------------------------------------------------------------
# result memoization: identical candidates verified once per process
# --------------------------------------------------------------------------
_RESULT_CACHE: OrderedDict = OrderedDict()
_RESULT_CACHE_MAX = 8192
_CACHE_STATS = {"hits": 0, "misses": 0}


def _cached(tag, check_fn, V_expr, fSR, GSR, bounds, kwargs):
    key = (
        tag,
        str(V_expr),
        np.asarray(bounds, dtype=float).tobytes(),
        repr(sorted(kwargs.items())),
    )
    if key in _RESULT_CACHE:
        _RESULT_CACHE.move_to_end(key)
        _CACHE_STATS["hits"] += 1
        return _RESULT_CACHE[key]
    result = check_fn(V_expr, fSR, GSR, bounds=bounds, **kwargs)
    _CACHE_STATS["misses"] += 1
    if result.status == "ok":  # never cache transient errors
        _RESULT_CACHE[key] = result
        while len(_RESULT_CACHE) > _RESULT_CACHE_MAX:
            _RESULT_CACHE.popitem(last=False)
    return result


def check_b_manifold_exact_gpu_cached(V_expr, fSR, GSR, bounds, **kwargs):
    """Memoized drop-in for ``check_b_manifold_exact`` (JAX/GPU engine).

    Key = exact expression string (constants baked in) + bounds + knobs, so
    a hit requires bit-identical inputs — zero accuracy risk, and GP clones
    and re-evaluated elites are verified exactly once.
    """
    return _cached("gpu", check_b_manifold_exact_gpu,
                   V_expr, fSR, GSR, bounds, kwargs)


def check_b_manifold_exact_cpu_cached(V_expr, fSR, GSR, bounds, **kwargs):
    """Same memoization wrapped around the UNMODIFIED CPU sympy check.

    The duplicate-elimination win is engine-independent; use this to get it
    with bit-identical CPU numbers (no GPU, no jit compile cost).
    """
    from src.b_manifold_check_exact import check_b_manifold_exact
    return _cached("cpu", check_b_manifold_exact,
                   V_expr, fSR, GSR, bounds, kwargs)
