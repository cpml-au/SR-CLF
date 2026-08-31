"""Fully GPU-resident batched Artstein falsifier for GPU2 training.

The optimized quantity is exactly ``a(x)`` under the scalar equality
``b(x)=0`` and the punctured box constraint.  There is no ``a-rho*b**2``
surrogate.  Dense deterministic axis chords find the equality manifold,
fixed-shape GPU bisection restores roots, and projected tangent ascent searches
for the largest feasible ``a``.  Candidate programs, roots, and optimizer
states remain on the device until one final result transfer.

This is a high-throughput numerical falsifier, not a formal certificate of
emptiness.  A positive feasible margin is a genuine counterexample; a final
candidate should still be audited with the original full exact checker.

GPU5_2 copy. Byte-for-byte ``srcGPU2.artstein_gpu`` apart from one added reduction:
``margin_mean_pos``, the mean of ``max(margin, 0)`` over the roots this
screen found. srcGPU2 is deliberately not modified -- this is a GPU5_2-owned
copy, the same way srcGPU3 was created rather than editing srcGPU2.

Why the cheap path needs it more than the exact path: ~2500 candidates are
screened per generation and only ~9-17 reach the exact checker, so the cheap
penalty is what actually scores the population. A dense signal only in the
exact path would be felt by <1% of the individuals.
"""

from __future__ import annotations

from collections import OrderedDict
import time

import jax
import jax.numpy as jnp
import numpy as np

from src.b_manifold_check import BManifoldResult
from srcGPU5_7.runtime_exact_candidate import (
    RuntimeExactCandidate,
    runtime_candidate_batch_bundle,
)


_GEOMETRY_CACHE: OrderedDict = OrderedDict()
_GEOMETRY_CACHE_MAX = 4


def _low_discrepancy_axis_chords(
    bounds, lines_per_axis, oblique_lines, samples
):
    """Deterministic 4-D axis and oblique full-box chords."""
    bounds = np.asarray(bounds, dtype=np.float64)
    key = (
        bounds.tobytes(),
        int(lines_per_axis),
        int(oblique_lines),
        int(samples),
    )
    if key in _GEOMETRY_CACHE:
        _GEOMETRY_CACHE.move_to_end(key)
        return _GEOMETRY_CACHE[key]

    n = bounds.shape[0]
    index = np.arange(1, int(lines_per_axis) + 1, dtype=np.float64)
    # Incommensurate rotations give deterministic, well-spread coordinates
    # without a SciPy/CPU random-number dependency.
    rotations = np.sqrt(np.asarray([2.0, 3.0, 5.0, 7.0]))
    unit = np.mod(index[:, None] * rotations[None, :], 1.0)
    span = bounds[:, 1] - bounds[:, 0]
    base = bounds[:, 0] + unit * span

    origins = np.repeat(base[:, None, :], n, axis=1)
    directions = np.zeros_like(origins)
    for axis in range(n):
        origins[:, axis, axis] = bounds[axis, 0]
        directions[:, axis, axis] = span[axis]
    # Flattening interleaves axes, so a fixed bracket cap cannot accidentally
    # retain only x1 lines when many sign changes are present.
    origins = origins.reshape((-1, n))
    directions = directions.reshape((-1, n))

    # Generic full-box chords close the axis-tangency blind spot without
    # host-side randomness.  The construction is fixed across candidates and
    # runs, just like the legacy exact check's fixed-seed random lines.
    if int(oblique_lines) > 0:
        oblique_index = np.arange(
            1, int(oblique_lines) + 1, dtype=np.float64
        )
        point_unit = np.mod(
            oblique_index[:, None]
            * np.sqrt(np.asarray([11.0, 13.0, 17.0, 19.0]))[None, :],
            1.0,
        )
        interior = bounds[:, 0] + point_unit * span
        raw_direction = 2.0 * np.mod(
            oblique_index[:, None]
            * np.sqrt(np.asarray([23.0, 29.0, 31.0, 37.0]))[None, :],
            1.0,
        ) - 1.0
        raw_direction /= np.linalg.norm(
            raw_direction, axis=1, keepdims=True
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            ta = (bounds[:, 0][None, :] - interior) / raw_direction
            tb = (bounds[:, 1][None, :] - interior) / raw_direction
        t0 = np.minimum(ta, tb).max(axis=1)
        t1 = np.maximum(ta, tb).min(axis=1)
        oblique_origins = interior + t0[:, None] * raw_direction
        oblique_directions = (t1 - t0)[:, None] * raw_direction
        origins = np.concatenate([origins, oblique_origins], axis=0)
        directions = np.concatenate([directions, oblique_directions], axis=0)
    parameters = np.linspace(0.0, 1.0, int(samples), dtype=np.float64)
    coordinates = (
        origins[:, None, :]
        + parameters[None, :, None] * directions[:, None, :]
    ).reshape((-1, n))
    result = (
        jnp.asarray(origins),
        jnp.asarray(directions),
        jnp.asarray(parameters),
        jnp.asarray(coordinates),
    )
    _GEOMETRY_CACHE[key] = result
    while len(_GEOMETRY_CACHE) > _GEOMETRY_CACHE_MAX:
        _GEOMETRY_CACHE.popitem(last=False)
    return result


def _runtime_options(base):
    return {
        "fusion_width": int(getattr(base, "GPU2_ARTSTEIN_FUSION", 4)),
        "lines_per_axis": int(
            getattr(base, "GPU2_ARTSTEIN_LINES_PER_AXIS", 1024)
        ),
        "line_samples": int(getattr(base, "GPU2_ARTSTEIN_LINE_SAMPLES", 801)),
        "oblique_lines": int(
            getattr(base, "GPU2_ARTSTEIN_OBLIQUE_LINES", 512)
        ),
        "max_brackets": int(
            getattr(base, "GPU2_ARTSTEIN_MAX_BRACKETS", 8192)
        ),
        "bisection_iterations": int(
            getattr(base, "GPU2_ARTSTEIN_BISECTION_ITERATIONS", 36)
        ),
        "starts": int(getattr(base, "GPU2_ARTSTEIN_STARTS", 256)),
        "iterations": int(getattr(base, "GPU2_ARTSTEIN_ITERATIONS", 24)),
        "initial_projection_steps": int(
            getattr(base, "GPU2_ARTSTEIN_INITIAL_PROJECTION_STEPS", 6)
        ),
        "projection_steps": int(
            getattr(base, "GPU2_ARTSTEIN_PROJECTION_STEPS", 2)
        ),
        "step_size": float(getattr(base, "GPU2_ARTSTEIN_STEP_SIZE", 0.05)),
        "report_points": int(
            getattr(base, "GPU2_ARTSTEIN_REPORT_POINTS", 64)
        ),
    }


def _cheap_runtime_options(base):
    """Fixed-shape options for the mandatory first-stage GPU screen."""
    return {
        "fusion_width": int(getattr(base, "GPU2_CHEAP_FUSION", 32)),
        "lines_per_axis": int(
            getattr(base, "GPU2_CHEAP_LINES_PER_AXIS", 224)
        ),
        "line_samples": int(getattr(base, "GPU2_CHEAP_LINE_SAMPLES", 225)),
        "oblique_lines": int(
            getattr(base, "GPU2_CHEAP_OBLIQUE_LINES", 112)
        ),
        "max_brackets": int(
            getattr(base, "GPU2_CHEAP_MAX_BRACKETS", 6144)
        ),
        "bisection_iterations": int(
            getattr(base, "GPU2_CHEAP_BISECTION_ITERATIONS", 30)
        ),
        # The cheap screen deliberately stops after direct root scoring.  The
        # full exact checker owns the expensive polishing stage.
        "starts": 0,
        "iterations": 0,
        "initial_projection_steps": 0,
        "projection_steps": 0,
        "step_size": 0.0,
        "report_points": int(
            getattr(base, "GPU2_CHEAP_REPORT_POINTS", 64)
        ),
    }


def check_artstein_gpu_many(
    candidates,
    *,
    bounds,
    gamma1,
    margin_tol,
    origin_tol,
    b_tol,
    fusion_width=4,
    lines_per_axis=1024,
    oblique_lines=512,
    line_samples=801,
    max_brackets=8192,
    bisection_iterations=36,
    starts=256,
    iterations=24,
    initial_projection_steps=6,
    projection_steps=2,
    step_size=0.05,
    report_points=64,
):
    """Search max a on b=0 for several runtime GP candidates in one launch."""
    candidates = tuple(candidates)
    if not candidates:
        return []
    if not all(isinstance(item, RuntimeExactCandidate) for item in candidates):
        raise TypeError("GPU Artstein checking requires runtime candidates")

    requested_count = len(candidates)
    fusion_width = max(requested_count, int(fusion_width))
    padded_candidates = candidates + (candidates[-1],) * (
        fusion_width - requested_count
    )
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.shape != (4, 2):
        raise ValueError("GPU Artstein currently requires four state bounds")
    max_brackets = max(int(starts), int(max_brackets))
    report_points = min(max(1, int(report_points)), max_brackets + int(starts))
    started = time.perf_counter()

    try:
        bundle = runtime_candidate_batch_bundle(padded_candidates)
        max_brackets = (
            (max_brackets + bundle.block_size - 1) // bundle.block_size
        ) * bundle.block_size
        origins, directions, parameters, coordinates = (
            _low_discrepancy_axis_chords(
                bounds,
                int(lines_per_axis),
                int(oblique_lines),
                int(line_samples),
            )
        )
        line_count = int(origins.shape[0])
        interval_count = int(line_samples) - 1

        # Stage 1: b-only manifold discovery for all programs on the A100.
        scan_values = bundle.b_batch(coordinates).reshape(
            fusion_width, line_count, int(line_samples)
        )
        lower_values = scan_values[:, :, :-1]
        upper_values = scan_values[:, :, 1:]
        finite_pair = jnp.isfinite(lower_values) & jnp.isfinite(upper_values)
        bracket_mask = finite_pair & (
            (lower_values == 0.0)
            | (upper_values == 0.0)
            | (jnp.signbit(lower_values) != jnp.signbit(upper_values))
        )

        flat_indices = []
        bracket_counts = []
        for candidate_index in range(fusion_width):
            flat_mask = bracket_mask[candidate_index].reshape(-1)
            count = jnp.minimum(jnp.count_nonzero(flat_mask), max_brackets)
            indices = jnp.nonzero(
                flat_mask, size=max_brackets, fill_value=0
            )[0]
            flat_indices.append(indices)
            bracket_counts.append(count)
        flat_indices = jnp.stack(flat_indices)
        bracket_counts = jnp.stack(bracket_counts)
        bracket_active = (
            jnp.arange(max_brackets)[None, :] < bracket_counts[:, None]
        )
        line_index = flat_indices // interval_count
        sample_index = flat_indices % interval_count
        bracket_origins = origins[line_index]
        bracket_directions = directions[line_index]
        lower_parameter = parameters[sample_index]
        upper_parameter = parameters[sample_index + 1]
        flat_scan = scan_values.reshape(fusion_width, -1)
        lower_flat_index = line_index * int(line_samples) + sample_index
        b_lower = jnp.take_along_axis(flat_scan, lower_flat_index, axis=1)

        # Stage 2: every retained equality bracket is bisected in parallel.
        roots = bundle.bisect_line(
            bracket_origins,
            bracket_directions,
            lower_parameter,
            upper_parameter,
            b_lower,
            int(bisection_iterations),
        )
        root_ab = bundle.ab_batch(roots)
        root_a = root_ab[:, 0, :]
        root_b = root_ab[:, 1, :]
        r0_sq = float(origin_tol * (1.0 + 1.0e-6)) ** 2
        root_norm_sq = jnp.sum(roots * roots, axis=2)
        root_valid = (
            bracket_active
            & jnp.isfinite(root_a)
            & jnp.isfinite(root_b)
            & (jnp.abs(root_b) <= b_tol)
            & (root_norm_sq > r0_sq)
        )

        # Optional tangent polishing.  The mandatory cheap screen sets
        # starts=0, so it pays only for scan+bisection+direct root scoring.
        if int(starts) > 0:
            seed_scores = jnp.where(root_valid, root_a, -jnp.inf)
            top_values, top_indices = jax.lax.top_k(
                seed_scores, int(starts)
            )
            seeds = jnp.take_along_axis(
                roots, top_indices[:, :, None], axis=1
            )
            seed_active = jnp.isfinite(top_values)
            polished, polished_valid, polished_a = (
                bundle.artstein_ascent_batch(
                    seeds,
                    seed_active,
                    jnp.asarray(bounds[:, 0]),
                    jnp.asarray(bounds[:, 1]),
                    jnp.asarray(r0_sq),
                    jnp.asarray(b_tol),
                    jnp.asarray(step_size),
                    int(iterations),
                    int(initial_projection_steps),
                    int(projection_steps),
                )
            )
        else:
            polished = jnp.empty((fusion_width, 0, 4), dtype=roots.dtype)
            polished_valid = jnp.empty(
                (fusion_width, 0), dtype=jnp.bool_
            )
            polished_a = jnp.empty(
                (fusion_width, 0), dtype=root_a.dtype
            )

        root_margin = root_a + gamma1
        polished_margin = polished_a + gamma1
        root_violation = root_valid & (root_margin > margin_tol)
        polished_violation = polished_valid & (polished_margin > margin_tol)
        all_points = jnp.concatenate([roots, polished], axis=1)
        all_valid = jnp.concatenate([root_valid, polished_valid], axis=1)
        all_margin = jnp.concatenate([root_margin, polished_margin], axis=1)
        all_violation = jnp.concatenate(
            [root_violation, polished_violation], axis=1
        )
        finite_margin = jnp.where(all_valid, all_margin, -jnp.inf)
        margin_max = jnp.max(finite_margin, axis=1)
        # GPU5_2: device-side sum of max(margin, 0) over the valid roots, so
        # the cheap screen can report a MEASURE of the violation and not just
        # its single worst point. One extra reduction over an array that is
        # already resident; no extra kernel launch, no extra device sync.
        margin_pos_totals = jnp.sum(
            jnp.where(all_valid, jnp.maximum(all_margin, 0.0), 0.0), axis=1
        )
        scan_root_totals = jnp.sum(root_valid, axis=1)
        polished_totals = jnp.sum(polished_valid, axis=1)
        root_totals = jnp.sum(all_valid, axis=1)
        violation_totals = jnp.sum(all_violation, axis=1)

        report_scores = jnp.where(all_violation, all_margin, -jnp.inf)
        report_values, report_indices = jax.lax.top_k(
            report_scores, report_points
        )
        violation_points = jnp.take_along_axis(
            all_points, report_indices[:, :, None], axis=1
        )
        report_valid = jnp.isfinite(report_values)

        # Exactly one device-to-host synchronization for all requested
        # candidates.  No optimizer iteration synchronizes with the CPU.
        (
            host_root_totals,
            host_violation_totals,
            host_margin_max,
            host_points,
            host_report_valid,
            host_bracket_counts,
            host_scan_root_totals,
            host_polished_totals,
            host_margin_pos_totals,
        ) = jax.device_get(
            (
                root_totals[:requested_count],
                violation_totals[:requested_count],
                margin_max[:requested_count],
                violation_points[:requested_count],
                report_valid[:requested_count],
                bracket_counts[:requested_count],
                scan_root_totals[:requested_count],
                polished_totals[:requested_count],
                margin_pos_totals[:requested_count],
            )
        )
        elapsed = time.perf_counter() - started

        results = []
        for index, candidate in enumerate(candidates):
            n_roots = int(host_root_totals[index])
            n_violations = int(host_violation_totals[index])
            value = float(host_margin_max[index])
            if not np.isfinite(value):
                value = np.nan
            # Zero discovered roots is a vacuous numerical screen, not a
            # software failure.  The second-stage policy sends it to exact.
            status = "ok"
            result = BManifoldResult(
                n_roots=n_roots,
                n_violations=n_violations,
                margin_max=value,
                violation_points=np.asarray(host_points[index])[
                    np.asarray(host_report_valid[index], dtype=bool)
                ],
                status=status,
            )
            result.gpu2_metrics = {
                "engine": "gpu_artstein",
                "artstein_total_s": elapsed,
                "total_s": elapsed,
                "device_syncs": 1,
                "scan_points": int(coordinates.shape[0]),
                "brackets": int(host_bracket_counts[index]),
                "roots_before_polish": int(host_scan_root_totals[index]),
                "polished_accepted": int(host_polished_totals[index]),
                "program_nodes": int(bundle.program_n_ops[index]),
                "fused_candidates": fusion_width,
                "a_max": value - float(gamma1),
            }
            result.gpu2_a_max = value - float(gamma1)
            # GPU5_2: mean of the positive part of the margin over every root
            # this screen found. NaN when no roots were found, which the
            # penalty treats as "not applicable" and charges nothing.
            result.margin_mean_pos = (
                float(host_margin_pos_totals[index]) / n_roots
                if n_roots > 0
                else np.nan
            )
            result.gpu2_metrics["margin_mean_pos"] = result.margin_mean_pos
            results.append(result)
        return results
    except Exception as exc:
        elapsed = time.perf_counter() - started
        results = []
        concise_error = str(exc).splitlines()[0][:400]
        for _ in candidates:
            result = BManifoldResult(
                0,
                0,
                np.nan,
                np.empty((0, 4)),
                f"error: {type(exc).__name__}: {concise_error}",
            )
            result.gpu2_metrics = {
                "engine": "gpu_artstein",
                "artstein_total_s": elapsed,
                "total_s": elapsed,
                "device_syncs": 0,
            }
            results.append(result)
        return results


def check_artstein_gpu_many_from_base(candidates, base):
    """Evaluate using the GPU2-only knobs exposed through Evaluate.py."""
    return check_artstein_gpu_many(
        candidates,
        bounds=base.SHGO_BOUNDS,
        gamma1=base.CLF_GAMMA1,
        margin_tol=base.MANIFOLD_MARGIN_TOL_EXACT,
        origin_tol=base.CLF_ORIGIN_EXCLUDE_RADIUS,
        b_tol=float(getattr(base, "GPU2_ARTSTEIN_B_TOL", 1.0e-8)),
        **_runtime_options(base),
    )


def check_artstein_gpu_cheap_many_from_base(candidates, base):
    """Mandatory direct b=0 screen before current-generation exact waves."""
    return check_artstein_gpu_many(
        candidates,
        bounds=base.SHGO_BOUNDS,
        gamma1=base.CLF_GAMMA1,
        margin_tol=base.MANIFOLD_MARGIN_TOL_EXACT,
        origin_tol=base.CLF_ORIGIN_EXCLUDE_RADIUS,
        b_tol=float(getattr(base, "GPU2_CHEAP_B_TOL", 1.0e-8)),
        **_cheap_runtime_options(base),
    )


def warm_artstein_gpu(base):
    """Compile and allocate the fixed-width Artstein graph before evolution."""
    fusion_width = int(getattr(base, "GPU2_ARTSTEIN_FUSION", 4))
    expression = (
        "add(add(mul(x1,x1),mul(x2,x2)),"
        "add(mul(x3,x3),mul(x4,x4)))"
    )
    dummy = RuntimeExactCandidate(expression, ())
    started = time.perf_counter()
    result = check_artstein_gpu_many_from_base(
        [dummy] * fusion_width, base
    )
    if not result or any(item.status.startswith("error:") for item in result):
        status = "failed: " + (
            result[0].status if result else "no warm-up result"
        )
    else:
        status = "ok"
    return {
        "status": status,
        "seconds": time.perf_counter() - started,
        "fusion_width": fusion_width,
    }


def warm_cheap_artstein_gpu(base):
    """Compile the mandatory cheap fixed-width graph before evolution."""
    fusion_width = int(getattr(base, "GPU2_CHEAP_FUSION", 32))
    expression = (
        "add(add(mul(x1,x1),mul(x2,x2)),"
        "add(mul(x3,x3),mul(x4,x4)))"
    )
    dummy = RuntimeExactCandidate(expression, ())
    started = time.perf_counter()
    result = check_artstein_gpu_cheap_many_from_base(
        [dummy] * fusion_width, base
    )
    first_error = next(
        (item.status for item in result if item.status != "ok"), None
    )
    return {
        "status": "ok" if first_error is None else f"failed: {first_error}",
        "seconds": time.perf_counter() - started,
        "fusion_width": fusion_width,
    }
