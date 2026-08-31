"""Exact analytic ||grad V|| at arbitrary points, for the GPU5_4 normalised
Artstein margin.

WHY THIS EXISTS
---------------
``a = grad(V).f`` and ``b = grad(V).g`` are both LINEAR in ``grad(V)``. Scaling
V by k > 0 leaves ``{b = 0}`` and ``sign(a)`` identical, so the Artstein
condition constrains only the DIRECTION field of ``grad(V)`` -- it is blind to
the gradient's magnitude. The penalty built on the raw margin is not, and that
asymmetry is an exploit: wherever the search cannot get ``grad(V).f < 0``, the
cheapest local move is to shrink ``grad(V)`` there, which drives ``a -> 0`` and
buys margin without fixing anything. It also flattens V, which is precisely
what the properness and positive-definiteness terms then charge for. The two
conditions look mutually exclusive because the search is sliding along that
one-parameter family of flatness, trading one penalty for the other.

Measured on run 122299 generation 76: ``margin_max = 8.64e-06`` (essentially
bought) with 441 of 194481 grid points at ``V - V(0) <= 0`` and 795 improper
points. The zero set and the near-zero margin are the same defect.

``a / ||grad V||`` is invariant under ``grad(V) -> lambda grad(V)``, so
flattening buys exactly nothing. It is also the natural quantity: the
directional derivative of V along the drift, per unit gradient.

SOUNDNESS
---------
``||grad V|| > 0``, so ``a / ||grad V|| < 0`` if and only if ``a < 0``. The
validity DECISION is therefore unchanged -- ``n_violations`` and the sign test
in the checkers are untouched. Only the magnitude used for fitness SHAPING
changes. Nothing here can certify a candidate the raw check would reject.

The gradient is the exact analytic one from the runtime dual-number
interpreter (``srcGPU5_7.runtime_exact_candidate._dual2_point``), the same engine
the exact checker uses for a and b -- no finite differences.
"""

from __future__ import annotations

import os
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from srcGPU5_7.runtime_exact_candidate import (
    MAX_CONSTANTS,
    _cartpole_fields,
    _dual2_point,
    get_runtime_bundle,
)

# Points are padded up to one of these bucket sizes so the vmapped interpreter
# compiles a handful of shapes instead of one per candidate.
_BUCKETS = (16, 64, 256, 1024, 4096, 16384)

# Hard ceiling on how many points are normalised in one call. Violating-root
# populations are typically 1e1-1e3; the cap only bites on degenerate blobs
# (run 122252's champion had 268189 violating roots), where a subsample of the
# worst-margin points is a faithful estimate of the maximum anyway.
_MAX_POINTS = int(os.environ.get("SYMCLF_GPU5_7_GRAD_NORM_MAX_POINTS", "16384"))


@lru_cache(maxsize=len(_BUCKETS))
def _a_and_grad_kernel(point_count):
    """Exact ``(a, ||grad V||)`` per point, one forward dual-number pass.

    ``a = grad(V) . f`` is recomputed here rather than read back from the
    checker because the batched GPU4 path reports only the aggregate
    ``margin_max`` and the violating-point set, not per-root margins. Same
    interpreter and same drift field the checker itself uses, so the values
    agree with its ``a`` to interpreter precision.
    """

    @jax.jit
    def kernel(opcodes, operands, literals, n_ops, parameters, points):
        def one(point):
            drift, _ = _cartpole_fields(point)
            value, gradient, _ = _dual2_point(
                opcodes, operands, literals, n_ops, parameters, point
            )
            # GPU5_7: V(point) is returned as well, so the exact stage can
            # price violations by their sublevel W = V - V(0) -- a violation
            # at W ~ 0 empties the certificate, one near the boundary barely
            # shrinks it. Same forward pass, zero extra cost.
            return value, gradient @ drift, jnp.sqrt(jnp.sum(gradient * gradient))

        return jax.vmap(one)(points)

    return kernel


def _bucket(count):
    for size in _BUCKETS:
        if count <= size:
            return size
    return _BUCKETS[-1]


def a_and_grad_norms(expression, constants, points):
    """Exact ``(V, a, ||grad V||)`` at ``points`` (shape ``(M, 4)``).

    Returns three float arrays of length M. Returns all-NaN arrays rather
    than raising if the candidate cannot be encoded: the caller then falls
    back to the raw margin, which is the GPU5_3 behaviour.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty
    if points.shape[1] != 4:
        raise ValueError(f"expected (M, 4) points; received {points.shape}")
    if points.shape[0] > _MAX_POINTS:
        points = points[:_MAX_POINTS]

    try:
        bundle, values = get_runtime_bundle(str(expression), constants)
    except Exception:
        nan = np.full(len(points), np.nan)
        return nan, nan, nan

    parameters = jnp.pad(
        jnp.asarray(values, dtype=jnp.float64),
        (0, MAX_CONSTANTS - int(np.asarray(values).size)),
    )
    real = len(points)
    size = _bucket(real)
    padded = np.concatenate(
        [points, np.repeat(points[:1], size - real, axis=0)], axis=0
    ) if size > real else points

    try:
        v_values, a_values, norms = _a_and_grad_kernel(size)(
            bundle.opcodes,
            bundle.operands,
            bundle.literals,
            bundle.n_ops,
            parameters,
            jnp.asarray(padded),
        )
    except Exception:
        nan = np.full(real, np.nan)
        return nan, nan, nan
    return (
        np.asarray(v_values, dtype=np.float64)[:real],
        np.asarray(a_values, dtype=np.float64)[:real],
        np.asarray(norms, dtype=np.float64)[:real],
    )
