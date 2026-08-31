"""Positive-definiteness check with the b=0 pipeline's architecture.

Mirrors ``srcGPU5_7.b_manifold_check_gpu3``:

* stage 1  dense GPU scan of V over the same lattice the b-scan already
  visits, so the coverage is the 2.6M-point one, not the 15^4 fitness grid;
* stage 2  the worst points are compacted on device;
* stage 3  SLSQP polish from the top-K seeds, minimising
      g(x) = V(x) - V(0) - pd_eps*||x||^2
  over the box with the origin ball kept out by the SAME inequality
  constraint the b-polish uses (z@z - r0^2 >= 0), with the exact gradient
  grad(V) - 2*pd_eps*x from the numba dual-number interpreter.

Simpler than the b-check: there is no b=0 equality constraint, so the polish
is a bound- and ball-constrained minimisation only.

Returns a ``PDResult`` shaped like ``BManifoldResult`` so the caller can gate
and penalise it with the same mentality as the Artstein result.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
import time

import os
import numpy as np
from scipy.optimize import minimize

import srcGPU5_7  # noqa: F401  (float64 before JAX)

import jax
import jax.numpy as jnp

# srcGPU2 only sets the JAX_ENABLE_X64 env var, which is a no-op if anything
# imported jax first. Then every float64 array silently becomes float32 and the
# Pallas kernel fails with "Invalid dtype for swap: Ref float64, Value float32".
# Force it here so GPU5 is robust to import order.
jax.config.update("jax_enable_x64", True)

from srcGPU5_7.b_manifold_check_gpu import _scan_geometry
from srcGPU5_7.grid_fitness import MAX_STACK_DEPTH, encode_expression
from srcGPU5_7.pallas_interpreter import DEFAULT_BLOCK_SIZE, evaluate_program_batch
from srcGPU5_7.runtime_exact_candidate import RuntimeExactCandidate
from srcGPU5_7.cpu_polish import _dual2


@dataclass
class PDResult:
    """Positive-definiteness verdict, shaped like BManifoldResult."""

    n_violations: int
    min_margin: float                 # min over the box of W - pd_eps*||x||^2
    violation_points: np.ndarray
    status: str
    positive_definite: bool = True
    min_point: tuple = ()
    gpu5_metrics: dict = field(default_factory=dict)


_PROGRAM_CACHE: OrderedDict = OrderedDict()
_PROGRAM_CACHE_MAX = 64


def _program(expression, constants, block_size):
    key = (str(expression), tuple(np.asarray(constants, dtype=float).reshape(-1)))
    if key in _PROGRAM_CACHE:
        _PROGRAM_CACHE.move_to_end(key)
        return _PROGRAM_CACHE[key]
    program = encode_expression(str(expression))
    values = np.asarray(constants, dtype=np.float64).reshape(-1)
    parameters = np.zeros(program.literals.shape[0] if False else 400)
    parameters[: values.size] = values
    entry = (
        np.asarray(program.opcodes, dtype=np.int32),
        np.asarray(program.operands, dtype=np.int32),
        np.asarray(program.literals, dtype=np.float64),
        int(program.n_ops),
        parameters,
    )
    _PROGRAM_CACHE[key] = entry
    while len(_PROGRAM_CACHE) > _PROGRAM_CACHE_MAX:
        _PROGRAM_CACHE.popitem(last=False)
    return entry


def _scan_values(opcodes, operands, literals, n_ops, parameters, points,
                 block_size, chunk):
    """Evaluate V at every row of ``points`` in block-aligned GPU chunks."""
    total = int(points.shape[0])
    out = []
    op = jnp.asarray(opcodes)[None, :]
    opr = jnp.asarray(operands)[None, :]
    lit = jnp.asarray(literals)[None, :]
    nop = jnp.asarray(np.asarray([n_ops], dtype=np.int32))
    par = jnp.asarray(parameters)[None, :]
    interpret = jax.default_backend() != "gpu"
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        piece = points[start:stop]
        count = stop - start
        padded = ((count + block_size - 1) // block_size) * block_size
        if padded != count:
            piece = jnp.pad(piece, ((0, padded - count), (0, 0)))
        values = evaluate_program_batch(
            op, opr, lit, nop, par, piece,
            max_stack_depth=MAX_STACK_DEPTH,
            block_size=block_size,
            interpret=interpret,
        )
        out.append(np.asarray(jax.device_get(values[0]))[:count])
    return np.concatenate(out) if out else np.empty((0,))


def check_positive_definite_gpu5(
    V_expr,
    bounds,
    pd_eps=1.0e-4,
    origin_tol=1.1e-3,
    scan_axes=(0, 1, 2, 3),
    scan_points=801,
    grid_points_per_axis=9,
    random_lines=200,
    random_line_points=401,
    rng_seed=0,
    mesh_points_per_axis=21,
    polish_top_k=40,
    polish_maxiter=60,
    pd_random_points=0,
    pd_reference_matrix=None,
    block_size=None,
    scan_chunk=262144,
):
    """Minimise W - pd_eps*Q(x) over the box. Negative => not a CLF.

    ``Q`` is ``||x||^2`` by default. Pass ``pd_reference_matrix=P`` (symmetric
    positive definite) to compare against ``x^T P x`` instead -- e.g. the LQR
    quadratic already used as the properness reference. Both are valid class-K
    bounds (lam_min*||x||^2 <= x^T P x <= lam_max*||x||^2), but they are NOT
    interchangeable in practice: the cart-pole P has condition number ~1712,
    so the required margin differs by three orders of magnitude between the
    stiff and soft directions. Using P makes the positive-definiteness test and
    the properness lower bound the SAME condition in the SAME metric, instead
    of two different comparisons doing overlapping jobs.
    """
    started = time.perf_counter()
    bounds = np.asarray(bounds, dtype=float)
    n = bounds.shape[0]
    metrics = {"engine": "gpu5", "scan_s": 0.0, "polish_s": 0.0,
               "total_s": 0.0, "scan_points": 0, "polish_starts": 0}
    try:
        if isinstance(V_expr, RuntimeExactCandidate):
            expression, constants = V_expr.expression, V_expr.constants
        else:
            raise TypeError("GPU5 expects a RuntimeExactCandidate")
        bs = int(block_size or DEFAULT_BLOCK_SIZE)
        opcodes, operands, literals, n_ops, parameters = _program(
            expression, constants, bs
        )

        # --- stage 1: dense GPU scan on the same lattice the b-scan uses ----
        geometry = _scan_geometry(
            bounds, scan_axes, scan_points, grid_points_per_axis,
            random_lines, random_line_points, rng_seed, mesh_points_per_axis,
        )
        coords = np.asarray(jax.device_get(geometry["coordinates"]))
        # Optional uniform random block. Measured: it did NOT improve the
        # 121910 semidefinite case (the shortfall there was the polish, not the
        # sampling) and cost ~+40% scan time, so it is off by default and kept
        # only as a knob for lattices that under-sample.
        if int(pd_random_points) > 0:
            rng = np.random.default_rng(int(rng_seed))
            extra = rng.uniform(bounds[:, 0], bounds[:, 1],
                                size=(int(pd_random_points), n))
            coords = np.concatenate([coords, extra], axis=0)
        metrics["scan_points"] = int(coords.shape[0])
        t0 = time.perf_counter()
        values = _scan_values(opcodes, operands, literals, n_ops, parameters,
                              jnp.asarray(coords), bs, int(scan_chunk))
        metrics["scan_s"] = time.perf_counter() - t0

        origin = np.zeros(n)
        v0 = float(_dual2(opcodes.astype(np.int64), operands.astype(np.int64),
                          literals, n_ops, parameters, origin)[0])
        r2 = np.sum(coords * coords, axis=1)
        if pd_reference_matrix is None:
            P = None
            quad = r2
        else:
            P = np.asarray(pd_reference_matrix, dtype=float).reshape(n, n)
            P = 0.5 * (P + P.T)                       # symmetrise
            quad = np.einsum("ij,jk,ik->i", coords, P, coords)
        g = values - v0 - pd_eps * quad
        keep = r2 > origin_tol * origin_tol
        g = np.where(keep & np.isfinite(g), g, np.inf)

        # --- stage 2: worst points become polish seeds ----------------------
        top_k = int(polish_top_k)
        order = np.argsort(g)[:top_k]
        order = order[np.isfinite(g[order])]
        metrics["polish_starts"] = int(order.size)
        best = float(g[order[0]]) if order.size else float("inf")
        best_at = coords[order[0]] if order.size else origin

        # --- stage 3: SLSQP polish, same constraints as the b-polish --------
        t0 = time.perf_counter()
        box = [tuple(bounds[i]) for i in range(n)]
        r0_sq = (origin_tol * (1.0 + 1e-6)) ** 2
        op64, opr64 = opcodes.astype(np.int64), operands.astype(np.int64)

        def g_and_grad(z):
            z = np.asarray(z, dtype=float)
            value, gradient, _ = _dual2(op64, opr64, literals, n_ops,
                                        parameters, z)
            if P is None:
                penalty, dpenalty = float(z @ z), 2.0 * z
            else:
                Pz = P @ z
                penalty, dpenalty = float(z @ Pz), 2.0 * Pz
            return (float(value) - v0 - pd_eps * penalty,
                    np.asarray(gradient, dtype=float) - pd_eps * dpenalty)

        def g_barrier(z):
            # Nelder-Mead cannot take the ball as a constraint, so bar it here.
            z = np.asarray(z, dtype=float)
            if float(z @ z) <= r0_sq:
                return np.inf
            return g_and_grad(z)[0]

        violations = []
        for index in order:
            z0 = np.clip(coords[index], bounds[:, 0], bounds[:, 1])
            # SLSQP with the exact gradient handles well-conditioned wells, but
            # it STALLS on a semidefinite valley where grad(g) vanishes along
            # the flat direction: on job 121910 it returns +1.19e-3 and never
            # goes negative, while Nelder-Mead finds the true -1.259e-05. Run
            # both from every seed and keep the better one; the derivative-free
            # pass costs ~0.2 s for the whole seed set.
            for method in ("SLSQP", "Nelder-Mead"):
                try:
                    if method == "SLSQP":
                        res = minimize(
                            lambda z: g_and_grad(z)[0],
                            z0,
                            jac=lambda z: g_and_grad(z)[1],
                            constraints=[{
                                "type": "ineq",
                                "fun": lambda z: float(z @ z) - r0_sq,
                                "jac": lambda z: 2.0 * z,
                            }],
                            bounds=box,
                            method="SLSQP",
                            options={"maxiter": int(polish_maxiter),
                                     "ftol": 1e-12},
                        )
                    else:
                        res = minimize(
                            g_barrier, z0, bounds=box, method="Nelder-Mead",
                            options={"maxiter": 400, "xatol": 1e-10,
                                     "fatol": 1e-14},
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
                except Exception:
                    continue
        metrics["polish_s"] = time.perf_counter() - t0
        metrics["total_s"] = time.perf_counter() - started

        points = (np.asarray(violations) if violations
                  else np.empty((0, n)))
        return PDResult(
            n_violations=int(points.shape[0]) + int(best <= 0.0 and not violations),
            min_margin=float(best),
            violation_points=points,
            status="ok",
            positive_definite=bool(best > 0.0),
            min_point=tuple(float(v) for v in np.asarray(best_at).reshape(-1)),
            gpu5_metrics=metrics,
        )
    except Exception as exc:
        metrics["total_s"] = time.perf_counter() - started
        return PDResult(
            0, float("nan"), np.empty((0, n)),
            f"error: {type(exc).__name__}: {exc}",
            positive_definite=False, gpu5_metrics=metrics,
        )


# --------------------------------------------------------------------------
# gate + penalty, same mentality as the Artstein result
# --------------------------------------------------------------------------
# GPU5_7 REVIEW FIX: 10.0 -> 0.0. With _cert_penalty now pricing PD through
# the ROA ramp (up to ROA_WEIGHT), this flat base plus the near-zero curve
# left a ~14-point STEP at depth -> 0+: the last infinitesimal improvement was
# worth 14.4 and every one before it ~1 per decade -- the same gate-shaped
# cliff removed from a_max, c_star and the axis shoulders. Set >0 only to
# restore a flat surcharge for failing PD at all.
PD_PENALTY_BASE = float(os.environ.get("SYMCLF_GPU5_7_PD_PENALTY_BASE", "0.0"))
PD_PENALTY_WEIGHT = 12000.0
PD_PENALTY_MAX = 8000.0
PD_M0 = 0.01


def pd_result_penalty(result, base=None):
    """Penalty for a positive-definiteness failure.

    Mirrors the Artstein penalty in BOTH of its parts: a smooth term,
    quadratic near zero and saturating, plus concave near-zero terms with
    non-vanishing slope at depth 0 (without them the penalty was pinned flat
    at 10.0 across the 3e-10..1e-5 depth range the population occupies).

    GPU5_7 fixes the two defects the GPU5_6 audit found:

    1. SCALE INVARIANCE. The GPU5_6 depth was ``-min_margin``, an ABSOLUTE
       number, so scaling V down by k scaled the penalty's argument down by k
       too -- the GP could discount a PD hole without changing whether V is
       positive definite, the same exploit the normalised Artstein margin
       closed. The depth is now divided by the reference quadratic x'Px at
       the minimiser (the same metric ``pd_eps`` compares against), which
       makes the argument the RELATIVE depth of the well and invariant under
       V -> kV.
    2. ONE CURVE. GPU5_6 duplicated the near-zero curve inline with the CPU
       base's original 4/64/256 coefficients, while the example shim cuts
       those coefficients 16x in ``_exact_near0_penalty`` -- so the PD term
       ran 4.37x heavier than every Artstein term it was meant to mirror.
       The curve is now read from ``base._exact_near0_penalty`` whenever the
       caller provides ``base``, so a single override rescales both
       consistently. The inline coefficients remain only as the no-``base``
       fallback.
    """
    if result is None or result.status != "ok":
        return float(PD_PENALTY_MAX)
    depth = -float(result.min_margin)          # >0 when not positive definite
    if not np.isfinite(depth) or depth <= 0.0:
        return 0.0

    # Relative depth: divide by x'Px (or ||x||^2) at the minimiser.
    minimum_point = np.asarray(
        getattr(result, "min_point", ()) or (), dtype=float
    ).reshape(-1)
    if base is not None and minimum_point.size:
        reference = getattr(base, "GPU5_PD_REFERENCE_MATRIX", None)
        if reference is not None:
            P = np.asarray(reference, dtype=float)
            P = 0.5 * (P + P.T)
            quad = float(minimum_point @ P @ minimum_point)
        else:
            quad = float(minimum_point @ minimum_point)
        depth = depth / max(quad, 1.0e-12)

    smooth = PD_PENALTY_WEIGHT * depth * depth / (depth + PD_M0)
    if base is not None and hasattr(base, "_exact_near0_penalty"):
        near_zero = float(base._exact_near0_penalty(depth))
    else:
        near_zero = (
            4.0 * np.sqrt(depth)
            + 64.0 * depth ** 0.25
            + 256.0 * depth ** (1.0 / 6.0)
        )
    return float(min(PD_PENALTY_MAX,
                     PD_PENALTY_BASE + smooth + near_zero))
