"""The training-grid (lattice) stage of the CPU pipeline: grid_score and the check cost's anchors.

For n states and m inputs, with the check deterministic_search (margins_mode = 1).  The grid
contributes

    grid_score = symbolic + invalid + margins_roa        (1e10 + symbolic when V is not finite or
                                                       |V| > 1e6 somewhere on the grid)

and the anchors the check cost needs: boundary_min, w_scale = median |W|, c_max, the
sublevel_fraction, and the grid ratios cv_grid / kappa_grid:

    cv_grid           = min over the grid without x = 0 of min(W/|x|^2, x.grad W/(2|x|^2))
    kappa_grid        = -max over {W > 0} of (a - rho*||b||)/W
    boundary_min      = min of W on the faces of the grid
    c_max             = ROA_RHO_MULTIPLIER * boundary_min
    sublevel_fraction = mean(W < c_max), 0 when c_max <= 0  (the fraction of the grid with W < c_max)

margins_roa is the check cost's region/rate rule (src/CPU/rule.py::roa_rule, the three hinges
included) written out on cv_grid, the gated kappa_grid and sublevel_fraction, + 1e6 when no grid
point has W > 0.

Gradients on the grid are finite differences (numpy gradient, edge_order=2, per-axis spacing) -- not
the exact bytecode gradients, which belong to the check.

This is an estimate on the training grid, not a search: it shapes the fitness; the verdict
(numerically verified or not) is the check's.
Tolerances: PD_EPS 1e-4, KAPPA_MIN 0.0, GATE_KAPPA_CAP 1e6; samples within 1e-12 of 0 (relative
to the axis) are set to 0 so the origin is on the grid.
Settings read here: grid.points_per_axis (SYMCLF_CPU_GRID_POINTS), system.box, verification.u_max,
verification.b_norm, verification.pd_eps, verification.kappa_min, fitness.region_weight,
fitness.region_target, fitness.hinge.*, fitness.non_pd_kappa_cap.  Environment only:
SYMCLF_CHECK_KAPPA_SCALE, SYMCLF_FIT_ROA_RHO_MULTIPLIER.
"""
from __future__ import annotations

from functools import lru_cache

import os

import numpy as np

from src.CPU import config
from src.CPU.encode import program_arrays
from src.CPU.evalnb import _M, _N, lattice_vgrad

_RINV = None


def _r_inverse_diagonal():
    """diag(R^-1) of the system (the cart-pole: R = 1e-4 I, so beta = 1e4 |b|^2)."""
    global _RINV
    if _RINV is None:
        if _M == 0:   # no input (Lyapunov-function mode): no input cost
            _RINV = np.empty(0)
        else:
            R = np.asarray(config.system().R_MATRIX, dtype=float).reshape(_M, _M)
            _RINV = np.diag(np.linalg.inv(R)).copy()
    return _RINV


@lru_cache(maxsize=4)
def grid_context(grid_points: int, box_key: tuple):
    """Grid points, per-axis spacing, the dynamics on the grid and |x|^2 (cached)."""
    import sympy as sp

    half = np.asarray(box_key, dtype=float)
    axes = [np.linspace(-h, h, grid_points) for h in half]
    # snap near-zero samples to 0.0 so the origin is the origin.
    axes = [np.where(np.abs(a) < 1.0e-12 * max(np.max(np.abs(a)), 1.0), 0.0, a) for a in axes]
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.stack([c.ravel() for c in mesh], axis=1)

    sys_mod = config.system()
    xs = sp.symbols(f"x1:{_N + 1}")
    cols = [points[:, i] for i in range(_N)]
    P = points.shape[0]

    def _column(entries):
        """Evaluate n sympy scalars on the grid, one lambdify each (constants broadcast)."""
        out = np.empty((P, _N))
        for i, expr in enumerate(entries):
            value = np.asarray(sp.lambdify(xs, expr, "numpy")(*cols), dtype=float)
            out[:, i] = value if value.shape == (P,) else np.full(P, float(value.reshape(-1)[0]))
        return out

    f_sym = sp.Matrix(sys_mod.fSR(*xs))
    G_sym = sp.Matrix(sys_mod.GSR(*xs))
    f_values = _column([f_sym[i] for i in range(_N)])
    if _M == 0:   # no input: an empty input matrix on the grid
        g_values = np.empty((0, P, _N))
    else:
        g_values = np.stack([_column([G_sym[i, k] for i in range(_N)]) for k in range(_M)], axis=0)
    return {
        "axes": axes,
        "shape": tuple(len(a) for a in axes),
        "points": np.ascontiguousarray(points),
        "spacing": np.asarray([float(a[1] - a[0]) for a in axes]),
        "f_values": f_values,
        "g_values": g_values,
        "radius_squared": np.sum(points * points, axis=1),
        "grid_points": grid_points,
    }


def _gradient_axis(values, spacing, axis):
    """numpy gradient(..., edge_order=2) on a uniform grid -- the kernel's stencil."""
    first = (-3.0 * np.take(values, 0, axis=axis) + 4.0 * np.take(values, 1, axis=axis)
             - np.take(values, 2, axis=axis)) / (2.0 * spacing)
    middle = (np.take(values, np.arange(2, values.shape[axis]), axis=axis)
              - np.take(values, np.arange(0, values.shape[axis] - 2), axis=axis)) / (2.0 * spacing)
    last = (3.0 * np.take(values, values.shape[axis] - 1, axis=axis)
            - 4.0 * np.take(values, values.shape[axis] - 2, axis=axis)
            + np.take(values, values.shape[axis] - 3, axis=axis)) / (2.0 * spacing)
    return np.concatenate([np.expand_dims(first, axis), middle, np.expand_dims(last, axis)], axis=axis)


def _boundary_min(values):
    candidates = []
    for axis in range(values.ndim):
        candidates.append(np.min(np.take(values, 0, axis=axis)))
        candidates.append(np.min(np.take(values, values.shape[axis] - 1, axis=axis)))
    return float(np.min(np.stack(candidates)))


def symbolic_structure_penalty(expression):
    """1e6 per nested exp/aq and per ignored state (the rule of
    examples/cartpole4d/BaseEvaluate.py::_symbolic_structure_penalty)."""
    from src.CPU.encode import encode_expression, PUSH_X

    text = str(expression)
    penalty = 0.0
    allowed = int(os.environ.get("SYMCLF_NEST_DEPTH", "1"))   # 1 = no exp in exp, no aq in aq (the default);
    for name in ("exp", "aq"):                                # 2, 3: that many levels allowed
        depth_hits = _nested_calls(text, name) if allowed <= 1 else _nest_depth(text, name) > allowed
        if depth_hits:
            penalty += 1.0e6
    program = encode_expression(text, _N)
    used = {int(op) for op, code in zip(program.operands[: program.n_ops],
                                        program.opcodes[: program.n_ops]) if code == PUSH_X}
    penalty += 1.0e6 * float(_N - len(used))
    return penalty


def _nest_depth(text, name):
    """The largest number of calls to `name` stacked inside one another (1: none nested)."""
    import ast

    def depth(node):
        inner = max((depth(c) for c in ast.iter_child_nodes(node)), default=0)
        return inner + (1 if isinstance(node, ast.Call) and getattr(node.func, "id", None) == name else 0)
    return depth(ast.parse(str(text), mode="eval").body)


def _nested_calls(text, name):
    """True when a call to `name` contains another call to `name`: src/SymFunctions.py's
    detect_nested_function_calls, the same test as the GP's complexity penalty
    (src/CPU/gp_engine.py::_extra).  A call inside some other function does not count.
    """
    from src.SymFunctions import detect_nested_function_calls

    return bool(detect_nested_function_calls(str(text), name))


def lattice_numbers(expression, constants, base=None):
    """The grid stage of one candidate (margins_mode = 1), as a dict."""
    ctx = grid_context(config.grid_points(), tuple(config.box()))
    prog = program_arrays(expression, constants, _N)
    values_flat, _, _ = lattice_vgrad(*prog, ctx["points"])   # raw V on the grid
    # the kernel scores raw V and subtracts V(0) -- same arithmetic, cancellation included
    origin_index = int(np.argmin(ctx["radius_squared"]))
    origin_value = float(values_flat[origin_index])
    relative_flat = values_flat - origin_value
    shape = ctx["shape"]
    relative = relative_flat.reshape(shape)

    gradients = np.stack(
        [_gradient_axis(relative, ctx["spacing"][axis], axis) for axis in range(_N)], axis=-1
    ).reshape(-1, _N)

    f_values = ctx["f_values"]
    g_values = ctx["g_values"]
    a_values = np.sum(gradients * f_values, axis=1)
    if _M == 0:   # no input: b is empty and ||b|| = 0, the rate term is a = grad V . f
        b_values = np.empty((gradients.shape[0], 0))
    else:
        b_values = np.stack([np.sum(gradients * g_values[k], axis=1) for k in range(_M)], axis=1)
    rinv = _r_inverse_diagonal()
    beta = np.sum(b_values * b_values * rinv[None, :], axis=1)      # b' R^-1 b
    radius_squared = ctx["radius_squared"]
    with np.errstate(all="ignore"):
        lambda_values = (np.sqrt(a_values * a_values + radius_squared * beta) + a_values) / (beta + 1.0e-6)
        vdot_values = a_values - beta * lambda_values
        bnorm = (np.sum(np.abs(b_values), axis=1) if config.B_NORM == "l1"
                 else np.linalg.norm(b_values, axis=1))

    valid_v = bool(np.all(np.isfinite(values_flat)) and np.isfinite(origin_value)
                   and np.max(np.abs(values_flat)) <= 1.0e6)
    invalid = 1.0e6 * (
        float(not np.all(np.isfinite(a_values)))
        + float(not np.all(np.isfinite(vdot_values)))
        + float(not np.all(np.isfinite(lambda_values)))
    )

    nonorigin = radius_squared > 0.0
    with np.errstate(all="ignore"):
        cv_w_grid = float(np.min(np.where(nonorigin, relative_flat / np.maximum(radius_squared, 1.0e-300), np.inf)))
        cv_rad_grid = float(np.min(np.where(
            nonorigin,
            np.sum(ctx["points"] * gradients, axis=1) / (2.0 * np.maximum(radius_squared, 1.0e-300)),
            np.inf,
        )))
    cv_grid = min(cv_w_grid, cv_rad_grid)

    w_pos = nonorigin & (relative_flat > 0.0)
    with np.errstate(all="ignore"):
        margins_ratio = np.where(w_pos, (a_values - config.RHO * bnorm) / np.where(w_pos, relative_flat, 1.0), -np.inf)
    margins_has_w = bool(np.any(w_pos))
    kappa_grid = float(-np.max(margins_ratio)) if margins_has_w else 0.0
    margins_pd_ok = bool(cv_grid > config.PD_EPS)
    verified = bool(margins_pd_ok and margins_has_w and kappa_grid > config.KAPPA_MIN)
    kappa_rule = kappa_grid if (config.GATE_KAPPA_CAP <= 0.0 or margins_pd_ok) else -config.GATE_KAPPA_CAP

    raw_boundary_min = _boundary_min(relative)
    w_scale = float(max(np.median(np.abs(relative_flat)), 1.0e-300))
    c_max = config.ROA_RHO_MULTIPLIER * raw_boundary_min
    sublevel_fraction = float(np.mean(relative_flat < c_max)) if c_max > 0.0 else 0.0

    if verified:
        margins_roa = config.ROA_WEIGHT * float(np.clip(1.0 - raw_boundary_min / (w_scale * config.ROA_C_TARGET), 0.0, 1.0))
    else:
        margins_roa = config.ROA_WEIGHT * (
            1.0 + np.log1p(max(-kappa_rule, 0.0) / config.KAPPA_SCALE)
            + 2.0 * max(1.0 - cv_grid / config.PD_EPS, 0.0)
        )
    if not margins_has_w:
        margins_roa += 1.0e6
    margins_roa += (
        config.PEN_CV * float(np.clip((config.PEN_CV_TARGET - cv_grid) / config.PEN_CV_TARGET, 0.0, 1.0))
        + config.PEN_KAPPA * float(np.clip((config.PEN_KAPPA_TARGET - kappa_rule) / config.PEN_KAPPA_TARGET, 0.0, 1.0))
        + config.PEN_VOL * float(np.clip((config.PEN_VOL_TARGET - sublevel_fraction) / config.PEN_VOL_TARGET, 0.0, 1.0))
    )

    symbolic = symbolic_structure_penalty(expression)
    grid_mse = invalid + margins_roa
    grid_score = (grid_mse if valid_v else 1.0e10) + symbolic
    return {
        "grid_score": float(grid_score),
        "symbolic": float(symbolic),
        "invalid": float(invalid),
        "roa": float(margins_roa),
        "valid_v": valid_v,
        "boundary_min": float(raw_boundary_min),
        "c_max": float(c_max),
        "sublevel_fraction": float(sublevel_fraction),
        "w_scale": float(w_scale),
        "cv_grid": float(cv_grid),
        "cv_w_grid": float(cv_w_grid),
        "cv_rad_grid": float(cv_rad_grid),
        "kappa_grid": float(kappa_grid),
        "grid_numerically_verified": verified,
    }
