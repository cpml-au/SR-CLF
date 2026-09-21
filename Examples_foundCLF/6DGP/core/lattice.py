"""The training-grid (lattice) stage in DET4 mode -- src4DCartPoleV8/grid_fitness._score_kernel's
det4 branch, re-expressed for (n states, m inputs) without JAX.

In DET4 mode the grid contributes
    pre_exact = symbolic + invalid + det4_roa
and the anchors the price needs: boundary_min, w_scale, c_max, certified volume, and the lattice
ratios cv_grid / kappa_grid.  Every other grid term of the 4-D kernel is multiplied by
legacy = 0 in this mode and is therefore not computed here.

Gradients on the grid are the SAME finite differences as the kernel (numpy gradient, edge_order=2,
per-axis spacing) -- not the exact bytecode gradients, which belong to the DET4 check.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from core import config
from core.encode import program_arrays
from core.evalnb import _M, _N, lattice_vgrad

_RINV = None


def _r_inverse_diagonal():
    """diag(R^-1) of the system (V8: R = 1e-4 I, so beta = 1e4 |b|^2)."""
    global _RINV
    if _RINV is None:
        R = np.asarray(config.system().R_MATRIX, dtype=float).reshape(_M, _M)
        _RINV = np.diag(np.linalg.inv(R)).copy()
    return _RINV


@lru_cache(maxsize=4)
def grid_context(grid_points: int, box_key: tuple):
    """Grid points, per-axis spacing, the dynamics on the grid and |x|^2 (cached)."""
    import sympy as sp

    half = np.asarray(box_key, dtype=float)
    axes = [np.linspace(-h, h, grid_points) for h in half]
    # V2: snap near-zero samples to exact 0 so the origin is the origin (grid_fitness._context).
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
    """BaseEvaluate._symbolic_structure_penalty: 1e6 per nested exp/aq and per ignored state."""
    from core.encode import encode_expression, PUSH_X

    text = str(expression)
    penalty = 0.0
    for name in ("exp", "aq"):
        depth_hits = _nested_calls(text, name)
        if depth_hits:
            penalty += 1.0e6
    program = encode_expression(text, _N)
    used = {int(op) for op, code in zip(program.operands[: program.n_ops],
                                        program.opcodes[: program.n_ops]) if code == PUSH_X}
    penalty += 1.0e6 * float(_N - len(used))
    return penalty


def _nested_calls(text, name):
    """True when a call to `name` contains another call to `name` (SymFunctions' rule)."""
    import ast

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return False
    stack = []

    class V(ast.NodeVisitor):
        found = False

        def visit_Call(self, node):  # noqa: N802
            is_target = isinstance(node.func, ast.Name) and node.func.id == name
            if is_target and stack:
                V.found = True
            stack.append(is_target)
            self.generic_visit(node)
            stack.pop()

    V.found = False
    V().visit(tree)
    return V.found


def lattice_numbers(expression, constants, base=None):
    """The det4-mode grid stage of one candidate; keys mirror grid_fitness' det4 outputs."""
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
        det4_ratio = np.where(w_pos, (a_values - config.RHO * bnorm) / np.where(w_pos, relative_flat, 1.0), -np.inf)
    det4_has_w = bool(np.any(w_pos))
    kappa_grid = float(-np.max(det4_ratio)) if det4_has_w else 0.0
    det4_pd_ok = bool(cv_grid > config.PD_EPS)
    certified = bool(det4_pd_ok and det4_has_w and kappa_grid > config.KAPPA_MIN)
    kappa_rule = kappa_grid if (config.GATE_KAPPA_CAP <= 0.0 or det4_pd_ok) else -config.GATE_KAPPA_CAP

    raw_boundary_min = _boundary_min(relative)
    w_scale = float(max(np.median(np.abs(relative_flat)), 1.0e-300))
    c_max = config.ROA_RHO_MULTIPLIER * raw_boundary_min
    certified_volume = float(np.mean(relative_flat < c_max)) if c_max > 0.0 else 0.0

    if certified:
        det4_roa = config.ROA_WEIGHT * float(np.clip(1.0 - raw_boundary_min / (w_scale * config.ROA_C_TARGET), 0.0, 1.0))
    else:
        det4_roa = config.ROA_WEIGHT * (
            1.0 + np.log1p(max(-kappa_rule, 0.0) / config.KAPPA_SCALE)
            + 2.0 * max(1.0 - cv_grid / config.PD_EPS, 0.0)
        )
    if not det4_has_w:
        det4_roa += 1.0e6
    det4_roa += (
        config.PEN_CV * float(np.clip((config.PEN_CV_TARGET - cv_grid) / config.PEN_CV_TARGET, 0.0, 1.0))
        + config.PEN_KAPPA * float(np.clip((config.PEN_KAPPA_TARGET - kappa_rule) / config.PEN_KAPPA_TARGET, 0.0, 1.0))
        + config.PEN_VOL * float(np.clip((config.PEN_VOL_TARGET - certified_volume) / config.PEN_VOL_TARGET, 0.0, 1.0))
    )

    symbolic = symbolic_structure_penalty(expression)
    grid_mse = invalid + det4_roa
    pre_exact = (grid_mse if valid_v else 1.0e10) + symbolic
    return {
        "pre_exact": float(pre_exact),
        "symbolic": float(symbolic),
        "invalid": float(invalid),
        "roa": float(det4_roa),
        "valid_v": valid_v,
        "boundary_min": float(raw_boundary_min),
        "c_max": float(c_max),
        "certified_volume": float(certified_volume),
        "w_scale": float(w_scale),
        "cv_grid": float(cv_grid),
        "cv_w_grid": float(cv_w_grid),
        "cv_rad_grid": float(cv_rad_grid),
        "kappa_grid": float(kappa_grid),
        "grid_certified": certified,
    }
