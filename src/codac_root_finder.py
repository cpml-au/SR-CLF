"""CODAC set-inversion search for Artstein counterexample boxes.

The search targets the closed, numerically meaningful bad set

  b(x) = 0,  a(x) >= a_min,  ||x|| >= origin_radius.

CODAC's SIVIA paving returns inner boxes that are certified inside this set
and boundary boxes that remain unresolved at the requested paving precision.
Boundary boxes are candidates, not certified counterexamples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sympy import lambdify, symbols


@dataclass(frozen=True)
class CodacRootResult:
    """Result of one CODAC paving of the bad set."""

    n_inner_boxes: int
    n_boundary_boxes: int
    n_outer_boxes: int
    inner_boxes: np.ndarray
    boundary_boxes: np.ndarray
    status: str


def _boxes_to_numpy(boxes, n_states: int) -> np.ndarray:
    """Convert CODAC IntervalVector boxes to (n_boxes, n_states, 2)."""
    values = []
    for box in boxes:
        values.append(
            [[float(box[i].lb()), float(box[i].ub())] for i in range(n_states)]
        )
    if not values:
        return np.empty((0, n_states, 2))
    return np.asarray(values, dtype=float)


def _codac_function(a_expr, b_expr):
    """Translate scalar SymPy a and b into a CODAC vector function."""
    import codac

    x_syms = symbols("x1:5")
    x = codac.VectorVar(4)
    # SymPy emits ordinary function names (sin, cos, exp, sqrt). Mapping them
    # to CODAC operators retains symbolic expressions and interval extensions.
    codac_namespace = {
        "sin": codac.sin,
        "cos": codac.cos,
        "exp": codac.exp,
        "sqrt": codac.sqrt,
        "tan": codac.tan,
    }
    r2_expr = sum(symbol * symbol for symbol in x_syms)
    expression = lambdify(
        x_syms,
        [b_expr, a_expr, r2_expr],
        modules=[codac_namespace],
    )
    b_codac, a_codac, r2_codac = expression(*(x[i] for i in range(4)))
    return codac.AnalyticFunction([x], codac.vec(b_codac, a_codac, r2_codac))


def find_codac_artstein_roots(
    a_expr,
    b_expr,
    bounds,
    *,
    a_min: float = 1e-10,
    origin_radius: float = 1e-3,
    paving_eps: float = 1e-3,
    verbose: bool = False,
) -> CodacRootResult:
    """Pave b=0, a>=a_min outside the origin exclusion ball.

    CODAC is imported lazily so that importing Evaluate remains valid on
    machines where the optional Python binding has not yet been installed.
    """
    bounds = np.asarray(bounds, dtype=float)
    n_states = bounds.shape[0] if bounds.ndim == 2 else 0
    empty = np.empty((0, n_states, 2))
    if bounds.shape != (4, 2):
        raise ValueError("CODAC root finder requires four (low, high) bounds")
    if not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("bounds must be finite with low < high")
    if not np.isfinite(a_min):
        raise ValueError("a_min must be finite")
    if origin_radius <= 0 or not np.isfinite(origin_radius):
        raise ValueError("origin_radius must be finite and positive")
    if paving_eps <= 0 or not np.isfinite(paving_eps):
        raise ValueError("paving_eps must be finite and positive")

    try:
        import codac
    except ImportError as exc:
        return CodacRootResult(
            0, 0, 0, empty, empty,
            f"unavailable: CODAC Python binding is not installed ({exc})",
        )

    try:
        function = _codac_function(a_expr, b_expr)
        domain = codac.IntervalVector(bounds.tolist())
        target = codac.IntervalVector(
            [
                [0.0, 0.0],
                [float(a_min), codac.oo],
                [float(origin_radius * origin_radius), codac.oo],
            ]
        )
        paving = codac.sivia(domain, function, target, float(paving_eps), verbose)
        inner = list(paving.boxes(paving.inner))
        boundary = list(paving.boxes(paving.bound))
        outer = list(paving.boxes(paving.outer))
        return CodacRootResult(
            n_inner_boxes=len(inner),
            n_boundary_boxes=len(boundary),
            n_outer_boxes=len(outer),
            inner_boxes=_boxes_to_numpy(inner, n_states),
            boundary_boxes=_boxes_to_numpy(boundary, n_states),
            status="ok",
        )
    except Exception as exc:
        return CodacRootResult(
            0, 0, 0, empty, empty, f"error: {type(exc).__name__}: {exc}"
        )
