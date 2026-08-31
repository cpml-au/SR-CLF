"""IBEX optimization of the Artstein constrained margin.

This verifier maximizes the exact symbolic ``a(x)`` on the bounded,
punctured near-manifold ``|b_i(x)| <= b_tolerance``.  A positive feasible
``a_max`` is an Artstein counterexample for the requested tolerance.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np

from src.ibex_counterexample_optimizers import (
    IbexBoxResult,
    _boxes,
    _minibex_expression,
    _run_box,
)


_NAMES = ("x1", "x2", "x3", "x4")


@dataclass(frozen=True)
class IbexArtsteinResult:
    """Aggregate result for ``max a(x)`` with ``b_i(x)`` near zero."""

    a_max: float
    a_upper_bound: float
    n_boxes: int
    n_complete: int
    timeout_seconds: float
    b_tolerance: float
    status: str
    boxes: tuple[IbexBoxResult, ...]


def _as_constraints(b_expr) -> tuple:
    """Accept the current scalar b expression or a sequence for multi-input."""
    if isinstance(b_expr, (list, tuple)):
        if not b_expr:
            raise ValueError("at least one Artstein b expression is required")
        return tuple(b_expr)
    return (b_expr,)


def _model_text(
    a_expr,
    b_expressions: tuple,
    box,
    origin_radius: float,
    b_tolerance: float,
) -> str:
    variables = "\n".join(
        f"  {name} in [{low:.17g}, {high:.17g}];"
        for name, (low, high) in zip(_NAMES, box)
    )
    functions = "\n\n".join(
        f"""function bfun_{index}(x1, x2, x3, x4)
  return {_minibex_expression(expression)};
end"""
        for index, expression in enumerate(b_expressions, start=1)
    )
    constraints = "\n\n".join(
        f"  -b_tol <= bfun_{index}(x1, x2, x3, x4);\n"
        f"   bfun_{index}(x1, x2, x3, x4) <= b_tol;"
        for index in range(1, len(b_expressions) + 1)
    )
    return f"""Constants
  eps = {origin_radius:.17g};
  b_tol = {b_tolerance:.17g};

Variables
{variables}

function afun(x1, x2, x3, x4)
  return {_minibex_expression(a_expr)};
end

{functions}

Minimize
  -afun(x1, x2, x3, x4);

Constraints
{constraints}

  x1^2 + x2^2 + x3^2 + x4^2 >= eps^2;
end
"""


def verify_clf_ibex_artstein(
    a_expr,
    b_expr,
    bounds,
    *,
    origin_radius: float = 1e-3,
    b_tolerance: float = 1e-10,
    timeout_seconds: float = 10.0,
    verbose: bool = False,
    work_dir: str | Path | None = None,
) -> IbexArtsteinResult:
    """Maximize ``a`` in 12 parallel IBEX boxes with ``b_i≈0``.

    ``a_max`` is ``max(-f_upper)``.  It is the largest feasible Artstein
    value observed by IBEX, so ``a_max > 0`` is a counterexample.  The global
    upper bound ``a_upper_bound=max(-f_lower)`` is conclusive only if every
    box completes.
    """
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (4, 2):
        raise ValueError("IBEX Artstein verifier requires four (low, high) bounds")
    if not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("bounds must be finite with low < high")
    if origin_radius <= 0 or not np.isfinite(origin_radius):
        raise ValueError("origin_radius must be finite and positive")
    if b_tolerance < 0 or not np.isfinite(b_tolerance):
        raise ValueError("b_tolerance must be finite and nonnegative")
    if timeout_seconds <= 0 or not np.isfinite(timeout_seconds):
        raise ValueError("timeout_seconds must be finite and positive")

    b_expressions = _as_constraints(b_expr)
    boxes = _boxes(bounds)

    def execute(directory: Path) -> tuple[IbexBoxResult, ...]:
        jobs = []
        for index, box in enumerate(boxes, start=1):
            model_path = directory / f"artstein_box_{index:02d}.mbx"
            model_path.write_text(
                _model_text(
                    a_expr, b_expressions, box, origin_radius, b_tolerance
                ),
                encoding="utf-8",
            )
            jobs.append(
                (index, str(model_path), str(directory / f"artstein_box_{index:02d}.cov"))
            )

        results = []
        with ProcessPoolExecutor(max_workers=len(jobs)) as executor:
            futures = {
                executor.submit(_run_box, index, model, cov, timeout_seconds): index
                for index, model, cov in jobs
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if verbose:
                    print(
                        f"IBEX Artstein box {result.index:02d}: "
                        f"{'complete' if result.complete else 'incomplete'} "
                        f"f=[{result.f_lower}, {result.f_upper}]"
                    )
        return tuple(sorted(results, key=lambda item: item.index))

    if work_dir is None:
        with tempfile.TemporaryDirectory(prefix="clf_ibex_artstein_") as directory:
            results = execute(Path(directory))
    else:
        directory = Path(work_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        results = execute(directory)

    feasible_values = [-item.f_upper for item in results if np.isfinite(item.f_upper)]
    upper_values = [-item.f_lower for item in results if np.isfinite(item.f_lower)]
    n_complete = sum(item.complete for item in results)
    if not feasible_values:
        status = "error: no IBEX objective bounds"
    elif n_complete == len(results):
        status = "complete"
    else:
        status = f"incomplete: {n_complete}/{len(results)} boxes completed"

    result = IbexArtsteinResult(
        a_max=float(max(feasible_values)) if feasible_values else np.nan,
        a_upper_bound=float(max(upper_values)) if upper_values else np.nan,
        n_boxes=len(results),
        n_complete=n_complete,
        timeout_seconds=float(timeout_seconds),
        b_tolerance=float(b_tolerance),
        status=status,
        boxes=results,
    )
    if verbose:
        print(
            "IBEX Artstein max a "
            f"(feasible lower bound): {result.a_max}; {result.status}"
        )
    return result
