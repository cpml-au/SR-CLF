"""IBEX-based falsification of the scalar CLF barrier.

The verifier maximizes a(x) - rho*b(x)**2 over a bounded, punctured state
box. The scalar input coefficient b makes b**2 equal to |b|**2. IBEX
minimizes its negative on 12 disjoint boxes in parallel.

The reported margin_max is the largest feasible barrier value found from
IBEX's incumbent bounds. A positive value is therefore a counterexample,
even if a time limit prevents a global certificate.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np


_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_OBJECTIVE_PATTERN = re.compile(
    rf"f\*\s+in\s*\[\s*({_FLOAT})\s*,\s*({_FLOAT})\s*\]"
)
_ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")
_NAMES = ("x1", "x2", "x3", "x4")
_SPLITS = (3, 2, 2, 1)


@dataclass(frozen=True)
class IbexBoxResult:
    """Result from one disjoint IBEX domain box."""

    index: int
    complete: bool
    timed_out: bool
    f_lower: float
    f_upper: float
    returncode: int
    detail: str


@dataclass(frozen=True)
class IbexBarrierResult:
    """Aggregate result for max(a - rho*b**2)."""

    margin_max: float
    margin_upper_bound: float
    n_boxes: int
    n_complete: int
    timeout_seconds: float
    status: str
    boxes: tuple[IbexBoxResult, ...]


def _minibex_expression(expr) -> str:
    """Convert the SymPy printer's exponent syntax to Minibex syntax."""
    return str(expr).replace("**", "^")


def _partitions(low: float, high: float, count: int):
    width = (high - low) / count
    return [(low + i * width, low + (i + 1) * width) for i in range(count)]


def _boxes(bounds: np.ndarray):
    pieces = [
        _partitions(float(low), float(high), count)
        for (low, high), count in zip(bounds, _SPLITS)
    ]
    return [
        (b1, b2, b3, b4)
        for b1 in pieces[0]
        for b2 in pieces[1]
        for b3 in pieces[2]
        for b4 in pieces[3]
    ]


def _model_text(a_expr, b_expr, rho: float, box, origin_radius: float) -> str:
    variables = "\n".join(
        f"  {name} in [{low:.17g}, {high:.17g}];"
        for name, (low, high) in zip(_NAMES, box)
    )
    return f"""Constants
  eps = {origin_radius:.17g};
  rho = {rho:.17g};

Variables
{variables}

function bfun(x1, x2, x3, x4)
  return {_minibex_expression(b_expr)};
end

function afun(x1, x2, x3, x4)
  return {_minibex_expression(a_expr)};
end

function barrierfun(x1, x2, x3, x4)
  // b is scalar, so b^2 = |b|^2.
  return afun(x1, x2, x3, x4) - rho*bfun(x1, x2, x3, x4)^2;
end

Minimize
  -barrierfun(x1, x2, x3, x4);

Constraints
  x1^2 + x2^2 + x3^2 + x4^2 >= eps^2;
end
"""


def _run_box(index: int, model_path: str, cov_path: str, timeout_seconds: float):
    command = [
        "ibexopt",
        model_path,
        "--rigor",
        "--kkt",
        "--abs-eps-f=1e-10",
        "--rel-eps-f=1e-10",
        "--simpl=2",
        "--output",
        cov_path,
        f"--timeout={timeout_seconds:.17g}",
    ]
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        output = _ANSI_PATTERN.sub("", completed.stdout)
        match = _OBJECTIVE_PATTERN.search(output)
        timed_out = "time limit" in output.lower()
        f_lower = float(match.group(1)) if match else np.nan
        f_upper = float(match.group(2)) if match else np.nan
        complete = completed.returncode == 0 and match is not None and not timed_out
        detail = output.splitlines()[-1] if output.splitlines() else "no IBEX output"
        return IbexBoxResult(
            index=index,
            complete=complete,
            timed_out=timed_out,
            f_lower=f_lower,
            f_upper=f_upper,
            returncode=int(completed.returncode),
            detail=detail,
        )
    except OSError as exc:
        return IbexBoxResult(
            index=index,
            complete=False,
            timed_out=False,
            f_lower=np.nan,
            f_upper=np.nan,
            returncode=-1,
            detail=f"{type(exc).__name__}: {exc}",
        )


def verify_clf_ibex_barrier(
    a_expr,
    b_expr,
    bounds,
    rho: float,
    *,
    origin_radius: float = 1e-5,
    timeout_seconds: float = 10.0,
    verbose: bool = False,
    work_dir: str | Path | None = None,
) -> IbexBarrierResult:
    """Falsify a - rho*b**2 <= 0 with 12 parallel IBEX subproblems.

    margin_max is max(-f_upper). Each f_upper is IBEX's feasible incumbent
    for f = -(a-rho*b**2), so a positive margin_max is an observed violation.
    margin_upper_bound is max(-f_lower); it is globally meaningful only when
    every box completes.
    """
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (4, 2):
        raise ValueError("IBEX barrier verifier requires four (low, high) bounds")
    if rho < 0 or not np.isfinite(rho):
        raise ValueError("rho must be finite and nonnegative")
    if origin_radius <= 0 or not np.isfinite(origin_radius):
        raise ValueError("origin_radius must be finite and positive")
    if timeout_seconds <= 0 or not np.isfinite(timeout_seconds):
        raise ValueError("timeout_seconds must be finite and positive")

    boxes = _boxes(bounds)

    def execute(directory: Path):
        jobs = []
        for index, box in enumerate(boxes, start=1):
            model_path = directory / f"box_{index:02d}.mbx"
            model_path.write_text(
                _model_text(a_expr, b_expr, rho, box, origin_radius),
                encoding="utf-8",
            )
            jobs.append((index, str(model_path), str(directory / f"box_{index:02d}.cov")))

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
                        f"IBEX box {result.index:02d}: "
                        f"{'complete' if result.complete else 'incomplete'} "
                        f"f=[{result.f_lower}, {result.f_upper}]"
                    )
        return tuple(sorted(results, key=lambda item: item.index))

    if work_dir is None:
        with tempfile.TemporaryDirectory(prefix="clf_ibex_") as directory:
            results = execute(Path(directory))
    else:
        directory = Path(work_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        results = execute(directory)

    feasible_margins = [-item.f_upper for item in results if np.isfinite(item.f_upper)]
    upper_margins = [-item.f_lower for item in results if np.isfinite(item.f_lower)]
    n_complete = sum(item.complete for item in results)
    if not feasible_margins:
        status = "error: no IBEX objective bounds"
    elif n_complete == len(results):
        status = "complete"
    else:
        status = f"incomplete: {n_complete}/{len(results)} boxes completed"

    result = IbexBarrierResult(
        margin_max=float(max(feasible_margins)) if feasible_margins else np.nan,
        margin_upper_bound=float(max(upper_margins)) if upper_margins else np.nan,
        n_boxes=len(results),
        n_complete=n_complete,
        timeout_seconds=float(timeout_seconds),
        status=status,
        boxes=results,
    )
    if verbose:
        print(
            "IBEX max barrier margin "
            f"(feasible lower bound): {result.margin_max}; {result.status}"
        )
    return result
