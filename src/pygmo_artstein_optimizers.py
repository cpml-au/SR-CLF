"""Pygmo numerical falsification of the exact Artstein condition.

Pygmo maximizes ``a`` subject to the actual equality constraints ``b_i=0``
and the punctured-domain inequality. Its champion is then SLSQP-polished on
the same constraints. No ``a-rho*b**2`` or other barrier merit is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import secrets

import numpy as np
import pygmo as pg
from scipy.optimize import minimize
from sympy import lambdify, symbols


@dataclass(frozen=True)
class PygmoArtsteinResult:
    """Observed maxima from Pygmo seeds polished on ``b_i(x)=0``."""

    a_max: float
    n_restarts: int
    n_polished: int
    n_feasible: int
    candidate_points: np.ndarray
    feasible_points: np.ndarray
    violation_points: np.ndarray
    status: str


def _as_constraint_tuple(b_expr) -> tuple:
    if isinstance(b_expr, (list, tuple)):
        if not b_expr:
            raise ValueError("at least one Artstein b expression is required")
        return tuple(b_expr)
    return (b_expr,)


def _scalar_value(function, x) -> float:
    value = np.asarray(function(*np.asarray(x, dtype=float)), dtype=float)
    if value.size != 1:
        raise ValueError("Artstein expression must evaluate to a scalar")
    return float(value.reshape(-1)[0])


def _make_algorithm(name: str, generations: int, seed: int):
    name = str(name).lower()
    if name == "cmaes":
        return pg.algorithm(pg.cmaes(gen=int(generations), seed=int(seed)))
    if name == "de":
        return pg.algorithm(pg.de(gen=int(generations), seed=int(seed)))
    if name in {"de1220", "de_1220", "pde"}:
        return pg.algorithm(pg.de1220(gen=int(generations), seed=int(seed)))
    raise ValueError(f"Unsupported Pygmo Artstein optimizer: {name}")


class _ArtsteinConstraintProblem:
    """Constrained Pygmo UDP for ``max a`` with exact Artstein constraints."""

    def __init__(
        self,
        a_function,
        b_functions,
        bounds,
        origin_radius: float,
        invalid_value: float,
    ):
        self.a_function = a_function
        self.b_functions = b_functions
        self.bounds = np.asarray(bounds, dtype=float)
        self.origin_radius = float(origin_radius)
        self.invalid_value = float(invalid_value)

    def fitness(self, x):
        x = np.asarray(x, dtype=float)
        try:
            a_value = _scalar_value(self.a_function, x)
            b_values = np.asarray(
                [_scalar_value(function, x) for function in self.b_functions],
                dtype=float,
            )
            # Pygmo expects equality constraints after the objective, then
            # inequalities c(x) <= 0. This is max(a) expressed as min(-a).
            return [
                -a_value,
                *b_values.tolist(),
                self.origin_radius**2 - float(x @ x),
            ]
        except Exception:
            return [self.invalid_value] + [self.invalid_value] * (
                len(self.b_functions) + 1
            )

    def get_bounds(self):
        return (self.bounds[:, 0], self.bounds[:, 1])

    def get_nec(self):
        return len(self.b_functions)

    def get_nic(self):
        return 1


def verify_clf_pygmo_artstein(
    a_expr,
    b_expr,
    bounds,
    *,
    optimizer: str = "de",
    generations: int = 80,
    population_size: int = 100,
    restarts: int = 4,
    origin_radius: float = 1e-3,
    b_tolerance: float = 1e-10,
    polish_maxiter: int = 500,
    violation_tol: float = 0.0,
    random_seed: int | None = None,
    verbose: bool = False,
) -> PygmoArtsteinResult:
    """Search for ``max a(x)`` on ``b_i(x)=0`` in a bounded punctured box.

    Pygmo's constrained self-adaptive wrapper handles the equality manifold;
    SLSQP only polishes its champion on those same exact constraints. A
    positive returned ``a_max`` is a concrete numerical violation; a negative
    result is evidence only, not a global verification.
    """
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must have shape (n_states, 2)")
    if not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("bounds must be finite with lower < upper")
    if generations < 1 or population_size < 2 or restarts < 1:
        raise ValueError("generations, population_size, and restarts must be positive")
    if origin_radius <= 0 or b_tolerance < 0 or polish_maxiter < 1:
        raise ValueError("invalid Artstein radius, tolerance, or polish iteration count")

    n_states = bounds.shape[0]
    x_symbols = symbols(f"x1:{n_states + 1}")
    a_function = lambdify(x_symbols, a_expr, "numpy")
    b_functions = [
        lambdify(x_symbols, expression, "numpy")
        for expression in _as_constraint_tuple(b_expr)
    ]
    if random_seed is None:
        random_seed = secrets.randbelow(2**32)
    random_seed = int(random_seed) % (2**32)

    problem = _ArtsteinConstraintProblem(
        a_function,
        b_functions,
        bounds,
        origin_radius,
        invalid_value=1e30,
    )
    seeds = []
    try:
        for restart in range(int(restarts)):
            seed = (random_seed + restart) % (2**32)
            constrained_problem = pg.problem(problem)
            constrained_problem.c_tol = [float(b_tolerance)] * len(b_functions) + [0.0]
            population = pg.population(
                constrained_problem, size=int(population_size), seed=seed
            )
            inner = _make_algorithm(optimizer, 1, seed)
            algorithm = pg.algorithm(
                pg.cstrs_self_adaptive(
                    iters=int(generations), algo=inner, seed=seed
                )
            )
            champion = algorithm.evolve(population)
            seeds.append(np.asarray(champion.champion_x, dtype=float))
    except Exception as exc:
        empty = np.empty((0, n_states), dtype=float)
        return PygmoArtsteinResult(
            a_max=np.nan,
            n_restarts=int(restarts),
            n_polished=0,
            n_feasible=0,
            candidate_points=empty,
            feasible_points=empty,
            violation_points=empty,
            status=f"{optimizer}_failed: {type(exc).__name__}: {exc}",
        )

    box = [tuple(row) for row in bounds]
    origin_radius_sq = float(origin_radius) ** 2
    constraints = [
        {
            "type": "eq",
            "fun": lambda x, function=function: _scalar_value(function, x),
        }
        for function in b_functions
    ]
    constraints.append(
        {
            "type": "ineq",
            "fun": lambda x: float(np.asarray(x, dtype=float) @ np.asarray(x, dtype=float))
            - origin_radius_sq,
        }
    )

    feasible_points = []
    polished = 0
    for seed in seeds:
        try:
            result = minimize(
                lambda x: -_scalar_value(a_function, x),
                np.clip(seed, bounds[:, 0], bounds[:, 1]),
                method="SLSQP",
                bounds=box,
                constraints=constraints,
                options={"maxiter": int(polish_maxiter), "ftol": 1e-12},
            )
            polished += 1
            point = np.asarray(result.x, dtype=float)
            b_values = np.asarray(
                [_scalar_value(function, point) for function in b_functions],
                dtype=float,
            )
            if (
                np.all(np.isfinite(point))
                and np.isfinite(_scalar_value(a_function, point))
                and np.linalg.norm(point) >= origin_radius * (1.0 - 1e-6)
                and np.all(np.abs(b_values) <= b_tolerance)
            ):
                feasible_points.append(point)
        except Exception:
            continue

    candidates = np.asarray(seeds, dtype=float).reshape((-1, n_states))
    feasible = np.asarray(feasible_points, dtype=float)
    if feasible.size == 0:
        feasible = np.empty((0, n_states), dtype=float)
        violations = np.empty((0, n_states), dtype=float)
        a_max = np.nan
        status = "no equality-feasible Artstein point"
    else:
        a_values = np.asarray([_scalar_value(a_function, point) for point in feasible])
        a_max = float(np.max(a_values))
        violations = feasible[a_values > float(violation_tol)]
        status = "counterexample_found" if violations.size else "no_counterexample_found"

    result = PygmoArtsteinResult(
        a_max=a_max,
        n_restarts=int(restarts),
        n_polished=polished,
        n_feasible=len(feasible),
        candidate_points=candidates,
        feasible_points=feasible,
        violation_points=violations,
        status=status,
    )
    if verbose:
        print(
            "Pygmo Artstein max a "
            f"{result.a_max}; {result.status}; "
            f"feasible={result.n_feasible}/{result.n_restarts}"
        )
    return result
