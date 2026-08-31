"""Pygmo Artstein falsifier on the raw numpy callable (complex-step a, b).

Same architecture as ``src/pygmo_artstein_optimizers.py`` (constrained UDP
max a s.t. b=0 outside the origin ball, ``cstrs_self_adaptive`` wrapper,
SLSQP polish of each champion on the exact constraints), but a = grad(V).f
and b = grad(V).g are computed by complex-step differentiation of the
candidate's numpy callable -- no sympy, no ``ind2MSE``. Gradient engine and
the non-analytic-primitive guard are shared with
``src.b_manifold_check_callable`` (status ``cs_incompatible`` on refusal).

``f`` and ``G_col`` follow the ``check_b_manifold_callable`` convention:
vectorized numeric dynamics, f(x1..xn) -> (n, M) and active input column
G_col(x1..xn) -> (n, M).
"""

from __future__ import annotations

import secrets

import numpy as np
import pygmo as pg
from scipy.optimize import minimize

from src.b_manifold_check_callable import _cs_selfcheck, cs_grad_batch
from src.pygmo_artstein_optimizers import PygmoArtsteinResult, _make_algorithm


def _make_ab(individual, consts, f, G_col, n):
    """Point evaluators a(x), b(x) via complex-step on the callable."""

    def ab_point(x):
        X = np.asarray(x, dtype=float).reshape(1, n)
        G = cs_grad_batch(individual, consts, X)
        fv = np.asarray(f(*X.T), dtype=float).reshape(n, 1).T
        gv = np.asarray(G_col(*X.T), dtype=float).reshape(n, 1).T
        return (
            float(np.sum(G * fv, axis=1)[0]),
            float(np.sum(G * gv, axis=1)[0]),
        )

    return ab_point


class _ArtsteinCSProblem:
    """Constrained Pygmo UDP for ``max a`` on ``b=0`` via complex-step."""

    def __init__(self, ab_point, bounds, origin_radius, invalid_value):
        self.ab_point = ab_point
        self.bounds = np.asarray(bounds, dtype=float)
        self.origin_radius = float(origin_radius)
        self.invalid_value = float(invalid_value)

    def fitness(self, x):
        x = np.asarray(x, dtype=float)
        try:
            a_value, b_value = self.ab_point(x)
            if not (np.isfinite(a_value) and np.isfinite(b_value)):
                raise ValueError("non-finite a or b")
            return [
                -a_value,
                b_value,
                self.origin_radius**2 - float(x @ x),
            ]
        except Exception:
            return [self.invalid_value] * 3

    def get_bounds(self):
        return (self.bounds[:, 0], self.bounds[:, 1])

    def get_nec(self):
        return 1

    def get_nic(self):
        return 1


def verify_clf_pygmo_artstein_cs(
    individual,
    consts,
    f,
    G_col,
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
    cs_selfcheck_tol: float = 1e-5,
    verbose: bool = False,
) -> PygmoArtsteinResult:
    """Search ``max a(x)`` on ``b(x)=0`` in the punctured box, sympy-free.

    A positive returned ``a_max`` is a concrete numerical violation; a
    negative result is evidence only, not a global verification. Returns
    status ``cs_incompatible`` (a_max NaN) if the callable fails the
    complex-step-vs-finite-difference gradient probe.
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
    empty = np.empty((0, n_states), dtype=float)
    if not _cs_selfcheck(individual, consts, bounds, cs_selfcheck_tol):
        return PygmoArtsteinResult(
            a_max=np.nan,
            n_restarts=int(restarts),
            n_polished=0,
            n_feasible=0,
            candidate_points=empty,
            feasible_points=empty,
            violation_points=empty,
            status="cs_incompatible",
        )

    ab_point = _make_ab(individual, consts, f, G_col, n_states)
    if random_seed is None:
        random_seed = secrets.randbelow(2**32)
    random_seed = int(random_seed) % (2**32)

    problem = _ArtsteinCSProblem(ab_point, bounds, origin_radius, invalid_value=1e30)
    seeds = []
    try:
        for restart in range(int(restarts)):
            seed = (random_seed + restart) % (2**32)
            constrained_problem = pg.problem(problem)
            constrained_problem.c_tol = [float(b_tolerance), 0.0]
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
        {"type": "eq", "fun": lambda x: ab_point(x)[1]},
        {
            "type": "ineq",
            "fun": lambda x: float(np.asarray(x, dtype=float) @ np.asarray(x, dtype=float))
            - origin_radius_sq,
        },
    ]

    feasible_points = []
    polished = 0
    for seed in seeds:
        try:
            result = minimize(
                lambda x: -ab_point(x)[0],
                np.clip(seed, bounds[:, 0], bounds[:, 1]),
                method="SLSQP",
                bounds=box,
                constraints=constraints,
                options={"maxiter": int(polish_maxiter), "ftol": 1e-12},
            )
            polished += 1
            point = np.asarray(result.x, dtype=float)
            a_value, b_value = ab_point(point)
            if (
                np.all(np.isfinite(point))
                and np.isfinite(a_value)
                and np.linalg.norm(point) >= origin_radius * (1.0 - 1e-6)
                and abs(b_value) <= b_tolerance
            ):
                feasible_points.append(point)
        except Exception:
            continue

    candidates = np.asarray(seeds, dtype=float).reshape((-1, n_states))
    feasible = np.asarray(feasible_points, dtype=float)
    if feasible.size == 0:
        feasible = empty
        violations = empty
        a_max = np.nan
        status = "no equality-feasible Artstein point"
    else:
        a_values = np.asarray([ab_point(point)[0] for point in feasible])
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
            "Pygmo Artstein (CS) max a "
            f"{result.a_max}; {result.status}; "
            f"feasible={result.n_feasible}/{result.n_restarts}"
        )
    return result
