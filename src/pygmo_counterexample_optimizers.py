from dataclasses import dataclass

import numpy as np
import pygmo as pg
import secrets

from src.numerical_point_verification import evaluate_clf_point, evaluate_v


@dataclass
class PygmoVerificationResult:
    valid: bool
    penalty: float
    v_min: float
    decay_margin_max: float
    counterexamples: np.ndarray
    v_counterexamples: np.ndarray
    vdot_counterexamples: np.ndarray
    ignored_vdot_counterexamples: np.ndarray
    status: str


class _CLFOptimizationProblem:
    def __init__(
        self,
        objective,
        individual,
        consts,
        f,
        G,
        Q,
        R,
        bounds,
        decay_rate,
        fd_step,
        invalid_value,
        rho=1.0,
        gamma1=0.0,
        sublevel_cap=None,
        pd_eps=0.0,
        origin_exclusion_radius=None,
    ):
        self.objective = objective
        self.individual = individual
        self.consts = consts
        self.f = f
        self.G = G
        self.Q = Q
        self.R = R
        self.bounds = np.asarray(bounds, dtype=float)
        self.decay_rate = decay_rate
        self.fd_step = fd_step
        self.invalid_value = invalid_value
        self.rho = rho
        self.gamma1 = gamma1
        self.sublevel_cap = sublevel_cap
        self.pd_eps = pd_eps
        self.origin_exclusion_radius = origin_exclusion_radius
        self.origin_value = evaluate_v(individual, np.zeros(len(bounds)), consts)

    def fitness(self, x):
        x = np.asarray(x, dtype=float)
        try:
            if self.objective == "V":
                # positive-definiteness with margin: V - V(0) must exceed
                # pd_eps*||x||^2, so semidefinite valleys (V == V(0) on a
                # manifold) give a strictly negative minimum DE can find
                value = (
                    evaluate_v(self.individual, x, self.consts)
                    - self.origin_value
                    - self.pd_eps * float(x @ x)
                )
            elif self.objective == "negative_barrier_margin":
                if (
                    self.origin_exclusion_radius is not None
                    and np.linalg.norm(x) <= self.origin_exclusion_radius
                ):
                    return [float(self.invalid_value)]
                values = evaluate_clf_point(
                    self.individual,
                    x,
                    self.consts,
                    self.f,
                    self.G,
                    self.Q,
                    self.R,
                    bounds=self.bounds,
                    decay_rate=self.decay_rate,
                    fd_step=self.fd_step,
                    rho=self.rho,
                    gamma1=self.gamma1,
                )
                value = -values["barrier_margin"]
                # restrict the barrier search to the certified sublevel set;
                # the sloped penalty guides the optimizer into the region
                if self.sublevel_cap is not None:
                    v_rel = (
                        evaluate_v(self.individual, x, self.consts)
                        - self.origin_value
                    )
                    if not np.isfinite(v_rel):
                        value = self.invalid_value
                    elif v_rel > self.sublevel_cap:
                        value += 1e3 + (v_rel - self.sublevel_cap)
            else:
                raise ValueError(f"Unknown objective: {self.objective}")
        except Exception:
            value = self.invalid_value

        if not np.isfinite(value):
            value = self.invalid_value
        return [float(value)]

    def get_bounds(self):
        return (self.bounds[:, 0], self.bounds[:, 1])


def _make_algorithm(name, generations, seed):
    name = name.lower()
    if name == "cmaes":
        return pg.algorithm(pg.cmaes(gen=generations, seed=seed))
    if name == "de":
        return pg.algorithm(pg.de(gen=generations, seed=seed))
    if name in {"de1220", "de_1220", "pde"}:
        return pg.algorithm(pg.de1220(gen=generations, seed=seed))
    raise ValueError(f"Unsupported pygmo optimizer: {name}")


def _optimize(problem, optimizer, generations, population_size, seed):
    algo = _make_algorithm(optimizer, generations, seed)
    pop = pg.population(pg.problem(problem), size=population_size, seed=seed)
    pop = algo.evolve(pop)
    return np.asarray(pop.champion_x, dtype=float), float(pop.champion_f[0])


def _localized_points(root, low_bound, up_bound, rng, n_samples, radius):
    if n_samples <= 0:
        return root.reshape(1, -1)

    uniform_offsets = rng.uniform(0.0, radius, size=(n_samples, len(root)))
    gaussian_offsets = rng.normal(0.0, radius / 15.0, size=(n_samples, len(root)))
    offsets = np.vstack((uniform_offsets, gaussian_offsets))
    minus = np.clip(root - offsets, low_bound, up_bound)
    plus = np.clip(root + offsets, low_bound, up_bound)
    return np.vstack((minus, plus, root.reshape(1, -1)))


def verify_clf_pygmo(
    individual,
    consts,
    f,
    G,
    Q,
    R,
    bounds,
    optimizer="de",
    decay_rate=0.001,
    generations=80,
    population_size=60,
    local_samples=300,
    local_radius=0.15,
    fd_step=1e-5,
    rho=1.0,
    gamma1=0.0,
    sublevel_cap=None,
    positive_tol=1e-10,
    pd_eps=1e-4,
    decay_tol=0.0,
    origin_tol=1e-9,
    counterexample_penalty=1.0,
    root_penalty=1000.0,
    invalid_penalty=1e6,
    ignore_vdot_ball_radius=None,
    ignore_vdot_abs_tol=None,
    random_seed=0,
):
    bounds = np.asarray(bounds, dtype=float)
    n_states = bounds.shape[0]

    if random_seed is None:
        random_seed = secrets.randbelow(2**32)
    random_seed = int(random_seed) % (2**32)

    rng = np.random.default_rng(random_seed)
    # rng = np.random.default_rng(random_seed)

    try:
        v_problem = _CLFOptimizationProblem(
            "V",
            individual,
            consts,
            f,
            G,
            Q,
            R,
            bounds,
            decay_rate,
            fd_step,
            invalid_penalty,
            rho=rho,
            gamma1=gamma1,
            pd_eps=pd_eps,
        )
        margin_problem = _CLFOptimizationProblem(
            "negative_barrier_margin",
            individual,
            consts,
            f,
            G,
            Q,
            R,
            bounds,
            decay_rate,
            fd_step,
            invalid_penalty,
            rho=rho,
            gamma1=gamma1,
            sublevel_cap=sublevel_cap,
            origin_exclusion_radius=ignore_vdot_ball_radius,
        )

        v_root, v_min = _optimize(
            v_problem, optimizer, generations, population_size, random_seed
        )
        margin_root, negative_margin_min = _optimize(
            margin_problem,
            optimizer,
            generations,
            population_size,
            (random_seed + 1) % (2**32),
        )
    except Exception as exc:
        return PygmoVerificationResult(
            valid=False,
            penalty=invalid_penalty,
            v_min=np.nan,
            decay_margin_max=np.nan,
            counterexamples=np.empty((0, n_states)),
            v_counterexamples=np.empty((0, n_states)),
            vdot_counterexamples=np.empty((0, n_states)),
            ignored_vdot_counterexamples=np.empty((0, n_states)),
            status=f"{optimizer}_failed: {exc}",
        )

    decay_margin_max = -negative_margin_min
    low_bound = bounds[:, 0]
    up_bound = bounds[:, 1]
    candidate_points = np.vstack(
        (
            _localized_points(
                v_root, low_bound, up_bound, rng, local_samples, local_radius
            ),
            _localized_points(
                margin_root, low_bound, up_bound, rng, local_samples, local_radius
            ),
        )
    )

    origin_value = evaluate_v(individual, np.zeros(n_states), consts)
    v_counterexamples = []
    vdot_counterexamples = []
    ignored_vdot_counterexamples = []
    point_eval_failures = 0
    first_point_error = None

    for point in candidate_points:
        if np.linalg.norm(point) <= origin_tol:
            continue

        try:
            v_value = evaluate_v(individual, point, consts) - origin_value
            values = evaluate_clf_point(
                individual,
                point,
                consts,
                f,
                G,
                Q,
                R,
                bounds=bounds,
                decay_rate=decay_rate,
                fd_step=fd_step,
                rho=rho,
                gamma1=gamma1,
            )
            barrier_value = values["barrier"]
            margin_value = values["barrier_margin"] #values["a"] #values["barrier_margin"]
        except Exception as exc:
            point_eval_failures += 1
            if first_point_error is None:
                first_point_error = f"{type(exc).__name__}: {exc}"
            continue

        # relative positivity: V - V(0) must grow at least like pd_eps*||x||^2
        # (positive_tol kept as the old absolute floor)
        if np.isfinite(v_value) and v_value < max(
            positive_tol, pd_eps * float(point @ point)
        ):
            v_counterexamples.append(point.copy())

        inside_sublevel = (
            sublevel_cap is None
            or (np.isfinite(v_value) and v_value < sublevel_cap)
        )
        if np.isfinite(margin_value) and margin_value >= -decay_tol*1 and inside_sublevel:
            should_ignore_vdot = (
                ignore_vdot_ball_radius is not None
                and ignore_vdot_abs_tol is not None
                and np.linalg.norm(point) <= ignore_vdot_ball_radius
                and abs(barrier_value) < ignore_vdot_abs_tol
            )
            if should_ignore_vdot:
                ignored_vdot_counterexamples.append(point.copy())
            else:
                vdot_counterexamples.append(point.copy())

    v_counterexamples = np.asarray(v_counterexamples, dtype=float)
    vdot_counterexamples = np.asarray(vdot_counterexamples, dtype=float)
    ignored_vdot_counterexamples = np.asarray(ignored_vdot_counterexamples, dtype=float)

    if v_counterexamples.size == 0:
        v_counterexamples = np.empty((0, n_states))
    if vdot_counterexamples.size == 0:
        vdot_counterexamples = np.empty((0, n_states))
    if ignored_vdot_counterexamples.size == 0:
        ignored_vdot_counterexamples = np.empty((0, n_states))

    counterexamples = np.vstack((v_counterexamples, vdot_counterexamples))
    if counterexamples.size == 0:
        counterexamples = np.empty((0, n_states))

    # penalty = counterexample_penalty * len(counterexamples)
    # if v_min < -positive_tol:
    #     penalty += root_penalty
    # if np.linalg.norm(margin_root) > origin_tol and decay_margin_max >= decay_tol:
    #     penalty += root_penalty
    penalty = counterexample_penalty * len(counterexamples)

    if v_counterexamples.size > 0:
        penalty += root_penalty

    if vdot_counterexamples.size > 0:
        penalty += root_penalty

    valid = penalty == 0
    status = "valid" if valid else "counterexample_found"
    if point_eval_failures:
        print(
            f"verify_clf_pygmo: {point_eval_failures} candidate-point "
            f"evaluations failed (first: {first_point_error})"
        )
        status = f"{status}; point_eval_failures={point_eval_failures}"

    return PygmoVerificationResult(
        valid=valid,
        penalty=float(penalty),
        v_min=float(v_min),
        decay_margin_max=float(decay_margin_max),
        counterexamples=counterexamples,
        v_counterexamples=v_counterexamples,
        vdot_counterexamples=vdot_counterexamples,
        ignored_vdot_counterexamples=ignored_vdot_counterexamples,
        status=status,
    )
