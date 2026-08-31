from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, shgo
from sympy import Matrix, lambdify, symbols

from src.SymVVdot_Calculations import compute_v_and_v_dotSR


@dataclass
class SHGOVerificationResult:
    valid: bool
    penalty: float
    v_min: float
    decay_margin_max: float
    counterexamples: np.ndarray
    v_counterexamples: np.ndarray
    vdot_counterexamples: np.ndarray
    ignored_vdot_counterexamples: np.ndarray
    status: str


def _as_scalar(value):
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return np.nan
    return float(arr.reshape(-1)[0])


def _safe_objective(func, invalid_value):
    def objective(x):
        try:
            value = _as_scalar(func(np.asarray(x, dtype=float)))
        except Exception:
            return invalid_value
        if not np.isfinite(value):
            return invalid_value
        return value

    return objective


def _run_shgo(func, bounds, n, iters, sampling_method):
    result = shgo(
        func,
        Bounds(bounds[:, 0], bounds[:, 1]),
        n=n,
        iters=iters,
        sampling_method=sampling_method,
    )
    if result.x is None or not np.isfinite(result.fun):
        raise RuntimeError(result.message)
    return np.asarray(result.x, dtype=float), float(result.fun)


def _localized_points(root, low_bound, up_bound, rng, n_samples, radius):
    if n_samples <= 0:
        return np.empty((0, len(root)))

    uniform_offsets = rng.uniform(0.0, radius, size=(n_samples, len(root)))
    gaussian_offsets = rng.normal(0.0, radius / 15.0, size=(n_samples, len(root)))
    offsets = np.vstack((uniform_offsets, gaussian_offsets))

    minus = np.clip(root - offsets, low_bound, up_bound)
    plus = np.clip(root + offsets, low_bound, up_bound)
    return np.vstack((minus, plus, root.reshape(1, -1)))


def verify_clf_shgo(
    expression,
    fSR,
    GSR,
    QSR,
    RSR,
    bounds,
    decay_rate=0.001,
    rho=1.0,
    gamma1=0.0,
    sublevel_cap=None,
    shgo_n=2048,
    shgo_iters=3,
    sampling_method="simplicial",
    local_samples=800,
    local_radius=0.001,
    positive_tol=1e-10,
    pd_eps=1e-4,
    decay_tol=0.0,
    origin_tol=1e-9,
    control_zero_tol=1e-6,
    counterexample_penalty=1.0,
    root_penalty=1000.0,
    invalid_penalty=1e6,
    ignore_vdot_ball_radius=None,
    ignore_vdot_abs_tol=None,
    random_seed=None,
):
    """
    SHGO-based numerical falsification for the CLF conditions used by symclf.

    The verifier searches for violations of:
        V(x) - V(0) > pd_eps*||x||^2
        a(x) - rho*|b(x)| + gamma1 < 0

    with a = grad(V).f and scalar b = grad(V).g for the active input column.
    This matches the DE verifier exactly.  The state-dependent
    decay_rate*||x||^2 term belongs to the separate Sontag-Vdot performance
    check and is deliberately not part of this Artstein/bounded-input check.
    Result fields keep their historical names (decay_margin_max,
    vdot_counterexamples) but refer to the barrier margin.
    """
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must have shape (n_states, 2)")

    n_states = bounds.shape[0]
    x_syms = symbols(f"x1:{n_states + 1}")
    if len(expression.free_symbols.intersection(set(x_syms))) < n_states:
        return SHGOVerificationResult(
            valid=False,
            penalty=invalid_penalty,
            v_min=np.nan,
            decay_margin_max=np.nan,
            counterexamples=np.empty((0, n_states)),
            v_counterexamples=np.empty((0, n_states)),
            vdot_counterexamples=np.empty((0, n_states)),
            ignored_vdot_counterexamples=np.empty((0, n_states)),
            status="missing_state_symbol",
        )

    origin_subs = {x_syms[i]: 0 for i in range(n_states)}
    expression = expression - expression.subs(origin_subs)

    (
        _V_expr,
        _Vdot_expr,
        _lambda_expr,
        _u_expr,
        _u2_expr,
        _u_mag_expr,
        control_terms,
        b_expr,
        a_expr,
    ) = compute_v_and_v_dotSR(expression, fSR, GSR, QSR, RSR, x_syms, None)

    v_func_raw = lambdify(x_syms, expression, "numpy")
    a_func_raw = lambdify(x_syms, a_expr, "numpy")
    b_func_raw = lambdify(x_syms, b_expr, "numpy")

    def v_value(x):
        return _as_scalar(v_func_raw(*x)) - pd_eps * float(np.dot(x, x))

    def barrier_and_margin(x):
        a_value = _as_scalar(a_func_raw(*x))
        b_value = _as_scalar(b_func_raw(*x))
        if not np.all(np.isfinite([a_value, b_value])):
            return np.nan, np.nan

        barrier_value = a_value - rho * abs(b_value)
        # margin = barrier_value + decay_rate * ||x||^2  # intentionally off
        return barrier_value, barrier_value + gamma1

    def barrier_margin(x):
        return barrier_and_margin(x)[1]

    def capped_negative_margin(x):
        if (
            ignore_vdot_ball_radius is not None
            and np.linalg.norm(x) <= ignore_vdot_ball_radius
        ):
            return invalid_penalty
        value = -barrier_margin(x)
        # restrict the barrier search to the certified sublevel set;
        # the sloped penalty guides the optimizer into the region
        if sublevel_cap is not None:
            v_rel = _as_scalar(v_func_raw(*x))
            if not np.isfinite(v_rel):
                return np.nan
            if v_rel > sublevel_cap:
                value += 1e3 + (v_rel - sublevel_cap)
        return value

    v_objective = _safe_objective(v_value, invalid_penalty)
    negative_margin_objective = _safe_objective(
        capped_negative_margin, invalid_penalty
    )

    rng = np.random.default_rng(random_seed)
    low_bound = bounds[:, 0]
    up_bound = bounds[:, 1]

    try:
        v_root, v_min = _run_shgo(
            v_objective, bounds, shgo_n, shgo_iters, sampling_method
        )
        margin_root, negative_margin_min = _run_shgo(
            negative_margin_objective, bounds, shgo_n, shgo_iters, sampling_method
        )
    except Exception as exc:
        return SHGOVerificationResult(
            valid=False,
            penalty=invalid_penalty,
            v_min=np.nan,
            decay_margin_max=np.nan,
            counterexamples=np.empty((0, n_states)),
            v_counterexamples=np.empty((0, n_states)),
            vdot_counterexamples=np.empty((0, n_states)),
            ignored_vdot_counterexamples=np.empty((0, n_states)),
            status=f"shgo_failed: {exc}",
        )

    decay_margin_max = -negative_margin_min
    v_counterexamples = []
    vdot_counterexamples = []
    ignored_vdot_counterexamples = []

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

    for point in candidate_points:
        if np.linalg.norm(point) <= origin_tol:
            continue

        v_rel_at_point = _as_scalar(v_func_raw(*point))
        barrier_at_point, margin_at_point = barrier_and_margin(point)
        if v_rel_at_point < max(positive_tol, pd_eps * float(point @ point)):
            v_counterexamples.append(point.copy())
        inside_sublevel = sublevel_cap is None or v_rel_at_point < sublevel_cap
        if margin_at_point >= decay_tol and inside_sublevel:
            should_ignore_vdot = (
                ignore_vdot_ball_radius is not None
                and ignore_vdot_abs_tol is not None
                and np.linalg.norm(point) <= ignore_vdot_ball_radius
                and abs(barrier_at_point) < ignore_vdot_abs_tol
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

    penalty = counterexample_penalty * len(counterexamples)
    if v_min < -positive_tol:
        penalty += root_penalty
    if np.linalg.norm(margin_root) > origin_tol and decay_margin_max >= decay_tol:
        penalty += root_penalty

    valid = penalty == 0
    return SHGOVerificationResult(
        valid=valid,
        penalty=float(penalty),
        v_min=float(v_min),
        decay_margin_max=float(decay_margin_max),
        counterexamples=counterexamples,
        v_counterexamples=v_counterexamples,
        vdot_counterexamples=vdot_counterexamples,
        ignored_vdot_counterexamples=ignored_vdot_counterexamples,
        status="valid" if valid else "counterexample_found",
    )
