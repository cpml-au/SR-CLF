import numpy as np


def as_scalar(value):
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return np.nan
    return float(arr.reshape(-1)[0])


def evaluate_v(individual, x, consts):
    x = np.asarray(x, dtype=float)
    return as_scalar(individual(*x, consts))


def finite_difference_grad(individual, x, consts, bounds=None, step=1e-5):
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x, dtype=float)

    for i in range(len(x)):
        h = step * max(1.0, abs(x[i]))
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h

        if bounds is not None:
            low = bounds[i][0]
            high = bounds[i][1]
            if x_plus[i] > high:
                x_plus[i] = x[i]
                x_minus[i] = max(low, x[i] - h)
            elif x_minus[i] < low:
                x_minus[i] = x[i]
                x_plus[i] = min(high, x[i] + h)

        denom = x_plus[i] - x_minus[i]
        if abs(denom) < np.finfo(float).eps:
            grad[i] = np.nan
        else:
            grad[i] = (
                evaluate_v(individual, x_plus, consts)
                - evaluate_v(individual, x_minus, consts)
            ) / denom

    return grad


def evaluate_clf_point(
    individual,
    x,
    consts,
    f,
    G,
    Q,
    R,
    bounds=None,
    decay_rate=0.01,
    fd_step=1e-5,
    rho=1.0,                   # bounded-input authority in a - rho*|b|
    gamma1=0.0,                # constant strictness margin outside origin ball
    # control_zero_tol=1e-6,     # kept only for backward compatibility
):
    x = np.asarray(x, dtype=float)
    V_value = evaluate_v(individual, x, consts)
    # V_value_PD = V_value - decay_rate * float(x @ x)
    grad = finite_difference_grad(individual, x, consts, bounds=bounds, step=fd_step)

    try:
        f_vector = np.asarray(f(*x), dtype=float).reshape(-1)
        G_matrix = np.asarray(G(*x), dtype=float)
        Q_matrix = np.asarray(Q(*x), dtype=float)
        R_matrix = np.asarray(R(*x), dtype=float)
        R_inv = np.linalg.inv(R_matrix)

        a_value = float(np.dot(grad, f_vector))
        b_vector = np.asarray(grad @ G_matrix, dtype=float).reshape(-1)
        b_norm_squared = float(b_vector @ R_inv @ b_vector)
        b_l1 = float(np.sum(np.abs(b_vector)))
        b_l11 = float(np.sum((b_vector)))
        x_norm_squared = float(x @ Q_matrix @ x)

        if not np.all(
            np.isfinite([V_value, *grad, a_value, b_l1, b_norm_squared, x_norm_squared])
        ):
            Vdot_value = np.nan
            margin = np.nan
            barrier_value = np.nan
            barrier_margin = np.nan
        else:
            # Sontag/min-norm Vdot (unbounded u): decay-rate check only.
            # It is <= 0 wherever b != 0 for ANY V, so it carries no CLF
            # validity information; validity lives in the barrier below.
            radicand = a_value * a_value + x_norm_squared * b_norm_squared
            Vdot_value = -np.sqrt(max(radicand, 0.0))
            margin = Vdot_value #+ decay_rate * x_norm_squared

            # Exact scalar bounded-input CLF condition on the punctured box:
            #   a(x) - rho*|b(x)| + gamma1 < 0.
            # At b = 0 this reduces to the strict Artstein check
            # a(x) + gamma1 < 0.
            barrier_value = a_value - rho * b_l1
            # barrier_margin = barrier_value + decay_rate * x_norm_squared
            barrier_margin = barrier_value + gamma1

    except Exception:
        a_value = np.nan
        b_norm_squared = np.nan
        b_l1 = np.nan
        x_norm_squared = np.nan
        Vdot_value = np.nan
        margin = np.nan
        barrier_value = np.nan
        barrier_margin = np.nan

    return {
        "x": x,
        "V": V_value,
        # "V_PD": V_value_PD,
        "grad": grad,
        "a": a_value,
        "b_norm_squared": b_norm_squared,
        "b_l1": b_l1,
        "x_norm_squared": x_norm_squared,
        "Vdot": Vdot_value,
        "decay_margin": margin,
        "barrier": barrier_value,
        "barrier_margin": barrier_margin,
    }
