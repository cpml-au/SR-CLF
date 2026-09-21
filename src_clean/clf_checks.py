"""Generic CLF-check machinery shared by example Evaluate modules.

System-agnostic pieces of the unified fitness evaluation, factored out of
examples/4DCartPoler/Evaluate.py + EvaluatePrime.py:

- certified sublevel set from the grid (boundary minimum of V - V(0));
- soft bounded-control CLF margin on the grid;
- single-point finite-difference evaluation of V, a = grad(V).f and
  b = grad(V).g (the active input column of G);
- DE falsifier for the CLF margin and the {b ~ 0, a > 0} manifold;
- closed-loop rollouts with the saturated min-norm (Sontag) controller.

System dynamics enter only as callables (f, G, Q, R) and all tuning
constants are explicit keyword arguments, so each example's Evaluate.py
owns its configuration. Single active input column is assumed
(input_index selects it, default 1 as in the cart-pole examples).
"""

import numpy as np
import pygmo as pg


def boundary_min(V_rel):
    """Minimum of V_rel over all faces of the grid boundary."""
    mn = np.inf
    for axis in range(V_rel.ndim):
        sl = [slice(None)] * V_rel.ndim
        sl[axis] = 0
        mn = min(mn, float(np.min(V_rel[tuple(sl)])))
        sl[axis] = -1
        mn = min(mn, float(np.min(V_rel[tuple(sl)])))
    return mn


def sublevel_cap_from_grid(V_vals, V_val_0):
    """Largest c such that {V - V(0) < c} stays inside the box: the minimum
    of V - V(0) over the grid boundary. CLF conditions are only certified
    (and falsified) inside this sublevel set."""
    return max(boundary_min(V_vals - V_val_0), 0.0)


def dynamics_on_grid(true_data, mesh, f, G, input_index=1,
                     cache_attr="_clf_dyn_cache"):
    """f and the active G column evaluated on the mesh, cached on the
    dataset (they do not depend on the candidate, so compute once per
    worker)."""
    cache = getattr(true_data, cache_attr, None)
    if cache is None:
        f_vec = np.asarray(f(*mesh), dtype=float)
        g_col = np.asarray(G(*mesh), dtype=float)[:, input_index]
        cache = (f_vec, g_col)
        setattr(true_data, cache_attr, cache)
    return cache


class _FalsifyProblem:
    def __init__(self, V_call, V_val_0, mode, bounds, c_star, f, G_point,
                 u_max, alpha, manifold_weight, fd_step, input_index):
        self.V_call = V_call
        self.V_val_0 = V_val_0
        self.mode = mode
        self.bounds = np.asarray(bounds, dtype=float)
        self.c_star = c_star
        self.f = f
        self.G_point = G_point
        self.u_max = u_max
        self.alpha = alpha
        self.manifold_weight = manifold_weight
        self.fd_step = fd_step
        self.input_index = input_index

    def fitness(self, z):
        try:
            V_value, a_value, b_value, _, _ = point_terms(
                self.V_call, z, self.f, self.G_point,
                fd_step=self.fd_step, input_index=self.input_index,
            )
            V_rel = V_value - self.V_val_0
            if self.mode == "margin":
                value = -(a_value - self.u_max * abs(b_value)
                          + self.alpha * max(V_rel, 0.0))
            else:  # "manifold": drive b to zero while a stays positive
                value = self.manifold_weight * b_value * b_value - a_value
            if V_rel > self.c_star:
                # sloped penalty keeps the search inside the certified
                # region while guiding the optimizer toward it
                value += 1e3 + (V_rel - self.c_star)
        except Exception:
            value = 1e6
        if not np.isfinite(value):
            value = 1e6
        return [float(value)]

    def get_bounds(self):
        return (self.bounds[:, 0], self.bounds[:, 1])


def rollout_penalty(V_call, V_val_0, f, G_point, Q, R, x0s, *,
                    t_final=1.5, dt=2e-3,
                    diverge_norm=1.0, diverge_penalty=1000.0,
                    v_increase_weight=50.0, final_v_weight=10.0,
                    converged_norm=1e-3, fd_step=1e-5, input_index=1):
    """Closed-loop RK4 rollouts under the saturated min-norm controller.
    Penalizes divergence, V increases along the trajectory, and net V
    growth. Returns (penalty, per-rollout details)."""
    n = len(x0s[0])
    origin = [0.0] * n
    r_scalar = float(np.asarray(R(*origin))[input_index, input_index])
    Q_point = np.asarray(Q(*origin), dtype=float)

    def rhs(x):
        V_value, a_value, b_value, f_vec, g_col = point_terms(
            V_call, x, f, G_point, fd_step=fd_step, input_index=input_index
        )
        bn2 = b_value * b_value / r_scalar
        if bn2 > 1e-18:
            xn2 = float(x @ Q_point @ x)
            lam = (np.sqrt(a_value * a_value + xn2 * bn2) + a_value) / bn2
            u = -(b_value / r_scalar) * lam
        else:
            u = 0.0
        u = float(u)
        return f_vec + g_col * u, V_value

    total = 0.0
    details = []
    n_steps = int(round(t_final / dt))

    for x0 in x0s:
        x = np.asarray(x0, dtype=float)
        diverged = False
        increases = 0
        steps_taken = 0
        V_prev = None
        V_begin = None
        V_last = None

        for _ in range(n_steps):
            try:
                k1, V_here = rhs(x)
                k2, _ = rhs(x + 0.5 * dt * k1)
                k3, _ = rhs(x + 0.5 * dt * k2)
                k4, _ = rhs(x + dt * k3)
            except Exception:
                diverged = True
                break

            if V_begin is None:
                V_begin = V_here
            if V_prev is not None and V_here > V_prev + 1e-8 + 1e-5 * abs(V_prev):
                increases += 1
            V_prev = V_here
            V_last = V_here
            steps_taken += 1

            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if not np.all(np.isfinite(x)) or np.linalg.norm(x) > diverge_norm:
                diverged = True
                break
            if np.linalg.norm(x) < converged_norm:
                break

        if diverged:
            roll = diverge_penalty
        else:
            frac_inc = increases / max(steps_taken, 1)
            roll = v_increase_weight * frac_inc
            if V_begin is not None and V_last is not None:
                rel_begin = max(V_begin - V_val_0, 1e-12)
                ratio = (V_last - V_val_0) / rel_begin
                roll += final_v_weight * max(0.0, ratio - 1.0)

        total += roll
        details.append({
            "x0": tuple(x0),
            "diverged": diverged,
            "steps": steps_taken,
            "final_norm": float(np.linalg.norm(x)),
            "V_increase_steps": increases,
            "penalty": float(roll),
        })

    return float(total), details
