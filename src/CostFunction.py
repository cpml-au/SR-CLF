import numpy as np
from sympy import symbols, Matrix, lambdify


def calculate_cost(
    t_vals,
    x_vals, 
    u_func_sym, 
    G, Q, R, 
    b_norm_squared_sym, 
    TRQlim, 
    skip_steps,
):
    """
    Calculate the cost function J = ∫ (x^T Q x + u^T R u) dt.

    Parameters:
        t_vals : numpy.ndarray
            Time points.
        x_vals : list of numpy.ndarray
            List of arrays for each state variable over time.
        u_func_sym : sympy.Expr
            Symbolic expression for control input u(x).
        b_norm_squared_sym : sympy.Expr or None
            Symbolic expression to deactivate control if too small.
        TRQlim (float or None): If it is not None, clamps control input to physical limits (TRQlim)
            Whether to clip the control inputs.
        skip_steps : int
            Number of initial steps to skip in cost computation.
        Q, R : callables
            Symbolic functions for Q(x) and R(x).

    Returns:
        total_cost : float
        total_cost1 : float (state cost)
        total_cost2 : float (control cost)
        u_vals : numpy.ndarray (evaluated control inputs over time)
    """

    n_states = len(x_vals)
    x_syms = symbols(f"x1:{n_states+1}")

    # Build full symbolic control input vector
    if isinstance(u_func_sym, Matrix):
        u_sym = u_func_sym
    else:
        # Assume user gives scalar control for second input (index 1)
        n_inputs = G(*x_syms).shape[1]
        u_entries = [0] * n_inputs
        u_entries[1] = u_func_sym
        u_sym = Matrix(u_entries)

    # Lambdify necessary functions
    u_func = lambdify(x_syms, u_sym, "numpy")
    if b_norm_squared_sym is not None:
        b_norm_squared_func = lambdify(x_syms, b_norm_squared_sym, "numpy")

    # Cost accumulators
    total_cost = 0
    total_cost1 = 0
    total_cost2 = 0
    u_val_list = []

    dt = t_vals[1] - t_vals[0]

    for i in range(len(t_vals)):
        if i < skip_steps:
            continue

        try:
            # Get current state
            x_now = np.array([x[i] for x in x_vals])

            # Evaluate Q, R, u
            Q_val = np.array(Q(*x_now), dtype=float)
            R_val = np.array(R(*x_now), dtype=float)
            u_val = np.array(u_func(*x_now), dtype=float).flatten()

            # Enforce b_norm_squared condition
            if b_norm_squared_sym is not None:
                b_norm_val = b_norm_squared_func(*x_now)
                if np.any(b_norm_val < 1e-12):
                    u_val[:] = 0

            # Apply control limits
            if TRQlim is not None:
                u_val = np.clip(u_val, -TRQlim, TRQlim)

            u_val_list.append(u_val)

            # Calculate cost terms
            cost1 = x_now.T @ Q_val @ x_now
            cost2 = u_val.T @ R_val @ u_val
            cost = cost1 + cost2

            total_cost1 += cost1 * dt
            total_cost2 += cost2 * dt
            total_cost += cost * dt

        except Exception as e:
            print(f"Skipping t={t_vals[i]} due to error: {e}")
            continue

    # Finalize u_vals array
    u_vals = np.array(u_val_list)

    return total_cost, total_cost1, total_cost2, u_vals
