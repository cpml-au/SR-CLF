import numpy as np
from scipy.integrate import solve_ivp
from sympy import symbols, Matrix, lambdify


def NumSol(
    u_func_sym,
    T, x0, f, G,
    b_norm_squared_sym=None,
    TRQlim=None,
    ):
    """
    Numerically solves a system of differential equations with symbolic control input.

    Parameters:
        u_func_sym : sympy.Expr
            Symbolic expression for control input u(x).
        T : float
            Total simulation time.
        x0 : list or numpy.ndarray
            Initial conditions for all state variables [x1_0, x2_0, ..., xn_0].
        b_norm_squared_sym : sympy.Expr or None
            Symbolic expression for b_norm_squared (optional).
        TRQlim (float or None): If it is not None, clamps control input to physical limits (TRQlim)
            Whether to clip the control inputs.
        f, G : callables
            Functions returning symbolic system dynamics f(x) and G(x).

    Returns:
        t_vals : numpy.ndarray
            Array of time values.
        x_vals : list of numpy.ndarray
            List of arrays for each state variable over time.
    """

    n_states = len(x0)

    # Define symbolic state variables dynamically
    x_syms = symbols(f"x1:{n_states+1}")

    # Symbolic system dynamics
    f_vector = f(*x_syms)  # shape (n_states, 1)
    G_matrix = G(*x_syms)  # shape (n_states, n_inputs)

    # Symbolic control input
    if isinstance(u_func_sym, Matrix):
        u_sym = u_func_sym
    else:
        # single input, inject into second actuator (index 1)
        n_inputs = G_matrix.shape[1]
        u_entries = [0] * n_inputs
        u_entries[1] = u_func_sym
        u_sym = Matrix(u_entries)

    # Lambdify functions
    f_func = lambdify(x_syms, f_vector, "numpy")
    G_func = lambdify(x_syms, G_matrix, "numpy")
    u_func = lambdify(x_syms, u_sym, "numpy")

    if b_norm_squared_sym is not None:
        b_norm_squared_func = lambdify(x_syms, b_norm_squared_sym, "numpy")

    # Define the dynamics for numerical integration
    def dynamics(t, x):
        # Unpack states
        state_vals = list(x)

        # Evaluate system functions
        f_val = np.array(f_func(*state_vals)).flatten()  # (n_states,)
        G_val = np.array(G_func(*state_vals))  # (n_states, n_inputs)
        u_val = np.array(u_func(*state_vals)).flatten()  # (n_inputs,)

        # Enforce b_norm_squared constraint
        if b_norm_squared_sym is not None:
            b_norm_squared_val = b_norm_squared_func(*state_vals)
            if np.any(b_norm_squared_val < 1e-12):
                u_val[:] = 0  # Set entire control to zero

        # Apply control limits
        if TRQlim is not None:
            u_val = np.clip(u_val, -TRQlim, TRQlim)

        # Compute \dot{x} = f + G @ u
        dxdt = f_val + G_val @ u_val
        return dxdt

    # Set initial conditions and time span
    t_span = (0, T)
    t_eval = np.linspace(0, T, 2000)

    # Solve ODE system
    solution = solve_ivp(dynamics, t_span, x0, t_eval=t_eval)

    # Extract results
    t_vals = solution.t
    x_vals = [
        solution.y[i] for i in range(n_states)
    ]  # List of arrays per state variable

    return t_vals, x_vals
