# Compute V, V_dot, and related terms numerically
import numpy as np
from sympy import symbols, lambdify
from src.Functions import safe_divide  # Import a safe division utility function


def compute_v_and_v_dot(V_function, state_grids, true_data, scales, f, G, Q, R, TRQlim, u_func, TrnOFFb=True):
    """
    Dimension-agnostic computation of V, V_dot, and control terms.

    Parameters:
        V_function (np.ndarray or sympy.Expr): Control Lyapunov Function (CLF)
        (either pre-evaluated or symbolic)
        state_grids (list of np.ndarray): List of 1D arrays defining the
        discretized domain per state dimension
        TRQlim (float or None): If it is not None, clamps control input to physical limits (TRQlim)
        u_func (sympy.Expr or None): Optional symbolic control function u(x)
        f, G, Q, R (callables): System dynamics and cost matrices as functions of state

    Returns:
        tuple: Computed quantities:
            - V_vals: CLF function
            - V_grad_mag: Gradient magnitude of V
            - V_dot_vals: Time derivative of V
            - lambda_x_vals: Optimal lambda term
            - u_vals: Optimal control inputs
            - u_mag: Control effort magnitude
            - a: Gradient dot f
            - b_norm_squared: Norm squared of control direction
            - x_norm_squared: Norm squared of state
    """
    # Create meshgrid over all state variables
    # mesh = np.meshgrid(*state_grids, indexing="ij")  # Each has shape [...grid]
    mesh = true_data.mesh
    n_states = len(state_grids)

    # Stack mesh into a final state grid [..., n_states]
    x_stack = np.stack(mesh, axis=-1)

    # Normalized mesh (for V evaluation)
    mesh_norm = [M / s for M, s in zip(mesh, scales)]
    state_grids_norm = [M / s for M, s in zip(state_grids, scales)]
    
    # Evaluate system dynamics and cost matrices
    f_vector = f(*mesh)  # shape (n_states, ...)
    # f_vector = np.array([fi.T for fi in f_vector])
    f_vector = np.array(f_vector)

    G_matrix = G(*mesh)  # shape (n_states, n_inputs, ...)
    Q_matrix = Q(*mesh)  # shape (n_states, n_states, ...)
    R_matrix = R(*mesh)  # shape (n_inputs, n_inputs, ...)

    # Evaluate or assign CLF function
    if isinstance(V_function, np.ndarray):
        V_vals = V_function
    else:
        x_syms = symbols(f"x1:{n_states+1}")
        V_func = lambdify(x_syms, V_function, modules="numpy")
        V_vals = V_func(*mesh_norm)
        # V_vals = V_vals.T

    # Compute gradient of V
    gradients = np.gradient(
        V_vals, *state_grids_norm, edge_order=2
    )  # one for each state var
    gradients = [g_i / s_i for g_i, s_i in zip(gradients, scales)]
    # V_grad = np.stack(
    #     gradients[::-1], axis=-1
    # )  # Reverse and stack → [∂V/∂x₁, ∂V/∂x₂, ..., ∂V/∂xₙ]
    V_grad = np.stack(gradients, axis=-1)
    V_grad_mag = np.linalg.norm(V_grad, axis=-1)

    # Compute control-related terms
    b_T = np.einsum("...i,ij...->...j", V_grad, G_matrix)  # shape [..., n_inputs]
    a = np.einsum("...i,i...->...", V_grad, f_vector)  # scalar field

    # Compute quadratic norms
    x_norm_squared = np.einsum("...i,ij,...j->...", x_stack, Q_matrix, x_stack)
    b_norm_squared = np.einsum("...i,ij,...j->...", b_T, np.linalg.inv(R_matrix), b_T)

    # Compute lambda_x values
    if not TrnOFFb:
        lambda_x_vals = safe_divide(
            np.sqrt(a**2 + x_norm_squared * b_norm_squared**2) + a, b_norm_squared, default=0
        )
    else:
        lambda_x_vals = (np.sqrt(a**2 + x_norm_squared * b_norm_squared) + a) / (b_norm_squared+1e-6)
        # lambda_x_vals = (np.sqrt(a**2 + x_norm_squared * b_norm_squared**2) + a) / (b_norm_squared+1e-6)
        #lambda_x_vals = (np.sqrt(a**2 + x_norm_squared * b_norm_squared**2) + a) / np.sqrt(b_norm_squared**2+1)

    # Compute optimal control input u
    if u_func is not None:
        x_syms = symbols(f"x1:{n_states+1}")
        u_func_num = lambdify(x_syms, u_func, modules="numpy")
        u_inputs = u_func_num(*mesh)  # Assume it returns shape like (n_inputs, ...)
        # u_inputs = u_inputs.T
        # Create u_vals with zeros, final shape: (100, 100, n_states)
        u_vals = np.zeros((*u_inputs.shape, n_states))

        # Fill the second "state slot" (index 1)
        u_vals[..., 1] = u_inputs
    else:
        u_vals = -np.einsum(
            "ij,...j->...i", np.linalg.inv(R_matrix), b_T * lambda_x_vals[..., None]
        )

    if TRQlim is not None:
        u_vals = np.clip(u_vals, -TRQlim, TRQlim)

    if not TrnOFFb:
        u_vals[b_norm_squared < 1e-6] = 0

    # Compute V̇ = ∇V·f + ∇V·(G·u) - the time derivative of the CLF function (V_dot)
    V_dot_f = np.einsum("...i,i...->...", V_grad, f_vector)
    G_u = np.einsum("ij...,...j->i...", G_matrix, u_vals)
    V_dot_G = np.einsum("...i,i...->...", V_grad, G_u)
    V_dot_vals = V_dot_f + V_dot_G

    # Compute control effort magnitude
    u_mag = np.einsum("...i,ij,...j->...", u_vals, R_matrix, u_vals)

    # Return computed results
    return (
        V_vals,
        V_grad_mag,
        V_dot_vals,
        lambda_x_vals,
        u_vals[..., 1],
        u_mag,
        a,
        [b_norm_squared, b_T.T[0], b_T.T[1]],
        x_norm_squared,
    )
