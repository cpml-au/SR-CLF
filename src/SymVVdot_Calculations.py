from sympy import Matrix, sqrt


# Define the gradient of V(x)
def V_x(V, x_syms):
    """Compute symbolic gradient of V with respect to given symbols."""
    return Matrix([V.diff(xi) for xi in x_syms])


def _compute_v_and_v_dotSR_terms(V_value, f, G, Q, R, x_syms, u_func):
    f_vector = f(*x_syms)  # shape (n_states, 1)
    G_matrix = G(*x_syms)  # shape (n_states, n_inputs)
    Q_matrix = Q(*x_syms)
    R_matrix = R(*x_syms)

    # State vector
    x_vector = Matrix(x_syms)

    # Gradient of V
    V_grad = V_x(V_value, x_syms)  # shape (n_states, 1)

    # Compute a and b_T
    a = (V_grad.T * f_vector)[0]
    b_T = V_grad.T * G_matrix  # shape (1, n_inputs)

    # Norms
    b_norm_squared = (b_T * R_matrix.inv() * b_T.T)[0]
    x_norm_squared = (x_vector.T * Q_matrix * x_vector)[0]

    # lambda_x
    """
    lambda_x = safe_divideSR(
        sqrt(a**2 + x_norm_squared * b_norm_squared**2) + a,
        b_norm_squared,
        default=0
    )
    """
    lambda_x = (sqrt(a**2 + x_norm_squared * b_norm_squared) + a) / (b_norm_squared+1e-6)
    # lambda_x = (sqrt(a**2 + x_norm_squared * b_norm_squared**2) + a) / (b_norm_squared+1e-6)

    # lambda_x = (sqrt(a**2 + x_norm_squared * b_norm_squared**2) + a) / sqrt(1 + b_norm_squared**2)

    # Control law
    if u_func is None:
        u_value = -R_matrix.inv() * b_T.T * lambda_x
    else:
        # user-supplied u_func is scalar, maps to second input (index 1)
        n_inputs = G_matrix.shape[1]
        u_entries = [0] * n_inputs
        u_entries[1] = u_func
        u_value = Matrix(u_entries)

    # Alternative control (without lambda_x scaling)
    u_value2 = -R_matrix.inv() * b_T.T

    # Time derivative of V
    V_dot = (V_grad.T * f_vector + V_grad.T * G_matrix * u_value)[0]

    # Control effort
    u_mag = (u_value.T * R_matrix * u_value)[0]

    return V_dot, lambda_x, u_value, u_value2, u_mag, [b_norm_squared, b_T[0], b_T[1]], b_T[0, 1], a


# Compute V and V_dot
def compute_v_and_v_dotSR(V_value, f, G, Q, R, x_syms, u_func):
    """
    Computes the symbolic Lyapunov derivative and control terms.

    The return signature is preserved for existing examples: only the second
    control component is returned as ``u_func``.
    """
    V_dot, lambda_x, u_value, u_value2, u_mag, control_terms, b1, a = (
        _compute_v_and_v_dotSR_terms(V_value, f, G, Q, R, x_syms, u_func)
    )

    # Return important terms
    return (
        V_value,
        V_dot,
        lambda_x,
        u_value[1],
        u_value2[1],
        u_mag,
        control_terms,
        b1,
        a,
    )


def compute_v_and_v_dotSR_full_control(V_value, f, G, Q, R, x_syms, u_func):
    """
    Same symbolic CLF terms as ``compute_v_and_v_dotSR``, but returns the full
    input vector. Use this for multi-input systems such as the Caltech fan.
    """
    V_dot, lambda_x, u_value, u_value2, u_mag, control_terms, b1, a = (
        _compute_v_and_v_dotSR_terms(V_value, f, G, Q, R, x_syms, u_func)
    )

    return (
        V_value,
        V_dot,
        lambda_x,
        u_value,
        u_value2,
        u_mag,
        control_terms,
        b1,
        a,
    )
