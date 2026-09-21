# System Dynamics Model
import numpy as np


# Define the system dynamics numerically
def f(x1, x2, x3, x4):
    """
    Nonlinear cart-pole system dynamics f(x) without control input.
    
    Args:
        x1 (float): cart position
        x2 (float): cart velocity
        x3 (float): pole angle (theta)
        x4 (float): pole angular velocity

    Returns:
        np.ndarray: [dx1/dt, dx2/dt, dx3/dt, dx4/dt]
    """
    # Constants
    g = -10       # gravity (m/s^2)
    L = 2         # length of pole (m)
    d = 1         # damping coefficient
    m = 1         # mass of pendulum (kg)
    M = 5         # mass of cart (kg)

    Sx = -np.sin(x3)
    Cx = -np.cos(x3)
    D = m * L * L * (M + m * (1 - Cx**2))

    dx1 = x2
    dx2 = (1 / D) * (-m**2 * L**2 * g * Cx * Sx + m * L**2 * (m * L * x4**2 * Sx - d * x2))
    dx3 = x4
    dx4 = (1 / D) * ((m + M) * m * g * L * Sx - m * L * Cx * (m * L * x4**2 * Sx - d * x2))

    return np.array([dx1, dx2, dx3, dx4])


def G(x1, x2, x3, x4):
    """
    Returns a batch of 4x4 control influence matrices evaluated over meshgrid inputs.
    
    Inputs:
        x1, x2, x3, x4: Meshgrid arrays (same shape)
    
    Output:
        G_matrix: shape (4, 4, *grid_shape)
    """
    # Constants
    g = -10
    L = 2
    d = 1
    m = 1
    M = 5

    Cx = -np.cos(x3)
    D = m * L * L * (M + m * (1 - Cx**2))  # shape: (N, N, N, N)
    u1 = m * L * L * (1 / D)  # shape: (N, N, N, N)
    u2 = m * L * Cx * (1 / D)  # shape: (N, N, N, N)

    grid_shape = x3.shape  # Should be e.g. (50, 50, 50, 50)
    
    # Initialize G_matrix: shape (4, 4, *grid_shape)
    G_matrix = np.zeros((4, 4) + grid_shape)

    # Assign only nonzero entries
    G_matrix[1, 1] = u1
    G_matrix[3, 1] = -u2

    return G_matrix


def Q(x1, x2, x3, x4):
    """
    Defines the state cost matrix Q(x) used in optimal control.
    
    Returns:
        ndarray: A (4×4) positive semi-definite matrix representing the weight on state errors.
    """
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def R(x1, x2, x3, x4):
    """
    Defines the control cost matrix R(x) used in optimal control.
    
    Returns:
        ndarray: A (4×4) positive definite matrix representing the cost of control effort.
    """
    return np.array([[0.0001, 0, 0, 0], [0, 0.0001, 0, 0], [0, 0, 0.0001, 0], [0, 0, 0, 0.0001]])


