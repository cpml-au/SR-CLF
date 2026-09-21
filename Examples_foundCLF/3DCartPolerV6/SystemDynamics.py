"""Reduced 3-D cart-pole: the full system minus the cart position.

States ``(z1, z2, z3) = (cart velocity, pole angle, pole angular rate)``,
one input (horizontal force on the cart).

THE REDUCTION IS EXACT. In examples/4DCartPoler/SystemDynamics.py the drift
``f`` never reads ``x1`` and ``G`` depends only on ``x3``, so ``(x2, x3, x4)``
is a closed subsystem -- this is not a linearisation or an approximation, it is
the same vector field with one coordinate projected out.

WHAT IT KEEPS: underactuation (1 input, 3 states), sin/cos coupling, a
state-dependent input gain, the 1/D denominator, and the saturation behaviour
near the b = 0 manifold.

WHAT IT REMOVES: the translation invariance in cart position. In 4-D that made
``f == 0`` on the entire x1 axis, so ``a = grad(V).f == 0`` there for ANY V and
a whole slab of ``{b = 0}`` certified for free -- the degeneracy the 4-D
FINDINGS document at length. It cannot occur here.

Same constants as the 4-D file: g = -10, L = 2, d = 1, m = 1, M = 5.
"""
import numpy as np

_G, _L, _D_COEF, _M_POLE, _M_CART = -10.0, 2.0, 1.0, 1.0, 5.0


def f(z1, z2, z3):
    """Drift of the reduced cart-pole -> [dz1, dz2, dz3]."""
    g, L, d, m, M = _G, _L, _D_COEF, _M_POLE, _M_CART
    Sx = -np.sin(z2)
    Cx = -np.cos(z2)
    D = m * L * L * (M + m * (1 - Cx**2))
    common = m * L * z3**2 * Sx - d * z1
    dz1 = (1 / D) * (-(m**2) * L**2 * g * Cx * Sx + m * L**2 * common)
    dz2 = z3
    dz3 = (1 / D) * ((m + M) * m * g * L * Sx - m * L * Cx * common)
    return np.array([dz1, dz2, dz3])


def G(z1, z2, z3):
    """Control influence, shape (3, 3, *grid_shape); column 1 is the input."""
    m, L, M = _M_POLE, _L, _M_CART
    Cx = -np.cos(z2)
    D = m * L * L * (M + m * (1 - Cx**2))
    u1 = m * L * L * (1 / D)
    u2 = m * L * Cx * (1 / D)
    G_matrix = np.zeros((3, 3) + np.shape(z2))
    G_matrix[0, 1] = u1
    G_matrix[2, 1] = -u2
    return G_matrix


def Q(z1, z2, z3):
    """State cost (identity), matching the 4-D file's convention."""
    return np.eye(3)


def R(z1, z2, z3):
    """Control cost, matching the 4-D file's 1e-4."""
    return np.eye(3) * 0.0001
