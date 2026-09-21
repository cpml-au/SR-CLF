"""Symbolic (sympy) dynamics of the reduced 3-D cart-pole.

GP variables map to the physical states as

    x1 = cart velocity        (was x2 in the 4-D system)
    x2 = pole angle theta     (was x3)
    x3 = pole angular rate    (was x4)

The names x1..x3 are kept because the GP primitive set and the program encoder
address variables positionally by those names.
"""
from sympy import Matrix, symbols, sin, cos

x1, x2, x3 = symbols("x1 x2 x3")

_G, _L, _D_COEF, _M_POLE, _M_CART = -10, 2, 1, 1, 5


def xSR(x1, x2, x3):
    return Matrix([x1, x2, x3])


def fSR(x1, x2, x3):
    g, L, d, m, M = _G, _L, _D_COEF, _M_POLE, _M_CART
    Sx = -sin(x2)
    Cx = -cos(x2)
    D = m * L * L * (M + m * (1 - Cx**2))
    common = m * L * x3**2 * Sx - d * x1
    f1 = (1 / D) * (-(m**2) * L**2 * g * Cx * Sx + m * L**2 * common)
    f2 = x3
    f3 = (1 / D) * ((m + M) * m * g * L * Sx - m * L * Cx * common)
    return Matrix([f1, f2, f3])


def GSR(x1, x2, x3):
    g, L, d, m, M = _G, _L, _D_COEF, _M_POLE, _M_CART
    Cx = -cos(x2)
    D = m * L * L * (M + m * (1 - Cx**2))
    u1 = m * L * L * (1 / D)
    u2 = m * L * Cx * (1 / D)
    # Column 1 is the input column, matching the 4-D file's convention.
    return Matrix([[0, u1, 0], [0, 0, 0], [0, -u2, 0]])


def QSR(x1, x2, x3):
    return Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def RSR(x1, x2, x3):
    return Matrix([[1e-4, 0, 0], [0, 1e-4, 0], [0, 0, 1e-4]])
