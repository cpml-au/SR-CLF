"""The 3-D cart-pole (system.name = cartpole3d): the 4-D cart-pole without the cart position, 3 states, 1 input.

States: x1 = cart velocity, x2 = pole angle theta, x3 = pole angular rate (the 4-D system's x2, x3, x4);
g = -10, L = 2, d = 1, m = 1, M = 5.  The single input is the force on the cart.  Q = I, R = 1e-4.
    Sx = -sin(x2),  Cx = -cos(x2),  D = m L^2 (M + m (1 - Cx^2)),  common = m L x3^2 Sx - d x1
    xdot = fSR(x) + GSR(x) u,   GSR[:, 0] = (m L L (1/D), 0, -m L Cx (1/D))
The box (0.25, 0.0494, 0.1155) is the 4-D box without the cart position.
system.name (SYMCLF_CPU_SYSTEM) selects this file; system.box (SYMCLF_BOX) replaces its box.
"""
import numpy as np
import sympy as sp

NAME = "cartpole3d"
N_STATES = 3
N_INPUTS = 1

BOX_HALF_WIDTHS = (0.25, 0.0494, 0.1155)
STATE_NAMES = ("x1", "x2", "x3")

_g = -10
_L = 2
_d = 1
_m = 1
_M = 5


def fSR(*x):
    x1, x2, x3 = x
    Sx = -sp.sin(x2)
    Cx = -sp.cos(x2)
    D = _m * _L * _L * (_M + _m * (1 - Cx ** 2))
    common = _m * _L * x3 ** 2 * Sx - _d * x1
    return sp.Matrix([
        (1 / D) * (-(_m ** 2) * _L ** 2 * _g * Cx * Sx + _m * _L ** 2 * common),
        x3,
        (1 / D) * ((_m + _M) * _m * _g * _L * Sx - _m * _L * Cx * common),
    ])


def GSR(*x):
    x1, x2, x3 = x
    Cx = -sp.cos(x2)
    D = _m * _L * _L * (_M + _m * (1 - Cx ** 2))
    u1 = _m * _L * _L * (1 / D)
    u2 = _m * _L * Cx * (1 / D)
    return sp.Matrix([[u1], [0], [-u2]])


def QSR(*x):
    return sp.eye(N_STATES)


def RSR(*x):
    return sp.Float(1e-4) * sp.eye(N_INPUTS)


Q_MATRIX = np.eye(N_STATES)
R_MATRIX = 1e-4 * np.eye(N_INPUTS)
