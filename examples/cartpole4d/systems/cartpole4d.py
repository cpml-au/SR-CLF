"""The 4-D cart-pole (system.name = cartpole4d), 4 states, 1 input.

Imported by src/CPU/config.py::system(); defines NAME, N_STATES = 4, N_INPUTS = 1, BOX_HALF_WIDTHS (the
box when SYMCLF_BOX is unset), STATE_NAMES, the sympy fSR (n, 1) and GSR (n, m) with every column an
input, QSR = I, RSR = 1e-4 I, and the numeric Q_MATRIX / R_MATRIX.  g = -10, L = 2, d = 1, m = 1, M = 5;
the single input is the force on the cart.

    Sx = -sin(x3),  Cx = -cos(x3),  D = m L^2 (M + m (1 - Cx^2))
    xdot = fSR(x) + GSR(x) u,   GSR[:, 0] = (0, m L L (1/D), 0, -m L Cx (1/D))

system.name (SYMCLF_CPU_SYSTEM) selects this file; system.box (SYMCLF_BOX) replaces its box.
"""
import numpy as np
import sympy as sp

NAME = "cartpole4d"
N_STATES = 4
N_INPUTS = 1

BOX_HALF_WIDTHS = (0.25, 0.25, 0.0494, 0.1155)
STATE_NAMES = ("x1", "x2", "x3", "x4")

_g = -10
_L = 2
_d = 1
_m = 1
_M = 5


def fSR(*x):
    x1, x2, x3, x4 = x
    Sx = -sp.sin(x3)
    Cx = -sp.cos(x3)
    D = _m * _L * _L * (_M + _m * (1 - Cx ** 2))
    return sp.Matrix([
        x2,
        (1 / D) * (-_m ** 2 * _L ** 2 * _g * Cx * Sx + _m * _L ** 2 * (_m * _L * x4 ** 2 * Sx - _d * x2)),
        x4,
        (1 / D) * ((_m + _M) * _m * _g * _L * Sx - _m * _L * Cx * (_m * _L * x4 ** 2 * Sx - _d * x2)),
    ])


def GSR(*x):
    x1, x2, x3, x4 = x
    Cx = -sp.cos(x3)
    D = _m * _L * _L * (_M + _m * (1 - Cx ** 2))
    u1 = _m * _L * _L * (1 / D)
    u2 = _m * _L * Cx * (1 / D)
    return sp.Matrix([[0], [u1], [0], [-u2]])


def QSR(*x):
    return sp.eye(N_STATES)


def RSR(*x):
    return sp.Float(1e-4) * sp.eye(N_INPUTS)


Q_MATRIX = np.eye(N_STATES)
R_MATRIX = 1e-4 * np.eye(N_INPUTS)
