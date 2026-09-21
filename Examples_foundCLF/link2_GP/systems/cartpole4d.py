"""The 4-D cart-pole of examples/4DCartPolerV8 -- the ACCEPTANCE system of this folder.

Identical to 4DCartPolerV8/SystemDynamicsSR.py (g = -10, L = 2, d = 1, m = 1, M = 5) with the
single active input column G[:, 1] written as the one column of a 4x1 G, so that this folder's
generic (n states, m inputs) code reproduces the published 4-D numbers with its own evaluators.
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
