"""2-D Example 1 of the paper (Freeman & Primbs 1996, the exact-value-function example), 2 states, 1 input.

    x1' = x2
    x2' = -x1 + x2 sinh(x1^2 + x2^2) + u

G = [[0], [1]].  Q = diag(0, 1) and R = 1 (Q is used by the 'are' template, and by kappa* when verification.kappa_norm is xqx).
The box is +-2 (on +-4 the term x2 sinh(x1^2 + x2^2) would need an input bound of ~1e14).
x1^2 + x2^2 is only a weak CLF for this system: a = 0 on b = 0.
"""
import numpy as np
import sympy as sp

NAME = "ex2d1"
N_STATES = 2
N_INPUTS = 1

BOX_HALF_WIDTHS = (2.0, 2.0)
STATE_NAMES = ("x1", "x2")


def fSR(*x):
    """Drift f(x) (sympy, 2x1)."""
    x1, x2 = x
    return sp.Matrix([x2, -x1 + x2 * sp.sinh(x1**2 + x2**2)])


def GSR(*x):
    """Input matrix G(x) (sympy, 2x1): the live column of the old [[0, 0], [0, 1]]."""
    G = sp.zeros(N_STATES, N_INPUTS)
    G[1, 0] = sp.Integer(1)
    return G


def QSR(*x):
    return sp.Matrix([[0, 0], [0, 1]])


def RSR(*x):
    return sp.eye(N_INPUTS)


Q_MATRIX = np.array([[0.0, 0.0], [0.0, 1.0]])
R_MATRIX = np.eye(N_INPUTS)
