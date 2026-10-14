"""2-D Example 2 of the paper (section 4.3, Eq. 20), 2 states, 1 input.

    x1' = x2
    x2' = -exp(x2) (x1 + x2/2) + (x2/2) exp(4 x1 + 3 x2) + exp(2 x1 + 2 x2) u

G = [[0], [exp(2 x1 + 2 x2)]].  Q = I and R = 1 (Q is used by the 'are' template, and by kappa* when verification.kappa_norm is xqx).
The box is +-2.
"""
import numpy as np
import sympy as sp

NAME = "ex2d2"
N_STATES = 2
N_INPUTS = 1

BOX_HALF_WIDTHS = (2.0, 2.0)
STATE_NAMES = ("x1", "x2")


def fSR(*x):
    """Drift f(x) (sympy, 2x1)."""
    x1, x2 = x
    return sp.Matrix([x2, -sp.exp(x2) * (x1 + 0.5 * x2) + 0.5 * x2 * sp.exp(4 * x1 + 3 * x2)])


def GSR(*x):
    """Input matrix G(x) (sympy, 2x1): the live column of the old [[0, 0], [0, exp(2 x1 + 2 x2)]]."""
    x1, x2 = x
    G = sp.zeros(N_STATES, N_INPUTS)
    G[1, 0] = sp.exp(2 * x1 + 2 * x2)
    return G


def QSR(*x):
    return sp.eye(N_STATES)


def RSR(*x):
    return sp.eye(N_INPUTS)


Q_MATRIX = np.eye(N_STATES)
R_MATRIX = np.eye(N_INPUTS)
