"""The reversed Van der Pol system of the paper (section 4.4, Eq. 21), 2 states, 1 input.

    x1' = -x2
    x2' = x1 + x2 (x1^2 - 1) + x2 (x1^2 - 1) u

The drift is the paper's, f1 = -x2.  G = [[0], [x2 (x1^2 - 1)]].  Q = I and R = 1 (Q is used by the 'are' template, and by kappa* when verification.kappa_norm is xqx).
The box is +-2, with no points left out around x1^2 = 1 and x2 = 0.
With f1 = -x2 every radial V has a = b, a weak CLF.
"""
import numpy as np
import sympy as sp

NAME = "vanderpol2"
N_STATES = 2
N_INPUTS = 1

BOX_HALF_WIDTHS = (2.0, 2.0)
STATE_NAMES = ("x1", "x2")


def fSR(*x):
    """Drift f(x) (sympy, 2x1): the paper's f, f1 = -x2."""
    x1, x2 = x
    return sp.Matrix([-x2, x1 + x2 * (x1**2 - 1)])


def GSR(*x):
    """Input matrix G(x) (sympy, 2x1): the live column of the old [[0, 0], [0, x2 (x1^2 - 1)]]."""
    x1, x2 = x
    G = sp.zeros(N_STATES, N_INPUTS)
    G[1, 0] = x2 * (x1**2 - 1)
    return G


def QSR(*x):
    return sp.eye(N_STATES)


def RSR(*x):
    return sp.eye(N_INPUTS)


Q_MATRIX = np.eye(N_STATES)
R_MATRIX = np.eye(N_INPUTS)
