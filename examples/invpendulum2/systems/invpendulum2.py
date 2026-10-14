"""The damped inverted pendulum of the paper (section 4.1, Eq. 17), 2 states, 1 input.

    x1' = x2
    x2' = (g / l) sin(x1) - b / (m l^2) x2 + 26.7 u,        g, l, b, m = 9.81, 0.5, 0.1, 0.15

G = [[0], [26.7]].  Q = I and R = 1 (Q is used by the 'are' template, and by kappa* when verification.kappa_norm is xqx).
The box is +-4.
"""
import numpy as np
import sympy as sp

NAME = "invpendulum2"
N_STATES = 2
N_INPUTS = 1

BOX_HALF_WIDTHS = (4.0, 4.0)
STATE_NAMES = ("x1", "x2")


def fSR(*x):
    """Drift f(x) (sympy, 2x1), with the old code's constants."""
    x1, x2 = x
    g, l, b, m = 9.81, 0.5, 0.1, 0.15
    A, B = g / l, -b / (m * l**2)
    return sp.Matrix([x2, A * sp.sin(x1) + B * x2])


def GSR(*x):
    """Input matrix G(x) (sympy, 2x1): the live column of the old [[0, 0], [0, 26.7]]."""
    G = sp.zeros(N_STATES, N_INPUTS)
    G[1, 0] = sp.Float(26.7)
    return G


def QSR(*x):
    return sp.eye(N_STATES)


def RSR(*x):
    return sp.eye(N_INPUTS)


Q_MATRIX = np.eye(N_STATES)
R_MATRIX = np.eye(N_INPUTS)
