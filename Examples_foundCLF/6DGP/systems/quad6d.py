"""Indoor micro quadrotor, angular-rotation subsystem (Bouabdallah et al. 2004), 6 states, 3 inputs.

    x1 roll      x2 = d/dt roll
    x3 pitch     x4 = d/dt pitch
    x5 yaw       x6 = d/dt yaw

    x1' = x2
    x2' = x4 x6 (Iy - Iz)/Ix - (JR/Ix) x4 Omega + (l/Ix) U1
    x3' = x4
    x4' = x2 x6 (Iz - Ix)/Iy + (JR/Iy) x2 Omega + (l/Iy) U2
    x5' = x6
    x6' = x2 x4 (Ix - Iy)/Iz + (l/Iz) U3        (= 0 with Ix = Iy)

with Omega = sin(x2) cos(x4), Ix = Iy = 2, Iz = 5, l = 1, JR = 1, and D = {|x_i| <= 3}.

The control-affine split the CLF search uses is  x' = f(x) + G(x) u,  u = (U1, U2, U3):
f is the drift above WITHOUT the U terms and G is the constant 6x3 matrix of the U coefficients.
The paper's own controller (U1 = -(Ix/l)(x1 - x1d) - k1 x2, ...) is NOT used anywhere -- u comes
from Sontag's formula with this V, exactly as in the 4-D examples.
"""
import numpy as np
import sympy as sp

NAME = "quad6d"
N_STATES = 6
N_INPUTS = 3

IX = 2.0
IY = 2.0
IZ = 5.0
L = 1.0
JR = 1.0

# D = {|x_i| <= 3} of the paper.
BOX_HALF_WIDTHS = (3.0, 3.0, 3.0, 3.0, 3.0, 3.0)

STATE_NAMES = ("x1", "x2", "x3", "x4", "x5", "x6")


def fSR(*x):
    """Drift f(x) (sympy, 6x1): the dynamics with U1 = U2 = U3 = 0."""
    x1, x2, x3, x4, x5, x6 = x
    omega = sp.sin(x2) * sp.cos(x4)
    return sp.Matrix([
        x2,
        x4 * x6 * (sp.Float(IY) - sp.Float(IZ)) / sp.Float(IX) - sp.Float(JR) / sp.Float(IX) * x4 * omega,
        x4,
        x2 * x6 * (sp.Float(IZ) - sp.Float(IX)) / sp.Float(IY) + sp.Float(JR) / sp.Float(IY) * x2 * omega,
        x6,
        x2 * x4 * (sp.Float(IX) - sp.Float(IY)) / sp.Float(IZ),
    ])


def GSR(*x):
    """Input matrix G(x) (sympy, 6x3): the coefficients of U1, U2, U3."""
    G = sp.zeros(N_STATES, N_INPUTS)
    G[1, 0] = sp.Float(L) / sp.Float(IX)
    G[3, 1] = sp.Float(L) / sp.Float(IY)
    G[5, 2] = sp.Float(L) / sp.Float(IZ)
    return G


def QSR(*x):
    return sp.eye(N_STATES)


def RSR(*x):
    return sp.Float(1e-4) * sp.eye(N_INPUTS)


Q_MATRIX = np.eye(N_STATES)
R_MATRIX = 1e-4 * np.eye(N_INPUTS)
