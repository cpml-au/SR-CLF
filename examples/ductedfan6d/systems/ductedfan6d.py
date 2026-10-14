"""Caltech ducted fan in hover mode, 6 states, 2 inputs.

    x1 = x        x2 = y        x3 = theta          (position and orientation of the fan)
    x4 = xdot     x5 = ydot     x6 = thetadot

    xdot'     = (-d xdot + u1 cos(theta) - u2 sin(theta)) / m
    ydot'     = (-d ydot + u1 sin(theta) - u2 cos(theta) - m g) / m
    thetadot' = r u1 / I

with g = 0.28, m = 11.2, I = 0.0462, r = 0.156, d = 0.1.

HOVER SHIFT (an assumption made here): with u = 0 the origin is NOT an equilibrium --
gravity gives ydot' = -g. The thrust is therefore measured from its hover value: u2 = u2' - m g,
i.e. u2' = 0 holds the fan against gravity at theta = 0. The system below is in u' = (u1, u2'),
f(0) = 0, and nothing else is changed. The source's own neural controller and neural V are NOT used:
u comes from Sontag's formula with the V this pipeline finds.

    f = [x4, x5, x6, (-d x4 + m g sin(x3))/m, (-d x5 + m g (cos(x3) - 1))/m, 0]
    G = [[0,0],[0,0],[0,0], [cos(x3)/m, -sin(x3)/m], [sin(x3)/m, -cos(x3)/m], [r/I, 0]]
"""
import numpy as np
import sympy as sp

NAME = "ductedfan6d"
N_STATES = 6
N_INPUTS = 2

G_ACC = 0.28
MASS = 11.2
INERTIA = 0.0462
RADIUS = 0.156
DRAG = 0.1

# The source gives no domain; the box is a choice: positions +-0.5 m, theta +-0.3 rad, velocities +-0.5.
BOX_HALF_WIDTHS = (0.5, 0.5, 0.3, 0.5, 0.5, 0.5)
STATE_NAMES = ("x", "y", "theta", "xdot", "ydot", "thetadot")


def fSR(*x):
    """Drift f(x) (sympy, 6x1): the hover-shifted dynamics with u1 = u2' = 0."""
    _x, _y, th, dx, dy, dth = x
    m, g, d = sp.Float(MASS), sp.Float(G_ACC), sp.Float(DRAG)
    return sp.Matrix([
        dx,
        dy,
        dth,
        (-d * dx + m * g * sp.sin(th)) / m,
        (-d * dy + m * g * (sp.cos(th) - sp.Integer(1))) / m,
        sp.Integer(0),
    ])


def GSR(*x):
    """Input matrix G(x) (sympy, 6x2) for u' = (u1, u2'), u2 = u2' - m g."""
    _x, _y, th, dx, dy, dth = x
    m, r, inertia = sp.Float(MASS), sp.Float(RADIUS), sp.Float(INERTIA)
    G = sp.zeros(N_STATES, N_INPUTS)
    G[3, 0] = sp.cos(th) / m
    G[4, 0] = sp.sin(th) / m
    G[5, 0] = r / inertia
    G[3, 1] = -sp.sin(th) / m
    G[4, 1] = -sp.cos(th) / m
    return G


def QSR(*x):
    return sp.eye(N_STATES)


def RSR(*x):
    return sp.Float(1e-4) * sp.eye(N_INPUTS)


Q_MATRIX = np.eye(N_STATES)
R_MATRIX = 1e-4 * np.eye(N_INPUTS)
