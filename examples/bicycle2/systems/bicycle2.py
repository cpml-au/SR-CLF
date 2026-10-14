"""Wheeled vehicle path following, kinematic bicycle in path coordinates,
2 states, 1 input. The target path is the unit circle, so the curvature is constant, kappa(s) = 1, and the
path position s drops out of the error dynamics:

    x1 = d_e  (distance error)      x2 = theta_e (angle error)
    d_e'     = v sin(theta_e)
    theta_e' = v tan(u) / L - v kappa cos(theta_e) / (1 - d_e kappa)

INPUT SUBSTITUTION AND NOMINAL STEERING (assumptions made here):
 * the steering angle enters through tan(u), which is not affine, so the input of the control-affine
   system is w = tan(u)/L (a bijection for |u| < pi/2); the steering angle is u = arctan(L w);
 * with w = 0 the origin is not an equilibrium on a curved path (theta_e' = -v kappa): the input is
   measured from the nominal steering of the circle, w = w' + kappa, so w' = 0 follows the circle.
Then f(0) = 0 and

    f = [v sin(x2), v kappa (1 - cos(x2)/(1 - x1 kappa))],   G = [[0], [v]].

v = 1 and L = 1 are choices (the source gives no values); the source's own controller
u = -0.8471 d_e - 1.6414 theta_e and its neural V are NOT used.
"""
import numpy as np
import sympy as sp

NAME = "bicycle2"
N_STATES = 2
N_INPUTS = 1

SPEED = 1.0
WHEELBASE = 1.0
CURVATURE = 1.0   # unit circle

# The source's domain is the ball ||x||_2 <= 0.8; the box that contains it (1 - d_e >= 0.2 > 0).
BOX_HALF_WIDTHS = (0.8, 0.8)
STATE_NAMES = ("d_e", "theta_e")


def fSR(*x):
    """Drift f(x) (sympy, 2x1) with the nominal steering of the circle applied (w' = 0)."""
    de, te = x
    v, k = sp.Float(SPEED), sp.Float(CURVATURE)
    return sp.Matrix([
        v * sp.sin(te),
        v * k * (sp.Integer(1) - sp.cos(te) / (sp.Integer(1) - de * k)),
    ])


def GSR(*x):
    """Input matrix G(x) (sympy, 2x1) for w' = tan(u)/L - kappa."""
    G = sp.zeros(N_STATES, N_INPUTS)
    G[1, 0] = sp.Float(SPEED)
    return G


def QSR(*x):
    return sp.eye(N_STATES)


def RSR(*x):
    return sp.Float(1e-4) * sp.eye(N_INPUTS)


Q_MATRIX = np.eye(N_STATES)
R_MATRIX = 1e-4 * np.eye(N_INPUTS)
