"""2-link planar robot balancing about the upright equilibrium (the user's screenshot, eq. 3),
4 states, 2 inputs.

    M(theta) theta'' + C(theta, theta') theta' + tau(theta) = B u
    M   = [a_ij cos(theta_j - theta_i)],   C = [-a_ij theta_j' sin(theta_j - theta_i)],
    tau = [-b_i sin(theta_i)],
    a_ii = I_i + m_i lc_i^2 + l_i^2 sum_(k>i) m_k,
    a_ij = a_ji = m_j l_i lc_j + l_i l_j sum_(k>j) m_k    (i < j),
    b_i  = (m_i lc_i + l_i sum_(k>i) m_k) g.

State order as in the source: x = [theta_1..theta_2, theta_1'..theta_2'], so

    f = [theta' ; -M^-1 (C theta' + tau)],     G = [0 ; M^-1 B].

With u = 0 the upright position is an equilibrium (tau(0) = 0, C(0, 0) = 0) and it is unstable -- that is
the balancing problem. The paper's own neural controllers and neural V are NOT used: u comes from Sontag's
formula with the V this pipeline finds (standing rule).

LINK PARAMETERS (documented assumptions, 2026-09-20): the source gives no numbers, so every link is a
uniform rod of mass 1 kg and length 1 m: lc_i = l_i / 2, I_i = m_i l_i^2 / 12, g = 9.81. B = identity (one
torque per joint; the source writes B = [1, ..., 1]^T with u in R^n, which only types as the identity).
The box contains the source's domain ||x||_2 <= 0.5.
"""
import numpy as np
import sympy as sp

NAME = "link2"
N_LINKS = 2
N_STATES = 4
N_INPUTS = 2

G_ACC = 9.81
MASS = tuple([1.0] * N_LINKS)
LENGTH = tuple([1.0] * N_LINKS)
LC = tuple(0.5 * length for length in LENGTH)
MOMENT = tuple(mass * length ** 2 / 12.0 for mass, length in zip(MASS, LENGTH))

BOX_HALF_WIDTHS = tuple([0.5] * N_STATES)
STATE_NAMES = tuple([f"theta{i + 1}" for i in range(N_LINKS)] + [f"dtheta{i + 1}" for i in range(N_LINKS)])


def _a(i, j):
    """a_ij of the source (0-based indices)."""
    if i == j:
        return MOMENT[i] + MASS[i] * LC[i] ** 2 + LENGTH[i] ** 2 * sum(MASS[k] for k in range(i + 1, N_LINKS))
    lo, hi = min(i, j), max(i, j)
    return MASS[hi] * LENGTH[lo] * LC[hi] + LENGTH[lo] * LENGTH[hi] * sum(MASS[k] for k in range(hi + 1, N_LINKS))


def _b(i):
    """b_i of the source."""
    return (MASS[i] * LC[i] + LENGTH[i] * sum(MASS[k] for k in range(i + 1, N_LINKS))) * G_ACC


def _terms(x):
    theta = list(x[:N_LINKS])
    dtheta = list(x[N_LINKS:])
    M = sp.Matrix(N_LINKS, N_LINKS, lambda i, j: sp.Float(_a(i, j)) * sp.cos(theta[j] - theta[i]))
    C = sp.Matrix(N_LINKS, N_LINKS, lambda i, j: -sp.Float(_a(i, j)) * dtheta[j] * sp.sin(theta[j] - theta[i]))
    tau = sp.Matrix([-sp.Float(_b(i)) * sp.sin(theta[i]) for i in range(N_LINKS)])
    return theta, dtheta, M, C, tau


def fSR(*x):
    """Drift f(x) (sympy, 4 x 1): the dynamics with u = 0."""
    _theta, dtheta, M, C, tau = _terms(x)
    acc = M.inv() * (-C * sp.Matrix(dtheta) - tau)
    return sp.Matrix(list(dtheta) + list(acc))


def GSR(*x):
    """Input matrix G(x) (sympy, 4 x 2): the joint torques through M^-1."""
    _theta, _dtheta, M, _C, _tau = _terms(x)
    Minv = M.inv()
    G = sp.zeros(N_STATES, N_INPUTS)
    for i in range(N_LINKS):
        for j in range(N_LINKS):
            G[N_LINKS + i, j] = Minv[i, j]
    return G


def QSR(*x):
    return sp.eye(N_STATES)


def RSR(*x):
    return sp.Float(1e-4) * sp.eye(N_INPUTS)


Q_MATRIX = np.eye(N_STATES)
R_MATRIX = 1e-4 * np.eye(N_INPUTS)
