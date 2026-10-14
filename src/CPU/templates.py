"""The quadratic templates: V(x) = sum_{i<=j} a_ij x_i x_j with `a` placeholders.

full  : every monomial of the quadratic form, n(n+1)/2 coefficients
        (10 coefficients for n = 4, 21 for n = 6).
are   : only the monomials that are non-zero in this system's Riccati (ARE) solution
        (9 for quad6d: x1^2, x1x2, x2^2, x3^2, x3x4, x4^2, x5^2, x5x6, x6^2).
The coefficients are NOT started from the Riccati solution: x0 defaults to all ones.  The Riccati
solution (are_solution) only gives the 'are' pattern and a reference quadratic for reports.

Tolerances: |P_ij| > 1e-12 marks a monomial of the Riccati solution as present.
Settings read here: system.template (SYMCLF_CPU_TEMPLATE: full | are), system.template_x0
(SYMCLF_CPU_TEMPLATE_X0).
"""
from __future__ import annotations

import numpy as np

from src.CPU import config


def monomials(n, kind="full", are_pattern=None):
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    if kind == "full":
        return pairs
    if kind == "are":
        if are_pattern is None:
            raise ValueError("kind='are' needs the ARE sparsity pattern")
        return [(i, j) for (i, j) in pairs if are_pattern[i, j]]
    raise ValueError(f"unknown template kind {kind}")


def template_expression(n, kind="full", are_pattern=None):
    """The DEAP prefix string add(add(...), mul(a, mul(xi, xj))) in the 4-D seed's own order."""
    terms = [f"mul(a, mul(x{i + 1}, x{j + 1}))" for (i, j) in monomials(n, kind, are_pattern)]
    expr = terms[0]
    for term in terms[1:]:
        expr = f"add({expr}, {term})"
    return expr


def quadratic_matrix(n, coefficients, kind="full", are_pattern=None):
    """P with V(x) = x' P x for the template's coefficient vector."""
    P = np.zeros((n, n))
    for c, (i, j) in zip(np.asarray(coefficients, float).reshape(-1),
                         monomials(n, kind, are_pattern)):
        if i == j:
            P[i, i] = c
        else:
            P[i, j] += 0.5 * c
            P[j, i] += 0.5 * c
    return P


def are_solution():
    """P of the ARE for the system's linearisation (Q, R from the system module).

    Used for the 'are' template pattern and as a REFERENCE quadratic in reports -- never as a
    starting point for the optimiser."""
    import sympy as sp
    from scipy.linalg import solve_continuous_are

    sys_mod = config.system()
    n, m = sys_mod.N_STATES, sys_mod.N_INPUTS
    if m == 0:
        raise ValueError("the ARE needs an input: for an autonomous system (N_INPUTS = 0) use the 'full' template")
    xs = sp.symbols(f"x1:{n + 1}")
    zero = {s: 0 for s in xs}
    A = np.array(sp.Matrix(sys_mod.fSR(*xs)).jacobian(list(xs)).subs(zero), dtype=float)
    B = np.array(sp.Matrix(sys_mod.GSR(*xs)).subs(zero), dtype=float).reshape(n, m)
    return solve_continuous_are(A, B, np.asarray(sys_mod.Q_MATRIX, float),
                                np.asarray(sys_mod.R_MATRIX, float)), A, B


def default_template():
    """(expression, x0, kind) from the environment: SYMCLF_CPU_TEMPLATE = full | are."""
    sys_mod = config.system()
    n = sys_mod.N_STATES
    kind = config.TEMPLATE_KIND
    pattern = None
    if kind == "are":
        P = are_solution()[0]
        pattern = np.abs(P) > 1e-12
    expr = template_expression(n, kind, pattern)
    x0 = np.full(len(monomials(n, kind, pattern)), config.TEMPLATE_X0)
    return expr, x0, kind
