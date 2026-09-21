"""The DEFINITE CLF conditions and their predicates -- one definition, shared by every check
in src/srcCPU and src/srcGPU.  Ported from examples/VerifierBenchmark/conditions.py; the 4-D
cart-pole box is NOT baked in here, every check takes ``bounds`` explicitly.

CONDITION 1 -- ARTSTEIN, DEFINITE

    for all x in box, ||x|| > ORIGIN_TOL :   b(x) = 0  =>  a(x) < -MARGIN_TOL

    a = grad(V).f(x),   b = grad(V).G(x)[:, input_index],   margin = a + gamma1
    VIOLATION iff margin > -MARGIN_TOL           VALID iff margin_max <= -MARGIN_TOL
    "on the manifold" means |b| <= B_TOL.  a == 0 IS a violation (0 > -1e-9): the historical
    predicate margin > 0 was semidefinite and let a V that merely fails to increase pass.

CONDITION 2 -- POSITIVE DEFINITENESS, DEFINITE

    for all x in box, ||x|| > ORIGIN_TOL :   W(x) = V(x) - V(0) > PD_EPS * ||x||^2
    VALID iff min(W - PD_EPS*||x||^2) > PD_TOL     (a minimum that touches zero fails)

CONDITION 3 -- DET3, the EXPONENTIAL bounded-input Artstein condition (kappa = 0 is DET2)

    m(x) = a(x) - rho*|b(x)| + kappa*W(x) + gamma1        over the punctured box
    strict     VALID iff max m <= -MARGIN_TOL        (the definite predicate, artstein_valid)
    non-strict VALID iff max m <= 0                  (artstein_valid_nonstrict)
    The barrier equals a on b = 0, so barrier holds => Artstein holds, never the reverse;
    rho -> inf recovers Artstein and kappa > 0 additionally asks for exponential decay of W.

A CLF here means conditions 1 AND 2 (or 3 AND 2).  Properness, c*, bloat and closed-loop
convergence are deliberately not part of these checks.
"""

from __future__ import annotations

import numpy as np

#: Points closer to the origin than this are excluded (a and b both vanish at 0).
ORIGIN_TOL = 1.1e-3

#: Definiteness margin: VALID requires margin_max <= -MARGIN_TOL.
MARGIN_TOL = 1.0e-9

#: |b| <= B_TOL counts as "on the manifold" (polish acceptance tolerance).
B_TOL = 1.0e-10

#: Additive constant in margin = a + gamma1.
DEFAULT_GAMMA1 = 0.0

#: W must dominate PD_EPS * ||x||^2; VALID requires pd_min_margin > PD_TOL.
PD_EPS = 1.0e-4
PD_TOL = 1.0e-9

#: DET2/DET3 sweep defaults (the benchmark's).
DEFAULT_RHOS = (50.0, 500.0, 1000.0)
DEFAULT_KAPPAS = (0.001, 0.01, 0.1, 1.0)


def artstein_violation(margin, margin_tol=MARGIN_TOL):
    """True when this margin counts as an Artstein violation (definite)."""
    margin = float(margin)
    if not np.isfinite(margin):
        return False
    return margin > -margin_tol


def artstein_valid(margin_max, margin_tol=MARGIN_TOL):
    """True when the worst margin over the whole manifold clears the band."""
    margin_max = float(margin_max)
    if not np.isfinite(margin_max):
        return False  # nothing found is not a proof
    return margin_max <= -margin_tol


def nonstrict_valid(margin_max):
    """The semidefinite / non-strict reading of the SAME number: max m <= 0."""
    margin_max = float(margin_max)
    return bool(np.isfinite(margin_max) and margin_max <= 0.0)


def pd_valid(pd_min_margin, pd_tol=PD_TOL):
    """True when W - PD_EPS*||x||^2 stays strictly positive (definite)."""
    pd_min_margin = float(pd_min_margin)
    if not np.isfinite(pd_min_margin):
        return False
    return pd_min_margin > pd_tol


def overall_valid(margin_max, pd_min_margin, margin_tol=MARGIN_TOL, pd_tol=PD_TOL):
    """A candidate is a CLF iff BOTH definite conditions hold."""
    return artstein_valid(margin_max, margin_tol) and pd_valid(pd_min_margin, pd_tol)


def describe(bounds=None):
    """One block of text stating exactly what is checked."""
    box = (
        ""
        if bounds is None
        else f"  box                {list(map(tuple, np.asarray(bounds, float)))}\n"
    )
    return f"""CONDITIONS CHECKED (DEFINITE)
{box}  origin excluded    ||x|| > {ORIGIN_TOL:g}

  1. Artstein    b(x) = 0  =>  a(x) < -{MARGIN_TOL:g}
       violation iff  margin > -{MARGIN_TOL:g}        (margin = a + gamma1)
       valid     iff  margin_max <= -{MARGIN_TOL:g}
       manifold   |b| <= {B_TOL:g}
  2. Positive definiteness    V(x) - V(0) > {PD_EPS:g} * ||x||^2
       valid     iff  min(W - {PD_EPS:g}*||x||^2) > {PD_TOL:g}
  3. DET3        m = a - rho*|b| + kappa*(V - V(0)) + gamma1
       strict valid iff max m <= -{MARGIN_TOL:g};  non-strict valid iff max m <= 0
       rho in {DEFAULT_RHOS}, kappa in {DEFAULT_KAPPAS}; kappa = 0 is DET2.
"""
