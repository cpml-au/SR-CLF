"""Minimal CPU base for the 4-D cart-pole (4DCartPolerV6 = 3DCartPolerV6 extended back to 4-D).

The 4-D examples import examples/4DCartPoler/Evaluate.py (1277 lines) as their
base. Almost all of it is 4-D-specific verification machinery the GPU shim
overrides anyway; auditing what the shim and src4DCartPoleV10 actually read from
the base leaves this short list. Writing it out is safer than porting the 4-D
file, because every constant here is deliberate rather than inherited.

GP variable mapping:  x1 = cart position, x2 = cart velocity, x3 = pole angle, x4 = pole rate
(examples/4DCartPoler/SystemDynamics.py).
"""
import os

import numpy as np

from src.SymFunctions import (
    DeapSimplifier,
    contains_symbol,
    detect_nested_function_calls,
    substitute_paramsCoef,
)
from SystemDynamics import f, G, Q, R                       # noqa: F401
from SystemDynamicsSR import fSR, GSR, QSR, RSR             # noqa: F401

N_STATES = 4

# --- box and CLF condition -------------------------------------------------
# V2: THE BOX IS MATCHED TO THE ARE FORM, NOT UNIFORM.
#
# The certified sublevel set of ANY quadratic inscribed in a box is thin by a
# factor sqrt(cond(P)) -- for the 3-D ARE form cond(P) = 682.4, sqrt = 26.1,
# and the measured ROA anisotropy of the ARE quadratic itself is 25.3. The
# "needle" is that number. The CORRECT answer has it too (its r_in is 0.0117
# against the gen-89 champion's 0.0118), so it is geometry, not a search
# pathology.
#
# Box SIZE is a no-op: +-0.25 and +-1.00 give bit-identical r_in, anisotropy
# and certified fraction. Box SHAPE is everything. Setting the half-widths
# proportional to 1/sqrt(diag(P_ARE)) gives, for the same V:
#
#     uniform +-0.25                      r_in 0.0424  aniso 26.1  certified 1.151%
#     matched [0.25, 0.0494, 0.1155]      r_in 0.1002  aniso 12.2  certified 4.839%
#
# 4.2x the certified volume and 2.4x the inscribed radius, for free, with no
# change to the fitness at all. It also makes the Cartesian lattice far less
# badly matched to the geometry -- under the uniform box only 17 of 9261 grid
# points landed inside the certified set.
#
# Physically this is just honest units: the three axes are cart velocity
# (m/s), pole angle (rad) and pole rate (rad/s). A uniform +-0.25 across three
# different physical units is arbitrary, and it is that choice which produced
# cond(P) = 682 in the first place.
# 4DCartPolerV6: the cart position x1 gets GPU5_7's +-0.25 (examples/4DCartPoler); the three
# shared states keep 3DCartPolerV6's matched half-widths.
_MATCHED = (0.25, 0.25, 0.0494, 0.1155)
BOX_HALF_WIDTHS = tuple(
    float(v) for v in os.environ.get(
        "SYMCLF_V2_BOX", ",".join(str(h) for h in _MATCHED)
    ).split(",")
)
if len(BOX_HALF_WIDTHS) != N_STATES:
    raise ValueError(
        f"SYMCLF_V2_BOX needs {N_STATES} comma-separated half-widths, "
        f"got {BOX_HALF_WIDTHS}"
    )
SHGO_BOUNDS = [(-h, h) for h in BOX_HALF_WIDTHS]
SHGO_DECAY_RATE = 0.0012
CLF_GAMMA1 = 0.0
# 2026-09-09: NO origin exclusion for the conditions. DET4's two numbers are RATIOS,
# c_v* = min W/|x|^2 and kappa* = min -(a - rho|b|)/W, which converge to the linearisation at
# x -> 0 (measured identical to 7 digits from r = 1.1e-3 down to r = 0); only x = 0 itself
# (a = b = W = 0) is excluded.  See README.md.
CLF_ORIGIN_EXCLUDE_RADIUS = 0.0
# The root-bisection Artstein check (the verdict) searches an ABSOLUTE margin a < -1e-9 at b = 0,
# which vanishes at the origin: without a ball every valid V "drains" to 0 and reports a
# violation.  It keeps a small NUMERICAL ball; it is not part of the condition being certified.
ABSOLUTE_CHECK_BALL_RADIUS = 1.1e-3

# --- manifold penalties (identical to the 4-D file) ------------------------
MANIFOLD_MARGIN_SAFETY = 1e-4
MANIFOLD_MARGIN_M0 = 0.01
MANIFOLD_EXACT_MARGIN_WEIGHT = 12000.0
MANIFOLD_EXACT_MARGIN_PENALTY_MAX = 8000.0
EXACT_NEAR0_SHIFT = 1e-7
MANIFOLD_VACUOUS_PENALTY = 100.0
MANIFOLD_EXACT_MSE_GATE = 100000000000000
MANIFOLD_EXACT_PENALTY_MAX = 100000.0
MANIFOLD_MARGIN_TOL_EXACT = 1e-12          # the shim negates this (strict test)
MANIFOLD_CHECK_ENABLED = False
MANIFOLD_EXACT_CHECK_ENABLED = True

# --- certified region ------------------------------------------------------
C_STAR_TARGET = 0.02
C_STAR_PENALTY_MAX = 500.0
ROA_COVERAGE_WEIGHT = 1.0

# --- origin needle probe (penalty disarmed by the shim; ratio still logged) -
ORIGIN_PROBE_ENABLED = True
ORIGIN_PROBE_K = 5e4
ORIGIN_PROBE_OFFSETS = (1e-4, 1e-2)
ORIGIN_PROBE_PENALTY = 100000.0

# --- gradient-magnitude targets (re-derived from the ARE form in the shim) --
V_GRAD_X1X2_TARGET = 1.0
V_GRAD_X3X4_TARGET = 1.0
V_GRAD_WEIGHT = 0.1
PROPERNESS_PENALTY_WEIGHT = 0.0

# --- rollout starts: (position, velocity, angle, rate) ---------------------
# 4DCartPolerV6: 3DCartPolerV6's four starts with cart position 0 (rollouts are off in det4
# mode; the grid kernel only evaluates V at these points).
ROLLOUT_ENABLED = False
_H = BOX_HALF_WIDTHS
ROLLOUT_X0S = (
    (0.0, 0.00 * _H[1], 0.04 * _H[2], 0.00 * _H[3]),
    (0.0, 0.20 * _H[1], 0.20 * _H[2], 0.00 * _H[3]),
    (0.0, -0.20 * _H[1], -0.20 * _H[2], 0.20 * _H[3]),
    (0.0, 0.32 * _H[1], 0.32 * _H[2], -0.20 * _H[3]),
)

# --- verifiers the GPU path does not use -----------------------------------
IBEX_VERIFIER_ENABLED = False
IBEX_ARTSTEIN_ENABLED = False
PYGMO_ARTSTEIN_ENABLED = False
CODAC_ROOT_ENABLED = False
NUMERICAL_VERIFIER = None


def _reference_quadratic(X1, X2, X3, X4):
    """Placeholder; the shim replaces this with the 4-D ARE solution."""
    return X1**2 + X2**2 + X3**2 + X4**2


def _sympy_expression(ind2MSE, consts):
    return DeapSimplifier(substitute_paramsCoef(str(ind2MSE), consts),
                          should_print=False)


def _manifold_margin_penalty(margin_max, weight, cap):
    """m = margin + SAFETY; W*m^2/(m+M0) capped. Zero when margin <= -SAFETY."""
    if not np.isfinite(margin_max):
        return 0.0
    m = margin_max + MANIFOLD_MARGIN_SAFETY
    if m <= 0.0:
        return 0.0
    return float(min(weight * m * m / (m + MANIFOLD_MARGIN_M0), cap))


def _exact_near0_penalty(margin_max):
    """Concave finish-line terms; the shim overrides the coefficients."""
    if not np.isfinite(margin_max):
        return 0.0
    m = max(margin_max, 0.0)
    ms = max(margin_max + EXACT_NEAR0_SHIFT, 0.0)
    return float(4.0 * np.sqrt(m) + 64.0 * m**0.25 + 256.0 * ms ** (1.0 / 6.0))


def _symbolic_structure_penalty(ind2MSE, consts):
    """1e6 per nested exp/aq and per state the expression ignores.

    Four states here (as in examples/4DCartPoler) -- an expression that never mentions x4 is
    as degenerate as one that never mentions x1.
    """
    penalty = 0.0
    text = str(ind2MSE)
    if detect_nested_function_calls(text, "exp"):
        penalty += 1e6
    if detect_nested_function_calls(text, "aq"):
        penalty += 1e6
    for name in ("x1", "x2", "x3", "x4"):
        if not contains_symbol(text, name):
            penalty += 1e6
    return penalty
