"""GPU5_7 Evaluate shim: the certified-region objective.

Loads the CPU base Evaluate (examples/4DCartPoler), swaps the exact manifold
check for the srcGPU5_7 GPU/SLSQP implementation, and configures the GPU5_7
fitness. Everything srcGPU5_7 needs is inside srcGPU5_7 -- no other srcGPU
package is imported anywhere in this run.

WHAT IS DIFFERENT FROM GPU5_6 (the decisive measurement, 2026-08-27):
run 122340's generation-7 champion was "certified" -- PD passes, exact
Artstein violations only above W = 0.0316, certified volume 0.328% -- and
DIVERGES in closed loop even from initial conditions inside its own
certified set (needs |u| up to 1.46e5 on the box, ~1e11 along trajectories).
Run 122304's champion is Artstein-INVALID on paper (57 roots) and converges
from every tested initial condition needing only |u| <= 59. The conditions
the fitness now scores are the ones that separate these two:

  * SATURATION:  u_required = max a/|b| on the grid, charged per decade
    above GPU5_7_SAT_U_TARGET.  (96051: 57, 122304: 59, 122340 gen-7: 1.5e5)
  * ROA:         certified level c_max and certified volume fraction --
    the actual objective, replacing the properness band / c_star target /
    coverage priors that were measured ANTI-correlated with it.
  * CERT DEPTH:  exact-stage violations priced by their W-placement.
  * ROLLOUT:     candidates that pass all static checks are integrated
    closed-loop with the saturated Sontag controller; divergence charged.

Priors switched OFF by default (knobs remain): properness band count,
V_GRAD magnitude targets, c_star target penalty, ROA coverage count.
"""

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_CPU_EXAMPLE = os.path.join(_ROOT, "examples", "4DCartPoler")
for _path in (_ROOT, _CPU_EXAMPLE):
    if _path not in sys.path:
        sys.path.append(_path)

from srcGPU5_7.b_manifold_check_gpu3 import (  # noqa: E402
    check_b_manifold_exact_gpu3_cached as check_b_manifold_exact_gpu_cached,
)
from srcGPU5_7.grid_fitness import make_gpu_evaluators  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_Evaluate_gpu2_cpu_base", os.path.join(_CPU_EXAMPLE, "Evaluate.py")
)
_base = importlib.util.module_from_spec(_spec)
sys.modules["_Evaluate_gpu2_cpu_base"] = _base
_spec.loader.exec_module(_base)

# The exact-manifold gate and penalty constants come from the CPU file.
_base.IBEX_VERIFIER_ENABLED = False
_base.IBEX_ARTSTEIN_ENABLED = False
_base.PYGMO_ARTSTEIN_ENABLED = False
_base.CODAC_ROOT_ENABLED = False
_base.NUMERICAL_VERIFIER = None
_base.MANIFOLD_CHECK_ENABLED = False
_base.MANIFOLD_EXACT_CHECK_ENABLED = True
_base.ROLLOUT_ENABLED = False

# --- mandatory cheap first stage (unchanged from GPU5_6) --------------------
_base.GPU2_CHEAP_ENABLED = (
    os.environ.get("SYMCLF_GPU2_CHEAP_ENABLED", "1") == "1"
)
_base.GPU2_CHEAP_FUSION = int(
    os.environ.get("SYMCLF_GPU2_CHEAP_FUSION", "32")
)
_base.GPU2_CHEAP_LINES_PER_AXIS = int(
    os.environ.get("SYMCLF_GPU2_CHEAP_LINES_PER_AXIS", "224")
)
_base.GPU2_CHEAP_LINE_SAMPLES = int(
    os.environ.get("SYMCLF_GPU2_CHEAP_LINE_SAMPLES", "225")
)
_base.GPU2_CHEAP_OBLIQUE_LINES = int(
    os.environ.get("SYMCLF_GPU2_CHEAP_OBLIQUE_LINES", "112")
)
_base.GPU2_CHEAP_MAX_BRACKETS = int(
    os.environ.get("SYMCLF_GPU2_CHEAP_MAX_BRACKETS", "6144")
)
_base.GPU2_CHEAP_BISECTION_ITERATIONS = int(
    os.environ.get("SYMCLF_GPU2_CHEAP_BISECTION_ITERATIONS", "30")
)
_base.GPU2_CHEAP_B_TOL = float(
    os.environ.get("SYMCLF_GPU2_CHEAP_B_TOL", "1e-8")
)
_base.GPU2_CHEAP_REPORT_POINTS = int(
    os.environ.get("SYMCLF_GPU2_CHEAP_REPORT_POINTS", "64")
)
# GPU5_5: 15 -> 40 seconds, and an overrun warns instead of raising (a 0.133s
# overrun on the 15s ceiling killed run 122300 after 12h19m).
_base.GPU2_CHEAP_BUDGET_SECONDS = float(
    os.environ.get("SYMCLF_GPU2_CHEAP_BUDGET_SECONDS", "40")
)
# GPU5_3: widened gate, so the population's whole frontier band is inside the
# exactly-checked region.
_base.GPU2_EXACT_A_MAX_GATE = float(
    os.environ.get("SYMCLF_GPU2_EXACT_A_MAX_GATE", "0.15")
)
_base.GPU2_EXACT_GATE_PENALTY_RATIO = float(
    os.environ.get("SYMCLF_GPU2_EXACT_GATE_PENALTY_RATIO", "1.1")
)
if _base.GPU2_CHEAP_BUDGET_SECONDS <= 0.0:
    raise ValueError("SYMCLF_GPU2_CHEAP_BUDGET_SECONDS must be positive")
if _base.GPU2_EXACT_A_MAX_GATE < 0.0:
    raise ValueError("SYMCLF_GPU2_EXACT_A_MAX_GATE must be nonnegative")
# GPU5_3 gate pricing: rejection is priced off the candidate's own cheap
# result at ESCALATION x, floored at the flat fee and capped at the price of
# the only other dodge (failing gate 1 on purpose).
_base.GPU5_7_GATE_PRICE_ESCALATION = float(
    os.environ.get("SYMCLF_GPU5_7_GATE_PRICE_ESCALATION", "2.0")
)
if "SYMCLF_GPU5_7_GATE_PRICE_CAP" in os.environ:
    _base.GPU5_7_GATE_PRICE_CAP = float(
        os.environ["SYMCLF_GPU5_7_GATE_PRICE_CAP"]
    )
if _base.GPU5_7_GATE_PRICE_ESCALATION < 1.0:
    raise ValueError(
        "SYMCLF_GPU5_7_GATE_PRICE_ESCALATION must be >= 1.0: below 1.0 the "
        "cheap screen's own understatement of the margin makes refusing the "
        "exact check cheaper than taking it"
    )
if _base.GPU2_EXACT_GATE_PENALTY_RATIO < 1.0:
    raise ValueError(
        "SYMCLF_GPU2_EXACT_GATE_PENALTY_RATIO must be >= 1.0: below 1.0 it is "
        "a discount for not being checked"
    )

# Retained only for standalone experiments with the heavier direct Artstein
# implementation.
_base.GPU2_ARTSTEIN_ENABLED = (
    os.environ.get("SYMCLF_GPU2_ARTSTEIN_ENABLED", "1") == "1"
)
_base.GPU2_ARTSTEIN_FUSION = int(
    os.environ.get("SYMCLF_GPU2_ARTSTEIN_FUSION", "4")
)
_base.GPU2_ARTSTEIN_LINES_PER_AXIS = int(
    os.environ.get("SYMCLF_GPU2_ARTSTEIN_LINES_PER_AXIS", "1024")
)
_base.GPU2_ARTSTEIN_LINE_SAMPLES = int(
    os.environ.get("SYMCLF_GPU2_ARTSTEIN_LINE_SAMPLES", "801")
)
_base.GPU2_ARTSTEIN_OBLIQUE_LINES = int(
    os.environ.get("SYMCLF_GPU2_ARTSTEIN_OBLIQUE_LINES", "512")
)
_base.GPU2_ARTSTEIN_MAX_BRACKETS = int(
    os.environ.get("SYMCLF_GPU2_ARTSTEIN_MAX_BRACKETS", "8192")
)
_base.GPU2_ARTSTEIN_BISECTION_ITERATIONS = int(
    os.environ.get("SYMCLF_GPU2_ARTSTEIN_BISECTION_ITERATIONS", "36")
)
_base.GPU2_ARTSTEIN_STARTS = int(
    os.environ.get("SYMCLF_GPU2_ARTSTEIN_STARTS", "256")
)
_base.GPU2_ARTSTEIN_ITERATIONS = int(
    os.environ.get("SYMCLF_GPU2_ARTSTEIN_ITERATIONS", "24")
)
_base.GPU2_ARTSTEIN_INITIAL_PROJECTION_STEPS = int(
    os.environ.get("SYMCLF_GPU2_ARTSTEIN_INITIAL_PROJECTION_STEPS", "6")
)
_base.GPU2_ARTSTEIN_PROJECTION_STEPS = int(
    os.environ.get("SYMCLF_GPU2_ARTSTEIN_PROJECTION_STEPS", "2")
)
_base.GPU2_ARTSTEIN_STEP_SIZE = float(
    os.environ.get("SYMCLF_GPU2_ARTSTEIN_STEP_SIZE", "0.05")
)
_base.GPU2_ARTSTEIN_B_TOL = float(
    os.environ.get("SYMCLF_GPU2_ARTSTEIN_B_TOL", "1e-8")
)
_base.GPU2_ARTSTEIN_REPORT_POINTS = int(
    os.environ.get("SYMCLF_GPU2_ARTSTEIN_REPORT_POINTS", "64")
)
# Exact-manifold polish: the reference SciPy SLSQP.
_base.GPU2_MANIFOLD_POLISH_TOP_K = int(
    os.environ.get("SYMCLF_GPU2_POLISH_TOP_K", "40")
)
_base.GPU2_MANIFOLD_POLISH_ITERATIONS = int(
    os.environ.get("SYMCLF_GPU2_POLISH_ITERATIONS", "60")
)
_base.GPU2_MANIFOLD_POLISH_B_TOL = float(
    os.environ.get("SYMCLF_GPU2_POLISH_B_TOL", "1e-8")
)
_base.GPU3_MESH_POINTS = int(os.environ.get("SYMCLF_GPU3_MESH_POINTS", "0"))

# --- GPU5 positive-definiteness check ---------------------------------------
_base.GPU5_PD_EPS = float(os.environ.get("SYMCLF_GPU5_PD_EPS", "1e-4"))
_base.GPU5_PD_POLISH_TOP_K = int(os.environ.get("SYMCLF_GPU5_PD_TOP_K", "40"))
_base.GPU5_PD_POLISH_ITERATIONS = int(
    os.environ.get("SYMCLF_GPU5_PD_ITERATIONS", "60")
)
_base.GPU5_PD_UNKNOWN_PENALTY = float(
    os.environ.get("SYMCLF_GPU5_PD_UNKNOWN_PENALTY", "8000")
)

# --- GPU5_2/5_3 measure terms (unchanged shapes) ----------------------------
_base.GPU5_7_VIOLATION_WEIGHT = float(
    os.environ.get("SYMCLF_GPU5_7_VIOLATION_WEIGHT", "300")
)
_base.GPU5_7_MEAN_MARGIN_SCALE = float(
    os.environ.get("SYMCLF_GPU5_7_MEAN_MARGIN_SCALE", "0.1")
)
_base.GPU5_7_GATE_MARGIN_ESCALATION = float(
    os.environ.get("SYMCLF_GPU5_7_GATE_MARGIN_ESCALATION", "2.0")
)
_base.GPU5_7_MEAN_MARGIN_WEIGHT = float(
    os.environ.get(
        "SYMCLF_GPU5_7_MEAN_MARGIN_WEIGHT",
        str(_base.MANIFOLD_EXACT_MARGIN_WEIGHT),
    )
)
_base.GPU5_7_MEAN_MARGIN_PENALTY_MAX = float(
    os.environ.get(
        "SYMCLF_GPU5_7_MEAN_MARGIN_PENALTY_MAX",
        str(_base.MANIFOLD_EXACT_MARGIN_PENALTY_MAX),
    )
)

import numpy as _np  # noqa: E402


def _reference_matrix(reference_fn, n=4):
    """Recover P from a quadratic form ref(x) = x^T P x by evaluation."""
    basis = _np.eye(n)
    matrix = _np.zeros((n, n))
    for i in range(n):
        matrix[i, i] = float(reference_fn(*basis[i]))
    for i in range(n):
        for j in range(i + 1, n):
            both = float(reference_fn(*(basis[i] + basis[j])))
            off = 0.5 * (both - matrix[i, i] - matrix[j, j])
            matrix[i, j] = matrix[j, i] = off
    return matrix


# --- GPU5_7: PRIORS OFF, OBJECTIVE ON ---------------------------------------
# Properness band count: OFF. It is a prior about V's shape relative to the
# reference quadratic. Measured on run 122304 it was 61% of the champion's
# fitness while the champion's actual defect (violations parked at W ~ 0) was
# uncharged. The realities it guarded against are covered by the search-based
# PD check (lower half) and the ROA volume term (upper half: an over-inflated
# V has a small certified volume because boundary_min stops scaling with it
# -- both scale together, so inflation buys nothing at all now).
_base.PROPERNESS_PENALTY_WEIGHT = float(
    os.environ.get("SYMCLF_GPU5_7_PROPERNESS_WEIGHT", "0")
)
# V_GRAD magnitude targets: OFF. An unnormalised absolute-error prior; the
# saturation and ROA terms are scale-invariant so the exploit it patched
# (inflating ||grad V||) no longer pays.
_base.V_GRAD_WEIGHT = float(
    os.environ.get("SYMCLF_GPU5_7_V_GRAD_WEIGHT", "0")
)
# c_star-vs-reference target: OFF (weights 0). Replaced by the ROA term,
# which prices the same quantity (boundary_min enters c_max) against the
# candidate's own violations rather than against the reference form.
_base.C_STAR_TARGET_RATIO = float(
    os.environ.get("SYMCLF_GPU5_7_C_STAR_TARGET_RATIO", "1.0")
)
_base.C_STAR_TARGET = float(
    os.environ.get("SYMCLF_GPU4_1_C_STAR_TARGET", "0.02")
)
_base.C_STAR_PENALTY_MAX = float(
    os.environ.get("SYMCLF_GPU5_7_C_STAR_PENALTY_MAX", "0")
)
_base.C_STAR_PENALTY_CAP = float(
    os.environ.get("SYMCLF_GPU5_7_C_STAR_PENALTY_CAP", "0")
)
_base.C_STAR_TAIL_WEIGHT = float(
    os.environ.get("SYMCLF_GPU5_7_C_STAR_TAIL_WEIGHT", "0")
)
# Coverage count: OFF. The certified-volume fraction is its continuous,
# scale-invariant replacement.
_base.ROA_COVERAGE_WEIGHT = float(
    os.environ.get("SYMCLF_GPU5_7_COVERAGE_WEIGHT", "0")
)

# --- GPU5_7: the saturation term --------------------------------------------
# u_required = max over the grid of a/|b| where a > 0: the control magnitude
# a bounded actuator needs to force decrease. Charged per DECADE above the
# target. Calibration (21^4 grid, finite-difference gradients):
#     96051 (verified CLF)      57        122340 gen-7 (diverges)  1.46e5
#     122304 champ (converges)  59        -- and the gen-7 champion needs
#     |u| ~ 1e11 along its actual trajectories, i.e. un-integrable.
# TARGET 1000 gives every known-good candidate a 17x margin and charges the
# gen-7 family ~2.2 decades. The closed-loop rollout clips at this same value.
_base.GPU5_7_SAT_U_TARGET = float(
    os.environ.get("SYMCLF_GPU5_7_SAT_U_TARGET", "1000")
)
_base.GPU5_7_SAT_WEIGHT = float(
    os.environ.get("SYMCLF_GPU5_7_SAT_WEIGHT", "150")
)

# --- GPU5_7: the ROA (certified region) term --------------------------------
# c_max = min(boundary_min, min W over violating grid points); the certified
# volume is the fraction of the box with W < c_max. VOL_TARGET 0.005 (0.5% of
# the box) sits just under the best known certified volume (96051: 0.657%),
# so a genuinely certified candidate can pay ~0 while everything else feels a
# monotone pull toward larger certified sets. The negative side (c_max < 0,
# scaled by the candidate's own median |W|) supplies slope where no
# certificate exists at all.
_base.GPU5_7_ROA_WEIGHT = float(
    os.environ.get("SYMCLF_GPU5_7_ROA_WEIGHT", "500")
)
# Target on the SCALE-FREE ratio c_max / median|W| -- not on raw c_max, which
# scales with V and would let the search buy the term by inflating V; not on
# certified volume, which is quantised and hard-zeroed for c_max <= 0 (the
# 500-point cliff that stalled run 122350 at c_max = -4.8e-5). Measured:
# 96051 sits at 0.0221, 122232 at 0.0274, ARE reference at 0.0044.
_base.GPU5_7_ROA_C_TARGET = float(
    # GPU5_7 REVIEW FIX: 0.02 -> 0.004. 0.02 was calibrated on the STRONGEST
    # certificates in the panel (96051 0.0212, 122232 0.0204) and the ARE
    # reference -- a CLF valid BY CONSTRUCTION -- sits at 0.0042 and so ate
    # nearly the whole 500-point ramp, scoring 384.45. That let 122340's final
    # champion, which DIVERGES in closed loop, score 69.41 and beat it 5.5x.
    # Calibrate to the WEAKEST known-valid certificate, not the strongest --
    # the same error as anchoring the properness band on an invalid matrix.
    os.environ.get("SYMCLF_GPU5_7_ROA_C_TARGET", "0.004")
)
_base.GPU5_7_ROA_NEG_WEIGHT = float(
    os.environ.get("SYMCLF_GPU5_7_ROA_NEG_WEIGHT", "250")
)

# --- GPU5_7: certificate depth (exact stage) --------------------------------
# Violations found by the exact Artstein/PD searches are priced by their
# W-placement: weight * clip(1 - min_w/boundary_min, 0, 2). Both current
# champions park their violations at W ~ 0 -- certificate empty, margin
# penalty tiny. This is the term that makes that placement expensive.
_base.GPU5_7_CERT_WEIGHT = float(
    os.environ.get("SYMCLF_GPU5_7_CERT_WEIGHT", "300")
)

# --- GPU5_7: the closed-loop rollout gate -----------------------------------
# A candidate with zero violations, PD passing, is integrated with the
# SATURATED Sontag controller (|u| <= ROLLOUT_UMAX, default = SAT_U_TARGET)
# from the CPU example's standard initial conditions. DIVERGENCE is charged
# (500/trajectory); failure to fully converge is not, since standard starts
# may legitimately sit outside a small certified region. This is the exact
# test the 122340 gen-7 "certified" champion fails.
_base.GPU5_7_ROLLOUT_ENABLED = (
    os.environ.get("SYMCLF_GPU5_7_ROLLOUT_ENABLED", "1") == "1"
)
_base.GPU5_7_ROLLOUT_UMAX = float(
    os.environ.get("SYMCLF_GPU5_7_ROLLOUT_UMAX",
                   str(_base.GPU5_7_SAT_U_TARGET))
)
_base.GPU5_7_ROLLOUT_T = float(
    os.environ.get("SYMCLF_GPU5_7_ROLLOUT_T", "10.0")
)
_base.GPU5_7_ROLLOUT_DT = float(
    os.environ.get("SYMCLF_GPU5_7_ROLLOUT_DT", "2e-3")
)
_base.GPU5_7_ROLLOUT_DIVERGE_NORM = float(
    os.environ.get("SYMCLF_GPU5_7_ROLLOUT_DIVERGE_NORM", "1.0")
)
_base.GPU5_7_ROLLOUT_FAIL_PENALTY = float(
    os.environ.get("SYMCLF_GPU5_7_ROLLOUT_FAIL_PENALTY", "500")
)

# --- exact near-zero sharpener: coefficients cut 16x (GPU5_3) ---------------
# Kept: infinite slope at margin 0 so a tiny positive margin cannot be parked
# on, without the 256-coefficient constant floor. pd_result_penalty reads this
# SAME curve (GPU5_7 fix), so one override rescales Artstein and PD together.
_NEAR0_SQRT = float(os.environ.get("SYMCLF_GPU5_7_NEAR0_SQRT", "0.25"))
_NEAR0_QUARTIC = float(os.environ.get("SYMCLF_GPU5_7_NEAR0_QUARTIC", "4"))
_NEAR0_SIXTH = float(os.environ.get("SYMCLF_GPU5_7_NEAR0_SIXTH", "64"))


def _exact_near0_penalty_gpu5_7(margin_max):
    if not _np.isfinite(margin_max):
        return 0.0
    m = max(float(margin_max), 0.0)
    ms = max(float(margin_max) + _base.EXACT_NEAR0_SHIFT, 0.0)
    return float(
        _NEAR0_SQRT * _np.sqrt(m)
        + _NEAR0_QUARTIC * m ** 0.25
        + _NEAR0_SIXTH * ms ** (1.0 / 6.0)
    )


_base._exact_near0_penalty = _exact_near0_penalty_gpu5_7

# --- GPU5_4: normalised Artstein margin + grid flatness ---------------------
# a/||grad V|| is scale-invariant, so flattening V buys no margin. The grid
# flatness count charges RELATIVE flat spots (slope below a fraction of the
# mean slope). Both keep their GPU5_6 calibration: zero false positives on
# 96051/122232, 432 hits on the run-122299 flattening exploit.
_base.GPU5_7_NORMALIZE_MARGIN = (
    os.environ.get("SYMCLF_GPU5_7_NORMALIZE_MARGIN", "1") == "1"
)
_base.GPU5_7_GRAD_NORM_FLOOR = float(
    os.environ.get("SYMCLF_GPU5_7_GRAD_NORM_FLOOR", "1e-3")
)
_base.GPU5_7_FLAT_GRAD_FLOOR = float(
    os.environ.get("SYMCLF_GPU5_7_FLAT_GRAD_FLOOR", "1e-3")
)
_base.GPU5_7_FLAT_GRAD_WEIGHT = float(
    os.environ.get("SYMCLF_GPU5_7_FLAT_GRAD_WEIGHT", "1.0")
)

# --- GPU5_5: strict violation predicate -------------------------------------
# viol = margin > -1e-9 in every checker (one number: MANIFOLD_MARGIN_TOL_
# EXACT). A degenerate a == 0 root is a violation instead of certifying free.
# 96051's true margin is -1.13e-07 (113x below the tolerance) so real CLFs
# pass untouched.
_base.GPU5_7_STRICT_MARGIN_TOL = float(
    os.environ.get("SYMCLF_GPU5_7_STRICT_MARGIN_TOL", "1e-9")
)
if _base.GPU5_7_STRICT_MARGIN_TOL <= 0.0:
    raise ValueError(
        "SYMCLF_GPU5_7_STRICT_MARGIN_TOL must be > 0: it is negated into "
        "MANIFOLD_MARGIN_TOL_EXACT, and a non-positive value restores the "
        "non-strict test that lets a == 0 certify for free"
    )
_base.MANIFOLD_MARGIN_TOL_EXACT = -_base.GPU5_7_STRICT_MARGIN_TOL
print(
    "GPU5_7 strict margin predicate: violation iff margin > "
    f"{_base.MANIFOLD_MARGIN_TOL_EXACT:.1e}  (a == 0 is now a violation)",
    flush=True,
)

# --- GPU5_5: x1-axis terms (kept as slope, weight-reduced role) --------------
# The saturation term subsumes the axis VERDICT (a == 0 on the axis forces
# b != 0 there), but on the degenerate-slab family a = b = 0 hides from the
# ratio a/|b|, so these remain the only CONTINUOUS slope out of that basin.
# They charge known-good candidates 0.17-0.27 points total.
_base.GPU5_7_AXIS_B_TARGET = float(
    os.environ.get("SYMCLF_GPU5_7_AXIS_B_TARGET", "1e-3")
)
_base.GPU5_7_AXIS_B_WEIGHT = float(
    os.environ.get("SYMCLF_GPU5_7_AXIS_B_WEIGHT", "300")
)
_base.GPU5_7_AXIS_V_TARGET = float(
    os.environ.get("SYMCLF_GPU5_7_AXIS_V_TARGET", "1e-2")
)
_base.GPU5_7_AXIS_V_WEIGHT = float(
    os.environ.get("SYMCLF_GPU5_7_AXIS_V_WEIGHT", "300")
)
_base.GPU5_7_AXIS_TAIL_WEIGHT = float(
    os.environ.get("SYMCLF_GPU5_7_AXIS_TAIL_WEIGHT", "0.5")
)
# V(0) is gauge; properness (when enabled) and PD measure W = V - V(0).
_base.GPU5_7_ORIGIN_PENALTY_WEIGHT = float(
    os.environ.get("SYMCLF_GPU5_7_ORIGIN_PENALTY_WEIGHT", "0")
)

# --- GPU5_6: the reference form is the ACTUAL ARE solution -------------------
# examples/4DCartPoler's hardcoded quadratic is NOT the LQR P for this system
# (off by up to 24%, fails Artstein with +0.412 on the linearisation and 209
# violating roots on the box). These coefficients are the true solution of
# the Algebraic Riccati Equation with the project's own Q = I, R = 1e-4
# (ARE residual 2.1e-11); the guard below re-verifies it at import.
_ARE_REFERENCE = (
    1.96536550864235,    # x1^2
    2.86256158256099,    # x1 x2
    -13.36889758391940,  # x1 x3
    -5.92512316512200,   # x1 x4
    2.14394502116904,    # x2^2
    -20.34964703488520,  # x2 x3
    -8.97085318640464,   # x2 x4
    50.81896563027710,   # x3^2
    43.68185565233150,   # x3 x4
    9.63929806560065,    # x4^2
)


def _reference_quadratic_are(X1, X2, X3, X4):
    c = _ARE_REFERENCE
    return (
        c[0] * X1**2 + c[1] * X1 * X2 + c[2] * X1 * X3 + c[3] * X1 * X4
        + c[4] * X2**2 + c[5] * X2 * X3 + c[6] * X2 * X4
        + c[7] * X3**2 + c[8] * X3 * X4 + c[9] * X4**2
    )


def _assert_reference_is_a_clf():
    """Fail at import if the reference stops satisfying Artstein."""
    A = _np.array([[0.0, 1.0, 0.0, 0.0],
                   [0.0, -0.2, 2.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0],
                   [0.0, -0.1, 6.0, 0.0]])
    B = _np.array([[0.0], [0.2], [0.0], [0.1]])
    P = _reference_matrix(_reference_quadratic_are)
    S = (P @ B).reshape(1, -1)
    _, _, vt = _np.linalg.svd(S)
    null = vt[1:].T                       # {x : B'Px = 0}
    M = P @ A + A.T @ P
    reduced = null.T @ M @ null
    eig = _np.linalg.eigvalsh(0.5 * (reduced + reduced.T))
    if eig.max() >= 0.0:
        raise ValueError(
            "GPU5_7 reference form does not satisfy the Artstein condition on "
            f"the linearisation: eig(PA+A'P)|_{{B'Px=0}} = {eig} -- the "
            "hardcoded reference is not the ARE solution"
        )
    return eig


_base._reference_quadratic = _reference_quadratic_are
_base.GPU5_PD_REFERENCE_MATRIX = _reference_matrix(_reference_quadratic_are)
_ARE_EIG = _assert_reference_is_a_clf()
print(
    "GPU5_7 reference = ARE solution; eig(PA+A'P) on {B'Px=0} = "
    f"{_np.round(_ARE_EIG, 6)}",
    flush=True,
)

# V_GRAD targets are still derived from the ARE reference for anyone who
# turns the weight back on; at the default weight 0 they are inert.
def _v_grad_targets(reference_fn, grid_points=21, half_width=0.25):
    axis = _np.linspace(-half_width, half_width, grid_points)
    mesh = _np.meshgrid(axis, axis, axis, axis, indexing="ij")
    values = reference_fn(*mesh)
    spacing = float(axis[1] - axis[0])
    g = [_np.gradient(values, spacing, axis=i) for i in range(4)]
    return (
        float(_np.mean(_np.hypot(g[0], g[1]))),
        float(_np.mean(_np.hypot(g[2], g[3]))),
    )


_G12, _G34 = _v_grad_targets(_reference_quadratic_are)
_base.V_GRAD_X1X2_TARGET = float(
    os.environ.get("SYMCLF_GPU5_7_V_GRAD_X1X2_TARGET", str(_G12))
)
_base.V_GRAD_X3X4_TARGET = float(
    os.environ.get("SYMCLF_GPU5_7_V_GRAD_X3X4_TARGET", str(_G34))
)

# Properness reference selector kept for parity (inert at weight 0).
_PROPER_REF = os.environ.get("SYMCLF_GPU4_1_PROPER_REF", "lqr").strip().lower()
if _PROPER_REF in ("identity", "id", "norm"):
    def _reference_identity(X1, X2, X3, X4):
        return X1**2 + X2**2 + X3**2 + X4**2
    _base._reference_quadratic = _reference_identity
elif _PROPER_REF in ("sum_squared", "sum", "sumsq"):
    def _reference_sum_squared(X1, X2, X3, X4):
        return (X1 + X2 + X3 + X4) ** 2
    _base._reference_quadratic = _reference_sum_squared
elif _PROPER_REF not in ("lqr", "quadratic", ""):
    raise ValueError(f"unknown SYMCLF_GPU4_1_PROPER_REF: {_PROPER_REF!r}")
print(f"GPU5_7 properness reference: {_PROPER_REF}", flush=True)

_base.check_b_manifold_exact = check_b_manifold_exact_gpu_cached
(
    _base.eval_MSE_gpu2,
    _base.eval_MSE_and_tune_constants,
) = make_gpu_evaluators(_base, check_b_manifold_exact_gpu_cached)

for _name in dir(_base):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_base, _name)
