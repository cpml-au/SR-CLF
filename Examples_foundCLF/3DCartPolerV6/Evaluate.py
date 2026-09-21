"""CP3D Evaluate shim: the certified-region objective.

Loads the CPU base Evaluate (examples/4DCartPoler), swaps the exact manifold
check for the src3DCartPoleV3 GPU/SLSQP implementation, and configures the GPU5_7
fitness. Everything src3DCartPoleV3 needs is inside src3DCartPoleV3 -- no other srcGPU
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
    above CP3D_SAT_U_TARGET.  (96051: 57, 122304: 59, 122340 gen-7: 1.5e5)
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
_CPU_EXAMPLE = _HERE   # 3-D: the base lives beside this file
for _path in (_ROOT, _CPU_EXAMPLE):
    if _path not in sys.path:
        sys.path.append(_path)

from src3DCartPoleV3.b_manifold_check_gpu3 import (  # noqa: E402
    check_b_manifold_exact_gpu3_cached as check_b_manifold_exact_gpu_cached,
)
from src3DCartPoleV3.grid_fitness import make_gpu_evaluators  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_Evaluate_cp3d_base", os.path.join(_HERE, "BaseEvaluate.py")
)
_base = importlib.util.module_from_spec(_spec)
sys.modules["_Evaluate_cp3d_base"] = _base
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
_base.CP3D_GATE_PRICE_ESCALATION = float(
    os.environ.get("SYMCLF_CP3D_GATE_PRICE_ESCALATION", "2.0")
)
if "SYMCLF_CP3D_GATE_PRICE_CAP" in os.environ:
    _base.CP3D_GATE_PRICE_CAP = float(
        os.environ["SYMCLF_CP3D_GATE_PRICE_CAP"]
    )
if _base.CP3D_GATE_PRICE_ESCALATION < 1.0:
    raise ValueError(
        "SYMCLF_CP3D_GATE_PRICE_ESCALATION must be >= 1.0: below 1.0 the "
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
_base.CP3D_VIOLATION_WEIGHT = float(
    os.environ.get("SYMCLF_CP3D_VIOLATION_WEIGHT", "300")
)
_base.CP3D_MEAN_MARGIN_SCALE = float(
    os.environ.get("SYMCLF_CP3D_MEAN_MARGIN_SCALE", "0.1")
)
_base.CP3D_GATE_MARGIN_ESCALATION = float(
    os.environ.get("SYMCLF_CP3D_GATE_MARGIN_ESCALATION", "2.0")
)
_base.CP3D_MEAN_MARGIN_WEIGHT = float(
    os.environ.get(
        "SYMCLF_CP3D_MEAN_MARGIN_WEIGHT",
        str(_base.MANIFOLD_EXACT_MARGIN_WEIGHT),
    )
)
_base.CP3D_MEAN_MARGIN_PENALTY_MAX = float(
    os.environ.get(
        "SYMCLF_CP3D_MEAN_MARGIN_PENALTY_MAX",
        str(_base.MANIFOLD_EXACT_MARGIN_PENALTY_MAX),
    )
)

import numpy as _np  # noqa: E402


def _reference_matrix(reference_fn, n=3):
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
    os.environ.get("SYMCLF_CP3D_PROPERNESS_WEIGHT", "0")
)
# V_GRAD magnitude targets: OFF. An unnormalised absolute-error prior; the
# saturation and ROA terms are scale-invariant so the exploit it patched
# (inflating ||grad V||) no longer pays.
_base.V_GRAD_WEIGHT = float(
    os.environ.get("SYMCLF_CP3D_V_GRAD_WEIGHT", "0")
)
# c_star-vs-reference target: OFF (weights 0). Replaced by the ROA term,
# which prices the same quantity (boundary_min enters c_max) against the
# candidate's own violations rather than against the reference form.
_base.C_STAR_TARGET_RATIO = float(
    os.environ.get("SYMCLF_CP3D_C_STAR_TARGET_RATIO", "1.0")
)
_base.C_STAR_TARGET = float(
    os.environ.get("SYMCLF_GPU4_1_C_STAR_TARGET", "0.02")
)
_base.C_STAR_PENALTY_MAX = float(
    os.environ.get("SYMCLF_CP3D_C_STAR_PENALTY_MAX", "0")
)
_base.C_STAR_PENALTY_CAP = float(
    os.environ.get("SYMCLF_CP3D_C_STAR_PENALTY_CAP", "0")
)
_base.C_STAR_TAIL_WEIGHT = float(
    os.environ.get("SYMCLF_CP3D_C_STAR_TAIL_WEIGHT", "0")
)
# Coverage count: OFF. The certified-volume fraction is its continuous,
# scale-invariant replacement.
_base.ROA_COVERAGE_WEIGHT = float(
    os.environ.get("SYMCLF_CP3D_COVERAGE_WEIGHT", "0")
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
_base.CP3D_SAT_U_TARGET = float(
    os.environ.get("SYMCLF_CP3D_SAT_U_TARGET", "1000")
)
_base.CP3D_SAT_WEIGHT = float(
    os.environ.get("SYMCLF_CP3D_SAT_WEIGHT", "150")
)

# --- GPU5_7: the ROA (certified region) term --------------------------------
# c_max = min(boundary_min, min W over violating grid points); the certified
# volume is the fraction of the box with W < c_max. VOL_TARGET 0.005 (0.5% of
# the box) sits just under the best known certified volume (96051: 0.657%),
# so a genuinely certified candidate can pay ~0 while everything else feels a
# monotone pull toward larger certified sets. The negative side (c_max < 0,
# scaled by the candidate's own median |W|) supplies slope where no
# certificate exists at all.
# CP3D: price for a candidate no GPU stage could evaluate (device OOM after
# every retry). Must be a FLOOR, never a discount -- charging only the gate-2
# fee made an OOM-degraded candidate score 3.7x-8.9x BETTER than the same
# candidate screened, turning a hardware failure into a reward. Default is
# MANIFOLD_EXACT_PENALTY_MAX, this codebase's existing "unknown means failed"
# value (what _exact_result_penalty returns for a non-ok result): bounded, and
# well above anything a real evaluation produces.
_base.CP3D_UNSCREENABLE_PENALTY = float(
    os.environ.get(
        "SYMCLF_CP3D_UNSCREENABLE_PENALTY",
        str(float(_base.MANIFOLD_EXACT_PENALTY_MAX)),
    )
)

_base.CP3D_ROA_WEIGHT = float(
    os.environ.get("SYMCLF_CP3D_ROA_WEIGHT", "500")
)
# Target on the SCALE-FREE ratio c_max / median|W| -- not on raw c_max, which
# scales with V and would let the search buy the term by inflating V; not on
# certified volume, which is quantised and hard-zeroed for c_max <= 0 (the
# 500-point cliff that stalled run 122350 at c_max = -4.8e-5). Measured:
# 96051 sits at 0.0221, 122232 at 0.0274, ARE reference at 0.0044.
_base.CP3D_ROA_C_TARGET = float(
    # CP3D REVIEW FIX: 0.02 -> 0.004. 0.02 was calibrated on the STRONGEST
    # certificates in the panel (96051 0.0212, 122232 0.0204) and the ARE
    # reference -- a CLF valid BY CONSTRUCTION -- sits at 0.0042 and so ate
    # nearly the whole 500-point ramp, scoring 384.45. That let 122340's final
    # champion, which DIVERGES in closed loop, score 69.41 and beat it 5.5x.
    # Calibrate to the WEAKEST known-valid certificate, not the strongest --
    # the same error as anchoring the properness band on an invalid matrix.
    os.environ.get("SYMCLF_CP3D_ROA_C_TARGET", "0.004")
)
_base.CP3D_ROA_NEG_WEIGHT = float(
    os.environ.get("SYMCLF_CP3D_ROA_NEG_WEIGHT", "250")
)

# --- GPU5_7: certificate depth (exact stage) --------------------------------
# Violations found by the exact Artstein/PD searches are priced by their
# W-placement: weight * clip(1 - min_w/boundary_min, 0, 2). Both current
# champions park their violations at W ~ 0 -- certificate empty, margin
# penalty tiny. This is the term that makes that placement expensive.
_base.CP3D_CERT_WEIGHT = float(
    os.environ.get("SYMCLF_CP3D_CERT_WEIGHT", "300")
)

# --- GPU5_7: the closed-loop rollout gate -----------------------------------
# A candidate with zero violations, PD passing, is integrated with the
# SATURATED Sontag controller (|u| <= ROLLOUT_UMAX, default = SAT_U_TARGET)
# from the CPU example's standard initial conditions. DIVERGENCE is charged
# (500/trajectory); failure to fully converge is not, since standard starts
# may legitimately sit outside a small certified region. This is the exact
# test the 122340 gen-7 "certified" champion fails.
_base.CP3D_ROLLOUT_ENABLED = (
    os.environ.get("SYMCLF_CP3D_ROLLOUT_ENABLED", "1") == "1"
)
_base.CP3D_ROLLOUT_UMAX = float(
    os.environ.get("SYMCLF_CP3D_ROLLOUT_UMAX",
                   str(_base.CP3D_SAT_U_TARGET))
)
_base.CP3D_ROLLOUT_T = float(
    os.environ.get("SYMCLF_CP3D_ROLLOUT_T", "10.0")
)
_base.CP3D_ROLLOUT_DT = float(
    os.environ.get("SYMCLF_CP3D_ROLLOUT_DT", "2e-3")
)
_base.CP3D_ROLLOUT_DIVERGE_NORM = float(
    os.environ.get("SYMCLF_CP3D_ROLLOUT_DIVERGE_NORM", "1.0")
)
_base.CP3D_ROLLOUT_FAIL_PENALTY = float(
    os.environ.get("SYMCLF_CP3D_ROLLOUT_FAIL_PENALTY", "500")
)

# --- exact near-zero sharpener: coefficients cut 16x (GPU5_3) ---------------
# Kept: infinite slope at margin 0 so a tiny positive margin cannot be parked
# on, without the 256-coefficient constant floor. pd_result_penalty reads this
# SAME curve (CP3D fix), so one override rescales Artstein and PD together.
_NEAR0_SQRT = float(os.environ.get("SYMCLF_CP3D_NEAR0_SQRT", "0.25"))
_NEAR0_QUARTIC = float(os.environ.get("SYMCLF_CP3D_NEAR0_QUARTIC", "4"))
_NEAR0_SIXTH = float(os.environ.get("SYMCLF_CP3D_NEAR0_SIXTH", "64"))


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
_base.CP3D_NORMALIZE_MARGIN = (
    os.environ.get("SYMCLF_CP3D_NORMALIZE_MARGIN", "1") == "1"
)
_base.CP3D_GRAD_NORM_FLOOR = float(
    os.environ.get("SYMCLF_CP3D_GRAD_NORM_FLOOR", "1e-3")
)
_base.CP3D_FLAT_GRAD_FLOOR = float(
    os.environ.get("SYMCLF_CP3D_FLAT_GRAD_FLOOR", "1e-3")
)
_base.CP3D_FLAT_GRAD_WEIGHT = float(
    os.environ.get("SYMCLF_CP3D_FLAT_GRAD_WEIGHT", "1.0")
)

# --- GPU5_5: strict violation predicate -------------------------------------
# viol = margin > -1e-9 in every checker (one number: MANIFOLD_MARGIN_TOL_
# EXACT). A degenerate a == 0 root is a violation instead of certifying free.
# 96051's true margin is -1.13e-07 (113x below the tolerance) so real CLFs
# pass untouched.
_base.CP3D_STRICT_MARGIN_TOL = float(
    os.environ.get("SYMCLF_CP3D_STRICT_MARGIN_TOL", "1e-9")
)
if _base.CP3D_STRICT_MARGIN_TOL <= 0.0:
    raise ValueError(
        "SYMCLF_CP3D_STRICT_MARGIN_TOL must be > 0: it is negated into "
        "MANIFOLD_MARGIN_TOL_EXACT, and a non-positive value restores the "
        "non-strict test that lets a == 0 certify for free"
    )
_base.MANIFOLD_MARGIN_TOL_EXACT = -_base.CP3D_STRICT_MARGIN_TOL
print(
    "CP3D strict margin predicate: violation iff margin > "
    f"{_base.MANIFOLD_MARGIN_TOL_EXACT:.1e}  (a == 0 is now a violation)",
    flush=True,
)

# --- GPU5_5: x1-axis terms (kept as slope, weight-reduced role) --------------
# The saturation term subsumes the axis VERDICT (a == 0 on the axis forces
# b != 0 there), but on the degenerate-slab family a = b = 0 hides from the
# ratio a/|b|, so these remain the only CONTINUOUS slope out of that basin.
# They charge known-good candidates 0.17-0.27 points total.
_base.CP3D_AXIS_B_TARGET = float(
    os.environ.get("SYMCLF_CP3D_AXIS_B_TARGET", "1e-3")
)
_base.CP3D_AXIS_B_WEIGHT = float(
    os.environ.get("SYMCLF_CP3D_AXIS_B_WEIGHT", "300")
)
_base.CP3D_AXIS_V_TARGET = float(
    os.environ.get("SYMCLF_CP3D_AXIS_V_TARGET", "1e-2")
)
_base.CP3D_AXIS_V_WEIGHT = float(
    os.environ.get("SYMCLF_CP3D_AXIS_V_WEIGHT", "300")
)
_base.CP3D_AXIS_TAIL_WEIGHT = float(
    os.environ.get("SYMCLF_CP3D_AXIS_TAIL_WEIGHT", "0.5")
)
# V(0) is gauge; properness (when enabled) and PD measure W = V - V(0).
_base.CP3D_ORIGIN_PENALTY_WEIGHT = float(
    os.environ.get("SYMCLF_CP3D_ORIGIN_PENALTY_WEIGHT", "0")
)

# --- 3-D ARE reference form ------------------------------------------------
#
# Solved from THIS system's own linearisation with the project's Q = I and
# R = 1e-4 (SystemDynamics.py). States (x1, x2, x3) = (cart velocity, pole
# angle, pole rate):
#
#     A = [[-0.2, 2, 0], [0, 0, 1], [-0.1, 6, 0]]      B = [0.2, 0, 0.1]
#     controllability rank 3/3 (det 0.01),  ARE residual 3.2e-12
#
#     P = [[ 0.971695, -4.658320, -2.044396],
#          [-4.658320, 24.844143, 10.348886],
#          [-2.044396, 10.348886,  4.554601]]
#
# Valid BY CONSTRUCTION: on {b = 0} = {B'Px = 0} the ARE gives
# x'(A'P + PA)x = -x'Qx, so the restricted eigenvalues are exactly [-1, -1].
# The 4-D project shipped a hardcoded matrix that was NOT its ARE solution
# (off by up to 24%, one restricted eigenvalue +0.412, 209 violations on the
# box), and every run before GPU5_6 was scored against a target that is not a
# CLF. The import-time guard below re-runs that test so it cannot recur here.
#
# cond(P) = 682 against 1712 in 4-D, so this system is also less anisotropic.
_ARE_REFERENCE = (
     0.11009588505,   # x1^2
    -2.11326179407,   # x1 x2
    -0.83072308244,   # x1 x3
     25.00000000000,   # x2^2
     17.92706326818,   # x2 x3
     4.01658510731,   # x3^2
)

def _reference_quadratic_are(X1, X2, X3):
    c = _ARE_REFERENCE
    return (
        c[0] * X1**2 + c[1] * X1 * X2 + c[2] * X1 * X3
        + c[3] * X2**2 + c[4] * X2 * X3 + c[5] * X3**2
    )


def _assert_reference_is_a_clf():
    A = _np.array([[-0.2, 2.0, 0.0], [0.0, 0.0, 1.0], [-0.1, 6.0, 0.0]])
    B = _np.array([[0.2], [0.0], [0.1]])
    P = _reference_matrix(_reference_quadratic_are, n=3)
    S = (P @ B).reshape(1, -1)
    _, _, vt = _np.linalg.svd(S)
    null = vt[1:].T
    reduced = null.T @ (P @ A + A.T @ P) @ null
    eig = _np.linalg.eigvalsh(0.5 * (reduced + reduced.T))
    if eig.max() >= 0.0:
        raise ValueError(
            "3-D reference form does not satisfy the Artstein condition on "
            f"the linearisation: eig(PA+A'P)|_{{B'Px=0}} = {eig}"
        )
    return eig


_base._reference_quadratic = _reference_quadratic_are
_base.GPU5_PD_REFERENCE_MATRIX = _reference_matrix(_reference_quadratic_are, n=3)
_ARE_EIG = _assert_reference_is_a_clf()
print(
    "3DCartPole reference = ARE solution; eig(PA+A'P) on {B'Px=0} = "
    f"{_np.round(_ARE_EIG, 6)}  (must be [-1, -1])",
    flush=True,
)


def _v_grad_targets(reference_fn, grid_points=41, half_width=0.25):
    """Mean gradient magnitudes of the reference on the fitness grid.

    Two numbers only, because there are three states: the (x1, x2) pair keeps
    the 4-D name, and the third target is the mean |d/dx3|.
    """
    axis = _np.linspace(-half_width, half_width, grid_points)
    mesh = _np.meshgrid(axis, axis, axis, indexing="ij")
    values = reference_fn(*mesh)
    spacing = float(axis[1] - axis[0])
    g = [_np.gradient(values, spacing, axis=i) for i in range(3)]
    return (
        float(_np.mean(_np.hypot(g[0], g[1]))),
        float(_np.mean(_np.abs(g[2]))),
    )


_G12, _G34 = _v_grad_targets(_reference_quadratic_are)
_base.V_GRAD_X1X2_TARGET = float(
    os.environ.get("SYMCLF_CP3D_V_GRAD_X1X2_TARGET", str(_G12))
)
_base.V_GRAD_X3X4_TARGET = float(
    os.environ.get("SYMCLF_CP3D_V_GRAD_X3X4_TARGET", str(_G34))
)
print(
    f"3DCartPole V_GRAD targets: {_base.V_GRAD_X1X2_TARGET:.4f} / "
    f"{_base.V_GRAD_X3X4_TARGET:.4f}",
    flush=True,
)

# (3-D: the properness-reference selector is removed -- its alternatives are
#  four-argument forms, the ARE quadratic above is the only reference, and
#  PROPERNESS_PENALTY_WEIGHT defaults to 0.)

# =============================================================================
# V2 KNOBS -- the four changes, and why each one is calibrated the way it is.
# Every default below was checked against the 3-D ARE quadratic FIRST. That
# form is proper by construction and passes the exact Artstein checker (228
# roots, 0 violations, margin -9.854e-05, PD ok), so it is the one candidate
# whose correctness is not in question. Three earlier terms were anchored to
# something the correct answer fails -- the properness band on a non-CLF
# matrix, u_required (ARE 294.6 against the DIVERGING gen 88's 15.05), and
# Euclidean anisotropy (ARE 25.2x, worst of three). Not again.
# =============================================================================

# (1) ANTI-NEEDLE. q(x) = W(x) / (x' P_ARE x); the penalty is std(log q).
#     Exactly 0 for the ARE form. Measured on run 122467:
#         3-D ARE (correct)   0.0000      gen 89 (converges)  1.1115
#         gen 176 (current)   1.1048      gen 88 (DIVERGES)   1.3923
#     First cheap, continuous, full-dimensional quantity that orders those
#     correctly. Scale-free: W -> kW shifts log q by a constant.
#     300 x 1.11 = 333 for a gen-89-like needle, against a length penalty of
#     ~0.13 -- deliberately dominant, because the length penalty is what the
#     search collapsed to at the plateau.
_base.V2_SHAPE_WEIGHT = float(os.environ.get("SYMCLF_V2_SHAPE_WEIGHT", "300"))

# (2) rho PINNED TO THE BOUNDARY (Yang/Dai, Lyapunov_Stable_NN_Controllers/
#     neural_lyapunov_training/lyapunov.py:426). The old c_max was
#     min(boundary_min, min_w_violating) -- a free variable that collapses to
#     the first violation, so shrinking the certified set was the cheapest way
#     to clear it. That single line WAS the needle incentive.
#     1.0 makes the certified set reach the box boundary; below 1.0 keeps it
#     strictly inside.
_base.V2_ROA_RHO_MULTIPLIER = float(
    os.environ.get("SYMCLF_V2_ROA_RHO_MULTIPLIER", "1.0")
)

# (3) Violations inside the pinned rho are now a CHARGE instead of something
#     that redefines the region. Continuous in (rho - min_W_violating)/rho:
#     0 when the first violation sits at or above rho, 1 when one reaches the
#     origin.
_base.V2_CERT_WEIGHT = float(os.environ.get("SYMCLF_V2_CERT_WEIGHT", "500"))

# (4) THE UNBOUNDED-SONTAG vdot COUNT IS OFF (weight 0).
#     vdot = a - 1e4 b^2 lambda is algebraically -sqrt(a^2 + 1e4 r^2 b^2)
#     (verified to 2.1e-07 across the grid), hence negative wherever b != 0.
#     min|b| on the lattice is 4.9e-05 and max vdot is -1.022e-01 against a
#     threshold of -2.2e-04: a factor of 460, at every point, for every
#     candidate. The drift is destabilising at 68% of the box and the term
#     still reports zero. It is a tautology, not a weak test. Still computed
#     and reported as a diagnostic; set to 1 to restore the old behaviour.
_base.V2_VDOT_WEIGHT = float(os.environ.get("SYMCLF_V2_VDOT_WEIGHT", "0"))

# (5) V3: POSITIVITY IS A MAGNITUDE, NOT A COUNT.
#     The old `v_violations` count was the wall around the correct answer.
#     Measured on perturbations of the ARE coefficients:
#         perturbation   count (= penalty)   magnitude x300   lambda_min(P)
#         ARE +-  2%                    0              0.0         +0.0163
#         ARE +-  5%                  174              0.1         -0.0500
#         ARE +- 10%                  724              2.5         -0.3037
#         ARE +- 25%                  506              1.5         -0.1751
#     Piecewise-constant, non-monotone, jumping by hundreds the moment P leaves
#     the PD cone -- and with Q=I, R=1e-4 the ARE sits ON that edge
#     (lambda_min = 0.0441; 57% of +-5% perturbations lose PD). Grid fitness was
#     0.00 at the ARE and ~2595 at +-5%, worse than a generic all-ones
#     quadratic. Neither the SEA nor finite-difference Adam could enter the
#     basin from any start. The magnitude has the same zero, is continuous, and
#     its gradient points back into the cone.
_base.V3_POSITIVITY_WEIGHT = float(
    os.environ.get("SYMCLF_V3_POSITIVITY_WEIGHT", "300")
)

# Terms explicitly NOT re-enabled, with the measurement that rules each out:
#   * Euclidean anisotropy  -- ARE 25.2x vs diverging gen 88's 10.8x
#   * origin_probe_ratio    -- ARE 24.8 vs diverging gen 88's 9.0
#   * u_required / SAT       -- ARE 294.6 vs diverging gen 88's 15.05
#   * raising ROA_C_TARGET  -- ARE c_ratio 9.30e-03, gen 89 9.36e-03:
#                              the correct answer and the needle are
#                              indistinguishable on that axis
#   * properness band       -- the original anchor bug; weight stays 0

# ============================================================================
# 2026-09-10: DET4 MODE (SYMCLF_EXACT_MODE=det4, the default here).  The fitness is
#     symbolic + invalid + length + ROA_RULE(c_v*, kappa*, boundary_min, w_scale)
# with the two DET4 numbers read by the GPU5_7 pipeline itself (lattice -> cheap GPU screen ->
# exact checker + PD check), same search and same scheduling as GPU5_7, the DET4 ratios as the
# polished quantities.  Every other grid term is switched off inside the kernel; no margin
# curves, rollouts or cert_penalty run.  See README_DET4.md.
# ============================================================================
_base.EXACT_MODE = os.environ.get("SYMCLF_EXACT_MODE", "det4")
_base.DET4_MODE = 1.0 if _base.EXACT_MODE == "det4" else 0.0
_base.DET4_RHO = float(os.environ.get("SYMCLF_DET4_RHO", str(_base.CP3D_SAT_U_TARGET)))   # u_max
_base.DET4_PD_EPS = float(os.environ.get("SYMCLF_DET4_PD_EPS", str(_base.GPU5_PD_EPS)))
_base.DET4_KAPPA_MIN = float(os.environ.get("SYMCLF_DET4_KAPPA_MIN", "0"))
_base.DET4_KAPPA_SCALE = float(os.environ.get("SYMCLF_DET4_KAPPA_SCALE", "1.0"))
_base.DET4_REFINE_TOP = int(os.environ.get("SYMCLF_DET4_REFINE_TOP", "3"))      # elites per gen
_base.DET4_REFINE_STEPS = int(os.environ.get("SYMCLF_DET4_REFINE_STEPS", "40"))  # descent evaluations
_base.DET4_REFINE_LR = float(os.environ.get("SYMCLF_DET4_REFINE_LR", "0.3"))
if _base.DET4_MODE > 0.0:
    print(
        f"3DCartPolerV6 DET4 mode: rho={_base.DET4_RHO:g} pd_eps={_base.DET4_PD_EPS:g} "
        f"kappa_min={_base.DET4_KAPPA_MIN:g} ROA weight={_base.CP3D_ROA_WEIGHT:g} "
        f"target={_base.CP3D_ROA_C_TARGET:g}; GPU5_7 scheduling (cheap screen on every gate-1 "
        f"pass, full check if max a/W <= {_base.GPU2_EXACT_A_MAX_GATE:g} or certified, "
        f"escalation {_base.CP3D_GATE_PRICE_ESCALATION:g}); elites {_base.DET4_REFINE_TOP} x "
        f"{_base.DET4_REFINE_STEPS} steps; origin exclusion 0 "
        f"(absolute root check ball {getattr(_base, 'ABSOLUTE_CHECK_BALL_RADIUS', None)})",
        flush=True,
    )

_base.check_b_manifold_exact = check_b_manifold_exact_gpu_cached
(
    _base.eval_MSE_gpu2,
    _base.eval_MSE_and_tune_constants,
) = make_gpu_evaluators(_base, check_b_manifold_exact_gpu_cached)

for _name in dir(_base):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_base, _name)
