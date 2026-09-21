"""V8 Evaluate shim (2026-09-15): the DET4 objective with ONE check.

Loads the CPU base (BaseEvaluate.py) and configures the knobs the V8 engine reads: the lattice
kernel's terms (grid_fitness), the DET4 rule (fitness) and the 4-D ARE reference. V7's checker
knobs (cheap screen, Artstein ascent, root polish, GPU5 PD, margins, rollouts, gate pricing) are
gone with their code. The history below is kept for the record.

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

# --- PD tolerance (the DET4_PD_EPS default) ----------------------------------
_base.GPU5_PD_EPS = float(os.environ.get("SYMCLF_GPU5_PD_EPS", "1e-4"))
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

# --- grid flatness count (legacy kernel term, inert in DET4 mode; read by the kernel args) --
_base.CP3D_FLAT_GRAD_FLOOR = float(
    os.environ.get("SYMCLF_CP3D_FLAT_GRAD_FLOOR", "1e-3")
)
_base.CP3D_FLAT_GRAD_WEIGHT = float(
    os.environ.get("SYMCLF_CP3D_FLAT_GRAD_WEIGHT", "1.0")
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

# --- 4-D ARE reference form (copied from examples/4DCartPolerGPU5_7/Evaluate.py) -----------
# GPU5_6/GPU5_7: the true ARE solution with the project's Q = I, R = 1e-4 (re-solved
# 2026-09-11: max |coefficient diff| 4.1e-13; lambda_min(P) 0.0440, cond(P) 1435). Used by the
# legacy grid terms and the smoke only -- the det4 price does not read the reference.
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
_base.GPU5_PD_REFERENCE_MATRIX = _reference_matrix(_reference_quadratic_are, n=4)
_ARE_EIG = _assert_reference_is_a_clf()
print(
    "4DCartPole reference = ARE solution (GPU5_7's); eig(PA+A'P) on {B'Px=0} = "
    f"{_np.round(_ARE_EIG, 6)}",
    flush=True,
)


# V_GRAD targets (GPU5_7's 4-D function); inert at the default weight 0.
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
    os.environ.get("SYMCLF_CP3D_V_GRAD_X1X2_TARGET", str(_G12))
)
_base.V_GRAD_X3X4_TARGET = float(
    os.environ.get("SYMCLF_CP3D_V_GRAD_X3X4_TARGET", str(_G34))
)
print(
    f"4DCartPole V_GRAD targets: {_base.V_GRAD_X1X2_TARGET:.4f} / "
    f"{_base.V_GRAD_X3X4_TARGET:.4f}",
    flush=True,
)

# (4DCartPolerV6: GPU5_7's properness-reference selector stays removed, as in 3DCartPolerV6;
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
        f"4DCartPolerV8 DET4 mode: rho={_base.DET4_RHO:g} pd_eps={_base.DET4_PD_EPS:g} "
        f"kappa_min={_base.DET4_KAPPA_MIN:g} ROA weight={_base.CP3D_ROA_WEIGHT:g} "
        f"target={_base.CP3D_ROA_C_TARGET:g}; one check (det4_fast) on every live tree; "
        f"elites {_base.DET4_REFINE_TOP} x {_base.DET4_REFINE_STEPS} steps; origin exclusion 0",
        flush=True,
    )

for _name in dir(_base):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_base, _name)
