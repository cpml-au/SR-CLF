"""Evaluation for 4D cart-pole CLF discovery.

1. Grid fitness: properness sandwich vs the reference quadratic, V > 0,
   min-norm Vdot decay-rate count, origin penalty, gradient targets.
2. Numerical verifier (de/cmaes/shgo): V positivity and the exact scalar
   bounded-input CLF validity condition on the punctured box
       a(x) - CLF_RHO*|b(x)|**2 + CLF_GAMMA1 < 0
   could be restricted to the certified sublevel set {V - V(0) < c*}.
4. ROA coverage penalty (extra constraint): rollout initial states must lie inside {V - V(0) < c*}.
5. Closed-loop rollouts with the saturated min-norm controller, gated.

Generic machinery lives in src/clf_checks.py; this module owns the
system dynamics bindings and all configuration constants.
"""

import numpy as np
from src.VVdot_Calculations import compute_v_and_v_dot
from src.SymFunctions import (
    contains_symbol,
    DeapSimplifier,
    substitute_paramsCoef,
    detect_nested_function_calls,
)
from src.shgo_numerical_verification import verify_clf_shgo
from src.pygmo_counterexample_optimizers import verify_clf_pygmo
from src.ibex_counterexample_optimizers import verify_clf_ibex_barrier
from src.b_manifold_check import check_b_manifold
from src.b_manifold_check_exact import check_b_manifold_exact
from src import clf_checks
from sympy import Matrix, symbols, diff
import pygmo as pg
from SystemDynamics import f, G, Q, R
from SystemDynamicsSR import fSR, GSR, QSR, RSR
import warnings


# --- IBEX barrier verifier (12 boxes run in parallel) ---
IBEX_VERIFIER_ENABLED = True
IBEX_VERIFIER_TIMEOUT_SECONDS = 5.0  # per box; clean margins escalate to 2x, then 3x
IBEX_VERIFIER_VERBOSE = False #True

# --- numerical verifier (de/de1220/cmaes/shgo) ---
NUMERICAL_VERIFIER = None  # options: "shgo", "cmaes", "de", "de1220", None
NUMERICAL_VERIFIER_RANDOM = True
SHGO_BOUNDS = [(-0.25, 0.25)] * 4
SHGO_DECAY_RATE = 0.0012
# Strictness margin for the Artstein/bounded-input checks.
# gamma1 = 0 is the PURE Artstein condition (a < 0 on {b=0}, x != 0): on the
# compact punctured box, continuity of a makes strict validity carry its own
# uniform margin per candidate, so no imposed margin is needed. Any gamma1 > 0
# rejects Artstein-valid candidates near the boundary, and a constant gamma1
# is unsatisfiable near the origin ball (measured on the Riccati quadratic:
# a = -0.2504*r^2 on b=0 roots, so a + 1e-6 > 0 for all roots with
# r < 2.0e-3 > CLF_ORIGIN_EXCLUDE_RADIUS). Anti-degeneracy duty (flat
# valleys, needles) is carried by the positivity checks and origin probe.
# Estimation noise in a is absorbed by MANIFOLD_MARGIN_TOL_* instead.
CLF_GAMMA1 = 0.0
MANIFOLD_MARGIN_TOL_FD = 1e-8      # FD gradient noise floor on a at roots
MANIFOLD_MARGIN_TOL_EXACT = 1e-12  # exact-derivative numerical floor
CLF_ORIGIN_EXCLUDE_RADIUS = 1.1e-3
SHGO_MSE_GATE = np.inf # 1.0  # 0.5
SHGO_RANDOM_SEED = 0
SHGO_IGNORE_VDOT_BALL_RADIUS = CLF_ORIGIN_EXCLUDE_RADIUS
SHGO_IGNORE_VDOT_ABS_TOL = 1e-6
PYGMO_VERIFY_GENERATIONS = 80#200
PYGMO_VERIFY_POPULATION_SIZE = 100#200
PYGMO_VERIFY_LOCAL_SAMPLES = 150#3000#3000
PYGMO_VERIFY_LOCAL_RADIUS = 0.01  # per-coordinate radius around each DE champion
PYGMO_VERIFY_FD_STEP = 1e-5
# --- adaptive DE falsifier: escalate search effort as the margin nears 0 ---
# Run 1 at normal settings (fresh seed) -> m1.
#   m1 >= FALSIFIER_ESCALATE_BELOW: gross violator, no escalation.
#   FALSIFIER_CLEAN_BELOW <= m1 < FALSIFIER_ESCALATE_BELOW: one extra run at
#       FALSIFIER_MID_MULT x samples (fresh seed).
#   m1 < FALSIFIER_CLEAN_BELOW (incl. negative = "looks clean"): one extra run
#       at FALSIFIER_CLEAN_MULT x samples (fresh seed).
# The penalty comes from whichever run found the BIGGER margin (monotone in
# search effort: more search can only raise or confirm the verdict).
FALSIFIER_ADAPTIVE_ENABLED = True
FALSIFIER_ESCALATE_BELOW = 0.05
FALSIFIER_CLEAN_BELOW = 0.0
FALSIFIER_MID_MULT = 2
FALSIFIER_CLEAN_MULT = 3
# gamma1 is already the strictness/numerical safety margin. Do not offset it
# again with a second acceptance tolerance in the barrier comparison.
DECAY_TOL = 0.0
V_MAGNITUDE_LIMIT = 1e6
V_GRAD_X1X2_TARGET = 3.6186
V_GRAD_X3X4_TARGET = 16.1497
V_GRAD_WEIGHT = 1.0

# --- symbolic subspace structure check ---
SUBSPACE_STRUCTURE_CHECK_ENABLED = False
SUBSPACE_STRUCTURE_PENALTY = 1e1

# Bounded scalar-input authority for the CLF condition checked consistently by
# DE and SHGO on the punctured box:
#     a(x) - rho*|b(x)|**2 + CLF_GAMMA1 < 0.
# At b = 0 this becomes the constant-margin Artstein condition
# a + CLF_GAMMA1 < 0. The former +SHGO_DECAY_RATE*||x||^2 term is deliberately
# not part of this check.
CLF_RHO = 80000#800 #800.0

ROA_COVERAGE_WEIGHT = 1.0     # per rollout x0 outside the certified {V < c*}

# --- b=0 manifold check (exact Artstein condition, gated) ---
# Solves {b(x)=0} by line scans + bisection and requires
#     a + CLF_GAMMA1 <= 0
# at every retained root outside the CLF_ORIGIN_EXCLUDE_RADIUS ball.
# Gate: runs only when final_mse < MANIFOLD_MSE_GATE; candidates that were
# NOT checked are floored at the gate, so fitness below the gate is only
# reachable by passing this check. Violators get an order-1000 penalty
# that puts them back above the gate.
MANIFOLD_CHECK_ENABLED = False
MANIFOLD_MSE_GATE = 10000
MANIFOLD_ROOT_PENALTY = 100.0
MANIFOLD_CEX_WEIGHT = 1.0
# Smooth margin mapping: m = margin + SAFETY; penalty = min(W*m^2/(m+M0), CAP)
# for m > 0, else 0. Quadratic near zero (gentle under margin-estimate noise),
# asymptotically linear, saturating at CAP near margin ~0.5. SAFETY=1e-4 means
# "clean" requires margin <= -1e-4: pushes through the knife edge instead of
# parking on it.
MANIFOLD_MARGIN_SAFETY = 1e-4
MANIFOLD_MARGIN_M0 = 0.01
MANIFOLD_MARGIN_WEIGHT = 12000.0        # FD manifold: cap 6000 near margin 0.5
MANIFOLD_MARGIN_PENALTY_MAX = 6000.0
MANIFOLD_EXACT_MARGIN_WEIGHT = 6000.0   # exact manifold: cap 1000 near margin 0.5
MANIFOLD_EXACT_MARGIN_PENALTY_MAX = 2000.0
# roots == 0 means the check found NO manifold: it proved nothing. Price that
# "unknown" state so manifold-free candidates cannot hide behind it.
MANIFOLD_VACUOUS_PENALTY = 100.0
# same medicine for the DE/shgo verifier penalty: smooth slope on the worst
# barrier margin (decay_margin_max -> 0) and the positivity margin (v_min -> 0+)
SHGO_MARGIN_WEIGHT = 1000.0 * 5.0
# flat penalty the DE/shgo verifier adds per counterexample class (was the
# hardcoded default inside verify_clf_pygmo; exposed here as a knob).
# NOTE: lowering this (e.g. to 100) aligns the shgo cliff with the manifold
# cliff and shrinks the seed-lottery fitness noise.
SHGO_ROOT_PENALTY = 0.5
# --- exact-gradient (symbolic) manifold check: final confirmation layer ---
# Same condition as the FD manifold check but with exact sympy derivatives
# (no fd_step error). Runs only for near-feasible candidates; margin penalty
# uses the same clip(WEIGHT*margin, MIN, MAX) scheme as the FD check, and the
# total is capped at MANIFOLD_EXACT_PENALTY_MAX.
MANIFOLD_EXACT_CHECK_ENABLED = False #True
MANIFOLD_EXACT_MSE_GATE = 1000000
MANIFOLD_EXACT_PENALTY_MAX = 50000.0
C_STAR_TARGET = 0.003
C_STAR_PENALTY_MAX = 500000.0

PROPERNESS_PENALTY_WEIGHT = 0.1 #4.0

# --- origin-consistency probe (surgical needle detector) ---
# A legitimate V has V(x) - V(0) ~ x'Px near the origin, so the ratio
# (V(x)-V(0))/||x||^2 at tiny offsets is bounded by lambda_max(P) (~50-60 for
# LQR-scale CLFs, ~3000 at the top of the 50x proper band). An origin "needle"
# (V pinned to ~0 in a <1e-8 ball, jumping to O(1) just outside) blows this
# ratio up to ~1e7. Probing V at +/-1e-4 and +/-1e-2 on each axis (16 evals)
# and flagging ratio > ORIGIN_PROBE_K catches the needle with huge margin and
# never touches legitimate candidates.
ORIGIN_PROBE_ENABLED = True
ORIGIN_PROBE_K = 5e4
ORIGIN_PROBE_OFFSETS = (1e-4, 1e-2)
ORIGIN_PROBE_PENALTY = 100000.0

# --- closed-loop rollouts (gated; the ground-truth check) ---
ROLLOUT_ENABLED = False
ROLLOUT_MSE_GATE = np.inf       # run only when everything else is near-feasible
ROLLOUT_T = 1.5
ROLLOUT_DT = 2e-3
ROLLOUT_X0S = (
    (0.0, 0.0, 0.01, 0.0),
    (0.05, 0.0, 0.05, 0.0),
    (-0.05, 0.05, -0.05, 0.05),
    (0.08, -0.05, 0.08, -0.05),
)
ROLLOUT_DIVERGE_NORM = 1.0
ROLLOUT_DIVERGE_PENALTY = 1000.0
ROLLOUT_V_INCREASE_WEIGHT = 50.0  # times the fraction of steps where V grew
ROLLOUT_FINAL_V_WEIGHT = 10.0     # times ReLU(V(T)/V(0) - 1)
ROLLOUT_CONVERGED_NORM = 1e-3


def _sympy_expression(ind2MSE, consts):
    expr_str = substitute_paramsCoef(str(ind2MSE), consts)
    return DeapSimplifier(expr_str, should_print=False)


def _sublevel_cap_from_grid(V_vals, V_val_0):
    """Largest c such that {V - V(0) < c} stays inside the box."""
    return clf_checks.sublevel_cap_from_grid(V_vals, V_val_0)


def _rollout_penalty(V_call, V_val_0):
    return clf_checks.rollout_penalty(
        V_call, V_val_0, f, _G_point, Q, R, ROLLOUT_X0S,
        t_final=ROLLOUT_T,
        dt=ROLLOUT_DT,
        diverge_norm=ROLLOUT_DIVERGE_NORM,
        diverge_penalty=ROLLOUT_DIVERGE_PENALTY,
        v_increase_weight=ROLLOUT_V_INCREASE_WEIGHT,
        final_v_weight=ROLLOUT_FINAL_V_WEIGHT,
        converged_norm=ROLLOUT_CONVERGED_NORM,
        fd_step=PYGMO_VERIFY_FD_STEP,
    )


def _G_point(x1, x2, x3, x4):
    L = 2.0
    m = 1.0
    M = 5.0

    cos_theta = -np.cos(x3)
    denominator = m * L * L * (M + m * (1.0 - cos_theta**2))
    u1 = m * L * L / denominator
    u2 = m * L * cos_theta / denominator

    G_matrix = np.zeros((4, 4))
    G_matrix[1, 1] = u1
    G_matrix[3, 1] = -u2
    return G_matrix


def _manifold_margin_penalty(margin_max, weight, cap):
    """Smooth margin penalty: m = margin + SAFETY; W*m^2/(m+M0) capped.

    Quadratic near zero (gentle under margin-estimate noise), linear for
    larger margins, saturating at `cap`. Zero when margin <= -SAFETY, so
    'clean' means a real safety margin, not the knife edge at 0."""
    if not np.isfinite(margin_max):
        return 0.0
    m = margin_max + MANIFOLD_MARGIN_SAFETY
    if m <= 0.0:
        return 0.0
    return float(min(weight * m * m / (m + MANIFOLD_MARGIN_M0), cap))


def _origin_probe_penalty(individual, consts, v_val_0, n_states=4):
    """Flag origin 'needles': V(x)-V(0) growing much faster than ||x||^2 at
    tiny per-axis offsets. Returns (penalty, worst_ratio)."""
    if not ORIGIN_PROBE_ENABLED:
        return 0.0, 0.0
    worst = 0.0
    for i in range(n_states):
        for d in ORIGIN_PROBE_OFFSETS:
            for s in (d, -d):
                x = np.zeros(n_states)
                x[i] = s
                try:
                    v = float(np.asarray(individual(*x, consts)).reshape(-1)[0])
                except Exception:
                    continue
                r2 = float(x @ x)
                if r2 > 0 and np.isfinite(v):
                    worst = max(worst, (v - v_val_0) / r2)
    penalty = ORIGIN_PROBE_PENALTY if worst > ORIGIN_PROBE_K else 0.0
    return penalty, worst


def _G_col_vec(x1, x2, x3, x4):
    """Active G column (input index 1), vectorized over array states."""
    L = 2.0
    m = 1.0
    M = 5.0
    cos_theta = -np.cos(np.asarray(x3, dtype=float))
    denominator = m * L * L * (M + m * (1.0 - cos_theta**2))
    zeros = np.zeros_like(cos_theta)
    return np.array(
        [zeros, m * L * L / denominator, zeros, -m * L * cos_theta / denominator]
    )


def _numerical_verifier_seed(random_seed=None):
    if random_seed is not None:
        return random_seed
    if NUMERICAL_VERIFIER_RANDOM:
        return None
    return SHGO_RANDOM_SEED


def _run_shgo_verification(ind2MSE, consts, sublevel_cap=None, random_seed=None):
    return verify_clf_shgo(
        _sympy_expression(ind2MSE, consts),
        fSR=fSR,
        GSR=GSR,
        QSR=QSR,
        RSR=RSR,
        bounds=SHGO_BOUNDS,
        decay_rate=SHGO_DECAY_RATE,
        decay_tol=DECAY_TOL,
        random_seed=_numerical_verifier_seed(random_seed),
        ignore_vdot_ball_radius=SHGO_IGNORE_VDOT_BALL_RADIUS,
        ignore_vdot_abs_tol=SHGO_IGNORE_VDOT_ABS_TOL,
        origin_tol=CLF_ORIGIN_EXCLUDE_RADIUS,
        rho=CLF_RHO,
        gamma1=CLF_GAMMA1,
        sublevel_cap=sublevel_cap,
        pd_eps=1e-4,
        root_penalty=SHGO_ROOT_PENALTY,
    )


def _run_pygmo_verification(
    individual, consts, optimizer, sublevel_cap=None, random_seed=None,
    local_samples=None,
):
    return verify_clf_pygmo(
        individual,
        consts,
        f=f,
        G=_G_point,
        Q=Q,
        R=R,
        bounds=SHGO_BOUNDS,
        optimizer=optimizer,
        decay_rate=SHGO_DECAY_RATE,
        generations=PYGMO_VERIFY_GENERATIONS,
        population_size=PYGMO_VERIFY_POPULATION_SIZE,
        local_samples=(
            PYGMO_VERIFY_LOCAL_SAMPLES if local_samples is None else local_samples
        ),
        local_radius=PYGMO_VERIFY_LOCAL_RADIUS,
        fd_step=PYGMO_VERIFY_FD_STEP,
        rho=CLF_RHO,
        gamma1=CLF_GAMMA1,
        sublevel_cap=sublevel_cap,
        decay_tol=DECAY_TOL,
        root_penalty=SHGO_ROOT_PENALTY,
        random_seed=_numerical_verifier_seed(random_seed),
        ignore_vdot_ball_radius=SHGO_IGNORE_VDOT_BALL_RADIUS,
        ignore_vdot_abs_tol=SHGO_IGNORE_VDOT_ABS_TOL,
        origin_tol=CLF_ORIGIN_EXCLUDE_RADIUS,
    )


def _run_numerical_verification(
    individual, ind2MSE, consts, sublevel_cap=None, random_seed=None
):
    if NUMERICAL_VERIFIER is None:
        return None

    verifier = str(NUMERICAL_VERIFIER).lower()
    if verifier == "shgo":
        return _run_shgo_verification(
            ind2MSE,
            consts,
            sublevel_cap=sublevel_cap,
            random_seed=random_seed,
        )
    if verifier in {"cmaes", "de", "de1220", "de_1220", "pde"}:
        return _run_pygmo_verification(
            individual,
            consts,
            verifier,
            sublevel_cap=sublevel_cap,
            random_seed=random_seed,
        )
    raise ValueError(f"Unsupported NUMERICAL_VERIFIER: {NUMERICAL_VERIFIER}")


def _run_adaptive_falsifier(individual, ind2MSE, consts, sublevel_cap=None):
    """Adaptive DE falsifier (see FALSIFIER_* knobs).

    Returns (result, info) where result is the verification result whose
    margin was largest across the run(s), and info is a dict for the report.
    """
    r1 = _run_numerical_verification(
        individual, ind2MSE, consts, sublevel_cap=sublevel_cap
    )
    info = {
        "falsifier_stage": "normal",
        "falsifier_m1": (
            float(r1.decay_margin_max) if r1 is not None else np.nan
        ),
        "falsifier_m2": np.nan,
    }
    if r1 is None:
        return None, info

    verifier = str(NUMERICAL_VERIFIER).lower()
    m1 = r1.decay_margin_max
    if (
        not FALSIFIER_ADAPTIVE_ENABLED
        or verifier == "shgo"                      # escalation is pygmo-only
        or not np.isfinite(m1)                     # run 1 failed: keep it
        or m1 >= FALSIFIER_ESCALATE_BELOW          # gross violator
    ):
        return r1, info

    mult = (
        FALSIFIER_CLEAN_MULT
        if m1 < FALSIFIER_CLEAN_BELOW
        else FALSIFIER_MID_MULT
    )
    r2 = _run_pygmo_verification(
        individual,
        consts,
        verifier,
        sublevel_cap=sublevel_cap,
        local_samples=int(PYGMO_VERIFY_LOCAL_SAMPLES * mult),
    )
    m2 = r2.decay_margin_max if r2 is not None else np.nan
    info["falsifier_m2"] = float(m2) if np.isfinite(m2) else np.nan
    # monotone: keep whichever run found the bigger margin
    if r2 is not None and np.isfinite(m2) and m2 > m1:
        info["falsifier_stage"] = f"escalated_x{mult}_used_run2"
        return r2, info
    info["falsifier_stage"] = f"escalated_x{mult}_kept_run1"
    return r1, info


def _run_ibex_barrier_verification(ind2MSE, consts, timeout_seconds):
    """Build exact a, b expressions and run IBEX on a - rho*b**2."""
    V_expr = _sympy_expression(ind2MSE, consts)
    x_syms = symbols("x1:5")
    grad = Matrix([diff(V_expr, symbol) for symbol in x_syms])
    f_vec = fSR(*x_syms)
    G_mat = GSR(*x_syms)
    a_expr = (grad.T * f_vec)[0]
    b_expr = sum(grad[i] * G_mat[i, 1] for i in range(4))
    return verify_clf_ibex_barrier(
        a_expr,
        b_expr,
        SHGO_BOUNDS,
        CLF_RHO,
        origin_radius=CLF_ORIGIN_EXCLUDE_RADIUS,
        timeout_seconds=timeout_seconds,
        verbose=IBEX_VERIFIER_VERBOSE,
    )


def _run_adaptive_ibex_barrier(ind2MSE, consts):
    """Escalate only clean IBEX runs: T, then 2T, then 3T per box."""
    timeout0 = float(IBEX_VERIFIER_TIMEOUT_SECONDS)
    results = [_run_ibex_barrier_verification(ind2MSE, consts, timeout0)]
    stage = "T"

    if np.isfinite(results[-1].margin_max) and results[-1].margin_max < 0.0:
        results.append(_run_ibex_barrier_verification(ind2MSE, consts, 2.0 * timeout0))
        stage = "2T"
        if np.isfinite(results[-1].margin_max) and results[-1].margin_max < 0.0:
            results.append(
                _run_ibex_barrier_verification(ind2MSE, consts, 5.0 * timeout0)
            )
            stage = "5T"

    # Keep the greatest feasible margin found across all runs. A larger value
    # is the stronger counterexample and therefore the only one relevant to
    # the margin-only penalty.
    result = max(
        results,
        key=lambda item: item.margin_max if np.isfinite(item.margin_max) else -np.inf,
    )
    return result, {
        "ibex_stage": stage,
        "ibex_runs": len(results),
        "ibex_timeouts_seconds": [item.timeout_seconds for item in results],
        "ibex_stage_margins": [item.margin_max for item in results],
    }


def _state_values(true_data):
    return [true_data.get_input(i) for i in range(4)]


def _ensure_mesh(true_data, x_vals):
    mesh = getattr(true_data, "mesh", None)
    if mesh is None or len(mesh) != 4:
        mesh = np.meshgrid(*x_vals, indexing="ij")
        true_data.mesh = mesh
        true_data.grid_shape = mesh[0].shape
        true_data.X1, true_data.X2, true_data.X3, true_data.X4 = mesh
    return mesh


def _evaluate_v(individual, mesh, consts):
    V_vals = individual(*mesh, consts)
    V_val_0 = individual(0.0, 0.0, 0.0, 0.0, consts)

    if (not np.isfinite(V_val_0)) or (not np.all(np.isfinite(V_vals))):
        return None, None, "invalid_V"
    if np.max(np.abs(V_vals)) > V_MAGNITUDE_LIMIT:
        return None, None, "V_magnitude_too_large"

    V_vals = np.asarray(V_vals)
    if V_vals.ndim == 0:
        V_vals = np.full(mesh[0].shape, V_vals)
    return V_vals, V_val_0, "ok"


def _reference_quadratic(X1, X2, X3, X4):
    return (

        # (X1 + X2 + X3 + X4)**2

        
        1.821 * X1**2
        + 2.318 * X1 * X2
        - 12.932 * X1 * X3
        - 4.837 * X1 * X4
        + 1.64 * X2**2
        - 18.72 * X2 * X3
        - 6.96 * X2 * X4
        + 56.19 * X3**2
        + 40.81 * X3 * X4
        + 7.612 * X4**2
        
    )

def _reference_quadratic_lower(X1, X2, X3, X4):
    return (

        (X1 + X2 + X3 + X4)**2

        
        # 1.821 * X1**2
        # + 2.318 * X1 * X2
        # - 12.932 * X1 * X3
        # - 4.837 * X1 * X4
        # + 1.64 * X2**2
        # - 18.72 * X2 * X3
        # - 6.96 * X2 * X4
        # + 56.19 * X3**2
        # + 40.81 * X3 * X4
        # + 7.612 * X4**2
        
    )


# def _subspace_structure_penalty(ind2MSE, consts):
#     """Require V restricted to important subspaces to still depend on
#     the corresponding nonzero coordinates.

#     This is a symbolic dependency check, not a proof of positivity.
#     It catches degenerate CLFs such as V = V(x3, x4), where
#     V(x1, x2, 0, 0) becomes identically constant/zero.
#     """
#     if not SUBSPACE_STRUCTURE_CHECK_ENABLED:
#         return 0.0

#     x1, x2, x3, x4 = symbols("x1 x2 x3 x4")

#     try:
#         expr = _sympy_expression(ind2MSE, consts)
#     except Exception:
#         return SUBSPACE_STRUCTURE_PENALTY

#     checks = (
#         # V(x1, x2, 0, 0) must depend on x1 and x2
#         # ("x1x2_plane", {x3: 0, x4: 0}, {"x1", "x2"}),

#         # V(0, 0, x3, x4) must depend on x3 and x4
#         # ("x3x4_plane", {x1: 0, x2: 0}, {"x3", "x4"}),

#         # Axis restrictions
#         # ("x1_axis", {x2: 0, x3: 0, x4: 0}, {"x1"}),
#         # ("x2_axis", {x1: 0, x3: 0, x4: 0}, {"x2"}),
#         # ("x3_axis", {x1: 0, x2: 0, x4: 0}, {"x3"}),
#         # ("x4_axis", {x1: 0, x2: 0, x3: 0}, {"x4"}),
#     )

#     penalty = 0.0

#     for _, substitutions, required_symbols in checks:
#         try:
#             restricted_expr = expr.subs(substitutions)
#             restricted_expr = DeapSimplifier(str(restricted_expr), should_print=False)
#             free_symbols = {str(s) for s in restricted_expr.free_symbols}

#             # Penalize if the restricted expression is constant/zero,
#             # or if it lost one of the required active coordinates.
#             if len(free_symbols) == 0:
#                 penalty += SUBSPACE_STRUCTURE_PENALTY
#                 continue

#             if not required_symbols.issubset(free_symbols):
#                 penalty += SUBSPACE_STRUCTURE_PENALTY

#         except Exception:
#             penalty += SUBSPACE_STRUCTURE_PENALTY

#     return penalty


# def _symbolic_structure_penalty(ind2MSE, consts):
#     penalty = 0.0

#     if detect_nested_function_calls(str(ind2MSE), "exp"):
#         penalty += 1e6
#     if detect_nested_function_calls(str(ind2MSE), "aq"):
#         penalty += 1e6

#     for symbol_name in ("x1", "x2", "x3", "x4"):
#         if not contains_symbol(str(ind2MSE), symbol_name):
#             penalty += 1e6

#     penalty += _subspace_structure_penalty(ind2MSE, consts)

#     return penalty


def _symbolic_structure_penalty(ind2MSE, consts):
    penalty = 0.0

    if detect_nested_function_calls(str(ind2MSE), "exp"):
        penalty += 1e6
    if detect_nested_function_calls(str(ind2MSE), "aq"):
        penalty += 1e6

    for symbol_name in ("x1", "x2", "x3", "x4"):
        if not contains_symbol(str(ind2MSE), symbol_name):
            penalty += 1e6

    return penalty


def eval_MSE_breakdown(
    individual, num_consts, toolbox, true_data, ind2MSE, consts, verify_shgo=True
):
    warnings.filterwarnings("ignore")

    report = {
        "status": "ok",
        "symbolic_structure_penalty": 0.0,
        "axis_gradient_penalty": 0.0,
        "axis_gradient_report": {},
        "proper_penalty": 0.0,
        "origin_probe_penalty": 0.0,
        "V_violations": 0.0,
        "V_dot_violations": 0.0,
        "invalid_penalty": 0.0,
        "origin_penalty": 0.0,
        "V_grad_mean": np.nan,
        "V_grad_x1x2_target_error": np.nan,
        "V_grad_x3x4_target_error": np.nan,
        "V_grad_penalty": 0.0,
        "grid_mse": 1e10,
        "numerical_verifier": NUMERICAL_VERIFIER,
        "shgo_ran": False,
        "shgo_penalty": 0.0,
        "shgo_counterexamples": 0,
        "shgo_V_counterexamples": 0,
        "shgo_Vdot_counterexamples": 0,
        "shgo_ignored_Vdot_counterexamples": 0,
        "shgo_V_counterexample_points": np.empty((0, 4)),
        "shgo_Vdot_counterexample_points": np.empty((0, 4)),
        "shgo_ignored_Vdot_counterexample_points": np.empty((0, 4)),
        "shgo_v_min": np.nan,
        "shgo_decay_margin_max": np.nan,
        "shgo_status": "not_run",
        "ibex_ran": False,
        "ibex_penalty": 0.0,
        "ibex_margin_max": np.nan,
        "ibex_margin_upper_bound": np.nan,
        "ibex_completed_boxes": 0,
        "ibex_total_boxes": 0,
        "ibex_status": "not_run",
        "ibex_stage": "not_run",
        "ibex_runs": 0,
        "ibex_timeouts_seconds": [],
        "ibex_stage_margins": [],
        # ROA coverage and unbounded closed-loop rollouts
        "c_star": np.nan,
        "c_star_penalty": 0.0,
        "roa_coverage_penalty": 0.0,
        "rollout_ran": False,
        "rollout_penalty": 0.0,
        "rollout_details": [],
        # "prime_grid_mse": 1e10,
        "final_mse": 1e10,
    }

    if not callable(individual):
        report["status"] = "not_callable"
        return report

    x_vals = _state_values(true_data)
    mesh = _ensure_mesh(true_data, x_vals)
    X1, X2, X3, X4 = mesh

    try:
        V_vals, V_val_0, status = _evaluate_v(individual, mesh, consts)
        if status != "ok":
            report["status"] = status
            return report

        report["symbolic_structure_penalty"] = _symbolic_structure_penalty(
            ind2MSE, consts
        )

        ref_vals = _reference_quadratic(X1, X2, X3, X4)
        # ref_vals_lower = 0.01 * _reference_quadratic_lower(X1, X2, X3, X4)
        upper_ref_bound = 50.0 * ref_vals #+ 1e-2
        lower_ref_bound = 0.1 * ref_vals #- 1e-2
        #  | (V_vals <= ref_vals_lower)
        
        # exclude the exact origin: ref(0)=0 makes the band [0,0] there, so
        # every candidate would fire once the (odd) grid contains x=0
        _r2_grid = X1**2 + X2**2 + X3**2 + X4**2
        report["proper_penalty"] = PROPERNESS_PENALTY_WEIGHT * float(
            np.count_nonzero(
                ((upper_ref_bound <= V_vals) | (V_vals <= lower_ref_bound))
                & (_r2_grid > 0)
            )
        )

        (
            _,
            V_grad_mag,
            V_dot_vals,
            lambda_vals,
            _,
            _,
            a_vals,
            _,
            x_norm_squared,
        ) = compute_v_and_v_dot(
            V_vals,
            x_vals,
            true_data,
            [1.0, 1.0, 1.0, 1.0],
            f,
            G,
            Q,
            R,
            TRQlim=None,
            u_func=None,
            TrnOFFb=True,
        )

        # relative positivity with margin (grid version of the pd_eps check):
        # V - V(0) must exceed 1e-4*||x||^2, so flat valleys score hundreds
        report["V_violations"] = float(
            np.count_nonzero(((V_vals - V_val_0) <= 1e-4 * _r2_grid) & (_r2_grid > 0))
        )
        # Decay-rate check on the min-norm Vdot (grid only): performance
        # requirement, as in the original framework. CLF validity is
        # checked separately by the verifier's barrier condition.
        report["V_dot_violations"] = float(
            np.count_nonzero(
                (V_dot_vals >= -SHGO_DECAY_RATE * x_norm_squared)
                & (x_norm_squared > 0)  # exact origin is trivially Vdot=0
            )
        )

        invalid_penalty = 0.0
        if np.any(np.isnan(V_vals)) or np.any(np.isinf(V_vals)):
            invalid_penalty += 1e6
        if np.any(np.isnan(a_vals)) or np.any(np.isinf(a_vals)):
            invalid_penalty += 1e6
        if np.any(np.isnan(V_dot_vals)) or np.any(np.isinf(V_dot_vals)):
            invalid_penalty += 1e6
        if np.any(np.isnan(lambda_vals)) or np.any(np.isinf(lambda_vals)):
            invalid_penalty += 1e6
        report["invalid_penalty"] = invalid_penalty

        report["origin_penalty"] = float(1000000 * (V_val_0**2 - 0))
        report["V_grad_mean"] = float(np.mean(V_grad_mag))

        V_grad_x1, V_grad_x2, V_grad_x3, V_grad_x4 = np.gradient(
            V_vals, *x_vals, edge_order=2
        )
        V_grad_x1x2 = np.mean(np.hypot(V_grad_x1, V_grad_x2))
        V_grad_x3x4 = np.mean(np.hypot(V_grad_x3, V_grad_x4))
        report["V_grad_x1x2_target_error"] = float(
            abs(V_grad_x1x2 - V_GRAD_X1X2_TARGET)
        )
        report["V_grad_x3x4_target_error"] = float(
            abs(V_grad_x3x4 - V_GRAD_X3X4_TARGET)
        )
        report["V_grad_penalty"] = V_GRAD_WEIGHT * (
            report["V_grad_x1x2_target_error"]
            + report["V_grad_x3x4_target_error"]
        )

        origin_probe_pen, origin_probe_ratio = _origin_probe_penalty(
            individual, consts, V_val_0, n_states=len(x_vals)
        )
        report["origin_probe_penalty"] = float(origin_probe_pen)
        report["origin_probe_ratio"] = float(origin_probe_ratio)

        grid_mse = (
            report["symbolic_structure_penalty"]
            + report["proper_penalty"]
            + report["V_violations"]
            + report["V_dot_violations"]
            + report["invalid_penalty"]
            + report["origin_penalty"]
            + report["V_grad_penalty"]
            + report["origin_probe_penalty"]
        )
        report["grid_mse"] = float(grid_mse)

        if verify_shgo and grid_mse < SHGO_MSE_GATE:
            shgo_result, falsifier_info = _run_adaptive_falsifier(
                individual,
                ind2MSE,
                consts,
                sublevel_cap=None, #_sublevel_cap_from_grid(V_vals, V_val_0),
            )
            report.update(falsifier_info)
            report["shgo_ran"] = True
            if shgo_result is not None:
                shgo_pen = float(shgo_result.penalty)
                if np.isfinite(shgo_result.decay_margin_max):
                    shgo_pen += SHGO_MARGIN_WEIGHT * max(
                        shgo_result.decay_margin_max, 0.0
                    )
                if np.isfinite(shgo_result.v_min):
                    shgo_pen += SHGO_MARGIN_WEIGHT * max(-shgo_result.v_min, 0.0)
                report["shgo_penalty"] = shgo_pen
                report["shgo_counterexamples"] = len(shgo_result.counterexamples)
                report["shgo_V_counterexamples"] = len(shgo_result.v_counterexamples)
                report["shgo_Vdot_counterexamples"] = len(
                    shgo_result.vdot_counterexamples
                )
                report["shgo_ignored_Vdot_counterexamples"] = len(
                    shgo_result.ignored_vdot_counterexamples
                )
                report["shgo_V_counterexample_points"] = shgo_result.v_counterexamples
                report["shgo_Vdot_counterexample_points"] = (
                    shgo_result.vdot_counterexamples
                )
                report["shgo_ignored_Vdot_counterexample_points"] = (
                    shgo_result.ignored_vdot_counterexamples
                )
                report["shgo_v_min"] = float(shgo_result.v_min)
                report["shgo_decay_margin_max"] = float(
                    shgo_result.decay_margin_max
                )
                report["shgo_status"] = shgo_result.status
            else:
                report["shgo_status"] = "disabled"
        elif verify_shgo:
            report["shgo_status"] = "skipped_by_mse_gate"

        # --- IBEX global barrier falsifier (margin penalty only) ---
        if (
            verify_shgo
            and IBEX_VERIFIER_ENABLED
            and grid_mse < SHGO_MSE_GATE
        ):
            report["ibex_ran"] = True
            try:
                ibex_result, ibex_info = _run_adaptive_ibex_barrier(
                    ind2MSE, consts
                )
                report.update(ibex_info)
                report["ibex_margin_max"] = float(ibex_result.margin_max)
                report["ibex_margin_upper_bound"] = float(
                    ibex_result.margin_upper_bound
                )
                report["ibex_completed_boxes"] = int(ibex_result.n_complete)
                report["ibex_total_boxes"] = int(ibex_result.n_boxes)
                report["ibex_status"] = ibex_result.status
                # Unlike DE/SHGO, IBEX contributes neither root nor count
                # penalties. A penalty is applied only for a positive
                # feasible barrier margin, i.e. a concrete counterexample.
                if np.isfinite(ibex_result.margin_max):
                    report["ibex_penalty"] = 20 * SHGO_MARGIN_WEIGHT * max(
                        ibex_result.margin_max, 0.0
                    )
            except Exception as exc:
                report["ibex_status"] = f"error: {type(exc).__name__}: {exc}"
        elif verify_shgo and IBEX_VERIFIER_ENABLED:
            report["ibex_status"] = "skipped_by_mse_gate"
        elif not IBEX_VERIFIER_ENABLED:
            report["ibex_status"] = "disabled"

        # --- ROA coverage and unbounded closed-loop rollouts ---
        c_star = _sublevel_cap_from_grid(V_vals, V_val_0)
        report["c_star"] = float(c_star)
        raw_min = clf_checks.boundary_min(V_vals - V_val_0)      # NOT clamped
        if not np.isfinite(raw_min):
            c_star_penalty = 10.0 * C_STAR_PENALTY_MAX
        else:
            shortfall = max(0.0, (C_STAR_TARGET - raw_min) / C_STAR_TARGET)  # > 1 when raw < 0
            c_star_penalty = C_STAR_PENALTY_MAX * min(shortfall, 10.0) ** 2

        # if not np.isfinite(c_star) or c_star <= 0:
        #     c_star_penalty = C_STAR_PENALTY_MAX
        # else:
        #     c_star_shortfall = max(0.0, (C_STAR_TARGET - c_star) / C_STAR_TARGET)
        #     c_star_penalty = C_STAR_PENALTY_MAX * c_star_shortfall**2
        report["c_star_penalty"] = float(c_star_penalty)

        V_call = lambda x1, x2, x3, x4: individual(
            x1, x2, x3, x4, consts
        )

        coverage = 0
        for x0 in ROLLOUT_X0S:
            V_x0 = float(np.asarray(V_call(*x0)))
            if not np.isfinite(V_x0) or (V_x0 - V_val_0) >= c_star:
                coverage += 1
        report["roa_coverage_penalty"] = ROA_COVERAGE_WEIGHT * coverage

        grid_mse = report["grid_mse"]

        rollout_penalty = 0.0
        if verify_shgo:
            pre_rollout = (
                grid_mse + report["shgo_penalty"] + report["ibex_penalty"]
                + report["c_star_penalty"]
                + report["roa_coverage_penalty"]
            )
            if ROLLOUT_ENABLED and pre_rollout < ROLLOUT_MSE_GATE:
                report["rollout_ran"] = True
                rollout_penalty, rollout_details = _rollout_penalty(V_call, V_val_0)
                report["rollout_penalty"] = rollout_penalty
                report["rollout_details"] = rollout_details

        report["final_mse"] = float(
            grid_mse
            + report["shgo_penalty"]
            + report["ibex_penalty"]
            + rollout_penalty
            + report["c_star_penalty"]
            + report["roa_coverage_penalty"]
        )

        # --- gated b=0 manifold check: fitness can only drop below the
        # gate by passing the exact Artstein condition on the solved roots
        report["manifold_ran"] = False
        report["manifold_penalty"] = 0.0
        if MANIFOLD_CHECK_ENABLED:
            if report["final_mse"] < MANIFOLD_MSE_GATE:
                man = check_b_manifold(
                    individual,
                    consts,
                    f,
                    _G_col_vec,
                    bounds=SHGO_BOUNDS,
                    gamma1=CLF_GAMMA1,
                    margin_tol=MANIFOLD_MARGIN_TOL_FD,
                    fd_step=PYGMO_VERIFY_FD_STEP,
                    origin_tol=CLF_ORIGIN_EXCLUDE_RADIUS,
                    # all four scan axes: catches b=0 branches that hug the
                    # x3/x4 corners and vary along x1/x2 (endpoint roots
                    # produce no sign-change bracket on the default axes)
                    scan_axes=(0, 1, 2, 3),
                )
                report["manifold_status"] = man.status
                if man.status == "ok":
                    report["manifold_ran"] = True
                    report["manifold_roots"] = man.n_roots
                    report["manifold_violations"] = man.n_violations
                    report["manifold_margin_max"] = man.margin_max
                    report["manifold_violation_points"] = man.violation_points
                    if man.n_violations > 0:
                        report["manifold_penalty"] = float(
                            MANIFOLD_ROOT_PENALTY
                            + MANIFOLD_CEX_WEIGHT * man.n_violations
                            + _manifold_margin_penalty(
                                man.margin_max,
                                MANIFOLD_MARGIN_WEIGHT,
                                MANIFOLD_MARGIN_PENALTY_MAX,
                            )
                        )
                    elif man.n_roots == 0:
                        # no manifold found: the check proved nothing
                        report["manifold_vacuous"] = True
                        report["manifold_penalty"] = float(
                            MANIFOLD_VACUOUS_PENALTY
                        )
                    report["final_mse"] = float(
                        report["final_mse"] + report["manifold_penalty"]
                    )
                else:
                    # check failed to run: treat as unchecked
                    report["final_mse"] += float(MANIFOLD_MSE_GATE) * 5
            else:
                # not checked: never report below the gate
                report["final_mse"] += float(MANIFOLD_MSE_GATE) * 4

        # --- exact-gradient (symbolic) manifold check: final confirmation
        # layer for near-feasible candidates; penalty capped
        report["manifold_exact_ran"] = False
        report["manifold_exact_penalty"] = 0.0
        if (
            MANIFOLD_EXACT_CHECK_ENABLED
            and report["final_mse"] < MANIFOLD_EXACT_MSE_GATE
        ):
            try:
                mex = check_b_manifold_exact(
                    _sympy_expression(ind2MSE, consts),
                    fSR,
                    GSR,
                    bounds=SHGO_BOUNDS,
                    gamma1=CLF_GAMMA1,
                    margin_tol=MANIFOLD_MARGIN_TOL_EXACT,
                    origin_tol=CLF_ORIGIN_EXCLUDE_RADIUS,
                    scan_axes=(0, 1, 2, 3),
                )
                report["manifold_exact_status"] = mex.status
            except Exception as exc:
                report["manifold_exact_status"] = f"error: {exc}"
                mex = None
            if mex is not None and mex.status == "ok":
                report["manifold_exact_ran"] = True
                report["manifold_exact_roots"] = mex.n_roots
                report["manifold_exact_violations"] = mex.n_violations
                report["manifold_exact_margin_max"] = mex.margin_max
                report["manifold_exact_violation_points"] = mex.violation_points
                if mex.n_violations > 0:
                    report["manifold_exact_penalty"] = float(
                        min(
                            MANIFOLD_EXACT_PENALTY_MAX,
                            100.0
                            + mex.n_violations * 0.01
                            + _manifold_margin_penalty(
                                mex.margin_max,
                                MANIFOLD_EXACT_MARGIN_WEIGHT,
                                MANIFOLD_EXACT_MARGIN_PENALTY_MAX,
                            ),
                        )
                    )
                elif mex.n_roots == 0:
                    # no manifold found: the check proved nothing
                    report["manifold_exact_vacuous"] = True
                    report["manifold_exact_penalty"] = float(
                        MANIFOLD_VACUOUS_PENALTY
                    )
                report["final_mse"] = float(
                    report["final_mse"] + report["manifold_exact_penalty"]
                )

        if np.isnan(report["final_mse"]) or np.isinf(report["final_mse"]):
            report["status"] = "nonfinite_mse"
            report["final_mse"] = 1e10
        return report

    except Exception as e:
        report["status"] = f"error: {e}"
        report["final_mse"] = 1e10
        return report


def eval_MSE_sol(
    individual,
    num_consts,
    toolbox,
    true_data,
    ind2MSE,
    consts,
    verify_shgo=False,
    pass_cost=None,
):
    warnings.filterwarnings("ignore")

    report = eval_MSE_breakdown(
        individual,
        num_consts,
        toolbox,
        true_data,
        ind2MSE,
        consts,
        verify_shgo=verify_shgo,
    )
    V_vals = None
    if report["status"] == "ok":
        x_vals = _state_values(true_data)
        mesh = _ensure_mesh(true_data, x_vals)
        V_vals, _, _ = _evaluate_v(individual, mesh, consts)

    return report["final_mse"], V_vals


def eval_MSE_and_tune_constants(
    individual, num_consts, toolbox, true_data, ind2MSE, pass_cost=None
):
    if num_consts > 0:
        x0 = np.ones(num_consts)

        class fitting_problem:
            def fitness(self, x):
                total_err, _ = eval_MSE_sol(
                    individual,
                    num_consts,
                    toolbox,
                    true_data,
                    ind2MSE,
                    x,
                    verify_shgo=False,
                    pass_cost=pass_cost,
                )
                return [total_err]

            def get_bounds(self):
                return (-10.0 * np.ones(num_consts), 10.0 * np.ones(num_consts))

        prb = pg.problem(fitting_problem())
        # algo = pg.algorithm(pg.pso(gen=10))
        algo = pg.algorithm(pg.sea(gen=10))
        pop = pg.population(prb, size=70)
        pop.push_back(x0)
        pop = algo.evolve(pop)
        consts = pop.champion_x
        MSE, _ = eval_MSE_sol(
            individual,
            num_consts,
            toolbox,
            true_data,
            ind2MSE,
            consts,
            verify_shgo=True,
            pass_cost=pass_cost,
        )
    else:
        MSE, _ = eval_MSE_sol(
            individual,
            num_consts,
            toolbox,
            true_data,
            ind2MSE,
            consts=[],
            verify_shgo=True,
            pass_cost=pass_cost,
        )
        consts = []
    return MSE, consts
