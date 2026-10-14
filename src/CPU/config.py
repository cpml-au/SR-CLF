"""The settings of the check and of the fitness rule, read once from the environment when imported.

The pipeline runs any system xdot = f(x) + G(x) u with n states and m inputs that is defined by a file
systems/<name>.py (fSR, GSR, N_STATES, N_INPUTS, BOX_HALF_WIDTHS, Q_MATRIX, R_MATRIX).  Every setting is
an environment variable (src/settings.py writes them from config.yaml).  The number behind each name
below is the default the code uses when the variable is not set; examples/<name>/config.yaml holds the
values of the runs.

Tolerances: PD_EPS 1e-4 (positive definite iff c_v* > PD_EPS), KAPPA_MIN 0.0 (numerically
verified iff also kappa* > KAPPA_MIN), PD_RATE_TOL 1e-6, B_TOL 1e-10, B_REL_TOL 1e-6,
GATE_KAPPA_CAP 1e6.
Settings read here (src/settings.py KEYMAP): system.name, system.box, system.template,
system.template_x0, grid.points_per_axis, verification.u_max, verification.b_norm,
verification.pd_eps, verification.kappa_min, verification.b_rel_tol, verification.project_ascent,
verification.origin_seeds, verification.kappa_norm, verification.lattice.*, fitness.region_weight,
fitness.region_target, fitness.hinge.*, fitness.non_pd_kappa_cap, fitness.length_reg,
fitness.nested_penalty, fitness.non_pd_weight.
Environment only (no key in config.yaml): SYMCLF_CHECK_KAPPA_SCALE, SYMCLF_CLF_GAMMA1,
SYMCLF_CHECK_B_TOL, SYMCLF_CHECK_POLISH_TOP_K, SYMCLF_CHECK_POLISH_MAXITER, SYMCLF_CHECK_SOBOL_SEEDS,
SYMCLF_CHECK_PD_RATE_TOL, SYMCLF_FIT_ROA_RHO_MULTIPLIER.
"""
from __future__ import annotations

import os

import numpy as np


def _f(name, default):
    raw = os.environ.get(name)
    return float(default if raw is None or raw == "" else raw)


def _i(name, default):
    raw = os.environ.get(name)
    return int(default if raw is None or raw == "" else raw)


def _s(name, default):
    raw = os.environ.get(name)
    return str(default if raw is None or raw == "" else raw)


# --- which system -----------------------------------------------------------
# system.name: the file systems/<name>.py.  The default is quad6d, the quadrotor of examples/quadrotor6d.
SYSTEM = _s("SYMCLF_CPU_SYSTEM", "quad6d")

# --- the settings of the check ---
RHO = _f("SYMCLF_CHECK_RHO", 1000.0)              # per-input actuator bound |u_i| <= RHO
PD_EPS = _f("SYMCLF_CHECK_PD_EPS", 1e-4)          # PD iff c_v* > PD_EPS
KAPPA_MIN = _f("SYMCLF_CHECK_KAPPA_MIN", 0.0)     # numerically verified iff also kappa* > KAPPA_MIN
KAPPA_SCALE = _f("SYMCLF_CHECK_KAPPA_SCALE", 1.0)
GAMMA1 = _f("SYMCLF_CLF_GAMMA1", 0.0)
GATE_KAPPA_CAP = _f("SYMCLF_CHECK_GATE_KAPPA_CAP", 1e6)
B_TOL = _f("SYMCLF_CHECK_B_TOL", 1e-10)
B_REL_TOL = _f("SYMCLF_CHECK_B_REL_TOL", 1e-6)
PROJECT_ASCENT = _i("SYMCLF_CHECK_PROJECT_ASCENT", 1) == 1
ORIGIN_SEEDS = _i("SYMCLF_CHECK_ORIGIN_SEEDS", 1) == 1
# What kappa* divides by (verification.kappa_norm):
#   "w"   -> W = V - V(0): kappa* = -max (a - rho||b||)/W  (the default)
#   "xqx" -> x'Qx with the system's Q_MATRIX: kappa* = -max (a - rho||b||)/x'Qx, the margin
#            a - rho||b|| <= -k x'Qx; where x'Qx = 0 only a - rho||b|| <= 0 is required: a violation
#            iff a - rho||b|| > XQX_NULL_TOL * (1 + |a| + rho||b||), counted as the ratio 1e30.
KAPPA_NORM = _s("SYMCLF_CHECK_KAPPA_NORM", "w").strip().lower()
if KAPPA_NORM not in ("w", "xqx"):
    raise ValueError(f"SYMCLF_CHECK_KAPPA_NORM must be w or xqx, not {KAPPA_NORM!r}")
XQX_NULL_TOL = 1e-12
ORIGIN_RADII = (1e-3, 1e-2, 3e-2, 1e-1)
POLISH_TOP_K = _i("SYMCLF_CHECK_POLISH_TOP_K", 40)
POLISH_MAXITER = _i("SYMCLF_CHECK_POLISH_MAXITER", 60)
SOBOL_SEEDS = _i("SYMCLF_CHECK_SOBOL_SEEDS", 64)
PD_RATE_TOL = _f("SYMCLF_CHECK_PD_RATE_TOL", 1e-6)   # the check's flag pd_valid: c_v* > PD_EPS + PD_RATE_TOL

# b = grad(V)' G is a VECTOR with one entry per input.  The rate ratio of the margins needs the
# best decrease a bounded actuator can buy, min_u (a + b.u):
#   l1 : |u_i| <= RHO per input (the clip)    -> a - RHO * sum_i |b_i|
#   l2 : ||u||_2 <= RHO (a round bound)       -> a - RHO * ||b||_2
# Single-input systems give the same number either way.
B_NORM = _s("SYMCLF_CPU_B_NORM", "l1")

# --- the fitness rule (src/CPU/rule.py) ---
ROA_WEIGHT = _f("SYMCLF_FIT_ROA_WEIGHT", 500.0)
ROA_C_TARGET = _f("SYMCLF_FIT_ROA_C_TARGET", 0.004)
ROA_RHO_MULTIPLIER = _f("SYMCLF_FIT_ROA_RHO_MULTIPLIER", 1.0)
PEN_CV = _f("SYMCLF_CHECK_PEN_CV", 10.0)
PEN_CV_TARGET = _f("SYMCLF_CHECK_PEN_CV_TARGET", 0.05)
PEN_KAPPA = _f("SYMCLF_CHECK_PEN_KAPPA", 30.0)
PEN_KAPPA_TARGET = _f("SYMCLF_CHECK_PEN_KAPPA_TARGET", 0.05)
PEN_VOL = _f("SYMCLF_CHECK_PEN_VOL", 20.0)
PEN_VOL_TARGET = _f("SYMCLF_CHECK_PEN_VOL_TARGET", 0.02)
DEGENERATE_PENALTY = 1.0e6
CHECK_FAILED_PENALTY = 1.0e5
LENGTH_REG = _f("SYMCLF_LENGTH_REG", 0.005)      # fitness.length_reg
NESTED_PENALTY = _f("SYMCLF_FIT_NESTED_PENALTY", 50.0)
# fitness.non_pd_weight: a graded charge for a tree that is not positive definite, on top of the rule:
# fitness += NON_PD_WEIGHT * f, f = the fraction of the check's scan points (x != 0) where
# min(W/|x|^2, x.grad V/(2|x|^2)) <= pd_eps + pd_rate_tol (the c_v* test at one point).  f = 0 for a
# positive-definite tree, so its fitness is unchanged; shrinking V towards 0 raises f.  0 = off (the default).
NON_PD_WEIGHT = _f("SYMCLF_FIT_NON_PD_WEIGHT", 0.0)

# --- the two point sets -----------------------------------------------------
# GRID_POINTS: the training grid (odd, so 0 is a point on every axis).
# The scan points of the check (src/CPU/geometry.py::scan_geometry): mesh^n + n * lines^(n-1) * samples
# + random_lines * random_line_points.
GRID_POINTS = _i("SYMCLF_CPU_GRID_POINTS", 7)
LATTICE_MESH = _i("SYMCLF_CPU_LATTICE_MESH", 7)
LATTICE_LINES = _i("SYMCLF_CPU_LATTICE_LINES", 3)
LATTICE_SAMPLES = _i("SYMCLF_CPU_LATTICE_SAMPLES", 801)
LATTICE_RANDOM_LINES = _i("SYMCLF_CPU_LATTICE_RANDOM_LINES", 200)
LATTICE_RANDOM_POINTS = _i("SYMCLF_CPU_LATTICE_RANDOM_POINTS", 401)
LATTICE_SEED = _i("SYMCLF_CPU_LATTICE_SEED", 0)

_CARTPOLE_DEFAULTS = dict(grid=21, mesh=21, lines=9, samples=801)


def system():
    """Import the selected system module (systems/<name>.py)."""
    import importlib

    return importlib.import_module(f"systems.{SYSTEM}")


def lattice_kwargs():
    """The arguments of scan_geometry.  For cartpole4d the numbers are mesh 21, lines 9, samples 801
    unless SYMCLF_CPU_LATTICE_MESH is set."""
    if SYSTEM == "cartpole4d" and "SYMCLF_CPU_LATTICE_MESH" not in os.environ:
        return dict(
            mesh_points_per_axis=_CARTPOLE_DEFAULTS["mesh"],
            grid_points_per_axis=_CARTPOLE_DEFAULTS["lines"],
            scan_points=_CARTPOLE_DEFAULTS["samples"],
            random_lines=LATTICE_RANDOM_LINES,
            random_line_points=LATTICE_RANDOM_POINTS,
            rng_seed=LATTICE_SEED,
        )
    return dict(
        mesh_points_per_axis=LATTICE_MESH,
        grid_points_per_axis=LATTICE_LINES,
        scan_points=LATTICE_SAMPLES,
        random_lines=LATTICE_RANDOM_LINES,
        random_line_points=LATTICE_RANDOM_POINTS,
        rng_seed=LATTICE_SEED,
    )


def grid_points():
    if SYSTEM == "cartpole4d" and "SYMCLF_CPU_GRID_POINTS" not in os.environ:
        return _CARTPOLE_DEFAULTS["grid"]
    return GRID_POINTS


def box():
    """Per-axis half-widths of the state box."""
    sys_mod = system()
    raw = os.environ.get("SYMCLF_BOX")
    if raw:
        half = tuple(float(v) for v in raw.split(","))
    else:
        half = tuple(float(v) for v in sys_mod.BOX_HALF_WIDTHS)
    if len(half) != sys_mod.N_STATES:
        raise ValueError(f"SYMCLF_BOX needs {sys_mod.N_STATES} half-widths, got {half}")
    return half


def bounds():
    return np.array([[-h, h] for h in box()], dtype=float)


# --- the CMA template -------------------------------------------------------
# full : all n(n+1)/2 monomials (21 for n = 6)
# are  : only the monomials the Riccati solution uses (9 for quad6d)
TEMPLATE_KIND = _s("SYMCLF_CPU_TEMPLATE", "full")
TEMPLATE_X0 = _f("SYMCLF_CPU_TEMPLATE_X0", 1.0)   # every coefficient starts here; never from a Riccati solution
