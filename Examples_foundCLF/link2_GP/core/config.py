"""Configuration of the self-contained 6-D CLF/CMA folder.

Every knob is an environment variable with the 4-D V8 default where one exists, so a run is
reproducible from the job script alone.  Nothing here imports the 4-D tree.
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
# quad6d      : the indoor micro quadrotor's angular subsystem (6 states, 3 inputs)
# cartpole4d  : the 4-D cart-pole of 4DCartPolerV8 -- the ACCEPTANCE system, so that this
#               folder can reproduce job 146800's published numbers with its own code.
SYSTEM = _s("SYMCLF6D_SYSTEM", "quad6d")

# --- DET4 knobs (V8 job script values) --------------------------------------
RHO = _f("SYMCLF_DET4_RHO", 1000.0)              # per-input actuator bound |u_i| <= RHO
PD_EPS = _f("SYMCLF_DET4_PD_EPS", 1e-4)          # PD iff c_v* > PD_EPS
KAPPA_MIN = _f("SYMCLF_DET4_KAPPA_MIN", 0.0)     # certified iff kappa* > KAPPA_MIN
KAPPA_SCALE = _f("SYMCLF_DET4_KAPPA_SCALE", 1.0)
GAMMA1 = _f("SYMCLF_CLF_GAMMA1", 0.0)
GATE_KAPPA_CAP = _f("SYMCLF_DET4_GATE_KAPPA_CAP", 1e6)
B_TOL = _f("SYMCLF_DET4_B_TOL", 1e-10)
B_REL_TOL = _f("SYMCLF_DET4_B_REL_TOL", 1e-6)
PROJECT_ASCENT = _i("SYMCLF_DET4_PROJECT_ASCENT", 1) == 1
ORIGIN_SEEDS = _i("SYMCLF_DET4_ORIGIN_SEEDS", 1) == 1
ORIGIN_RADII = (1e-3, 1e-2, 3e-2, 1e-1)
POLISH_TOP_K = _i("SYMCLF_DET4_POLISH_TOP_K", 40)
POLISH_MAXITER = _i("SYMCLF_DET4_POLISH_MAXITER", 60)
SOBOL_SEEDS = _i("SYMCLF_DET4_SOBOL_SEEDS", 64)
PD_RATE_TOL = _f("SYMCLF_DET4_PD_RATE_TOL", 1e-6)   # srcCPU/det4_check_cpu.PD_RATE_TOL

# b = grad(V)' G is a VECTOR with one entry per input.  The DET4 ratio needs the best decrease a
# bounded actuator can buy, min_u (a + b.u):
#   l1 : |u_i| <= RHO per input (V8's clip)   -> a - RHO * sum_i |b_i|
#   l2 : ||u||_2 <= RHO (a round bound)       -> a - RHO * ||b||_2
# Single-input systems give the same number either way, so the 4-D acceptance is unaffected.
B_NORM = _s("SYMCLF6D_B_NORM", "l1")

# --- price rule (V6/V8 weights) ---------------------------------------------
ROA_WEIGHT = _f("SYMCLF_CP3D_ROA_WEIGHT", 500.0)
ROA_C_TARGET = _f("SYMCLF_CP3D_ROA_C_TARGET", 0.004)
ROA_RHO_MULTIPLIER = _f("SYMCLF_V2_ROA_RHO_MULTIPLIER", 1.0)
PEN_CV = _f("SYMCLF_DET4_PEN_CV", 10.0)
PEN_CV_TARGET = _f("SYMCLF_DET4_PEN_CV_TARGET", 0.05)
PEN_KAPPA = _f("SYMCLF_DET4_PEN_KAPPA", 30.0)
PEN_KAPPA_TARGET = _f("SYMCLF_DET4_PEN_KAPPA_TARGET", 0.05)
PEN_VOL = _f("SYMCLF_DET4_PEN_VOL", 20.0)
PEN_VOL_TARGET = _f("SYMCLF_DET4_PEN_VOL_TARGET", 0.02)
DEGENERATE_PENALTY = 1.0e6
EXACT_PENALTY_MAX = 1.0e5
LENGTH_REG = _f("SYMCLF_LENGTH_REG", 0.005)      # config.yaml penalty.reg_param
NESTED_PENALTY = _f("SYMCLF_CP3D_NESTED_PENALTY", 50.0)

# --- the two lattices -------------------------------------------------------
# GRID_POINTS: the training grid (odd, so 0 is a sample on every axis), V8 4-D used 21^4.
# The DET4 scan lattice is the referee's scan_geometry: mesh^n + n * lines^(n-1) * samples
# + random_lines * random_line_points.
GRID_POINTS = _i("SYMCLF6D_GRID_POINTS", 7)
LATTICE_MESH = _i("SYMCLF6D_LATTICE_MESH", 7)
LATTICE_LINES = _i("SYMCLF6D_LATTICE_LINES", 3)
LATTICE_SAMPLES = _i("SYMCLF6D_LATTICE_SAMPLES", 801)
LATTICE_RANDOM_LINES = _i("SYMCLF6D_LATTICE_RANDOM_LINES", 200)
LATTICE_RANDOM_POINTS = _i("SYMCLF6D_LATTICE_RANDOM_POINTS", 401)
LATTICE_SEED = _i("SYMCLF6D_LATTICE_SEED", 0)

_CARTPOLE_DEFAULTS = dict(grid=21, mesh=21, lines=9, samples=801)


def system():
    """Import the selected system module (systems/<name>.py)."""
    import importlib

    return importlib.import_module(f"systems.{SYSTEM}")


def lattice_kwargs():
    """scan_geometry knobs; the 4-D acceptance system keeps V8's referee lattice."""
    if SYSTEM == "cartpole4d" and "SYMCLF6D_LATTICE_MESH" not in os.environ:
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
    if SYSTEM == "cartpole4d" and "SYMCLF6D_GRID_POINTS" not in os.environ:
        return _CARTPOLE_DEFAULTS["grid"]
    return GRID_POINTS


def box():
    """Per-axis half-widths of the state box."""
    sys_mod = system()
    raw = os.environ.get("SYMCLF_V2_BOX")
    if raw:
        half = tuple(float(v) for v in raw.split(","))
    else:
        half = tuple(float(v) for v in sys_mod.BOX_HALF_WIDTHS)
    if len(half) != sys_mod.N_STATES:
        raise ValueError(f"SYMCLF_V2_BOX needs {sys_mod.N_STATES} half-widths, got {half}")
    return half


def bounds():
    return np.array([[-h, h] for h in box()], dtype=float)


# --- the CMA template -------------------------------------------------------
# full : all n(n+1)/2 monomials (21 in 6-D) -- the literal extension of the 4-D seed_expr2
# are  : only the monomials the ARE solution uses (9 in 6-D)
TEMPLATE_KIND = _s("SYMCLF6D_TEMPLATE", "full")
TEMPLATE_X0 = _f("SYMCLF6D_TEMPLATE_X0", 1.0)   # no LQR/ARE initialisation (standing rule)
