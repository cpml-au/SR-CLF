"""The check-cost rule of the CPU pipeline and the fitness of a candidate at fixed constants.

    fitness            = grid_score + check_cost + complexity_penalty
    check_cost         = roa_rule(c_v*, kappa*, grid anchors) - the grid's own ROA term (margins_roa)
    complexity_penalty = NESTED_PENALTY * (nested trig + nested exp + nested aq) + LENGTH_REG * n_nodes
                         (0/1 flags: the formula the GP charges, src/CPU/gp_engine.py::_extra)
    roa_rule, numerically verified: ROA_WEIGHT * clip(1 - boundary_min/(w_scale*ROA_C_TARGET), 0, 1)
                                    + hinges
              otherwise:            ROA_WEIGHT * (1 + log1p(max(-kappa*, 0)/KAPPA_SCALE))
                                    + 2 ROA_WEIGHT * max(1 - c_v*/PD_EPS, 0) + hinges
    hinges             = sum of weight * clip((target - value)/target, 0, 1) over c_v*, kappa*
                         and the grid's sublevel_fraction

grid_score and the anchors (boundary_min, w_scale, sublevel_fraction) come from the training-grid
stage (src/CPU/lattice.py), c_v* and kappa* from the check (src/CPU/deterministic_search.py); the
complexity penalty is the `extra` argument.  A dead tree (symbolic or invalid >= 1e6, grid_score
not finite or >= 1e10) pays grid_score + complexity_penalty and skips the check.

Numerically verified iff c_v* > PD_EPS and kappa* > KAPPA_MIN.  That is the result of a search: no
violation was found.  It is not a proof.
Tolerances: PD_EPS 1e-4, KAPPA_MIN 0.0; a candidate that is not positive definite is charged
kappa* = -GATE_KAPPA_CAP (1e6); CHECK_FAILED_PENALTY 1e5 (c_v* unknown, or the check failed);
DEGENERATE_PENALTY 1e6 (no W > 0 region at all).
Settings read here: fitness.region_weight, fitness.region_target, fitness.hinge.*,
fitness.non_pd_kappa_cap, fitness.length_reg, fitness.nested_penalty, fitness.non_pd_weight,
verification.pd_eps, verification.kappa_min.  Environment only: SYMCLF_CHECK_KAPPA_SCALE.
"""
from __future__ import annotations

import numpy as np

from src.CPU import config
from src.CPU.encode import encode_expression
from src.CPU.evalnb import _N
from src.CPU.lattice import lattice_numbers


def check_penalty_terms(cv, kappa, volume):
    """(c_v term, kappa term, volume term) -- the three hinges; the volume is the grid's
    sublevel_fraction.  None/NaN pays the full weight."""
    def hinge(w, t, v):
        if w <= 0.0:
            return 0.0
        if v is None or not np.isfinite(v):
            return float(w)
        return float(w * np.clip((t - float(v)) / t, 0.0, 1.0))

    return (hinge(config.PEN_CV, config.PEN_CV_TARGET, cv),
            hinge(config.PEN_KAPPA, config.PEN_KAPPA_TARGET, kappa),
            hinge(config.PEN_VOL, config.PEN_VOL_TARGET, volume))


def check_gate(numbers):
    """fitness.non_pd_kappa_cap: a candidate that is not positive definite is charged kappa* = -cap
    instead of its own kappa*."""
    if (config.GATE_KAPPA_CAP <= 0.0 or not numbers or numbers.get("status") != "ok"
            or bool(numbers.get("pd_valid"))):
        return numbers
    w_pos = numbers.get("kappa") is not None or numbers.get("lattice_kappa") is not None
    return dict(numbers, kappa=(-config.GATE_KAPPA_CAP if w_pos else None),
                kappa_ungated=numbers.get("kappa"), kappa_gated=True,
                worst_point=None, worst_r=None, worst_W=None, worst_abs_b=None)


def numerically_verified(numbers):
    """True iff the check ran and c_v* > pd_eps and kappa* > kappa_min (both finite).

    This is the verdict of a numerical search (a scan plus local searches): no violation was
    found.  It is not a proof.
    """
    if not numbers or numbers.get("status") != "ok":
        return False
    cv, kappa = numbers.get("cv"), numbers.get("kappa")
    return bool(cv is not None and np.isfinite(cv) and cv > config.PD_EPS
                and kappa is not None and np.isfinite(kappa) and kappa > config.KAPPA_MIN)


def roa_rule(cv, kappa, boundary_min, w_scale, verified, volume=None):
    if verified:
        ok = (boundary_min is not None and w_scale is not None and np.isfinite(boundary_min)
              and np.isfinite(w_scale) and w_scale > 0.0)
        level = float(boundary_min) / (float(w_scale) * config.ROA_C_TARGET) if ok else 0.0
        return float(config.ROA_WEIGHT * np.clip(1.0 - level, 0.0, 1.0)
                     + sum(check_penalty_terms(cv, kappa, volume)))
    if cv is None or not np.isfinite(cv):
        return float(config.CHECK_FAILED_PENALTY)          # unknown means failed
    if kappa is None:
        return float(config.DEGENERATE_PENALTY)         # no W > 0 region at all
    kappa = 0.0 if not np.isfinite(kappa) else float(kappa)
    return float(
        config.ROA_WEIGHT * (1.0 + np.log1p(max(-kappa, 0.0) / config.KAPPA_SCALE))
        + 2.0 * config.ROA_WEIGHT * max(1.0 - float(cv) / config.PD_EPS, 0.0)
        + sum(check_penalty_terms(cv, kappa, volume))
    )


def complexity_penalty(expression):
    """The complexity penalty the GP charges a tree (src/CPU/gp_engine.py::_extra), from the expression alone:

        NESTED_PENALTY * (nested trig + nested exp + nested aq) + LENGTH_REG * n_nodes

    with the same 0/1 flags -- flex's nested-trigonometric test (src/SymFunctions.py::check_nested_trig_fn:
    a sin or cos inside a sin or cos) and detect_nested_function_calls for exp and aq -- and n_nodes the
    tree's length (the encoded program has one op per node), in the same order of operations, so
    fitness_fixed_constants without `extra` gives the fitness the GP recorded.
    """
    from src.SymFunctions import check_nested_trig_fn, detect_nested_function_calls

    text = str(expression)
    length = encode_expression(text, _N).n_ops
    nested = float(check_nested_trig_fn(text)) + detect_nested_function_calls(text, "exp") \
        + detect_nested_function_calls(text, "aq")
    return float(config.NESTED_PENALTY * nested + config.LENGTH_REG * float(length))


def fitness_fixed_constants(expression, constants, check_fn=None, grid=None, extra=None):
    """The fitness of (expression, constants): grid stage -> dead rule -> check -> check cost."""
    from src.CPU.deterministic_search import deterministic_search

    check_fn = deterministic_search if check_fn is None else check_fn
    grid = lattice_numbers(expression, constants) if grid is None else grid
    extra = complexity_penalty(expression) if extra is None else float(extra)
    grid_score = float(grid["grid_score"])
    dead = bool(grid["symbolic"] >= 1.0e6 or grid["invalid"] >= 1.0e6
                or not np.isfinite(grid_score) or grid_score >= 1.0e10)
    if dead:
        return {"fitness": float(grid_score + extra), "grid_score": grid_score, "complexity_penalty": extra,
                "check_cost": 0.0, "dead": True, "numerically_verified": False, "grid": grid, "check": None}
    numbers = check_gate(check_fn(expression, constants))
    if numbers is None or numbers.get("status") != "ok":
        cost = float(config.CHECK_FAILED_PENALTY)
    else:
        cost = roa_rule(numbers.get("cv"), numbers.get("kappa"), grid["boundary_min"],
                         grid["w_scale"], numerically_verified(numbers),
                         volume=grid["sublevel_fraction"])
        numbers["penalty_terms"] = check_penalty_terms(
            numbers.get("cv"), numbers.get("kappa"), grid["sublevel_fraction"])
        fraction = numbers.get("non_pd_fraction")
        if config.NON_PD_WEIGHT > 0.0 and fraction is not None and np.isfinite(fraction):
            charge = float(config.NON_PD_WEIGHT * fraction)      # fitness.non_pd_weight, on top of the rule
            cost += charge
            numbers["penalty_terms"] = tuple(numbers["penalty_terms"]) + (charge,)
    cost -= float(grid["roa"]) if np.isfinite(grid["roa"]) else 0.0
    return {
        "fitness": float(grid_score + cost + extra),
        "grid_score": grid_score, "complexity_penalty": extra, "check_cost": float(cost), "dead": False,
        "numerically_verified": bool(numerically_verified(numbers)), "grid": grid, "check": numbers,
    }
