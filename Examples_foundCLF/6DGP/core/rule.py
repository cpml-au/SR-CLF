"""The V8 price rule and the fitness of a candidate at fixed constants.

fitness = pre_exact + (ROA_RULE(DET4 numbers, grid anchors) - grid ROA term) + extra
with pre_exact and the anchors from the lattice stage, the DET4 numbers from the check, and
extra = nested-call charge + length regulariser.  Verbatim arithmetic of
src4DCartPoleV8/fitness.py (_roa_rule, _det4_penalty_terms, _det4_certified, _det4_gate,
_det4_price_from_entry, _price_entry, fitness_fixed_constants + the D4 dead rule).
"""
from __future__ import annotations

import numpy as np

from core import config
from core.encode import encode_expression
from core.evalnb import _N
from core.lattice import _nested_calls, lattice_numbers


def det4_penalty_terms(cv, kappa, volume):
    """(c_v term, kappa term, volume term); None/NaN pays the full weight."""
    def hinge(w, t, v):
        if w <= 0.0:
            return 0.0
        if v is None or not np.isfinite(v):
            return float(w)
        return float(w * np.clip((t - float(v)) / t, 0.0, 1.0))

    return (hinge(config.PEN_CV, config.PEN_CV_TARGET, cv),
            hinge(config.PEN_KAPPA, config.PEN_KAPPA_TARGET, kappa),
            hinge(config.PEN_VOL, config.PEN_VOL_TARGET, volume))


def det4_gate(numbers):
    """Option B: a non-PD candidate is charged kappa* = -cap instead of its own kappa."""
    if (config.GATE_KAPPA_CAP <= 0.0 or not numbers or numbers.get("status") != "ok"
            or bool(numbers.get("pd_valid"))):
        return numbers
    w_pos = numbers.get("kappa") is not None or numbers.get("lattice_kappa") is not None
    return dict(numbers, kappa=(-config.GATE_KAPPA_CAP if w_pos else None),
                kappa_ungated=numbers.get("kappa"), kappa_gated=True,
                worst_point=None, worst_r=None, worst_W=None, worst_abs_b=None)


def det4_certified(numbers):
    """D1: the certificate is c_v* > pd_eps and kappa* > kappa_min (no root verdict in V8)."""
    if not numbers or numbers.get("status") != "ok":
        return False
    cv, kappa = numbers.get("cv"), numbers.get("kappa")
    return bool(cv is not None and np.isfinite(cv) and cv > config.PD_EPS
                and kappa is not None and np.isfinite(kappa) and kappa > config.KAPPA_MIN)


def roa_rule(cv, kappa, boundary_min, w_scale, certified, volume=None):
    if certified:
        ok = (boundary_min is not None and w_scale is not None and np.isfinite(boundary_min)
              and np.isfinite(w_scale) and w_scale > 0.0)
        level = float(boundary_min) / (float(w_scale) * config.ROA_C_TARGET) if ok else 0.0
        return float(config.ROA_WEIGHT * np.clip(1.0 - level, 0.0, 1.0)
                     + sum(det4_penalty_terms(cv, kappa, volume)))
    if cv is None or not np.isfinite(cv):
        return float(config.EXACT_PENALTY_MAX)          # unknown means failed
    if kappa is None:
        return float(config.DEGENERATE_PENALTY)         # no W > 0 region at all
    kappa = 0.0 if not np.isfinite(kappa) else float(kappa)
    return float(
        config.ROA_WEIGHT * (1.0 + np.log1p(max(-kappa, 0.0) / config.KAPPA_SCALE))
        + 2.0 * config.ROA_WEIGHT * max(1.0 - float(cv) / config.PD_EPS, 0.0)
        + sum(det4_penalty_terms(cv, kappa, volume))
    )


def extra_penalty(expression):
    """NESTED_PENALTY per nested trig/exp/aq class + reg_param * tree length (fitness_pre_exact)."""
    text = str(expression)
    program = encode_expression(text, _N)
    nested = float(_nested_calls(text, "sin")) + float(_nested_calls(text, "cos")) \
        + float(_nested_calls(text, "exp")) + float(_nested_calls(text, "aq"))
    return float(config.NESTED_PENALTY * nested + config.LENGTH_REG * program.n_ops)


def fitness_fixed_constants(expression, constants, det4_fn=None, grid=None, extra=None):
    """The V8 fitness of (expression, constants): lattice stage -> dead rule -> check -> price."""
    from core.det4 import det4 as _det4

    det4_fn = _det4 if det4_fn is None else det4_fn
    grid = lattice_numbers(expression, constants) if grid is None else grid
    extra = extra_penalty(expression) if extra is None else float(extra)
    pre_exact = float(grid["pre_exact"])
    dead = bool(grid["symbolic"] >= 1.0e6 or grid["invalid"] >= 1.0e6
                or not np.isfinite(pre_exact) or pre_exact >= 1.0e10)
    if dead:
        return {"fitness": float(pre_exact + extra), "pre_exact": pre_exact, "extra": extra,
                "price": 0.0, "dead": True, "certified": False, "grid": grid, "det4": None}
    numbers = det4_gate(det4_fn(expression, constants))
    if numbers is None or numbers.get("status") != "ok":
        price = float(config.EXACT_PENALTY_MAX)
    else:
        price = roa_rule(numbers.get("cv"), numbers.get("kappa"), grid["boundary_min"],
                         grid["w_scale"], det4_certified(numbers),
                         volume=grid["certified_volume"])
        numbers["penalty_terms"] = det4_penalty_terms(
            numbers.get("cv"), numbers.get("kappa"), grid["certified_volume"])
    price -= float(grid["roa"]) if np.isfinite(grid["roa"]) else 0.0
    return {
        "fitness": float(pre_exact + price + extra),
        "pre_exact": pre_exact, "extra": extra, "price": float(price), "dead": False,
        "certified": bool(det4_certified(numbers)), "grid": grid, "det4": numbers,
    }
