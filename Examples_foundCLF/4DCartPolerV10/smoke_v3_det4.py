"""4DCartPolerV8 smoke (CPU, Pallas interpret): the V8 pipeline = grid ledger, THE CHECK (det4_fast) against
the CPU referee (src/srcCPU DET4, same switches), the V6 rule and penalties, the price-based elite selection,
the tuner exits, the D4 dead tree.

    JAX_PLATFORMS=cpu SYMCLF_GPU2_ALLOW_CPU=1 SYMCLF_GPU2_PALLAS_INTERPRET=1 \\
    SYMCLF_GPU2_TUNER_RANDOM_POPULATION=4 SYMCLF_GPU2_TUNER_GENERATIONS=1 \\
    SYMCLF_TUNER_REFINE_STEPS=0 PYTHONPATH=$PWD:$PWD/../.. python smoke_v3_det4.py

Checks (each prints PASS/FAIL, exit 1 on any FAIL):
  1. grid ledger: sum(GRID_TERM_NAMES terms) + symbolic == pre_exact; the ARE certified on the lattice with
     grid score = its V6 lattice terms (ROA rule 0); all-ones not certified on the lattice (kappa <= kappa_min)
  2. the check: the ARE is det4-certified and its fitness = length + its V6 terms (ROA rule 0); all-ones is not
     certified and its fitness = the rule on its det4 numbers + length; both vs the referee (kappa* new - ref
     <= 1e-6, c_v* 1e-6; the referee runs with the same PROJECT_ASCENT / ORIGIN_SEEDS switches)
  3. V6: the penalised rule's ranking (ARE < gen 27 < gen 147 < plateau), the price-based elite selection,
     the tuner exits bit-identical to the full Adam
  4. the bad tree (missing x3, x4): fitness > 1e6; under SYMCLF_DET4_SKIP_DEAD=1 it is dead (no det4 numbers)
  5. (2026-09-15, the CMA stage) the fitness at fixed constants reproduces job 146800's 5.972398158733319 on its
     certified quadratic; cma_refine_elite's plumbing (18 evaluations, keep-best, structure cache)
"""
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("SYMCLF_GPU2_ALLOW_CPU", "1")
os.environ.setdefault("SYMCLF_GPU2_PALLAS_INTERPRET", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("SYMCLF_EXACT_MODE", "det4")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT]

import numpy as np  # noqa: E402
from deap import gp  # noqa: E402
from flex.gp import util  # noqa: E402
from flex.gp.primitives import add_primitives_to_pset_from_dict  # noqa: E402

import Evaluate  # noqa: E402
import src.Functions  # noqa: E402
from src.srcCPU.det4_check_cpu import check_det4_cpu  # noqa: E402
from src.srcCPU.symbolic import sympy_expression  # noqa: E402
from src4DCartPoleV10.fitness import (  # noqa: E402
    _det4_certified,
    _det4_penalty_terms,
    fitness_finish_det4,
    fitness_pre_exact,
)
from src4DCartPoleV10.grid_fitness import GRID_TERM_NAMES, gpu_pre_exact_mse_many  # noqa: E402

base = Evaluate._base
assert base.DET4_MODE > 0.0, "run with SYMCLF_EXACT_MODE=det4"
RHO = float(base.DET4_RHO)
H = base.BOX_HALF_WIDTHS
axes = [np.linspace(-h, h, 21) for h in H]   # run.py GRID_POINTS
X = np.meshgrid(*axes, indexing="ij")
td = src.Functions.Dataset("true_data", axes, None)
td.X1, td.X2, td.X3, td.X4, td.grid_shape, td.mesh = X[0], X[1], X[2], X[3], X[0].shape, list(X)

C = Evaluate._ARE_REFERENCE
T = tuple((f"x{i}", f"x{j}") for i in range(1, 5) for j in range(i, 5))   # _ARE_REFERENCE order
pp = [f"mul({c!r}, mul({u}, {v}))" for c, (u, v) in zip(C, T)]
ARE = pp[0]
for q in pp[1:]:
    ARE = f"add({ARE}, {q})"
ONES = "add(add(mul(x1, x1), mul(x2, x2)), add(mul(x3, x3), mul(x4, x4)))"
BAD = "sin(sin(mul(x1, x2)))"   # exp/aq are no longer primitives (2026-09-10); misses x3 and x4 -> symbolic 2e6
names = ["ARE (CLF)", "all-ones", "bad tree"]
exprs = [ARE, ONES, BAD]

fails = []


def check(cond, text):
    print(("PASS " if cond else "FAIL ") + text, flush=True)
    if not cond:
        fails.append(text)


def referee(expr, consts):
    V = sympy_expression(expr, list(consts))
    r = check_det4_cpu(
        V, fSR=base.fSR, GSR=base.GSR, bounds=base.SHGO_BOUNDS, rhos=(RHO,),
        origin_tol=0.0, gate_on_pd=False,
        # the referee in step with DET4Fast4D_V10's switches
        project_ascent=os.environ.get("SYMCLF_DET4_PROJECT_ASCENT", "0") == "1",
        origin_seeds=os.environ.get("SYMCLF_DET4_ORIGIN_SEEDS", "0") == "1",
    )
    row = r.rows[RHO]
    return {"cv": r.pd_rate, "cv_w": r.pd_rate_w, "cv_rad": r.pd_rate_radial,
            "kappa": row.kappa_star, "status": r.status}


t_all = time.time()
# 1. grid ledger
pre, det = gpu_pre_exact_mse_many(exprs, [np.empty(0)] * 3, td, base)
for k, name in enumerate(names):
    terms = dict(zip(GRID_TERM_NAMES, det["terms"][k].tolist()))
    total = sum(terms.values()) + float(det["symbolic"][k])
    ok = abs(pre[k] - total) <= 1e-9 * max(1.0, abs(pre[k])) or pre[k] >= 1e10
    check(
        ok,
        f"ledger {name}: pre_exact {pre[k]:.6g} = sum(terms)+symbolic {total:.6g}"
        f" (lattice c_v {det['cv_grid'][k]:.4g} kappa {det['kappa_grid'][k]:.4g})",
    )
lat_pen = sum(_det4_penalty_terms(det["cv_grid"][0], det["kappa_grid"][0], det["certified_volume"][0]))
check(
    det["cv_grid"][0] > base.DET4_PD_EPS and det["kappa_grid"][0] > base.DET4_KAPPA_MIN
    and abs(pre[0] - lat_pen) <= 1e-9 * max(1.0, abs(pre[0])),
    f"ARE certified on the lattice, grid score {pre[0]:.4g} = its V6 lattice terms {lat_pen:.4g} (ROA rule 0)",
)
check(
    det["kappa_grid"][1] <= base.DET4_KAPPA_MIN and pre[1] > 500.0,   # 4-D: kappa = -0 (x1 axis)
    f"all-ones not certified on the lattice (kappa {det['kappa_grid'][1]:.4g} <= kappa_min"
    f" {base.DET4_KAPPA_MIN:g}), grid score {pre[1]:.4g}",
)

# 2. the two stages: S0+S1 then THE CHECK
_, cfg = util.load_config_data(os.path.join(HERE, "config.yaml"))
pset = gp.PrimitiveSetTyped("MAIN", [float, float, float, float], float)
pset.renameArguments(ARG0="x1", ARG1="x2", ARG2="x3", ARG3="x4")
pset = add_primitives_to_pset_from_dict(pset, cfg["gp"]["primitives"])
pset.addTerminal(object, float, "a")
inds = [gp.PrimitiveTree.from_string(e, pset) for e in exprs]
entries = fitness_pre_exact(
    inds, true_data=td, penalty=cfg["gp"]["penalty"], base=base, tuner_fusion=2
)
entries = fitness_finish_det4(entries, base=base)
d4 = {n: (e.get("det4_numbers") or {}) for n, e in zip(names, entries)}
fit = {n: float(e["fitness"][0]) for n, e in zip(names, entries)}
are_e = entries[0]
check(
    are_e.get("det4_certified") is True and _det4_certified(d4["ARE (CLF)"], base),
    f"the check certifies the ARE: kappa* {d4['ARE (CLF)'].get('kappa')} c_v* {d4['ARE (CLF)'].get('cv')}"
    f" (source {are_e.get('det4_source')}, {d4['ARE (CLF)'].get('seconds', 0.0):.1f} s)",
)
are_pen = sum(_det4_penalty_terms(d4["ARE (CLF)"].get("cv"), d4["ARE (CLF)"].get("kappa"),
                                  are_e.get("grid_certified_volume")))
check(
    abs(fit["ARE (CLF)"] - (are_e["extra_penalty"] + are_pen)) < 1e-9,
    f"ARE fitness = length + its V6 terms (ROA rule 0): {fit['ARE (CLF)']:.6g}"
    f" (extra {are_e['extra_penalty']:.6g} + V6 {are_pen:.6g})",
)
w = float(base.CP3D_ROA_WEIGHT)
ones_e = entries[1]
kappa_ones = float(d4["all-ones"]["kappa"])
cv_ones = float(d4["all-ones"]["cv"])
rule_ones = (
    w * (1.0 + np.log1p(max(-kappa_ones, 0.0)))
    + 2.0 * w * max(1.0 - cv_ones / float(base.DET4_PD_EPS), 0.0)
    + sum(_det4_penalty_terms(cv_ones, kappa_ones, ones_e.get("grid_certified_volume")))
)
check(
    ones_e.get("det4_certified") is False
    and abs(fit["all-ones"] - (rule_ones + ones_e["extra_penalty"])) < 1e-6,
    f"all-ones not certified; fitness = the rule on its det4 numbers (kappa* {kappa_ones:.4f},"
    f" c_v* {cv_ones:.4f}) + length: {fit['all-ones']:.4f} vs {rule_ones + ones_e['extra_penalty']:.4f}",
)
for n, e in (("ARE (CLF)", ARE), ("all-ones", ONES)):
    ref = referee(e, [])
    got = d4[n]
    # kappa*: never HIGHER than the referee (a higher kappa* = a missed point); lower is a better-found max
    dk = float(got["kappa"]) - float(ref["kappa"])
    dc = abs(float(got["cv"]) - float(ref["cv"]))
    check(
        dk <= 1e-6 and dc < 1e-6,
        f"{n} vs referee: kappa* {got['kappa']:.9f} / {ref['kappa']:.9f} (new - ref {dk:+.2e});"
        f" c_v* {got['cv']:.9f} / {ref['cv']:.9f} (diff {dc:.2e})",
    )

# 2b. the fitness at FIXED constants (the CMA stage's objective): job 146800's certified quadratic
#     (examples/4DCartPolerV7Gurobi, CMA-ES on the V7 fitness, referee-audited; V7 fitness 5.972398158733319)
from src.SymFunctions import detect_nested_function_calls, get_features_batch  # noqa: E402
from src4DCartPoleV10.fitness import _NESTED_CALL_PENALTY, fitness_fixed_constants  # noqa: E402
from src4DCartPoleV10 import ray_fitness as RF  # noqa: E402

SEED2 = ("add(add(add(add(add(add(add(add(add(mul(a, mul(x1, x1)), mul(a, mul(x1, x2))), mul(a, mul(x1, x3))), "
         "mul(a, mul(x1, x4))), mul(a, mul(x2, x2))), mul(a, mul(x2, x3))), mul(a, mul(x2, x4))), "
         "mul(a, mul(x3, x3))), mul(a, mul(x3, x4))), mul(a, mul(x4, x4)))")   # run.py seed_expr2
C146800 = [0.048198739306794, 0.02242609132683108, -0.6814929779488536, -0.07891171609873011,
           0.06499410224542178, -0.3604573865991704, -0.5090843613379541, 4.770118393296521,
           1.5200301678935837, 1.546968743880691]


def extra_of(expr):
    """The extra penalty exactly as fitness_pre_exact computes it (nested calls + length)."""
    length, nested, _ = get_features_batch([gp.PrimitiveTree.from_string(expr, pset)])
    return float(
        _NESTED_CALL_PENALTY
        * (nested[0] + detect_nested_function_calls(expr, "exp") + detect_nested_function_calls(expr, "aq"))
        + float(cfg["gp"]["penalty"]["reg_param"]) * length[0]
    )


fx = fitness_fixed_constants(
    [{"expression": SEED2, "consts": np.asarray(C146800), "extra_penalty": extra_of(SEED2)}],
    base=base, true_data=td,
)[0]
check(
    abs(float(fx["fitness"][0]) - 5.972398158733319) < 1e-9 and fx.get("det4_certified") is True,
    f"fixed-constants fitness of the 146800 quadratic: {float(fx['fitness'][0]):.12f} (job 146800:"
    f" 5.972398158733319), certified {fx.get('det4_certified')},"
    f" kappa* {(fx.get('det4_numbers') or {}).get('kappa')}",
)

# 2c. the CMA stage's plumbing (pycma, the 146800 recipe, in-process evaluation): 2 generations of 9 on the
#     all-ones quadratic with one constant; keep-best; the structure cache
ONES_A = "mul(a, add(add(mul(x1, x1), mul(x2, x2)), add(mul(x3, x3), mul(x4, x4))))"
e0 = fitness_fixed_constants(
    [{"expression": ONES_A, "consts": np.asarray([1.0]), "extra_penalty": 0.0}], base=base, true_data=td
)[0]
done = set()
best_entry, info = RF.cma_refine_elite(
    e0, lambda items: fitness_fixed_constants(items, base=base, true_data=td),
    evals=18, popsize=9, seed=1, structure_done=done,
)
check(
    info["evals"] >= 18 and ONES_A in done and info["after"] <= info["before"]
    and (best_entry is None or float(best_entry["fitness"][0]) == info["after"]),
    f"CMA stage plumbing: {info['evals']} evaluations in {info['seconds']:.0f} s, before {info['before']:.4f}"
    f" after {info['after']:.4f}, improved {info['improved']}, structure cached",
)

# 3. V6: positivity penalties, price-selected elites, tuner exits
from src4DCartPoleV10.fitness import _roa_rule  # noqa: E402
from src4DCartPoleV10 import grid_fitness as GF, ray_fitness as RF  # noqa: E402

V6 = os.environ.get("SYMCLF_DET4_PEN_KAPPA", "0") not in ("0", "0.0")
if V6:
    knobs = tuple(float(os.environ.get(k, d)) for k, d in (
        ("SYMCLF_DET4_PEN_CV", "0"), ("SYMCLF_DET4_PEN_CV_TARGET", "0.05"),
        ("SYMCLF_DET4_PEN_KAPPA", "0"), ("SYMCLF_DET4_PEN_KAPPA_TARGET", "0.05"),
        ("SYMCLF_DET4_PEN_VOL", "0"), ("SYMCLF_DET4_PEN_VOL_TARGET", "0.02")))
    default_knobs = knobs == (10.0, 0.05, 30.0, 0.05, 20.0, 0.02)
    p27 = _roa_rule(0.303993, 0.016685, 0.01921, 0.1458, base, True, volume=0.0446)
    p147 = _roa_rule(0.0076387, 0.0741924, 0.0004962, 0.09992, base, True, volume=0.001727)
    p119 = _roa_rule(0.005704, 0.0011164, 0.000371, 0.09212, base, False, volume=0.001523)
    check(
        p27 < p147 < 500.0 < p119 and _det4_penalty_terms(0.06362, 0.6493, 0.07648) == (0.0, 0.0, 0.0)
        and (not default_knobs or (abs(p27 - 19.99) < 0.05 and abs(p147 - 26.75) < 0.05 and abs(p119 - 556.67) < 0.05)),
        f"V6 rule: ARE penalty 0; gen 27 {p27:.2f} < gen 147 (needle) {p147:.2f} < plateau gen 119 {p119:.2f}",
    )
    mk = lambda pr, key, consts=(1.0,): {"pre_exact": pr, "det4_price": 0.0, "extra_penalty": 0.0,  # noqa: E731
                                        "det4_numbers": {"status": "ok"}, "consts": np.asarray(consts),
                                        "exact_key": key}
    inter = [mk(700, "a"), mk(501, "b"), mk(505, "c"), mk(9000, "d", ()), mk(501, "b"), mk(503, "e")]
    flagged = RF._select_refine_by_price(inter, 3)
    check(
        [inter[i]["exact_key"] for i in flagged] == ["b", "e", "c"] and all(inter[i].get("needs_refine") for i in flagged),
        f"V6 elite selection: top-3 distinct by price -> keys {[inter[i]['exact_key'] for i in flagged]} (no-constant and duplicate entries skipped)",
    )
    ONES_A = "mul(a, add(add(mul(x1, x1), mul(x2, x2)), add(mul(x3, x3), mul(x4, x4))))"
    same = True
    for e_ in (ONES_A, f"mul(a, {ARE})"):
        GF._ADAM_EXIT_ZERO = False
        v0_, x0_ = GF.refine_constants(e_, np.array([1.0]), 1, td, base, steps=3)
        GF._ADAM_EXIT_ZERO = True
        v1_, x1_ = GF.refine_constants(e_, np.array([1.0]), 1, td, base, steps=3)
        same = same and v0_ == v1_ and np.array_equal(x0_, x1_)
    check(same, "V6 tuner exits (floor / zero gradient) are bit-identical to the full 40-step Adam")

# 3b. V9: the CMA gate and count on fake priced entries (fitness = pre_exact + price + extra here)
mkc = lambda fit, expr, consts=(1.0,): {"pre_exact": fit, "det4_price": 0.0, "extra_penalty": 0.0,  # noqa: E731
                                        "det4_numbers": {"status": "ok"}, "consts": np.asarray(consts),
                                        "expression": expr, "fitness": (fit,)}
inter2 = [mkc(5000.0, "A"), mkc(2999.0, "B"), mkc(1200.0, "C"), mkc(1200.0, "C"), mkc(800.0, "D", ()), mkc(900.0, "E")]
g1 = RF._select_cma_elites(inter2, set(), count=1, max_fitness=3000.0)
g3 = RF._select_cma_elites(inter2, {"E"}, count=3, max_fitness=3000.0)
g0 = RF._select_cma_elites([mkc(4000.0, "F")], set(), count=1, max_fitness=3000.0)
check(
    [inter2[i]["expression"] for i in g1] == ["E"] and [inter2[i]["expression"] for i in g3] == ["C", "B"] and g0 == [],
    f"V9 CMA gate/count: best below 3000 -> {[inter2[i]['expression'] for i in g1]}; count 3 skipping E, distinct,"
    f" gated -> {[inter2[i]['expression'] for i in g3]}; nothing below 3000 -> {g0}",
)

# 4. the bad tree: dead under D4 (no check), far above in any case
skip_dead = os.environ.get("SYMCLF_DET4_SKIP_DEAD", "0") == "1"
bad_e = entries[2]
check(
    fit["bad tree"] > 1e6
    and (not skip_dead or (bad_e.get("dead") is True and bad_e.get("det4_numbers") is None)),
    f"bad tree stays far above: fitness {fit['bad tree']:.4g} (dead={bad_e.get('dead')}, skip_dead={skip_dead},"
    f" det4 numbers={'none' if bad_e.get('det4_numbers') is None else 'present'})",
)
verdict = "SMOKE V8 PASS" if not fails else f"SMOKE V8 FAIL: {fails}"
print(f"[{time.time() - t_all:.0f}s]  {verdict}")
sys.exit(1 if fails else 0)
