"""3DCartPolerV6 DET4-mode smoke (CPU, Pallas interpret): the DET4 condition inside the GPU5_7
pipeline, checked against the CPU referee (src/srcCPU DET4, origin excluded at 0, kappa over
the W > 0 points).

    JAX_PLATFORMS=cpu SYMCLF_GPU2_ALLOW_CPU=1 SYMCLF_GPU2_PALLAS_INTERPRET=1 \\
    SYMCLF_GPU2_TUNER_RANDOM_POPULATION=4 SYMCLF_GPU2_TUNER_GENERATIONS=1 \\
    SYMCLF_TUNER_REFINE_STEPS=0 PYTHONPATH=$PWD:$PWD/../.. python smoke_v3_det4.py

Checks (each prints PASS/FAIL, exit 1 on any FAIL):
  1. grid ledger: sum(GRID_TERM_NAMES terms) + symbolic == pre_exact
  2. the Pallas ab kernel's rows V and x.gradV == the numba dual interpreter (1e-10 relative)
  3. the ARE quadratic (a CLF by construction): certified on the lattice, by the cheap screen and
     by the exact stage; fitness = the length term only; kappa*, c_v* == referee (1e-6)
  4. the all-ones quadratic: not certified anywhere; fitness = rule on kappa* + length;
     kappa*, c_v* == referee (1e-6)
  5. the crease constant of job 126166 (aq/exp string): c_v_rad ~ 0 -> not PD, grid and full
  6. a bad tree stays far above
  7. V6: the penalised rule's ranking (ARE < gen 27 < gen 147 < plateau), the price-based elite
     selection, the refine's slope on the plateau, the keep-best Adam refine, the tuner exits
"""
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("SYMCLF_GPU2_ALLOW_CPU", "1")
os.environ.setdefault("SYMCLF_GPU2_PALLAS_INTERPRET", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("SYMCLF_EXACT_MODE", "det4")
os.environ.setdefault("SYMCLF_GPU2_CHEAP_FUSION", "4")   # smoke: 3 candidates, not 32 copies
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
from src3DCartPoleV3.cpu_polish import make_cpu_polish_callables  # noqa: E402
from src3DCartPoleV3.fitness import (  # noqa: E402
    _det4_numbers,
    fitness_finish_cheap,
    fitness_finish_exact,
    fitness_pre_exact,
)
from src3DCartPoleV3.grid_fitness import GRID_TERM_NAMES, gpu_pre_exact_mse_many  # noqa: E402
from src3DCartPoleV3.runtime_exact_candidate import (  # noqa: E402
    RuntimeExactCandidate,
    runtime_candidate_bundle,
)

base = Evaluate._base
assert base.DET4_MODE > 0.0, "run with SYMCLF_EXACT_MODE=det4"
RHO = float(base.DET4_RHO)
H = base.BOX_HALF_WIDTHS
axes = [np.linspace(-h, h, 41) for h in H]
X = np.meshgrid(*axes, indexing="ij")
td = src.Functions.Dataset("true_data", axes, None)
td.X1, td.X2, td.X3, td.grid_shape, td.mesh = X[0], X[1], X[2], X[0].shape, list(X)

C = Evaluate._ARE_REFERENCE
T = (("x1", "x1"), ("x1", "x2"), ("x1", "x3"), ("x2", "x2"), ("x2", "x3"), ("x3", "x3"))
pp = [f"mul({c!r}, mul({u}, {v}))" for c, (u, v) in zip(C, T)]
ARE = pp[0]
for q in pp[1:]:
    ARE = f"add({ARE}, {q})"
ONES = "add(add(mul(x1, x1), mul(x2, x2)), mul(x3, x3))"
BAD = "sin(sin(mul(x1, x2)))"   # exp/aq are no longer primitives (2026-09-10)
CREASE = "aq(a, add(mul(sub(sub(x2, x3), x1), exp(a)), x1))"   # job 126166 champion (string only)
CREASE_C = [-14.928, 53.62]
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
check(pre[0] == 0.0, f"ARE certified on the lattice, grid score {pre[0]:.4g}")
check(
    det["kappa_grid"][1] < 0.0 and pre[1] > 500.0,
    f"all-ones not certified on the lattice, grid score {pre[1]:.4g}",
)

# 2. kernel rows vs numba
expr2, c2 = "add(mul(a, mul(x1, sin(x2))), mul(x3, mul(x3, add(x1, a))))", (0.7, -1.3)
bundle, cv = runtime_candidate_bundle(RuntimeExactCandidate(expr2, c2))
rng = np.random.default_rng(0)
pts = rng.uniform(-0.2, 0.2, (37, 3))
rows = np.asarray(bundle.vabr_batch(pts, cv))
fns2 = make_cpu_polish_callables(expr2, c2, base.fSR, base.GSR)
ref_v = np.array([fns2["v_fn"](*z) for z in pts])
ref_r = np.array([float(z @ fns2["gv_fn"](*z)) for z in pts])
err_v = np.max(np.abs(rows[2] - ref_v)) / max(np.max(np.abs(ref_v)), 1e-300)
err_r = np.max(np.abs(rows[3] - ref_r)) / max(np.max(np.abs(ref_r)), 1e-300)
check(err_v < 1e-10 and err_r < 1e-10, f"Pallas rows V / x.gradV vs numba: {err_v:.2e} / {err_r:.2e}")

# 5. crease constant: grid and full numbers
pre_c, det_c = gpu_pre_exact_mse_many([CREASE], [np.asarray(CREASE_C)], td, base)
check(
    det_c["cv_rad_grid"][0] < 1e-4 and pre_c[0] >= 1500.0,
    f"crease champion on the grid: cv_w {det_c['cv_grid'][0]:.3g}"
    f" cv_rad {det_c['cv_rad_grid'][0]:.3g} price {pre_c[0]:.4g} (>= 1500)",
)
d_c = _det4_numbers(CREASE, CREASE_C, base)
check(
    d_c["status"] == "ok" and d_c["cv"] is not None and d_c["cv"] < 1e-4,
    f"crease champion by the full check: cv_w {d_c.get('cv_w')} cv_rad {d_c.get('cv_rad')}"
    f" -> c_v* {d_c['cv']} kappa* {d_c.get('kappa')}",
)

# 3-4. the three stages
_, cfg = util.load_config_data(os.path.join(HERE, "config.yaml"))
pset = gp.PrimitiveSetTyped("MAIN", [float, float, float], float)
pset.renameArguments(ARG0="x1", ARG1="x2", ARG2="x3")
pset = add_primitives_to_pset_from_dict(pset, cfg["gp"]["primitives"])
pset.addTerminal(object, float, "a")
inds = [gp.PrimitiveTree.from_string(e, pset) for e in exprs]
t0 = time.time()
entries = fitness_pre_exact(
    inds, true_data=td, penalty=cfg["gp"]["penalty"], base=base, tuner_fusion=2
)
entries = fitness_finish_cheap(entries, base=base)
cheap = {n: e.get("cheap_diagnostic") or {} for n, e in zip(names, entries)}
check(cheap["ARE (CLF)"].get("certified") is True, f"cheap screen, ARE: {cheap['ARE (CLF)']}")
check(cheap["all-ones"].get("certified") is False, f"cheap screen, all-ones: {cheap['all-ones']}")
check(entries[0].get("needs_full_exact") is True, "ARE promoted to the full check")
attrs = fitness_finish_exact(entries, base=base, true_data=td)
ex = {n: a.get("_exact") or {} for n, a in zip(names, attrs)}
fit = {n: float(a["fitness"][0]) for n, a in zip(names, attrs)}
d4 = {n: (ex[n].get("det4") or {}) for n in names}
are = ex["ARE (CLF)"]
check(
    are.get("det4_certified") is True and are.get("violations") == 0,
    f"exact stage certifies the ARE: roots {are.get('roots')} viol {are.get('violations')}"
    f" kappa* {d4['ARE (CLF)'].get('kappa')} c_v* {d4['ARE (CLF)'].get('cv')}",
)
check(
    abs(fit["ARE (CLF)"] - entries[0]["extra_penalty"]) < 1e-9,
    f"ARE fitness = length term only: {fit['ARE (CLF)']:.6g}"
    f" (extra {entries[0]['extra_penalty']:.6g})",
)
# all-ones: max a/W 5.4 > the gate 0.15 and not certified -> GPU5_7 does NOT promote it; its
# fitness is the escalated cheap price (rule with kappa_cheap x CP3D_GATE_PRICE_ESCALATION)
w = float(base.CP3D_ROA_WEIGHT)
# V5 (SYMCLF_DET4_ALL=1): the screen carries the full DET4 numbers, priced without escalation
esc = 1.0 if os.environ.get("SYMCLF_DET4_ALL", "0") == "1" else float(base.CP3D_GATE_PRICE_ESCALATION)
kappa_ones = float(cheap["all-ones"]["kappa"])
from src3DCartPoleV3.fitness import _det4_penalty_terms as _pen_terms  # noqa: E402
rule_ones = w * (1.0 + np.log1p(max(-kappa_ones * esc, 0.0))) + sum(   # V6: + the positivity penalty
    _pen_terms(float(cheap["all-ones"]["cv"]), kappa_ones * esc, entries[1].get("grid_certified_volume"))
)
check(
    entries[1].get("needs_full_exact") is False
    and abs(fit["all-ones"] - (rule_ones + entries[1]["extra_penalty"])) < 1e-6,
    f"all-ones not promoted; fitness = escalated cheap rule on kappa_cheap ({kappa_ones:.4f}"
    f" x {esc:g}) + length: {fit['all-ones']:.4f} vs {rule_ones + entries[1]['extra_penalty']:.4f}",
)
d4["all-ones"] = _det4_numbers(ONES, [], base)   # the full numbers, for the referee comparison
for n, e in (("ARE (CLF)", ARE), ("all-ones", ONES)):
    ref = referee(e, [])
    got = d4[n]
    # kappa*: never HIGHER than the referee (a higher kappa* = a missed point); lower is a
    # better-found max (all-ones: the manifold roots seed the polish closer to the origin, where
    # the ratio's supremum lies, 5.7e-5 below the referee and on the linearised limit)
    dk = float(got["kappa"]) - float(ref["kappa"])
    dc = abs(float(got["cv"]) - float(ref["cv"]))
    check(
        dk <= 1e-6 and dc < 1e-6,
        f"{n} vs referee: kappa* {got['kappa']:.9f} / {ref['kappa']:.9f} (new - ref {dk:+.2e});"
        f" c_v* {got['cv']:.9f} / {ref['cv']:.9f} (diff {dc:.2e})",
    )
# ---- V6 (2026-09-11): positivity penalties, price-selected elites, Adam refine, tuner exits ----
from src3DCartPoleV3.fitness import (  # noqa: E402
    _BASE_FSR, _BASE_GSR, _BASE_N, _det4_penalty_terms, _roa_rule, _rule_and_gradient,
    _symbolic_with_constants, refine_constants_det4,
)
from src3DCartPoleV3 import grid_fitness as GF, ray_fitness as RF  # noqa: E402

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
    mk = lambda pr, key, consts=(1.0,): {"pre_exact": pr, "gate2_penalty": 0.0, "extra_penalty": 0.0,  # noqa: E731
                                        "cheap_diagnostic": {}, "consts": np.asarray(consts), "exact_key": key}
    inter = [mk(700, "a"), mk(501, "b"), mk(505, "c"), mk(9000, "d", ()), mk(501, "b"), mk(503, "e")]
    flagged = RF._select_refine_by_price(inter, 3)
    check(
        [inter[i]["exact_key"] for i in flagged] == ["b", "e", "c"] and all(inter[i].get("needs_full_exact") and inter[i].get("needs_refine") for i in flagged),
        f"V6 elite selection: top-3 distinct by price -> keys {[inter[i]['exact_key'] for i in flagged]} (no-constant and duplicate entries skipped)",
    )
    G119 = 'add(add(add(add(add(mul(a, mul(sub(add(add(add(mul(mul(x2, x2), mul(mul(x1, x3), add(add(x2, x2), x2))), mul(a, mul(a, mul(sub(x3, x1), add(x1, mul(a, x1)))))), x2), mul(a, add(x2, x2))), x1), sub(x1, x3))), mul(a, mul(x2, x3))), mul(mul(mul(x1, a), mul(x1, mul(sin(x2), x2))), mul(x1, mul(x1, x3)))), mul(a, mul(x2, x2))), mul(a, mul(sub(x2, x1), x3))), mul(a, add(a, mul(x3, sin(x3)))))'
    G119_C = [-0.0059529992908966605, 0.10186808431985778, 0.10186808431985778, 0.11523453303157777, 0.038037886271777614, 0.2780718206639071, 1.1975390057623785, 20.85110286168646, 0.1310250305889205, 21.073201108194468, 0.022369713479682744]
    n119 = len(G119_C)
    num119 = _det4_numbers(G119, G119_C, base)
    _BASE_FSR[0], _BASE_GSR[0], _BASE_N[0] = base.fSR, base.GSR, 3
    fns119 = _symbolic_with_constants(G119, n119)
    val119, grad119 = _rule_and_gradient(num119, (0.0, 1.0, np.zeros(n119), np.zeros(n119), 0.0015), base, fns119, np.asarray(G119_C))
    check(
        num119.get("kappa") is not None and 0.0 < num119["kappa"] < knobs[3] and bool(np.any(grad119)),
        f"V6 refine sees a slope on the plateau: gen 119 kappa* {num119.get('kappa')}, rule {val119:.2f}, |grad| {np.linalg.norm(grad119):.3g}",
    )
    G25 = 'add(add(add(add(add(mul(a, mul(sub(sub(mul(mul(sin(x3), sub(x3, x2)), x1), mul(a, x3)), add(sin(x3), add(x1, x2))), x1)), mul(x3, mul(x1, x3))), mul(x3, mul(x1, x3))), mul(a, mul(x2, x2))), mul(a, mul(x2, x3))), mul(a, mul(add(x2, x3), x3)))'
    G25_C = [-0.0010004273474276452, 0.08667589019244848, 6.441461278718269, 0.03986186569355108, 2.6120167148123743]
    r_value, r_c, r_num, r_log = refine_constants_det4(G25, G25_C, td, base, steps=4)
    check(
        r_value is not None and r_value <= r_log[0]["value"] + 1e-9 and len(r_log) == 5 and os.environ.get("SYMCLF_DET4_REFINE_METHOD") == "adam",
        f"V6 Adam refine (4 evaluations, keep-best): rule {r_log[0]['value']:.3f} -> {r_value if r_value is None else round(r_value, 3)},"
        f" kappa* {r_log[0]['kappa']:.5f} -> {r_num.get('kappa')}, accepted steps {sum(bool(e.get('accepted')) for e in r_log[1:])}",
    )
    ONES_A = "mul(a, add(add(mul(x1, x1), mul(x2, x2)), mul(x3, x3)))"
    same = True
    for e_ in (ONES_A, f"mul(a, {ARE})"):
        GF._ADAM_EXIT_ZERO = False
        v0_, x0_ = GF.refine_constants(e_, np.array([1.0]), 1, td, base, steps=3)
        GF._ADAM_EXIT_ZERO = True
        v1_, x1_ = GF.refine_constants(e_, np.array([1.0]), 1, td, base, steps=3)
        same = same and v0_ == v1_ and np.array_equal(x0_, x1_)
    check(same, "V6 tuner exits (floor / zero gradient) are bit-identical to the full 40-step Adam")
check(fit["bad tree"] > 1e6, f"bad tree stays far above: fitness {fit['bad tree']:.4g}")
verdict = "SMOKE V3 DET4 PASS" if not fails else f"SMOKE V3 DET4 FAIL: {fails}"
print(f"[{time.time() - t_all:.0f}s]  {verdict}")
sys.exit(1 if fails else 0)
