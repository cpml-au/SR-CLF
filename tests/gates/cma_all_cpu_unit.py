"""Unit test of option (b)'s selection (src/cma_all.py::select) on entries as src/CPU/gp_engine.py builds them
(the check's numbers under "check").

    cd <repo> && python tests/gates/cma_all_cpu_unit.py
Checks: dead / unchecked / constant-free trees out, one representative per structure, verified first by fitness,
then by kappa*, no kappa* last, lifetime caps, whole CMA iterations, the per-island share, and the arrival constants and
the generation / island of a mapper call.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np                       # noqa: E402
from src import cma_all as C             # noqa: E402

os.environ.pop("SYMCLF_N_ISLANDS", None)


def entry(expr, fit, kappa, verified=False, dead=False, consts=(1.0, 2.0), checked=True):
    return {"expression": expr, "fitness": (fit,), "consts": np.asarray(consts, float), "dead": dead,
            "numerically_verified": verified, "check": ({"kappa": kappa} if checked else None)}


inter = [
    entry("A", 50.0, 0.1), entry("B", 0.30, 1.0, verified=True), entry("C", 900.0, 0.9),
    entry("B", 0.29, 1.0, verified=True), entry("D", 1e8, None, dead=True), entry("E", 20.0, 0.5, consts=()),
    entry("F", 10.0, None, checked=False), entry("G", 0.40, 1.1, verified=True), entry("H", 70.0, None),
]
numbers = lambda e: e.get("check")   # noqa: E731
kw = dict(first=108, last=18, cap_verified=5000, cap_other=3000, numbers_of=numbers)
state = {"C": {"blob": b"state-C", "evals": 2990, "verified": False},
         "A": {"blob": b"state-A", "evals": 2946, "verified": False}}
sel, budgets, blobs, groups = C.select(inter, state, 18, total=1800, **kw)
order = [inter[i]["expression"] for i in sel]
ok = (order == ["B", "G", "A", "H"] and sel[0] == 3 and budgets == [108, 72, 54, 18] and blobs[2] == b"state-A"
      and blobs[0] is None and groups["B"] == [1, 3] and all(b % 18 == 0 for b in budgets))
print("selection:", order, budgets, "->", "ok" if ok else "BAD (want B G A H, [108, 72, 54, 18])")
sel_c, budgets_c, _, _ = C.select(inter, state, 18, total=36, **kw)   # ceiling 36: two iterations -> the best two
good = [inter[i]["expression"] for i in sel_c] == ["B", "G"] and budgets_c == [18, 18]
print("ceiling 36 (two iterations):", [inter[i]["expression"] for i in sel_c], budgets_c, "->", "ok" if good else "BAD")
ok &= good
os.environ["SYMCLF_GEN_POPULATION"] = "45"   # the 9 entries are 1/5 of the population: an island call
sel5, budgets5, _, _ = C.select(inter, state, 18, total=9000, **kw)
good = sel5 == sel and budgets5 == budgets
print("island call (9 of 45 trees, total 9000 -> 1800):", budgets5, "->", "ok" if good else "BAD")
ok &= good
os.environ["SYMCLF_GEN_POPULATION"] = "9"    # the 9 entries are the whole population: a generation call
selg, budgetsg, _, _ = C.select(inter, state, 18, total=1800, **kw)
good = selg == sel and budgetsg == budgets
print("generation call (9 of 9 trees, total 1800 -> 1800):", budgetsg, "->", "ok" if good else "BAD")
ok &= good
os.environ.pop("SYMCLF_GEN_POPULATION")
os.environ["SYMCLF_N_ISLANDS"] = "5"
gi = [C.generation_island(k) for k in (0, 4, 5, 6, 9)]
good = gi == [(0, 1), (0, 5), (1, "all"), (2, "all"), (5, "all")]
print("generation / island of map calls 0, 4, 5, 6, 9 (5 islands):", gi, "->", "ok" if good else "BAD")
ok &= good
os.environ.pop("SYMCLF_N_ISLANDS")
a = C.arrival_constants([2.0, 3.0], 2)
b = C.arrival_constants([2.0, np.nan], 2)
c = C.arrival_constants([2.0], 2)
d = C.arrival_constants(None, 3)
good = list(a) == [2.0, 3.0] and list(b) == [1.0, 1.0] and list(c) == [1.0, 1.0] and list(d) == [1.0, 1.0, 1.0]
print("arrival constants (parent / non-finite / wrong count / none):", list(a), list(b), list(c), list(d),
      "->", "ok" if good else "BAD")
ok &= good
print("CMA ALL CPU UNIT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
