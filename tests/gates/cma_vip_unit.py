"""Unit test of option (c)'s selection (src/cma_all.py::select_vip, merge_seed_copies).

    cd <repo> && python tests/gates/cma_vip_unit.py
Checks, with the numbers below (popsize 18): generation 0 -- each island copy of the seed is its own
search (key S#copy<island>, pycma seed base + island) of 4986 = 5000 in whole iterations, the island call draws its
share of VIP 2 (2 of the best 4) with 2484 each and gives the standard class its share (20 000) between 486 and 18;
the merge keeps the best copy; a later generation gives the merged seed 990 (6000 - 4986 left, at most 1000), draws
10 of the best 20 (486 for a structure drawn before, none with VIP-2 lifetime spent), the standard class 100 000;
the draw is reproducible; no call exceeds total_max x its share; a small total_max trims VIP 2 and the standard class.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np                       # noqa: E402
from src import cma_all as C             # noqa: E402

POP = 18
KW = dict(seed_expressions=["S", "SOS"], base_seed=2, seed_first=5000, seed_again=1000, seed_cap=6000, vip_pool=20,
          vip_count=10, vip_first=2500, vip_again=500, vip_cap=3000, std_total=100000, std_first=500, std_last=18,
          cap_verified=5000, cap_other=3000, total_max=150000, draw_seed=0,
          numbers_of=lambda e: e.get("check"))
failures = []


def check(name, condition, detail=""):
    print(f"{name}: {'ok' if condition else 'BAD'} {detail}")
    if not condition:
        failures.append(name)


def entry(expr, fit, kappa, verified=False, consts=10):
    return {"expression": expr, "fitness": (fit,), "consts": np.ones(consts), "dead": False,
            "numerically_verified": verified, "check": {"kappa": kappa}}


def population(n_others):
    out = [entry("S", 1500.0, -4.7), entry("SOS", 600.0, 0.0, consts=0), entry("V", 0.3, 1.2, verified=True)]
    out += [entry(f"T{i:03d}", 700.0 + i, 0.5 - 0.01 * i) for i in range(n_others)]
    return out


# ---- generation 0, island call 1 and 2 (5 islands, 50 of 250 trees: share 0.2) ----------------------------------
os.environ["SYMCLF_GEN_POPULATION"] = "250"
island = population(47)                                  # 50 entries
state = {}
runs, groups, summary = C.select_vip(island, state, POP, map_call=0, n_islands=5, **KW)
vip1 = [r for r in runs if r["klass"] == "vip1"]
vip2 = [r for r in runs if r["klass"] == "vip2"]
std = [r for r in runs if r["klass"] == "std"]
check("gen 0 island 1: the seed copy", len(vip1) == 1 and vip1[0]["key"] == "S#copy1" and vip1[0]["seed"] == 3
      and vip1[0]["budget"] == 4986 and vip1[0]["blob"] is None, str(vip1))
check("gen 0 island 1: VIP 2 = 2 of the best 4", len(vip2) == 2 and len(summary["pool"]) == 4
      and all(r["budget"] == 2484 for r in vip2) and set(summary["drawn"]) <= set(summary["pool"]),
      f"pool {summary['pool']} drawn {summary['drawn']}")
check("gen 0 island 1: the sum of squares (no constants) is not run", all(r["key"] != "SOS" for r in runs))
check("gen 0 island 1: standard 486..18 in whole iterations under its 20 000",
      std and max(r["budget"] for r in std) == 486 and min(r["budget"] for r in std) == 18
      and all(r["budget"] % POP == 0 for r in std) and sum(r["budget"] for r in std) <= 20000
      and summary["std_ceiling"] == 20000, f"{len(std)} structures, {sum(r['budget'] for r in std)} evals")
check("gen 0 island 1: the call's total within 150 000 x 0.2", summary["total"] <= 30000, str(summary["total"]))
check("gen 0 island 1: launch order VIP 1, VIP 2, standard", [r["klass"] for r in runs[:3]] == ["vip1", "vip2", "vip2"])
state["S#copy1"] = {"blob": b"copy1", "evals": 4986, "verified": False, "vip": True, "best": 5.0}
runs2, _, _ = C.select_vip(island, state, POP, map_call=1, n_islands=5, **KW)
s2 = [r for r in runs2 if r["klass"] == "vip1"]
check("gen 0 island 2: an independent copy", len(s2) == 1 and s2[0]["key"] == "S#copy2" and s2[0]["seed"] == 4
      and s2[0]["budget"] == 4986 and s2[0]["blob"] is None, str(s2))

# ---- the merge ----------------------------------------------------------------------------------------------------
state = {"S#copy1": {"blob": b"c1", "evals": 4986, "best": 5.0, "vip": True},
         "S#copy2": {"blob": b"c2", "evals": 4986, "best": 3.0, "vip": True, "best_x": [2.0] * 10},
         "S#copy3": {"blob": b"c3", "evals": 4986, "best": 4.0, "vip": True}, "T001": {"blob": b"t", "evals": 36}}
merged = C.merge_seed_copies(state, ["S", "SOS"])
check("merge: the best copy kept under the plain key", merged == [("S", 2, 3.0, 4986, [1, 2, 3])]
      and state["S"]["blob"] == b"c2" and not any("#copy" in k for k in state) and "T001" in state, str(merged))
check("merge: a second call does nothing", C.merge_seed_copies(state, ["S", "SOS"]) == [])

# ---- a later generation (one call of the whole population: share 1.0) ---------------------------------------------
os.environ["SYMCLF_GEN_POPULATION"] = "60"
later = population(57)                                   # 60 entries
state["T000"] = {"blob": b"t0", "evals": 2484, "vip2": True, "vip": True}       # drawn before: 486 more if drawn
state["T002"] = {"blob": b"t2", "evals": 2995, "vip2": True, "vip": True}       # VIP-2 lifetime spent: out of the pool
runs, groups, summary = C.select_vip(later, state, POP, map_call=5, n_islands=5, **KW)
vip1 = [r for r in runs if r["klass"] == "vip1"]
vip2 = {r["key"]: r for r in runs if r["klass"] == "vip2"}
std = [r for r in runs if r["klass"] == "std"]
check("later: the merged seed gets 990 more", len(vip1) == 1 and vip1[0]["key"] == "S" and vip1[0]["budget"] == 990
      and vip1[0]["blob"] == b"c2", str(vip1))
check("later: VIP 2 = 10 of the best 20", len(vip2) == 10 and len(summary["pool"]) == 20 and "T002" not in summary["pool"]
      and set(vip2) <= set(summary["pool"]), f"drawn {sorted(vip2)}")
check("later: a verified structure ranks first in the pool", summary["pool"][0] == "V", str(summary["pool"][:3]))
check("later: budgets 2484 the first time (486 if drawn before)",
      all(r["budget"] == (486 if k == "T000" else 2484) for k, r in vip2.items()), str({k: r["budget"] for k, r in vip2.items()}))
check("later: standard excludes the drawn, ceiling 100 000", not ({r["key"] for r in std} & set(vip2))
      and summary["std_ceiling"] == 100000 and max(r["budget"] for r in std) == 486, f"{len(std)} standard")
check("later: the total within 150 000", summary["total"] <= 150000, str(summary["total"]))
again, _, summary_again = C.select_vip(later, state, POP, map_call=5, n_islands=5, **KW)
check("later: the draw is reproducible", [(r["key"], r["budget"]) for r in again] == [(r["key"], r["budget"]) for r in runs])
other, _, summary_other = C.select_vip(later, state, POP, map_call=6, n_islands=5, **KW)
check("later: another call draws anew", summary_other["drawn"] != summary["drawn"],
      f"{summary['drawn']} vs {summary_other['drawn']}")
seen = dict(state, **{f"T{i:03d}": {"blob": b"t", "evals": 36, "vip2": True, "vip": True} for i in range(25)})
runs_seen, _, _ = C.select_vip(later, seen, POP, map_call=5, n_islands=5, **KW)
again_budgets = [r["budget"] for r in runs_seen if r["klass"] == "vip2"]
check("later: a structure drawn before gets 486 (500 in whole iterations)", len(again_budgets) == 10
      and all(b == 486 for b in again_budgets), str(again_budgets))

# ---- a small total_max trims VIP 2 and the standard class ---------------------------------------------------------
small = dict(KW, total_max=6000)
runs, _, summary = C.select_vip(later, state, POP, map_call=5, n_islands=5, **small)
check("trim: total within 6000, the seed first, no standard", summary["total"] <= 6000 and runs[0]["klass"] == "vip1"
      and not [r for r in runs if r["klass"] == "std"], f"{[(r['klass'], r['budget']) for r in runs]}")

print("CMA VIP UNIT:", "PASS" if not failures else f"FAIL {failures}")
sys.exit(1 if failures else 0)
