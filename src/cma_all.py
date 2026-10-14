"""Which trees get how many CMA evaluations: the constant-tuning options (b) and (c).

Option (b), "CMA for all" (stages.tune.mode: cma_all).  Every tree is checked once at the constants it
arrives with.  Then CMA-ES runs on EVERY live structure, ranked: numerically verified first (by fitness),
then the others by kappa* (highest first).  The budget is linear in the rank, from max_evals (rank 1) to
min_evals (the last rank), in whole CMA iterations; never more than total_evals per generation (one call
of the tree-handling function spends its share of that: call_share); and a structure has a lifetime cap
(cap_verified once it is numerically verified, cap_other before).

Option (c), "VIP classes" (stages.tune.mode: cma_vip), is described above select_vip below.

Used by src/CPU/gp_engine.py.  Python and numpy only.
"""
from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np


def cma_all_budgets(n, *, first, last, ceiling, popsize):
    """Evaluations per rank for n ranked structures: rank 1 gets `first` (108), the
    last rank `last` (18), linear in between, in whole CMA iterations of `popsize` (18) evaluations; the sum never
    exceeds `ceiling` (the generation's 100 000, or a call's share of it).  When the linear profile would exceed the
    ceiling, the iterations above the first one shrink in proportion (floor), every structure keeping at least one
    iteration; when even one iteration each does not fit, only the best floor(ceiling / popsize) ranks get one and the
    others get 0 (they are not run)."""
    if n <= 0:
        return []
    popsize = max(1, int(popsize))
    first_it = max(1, int(round(float(first) / popsize)))
    last_it = max(1, min(first_it, int(round(float(last) / popsize))))
    if n == 1:
        its = [first_it]
    else:
        its = [int(round(last_it + (first_it - last_it) * (n - 1 - r) / (n - 1))) for r in range(n)]
    room = int(float(ceiling) // popsize)            # whole iterations the ceiling allows
    if sum(its) > room:
        if n > room:                                 # not even one iteration each: the best ranks only
            its = [1] * max(0, room) + [0] * (n - max(0, room))
        else:
            extra = [i - 1 for i in its]
            scale = (room - n) / float(sum(extra)) if sum(extra) else 0.0
            its = [1 + int(e * scale) for e in extra]
    return [i * popsize for i in its]


def call_share(total, n_candidates):
    """The part of the generation's total one mapper call spends: total * (trees in the call / whole population).

    flex evaluates the initial population island by island (5 calls of 500 trees: each gets 1/5) and every later
    generation in ONE call with all islands (2500 trees: the whole total).  SYMCLF_GEN_POPULATION = the whole
    population (num_individuals x num_islands, set by src/CPU/run.py); unset: the whole total."""
    population = int(os.environ.get("SYMCLF_GEN_POPULATION", "0") or 0)
    if population <= 0:
        return total
    return total * min(1.0, float(n_candidates) / population)


def generation_island(map_call):
    """(generation, island) of the map_call-th mapper call (0-based): the first n_islands calls are the initial
    population's islands (generation 0, island 1..n); every later call is one whole generation (island "all")."""
    n_islands = int(os.environ.get("SYMCLF_N_ISLANDS", "0") or 0)
    if n_islands <= 0:
        return "", ""
    if map_call < n_islands:
        return 0, map_call + 1
    return map_call - n_islands + 1, "all"


def arrival_constants(inherited, n_constants):
    """The constants a tree arrives with: its parent's when the count matches and they are finite, else all ones
    (never the coefficients of a Riccati solution)."""
    if inherited is not None:
        inh = np.asarray(inherited, dtype=np.float64).reshape(-1)
        if inh.size == int(n_constants) and np.all(np.isfinite(inh)):
            return inh.copy()
    return np.ones(int(n_constants), dtype=np.float64)


def select(entries, state, popsize, *, total, first, last, cap_verified, cap_other, numbers_of):
    """(representative indices in rank order, budgets, resumed pycma blobs, groups).

    entries: the mapper's entries after the check (keys expression, consts, fitness, dead, numerically_verified);
    state: expression -> {"blob", "evals" (lifetime), "verified"}; numbers_of(entry) -> the check's numbers (a dict
    with "kappa") or None when the tree was not checked (then it is left out).  groups maps each structure to the
    indices of all its individuals (they all receive the CMA's best)."""
    groups, best = OrderedDict(), {}
    for index, entry in enumerate(entries):
        if entry.get("dead") or numbers_of(entry) is None:
            continue
        consts, fitness = entry.get("consts"), entry.get("fitness")
        if consts is None or np.asarray(consts).size == 0 or not fitness or not np.isfinite(float(fitness[0])):
            continue
        expression = str(entry["expression"])
        groups.setdefault(expression, []).append(index)
        if expression not in best or float(fitness[0]) < float(entries[best[expression]]["fitness"][0]):
            best[expression] = index
    candidates = []
    for expression, index in best.items():
        held = state.get(expression, {})
        verified = bool(entries[index].get("numerically_verified")) or bool(held.get("verified"))
        remaining = (cap_verified if verified else cap_other) - int(held.get("evals", 0))
        if remaining < int(popsize):
            continue   # its lifetime allowance is spent
        kappa = (numbers_of(entries[index]) or {}).get("kappa")
        kappa = float(kappa) if kappa is not None and np.isfinite(float(kappa)) else -np.inf
        key = (0, float(entries[index]["fitness"][0])) if verified else (1, -kappa)
        candidates.append((key, index, expression, remaining))
    candidates.sort(key=lambda item: item[0])
    budgets = cma_all_budgets(len(candidates), first=first, last=last, ceiling=call_share(total, len(entries)),
                              popsize=popsize)
    selected, out_budgets, blobs = [], [], []
    for (_, index, expression, remaining), budget in zip(candidates, budgets):
        budget = min(int(budget), (int(remaining) // int(popsize)) * int(popsize))
        if budget <= 0:
            continue   # outside the ceiling this generation
        selected.append(index)
        out_budgets.append(budget)
        blobs.append(state.get(expression, {}).get("blob"))
    return selected, out_budgets, blobs, groups


# ------------------------------------------------------------------------------------------------------------------
# Option (c), "VIP classes" (stages.tune.mode: cma_vip).
#   VIP 1   the seed template: every generation-0 island copy runs its OWN CMA (its own key and pycma seed) of up to
#           seed_first; from generation 1 on the copies are MERGED into the best one, and each reappearance of the
#           seed resumes it with up to seed_again more, lifetime seed_cap.
#   VIP 2   vip_count structures drawn at random from the best vip_pool of the ranked list (the seed excluded; only
#           structures with VIP-2 lifetime left): up to vip_first the first time a structure is drawn, vip_again on
#           later draws, lifetime vip_cap (all its CMA evaluations count).
#   standard  every other ranked structure: option (b)'s rule between std_first (rank 1) and std_last, sharing
#           std_total, lifetime caps cap_verified / cap_other.
# The ranking is option (b)'s (numerically verified first by fitness, then kappa* highest first).  A generation-0
# island call spends its share of vip_pool, vip_count, std_total and total_max (call_share); the seed's runs are not
# shared out.  The sum of one call's budgets never exceeds total_max x its share.
# ------------------------------------------------------------------------------------------------------------------
_COPY = "#copy"


def vip_copy_key(expression, copy):
    """The CMA state key of the seed's generation-0 copy in island `copy` (1-based)."""
    return f"{expression}{_COPY}{int(copy)}"


def merge_seed_copies(state, seed_expressions):
    """Generation >= 1: each seed's generation-0 copies -> ONE state under the plain expression, the copy with the
    lowest best fitness (ties: the lowest island); the other copies are dropped.  Returns [(expression, island kept,
    its best fitness, its lifetime evaluations, the islands merged)]; a second call finds no copies and does nothing."""
    merged = []
    for expression in seed_expressions:
        prefix = str(expression) + _COPY
        copies = sorted((int(key[len(prefix):]), key) for key in list(state)
                        if key.startswith(prefix) and key[len(prefix):].isdigit())
        if not copies:
            continue
        kept_island, kept_key = min(copies, key=lambda c: (float(state[c[1]].get("best", np.inf)), c[0]))
        kept = dict(state[kept_key])
        kept["merged_from"] = [island for island, _ in copies]
        for _, key in copies:
            del state[key]
        state[str(expression)] = kept
        merged.append((str(expression), kept_island, float(kept.get("best", np.inf)), int(kept.get("evals", 0)),
                       kept["merged_from"]))
    return merged


def _structures(entries, numbers_of):
    """One representative (the best fitness) per structure and every index of the structure: the same rules as
    select() -- dead, unchecked, constant-free and non-finite entries are left out."""
    groups, best = OrderedDict(), {}
    for index, entry in enumerate(entries):
        if entry.get("dead") or numbers_of(entry) is None:
            continue
        consts, fitness = entry.get("consts"), entry.get("fitness")
        if consts is None or np.asarray(consts).size == 0 or not fitness or not np.isfinite(float(fitness[0])):
            continue
        expression = str(entry["expression"])
        groups.setdefault(expression, []).append(index)
        if expression not in best or float(fitness[0]) < float(entries[best[expression]]["fitness"][0]):
            best[expression] = index
    return groups, best


def select_vip(entries, state, popsize, *, map_call, n_islands, seed_expressions, base_seed,
               seed_first, seed_again, seed_cap, vip_pool, vip_count, vip_first, vip_again, vip_cap,
               std_total, std_first, std_last, cap_verified, cap_other, total_max, draw_seed, numbers_of):
    """Option (c): (runs in launch order, groups, summary).  A run is {"index", "budget", "blob", "key", "seed",
    "klass"}: the representative entry, its evaluations this call (whole CMA iterations), the pycma blob to resume
    (None = a new search from the entry's constants), the state key its result goes to, the pycma seed of a new
    search, and the class (vip1, vip2, std).  state: key -> {"blob", "evals" (lifetime), "verified", "vip2"}."""
    popsize = max(1, int(popsize))

    def whole(budget):
        return (max(0, int(budget)) // popsize) * popsize

    generation0 = int(n_islands) > 0 and int(map_call) < int(n_islands)
    share = call_share(1.0, len(entries))
    groups, best = _structures(entries, numbers_of)
    seeds = {str(s) for s in seed_expressions}
    runs = []

    for expression, index in best.items():                       # VIP 1: the seed template
        if expression not in seeds:
            continue
        if generation0:
            island = int(map_call) + 1
            key, run_seed = vip_copy_key(expression, island), int(base_seed) + island
            held = state.get(key, {})
            budget = whole(min(int(seed_first), int(seed_cap) - int(held.get("evals", 0))))
        else:
            key, run_seed = expression, int(base_seed)
            held = state.get(key, {})
            step = int(seed_again) if held else int(seed_first)
            budget = whole(min(step, int(seed_cap) - int(held.get("evals", 0))))
        if budget >= popsize:
            runs.append({"index": index, "budget": budget, "blob": held.get("blob"), "key": key, "seed": run_seed,
                         "klass": "vip1"})

    ranked = []                                                   # the others, ranked as option (b)
    for expression, index in best.items():
        if expression in seeds:
            continue
        held = state.get(expression, {})
        verified = bool(entries[index].get("numerically_verified")) or bool(held.get("verified"))
        kappa = (numbers_of(entries[index]) or {}).get("kappa")
        kappa = float(kappa) if kappa is not None and np.isfinite(float(kappa)) else -np.inf
        rank_key = (0, float(entries[index]["fitness"][0])) if verified else (1, -kappa)
        ranked.append((rank_key, index, expression, verified, int(held.get("evals", 0))))
    ranked.sort(key=lambda item: item[0])

    n_pool = max(1, int(round(float(vip_pool) * share)))          # VIP 2: a random draw from the best of the list
    n_draw = max(1, int(round(float(vip_count) * share)))
    pool = [item for item in ranked if int(vip_cap) - item[4] >= popsize][:n_pool]
    rng = np.random.default_rng([int(draw_seed), int(map_call)])
    drawn = sorted(rng.choice(len(pool), size=min(n_draw, len(pool)), replace=False).tolist()) if pool else []
    drawn_expressions = set()
    vip2 = []
    for position in drawn:
        _, index, expression, _, used = pool[position]
        held = state.get(expression, {})
        step = int(vip_again) if held.get("vip2") else int(vip_first)
        budget = whole(min(step, int(vip_cap) - used))
        if budget >= popsize:
            vip2.append({"index": index, "budget": budget, "blob": held.get("blob"), "key": expression,
                         "seed": int(base_seed), "klass": "vip2"})
            drawn_expressions.add(expression)
    vip2.sort(key=lambda run: -run["budget"])                     # the longest searches start first
    room, kept = float(total_max) * share, []
    for run in runs + vip2:                                       # VIP 1, then VIP 2: never beyond the call's limit
        budget = min(int(run["budget"]), whole(room))
        if budget >= popsize:
            run["budget"] = budget
            kept.append(run)
            room -= budget
        elif run["klass"] == "vip2":
            drawn_expressions.discard(run["key"])                 # not run as VIP 2: it stays in the standard class
    runs = kept

    vip_allotted = sum(run["budget"] for run in runs)             # standard: option (b)'s rule on the rest
    ceiling = max(0.0, min(float(std_total) * share, float(total_max) * share - vip_allotted))
    standard = []
    for rank_key, index, expression, verified, used in ranked:
        if expression in drawn_expressions:
            continue
        remaining = (int(cap_verified) if verified else int(cap_other)) - used
        if remaining < popsize:
            continue
        standard.append((index, expression, remaining))
    budgets = cma_all_budgets(len(standard), first=max(popsize, whole(std_first)), last=max(popsize, whole(std_last)),
                              ceiling=ceiling, popsize=popsize)   # whole iterations never above std_first (500 -> 486)
    for (index, expression, remaining), budget in zip(standard, budgets):
        budget = min(int(budget), whole(remaining))
        if budget <= 0:
            continue
        runs.append({"index": index, "budget": budget, "blob": state.get(expression, {}).get("blob"), "key": expression,
                     "seed": int(base_seed), "klass": "std"})

    summary = {"generation0": generation0, "share": share, "pool": [pool[p][2] for p in range(len(pool))],
               "drawn": [run["key"] for run in runs if run["klass"] == "vip2"], "vip_allotted": vip_allotted,
               "std_ceiling": ceiling,
               "limit": float(total_max) * share, "total": sum(run["budget"] for run in runs)}
    assert summary["total"] <= summary["limit"] + 1e-9, summary   # the ceiling: never above total_max
    return runs, groups, summary
