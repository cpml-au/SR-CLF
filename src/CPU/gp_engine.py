"""What happens to the trees of one generation (the CPU pipeline).

flex hands every generation's trees to persistent_actor_mapper, which returns the constants and the fitness
of each tree:

  gate    a tree with stages.gate.max_length nodes or more, or one the encoder cannot read: fitness 1e8.
  grid    the grid stage at the constants the tree arrives with (its parent's; all ones when the number of
          constants differs).  A dead tree (a state missing, a nested exp or aq, V not finite) keeps its grid
          score and is not checked.
  check   the check and the fitness rule, once per distinct (expression, constants); the results are kept for
          the whole run.
  cma     CMA-ES on the constants of the live trees, with the same fitness as objective (stages.tune.mode):
            cma_all   option (b): every live structure; src/cma_all.py::select decides who gets how many
                      evaluations
            cma_vip   option (c): the seed template first, then a draw from the best, then the rest;
                      src/cma_all.py::select_vip
          Trees with the same expression share one search.  A search is saved when its budget of the
          generation is spent and continues the next time the structure appears.

    fitness            = grid_score + check_cost + complexity_penalty                     (src/CPU/rule.py)
    complexity_penalty = nested_penalty * (nested trig + nested exp + nested aq) + length_reg * nodes

The work runs on Ray.  One single-thread worker per core (CPUFitnessActor) does the grid stage, the checks and
the CMA evaluations.  Each CMA search is its own Ray task, because pycma seeds numpy's global random generator
and two searches must not share a process.

A result is "numerically verified" when c_v* > verification.pd_eps and kappa* > verification.kappa_min: the
verdict of a search, not a proof.  Every verified (expression, constants) pair is appended to
<job>_numerically_verified.csv in the run folder.

Settings read here: stages.gate.max_length, stages.tune.mode, stages.cma.{popsize, seed, max_restarts},
stages.cma_all.*, stages.cma_vip.*, fitness.{nested_penalty, length_reg}; environment only:
SYMCLF_CMA_POOL_LOOPS.
"""
from __future__ import annotations

import os
import sys
import time
from collections import OrderedDict

import numpy as np
import ray

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))   # src/CPU -> src -> repository root
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---- settings: config.yaml -> src/settings.py -> the environment, read once when this module is imported
MAX_TREE_LENGTH = int(os.environ.get("SYMCLF_MAX_TREE_LENGTH", "90"))   # stages.gate.max_length
NESTED_PENALTY = float(os.environ.get("SYMCLF_FIT_NESTED_PENALTY", "50"))   # fitness.nested_penalty
CMA_ELITE_POPSIZE = int(os.environ.get("SYMCLF_CMA_ELITE_POPSIZE", "18"))   # stages.cma.popsize: evaluations per CMA iteration
CMA_ELITE_SEED = int(os.environ.get("SYMCLF_CMA_ELITE_SEED", "1"))   # stages.cma.seed: the seed of every CMA search
CMA_MAX_RESTARTS = int(os.environ.get("SYMCLF_CMA_MAX_RESTARTS", "100"))   # stages.cma.max_restarts: restarts of one search before it gives up
CMA_LOOPS = int(os.environ.get("SYMCLF_CMA_POOL_LOOPS", "0"))   # CMA searches running at the same time; 0 = workers / popsize
TUNE_MODE = (os.environ.get("SYMCLF_TUNE_MODE") or "").strip().lower()   # stages.tune.mode
if TUNE_MODE not in ("cma_all", "cma_vip"):
    raise ValueError(f"stages.tune.mode (SYMCLF_TUNE_MODE) must be cma_all or cma_vip, not {TUNE_MODE!r}")
# option (b), cma_all (src/cma_all.py::select): every live structure, ranked; per generation the first rank
# gets CMA_ALL_MAX evaluations and the last CMA_ALL_MIN, never more than CMA_ALL_TOTAL together; a structure
# stops at its lifetime cap (CMA_ALL_CAP_VERIFIED once numerically verified, CMA_ALL_CAP_OTHER before).
CMA_ALL_TOTAL = int(os.environ.get("SYMCLF_CMA_ALL_TOTAL_EVALS", "100000"))
CMA_ALL_MAX = int(os.environ.get("SYMCLF_CMA_ALL_MAX_EVALS", "108"))   # rank 1, per generation
CMA_ALL_MIN = int(os.environ.get("SYMCLF_CMA_ALL_MIN_EVALS", "18"))    # the last rank, per generation
CMA_ALL_CAP_VERIFIED = int(os.environ.get("SYMCLF_CMA_ALL_CAP_VERIFIED", "5000"))
CMA_ALL_CAP_OTHER = int(os.environ.get("SYMCLF_CMA_ALL_CAP_OTHER", "3000"))
# option (c), cma_vip (src/cma_all.py::select_vip): the seed template (SEED_FIRST per island copy in
# generation 0, SEED_AGAIN each later time, lifetime SEED_CAP), then VIP_COUNT trees drawn from the best
# VIP_POOL (FIRST, AGAIN, CAP), then all the others as in option (b) between STD_MAX and STD_MIN sharing
# STD_TOTAL; never more than TOTAL_MAX evaluations per generation.
CMA_VIP_SEED_FIRST = int(os.environ.get("SYMCLF_CMA_VIP_SEED_FIRST", "5000"))
CMA_VIP_SEED_AGAIN = int(os.environ.get("SYMCLF_CMA_VIP_SEED_AGAIN", "1000"))
CMA_VIP_SEED_CAP = int(os.environ.get("SYMCLF_CMA_VIP_SEED_CAP", "6000"))
CMA_VIP_POOL = int(os.environ.get("SYMCLF_CMA_VIP_POOL", "20"))
CMA_VIP_COUNT = int(os.environ.get("SYMCLF_CMA_VIP_COUNT", "10"))
CMA_VIP_FIRST = int(os.environ.get("SYMCLF_CMA_VIP_FIRST", "2500"))
CMA_VIP_AGAIN = int(os.environ.get("SYMCLF_CMA_VIP_AGAIN", "500"))
CMA_VIP_CAP = int(os.environ.get("SYMCLF_CMA_VIP_CAP", "3000"))
CMA_VIP_STD_TOTAL = int(os.environ.get("SYMCLF_CMA_VIP_STD_TOTAL", "100000"))
CMA_VIP_STD_MAX = int(os.environ.get("SYMCLF_CMA_VIP_STD_MAX", "500"))
CMA_VIP_STD_MIN = int(os.environ.get("SYMCLF_CMA_VIP_STD_MIN", "18"))
CMA_VIP_TOTAL_MAX = int(os.environ.get("SYMCLF_CMA_VIP_TOTAL_MAX", "150000"))
CMA_VIP_DRAW_SEED = int(os.environ.get("SYMCLF_CMA_VIP_DRAW_SEED", "0"))

# ---- what one run remembers (in the driver process)
_CHECK_CACHE = OrderedDict()   # (expression, constants) -> the check's result and cost
_CHECK_CACHE_LIMIT = 65536
SEED_EXPRESSIONS = []          # the seed trees; set by the runner (src/CPU/run.py)
_SEED_CONSTANTS = [None]


def _seed_constants():
    """{seed expression: its constants} of SYMCLF_CPU_SEED_CONSTANTS, read once per process; {} when unset."""
    if _SEED_CONSTANTS[0] is None:
        path = os.environ.get("SYMCLF_CPU_SEED_CONSTANTS", "")
        if path:
            import json   # noqa: PLC0415
            with open(path) as handle:
                _SEED_CONSTANTS[0] = json.load(handle)
        else:
            _SEED_CONSTANTS[0] = {}
    return _SEED_CONSTANTS[0]
_CMA_ALL_STATE = {}   # expression -> {"blob": the saved search, "evals": lifetime evaluations, "verified", ...}
_VIP_MERGED = [False]   # option (c): the seed copies of generation 0 are merged once, at the first later call
# every numerically verified (expression, constants) pair of the run goes, once, to
# <job>_numerically_verified.csv in the run folder (logging only)
_MAP_CALL = [0]
_VERIFIED_SEEN = set()
_VERIFIED_FIELDS = ("job", "map_call", "generation", "island", "stage", "fitness", "c_v", "kappa",
                    "grid_sublevel_fraction", "cma_numerically_verified_at", "n_nodes", "n_constants",
                    "expression", "constants", "time")


def _record_numerically_verified(entries, map_call):
    """Append the numerically verified entries not seen before; generation / island from
    src/cma_all.py::generation_island (SYMCLF_N_ISLANDS, set by src/CPU/run.py)."""
    import csv    # noqa: PLC0415
    import json   # noqa: PLC0415
    from src.cma_all import generation_island   # noqa: PLC0415
    generation, island = generation_island(map_call)
    rows = []
    for entry in entries:
        if not entry.get("numerically_verified"):
            continue
        consts = entry.get("consts")
        consts = [] if consts is None else [float(v) for v in np.asarray(consts, dtype=float).reshape(-1)]
        expression = str(entry.get("expression"))
        key = (expression, tuple(consts))
        if key in _VERIFIED_SEEN:
            continue
        _VERIFIED_SEEN.add(key)
        numbers = entry.get("check") if isinstance(entry.get("check"), dict) else {}
        cma = entry.get("cma") or {}
        grid = entry.get("grid") if isinstance(entry.get("grid"), dict) else {}
        rows.append({
            "job": os.environ.get("SLURM_JOB_ID", "local"),
            "map_call": map_call,
            "generation": generation,
            "island": island,
            "stage": "cma" if (int(cma.get("evals") or 0) > 0 or entry.get("from_cma")) else "tune",
            "fitness": repr(float(entry["fitness"][0])),
            "c_v": repr(numbers.get("cv")),
            "kappa": repr(numbers.get("kappa")),
            "grid_sublevel_fraction": repr(grid.get("sublevel_fraction")),
            "cma_numerically_verified_at": cma.get("numerically_verified_at", ""),
            "n_nodes": expression.count("(") + expression.count(",") + 1,
            "n_constants": len(consts),
            "expression": expression,
            "constants": json.dumps([repr(v) for v in consts]),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
    if not rows:
        return 0
    path = f"{os.environ.get('SLURM_JOB_ID', 'local')}_numerically_verified.csv"
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_VERIFIED_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


# ----------------------------------------------------------------------------------------------- one tree
def _grid(expression, constants):
    from src.CPU.lattice import lattice_numbers   # noqa: PLC0415

    grid = lattice_numbers(expression, np.asarray(constants, dtype=np.float64))
    value = float(grid["grid_score"])
    return (value if np.isfinite(value) and value < 1.0e10 else 1.0e10), grid


def _extra(expression, length, nested_trig):
    from src.SymFunctions import detect_nested_function_calls   # noqa: PLC0415

    nested = float(nested_trig) + detect_nested_function_calls(expression, "exp") \
        + detect_nested_function_calls(expression, "aq")
    reg = float(os.environ.get("SYMCLF_LENGTH_REG", "0.005"))
    return float(NESTED_PENALTY * nested + reg * float(length))


def stage_tune(item):
    """gate + grid stage for one tree: {expression, length, nested_trig, arrival} -> its constants and grid numbers."""
    from src.CPU.encode import encode_expression   # noqa: PLC0415
    from src.CPU.evalnb import _N                   # noqa: PLC0415

    expression, length = str(item["expression"]), int(item["length"])
    if length >= MAX_TREE_LENGTH:
        return {"expression": expression, "consts": None, "fitness": (1.0e8,), "needs_check": False, "gate": "length"}
    try:
        n = int(encode_expression(expression, _N).n_constants)
    except Exception:
        return {"expression": expression, "consts": None, "fitness": (1.0e8,), "needs_check": False, "gate": "encode"}
    extra = _extra(expression, length, item.get("nested_trig", 0))
    started = time.perf_counter()
    if n == 0:
        value, grid = _grid(expression, np.zeros(0))
        consts = np.zeros(0)
    else:   # the constants the tree arrives with (its parent's, or all ones); CMA tunes them afterwards
        from src.cma_all import arrival_constants   # noqa: PLC0415
        consts = arrival_constants(item.get("arrival"), n)
        # a seed tree that comes with its own constants (SYMCLF_CPU_SEED_CONSTANTS, a json file
        # {expression: [constants]}; unset = off) starts from them instead of all ones
        given = _seed_constants().get(expression)
        if item.get("arrival") is None and given is not None and len(given) == n:
            consts = np.asarray(given, dtype=np.float64)
            print(f"seed constants given: {n} constants, tree of {length} nodes", flush=True)
        value, grid = _grid(expression, consts)
    grid_score = float(grid["grid_score"])
    dead = bool(grid["symbolic"] >= 1.0e6 or grid["invalid"] >= 1.0e6
                or not np.isfinite(grid_score) or grid_score >= 1.0e10)
    entry = {"expression": expression, "consts": consts, "complexity_penalty": extra, "grid": grid,
             "grid_score": grid_score, "dead": dead, "needs_check": not dead,
             "tune_seconds": time.perf_counter() - started}
    if dead:
        entry["fitness"] = (float(grid_score + extra),)
    return entry


def stage_check(entry):
    """check for one live tree: the check + the check cost (src/CPU/rule.py), at the tuned constants and
    their grid numbers."""
    from src.CPU.rule import fitness_fixed_constants   # noqa: PLC0415

    started = time.perf_counter()
    r = fitness_fixed_constants(entry["expression"], entry["consts"], grid=entry["grid"], extra=entry["complexity_penalty"])
    out = dict(entry)
    out.update({"fitness": (float(r["fitness"]),), "check": r["check"], "check_cost": r["check_cost"],
                "numerically_verified": r["numerically_verified"], "dead": r["dead"], "needs_check": False,
                "check_seconds": time.perf_counter() - started})
    return out


def fixed_constants(item):
    """The fitness at GIVEN constants (the CMA objective): grid stage -> dead rule -> check -> check cost."""
    from src.CPU.rule import fitness_fixed_constants   # noqa: PLC0415

    r = fitness_fixed_constants(item["expression"], np.asarray(item["consts"], dtype=np.float64),
                                extra=item["complexity_penalty"])
    return {"expression": item["expression"], "consts": np.asarray(item["consts"], dtype=np.float64),
            "complexity_penalty": item["complexity_penalty"], "grid": r["grid"], "grid_score": r["grid_score"],
            "fitness": (float(r["fitness"]),), "check": r["check"], "check_cost": r["check_cost"],
            "numerically_verified": r["numerically_verified"], "dead": r["dead"], "needs_check": False}


# ----------------------------------------------------------------------------------------------- the actors
@ray.remote(num_cpus=1, max_restarts=0)
class CPUFitnessActor:
    """One single-thread worker (NUMBA_NUM_THREADS=1: the most evaluations per core, measured on the
    4-D cart-pole)."""

    def __init__(self):
        from src.CPU import config   # noqa: F401,PLC0415  (reads the environment once)

    def ready(self):
        import socket   # noqa: PLC0415
        return f"{socket.gethostname()}:{os.environ.get('NUMBA_NUM_THREADS')}"

    def tune(self, items):
        return [stage_tune(item) for item in items]

    def check(self, entries):
        return [stage_check(entry) for entry in entries]

    def fixed(self, items):
        return [fixed_constants(item) for item in items]


def create_actor_pool(count):
    options = CPUFitnessActor.options(runtime_env={"env_vars": {"NUMBA_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}})
    actors = [options.remote() for _ in range(int(count))]
    hosts = ray.get([a.ready.remote() for a in actors])
    return actors, hosts


def _actor_map(items, *, actors, batch_size, method_name):
    """Hand the batches to the workers: every worker gets a new batch as soon as it has finished one."""
    batches = [(start, items[start:start + batch_size]) for start in range(0, len(items), batch_size)]
    results = [None] * len(items)
    active = {}
    next_batch = 0

    def submit(actor, start, batch):
        active[getattr(actor, method_name).remote(batch)] = (actor, start, len(batch))

    for actor in actors:
        if next_batch >= len(batches):
            break
        start, batch = batches[next_batch]
        submit(actor, start, batch)
        next_batch += 1
    while active:
        ready, _ = ray.wait(list(active), num_returns=1)
        actor, start, size = active.pop(ready[0])
        out = ray.get(ready[0])
        if len(out) != size:
            raise RuntimeError(f"actor returned the wrong {method_name} batch length")
        results[start:start + size] = out
        if next_batch < len(batches):
            new_start, batch = batches[next_batch]
            submit(actor, new_start, batch)
            next_batch += 1
    return results


# ----------------------------------------------------------------------------------------------- cma: the CMA stage
def cma_refine_elite_resumable(elite, evaluate, *, evals, popsize, seed, state=None):
    """The CMA-ES search of one structure (sigma0 1, stds 0.3 max(|c0|, 0.1), popsize, seed, keep-best)
    as a plain pycma ask/tell loop that can be SLICED: `state` is the blob of an earlier slice -- (the pycma object,
    numpy's global RNG state, the restart count, the running best point) -- so the search continues exactly where it
    stopped. A warm restart from the running best follows any stop of pycma's own criteria (at most
    SYMCLF_CMA_MAX_RESTARTS). Returns (best_entry or None, info, blob)."""
    import pickle   # noqa: PLC0415
    import cma      # noqa: PLC0415

    expression = str(elite["expression"])
    c0 = np.asarray(elite["consts"], dtype=np.float64).reshape(-1)
    f0 = float(elite["fitness"][0])
    extra = float(elite["complexity_penalty"])
    started = time.perf_counter()
    best, best_entry, verified_at, n, n_verified = f0, None, None, 0, 0

    def _new_es(x0, run_seed):
        x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
        return cma.CMAEvolutionStrategy(x0.tolist(), 1.0, {
            "CMA_stds": (0.3 * np.maximum(np.abs(x0), 0.1)).tolist(), "popsize": int(popsize),
            "seed": int(run_seed), "verbose": -9, "verb_disp": 0, "verb_log": 0})

    if state is None:
        es = _new_es(c0, seed)
        restarts, rng, best_x = 0, np.random.get_state(), c0.copy()
    else:
        es, rng, restarts, best_x = pickle.loads(state)
    np.random.set_state(rng)

    exhausted = False
    while n < int(evals):
        if es.stop():
            if restarts >= CMA_MAX_RESTARTS:
                exhausted = True
                break
            restarts += 1
            es = _new_es(best_x, int(seed) + 1000 * restarts)
        xs = es.ask()
        results = evaluate([{"expression": expression, "consts": np.asarray(x, dtype=np.float64),
                             "complexity_penalty": extra} for x in xs])
        fits = []
        for result in results:
            value = float(result["fitness"][0])
            n += 1
            if result.get("numerically_verified"):
                n_verified += 1
                if verified_at is None:
                    verified_at = n
            if np.isfinite(value) and value < best - 1.0e-9:
                best, best_entry = value, result
                best_x = np.asarray(result["consts"], dtype=np.float64).reshape(-1)
            fits.append(value if np.isfinite(value) else 1.0e12)
        es.tell(xs, fits)
    blob = pickle.dumps((es, np.random.get_state(), restarts, best_x))
    info = {"n_constants": int(c0.size), "before": f0, "after": float(best),
            "improved": best_entry is not None, "evals": int(n), "popsize": int(popsize), "seed": int(seed),
            "numerically_verified_at": verified_at, "n_numerically_verified": int(n_verified), "restarts": int(restarts),
            "exhausted": bool(exhausted), "resumed": state is not None,
            "seconds": time.perf_counter() - started}
    return best_entry, info, blob


@ray.remote(num_cpus=0, max_retries=0)
def _cma_loop_task_resumable(elite, actors, slot, evals, popsize, seed, state):
    rotated = actors[(slot * popsize) % len(actors):] + actors[:(slot * popsize) % len(actors)]

    def evaluate(items):
        return _actor_map(items, actors=rotated, batch_size=1, method_name="fixed")

    return cma_refine_elite_resumable(elite, evaluate, evals=evals, popsize=popsize, seed=seed, state=state)


def _cma_all_stage(entries, actors, cma_infos):
    """Option (b): CMA on every live structure (src/cma_all.py: ranking, budgets, caps), all loops at once, each loop
    in its own Ray task with its population on a rotated slice of the actor pool; every individual of a structure
    receives the CMA's best when it is better.  Edits `entries` in place, appends to `cma_infos`."""
    from src.cma_all import select   # noqa: PLC0415

    selected, budgets, blobs, groups = select(
        entries, _CMA_ALL_STATE, CMA_ELITE_POPSIZE, total=CMA_ALL_TOTAL, first=CMA_ALL_MAX, last=CMA_ALL_MIN,
        cap_verified=CMA_ALL_CAP_VERIFIED, cap_other=CMA_ALL_CAP_OTHER,
        numbers_of=lambda e: e.get("check") if isinstance(e.get("check"), dict) else None)
    loops = CMA_LOOPS or max(1, len(actors) // max(1, CMA_ELITE_POPSIZE))
    pending, next_slot = {}, 0

    def _launch(slot):
        pending[_cma_loop_task_resumable.remote(entries[selected[slot]], actors, slot, int(budgets[slot]),
                                                CMA_ELITE_POPSIZE, CMA_ELITE_SEED, blobs[slot])] = slot

    while next_slot < min(loops, len(selected)):
        _launch(next_slot)
        next_slot += 1
    while pending:
        ready, _ = ray.wait(list(pending), num_returns=1)
        slot = pending.pop(ready[0])
        best_entry, info, blob = ray.get(ready[0])
        index = selected[slot]
        expression = str(entries[index]["expression"])
        held = _CMA_ALL_STATE.setdefault(expression, {"blob": None, "evals": 0, "verified": False})
        held["evals"] += int(info["evals"])
        held["blob"] = blob
        if info.get("numerically_verified_at") is not None or entries[index].get("numerically_verified"):
            held["verified"] = True
        info["evals_total"] = held["evals"]
        if best_entry is not None:
            for i in groups[expression]:
                if float(best_entry["fitness"][0]) < float(entries[i]["fitness"][0]):
                    entries[i] = dict(best_entry, from_cma=True)
        representative = dict(entries[index])
        representative["cma"] = info
        entries[index] = representative
        cma_infos.append(info)
        if next_slot < len(selected):
            _launch(next_slot)
            next_slot += 1
    spent = sum(int(i["evals"]) for i in cma_infos)
    print(f"cma_all: {len(selected)} structures ranked, {sum(budgets)} evaluations allotted "
          f"({min(budgets) if budgets else 0}..{max(budgets) if budgets else 0} per structure), {spent} spent, "
          f"{sum(1 for i in cma_infos if i.get('numerically_verified_at') is not None)} numerically verified, "
          f"{loops} loops in flight, {len(_CMA_ALL_STATE)} structures tracked", flush=True)


def _vip_prepare(items):
    """Option (c), before the tune stage of a mapper call: at the first call after generation 0 the seed's island
    copies are merged into the best one (src/cma_all.py::merge_seed_copies); every item of a VIP structure (VIP 1 or
    VIP 2 before) then arrives with the best constants its CMA has found, instead of its parent's."""
    from src.cma_all import merge_seed_copies   # noqa: PLC0415

    n_islands = int(os.environ.get("SYMCLF_N_ISLANDS", "0") or 0)
    if _MAP_CALL[0] >= n_islands and not _VIP_MERGED[0]:
        _VIP_MERGED[0] = True
        for expression, island, best, evals, islands in merge_seed_copies(_CMA_ALL_STATE, SEED_EXPRESSIONS):
            print(f"cma_vip merge: seed [{expression[:50]}...] copies of islands {islands} -> island {island} "
                  f"(best {best:.6g}, {evals} evaluations)", flush=True)
    for item in items:
        held = _CMA_ALL_STATE.get(item["expression"])
        if held and held.get("vip") and held.get("best_x") is not None:
            item["arrival"] = np.asarray(held["best_x"], dtype=np.float64)


def _cma_vip_stage(entries, actors, cma_infos):
    """Option (c): _cma_all_stage with the budgets of src/cma_all.py::select_vip (VIP 1, VIP 2, standard); each run
    resumes the state under its key (a seed copy of generation 0 has its own key and pycma seed).  Edits `entries` in
    place, appends to `cma_infos`."""
    from src.cma_all import select_vip   # noqa: PLC0415

    runs, groups, summary = select_vip(
        entries, _CMA_ALL_STATE, CMA_ELITE_POPSIZE, map_call=_MAP_CALL[0],
        n_islands=int(os.environ.get("SYMCLF_N_ISLANDS", "0") or 0), seed_expressions=SEED_EXPRESSIONS,
        base_seed=CMA_ELITE_SEED, seed_first=CMA_VIP_SEED_FIRST, seed_again=CMA_VIP_SEED_AGAIN, seed_cap=CMA_VIP_SEED_CAP,
        vip_pool=CMA_VIP_POOL, vip_count=CMA_VIP_COUNT, vip_first=CMA_VIP_FIRST, vip_again=CMA_VIP_AGAIN,
        vip_cap=CMA_VIP_CAP, std_total=CMA_VIP_STD_TOTAL, std_first=CMA_VIP_STD_MAX, std_last=CMA_VIP_STD_MIN,
        cap_verified=CMA_ALL_CAP_VERIFIED, cap_other=CMA_ALL_CAP_OTHER, total_max=CMA_VIP_TOTAL_MAX,
        draw_seed=CMA_VIP_DRAW_SEED, numbers_of=lambda e: e.get("check") if isinstance(e.get("check"), dict) else None)
    loops = CMA_LOOPS or max(1, len(actors) // max(1, CMA_ELITE_POPSIZE))
    pending, next_slot = {}, 0

    def _launch(slot):
        run = runs[slot]
        pending[_cma_loop_task_resumable.remote(entries[run["index"]], actors, slot, int(run["budget"]),
                                                CMA_ELITE_POPSIZE, int(run["seed"]), run["blob"])] = slot

    while next_slot < min(loops, len(runs)):
        _launch(next_slot)
        next_slot += 1
    while pending:
        ready, _ = ray.wait(list(pending), num_returns=1)
        slot = pending.pop(ready[0])
        best_entry, info, blob = ray.get(ready[0])
        run = runs[slot]
        index = run["index"]
        expression = str(entries[index]["expression"])
        held = _CMA_ALL_STATE.setdefault(run["key"], {"blob": None, "evals": 0, "verified": False})
        held["evals"] += int(info["evals"])
        held["blob"] = blob
        if info.get("numerically_verified_at") is not None or entries[index].get("numerically_verified"):
            held["verified"] = True
        if run["klass"] != "std":
            held["vip"] = True
        if run["klass"] == "vip2":
            held["vip2"] = True
        found = float(info["after"])   # the best of this slice (its start point when nothing was better)
        if np.isfinite(found) and found < float(held.get("best", np.inf)):
            held["best"] = found
            held["best_x"] = np.asarray((best_entry or entries[index])["consts"], dtype=np.float64).tolist()
        info["evals_total"] = held["evals"]
        info["vip_class"] = run["klass"]
        if best_entry is not None:
            for i in groups[expression]:
                if float(best_entry["fitness"][0]) < float(entries[i]["fitness"][0]):
                    entries[i] = dict(best_entry, from_cma=True)
        representative = dict(entries[index])
        representative["cma"] = info
        entries[index] = representative
        cma_infos.append(info)
        if run["klass"] != "std":
            print(f"cma_vip {run['klass']} [{run['key'][:60]}]: {info['before']:.6g} -> {info['after']:.6g} in "
                  f"{int(info['evals'])} evals (lifetime {held['evals']}), numerically_verified_at "
                  f"{info['numerically_verified_at']}, seed {run['seed']}, resumed {info.get('resumed')}", flush=True)
        if next_slot < len(runs):
            _launch(next_slot)
            next_slot += 1
    spent = {k: sum(int(i["evals"]) for i in cma_infos if i.get("vip_class") == k) for k in ("vip1", "vip2", "std")}
    count = {k: sum(1 for r in runs if r["klass"] == k) for k in ("vip1", "vip2", "std")}
    std = [r["budget"] for r in runs if r["klass"] == "std"]
    print(f"cma_vip: vip1 {count['vip1']} runs {spent['vip1']} evals, vip2 {count['vip2']} of a pool of "
          f"{len(summary['pool'])} {spent['vip2']} evals, standard {count['std']} structures "
          f"({min(std) if std else 0}..{max(std) if std else 0} each, ceiling {summary['std_ceiling']:.0f}) "
          f"{spent['std']} evals; allotted {summary['total']} of the limit {summary['limit']:.0f}, "
          f"{sum(1 for i in cma_infos if i.get('numerically_verified_at') is not None)} numerically verified, "
          f"{loops} loops in flight, {len(_CMA_ALL_STATE)} states held", flush=True)


# ----------------------------------------------------------------------------------------------- the mapper
def _key(entry):
    return (entry["expression"], tuple(np.round(np.asarray(entry["consts"], dtype=np.float64), 15).tolist()))


def _breakdown(e):
    g = e.get("grid") or {}
    d4 = e.get("check") or {}
    return {
        "grid_score": e.get("grid_score"), "complexity_penalty": e.get("complexity_penalty"), "dead": bool(e.get("dead")),
        "grid_roa": g.get("roa"), "grid_cv": g.get("cv_grid"), "grid_kappa": g.get("kappa_grid"),
        "grid_boundary_min": g.get("boundary_min"), "grid_w_scale": g.get("w_scale"),
        "grid_sublevel_fraction": g.get("sublevel_fraction"),
        "check": {k: d4.get(k) for k in ("cv", "kappa", "worst_r", "penalty_terms", "pd_valid")} if d4 else None,
        "check_cost": e.get("check_cost"), "numerically_verified": e.get("numerically_verified"), "cma": e.get("cma"),
        "gate": e.get("gate"),
    }


def persistent_actor_mapper(_function, individuals, *, actors, batch_size=4):
    from src.SymFunctions import get_features_batch   # noqa: PLC0415

    individuals = list(individuals)
    if not individuals:
        return []
    lengths, nested, _ = get_features_batch(individuals)
    for i, ind in enumerate(individuals):   # log line: what arrives for the seed trees
        if str(ind) in SEED_EXPRESSIONS:
            print(f"seed item: nodes={len(ind)} features_length={int(lengths[i])} nested={float(nested[i])} "
                  f"n_features={len(lengths)} n_individuals={len(individuals)} expr={str(ind)[:40]}", flush=True)
    items = [{"expression": str(ind), "length": int(lengths[i]), "nested_trig": float(nested[i])}
             for i, ind in enumerate(individuals)]
    for i, ind in enumerate(individuals):   # the parent's constants travel with the offspring
        items[i]["arrival"] = getattr(ind, "consts", None)
    if TUNE_MODE == "cma_vip":   # option (c): the seed copies merged; a VIP structure arrives with its best constants
        _vip_prepare(items)

    t0 = time.perf_counter()
    entries = _actor_map(items, actors=actors, batch_size=int(batch_size), method_name="tune")
    tune_s = time.perf_counter() - t0

    # check: once per distinct (expression, constants), driver cache across generations
    t0 = time.perf_counter()
    groups = OrderedDict()
    for i, e in enumerate(entries):
        if e.get("needs_check"):
            groups.setdefault(_key(e), []).append(i)
    todo, hits = [], 0
    for key, positions in groups.items():
        cached = _CHECK_CACHE.get(key)
        if cached is None:
            todo.append(key)
            continue
        _CHECK_CACHE.move_to_end(key)
        hits += len(positions)
        for i in positions:
            entries[i] = dict(entries[i], **cached)
    if todo:
        checked = _actor_map([entries[groups[k][0]] for k in todo], actors=actors, batch_size=1, method_name="check")
        for key, e in zip(todo, checked):
            cached = {k: e[k] for k in ("fitness", "check", "check_cost", "numerically_verified", "dead", "needs_check")}
            _CHECK_CACHE[key] = cached
            while len(_CHECK_CACHE) > _CHECK_CACHE_LIMIT:
                _CHECK_CACHE.popitem(last=False)
            for i in groups[key]:
                entries[i] = dict(entries[i], **cached)
    check_s = time.perf_counter() - t0

    # cma: CMA-ES on the constants of the live structures, one Ray task per search
    t0 = time.perf_counter()
    cma_infos = []
    for e in entries:   # log line: what the seed trees look like when they reach the CMA stage
        if str(e.get("expression", "")) in SEED_EXPRESSIONS:
            d4 = e.get("check") or {}
            print(f"seed entry: len={len(str(e.get('expression', '')))} gate={e.get('gate')} dead={e.get('dead')} "
                  f"consts={'None' if e.get('consts') is None else np.asarray(e['consts']).size} fitness={e.get('fitness')} "
                  f"needs_check={e.get('needs_check')} check_status={d4.get('status') if isinstance(d4, dict) else None} "
                  f"grid_score={e.get('grid_score')} keys={sorted(e.keys())[:12]}", flush=True)
    if TUNE_MODE == "cma_all":
        _cma_all_stage(entries, actors, cma_infos)
    else:
        _cma_vip_stage(entries, actors, cma_infos)
    cma_seconds = time.perf_counter() - t0

    n_new_verified = _record_numerically_verified(entries, _MAP_CALL[0])   # logging only
    _MAP_CALL[0] += 1
    verified = sum(1 for e in entries if e.get("numerically_verified"))
    fits = [float(e["fitness"][0]) for e in entries if e.get("fitness")]
    tune = [e["tune_seconds"] for e in entries if e.get("tune_seconds") is not None]
    print("stages: "
          f"candidates={len(individuals)} tune_s={tune_s:.1f} tune_mean_s={np.mean(tune) if tune else 0:.2f} "
          f"check_candidates={sum(len(p) for p in groups.values())} check_unique={len(groups)} check_computed={len(todo)} "
          f"check_cached={hits} check_s={check_s:.1f} dead={sum(1 for e in entries if e.get('dead'))} "
          f"numerically_verified={verified} fitness_min={min(fits, default=float('nan')):.6g} "
          f"cma_runs={len(cma_infos)} cma_numerically_verified={sum(1 for c in cma_infos if c['numerically_verified_at'] is not None)} "
          f"cma_best={min((c['after'] for c in cma_infos), default=float('nan')):.6g} cma_s={cma_seconds:.1f} "
          f"cache_entries={len(_CHECK_CACHE)} "
          f"seed_exprs={len(SEED_EXPRESSIONS)} seeds_in_entries={sum(1 for e in entries if str(e.get('expression', '')) in SEED_EXPRESSIONS)} "
          f"states={len(_CMA_ALL_STATE)} numerically_verified_new={n_new_verified}", flush=True)
    return [{"consts": e.get("consts"), "fitness": e["fitness"], "breakdown": _breakdown(e)} for e in entries]
