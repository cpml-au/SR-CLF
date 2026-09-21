"""6-D GP engine: the 4-D V9 pipeline on the 6-D quadrotor, CPU only (2026-09-19, the user's order: "the same as
4D but the dynamics and the settings of the 6D folder").

Per generation (the mapper below replaces Flex's task mapper, as gpu_ray.py does in 4-D):
  S0  gate          tree length >= SYMCLF_MAX_TREE_LENGTH or an unencodable tree -> fitness 1e8 (V8/V9 rule)
  S1  tuner         the constants of every tree: the V9 SEA (population 35 uniform in [-R, R] + all-ones,
                    SEA_GENERATIONS steady-state steps: one offspring from the best, each coordinate mutated with
                    probability 1/n, perturbation best*exp(U[-2, 2]) with probability 0.5 else a fresh uniform
                    draw, replaces the worst when not worse), then the V9 central-difference Adam (40 steps,
                    lr 0.3, scale-aware FD step 0.08|x|, exits at the floor / at zero gradient). The objective is
                    the lattice stage's pre_exact (core.lattice.lattice_numbers, the 6-D port of the 4-D grid
                    kernel's det4-mode terms; 7^6 grid). 4-D ran this on the GPU; here it runs on the CPU actors.
  S2  the check     det4 (core.det4, the referee's algorithm for n states and m inputs) on every live tree, once
                    per distinct (expression, constants): driver cache across generations, as in 4-D.
                    Price and fitness = core.rule (the V8 arithmetic; fitness = pre_exact + price + extra).
  S2b CMA           pycma BIPOP (the 146800 recipe: sigma0 1, stds 0.3 max(|c0|, 0.1), fixed seed, keep-best) on
                    the best new structures under the gate, SYMCLF_CMA_ELITE_COUNT per generation, each loop in
                    its own Ray task (pycma seeds numpy's global stream: loops must not share a process, V10 test
                    147721), its population evaluated on the actor pool.
  S3  (4-D's det4-gradient refine of 18 elites is not ported: in 6-D b is a vector; CMA covers the constants.)
extra = NESTED_PENALTY x (nested trig + nested exp + nested aq flags) + reg_param x tree length (4-D formula).
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import ray

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (HERE, PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MAX_TREE_LENGTH = int(os.environ.get("SYMCLF_MAX_TREE_LENGTH", "90"))
SEA_POPULATION = int(os.environ.get("SYMCLF_GPU2_TUNER_RANDOM_POPULATION", "35")) + 1   # + the all-ones member
SEA_GENERATIONS = int(os.environ.get("SYMCLF_GPU2_TUNER_GENERATIONS", "5"))
CONST_RANGE = float(os.environ.get("SYMCLF_TUNER_CONST_RANGE", "30"))
PERTURB_FRAC = float(os.environ.get("SYMCLF_TUNER_CONST_PERTURB_FRAC", "0.5"))
PERTURB_SCALE = float(os.environ.get("SYMCLF_TUNER_CONST_PERTURB_SCALE", "2.0"))
REFINE_STEPS = int(os.environ.get("SYMCLF_TUNER_REFINE_STEPS", "40"))
REFINE_LR = float(os.environ.get("SYMCLF_TUNER_REFINE_LR", "0.30"))
REFINE_FD = float(os.environ.get("SYMCLF_TUNER_REFINE_FD", "0.08"))
SKIP_ADAM_AT_ZERO = os.environ.get("SYMCLF_TUNER_SKIP_ADAM_AT_ZERO", "1") == "1"
ADAM_EXIT_ZERO = os.environ.get("SYMCLF_TUNER_ADAM_EXIT_ZERO_GRAD", "1") == "1"
ADAM_PATIENCE = int(os.environ.get("SYMCLF_TUNER_ADAM_PATIENCE", "0"))
NESTED_PENALTY = float(os.environ.get("SYMCLF_CP3D_NESTED_PENALTY", "50"))
CMA_ELITE_EVALS = int(os.environ.get("SYMCLF_CMA_ELITE_EVALS", "0"))
CMA_ELITE_COUNT = max(1, int(os.environ.get("SYMCLF_CMA_ELITE_COUNT", "1")))
CMA_ELITE_MAX_FITNESS = float(os.environ.get("SYMCLF_CMA_ELITE_MAX_FITNESS", "0"))
CMA_ELITE_POPSIZE = int(os.environ.get("SYMCLF_CMA_ELITE_POPSIZE", "18"))
CMA_ELITE_SEED = int(os.environ.get("SYMCLF_CMA_ELITE_SEED", "1"))
TUNER_SEED = int(os.environ.get("SYMCLF_GPU2_TUNER_SEED", "0"))
# 2026-09-19 (the user): skip the Adam refine for STRUCTURALLY dead trees (the symbolic structure penalty >= 1e6:
# a missing state, a nested exp/aq), which no constant can revive; the CMA stage never selects dead trees.
SKIP_DEAD_ADAM = os.environ.get("SYMCLF6D_SKIP_DEAD_ADAM", "0") == "1"

_DET4_CACHE = OrderedDict()   # (expression, constants) -> the priced result of the check, across generations
_DET4_CACHE_LIMIT = 65536
_CMA_DONE = set()             # structures already CMA-refined in this run (once each, as in 4-D)
_CMA_STATE = {}               # V10 port (2026-09-20): expression -> {"blob": bytes | None, "evals": int}
# 2026-09-20 (the user): the champion is carried by elitism WITHOUT being re-evaluated, so its structure is not
# among a generation's candidates and its CMA could never continue. The best entry seen is kept here and
# injected as candidate #0 of every CMA stage.
_BEST_ENTRY = [None]
# 2026-09-20 (the user): the seeded template must be CERTIFIED IN GENERATION 1, as it is in the <sys>_LQR
# folder, where CMA starts from all-ones and certifies (3-link at eval 17, ducted fan at 1128). In the GP the
# template's constants come from the S1 tuner, which leaves it non positive definite, and the kappa rule then
# refuses it for CMA -- so it never got the run that certifies it. The seed structures are therefore forced
# into the CMA stage, and their FIRST run starts from all-ones instead of the tuner's constants.
SEED_EXPRESSIONS = []          # set by run.py
_SEED_ENTRIES = {}             # 2026-09-20: the seeds are only evaluated in generation 1; keep them
                               # so they can be injected until they have spent their cap
# V10 port (2026-09-20, the user's order "adopt the GPs to the V10"): the CMA budget is a LIFETIME cap per structure
# spent in SLICES that RESUME the structure's own search (pycma state + numpy RNG state pickled; a resumed run is
# bit-identical to a continuous one, tests 147860/147872), the V9 once-per-structure rule is off, the candidates are
# ranked by how close the check puts them to certification, and the FIRST elite of a generation may spend up to
# SYMCLF_CMA_FIRST_MAX_EVALS over its life (certification lands near eval 2000 and the fitness falls afterwards).
CMA_RESUMABLE = os.environ.get("SYMCLF_CMA_RESUMABLE", "0") == "1"
CMA_SELECT = os.environ.get("SYMCLF_CMA_SELECT", "fitness")
CMA_DEEP_COUNT = int(os.environ.get("SYMCLF_CMA_DEEP_COUNT", "6"))
CMA_DEEP_EVALS = int(os.environ.get("SYMCLF_CMA_DEEP_EVALS", "2500"))
CMA_SLICE_COUNT = int(os.environ.get("SYMCLF_CMA_SLICE_COUNT", "20"))
CMA_SLICE_EVALS = int(os.environ.get("SYMCLF_CMA_SLICE_EVALS", "500"))
CMA_MAX_EVALS = int(os.environ.get("SYMCLF_CMA_MAX_EVALS_PER_STRUCTURE", "2500"))
CMA_FIRST_MAX_EVALS = int(os.environ.get("SYMCLF_CMA_FIRST_MAX_EVALS", "5000"))
CMA_MAX_RESTARTS = int(os.environ.get("SYMCLF_CMA_MAX_RESTARTS", "100"))   # 2026-09-20: restarts do not retire a structure
# 2026-09-20 (the user's order): INJECTION off by default -- an injected entry (champion or seed) belongs to no
# individual, so its CMA result never reaches the population. The seeds are instead forced in as POPULATION
# candidates (with all-ones constants whatever their evaluation returned) and the rank-0 candidate gets its
# whole allowance in one stage, so the certified result lands on an individual, as in V9.
CMA_INJECT_CHAMPION = os.environ.get("SYMCLF_CMA_INJECT_CHAMPION", "0") == "1"
CMA_LOOPS = int(os.environ.get("SYMCLF_CMA_POOL_LOOPS", "0"))   # 0 = actors / popsize


# ----------------------------------------------------------------------------------------------- S1: the tuner
def _grid(expression, constants):
    from core.lattice import lattice_numbers   # noqa: PLC0415

    grid = lattice_numbers(expression, np.asarray(constants, dtype=np.float64))
    value = float(grid["pre_exact"])
    return (value if np.isfinite(value) and value < 1.0e10 else 1.0e10), grid


def _sea(expression, n, rng):
    """The V9 SEA (grid_fitness._device_tuner_kernel) on the CPU, one tree at a time."""
    population = np.concatenate([rng.uniform(-CONST_RANGE, CONST_RANGE, (SEA_POPULATION - 1, n)),
                                 np.ones((1, n))], axis=0)
    scored = [_grid(expression, row) for row in population]
    scores = np.array([s for s, _ in scored])
    grids = [g for _, g in scored]
    for _ in range(SEA_GENERATIONS):
        best = population[int(np.argmin(scores))]
        mutation = np.zeros(n, dtype=bool)
        while not mutation.any():
            mutation = rng.uniform(size=n) < 1.0 / n
        replacement = rng.uniform(-CONST_RANGE, CONST_RANGE, n)
        perturbed = best * np.exp(rng.uniform(-PERTURB_SCALE, PERTURB_SCALE, n))
        proposal = np.where(rng.uniform(size=n) < PERTURB_FRAC, perturbed, replacement)
        offspring = np.where(mutation, proposal, best)
        s, g = _grid(expression, offspring)
        worst = int(np.argmax(scores))
        if s <= scores[worst]:
            population[worst], scores[worst], grids[worst] = offspring, s, g
    k = int(np.argmin(scores))
    return float(scores[k]), population[k].copy(), grids[k]


def _adam(expression, x0, value0, grid0):
    """The V9 central-difference Adam (grid_fitness.refine_constants), never worse than its start."""
    n = x0.size
    x = x0.astype(float).copy()
    m, v = np.zeros(n), np.zeros(n)
    best_value, best_x, best_grid = float(value0), x.copy(), grid0
    stall = 0
    for step in range(1, REFINE_STEPS + 1):
        h = np.maximum(REFINE_FD * np.abs(x), 1.0e-3)
        v0, g0 = _grid(expression, x)
        if not np.isfinite(v0):
            break
        values = []
        for i in range(n):
            for direction in (1.0, -1.0):
                shifted = x.copy()
                shifted[i] += direction * h[i]
                values.append(_grid(expression, shifted)[0])
        improved = v0 < best_value
        if improved:
            best_value, best_x, best_grid = float(v0), x.copy(), g0
        stall = 0 if improved else stall + 1
        gradient = np.array([(values[2 * i] - values[2 * i + 1]) / (2.0 * h[i]) for i in range(n)])
        gradient = np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
        if ADAM_EXIT_ZERO and (v0 <= 0.0 or (not np.any(gradient) and not np.any(m))):
            break
        if ADAM_PATIENCE > 0 and stall >= ADAM_PATIENCE:
            break
        m = 0.9 * m + 0.1 * gradient
        v = 0.999 * v + 0.001 * gradient * gradient
        m_hat = m / (1.0 - 0.9 ** step)
        v_hat = v / (1.0 - 0.999 ** step)
        scale = np.maximum(np.abs(x), 1.0e-2)
        x = np.clip(x - REFINE_LR * scale * m_hat / (np.sqrt(v_hat) + 1.0e-8), -1.0e3, 1.0e3)
    final, grid_final = _grid(expression, x)
    if np.isfinite(final) and final < best_value:
        best_value, best_x, best_grid = float(final), x.copy(), grid_final
    return best_value, best_x, best_grid


def _extra(expression, length, nested_trig):
    from src.SymFunctions import detect_nested_function_calls   # noqa: PLC0415

    nested = float(nested_trig) + detect_nested_function_calls(expression, "exp") \
        + detect_nested_function_calls(expression, "aq")
    reg = float(os.environ.get("SYMCLF_LENGTH_REG", "0.005"))
    return float(NESTED_PENALTY * nested + reg * float(length))


def stage_s1(item):
    """S0 + S1 for one tree: {expression, length, nested_trig} -> the tuned constants and the grid numbers."""
    from core.encode import encode_expression   # noqa: PLC0415
    from core.evalnb import _N                   # noqa: PLC0415

    expression, length = str(item["expression"]), int(item["length"])
    if length >= MAX_TREE_LENGTH:
        return {"expression": expression, "consts": None, "fitness": (1.0e8,), "needs_det4": False, "gate": "length"}
    try:
        n = int(encode_expression(expression, _N).n_constants)
    except Exception:
        return {"expression": expression, "consts": None, "fitness": (1.0e8,), "needs_det4": False, "gate": "encode"}
    extra = _extra(expression, length, item.get("nested_trig", 0))
    started = time.perf_counter()
    if n == 0:
        value, grid = _grid(expression, np.zeros(0))
        consts = np.zeros(0)
    else:
        digest = hashlib.blake2b(expression.encode(), digest_size=8).digest()
        rng = np.random.default_rng([TUNER_SEED, int.from_bytes(digest, "little")])
        value, consts, grid = _sea(expression, n, rng)
        structurally_dead = SKIP_DEAD_ADAM and float(grid["symbolic"]) >= 1.0e6
        if REFINE_STEPS > 0 and not (SKIP_ADAM_AT_ZERO and value <= 0.0) and not structurally_dead:
            value, consts, grid = _adam(expression, consts, value, grid)
    pre_exact = float(grid["pre_exact"])
    dead = bool(grid["symbolic"] >= 1.0e6 or grid["invalid"] >= 1.0e6
                or not np.isfinite(pre_exact) or pre_exact >= 1.0e10)
    entry = {"expression": expression, "consts": consts, "extra_penalty": extra, "grid": grid,
             "pre_exact": pre_exact, "dead": dead, "needs_det4": not dead,
             "tune_seconds": time.perf_counter() - started}
    if dead:
        entry["fitness"] = (float(pre_exact + extra),)
    return entry


def stage_s2(entry):
    """S2 for one live tree: the check + the price (core.rule), at the tuned constants and their grid numbers."""
    from core.rule import fitness_fixed_constants   # noqa: PLC0415

    started = time.perf_counter()
    r = fitness_fixed_constants(entry["expression"], entry["consts"], grid=entry["grid"], extra=entry["extra_penalty"])
    out = dict(entry)
    out.update({"fitness": (float(r["fitness"]),), "det4": r["det4"], "price": r["price"],
                "det4_certified": r["certified"], "dead": r["dead"], "needs_det4": False,
                "det4_seconds": time.perf_counter() - started})
    return out


def fixed_constants(item):
    """The 6-D fitness at GIVEN constants (the CMA objective): lattice -> dead rule -> check -> price."""
    from core.rule import fitness_fixed_constants   # noqa: PLC0415

    r = fitness_fixed_constants(item["expression"], np.asarray(item["consts"], dtype=np.float64),
                                extra=item["extra_penalty"])
    return {"expression": item["expression"], "consts": np.asarray(item["consts"], dtype=np.float64),
            "extra_penalty": item["extra_penalty"], "grid": r["grid"], "pre_exact": r["pre_exact"],
            "fitness": (float(r["fitness"]),), "det4": r["det4"], "price": r["price"],
            "det4_certified": r["certified"], "dead": r["dead"], "needs_det4": False}


# ----------------------------------------------------------------------------------------------- the actors
@ray.remote(num_cpus=1, max_restarts=0)
class CPUFitnessActor:
    """One single-thread worker (NUMBA_NUM_THREADS=1: the most evaluations per core, measured in 4-D)."""

    def __init__(self):
        from core import config   # noqa: F401,PLC0415  (reads the environment once)

    def ready(self):
        import socket   # noqa: PLC0415
        return f"{socket.gethostname()}:{os.environ.get('NUMBA_NUM_THREADS')}"

    def s1(self, items):
        return [stage_s1(item) for item in items]

    def s2(self, entries):
        return [stage_s2(entry) for entry in entries]

    def fixed(self, items):
        return [fixed_constants(item) for item in items]


def create_actor_pool(count):
    options = CPUFitnessActor.options(runtime_env={"env_vars": {"NUMBA_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}})
    actors = [options.remote() for _ in range(int(count))]
    hosts = ray.get([a.ready.remote() for a in actors])
    return actors, hosts


def _actor_map(items, *, actors, batch_size, method_name):
    """Work-stealing actor map (verbatim logic of the 4-D ray_fitness._actor_map)."""
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
            raise RuntimeError(f"6-D actor returned the wrong {method_name} batch length")
        results[start:start + size] = out
        if next_batch < len(batches):
            new_start, batch = batches[next_batch]
            submit(actor, new_start, batch)
            next_batch += 1
    return results


# ----------------------------------------------------------------------------------------------- S2b: CMA
def cma_refine_elite(elite, evaluate, *, evals, popsize, seed):
    """The 4-D V9 cma_refine_elite, unchanged recipe (fmin2 BIPOP, sigma0 1, stds 0.3 max(|c0|, 0.1), keep-best)."""
    import cma   # noqa: PLC0415

    expression = str(elite["expression"])
    c0 = np.asarray(elite["consts"], dtype=np.float64).reshape(-1)
    f0 = float(elite["fitness"][0])
    extra = float(elite["extra_penalty"])
    state = {"n": 0, "best": f0, "best_entry": None, "certified_at": None, "n_certified": 0}
    started = time.perf_counter()

    def objective(xs):
        results = evaluate([{"expression": expression, "consts": np.asarray(x, dtype=np.float64),
                             "extra_penalty": extra} for x in xs])
        fits = []
        for result in results:
            value = float(result["fitness"][0])
            state["n"] += 1
            if result.get("det4_certified"):
                state["n_certified"] += 1
                if state["certified_at"] is None:
                    state["certified_at"] = state["n"]
            if np.isfinite(value) and value < state["best"] - 1.0e-9:
                state["best"], state["best_entry"] = value, result
            fits.append(value if np.isfinite(value) else 1.0e12)
        return fits

    opts = {"CMA_stds": (0.3 * np.maximum(np.abs(c0), 0.1)).tolist(), "popsize": int(popsize), "seed": int(seed),
            "verbose": -9, "verb_disp": 0, "verb_log": 0, "termination_callback": [lambda es: state["n"] >= int(evals)]}
    cma.fmin2(None, c0.tolist(), 1.0, opts, restarts=9, bipop=True, parallel_objective=objective)
    info = {"n_constants": int(c0.size), "before": f0, "after": float(state["best"]),
            "improved": state["best_entry"] is not None, "evals": int(state["n"]), "popsize": int(popsize),
            "seed": int(seed), "certified_at": state["certified_at"], "n_certified": int(state["n_certified"]),
            "seconds": time.perf_counter() - started}
    return state["best_entry"], info


def cma_refine_elite_resumable(elite, evaluate, *, evals, popsize, seed, state=None):
    """V10 port: the same recipe as cma_refine_elite (sigma0 1, stds 0.3 max(|c0|, 0.1), popsize, seed, keep-best)
    as a plain pycma ask/tell loop that can be SLICED: `state` is the blob of an earlier slice -- (the pycma object,
    numpy's global RNG state, the restart count, the running best point) -- so the search continues exactly where it
    stopped. A warm restart from the running best follows any stop of pycma's own criteria (at most
    SYMCLF_CMA_MAX_RESTARTS). Returns (best_entry or None, info, blob)."""
    import pickle   # noqa: PLC0415
    import cma      # noqa: PLC0415

    expression = str(elite["expression"])
    c0 = np.asarray(elite["consts"], dtype=np.float64).reshape(-1)
    f0 = float(elite["fitness"][0])
    extra = float(elite["extra_penalty"])
    started = time.perf_counter()
    best, best_entry, certified_at, n, n_certified = f0, None, None, 0, 0

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
                             "extra_penalty": extra} for x in xs])
        fits = []
        for result in results:
            value = float(result["fitness"][0])
            n += 1
            if result.get("det4_certified"):
                n_certified += 1
                if certified_at is None:
                    certified_at = n
            if np.isfinite(value) and value < best - 1.0e-9:
                best, best_entry = value, result
                best_x = np.asarray(result["consts"], dtype=np.float64).reshape(-1)
            fits.append(value if np.isfinite(value) else 1.0e12)
        es.tell(xs, fits)
    blob = pickle.dumps((es, np.random.get_state(), restarts, best_x))
    info = {"n_constants": int(c0.size), "before": f0, "after": float(best),
            "improved": best_entry is not None, "evals": int(n), "popsize": int(popsize), "seed": int(seed),
            "certified_at": certified_at, "n_certified": int(n_certified), "restarts": int(restarts),
            "exhausted": bool(exhausted), "resumed": state is not None,
            "seconds": time.perf_counter() - started}
    return best_entry, info, blob


@ray.remote(num_cpus=0, max_retries=0)
def _cma_loop_task_resumable(elite, actors, slot, evals, popsize, seed, state):
    rotated = actors[(slot * popsize) % len(actors):] + actors[:(slot * popsize) % len(actors)]

    def evaluate(items):
        return _actor_map(items, actors=rotated, batch_size=1, method_name="fixed")

    return cma_refine_elite_resumable(elite, evaluate, evals=evals, popsize=popsize, seed=seed, state=state)


def _select_cma_elites_kappa(entries, count, max_fitness, exhausted):
    """V10 port: positive definite (c_v* > pd_eps and not the non-PD gate) and the highest DET4 kappa* first --
    how close the check puts the structure to certification -- then fitness; distinct structures."""
    pd_eps = float(os.environ.get("SYMCLF_DET4_PD_EPS", "1e-4"))
    ranked = []
    for i, e in enumerate(entries):
        d4 = e.get("det4")
        if (d4 is None or e.get("consts") is None or np.asarray(e["consts"]).size == 0 or e.get("dead")
                or not e.get("fitness") or not np.isfinite(float(e["fitness"][0]))
                or str(e["expression"]) in exhausted
                or (max_fitness > 0.0 and float(e["fitness"][0]) >= max_fitness)):
            continue
        kappa, cv = d4.get("kappa"), d4.get("cv")
        if kappa is None or cv is None or not np.isfinite(kappa) or not np.isfinite(cv):
            continue
        if kappa <= -1.0e5 or cv <= pd_eps:
            continue
        ranked.append((-float(kappa), float(e["fitness"][0]), i))
    ranked.sort()
    chosen, seen = [], set()
    for _, _, i in ranked:
        expression = str(entries[i]["expression"])
        if expression in seen:
            continue
        seen.add(expression)
        chosen.append(i)
        if len(chosen) >= int(count):
            break
    return chosen


def _champion_key(entry):
    """2026-09-20: certified first, then positive definite, then fitness -- a degenerate structure has c_v* = 0 and
    a price near 0, so by fitness alone it outranks a certified champion and steals the budget."""
    d4 = entry.get("det4") or {}
    cv = d4.get("cv")
    pd_eps = float(os.environ.get("SYMCLF_DET4_PD_EPS", "1e-4"))
    tier = 0 if entry.get("det4_certified") else (1 if (cv is not None and np.isfinite(cv) and cv > pd_eps) else 2)
    return (tier, float(entry["fitness"][0]))


def _select_cma_tiers(entries, cma_state, max_fitness):
    """V10 port: the tiers of one generation. CMA_DEEP_COUNT structures get CMA_DEEP_EVALS, the next
    CMA_SLICE_COUNT get CMA_SLICE_EVALS, each truncated to what is left of the structure's cap -- the first elite
    of the generation may go to CMA_FIRST_MAX_EVALS, every other structure stops at CMA_MAX_EVALS."""
    top_cap = max(CMA_MAX_EVALS, CMA_FIRST_MAX_EVALS)
    exhausted = {key for key, value in cma_state.items() if int(value["evals"]) >= top_cap}
    want = max(1, CMA_DEEP_COUNT + CMA_SLICE_COUNT)
    # 2026-09-20 (the user): a structure that already holds CMA state with budget left goes FIRST -- it is the one
    # that "had 2500 and gets another 2500". Ranking by kappa* alone let a fresh structure take the top slot and the
    # champion never resumed (4-D job 147877, GEN 2: 0 resumed).
    seeds_first, seen = [], set()
    for index, entry in enumerate(entries):
        expression = str(entry.get("expression", ""))
        if expression in SEED_EXPRESSIONS and expression not in seen and expression not in exhausted:
            # 2026-09-20: whatever S1 returned for the seed (jobs 147944/147946: it came back without constants), it
            # is a CMA candidate of the POPULATION -- its constants are all-ones for the first run (_seed_start), so
            # only the count is needed here.
            if entry.get("consts") is None or np.asarray(entry["consts"]).size == 0:
                from core.encode import encode_expression   # noqa: PLC0415
                from core.evalnb import _N                   # noqa: PLC0415
                try:
                    n = int(encode_expression(expression, _N).n_constants)
                except Exception:
                    continue
                if n == 0:
                    continue
                fixed = dict(entry)
                fixed["consts"] = np.ones(n)
                fixed.setdefault("extra_penalty", _extra(expression, int(entry.get("length", len(expression) // 8)),
                                                         float(entry.get("nested_trig", 0))))
                fixed["fitness"] = (1.0e12,)
                fixed["dead"] = False
                entries[index] = fixed
            seen.add(expression)
            seeds_first.append(index)
    resumable = []
    for index, entry in enumerate(entries):
        expression = str(entry.get("expression", ""))
        saved = cma_state.get(expression)
        if (saved is None or saved.get("blob") is None or int(saved["evals"]) >= top_cap or expression in seen
                or entry.get("consts") is None or np.asarray(entry["consts"]).size == 0 or entry.get("dead")
                or not entry.get("fitness") or not np.isfinite(float(entry["fitness"][0]))):
            continue
        seen.add(expression)
        resumable.append(index)
    resumable.sort(key=lambda i: float(entries[i]["fitness"][0]))
    resumable = seeds_first + resumable
    if CMA_SELECT == "kappa":
        picked = _select_cma_elites_kappa(entries, want, max_fitness, exhausted | seen)
        if len(resumable) + len(picked) < want:
            # 2026-09-20: too few positive-definite candidates starve the stage (2-link GP: 1 structure per
            # generation for 100 generations) -- fill the rest by fitness, as V9 did.
            already = {str(entries[i]["expression"]) for i in resumable + picked}
            for index in _select_cma_elites(entries, want, max_fitness, exhausted | seen | already):
                if str(entries[index]["expression"]) not in already:
                    already.add(str(entries[index]["expression"]))
                    picked.append(index)
                if len(resumable) + len(picked) >= want:
                    break
        ranked = resumable + picked
    else:
        ranked = resumable + _select_cma_elites(entries, want, max_fitness, exhausted | seen)
    selected, budgets, states = [], [], []
    for rank, index in enumerate(ranked):
        expression = str(entries[index]["expression"])
        used = int(cma_state.get(expression, {}).get("evals", 0))
        if rank == 0:   # the best candidate of the generation: its whole allowance in this stage (V9 behaviour)
            budget = max(0, CMA_FIRST_MAX_EVALS - used)
        else:
            base = CMA_DEEP_EVALS if rank < CMA_DEEP_COUNT else CMA_SLICE_EVALS
            budget = max(0, min(int(base), CMA_MAX_EVALS - used))
        if budget <= 0:
            continue
        selected.append(index)
        budgets.append(budget)
        states.append(cma_state.get(expression, {}).get("blob"))
    return selected, budgets, states


@ray.remote(num_cpus=0, max_retries=0)
def _cma_loop_task(elite, actors, slot, evals, popsize, seed):
    rotated = actors[(slot * popsize) % len(actors):] + actors[:(slot * popsize) % len(actors)]

    def evaluate(items):
        return _actor_map(items, actors=rotated, batch_size=1, method_name="fixed")

    return cma_refine_elite(elite, evaluate, evals=evals, popsize=popsize, seed=seed)


def _select_cma_elites(entries, count, max_fitness, exhausted=None):
    """V9's rule: checked, with constants, not dead, finite, under the gate, a structure not yet CMA-refined,
    best first, distinct structures. `exhausted` (V10 port) replaces _CMA_DONE in the resumable mode."""
    skip = _CMA_DONE if exhausted is None else exhausted
    ranked = sorted(
        (float(e["fitness"][0]), i) for i, e in enumerate(entries)
        if e.get("det4") is not None and e.get("consts") is not None and np.asarray(e["consts"]).size > 0
        and not e.get("dead") and e.get("fitness") and np.isfinite(float(e["fitness"][0]))
        and e["expression"] not in skip and (max_fitness <= 0.0 or float(e["fitness"][0]) < max_fitness)
    )
    chosen, seen = [], set()
    for _, i in ranked:
        if entries[i]["expression"] in seen:
            continue
        seen.add(entries[i]["expression"])
        chosen.append(i)
        if len(chosen) >= int(count):
            break
    return chosen


# ----------------------------------------------------------------------------------------------- the mapper
def _key(entry):
    return (entry["expression"], tuple(np.round(np.asarray(entry["consts"], dtype=np.float64), 15).tolist()))


def _breakdown(e):
    g = e.get("grid") or {}
    d4 = e.get("det4") or {}
    return {
        "pre_exact": e.get("pre_exact"), "extra_penalty": e.get("extra_penalty"), "dead": bool(e.get("dead")),
        "grid_roa": g.get("roa"), "grid_cv": g.get("cv_grid"), "grid_kappa": g.get("kappa_grid"),
        "grid_boundary_min": g.get("boundary_min"), "grid_w_scale": g.get("w_scale"),
        "grid_certified_volume": g.get("certified_volume"),
        "det4": {k: d4.get(k) for k in ("cv", "kappa", "worst_r", "penalty_terms", "pd_valid")} if d4 else None,
        "det4_price": e.get("price"), "det4_certified": e.get("det4_certified"), "cma": e.get("cma"),
        "gate": e.get("gate"),
    }


def persistent_actor_mapper(_function, individuals, *, actors, batch_size=4):
    from src.SymFunctions import get_features_batch   # noqa: PLC0415

    individuals = list(individuals)
    if not individuals:
        return []
    lengths, nested, _ = get_features_batch(individuals)
    for i, ind in enumerate(individuals):   # 2026-09-20 diagnostic: what the driver sends for the seeds
        if str(ind) in SEED_EXPRESSIONS:
            print(f"6D seed item: nodes={len(ind)} features_length={int(lengths[i])} nested={float(nested[i])} "
                  f"n_features={len(lengths)} n_individuals={len(individuals)} expr={str(ind)[:40]}", flush=True)
    items = [{"expression": str(ind), "length": int(lengths[i]), "nested_trig": float(nested[i])}
             for i, ind in enumerate(individuals)]

    t0 = time.perf_counter()
    entries = _actor_map(items, actors=actors, batch_size=int(batch_size), method_name="s1")
    s1_seconds = time.perf_counter() - t0

    # S2: once per distinct (expression, constants), driver cache across generations
    t0 = time.perf_counter()
    groups = OrderedDict()
    for i, e in enumerate(entries):
        if e.get("needs_det4"):
            groups.setdefault(_key(e), []).append(i)
    todo, hits = [], 0
    for key, positions in groups.items():
        cached = _DET4_CACHE.get(key)
        if cached is None:
            todo.append(key)
            continue
        _DET4_CACHE.move_to_end(key)
        hits += len(positions)
        for i in positions:
            entries[i] = dict(entries[i], **cached)
    if todo:
        checked = _actor_map([entries[groups[k][0]] for k in todo], actors=actors, batch_size=1, method_name="s2")
        for key, e in zip(todo, checked):
            cached = {k: e[k] for k in ("fitness", "det4", "price", "det4_certified", "dead", "needs_det4")}
            _DET4_CACHE[key] = cached
            while len(_DET4_CACHE) > _DET4_CACHE_LIMIT:
                _DET4_CACHE.popitem(last=False)
            for i in groups[key]:
                entries[i] = dict(entries[i], **cached)
    s2_seconds = time.perf_counter() - t0

    # S2b: CMA on the best new structures under the gate, one Ray task per structure
    t0 = time.perf_counter()
    cma_infos = []
    for e in entries:   # 2026-09-20 diagnostic: what the seeds look like when they reach the CMA stage
        if str(e.get("expression", "")) in SEED_EXPRESSIONS:
            d4 = e.get("det4") or {}
            print(f"6D seed entry: len={len(str(e.get('expression', '')))} gate={e.get('gate')} dead={e.get('dead')} "
                  f"consts={'None' if e.get('consts') is None else np.asarray(e['consts']).size} fitness={e.get('fitness')} "
                  f"needs_det4={e.get('needs_det4')} det4_status={d4.get('status') if isinstance(d4, dict) else None} "
                  f"pre_exact={e.get('pre_exact')} keys={sorted(e.keys())[:12]}", flush=True)
    if CMA_ELITE_EVALS > 0 and CMA_RESUMABLE:
        # V10 port: tiers + lifetime caps + resumed states, at most `loops` structures in flight
        selected, budgets, states = _select_cma_tiers(entries, _CMA_STATE, CMA_ELITE_MAX_FITNESS)
        champion = _BEST_ENTRY[0]
        champion_expr = None if champion is None else str(champion["expression"])
        top_cap_now = max(CMA_MAX_EVALS, CMA_FIRST_MAX_EVALS)
        champion_used = 0 if champion is None else int(_CMA_STATE.get(champion_expr, {}).get("evals", 0))
        if champion is not None and champion_used >= top_cap_now:
            # 2026-09-20: a retired champion must release the slot, otherwise its fitness blocks every later champion
            _BEST_ENTRY[0] = champion = None
            champion_expr, champion_used = None, 0
        # 2026-09-20 (the user): the injected champion must pass the same gate as every other candidate (the SEEDS
        # stay exempt on purpose -- the template must get its CMA even when its tuner fitness is above the gate).
        use_champion = (CMA_INJECT_CHAMPION and champion is not None and champion_used < top_cap_now
                        and (CMA_ELITE_MAX_FITNESS <= 0.0 or float(champion["fitness"][0]) < CMA_ELITE_MAX_FITNESS)
                        and champion_expr not in {str(entries[i]["expression"]) for i in selected})
        def _seed_start(entry):
            """A seed structure's FIRST CMA starts from all-ones, like the <sys>_LQR run that certifies it."""
            expression = str(entry.get("expression", ""))
            if expression in SEED_EXPRESSIONS and _CMA_STATE.get(expression) is None:
                out = dict(entry)
                out["consts"] = np.ones(np.asarray(entry["consts"], dtype=float).reshape(-1).size)
                return out
            return entry

        injected = []
        if use_champion:
            injected.append((dict(champion), champion_expr, champion_used))
        present = {str(entries[i]["expression"]) for i in selected} | {e for _, e, _ in injected}
        for expression, entry in (_SEED_ENTRIES.items() if CMA_INJECT_CHAMPION else ()):   # injection off by default
            used = int(_CMA_STATE.get(expression, {}).get("evals", 0))
            if expression not in present and used < top_cap_now:
                injected.append((_seed_start(dict(entry)), expression, used))
                present.add(expression)
        elites_in = [e for e, _, _ in injected] + [_seed_start(entries[i]) for i in selected]
        budgets = [max(0, min(CMA_DEEP_EVALS, top_cap_now - used)) for _, _, used in injected] + budgets
        states = [_CMA_STATE.get(expression, {}).get("blob") for _, expression, _ in injected] + states
        loops = CMA_LOOPS or max(1, len(actors) // max(1, CMA_ELITE_POPSIZE))
        pending, next_slot = {}, 0
        n_deep = sum(1 for b in budgets if int(b) >= CMA_DEEP_EVALS)
        while next_slot < min(loops, len(elites_in)):
            pending[_cma_loop_task_resumable.remote(
                elites_in[next_slot], actors, next_slot, int(budgets[next_slot]),
                CMA_ELITE_POPSIZE, CMA_ELITE_SEED, states[next_slot])] = next_slot
            next_slot += 1
        while pending:
            ready, _ = ray.wait(list(pending), num_returns=1)
            slot = pending.pop(ready[0])
            best_entry, info, blob = ray.get(ready[0])
            if slot < len(injected):
                _, inj_expr, inj_used = injected[slot]
                used = inj_used + int(info["evals"])
                done = used >= top_cap_now or int(info["evals"]) == 0
                info["evals_total"] = top_cap_now if done else used
                _CMA_STATE[inj_expr] = {"blob": None if done else blob,
                                        "evals": top_cap_now if done else used,
                                        "best": float(info["after"])}
                if best_entry is not None:
                    _BEST_ENTRY[0] = {k: best_entry.get(k) for k in
                                      ("expression", "consts", "fitness", "extra_penalty", "det4", "det4_certified")}
                cma_infos.append(info)
                print(f"6D injected CMA [{inj_expr[:40]}...]: {info['before']:.6g} -> {info['after']:.6g} in {int(info['evals'])} evals "
                      f"(total {int(info['evals_total'])}/{top_cap_now}), certified_at {info['certified_at']}, "
                      f"restarts {int(info['restarts'])}", flush=True)
                if best_entry is not None:   # the champion is not an individual, so persist it here
                    import json as _json
                    with open(f"{os.environ.get('SLURM_JOB_ID', 'local')}_champion_cma.jsonl", "a") as _fh:
                        _fh.write(_json.dumps({
                            "expression": str(best_entry.get("expression")),
                            "constants": [float(v) for v in np.asarray(best_entry.get("consts"), dtype=float).reshape(-1)],
                            "fitness": float(best_entry["fitness"][0]),
                            "certified": bool(best_entry.get("det4_certified")),
                            "det4": best_entry.get("det4"),
                            "cma": {k: info.get(k) for k in ("before", "after", "evals", "evals_total",
                                                             "certified_at", "restarts", "popsize", "seed")},
                        }, default=float) + "\n")
                if next_slot < len(elites_in):
                    pending[_cma_loop_task_resumable.remote(
                        elites_in[next_slot], actors, next_slot, int(budgets[next_slot]),
                        CMA_ELITE_POPSIZE, CMA_ELITE_SEED, states[next_slot])] = next_slot
                    next_slot += 1
                continue
            i = selected[slot - len(injected)]
            expression = str(entries[i]["expression"])
            top_cap = max(CMA_MAX_EVALS, CMA_FIRST_MAX_EVALS)
            used = int(_CMA_STATE.get(expression, {}).get("evals", 0)) + int(info["evals"])
            done = used >= top_cap or int(info["evals"]) == 0   # a slice with no evaluation is finished
            info["evals_total"] = top_cap if done else used
            _CMA_STATE[expression] = {"blob": None if done else blob, "evals": top_cap if done else used,
                                      "best": float(info["after"])}
            e = dict(entries[i]) if best_entry is None else dict(best_entry)
            e["cma"] = info
            entries[i] = e
            cma_infos.append(info)
            if next_slot < len(elites_in):
                pending[_cma_loop_task_resumable.remote(
                    elites_in[next_slot], actors, next_slot, int(budgets[next_slot]),
                    CMA_ELITE_POPSIZE, CMA_ELITE_SEED, states[next_slot])] = next_slot
                next_slot += 1
        if selected:
            print(f"6D cma (resumable): {len(selected)} structures ({n_deep} deep, {len(selected) - n_deep} slices), "
                  f"{loops} loops in flight, {sum(i['evals'] for i in cma_infos)} evals, "
                  f"{sum(1 for i in cma_infos if i.get('certified_at') is not None)} certified, "
                  f"{sum(1 for i in cma_infos if i.get('resumed'))} resumed, "
                  f"restarts {[int(i.get('restarts', 0)) for i in cma_infos][:6]}, states held {len(_CMA_STATE)}", flush=True)
    elif CMA_ELITE_EVALS > 0:
        chosen = _select_cma_elites(entries, CMA_ELITE_COUNT, CMA_ELITE_MAX_FITNESS)
        refs = {_cma_loop_task.remote(entries[i], actors, slot, CMA_ELITE_EVALS, CMA_ELITE_POPSIZE, CMA_ELITE_SEED): i
                for slot, i in enumerate(chosen)}
        for ref, i in refs.items():
            best_entry, info = ray.get(ref)
            _CMA_DONE.add(entries[i]["expression"])
            e = dict(entries[i]) if best_entry is None else dict(best_entry)
            e["cma"] = info
            entries[i] = e
            cma_infos.append(info)
    cma_seconds = time.perf_counter() - t0

    for e in entries:   # 2026-09-20: remember the seeds so they can be injected later
        if str(e.get("expression", "")) in SEED_EXPRESSIONS and e.get("consts") is not None \
                and np.asarray(e["consts"]).size > 0 and str(e["expression"]) not in _SEED_ENTRIES:
            _SEED_ENTRIES[str(e["expression"])] = {k: e.get(k) for k in
                                                   ("expression", "consts", "fitness", "extra_penalty", "det4",
                                                    "det4_certified")}

    for e in entries:   # 2026-09-20: remember the champion for the next CMA stage
        if e.get("fitness") and np.isfinite(float(e["fitness"][0])) and e.get("consts") is not None \
                and np.asarray(e["consts"]).size > 0 and not e.get("dead"):
            if _BEST_ENTRY[0] is None or _champion_key(e) < _champion_key(_BEST_ENTRY[0]):
                _BEST_ENTRY[0] = {k: e.get(k) for k in
                                  ("expression", "consts", "fitness", "extra_penalty", "det4", "det4_certified")}

    certified = sum(1 for e in entries if e.get("det4_certified"))
    fits = [float(e["fitness"][0]) for e in entries if e.get("fitness")]
    tune = [e["tune_seconds"] for e in entries if e.get("tune_seconds") is not None]
    print("6D stages: "
          f"candidates={len(individuals)} s1_s={s1_seconds:.1f} tune_mean_s={np.mean(tune) if tune else 0:.2f} "
          f"det4_candidates={sum(len(p) for p in groups.values())} det4_unique={len(groups)} det4_checked={len(todo)} "
          f"det4_cached={hits} s2_s={s2_seconds:.1f} dead={sum(1 for e in entries if e.get('dead'))} "
          f"certified={certified} fitness_min={min(fits, default=float('nan')):.6g} "
          f"cma_runs={len(cma_infos)} cma_certified={sum(1 for c in cma_infos if c['certified_at'] is not None)} "
          f"cma_best={min((c['after'] for c in cma_infos), default=float('nan')):.6g} cma_s={cma_seconds:.1f} "
          f"cache_entries={len(_DET4_CACHE)} "
          f"seed_exprs={len(SEED_EXPRESSIONS)} seeds_in_entries={sum(1 for e in entries if str(e.get('expression', '')) in SEED_EXPRESSIONS)} "
          f"states={len(_CMA_STATE)}", flush=True)
    return [{"consts": e.get("consts"), "fitness": e["fitness"], "breakdown": _breakdown(e)} for e in entries]
