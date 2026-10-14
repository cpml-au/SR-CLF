"""The runner, the same for every example: one flex GP run on the Ray workers of src/CPU.

    python -m src.CPU.run examples/<name>        (from the repository root; tools/cluster.sh does this)

The example folder is the argument and everything else is found from it:

  config.yaml    every numerical setting (src/settings.py, applied here BEFORE the pipeline is imported)
                 and flex's gp block (flex.gp.util.load_config_data)
  systems/       the folder goes on sys.path and PYTHONPATH, so systems/<system.name>.py is the system;
                 n, the number of state variables of the trees and of the seeds, is its N_STATES,
                 never a number written here
  runs/<job>/    this run's records (the working directory of the run):
                 <job>_best_per_generation.jsonl, <job>_numerically_verified.csv,
                 <job>_best_expression.txt, <job>_final_audit.json, and config.yaml,
                 effective_env.txt, code_version.txt; SYMCLF_RUNS_ROOT replaces <example>/runs
  reference/     the records of the reference jobs (the run does not use them)

The run: the flex regressor of the gp block, trees over x1..xn with the gp block's primitives and one
constant terminal `a`, two seed trees (a sum of squares with one cross term and no constants, and the
full quadratic template with n(n+1)/2 `a` placeholders -- never the coefficients of a Riccati
solution), the tree-handling function of src/CPU/gp_engine.py (gate, grid stage, check, CMA), a GEN
line and a jsonl record per generation, and at the end, as a report only, the audit of the best tree
(python -m src.CPU.audit, an independent re-computation).

    fitness = grid_score + check_cost + complexity_penalty

Settings read here: backend.driver_cpus (SYMCLF_CPU_DRIVER_CPUS), backend.check_batch.cpu
  (SYMCLF_CPU_S1_BATCH), verification.final (SYMCLF_CPU_FINAL_AUDIT), gp_runtime.seeds, gp_runtime.seed_trees.
Environment only: SYMCLF_CONFIG (another settings file), SYMCLF_OPT_CONFIG (another gp block),
  SYMCLF_OPT_* (one GP setting for one job; SYMCLF_OPT_GENERATIONS=1 = one generation), SYMCLF_RUNS_ROOT,
  SLURM_JOB_ID, RAY_ADDRESS.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))   # src/CPU -> src -> repository root


def paths(example_dir):
    """The folders and files of one example (see the module docstring); SYMCLF_RUNS_ROOT replaces runs/."""
    example_dir = os.path.abspath(example_dir)
    return {
        "example": example_dir,
        "config": os.environ.get("SYMCLF_CONFIG") or os.path.join(example_dir, "config.yaml"),
        "systems": os.path.join(example_dir, "systems"),
        "reference": os.path.join(example_dir, "reference"),
        "runs": os.environ.get("SYMCLF_RUNS_ROOT") or os.path.join(example_dir, "runs"),
    }


def _apply_env_overrides(regressor_params, config_file_data):
    """The GP settings a job can replace (SYMCLF_OPT_*): population, islands, generations, operators, reg_param,
    migration."""
    def _f(name, cast, current):
        raw = os.environ.get(name)
        return current if raw is None or raw == "" else cast(raw)

    gp_cfg = config_file_data["gp"]
    islands = _f("SYMCLF_OPT_NUM_ISLANDS", int, int(regressor_params["num_islands"]))
    total = _f("SYMCLF_OPT_POPULATION", int, int(regressor_params["num_individuals"]) * islands)
    regressor_params["num_islands"] = islands
    os.environ["SYMCLF_N_ISLANDS"] = str(int(islands))   # the CSV generation / island (src/cma_all.py)
    os.environ["SYMCLF_GEN_POPULATION"] = str(int(total))   # option (b): a call spends its share of the population
    regressor_params["num_individuals"] = max(1, total // max(1, islands))
    regressor_params["generations"] = _f("SYMCLF_OPT_GENERATIONS", int, int(regressor_params["generations"]))
    regressor_params["crossover_prob"] = _f("SYMCLF_OPT_CROSSOVER_PROB", float, float(regressor_params["crossover_prob"]))
    regressor_params["mut_prob"] = _f("SYMCLF_OPT_MUT_PROB", float, float(regressor_params["mut_prob"]))
    penalty = dict(gp_cfg["penalty"])
    penalty["reg_param"] = _f("SYMCLF_OPT_REG_PARAM", float, float(penalty["reg_param"]))
    gp_cfg["penalty"] = penalty
    if regressor_params.get("mig_freq") is not None:
        regressor_params["mig_freq"] = _f("SYMCLF_OPT_MIGRATION_FREQ", int, int(regressor_params["mig_freq"]))
    if regressor_params.get("mig_frac") is not None:
        regressor_params["mig_frac"] = _f("SYMCLF_OPT_MIGRATION_FRAC", float, float(regressor_params["mig_frac"]))
    print(f"GP hyperparameters: population={regressor_params['num_individuals'] * islands} "
          f"(num_individuals={regressor_params['num_individuals']} x num_islands={islands}), "
          f"generations={regressor_params['generations']}, crossover={regressor_params['crossover_prob']}, "
          f"mutation={regressor_params['mut_prob']}, reg_param={penalty['reg_param']}, "
          f"mig_freq={regressor_params.get('mig_freq')}, mig_frac={regressor_params.get('mig_frac')}", flush=True)
    return regressor_params, config_file_data


def assign_attributes(individuals, attributes):
    for individual, values in zip(individuals, attributes):
        individual.consts = values["consts"]
        individual.fitness.values = values["fitness"]
        individual.breakdown = values.get("breakdown")


def _json_default(value):
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return str(value)


def _g(value, nd=4):
    import numpy as np

    try:
        value = float(value)
    except (TypeError, ValueError):
        return "na"
    return "na" if not np.isfinite(value) else f"{value:.{nd}g}"


def format_breakdown(bd):
    if not bd:
        return "breakdown: none"
    if bd.get("gate"):
        return f"gate: {bd['gate']}"
    d4 = bd.get("check") or {}
    if bd.get("dead"):
        stage = "dead (grid score stands, no check)"
    elif d4:
        pen = d4.get("penalty_terms")
        stage = (f"check check_cost={_g(bd.get('check_cost'))} c_v*={_g(d4.get('cv'))} kappa*={_g(d4.get('kappa'))}"
                 f" numerically_verified={bd.get('numerically_verified')} worst_r={_g(d4.get('worst_r'), 3)}"
                 + ("" if not pen else " pen=" + "/".join(_g(t, 3) for t in pen)))
    else:
        stage = "no check numbers"
    cm = bd.get("cma") or {}
    if cm.get("after") is not None:
        stage += (f" | cma {_g(cm.get('before'))}->{_g(cm.get('after'))} ({cm.get('evals')} evals,"
                  f" numerically_verified_at {cm.get('numerically_verified_at')}, {_g(cm.get('seconds'), 3)} s)")
    return (f"grid={_g(bd.get('grid_score'))} [roa={_g(bd.get('grid_roa'), 3)}] lattice c_v={_g(bd.get('grid_cv'))}"
            f" kappa={_g(bd.get('grid_kappa'))} | complexity={_g(bd.get('complexity_penalty'))} | {stage}"
            f" | boundary_min={_g(bd.get('grid_boundary_min'))} volume={_g(bd.get('grid_sublevel_fraction'), 3)}")


def make_generation_logger(filename):
    import numpy as np

    generation = 0

    def log_best(best_individuals):
        nonlocal generation
        generation += 1
        best = best_individuals[0]
        consts = getattr(best, "consts", None)
        record = {"generation": generation, "expression": str(best),
                  "constants": None if consts is None else np.asarray(consts, dtype=float).tolist(),
                  "fitness": float(best.fitness.values[0]), "breakdown": getattr(best, "breakdown", None)}
        print(f"GEN {generation} best fitness={record['fitness']:.6g} len={len(best)} :: "
              + format_breakdown(record["breakdown"]), flush=True)
        with open(filename, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=_json_default) + "\n")

    return log_best


def fitness_marker(*_args, **_kwargs):
    raise RuntimeError("the fitness runs only through the actor mapper")


def enable_actor_fitness(regressor_class, *, actors, batch_size):
    from src.CPU.gp_engine import persistent_actor_mapper

    def register_map(self, toolbox):
        if not self.multiprocessing:
            raise RuntimeError("the actor mapper requires multiprocessing=True")
        toolbox.register("map", persistent_actor_mapper, actors=actors, batch_size=int(batch_size))

    regressor_class._GPSymbolicRegressor__register_map = register_map


def seed_expressions(n):
    """The two seed trees of a run over x1..xn: a sum of squares with one cross term and no constants,
    and the full quadratic template with `a` placeholders (n(n+1)/2 monomials) -- never the coefficients
    of a Riccati solution."""
    from src.CPU.templates import template_expression

    squares = [f"mul(x{i}, x{i})" for i in range(1, n + 1)]
    seed_expr = squares[0]
    for term in squares[1:] + ["mul(x1, x2)"]:
        seed_expr = f"add({seed_expr}, {term})"
    return seed_expr, template_expression(n, "full")


def _record_run_folder(run_dir, p):
    """config.yaml, the effective environment and the code version next to the records."""
    import shutil

    shutil.copyfile(p["config"], os.path.join(run_dir, "config.yaml"))
    with open(os.path.join(run_dir, "effective_env.txt"), "w") as fh:
        for name in sorted(os.environ):
            if name.startswith(("SYMCLF", "FLEX_", "NUMBA_", "OMP_", "MKL_", "OPENBLAS_")):
                fh.write(f"{name}={os.environ[name]}\n")
    try:
        head = subprocess.run(["git", "-C", PROJECT_ROOT, "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip() or "unknown"
        status = subprocess.run(["git", "-C", PROJECT_ROOT, "status", "--short", "--untracked-files=no"],
                                capture_output=True, text=True).stdout
    except OSError:
        head, status = "unknown", ""
    with open(os.path.join(run_dir, "code_version.txt"), "w") as fh:
        fh.write(f"commit {os.environ.get('SYMCLF_CODE_COMMIT', head)}\ncode folder {PROJECT_ROOT}\n")
        fh.write("".join(f"modified: {line}\n" for line in status.splitlines()))


def main(example_dir):
    p = paths(example_dir)
    if not os.path.isfile(p["config"]):
        raise SystemExit(f"no config.yaml in {p['example']}")
    for _p in (PROJECT_ROOT, p["example"]):   # the example first: `systems` is its systems/ package
        if _p in sys.path:
            sys.path.remove(_p)
        sys.path.insert(0, _p)
    os.environ["PYTHONPATH"] = os.pathsep.join([p["example"], PROJECT_ROOT, os.environ.get("PYTHONPATH", "")])

    # Every numerical setting of this run comes from config.yaml (docs/SETTINGS_CPU.md): src/settings.py
    # turns its keys into the environment variables the code reads when it is imported, so this has to
    # happen BEFORE the imports below.  A variable that is already set wins (the sbatch line).
    from src import settings

    settings.apply(p["config"], backend="cpu")
    # `python -m src.CPU.run` imported the src.CPU package before the settings were applied: set its
    # numba cache folder again for the system of this run, before anything is compiled
    from src.CPU import apply_numba_cache_dir

    apply_numba_cache_dir()

    import numpy as np
    import ray
    from deap import gp
    from flex.gp import util
    from flex.gp.primitives import add_primitives_to_pset_from_dict
    from flex.gp.regressor import GPSymbolicRegressor

    import src.Functions
    from src.PredictScoreFuncs import predict, score
    from src.CPU import config as C6
    from src.CPU import gp_engine
    from src.CPU.gp_engine import create_actor_pool

    n = int(C6.system().N_STATES)
    job_id = os.environ.get("SLURM_JOB_ID")
    run_dir = os.path.join(p["runs"], job_id or "local")
    os.makedirs(run_dir, exist_ok=True)
    os.chdir(run_dir)   # every record of the run lands here (gp_engine writes the verified pairs into the cwd)
    _record_run_folder(run_dir, p)
    tag = job_id or "local"
    best_history_file = os.path.join(run_dir, f"{tag}_best_per_generation.jsonl")
    print(f"example {p['example']} | run folder {run_dir}", flush=True)

    if not ray.is_initialized():
        ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))
    resources = ray.cluster_resources()
    print(f"Ray resources: {resources}", flush=True)
    print(f"system {C6.system().NAME}: box {C6.box()} rho {C6.RHO} b-norm {C6.B_NORM} grid {C6.grid_points()}^{n} | "
          f"hinges cv {C6.PEN_CV}@{C6.PEN_CV_TARGET} kappa {C6.PEN_KAPPA}@{C6.PEN_KAPPA_TARGET} "
          f"vol {C6.PEN_VOL}@{C6.PEN_VOL_TARGET}", flush=True)

    regressor_params, config_file_data = util.load_config_data(os.environ.get("SYMCLF_OPT_CONFIG", p["config"]))
    regressor_params, config_file_data = _apply_env_overrides(regressor_params, config_file_data)
    with open(best_history_file, "w", encoding="utf-8"):
        pass

    pset = gp.PrimitiveSetTyped("MAIN", [float] * n, float)
    pset.renameArguments(**{f"ARG{i}": f"x{i + 1}" for i in range(n)})
    pset = add_primitives_to_pset_from_dict(pset, config_file_data["gp"]["primitives"])
    pset.addTerminal(object, float, "a")
    penalty = config_file_data["gp"]["penalty"]

    # the training data object Flex needs (the fitness itself builds its own training grid, grid_points()^n,
    # in src/CPU/lattice.py)
    axes = [np.linspace(-h, h, C6.grid_points()) for h in C6.box()]
    train_data = src.Functions.Dataset("true_data", axes, None)

    seed_expr, seed_expr2 = seed_expressions(n)
    # gp_runtime.seed_trees replaces the two default seed trees; unset -> the defaults
    custom = [t.strip() for t in os.environ.get("SYMCLF_CPU_SEED_TREES", "").split(";") if t.strip()]
    seed_list = custom or [seed_expr, seed_expr2]
    # the tree-handling function needs to know the seed trees (option (c) tunes them first)
    gp_engine.SEED_EXPRESSIONS = list(seed_list)
    # gp_runtime.seeds false -> no seed tree at all (the search starts from random trees only)
    use_seeds = os.environ.get("SYMCLF_CPU_SEEDS", "1") == "1"
    if not use_seeds:
        gp_engine.SEED_EXPRESSIONS = []
    described = f"{len(custom)} seed tree(s) of gp_runtime.seed_trees: {custom}" if custom else "the two seed trees"
    print(f"seeds: {described if use_seeds else 'NONE (gp_runtime.seeds false)'}", flush=True)

    cpu_total = int(resources.get("CPU", 1))
    reserve = int(os.environ.get("SYMCLF_CPU_DRIVER_CPUS", "2"))
    actors, hosts = create_actor_pool(max(1, cpu_total - reserve))
    from collections import Counter
    print(f"scheduling: {len(actors)} single-thread CPU actors on {dict(Counter(h.split(':')[0] for h in hosts))} | "
          f"constant tuning {gp_engine.TUNE_MODE} | max_len {gp_engine.MAX_TREE_LENGTH} | "
          f"CMA popsize {gp_engine.CMA_ELITE_POPSIZE}, seed {gp_engine.CMA_ELITE_SEED}", flush=True)
    batch = int(os.environ.get("SYMCLF_CPU_S1_BATCH", "2"))
    enable_actor_fitness(GPSymbolicRegressor, actors=actors, batch_size=batch)

    gpsr = GPSymbolicRegressor(
        pset_config=pset, fitness=fitness_marker, score_func=score, predict_func=predict,
        common_data={"true_data": train_data, "penalty": penalty}, callback_func=assign_attributes,
        custom_logger=make_generation_logger(best_history_file), print_log=True, batch_size=batch,
        seed_str=list(seed_list) if use_seeds else None, **regressor_params,
    )
    tic = time.time()
    gpsr.fit(train_data)
    toc = time.time()
    best = gpsr.get_best_individuals(1)[0]
    consts = getattr(best, "consts", None)
    print("Best expression:", str(best), "\nBest constants:", consts, "\nElapsed:", toc - tic, flush=True)
    with open(os.path.join(run_dir, f"{tag}_best_expression.txt"), "w") as fh:
        fh.write(f"Best Expression:\n{best}\nBest Fitness:\n{best.fitness.values[0]}\nBest Parameters:\n{consts}\n")

    if os.environ.get("SYMCLF_CPU_FINAL_AUDIT", "1") == "1" and consts is not None and len(consts):
        cmd = [sys.executable, "-m", "src.CPU.audit", "--expression", str(best),
               "--constants", ",".join(repr(float(c)) for c in consts),
               "--json", os.path.join(run_dir, f"{tag}_final_audit.json")]
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        print("final reference audit (audit.py):\n" + out.stdout[-3000:] + out.stderr[-1000:], flush=True)
    ray.shutdown()


if __name__ == "__main__":
    if len(sys.argv) != 2 or not os.path.isdir(sys.argv[1]):
        raise SystemExit("usage: python -m src.CPU.run examples/<name>")
    main(sys.argv[1])
