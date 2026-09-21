"""6-D GP: symbolic regression of a control Lyapunov function for the 6-D quadrotor (examples/6D's system),
with the 4-D V9 pipeline (gp6d_engine.py) on CPU Ray actors. Mirrors examples/4DCartPolerV9/run.py: the same
Flex regressor, config.yaml, seeds of the same kind, the same generation log and jsonl; only the dynamics, the
6 variables and the 6-D fitness settings (examples/6D/env.sh) differ."""
import json
import os
import subprocess
import sys
import time

example_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(example_dir, "../../"))
for _p in (project_root, example_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ["PYTHONPATH"] = os.pathsep.join([example_dir, project_root, os.environ.get("PYTHONPATH", "")])

import numpy as np  # noqa: E402
import ray  # noqa: E402
from deap import gp  # noqa: E402
from flex.gp import util  # noqa: E402
from flex.gp.primitives import add_primitives_to_pset_from_dict  # noqa: E402
from flex.gp.regressor import GPSymbolicRegressor  # noqa: E402

import src.Functions  # noqa: E402
from src.PredictScoreFuncs import predict, score  # noqa: E402
from core import config as C6  # noqa: E402
from core.templates import template_expression  # noqa: E402
import gp6d_engine  # noqa: E402
from gp6d_engine import create_actor_pool, persistent_actor_mapper  # noqa: E402

N = C6.system().N_STATES   # 2026-09-20: was hardcoded 6 -- the 2-link (4) and bicycle (2) GP runs built
# 6-variable trees and seeds (job 147956 diagnostics: seeds of 27/125 nodes), which the actors gated
JOB_ID = os.environ.get("SLURM_JOB_ID")
BEST_HISTORY_FILE = os.path.join(example_dir, f"{JOB_ID}_best_per_generation.jsonl" if JOB_ID else "best_per_generation.jsonl")


def _apply_env_overrides(regressor_params, config_file_data):
    """The 4-D V9 overrides (SYMCLF_OPT_*): population, islands, generations, operators, reg_param, migration."""
    def _f(name, cast, current):
        raw = os.environ.get(name)
        return current if raw is None or raw == "" else cast(raw)

    gp_cfg = config_file_data["gp"]
    islands = _f("SYMCLF_OPT_NUM_ISLANDS", int, int(regressor_params["num_islands"]))
    total = _f("SYMCLF_OPT_POPULATION", int, int(regressor_params["num_individuals"]) * islands)
    regressor_params["num_islands"] = islands
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
    print(f"6D GP hyperparameters: population={regressor_params['num_individuals'] * islands} "
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
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return str(value)


def _g(value, nd=4):
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
    d4 = bd.get("det4") or {}
    if bd.get("dead"):
        stage = "dead (grid price stands, no check)"
    elif d4:
        pen = d4.get("penalty_terms")
        stage = (f"det4 price={_g(bd.get('det4_price'))} c_v*={_g(d4.get('cv'))} kappa*={_g(d4.get('kappa'))}"
                 f" certified={bd.get('det4_certified')} worst_r={_g(d4.get('worst_r'), 3)}"
                 + ("" if not pen else " pen=" + "/".join(_g(t, 3) for t in pen)))
    else:
        stage = "no det4 numbers"
    cm = bd.get("cma") or {}
    if cm.get("after") is not None:
        stage += (f" | cma {_g(cm.get('before'))}->{_g(cm.get('after'))} ({cm.get('evals')} evals,"
                  f" certified_at {cm.get('certified_at')}, {_g(cm.get('seconds'), 3)} s)")
    return (f"grid={_g(bd.get('pre_exact'))} [roa={_g(bd.get('grid_roa'), 3)}] lattice c_v={_g(bd.get('grid_cv'))}"
            f" kappa={_g(bd.get('grid_kappa'))} | extra={_g(bd.get('extra_penalty'))} | {stage}"
            f" | boundary_min={_g(bd.get('grid_boundary_min'))} volume={_g(bd.get('grid_certified_volume'), 3)}")


def make_generation_logger(filename):
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
    raise RuntimeError("the 6-D fitness runs only through the actor mapper")


def enable_actor_fitness(regressor_class, *, actors, batch_size):
    def register_map(self, toolbox):
        if not self.multiprocessing:
            raise RuntimeError("the 6-D actor mapper requires multiprocessing=True")
        toolbox.register("map", persistent_actor_mapper, actors=actors, batch_size=int(batch_size))

    regressor_class._GPSymbolicRegressor__register_map = register_map


def main():
    if not ray.is_initialized():
        ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))
    resources = ray.cluster_resources()
    print(f"6D Ray resources: {resources}", flush=True)
    print(f"6D system {C6.system().NAME}: box {C6.box()} rho {C6.RHO} b-norm {C6.B_NORM} grid {C6.grid_points()}^6 | "
          f"hinges cv {C6.PEN_CV}@{C6.PEN_CV_TARGET} kappa {C6.PEN_KAPPA}@{C6.PEN_KAPPA_TARGET} "
          f"vol {C6.PEN_VOL}@{C6.PEN_VOL_TARGET}", flush=True)

    regressor_params, config_file_data = util.load_config_data(os.environ.get("SYMCLF_OPT_CONFIG", "config.yaml"))
    regressor_params, config_file_data = _apply_env_overrides(regressor_params, config_file_data)
    with open(BEST_HISTORY_FILE, "w", encoding="utf-8"):
        pass

    pset = gp.PrimitiveSetTyped("MAIN", [float] * N, float)
    pset.renameArguments(**{f"ARG{i}": f"x{i + 1}" for i in range(N)})
    pset = add_primitives_to_pset_from_dict(pset, config_file_data["gp"]["primitives"])
    pset.addTerminal(object, float, "a")
    penalty = config_file_data["gp"]["penalty"]

    # the training data object Flex needs (the fitness itself builds its own 7^6 grid in core.lattice)
    axes = [np.linspace(-h, h, C6.grid_points()) for h in C6.box()]
    train_data = src.Functions.Dataset("true_data", axes, None)

    # seeds of the 4-D kind: a coefficient-free sum of squares with one cross term, and the full quadratic
    # template with `a` placeholders (21 monomials) -- never the ARE's coefficients (standing rule)
    squares = [f"mul(x{i}, x{i})" for i in range(1, N + 1)]
    seed_expr = squares[0]
    for term in squares[1:] + ["mul(x1, x2)"]:
        seed_expr = f"add({seed_expr}, {term})"
    seed_expr2 = template_expression(N, "full")
    # 2026-09-20 (the user): the CMA stage must certify these in generation 1, as the <sys>_LQR run does
    gp6d_engine.SEED_EXPRESSIONS = [seed_expr, seed_expr2]

    cpu_total = int(resources.get("CPU", 1))
    reserve = int(os.environ.get("SYMCLF6D_DRIVER_CPUS", "2"))
    actors, hosts = create_actor_pool(max(1, cpu_total - reserve))
    from collections import Counter
    print(f"6D scheduling: {len(actors)} single-thread CPU actors on {dict(Counter(h.split(':')[0] for h in hosts))} | "
          f"tuner SEA {int(os.environ.get('SYMCLF_GPU2_TUNER_RANDOM_POPULATION', '35')) + 1} x "
          f"{os.environ.get('SYMCLF_GPU2_TUNER_GENERATIONS', '5')} + Adam {os.environ.get('SYMCLF_TUNER_REFINE_STEPS', '40')} "
          f"| max_len {os.environ.get('SYMCLF_MAX_TREE_LENGTH', '90')} | CMA {os.environ.get('SYMCLF_CMA_ELITE_COUNT', '1')} x "
          f"{os.environ.get('SYMCLF_CMA_ELITE_EVALS', '0')} evals, popsize {os.environ.get('SYMCLF_CMA_ELITE_POPSIZE', '18')}, "
          f"seed {os.environ.get('SYMCLF_CMA_ELITE_SEED', '1')}, gate {os.environ.get('SYMCLF_CMA_ELITE_MAX_FITNESS', '0')}",
          flush=True)
    batch = int(os.environ.get("SYMCLF6D_S1_BATCH", "2"))
    enable_actor_fitness(GPSymbolicRegressor, actors=actors, batch_size=batch)

    gpsr = GPSymbolicRegressor(
        pset_config=pset, fitness=fitness_marker, score_func=score, predict_func=predict,
        common_data={"true_data": train_data, "penalty": penalty}, callback_func=assign_attributes,
        custom_logger=make_generation_logger(BEST_HISTORY_FILE), print_log=True, batch_size=batch,
        seed_str=[seed_expr, seed_expr2], **regressor_params,
    )
    tic = time.time()
    gpsr.fit(train_data)
    toc = time.time()
    best = gpsr.get_best_individuals(1)[0]
    consts = getattr(best, "consts", None)
    print("Best expression:", str(best), "\nBest constants:", consts, "\nElapsed:", toc - tic, flush=True)
    with open(os.path.join(example_dir, f"{JOB_ID or 'local'}_best_expression.txt"), "w") as fh:
        fh.write(f"Best Expression:\n{best}\nBest Fitness:\n{best.fitness.values[0]}\nBest Parameters:\n{consts}\n")

    if os.environ.get("SYMCLF6D_FINAL_AUDIT", "1") == "1" and consts is not None and len(consts):
        cmd = [sys.executable, os.path.join(example_dir, "audit.py"), "--expression", str(best),
               "--constants", ",".join(repr(float(c)) for c in consts),
               "--json", os.path.join(example_dir, f"{JOB_ID or 'local'}_final_audit.json")]
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=example_dir)
        print("6D final referee audit (audit.py):\n" + out.stdout[-3000:] + out.stderr[-1000:], flush=True)
    ray.shutdown()


if __name__ == "__main__":
    main()
