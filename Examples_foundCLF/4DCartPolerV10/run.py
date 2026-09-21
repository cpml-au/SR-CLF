import os
import sys
import json #

# Absolute paths used by both the Ray driver and its subprocesses.
example_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(
    os.path.join(example_dir, "../../")
)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if example_dir not in sys.path:
    sys.path.insert(0, example_dir)

# Make sure ``src.Fitness`` resolves this folder's Evaluate.py in Ray workers.
os.environ["PYTHONPATH"] = os.pathsep.join(
    [example_dir, project_root, os.environ.get("PYTHONPATH", "")]
)

import numpy as np
from deap import gp
from flex.gp.regressor import GPSymbolicRegressor
import ray
from flex.gp import util
from flex.gp.primitives import add_primitives_to_pset_from_dict
from src.PredictScoreFuncs import predict, score
from src4DCartPoleV10.ray_fitness import create_actor_pool, create_cpu_pool
import src.Functions
import time

from gpu_ray import enable_persistent_gpu_fitness


# Long-lived Ray actors, with CUDA visibility assigned by Ray. V8: the tuner runs on the
# actor's GPU share, the check (det4_fast, numba) on its cores; ACTORS_PER_GPU actors share
# each physical GPU (fractional num_gpus), turning the 32 host cores into parallel checks.
# Each actor preallocates 1/ACTORS_PER_GPU of the GPU (set the XLA mem
# fraction accordingly in the Slurm script).
# ACTORS_PER_GPU is capped by the GPU-memory-heavy constant tuner (~7.2 GiB
# per actor, measured on job 99799), NOT by the exact check (which is CPU-bound
# and GPU-tiny). 3 actors/GPU with MEM_FRACTION 0.26 (=10.4 GiB each) fits the
# tuner with headroom; 4/GPU at 0.18 OOM'd. Raise only if the tuner footprint
# is reduced (smaller SYMCLF_GPU2_TUNER_FUSION).
GPU_COUNT = max(1, int(os.environ.get("SYMCLF_GPU2_TOTAL_GPUS", "3")))
ACTORS_PER_GPU = max(1, int(os.environ.get("SYMCLF_GPU3_ACTORS_PER_GPU", "3")))
WORKER_COUNT = GPU_COUNT * ACTORS_PER_GPU
GPU_FRACTION = 1.0 / ACTORS_PER_GPU


def _apply_env_overrides(regressor_params, config_file_data):
    """Let an Optuna trial set every searched GP hyperparameter by env var.

    This replaces the template-rendering approach of examples/4DCartPolerOpt:
    the trial passes SYMCLF_OPT_* through ``sbatch --export`` and nothing on
    disk has to be rewritten per trial. Unset variables keep config.yaml's
    value, so a plain ``sbatch jobGPU4Opt.slurm`` behaves exactly like GPU4_1.
    """
    def _f(name, cast, current):
        raw = os.environ.get(name)
        return current if raw is None or raw == "" else cast(raw)

    gp_cfg = config_file_data["gp"]

    # Total population is num_individuals * num_islands. The sweep fixes the
    # total (default 2500) and derives the per-island count.
    islands = _f("SYMCLF_OPT_NUM_ISLANDS", int,
                 int(regressor_params["num_islands"]))
    total_pop = _f("SYMCLF_OPT_POPULATION", int,
                   int(regressor_params["num_individuals"]) * islands)
    per_island = max(1, total_pop // max(1, islands))

    regressor_params["num_islands"] = islands
    regressor_params["num_individuals"] = per_island
    regressor_params["generations"] = _f(
        "SYMCLF_OPT_GENERATIONS", int, int(regressor_params["generations"]))
    regressor_params["crossover_prob"] = _f(
        "SYMCLF_OPT_CROSSOVER_PROB", float,
        float(regressor_params["crossover_prob"]))
    regressor_params["mut_prob"] = _f(
        "SYMCLF_OPT_MUT_PROB", float, float(regressor_params["mut_prob"]))

    penalty = dict(gp_cfg["penalty"])
    penalty["reg_param"] = _f("SYMCLF_OPT_REG_PARAM", float,
                              float(penalty["reg_param"]))
    gp_cfg["penalty"] = penalty

    # flex flattens multi_island.migration to mig_freq / mig_frac
    if regressor_params.get("mig_freq") is not None:
        regressor_params["mig_freq"] = _f(
            "SYMCLF_OPT_MIGRATION_FREQ", int, int(regressor_params["mig_freq"]))
    if regressor_params.get("mig_frac") is not None:
        regressor_params["mig_frac"] = _f(
            "SYMCLF_OPT_MIGRATION_FRAC", float,
            float(regressor_params["mig_frac"]))

    print(
        "GPU5 GP hyperparameters: "
        f"population={per_island * islands} "
        f"(num_individuals={per_island} x num_islands={islands}), "
        f"generations={regressor_params['generations']}, "
        f"crossover={regressor_params['crossover_prob']}, "
        f"mutation={regressor_params['mut_prob']}, "
        f"reg_param={penalty['reg_param']}, "
        f"mig_freq={regressor_params.get('mig_freq')}, "
        f"mig_frac={regressor_params.get('mig_frac')}",
        flush=True,
    )
    return regressor_params, config_file_data


def assign_attributes(individuals, attributes):
    for individual, values in zip(individuals, attributes):
        individual.consts = values["consts"]
        individual.fitness.values = values["fitness"]
        # 2026-09-09: the per-stage fitness ledger (ray_fitness._breakdown) rides on the
        # individual so the generation logger can print the champion's breakdown.
        individual.breakdown = values.get("breakdown")


# 2026-09-09: certificate pair (c*, kappa*(Omega_c*)) of the champion every N generations on the
# driver CPU (certify.py, ~20-40 s per call); 0 = off. It is a report, never a fitness.
CERTIFY_EVERY = int(os.environ.get("SYMCLF_CERTIFY_EVERY", "0"))


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


def _pen(terms):
    """V6: the three positivity penalty terms (c_v*, kappa*, certified volume) when present."""
    if not terms:
        return ""
    return " pen=" + "/".join(_g(t, 3) for t in terms)


def format_breakdown(bd):
    """One line: fitness = grid (terms) + extra (nested + length) + det4 price."""
    if not bd:
        return "breakdown: none (individual not evaluated by the actor mapper)"
    terms = bd.get("grid_terms") or {}
    nonzero = " ".join(f"{k}={_g(v, 3)}" for k, v in terms.items() if v)
    d4 = bd.get("det4") or {}
    if bd.get("dead"):
        stage = "dead (grid price stands, no check)"
    elif d4:
        stage = (
            f"det4[{bd.get('det4_source')}] price={_g(bd.get('det4_price'))}"
            f" c_v*={_g(d4.get('cv'))} kappa*={_g(d4.get('kappa'))}"
            f" certified={bd.get('det4_certified')} worst_r={_g(d4.get('worst_r'), 3)}"
            + _pen(d4.get("penalty_terms"))
        )
    else:
        stage = "no det4 numbers"
    cm = bd.get("cma") or {}
    if cm.get("after") is not None:
        stage += (
            f" | cma {_g(cm.get('before'))}->{_g(cm.get('after'))} ({cm.get('evals')} evals,"
            f" certified_at {cm.get('certified_at')}, {_g(cm.get('seconds'), 3)} s)"
        )
    rf = bd.get("refined") or {}
    if rf.get("rule_after") is not None:
        stage += (
            f" | refine {_g(rf.get('rule_before'))}->{_g(rf.get('rule_after'))}"
            f" ({rf.get('steps')} steps, {_g(rf.get('seconds'), 3)} s)"
        )
    return (
        f"grid={_g(bd.get('pre_exact'))} [{nonzero or 'all terms 0'}]"
        f" lattice c_v={_g(bd.get('grid_cv'))} kappa={_g(bd.get('grid_kappa'))}"
        f" | extra={_g(bd.get('extra_penalty'))} | {stage}"
        f" | boundary_min={_g(bd.get('grid_boundary_min'))} w_scale={_g(bd.get('grid_w_scale'), 3)}"
    )


def gpu2_actor_fitness_marker(*_args, **_kwargs):
    """Flex registration marker; persistent actors perform the real fitness."""
    raise RuntimeError("V8 fitness marker must only be used by the actor mapper")


# Define the four-dimensional cart-pole training grid.
# 4-D cart-pole: (x1, x2, x3, x4) = (cart position, cart velocity, pole angle, pole rate).
# ODD count is required: the grid must contain 0.0 on every axis so the origin
# and the coordinate planes are sampled exactly.
#
# 4DCartPolerV6: 21 per axis, the 4-D file's (GPU5_7) count, 21^4 = 194,481 points. 3DCartPolerV6
# used 41 because dropping a dimension made 41^3 = 68,921 affordable; 41^4 = 2,825,761 is 41x the
# 3-D grid and does not fit the tuner's per-actor GPU budget.
GRID_POINTS = 21
# V2: PER-AXIS half-widths, matched to the ARE form -- see BaseEvaluate.py.
# Taken from the base so the training grid, SHGO_BOUNDS and the rollout ICs
# cannot silently disagree.
import Evaluate as _Eval  # noqa: E402

_H = _Eval.BOX_HALF_WIDTHS
x_Domain = max(_H)
x1_vals = np.linspace(-_H[0], _H[0], GRID_POINTS)
x2_vals = np.linspace(-_H[1], _H[1], GRID_POINTS)
x3_vals = np.linspace(-_H[2], _H[2], GRID_POINTS)
x4_vals = np.linspace(-_H[3], _H[3], GRID_POINTS)
print(f"V2 box half-widths = {_H}  grid = {GRID_POINTS}^4", flush=True)
X1, X2, X3, X4 = np.meshgrid(x1_vals, x2_vals, x3_vals, x4_vals, indexing="ij")

JOB_ID = os.environ.get("SLURM_JOB_ID")
BEST_HISTORY_FILE = os.path.join(
    os.path.dirname(__file__),
    f"{JOB_ID}_best_per_generation.jsonl" if JOB_ID else "best_per_generation.jsonl",
)


def make_generation_logger(filename):
    generation = 0

    def log_best(best_individuals):
        nonlocal generation
        generation += 1

        best = best_individuals[0]
        consts = getattr(best, "consts", None)

        if consts is not None:
            consts = np.asarray(consts, dtype=float).tolist()

        record = {
            "generation": generation,
            "expression": str(best),
            "constants": consts,
            "fitness": float(best.fitness.values[0]),
            # 2026-09-09: every fitness term of the champion (format_breakdown / README)
            "breakdown": getattr(best, "breakdown", None),
        }
        print(
            f"GEN {generation} best fitness={record['fitness']:.6g} len={len(best)} :: "
            + format_breakdown(record["breakdown"]),
            flush=True,
        )
        if CERTIFY_EVERY > 0 and generation % CERTIFY_EVERY == 0 and consts is not None:
            from certify import certify, format_certificate  # noqa: PLC0415

            record["certificate"] = certify(str(best), consts, _Eval)
            print(f"GEN {generation} " + format_certificate(record["certificate"]), flush=True)

        with open(filename, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, default=_json_default) + "\n")

    return log_best


def _referee_audit(expression, constants):
    """END: the CPU referee (src/srcCPU DET4, sympy evaluator) on the champion with the run's DET4
    switches; the audit of THE CHECK, report only (replaces V7's GPU root audit)."""
    from src.srcCPU.det4_check_cpu import check_det4_cpu  # noqa: PLC0415
    from src.srcCPU.symbolic import sympy_expression  # noqa: PLC0415

    started = time.time()
    rho = float(_Eval.DET4_RHO)
    V = sympy_expression(str(expression), [float(c) for c in (constants or [])])
    r = check_det4_cpu(
        V, fSR=_Eval.fSR, GSR=_Eval.GSR, bounds=_Eval.SHGO_BOUNDS, rhos=(rho,),
        origin_tol=0.0, gate_on_pd=False,
        project_ascent=os.environ.get("SYMCLF_DET4_PROJECT_ASCENT", "0") == "1",
        origin_seeds=os.environ.get("SYMCLF_DET4_ORIGIN_SEEDS", "0") == "1",
    )
    row = r.rows[rho]
    return {
        "status": r.status, "c_v_star": r.pd_rate, "pd_valid": bool(r.pd_valid),
        "kappa_star": row.kappa_star, "artstein_valid": bool(row.artstein_valid),
        "worst_r": row.worst_r, "worst_abs_b": row.worst_abs_b, "seconds": time.time() - started,
    }


def main():
    if not ray.is_initialized():
        ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))
    cluster_gpus = int(ray.cluster_resources().get("GPU", 0))
    if cluster_gpus < GPU_COUNT:
        raise RuntimeError(
            f"V8 requested {GPU_COUNT} GPUs, but Ray exposes {cluster_gpus}."
        )
    print(f"V8 Ray resources: {ray.cluster_resources()}", flush=True)

    yamlfile = os.environ.get("SYMCLF_OPT_CONFIG", "config.yaml")
    filename = yamlfile

    regressor_params, config_file_data = util.load_config_data(filename)
    regressor_params, config_file_data = _apply_env_overrides(
        regressor_params, config_file_data
    )

    # Clear history once, on the Ray driver.
    with open(BEST_HISTORY_FILE, "w", encoding="utf-8"):
        pass

    generation_logger = make_generation_logger(BEST_HISTORY_FILE)

    pset = gp.PrimitiveSetTyped(
        "MAIN",
        [float, float, float, float],
        float,
    )

    pset.renameArguments(ARG0="x1", ARG1="x2", ARG2="x3", ARG3="x4")
    pset = add_primitives_to_pset_from_dict(pset, config_file_data["gp"]["primitives"])
    penalty = config_file_data["gp"]["penalty"]

    train_data = src.Functions.Dataset(
        "true_data", [x1_vals, x2_vals, x3_vals, x4_vals], None
    )
    # attach grid ONCE
    train_data.X1 = X1
    train_data.X2 = X2
    train_data.X3 = X3
    train_data.X4 = X4
    train_data.grid_shape = X1.shape
    train_data.mesh = [X1, X2, X3, X4]

    common_data = {"true_data": train_data, "penalty": penalty}
    callback_func = assign_attributes
    pset.addTerminal(object, float, "a")

    # 4-D seeds, copied from examples/4DCartPolerGPU5_7/run.py. Coefficient-free by design: the GP
    # is given the STRUCTURE of a quadratic and must find the coefficients through the tuner.
    seed_expr = "add(add(add(mul(x1, x1), mul(x2, x2)), add(mul(x3, x3), mul(x4, x4))), mul(x1, x2))"
    seed_expr2 = "add(add(add(add(add(add(add(add(add(mul(a, mul(x1, x1)), mul(a, mul(x1, x2))), mul(a, mul(x1, x3))), mul(a, mul(x1, x4))), mul(a, mul(x2, x2))), mul(a, mul(x2, x3))), mul(a, mul(x2, x4))), mul(a, mul(x3, x3))), mul(a, mul(x3, x4))), mul(a, mul(x4, x4)))"

    total_population = (
        int(regressor_params["num_individuals"])
        * int(regressor_params["num_islands"])
    )
    raw_batch_size = os.environ.get("SYMCLF_GPU2_BATCH_SIZE", "16").lower()
    if raw_batch_size == "auto":
        gpu_batch_size = 16
    else:
        gpu_batch_size = max(1, int(raw_batch_size))
    # V8: det4_fast candidates per actor call (CPU, sequential inside the actor).
    det4_batch_size = max(1, int(os.environ.get("SYMCLF_DET4_BATCH_SIZE", "32")))
    tuner_fusion = max(
        1, int(os.environ.get("SYMCLF_GPU2_TUNER_FUSION", "16"))
    )
    total_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "32"))
    cpus_per_actor = max(1, total_cpus // WORKER_COUNT)
    actors, actor_devices = create_actor_pool(
        train_data,
        penalty,
        WORKER_COUNT,
        tuner_fusion=tuner_fusion,
        gpu_fraction=GPU_FRACTION,
        cpus_per_actor=cpus_per_actor,
    )
    # V10 (2026-09-19): the CMA pool -- single-thread det4 actors on the nodes advertising `det4pool`
    cpu_actors, cpu_hosts = [], []
    if os.environ.get("SYMCLF_CMA_POOL", "0") == "1":
        cpu_actors, cpu_hosts = create_cpu_pool(int(os.environ.get("SYMCLF_CMA_POOL_ACTORS", "0")))
        from collections import Counter
        print(f"V10 cma pool: {len(cpu_actors)} det4 actors on {dict(Counter(h.split(':')[0] for h in cpu_hosts))}",
              flush=True)
    enable_persistent_gpu_fitness(
        GPSymbolicRegressor,
        actors=actors,
        pre_batch_size=gpu_batch_size,
        det4_batch_size=det4_batch_size,
        cpu_actors=cpu_actors,
    )
    print(
        f"V8 scheduling: {WORKER_COUNT} persistent actors "
        f"({GPU_COUNT} GPUs x {ACTORS_PER_GPU} actors/GPU, "
        f"gpu_fraction={GPU_FRACTION:.3f}, cpus/actor={cpus_per_actor}), "
        f"pre_batch_size={gpu_batch_size}, det4_batch_size={det4_batch_size}, "
        f"refine_top={os.environ.get('SYMCLF_DET4_REFINE_TOP', '3')}, "
        f"skip_dead={os.environ.get('SYMCLF_DET4_SKIP_DEAD', '0')}, "
        f"cma_elite_evals={os.environ.get('SYMCLF_CMA_ELITE_EVALS', '0')}, "
        f"tuner_fusion={tuner_fusion}, "
        f"population={total_population}",
        flush=True,
    )
    print(f"V8 actor devices: {actor_devices}", flush=True)

    gpsr = GPSymbolicRegressor(
        pset_config=pset,
        fitness=gpu2_actor_fitness_marker,
        score_func=score,
        predict_func=predict,
        common_data=common_data,
        callback_func=callback_func,
        custom_logger=generation_logger,
        print_log=True,
        batch_size=gpu_batch_size,
        seed_str=[seed_expr, seed_expr2],
        # max_height=10,
        **regressor_params,
    )

    tic = time.time()
    gpsr.fit(train_data)
    toc = time.time()

    best_ind = gpsr.get_best_individuals(1)[0]   # Access the best individual

    best_parameters = getattr(best_ind, "consts", None)  # Save the best parameters
    if best_parameters is not None:
        print("Best parameters = ", best_parameters)

    if os.environ.get("SYMCLF_V8_FINAL_AUDIT", "1") == "1":
        try:
            final_audit = _referee_audit(
                str(best_ind), [] if best_parameters is None else best_parameters
            )
        except Exception as exc:  # report only, never lose the run's result
            final_audit = {"status": f"error: {type(exc).__name__}: {str(exc)[:200]}"}
        print("V8 final referee audit (src/srcCPU DET4): ", final_audit, flush=True)

    print("Elapsed time = ", toc - tic)
    time_per_individual = (toc - tic) / (
        gpsr.generations * gpsr.num_individuals * gpsr.num_islands
    )
    print("Time per individual = ", time_per_individual)
    print("Individuals per sec = ", 1 / time_per_individual)

    # Access and save the best individual
    best_expression = str(best_ind)  # Save the best expression
    best_fitness = gpsr.get_train_fit_history()[-1]  # Save the best fitness score

    # Write best expression to a file
    with open("best_expression.txt", "w") as file:
        file.write(f"Best Expression:\n{best_expression}\n")
        file.write(f"Best Fitness:\n{best_fitness}\n")
        file.write(f"Best Parameters:\n{best_parameters}\n")

    print("Best Expression and Fitness saved to 'best_expression.txt'.")
    ray.shutdown()


if __name__ == "__main__":
    main()
