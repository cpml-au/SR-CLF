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
from srcGPU5_7.ray_fitness import create_actor_pool
import src.Functions
import time

from gpu_ray import enable_persistent_gpu_fitness


# Long-lived Ray actors, with CUDA visibility assigned by Ray. The full exact
# check is CPU-bound (SciPy SLSQP polish; the GPU sits ~2% idle), so we
# oversubscribe: ACTORS_PER_GPU actors share each physical GPU (fractional
# num_gpus), turning the 32 host cores into that many parallel exact checks.
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


def gpu2_actor_fitness_marker(*_args, **_kwargs):
    """Flex registration marker; persistent actors perform the real fitness."""
    raise RuntimeError("GPU5 fitness marker must only be used by the actor mapper")


# Define the four-dimensional cart-pole training grid.
GRID_POINTS = 21  # odd: grid contains x3=x4=0, so the decay check sees the plane
# 21^4 = 194481 points (3.8x the 15^4 = 50625 used before). The grid cannot
# certify a pointwise condition either way -- the search-based PD check does
# that -- but a denser grid gives the GP less room to hide violations between
# samples, at ~3.8x the grid-fitness cost.
x_Domain = 0.25
x1_vals = np.linspace(-x_Domain, x_Domain, GRID_POINTS)
x2_vals = np.linspace(-x_Domain, x_Domain, GRID_POINTS)
x3_vals = np.linspace(-x_Domain, x_Domain, GRID_POINTS)
x4_vals = np.linspace(-x_Domain, x_Domain, GRID_POINTS)
X1, X2, X3, X4 = np.meshgrid(
    x1_vals, x2_vals, x3_vals, x4_vals, indexing="ij"
)

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
        }

        with open(filename, "a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

    return log_best


def main():
    if not ray.is_initialized():
        ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))
    cluster_gpus = int(ray.cluster_resources().get("GPU", 0))
    if cluster_gpus < GPU_COUNT:
        raise RuntimeError(
            f"GPU5 requested {GPU_COUNT} GPUs, but Ray exposes {cluster_gpus}."
        )
    print(f"GPU5 Ray resources: {ray.cluster_resources()}", flush=True)

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

    # seed_expr = "add(add(mul(x1, x1), mul(x2, x2)), add(mul(x3, x3), mul(x4, x4)))"
    seed_expr = "add(add(add(mul(x1, x1), mul(x2, x2)), add(mul(x3, x3), mul(x4, x4))), mul(x1, x2))"

    # seed_expr = "add(add(add(add(add(add(add(add(add(mul(1.965366, mul(x1, x1)), mul(2.862562, mul(x1, x2))), mul(-13.368898, mul(x1, x3))), mul(-5.925124, mul(x1, x4))), mul(2.143945, mul(x2, x2))), mul(-20.349648, mul(x2, x3))), mul(-8.970854, mul(x2, x4))), mul(50.818966, mul(x3, x3))), mul(43.681856, mul(x3, x4))), mul(9.639298, mul(x4, x4)))"
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
    # GPU5 verifies a whole batch of candidates per exact GPU call (batched
    # scan + batched line-bisection). Larger batches amortize the launch-bound
    # bisection further; 24 is a safe default per actor.
    exact_batch_size = max(
        1, int(os.environ.get("SYMCLF_GPU4_EXACT_BATCH_SIZE", "24"))
    )
    cheap_batch_size = max(
        1, int(os.environ.get("SYMCLF_GPU2_CHEAP_FUSION", "32"))
    )
    exact_wave_first_limit = max(
        1,
        int(os.environ.get("SYMCLF_GPU2_EXACT_WAVE1_CHEAP_MAX", "500")),
    )
    exact_wave_second_limit = max(
        exact_wave_first_limit,
        int(os.environ.get("SYMCLF_GPU2_EXACT_WAVE2_CHEAP_MAX", "1500")),
    )
    full_exact_enabled = (
        os.environ.get("SYMCLF_GPU2_FULL_EXACT_ENABLED", "1") == "1"
    )
    tuner_fusion = max(
        1, int(os.environ.get("SYMCLF_GPU2_TUNER_FUSION", "16"))
    )
    total_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "32"))
    cpus_per_actor = max(1, total_cpus // WORKER_COUNT)
    actors, exact_cache, actor_devices = create_actor_pool(
        train_data,
        penalty,
        WORKER_COUNT,
        tuner_fusion=tuner_fusion,
        full_exact_enabled=full_exact_enabled,
        gpu_fraction=GPU_FRACTION,
        cpus_per_actor=cpus_per_actor,
    )
    enable_persistent_gpu_fitness(
        GPSymbolicRegressor,
        actors=actors,
        exact_cache=exact_cache,
        pre_batch_size=gpu_batch_size,
        cheap_batch_size=cheap_batch_size,
        exact_batch_size=exact_batch_size,
        exact_wave_first_limit=exact_wave_first_limit,
        exact_wave_second_limit=exact_wave_second_limit,
        full_exact_enabled=full_exact_enabled,
    )
    print(
        f"GPU5 scheduling: {WORKER_COUNT} persistent actors "
        f"({GPU_COUNT} GPUs x {ACTORS_PER_GPU} actors/GPU, "
        f"gpu_fraction={GPU_FRACTION:.3f}, cpus/actor={cpus_per_actor}), "
        f"pre_batch_size={gpu_batch_size}, exact_batch_size={exact_batch_size}, "
        f"cheap_batch_size={cheap_batch_size}, "
        f"exact_wave_cheap_limits=({exact_wave_first_limit},"
        f"{exact_wave_second_limit}), "
        f"full_exact_enabled={int(full_exact_enabled)}, "
        f"tuner_fusion={tuner_fusion}, "
        f"population={total_population}",
        flush=True,
    )
    print(f"GPU5 actor devices: {actor_devices}", flush=True)

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

    if (
        full_exact_enabled
        and os.environ.get("SYMCLF_GPU2_FINAL_EXACT_AUDIT", "1") == "1"
    ):
        final_audit = ray.get(
            actors[0].audit_full_exact.remote(
                str(best_ind),
                [] if best_parameters is None else best_parameters,
            )
        )
        print("GPU5 final full exact audit: ", final_audit, flush=True)

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
    print("GPU5 exact cache: ", ray.get(exact_cache.stats.remote()), flush=True)
    ray.shutdown()


if __name__ == "__main__":
    main()
