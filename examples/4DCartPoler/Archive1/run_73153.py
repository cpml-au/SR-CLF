import os
import sys
import json #

# Absolute path to project root (../../ from run.py)
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Make sure Ray subprocesses also see this
os.environ["PYTHONPATH"] = project_root + ":" + os.environ.get("PYTHONPATH", "")

import numpy as np
from deap import gp
from flex.gp.regressor import GPSymbolicRegressor
import ray
from flex.gp import util
from flex.gp.primitives import add_primitives_to_pset_from_dict
from src.Fitness import fitness, assign_attributes
from src.PredictScoreFuncs import predict, score
import src.Functions
import time


# Define the four-dimensional cart-pole training grid.
GRID_POINTS = 21  # odd: grid contains x3=x4=0, so the decay check sees the plane
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
    yamlfile = "config.yaml"
    filename = yamlfile

    regressor_params, config_file_data = util.load_config_data(filename)

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
    seed_expr = "add(add(add(mul(x1, x1), mul(x2, x2)), add(mul(x3, x3), mul(x4, x4))), add(mul(x1, x2), mul(x1, x2))))"

    gpsr = GPSymbolicRegressor(
        pset_config=pset,
        fitness=fitness,
        score_func=score,
        predict_func=predict,
        common_data=common_data,
        callback_func=callback_func,
        custom_logger=generation_logger,
        print_log=True,
        batch_size=1,
        seed_str=[seed_expr],
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
