"""The predict / score hooks of flex's GPSymbolicRegressor: stubs that return None.

The regressor requires a predict_func (score_func serves its validation and its score method).  The
runner (src/CPU/run.py) passes these two, runs with validate: False and never calls predict or score:
the fitness comes from src/CPU/gp_engine.py.  Each docstring keeps a former body, which calls
eval_MSE_sol (not in this repository).
"""


def predict(individuals_str, toolbox, true_data, penalty):
    """
    callables = compile_individuals(toolbox, individuals_str)

    u = [None] * len(individuals_str)

    for i, ind in enumerate(callables):
        _, u[i] = eval_MSE_sol(ind, true_data)
    """
    return None


def score(individuals_str, toolbox, true_data, penalty):
    """
    callables = compile_individuals(toolbox, individuals_str)

    MSE = [None] * len(individuals_str)

    for i, ind in enumerate(callables):
        MSE[i], _ = eval_MSE_sol(ind, true_data)
    """
    return None
