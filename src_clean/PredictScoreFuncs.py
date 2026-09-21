# predict and score functions


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
