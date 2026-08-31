from src.SymFunctions import (
    get_features_batch,
    compile_individuals_with_consts,
    detect_nested_function_calls,
)
from Evaluate import eval_MSE_and_tune_constants


def assign_attributes(individuals, attributes):
    for ind, attr in zip(individuals, attributes):
        ind.consts = attr["consts"]
        ind.fitness.values = attr["fitness"]


def fitness(individuals_str, toolbox, **kwargs):
    """
    Computes the fitness of a batch of individuals in a symbolic regression problem.

    Parameters:
    individuals_str : list of str
        List of individuals represented as string expressions.
    toolbox : deap.base.Toolbox
        DEAP toolbox for evolutionary computations.
    true_data : numpy.ndarray
        The dataset against which the individuals are evaluated.
    penalty : dict
        A dictionary containing penalty parameters for fitness calculation.

    Returns:
    list of tuple
        A list containing fitness values for each individual.
    """
    true_data = kwargs["true_data"]
    penalty = kwargs["penalty"]

    # Compile individuals into callable functions
    # callables = util.compile_individuals(toolbox, individuals_str)

    # Compile individuals into callable functions
    callables, const_counts = compile_individuals_with_consts(
        toolbox, individuals_str, special_term_name="a"
    )

    # Extract features from the individuals
    individ_length, nested_trigs, num_trigs = get_features_batch(individuals_str)

    # Initialize lists to store MSE and fitness values
    MSE = [None] * len(individuals_str)
    fitnesses = [None] * len(individuals_str)
    attributes = [None] * len(individuals_str)

    # Compute fitness for each individual
    for i, (ind, num_consts) in enumerate(zip(callables, const_counts)):
        # If the individual's length exceeds _, assign a very high fitness (penalty)
        if individ_length[i] >= 400:
            consts = None
            fitnesses[i] = (1e8,)
        else:
            # Evaluate the Mean Squared Error (MSE) of the individual
            ind2MSE = individuals_str[i]
            MSE[i], consts = eval_MSE_and_tune_constants(
                ind, num_consts, toolbox, true_data, ind2MSE
            )

            # Check nested "exp" and "aq"
            nested_exp = detect_nested_function_calls(str(ind2MSE), "exp")
            nested_aq  = detect_nested_function_calls(str(ind2MSE), "aq")

            # Compute the fitness function with penalties for complexity
            fitnesses[i] = (
                MSE[i]
                + 100000 * (
                    nested_trigs[i]
                    + nested_exp
                    + nested_aq
                )
                + penalty["reg_param"] * individ_length[i],
                # + 0.005 * individ_length[i],
            )
        attributes[i] = {"consts": consts, "fitness": fitnesses[i]}

    return attributes
