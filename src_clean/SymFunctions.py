# Define Functions related to individuals
from flex.gp import util
import re
from sympy import sympify, preorder_traversal, symbols, exp, Function

import numpy as np
from deap import gp


def compile_individuals_with_consts(
    toolbox, individuals_str_batch, special_term_name="a"
):
    """
    Compiles a batch of individuals (symbolic expressions) while handling special constants.

    Parameters:
        toolbox (object): A DEAP (Distributed Evolutionary Algorithms in Python) toolbox
        containing the compile method.
        individuals_str_batch (list): A list of DEAP GP trees representing symbolic expressions.
        special_term_name (str): The prefix used for identifying special constant terms.

    Returns:
        tuple: A tuple containing:
            - list: A list of compiled symbolic expressions.
            - list: A list of integer counts representing the number of special
            constants in each individual.
    """
    compiled_batch = []
    const_counts = []

    for tree in individuals_str_batch:
        const_idx = 0
        tree_clone = toolbox.clone(tree)

        for i, node in enumerate(tree_clone):
            if isinstance(node, gp.Terminal) and node.name[:3] != "ARG":
                if node.name == special_term_name:
                    new_node_name = f"{special_term_name}[{const_idx}]"
                    tree_clone[i] = gp.Terminal(new_node_name, True, float)
                    const_idx += 1

        individual = toolbox.compile(expr=tree_clone, extra_args=[special_term_name])
        compiled_batch.append(individual)
        const_counts.append(const_idx)

    return compiled_batch, const_counts


def check_trig_fn(ind):
    """
    Counts the occurrences of trigonometric functions (sin, cos) in a symbolic expression.

    Parameters:
        ind (str): A symbolic expression in string format.

    Returns:
        int: The count of trigonometric functions present in the expression.
    """
    return len(re.findall("cos", str(ind))) + len(re.findall("sin", str(ind)))


def check_nested_trig_fn(ind):
    """
    Detects nested trigonometric functions in a symbolic expression.

    Parameters:
        ind (str): A symbolic expression in string format.

    Returns:
        int: The count of nested trigonometric functions detected.
    """
    return util.detect_nested_trigonometric_functions(str(ind))


def get_features_batch(
    individuals_str_batch,
    individ_feature_extractors=[len, check_nested_trig_fn, check_trig_fn],
):
    """
    Extracts multiple features from a batch of symbolic expressions.

    Parameters:
        individuals_str_batch (list of str): A list of symbolic expressions.
        individ_feature_extractors (list of functions, optional): A list of
        functions to extract features.
            Defaults to [len, check_nested_trig_fn, check_trig_fn].

    Returns:
        tuple: Three lists containing:
            - individ_length (list): Length of each expression.
            - nested_trigs (list): Number of nested trigonometric functions in each expression.
            - num_trigs (list): Total number of trigonometric functions in each expression.
    """

    features_batch = [
        [fe(i) for i in individuals_str_batch] for fe in individ_feature_extractors
    ]

    individ_length = features_batch[0]
    nested_trigs = features_batch[1]
    num_trigs = features_batch[2]
    return individ_length, nested_trigs, num_trigs


def read_expressionCoef(file_path):
    """
    Reads a symbolic expression from the second line of a text file.

    Parameters:
        file_path (str): The path to the file.

    Returns:
        str or None: The expression (if found), otherwise None.
    """

    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
            if len(lines) > 5:
                expression = lines[1].strip()  # Best Expression
                BestFitness = lines[3].strip()  # Best Fitness
                BestParameters = "".join(lines[5:]).strip()  # Best Parameters (
                # could be multiple lines)
                BestParameters = np.fromstring(BestParameters.strip("[]"), sep=" ")
                return expression, BestFitness, BestParameters
            else:
                print("File does not contain enough lines.")
                return None, None, None
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None, None, None


def substitute_paramsCoef(expr, params):
    a_matches = list(re.finditer(r"\ba\b", expr))  # matches 'a' as a whole word
    if len(a_matches) > len(params):
        raise ValueError("More 'a' tokens in expression than provided parameters.")

    # Replace each match with the corresponding parameter, in reverse order to avoid index shift
    for i, match in enumerate(reversed(a_matches)):
        start, end = match.span()
        expr = expr[:start] + str(params[-(i + 1)]) + expr[end:]
    return expr


def DeapSimplifier(ind, scales=None, should_print=True):
    """
    Processes and simplifies a symbolic expression using custom operators.

    Parameters:
        ind (str): A string representing the symbolic expression.
        should_print (bool): If True, prints the original and simplified expressions.

    Returns:
        sympy expression: A simplified version of the input expression.
    """
    locals = {
        "sub": lambda x, y: x - y,
        "div": lambda x, y: x / y,
        "aq": lambda x, y: x / (1 + y**2) ** 0.5,
        "mul": lambda x, y: x * y,
        "add": lambda x, y: x + y,
        "neg": lambda x: -x,
        "pow": lambda x, y: x**y,
    }

    if should_print:
        print(f"Original Expression: {ind}")

    expr = sympify(str(ind), locals=locals)

    if should_print:
        print(f"Simplified Expression: {expr}")

    # Optional normalization: x1..xN -> (x1/s1)..(xN/sN)
    if scales is not None:
        n = len(scales)
        x_syms = symbols(f"x1:{n+1}")  # (x1, x2, ..., xN)
        subs_map = {x_syms[i]: x_syms[i] / float(scales[i]) for i in range(n)}
        expr = expr.subs(subs_map)
        if should_print:
            print(f"Simplified ReScaled Expression: {expr}")

    return expr


def detect_nested_function_calls(expr: str, fn_name: str):
    """
    Returns 1 if expr contains nested fn_name(...) inside fn_name(...), else 0.

    Pure string-based scan. Assumes calls look like: fn_name( ... )
    """
    expr = expr.replace(" ", "")
    n = len(expr)
    fn = fn_name.lower()
    fn_len = len(fn)

    nested = 0
    # depth inside current fn call's parentheses (counts parentheses)
    depth = 0
    i = 0

    while i < n and not nested:
        # detect function name at position i
        if expr[i:i+fn_len].lower() == fn:
            # must be a call: next non-space char should be '('
            j = i + fn_len
            if j < n and expr[j] == "(":
                # if we are already inside a call of the same function => nested
                if depth > 0:
                    nested = 1
                    break
                depth = 1  # enter first level of this fn's parentheses
                i = j  # move to '('
        elif expr[i] == "(" and depth > 0:
            depth += 1
        elif expr[i] == ")" and depth > 0:
            depth -= 1

        i += 1

    return nested


def contains_symbol(expr: str, sym: str) -> bool:
    # matches whole identifier: x1 but not x10 or ax1
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(sym)}(?![A-Za-z0-9_])", expr) is not None
