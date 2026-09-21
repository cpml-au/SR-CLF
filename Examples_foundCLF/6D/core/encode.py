"""GP expression -> postfix bytecode (verbatim from src4DCartPoleV8/grid_fitness.py, with the
state terminals generalised to x1..xN).  Same opcodes, same limits, same validation."""
from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

MAX_PROGRAM_NODES = 400
MAX_CONSTANTS = 400
MAX_STACK_DEPTH = 32

PUSH_X = 0
PUSH_PARAMETER = 1
PUSH_LITERAL = 2
ADD = 3
SUB = 4
MUL = 5
AQ = 6
NEG = 7
SIN = 8
EXP = 9

_FUNCTIONS = {
    "add": (ADD, 2),
    "sub": (SUB, 2),
    "mul": (MUL, 2),
    "aq": (AQ, 2),
    "neg": (NEG, 1),
    "sin": (SIN, 1),
    "exp": (EXP, 1),
}


@dataclass(frozen=True)
class EncodedProgram:
    opcodes: np.ndarray
    operands: np.ndarray
    literals: np.ndarray
    n_ops: int
    n_constants: int


@lru_cache(maxsize=16384)
def encode_expression(expression: str, n_states: int = 6) -> EncodedProgram:
    """Convert a DEAP prefix-call string to a padded postfix program."""
    root = ast.parse(str(expression), mode="eval").body
    instructions = []
    constant_index = 0
    names = {f"x{i + 1}" for i in range(int(n_states))}

    def visit(node):
        nonlocal constant_index
        if isinstance(node, ast.Name):
            if node.id in names:
                instructions.append((PUSH_X, int(node.id[1:]) - 1, 0.0))
                return
            if node.id == "a":
                instructions.append((PUSH_PARAMETER, constant_index, 0.0))
                constant_index += 1
                return
            raise ValueError(f"Unsupported terminal: {node.id}")
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            instructions.append((PUSH_LITERAL, 0, float(node.value)))
            return
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            visit(node.operand)
            instructions.append((NEG, 0, 0.0))
            return
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            function = node.func.id
            if function not in _FUNCTIONS:
                raise ValueError(f"Unsupported primitive: {function}")
            opcode, arity = _FUNCTIONS[function]
            if len(node.args) != arity:
                raise ValueError(f"{function} expects {arity} argument(s)")
            for argument in node.args:
                visit(argument)
            instructions.append((opcode, 0, 0.0))
            return
        raise ValueError(f"Unsupported syntax: {ast.dump(node)}")

    visit(root)
    if len(instructions) > MAX_PROGRAM_NODES:
        raise ValueError(f"program has {len(instructions)} nodes; limit is {MAX_PROGRAM_NODES}")
    if constant_index > MAX_CONSTANTS:
        raise ValueError("expression has too many tunable constants")

    depth = 0
    max_depth = 0
    for opcode, _, _ in instructions:
        if opcode in {PUSH_X, PUSH_PARAMETER, PUSH_LITERAL}:
            depth += 1
        elif opcode in {ADD, SUB, MUL, AQ}:
            depth -= 1
        if depth < 1:
            raise ValueError("Malformed postfix program")
        max_depth = max(max_depth, depth)
    if depth != 1:
        raise ValueError("Malformed expression stack")
    if max_depth > MAX_STACK_DEPTH:
        raise ValueError(f"expression needs stack depth {max_depth}; limit is {MAX_STACK_DEPTH}")

    opcodes = np.zeros(MAX_PROGRAM_NODES, dtype=np.int32)
    operands = np.zeros(MAX_PROGRAM_NODES, dtype=np.int32)
    literals = np.zeros(MAX_PROGRAM_NODES, dtype=np.float64)
    for index, (opcode, operand, literal) in enumerate(instructions):
        opcodes[index] = opcode
        operands[index] = operand
        literals[index] = literal
    return EncodedProgram(
        opcodes=opcodes,
        operands=operands,
        literals=literals,
        n_ops=len(instructions),
        n_constants=constant_index,
    )


def program_arrays(expression, constants, n_states):
    """(opcodes, operands, literals, n_ops, parameters) for the numba kernels."""
    program = encode_expression(str(expression), int(n_states))
    params = np.zeros(MAX_CONSTANTS)
    cvals = np.asarray(constants, dtype=float).reshape(-1)
    if cvals.size != program.n_constants:
        raise ValueError(f"expected {program.n_constants} constants, got {cvals.size}")
    params[: cvals.size] = cvals
    return (
        np.asarray(program.opcodes, dtype=np.int64),
        np.asarray(program.operands, dtype=np.int64),
        np.asarray(program.literals, dtype=np.float64),
        int(program.n_ops),
        params,
    )
