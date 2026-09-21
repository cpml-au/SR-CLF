"""Runtime program encoder for the vendored GPU engine (the encoder subset of
srcGPU5_9/grid_fitness.py; the GP grid-fitness kernels and the tuner are not vendored).

GP trees are encoded as fixed-width postfix instructions: the Pallas kernels have one shape
for every tree, so structural mutation never triggers a new XLA compile.  The grammar is
{add, sub, mul, aq, neg, sin, exp} over x1..x4 and the constant placeholder ``a``.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

MAX_PROGRAM_NODES = 400
MAX_CONSTANTS = 400
# Flex applies a static height limit of 17 to crossover and mutation. A
# postfix evaluator needs at most height+1 live values, so 32 leaves ample
# headroom without allocating a 400-row stack over the entire 21^4 grid.
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
def encode_expression(expression: str) -> EncodedProgram:
    """Convert a DEAP prefix-call string to a padded postfix program."""
    root = ast.parse(str(expression), mode="eval").body
    instructions = []
    constant_index = 0

    def visit(node):
        nonlocal constant_index
        if isinstance(node, ast.Name):
            if node.id in {"x1", "x2", "x3", "x4"}:
                instructions.append((PUSH_X, int(node.id[1:]) - 1, 0.0))
                return
            if node.id == "a":
                instructions.append((PUSH_PARAMETER, constant_index, 0.0))
                constant_index += 1
                return
            raise ValueError(f"Unsupported GPU2 terminal: {node.id}")
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
                raise ValueError(f"Unsupported GPU2 primitive: {function}")
            opcode, arity = _FUNCTIONS[function]
            if len(node.args) != arity:
                raise ValueError(f"{function} expects {arity} argument(s)")
            for argument in node.args:
                visit(argument)
            instructions.append((opcode, 0, 0.0))
            return
        raise ValueError(f"Unsupported GPU2 syntax: {ast.dump(node)}")

    visit(root)
    if len(instructions) > MAX_PROGRAM_NODES:
        raise ValueError(
            f"GPU2 program has {len(instructions)} nodes; limit is {MAX_PROGRAM_NODES}"
        )
    if constant_index > MAX_CONSTANTS:
        raise ValueError("GPU2 expression has too many tunable constants")

    depth = 0
    max_depth = 0
    for opcode, _, _ in instructions:
        if opcode in {PUSH_X, PUSH_PARAMETER, PUSH_LITERAL}:
            depth += 1
        elif opcode in {ADD, SUB, MUL, AQ}:
            depth -= 1
        if depth < 1:
            raise ValueError("Malformed GPU2 postfix program")
        max_depth = max(max_depth, depth)
    if depth != 1:
        raise ValueError("Malformed GPU2 expression stack")
    if max_depth > MAX_STACK_DEPTH:
        raise ValueError(
            f"GPU2 expression needs stack depth {max_depth}; "
            f"limit is {MAX_STACK_DEPTH}"
        )

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
