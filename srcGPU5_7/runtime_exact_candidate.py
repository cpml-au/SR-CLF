"""Compile-once runtime derivatives for GPU2 exact manifold verification.

The GP tree is runtime opcode data, not Python/JAX source. Consequently a new
tree does not create a new XLA program. The large b-field and every bisection
use a float64 Pallas forward-derivative interpreter. The small SQP population
uses a fixed-shape second-order forward interpreter to obtain exact a, b,
grad(a), and grad(b) without finite differences or per-tree autodiff/JIT.

This module is deliberately specific to the four-state cart-pole GPU2 case.
It preserves the same analytic dynamics and the same SQP/bisection algorithms
as :mod:`srcGPU5_7.jax_candidate`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os

import srcGPU5_7  # noqa: F401  (configure float64 before importing JAX)

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

from srcGPU5_7.grid_fitness import (
    ADD,
    AQ,
    EXP,
    MAX_CONSTANTS,
    MAX_PROGRAM_NODES,
    MAX_STACK_DEPTH,
    MUL,
    NEG,
    PUSH_LITERAL,
    PUSH_PARAMETER,
    PUSH_X,
    SIN,
    SUB,
    encode_expression,
)
from srcGPU5_7.pallas_interpreter import DEFAULT_BLOCK_SIZE


@dataclass(frozen=True)
class RuntimeExactCandidate:
    """Prefix GP program and its fitted constants for exact verification."""

    expression: str
    constants: tuple[float, ...]


def _cartpole_fields(x):
    """Return the drift and active G[:, 1] column used by Evaluate.py."""
    x1, x2, x3, x4 = x
    del x1
    sin_theta = -jnp.sin(x3)
    cos_theta = -jnp.cos(x3)
    denominator = 4.0 * (6.0 - cos_theta * cos_theta)
    common = 2.0 * x4 * x4 * sin_theta - x2
    drift = jnp.stack(
        [
            x2,
            (40.0 * cos_theta * sin_theta + 4.0 * common) / denominator,
            x4,
            (-120.0 * sin_theta - 2.0 * cos_theta * common) / denominator,
        ]
    )
    control = jnp.stack(
        [
            jnp.asarray(0.0, x.dtype),
            4.0 / denominator,
            jnp.asarray(0.0, x.dtype),
            -2.0 * cos_theta / denominator,
        ]
    )
    return drift, control


@lru_cache(maxsize=12)
def _b_interpreter(
    padded_point_count: int,
    block_size: int,
    interpret: bool,
):
    """Pallas dual-number interpreter for b = directional_G(V).

    Only value and one exact directional derivative are needed during root
    scanning/bisection. The caller uses fixed point chunks so scratch remains
    far below Triton's 32-bit byte-offset boundary.
    """
    if padded_point_count % block_size:
        raise ValueError("runtime exact point count must be block aligned")

    def kernel(
        opcodes_ref,
        operands_ref,
        literals_ref,
        n_ops_ref,
        parameters_ref,
        points_ref,
        output_ref,
        stack_ref,
    ):
        lane = jnp.arange(block_size, dtype=jnp.int32)
        x3 = pl.load(points_ref, (lane, 2))
        cos_theta = -jnp.cos(x3)
        denominator = 4.0 * (6.0 - cos_theta * cos_theta)
        control_x2 = 4.0 / denominator
        control_x4 = -2.0 * cos_theta / denominator

        def store_item(pointer, value, derivative):
            pl.store(stack_ref, (pointer, 0, lane), value)
            pl.store(stack_ref, (pointer, 1, lane), derivative)

        def load_item(pointer):
            return (
                pl.load(stack_ref, (pointer, 0, lane)),
                pl.load(stack_ref, (pointer, 1, lane)),
            )

        zeros = jnp.zeros((block_size,), dtype=jnp.float64)

        def push_x(args):
            pointer, operand, _ = args
            value = pl.load(points_ref, (lane, operand))
            derivative = jnp.where(
                operand == 1,
                control_x2,
                jnp.where(
                    operand == 3,
                    control_x4,
                    zeros,
                ),
            )
            store_item(pointer, value, derivative)
            return pointer + 1

        def push_parameter(args):
            pointer, operand, _ = args
            value = jnp.broadcast_to(
                pl.load(parameters_ref, (operand,)), (block_size,)
            )
            store_item(pointer, value, zeros)
            return pointer + 1

        def push_literal(args):
            pointer, _, literal = args
            store_item(
                pointer, jnp.broadcast_to(literal, (block_size,)), zeros
            )
            return pointer + 1

        def add(args):
            pointer, _, _ = args
            left, dl = load_item(pointer - 2)
            right, dr = load_item(pointer - 1)
            store_item(pointer - 2, left + right, dl + dr)
            return pointer - 1

        def sub(args):
            pointer, _, _ = args
            left, dl = load_item(pointer - 2)
            right, dr = load_item(pointer - 1)
            store_item(pointer - 2, left - right, dl - dr)
            return pointer - 1

        def mul(args):
            pointer, _, _ = args
            left, dl = load_item(pointer - 2)
            right, dr = load_item(pointer - 1)
            store_item(pointer - 2, left * right, dl * right + dr * left)
            return pointer - 1

        def aq(args):
            pointer, _, _ = args
            left, dl = load_item(pointer - 2)
            right, dr = load_item(pointer - 1)
            scale = jax.lax.rsqrt(1.0 + right * right)
            scale_prime = -right * scale * scale * scale
            store_item(
                pointer - 2,
                left * scale,
                dl * scale + dr * left * scale_prime,
            )
            return pointer - 1

        def neg(args):
            pointer, _, _ = args
            value, derivative = load_item(pointer - 1)
            store_item(pointer - 1, -value, -derivative)
            return pointer

        def sin(args):
            pointer, _, _ = args
            value, derivative = load_item(pointer - 1)
            store_item(
                pointer - 1, jnp.sin(value), derivative * jnp.cos(value)
            )
            return pointer

        def exp(args):
            pointer, _, _ = args
            value, derivative = load_item(pointer - 1)
            exponential = jnp.exp(value)
            store_item(pointer - 1, exponential, derivative * exponential)
            return pointer

        branches = (
            push_x,
            push_parameter,
            push_literal,
            add,
            sub,
            mul,
            aq,
            neg,
            sin,
            exp,
        )

        def body(index, pointer):
            return jax.lax.switch(
                pl.load(opcodes_ref, (index,)),
                branches,
                (
                    pointer,
                    pl.load(operands_ref, (index,)),
                    pl.load(literals_ref, (index,)),
                ),
            )

        pointer = jax.lax.fori_loop(
            0, pl.load(n_ops_ref, ()), body, jnp.int32(0)
        )
        _, derivative = load_item(pointer - 1)
        pl.store(output_ref, (lane,), derivative)

    call = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((padded_point_count,), jnp.float64),
            jax.ShapeDtypeStruct(
                (MAX_STACK_DEPTH, 2, padded_point_count), jnp.float64
            ),
        ),
        grid=(padded_point_count // block_size,),
        in_specs=(
            pl.BlockSpec((MAX_PROGRAM_NODES,), lambda _: (0,)),
            pl.BlockSpec((MAX_PROGRAM_NODES,), lambda _: (0,)),
            pl.BlockSpec((MAX_PROGRAM_NODES,), lambda _: (0,)),
            pl.BlockSpec((), lambda _: ()),
            pl.BlockSpec((MAX_CONSTANTS,), lambda _: (0,)),
            pl.BlockSpec((block_size, 4), lambda block: (block, 0)),
        ),
        out_specs=(
            pl.BlockSpec((block_size,), lambda block: (block,)),
            pl.BlockSpec(
                (MAX_STACK_DEPTH, 2, block_size),
                lambda block: (0, 0, block),
            ),
        ),
        compiler_params=plgpu.TritonCompilerParams(num_warps=4, num_stages=1),
        interpret=interpret,
        name="gpu2_runtime_exact_b",
    )

    @jax.jit
    def evaluate(opcodes, operands, literals, n_ops, parameters, points):
        values, _ = call(
            opcodes, operands, literals, n_ops, parameters, points
        )
        return values

    return evaluate


@lru_cache(maxsize=12)
def _b_many_interpreter(
    candidate_count: int,
    padded_point_count: int,
    block_size: int,
    interpret: bool,
):
    """Vectorize the fixed-shape Pallas b interpreter across programs."""
    evaluate_one = _b_interpreter(
        padded_point_count, block_size, interpret
    )

    @jax.jit
    def evaluate(opcodes, operands, literals, n_ops, parameters, points):
        return jax.vmap(
            lambda op, arg, lit, count, params: evaluate_one(
                op, arg, lit, count, params, points
            )
        )(opcodes, operands, literals, n_ops, parameters)

    return evaluate


@lru_cache(maxsize=12)
def _ab_interpreter(
    padded_point_count: int,
    block_size: int,
    interpret: bool,
):
    """Pallas value+gradient interpreter returning analytic a and b."""
    if padded_point_count % block_size:
        raise ValueError("runtime exact point count must be block aligned")

    def kernel(
        opcodes_ref,
        operands_ref,
        literals_ref,
        n_ops_ref,
        parameters_ref,
        points_ref,
        output_ref,
        stack_ref,
    ):
        lane = jnp.arange(block_size, dtype=jnp.int32)

        def store_item(pointer, value, gradient):
            pl.store(stack_ref, (pointer, 0, lane), value)
            for dimension in range(4):
                pl.store(
                    stack_ref,
                    (pointer, dimension + 1, lane),
                    gradient[dimension],
                )

        def load_item(pointer):
            value = pl.load(stack_ref, (pointer, 0, lane))
            # Keep the four gradient components as independent lane vectors.
            # A stacked [4, block] value lowers to ``concatenate`` under the
            # vmapped Pallas kernel, which Triton does not implement.
            gradient = tuple(
                pl.load(stack_ref, (pointer, dimension + 1, lane))
                for dimension in range(4)
            )
            return value, gradient

        zero_lanes = jnp.zeros((block_size,), dtype=jnp.float64)
        zeros = (zero_lanes, zero_lanes, zero_lanes, zero_lanes)

        def push_x(args):
            pointer, operand, _ = args
            value = pl.load(points_ref, (lane, operand))
            gradient = tuple(
                jnp.where(
                    operand == dimension,
                    jnp.ones((block_size,), dtype=jnp.float64),
                    zero_lanes,
                )
                for dimension in range(4)
            )
            store_item(pointer, value, gradient)
            return pointer + 1

        def push_parameter(args):
            pointer, operand, _ = args
            value = jnp.broadcast_to(
                pl.load(parameters_ref, (operand,)), (block_size,)
            )
            store_item(pointer, value, zeros)
            return pointer + 1

        def push_literal(args):
            pointer, _, literal = args
            store_item(
                pointer, jnp.broadcast_to(literal, (block_size,)), zeros
            )
            return pointer + 1

        def add(args):
            pointer, _, _ = args
            left, gl = load_item(pointer - 2)
            right, gr = load_item(pointer - 1)
            store_item(
                pointer - 2,
                left + right,
                tuple(gl[dimension] + gr[dimension] for dimension in range(4)),
            )
            return pointer - 1

        def sub(args):
            pointer, _, _ = args
            left, gl = load_item(pointer - 2)
            right, gr = load_item(pointer - 1)
            store_item(
                pointer - 2,
                left - right,
                tuple(gl[dimension] - gr[dimension] for dimension in range(4)),
            )
            return pointer - 1

        def mul(args):
            pointer, _, _ = args
            left, gl = load_item(pointer - 2)
            right, gr = load_item(pointer - 1)
            store_item(
                pointer - 2,
                left * right,
                tuple(
                    gl[dimension] * right + gr[dimension] * left
                    for dimension in range(4)
                ),
            )
            return pointer - 1

        def aq(args):
            pointer, _, _ = args
            left, gl = load_item(pointer - 2)
            right, gr = load_item(pointer - 1)
            scale = jax.lax.rsqrt(1.0 + right * right)
            scale_prime = -right * scale * scale * scale
            store_item(
                pointer - 2,
                left * scale,
                tuple(
                    gl[dimension] * scale
                    + gr[dimension] * (left * scale_prime)
                    for dimension in range(4)
                ),
            )
            return pointer - 1

        def neg(args):
            pointer, _, _ = args
            value, gradient = load_item(pointer - 1)
            store_item(
                pointer - 1,
                -value,
                tuple(-gradient[dimension] for dimension in range(4)),
            )
            return pointer

        def sin(args):
            pointer, _, _ = args
            value, gradient = load_item(pointer - 1)
            store_item(
                pointer - 1,
                jnp.sin(value),
                tuple(
                    gradient[dimension] * jnp.cos(value)
                    for dimension in range(4)
                ),
            )
            return pointer

        def exp(args):
            pointer, _, _ = args
            value, gradient = load_item(pointer - 1)
            exponential = jnp.exp(value)
            store_item(
                pointer - 1,
                exponential,
                tuple(
                    gradient[dimension] * exponential
                    for dimension in range(4)
                ),
            )
            return pointer

        branches = (
            push_x,
            push_parameter,
            push_literal,
            add,
            sub,
            mul,
            aq,
            neg,
            sin,
            exp,
        )

        def body(index, pointer):
            return jax.lax.switch(
                pl.load(opcodes_ref, (index,)),
                branches,
                (
                    pointer,
                    pl.load(operands_ref, (index,)),
                    pl.load(literals_ref, (index,)),
                ),
            )

        pointer = jax.lax.fori_loop(
            0, pl.load(n_ops_ref, ()), body, jnp.int32(0)
        )
        _, gradient = load_item(pointer - 1)
        x2 = pl.load(points_ref, (lane, 1))
        x3 = pl.load(points_ref, (lane, 2))
        x4 = pl.load(points_ref, (lane, 3))
        sin_theta = -jnp.sin(x3)
        cos_theta = -jnp.cos(x3)
        denominator = 4.0 * (6.0 - cos_theta * cos_theta)
        common = 2.0 * x4 * x4 * sin_theta - x2
        f2 = (40.0 * cos_theta * sin_theta + 4.0 * common) / denominator
        f4 = (-120.0 * sin_theta - 2.0 * cos_theta * common) / denominator
        a_value = (
            gradient[0] * x2
            + gradient[1] * f2
            + gradient[2] * x4
            + gradient[3] * f4
        )
        b_value = (
            gradient[1] * (4.0 / denominator)
            + gradient[3] * (-2.0 * cos_theta / denominator)
        )
        pl.store(output_ref, (0, lane), a_value)
        pl.store(output_ref, (1, lane), b_value)

    call = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((2, padded_point_count), jnp.float64),
            jax.ShapeDtypeStruct(
                (MAX_STACK_DEPTH, 5, padded_point_count), jnp.float64
            ),
        ),
        grid=(padded_point_count // block_size,),
        in_specs=(
            pl.BlockSpec((MAX_PROGRAM_NODES,), lambda _: (0,)),
            pl.BlockSpec((MAX_PROGRAM_NODES,), lambda _: (0,)),
            pl.BlockSpec((MAX_PROGRAM_NODES,), lambda _: (0,)),
            pl.BlockSpec((), lambda _: ()),
            pl.BlockSpec((MAX_CONSTANTS,), lambda _: (0,)),
            pl.BlockSpec((block_size, 4), lambda block: (block, 0)),
        ),
        out_specs=(
            pl.BlockSpec((2, block_size), lambda block: (0, block)),
            pl.BlockSpec(
                (MAX_STACK_DEPTH, 5, block_size),
                lambda block: (0, 0, block),
            ),
        ),
        compiler_params=plgpu.TritonCompilerParams(num_warps=4, num_stages=1),
        interpret=interpret,
        name="gpu2_runtime_exact_ab",
    )

    @jax.jit
    def evaluate(opcodes, operands, literals, n_ops, parameters, points):
        values, _ = call(
            opcodes, operands, literals, n_ops, parameters, points
        )
        return values

    return evaluate


@lru_cache(maxsize=12)
def _ab_many_interpreter(
    candidate_count: int,
    padded_point_count: int,
    block_size: int,
    interpret: bool,
):
    """Vectorize exact a,b evaluation across fixed-width programs."""
    evaluate_one = _ab_interpreter(
        padded_point_count, block_size, interpret
    )

    @jax.jit
    def evaluate(opcodes, operands, literals, n_ops, parameters, points):
        return jax.vmap(evaluate_one)(
            opcodes, operands, literals, n_ops, parameters, points
        )

    return evaluate


def _set_dual(state, index, value, gradient, hessian, pointer):
    _, values, gradients, hessians = state
    return (
        pointer,
        values.at[index].set(value),
        gradients.at[index].set(gradient),
        hessians.at[index].set(hessian),
    )


def _dual2_point(opcodes, operands, literals, n_ops, parameters, x):
    """Forward value/gradient/Hessian interpreter for one point."""
    dtype = x.dtype
    zero_g = jnp.zeros((4,), dtype=dtype)
    zero_h = jnp.zeros((4, 4), dtype=dtype)
    identity = jnp.eye(4, dtype=dtype)
    initial = (
        jnp.int32(0),
        jnp.zeros((MAX_STACK_DEPTH,), dtype=dtype),
        jnp.zeros((MAX_STACK_DEPTH, 4), dtype=dtype),
        jnp.zeros((MAX_STACK_DEPTH, 4, 4), dtype=dtype),
    )

    def push_x(state, operand, _):
        pointer = state[0]
        return _set_dual(
            state, pointer, x[operand], identity[operand], zero_h, pointer + 1
        )

    def push_parameter(state, operand, _):
        pointer = state[0]
        return _set_dual(
            state, pointer, parameters[operand], zero_g, zero_h, pointer + 1
        )

    def push_literal(state, _, literal):
        pointer = state[0]
        return _set_dual(
            state, pointer, literal, zero_g, zero_h, pointer + 1
        )

    def add(state, _, __):
        pointer, values, gradients, hessians = state
        left, right = pointer - 2, pointer - 1
        return _set_dual(
            state,
            left,
            values[left] + values[right],
            gradients[left] + gradients[right],
            hessians[left] + hessians[right],
            pointer - 1,
        )

    def sub(state, _, __):
        pointer, values, gradients, hessians = state
        left, right = pointer - 2, pointer - 1
        return _set_dual(
            state,
            left,
            values[left] - values[right],
            gradients[left] - gradients[right],
            hessians[left] - hessians[right],
            pointer - 1,
        )

    def product(state, left, right, value_right, grad_right, hess_right):
        pointer, values, gradients, hessians = state
        value_left = values[left]
        grad_left = gradients[left]
        hess_left = hessians[left]
        value = value_left * value_right
        gradient = grad_left * value_right + grad_right * value_left
        hessian = (
            hess_left * value_right
            + hess_right * value_left
            + jnp.outer(grad_left, grad_right)
            + jnp.outer(grad_right, grad_left)
        )
        return _set_dual(
            state, left, value, gradient, hessian, pointer - 1
        )

    def mul(state, _, __):
        pointer, values, gradients, hessians = state
        right = pointer - 1
        return product(
            state,
            pointer - 2,
            right,
            values[right],
            gradients[right],
            hessians[right],
        )

    def aq(state, _, __):
        pointer, values, gradients, hessians = state
        left, right = pointer - 2, pointer - 1
        y = values[right]
        scale = jax.lax.rsqrt(1.0 + y * y)
        scale_prime = -y * scale**3
        scale_second = (2.0 * y * y - 1.0) * scale**5
        grad_scale = scale_prime * gradients[right]
        hess_scale = (
            scale_prime * hessians[right]
            + scale_second * jnp.outer(gradients[right], gradients[right])
        )
        return product(
            state, left, right, scale, grad_scale, hess_scale
        )

    def neg(state, _, __):
        pointer, values, gradients, hessians = state
        index = pointer - 1
        return _set_dual(
            state,
            index,
            -values[index],
            -gradients[index],
            -hessians[index],
            pointer,
        )

    def unary(state, first, second):
        pointer, values, gradients, hessians = state
        index = pointer - 1
        gradient = first * gradients[index]
        hessian = (
            first * hessians[index]
            + second * jnp.outer(gradients[index], gradients[index])
        )
        return gradient, hessian

    def sin(state, _, __):
        pointer, values, gradients, hessians = state
        index = pointer - 1
        value = values[index]
        gradient, hessian = unary(state, jnp.cos(value), -jnp.sin(value))
        return _set_dual(
            state, index, jnp.sin(value), gradient, hessian, pointer
        )

    def exp(state, _, __):
        pointer, values, gradients, hessians = state
        index = pointer - 1
        value = jnp.exp(values[index])
        gradient, hessian = unary(state, value, value)
        return _set_dual(state, index, value, gradient, hessian, pointer)

    branches = (
        push_x,
        push_parameter,
        push_literal,
        add,
        sub,
        mul,
        aq,
        neg,
        sin,
        exp,
    )

    def body(index, state):
        return jax.lax.switch(
            opcodes[index],
            branches,
            state,
            operands[index],
            literals[index],
        )

    pointer, values, gradients, hessians = jax.lax.fori_loop(
        0, n_ops, body, initial
    )
    index = pointer - 1
    return values[index], gradients[index], hessians[index]


def _set_hvp_dual(state, index, value, gradient, hvps, pointer):
    _, values, gradients, stack_hvps = state
    return (
        pointer,
        values.at[index].set(value),
        gradients.at[index].set(gradient),
        stack_hvps.at[index].set(hvps),
    )


def _dual_hvp_point(
    opcodes, operands, literals, n_ops, parameters, x, directions
):
    """Exact value/gradient and H(V)@directions without forming H(V).

    The Artstein polish only needs H(V)f and H(V)G. Carrying those two
    Hessian-vector products uses 13 float64 channels per live stack entry
    instead of the 21 channels required by the complete 4x4 Hessian.
    """
    dtype = x.dtype
    zero_g = jnp.zeros((4,), dtype=dtype)
    zero_hvps = jnp.zeros((2, 4), dtype=dtype)
    identity = jnp.eye(4, dtype=dtype)
    initial = (
        jnp.int32(0),
        jnp.zeros((MAX_STACK_DEPTH,), dtype=dtype),
        jnp.zeros((MAX_STACK_DEPTH, 4), dtype=dtype),
        jnp.zeros((MAX_STACK_DEPTH, 2, 4), dtype=dtype),
    )

    def push_x(state, operand, _):
        pointer = state[0]
        return _set_hvp_dual(
            state,
            pointer,
            x[operand],
            identity[operand],
            zero_hvps,
            pointer + 1,
        )

    def push_parameter(state, operand, _):
        pointer = state[0]
        return _set_hvp_dual(
            state,
            pointer,
            parameters[operand],
            zero_g,
            zero_hvps,
            pointer + 1,
        )

    def push_literal(state, _, literal):
        pointer = state[0]
        return _set_hvp_dual(
            state, pointer, literal, zero_g, zero_hvps, pointer + 1
        )

    def add(state, _, __):
        pointer, values, gradients, hvps = state
        left, right = pointer - 2, pointer - 1
        return _set_hvp_dual(
            state,
            left,
            values[left] + values[right],
            gradients[left] + gradients[right],
            hvps[left] + hvps[right],
            pointer - 1,
        )

    def sub(state, _, __):
        pointer, values, gradients, hvps = state
        left, right = pointer - 2, pointer - 1
        return _set_hvp_dual(
            state,
            left,
            values[left] - values[right],
            gradients[left] - gradients[right],
            hvps[left] - hvps[right],
            pointer - 1,
        )

    def product(state, left, value_right, grad_right, hvps_right):
        pointer, values, gradients, hvps = state
        value_left = values[left]
        grad_left = gradients[left]
        hvps_left = hvps[left]
        left_directional = directions @ grad_left
        right_directional = directions @ grad_right
        value = value_left * value_right
        gradient = grad_left * value_right + grad_right * value_left
        product_hvps = (
            hvps_left * value_right
            + hvps_right * value_left
            + grad_left[None, :] * right_directional[:, None]
            + grad_right[None, :] * left_directional[:, None]
        )
        return _set_hvp_dual(
            state, left, value, gradient, product_hvps, pointer - 1
        )

    def mul(state, _, __):
        pointer, values, gradients, hvps = state
        right = pointer - 1
        return product(
            state,
            pointer - 2,
            values[right],
            gradients[right],
            hvps[right],
        )

    def aq(state, _, __):
        pointer, values, gradients, hvps = state
        left, right = pointer - 2, pointer - 1
        y = values[right]
        scale = jax.lax.rsqrt(1.0 + y * y)
        scale_prime = -y * scale**3
        scale_second = (2.0 * y * y - 1.0) * scale**5
        grad_right = gradients[right]
        grad_scale = scale_prime * grad_right
        directional_right = directions @ grad_right
        hvps_scale = (
            scale_prime * hvps[right]
            + scale_second
            * directional_right[:, None]
            * grad_right[None, :]
        )
        return product(state, left, scale, grad_scale, hvps_scale)

    def neg(state, _, __):
        pointer, values, gradients, hvps = state
        index = pointer - 1
        return _set_hvp_dual(
            state,
            index,
            -values[index],
            -gradients[index],
            -hvps[index],
            pointer,
        )

    def unary(state, first, second):
        pointer, _, gradients, hvps = state
        index = pointer - 1
        gradient = gradients[index]
        directional = directions @ gradient
        result_gradient = first * gradient
        result_hvps = (
            first * hvps[index]
            + second * directional[:, None] * gradient[None, :]
        )
        return result_gradient, result_hvps

    def sin(state, _, __):
        pointer, values, _, _ = state
        index = pointer - 1
        value = values[index]
        gradient, hvps = unary(state, jnp.cos(value), -jnp.sin(value))
        return _set_hvp_dual(
            state, index, jnp.sin(value), gradient, hvps, pointer
        )

    def exp(state, _, __):
        pointer, values, _, _ = state
        index = pointer - 1
        value = jnp.exp(values[index])
        gradient, hvps = unary(state, value, value)
        return _set_hvp_dual(state, index, value, gradient, hvps, pointer)

    branches = (
        push_x,
        push_parameter,
        push_literal,
        add,
        sub,
        mul,
        aq,
        neg,
        sin,
        exp,
    )

    def body(index, state):
        return jax.lax.switch(
            opcodes[index],
            branches,
            state,
            operands[index],
            literals[index],
        )

    pointer, values, gradients, hvps = jax.lax.fori_loop(
        0, n_ops, body, initial
    )
    index = pointer - 1
    return values[index], gradients[index], hvps[index]


def _abg_one(opcodes, operands, literals, n_ops, parameters, x):
    drift, control = _cartpole_fields(x)
    # The dynamics are fixed, so these two Jacobians are part of one shared
    # compiled graph and never vary with the GP tree.
    jac_drift, jac_control = jax.jacfwd(_cartpole_fields)(x)
    _, gradient, hvps = _dual_hvp_point(
        opcodes,
        operands,
        literals,
        n_ops,
        parameters,
        x,
        jnp.stack([drift, control]),
    )
    a_value = gradient @ drift
    b_value = gradient @ control
    ga = hvps[0] + jac_drift.T @ gradient
    gb = hvps[1] + jac_control.T @ gradient
    return a_value, b_value, ga, gb


@jax.jit
def _abg_batch(opcodes, operands, literals, n_ops, parameters, points):
    return jax.vmap(
        lambda point: _abg_one(
            opcodes, operands, literals, n_ops, parameters, point
        )
    )(points)


@lru_cache(maxsize=12)
def _bisect_axis_kernel(point_count, iterations, block_size, interpret):
    evaluate = _b_interpreter(point_count, block_size, interpret)

    @jax.jit
    def bisect(
        opcodes,
        operands,
        literals,
        n_ops,
        parameters,
        points,
        onehot,
        lo,
        hi,
        b_lo,
    ):
        def condition(state):
            iteration, lower, upper, _ = state
            middle = 0.5 * (lower + upper)
            movable = jnp.any((middle != lower) & (middle != upper))
            return (iteration < iterations) & movable

        def body(state):
            iteration, lower, upper, lower_value = state
            middle = 0.5 * (lower + upper)
            middle_points = (
                points * (1.0 - onehot) + middle[:, None] * onehot
            )
            middle_value = evaluate(
                opcodes,
                operands,
                literals,
                n_ops,
                parameters,
                middle_points,
            )
            left = jnp.sign(middle_value) * jnp.sign(lower_value) > 0
            return (
                iteration + 1,
                jnp.where(left, middle, lower),
                jnp.where(left, upper, middle),
                jnp.where(left, middle_value, lower_value),
            )

        _, lower, upper, _ = jax.lax.while_loop(
            condition, body, (jnp.int32(0), lo, hi, b_lo)
        )
        middle = 0.5 * (lower + upper)
        return points * (1.0 - onehot) + middle[:, None] * onehot

    return bisect


@lru_cache(maxsize=12)
def _bisect_line_kernel(point_count, iterations, block_size, interpret):
    evaluate = _b_interpreter(point_count, block_size, interpret)

    @jax.jit
    def bisect(
        opcodes,
        operands,
        literals,
        n_ops,
        parameters,
        origins,
        directions,
        lo,
        hi,
        b_lo,
    ):
        def condition(state):
            iteration, lower, upper, _ = state
            middle = 0.5 * (lower + upper)
            movable = jnp.any((middle != lower) & (middle != upper))
            return (iteration < iterations) & movable

        def body(state):
            iteration, lower, upper, lower_value = state
            middle = 0.5 * (lower + upper)
            middle_points = origins + middle[:, None] * directions
            middle_value = evaluate(
                opcodes,
                operands,
                literals,
                n_ops,
                parameters,
                middle_points,
            )
            left = jnp.sign(middle_value) * jnp.sign(lower_value) > 0
            return (
                iteration + 1,
                jnp.where(left, middle, lower),
                jnp.where(left, upper, middle),
                jnp.where(left, middle_value, lower_value),
            )

        _, lower, upper, _ = jax.lax.while_loop(
            condition, body, (jnp.int32(0), lo, hi, b_lo)
        )
        return origins + (0.5 * (lower + upper))[:, None] * directions

    return bisect


@lru_cache(maxsize=12)
def _bisect_line_many_kernel(
    candidate_count, point_count, iterations, block_size, interpret
):
    bisect_one = _bisect_line_kernel(
        point_count, iterations, block_size, interpret
    )

    @jax.jit
    def bisect(
        opcodes,
        operands,
        literals,
        n_ops,
        parameters,
        origins,
        directions,
        lo,
        hi,
        b_lo,
    ):
        return jax.vmap(bisect_one)(
            opcodes,
            operands,
            literals,
            n_ops,
            parameters,
            origins,
            directions,
            lo,
            hi,
            b_lo,
        )

    return bisect


@lru_cache(maxsize=8)
def _polish_kernel(iterations, projection_steps, line_search_steps):
    """Unchanged batched SQP algorithm over runtime exact derivatives."""

    @jax.jit
    def polish(
        opcodes,
        operands,
        literals,
        n_ops,
        parameters,
        seeds,
        lower,
        upper,
        r0_sq,
        b_tol,
        step_size,
    ):
        dtype = seeds.dtype
        eps = jnp.asarray(1.0e-24, dtype=dtype)

        def abg_batch(points):
            return _abg_batch(
                opcodes,
                operands,
                literals,
                n_ops,
                parameters,
                points,
            )

        def enforce_constraints(points):
            def project_once(_, current):
                _, b_value, _, gb = abg_batch(current)
                norm_sq = jnp.sum(current * current, axis=1)
                active = norm_sq < r0_sq
                radial = 2.0 * current
                m11 = jnp.sum(gb * gb, axis=1) + eps
                m12 = jnp.sum(gb * radial, axis=1)
                m22 = jnp.sum(radial * radial, axis=1) + eps
                c2 = jnp.where(active, norm_sq - r0_sq, 0.0)
                determinant = m11 * m22 - m12 * m12 + eps
                lambda_both_1 = (
                    b_value * m22 - c2 * m12
                ) / determinant
                lambda_both_2 = (
                    c2 * m11 - b_value * m12
                ) / determinant
                lambda_1 = jnp.where(
                    active, lambda_both_1, b_value / m11
                )
                lambda_2 = jnp.where(active, lambda_both_2, 0.0)
                correction = (
                    lambda_1[:, None] * gb
                    + lambda_2[:, None] * radial
                )
                projected = jnp.clip(current - correction, lower, upper)
                return jnp.where(jnp.isfinite(projected), projected, current)

            return jax.lax.fori_loop(
                0,
                projection_steps,
                project_once,
                jnp.clip(points, lower, upper),
            )

        def feasible_values(points):
            a_value, b_value, _, _ = abg_batch(points)
            norm_sq = jnp.sum(points * points, axis=1)
            valid = (
                jnp.isfinite(a_value)
                & jnp.isfinite(b_value)
                & (jnp.abs(b_value) <= b_tol)
                & (norm_sq > r0_sq)
            )
            return a_value, valid

        points = enforce_constraints(seeds)
        initial_a, initial_valid = feasible_values(points)
        best_a = jnp.where(initial_valid, initial_a, -jnp.inf)
        best_points = points
        factors = 0.5 ** jnp.arange(line_search_steps, dtype=dtype)
        identity = jnp.eye(4, dtype=dtype)
        hessians = jnp.broadcast_to(
            identity, (points.shape[0], 4, 4)
        )

        def ascent_step(_, state):
            current, best_current, best_values, current_hessians = state
            a_value, b_value, ga, gb = abg_batch(current)
            grad_f = -ga
            gb_norm_sq = jnp.sum(gb * gb, axis=1) + eps
            norm_sq = jnp.sum(current * current, axis=1)
            radial = 2.0 * current
            sphere_active = norm_sq <= r0_sq * (1.0 + 1.0e-5)
            constraint_jac = jnp.stack(
                [gb, jnp.where(sphere_active[:, None], radial, 0.0)],
                axis=1,
            )
            constraint_value = jnp.stack(
                [
                    b_value,
                    jnp.where(sphere_active, norm_sq - r0_sq, 0.0),
                ],
                axis=1,
            )
            dual_regularization = jnp.stack(
                [
                    jnp.full_like(b_value, 1.0e-12),
                    jnp.where(sphere_active, 1.0e-12, 1.0),
                ],
                axis=1,
            )
            top = jnp.concatenate(
                [current_hessians, jnp.swapaxes(constraint_jac, 1, 2)],
                axis=2,
            )
            dual_block = -jax.vmap(jnp.diag)(dual_regularization)
            bottom = jnp.concatenate([constraint_jac, dual_block], axis=2)
            kkt = jnp.concatenate([top, bottom], axis=1)
            rhs = -jnp.concatenate([grad_f, constraint_value], axis=1)
            solution = jnp.linalg.solve(kkt, rhs[..., None])[..., 0]
            direction = solution[:, :4]
            multipliers = solution[:, 4:]

            tangent = ga - (
                jnp.sum(ga * gb, axis=1) / gb_norm_sq
            )[:, None] * gb
            tangent_norm = jnp.linalg.norm(tangent, axis=1, keepdims=True)
            tangent_direction = jnp.where(
                tangent_norm > 1.0e-18,
                tangent / jnp.maximum(tangent_norm, 1.0e-18),
                0.0,
            )
            solved = jnp.all(jnp.isfinite(solution), axis=1)
            direction = jnp.where(
                solved[:, None], direction, tangent_direction
            )
            multipliers = jnp.where(solved[:, None], multipliers, 0.0)
            direction_norm = jnp.linalg.norm(
                direction, axis=1, keepdims=True
            )
            trust_scale = jnp.minimum(
                1.0, step_size / jnp.maximum(direction_norm, 1.0e-18)
            )
            direction = direction * trust_scale
            trials = (
                current[:, None, :]
                + factors[None, :, None] * direction[:, None, :]
            )
            trial_shape = trials.shape
            trials = enforce_constraints(trials.reshape((-1, 4)))
            trials = trials.reshape(trial_shape)
            choices = jnp.concatenate([current[:, None, :], trials], axis=1)
            flat_choices = choices.reshape((-1, 4))
            choice_a, choice_b, _, _ = abg_batch(flat_choices)
            choice_a = choice_a.reshape((current.shape[0], -1))
            choice_b = choice_b.reshape((current.shape[0], -1))
            choice_norm = jnp.sum(choices * choices, axis=2)
            search_tol = jnp.maximum(100.0 * b_tol, 1.0e-10)
            search_valid = (
                jnp.isfinite(choice_a)
                & jnp.isfinite(choice_b)
                & (jnp.abs(choice_b) <= search_tol)
                & (choice_norm > r0_sq)
            )
            penalty = jnp.maximum(
                10.0, 2.0 * jnp.max(jnp.abs(multipliers), axis=1)
            )
            sphere_error = jnp.maximum(0.0, r0_sq - choice_norm)
            choice_merit = (
                -choice_a
                + penalty[:, None] * (jnp.abs(choice_b) + sphere_error)
            )
            choice_merit = jnp.where(search_valid, choice_merit, jnp.inf)
            selected = jnp.argmin(choice_merit, axis=1)
            new_points = jnp.take_along_axis(
                choices, selected[:, None, None], axis=1
            )[:, 0, :]

            new_a, new_b, new_ga, new_gb = abg_batch(new_points)
            new_norm_sq = jnp.sum(new_points * new_points, axis=1)
            new_valid = (
                jnp.isfinite(new_a)
                & jnp.isfinite(new_b)
                & (jnp.abs(new_b) <= b_tol)
                & (new_norm_sq > r0_sq)
            )
            improve = new_valid & (new_a > best_values)
            best_current = jnp.where(
                improve[:, None], new_points, best_current
            )
            best_values = jnp.where(improve, new_a, best_values)

            old_grad_lagrangian = (
                grad_f
                + multipliers[:, :1] * gb
                + multipliers[:, 1:2]
                * jnp.where(sphere_active[:, None], radial, 0.0)
            )
            new_sphere_active = new_norm_sq <= r0_sq * (1.0 + 1.0e-5)
            new_grad_lagrangian = (
                -new_ga
                + multipliers[:, :1] * new_gb
                + multipliers[:, 1:2]
                * jnp.where(
                    new_sphere_active[:, None], 2.0 * new_points, 0.0
                )
            )
            displacement = new_points - current
            gradient_change = new_grad_lagrangian - old_grad_lagrangian
            hs = jnp.einsum("bij,bj->bi", current_hessians, displacement)
            s_h_s = jnp.sum(displacement * hs, axis=1)
            y_s = jnp.sum(gradient_change * displacement, axis=1)
            theta = jnp.where(
                y_s >= 0.2 * s_h_s,
                1.0,
                0.8 * s_h_s / jnp.maximum(s_h_s - y_s, eps),
            )
            damped_y = (
                theta[:, None] * gradient_change
                + (1.0 - theta)[:, None] * hs
            )
            r_s = jnp.sum(damped_y * displacement, axis=1)
            candidate_hessian = (
                current_hessians
                - jnp.einsum("bi,bj->bij", hs, hs)
                / jnp.maximum(s_h_s, eps)[:, None, None]
                + jnp.einsum("bi,bj->bij", damped_y, damped_y)
                / jnp.maximum(r_s, eps)[:, None, None]
            )
            update_valid = (
                (s_h_s > 1.0e-18)
                & (r_s > 1.0e-18)
                & jnp.all(jnp.isfinite(candidate_hessian), axis=(1, 2))
            )
            current_hessians = jnp.where(
                update_valid[:, None, None],
                candidate_hessian,
                current_hessians,
            )
            current_hessians = 0.5 * (
                current_hessians
                + jnp.swapaxes(current_hessians, 1, 2)
            )
            return new_points, best_current, best_values, current_hessians

        points, best_points, best_a, _ = jax.lax.fori_loop(
            0,
            iterations,
            ascent_step,
            (points, best_points, best_a, hessians),
        )
        points = enforce_constraints(points)
        final_a, final_valid = feasible_values(points)
        improve = final_valid & (final_a > best_a)
        best_points = jnp.where(improve[:, None], points, best_points)
        best_a = jnp.where(improve, final_a, best_a)
        return best_points, jnp.isfinite(best_a), best_a

    return polish


@lru_cache(maxsize=8)
def _polish_many_kernel(
    candidate_count, iterations, projection_steps, line_search_steps
):
    polish_one = _polish_kernel(
        iterations, projection_steps, line_search_steps
    )

    @jax.jit
    def polish(
        opcodes,
        operands,
        literals,
        n_ops,
        parameters,
        seeds,
        lower,
        upper,
        r0_sq,
        b_tol,
        step_size,
    ):
        return jax.vmap(
            lambda op, arg, lit, count, params, candidate_seeds: polish_one(
                op,
                arg,
                lit,
                count,
                params,
                candidate_seeds,
                lower,
                upper,
                r0_sq,
                b_tol,
                step_size,
            )
        )(opcodes, operands, literals, n_ops, parameters, seeds)

    return polish


@lru_cache(maxsize=8)
def _artstein_ascent_kernel(
    iterations, initial_projection_steps, projection_steps
):
    """GPU-resident projected ascent for max a(x) subject to b(x)=0.

    This is deliberately not a penalty/barrier objective.  Newton restoration
    enforces ``b=0`` and the ascent direction is the projection of ``grad(a)``
    onto the tangent space of that equality.  Thousands of starts can advance
    together, unlike the latency-bound SQP line search used by ``polish``.
    """

    @jax.jit
    def optimize(
        opcodes,
        operands,
        literals,
        n_ops,
        parameters,
        seeds,
        seed_active,
        lower,
        upper,
        r0_sq,
        b_tol,
        step_size,
    ):
        dtype = seeds.dtype
        eps = jnp.asarray(1.0e-24, dtype=dtype)
        minimum_step = jnp.asarray(1.0e-8, dtype=dtype)
        maximum_step = jnp.asarray(2.0, dtype=dtype) * step_size

        def abg_batch(points):
            return _abg_batch(
                opcodes,
                operands,
                literals,
                n_ops,
                parameters,
                points,
            )

        def keep_outside_origin(points):
            norm_sq = jnp.sum(points * points, axis=1)
            norm = jnp.sqrt(jnp.maximum(norm_sq, eps))
            fallback = jnp.zeros_like(points).at[:, 0].set(jnp.sqrt(r0_sq))
            radial = points * (
                jnp.sqrt(r0_sq) * (1.0 + 1.0e-6)
                / jnp.maximum(norm, jnp.sqrt(eps))
            )[:, None]
            radial = jnp.where(
                (norm > jnp.sqrt(eps))[:, None], radial, fallback
            )
            return jnp.where((norm_sq <= r0_sq)[:, None], radial, points)

        def project(points, count):
            def project_once(_, current):
                _, b_value, _, gb = abg_batch(current)
                gb_norm_sq = jnp.sum(gb * gb, axis=1) + eps
                correction = (b_value / gb_norm_sq)[:, None] * gb
                projected = jnp.clip(current - correction, lower, upper)
                projected = keep_outside_origin(projected)
                finite = jnp.all(jnp.isfinite(projected), axis=1)
                return jnp.where(finite[:, None], projected, current)

            return jax.lax.fori_loop(
                0,
                count,
                project_once,
                keep_outside_origin(jnp.clip(points, lower, upper)),
            )

        points = project(seeds, initial_projection_steps)
        a_value, b_value, _, _ = abg_batch(points)
        norm_sq = jnp.sum(points * points, axis=1)
        feasible = (
            seed_active
            & jnp.isfinite(a_value)
            & jnp.isfinite(b_value)
            & (jnp.abs(b_value) <= b_tol)
            & (norm_sq > r0_sq)
        )
        best_points = points
        best_a = jnp.where(feasible, a_value, -jnp.inf)
        steps = jnp.full((points.shape[0],), step_size, dtype=dtype)

        def ascent_step(_, state):
            current, current_steps, current_best, current_best_a = state
            a_now, b_now, ga, gb = abg_batch(current)
            gb_norm_sq = jnp.sum(gb * gb, axis=1) + eps
            tangent = ga - (
                jnp.sum(ga * gb, axis=1) / gb_norm_sq
            )[:, None] * gb

            # Respect active box faces before taking the tangent step.
            face_tol = jnp.asarray(1.0e-12, dtype=dtype)
            blocked_low = (current <= lower + face_tol) & (tangent < 0.0)
            blocked_high = (current >= upper - face_tol) & (tangent > 0.0)
            tangent = jnp.where(blocked_low | blocked_high, 0.0, tangent)

            # At the puncture boundary remove the inward radial component.
            norm_sq_now = jnp.sum(current * current, axis=1)
            radial_dot = jnp.sum(tangent * current, axis=1)
            radial_scale = radial_dot / jnp.maximum(norm_sq_now, eps)
            inward = (
                (norm_sq_now <= r0_sq * (1.0 + 1.0e-5))
                & (radial_dot < 0.0)
            )
            tangent = jnp.where(
                inward[:, None],
                tangent - radial_scale[:, None] * current,
                tangent,
            )
            tangent_norm = jnp.linalg.norm(tangent, axis=1)
            direction = tangent / jnp.maximum(tangent_norm, 1.0e-18)[:, None]
            proposal = current + current_steps[:, None] * direction
            proposal = project(proposal, projection_steps)
            a_new, b_new, _, _ = abg_batch(proposal)
            norm_sq_new = jnp.sum(proposal * proposal, axis=1)
            valid_new = (
                seed_active
                & jnp.isfinite(a_new)
                & jnp.isfinite(b_new)
                & (jnp.abs(b_new) <= b_tol)
                & (norm_sq_new > r0_sq)
            )
            valid_now = (
                seed_active
                & jnp.isfinite(a_now)
                & jnp.isfinite(b_now)
                & (jnp.abs(b_now) <= b_tol)
                & (norm_sq_now > r0_sq)
            )
            improves_a = a_new > a_now + 1.0e-14
            improves_feasibility = jnp.abs(b_new) < jnp.abs(b_now)
            accept = valid_new & (~valid_now | improves_a)
            accept = accept | (
                seed_active
                & ~valid_now
                & jnp.isfinite(a_new)
                & jnp.isfinite(b_new)
                & improves_feasibility
            )
            current = jnp.where(accept[:, None], proposal, current)
            current_steps = jnp.where(
                accept,
                jnp.minimum(current_steps * 1.08, maximum_step),
                jnp.maximum(current_steps * 0.5, minimum_step),
            )

            improve_best = valid_new & (a_new > current_best_a)
            current_best = jnp.where(
                improve_best[:, None], proposal, current_best
            )
            current_best_a = jnp.where(
                improve_best, a_new, current_best_a
            )
            return current, current_steps, current_best, current_best_a

        points, _, best_points, best_a = jax.lax.fori_loop(
            0,
            iterations,
            ascent_step,
            (points, steps, best_points, best_a),
        )
        best_points = project(best_points, initial_projection_steps)
        final_a, final_b, _, _ = abg_batch(best_points)
        final_norm_sq = jnp.sum(best_points * best_points, axis=1)
        final_valid = (
            seed_active
            & jnp.isfinite(final_a)
            & jnp.isfinite(final_b)
            & (jnp.abs(final_b) <= b_tol)
            & (final_norm_sq > r0_sq)
        )
        improve = final_valid & (final_a > best_a)
        best_a = jnp.where(improve, final_a, best_a)
        return best_points, jnp.isfinite(best_a), best_a

    return optimize


@lru_cache(maxsize=8)
def _artstein_ascent_many_kernel(
    candidate_count, iterations, initial_projection_steps, projection_steps
):
    optimize_one = _artstein_ascent_kernel(
        iterations, initial_projection_steps, projection_steps
    )

    @jax.jit
    def optimize(
        opcodes,
        operands,
        literals,
        n_ops,
        parameters,
        seeds,
        seed_active,
        lower,
        upper,
        r0_sq,
        b_tol,
        step_size,
    ):
        return jax.vmap(
            lambda op, arg, lit, count, params, candidate_seeds, active: (
                optimize_one(
                    op,
                    arg,
                    lit,
                    count,
                    params,
                    candidate_seeds,
                    active,
                    lower,
                    upper,
                    r0_sq,
                    b_tol,
                    step_size,
                )
            )
        )(
            opcodes,
            operands,
            literals,
            n_ops,
            parameters,
            seeds,
            seed_active,
        )

    return optimize


class RuntimeExactBundle:
    """Candidate data wrapper around shared fixed-shape exact kernels."""

    def __init__(self, expression, block_size=DEFAULT_BLOCK_SIZE):
        program = encode_expression(str(expression))
        self.opcodes = jax.device_put(program.opcodes)
        self.operands = jax.device_put(program.operands)
        self.literals = jax.device_put(program.literals)
        self.n_ops = jax.device_put(np.int32(program.n_ops))
        self.program_n_ops = int(program.n_ops)
        self.n_params = int(program.n_constants)
        self.block_size = int(block_size)
        requested_chunk = int(
            os.environ.get("SYMCLF_GPU2_EXACT_SCAN_CHUNK", "262144")
        )
        self.scan_chunk = max(
            self.block_size,
            (requested_chunk // self.block_size) * self.block_size,
        )
        self.interpret = jax.default_backend() != "gpu"

    def _parameters(self, values):
        values = jnp.asarray(values, dtype=jnp.float64).reshape(-1)
        return jnp.pad(values, (0, MAX_CONSTANTS - values.shape[0]))

    def _ab(self, points, values):
        real_count = points.shape[0]
        parameters = self._parameters(values)

        # The full exact geometry has about 2.61M points. A single
        # value+gradient stack would be ~3.34 GB and can cross Triton's 32-bit
        # byte-offset boundary. Fixed GPU chunks preserve every point while
        # keeping scratch addressing and allocation safely below 2 GB. Calls
        # are asynchronous device launches; there is no per-chunk device_get.
        if real_count > self.scan_chunk:
            evaluate = _ab_interpreter(
                self.scan_chunk, self.block_size, self.interpret
            )
            parts = []
            for start in range(0, real_count, self.scan_chunk):
                stop = min(start + self.scan_chunk, real_count)
                chunk = points[start:stop]
                chunk_count = stop - start
                if chunk_count < self.scan_chunk:
                    chunk = jnp.pad(
                        chunk,
                        ((0, self.scan_chunk - chunk_count), (0, 0)),
                    )
                part = evaluate(
                    self.opcodes,
                    self.operands,
                    self.literals,
                    self.n_ops,
                    parameters,
                    chunk,
                )
                parts.append(part[:, :chunk_count])
            return jnp.concatenate(parts, axis=1)

        padded_count = (
            (real_count + self.block_size - 1) // self.block_size
        ) * self.block_size
        if padded_count != real_count:
            points = jnp.pad(
                points, ((0, padded_count - real_count), (0, 0))
            )
        result = _ab_interpreter(
            padded_count, self.block_size, self.interpret
        )(
            self.opcodes,
            self.operands,
            self.literals,
            self.n_ops,
            parameters,
            points,
        )
        return result[:, :real_count]

    def _b(self, points, values):
        real_count = points.shape[0]
        parameters = self._parameters(values)
        if real_count > self.scan_chunk:
            evaluate = _b_interpreter(
                self.scan_chunk, self.block_size, self.interpret
            )
            parts = []
            for start in range(0, real_count, self.scan_chunk):
                stop = min(start + self.scan_chunk, real_count)
                chunk = points[start:stop]
                chunk_count = stop - start
                if chunk_count < self.scan_chunk:
                    chunk = jnp.pad(
                        chunk,
                        ((0, self.scan_chunk - chunk_count), (0, 0)),
                    )
                part = evaluate(
                    self.opcodes,
                    self.operands,
                    self.literals,
                    self.n_ops,
                    parameters,
                    chunk,
                )
                parts.append(part[:chunk_count])
            return jnp.concatenate(parts, axis=0)

        padded_count = (
            (real_count + self.block_size - 1) // self.block_size
        ) * self.block_size
        if padded_count != real_count:
            points = jnp.pad(
                points, ((0, padded_count - real_count), (0, 0))
            )
        result = _b_interpreter(
            padded_count, self.block_size, self.interpret
        )(
            self.opcodes,
            self.operands,
            self.literals,
            self.n_ops,
            parameters,
            points,
        )
        return result[:real_count]

    def ab_batch(self, points, values):
        result = self._ab(points, values)
        return result[0], result[1]

    def b_batch(self, points, values):
        return self._b(points, values)

    def abg_batch(self, points, values):
        return _abg_batch(
            self.opcodes,
            self.operands,
            self.literals,
            self.n_ops,
            self._parameters(values),
            points,
        )

    def bisect_axis(self, points, onehot, lo, hi, b_lo, values, iterations):
        kernel = _bisect_axis_kernel(
            int(points.shape[0]),
            int(iterations),
            self.block_size,
            self.interpret,
        )
        return kernel(
            self.opcodes,
            self.operands,
            self.literals,
            self.n_ops,
            self._parameters(values),
            points,
            onehot,
            lo,
            hi,
            b_lo,
        )

    def bisect_line(self, origins, directions, lo, hi, b_lo, values, iterations):
        kernel = _bisect_line_kernel(
            int(origins.shape[0]),
            int(iterations),
            self.block_size,
            self.interpret,
        )
        return kernel(
            self.opcodes,
            self.operands,
            self.literals,
            self.n_ops,
            self._parameters(values),
            origins,
            directions,
            lo,
            hi,
            b_lo,
        )

    def polish_batch(
        self,
        seeds,
        values,
        lower,
        upper,
        r0_sq,
        b_tol,
        step_size,
        iterations,
        projection_steps,
        line_search_steps,
    ):
        kernel = _polish_kernel(
            int(iterations),
            int(projection_steps),
            int(line_search_steps),
        )
        return kernel(
            self.opcodes,
            self.operands,
            self.literals,
            self.n_ops,
            self._parameters(values),
            seeds,
            lower,
            upper,
            r0_sq,
            b_tol,
            step_size,
        )


class RuntimeExactBatchBundle:
    """Fixed-width programs evaluated together on one GPU."""

    def __init__(self, candidates, block_size=DEFAULT_BLOCK_SIZE):
        candidates = tuple(candidates)
        if not candidates:
            raise ValueError("runtime exact batch cannot be empty")
        programs = [encode_expression(str(item.expression)) for item in candidates]
        self.candidate_count = len(candidates)
        self.opcodes = jax.device_put(
            np.stack([program.opcodes for program in programs])
        )
        self.operands = jax.device_put(
            np.stack([program.operands for program in programs])
        )
        self.literals = jax.device_put(
            np.stack([program.literals for program in programs])
        )
        self.n_ops = jax.device_put(
            np.asarray([program.n_ops for program in programs], dtype=np.int32)
        )
        self.program_n_ops = tuple(int(program.n_ops) for program in programs)
        parameter_rows = []
        for candidate, program in zip(candidates, programs):
            values = np.asarray(candidate.constants, dtype=np.float64).reshape(-1)
            if values.size != program.n_constants:
                raise ValueError(
                    f"runtime exact expected {program.n_constants} constants; "
                    f"received {values.size}"
                )
            parameter_rows.append(
                np.pad(values, (0, MAX_CONSTANTS - values.size))
            )
        self.parameters = jax.device_put(np.stack(parameter_rows))
        self.block_size = int(block_size)
        requested_chunk = int(
            os.environ.get("SYMCLF_GPU2_EXACT_SCAN_CHUNK", "262144")
        )
        self.scan_chunk = max(
            self.block_size,
            (requested_chunk // self.block_size) * self.block_size,
        )
        self.interpret = jax.default_backend() != "gpu"

    def b_batch(self, points):
        """Evaluate all candidates on one shared point array."""
        real_count = int(points.shape[0])
        parts = []
        for start in range(0, real_count, self.scan_chunk):
            stop = min(start + self.scan_chunk, real_count)
            chunk = points[start:stop]
            chunk_count = stop - start
            padded_count = (
                (chunk_count + self.block_size - 1) // self.block_size
            ) * self.block_size
            if padded_count != chunk_count:
                chunk = jnp.pad(
                    chunk, ((0, padded_count - chunk_count), (0, 0))
                )
            evaluate = _b_many_interpreter(
                self.candidate_count,
                padded_count,
                self.block_size,
                self.interpret,
            )
            part = evaluate(
                self.opcodes,
                self.operands,
                self.literals,
                self.n_ops,
                self.parameters,
                chunk,
            )
            parts.append(part[:, :chunk_count])
        return jnp.concatenate(parts, axis=1)

    def ab_batch(self, points):
        """Evaluate a,b for candidate-specific points [C,M,4]."""
        real_count = int(points.shape[1])
        parts = []
        for start in range(0, real_count, self.scan_chunk):
            stop = min(start + self.scan_chunk, real_count)
            chunk = points[:, start:stop, :]
            chunk_count = stop - start
            padded_count = (
                (chunk_count + self.block_size - 1) // self.block_size
            ) * self.block_size
            if padded_count != chunk_count:
                chunk = jnp.pad(
                    chunk,
                    ((0, 0), (0, padded_count - chunk_count), (0, 0)),
                )
            evaluate = _ab_many_interpreter(
                self.candidate_count,
                padded_count,
                self.block_size,
                self.interpret,
            )
            result = evaluate(
                self.opcodes,
                self.operands,
                self.literals,
                self.n_ops,
                self.parameters,
                chunk,
            )
            parts.append(result[:, :, :chunk_count])
        return jnp.concatenate(parts, axis=2)

    def bisect_line(self, origins, directions, lo, hi, b_lo, iterations):
        point_count = int(origins.shape[1])
        kernel = _bisect_line_many_kernel(
            self.candidate_count,
            point_count,
            int(iterations),
            self.block_size,
            self.interpret,
        )
        return kernel(
            self.opcodes,
            self.operands,
            self.literals,
            self.n_ops,
            self.parameters,
            origins,
            directions,
            lo,
            hi,
            b_lo,
        )

    def polish_batch(
        self,
        seeds,
        lower,
        upper,
        r0_sq,
        b_tol,
        step_size,
        iterations,
        projection_steps,
        line_search_steps,
    ):
        kernel = _polish_many_kernel(
            self.candidate_count,
            int(iterations),
            int(projection_steps),
            int(line_search_steps),
        )
        return kernel(
            self.opcodes,
            self.operands,
            self.literals,
            self.n_ops,
            self.parameters,
            seeds,
            lower,
            upper,
            r0_sq,
            b_tol,
            step_size,
        )

    def artstein_ascent_batch(
        self,
        seeds,
        seed_active,
        lower,
        upper,
        r0_sq,
        b_tol,
        step_size,
        iterations,
        initial_projection_steps,
        projection_steps,
    ):
        """Maximize a on b=0 for all candidates and starts on-device."""
        kernel = _artstein_ascent_many_kernel(
            self.candidate_count,
            int(iterations),
            int(initial_projection_steps),
            int(projection_steps),
        )
        return kernel(
            self.opcodes,
            self.operands,
            self.literals,
            self.n_ops,
            self.parameters,
            seeds,
            seed_active,
            lower,
            upper,
            r0_sq,
            b_tol,
            step_size,
        )


def get_runtime_bundle(expression, constants, block_size=DEFAULT_BLOCK_SIZE):
    bundle = RuntimeExactBundle(expression, block_size=block_size)
    constants = np.asarray(constants, dtype=np.float64).reshape(-1)
    if constants.size != bundle.n_params:
        raise ValueError(
            f"runtime exact expected {bundle.n_params} constants; "
            f"received {constants.size}"
        )
    return bundle, constants


def runtime_candidate_bundle(candidate):
    """Build a data-only bundle using the configured Pallas block size."""
    block_size = int(
        os.environ.get("SYMCLF_GPU2_PALLAS_BLOCK_SIZE", DEFAULT_BLOCK_SIZE)
    )
    return get_runtime_bundle(
        candidate.expression,
        candidate.constants,
        block_size=block_size,
    )


def runtime_candidate_batch_bundle(candidates):
    """Build one fixed-shape bundle for several exact candidates."""
    block_size = int(
        os.environ.get("SYMCLF_GPU2_PALLAS_BLOCK_SIZE", DEFAULT_BLOCK_SIZE)
    )
    return RuntimeExactBatchBundle(candidates, block_size=block_size)
