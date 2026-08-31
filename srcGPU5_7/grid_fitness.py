"""Fixed-program JAX evaluator for the complete 4D cart-pole grid fitness.

GP trees are encoded as runtime postfix instructions. The JAX kernel has one
fixed shape for every tree, so structural mutation does not trigger a new XLA
compile. Constants are runtime data as well. GPU2 tuning keeps its population,
scores, selection, mutation, and SEA generations on-device; only the final
champion score and constants return to the host.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from srcGPU5_7 import GPUUnavailableError, require_gpu
from srcGPU5_7.pallas_interpreter import DEFAULT_BLOCK_SIZE, evaluate_program_batch


# --- GPU4_1 properness sandwich -------------------------------------------
# Widened band vs the srcGPU2 default (0.1x .. 50x the reference quadratic).
# A wider band admits more structurally-different candidates as "proper",
# which is the diversity lever for the premature-convergence problem seen in
# job 99824. Override per run with the env vars.
PROPERNESS_LOWER = float(os.environ.get("SYMCLF_GPU4_1_PROPER_LOWER", "0.05"))
PROPERNESS_UPPER = float(os.environ.get("SYMCLF_GPU4_1_PROPER_UPPER", "500.0"))

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


def _evaluate_program(opcodes, operands, literals, n_ops, parameters, points):
    """Evaluate one runtime program at every row of ``points``."""
    stack = jnp.zeros((MAX_STACK_DEPTH, points.shape[0]), dtype=jnp.float64)

    def push_x(args):
        stack_, pointer, operand, _, points_, _ = args
        return stack_.at[pointer].set(points_[:, operand]), pointer + 1

    def push_parameter(args):
        stack_, pointer, operand, _, _, parameters_ = args
        return stack_.at[pointer].set(parameters_[operand]), pointer + 1

    def push_literal(args):
        stack_, pointer, _, literal, _, _ = args
        return stack_.at[pointer].set(literal), pointer + 1

    def binary(args, operation):
        stack_, pointer, _, _, _, _ = args
        left = stack_[pointer - 2]
        right = stack_[pointer - 1]
        return stack_.at[pointer - 2].set(operation(left, right)), pointer - 1

    def unary(args, operation):
        stack_, pointer, _, _, _, _ = args
        return stack_.at[pointer - 1].set(operation(stack_[pointer - 1])), pointer

    branches = (
        push_x,
        push_parameter,
        push_literal,
        lambda args: binary(args, lambda x, y: x + y),
        lambda args: binary(args, lambda x, y: x - y),
        lambda args: binary(args, lambda x, y: x * y),
        lambda args: binary(args, lambda x, y: x / jnp.sqrt(1.0 + y * y)),
        lambda args: unary(args, lambda x: -x),
        lambda args: unary(args, jnp.sin),
        lambda args: unary(args, jnp.exp),
    )

    def body(index, state):
        stack_, pointer = state
        args = (
            stack_, pointer, operands[index], literals[index], points, parameters
        )
        return jax.lax.switch(opcodes[index], branches, args)

    stack, pointer = jax.lax.fori_loop(0, n_ops, body, (stack, 0))
    return stack[pointer - 1]


def _gradient_axis(values, spacing, axis):
    """NumPy ``gradient(..., edge_order=2)`` stencil for a uniform grid."""
    first = (
        -3.0 * jnp.take(values, 0, axis=axis)
        + 4.0 * jnp.take(values, 1, axis=axis)
        - jnp.take(values, 2, axis=axis)
    ) / (2.0 * spacing)
    middle = (
        jnp.take(values, jnp.arange(2, values.shape[axis]), axis=axis)
        - jnp.take(values, jnp.arange(0, values.shape[axis] - 2), axis=axis)
    ) / (2.0 * spacing)
    last = (
        3.0 * jnp.take(values, values.shape[axis] - 1, axis=axis)
        - 4.0 * jnp.take(values, values.shape[axis] - 2, axis=axis)
        + jnp.take(values, values.shape[axis] - 3, axis=axis)
    ) / (2.0 * spacing)
    return jnp.concatenate(
        [jnp.expand_dims(first, axis), middle, jnp.expand_dims(last, axis)],
        axis=axis,
    )


def _boundary_min(values):
    candidates = []
    for axis in range(values.ndim):
        candidates.append(jnp.min(jnp.take(values, 0, axis=axis)))
        candidates.append(jnp.min(jnp.take(values, values.shape[axis] - 1, axis=axis)))
    return jnp.min(jnp.stack(candidates))


@lru_cache(maxsize=8)
def _score_kernel(grid_points: int):
    shape = (grid_points,) * 4

    @jax.jit
    def kernel(
        evaluated,
        f_values,
        g_values,
        reference,
        radius_squared,
        probe_radius_squared,
        spacing,
        properness_weight,
        decay_rate,
        gradient_target_12,
        gradient_target_34,
        gradient_weight,
        origin_probe_k,
        origin_probe_penalty,
        c_star_target,
        c_star_penalty_max,
        c_star_penalty_cap,
        c_star_tail_weight,
        flat_floor,
        flat_weight,
        axis_mask,
        axis_b_target,
        axis_b_weight,
        axis_v_target,
        axis_v_weight,
        axis_tail_weight,
        origin_penalty_weight,
        coverage_weight,
        sat_weight,
        sat_u_target,
        roa_weight,
        roa_c_target,
        roa_neg_weight,
    ):
        n_grid = grid_points**4
        values = evaluated[:n_grid].reshape(shape)
        origin_value = evaluated[n_grid]
        probes = evaluated[n_grid + 1:n_grid + 17]
        rollout_values = evaluated[n_grid + 17:n_grid + 21]

        gradients = jnp.stack(
            [_gradient_axis(values, spacing, axis) for axis in range(4)], axis=-1
        )
        gradients_flat = gradients.reshape((-1, 4))
        values_flat = values.reshape(-1)

        a_values = jnp.sum(gradients_flat * f_values, axis=1)
        b_values = jnp.sum(gradients_flat * g_values, axis=1)
        b_norm_squared = 1.0e4 * b_values * b_values
        lambda_values = (
            jnp.sqrt(a_values * a_values + radius_squared * b_norm_squared)
            + a_values
        ) / (b_norm_squared + 1.0e-6)
        vdot_values = a_values - 1.0e4 * b_values * b_values * lambda_values

        nonorigin = radius_squared > 0.0
        # GPU5_5: properness is measured on W = V - V(0), not on raw V.
        #
        # V and V + c are the SAME control-Lyapunov function: a = grad(V).f,
        # b = grad(V).g and Vdot are all identical, and positive definiteness is
        # defined on W = V - V(0). Every other grid term already subtracts V(0)
        # -- v_violations, c_star, coverage, the origin probe. Properness was
        # the ONE term reading raw V, and it is the only reason the additive
        # constant ever mattered.
        #
        # That made the 1e6 * V(0)^2 penalty necessary, and it was expensive:
        # on the run-122300 champion it was 183.09 of a 266.26 grid total --
        # 68.8% of the fitness spent on a gauge choice with no effect on whether
        # V is a CLF. Run 122299 paid 157.14. With properness on W the penalty
        # is redundant and GPU5_7_ORIGIN_PENALTY_WEIGHT defaults to 0.
        relative_values_flat = values_flat - origin_value
        proper = properness_weight * jnp.count_nonzero(
            (
                (PROPERNESS_UPPER * reference <= relative_values_flat)
                | (relative_values_flat <= PROPERNESS_LOWER * reference)
            )
            & nonorigin
        )
        v_violation_mask = (
            (relative_values_flat <= 1.0e-4 * radius_squared) & nonorigin
        )
        vdot_violation_mask = (
            (vdot_values >= -decay_rate * radius_squared) & nonorigin
        )
        v_violations = jnp.count_nonzero(v_violation_mask)
        vdot_violations = jnp.count_nonzero(vdot_violation_mask)

        # ``_evaluate_v`` in the CPU evaluator rejects a non-finite or
        # over-limit V before scoring. The remaining three fields contribute
        # one 1e6 penalty each, exactly as in Evaluate.py.
        valid_v = (
            jnp.all(jnp.isfinite(values_flat))
            & jnp.isfinite(origin_value)
            & (jnp.max(jnp.abs(values_flat)) <= 1.0e6)
        )
        invalid = 1.0e6 * (
            (~jnp.all(jnp.isfinite(a_values))).astype(jnp.float64)
            + (~jnp.all(jnp.isfinite(vdot_values))).astype(jnp.float64)
            + (~jnp.all(jnp.isfinite(lambda_values))).astype(jnp.float64)
        )
        # GPU5_5: gauge, not a defect -- see the properness note above.
        # Weight 0 by default; set GPU5_7_ORIGIN_PENALTY_WEIGHT to 1e6 to
        # restore the GPU5_4 behaviour.
        origin_penalty_value = (
            origin_penalty_weight * origin_value * origin_value
        )

        gradient_magnitude = jnp.linalg.norm(gradients_flat, axis=1)
        target_error_12 = jnp.abs(
            jnp.mean(jnp.hypot(gradients_flat[:, 0], gradients_flat[:, 1]))
            - gradient_target_12
        )
        target_error_34 = jnp.abs(
            jnp.mean(jnp.hypot(gradients_flat[:, 2], gradients_flat[:, 3]))
            - gradient_target_34
        )
        gradient_penalty_value = gradient_weight * (target_error_12 + target_error_34)

        probe_ratio = jnp.max(
            jnp.maximum(0.0, (probes - origin_value) / probe_radius_squared)
        )
        probe_penalty_value = jnp.where(
            probe_ratio > origin_probe_k, origin_probe_penalty, 0.0
        )

        # GPU5_4: flat-gradient count.
        #
        # a = grad(V).f and b = grad(V).g are both LINEAR in grad(V), so
        # wherever the search cannot get a < 0 on {b = 0} the cheapest move is
        # to shrink grad(V) there: a -> 0 buys Artstein margin without fixing
        # anything. The same flattening is what the properness and V>0 counts
        # then charge for, so the two conditions appear mutually exclusive --
        # the search is sliding along one parameter, the local flatness of V.
        # Run 122299 generation 76 is that state exactly: margin_max = 8.6e-06
        # with 441 grid points at V - V(0) <= 0 and 795 improper points.
        #
        # Normalising the Artstein margin (srcGPU5_7/grad_norm_gpu5_7.py)
        # removes the payoff where the margin is graded; this removes it
        # everywhere else. The threshold is a FRACTION OF THE MEAN gradient
        # magnitude, so the term is invariant under V -> kV -- it forbids
        # RELATIVE flatness, which is the actual defect, and cannot be dodged
        # by rescaling V the way an absolute floor could.
        # Compared per unit ||x||, not absolutely. For any well-behaved V,
        # ||grad V|| -> 0 at the origin (a quadratic gives ||grad V|| ~ 2*lam*
        # ||x||), so an absolute floor flags the whole near-origin shell on a
        # perfectly good CLF -- measured, it charged the certified run-122232
        # champion 288 points. The RATIO ||grad V|| / ||x|| is bounded below by
        # 2*lam_min for a quadratic, so it isolates genuine flat spots and is
        # still invariant under V -> kV.
        radius = jnp.sqrt(jnp.maximum(radius_squared, 1.0e-300))
        slope = gradient_magnitude / radius
        slope_mean = jnp.sum(jnp.where(nonorigin, slope, 0.0)) / jnp.maximum(
            jnp.count_nonzero(nonorigin).astype(jnp.float64), 1.0
        )
        flat_violations = jnp.count_nonzero(
            (slope < flat_floor * slope_mean) & nonorigin
        )
        flat_penalty_value = flat_weight * flat_violations.astype(jnp.float64)

        # GPU5_5: the x1-axis control-authority term.
        #
        # The cart-pole drift never reads x1 (translation symmetry), so f == 0
        # on the ENTIRE x1 axis and therefore a = grad(V).f == 0 there for any V
        # whatsoever. If b = 0 there too, every axis point is a b=0 root with
        # margin EXACTLY zero, which `margin > margin_tol` (1e-12, gamma1 = 0)
        # classifies as NON-violating. A whole slab of the manifold certifies
        # for free, so the search has no reason to lift V off the axis -- and
        # V == 0 on a line through the origin also forces boundary_min = 0,
        # v_violations, and the flat-gradient count.
        #
        # Measured on the run-122300 champion, that ONE structural fact was
        # 308 of its 506 grid points (V>0 20 + flat 20 + c_star 263.9 + cov 4),
        # frozen for 60+ generations, because every one of those terms is an
        # integer count or a saturated constant: no continuous change can move
        # them. That is why fitness sits at ~800 and then drops to ~4 in a
        # single generation -- run 96051 held 833.52 for 7000 generations
        # before dropping to 4.48 in one step.
        #
        # This term is the continuous counterpart. On the axis b is
        # proportional to 2*V_x2 + V_x4, which varies smoothly with V's
        # coefficients, so it supplies a GRADIENT toward the nonzero x1-x2 /
        # x1-x4 cross-coupling a valid CLF must have. Both |b| and the mean
        # gradient magnitude scale linearly with V, so the ratio is invariant
        # under V -> kV; ||x|| makes it invariant along the axis as well.
        #
        #   min |b| / (||x|| * mean||grad V||)   measured:
        #     LQR reference   1.198e-03     53 champion   8.196e-06
        #     96051 (works)   6.994e-03     54 champion   2.012e-05
        #     122232          5.398e-03
        #
        # a 60x-850x gap. The target sits at the weakest GOOD value so the LQR
        # form itself is charged nothing.
        axis_count = jnp.sum(axis_mask)
        nonorigin_count = jnp.maximum(
            jnp.count_nonzero(nonorigin).astype(jnp.float64), 1.0
        )
        grad_mean = jnp.sum(
            jnp.where(nonorigin, gradient_magnitude, 0.0)
        ) / nonorigin_count

        def _axis_charge(ratio, target, weight):
            """LINEAR shoulder plus a never-flat tail.

            GPU5_5's first shape was ``weight * clip(1 - r/target, 0, 1)**2``.
            Its derivative is ``-2*weight/target * (1 - r/target)``, which goes
            to ZERO as r approaches the target -- the pull dies exactly at the
            threshold. Run 122301 parked at r = 1.036e-03 against a 1e-03
            target: 3.6% over, penalty 0.00, and the defect still present. Same
            gate-parking as everywhere else in this codebase.

            Linear keeps the full slope ``-weight/target`` right up to the
            target, and the ``tail*target/(target+r)`` term never becomes
            exactly flat, so more is always strictly better. The tail weight is
            deliberately tiny (default 0.5): a solved CLF pays ~0.2-0.3 points
            against a total fitness near 4, while a degenerate candidate pays
            the full weight.
            """
            below = weight * jnp.clip(1.0 - ratio / target, 0.0, 1.0)
            tail = axis_tail_weight * target / (target + jnp.maximum(ratio, 0.0))
            return below + tail

        # (a) CONTROL AUTHORITY on the axis: b must not vanish, or the axis
        #     joins {b=0} where a == 0 identically.
        axis_b_ratio = jnp.min(
            jnp.where(
                axis_mask > 0.0,
                jnp.abs(b_values) / jnp.maximum(radius * grad_mean, 1.0e-300),
                jnp.inf,
            )
        )

        # (b) POSITIVE DEFINITENESS on the axis: V must actually rise off the
        #     axis, or boundary_min pins to 0 and the V>0 count locks.
        #
        #     Run 122301 satisfied (a) and failed (b): axis_b_ratio = 1.036e-03
        #     (above target, charged nothing) while V was CONSTANT along the
        #     axis -- axis_v_ratio exactly 0.0, 21 zero points, boundary_min 0,
        #     c_star pinned at 263.86. It moved the degeneracy rather than
        #     removing it, so (a) alone was never sufficient.
        #
        #     min (V - V(0)) / (||x||^2 * mean over box of (V - V(0))/||x||^2)
        #     measured:  LQR 1.083e-01 | 96051 3.469e-02 | 122232 3.446e-02
        #                54  1.590e-03 | 53    0.0       | 55     0.0
        relative_flat = values_flat - origin_value
        curvature = relative_flat / jnp.maximum(radius_squared, 1.0e-300)
        curvature_mean = jnp.abs(
            jnp.sum(jnp.where(nonorigin, curvature, 0.0)) / nonorigin_count
        )
        axis_v_ratio = jnp.min(
            jnp.where(
                axis_mask > 0.0,
                curvature / jnp.maximum(curvature_mean, 1.0e-300),
                jnp.inf,
            )
        )

        axis_penalty_value = jnp.where(
            axis_count > 0.0,
            _axis_charge(axis_b_ratio, axis_b_target, axis_b_weight)
            + _axis_charge(axis_v_ratio, axis_v_target, axis_v_weight),
            0.0,
        )

        # --- GPU5_7: SATURATION (bounded-control feasibility) ---------------
        #
        # THE FINDING. Run 122340's gen-7 champion was "certified" -- PD
        # passes, Artstein violations only above W = 0.0316, c_max > 0 -- and
        # still DIVERGES in closed loop from initial conditions inside its own
        # certified set. Run 122304's champion is Artstein-INVALID on paper
        # (57 roots) and converges from every tested initial condition. The
        # quantity that separates them, and that no existing term measured:
        #
        #     u_required = max over the box of a / |b|  (where a > 0)
        #       96051 (verified CLF)   57      | 122304 champ (works)     59
        #       122340 gen-7 (diverges)  1.46e5, |u| ~ 1e11 along its orbits
        #
        # Artstein's condition (a < 0 exactly on {b = 0}) assumes UNBOUNDED u.
        # Any real actuator (and any integrator) has |u| <= u_max, so decrease
        # requires a - u_max*|b| < 0 EVERYWHERE off the origin, not just on
        # the manifold. a and b are both linear in grad(V), so a/|b| is
        # invariant under V -> kV -- unlike almost every term this replaces.
        # The charge is per DECADE above the target, so five orders of
        # magnitude of infeasibility cost 5x, smooth, never flat.
        #
        # This term also subsumes the x1-axis loophole structurally: f == 0 on
        # the axis forces a == 0 there for any V, so feasibility a < u_max*|b|
        # demands b != 0 on the axis -- what GPU5_5's axis-b patch enforced by
        # hand. (The axis terms stay, weight-reduced, purely as continuous
        # slope out of the degenerate-slab family, where a = b = 0 hides from
        # this ratio.)
        # Two knowingly-accepted properties of the max: (1) it is a minimax --
        # gen-7's p99.9 of a/|b| is 339 against a max of 1.76e5, the signal can
        # live in one grid point (the rollout gate is the backstop if this
        # turns noisy; a p99.9 or soft-max is the fallback). (2) epsilon_b
        # means a genuine Artstein violation (a > 0 at b ~ 0) shows up as
        # u_required ~ 1e9 * a / grad_mean, so the term partly re-prices
        # Artstein violations rather than measuring saturation independently
        # -- acceptable, since a > 0 on {b = 0} is exactly the |u| = inf case
        # of bounded-control infeasibility.
        b_abs = jnp.abs(b_values)
        epsilon_b = 1.0e-9 * grad_mean + 1.0e-300
        u_needed = jnp.where(
            nonorigin & (a_values > 0.0),
            a_values / (b_abs + epsilon_b),
            0.0,
        )
        u_required = jnp.max(u_needed)
        sat_penalty_value = sat_weight * jnp.maximum(
            0.0,
            jnp.log10(jnp.maximum(u_required, 1.0e-300) / sat_u_target),
        )

        # --- GPU5_7: the certified-region (ROA) objective --------------------
        #
        # The old properness band, c_star target and coverage count were
        # PRIORS about what V should look like; measured on 122340 they were
        # ANTI-correlated with the thing we want (the fitness discarded its
        # own certified gen-7 champion, 8406.5 -> 197 while certified volume
        # went 0.328% -> 0). This term IS the thing we want:
        #
        #     c_max = min(boundary_min, min W over violating grid points)
        #     vol   = fraction of the box with W < c_max   (the certified set)
        #
        # where "violating" = V>0 violation, Vdot violation, or saturation-
        # infeasible (a > u_max*|b|). Everything is priced on the SCALE-FREE
        # ratio c_ratio = c_max / median|W|: c_max alone scales with V -> kV,
        # so a raw-c_max target would let the search buy the whole term by
        # inflating V (every other 5_7 term is scale-invariant, so that
        # inflation is otherwise free).
        #
        # TWO REVIEW FIXES, both found live:
        #
        # 1. (run 122349, gen 14) The negative side was clip(., 0, 1) -- flat
        #    for every candidate with c_max <= -w_scale, the same shelf defect
        #    removed from c_star, the axis shoulders and the PD count; and the
        #    early population lives exactly there. Now log1p: same slope at
        #    shallow negatives, monotone and unbounded at depth.
        #
        # 2. (run 122350) The positive side was priced on certified VOLUME,
        #    hard-zeroed by where(c_max > 0, ., 0.0). Result: a flat 500
        #    across the entire negative side and a CLIFF at c_max = 0. The
        #    live champion stood at c_max = -4.8e-5 -- 98% of the way to the
        #    biggest prize in the fitness -- and closing the last 5e-5 was
        #    worth 0.27 points of guidance until the discontinuous 500 fired.
        #    Now the ramp is in c_ratio itself: continuous through zero, full
        #    slope roa_weight/roa_c_target on 0 < c_ratio < target, zero at
        #    roa_c_target = 0.02 (96051 sits at 0.0221, 122232 at 0.0274, the
        #    ARE reference at 0.0044 -- measured, not guessed). Volume is
        #    still computed and reported, it just no longer prices anything:
        #    it is quantised in 1/21^4 steps and constant-zero on half the
        #    domain, both fatal for a search signal.
        sat_infeasible_mask = nonorigin & (
            a_values > sat_u_target * b_abs
        )
        violating_mask = (
            v_violation_mask | vdot_violation_mask | sat_infeasible_mask
        )
        relative_values = values - origin_value
        raw_boundary_min = _boundary_min(relative_values)
        min_w_violating = jnp.min(
            jnp.where(violating_mask, relative_values_flat, jnp.inf)
        )
        c_max = jnp.minimum(raw_boundary_min, min_w_violating)
        certified_volume = jnp.where(
            c_max > 0.0,
            jnp.mean((relative_values_flat < c_max).astype(jnp.float64)),
            0.0,
        )
        w_scale = jnp.maximum(
            jnp.median(jnp.abs(relative_values_flat)), 1.0e-300
        )
        c_ratio = c_max / w_scale
        # GPU5_7 REVIEW FIX 3: one slope-matched function instead of two
        # halves that meet with a 383x slope discontinuity.
        #
        # The two-piece form was  clip(1 - z, 0, 1) + neg*log1p(-c_ratio)  with
        # z = c_ratio/roa_c_target. For ANY c_ratio < 0 the first term has
        # 1 - z > 1 and clips to 1, so it is FLAT 500 across the whole negative
        # side; only the log1p tail moves. Measured on run 122354 gen 42
        # (c_ratio = -1.2279e-03): total 500.31, of which 500.00 is the dead
        # clip and 0.31 is the tail. The slope there is -250 per unit c_ratio,
        # while the instant c_ratio crosses zero it becomes -125,000. The
        # search sits exactly where the gradient dies.
        #
        # Replacement, with z = c_ratio / roa_c_target:
        #     z >= 1      ->  0                     certificate at target
        #     0 <= z < 1  ->  1 - z                 linear, slope -1
        #     z < 0       ->  1 + log1p(-z)         continuous, slope -1 at 0
        # C^1 at z = 0 (value 1, slope -1 from both sides), so the pull is
        # roa_weight/roa_c_target = 125,000 per unit c_ratio right up to the
        # crossing, and logarithmic beyond so a deep-negative certificate never
        # goes nuclear the way a raw unclip would (c_ratio = -0.5 costs 2918,
        # not 63,000).
        #
        #     c_ratio     old      new       slope at -1.23e-03
        #    -0.00123  500.31    634.1      -250  ->  -95,641
        #    -0.00500  501.25    905.5
        #    -0.05000  512.20   1801.3
        #    -0.50000  601.37   2918.1
        #
        # This acts on c_max, i.e. the single worst point of the certificate.
        # The other sub-threshold points keep their own +1 each from the V>0
        # count; this is extra pressure on the one that sets the certificate,
        # not a new signal that the rest exist.
        #
        # roa_neg_weight is superseded and no longer read.
        roa_z = c_ratio / roa_c_target
        roa_penalty_value = roa_weight * jnp.where(
            roa_z >= 0.0,
            jnp.clip(1.0 - roa_z, 0.0, 1.0),
            1.0 + jnp.log1p(jnp.maximum(-roa_z, 0.0)),
        )

        grid_mse = (
            proper.astype(jnp.float64)
            + v_violations.astype(jnp.float64)
            + vdot_violations.astype(jnp.float64)
            + invalid
            + origin_penalty_value
            + gradient_penalty_value
            + probe_penalty_value
            + flat_penalty_value
            + axis_penalty_value
            + sat_penalty_value
            + roa_penalty_value
        )

        c_star = jnp.maximum(raw_boundary_min, 0.0)
        shortfall = jnp.maximum(
            0.0, (c_star_target - raw_boundary_min) / c_star_target
        )
        # GPU5_3: soft, monotone, no dead zone.
        #
        # GPU5_2 charged ``500000 * min(shortfall, 10)**2``. Two consequences,
        # both measured on runs 122232/122235/122252/122253:
        #   * SCALE. At the target a 5% drop in boundary_min cost 526 points
        #     while solving the ENTIRE Artstein problem from the champion's
        #     margin_max=0.0053 was worth 361. Selection could not see the CLF
        #     conditions at all; it only saw "do not move". Every champion of
        #     every run parked at boundary_min = 0.0204 against a 0.02 target.
        #   * SATURATION. Past shortfall=10 the term was a flat 5.0e7 with zero
        #     gradient. `fmax` sat at 5.20e5-5.38e5 in essentially every
        #     generation of every run and 41-63% of the population was stuck
        #     there, contributing nothing.
        #
        # The replacement is the codebase's own margin idiom -- quadratic near
        # zero, asymptotically linear, saturating at a cap -- with the cap set
        # to 10x the weight rather than 100x, plus a log tail that never goes
        # flat so a hopeless candidate still has a direction to move in.
        #   shortfall 0.05 ->    1.2      (was    526)
        #   shortfall 1.00 ->  263.7      (was 500000)
        #   shortfall 10.0 -> 4595.9      (was 5.0e7, flat from here)
        #   shortfall 1e3  -> 5138.2      (still strictly increasing)
        c_star_penalty_value = jnp.minimum(
            c_star_penalty_max * shortfall * shortfall / (shortfall + 1.0),
            c_star_penalty_cap,
        ) + c_star_tail_weight * jnp.log1p(shortfall)
        coverage = jnp.count_nonzero(
            (~jnp.isfinite(rollout_values))
            | ((rollout_values - origin_value) >= c_star)
        )
        pre_exact_valid = (
            grid_mse + c_star_penalty_value
            + coverage_weight * coverage.astype(jnp.float64)
        )
        pre_exact = jnp.where(valid_v, pre_exact_valid, 1.0e10)
        # GPU5_7: boundary_min / c_max / volume / u_required travel with the
        # score so the exact stage can price certificate depth and the logs
        # can report the certified region every generation.
        return (
            pre_exact,
            grid_mse,
            jnp.mean(gradient_magnitude),
            probe_ratio,
            raw_boundary_min,
            c_max,
            certified_volume,
            u_required,
            w_scale,
        )

    return kernel


@lru_cache(maxsize=8)
def _grid_kernel(grid_points: int):
    """Legacy scalar wrapper retained for parity tests and CPU fallback."""
    score = _score_kernel(grid_points)

    @jax.jit
    def kernel(
        opcodes,
        operands,
        literals,
        n_ops,
        parameters,
        points,
        *score_arguments,
    ):
        evaluated = _evaluate_program(
            opcodes, operands, literals, n_ops, parameters, points
        )
        return score(evaluated, *score_arguments)

    return kernel


@lru_cache(maxsize=8)
def _grid_batch_kernel(grid_points: int):
    """Vectorize the fixed-program kernel over constant vectors.

    The GP program and 21^4 grid are shared by the complete tuner population;
    only the padded constants row changes. This turns a complete tuner
    population into one compiled GPU call without changing its fitness
    formula.
    """
    scalar_kernel = _grid_kernel(grid_points)

    @jax.jit
    def kernel(
        opcodes,
        operands,
        literals,
        n_ops,
        parameters_batch,
        points,
        f_values,
        g_values,
        reference,
        radius_squared,
        probe_radius_squared,
        spacing,
        properness_weight,
        decay_rate,
        gradient_target_12,
        gradient_target_34,
        gradient_weight,
        origin_probe_k,
        origin_probe_penalty,
        c_star_target,
        c_star_penalty_max,
        c_star_penalty_cap,
        c_star_tail_weight,
        flat_floor,
        flat_weight,
        axis_mask,
        axis_b_target,
        axis_b_weight,
        axis_v_target,
        axis_v_weight,
        axis_tail_weight,
        origin_penalty_weight,
        coverage_weight,
        sat_weight,
        sat_u_target,
        roa_weight,
        roa_c_target,
        roa_neg_weight,
    ):
        return jax.vmap(
            lambda parameters: scalar_kernel(
                opcodes,
                operands,
                literals,
                n_ops,
                parameters,
                points,
                f_values,
                g_values,
                reference,
                radius_squared,
                probe_radius_squared,
                spacing,
                properness_weight,
                decay_rate,
                gradient_target_12,
                gradient_target_34,
                gradient_weight,
                origin_probe_k,
                origin_probe_penalty,
                c_star_target,
                c_star_penalty_max,
                c_star_penalty_cap,
                c_star_tail_weight,
                flat_floor,
                flat_weight,
                axis_mask,
                axis_b_target,
                axis_b_weight,
                axis_v_target,
                axis_v_weight,
                axis_tail_weight,
                origin_penalty_weight,
                coverage_weight,
                sat_weight,
                sat_u_target,
                roa_weight,
                roa_c_target,
                roa_neg_weight,
            )
        )(parameters_batch)

    return kernel


@lru_cache(maxsize=8)
def _score_batch_kernel(grid_points: int):
    """Score already-evaluated programs without materializing GP stacks."""
    scalar_score = _score_kernel(grid_points)

    @jax.jit
    def kernel(evaluated_batch, *score_arguments):
        return jax.vmap(
            lambda evaluated: scalar_score(evaluated, *score_arguments)
        )(evaluated_batch)

    return kernel


_CONTEXTS = {}


def _context(true_data, base):
    key = (id(true_data), tuple(len(true_data.get_input(i)) for i in range(4)))
    if key in _CONTEXTS:
        return _CONTEXTS[key]

    axes = [np.asarray(true_data.get_input(i), dtype=np.float64) for i in range(4)]
    if len(set(map(len, axes))) != 1:
        raise ValueError("GPU2 requires the same grid count on all four axes")
    grid_points = len(axes[0])
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.stack([component.ravel() for component in mesh], axis=1)

    x1, x2, x3, x4 = points.T
    sin_theta = -np.sin(x3)
    cos_theta = -np.cos(x3)
    denominator = 4.0 * (5.0 + 1.0 * (1.0 - cos_theta**2))
    f_values = np.stack(
        [
            x2,
            (-4.0 * -10.0 * cos_theta * sin_theta
             + 4.0 * (2.0 * x4**2 * sin_theta - x2)) / denominator,
            x4,
            (6.0 * -10.0 * 2.0 * sin_theta
             - 2.0 * cos_theta * (2.0 * x4**2 * sin_theta - x2)) / denominator,
        ],
        axis=1,
    )
    g_values = np.stack(
        [np.zeros_like(x1), 4.0 / denominator, np.zeros_like(x1),
         -2.0 * cos_theta / denominator],
        axis=1,
    )
    reference = base._reference_quadratic(*mesh).ravel()
    radius_squared = np.sum(points * points, axis=1)

    probes = []
    probe_r2 = []
    for axis in range(4):
        for offset in base.ORIGIN_PROBE_OFFSETS:
            for sign in (-1.0, 1.0):
                point = np.zeros(4)
                point[axis] = sign * offset
                probes.append(point)
                probe_r2.append(offset * offset)
    rollout_points = np.asarray(base.ROLLOUT_X0S, dtype=np.float64)
    all_points = np.concatenate(
        [points, np.zeros((1, 4)), np.asarray(probes), rollout_points], axis=0
    )
    evaluation_point_count = len(all_points)
    block_size = int(
        os.environ.get("SYMCLF_GPU2_PALLAS_BLOCK_SIZE", DEFAULT_BLOCK_SIZE)
    )
    padded_point_count = (
        (evaluation_point_count + block_size - 1) // block_size
    ) * block_size
    if padded_point_count > evaluation_point_count:
        all_points = np.pad(
            all_points,
            ((0, padded_point_count - evaluation_point_count), (0, 0)),
            mode="constant",
        )

    # GPU5_3: the c_star target is expressed as a MULTIPLE of the reference
    # form's own boundary minimum instead of an absolute number, so it is
    # invariant to how V is scaled and to the box size. Measured on the 21^4
    # cart-pole box the LQR reference has boundary_min = 0.0033631, while
    # GPU5_2 shipped an absolute C_STAR_TARGET of 0.02 -- 6.0x the reference's
    # own value. Every champion of runs 122232/122235/122253 sat within a few
    # percent of that number because the penalty around it was a cliff.
    reference_grid = np.asarray(reference, dtype=np.float64).reshape(
        (grid_points,) * 4
    )
    reference_boundary_min = float(
        np.min(
            [
                bound
                for axis in range(4)
                for bound in (
                    np.min(np.take(reference_grid, 0, axis=axis)),
                    np.min(
                        np.take(
                            reference_grid,
                            reference_grid.shape[axis] - 1,
                            axis=axis,
                        )
                    ),
                )
            ]
        )
    )

    # GPU5_5: the x1 axis {x2 = x3 = x4 = 0} as a mask over the flattened grid.
    # With an ODD grid count 0.0 is a sample value, so the axis is already a
    # subset of the fitness grid -- the axis term costs no extra evaluations.
    # An even grid count leaves the mask empty and the term switches itself off.
    axis_mask = (
        (np.abs(points[:, 1]) < 1.0e-12)
        & (np.abs(points[:, 2]) < 1.0e-12)
        & (np.abs(points[:, 3]) < 1.0e-12)
        & (radius_squared > base.CLF_ORIGIN_EXCLUDE_RADIUS ** 2)
    ).astype(np.float64)

    context = {
        "grid_points": grid_points,
        "reference_boundary_min": reference_boundary_min,
        "axis_mask": jax.device_put(axis_mask),
        "evaluation_point_count": evaluation_point_count,
        "pallas_block_size": block_size,
        "points": jax.device_put(all_points),
        "f_values": jax.device_put(f_values),
        "g_values": jax.device_put(g_values),
        "reference": jax.device_put(reference),
        "radius_squared": jax.device_put(radius_squared),
        "probe_radius_squared": jax.device_put(np.asarray(probe_r2)),
        "spacing": float(axes[0][1] - axes[0][0]),
    }
    _CONTEXTS[key] = context
    return context


def _pad_constants(values, expected):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) != expected:
        raise ValueError(f"expected {expected} constants, received {len(values)}")
    padded = np.zeros(MAX_CONSTANTS, dtype=np.float64)
    padded[:len(values)] = values
    return padded


def _pad_constants_batch(values, expected):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != expected:
        raise ValueError(
            f"expected a (batch, {expected}) constants array; received {values.shape}"
        )
    padded = np.zeros((values.shape[0], MAX_CONSTANTS), dtype=np.float64)
    padded[:, :expected] = values
    return padded


def _c_star_target(context, base):
    """GPU5_3: absolute c_star target derived from the reference form itself.

    ``C_STAR_TARGET_RATIO`` is a multiple of the reference quadratic's own
    boundary minimum on this box, so the requirement travels with the box and
    with however V happens to be scaled. Ratio 1.0 means "certify a sublevel
    set at least as large as the reference's". If ``C_STAR_TARGET_RATIO`` is
    absent the legacy absolute ``C_STAR_TARGET`` is used unchanged.
    """
    ratio = getattr(base, "C_STAR_TARGET_RATIO", None)
    if ratio is None:
        return float(base.C_STAR_TARGET)
    return float(ratio) * float(context["reference_boundary_min"])


def _common_kernel_arguments(program, context, base):
    """Arguments shared by scalar and batched grid kernels."""
    return (
        jax.device_put(program.opcodes),
        jax.device_put(program.operands),
        jax.device_put(program.literals),
        np.int32(program.n_ops),
        context["points"],
        context["f_values"],
        context["g_values"],
        context["reference"],
        context["radius_squared"],
        context["probe_radius_squared"],
        context["spacing"],
        float(base.PROPERNESS_PENALTY_WEIGHT),
        float(base.SHGO_DECAY_RATE),
        float(base.V_GRAD_X1X2_TARGET),
        float(base.V_GRAD_X3X4_TARGET),
        float(base.V_GRAD_WEIGHT),
        float(base.ORIGIN_PROBE_K),
        float(base.ORIGIN_PROBE_PENALTY if base.ORIGIN_PROBE_ENABLED else 0.0),
        _c_star_target(context, base),
        float(base.C_STAR_PENALTY_MAX),
        float(getattr(base, "C_STAR_PENALTY_CAP",
                      10.0 * float(base.C_STAR_PENALTY_MAX))),
        float(getattr(base, "C_STAR_TAIL_WEIGHT", 20.0)),
        float(getattr(base, "GPU5_7_FLAT_GRAD_FLOOR", 0.02)),
        float(getattr(base, "GPU5_7_FLAT_GRAD_WEIGHT", 1.0)),
        context["axis_mask"],
        float(getattr(base, "GPU5_7_AXIS_B_TARGET", 1.0e-3)),
        float(getattr(base, "GPU5_7_AXIS_B_WEIGHT", 300.0)),
        float(getattr(base, "GPU5_7_AXIS_V_TARGET", 1.0e-2)),
        float(getattr(base, "GPU5_7_AXIS_V_WEIGHT", 300.0)),
        float(getattr(base, "GPU5_7_AXIS_TAIL_WEIGHT", 2.0)),
        float(getattr(base, "GPU5_7_ORIGIN_PENALTY_WEIGHT", 0.0)),
        float(base.ROA_COVERAGE_WEIGHT),
        float(getattr(base, "GPU5_7_SAT_WEIGHT", 150.0)),
        float(getattr(base, "GPU5_7_SAT_U_TARGET", 1000.0)),
        float(getattr(base, "GPU5_7_ROA_WEIGHT", 500.0)),
        float(getattr(base, "GPU5_7_ROA_C_TARGET", 0.02)),
        float(getattr(base, "GPU5_7_ROA_NEG_WEIGHT", 250.0)),
    )


def _score_arguments(context, base):
    return (
        context["f_values"],
        context["g_values"],
        context["reference"],
        context["radius_squared"],
        context["probe_radius_squared"],
        context["spacing"],
        float(base.PROPERNESS_PENALTY_WEIGHT),
        float(base.SHGO_DECAY_RATE),
        float(base.V_GRAD_X1X2_TARGET),
        float(base.V_GRAD_X3X4_TARGET),
        float(base.V_GRAD_WEIGHT),
        float(base.ORIGIN_PROBE_K),
        float(base.ORIGIN_PROBE_PENALTY if base.ORIGIN_PROBE_ENABLED else 0.0),
        _c_star_target(context, base),
        float(base.C_STAR_PENALTY_MAX),
        float(getattr(base, "C_STAR_PENALTY_CAP",
                      10.0 * float(base.C_STAR_PENALTY_MAX))),
        float(getattr(base, "C_STAR_TAIL_WEIGHT", 20.0)),
        float(getattr(base, "GPU5_7_FLAT_GRAD_FLOOR", 0.02)),
        float(getattr(base, "GPU5_7_FLAT_GRAD_WEIGHT", 1.0)),
        context["axis_mask"],
        float(getattr(base, "GPU5_7_AXIS_B_TARGET", 1.0e-3)),
        float(getattr(base, "GPU5_7_AXIS_B_WEIGHT", 300.0)),
        float(getattr(base, "GPU5_7_AXIS_V_TARGET", 1.0e-2)),
        float(getattr(base, "GPU5_7_AXIS_V_WEIGHT", 300.0)),
        float(getattr(base, "GPU5_7_AXIS_TAIL_WEIGHT", 2.0)),
        float(getattr(base, "GPU5_7_ORIGIN_PENALTY_WEIGHT", 0.0)),
        float(base.ROA_COVERAGE_WEIGHT),
        float(getattr(base, "GPU5_7_SAT_WEIGHT", 150.0)),
        float(getattr(base, "GPU5_7_SAT_U_TARGET", 1000.0)),
        float(getattr(base, "GPU5_7_ROA_WEIGHT", 500.0)),
        float(getattr(base, "GPU5_7_ROA_C_TARGET", 0.02)),
        float(getattr(base, "GPU5_7_ROA_NEG_WEIGHT", 250.0)),
    )


@lru_cache(maxsize=8)
def _device_tuner_kernel(
    grid_points: int,
    population_size: int,
    generations: int,
    fusion_width: int,
    evaluation_point_count: int,
    block_size: int,
    interpret: bool,
):
    """One compiled, device-resident SEA run for several GP programs."""

    def score_rows(
        opcodes,
        operands,
        literals,
        n_ops,
        parameters,
        points,
        score_arguments,
    ):
        evaluated = evaluate_program_batch(
            opcodes,
            operands,
            literals,
            n_ops,
            parameters,
            points,
            max_stack_depth=MAX_STACK_DEPTH,
            block_size=block_size,
            interpret=interpret,
        )
        evaluated = evaluated[:, :evaluation_point_count]
        pre_exact = _score_batch_kernel(grid_points)(
            evaluated, *score_arguments
        )[0]
        return jnp.where(
            jnp.isfinite(pre_exact) & (pre_exact < 1.0e10),
            pre_exact,
            1.0e10,
        )

    @jax.jit
    def tune(
        opcodes,
        operands,
        literals,
        n_ops,
        dimensions,
        key,
        points,
        *score_arguments,
    ):
        valid_dimensions = (
            jnp.arange(MAX_CONSTANTS)[None, :] < dimensions[:, None]
        )
        key, initial_key = jax.random.split(key)
        initial = jax.random.uniform(
            initial_key,
            (fusion_width, population_size - 1, MAX_CONSTANTS),
            minval=-10.0,
            maxval=10.0,
            dtype=jnp.float64,
        )
        initial = jnp.where(valid_dimensions[:, None, :], initial, 0.0)
        ones = valid_dimensions[:, None, :].astype(jnp.float64)
        population = jnp.concatenate([initial, ones], axis=1)

        repeated_opcodes = jnp.repeat(
            opcodes[:, None, :], population_size, axis=1
        ).reshape((fusion_width * population_size, MAX_PROGRAM_NODES))
        repeated_operands = jnp.repeat(
            operands[:, None, :], population_size, axis=1
        ).reshape((fusion_width * population_size, MAX_PROGRAM_NODES))
        repeated_literals = jnp.repeat(
            literals[:, None, :], population_size, axis=1
        ).reshape((fusion_width * population_size, MAX_PROGRAM_NODES))
        repeated_n_ops = jnp.repeat(n_ops, population_size, axis=0)
        scores = score_rows(
            repeated_opcodes,
            repeated_operands,
            repeated_literals,
            repeated_n_ops,
            population.reshape(
                (fusion_width * population_size, MAX_CONSTANTS)
            ),
            points,
            score_arguments,
        ).reshape((fusion_width, population_size))

        candidate_indices = jnp.arange(fusion_width)

        def sea_step(_, state):
            population_, scores_, key_ = state
            best_indices = jnp.argmin(scores_, axis=1)
            best = population_[candidate_indices, best_indices]

            # Pagmo SEA mutates each coordinate with probability 1/d and
            # resamples until at least one active coordinate is selected.
            mutation = jnp.zeros(
                (fusion_width, MAX_CONSTANTS), dtype=jnp.bool_
            )
            complete = jnp.zeros((fusion_width,), dtype=jnp.bool_)

            def needs_mutation(loop_state):
                return ~jnp.all(loop_state[2])

            def draw_mutation(loop_state):
                key__, mutation__, complete__ = loop_state
                key__, draw_key = jax.random.split(key__)
                draws = jax.random.uniform(
                    draw_key, (fusion_width, MAX_CONSTANTS)
                )
                proposal = (
                    draws < (1.0 / dimensions.astype(jnp.float64))[:, None]
                ) & valid_dimensions
                proposal_complete = jnp.any(proposal, axis=1)
                mutation__ = jnp.where(
                    complete__[:, None], mutation__, proposal
                )
                complete__ = complete__ | proposal_complete
                return key__, mutation__, complete__

            key_, mutation, _ = jax.lax.while_loop(
                needs_mutation,
                draw_mutation,
                (key_, mutation, complete),
            )
            key_, value_key = jax.random.split(key_)
            replacement = jax.random.uniform(
                value_key,
                (fusion_width, MAX_CONSTANTS),
                minval=-10.0,
                maxval=10.0,
                dtype=jnp.float64,
            )
            offspring = jnp.where(mutation, replacement, best)
            offspring_scores = score_rows(
                opcodes,
                operands,
                literals,
                n_ops,
                offspring,
                points,
                score_arguments,
            )
            worst_indices = jnp.argmax(scores_, axis=1)
            worst_scores = scores_[candidate_indices, worst_indices]
            accept = offspring_scores <= worst_scores
            old_worst = population_[candidate_indices, worst_indices]
            population_ = population_.at[
                candidate_indices, worst_indices
            ].set(jnp.where(accept[:, None], offspring, old_worst))
            scores_ = scores_.at[candidate_indices, worst_indices].set(
                jnp.where(accept, offspring_scores, worst_scores)
            )
            return population_, scores_, key_

        population, scores, key = jax.lax.fori_loop(
            0,
            generations,
            sea_step,
            (population, scores, key),
        )
        best_indices = jnp.argmin(scores, axis=1)
        return (
            scores[candidate_indices, best_indices],
            population[candidate_indices, best_indices],
        )

    return tune


def gpu_tune_programs(
    expressions,
    n_constants,
    true_data,
    base,
    *,
    population_size,
    generations,
    fusion_width,
    seed,
):
    """Tune a fixed-width candidate group with one final host transfer."""
    require_gpu()
    expressions = list(expressions)
    dimensions = list(map(int, n_constants))
    real_count = len(expressions)
    if not expressions or real_count > fusion_width:
        raise ValueError("invalid GPU2 tuner group size")
    if any(dimension <= 0 or dimension > MAX_CONSTANTS for dimension in dimensions):
        raise ValueError("GPU2 tuner dimensions must be in [1, MAX_CONSTANTS]")

    while len(expressions) < fusion_width:
        expressions.append(expressions[0])
        dimensions.append(dimensions[0])
    programs = [encode_expression(str(expression)) for expression in expressions]
    opcodes = np.stack([program.opcodes for program in programs])
    operands = np.stack([program.operands for program in programs])
    literals = np.stack([program.literals for program in programs])
    n_ops = np.asarray([program.n_ops for program in programs], dtype=np.int32)

    context = _context(true_data, base)
    interpret = (
        os.environ.get("SYMCLF_GPU2_PALLAS_INTERPRET", "0") == "1"
        or jax.default_backend() != "gpu"
    )
    tune = _device_tuner_kernel(
        context["grid_points"],
        int(population_size),
        int(generations),
        int(fusion_width),
        int(context["evaluation_point_count"]),
        int(context["pallas_block_size"]),
        bool(interpret),
    )
    device_result = tune(
        jax.device_put(opcodes),
        jax.device_put(operands),
        jax.device_put(literals),
        jax.device_put(n_ops),
        jax.device_put(np.asarray(dimensions, dtype=np.int32)),
        jax.random.PRNGKey(np.uint32(seed)),
        context["points"],
        *_score_arguments(context, base),
    )
    champion_scores, champion_constants = jax.device_get(device_result)
    symbolic = np.asarray(
        [
            float(base._symbolic_structure_penalty(expression, []))
            for expression in expressions[:real_count]
        ],
        dtype=float,
    )
    champion_scores = np.asarray(
        champion_scores, dtype=float
    )[:real_count].copy()
    champion_scores += symbolic
    champions = [
        np.asarray(champion_constants[index], dtype=float)[:dimensions[index]].copy()
        for index in range(real_count)
    ]
    return champion_scores.tolist(), champions


def _encoded_program_batch(expressions, constants_rows):
    programs = [encode_expression(str(expression)) for expression in expressions]
    opcodes = np.stack([program.opcodes for program in programs])
    operands = np.stack([program.operands for program in programs])
    literals = np.stack([program.literals for program in programs])
    n_ops = np.asarray([program.n_ops for program in programs], dtype=np.int32)
    parameters = np.stack(
        [
            _pad_constants(constants, program.n_constants)
            for program, constants in zip(programs, constants_rows)
        ]
    )
    return opcodes, operands, literals, n_ops, parameters


def gpu_pre_exact_mse_many(expressions, constants_rows, true_data, base):
    """Score different GP programs/constant vectors in one Pallas GPU call."""
    require_gpu()
    expressions = list(expressions)
    constants_rows = list(constants_rows)
    if not expressions or len(expressions) != len(constants_rows):
        raise ValueError("expressions and constants_rows must have equal nonzero size")

    context = _context(true_data, base)
    arrays = _encoded_program_batch(expressions, constants_rows)
    interpret = (
        os.environ.get("SYMCLF_GPU2_PALLAS_INTERPRET", "0") == "1"
        or jax.default_backend() != "gpu"
    )
    row_chunk = max(
        4, int(os.environ.get("SYMCLF_GPU2_EVAL_ROW_CHUNK", "16"))
    )
    score_parts = [[] for _ in range(9)]   # GPU5_7: +w_scale
    for start in range(0, len(expressions), row_chunk):
        stop = min(start + row_chunk, len(expressions))
        real_rows = stop - start
        # Only two stable compiled row shapes are used: four for scalar/SEA
        # calls and row_chunk for population batches.
        target_rows = 4 if real_rows <= 4 else row_chunk
        chunk_arrays = []
        for array in arrays:
            chunk = array[start:stop]
            if real_rows < target_rows:
                chunk = np.concatenate(
                    [chunk, np.repeat(chunk[:1], target_rows - real_rows, axis=0)],
                    axis=0,
                )
            chunk_arrays.append(chunk)

        evaluated = evaluate_program_batch(
            *(jax.device_put(array) for array in chunk_arrays),
            context["points"],
            max_stack_depth=MAX_STACK_DEPTH,
            block_size=context["pallas_block_size"],
            interpret=interpret,
        )
        evaluated = evaluated[:, :context["evaluation_point_count"]]
        chunk_result = _score_batch_kernel(context["grid_points"])(
            evaluated, *_score_arguments(context, base)
        )
        for parts, values in zip(score_parts, chunk_result):
            parts.append(np.asarray(values, dtype=float)[:real_rows])
    result = tuple(np.concatenate(parts) for parts in score_parts)
    symbolic = np.asarray(
        [
            float(base._symbolic_structure_penalty(expression, constants))
            for expression, constants in zip(expressions, constants_rows)
        ],
        dtype=float,
    )
    pre_exact = np.asarray(result[0], dtype=float) + symbolic
    return pre_exact, {
        "grid_mse": np.asarray(result[1], dtype=float) + symbolic,
        "V_grad_mean": np.asarray(result[2], dtype=float),
        "origin_probe_ratio": np.asarray(result[3], dtype=float),
        # GPU5_7: certified-region diagnostics, per candidate.
        "boundary_min": np.asarray(result[4], dtype=float),
        "c_max": np.asarray(result[5], dtype=float),
        "certified_volume": np.asarray(result[6], dtype=float),
        "u_required": np.asarray(result[7], dtype=float),
        "w_scale": np.asarray(result[8], dtype=float),
    }


def gpu_pre_exact_mse(expression, constants, true_data, base):
    """Return GPU grid+c-star+coverage MSE before the exact-manifold gate."""
    values, details = gpu_pre_exact_mse_many(
        [expression], [constants], true_data, base
    )
    return float(values[0]), {
        "grid_mse": float(details["grid_mse"][0]),
        "V_grad_mean": float(details["V_grad_mean"][0]),
        "origin_probe_ratio": float(details["origin_probe_ratio"][0]),
        "boundary_min": float(details["boundary_min"][0]),
        "c_max": float(details["c_max"][0]),
        "certified_volume": float(details["certified_volume"][0]),
        "u_required": float(details["u_required"][0]),
        "w_scale": float(details["w_scale"][0]),
    }


def gpu_pre_exact_mse_batch(expression, constants_batch, true_data, base):
    """Grid-only fitness for a batch of tuner decision vectors."""
    program = encode_expression(str(expression))
    constant_values = np.asarray(constants_batch, dtype=np.float64)
    if constant_values.ndim != 2 or constant_values.shape[1] != program.n_constants:
        raise ValueError(
            f"expected a (batch, {program.n_constants}) constants array; "
            f"received {constant_values.shape}"
        )
    values, _ = gpu_pre_exact_mse_many(
        [expression] * len(constant_values),
        list(constant_values),
        true_data,
        base,
    )
    return np.where(np.isfinite(values) & (values < 1.0e10), values, 1.0e10)


def make_gpu_evaluators(base, exact_check):
    """Create drop-in MSE/tuning functions bound to the GPU2 Evaluate shim."""
    tuner_call = 0

    def evaluate_grid_mse(ind2mse, constants, true_data):
        try:
            mse, _ = gpu_pre_exact_mse(ind2mse, constants, true_data, base)
            return float(mse) if np.isfinite(mse) and mse < 1.0e10 else 1.0e10
        except GPUUnavailableError:
            raise
        except Exception:
            return 1.0e10

    def evaluate_grid_mse_batch(ind2mse, constants_batch, true_data):
        constants_batch = np.asarray(constants_batch, dtype=float)
        try:
            return gpu_pre_exact_mse_batch(
                ind2mse, constants_batch, true_data, base
            )
        except GPUUnavailableError:
            raise
        except Exception:
            return np.full(constants_batch.shape[0], 1.0e10)

    def evaluate_mse(ind2mse, constants, true_data):
        """Final candidate score: grid first, then exactly one exact check."""
        try:
            mse = evaluate_grid_mse(ind2mse, constants, true_data)
            if mse >= 1.0e10:
                return 1.0e10
            if base.MANIFOLD_EXACT_CHECK_ENABLED:
                if mse < base.MANIFOLD_EXACT_MSE_GATE:
                    try:
                        result = exact_check(
                            base._sympy_expression(ind2mse, constants),
                            base.fSR,
                            base.GSR,
                            bounds=base.SHGO_BOUNDS,
                            gamma1=base.CLF_GAMMA1,
                            margin_tol=base.MANIFOLD_MARGIN_TOL_EXACT,
                            origin_tol=base.CLF_ORIGIN_EXCLUDE_RADIUS,
                            scan_axes=(0, 1, 2, 3),
                        )
                    except Exception:
                        result = None
                    penalty = 0.0
                    if result is not None and result.status == "ok":
                        if result.n_violations > 0:
                            penalty = min(
                                base.MANIFOLD_EXACT_PENALTY_MAX,
                                10.0
                                + result.n_violations * 0.01
                                + base._manifold_margin_penalty(
                                    result.margin_max,
                                    base.MANIFOLD_EXACT_MARGIN_WEIGHT,
                                    base.MANIFOLD_EXACT_MARGIN_PENALTY_MAX,
                                ),
                            )
                        elif result.n_roots == 0:
                            penalty = base.MANIFOLD_VACUOUS_PENALTY
                        if result.n_roots > 0:
                            penalty += base._exact_near0_penalty(result.margin_max)
                    mse += penalty
                else:
                    mse += float(base.MANIFOLD_EXACT_MSE_GATE) * 1.3
            return float(mse) if np.isfinite(mse) else 1.0e10
        except GPUUnavailableError:
            raise
        except Exception:
            return 1.0e10

    def eval_mse_and_tune_constants(
        individual, num_consts, toolbox, true_data, ind2MSE, pass_cost=None
    ):
        nonlocal tuner_call
        del individual, toolbox, pass_cost
        if num_consts > 0:
            random_population = int(
                os.environ.get("SYMCLF_GPU2_TUNER_RANDOM_POPULATION", "35")
            )
            generations = int(
                os.environ.get("SYMCLF_GPU2_TUNER_GENERATIONS", "5")
            )
            base_seed = int(os.environ.get("SYMCLF_GPU2_TUNER_SEED", "0"))
            seed = (base_seed + os.getpid() + tuner_call) & 0xFFFFFFFF
            tuner_call += 1
            _, champions = gpu_tune_programs(
                [ind2MSE],
                [num_consts],
                true_data,
                base,
                population_size=random_population + 1,
                generations=generations,
                fusion_width=1,
                seed=seed,
            )
            constants = champions[0]
        else:
            constants = np.empty(0, dtype=float)
        return evaluate_mse(ind2MSE, constants, true_data), constants

    return evaluate_mse, eval_mse_and_tune_constants
