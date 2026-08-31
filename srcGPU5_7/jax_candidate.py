"""Compile-once-per-tree-shape JAX kernels for a sympy candidate V.

Three of the accepted speed items live here:

* **Compile once per tree shape** — every Float constant in V is abstracted
  to a parameter symbol, so a tree whose constants are being retuned (or a
  clone with identical structure) maps to the SAME jitted kernels; constants
  arrive as a runtime array and never trigger recompilation.
* **Exact derivatives without symbolic diff** — a = grad(V)·f and
  b = grad(V)·g come from ``jax.grad`` of the compiled V: machine-precision
  exact (it differentiates the actual computation graph; no finite
  differences anywhere), with none of sympy diff's big-tree blowup and none
  of complex-step's 4x complex-arithmetic overhead.
* **Fused a,b evaluation** — one jitted function returns both; XLA's CSE
  shares the gradient subgraph between them.
* **GPU-only constrained polishing** — all selected b=0 roots are optimized
  together by a fixed-shape SQP/KKT kernel with a damped-BFGS Hessian,
  parallel merit-function line search, and Newton constraint restoration.
  There are no per-root Python callbacks or SciPy solves in the GPU2 path.

Accuracy contract: everything is float64 (enforced by ``srcGPU2.__init__``).
"""

from collections import OrderedDict

import srcGPU5_7  # noqa: F401  (configure JAX before backend initialization)

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp
from sympy import Symbol, lambdify, symbols

_BUNDLE_CACHE: OrderedDict = OrderedDict()
_BUNDLE_CACHE_MAX = 128
_DYN_CACHE: dict = {}


def abstract_floats(expr):
    """Replace every Float atom of ``expr`` with a parameter symbol.

    Returns (template, param_symbols, values). Two expressions with the same
    tree shape but different tuned constants share one template — and hence
    one set of compiled kernels.
    """
    floats = []
    for atom in sp.preorder_traversal(expr):
        if isinstance(atom, sp.Float) and atom not in floats:
            floats.append(atom)
    params = [Symbol(f"_p{i}") for i in range(len(floats))]
    template = expr.xreplace(dict(zip(floats, params)))
    values = np.asarray([float(f) for f in floats], dtype=np.float64)
    return template, params, values


def _dynamics_jax(fSR, GSR, n, input_index):
    """JAX-compiled drift components and active G column (built once)."""
    key = (id(fSR), id(GSR), int(n), int(input_index))
    if key not in _DYN_CACHE:
        xs = symbols(f"x1:{n + 1}")
        f_list = list(fSR(*xs))
        g_list = [GSR(*xs)[i, input_index] for i in range(n)]
        _DYN_CACHE[key] = (
            lambdify(xs, f_list, modules="jax"),
            lambdify(xs, g_list, modules="jax"),
        )
    return _DYN_CACHE[key]


class CandidateBundle:
    """Jitted kernels for one tree template.

    All kernels take the constants vector ``c`` as a runtime argument:
    retuning constants reuses every compiled kernel unchanged.
    """

    def __init__(self, template, params, n, fj, gj):
        self.n = int(n)
        self.n_params = len(params)
        xs = symbols(f"x1:{n + 1}")
        Vj = lambdify(tuple(xs) + tuple(params), template, modules="jax")
        k = len(params)

        def V_s(x, c):
            return Vj(*[x[i] for i in range(n)], *[c[j] for j in range(k)])

        def ab_s(x, c):
            g = jax.grad(V_s, argnums=0)(x, c)
            comps = [x[i] for i in range(n)]
            fv = jnp.stack([jnp.asarray(v, dtype=jnp.float64)
                            for v in fj(*comps)])
            gv = jnp.stack([jnp.asarray(v, dtype=jnp.float64)
                            for v in gj(*comps)])
            return g @ fv, g @ gv

        self._ab_s = ab_s
        self.ab_batch = jax.jit(jax.vmap(ab_s, in_axes=(0, None)))
        self.b_batch = jax.jit(
            jax.vmap(lambda x, c: ab_s(x, c)[1], in_axes=(0, None))
        )
        self.a_scalar = jax.jit(lambda x, c: ab_s(x, c)[0])
        self.b_scalar = jax.jit(lambda x, c: ab_s(x, c)[1])
        # Exact jacobians for constrained polishing (same role as the exact
        # module's ga_fn/gb_fn — one more derivative level, via jax.grad).
        self.ga_scalar = jax.jit(
            jax.grad(lambda x, c: ab_s(x, c)[0], argnums=0)
        )
        self.gb_scalar = jax.jit(
            jax.grad(lambda x, c: ab_s(x, c)[1], argnums=0)
        )

        ga_s = jax.grad(lambda x, c: ab_s(x, c)[0], argnums=0)
        gb_s = jax.grad(lambda x, c: ab_s(x, c)[1], argnums=0)

        def polish_batch(
            seeds,
            c,
            lower,
            upper,
            r0_sq,
            b_tol,
            step_size,
            iters,
            projection_steps,
            line_search_steps,
        ):
            """Batched nonlinear SQP on ``b(x)=0`` on the GPU.

            Each start has its own 4x4 damped-BFGS Hessian. Its equality QP
            is solved from the 6x6 KKT system (b=0 plus the active puncture
            boundary), line-search candidates are evaluated together, and
            every accepted trial is restored by batched Newton projection.
            """

            dtype = seeds.dtype
            eps = jnp.asarray(1.0e-24, dtype=dtype)

            def abg_batch(x):
                def one(z):
                    av, bv = ab_s(z, c)
                    return av, bv, ga_s(z, c), gb_s(z, c)

                return jax.vmap(one)(x)

            def enforce_constraints(x):
                """Alternating Newton projection for b=0 and ||x||>=r0."""

                def project_once(_, points):
                    _, bv, _, gbv = abg_batch(points)
                    norm_sq = jnp.sum(points * points, axis=1)
                    active = norm_sq < r0_sq
                    radial = 2.0 * points

                    m11 = jnp.sum(gbv * gbv, axis=1) + eps
                    m12 = jnp.sum(gbv * radial, axis=1)
                    m22 = jnp.sum(radial * radial, axis=1) + eps
                    c2 = jnp.where(active, norm_sq - r0_sq, 0.0)
                    determinant = m11 * m22 - m12 * m12 + eps

                    lambda_both_1 = (bv * m22 - c2 * m12) / determinant
                    lambda_both_2 = (c2 * m11 - bv * m12) / determinant
                    lambda_1 = jnp.where(active, lambda_both_1, bv / m11)
                    lambda_2 = jnp.where(active, lambda_both_2, 0.0)
                    correction = (
                        lambda_1[:, None] * gbv
                        + lambda_2[:, None] * radial
                    )
                    projected = jnp.clip(points - correction, lower, upper)
                    return jnp.where(jnp.isfinite(projected), projected, points)

                return jax.lax.fori_loop(
                    0, projection_steps, project_once, jnp.clip(x, lower, upper)
                )

            def feasible_values(x):
                av, bv, _, _ = abg_batch(x)
                norm_sq = jnp.sum(x * x, axis=1)
                valid = (
                    jnp.isfinite(av)
                    & jnp.isfinite(bv)
                    & (jnp.abs(bv) <= b_tol)
                    & (norm_sq > r0_sq)
                )
                return av, valid

            x = enforce_constraints(seeds)
            initial_a, initial_valid = feasible_values(x)
            best_a = jnp.where(initial_valid, initial_a, -jnp.inf)
            best_x = x
            factors = 0.5 ** jnp.arange(line_search_steps, dtype=dtype)
            identity = jnp.eye(n, dtype=dtype)
            hessian = jnp.broadcast_to(identity, (x.shape[0], n, n))

            def ascent_step(iteration, state):
                points, best_points, best_values, hessians = state
                av, bv, gav, gbv = abg_batch(points)
                grad_f = -gav
                gb_norm_sq = jnp.sum(gbv * gbv, axis=1) + eps
                norm_sq = jnp.sum(points * points, axis=1)
                radial = 2.0 * points
                sphere_active = norm_sq <= r0_sq * (1.0 + 1.0e-5)
                constraint_jac = jnp.stack(
                    [gbv, jnp.where(sphere_active[:, None], radial, 0.0)],
                    axis=1,
                )
                constraint_value = jnp.stack(
                    [bv, jnp.where(sphere_active, norm_sq - r0_sq, 0.0)],
                    axis=1,
                )

                # Equality-QP step: min 1/2 d' B d + grad(f)'d,
                # subject to J d + c = 0. The inactive sphere row is
                # decoupled with unit dual regularization.
                dual_regularization = jnp.stack(
                    [
                        jnp.full_like(bv, 1.0e-12),
                        jnp.where(sphere_active, 1.0e-12, 1.0),
                    ],
                    axis=1,
                )
                top = jnp.concatenate(
                    [hessians, jnp.swapaxes(constraint_jac, 1, 2)], axis=2
                )
                dual_block = -jax.vmap(jnp.diag)(dual_regularization)
                bottom = jnp.concatenate([constraint_jac, dual_block], axis=2)
                kkt = jnp.concatenate([top, bottom], axis=1)
                rhs = -jnp.concatenate([grad_f, constraint_value], axis=1)
                solution = jnp.linalg.solve(kkt, rhs[..., None])[..., 0]
                direction = solution[:, :n]
                multipliers = solution[:, n:]

                # Singular KKT systems fall back to the exact tangent
                # gradient, still respecting b=0 after restoration.
                tangent = gav - (
                    jnp.sum(gav * gbv, axis=1) / gb_norm_sq
                )[:, None] * gbv
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
                multipliers = jnp.where(
                    solved[:, None], multipliers, 0.0
                )
                direction_norm = jnp.linalg.norm(direction, axis=1, keepdims=True)
                trust_scale = jnp.minimum(
                    1.0, step_size / jnp.maximum(direction_norm, 1.0e-18)
                )
                direction = direction * trust_scale
                trials = (
                    points[:, None, :]
                    + factors[None, :, None] * direction[:, None, :]
                )
                trial_shape = trials.shape
                trials = enforce_constraints(trials.reshape((-1, n)))
                trials = trials.reshape(trial_shape)

                # Keep the current point as the zero-step line-search choice.
                choices = jnp.concatenate([points[:, None, :], trials], axis=1)
                flat_choices = choices.reshape((-1, n))
                choice_a, choice_b, _, _ = abg_batch(flat_choices)
                choice_a = choice_a.reshape((points.shape[0], -1))
                choice_b = choice_b.reshape((points.shape[0], -1))
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
                    + penalty[:, None]
                    * (jnp.abs(choice_b) + sphere_error)
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
                best_points = jnp.where(improve[:, None], new_points, best_points)
                best_values = jnp.where(improve, new_a, best_values)

                # Powell-damped BFGS update of the Lagrangian Hessian.
                old_grad_lagrangian = (
                    grad_f
                    + multipliers[:, :1] * gbv
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
                displacement = new_points - points
                gradient_change = new_grad_lagrangian - old_grad_lagrangian
                hs = jnp.einsum("bij,bj->bi", hessians, displacement)
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
                    hessians
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
                hessians = jnp.where(
                    update_valid[:, None, None], candidate_hessian, hessians
                )
                hessians = 0.5 * (
                    hessians + jnp.swapaxes(hessians, 1, 2)
                )
                return new_points, best_points, best_values, hessians

            x, best_x, best_a, _ = jax.lax.fori_loop(
                0, iters, ascent_step, (x, best_x, best_a, hessian)
            )
            # A final stronger restoration is cheap and makes the acceptance
            # test use the requested b tolerance, not the search tolerance.
            x = enforce_constraints(x)
            final_a, final_valid = feasible_values(x)
            improve = final_valid & (final_a > best_a)
            best_x = jnp.where(improve[:, None], x, best_x)
            best_a = jnp.where(improve, final_a, best_a)
            return best_x, jnp.isfinite(best_a), best_a

        self.polish_batch = jax.jit(
            polish_batch,
            static_argnums=(7, 8, 9),
        )

        def bisect_axis(P, onehot, lo, hi, blo, c, iters):
            """Fixed-count bisection along one axis for all brackets at once.

            ``iters`` is static (identical count to the CPU exact check; no
            early stopping — accuracy contract).
            """

            def body(_, state):
                lo_, hi_, blo_ = state
                mid = 0.5 * (lo_ + hi_)
                Pm = P * (1.0 - onehot) + mid[:, None] * onehot
                bm = jax.vmap(lambda x: ab_s(x, c)[1])(Pm)
                left = jnp.sign(bm) * jnp.sign(blo_) > 0
                return (
                    jnp.where(left, mid, lo_),
                    jnp.where(left, hi_, mid),
                    jnp.where(left, bm, blo_),
                )

            lo, hi, blo = jax.lax.fori_loop(0, iters, body, (lo, hi, blo))
            return P * (1.0 - onehot) + (0.5 * (lo + hi))[:, None] * onehot

        def bisect_line(P0, D, lo, hi, blo, c, iters):
            """Fixed-count bisection along arbitrary lines x = P0 + t*D."""

            def body(_, state):
                lo_, hi_, blo_ = state
                mid = 0.5 * (lo_ + hi_)
                Xm = P0 + mid[:, None] * D
                bm = jax.vmap(lambda x: ab_s(x, c)[1])(Xm)
                left = jnp.sign(bm) * jnp.sign(blo_) > 0
                return (
                    jnp.where(left, mid, lo_),
                    jnp.where(left, hi_, mid),
                    jnp.where(left, bm, blo_),
                )

            lo, hi, blo = jax.lax.fori_loop(0, iters, body, (lo, hi, blo))
            return P0 + (0.5 * (lo + hi))[:, None] * D

        self.bisect_axis = jax.jit(bisect_axis, static_argnums=6)
        self.bisect_line = jax.jit(bisect_line, static_argnums=6)


def get_bundle(V_expr, fSR, GSR, input_index, n):
    """Bundle for ``V_expr``, cached by tree template.

    Returns (bundle, constant_values). A cache hit skips lambdify and every
    jit compile — the constants-tuning loop and structural clones pay the
    build exactly once.
    """
    template, params, values = abstract_floats(V_expr)
    key = (sp.srepr(template), id(fSR), id(GSR), int(input_index), int(n))
    if key in _BUNDLE_CACHE:
        _BUNDLE_CACHE.move_to_end(key)
        return _BUNDLE_CACHE[key], values
    fj, gj = _dynamics_jax(fSR, GSR, n, input_index)
    bundle = CandidateBundle(template, params, n, fj, gj)
    _BUNDLE_CACHE[key] = bundle
    while len(_BUNDLE_CACHE) > _BUNDLE_CACHE_MAX:
        _BUNDLE_CACHE.popitem(last=False)
    return bundle, values
