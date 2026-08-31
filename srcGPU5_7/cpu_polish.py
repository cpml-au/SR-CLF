"""Numba CPU evaluator of a, b, grad(a), grad(b) for the SLSQP polish.

Replaces the per-check sympy ``diff``/``lambdify`` build (measured ~0.6 s for a
len-900 tree, plus ~330 us per callback eval) with:

* a compile-once numba dual-number interpreter that returns V's value,
  gradient and Hessian at one point directly from the runtime bytecode
  (same opcode data the GPU scan uses), and
* the fixed cart-pole drift/control and their Jacobians, built once from the
  reference ``fSR``/``GSR`` at import.

The assembly is exactly ``srcGPU5_7.runtime_exact_candidate._abg_one``:

    a  = grad(V) . drift
    b  = grad(V) . control
    ga = H(V) . drift   + Jf^T . grad(V)
    gb = H(V) . control + Jg^T . grad(V)

so the numbers are the same exact derivatives the reference SLSQP polish uses
(autodiff == symbolic derivatives), only evaluated from bytecode instead of a
freshly lambdified sympy graph. The numba kernel compiles once (opcodes are
runtime arguments, not code), then every candidate reuses it.
"""

from functools import lru_cache

import numpy as np
from numba import njit
from sympy import Matrix, lambdify, symbols

from srcGPU5_7.grid_fitness import (
    ADD,
    AQ,
    EXP,
    MAX_CONSTANTS,
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

_N = 4  # cart-pole state dimension


@njit(cache=True, fastmath=False)
def _product(sval, sgrad, shess, l, vr, gr, hr):
    """In-place stack[l] <- stack[l] * (vr, gr, hr) dual; product rule.

    gr, hr are the value-right gradient/Hessian (length-4 / 4x4). Uses saved
    copies of the left operand so the update is not corrupted by aliasing.
    """
    vl = sval[l]
    gl = sgrad[l].copy()
    hl = shess[l].copy()
    sval[l] = vl * vr
    for a in range(_N):
        for b in range(_N):
            shess[l, a, b] = (
                hl[a, b] * vr
                + hr[a, b] * vl
                + gl[a] * gr[b]
                + gr[a] * gl[b]
            )
    for a in range(_N):
        sgrad[l, a] = gl[a] * vr + gr[a] * vl


@njit(cache=True, fastmath=False)
def _dual2(opcodes, operands, literals, n_ops, parameters, x):
    """Forward value/gradient/Hessian of V at one point x (4-vector).

    Mirrors ``srcGPU5_7.runtime_exact_candidate._dual2_point`` exactly.
    """
    sval = np.zeros(MAX_STACK_DEPTH)
    sgrad = np.zeros((MAX_STACK_DEPTH, _N))
    shess = np.zeros((MAX_STACK_DEPTH, _N, _N))
    p = 0
    for i in range(n_ops):
        op = opcodes[i]
        operand = operands[i]
        if op == PUSH_X:
            sval[p] = x[operand]
            for a in range(_N):
                sgrad[p, a] = 1.0 if a == operand else 0.0
                for b in range(_N):
                    shess[p, a, b] = 0.0
            p += 1
        elif op == PUSH_PARAMETER:
            sval[p] = parameters[operand]
            for a in range(_N):
                sgrad[p, a] = 0.0
                for b in range(_N):
                    shess[p, a, b] = 0.0
            p += 1
        elif op == PUSH_LITERAL:
            sval[p] = literals[i]
            for a in range(_N):
                sgrad[p, a] = 0.0
                for b in range(_N):
                    shess[p, a, b] = 0.0
            p += 1
        elif op == ADD or op == SUB:
            l = p - 2
            r = p - 1
            s = 1.0 if op == ADD else -1.0
            sval[l] = sval[l] + s * sval[r]
            for a in range(_N):
                sgrad[l, a] = sgrad[l, a] + s * sgrad[r, a]
                for b in range(_N):
                    shess[l, a, b] = shess[l, a, b] + s * shess[r, a, b]
            p -= 1
        elif op == MUL:
            l = p - 2
            r = p - 1
            _product(sval, sgrad, shess, l, sval[r], sgrad[r], shess[r])
            p -= 1
        elif op == AQ:
            l = p - 2
            r = p - 1
            y = sval[r]
            scale = 1.0 / np.sqrt(1.0 + y * y)
            scale_p = -y * scale ** 3
            scale_s = (2.0 * y * y - 1.0) * scale ** 5
            gr = sgrad[r]
            hr = shess[r]
            grad_scale = np.empty(_N)
            hess_scale = np.empty((_N, _N))
            for a in range(_N):
                grad_scale[a] = scale_p * gr[a]
                for b in range(_N):
                    hess_scale[a, b] = scale_p * hr[a, b] + scale_s * gr[a] * gr[b]
            _product(sval, sgrad, shess, l, scale, grad_scale, hess_scale)
            p -= 1
        elif op == NEG:
            idx = p - 1
            sval[idx] = -sval[idx]
            for a in range(_N):
                sgrad[idx, a] = -sgrad[idx, a]
                for b in range(_N):
                    shess[idx, a, b] = -shess[idx, a, b]
        elif op == SIN:
            idx = p - 1
            v = sval[idx]
            first = np.cos(v)
            second = -np.sin(v)
            gl = sgrad[idx].copy()
            for a in range(_N):
                for b in range(_N):
                    shess[idx, a, b] = first * shess[idx, a, b] + second * gl[a] * gl[b]
            for a in range(_N):
                sgrad[idx, a] = first * gl[a]
            sval[idx] = np.sin(v)
        elif op == EXP:
            idx = p - 1
            v = np.exp(sval[idx])
            gl = sgrad[idx].copy()
            for a in range(_N):
                for b in range(_N):
                    shess[idx, a, b] = v * shess[idx, a, b] + v * gl[a] * gl[b]
            for a in range(_N):
                sgrad[idx, a] = v * gl[a]
            sval[idx] = v
    idx = p - 1
    return sval[idx], sgrad[idx].copy(), shess[idx].copy()


@lru_cache(maxsize=4)
def _field_callables(input_index, fSR, GSR):
    """Build cart-pole drift/control and Jacobians once from the reference
    dynamics. Cached on (input_index, fSR, GSR) identity."""
    xs = symbols(f"x1:{_N + 1}")
    drift = fSR(*xs)
    control = Matrix([GSR(*xs)[i, input_index] for i in range(_N)])
    state = list(xs)
    jac_d = drift.jacobian(state)
    jac_c = control.jacobian(state)
    f_d = lambdify(xs, list(drift), "numpy")
    f_c = lambdify(xs, list(control), "numpy")
    f_jd = lambdify(xs, jac_d.tolist(), "numpy")
    f_jc = lambdify(xs, jac_c.tolist(), "numpy")

    def fields(z):
        drift_v = np.asarray(f_d(*z), dtype=float).reshape(_N)
        control_v = np.asarray(f_c(*z), dtype=float).reshape(_N)
        jd = np.asarray(f_jd(*z), dtype=float).reshape(_N, _N)
        jc = np.asarray(f_jc(*z), dtype=float).reshape(_N, _N)
        return drift_v, control_v, jd, jc

    return fields


def make_cpu_polish_callables(expression, constants, fSR, GSR, gamma1=0.0,
                              input_index=1):
    """Return SciPy-ready {a_fn, m_fn, ga_fn, b_fn, gb_fn} for the polish.

    Same call surface as the sympy ``_build_reference_callables`` output, but
    evaluated via the numba bytecode dual-number kernel (no per-check sympy
    build). A 1-entry cache reuses (a,b,ga,gb) when SciPy queries fun and jac
    at the same point.
    """
    program = encode_expression(str(expression))
    opcodes = np.asarray(program.opcodes, dtype=np.int64)
    operands = np.asarray(program.operands, dtype=np.int64)
    literals = np.asarray(program.literals, dtype=np.float64)
    n_ops = int(program.n_ops)
    params = np.zeros(MAX_CONSTANTS, dtype=np.float64)
    cvals = np.asarray(constants, dtype=np.float64).reshape(-1)
    params[: cvals.shape[0]] = cvals

    fields = _field_callables(int(input_index), fSR, GSR)
    cache = {"z": None, "abg": None}

    def _abg(z):
        z = np.asarray(z, dtype=np.float64).reshape(_N)
        cz = cache["z"]
        if cz is not None and cz[0] == z[0] and cz[1] == z[1] \
                and cz[2] == z[2] and cz[3] == z[3]:
            return cache["abg"]
        _, grad, hess = _dual2(opcodes, operands, literals, n_ops, params, z)
        drift, control, jd, jc = fields(z)
        a = float(grad @ drift)
        b = float(grad @ control)
        ga = hess @ drift + jd.T @ grad
        gb = hess @ control + jc.T @ grad
        out = (a, b, ga, gb)
        cache["z"] = z.copy()
        cache["abg"] = out
        return out

    g1 = float(gamma1)
    return {
        "a_fn": lambda *z: _abg(np.asarray(z))[0],
        "m_fn": lambda *z: _abg(np.asarray(z))[0] + g1,
        "ga_fn": lambda *z: _abg(np.asarray(z))[2],
        "b_fn": lambda *z: _abg(np.asarray(z))[1],
        "gb_fn": lambda *z: _abg(np.asarray(z))[3],
    }
