"""Numba evaluators for a candidate V on an (n states, m inputs) control-affine system.

Ported from examples/DET4Fast4D_V8/evalnb.py and src4DCartPoleV8/cpu_polish.py (the value/
gradient/Hessian bytecode interpreters and the sympy-compiled dynamics), with ONE change of
substance: b = grad(V)' G is a VECTOR with one entry per input, so

    a      = grad(V) . f                      (scalar, unchanged)
    b_k    = grad(V) . G[:, k]                (k = 0 .. m-1)
    grad a = H f + Jf^T grad V                (unchanged)
    grad b_k = H G[:, k] + JG_k^T grad V      (per input)

and the DET4 rate uses the bounded-input optimum  min_u (a + b.u):
    l1 (|u_i| <= rho, V8's per-input clip) -> a - rho * sum_k |b_k|
    l2 (||u|| <= rho)                      -> a - rho * ||b||_2
For m = 1 both reduce to the 4-D formula a - rho |b|, so the acceptance system is unaffected.
"""
from __future__ import annotations

import math
import os

import numpy as np
import sympy as sp

from core import config

_sys = config.system()
_N = int(_sys.N_STATES)
_M = int(_sys.N_INPUTS)
_L1 = config.B_NORM == "l1"

from numba import njit, prange  # noqa: E402  (core/__init__ set NUMBA_CACHE_DIR)

from core.encode import (  # noqa: E402
    ADD, AQ, EXP, MAX_CONSTANTS, MAX_STACK_DEPTH, MUL, NEG, PUSH_LITERAL,
    PUSH_PARAMETER, PUSH_X, SIN, SUB, encode_expression, program_arrays,
)

# offsets inside the flat tuple returned by _FIELDS
_OFF_G = _N                      # g[k, i] at _OFF_G + k * _N + i
_OFF_JF = _N + _M * _N           # Jf[i, j] at _OFF_JF + i * _N + j
_OFF_JG = _OFF_JF + _N * _N      # JG[k, i, j] at _OFF_JG + k * _N * _N + i * _N + j

_K_FIELDS = _N + _M * _N + _N * _N + _M * _N * _N
_FIELDS = None


def set_dynamics(fSR=None, GSR=None):
    """Compile fields(x, out) -> writes (f, G columns, Jf, JG) from the system's sympy dynamics."""
    global _FIELDS
    if _FIELDS is not None:
        return _FIELDS
    fSR = _sys.fSR if fSR is None else fSR
    GSR = _sys.GSR if GSR is None else GSR
    xs = sp.symbols(f"x1:{_N + 1}")
    drift = sp.Matrix(fSR(*xs))
    G = sp.Matrix(GSR(*xs))
    if G.shape != (_N, _M):
        raise ValueError(f"G must be {_N}x{_M}, got {G.shape}")
    jd = drift.jacobian(list(xs))
    parts = [sp.pycode(e) for e in list(drift)]
    for k in range(_M):
        parts += [sp.pycode(G[i, k]) for i in range(_N)]
    parts += [sp.pycode(jd[i, j]) for i in range(_N) for j in range(_N)]
    for k in range(_M):
        jc = sp.Matrix([G[i, k] for i in range(_N)]).jacobian(list(xs))
        parts += [sp.pycode(jc[i, j]) for i in range(_N) for j in range(_N)]
    if len(parts) != _K_FIELDS:
        raise RuntimeError(f"field count {len(parts)} != {_K_FIELDS}")
    unpack = "\n    ".join(f"x{i + 1} = x[{i}]" for i in range(_N))
    body = "\n    ".join(f"out[{i}] = float({p})" for i, p in enumerate(parts))
    src = f"def fields(x, out):\n    {unpack}\n    {body}\n    return out\n"
    ns = {"math": math}
    exec(src, ns)  # noqa: S102  (our own generated code)
    _FIELDS = njit(cache=False)(ns["fields"])
    return _FIELDS


set_dynamics()


@njit(cache=True, fastmath=False, nogil=True)
def _dual1(opcodes, operands, literals, n_ops, parameters, x):
    """Value and gradient of V at x from the bytecode."""
    sval = np.zeros(MAX_STACK_DEPTH)
    sgrad = np.zeros((MAX_STACK_DEPTH, _N))
    p = 0
    for i in range(n_ops):
        op = opcodes[i]
        operand = operands[i]
        if op == PUSH_X:
            sval[p] = x[operand]
            for a in range(_N):
                sgrad[p, a] = 1.0 if a == operand else 0.0
            p += 1
        elif op == PUSH_PARAMETER:
            sval[p] = parameters[operand]
            for a in range(_N):
                sgrad[p, a] = 0.0
            p += 1
        elif op == PUSH_LITERAL:
            sval[p] = literals[i]
            for a in range(_N):
                sgrad[p, a] = 0.0
            p += 1
        elif op == ADD or op == SUB:
            l = p - 2
            r = p - 1
            s = 1.0 if op == ADD else -1.0
            sval[l] = sval[l] + s * sval[r]
            for a in range(_N):
                sgrad[l, a] = sgrad[l, a] + s * sgrad[r, a]
            p -= 1
        elif op == MUL:
            l = p - 2
            r = p - 1
            vl = sval[l]
            vr = sval[r]
            for a in range(_N):
                sgrad[l, a] = sgrad[l, a] * vr + sgrad[r, a] * vl
            sval[l] = vl * vr
            p -= 1
        elif op == AQ:
            l = p - 2
            r = p - 1
            y = sval[r]
            scale = 1.0 / np.sqrt(1.0 + y * y)
            scale_p = -y * scale ** 3
            vl = sval[l]
            for a in range(_N):
                sgrad[l, a] = sgrad[l, a] * scale + (scale_p * sgrad[r, a]) * vl
            sval[l] = vl * scale
            p -= 1
        elif op == NEG:
            idx = p - 1
            sval[idx] = -sval[idx]
            for a in range(_N):
                sgrad[idx, a] = -sgrad[idx, a]
        elif op == SIN:
            idx = p - 1
            v = sval[idx]
            first = np.cos(v)
            for a in range(_N):
                sgrad[idx, a] = first * sgrad[idx, a]
            sval[idx] = np.sin(v)
        elif op == EXP:
            idx = p - 1
            v = np.exp(sval[idx])
            for a in range(_N):
                sgrad[idx, a] = v * sgrad[idx, a]
            sval[idx] = v
    idx = p - 1
    return sval[idx], sgrad[idx].copy()


@njit(cache=True, fastmath=False, nogil=True)
def _wdelta(opcodes, operands, literals, n_ops, parameters, x):
    """W = V(x) - V(0) without the cancellation: every node carries (value at 0, increment)."""
    sc = np.zeros(MAX_STACK_DEPTH)
    sd = np.zeros(MAX_STACK_DEPTH)
    p = 0
    for i in range(n_ops):
        op = opcodes[i]
        operand = operands[i]
        if op == PUSH_X:
            sc[p] = 0.0
            sd[p] = x[operand]
            p += 1
        elif op == PUSH_PARAMETER:
            sc[p] = parameters[operand]
            sd[p] = 0.0
            p += 1
        elif op == PUSH_LITERAL:
            sc[p] = literals[i]
            sd[p] = 0.0
            p += 1
        elif op == ADD:
            sc[p - 2] = sc[p - 2] + sc[p - 1]
            sd[p - 2] = sd[p - 2] + sd[p - 1]
            p -= 1
        elif op == SUB:
            sc[p - 2] = sc[p - 2] - sc[p - 1]
            sd[p - 2] = sd[p - 2] - sd[p - 1]
            p -= 1
        elif op == MUL:
            cl = sc[p - 2]
            dl = sd[p - 2]
            cr = sc[p - 1]
            dr = sd[p - 1]
            sd[p - 2] = dl * (cr + dr) + cl * dr
            sc[p - 2] = cl * cr
            p -= 1
        elif op == AQ:
            cl = sc[p - 2]
            dl = sd[p - 2]
            cy = sc[p - 1]
            dy = sd[p - 1]
            a0 = math.sqrt(1.0 + cy * cy)
            a1 = math.sqrt(1.0 + (cy + dy) * (cy + dy))
            ds = -dy * (2.0 * cy + dy) / (a1 * a0 * (a1 + a0))
            sd[p - 2] = dl / a1 + cl * ds
            sc[p - 2] = cl / a0
            p -= 1
        elif op == NEG:
            sc[p - 1] = -sc[p - 1]
            sd[p - 1] = -sd[p - 1]
        elif op == SIN:
            c = sc[p - 1]
            d = sd[p - 1]
            sd[p - 1] = 2.0 * math.cos(c + 0.5 * d) * math.sin(0.5 * d)
            sc[p - 1] = math.sin(c)
        elif op == EXP:
            c = sc[p - 1]
            d = sd[p - 1]
            e0 = math.exp(c)
            sd[p - 1] = e0 * math.expm1(d)
            sc[p - 1] = e0
    return sd[p - 1]


@njit(cache=True, fastmath=False, nogil=True)
def _product(sval, sgrad, shess, l, vr, gr, hr):
    """stack[l] <- stack[l] * (vr, gr, hr); product rule (cpu_polish._product)."""
    vl = sval[l]
    gl = sgrad[l].copy()
    hl = shess[l].copy()
    sval[l] = vl * vr
    for a in range(_N):
        for b in range(_N):
            shess[l, a, b] = hl[a, b] * vr + hr[a, b] * vl + gl[a] * gr[b] + gr[a] * gl[b]
    for a in range(_N):
        sgrad[l, a] = gl[a] * vr + gr[a] * vl


@njit(cache=True, fastmath=False, nogil=True)
def _dual2(opcodes, operands, literals, n_ops, parameters, x):
    """Value, gradient and Hessian of V at x (cpu_polish._dual2)."""
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


@njit(cache=True, fastmath=False, nogil=True)
def bnorm(b):
    """rho-multiplier of the bounded-input optimum: sum |b_k| (l1) or ||b||_2 (l2)."""
    if _L1:
        s = 0.0
        for k in range(_M):
            s += abs(b[k])
        return s
    s = 0.0
    for k in range(_M):
        s += b[k] * b[k]
    return np.sqrt(s)


@njit(cache=False, fastmath=False, nogil=True)
def _abvr_point(opcodes, operands, literals, n_ops, parameters, x):
    """W, a, b (m-vector), x.gradV, gradV at one point (value + gradient only)."""
    _, g = _dual1(opcodes, operands, literals, n_ops, parameters, x)
    v = _wdelta(opcodes, operands, literals, n_ops, parameters, x)
    fld = _FIELDS(x, np.empty(_K_FIELDS))
    a = 0.0
    for i in range(_N):
        a += g[i] * fld[i]
    b = np.empty(_M)
    for k in range(_M):
        s = 0.0
        for i in range(_N):
            s += g[i] * fld[_OFF_G + k * _N + i]
        b[k] = s
    rad = 0.0
    for i in range(_N):
        rad += g[i] * x[i]
    return v, a, b, rad, g


@njit(cache=False, fastmath=False, nogil=True)
def point_full(opcodes, operands, literals, n_ops, parameters, x):
    """W, a, b (m), grad a (n), grad b (m x n), grad V (n) -- the SLSQP polish callables.
    ga = H f + Jf^T g ;  gb_k = H G[:,k] + JG_k^T g   (exact derivatives, as in cpu_polish)."""
    _, g, h = _dual2(opcodes, operands, literals, n_ops, parameters, x)
    v = _wdelta(opcodes, operands, literals, n_ops, parameters, x)
    fld = _FIELDS(x, np.empty(_K_FIELDS))
    drift = np.empty(_N)
    for i in range(_N):
        drift[i] = fld[i]
    ctrl = np.empty((_M, _N))
    for k in range(_M):
        for i in range(_N):
            ctrl[k, i] = fld[_OFF_G + k * _N + i]
    jd = np.empty((_N, _N))
    for i in range(_N):
        for j in range(_N):
            jd[i, j] = fld[_OFF_JF + i * _N + j]
    a = 0.0
    for i in range(_N):
        a += g[i] * drift[i]
    b = np.empty(_M)
    for k in range(_M):
        s = 0.0
        for i in range(_N):
            s += g[i] * ctrl[k, i]
        b[k] = s
    ga = h @ drift + jd.T @ g
    gb = np.empty((_M, _N))
    for k in range(_M):
        col = np.empty(_N)
        for i in range(_N):
            col[i] = ctrl[k, i]
        jc = np.empty((_N, _N))
        for i in range(_N):
            for j in range(_N):
                jc[i, j] = fld[_OFF_JG + k * _N * _N + i * _N + j]
        row = h @ col + jc.T @ g
        for j in range(_N):
            gb[k, j] = row[j]
    return v, a, b, ga, gb, g


@njit(cache=False, fastmath=False, parallel=True)
def lattice_eval(opcodes, operands, literals, n_ops, parameters, coords):
    """W, a, b (P x m), x.gradV on every row of coords, parallel over the rows."""
    P = coords.shape[0]
    V = np.empty(P)
    A = np.empty(P)
    B = np.empty((P, _M))
    R = np.empty(P)
    for i in prange(P):
        x = coords[i]
        v, a, b, rad, _ = _abvr_point(opcodes, operands, literals, n_ops, parameters, x)
        V[i] = v
        A[i] = a
        for k in range(_M):
            B[i, k] = b[k]
        R[i] = rad
    return V, A, B, R


@njit(cache=False, fastmath=False, parallel=True)
def lattice_vgrad(opcodes, operands, literals, n_ops, parameters, coords):
    """Raw V, W = V - V(0) and grad V on every row (the training-grid stage scores raw V)."""
    P = coords.shape[0]
    Vv = np.empty(P)
    W = np.empty(P)
    Gr = np.empty((P, _N))
    for i in prange(P):
        x = coords[i]
        v, g = _dual1(opcodes, operands, literals, n_ops, parameters, x)
        Vv[i] = v
        W[i] = _wdelta(opcodes, operands, literals, n_ops, parameters, x)
        for j in range(_N):
            Gr[i, j] = g[j]
    return Vv, W, Gr
