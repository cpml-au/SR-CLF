"""Candidate -> sympy V, a, b and the numpy callables the checks need.

Dynamics enter as callables ``fSR(*x_syms) -> sympy Matrix (n, 1)`` and
``GSR(*x_syms) -> sympy Matrix (n, n_inputs)``, exactly the SystemDynamicsSR objects of the
examples; the state dimension is ``len(bounds)`` everywhere, nothing is hard-coded to 4.

    V           = DeapSimplifier(substitute_paramsCoef(expression, constants))
    a           = grad(V) . fSR(x)
    b           = grad(V) . GSR(x)[:, input_index]

``to_prefix`` is the inverse of the first line (sympy -> GP prefix string + constants); it is
what lets a sympy V reach the vendored Pallas engine (src/srcGPU/gpu5_9), whose encoder only
reads {add, sub, mul, aq, neg, sin, exp, x_i, a}.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from src.SymFunctions import DeapSimplifier, substitute_paramsCoef


def state_symbols(n):
    """(x1, ..., xn)."""
    return sp.symbols(f"x1:{n + 1}")


def sympy_expression(expression, constants=None):
    """GP prefix string + fitted constants -> sympy V with the constants substituted.
    A sympy expression is passed through unchanged."""
    if isinstance(expression, sp.Basic):
        return expression
    values = (
        []
        if constants is None
        else list(np.asarray(constants, dtype=float).reshape(-1))
    )
    return DeapSimplifier(
        substitute_paramsCoef(str(expression), values), should_print=False
    )


def ab_expressions(V, fSR, GSR, x_syms, input_index=1):
    """a = grad(V).f and b = grad(V).G[:, input_index] as sympy expressions."""
    grad = sp.Matrix([sp.diff(V, s) for s in x_syms])
    f_vec = fSR(*x_syms)
    G_mat = GSR(*x_syms)
    a = (grad.T * f_vec)[0]
    b = sum(grad[i] * G_mat[i, input_index] for i in range(len(x_syms)))
    return a, b


def build_symbolic(expression, constants, fSR, GSR, n_states=4, input_index=1):
    """(V, a, b) as sympy expressions for one candidate."""
    x_syms = state_symbols(n_states)
    V = sympy_expression(expression, constants)
    a, b = ab_expressions(V, fSR, GSR, x_syms, input_index)
    return V, a, b


def lambdify_abv(V, a, b, x_syms):
    """Vectorised numpy callables (called as fn(*X) with X of shape (n, M))."""
    return (
        sp.lambdify(x_syms, V, "numpy"),
        sp.lambdify(x_syms, a, "numpy"),
        sp.lambdify(x_syms, b, "numpy"),
    )


def polish_callables(V, a, b, x_syms):
    """SciPy-ready callables for the polishes, keyed as in run_det3_gpu.py:

    ``V, a, b``: z (n,) -> float;  ``gV, ga, gb``: z (n,) -> (n,) exact gradients;  ``v0`` = V(0);
    ``V_raw, a_raw, b_raw``: the vectorised lambdified functions (fn(*columns)).
    """
    L = lambda e: sp.lambdify(x_syms, e, "numpy")  # noqa: E731
    raw = {
        "V": L(V),
        "gV": L([sp.diff(V, s) for s in x_syms]),
        "a": L(a),
        "ga": L([sp.diff(a, s) for s in x_syms]),
        "b": L(b),
        "gb": L([sp.diff(b, s) for s in x_syms]),
    }

    def scalar(fn):
        def f(z):
            with np.errstate(all="ignore"):
                return float(fn(*z))

        return f

    def vector(fn):
        def f(z):
            with np.errstate(all="ignore"):
                return np.asarray(fn(*z), dtype=float).reshape(-1)

        return f

    fns = {
        k: (scalar(v) if k in ("V", "a", "b") else vector(v)) for k, v in raw.items()
    }
    fns["V_raw"], fns["a_raw"], fns["b_raw"] = raw["V"], raw["a"], raw["b"]
    fns["gV_raw"] = [
        L(sp.diff(V, s)) for s in x_syms
    ]  # vectorised partials (radial PD)
    fns["v0"] = fns["V"](np.zeros(len(x_syms)))
    return fns


# --------------------------------------------------------------------------
# sympy -> GP prefix form (for the Pallas engine), from build_valid_batch.py
# --------------------------------------------------------------------------


def _aq_v(base):
    """base == v**2 + 1  ->  v, else None."""
    if not base.is_Add or len(base.args) != 2:
        return None
    one = [t for t in base.args if t.is_Number and float(t) == 1.0]
    sq = [t for t in base.args if t.is_Pow and t.exp == 2]
    if len(one) == 1 and len(sq) == 1:
        return sq[0].base
    return None


def to_prefix(expr):
    """sympy -> (prefix string over {add, sub, mul, aq, neg, sin, exp}, constants).

    Every number becomes an ``a`` placeholder whose value is appended to ``constants`` in
    text order (the order substitute_paramsCoef consumes them); (v**2 + 1)**-0.5 factors
    become aq(., v); integer powers become repeated mul.  Raises on anything else.
    """
    consts = []

    def num(v):
        consts.append(float(v))
        return "a"

    def fold(op, parts):
        out = parts[0]
        for p in parts[1:]:
            out = f"{op}({out}, {p})"
        return out

    def rec(e):
        if e.is_Symbol:
            return str(e)
        if e.is_Number:
            return num(e)
        if e.is_Add:
            return fold("add", [rec(t) for t in e.args])
        if e.is_Mul:
            aq_vs, rest = [], []
            for f in e.args:
                if f.is_Pow and float(f.exp) == -0.5 and _aq_v(f.base) is not None:
                    aq_vs.append(_aq_v(f.base))
                else:
                    rest.append(f)
            s = fold("mul", [rec(t) for t in rest]) if rest else num(1.0)
            for v in aq_vs:
                s = f"aq({s}, {rec(v)})"
            return s
        if e.is_Pow:
            base, ex = e.args
            if ex.is_Integer and int(ex) >= 2:
                return fold("mul", [rec(base) for _ in range(int(ex))])
            if float(ex) == -0.5 and _aq_v(base) is not None:
                return f"aq({num(1.0)}, {rec(_aq_v(base))})"
            raise ValueError(f"unsupported power {e}")
        name = type(e).__name__
        if name in ("exp", "sin") and len(e.args) == 1:
            return f"{name}({rec(e.args[0])})"
        raise ValueError(f"unsupported node {name} in {e}")

    return rec(sp.sympify(expr)), consts
