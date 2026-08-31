from sympy import symbols, Function, Integer, Rational, Float
from sympy import sin, cos, tanh, sinh, sqrt, exp, asin

def _real_lit(x):
    s = str(x)
    return s if ("." in s or "e" in s or "E" in s) else f"{s}.0"

def sympy_to_smt(e):
    # Symbols
    if e.is_Symbol:
        return e.name

    # Numbers
    if e.is_Number:
        if isinstance(e, Rational):
            return f"(/ {_real_lit(e.p)} {_real_lit(e.q)})"
        if isinstance(e, Integer):
            return _real_lit(int(e))
        if isinstance(e, Float):
            return _real_lit(e.evalf(17))
        return _real_lit(e.evalf(17))

    # Addition
    if e.is_Add:
        return f"(+ {' '.join(sympy_to_smt(a) for a in e.args)})"

    # Multiplication
    if e.is_Mul:
        return f"(* {' '.join(sympy_to_smt(a) for a in e.args)})"

    # Powers
    if e.is_Pow:
        base, expn = e.as_base_exp()

        def half_integer_power_to_smt(base_expr, numerator):
            if numerator == 0:
                return "1.0"

            abs_num = abs(numerator)
            factors = []
            if abs_num // 2:
                factors.extend(sympy_to_smt(base_expr) for _ in range(abs_num // 2))
            if abs_num % 2:
                factors.append(f"(sqrt {sympy_to_smt(base_expr)})")

            if len(factors) == 1:
                value = factors[0]
            else:
                value = f"(* {' '.join(factors)})"

            if numerator < 0:
                return f"(/ 1.0 {value})"
            return value

        # sqrt: exponent = 1/2
        if isinstance(expn, Rational) and expn.p == 1 and expn.q == 2:
            return f"(sqrt {sympy_to_smt(base)})"

        # Half-integer powers, e.g. x**(-3/2) from aq/sqrt expressions.
        if isinstance(expn, Rational) and expn.q == 2:
            return half_integer_power_to_smt(base, int(expn.p))

        # Integer exponents
        if expn.is_Integer:
            n = int(expn)
            if n == 0:
                return "1.0"
            if n == 1:
                return sympy_to_smt(base)
            if n > 1:
                return f"(* {' '.join(sympy_to_smt(base) for _ in range(n))})"
            if n == -1:
                return f"(/ 1.0 {sympy_to_smt(base)})"
            # n < -1
            return f"(/ 1.0 (* {' '.join(sympy_to_smt(base) for _ in range(-n))}))"

        # Float-looking integer, e.g., 2.0
        if expn.is_Float and float(expn).is_integer():
            n = int(float(expn))
            return sympy_to_smt(base) if n == 1 else f"(* {' '.join(sympy_to_smt(base) for _ in range(n))})"

        # Float-looking half integer, e.g. -1.5.
        if expn.is_Float:
            doubled = 2.0 * float(expn)
            if doubled.is_integer():
                return half_integer_power_to_smt(base, int(doubled))

        raise NotImplementedError(f"Non-integer power: {e}")

    # Elementary functions
    fn_map = {sin: "sin", cos: "cos", tanh: "tanh", sinh: "sinh", sqrt: "sqrt", exp: "exp", asin: "arcsin"}
    if e.func in fn_map:
        return f"({fn_map[e.func]} {' '.join(sympy_to_smt(a) for a in e.args)})"

    if isinstance(e.func, Function):
        return f"({e.func.__name__} {' '.join(sympy_to_smt(a) for a in e.args)})"

    raise NotImplementedError(f"Cannot print: {e} ({type(e)})")


# -----------------------
# SMT2 file writer
# -----------------------
def _as_bounds(domain, x_syms):
    if np_is_sequence(domain) and len(domain) == len(x_syms):
        bounds = []
        for item in domain:
            if np_is_sequence(item) and len(item) == 2:
                bounds.append((float(item[0]), float(item[1])))
            else:
                value = float(item)
                bounds.append((-value, value))
        return bounds

    value = float(domain)
    return [(-value, value) for _ in x_syms]


def np_is_sequence(value):
    return isinstance(value, (list, tuple))


def _outside_origin_ball_assertion(x_syms, origin_radius):
    radius_squared = float(origin_radius) ** 2
    norm_squared = "(+ " + " ".join(f"(* {x.name} {x.name})" for x in x_syms) + ")"
    return f"(assert (>= {norm_squared} {_real_lit(radius_squared)}))"


def write_smt2(
    filename,
    a=None,
    b=None,
    vdot=None,
    domain=2.0,
    tol=1e-5,
    boxes="both",
    x_syms=None,
    origin_radius=None,
):
    """
    Writes an SMT-LIB2 file.
      - Pass either (a and b) or (vdot).
      - a, b, vdot are SymPy expressions in x1, x2, ...
      - boxes: "both" | "neg" | "pos"
    """
    if (vdot is None) == (a is None or b is None):
        raise ValueError("Provide either vdot, or both a and b (but not both).")

    if x_syms is None:
        x_syms = symbols("x1 x2")
    x_syms = list(x_syms)
    bounds = _as_bounds(domain, x_syms)

    lines = []
    lines.append("(set-logic QF_NRA)")
    for x in x_syms:
        lines.append(f"(declare-fun {x.name} () Real)")

    args = " ".join(f"({x.name} Real)" for x in x_syms)
    call_args = " ".join(x.name for x in x_syms)

    if vdot is not None:
        lines.append(f"(define-fun vdot ({args}) Real {sympy_to_smt(vdot)})")
    else:
        lines.append(f"(define-fun a ({args}) Real {sympy_to_smt(a)})")
        lines.append(f"(define-fun b ({args}) Real {sympy_to_smt(b)})")

    if origin_radius is not None:
        box_terms = [
            f"(<= {_real_lit(low)} {x.name}) (<= {x.name} {_real_lit(high)})"
            for x, (low, high) in zip(x_syms, bounds)
        ]
        lines.append(f"(assert (and {' '.join(box_terms)}))")
        lines.append(_outside_origin_ball_assertion(x_syms, origin_radius))
    elif boxes == "both":
        pos_terms = []
        neg_terms = []
        for x, (low, high) in zip(x_syms, bounds):
            extent = max(abs(low), abs(high))
            pos_terms.append(f"(<= 0.001 {x.name}) (<= {x.name} {_real_lit(extent)})")
            neg_terms.append(f"(<= -{_real_lit(extent)} {x.name}) (<= {x.name} -0.001)")
        lines.append(f"(assert (or (and {' '.join(pos_terms)}) (and {' '.join(neg_terms)})))")
    elif boxes == "neg":
        neg_terms = []
        for x, (low, high) in zip(x_syms, bounds):
            extent = max(abs(low), abs(high))
            neg_terms.append(f"(<= -{_real_lit(extent)} {x.name}) (<= {x.name} -0.001)")
        lines.append(f"(assert (and {' '.join(neg_terms)}))")
    elif boxes == "pos":
        pos_terms = []
        for x, (low, high) in zip(x_syms, bounds):
            extent = max(abs(low), abs(high))
            pos_terms.append(f"(<= 0.001 {x.name}) (<= {x.name} {_real_lit(extent)})")
        lines.append(f"(assert (and {' '.join(pos_terms)}))")
    else:
        raise ValueError('boxes must be "both", "neg", or "pos".')

    if vdot is not None:
        lines.append(f"(assert (> (vdot {call_args}) {tol}))")
    else:
        lines.append(f"(assert (= (b {call_args}) 0.0))")
        lines.append(f"(assert (> (a {call_args}) {tol}))")

    lines.append("(check-sat)")
    lines.append("(exit)")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
