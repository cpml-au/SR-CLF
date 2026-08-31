from dataclasses import asdict, dataclass

import numpy as np
import sympy as sp


@dataclass
class DrealConditionReport:
    name: str
    ran: bool
    passed: bool
    penalty: float
    status: str
    counterexample: str | None = None


@dataclass
class DrealCLFReport:
    passed: bool
    total_penalty: float
    conditions: dict

    def as_dict(self):
        return {
            "passed": self.passed,
            "total_penalty": self.total_penalty,
            "conditions": {
                name: asdict(report) for name, report in self.conditions.items()
            },
        }


def _require_dreal():
    try:
        import dreal
    except ImportError as exc:
        raise ImportError(
            "dreal Python package is required. Use the flex311 environment."
        ) from exc
    return dreal


def _normalize_bounds(bounds, n_states):
    if bounds is None:
        raise ValueError("bounds must be provided for bounded dReal verification")

    if np.isscalar(bounds):
        bound = float(bounds)
        return [(-bound, bound)] * n_states

    if len(bounds) != n_states:
        raise ValueError("bounds length must match number of state symbols")

    normalized = []
    for item in bounds:
        if np.isscalar(item):
            bound = float(item)
            normalized.append((-bound, bound))
        else:
            low, high = item
            normalized.append((float(low), float(high)))
    return normalized


def _as_sympy_vector(values):
    if values is None:
        return None
    if callable(values):
        raise TypeError("Pass symbolic expressions, not numeric callables")
    return [sp.sympify(value) for value in values]


def _as_sympy_matrix(values):
    if values is None:
        return None
    if callable(values):
        raise TypeError("Pass symbolic expressions, not numeric callables")
    mat = sp.Matrix(values)
    if mat.cols == 0:
        return None
    return mat


def _dreal_abs(expr, dreal):
    return dreal.sqrt(expr * expr)


def _sympy_to_dreal(expr, symbol_map, dreal):
    expr = sp.sympify(expr)

    if expr in symbol_map:
        return symbol_map[expr]

    if expr == sp.E:
        return float(sp.E)
    if expr == sp.pi:
        return float(sp.pi)
    if expr.is_Number:
        return float(expr)

    if isinstance(expr, sp.Symbol):
        raise ValueError(f"Unknown symbol in expression: {expr}")

    if isinstance(expr, sp.Add):
        terms = [_sympy_to_dreal(arg, symbol_map, dreal) for arg in expr.args]
        result = terms[0]
        for term in terms[1:]:
            result = result + term
        return result

    if isinstance(expr, sp.Mul):
        factors = [_sympy_to_dreal(arg, symbol_map, dreal) for arg in expr.args]
        result = factors[0]
        for factor in factors[1:]:
            result = result * factor
        return result

    if isinstance(expr, sp.Pow):
        base = _sympy_to_dreal(expr.base, symbol_map, dreal)
        exponent = expr.exp
        if exponent.is_Integer:
            return base ** int(exponent)
        if exponent.is_Rational:
            return base ** float(exponent)
        return base ** _sympy_to_dreal(exponent, symbol_map, dreal)

    if isinstance(expr, sp.Abs):
        return _dreal_abs(_sympy_to_dreal(expr.args[0], symbol_map, dreal), dreal)

    if expr.func == sp.sin:
        return dreal.sin(_sympy_to_dreal(expr.args[0], symbol_map, dreal))
    if expr.func == sp.cos:
        return dreal.cos(_sympy_to_dreal(expr.args[0], symbol_map, dreal))
    if expr.func == sp.tan:
        return dreal.tan(_sympy_to_dreal(expr.args[0], symbol_map, dreal))
    if expr.func == sp.tanh:
        return dreal.tanh(_sympy_to_dreal(expr.args[0], symbol_map, dreal))
    if expr.func == sp.exp:
        return dreal.exp(_sympy_to_dreal(expr.args[0], symbol_map, dreal))
    if expr.func == sp.log:
        return dreal.log(_sympy_to_dreal(expr.args[0], symbol_map, dreal))
    if expr.func == sp.sqrt:
        return dreal.sqrt(_sympy_to_dreal(expr.args[0], symbol_map, dreal))
    if expr.func == sp.Min:
        args = [_sympy_to_dreal(arg, symbol_map, dreal) for arg in expr.args]
        result = args[0]
        for arg in args[1:]:
            result = dreal.Min(result, arg)
        return result
    if expr.func == sp.Max:
        args = [_sympy_to_dreal(arg, symbol_map, dreal) for arg in expr.args]
        result = args[0]
        for arg in args[1:]:
            result = dreal.Max(result, arg)
        return result

    raise NotImplementedError(f"Unsupported SymPy expression for dReal: {expr}")


def _logical_and(items, dreal):
    normalized = []
    for item in items:
        if isinstance(item, (bool, np.bool_)):
            normalized.append(dreal.Formula.TRUE() if item else dreal.Formula.FALSE())
        else:
            normalized.append(item)
    items = normalized
    if not items:
        return dreal.Formula.TRUE()
    result = items[0]
    for item in items[1:]:
        result = dreal.logical_and(result, item)
    return result


def _box_constraints(dvars, bounds):
    formulas = []
    for var, (low, high) in zip(dvars, bounds):
        formulas.append(var >= low)
        formulas.append(var <= high)
    return formulas


def _near_zero_formula(expr, tol):
    return [expr >= -tol, expr <= tol]


def _check_exists(name, formulas, delta, penalty, dreal):
    query = _logical_and(formulas, dreal)
    result = dreal.CheckSatisfiability(query, delta)
    if result is None:
        return DrealConditionReport(
            name=name,
            ran=True,
            passed=True,
            penalty=0.0,
            status="unsat",
            counterexample=None,
        )
    return DrealConditionReport(
        name=name,
        ran=True,
        passed=False,
        penalty=float(penalty),
        status="delta_sat",
        counterexample=str(result),
    )


def _origin_penalty(V_expr, x_syms, thresholds, penalties):
    origin_subs = {sym: 0 for sym in x_syms}
    try:
        v0 = float(sp.N(sp.sympify(V_expr).subs(origin_subs)))
        abs_v0 = abs(v0)
    except Exception as exc:
        return DrealConditionReport(
            name="origin",
            ran=True,
            passed=False,
            penalty=float(penalties[-1]),
            status=f"failed_to_evaluate_origin: {exc}",
            counterexample=None,
        )

    tight_tol, loose_tol = thresholds
    tight_penalty, loose_penalty, fail_penalty = penalties
    if abs_v0 <= tight_tol:
        penalty = tight_penalty
        passed = True
        status = f"|V(0)|={abs_v0:.6g} <= {tight_tol:g}"
    elif abs_v0 <= loose_tol:
        penalty = loose_penalty
        passed = True
        status = f"|V(0)|={abs_v0:.6g} <= {loose_tol:g}"
    else:
        penalty = fail_penalty
        passed = False
        status = f"|V(0)|={abs_v0:.6g} > {loose_tol:g}"

    return DrealConditionReport(
        name="origin",
        ran=True,
        passed=passed,
        penalty=float(penalty),
        status=status,
        counterexample=None,
    )


def check_clf_conditions_dreal(
    V_expr,
    f_exprs,
    G_exprs=None,
    x_syms=None,
    bounds=None,
    checks=("origin", "positive", "vdot_negative", "a_b_zero", "artstein"),
    vdot_expr=None,
    origin_radius=1e-3,
    delta=1e-6,
    zero_tol=1e-8,
    a_tol=0.0,
    b_tol=1e-8,
    decay_rate=1e-5,
    rho=1.0,
    gamma=0.0,
    violation_penalty=100.0,
    origin_thresholds=(1e-6, 1e-3),
    origin_penalties=(0.0, 10.0, 100.0),
):
    """Formally check selected Lyapunov/CLF conditions with dReal Python.

    The function checks violations as existential formulas on a bounded box.
    A condition passes when dReal returns unsat for the violation query.

    Supported checks:
      origin: tiered penalty on |V(0)|.
      positive: no x outside origin ball with V(x) <= 0.
      vdot_negative: no x outside origin ball with Vdot(x) >= 0.
      a_b_zero: no x outside origin ball with a(x) ~= 0 and b(x) ~= 0.
      artstein: no x outside origin ball with b(x) ~= 0 and a(x) >= 0.
      artstein_norm_decay: no b(x) ~= 0 point with a(x) >= -decay_rate*||x||.
      barrier: no x outside origin ball with a(x) - rho*|b(x)|_1 + gamma >= 0.
      a_negative: no x outside origin ball with a(x) >= 0.
    """
    dreal = _require_dreal()

    V_expr = sp.sympify(V_expr)
    if x_syms is None:
        x_syms = sorted(V_expr.free_symbols, key=lambda s: str(s))
    else:
        x_syms = list(x_syms)

    n_states = len(x_syms)
    bounds = _normalize_bounds(bounds, n_states)
    f_vec = _as_sympy_vector(f_exprs)
    if len(f_vec) != n_states:
        raise ValueError("f_exprs length must match number of state symbols")

    G_mat = _as_sympy_matrix(G_exprs)
    if G_mat is not None and G_mat.rows != n_states:
        raise ValueError("G_exprs must have one row per state")

    checks = tuple(checks)
    dvars = [dreal.Variable(str(sym)) for sym in x_syms]
    symbol_map = dict(zip(x_syms, dvars))

    V_d = _sympy_to_dreal(V_expr, symbol_map, dreal)
    grad_exprs = [sp.diff(V_expr, sym) for sym in x_syms]
    a_expr = sum(grad * f_i for grad, f_i in zip(grad_exprs, f_vec))
    a_d = _sympy_to_dreal(a_expr, symbol_map, dreal)

    if vdot_expr is None:
        vdot_d = a_d
    else:
        vdot_d = _sympy_to_dreal(vdot_expr, symbol_map, dreal)

    b_exprs = []
    if G_mat is not None:
        for col in range(G_mat.cols):
            b_exprs.append(
                sum(grad_exprs[row] * G_mat[row, col] for row in range(n_states))
            )
    b_ds = [_sympy_to_dreal(expr, symbol_map, dreal) for expr in b_exprs]

    norm_sq = sum(var * var for var in dvars)
    norm = dreal.sqrt(norm_sq)
    domain = _box_constraints(dvars, bounds)
    outside_origin = [norm_sq >= origin_radius * origin_radius]
    base = domain + outside_origin

    reports = {}

    if "origin" in checks:
        reports["origin"] = _origin_penalty(
            V_expr, x_syms, origin_thresholds, origin_penalties
        )

    if "positive" in checks:
        reports["positive"] = _check_exists(
            "positive",
            base + [V_d <= 0.0],
            delta,
            violation_penalty,
            dreal,
        )

    if "vdot_negative" in checks:
        reports["vdot_negative"] = _check_exists(
            "vdot_negative",
            base + [vdot_d >= 0.0],
            delta,
            violation_penalty,
            dreal,
        )

    b_zero = []
    for b_d in b_ds:
        b_zero.extend(_near_zero_formula(b_d, b_tol))

    if "a_b_zero" in checks:
        if not b_ds:
            reports["a_b_zero"] = DrealConditionReport(
                name="a_b_zero",
                ran=True,
                passed=False,
                penalty=float(violation_penalty),
                status="requires G_exprs/b(x)",
            )
        else:
            reports["a_b_zero"] = _check_exists(
                "a_b_zero",
                base + _near_zero_formula(a_d, zero_tol) + b_zero,
                delta,
                violation_penalty,
                dreal,
            )

    if "artstein" in checks:
        if not b_ds:
            reports["artstein"] = DrealConditionReport(
                name="artstein",
                ran=True,
                passed=False,
                penalty=float(violation_penalty),
                status="requires G_exprs/b(x)",
            )
        else:
            reports["artstein"] = _check_exists(
                "artstein",
                base + b_zero + [a_d >= -a_tol],
                delta,
                violation_penalty,
                dreal,
            )

    if "artstein_norm_decay" in checks:
        if not b_ds:
            reports["artstein_norm_decay"] = DrealConditionReport(
                name="artstein_norm_decay",
                ran=True,
                passed=False,
                penalty=float(violation_penalty),
                status="requires G_exprs/b(x)",
            )
        else:
            reports["artstein_norm_decay"] = _check_exists(
                "artstein_norm_decay",
                base + b_zero + [a_d + decay_rate * norm >= -a_tol],
                delta,
                violation_penalty,
                dreal,
            )

    if "barrier" in checks:
        if not b_ds:
            reports["barrier"] = DrealConditionReport(
                name="barrier",
                ran=True,
                passed=False,
                penalty=float(violation_penalty),
                status="requires G_exprs/b(x)",
            )
        else:
            b_l1 = sum(_dreal_abs(b_d, dreal) for b_d in b_ds)
            reports["barrier"] = _check_exists(
                "barrier",
                base + [a_d - rho * b_l1 + gamma >= 0.0],
                delta,
                violation_penalty,
                dreal,
            )

    if "a_negative" in checks:
        reports["a_negative"] = _check_exists(
            "a_negative",
            base + [a_d >= -a_tol],
            delta,
            violation_penalty,
            dreal,
        )

    total_penalty = float(sum(report.penalty for report in reports.values()))
    passed = all(report.passed for report in reports.values())
    return DrealCLFReport(
        passed=passed,
        total_penalty=total_penalty,
        conditions=reports,
    )
