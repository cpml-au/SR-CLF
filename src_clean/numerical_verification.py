from sympy import symbols, lambdify
import numpy as np
from src.SymVVdot_Calculations import compute_v_and_v_dotSR
from SystemDynamicsSR import fSR, GSR, QSR, RSR

def a_violation_numcheck(expression, res=5000):
    x1, x2 = symbols('x1 x2')

    # Compute symbolic pieces
    x_syms = [x1, x2]
    (
        _V, _V_grad, _Vdot, u_func, _lambda, _u_mag,
        b_norm_squared, b1, a
    ) = compute_v_and_v_dotSR(expression, fSR, GSR, QSR, RSR, x_syms, None)

    # Robust scalar expression for the equality "curve"
    try:
        if hasattr(b_norm_squared, '__iter__'):
            b_eq_expr = sum(list(b_norm_squared))
        else:
            b_eq_expr = b_norm_squared
    except Exception:
        b_eq_expr = b_norm_squared

    # Domain / grid
    xmin, xmax = -2.0, 2.0
    ymin, ymax = -2.0, 2.0
    res = res

    # x_check = np.linspace(xmin, xmax, res)
    # y_check = np.linspace(ymin, ymax, res)

    x_check = np.random.uniform(xmin, xmax, res)
    y_check = np.random.uniform(ymin, ymax, res)
    
    X_check, Y_check = np.meshgrid(x_check, y_check, indexing='xy')

    eq_fn   = lambdify((x1, x2), b_eq_expr, "numpy")
    ineq_fn = lambdify((x1, x2), a,          "numpy")

    # Evaluate; broadcast scalars to grid shape
    eq_vals_raw   = eq_fn(X_check, Y_check)
    ineq_vals_raw = ineq_fn(X_check, Y_check)

    # Ensure both are numpy arrays with the grid shape
    eq_vals   = np.asarray(eq_vals_raw)
    ineq_vals = np.asarray(ineq_vals_raw)

    if eq_vals.ndim == 0:
        eq_vals = np.full_like(X_check, eq_vals, dtype=float)
    if ineq_vals.ndim == 0:
        ineq_vals = np.full_like(X_check, ineq_vals, dtype=float)

    # Find approximate curve |eq| < tol
    tol = 1e-6
    curve_points = np.abs(eq_vals) < tol
    num_curve_pts = np.count_nonzero(curve_points)
    if num_curve_pts == 0:
        # No curve found on this grid -> treat as OK (or return a small penalty if desired)
        return 0

    # Check a(x) < 0 on the curve
    ineq_on_curve = ineq_vals[curve_points]
    all_inside = np.all(ineq_on_curve < 1e-5)

    return 1000 if not all_inside else 0
