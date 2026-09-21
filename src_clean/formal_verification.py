import os
import uuid
import subprocess
import numpy as np
from sympy import symbols, sin, cos, tanh, sqrt, exp
from src.SymVVdot_Calculations import compute_v_and_v_dotSR
from SystemDynamicsSR import fSR, xSR, GSR, QSR, RSR
from src.Sympy2SMT2 import write_smt2


def get_sif():
    d = os.path.abspath(os.getcwd())
    while True:
        if os.path.basename(d) == "examples":
            return os.path.join(d, "dreal4_latest.sif")
        parent = os.path.dirname(d)
        if parent == d:
            raise FileNotFoundError("could not find 'examples' directory")
        d = parent


SIF_PATH = get_sif()


def _symbols_from_inputs(x_vals=None, x_syms=None):
    if x_syms is not None:
        return list(x_syms)
    if x_vals is not None:
        return list(symbols(f"x1:{len(x_vals) + 1}"))
    return list(symbols("x1 x2"))


def _domain_from_inputs(domain, x_vals, n_states):
    if domain is not None:
        return domain
    if x_vals is None:
        return 2.0

    bounds = []
    for values in x_vals:
        arr = np.asarray(values, dtype=float)
        low = float(np.nanmin(arr))
        high = float(np.nanmax(arr))
        extent = max(abs(low), abs(high))
        bounds.append(extent)
    if len(bounds) != n_states:
        raise ValueError("x_vals length must match the number of state symbols")
    return bounds


def a_violation_check(
    expression,
    domain=None,
    tol=1e-5,
    delta=1e-3,
    x_vals=None,
    x_syms=None,
    fSR_func=fSR,
    GSR_func=GSR,
    QSR_func=QSR,
    RSR_func=RSR,
    b_index=2,
    origin_radius=None,
    boxes="both",
):
    x_syms = _symbols_from_inputs(x_vals=x_vals, x_syms=x_syms)
    domain = _domain_from_inputs(domain, x_vals, len(x_syms))

    # compute SymPy expressions
    _V, V_dot, *_rest, b_norm_squared, b1, a = compute_v_and_v_dotSR(
        expression, fSR_func, GSR_func, QSR_func, RSR_func, x_syms, None
    )
    b_sym = b_norm_squared[b_index]  # default keeps the previous 2D indexing
    a_sym = a
    vdot_sym = V_dot

    # out dir + UNIQUE filenames (safe in parallel)
    out_dir = "SMT_expresions"
    os.makedirs(out_dir, exist_ok=True)
    uniq = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    ab_path = os.path.join(out_dir, f"ab_case_{uniq}.smt2")
    vdot_path = os.path.join(out_dir, f"vdot_case_{uniq}.smt2")

    # write SMT2
    write_smt2(
        ab_path,
        a=a_sym,
        b=b_sym,
        domain=domain,
        tol=tol,
        boxes=boxes,
        x_syms=x_syms,
        origin_radius=origin_radius,
    )
    write_smt2(
        vdot_path,
        vdot=vdot_sym,
        domain=domain,
        tol=tol,
        boxes=boxes,
        x_syms=x_syms,
        origin_radius=origin_radius,
    )

    # run dReal
    def run_dreal(path, delta=1e-3):
        proc = subprocess.run(
            ["dreal", "--model", "--precision", str(delta), path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return proc.stdout.strip()

    try:
        out1 = run_dreal(ab_path)
        out2 = run_dreal(vdot_path)
    finally:
        # cleanup (best-effort)
        for p in (ab_path, vdot_path):
            try:
                os.remove(p)
            except OSError:
                pass

    # delta-sat if output is NOT "unsat"
    sat1 = "unsat" not in out1.lower()
    sat2 = "unsat" not in out2.lower()
    score = 1000 if (sat1 or sat2) else 0
    return score, out1, out2
