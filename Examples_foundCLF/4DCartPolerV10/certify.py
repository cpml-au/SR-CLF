"""certify.py -- the certificate pair (c*, kappa*(Omega_c*)) of one V3 individual, on the CPU.

    python certify.py best_per_generation.jsonl --last
    python certify.py --expression "add(mul(a, mul(x1, x1)), ...)" --constants 0.11,-2.11,...

Same environment as run.py (SYMCLF_EXACT_MODE=det4, SYMCLF_DET4_RHO, ...): the individual is the
GP tree with its constants and V is the tree itself (no wrap, no anchor in this example).

Every number comes from the CPU checks in src/srcCPU (2026-09-08/09):

  c_v*         min W/||x||^2 over the punctured box (DET4 PD rate; V is PD iff > PD_EPS)
  kappa*       min -(a - rho|b|)/W over the punctured box (DET4). Only x = 0 is excluded;
               this is the run's bc_rate at the same rho (bounded check, kappa - phi_max).
  boundary_min min W over the six faces of the box (face mesh + L-BFGS-B polish)
  c_viol       min W over the Artstein violators of the b = 0 manifold: exact-check roots
               (bisection, absolute margin a > -1e-9 inside a 1.1e-3 numerical ball) followed
               by the W-descent min_w_violator (min W s.t. b = 0, margin >= -tol)
  c*           min(boundary_min, c_viol) -- the certified level (Khalil Thm 4.x / Chesi);
               0 when V is not PD
  kappa*(c*)   min -(a - rho|b|)/W over {W <= c*}: DET4 restricted to the certified set
  volume       fraction of the run's 41^3 grid with W < c*

The whole thing is a verdict-time report: nothing here feeds the fitness.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scipy.optimize import minimize  # noqa: E402

from src.srcCPU import conditions as C  # noqa: E402
from src.srcCPU.det3_check_cpu import LATTICE  # noqa: E402
from src.srcCPU.det4_check_cpu import det4_from_lattice  # noqa: E402
from src.srcCPU.exact_check_cpu import (  # noqa: E402
    check_b_manifold_exact,
    eval_lattice,
    min_w_violator,
    scan_geometry,
)
from src.srcCPU.symbolic import (  # noqa: E402
    ab_expressions,
    polish_callables,
    state_symbols,
    sympy_expression,
)


def _load_base():
    import Evaluate  # noqa: PLC0415  (examples/3DCartPolerV7c/Evaluate.py)

    return Evaluate._base


def boundary_min_w(fns, bounds, face_points=101, polish_top=3):
    """min W over the faces of the box: a face mesh, then L-BFGS-B on the face."""
    bounds = np.asarray(bounds, float)
    n = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    V, gV, v0 = fns["V"], fns["gV"], fns["v0"]
    best, best_x = np.inf, None
    for axis in range(n):
        others = [i for i in range(n) if i != axis]
        lin = [np.linspace(lo[i], hi[i], face_points) for i in others]
        mesh = np.meshgrid(*lin, indexing="ij")
        for side in (lo[axis], hi[axis]):
            pts = np.empty((mesh[0].size, n))
            pts[:, axis] = side
            for k, i in enumerate(others):
                pts[:, i] = mesh[k].ravel()
            W = eval_lattice(fns["V_raw"], pts) - v0
            W = np.where(np.isfinite(W), W, np.inf)
            for j in np.argsort(W)[:polish_top]:
                if not np.isfinite(W[j]):
                    continue
                if W[j] < best:
                    best, best_x = float(W[j]), pts[j].copy()
                box = [(lo[i], hi[i]) if i != axis else (side, side) for i in range(n)]
                try:
                    res = minimize(
                        lambda z: V(z) - v0,
                        pts[j],
                        jac=gV,
                        bounds=box,
                        method="L-BFGS-B",
                    )
                    z = np.clip(res.x, lo, hi)
                    z[axis] = side
                    w = V(z) - v0
                    if np.isfinite(w) and w < best:
                        best, best_x = float(w), z
                except Exception:
                    continue
    return best, best_x


def certify(
    expression,
    constants,
    base,
    rho=None,
    gamma1=0.0,
    input_index=1,
    abs_ball=None,
    top_k=40,
    maxiter=60,
    grid_points=41,
):
    """The certificate pair of one individual; returns a JSON-able dict."""
    t_all = time.perf_counter()
    rho = float(getattr(base, "BC_RHO", 1000.0)) if rho is None else float(rho)
    abs_ball = (
        float(getattr(base, "ABSOLUTE_CHECK_BALL_RADIUS", C.ORIGIN_TOL))
        if abs_ball is None
        else float(abs_ball)
    )
    bounds = np.asarray(base.SHGO_BOUNDS, float)
    n = bounds.shape[0]
    out = {
        "status": "ok",
        "expression": str(expression),
        "constants": [float(c) for c in constants],
        "rho": rho,
        "abs_ball": abs_ball,
    }
    try:
        # --- symbolic V, a, b and the CPU callables ---------------------------------
        t0 = time.perf_counter()
        # the tree with its constants is V itself here (V3: no wrap); base._sympy_expression
        # returns a string, sympy_expression makes the sympy object the CPU checks need
        V = sympy_expression(str(expression), list(constants))
        x_syms = state_symbols(n)
        a, b = ab_expressions(V, base.fSR, base.GSR, x_syms, input_index)
        fns = polish_callables(V, a, b, x_syms)
        out["build_s"] = time.perf_counter() - t0

        # --- DET4 lattice (origin excluded only at 0) -------------------------------
        t0 = time.perf_counter()
        geometry = scan_geometry(bounds, **LATTICE)
        coords = geometry["coordinates"]
        keep = np.einsum("ij,ij->i", coords, coords) > 0.0
        A = eval_lattice(fns["a_raw"], coords)
        B = eval_lattice(fns["b_raw"], coords)
        W = eval_lattice(fns["V_raw"], coords) - fns["v0"]
        out["scan_s"] = time.perf_counter() - t0

        # --- DET4 on the whole box: c_v* and kappa* ---------------------------------
        t0 = time.perf_counter()
        d4 = det4_from_lattice(
            fns,
            A,
            B,
            W,
            coords,
            keep,
            bounds,
            (rho,),
            gamma1,
            origin_tol=0.0,
            top_k=top_k,
            maxiter=maxiter,
        )
        row = d4.rows[rho]
        out.update(
            {
                "c_v_star": d4.pd_rate,
                "pd_valid": bool(d4.pd_valid),
                "pd_worst_r": d4.pd_worst_r,
                "kappa_star": row.kappa_star,
                "kappa_valid": bool(row.artstein_valid),
                "kappa_worst_r": row.worst_r,
                "kappa_worst_W": row.worst_W,
                "kappa_worst_abs_b": row.worst_abs_b,
                "det4_s": time.perf_counter() - t0,
            }
        )

        # --- exact check (absolute margin, numerical ball) + W-descent -> c_viol ----
        t0 = time.perf_counter()
        ex = check_b_manifold_exact(
            V,
            base.fSR,
            base.GSR,
            bounds,
            gamma1=gamma1,
            input_index=input_index,
            origin_tol=abs_ball,
            polish_top_k=top_k,
            polish_maxiter=maxiter,
        )
        out.update(
            {
                "exact_status": ex.status,
                "n_roots": int(ex.n_roots),
                "n_violations": int(ex.n_violations),
                "margin_max": float(ex.margin_max),
            }
        )
        c_viol, x_viol = np.inf, None
        if ex.status == "ok" and ex.n_violations > 0:
            c_viol, x_viol = min_w_violator(
                fns,
                ex.violation_points,
                bounds,
                gamma1,
                abs_ball,
                top_k=top_k,
                maxiter=maxiter,
            )
            raw = (
                eval_lattice(fns["V_raw"], np.asarray(ex.violation_points, float))
                - fns["v0"]
            )
            raw = raw[np.isfinite(raw)]
            out["c_viol_raw"] = float(np.min(raw)) if raw.size else None
        out["c_viol"] = None if not np.isfinite(c_viol) else float(c_viol)
        out["x_viol"] = None if x_viol is None else [float(v) for v in x_viol]
        out["exact_s"] = time.perf_counter() - t0

        # --- boundary and the certified level ---------------------------------------
        t0 = time.perf_counter()
        bmin, bx = boundary_min_w(fns, bounds)
        out["boundary_min"] = None if not np.isfinite(bmin) else float(bmin)
        c_star = 0.0 if not d4.pd_valid else float(min(bmin, c_viol))
        c_star = max(c_star, 0.0) if np.isfinite(c_star) else 0.0
        out["c_star"] = c_star
        out["c_star_binding"] = (
            "not_pd"
            if not d4.pd_valid
            else ("violator" if c_viol <= bmin else "boundary")
        )
        # volume on the run's grid
        lin = [np.linspace(lo, hi, grid_points) for lo, hi in bounds]
        mesh = np.meshgrid(*lin, indexing="ij")
        pts = np.stack([m.ravel() for m in mesh], axis=1)
        Wg = eval_lattice(fns["V_raw"], pts) - fns["v0"]
        out["certified_volume"] = (
            float(np.mean(np.isfinite(Wg) & (Wg < c_star))) if c_star > 0.0 else 0.0
        )
        out["boundary_s"] = time.perf_counter() - t0

        # --- DET4 restricted to the certified set -----------------------------------
        t0 = time.perf_counter()
        if c_star > 0.0:
            d4c = det4_from_lattice(
                fns,
                A,
                B,
                W,
                coords,
                keep,
                bounds,
                (rho,),
                gamma1,
                origin_tol=0.0,
                top_k=top_k,
                maxiter=maxiter,
                sublevel=c_star,
            )
            rc = d4c.rows[rho]
            out.update(
                {
                    "kappa_star_c": rc.kappa_star,
                    "kappa_c_valid": bool(rc.artstein_valid),
                    "kappa_c_worst_r": rc.worst_r,
                    "kappa_c_worst_W": rc.worst_W,
                }
            )
        else:
            out.update({"kappa_star_c": None, "kappa_c_valid": False})
        out["sublevel_s"] = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        out["status"] = f"error: {type(exc).__name__}: {str(exc)[:200]}"
    out["total_s"] = time.perf_counter() - t_all
    return out


def _g(v, nd=4):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "na"
    return "na" if not np.isfinite(v) else f"{v:.{nd}g}"


def format_certificate(c):
    """One line for the log."""
    if c.get("status") != "ok":
        return f"certificate: {c.get('status')}"
    return (
        f"certificate rho={_g(c['rho'])}: PD={'yes' if c['pd_valid'] else 'NO'}"
        f" c_v*={_g(c['c_v_star'])}"
        f" | kappa*={_g(c['kappa_star'])} ({'valid' if c['kappa_valid'] else 'invalid'},"
        f" worst r={_g(c['kappa_worst_r'], 3)} W={_g(c['kappa_worst_W'], 3)})"
        f" | exact roots={c['n_roots']} viol={c['n_violations']}"
        f" margin_max={_g(c['margin_max'], 3)}"
        f" | boundary_min={_g(c['boundary_min'])} c_viol={_g(c['c_viol'])}"
        f" -> c*={_g(c['c_star'])} ({c['c_star_binding']}) vol={_g(c['certified_volume'], 3)}"
        f" | kappa*(c*)={_g(c['kappa_star_c'])} [{c['total_s']:.0f}s]"
    )


def load_records(path):
    """Records with an expression and constants (jsonl or a single json object)."""
    records = []
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    try:
        obj = json.loads(text)
        records = obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        for line in text.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return [r for r in records if "expression" in r and r.get("constants") is not None]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("records", nargs="?", help="jsonl/json with expression + constants")
    ap.add_argument("--expression")
    ap.add_argument("--constants", help="comma-separated")
    ap.add_argument("--rho", type=float, default=None, help="default: base.BC_RHO")
    ap.add_argument(
        "--last", action="store_true", help="only the last record of the file"
    )
    ap.add_argument("--out", default=None, help="append the JSON results here")
    args = ap.parse_args()

    if args.expression:
        records = [
            {
                "expression": args.expression,
                "constants": [float(v) for v in args.constants.split(",")],
            }
        ]
    elif args.records:
        records = load_records(args.records)
        if args.last:
            records = records[-1:]
    else:
        ap.error("give a records file or --expression/--constants")

    base = _load_base()
    rho = args.rho or getattr(base, "BC_RHO", 1000.0)
    ball = getattr(base, "ABSOLUTE_CHECK_BALL_RADIUS", C.ORIGIN_TOL)
    print(
        f"certify: box {tuple(float(h) for _, h in base.SHGO_BOUNDS)} rho {rho}"
        f" abs_ball {ball} origin_exclusion(ratios) 0",
        flush=True,
    )
    for k, rec in enumerate(records):
        cert = certify(rec["expression"], rec["constants"], base, rho=args.rho)
        tag = rec.get("generation", rec.get("index", k))
        print(f"[{tag}] {rec['expression']}")
        print("     " + format_certificate(cert), flush=True)
        if args.out:
            with open(args.out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"record": tag, **cert}) + "\n")


if __name__ == "__main__":
    main()
