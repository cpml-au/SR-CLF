#!/usr/bin/env python3
"""Closed-loop check for saved 3-D cart-pole champions.

Runs the SATURATED Sontag controller from every ROLLOUT_X0S, UNCONDITIONALLY
-- the fitness only rolls out candidates that already passed every static
check, which is exactly the ones we do not need to ask about. Reports, per
initial condition: final ||x||, whether it diverged, and the peak |u| the
controller actually demanded.

    python rollout_check.py 122467_best_per_generation.jsonl 88 89
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("SYMCLF_GPU2_ALLOW_CPU", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

HERE = Path(__file__).resolve().parent
for path in (HERE.parents[1], HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402

import Evaluate  # noqa: E402
import src.Functions  # noqa: E402
from src3DCartPoleV3.fitness import (  # noqa: E402
    _cartpole_fields_np, _compute_exact_result, _attach_pd, _pd_program,
)
from src3DCartPoleV3.cpu_polish import _dual2 as _cpu_dual2  # noqa: E402


def rollout(expression, constants, base, u_max=None, horizon=None, dt=None):
    opcodes, operands, literals, n_ops, parameters = _pd_program(
        expression, np.asarray(constants, dtype=float), 0
    )
    op64, opr64 = opcodes.astype(np.int64), operands.astype(np.int64)

    def grad_V(state):
        return np.asarray(
            _cpu_dual2(op64, opr64, literals, n_ops, parameters, state)[1],
            dtype=float,
        )

    u_max = float(u_max if u_max is not None
                  else getattr(base, "CP3D_ROLLOUT_UMAX", 1000.0))
    horizon = float(horizon if horizon is not None
                    else getattr(base, "CP3D_ROLLOUT_T", 10.0))
    dt = float(dt if dt is not None else getattr(base, "CP3D_ROLLOUT_DT", 2e-3))
    diverge = float(getattr(base, "CP3D_ROLLOUT_DIVERGE_NORM", 1.0))
    peak = {"u": 0.0}

    def closed_loop(state):
        drift, gain = _cartpole_fields_np(state)
        g = grad_V(state)
        a = float(g @ drift)
        b = float(g @ gain)
        bs = 1.0e4 * b * b
        r2 = float(state @ state)
        lam = (np.sqrt(a * a + r2 * bs) + a) / (bs + 1.0e-6)
        u = float(np.clip(-1.0e4 * b * lam, -u_max, u_max))
        peak["u"] = max(peak["u"], abs(u))
        return drift + gain * u

    steps = int(horizon / dt)
    out = []
    for start in np.asarray(base.ROLLOUT_X0S, dtype=float).reshape(-1, 3):
        state = start.copy()
        peak["u"] = 0.0
        bad, t_bad = False, None
        for k in range(steps):
            k1 = closed_loop(state)
            k2 = closed_loop(state + 0.5 * dt * k1)
            k3 = closed_loop(state + 0.5 * dt * k2)
            k4 = closed_loop(state + dt * k3)
            state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if not np.all(np.isfinite(state)) or np.linalg.norm(state) > diverge:
                bad, t_bad = True, k * dt
                break
        n0 = float(np.linalg.norm(start))
        nf = float(np.linalg.norm(state)) if np.all(np.isfinite(state)) else float("inf")
        out.append((start, n0, nf, bad, t_bad, peak["u"]))
    return out


def main():
    path = Path(sys.argv[1])
    wanted = {int(g) for g in sys.argv[2:]} or None
    rows = [json.loads(l) for l in path.open() if l.strip()]
    picks = [r for r in rows if wanted is None or r["generation"] in wanted]

    axis = np.linspace(-0.25, 0.25, 21)
    data = src.Functions.Dataset("true_data", [axis] * 3, None)
    data.mesh = list(np.meshgrid(*[axis] * 3, indexing="ij"))
    from src3DCartPoleV3.grid_fitness import gpu_pre_exact_mse

    for rec in picks:
        gen, expr = rec["generation"], rec["expression"]
        consts = np.asarray(rec.get("constants", []), dtype=float)
        print("=" * 78)
        print(f"GEN {gen}   recorded fitness {rec['fitness']:.6f}")
        print(f"  {expr}")
        print(f"  constants {np.round(consts, 6).tolist()}")

        pre, entry = gpu_pre_exact_mse(expr, consts, data, Evaluate)
        print(f"\n  pre_exact (grid+ROA+sat) = {pre:.6f}")
        for k in sorted(entry):
            print(f"      {k:<22}{float(entry[k]):.6g}")

        res = _compute_exact_result(expr, consts, Evaluate)
        res = _attach_pd(res, expr, consts, Evaluate)
        pd = getattr(res, "gpu5_pd", None)
        print(f"\n  EXACT roots={getattr(res,'n_roots',None)} "
              f"viol={getattr(res,'n_violations',None)} "
              f"margin_max={float(getattr(res,'margin_max',float('nan'))):+.6e}")
        if pd is not None:
            print(f"  PD status={pd.status} positive_definite={pd.positive_definite}")

        print(f"\n  CLOSED LOOP (saturated Sontag, u_max="
              f"{getattr(Evaluate,'CP3D_ROLLOUT_UMAX',1000.0):g}, "
              f"T={getattr(Evaluate,'CP3D_ROLLOUT_T',10.0):g}s, "
              f"diverge at ||x||>"
              f"{getattr(Evaluate,'CP3D_ROLLOUT_DIVERGE_NORM',1.0):g})")
        print(f"    {'x0 (v, theta, thetadot)':<30}{'||x0||':>9}{'||xT||':>12}"
              f"{'verdict':>12}{'peak |u|':>12}")
        for start, n0, nf, bad, t_bad, pu in rollout(expr, consts, Evaluate):
            verdict = f"DIVERGED@{t_bad:.2f}s" if bad else (
                "converged" if nf < 0.02 * max(n0, 1e-9) else
                "converged" if nf < 1e-3 else "stalled")
            print(f"    {str(np.round(start,4)):<30}{n0:9.4f}{nf:12.4e}"
                  f"{verdict:>12}{pu:12.4g}")


if __name__ == "__main__":
    main()
