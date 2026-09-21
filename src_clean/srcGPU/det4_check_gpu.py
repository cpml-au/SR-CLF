"""DET4 on the GPU: examples/VerifierBenchmark/run_det4_gpu.py as a check function.

The condition, the seeds and every polish are src.srcCPU.det4_check_cpu's (see its docstring);
only the lattice evaluation of a, b, V moves to the GPU, through the same two engines as
det3_check_gpu.py:

  engine="jax"     ab_batch / V_batch of the compact JAX bundle (any plant, any dimension)
  engine="pallas"  the srcGPU5_9 a/b kernel and PD scan kernel (4-D cart-pole only)

c_v* (normalised PD) once, then kappa* per rho; both verdicts from the ratios.
"""

from __future__ import annotations

import time

import numpy as np

import src.srcGPU  # noqa: F401  (float64 + XLA env before JAX)

import jax

jax.config.update("jax_enable_x64", True)

from src.srcCPU import conditions as C  # noqa: E402
from src.srcCPU.det3_check_cpu import (  # noqa: E402
    LATTICE,
    POLISH_MAXITER,
    POLISH_TOP_K,
    SOBOL_SEEDS,
)
from src.srcCPU.det4_check_cpu import (  # noqa: E402
    DEFAULT_RHOS,
    KAPPA_LADDER,
    KAPPA_MIN,
    PD_RATE_TOL,
    DET4Result,
    det4_from_lattice,
)
from src.srcCPU.exact_check_cpu import scan_geometry  # noqa: E402
from src.srcCPU.symbolic import (  # noqa: E402
    ab_expressions,
    polish_callables,
    state_symbols,
    sympy_expression,
)
from src.srcGPU.det3_check_gpu import lattice_jax, lattice_pallas  # noqa: E402
from src.srcGPU.exact_check_gpu import SCAN_CHUNK  # noqa: E402


def check_det4_gpu(
    expression,
    constants=None,
    *,
    fSR,
    GSR,
    bounds,
    rhos=DEFAULT_RHOS,
    gamma1=C.DEFAULT_GAMMA1,
    input_index=1,
    engine="jax",
    lattice=None,
    pd_eps=C.PD_EPS,
    pd_rate_tol=PD_RATE_TOL,
    kappa_min=KAPPA_MIN,
    kappa_ladder=KAPPA_LADDER,
    origin_tol=C.ORIGIN_TOL,
    b_tol=C.B_TOL,
    top_k=POLISH_TOP_K,
    maxiter=POLISH_MAXITER,
    sobol_seeds=SOBOL_SEEDS,
    scan_chunk=SCAN_CHUNK,
    sublevel=None,
    gate_on_pd=True,
):
    """DET4 for one candidate (GP prefix string + constants, or a sympy V), lattice on the
    GPU, polishes on the CPU.  ``sublevel``: kappa* over {W <= sublevel} only;
    ``gate_on_pd=False``: kappa* over the W > 0 region even when PD fails."""
    bounds = np.asarray(bounds, dtype=float)
    n = bounds.shape[0]
    result = DET4Result()
    try:
        t0 = time.perf_counter()
        x_syms = state_symbols(n)
        V = sympy_expression(expression, constants)
        a, b = ab_expressions(V, fSR, GSR, x_syms, input_index)
        fns = polish_callables(V, a, b, x_syms)  # CPU callables for the polishes
        result.build_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        coords = scan_geometry(bounds, **{**LATTICE, **(lattice or {})})["coordinates"]
        keep = np.einsum("ij,ij->i", coords, coords) > origin_tol**2
        if engine == "jax":
            A, B, Vl = lattice_jax(V, fSR, GSR, coords, input_index, scan_chunk)
        elif engine == "pallas":
            A, B, Vl = lattice_pallas(expression, constants, coords)
        else:
            raise ValueError(f"unknown engine {engine!r} (jax | pallas)")
        W = Vl - fns["v0"]
        result.scan_s = time.perf_counter() - t0

        det4_from_lattice(
            fns,
            A,
            B,
            W,
            coords,
            keep,
            bounds,
            rhos,
            gamma1,
            pd_eps,
            pd_rate_tol,
            kappa_min,
            kappa_ladder,
            origin_tol,
            b_tol,
            top_k,
            maxiter,
            sobol_seeds,
            result,
            sublevel,
            gate_on_pd,
        )
    except Exception as exc:
        result.status = f"error: {type(exc).__name__}: {str(exc)[:200]}"
    return result
