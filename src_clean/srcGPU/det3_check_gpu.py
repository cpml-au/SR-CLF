"""DET3 on the GPU: examples/VerifierBenchmark/run_det3_gpu.py as a check function.

The condition, the seeds, the manifold ascent and the (rho, kappa) polish are
src.srcCPU.det3_check_cpu's (see its docstring); only the lattice evaluation of a, b, V and the
folded-in exact Artstein + PD checks move to the GPU:

  engine="jax"     a, b via ab_batch and V via V_batch of the compact JAX bundle
                   (exact_check_gpu.get_jax_candidate), chunked; exact + PD from
                   exact_check_gpu (engine="jax").  Any plant, any dimension.
  engine="pallas"  a, b via the srcGPU5_9 a/b kernel (bundle.ab_batch), V via the PD scan
                   kernel (_scan_values); exact + PD = GPU3 + GPU5.  4-D cart-pole only.

Every (rho, kappa) row: margin_max = max over lattice U polished U manifold U exact-check
violation points outside the origin ball; strict (definite) and non-strict verdicts from it.
"""

from __future__ import annotations

import os
import time

import numpy as np

import src.srcGPU  # noqa: F401  (float64 + XLA env before JAX)

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from src.srcCPU import conditions as C  # noqa: E402
from src.srcCPU.det3_check_cpu import (  # noqa: E402
    LATTICE,
    POLISH_MAXITER,
    POLISH_TOP_K,
    SOBOL_SEEDS,
    DET3Result,
    det3_from_lattice,
)
from src.srcCPU.exact_check_cpu import scan_geometry  # noqa: E402
from src.srcCPU.symbolic import (  # noqa: E402
    ab_expressions,
    polish_callables,
    state_symbols,
    sympy_expression,
)
from src.srcGPU.exact_check_gpu import (  # noqa: E402
    SCAN_CHUNK,
    eval_on_device,
    get_jax_candidate,
    runtime_candidate,
    verify_exact_gpu,
)


def lattice_jax(V, fSR, GSR, coords, input_index=1, chunk=SCAN_CHUNK):
    """(A, B, V_lattice) on the host from the compact JAX bundle."""
    n = coords.shape[1]
    bundle, c = get_jax_candidate(V, fSR, GSR, n, input_index)
    c = jnp.asarray(c)
    A, B = eval_on_device(bundle.ab_batch, coords, c, chunk)
    Vl = eval_on_device(bundle.V_batch, coords, c, chunk)
    return A, B, Vl


def lattice_pallas(expression, constants, coords):
    """(A, B, V_lattice) on the host from the srcGPU5_9 Pallas kernels (4-D cart-pole)."""
    from src.srcGPU.gpu5_9.pallas_interpreter import DEFAULT_BLOCK_SIZE
    from src.srcGPU.gpu5_9.pd_check_gpu5_9 import _program, _scan_values
    from src.srcGPU.gpu5_9.runtime_exact_candidate import runtime_candidate_bundle

    cand = runtime_candidate(expression, constants)
    bs = int(os.environ.get("SYMCLF_GPU2_PALLAS_BLOCK_SIZE", DEFAULT_BLOCK_SIZE))
    bundle, cvals = runtime_candidate_bundle(cand)
    coords_dev = jnp.asarray(coords)
    a_dev, b_dev = bundle.ab_batch(coords_dev, cvals)
    A = np.asarray(jax.device_get(a_dev), dtype=float).reshape(-1)
    B = np.asarray(jax.device_get(b_dev), dtype=float).reshape(-1)
    opc, opr, lit, n_ops, params = _program(cand.expression, cand.constants, bs)
    Vl = np.asarray(
        _scan_values(opc, opr, lit, n_ops, params, coords_dev, bs, 262144), dtype=float
    ).reshape(-1)
    return A, B, Vl


def check_det3_gpu(
    expression,
    constants=None,
    *,
    fSR,
    GSR,
    bounds,
    rhos=C.DEFAULT_RHOS,
    kappas=C.DEFAULT_KAPPAS,
    gamma1=C.DEFAULT_GAMMA1,
    input_index=1,
    with_pd=True,
    with_exact=True,
    engine="jax",
    lattice=None,
    origin_tol=C.ORIGIN_TOL,
    b_tol=C.B_TOL,
    margin_tol=C.MARGIN_TOL,
    top_k=POLISH_TOP_K,
    maxiter=POLISH_MAXITER,
    sobol_seeds=SOBOL_SEEDS,
    scan_chunk=SCAN_CHUNK,
):
    """DET3 for one candidate (GP prefix string + constants, or a sympy V) at every
    (rho, kappa), lattice and exact/PD checks on the GPU.  kappa = 0 gives the DET2 row.
    """
    bounds = np.asarray(bounds, dtype=float)
    n = bounds.shape[0]
    result = DET3Result()
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

        extra = None
        if with_exact or with_pd:
            ex = verify_exact_gpu(
                expression,
                constants,
                fSR=fSR,
                GSR=GSR,
                bounds=bounds,
                gamma1=gamma1,
                input_index=input_index,
                with_artstein=with_exact,
                with_pd=with_pd,
                engine=engine,
                artstein_kwargs=dict(
                    margin_tol=-margin_tol,
                    origin_tol=origin_tol,
                    polish_b_tol=b_tol,
                    polish_top_k=top_k,
                    polish_maxiter=maxiter,
                ),
                pd_kwargs=dict(
                    origin_tol=origin_tol, polish_top_k=top_k, polish_maxiter=maxiter
                ),
            )
            if with_exact:
                result.exact = ex.artstein
                extra = ex.artstein.violation_points
                result.exact_s = ex.artstein_s
            if with_pd:
                result.pd = ex.pd
                result.pd_valid = ex.pd_valid if ex.pd.status == "ok" else False
                result.pd_s = ex.pd_s

        result.rows = det3_from_lattice(
            fns,
            A,
            B,
            W,
            coords,
            keep,
            bounds,
            rhos,
            kappas,
            gamma1,
            extra,
            origin_tol,
            b_tol,
            margin_tol,
            top_k,
            maxiter,
            sobol_seeds,
        )
    except Exception as exc:
        result.status = f"error: {type(exc).__name__}: {str(exc)[:200]}"
    return result
