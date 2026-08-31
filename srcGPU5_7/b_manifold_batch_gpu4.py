"""Batched exact b=0 manifold check: many candidates per GPU call.

Same accuracy as ``srcGPU5_7.b_manifold_check_gpu3`` (numba dual-number polish,
reference decision logic) but the two launch-overhead-bound GPU stages run
once for the whole batch:

* scan: ``RuntimeExactBatchBundle.b_batch`` evaluates all C candidates on the
  shared 2.61M-point lattice in one fused pass (C rows out), instead of C
  separate 2.61M-point launches;
* bisection: every candidate's sign-change brackets are padded to the batch
  maximum and refined in ONE ``bisect_line`` call. Bisection is 60 sequential
  tiny kernel launches, so doing it per-candidate is ~34x slower than doing a
  batch together (measured) -- this is the main win.

Root extraction (host masks) and the SLSQP polish stay per-candidate: the
polish is CPU numba (``srcGPU5_7.cpu_polish``) and cannot be GPU-batched without
returning to the inaccurate GPU-SQP polish GPU2 used.

Per-candidate results are identical to running GPU3 one at a time (verified);
the batch only changes how the GPU work is dispatched.
"""

import time

import numpy as np
from scipy.optimize import minimize

import srcGPU5_7  # noqa: F401  (float64 before JAX)

import jax
import jax.numpy as jnp

# srcGPU2 only sets the JAX_ENABLE_X64 env var, which is a no-op if anything
# imported jax first; float64 arrays then silently become float32 and the
# Pallas kernels fail with "Invalid dtype for swap". Force it explicitly.
jax.config.update("jax_enable_x64", True)

from src.b_manifold_check import BManifoldResult
from srcGPU5_7.b_manifold_check_gpu import _device_nonzero_many, _scan_geometry
from srcGPU5_7.runtime_exact_candidate import (
    RuntimeExactCandidate,
    runtime_candidate_batch_bundle,
)
from srcGPU5_7.cpu_polish import make_cpu_polish_callables


def _batch_extract(scan_host, geometry, n, scan_points,
                   random_lines, random_line_points, zero_tol, C):
    """Extract every candidate's roots/brackets from the host b-field in one
    pass -- pure numpy, no per-candidate device round trips.

    ``scan_host`` is the host (C, P) b-field. For each scan field the mask is
    formed over the whole (C, ...) block and one ``np.nonzero`` yields a
    candidate-index axis plus the field coordinates; grouping by that axis
    reproduces, per candidate and in the same order, exactly what the GPU3
    single-candidate extraction (``_device_nonzero_many`` per candidate)
    produced. Bit-identical brackets -> bit-identical roots.
    """
    per_zero = [[] for _ in range(C)]
    acc = [{"o": [], "d": [], "lo": [], "hi": [], "blo": []}
           for _ in range(C)]

    def add(i, o, d, lo, hi, blo):
        acc[i]["o"].append(o)
        acc[i]["d"].append(d)
        acc[i]["lo"].append(lo)
        acc[i]["hi"].append(hi)
        acc[i]["blo"].append(blo)

    # --- stage 0: mesh -------------------------------------------------------
    if geometry["mesh"] is not None:
        axes_lin, g, mesh_slice = geometry["mesh"]
        bm = scan_host[:, mesh_slice].reshape((C,) + (g,) * n)
        cz = np.nonzero(np.abs(bm) <= zero_tol)
        c = cz[0]
        coords = cz[1:]
        for i in range(C):
            sel = c == i
            if np.any(sel):
                per_zero[i].append(
                    np.stack([axes_lin[k][coords[k][sel]] for k in range(n)],
                             axis=1)
                )
        sg = np.sign(bm)
        for axis in range(n):
            lo_sl = (slice(None),) + tuple(
                slice(None, -1) if k == axis else slice(None)
                for k in range(n))
            hi_sl = (slice(None),) + tuple(
                slice(1, None) if k == axis else slice(None)
                for k in range(n))
            mc = np.nonzero(sg[lo_sl] * sg[hi_sl] < 0)
            c = mc[0]
            coords = mc[1:]
            for i in range(C):
                sel = c == i
                if not np.any(sel):
                    continue
                idx = [coords[k][sel] for k in range(n)]
                points = np.stack([axes_lin[k][idx[k]] for k in range(n)],
                                  axis=1)
                origins = points.copy()
                origins[:, axis] = 0.0
                directions = np.zeros_like(points)
                directions[:, axis] = 1.0
                blo = bm[(np.full(idx[0].shape, i),) + tuple(idx)]
                add(i, origins, directions,
                    axes_lin[axis][idx[axis]].astype(float),
                    axes_lin[axis][idx[axis] + 1].astype(float), blo)

    # --- stage 1: axis-aligned line scans ------------------------------------
    for axis, others, combos, s, scan_slice in geometry["axes"]:
        L = combos.shape[0]
        bv = scan_host[:, scan_slice].reshape(C, L, scan_points)
        zc = np.nonzero(np.abs(bv) <= zero_tol)
        c, zl, zs = zc
        for i in range(C):
            sel = c == i
            if np.any(sel):
                zpts = np.empty((int(sel.sum()), n))
                for k, coordinate in enumerate(others):
                    zpts[:, coordinate] = combos[zl[sel], k]
                zpts[:, axis] = s[zs[sel]]
                per_zero[i].append(zpts)
        sg = np.sign(bv)
        ac = np.nonzero(sg[:, :, :-1] * sg[:, :, 1:] < 0)
        c, al, asi = ac
        for i in range(C):
            sel = c == i
            if not np.any(sel):
                continue
            ci = al[sel]
            si = asi[sel]
            points = np.empty((ci.size, n))
            for k, coordinate in enumerate(others):
                points[:, coordinate] = combos[ci, k]
            origins = points.copy()
            origins[:, axis] = 0.0
            directions = np.zeros_like(points)
            directions[:, axis] = 1.0
            add(i, origins, directions, s[si].astype(float),
                s[si + 1].astype(float), bv[i, ci, si])

    # --- stage 2: random-direction line scans --------------------------------
    if geometry["random"] is not None:
        P0, D, T, random_slice = geometry["random"]
        bv = scan_host[:, random_slice].reshape(
            C, random_lines, random_line_points
        )
        sg = np.sign(bv)
        rc = np.nonzero(sg[:, :, :-1] * sg[:, :, 1:] < 0)
        c, rl, rt = rc
        for i in range(C):
            sel = c == i
            if not np.any(sel):
                continue
            li = rl[sel]
            ti = rt[sel]
            add(i, P0[li], D[li], T[li, ti].astype(float),
                T[li, ti + 1].astype(float), bv[i, li, ti])

    per_brackets = [None] * C
    for i in range(C):
        if acc[i]["o"]:
            per_brackets[i] = {
                "origins": np.concatenate(acc[i]["o"], axis=0),
                "directions": np.concatenate(acc[i]["d"], axis=0),
                "lo": np.concatenate(acc[i]["lo"], axis=0),
                "hi": np.concatenate(acc[i]["hi"], axis=0),
                "blo": np.concatenate(acc[i]["blo"], axis=0),
            }
    return per_zero, per_brackets


def check_b_manifold_exact_gpu4_batch(
    candidates,
    fSR,
    GSR,
    bounds,
    gamma1=0.0,
    input_index=1,
    scan_axes=(2, 3),
    scan_points=801,
    grid_points_per_axis=9,
    margin_tol=0.0,
    origin_tol=1e-9,
    refine_iters=60,
    zero_tol=1e-14,
    random_lines=200,
    random_line_points=401,
    rng_seed=0,
    polish_top_k=40,
    polish_maxiter=60,
    polish_b_tol=1e-8,
    mesh_points_per_axis=0,
):
    """Batched exact manifold check. ``candidates`` is a list of
    ``RuntimeExactCandidate``; returns a list of ``BManifoldResult`` (one per
    candidate, same order). Same knobs/decision logic as GPU3.
    """
    started = time.perf_counter()
    candidates = list(candidates)
    C = len(candidates)
    bounds = np.asarray(bounds, dtype=float)
    n = bounds.shape[0]
    results = [None] * C
    metrics = {"engine": "gpu4_batch", "candidate_count": C,
               "scan_s": 0.0, "extract_s": 0.0, "bisect_s": 0.0,
               "polish_s": 0.0, "final_s": 0.0, "total_s": 0.0}

    if C == 0:
        return results

    bundle = runtime_candidate_batch_bundle(candidates)
    geometry = _scan_geometry(
        bounds, scan_axes, scan_points, grid_points_per_axis,
        random_lines, random_line_points, rng_seed, mesh_points_per_axis,
    )

    # ---- batched scan: all candidates, one fused pass -----------------------
    t0 = time.perf_counter()
    scan_values = bundle.b_batch(geometry["coordinates"])  # (C, P)
    scan_values = jax.device_get(scan_values)
    metrics["scan_s"] = time.perf_counter() - t0

    # ---- batched bracket extraction (host numpy, no per-candidate syncs) ----
    t0 = time.perf_counter()
    per_zero, per_brackets = _batch_extract(
        scan_values, geometry, n, scan_points,
        random_lines, random_line_points, zero_tol, C,
    )
    metrics["extract_s"] = time.perf_counter() - t0

    # ---- batched bisection: pad brackets to the batch max, one call ---------
    t0 = time.perf_counter()
    counts = [0 if b is None else b["origins"].shape[0] for b in per_brackets]
    max_m = max(counts) if counts else 0
    bisected_roots = [None] * C
    if max_m > 0:
        # the batched bisect kernel requires a block-aligned point count
        block = int(bundle.block_size)
        max_m = ((max_m + block - 1) // block) * block
        pad = lambda a, m: (  # noqa: E731
            a if a.shape[0] == m
            else np.concatenate([a, np.repeat(a[:1], m - a.shape[0], axis=0)],
                                axis=0)
        )
        O = np.zeros((C, max_m, n))
        Dd = np.zeros((C, max_m, n))
        Dd[:, :, 0] = 1.0  # any nonzero direction for pad rows
        Lo = np.zeros((C, max_m))
        Hi = np.zeros((C, max_m))
        Blo = np.zeros((C, max_m))
        for i, b in enumerate(per_brackets):
            if b is None or b["origins"].shape[0] == 0:
                continue
            m = b["origins"].shape[0]
            O[i, :m] = b["origins"]
            Dd[i, :m] = b["directions"]
            Lo[i, :m] = b["lo"]
            Hi[i, :m] = b["hi"]
            Blo[i, :m] = b["blo"]
        roots = bundle.bisect_line(
            jnp.asarray(O), jnp.asarray(Dd), jnp.asarray(Lo),
            jnp.asarray(Hi), jnp.asarray(Blo), int(refine_iters),
        )
        roots = np.asarray(jax.device_get(roots))  # (C, max_m, n)
        for i in range(C):
            if counts[i] > 0:
                bisected_roots[i] = roots[i, :counts[i]]
    metrics["bisect_s"] = time.perf_counter() - t0

    # roots per candidate (n, K_i)
    Rs = []
    for i in range(C):
        root_list = list(per_zero[i])
        if bisected_roots[i] is not None:
            root_list.append(bisected_roots[i])
        Rs.append(
            np.concatenate(root_list, axis=0).T if root_list
            else np.empty((n, 0))
        )

    def batched_a(root_columns):
        """a-value for every candidate's roots via one batched ab_batch.

        root_columns: list of (n, K_i). Returns list of (K_i,) a arrays.
        """
        ks = [R.shape[1] for R in root_columns]
        mk = max(ks) if ks else 0
        if mk == 0:
            return [np.empty((0,)) for _ in root_columns]
        pts = np.zeros((C, mk, n))
        for i, R in enumerate(root_columns):
            if ks[i] > 0:
                pts[i, : ks[i]] = R.T
        ab = np.asarray(jax.device_get(bundle.ab_batch(jnp.asarray(pts))))
        return [ab[i, 0, : ks[i]] for i in range(C)]

    # ---- batched seed scoring -> per-candidate top-K --------------------
    t0 = time.perf_counter()
    a_seeds = batched_a(Rs)
    metrics["final_s"] += time.perf_counter() - t0

    # ---- per-candidate numba SLSQP polish (CPU) -------------------------
    t0 = time.perf_counter()
    r0_sq = (origin_tol * (1.0 + 1e-6)) ** 2
    box = [tuple(bounds[j]) for j in range(n)]
    for i, cand in enumerate(candidates):
        R = Rs[i]
        if polish_top_k <= 0 or R.shape[1] == 0:
            continue
        m_seed = a_seeds[i] + gamma1
        m_seed = np.where(np.isfinite(m_seed), m_seed, -np.inf)
        order = np.argsort(m_seed)[-int(polish_top_k):]
        fns = make_cpu_polish_callables(
            cand.expression, cand.constants, fSR, GSR, gamma1, input_index
        )
        a_fn, b_fn = fns["a_fn"], fns["b_fn"]
        ga_fn, gb_fn = fns["ga_fn"], fns["gb_fn"]
        polished = []
        for idx in order:
            x0 = np.clip(R[:, idx], bounds[:, 0], bounds[:, 1])
            try:
                res = minimize(
                    lambda z: -float(a_fn(*z)), x0,
                    jac=lambda z: -np.asarray(ga_fn(*z), float).ravel(),
                    constraints=[
                        {"type": "eq", "fun": lambda z: float(b_fn(*z)),
                         "jac": lambda z: np.asarray(gb_fn(*z), float).ravel()},
                        {"type": "ineq",
                         "fun": lambda z: float(z @ z) - r0_sq,
                         "jac": lambda z: 2.0 * z}],
                    bounds=box, method="SLSQP",
                    options={"maxiter": int(polish_maxiter), "ftol": 1e-12})
                z = np.clip(res.x, bounds[:, 0], bounds[:, 1])
                if np.all(np.isfinite(z)) and abs(float(b_fn(*z))) \
                        <= polish_b_tol:
                    polished.append(z)
            except Exception:
                continue
        if polished:
            Rs[i] = np.concatenate([R, np.array(polished).T], axis=1)
    metrics["polish_s"] = time.perf_counter() - t0

    # ---- batched final scoring + reference decision logic ---------------
    t0 = time.perf_counter()
    a_finals = batched_a(Rs)
    for i in range(C):
        R = Rs[i]
        if R.shape[1] == 0:
            results[i] = BManifoldResult(0, 0, np.nan, np.empty((0, n)), "ok")
            results[i].margin_mean_pos = np.nan
            continue
        margin = a_finals[i] + gamma1
        nonorigin = np.sqrt(np.sum(R ** 2, axis=0)) > origin_tol
        viol = nonorigin & np.isfinite(margin) & (margin > margin_tol)
        finite = margin[nonorigin & np.isfinite(margin)]
        margin_max = float(finite.max()) if finite.size else np.nan
        results[i] = BManifoldResult(
            n_roots=int(nonorigin.sum()),
            n_violations=int(viol.sum()),
            margin_max=margin_max,
            violation_points=R[:, viol].T,
            status="ok",
        )
        # GPU5_2, identical definition to srcGPU3 (purely additive).
        results[i].margin_mean_pos = (
            float(np.maximum(finite, 0.0).mean()) if finite.size else np.nan
        )
    metrics["final_s"] += time.perf_counter() - t0
    metrics["total_s"] = time.perf_counter() - started
    # per-candidate metrics keyed like GPU3 so the existing stage logging works
    per_metrics = dict(metrics)
    per_metrics["scan_and_masks_s"] = (
        metrics["scan_s"] + metrics["extract_s"]
    ) / max(1, C)
    per_metrics["bisection_s"] = metrics["bisect_s"] / max(1, C)
    per_metrics["polish_s"] = metrics["polish_s"] / max(1, C)
    per_metrics["final_score_s"] = metrics["final_s"] / max(1, C)
    for r in results:
        if r is not None:
            r.gpu4_metrics = dict(metrics)
            r.gpu2_metrics = per_metrics  # alias for GPU3-style diagnostics
    return results
