"""GPU5_7 GP fitness: GPU5_6's pipeline scored on the REAL closed-loop
conditions.

Same staging as GPU5_6 (grid -> MSE gate -> cheap Artstein screen -> a_max
gate -> batched exact check + PD check), same gate pricing, same tuner OOM
retry. What changed, and why (measured on runs 122304 / 122340):

1. The grid stage adds the SATURATION and ROA terms and zeroes the priors
   (properness band, V_GRAD targets, c_star target, coverage). See
   ``grid_fitness._score_kernel`` -- the gen-7 "certified" champion of run
   122340 needs |u| = 1.46e5 (1e11 along its orbits) and diverges from inside
   its own certified set, while the "invalid" 122304 champion needs |u| = 59
   and converges from everywhere tested. Feasibility with bounded control,
   ``a < u_max*|b|`` off the origin, is the condition both previous fitness
   functions and both exact checkers were blind to.
2. The exact stage prices CERTIFICATE DEPTH: each violation the exact
   Artstein/PD searches find is priced by how low in W = V - V(0) it sits
   (``_cert_penalty``). A violation at W ~ 0 empties the certified sublevel
   set; one near the box boundary barely shrinks it. Both current champions
   park their violations at W ~ 0 exactly because nothing charged for it.
3. A closed-loop ROLLOUT gate (``_attach_rollout``): a candidate that passes
   every static check is integrated with the SATURATED Sontag controller
   from the standard initial conditions; divergence is charged. This is the
   test the gen-7 champion actually failed.
4. The PD penalty is scale-invariant and reads the shared near-zero curve
   (see ``pd_check_gpu5_7.pd_result_penalty``).

Inherited from GPU5_3..5_6 and kept: monotone gate pricing (rejection is
never cheaper than the check), max() instead of cheap+exact double charge,
violating FRACTION not count, nested-call charge 50, the strict margin
predicate (a == 0 is a violation), and the ARE reference form.
"""
from __future__ import annotations

import hashlib
import os
import time
import uuid

import numpy as np

from src.SymFunctions import detect_nested_function_calls, get_features_batch
from srcGPU5_7.artstein_gpu5_7 import (
    check_artstein_gpu_cheap_many_from_base,
)
from srcGPU5_7.b_manifold_check_gpu3 import (
    check_b_manifold_exact_gpu3_cached as check_b_manifold_exact_gpu_cached,
)
from srcGPU5_7.b_manifold_batch_gpu4 import check_b_manifold_exact_gpu4_batch
from srcGPU5_7.pd_check_gpu5_7 import (
    check_positive_definite_gpu5,
    pd_result_penalty,
    _program as _pd_program,
)
from srcGPU5_7.cpu_polish import _dual2 as _cpu_dual2
from srcGPU5_7.grad_norm_gpu5_7 import a_and_grad_norms
from srcGPU5_7.grid_fitness import (
    encode_expression,
    gpu_pre_exact_mse_many,
    gpu_tune_programs,
)
from srcGPU5_7.runtime_exact_candidate import RuntimeExactCandidate


_INITIAL_RANDOM_POPULATION = int(
    os.environ.get("SYMCLF_GPU2_TUNER_RANDOM_POPULATION", "35")
)
_SEA_GENERATIONS = int(
    os.environ.get("SYMCLF_GPU2_TUNER_GENERATIONS", "5")
)
_TUNER_SEED = int(os.environ.get("SYMCLF_GPU2_TUNER_SEED", "0"))
_TUNER_CALL = 0

# GPU5_3: per-nested-function-class charge, was a hardcoded 100000.
_NESTED_CALL_PENALTY = float(
    os.environ.get("SYMCLF_GPU5_7_NESTED_PENALTY", "50")
)

# GPU5_6: smallest SEA population an OOM retry may fall back to.
_TUNER_MIN_POPULATION = max(
    2, int(os.environ.get("SYMCLF_GPU5_7_TUNER_MIN_POPULATION", "5"))
)
_TUNER_OOM_RETRIES = 0
# Widest fusion / largest SEA population known to fit on this process's device.
# Lowered permanently the first time an OOM forces a smaller setting.
_TUNER_WIDTH_CAP = 1 << 30
_TUNER_POPULATION_CAP = 1 << 30

_OOM_MARKERS = (
    "RESOURCE_EXHAUSTED",
    "Out of memory",
    "OutOfMemory",
    "CUDA_ERROR_OUT_OF_MEMORY",
)


def _is_oom(exc):
    """True for the device out-of-memory failures the tuner can retry.

    JAX surfaces this as a bare ValueError or an XlaRuntimeError depending on
    where it is raised, so match on the message rather than the type. Anything
    that is not an OOM must propagate untouched -- a silent retry loop around a
    real bug would be far worse than the crash it replaces.
    """
    text = f"{type(exc).__name__}: {exc}"
    return any(marker in text for marker in _OOM_MARKERS)


def _finite_scores(values):
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values) & (values < 1.0e10), values, 1.0e10)


def _score_rows(expressions, constants, true_data, base):
    values, _ = gpu_pre_exact_mse_many(
        expressions, constants, true_data, base
    )
    return _finite_scores(values)


def _tune_group(expressions, n_constants, true_data, base, fusion_width):
    """Tune programs with one device-resident SEA run and one final transfer.

    GPU5_6: an out-of-memory failure here retries at a smaller working set
    instead of killing the run.

    The tuner's device allocation scales with
    ``population_size x fusion_width x grid_points**4``. On the A100s that is a
    single ~6.89 GiB request against a pool already ~35 GiB preallocated across
    three actors per GPU, and it has now killed three separate runs at the
    identical size -- jobs 122252, 122299 and 122339 all died on
    ``RESOURCE_EXHAUSTED: Out of memory while trying to allocate 7397782584
    bytes``, after hours of useful work and after several *recoverable* OOMs on
    the same allocation. 122339 lost 102 generations at fitness 392.80.

    The allocation is a throughput choice, not a correctness one: halving the
    fusion width halves the memory and produces the same tuned constants from
    two passes instead of one. So an OOM degrades rather than aborts. Each
    retry halves the fusion width, then the SEA population, and only a failure
    at fusion width 1 with a minimal population is genuinely fatal.
    """
    global _TUNER_CALL
    seed = (
        _TUNER_SEED
        + os.getpid()
        + 0x9E3779B9 * _TUNER_CALL
    ) & 0xFFFFFFFF
    _TUNER_CALL += 1

    # Remember the widest setting that has actually worked in this process.
    # Without this every call would re-attempt the failing allocation, paying
    # one or two doomed ~6.9 GiB requests per group for the rest of the run.
    global _TUNER_WIDTH_CAP, _TUNER_POPULATION_CAP
    width = max(1, min(int(fusion_width), _TUNER_WIDTH_CAP))
    population = min(_INITIAL_RANDOM_POPULATION + 1, _TUNER_POPULATION_CAP)
    attempt = 0
    while True:
        try:
            # gpu_tune_programs requires len(group) <= fusion_width, so a
            # narrower width means MORE passes over the same group, not a
            # truncated one. Every expression is still tuned, with the same
            # population and generations; only the device working set shrinks.
            scores = []
            constants = []
            for start in range(0, len(expressions), width):
                stop = min(start + width, len(expressions))
                chunk_scores, chunk_constants = gpu_tune_programs(
                    expressions[start:stop],
                    n_constants[start:stop],
                    true_data,
                    base,
                    population_size=population,
                    generations=_SEA_GENERATIONS,
                    fusion_width=width,
                    seed=seed + start,
                )
                scores.extend(list(chunk_scores)[:stop - start])
                constants.extend(list(chunk_constants)[:stop - start])
            if len(scores) != len(expressions):
                raise RuntimeError(
                    "GPU5_6 tuner returned "
                    f"{len(scores)} results for {len(expressions)} programs"
                )
            _TUNER_WIDTH_CAP = min(_TUNER_WIDTH_CAP, width)
            _TUNER_POPULATION_CAP = min(_TUNER_POPULATION_CAP, population)
            return scores, constants
        except Exception as exc:
            if not _is_oom(exc):
                raise
            if width > 1:
                width = max(1, width // 2)
            elif population > _TUNER_MIN_POPULATION:
                population = max(_TUNER_MIN_POPULATION, population // 2)
            else:
                raise
            attempt += 1
            global _TUNER_OOM_RETRIES
            _TUNER_OOM_RETRIES += 1
            if _TUNER_OOM_RETRIES <= 5 or _TUNER_OOM_RETRIES % 50 == 0:
                print(
                    "GPU5_6 tuner OOM -- retrying smaller "
                    f"(fusion={width}, population={population}, "
                    f"attempt {attempt}, total retries {_TUNER_OOM_RETRIES}): "
                    f"{str(exc).splitlines()[0][:140]}",
                    flush=True,
                )


def _exact_cache_key(expression, constants, base):
    digest = hashlib.sha256()
    digest.update(str(expression).encode("utf-8"))
    digest.update(np.asarray(constants, dtype=np.float64).tobytes())
    digest.update(np.asarray(base.SHGO_BOUNDS, dtype=np.float64).tobytes())
    digest.update(
        repr(
            (
                base.CLF_GAMMA1,
                base.MANIFOLD_MARGIN_TOL_EXACT,
                base.CLF_ORIGIN_EXCLUDE_RADIUS,
                getattr(base, "GPU2_MANIFOLD_POLISH_TOP_K", 40),
                getattr(base, "GPU2_MANIFOLD_POLISH_ITERATIONS", 60),
                getattr(base, "GPU2_MANIFOLD_POLISH_B_TOL", 1.0e-8),
                int(
                    getattr(
                        base,
                        "GPU3_MESH_POINTS",
                        os.environ.get("SYMCLF_GPU3_MESH_POINTS", "0"),
                    )
                ),
                # GPU5_7: the rollout verdict is stored ON the cached result,
                # so its knobs must be part of the key.
                bool(getattr(base, "GPU5_7_ROLLOUT_ENABLED", True)),
                float(getattr(base, "GPU5_7_ROLLOUT_UMAX",
                              getattr(base, "GPU5_7_SAT_U_TARGET", 1000.0))),
                float(getattr(base, "GPU5_7_ROLLOUT_T", 10.0)),
                float(getattr(base, "GPU5_7_ROLLOUT_DT", 2.0e-3)),
            )
        ).encode("ascii")
    )
    return digest.hexdigest()


def _exact_check_kwargs(base):
    return {
        "bounds": base.SHGO_BOUNDS,
        "gamma1": base.CLF_GAMMA1,
        "margin_tol": base.MANIFOLD_MARGIN_TOL_EXACT,
        "origin_tol": base.CLF_ORIGIN_EXCLUDE_RADIUS,
        "scan_axes": (0, 1, 2, 3),
        "polish_top_k": getattr(base, "GPU2_MANIFOLD_POLISH_TOP_K", 40),
        "polish_maxiter": getattr(
            base, "GPU2_MANIFOLD_POLISH_ITERATIONS", 60
        ),
        "polish_b_tol": getattr(
            base, "GPU2_MANIFOLD_POLISH_B_TOL", 1.0e-8
        ),
        # 0 = reference-strict root population (the default). Set the
        # GPU3_MESH_POINTS attribute / SYMCLF_GPU3_MESH_POINTS env to 21 to
        # reproduce the GPU2 additive mesh-bracket harvest.
        "mesh_points_per_axis": int(
            getattr(
                base,
                "GPU3_MESH_POINTS",
                os.environ.get("SYMCLF_GPU3_MESH_POINTS", "0"),
            )
        ),
    }


def _runtime_candidate(expression, constants):
    return RuntimeExactCandidate(
        expression=str(expression),
        constants=tuple(np.asarray(constants, dtype=np.float64).reshape(-1)),
    )


_PD_ERROR_REPORTED = False


def _attach_pd(result, expression, constants, base):
    """Run the GPU5 positive-definiteness check and attach it to the result.

    Positive definiteness of W = V - V(0) is verified with the same pipeline as
    the Artstein condition (dense GPU scan -> SLSQP polish with the origin ball
    as an inequality constraint), because the 15^4 fitness grid cannot see the
    violations: job 121524 reads +9.1e-4 on the grid and -0.0916 in truth.
    A candidate that fails this is not a Lyapunov function at all, whatever its
    b=0 margin says.
    """
    if result is None:
        return result
    try:
        pd = check_positive_definite_gpu5(
            _runtime_candidate(expression, constants),
            bounds=base.SHGO_BOUNDS,
            pd_eps=float(getattr(base, "GPU5_PD_EPS", 1.0e-4)),
            origin_tol=base.CLF_ORIGIN_EXCLUDE_RADIUS,
            polish_top_k=int(getattr(base, "GPU5_PD_POLISH_TOP_K", 40)),
            polish_maxiter=int(getattr(base, "GPU5_PD_POLISH_ITERATIONS", 60)),
            pd_reference_matrix=getattr(base, "GPU5_PD_REFERENCE_MATRIX", None),
        )
    except Exception as exc:  # fail closed: unknown PD counts as a failure
        pd = None
        message = f"{type(exc).__name__}: {exc}"
        setattr(result, "gpu5_pd_error", message)
        # This used to be silent, and a silent failure here is expensive: every
        # exact candidate then pays GPU5_PD_UNKNOWN_PENALTY (8000), which is a
        # CONSTANT with no gradient. On run 122230 that constant was 86% of the
        # fitness and pinned the search at ~9252 for hundreds of generations
        # while every other term was already solved. Shout once per process.
        global _PD_ERROR_REPORTED
        if not _PD_ERROR_REPORTED:
            _PD_ERROR_REPORTED = True
            print(f"GPU5 PD CHECK FAILING -- every exact candidate is paying "
                  f"the {float(getattr(base, 'GPU5_PD_UNKNOWN_PENALTY', 8000.0)):.0f} "
                  f"unknown-PD penalty: {message}", flush=True)
    setattr(result, "gpu5_pd", pd)

    # GPU5_7: the W-level of the PD minimiser, for the certificate-depth
    # price. At the minimiser g = W - pd_eps*Q is min_margin, so
    # W = min_margin + pd_eps*Q(min_point) -- no extra evaluation needed.
    # Only attached when PD actually fails; a passing check does not bound
    # the certificate.
    if pd is not None and pd.status == "ok" and not pd.positive_definite:
        try:
            minimum_point = np.asarray(pd.min_point, dtype=float).reshape(-1)
            reference = getattr(base, "GPU5_PD_REFERENCE_MATRIX", None)
            if reference is not None:
                P = np.asarray(reference, dtype=float)
                quad = float(minimum_point @ (0.5 * (P + P.T)) @ minimum_point)
            else:
                quad = float(minimum_point @ minimum_point)
            pd_eps = float(getattr(base, "GPU5_PD_EPS", 1.0e-4))
            setattr(
                result,
                "gpu5_7_pd_min_w",
                float(pd.min_margin) + pd_eps * quad,
            )
        except Exception:
            pass
    return result


def _compute_exact_result(expression, constants, base):
    """Compute one full exact result and preserve software failures."""
    kwargs = _exact_check_kwargs(base)
    try:
        result = check_b_manifold_exact_gpu_cached(
            _runtime_candidate(expression, constants),
            base.fSR,
            base.GSR,
            **kwargs,
        )
        if result.status != "ok":
            result = check_b_manifold_exact_gpu_cached(
                base._sympy_expression(expression, constants),
                base.fSR,
                base.GSR,
                **kwargs,
            )
        result = _attach_pd(result, expression, constants, base)
        result = _attach_normalized_margins(result, expression, constants, base)
        return _attach_rollout(result, expression, constants, base)
    except Exception as exc:
        concise = str(exc).splitlines()[0][:500]
        raise RuntimeError(
            f"full exact implementation error ({type(exc).__name__}): {concise}"
        ) from exc


def _shared_exact_result(expression, constants, base, cache_actor):
    key = _exact_cache_key(expression, constants, base)
    owner = uuid.uuid4().hex
    if cache_actor is not None:
        import ray

        disposition, cached = ray.get(cache_actor.reserve.remote(key, owner))
        if disposition == "hit":
            cached.gpu2_cache = "cluster_hit"
            cached.gpu2_fused_fallback = None
            return cached
        if disposition == "wait":
            while True:
                state, cached = ray.get(cache_actor.state.remote(key))
                if state == "hit":
                    cached.gpu2_cache = "cluster_wait"
                    cached.gpu2_fused_fallback = None
                    return cached
                if state == "missing":
                    disposition, cached = ray.get(
                        cache_actor.reserve.remote(key, owner)
                    )
                    if disposition == "owner":
                        break
                    if disposition == "hit":
                        cached.gpu2_cache = "cluster_hit"
                        cached.gpu2_fused_fallback = None
                        return cached
                time.sleep(0.02)

    # This is deliberately the original full-resolution exact checker.  The
    # cheap direct Artstein screen is a separate mandatory first stage.
    try:
        result = _compute_exact_result(expression, constants, base)
    except Exception:
        if cache_actor is not None:
            import ray

            ray.get(cache_actor.abandon.remote(key, owner))
        raise

    if result is not None:
        result.gpu2_cache = "computed"

    if cache_actor is not None:
        import ray

        if result is not None and result.status == "ok":
            ray.get(cache_actor.publish.remote(key, owner, result))
        else:
            ray.get(cache_actor.abandon.remote(key, owner))
    return result


def _compute_exact_results_batch(entries, base):
    """GPU4: one batched exact GPU call for the actor's whole candidate batch.

    Any candidate the batch path fails on (or a non-ok status) falls back to
    the GPU3 single-candidate check, so a batched result is never less
    accurate or less robust than GPU3.
    """
    kwargs = _exact_check_kwargs(base)
    candidates = [
        _runtime_candidate(entry["expression"], entry["consts"])
        for entry in entries
    ]
    try:
        results = check_b_manifold_exact_gpu4_batch(
            candidates, base.fSR, base.GSR, **kwargs
        )
    except Exception:
        results = [None] * len(entries)
    output = []
    for entry, result in zip(entries, results):
        if result is None or result.status != "ok":
            result = _compute_exact_result(
                entry["expression"], entry["consts"], base
            )
        if result is not None:
            setattr(result, "gpu2_cache", getattr(result, "gpu2_cache",
                                                  "computed"))
            if getattr(result, "gpu5_pd", None) is None:
                result = _attach_pd(result, entry["expression"],
                                    entry["consts"], base)
            result = _attach_normalized_margins(
                result, entry["expression"], entry["consts"], base
            )
            result = _attach_rollout(
                result, entry["expression"], entry["consts"], base
            )
        output.append(result)
    return output


def _shared_exact_results_many(entries, base, cache_actor):
    """GPU4 batches the actor's assigned candidates into one GPU call. The
    cross-worker cache dedup is skipped here (the scheduler already dedups
    within a generation); process-local memoization still applies."""
    return _compute_exact_results_batch(entries, base)




# --- GPU5_4: the normalised Artstein margin ---------------------------------
# a and b are both LINEAR in grad(V), so the Artstein condition is invariant
# under V -> kV while the penalty built on the raw margin is not. Wherever the
# search cannot get a < 0, the cheapest move is to shrink grad(V) there: a -> 0
# buys margin without fixing anything, and the same flattening is what the
# properness and V>0 terms charge for. That is why properness and a negative
# margin looked mutually exclusive -- both were being traded along one
# parameter, the local flatness of V.
#
# a / ||grad V|| removes the payoff exactly. ||grad V|| > 0, so the SIGN is
# unchanged and the validity decision (n_violations, the sign test inside the
# checkers) is untouched; only the shaping magnitude changes. See
# srcGPU5_7/grad_norm_gpu5_7.py.
_NORMALIZE_MARGIN = os.environ.get(
    "SYMCLF_GPU5_7_NORMALIZE_MARGIN", "1"
) == "1"


def _grad_norm_floor(base):
    """Relative floor under ||grad V||, as a fraction of the median over the
    normalised points. A root where V is genuinely flat has a = 0 by identity,
    not by control authority; dividing by a vanishing norm would report a
    meaningless ratio, so the floor keeps the quotient finite and, because it
    is relative, keeps the whole term scale-invariant."""
    return float(getattr(base, "GPU5_7_GRAD_NORM_FLOOR", 1.0e-3))


def _attach_normalized_margins(result, expression, constants, base):
    """Attach ``norm_margin_max`` / ``norm_margin_mean_pos`` to a check result.

    Both are computed from ``violation_points`` alone, which is exact for the
    penalty: a non-violating root contributes 0 to ``max(m, 0)`` and to
    ``mean(max(m, 0))``, so only the violating set can move either term. With
    no violations both are 0 and the penalty is 0 either way.

    ``a`` is recomputed at those points with the same dual-number interpreter
    and drift field the checker uses, because the batched GPU4 path reports
    only the aggregate ``margin_max``, not per-root margins.

    The polish maximises the RAW margin, so the root maximising the normalised
    ratio need not be among the polished ones. That is a shaping approximation,
    not a soundness gap: no violating point is dropped, the sign is exact, and
    certification never reads these fields.
    """
    if result is None or not _NORMALIZE_MARGIN:
        return result
    n_roots = int(getattr(result, "n_roots", 0) or 0)
    if n_roots <= 0:
        return result
    points = getattr(result, "violation_points", None)
    points = np.asarray(
        points if points is not None else np.empty((0, 4)), dtype=np.float64
    )
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 4:
        setattr(result, "norm_margin_max", 0.0)
        setattr(result, "norm_margin_mean_pos", 0.0)
        setattr(result, "gpu5_7_min_w_viol", float("inf"))
        return result

    v_values, a_values, norms = a_and_grad_norms(expression, constants, points)

    # GPU5_7: the certificate depth of the violating set. W = V - V(0) at
    # the worst-placed violation bounds the certifiable level c_max from
    # above, so the exact stage can price HOW MUCH of the sublevel structure
    # the violations destroy (see _cert_penalty). One extra 1-point kernel
    # call for V(0); the per-root values ride along with the margins.
    try:
        v_origin = float(
            a_and_grad_norms(expression, constants, np.zeros((1, 4)))[0][0]
        )
        w_values = v_values - v_origin
        finite_w = w_values[np.isfinite(w_values)]
        setattr(
            result,
            "gpu5_7_min_w_viol",
            float(finite_w.min()) if finite_w.size else float("inf"),
        )
    except Exception:
        pass  # attribute absent -> _cert_penalty charges nothing

    good = np.isfinite(a_values) & np.isfinite(norms) & (norms > 0.0)
    if not np.any(good):
        return result  # fall back to the raw margin

    margins = a_values[good] + float(getattr(base, "CLF_GAMMA1", 0.0))
    norms = norms[good]
    # Relative floor: a root where V is genuinely flat has a = 0 by identity,
    # not by control authority, so the quotient there is meaningless. Flooring
    # at a FRACTION of the median keeps the whole term scale-invariant, which
    # is the entire point of normalising.
    floor = _grad_norm_floor(base) * float(np.median(norms))
    denominator = np.maximum(norms, max(floor, 1.0e-300))
    with np.errstate(all="ignore"):
        normalized = margins / denominator
    normalized = normalized[np.isfinite(normalized)]
    if normalized.size == 0:
        return result
    positive = np.maximum(normalized, 0.0)
    setattr(result, "norm_margin_max", float(normalized.max()))
    # Same denominator as the raw mean: positive part summed over the violating
    # roots, divided by ALL non-origin roots.
    setattr(
        result,
        "norm_margin_mean_pos",
        float(positive.sum() / float(n_roots)),
    )
    return result


def _penalty_margin(result):
    """The margin the penalties are shaped on.

    ``norm_margin_max`` when the normalised value was attached, otherwise the
    raw ``margin_max`` (GPU5_3 behaviour). Both have the same sign, so swapping
    between them can never turn a rejected candidate into an accepted one.
    """
    value = getattr(result, "norm_margin_max", None)
    if value is None or not np.isfinite(value):
        return result.margin_max
    # A valid check (no violations) has nothing positive to normalise; keep the
    # raw negative margin so the near-zero sharpener still pushes candidates
    # past the knife edge instead of parking on it.
    if int(getattr(result, "n_violations", 0) or 0) <= 0:
        return result.margin_max
    return float(value)


def _violation_weight(base):
    """Charge for a fully-violating b=0 set. GPU5_3 default 300."""
    return float(getattr(base, "GPU5_7_VIOLATION_WEIGHT", 300.0))


def _violating_fraction(result):
    """Fraction of retained b=0 roots that violate the Artstein condition.

    GPU5/GPU5_2 charged the raw COUNT (n_violations * 0.01, then * 0.3). The
    count is dominated by how many roots the scan happened to retain, which is
    a property of the candidate's geometry rather than of how bad it is: on the
    same scan settings the run-122235 champion has 1948 roots and the run-122252
    champion 539778. The count therefore priced the second candidate's
    violations at 80457 and the first candidate's at 19.5 -- and since every way
    of dodging the exact check is bounded, an UNBOUNDED honest price is exactly
    what makes dodging profitable.

    The fraction is bounded in [0, 1], comparable across candidates, and still
    falls monotonically as the violating measure shrinks, which is the signal
    the count was added for.
    """
    roots = int(getattr(result, "n_roots", 0) or 0)
    if roots <= 0:
        return 0.0
    return min(1.0, float(result.n_violations) / float(roots))


def _mean_pos_margin_penalty(result, base):
    """GPU5_2: the same penalty shape as the margin_max term, fed the MEAN of
    the positive part of the margin over every b=0 root instead of the single
    worst one.

    Why this term exists. ``margin_max`` is a minimax objective read off one
    point out of ``n_roots`` (typically ~1e3). A candidate that repairs 97% of
    the violating manifold but leaves the worst point alone scores the same as
    one that repairs nothing, so almost every useful mutation is invisible to
    the GP and the search creeps. ``mean(max(margin, 0))`` falls in proportion
    to the violating measure, which pays the GP for the work it actually did.

    Shape is deliberately identical to the max terms -- same
    ``_manifold_margin_penalty`` quadratic/linear/saturating curve and the same
    concave ``4*sqrt + 64*^(1/4) + 256*^(1/6)`` finish-line terms -- with one
    necessary change of origin. The max terms are built around "valid" meaning
    ``margin_max <= -MANIFOLD_MARGIN_SAFETY``, so they add SAFETY/SHIFT before
    evaluating. The mean of a positive part is >= 0 by construction and its
    "valid" state is exactly 0, so the offsets are dropped: feeding a
    non-negative quantity through the shifted curves would leave a constant
    ~12-point floor on every candidate including perfect ones -- a pure offset
    with no gradient, the same defect as the old fail-closed PD constant.
    With no offset the term is continuous, vanishes exactly when no root
    violates, and keeps the infinite slope at 0 that stops margin-parking.

    Returns 0.0 when the checker did not report ``margin_mean_pos``: the cheap
    screen builds a BManifoldResult without the attribute, and the cheap path
    is unchanged in GPU5_2 by design. Attribute missing means "not applicable",
    never "unknown" -- do not fail closed here.
    """
    # GPU5_4: prefer the normalised mean when the checker/attacher provided it.
    m = getattr(result, "norm_margin_mean_pos", None)
    if m is None:
        m = getattr(result, "margin_mean_pos", None)
    if m is None or not np.isfinite(m):
        return 0.0
    m = max(float(m), 0.0)
    if m <= 0.0:
        return 0.0
    weight = float(
        getattr(
            base,
            "GPU5_7_MEAN_MARGIN_WEIGHT",
            base.MANIFOLD_EXACT_MARGIN_WEIGHT,
        )
    )
    cap = float(
        getattr(
            base,
            "GPU5_7_MEAN_MARGIN_PENALTY_MAX",
            base.MANIFOLD_EXACT_MARGIN_PENALTY_MAX,
        )
    )
    smooth = min(
        weight * m * m / (m + base.MANIFOLD_MARGIN_M0), cap
    )
    # GPU5_3: read the SAME curve as the margin_max term instead of a
    # hardcoded copy of it, so the coefficient cut applied in the example's
    # Evaluate shim rescales both consistently. Only reached for m > 0
    # (guarded above), so the SHIFT inside base._exact_near0_penalty cannot
    # put a constant floor on a candidate whose violating measure is zero.
    near0 = base._exact_near0_penalty(m)
    # SCALE multiplies the WHOLE term, not just `weight`. At the magnitudes
    # this term actually sees (mean_pos ~ 4e-4 on a near-miss) the concave
    # near0 part is 80.25 of 80.48 and the weighted smooth part is 0.23, so
    # raising `weight` alone moves the term by <1%: tripling it takes 80.48 to
    # 80.94, which is a no-op. SCALE is the knob that actually changes the
    # term's authority against the margin_max penalty.
    scale = float(getattr(base, "GPU5_7_MEAN_MARGIN_SCALE", 1.0))
    return float(scale * (smooth + near0))


# --- GPU5_7: certificate depth ----------------------------------------------
def _cert_penalty(result, base):
    """Certificate depth, priced through the SAME ROA ramp as the grid stage.

    GPU5_7 REVIEW FIX. PD failure and "the certificate is empty" are the SAME
    EVENT -- if W dips below zero anywhere, c_max <= 0 by definition. Until now
    they were priced by two different mechanisms in two different stages:

      * the grid ROA ramp sees only 21^4 points and misses a well that sits
        between samples -- on run 122350's champion it read c_ratio = -4.88e-04,
        essentially zero;
      * the exact PD check finds the real well (min_point (0.25, 0.25, ...))
        and charged it 28.32 through pd_result_penalty.

    So the ONE condition still blocking the search was worth 4.8% of a 595
    fitness, while the ROA ramp it should have been driving was worth 500. And
    the old form of this function stood down entirely whenever
    ``boundary_min <= 0`` -- exactly the case it most needed to price.

    Now the exact stage recomputes the certificate from the exact geometry

        c_max_exact = min(grid boundary_min,
                          min W over exact Artstein violation points,
                          W at the PD minimiser)
        c_ratio     = c_max_exact / w_scale          (w_scale from the grid)

    and charges the identical ramp the grid uses, so grid and exact speak one
    currency and optimise one objective. A PD failure now costs up to
    ROA_WEIGHT (500) instead of 28. Scale-invariant: every term in the ratio
    scales with V.

    The grid ROA term is subtracted out for candidates that reach here, so a
    defect the grid already saw is not charged twice; the exact stage's view is
    strictly the better one.
    """
    roa_weight = float(getattr(base, "GPU5_7_ROA_WEIGHT", 500.0))
    roa_neg = float(getattr(base, "GPU5_7_ROA_NEG_WEIGHT", 250.0))
    target = float(getattr(base, "GPU5_7_ROA_C_TARGET", 0.004))
    if roa_weight <= 0.0 and roa_neg <= 0.0:
        return 0.0

    w_scale = getattr(result, "gpu5_7_w_scale", None)
    if w_scale is None or not np.isfinite(w_scale) or w_scale <= 0.0:
        return 0.0                      # no anchor: leave the grid term alone

    candidates = []
    boundary_min = getattr(result, "gpu5_7_boundary_min", None)
    if boundary_min is not None and np.isfinite(boundary_min):
        candidates.append(float(boundary_min))
    for attribute in ("gpu5_7_min_w_viol", "gpu5_7_pd_min_w"):
        value = getattr(result, attribute, None)
        if value is not None and np.isfinite(value):
            candidates.append(float(value))
    if not candidates:
        return 0.0

    c_ratio = min(candidates) / w_scale
    exact_charge = roa_weight * float(
        np.clip(1.0 - c_ratio / target, 0.0, 1.0)
    ) + roa_neg * float(np.log1p(max(-c_ratio, 0.0)))

    # Refund what the grid ROA term already charged for the same certificate,
    # so one defect is priced once. Never returns a credit.
    grid_ratio = (
        float(boundary_min) / w_scale
        if boundary_min is not None and np.isfinite(boundary_min)
        else c_ratio
    )
    grid_charge = roa_weight * float(
        np.clip(1.0 - grid_ratio / target, 0.0, 1.0)
    ) + roa_neg * float(np.log1p(max(-grid_ratio, 0.0)))
    return float(max(exact_charge - grid_charge, 0.0))



# --- GPU5_7: the closed-loop rollout gate ------------------------------------
def _cartpole_fields_np(state):
    """Drift f and input column g of the 4-D cart-pole, one state (numpy)."""
    x2, x3, x4 = state[1], state[2], state[3]
    sin_theta = -np.sin(x3)
    cos_theta = -np.cos(x3)
    denominator = 4.0 * (5.0 + 1.0 * (1.0 - cos_theta ** 2))
    drift = np.array([
        x2,
        (-4.0 * -10.0 * cos_theta * sin_theta
         + 4.0 * (2.0 * x4 ** 2 * sin_theta - x2)) / denominator,
        x4,
        (6.0 * -10.0 * 2.0 * sin_theta
         - 2.0 * cos_theta * (2.0 * x4 ** 2 * sin_theta - x2)) / denominator,
    ])
    gain = np.array([0.0, 4.0 / denominator, 0.0, -2.0 * cos_theta / denominator])
    return drift, gain


def _attach_rollout(result, expression, constants, base):
    """Integrate the closed loop with the SATURATED Sontag controller.

    Run 122340's generation-7 champion passed PD, passed the exact Artstein
    check inside its certified set, had c_max > 0 -- and diverges from inside
    that very set, because following its V requires |u| ~ 1e11. No static
    check replaces actually closing the loop, so a candidate that passes
    everything else gets integrated before it is allowed a near-zero penalty.

    Eligibility is deliberately strict (zero violations, PD holds), so this
    costs a few CPU-seconds only on the rare candidates that reach it --
    approximately zero per generation until the search is winning. Divergence
    (||x|| beyond GPU5_7_ROLLOUT_DIVERGE_NORM, or non-finite) is what is
    charged; failure to fully converge is NOT, because standard initial
    conditions can legitimately sit outside a small certified region.
    """
    if result is None or result.status != "ok":
        return result
    if not bool(getattr(base, "GPU5_7_ROLLOUT_ENABLED", True)):
        return result
    pd = getattr(result, "gpu5_pd", None)
    eligible = (
        int(getattr(result, "n_violations", 1) or 0) == 0
        and int(getattr(result, "n_roots", 0) or 0) > 0
        and pd is not None
        and pd.status == "ok"
        and bool(pd.positive_definite)
    )
    if not eligible:
        return result

    try:
        opcodes, operands, literals, n_ops, parameters = _pd_program(
            expression, constants, 0
        )
        op64 = opcodes.astype(np.int64)
        opr64 = operands.astype(np.int64)

        def gradient_of_v(state):
            return np.asarray(
                _cpu_dual2(op64, opr64, literals, n_ops, parameters, state)[1],
                dtype=float,
            )

        u_max = float(
            getattr(base, "GPU5_7_ROLLOUT_UMAX",
                    getattr(base, "GPU5_7_SAT_U_TARGET", 1000.0))
        )
        horizon = float(getattr(base, "GPU5_7_ROLLOUT_T", 10.0))
        dt = float(getattr(base, "GPU5_7_ROLLOUT_DT", 2.0e-3))
        diverge_norm = float(
            getattr(base, "GPU5_7_ROLLOUT_DIVERGE_NORM", 1.0)
        )
        starts = np.asarray(
            getattr(base, "ROLLOUT_X0S"), dtype=float
        ).reshape(-1, 4)

        def closed_loop(state):
            drift, gain = _cartpole_fields_np(state)
            gradient = gradient_of_v(state)
            a_value = float(gradient @ drift)
            b_value = float(gradient @ gain)
            b_scaled = 1.0e4 * b_value * b_value
            radius_sq = float(state @ state)
            lam = (
                np.sqrt(a_value * a_value + radius_sq * b_scaled) + a_value
            ) / (b_scaled + 1.0e-6)
            control = float(np.clip(-1.0e4 * b_value * lam, -u_max, u_max))
            return drift + gain * control

        steps = int(horizon / dt)
        diverged = 0
        final_norms = []
        for start in starts:
            state = np.asarray(start, dtype=float).copy()
            bad = False
            for _ in range(steps):
                k1 = closed_loop(state)
                k2 = closed_loop(state + 0.5 * dt * k1)
                k3 = closed_loop(state + 0.5 * dt * k2)
                k4 = closed_loop(state + dt * k3)
                state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                if (
                    not np.all(np.isfinite(state))
                    or float(np.linalg.norm(state)) > diverge_norm
                ):
                    bad = True
                    break
            diverged += int(bad)
            final_norms.append(
                float(np.linalg.norm(state)) if np.all(np.isfinite(state))
                else float("inf")
            )
        setattr(result, "gpu5_7_rollout", {
            "diverged": int(diverged),
            "n_starts": int(len(starts)),
            "final_norms": final_norms,
            "u_max": u_max,
        })
    except Exception as exc:
        # Fail closed: an un-integrable candidate must not look certified.
        setattr(result, "gpu5_7_rollout", {
            "diverged": 1,
            "n_starts": 0,
            "error": f"{type(exc).__name__}: {exc}",
        })
    return result


def _exact_result_penalty(result, base):
    """CPU-Evaluate-equivalent margin penalty for a valid check result."""
    if result is None or result.status != "ok":
        return float(base.MANIFOLD_EXACT_PENALTY_MAX)

    # GPU5_4: every margin the PENALTY sees is the normalised one when it is
    # available. n_violations, n_roots and the sign test are untouched -- the
    # validity decision is identical to GPU5_3.
    margin_for_penalty = _penalty_margin(result)

    penalty = 0.0
    if result.n_violations > 0:
        penalty = min(
            base.MANIFOLD_EXACT_PENALTY_MAX,
            10.0
            # GPU5_3: violating FRACTION x weight, not raw count -- see
            # _violating_fraction. Bounded, and comparable between a candidate
            # whose b=0 set is a 1948-root manifold and one whose b=0 set is a
            # 539778-point blob.
            + _violating_fraction(result) * _violation_weight(base)
            + base._manifold_margin_penalty(
                margin_for_penalty,
                base.MANIFOLD_EXACT_MARGIN_WEIGHT,
                base.MANIFOLD_EXACT_MARGIN_PENALTY_MAX,
            ),
        )
    elif result.n_roots == 0:
        penalty = base.MANIFOLD_VACUOUS_PENALTY

    if result.n_roots > 0:
        penalty += base._exact_near0_penalty(margin_for_penalty)
        # GPU5_2: added measure term. Exact path only -- the cheap screen has
        # no margin_mean_pos, so this is a no-op there.
        penalty += _mean_pos_margin_penalty(result, base)

    # Positive definiteness is gated exactly like the Artstein margin: a
    # failure adds its own capped penalty, so the GP is pushed away from
    # sign-changing V instead of being rewarded for a good b=0 margin on a
    # function that is not a Lyapunov candidate.
    # _exact_result_penalty is called for BOTH the cheap screen and the full
    # exact result. Only the exact path runs _attach_pd, so a cheap result has
    # no gpu5_pd ATTRIBUTE AT ALL -- that is "not applicable", not "unknown",
    # and charging it the fail-closed penalty put a constant +8000 on every
    # single candidate. On run 122231 that was 8000 of an 11504 cheap penalty,
    # i.e. 66% of the whole fitness, identical for every individual and
    # therefore pure offset with no gradient. Distinguish the two cases:
    #   attribute missing -> cheap result, PD not applicable
    #   attribute is None -> exact result whose PD check raised -> fail closed
    if hasattr(result, "gpu5_pd"):
        pd = getattr(result, "gpu5_pd")
        if pd is None:
            penalty += float(
                getattr(base, "GPU5_PD_UNKNOWN_PENALTY", 8000.0)
            )
        else:
            # GPU5_7: base is passed so the depth is scale-invariant and the
            # near-zero curve is the SHARED override, not a 4.37x inline copy.
            penalty += pd_result_penalty(pd, base)

    # GPU5_7: violations are also priced by their sublevel placement -- a
    # violation at W ~ 0 empties the certificate. Cheap results get the same
    # treatment (their violating points pass through the same attacher), so
    # both sides of the a_max gate price placement consistently.
    penalty += _cert_penalty(result, base)

    # GPU5_7: the closed-loop gate. Only ever attached to a candidate that
    # passed every static check; divergence under the saturated Sontag
    # controller is what run 122340's gen-7 "certified" champion actually
    # does, so it must never score as solved.
    rollout = getattr(result, "gpu5_7_rollout", None)
    if rollout is not None and int(rollout.get("diverged", 0)) > 0:
        penalty += float(
            getattr(base, "GPU5_7_ROLLOUT_FAIL_PENALTY", 500.0)
        ) * float(rollout["diverged"])

    # Internal consistency: margin_max is the maximum of the same margin
    # array used to count violations. A positive maximum must be penalized.
    if (
        np.isfinite(result.margin_max)
        and result.margin_max > base.MANIFOLD_MARGIN_TOL_EXACT
        and result.n_violations <= 0
    ):
        return float(base.MANIFOLD_EXACT_PENALTY_MAX)
    return float(penalty)


class _MeanProxy:
    """Minimal stand-in so the gate price can reuse the real mean-margin
    penalty instead of duplicating its curve."""

    __slots__ = ("margin_mean_pos",)

    def __init__(self, margin_mean_pos):
        self.margin_mean_pos = margin_mean_pos


def _gate2_floor(base):
    """Flat price for a candidate that never reached the cheap screen.

    This path -- grid MSE failed gate 1 -- has no result to price from, so the
    price must be a constant. That is safe here because failing gate 1 is not
    an escape route: it already costs ``mse >= MANIFOLD_EXACT_MSE_GATE`` plus
    ``1.3 * MANIFOLD_EXACT_MSE_GATE`` on top, i.e. >= 23000 before this term.

    The reference is what the exact check would charge a candidate sitting
    exactly on the a_max gate, quoted at ESCALATION x the gate because the
    cheap screen searches ~412 roots where the exact check searches ~2141 and
    therefore understates the margin (measured 1.03x-1.68x across six
    champions of run 122232, mean 1.43x).
    """
    threshold = float(getattr(base, "GPU2_EXACT_A_MAX_GATE", 0.05))
    gamma1 = float(getattr(base, "CLF_GAMMA1", 0.0))
    ratio = float(getattr(base, "GPU2_EXACT_GATE_PENALTY_RATIO", 1.1))
    escalation = float(getattr(base, "GPU5_7_GATE_MARGIN_ESCALATION", 2.0))
    mean_fraction = float(getattr(base, "GPU5_7_GATE_MEAN_FRACTION", 0.02))

    m = escalation * threshold + gamma1
    reference = (
        10.0
        + float(getattr(base, "GPU5_7_GATE_VIOLATING_FRACTION", 0.05))
        * _violation_weight(base)
        + base._manifold_margin_penalty(
            m,
            base.MANIFOLD_EXACT_MARGIN_WEIGHT,
            base.MANIFOLD_EXACT_MARGIN_PENALTY_MAX,
        )
        + base._exact_near0_penalty(m)
        + _mean_pos_margin_penalty(_MeanProxy(mean_fraction * m), base)
    )
    return float(ratio * reference)


def _second_gate_penalty(base, result=None):
    """Price charged to a candidate the cheap screen refused to promote.

    GPU5_2 made this a FLAT FEE, on the argument that a candidate-dependent
    price lets the population escape by getting worse on the grid so it is
    never screened (run 122234). The flat fee has a worse failure mode, and it
    is the one that actually happened: a flat fee is a CAP on an unbounded
    penalty, so refusing the check becomes the cheap way out for exactly the
    candidates that most deserve it.

    Measured on the run-122252 champion. Its true exact result is
    ``roots=539778, violations=268189, margin_max=10.03``, which
    ``_exact_result_penalty`` prices at 113958. Its RECORDED fitness was 4139,
    because the cheap screen rejected it at a_max > 0.05 and it paid the fee
    instead -- 63x cheaper than being looked at. Run 122235 shows the search
    finding that door and standing in it: the population median cheap a_max sat
    at 0.0514-0.0522 against a 0.05 gate for 2000 consecutive generations,
    100% of generations from gen ~1350 on.

    GPU5_3 prices rejection off the candidate's OWN cheap result, escalated:

        price = clip(ESCALATION * cheap_penalty, floor, cap)

    * MONOTONE in a_max, so there is no ledge at the gate to stand on.
    * ESCALATION >= 1 (default 2.0) covers the cheap screen's understatement,
      so refusing the check is never cheaper than taking it.
    * The floor is the flat GPU5_2 price, so a candidate just inside the gate
      is never charged less than one just outside it.
    * The cap closes run 122234's escape: it is the price of failing gate 1
      (1.3 * MANIFOLD_EXACT_MSE_GATE), so deliberately wrecking the grid score
      to avoid the screen can never be strictly cheaper than being screened.
      Only candidates with cheap_penalty >= cap/ESCALATION (a_max ~0.6, i.e.
      already hopeless) reach it; the whole frontier region stays monotone.

    ``result`` is the cheap ``BManifoldResult``. Called without one -- the
    gate-1 path -- it returns the flat floor, which is correct there.
    """
    floor = _gate2_floor(base)
    if result is None:
        return floor
    escalation = float(getattr(base, "GPU5_7_GATE_PRICE_ESCALATION", 2.0))
    cap = getattr(base, "GPU5_7_GATE_PRICE_CAP", None)
    if cap is None:
        # The other way out of the exact check is to fail gate 1 on purpose.
        # That costs raising grid MSE to MANIFOLD_EXACT_MSE_GATE, plus the
        # 1.3 x gate surcharge, plus this floor. Pricing the cap at that total
        # makes the two exits cost the same, so neither is a strictly cheaper
        # dodge -- which is what run 122234 exploited when the price was
        # a_max-shaped with no cap at all.
        cap = 2.3 * float(base.MANIFOLD_EXACT_MSE_GATE) + floor
    cap = float(cap)
    priced = escalation * float(_exact_result_penalty(result, base))
    if not np.isfinite(priced):
        return floor
    return float(min(max(priced, floor), cap))


def _apply_exact(pre_exact, expression, constants, base, cache_actor):
    mse = float(pre_exact)
    if not np.isfinite(mse) or mse >= 1.0e10:
        return 1.0e10
    if not base.MANIFOLD_EXACT_CHECK_ENABLED:
        return mse
    if mse >= base.MANIFOLD_EXACT_MSE_GATE:
        return (
            mse
            + float(base.MANIFOLD_EXACT_MSE_GATE) * 1.3
            + _second_gate_penalty(base)
        )

    result = _shared_exact_result(expression, constants, base, cache_actor)
    return mse + _exact_result_penalty(result, base)


def _evaluate_pre_exact_valid_batch(
    expressions,
    n_constants,
    true_data,
    base,
    fusion_width,
):
    scores = [None] * len(expressions)
    constants = [None] * len(expressions)

    plain = [index for index, count in enumerate(n_constants) if count == 0]
    if plain:
        plain_values = _score_rows(
            [expressions[index] for index in plain],
            [np.empty(0)] * len(plain),
            true_data,
            base,
        )
        for index, value in zip(plain, plain_values):
            constants[index] = np.empty(0)
            scores[index] = float(value)

    tuned = [index for index, count in enumerate(n_constants) if count > 0]
    for start in range(0, len(tuned), fusion_width):
        group = tuned[start:start + fusion_width]
        group_scores, group_constants = _tune_group(
            [expressions[index] for index in group],
            [n_constants[index] for index in group],
            true_data,
            base,
            fusion_width,
        )
        for index, value, fitted in zip(group, group_scores, group_constants):
            scores[index] = value
            constants[index] = fitted

    return scores, constants


def fitness_pre_exact(
    individuals,
    *,
    true_data,
    penalty,
    base,
    tuner_fusion=16,
):
    """Run validation/grid/tuning and apply the first MSE gate only."""
    individual_length, nested_trigs, _ = get_features_batch(individuals)
    expressions = [str(individual) for individual in individuals]
    intermediate = [None] * len(individuals)

    valid_indices = []
    valid_expressions = []
    valid_constants = []
    for index, expression in enumerate(expressions):
        if individual_length[index] >= 400:
            intermediate[index] = {
                "consts": None,
                "fitness": (1.0e8,),
                "needs_cheap": False,
            }
            continue
        try:
            program = encode_expression(expression)
        except Exception:
            intermediate[index] = {
                "consts": None,
                "fitness": (1.0e8,),
                "needs_cheap": False,
            }
            continue
        valid_indices.append(index)
        valid_expressions.append(expression)
        valid_constants.append(program.n_constants)

    if valid_indices:
        mse, constants = _evaluate_pre_exact_valid_batch(
            valid_expressions,
            valid_constants,
            true_data,
            base,
            max(1, int(tuner_fusion)),
        )
        for local, index in enumerate(valid_indices):
            nested_exp = detect_nested_function_calls(expressions[index], "exp")
            nested_aq = detect_nested_function_calls(expressions[index], "aq")
            # GPU5_3: 100000 -> 50 per nested function class.
            #
            # detect_nested_function_calls returns a 0/1 FLAG per function, so
            # this charged 100000 for the mere presence of exp() inside exp(),
            # aq() inside aq(), or a trig inside a trig -- lethal, and lethal
            # far more often than intended. Sampling random trees from this
            # run's primitive set: 57% of ~18-node trees carry one, 98% of
            # ~51-node trees, 100% at ~66 nodes, against champions averaging
            # 126-252 nodes. In 15199 champion generations across all nine
            # GPU5* runs, not ONE champion ever contained a nested exp, aq or
            # sin -- the rule was a perfect absorbing filter, confining the
            # population to a vanishingly thin syntactic subspace and burning
            # most of the offspring stream on pre-doomed individuals.
            #
            # 50 keeps the pressure against exp(exp(...)) blow-ups (it is 10x
            # the length regulariser on a 1000-node tree) without making the
            # structure lethal. Set SYMCLF_GPU5_7_NESTED_PENALTY to restore the
            # old behaviour; the other lever is dropping `aq` from the primitive
            # set in config.yaml, since exp already supplies the nonlinearity.
            extra_penalty = float(
                _NESTED_CALL_PENALTY
                * (nested_trigs[index] + nested_exp + nested_aq)
                + penalty["reg_param"] * individual_length[index]
            )
            pre_exact = float(mse[local])
            needs_cheap = bool(
                base.MANIFOLD_EXACT_CHECK_ENABLED
                and np.isfinite(pre_exact)
                and pre_exact < base.MANIFOLD_EXACT_MSE_GATE
            )
            if needs_cheap:
                fitness_value = None
            else:
                fitness_value = pre_exact
                if (
                    base.MANIFOLD_EXACT_CHECK_ENABLED
                    and np.isfinite(pre_exact)
                    and pre_exact < 1.0e10
                ):
                    # Missing gate 1 carries both the established 1.3*gate
                    # price and the second-gate price, exactly as requested.
                    fitness_value += (
                        float(base.MANIFOLD_EXACT_MSE_GATE) * 1.3
                        + _second_gate_penalty(base)
                    )
                fitness_value = float(fitness_value + extra_penalty)
            intermediate[index] = {
                "consts": constants[local],
                "fitness": None if fitness_value is None else (fitness_value,),
                "needs_cheap": needs_cheap,
                "pre_exact": pre_exact,
                "expression": expressions[index],
                "exact_key": (
                    _exact_cache_key(
                        expressions[index], constants[local], base
                    )
                    if needs_cheap
                    else None
                ),
                "extra_penalty": extra_penalty,
                "gate2_penalty": _second_gate_penalty(base),
                "exact_a_gate": float(
                    getattr(base, "GPU2_EXACT_A_MAX_GATE", 0.05)
                ),
                "cheap_budget_seconds": float(
                    getattr(base, "GPU2_CHEAP_BUDGET_SECONDS", 15.0)
                ),
            }

    # GPU5_7: certified-region grid details for every gate-passing candidate.
    # boundary_min anchors the exact stage's certificate-depth price
    # (_cert_penalty); c_max / volume / u_required feed the per-generation
    # logs. One extra batched GPU call over the ~10-20% of the population
    # that passed gate 1 -- measured cost well under a second.
    gate_passing = [
        index for index, entry in enumerate(intermediate)
        if entry is not None and entry.get("needs_cheap", False)
    ]
    if gate_passing:
        try:
            _, grid_details = gpu_pre_exact_mse_many(
                [intermediate[index]["expression"] for index in gate_passing],
                [intermediate[index]["consts"] for index in gate_passing],
                true_data,
                base,
            )
            for local, index in enumerate(gate_passing):
                intermediate[index]["grid_boundary_min"] = float(
                    grid_details["boundary_min"][local]
                )
                intermediate[index]["grid_c_max"] = float(
                    grid_details["c_max"][local]
                )
                intermediate[index]["grid_certified_volume"] = float(
                    grid_details["certified_volume"][local]
                )
                intermediate[index]["grid_u_required"] = float(
                    grid_details["u_required"][local]
                )
                intermediate[index]["grid_w_scale"] = float(
                    grid_details["w_scale"][local]
                )
        except Exception:
            pass  # _cert_penalty stands down without a boundary anchor
    return intermediate


def fitness_finish_cheap(entries, *, base):
    """Run the mandatory cheap direct b=0 screen for all gate-1 passes."""
    cheap_indices = [
        index for index, entry in enumerate(entries)
        if entry.get("needs_cheap", False)
    ]
    if not cheap_indices:
        return entries

    candidates = [
        _runtime_candidate(entries[index]["expression"], entries[index]["consts"])
        for index in cheap_indices
    ]
    checked = check_artstein_gpu_cheap_many_from_base(candidates, base)
    if len(checked) != len(cheap_indices):
        raise RuntimeError("GPU2 cheap checker returned the wrong result count")

    output = list(entries)
    a_gate = float(getattr(base, "GPU2_EXACT_A_MAX_GATE", 0.05))
    for index, result in zip(cheap_indices, checked):
        if result is None or result.status != "ok":
            status = "missing" if result is None else str(result.status)
            raise RuntimeError(f"GPU2 cheap Artstein checker failed: {status}")
        entry = dict(entries[index])
        result = _attach_normalized_margins(
            result, entries[index]["expression"], entries[index]["consts"], base
        )
        # GPU5_7: anchor for the certificate-depth price, attached BEFORE
        # pricing so the cheap penalty and the gate2 re-price both see it.
        setattr(
            result,
            "gpu5_7_boundary_min",
            entries[index].get("grid_boundary_min", None),
        )
        setattr(
            result,
            "gpu5_7_w_scale",
            entries[index].get("grid_w_scale", None),
        )
        cheap_penalty = _exact_result_penalty(result, base)
        a_max = float(
            getattr(
                result,
                "gpu2_a_max",
                float(result.margin_max) - float(base.CLF_GAMMA1),
            )
        )
        # A vacuous cheap search (no roots / NaN a_max) must go to exact;
        # only a finite result above the threshold is cheap-rejected.
        needs_full_exact = bool(
            result.n_roots == 0 or not np.isfinite(a_max) or a_max <= a_gate
        )
        entry.update(
            {
                "needs_cheap": False,
                "needs_full_exact": needs_full_exact,
                "cheap_penalty": float(cheap_penalty),
                # GPU5_3: re-price the second gate from this candidate's own
                # cheap result. fitness_pre_exact could only install the flat
                # floor because it had no result yet. Doing it here makes the
                # rejection price monotone in a_max, and it also fixes the
                # deferred-exact path in ray_fitness, which charges this same
                # field when a gate-passing candidate misses its wave.
                "gate2_penalty": float(
                    _second_gate_penalty(base, result)
                ),
                "cheap_a_max": a_max,
                "cheap_diagnostic": {
                    "status": result.status,
                    "roots": int(result.n_roots),
                    "violations": int(result.n_violations),
                    "a_max": a_max,
                    "margin_max": float(result.margin_max),
                    "penalty": float(cheap_penalty),
                    "metrics": dict(getattr(result, "gpu2_metrics", {})),
                },
            }
        )
        if needs_full_exact:
            entry["fitness"] = None
        else:
            # GPU5_3: gate2_penalty is now ESCALATION * cheap_penalty, so it
            # already CONTAINS the cheap charge; adding cheap_penalty on top of
            # it would price the same screen three times. Keeping the two sides
            # of the gate directly comparable is the whole point of the change:
            #     refused : pre_exact + ESCALATION * cheap_penalty
            #     checked : pre_exact + max(cheap_penalty, exact_penalty)
            # With ESCALATION 2.0 against the measured 1.03x-1.68x screen
            # understatement, crossing the gate is always the cheaper move and
            # both sides are monotone in the margin.
            entry["fitness"] = (
                float(
                    entry["pre_exact"]
                    + entry["gate2_penalty"]
                    + entry["extra_penalty"]
                ),
            )
        output[index] = entry
    return output


def fitness_finish_exact(entries, *, base, exact_cache=None):
    """Finish the selected current-generation entries with full exact GPU checks."""
    exact_indices = [
        index for index, entry in enumerate(entries)
        if entry.get("needs_full_exact", entry.get("needs_exact", False))
    ]
    exact_results = _shared_exact_results_many(
        [entries[index] for index in exact_indices], base, exact_cache
    ) if exact_indices else []
    result_by_index = dict(zip(exact_indices, exact_results))
    attributes = []
    for index, entry in enumerate(entries):
        if entry.get("needs_full_exact", entry.get("needs_exact", False)):
            result = result_by_index[index]
            if result is None or result.status != "ok":
                status = "missing" if result is None else str(result.status)
                raise RuntimeError(f"GPU2 full exact checker failed: {status}")
            setattr(
                result,
                "gpu5_7_boundary_min",
                entry.get("grid_boundary_min", None),
            )
            setattr(
                result,
                "gpu5_7_w_scale",
                entry.get("grid_w_scale", None),
            )
            exact_penalty = _exact_result_penalty(result, base)
            # GPU5_3: max(), not sum(). cheap_penalty and exact_penalty are
            # both _exact_result_penalty of the SAME physical condition -- one
            # read off a ~412-root screen, the other off a ~2141-root search --
            # so adding them charges one violation twice. On the run-122235
            # champion that was cheap 281 + exact 361, i.e. 44% of its whole
            # 647.64 fitness was a duplicate. max() keeps the conservative
            # reading (the exact search normally dominates) without paying for
            # the same defect twice.
            value = (
                entry["pre_exact"]
                + max(float(entry.get("cheap_penalty", 0.0)), exact_penalty)
                + entry["extra_penalty"]
            )
            diagnostic = {
                "status": "missing" if result is None else result.status,
                "roots": 0 if result is None else int(result.n_roots),
                "violations": (
                    0 if result is None else int(result.n_violations)
                ),
                "margin_max": (
                    np.nan if result is None else float(result.margin_max)
                ),
                "penalty": float(exact_penalty),
                "cache": (
                    "missing"
                    if result is None
                    else getattr(result, "gpu2_cache", "unknown")
                ),
                "metrics": (
                    {}
                    if result is None
                    else dict(getattr(result, "gpu2_metrics", {}))
                ),
                "fused_fallback": (
                    None
                    if result is None
                    else getattr(result, "gpu2_fused_fallback", None)
                ),
                # GPU5_7: the certified-region ledger for this candidate.
                "grid_boundary_min": entry.get("grid_boundary_min"),
                "grid_c_max": entry.get("grid_c_max"),
                "grid_certified_volume": entry.get("grid_certified_volume"),
                "grid_u_required": entry.get("grid_u_required"),
                "min_w_viol": (
                    None if result is None
                    else getattr(result, "gpu5_7_min_w_viol", None)
                ),
                "cert_penalty": (
                    None if result is None else _cert_penalty(result, base)
                ),
                "rollout": (
                    None if result is None
                    else getattr(result, "gpu5_7_rollout", None)
                ),
            }
            attributes.append(
                {
                    "consts": entry["consts"],
                    "fitness": (float(value),),
                    "_exact": diagnostic,
                }
            )
        else:
            attributes.append(
                {"consts": entry["consts"], "fitness": entry["fitness"]}
            )
    return attributes


def fitness(
    individuals,
    *,
    true_data,
    penalty,
    base,
    exact_cache=None,
    tuner_fusion=16,
):
    """Compatibility wrapper that executes both GPU2 fitness stages."""
    entries = fitness_pre_exact(
        individuals,
        true_data=true_data,
        penalty=penalty,
        base=base,
        tuner_fusion=tuner_fusion,
    )
    entries = fitness_finish_cheap(entries, base=base)
    return fitness_finish_exact(entries, base=base, exact_cache=exact_cache)
