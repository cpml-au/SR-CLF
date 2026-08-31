"""GPU5 persistent one-GPU Ray actors and cross-worker exact-result cache.

Copied from ``srcGPU2.ray_fitness`` with two changes: actors run the GPU3
fitness modules (reference-SLSQP exact polish), and the per-generation exact
wave budget is multiplied by ``SYMCLF_GPU3_EXACT_WAVE_MULT`` (default 6)
because one GPU5 exact check costs about one second instead of seven.
"""

from __future__ import annotations

from collections import OrderedDict
import os
import time

import ray


# Driver-owned completed-result cache. There is deliberately no cross-generation
# work queue: every generation spends its exact budget only on current candidates.
_FULL_EXACT_RESULTS = OrderedDict()
_CHEAP_BUDGET_WARNINGS = 0
_FULL_EXACT_RESULT_LIMIT = 32768


_EXACT_WAVE_MULT = max(
    1, int(os.environ.get("SYMCLF_GPU3_EXACT_WAVE_MULT", "6"))
)

# Per-generation exact budget (candidates verified, across all GPUs). The
# scan/sympy de-contention fix in b_manifold_check_gpu3 cut the per-check cost
# ~4x on the A100 (the host-side scan no longer contends with the sympy-build
# thread), so the same wall-clock exact stage now affords ~4x the GPU2-era
# capacity. missing_current is priority-sorted (smallest cheap a_max first =
# most-likely-valid first), so covering the best `budget` candidates verifies
# exactly the ones worth verifying; any deferred remainder is the borderline-
# worst tail, which the flat provisional penalty already punishes correctly.
# Set 0 to fall back to the legacy cheap-count wave tiers (`_EXACT_WAVE_MULT`).
_EXACT_MAX_PER_GEN = max(
    0, int(os.environ.get("SYMCLF_GPU3_EXACT_MAX_PER_GEN", "288"))
)


def _exact_wave_count(cheap_candidates, *, first_limit=500, second_limit=1500):
    """Legacy cheap-count wave tiers (used only when EXACT_MAX_PER_GEN=0)."""
    count = int(cheap_candidates)
    if count <= 0:
        return 0
    if count <= int(first_limit):
        return 1 * _EXACT_WAVE_MULT
    if count <= int(second_limit):
        return 2 * _EXACT_WAVE_MULT
    return 3 * _EXACT_WAVE_MULT


def _exact_priority(entry):
    """Vacuous cheap checks first, then the smallest finite cheap a_max."""
    value = float(entry.get("cheap_a_max", float("nan")))
    if value != value:
        return (0, 0.0)
    return (1, value)


def _percentile(values, fraction):
    """Small dependency-free percentile helper for stage diagnostics."""
    if not values:
        return float("nan")
    ordered = sorted(float(value) for value in values)
    index = int(round((len(ordered) - 1) * float(fraction)))
    return ordered[index]


def _metric_mean(diagnostics, name):
    values = [
        float(item.get("metrics", {}).get(name, float("nan")))
        for item in diagnostics
    ]
    values = [value for value in values if value == value]
    return sum(values) / len(values) if values else float("nan")


def _metric_max(diagnostics, name):
    values = [
        float(item.get("metrics", {}).get(name, float("nan")))
        for item in diagnostics
    ]
    values = [value for value in values if value == value]
    return max(values, default=float("nan"))


@ray.remote(num_cpus=0)
class ExactResultCache:
    """Cluster-wide memoization with reservations to suppress duplicate work."""

    def __init__(self, max_entries=16384):
        self.max_entries = int(max_entries)
        self.results = OrderedDict()
        self.inflight = {}
        self.hits = 0
        self.misses = 0
        self.waits = 0

    def reserve(self, key, owner):
        if key in self.results:
            self.results.move_to_end(key)
            self.hits += 1
            return "hit", self.results[key]
        if key in self.inflight:
            self.waits += 1
            return "wait", None
        self.inflight[key] = owner
        self.misses += 1
        return "owner", None

    def state(self, key):
        if key in self.results:
            self.results.move_to_end(key)
            return "hit", self.results[key]
        return ("pending", None) if key in self.inflight else ("missing", None)

    def publish(self, key, owner, result):
        if self.inflight.get(key) == owner:
            self.inflight.pop(key, None)
            self.results[key] = result
            self.results.move_to_end(key)
            while len(self.results) > self.max_entries:
                self.results.popitem(last=False)
        return True

    def abandon(self, key, owner):
        if self.inflight.get(key) == owner:
            self.inflight.pop(key, None)
        return True

    def stats(self):
        return {
            "entries": len(self.results),
            "inflight": len(self.inflight),
            "hits": self.hits,
            "misses": self.misses,
            "waits": self.waits,
        }


@ray.remote(num_cpus=1, num_gpus=1, max_restarts=0)
class GPUFitnessActor:
    """Long-lived worker owning exactly one CUDA device and its JAX caches."""

    def __init__(self, true_data, penalty, exact_cache, tuner_fusion=16):
        # Imports happen after Ray has assigned CUDA_VISIBLE_DEVICES.
        import Evaluate
        from srcGPU5_7 import require_gpu
        from srcGPU5_7.grid_fitness import _context

        self.true_data = true_data
        self.penalty = penalty
        self.base = Evaluate._base
        self.exact_cache = exact_cache
        self.tuner_fusion = int(tuner_fusion)
        devices = require_gpu()
        _context(self.true_data, self.base)
        self.device = ", ".join(str(device) for device in devices)

    def ready(self):
        return self.device

    def evaluate(self, individuals):
        from srcGPU5_7.fitness import fitness

        return fitness(
            individuals,
            true_data=self.true_data,
            penalty=self.penalty,
            base=self.base,
            exact_cache=self.exact_cache,
            tuner_fusion=self.tuner_fusion,
        )

    def evaluate_pre_exact(self, individuals):
        from srcGPU5_7.fitness import fitness_pre_exact

        return fitness_pre_exact(
            individuals,
            true_data=self.true_data,
            penalty=self.penalty,
            base=self.base,
            tuner_fusion=self.tuner_fusion,
        )

    def evaluate_exact(self, entries):
        from srcGPU5_7.fitness import fitness_finish_exact

        return fitness_finish_exact(
            entries,
            base=self.base,
            exact_cache=self.exact_cache,
        )

    def evaluate_cheap(self, entries):
        from srcGPU5_7.fitness import fitness_finish_cheap

        return fitness_finish_cheap(entries, base=self.base)

    def warm_artstein(self):
        if not getattr(self.base, "GPU2_CHEAP_ENABLED", True):
            return {"status": "disabled", "seconds": 0.0}
        from srcGPU5_7.artstein_gpu5_7 import warm_cheap_artstein_gpu

        return warm_cheap_artstein_gpu(self.base)

    def warm_full_exact(self):
        """Compile the original full exact graph on this actor's A100."""
        from srcGPU5_7.fitness import _compute_exact_result

        expression = (
            "add(add(mul(x1,x1),mul(x2,x2)),"
            "add(mul(x3,x3),mul(x4,x4)))"
        )
        started = time.perf_counter()
        try:
            result = _compute_exact_result(expression, (), self.base)
            status = "ok" if result.status == "ok" else str(result.status)
        except Exception as exc:
            status = (
                f"failed: {type(exc).__name__}: "
                f"{str(exc).splitlines()[0][:400]}"
            )
        return {"status": status, "seconds": time.perf_counter() - started}

    def audit_full_exact(self, expression, constants):
        """Run the original full GPU manifold checker for final acceptance."""
        from srcGPU5_7.fitness import (
            _compute_exact_result,
            _exact_result_penalty,
        )

        result = _compute_exact_result(expression, constants, self.base)
        if result is None or result.status != "ok":
            status = "missing" if result is None else str(result.status)
            raise RuntimeError(f"GPU5 final full exact audit failed: {status}")
        return {
            "status": result.status,
            "roots": int(result.n_roots),
            "violations": int(result.n_violations),
            "margin_max": float(result.margin_max),
            "penalty": float(_exact_result_penalty(result, self.base)),
            "metrics": dict(getattr(result, "gpu2_metrics", {})),
        }


def _actor_map(items, *, actors, batch_size, method_name):
    """Work-stealing actor map for one fitness stage."""
    batches = [
        (start, items[start:start + batch_size])
        for start in range(0, len(items), batch_size)
    ]
    results = [None] * len(items)
    active = {}
    next_batch = 0

    def submit(actor, start, batch):
        method = getattr(actor, method_name)
        active[method.remote(batch)] = (actor, start, len(batch))

    for actor in actors:
        if next_batch >= len(batches):
            break
        start, batch = batches[next_batch]
        submit(actor, start, batch)
        next_batch += 1

    while active:
        ready, _ = ray.wait(list(active), num_returns=1)
        reference = ready[0]
        actor, start, size = active.pop(reference)
        batch_result = ray.get(reference)
        if len(batch_result) != size:
            raise RuntimeError(
                f"GPU5 actor returned the wrong {method_name} batch length"
            )
        results[start:start + size] = batch_result

        if next_batch < len(batches):
            new_start, batch = batches[next_batch]
            submit(actor, new_start, batch)
            next_batch += 1
    return results


def persistent_actor_mapper(
    _function,
    individuals,
    *,
    actors,
    pre_batch_size,
    cheap_batch_size=32,
    exact_batch_size=1,
    exact_wave_first_limit=500,
    exact_wave_second_limit=1500,
    full_exact_enabled=True,
    exact_cache=None,
):
    """MSE gate -> mandatory cheap screen -> bounded current exact waves."""
    individuals = list(individuals)
    if not individuals:
        return []
    # GPU5 batches candidates per exact call; exact_batch_size candidates are
    # verified together in one batched GPU scan+bisection.
    exact_batch_size = max(1, int(exact_batch_size))

    started = time.perf_counter()
    intermediate = _actor_map(
        individuals,
        actors=actors,
        batch_size=int(pre_batch_size),
        method_name="evaluate_pre_exact",
    )
    pre_seconds = time.perf_counter() - started

    cheap_positions = [
        index
        for index, entry in enumerate(intermediate)
        if entry.get("needs_cheap", False)
    ]
    # Deduplicate identical expressions/constants on the driver, while every
    # distinct candidate below gate 1 still receives the cheap check.
    cheap_groups = OrderedDict()
    for index in cheap_positions:
        key = intermediate[index].get("exact_key") or ("position", index)
        cheap_groups.setdefault(key, []).append(index)
    cheap_representatives = [group[0] for group in cheap_groups.values()]

    cheap_started = time.perf_counter()
    if cheap_representatives:
        cheap_checked = _actor_map(
            [intermediate[index] for index in cheap_representatives],
            actors=actors,
            batch_size=max(1, int(cheap_batch_size)),
            method_name="evaluate_cheap",
        )
        for (key, positions), checked_entry in zip(
            cheap_groups.items(), cheap_checked
        ):
            for duplicate_number, index in enumerate(positions):
                entry = dict(checked_entry)
                entry["consts"] = intermediate[index]["consts"]
                entry["pre_exact"] = intermediate[index]["pre_exact"]
                entry["extra_penalty"] = intermediate[index]["extra_penalty"]
                entry["exact_key"] = key
                if duplicate_number:
                    entry["cheap_diagnostic"] = dict(
                        entry.get("cheap_diagnostic", {})
                    )
                    entry["cheap_diagnostic"]["cache"] = "driver_dedup"
                if not entry.get("needs_full_exact", False):
                    # GPU5_3: gate2_penalty already contains the escalated
                    # cheap charge -- see fitness_finish_cheap.
                    entry["fitness"] = (
                        float(
                            entry["pre_exact"]
                            + entry["gate2_penalty"]
                            + entry["extra_penalty"]
                        ),
                    )
                intermediate[index] = entry
    cheap_seconds = time.perf_counter() - cheap_started
    # The budget value is copied into each gate-1-passing entry by fitness;
    # reading it here avoids another actor round trip.
    cheap_budget = float(
        next(
            (
                entry.get("cheap_budget_seconds", 15.0)
                for entry in intermediate
                if entry.get("needs_full_exact", False)
                or "cheap_diagnostic" in entry
            ),
            15.0,
        )
    )
    if cheap_representatives and cheap_seconds > cheap_budget:
        # GPU5_5: warn, do not raise.
        #
        # This was a hard RuntimeError and it KILLED run 122300 after 12h19m at
        # generation 323 -- with the best fitness of any run in the family
        # (317.0, properness down to 2, V>0 down to 0, boundary_min positive) --
        # for overrunning by 0.133s on a 15.000s ceiling. 0.9%.
        #
        # The budget is a safety ceiling on a screen whose cost varies with how
        # many candidates clear gate 1 and how much the exact stage is
        # contending for the same GPUs; it was never a correctness invariant.
        # Losing the whole run to a transient scheduling hiccup is strictly
        # worse than one slow generation. Measured cheap_s: 122299 p50 1.51s,
        # 122300 p50 9.61s / max 14.35s, 122301 p50 11.47s / max 14.28s -- the
        # last two run at 60-95% of the old ceiling continuously.
        global _CHEAP_BUDGET_WARNINGS
        _CHEAP_BUDGET_WARNINGS += 1
        if _CHEAP_BUDGET_WARNINGS <= 5 or _CHEAP_BUDGET_WARNINGS % 100 == 0:
            print(
                "GPU5 cheap-check budget exceeded (continuing): "
                f"{cheap_seconds:.3f}s > {cheap_budget:.3f}s "
                f"[occurrence {_CHEAP_BUDGET_WARNINGS}]",
                flush=True,
            )

    exact_positions = [
        index for index, entry in enumerate(intermediate)
        if entry.get("needs_full_exact", False)
    ]
    # Deduplicate exact work within the current generation. Completed results
    # remain reusable, but unscheduled candidates are never carried forward.
    exact_groups = OrderedDict()
    for index in exact_positions:
        key = intermediate[index].get("exact_key") or ("position", index)
        exact_groups.setdefault(key, []).append(index)
    missing_current = [
        (key, dict(intermediate[positions[0]]))
        for key, positions in exact_groups.items()
        if key not in _FULL_EXACT_RESULTS
    ]
    missing_current.sort(key=lambda item: _exact_priority(item[1]))

    cache_before = (
        ray.get(exact_cache.stats.remote()) if exact_cache is not None else {}
    )
    exact_started = time.perf_counter()
    # Each wave owns one candidate per A100. _actor_map work-steals across the
    # complete current list, so all six actors remain occupied until fewer than
    # six candidates remain in the final partial wave.
    wave_size = len(actors)
    if not full_exact_enabled:
        wave_count = 0
        exact_capacity = 0
    elif _EXACT_MAX_PER_GEN > 0:
        # Verify the best `_EXACT_MAX_PER_GEN` current-generation candidates
        # (priority-sorted above); wave_count is derived only for logging.
        exact_capacity = _EXACT_MAX_PER_GEN
        wave_count = (exact_capacity + wave_size - 1) // max(1, wave_size)
    else:
        wave_count = _exact_wave_count(
            len(cheap_positions),
            first_limit=exact_wave_first_limit,
            second_limit=exact_wave_second_limit,
        )
        exact_capacity = wave_size * wave_count
    scheduled = missing_current[:exact_capacity]
    exact_deferred = max(0, len(missing_current) - len(scheduled))
    scheduled_results = []
    scheduled_diagnostics = []
    if scheduled:
        scheduled_results = _actor_map(
            [entry for _, entry in scheduled],
            actors=actors,
            batch_size=exact_batch_size,
            method_name="evaluate_exact",
        )
    for (key, _), checked in zip(scheduled, scheduled_results):
        diagnostic = checked.get("_exact")
        if not diagnostic or diagnostic.get("status") != "ok":
            status = "missing diagnostic" if not diagnostic else diagnostic.get("status")
            raise RuntimeError(f"GPU5 full exact checker failed: {status}")
        scheduled_diagnostics.append(diagnostic)
        _FULL_EXACT_RESULTS[key] = dict(diagnostic)
        _FULL_EXACT_RESULTS.move_to_end(key)
        while len(_FULL_EXACT_RESULTS) > _FULL_EXACT_RESULT_LIMIT:
            _FULL_EXACT_RESULTS.popitem(last=False)
    exact_seconds = time.perf_counter() - exact_started

    results = [None] * len(individuals)
    pending_instances = 0
    exact_diagnostics = []
    for index, entry in enumerate(intermediate):
        if not entry.get("needs_full_exact", False):
            results[index] = {
                "consts": entry["consts"],
                "fitness": entry["fitness"],
            }
            continue
        key = entry.get("exact_key") or ("position", index)
        diagnostic = _FULL_EXACT_RESULTS.get(key)
        if diagnostic is None:
            # Gate-passing candidate that missed this generation's exact wave.
            # In GPU5_3 gate2_penalty was re-priced from its own cheap result
            # in fitness_finish_cheap, so a deferral is charged something
            # monotone in a_max instead of a flat fee that would flatten the
            # frontier exactly where the search needs resolution.
            pending_instances += 1
            exact_penalty = float(entry["gate2_penalty"])
        else:
            exact_penalty = float(diagnostic["penalty"])
            exact_diagnostics.append(diagnostic)
        results[index] = {
            "consts": entry["consts"],
            # GPU5_3: max(), not sum() -- see fitness_finish_exact. Charging
            # cheap_penalty + exact_penalty prices the same violation twice.
            "fitness": (
                float(
                    entry["pre_exact"]
                    + max(float(entry["cheap_penalty"]), exact_penalty)
                    + entry["extra_penalty"]
                ),
            ),
        }

    cache_after = (
        ray.get(exact_cache.stats.remote()) if exact_cache is not None else {}
    )
    cache_delta = {
        name: int(cache_after.get(name, 0) - cache_before.get(name, 0))
        for name in ("hits", "misses", "waits")
    }
    cheap_rejected = sum(
        "cheap_diagnostic" in entry
        and not entry.get("needs_full_exact", False)
        for entry in intermediate
    )
    cheap_a_values = [
        float(entry.get("cheap_a_max", float("nan")))
        for entry in intermediate
        if "cheap_diagnostic" in entry
    ]
    cheap_a_values = [value for value in cheap_a_values if value == value]
    exact_margins = [
        float(item.get("margin_max", float("nan")))
        for item in exact_diagnostics
    ]
    exact_margins = [value for value in exact_margins if value == value]
    gate2_values = [
        float(entry.get("gate2_penalty", float("nan")))
        for entry in intermediate
        if "gate2_penalty" in entry
    ]
    gate2_values = [value for value in gate2_values if value == value]
    exact_a_gates = [
        float(entry.get("exact_a_gate", float("nan")))
        for entry in intermediate
        if "exact_a_gate" in entry
    ]
    exact_a_gates = [value for value in exact_a_gates if value == value]
    print(
        "GPU5 stages: "
        f"candidates={len(individuals)} "
        f"pre_s={pre_seconds:.3f} "
        f"cheap_candidates={len(cheap_positions)} "
        f"cheap_unique={len(cheap_representatives)} "
        f"cheap_s={cheap_seconds:.3f}/{cheap_budget:.3f} "
        f"cheap_rejected={cheap_rejected} "
        f"cheap_a_min={min(cheap_a_values, default=float('nan')):.12g} "
        f"cheap_a_p50={_percentile(cheap_a_values, 0.50):.12g} "
        f"cheap_a_p95={_percentile(cheap_a_values, 0.95):.12g} "
        f"cheap_a_max={max(cheap_a_values, default=float('nan')):.12g} "
        f"exact_a_gate={next(iter(exact_a_gates), float('nan')):.12g} "
        f"exact_provisional_penalty={next(iter(gate2_values), float('nan')):.12g} "
        f"exact_candidates={len(exact_positions)} "
        f"exact_unique={len(exact_groups)} "
        f"full_exact_enabled={int(bool(full_exact_enabled))} "
        f"exact_capacity={exact_capacity} "
        f"exact_waves={wave_count} "
        f"exact_scheduled={len(scheduled)} "
        f"exact_deferred={exact_deferred} "
        f"exact_wave={len(scheduled)}/{exact_capacity} "
        f"exact_queue_depth=0 "
        f"exact_pending_instances={pending_instances} "
        f"exact_s={exact_seconds:.3f} "
        f"exact_ok={len(scheduled_results)} "
        f"exact_margin_max={max(exact_margins, default=float('nan')):.12g} "
        f"exact_scan_mean_s={_metric_mean(scheduled_diagnostics, 'scan_and_masks_s'):.3f} "
        f"exact_bisect_mean_s={_metric_mean(scheduled_diagnostics, 'bisection_s'):.3f} "
        f"exact_polish_mean_s={_metric_mean(scheduled_diagnostics, 'polish_s'):.3f} "
        f"exact_final_mean_s={_metric_mean(scheduled_diagnostics, 'final_score_s'):.3f} "
        f"exact_margin_abs_b_max={_metric_max(scheduled_diagnostics, 'margin_point_abs_b'):.3g} "
        f"cache_hits={cache_delta['hits']} "
        f"cache_misses={cache_delta['misses']} "
        f"cache_waits={cache_delta['waits']}",
        flush=True,
    )
    return results


def create_actor_pool(
    true_data,
    penalty,
    worker_count,
    tuner_fusion=16,
    full_exact_enabled=True,
    gpu_fraction=1.0,
    cpus_per_actor=1,
):
    cache = ExactResultCache.remote()
    # gpu_fraction < 1 packs multiple actors onto one physical GPU (Ray sets
    # the same CUDA_VISIBLE_DEVICES for them). The exact check is CPU-bound, so
    # this trades ~idle GPU for host-core parallelism. Each actor still
    # preallocates its XLA_PYTHON_CLIENT_MEM_FRACTION slice, so that env must be
    # <= gpu_fraction * (headroom) to avoid OOM when actors co-reside.
    actor_options = GPUFitnessActor.options(
        num_gpus=float(gpu_fraction),
        num_cpus=max(1, int(cpus_per_actor)),
    )
    actors = [
        actor_options.remote(true_data, penalty, cache, tuner_fusion)
        for _ in range(int(worker_count))
    ]
    devices = ray.get([actor.ready.remote() for actor in actors])
    warmups = ray.get([actor.warm_artstein.remote() for actor in actors])
    print(
        "GPU5 Artstein warm-up: "
        + ", ".join(
            f"gpu{index}={item.get('status')}:{item.get('seconds', 0.0):.2f}s"
            for index, item in enumerate(warmups)
        ),
        flush=True,
    )
    failures = [
        f"gpu{index}: {item.get('status')}"
        for index, item in enumerate(warmups)
        if item.get("status") != "ok"
    ]
    if failures:
        raise RuntimeError(
            "GPU5 cheap verifier warm-up failed; refusing to assign fitness: "
            + "; ".join(failures)
        )
    if not full_exact_enabled:
        print("GPU5 full exact warm-up: disabled", flush=True)
        return actors, cache, devices
    exact_warmups = ray.get([actor.warm_full_exact.remote() for actor in actors])
    print(
        "GPU5 full exact warm-up: "
        + ", ".join(
            f"gpu{index}={item.get('status')}:{item.get('seconds', 0.0):.2f}s"
            for index, item in enumerate(exact_warmups)
        ),
        flush=True,
    )
    exact_failures = [
        f"gpu{index}: {item.get('status')}"
        for index, item in enumerate(exact_warmups)
        if item.get("status") != "ok"
    ]
    if exact_failures:
        raise RuntimeError(
            "GPU5 full exact warm-up failed; refusing to assign fitness: "
            + "; ".join(exact_failures)
        )
    return actors, cache, devices
