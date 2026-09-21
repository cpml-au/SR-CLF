"""Vendored srcGPU5_9 GPU engine (symclf-main_rnv, 2026-09): the Pallas bytecode interpreter
and the checks built on it.  Imported by src/srcGPU/exact_check_gpu.py and
src/srcGPU/det3_check_gpu.py as ``engine="pallas"``.

    grid_fitness.py            program encoder (prefix string -> postfix opcodes)
    pallas_interpreter.py      V on a point batch (used by the PD scan)
    runtime_exact_candidate.py b / (a, b) forward-derivative kernels, bisection, GPU SQP
    jax_candidate.py           the sympy -> JAX bundle (legacy sympy callers)
    cpu_polish.py              numba dual-number a, b, grad a, grad b for the SLSQP polish
    b_manifold_check_gpu.py    scan geometry + the GPU2-style check (GPU SQP polish)
    b_manifold_check_gpu3.py   GPU scans + the reference SLSQP polish (the exact contract)
    b_manifold_batch_gpu4.py   GPU3 batched over candidates (bit-identical, ~10x faster)
    pd_check_gpu5_9.py         positive definiteness, same pipeline
    artstein_gpu5_9.py         the cheap fully-GPU screen (224 chords/axis, no polish)

CAVEAT: the Pallas kernels hard-code the 4-D cart-pole drift and control column
(``_cartpole_fields`` and the b-kernel in runtime_exact_candidate.py); ``fSR``/``GSR`` only
reach the CPU polish.  For any other plant or dimension use engine="jax".  Requires the
flex env (JAX 0.5 + CUDA jaxlib, numba); float64 is forced below and must be set before JAX
creates any array.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_ENABLE_X64", "true")

_cache_dir = os.environ.get("SYMCLF_GPU2_JAX_CACHE_DIR")
if _cache_dir:
    os.makedirs(_cache_dir, exist_ok=True)
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", _cache_dir)
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0.5")


_DEVICE_REPORTED = False


class GPUUnavailableError(RuntimeError):
    """Raised when a GPU5_9 Ray task cannot see a CUDA JAX backend."""


def initialize_jax():
    """Import/configure JAX lazily after a Ray actor owns its CUDA device."""
    import jax

    jax.config.update("jax_enable_x64", True)
    if _cache_dir:
        jax.config.update("jax_compilation_cache_dir", _cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
    return jax


def require_gpu():
    """Fail clearly inside a Ray task if it did not receive a CUDA device."""
    global _DEVICE_REPORTED
    jax = initialize_jax()
    devices = jax.devices()
    has_gpu = any(device.platform == "gpu" for device in devices)
    allow_cpu = os.environ.get("SYMCLF_GPU2_ALLOW_CPU", "0") == "1"
    if not has_gpu and not allow_cpu:
        raise GPUUnavailableError(
            "GPU5_9 fitness task has no JAX GPU. Check Ray num_gpus and CUDA jaxlib."
        )
    if not _DEVICE_REPORTED:
        print(
            "GPU5_9 JAX devices: " + ", ".join(str(device) for device in devices),
            flush=True,
        )
        _DEVICE_REPORTED = True
    return devices
