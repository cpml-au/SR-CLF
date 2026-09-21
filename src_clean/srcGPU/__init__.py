"""GPU checks (ported from symclf-main_rnv/examples/VerifierBenchmark and srcGPU5_9, 2026-09).

exact_check_gpu.py   exact Artstein + PD with the scans and the bisection on the GPU, the
                     reference SciPy polish and verdict on the CPU (the GPU3 split)
det3_check_gpu.py    DET3 with a, b, V on the lattice from the GPU
gpu5_9/              the vendored Pallas bytecode engine (engine="pallas", 4-D cart-pole only)

Two engines, selected per call:
  engine="jax"     sympy V -> JAX (float64, a/b by autodiff), kernels compiled once per tree
                   template, constants are runtime data; any plant, any dimension.
  engine="pallas"  the vendored srcGPU5_9 engine: no per-tree XLA compile, batched exact
                   check and cheap screen; needs the GP prefix string + constants (a sympy V
                   is converted with symbolic.to_prefix) and the 4-D cart-pole dynamics.

float64 must be configured before JAX creates any array, hence the environment defaults
here (imported by both modules before jax).  Use the flex env (JAX 0.5 + CUDA jaxlib).
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_ENABLE_X64", "true")


def initialize_jax():
    """Import JAX with float64 enabled; returns the module."""
    import jax

    jax.config.update("jax_enable_x64", True)
    return jax
