"""GPU5_7: the certified-region objective, self-sufficient package.

WHAT CHANGED AND WHY (measured on runs 122304 / 122340, 2026-08-27)
--------------------------------------------------------------------
The decisive experiment: run 122340's generation-7 champion was "certified"
(PD passes, Artstein violations only above W = 0.0316, c_max > 0, 0.328% of
the box) and yet DIVERGES in closed loop even from initial conditions inside
its own certified sublevel set. Run 122304's champion is Artstein-INVALID on
paper (57 violating roots) and converges cleanly from every tested initial
condition. The discriminator neither check saw is CONTROL AUTHORITY:

    u_required = max over the box of  a / |b|   (where a > 0)

      96051  (verified CLF, converges)   u_req =  57
      122304 champion      (converges)   u_req =  59
      122340 gen-7 champ    (diverges)    u_req = 1.46e5

The pure Artstein condition (a < 0 exactly on {b = 0}) assumes UNBOUNDED
control. With any bounded actuator |u| <= u_max the achievable decrease is
a - u_max*|b|, which must be negative EVERYWHERE off the origin, not just on
the manifold. The gen-7 champion needed |u| ~ 1e11 along its trajectories --
no integrator and no actuator can follow that, so the mathematical
certificate was physically empty.

GPU5_7 therefore scores candidates on the REAL conditions only:

  1. SATURATION term (new): u_required on the grid, charged in decades above
     a target actuator bound. Scale-invariant (a/|b| is invariant under
     V -> kV). This single physical condition also subsumes the x1-axis
     loophole: f == 0 on the axis makes a == 0 there, so feasibility demands
     b != 0 on the axis -- exactly what the axis-b patch enforced by hand.
  2. ROA term (new): the certified level c_max = min(boundary_min,
     min W over violating grid points) and the certified volume fraction
     vol{W < c_max}/vol(box). Replaces the properness band, the c_star
     target, and the coverage count -- all of which were priors that fought
     the objective (the old fitness was measured ANTI-correlated with
     certified volume: it drove 122340 away from its own certified gen-7).
  3. CERTIFICATE-DEPTH term (new, exact stage): violations found by the
     exact Artstein/PD searches are priced by HOW LOW in W they sit
     (min W(violation)/boundary_min), because a violation at W ~ 0 empties
     the certificate while one near the boundary barely shrinks it.
  4. ROLLOUT gate (new, exact stage): a candidate that passes everything is
     integrated closed-loop with the SATURATED Sontag controller; divergence
     is charged. This is the test the gen-7 champion actually failed.
  5. PD penalty fixes: depth is normalised by the reference quadratic at the
     minimiser (scale-invariant, so shrinking V no longer discounts a PD
     hole), and the near-zero curve is read from the shared
     ``_exact_near0_penalty`` override instead of a hardcoded 4.37x copy.

Removed from the default objective (weights 0, knobs remain): properness
band count, V_GRAD magnitude targets, c_star-vs-reference target, coverage
count, the 1e6*V(0)^2 gauge penalty. Kept: V>0 and Vdot grid counts, the
strict margin predicate (a == 0 is a violation), the ARE reference form with
its import-time validity guard, the axis-b/axis-V and flat-gradient terms
(they charge valid CLFs ~0.2 points and provide the only continuous slope
out of the degenerate-slab family), tuner OOM retry, and the whole
cheap-screen / gate-pricing structure of GPU5_6.

SELF-SUFFICIENT: this package vendors every GPU module it needs
(pallas interpreter, runtime dual-number engine, GPU3 exact checker, GPU4
batch checker, cheap Artstein screen). It imports only ``src`` (the CPU
core) and itself -- no srcGPU2/3/4/5/5_x.
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
    """Raised when a GPU5_7 Ray task cannot see a CUDA JAX backend."""


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
            "GPU5_7 fitness task has no JAX GPU. Check Ray num_gpus and CUDA jaxlib."
        )
    if not _DEVICE_REPORTED:
        print(
            "GPU5_7 JAX devices: " + ", ".join(str(device) for device in devices),
            flush=True,
        )
        _DEVICE_REPORTED = True
    return devices
