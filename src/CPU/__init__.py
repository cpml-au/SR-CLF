"""The CPU pipeline's package: one numba cache folder per (system, n, m, b-norm).

This runs before any module of the package, and so before numba is imported.  It points NUMBA_CACHE_DIR at
<repository>/.numba_cache/<tag>, tag = <system>_n<N>_m<M>_<b-norm> (``nx_mx`` in place of the dimensions
when the system file cannot be read).

Why one folder each: numba's disk cache keys on the function source and the argument types, NOT on module
globals, so a kernel compiled with n = 4 would be loaded unchanged into an n = 6 process (the Hessian then
arrives as a 4x4 against a 6-vector).  A folder per (system, n, m, b-norm) keeps the cache safe; it must be
set before `import numba` anywhere.

``apply_numba_cache_dir()`` can be called again once the environment is final (cma_run.py does), so that a
value inherited from a parent process with another system never wins.  SYMCLF_CPU_NUMBA_CACHE_DIR names
another folder.

Reading the dimensions imports the system file -- sympy and numpy only, never numba.
``example_dir(system)`` is the folder that holds it (also used by cma_run.py, audit.py and smoke.py).

Settings read here: system.name (SYMCLF_CPU_SYSTEM), verification.b_norm (SYMCLF_CPU_B_NORM).
"""
import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_os.path.dirname(_HERE))   # the repository root (src/CPU -> src -> repo)


def example_dir(system):
    """The example folder whose ``systems/`` holds ``<system>.py``, or None.

    ``examples/<system>`` when it holds the module (cartpole4d, ductedfan6d, link2, ...); otherwise the
    one ``examples/<folder>`` that does -- a folder may be named after the system rather than after its
    module (examples/quadrotor6d keeps the module name of its runs, systems/quad6d.py, so that NAME and
    the numba cache tag stay those of the runs).  None when no folder, or more than one, holds it.
    """
    import glob

    own = _os.path.join(_ROOT, "examples", system)
    if _os.path.isfile(_os.path.join(own, "systems", system + ".py")):
        return own
    hits = sorted(glob.glob(_os.path.join(_ROOT, "examples", "*", "systems", system + ".py")))
    return _os.path.dirname(_os.path.dirname(hits[0])) if len(hits) == 1 else None


def _dims(system):
    """``(N_STATES, N_INPUTS)`` of ``systems/<system>.py``, or None when it cannot be resolved.

    The unchanged loader contract is ``systems.<name>`` with the example folder on sys.path; when
    that is not set up yet (the package can be imported before the run script arranges it), the
    file is read directly from ``<example_dir(system)>/systems/<system>.py`` under a private module
    name, so nothing in sys.modules is shadowed.  Never raises: an unresolvable system only costs
    the dimensions in the tag.
    """
    import importlib
    import importlib.util

    module = None
    try:
        module = importlib.import_module("systems." + system)
    except Exception:  # noqa: BLE001 - not on sys.path (yet), or not importable
        folder = example_dir(system)
        path = _os.path.join(folder, "systems", system + ".py") if folder else ""
        if path and _os.path.isfile(path):
            try:
                spec = importlib.util.spec_from_file_location("_symclf_cpu_dims_" + system, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception:  # noqa: BLE001
                module = None
    try:
        return int(module.N_STATES), int(module.N_INPUTS)
    except Exception:  # noqa: BLE001 - no module, or a system without the n x m contract
        return None


def cache_tag():
    """``<system>_n<N>_m<M>_<b-norm>``; ``_nx_mx_`` when the system module cannot be resolved."""
    system = _os.environ.get("SYMCLF_CPU_SYSTEM", "quad6d") or "quad6d"
    b_norm = _os.environ.get("SYMCLF_CPU_B_NORM", "l1") or "l1"
    dims = _dims(system)
    return "{}_{}_{}".format(system, "n{}_m{}".format(*dims) if dims else "nx_mx", b_norm)


def apply_numba_cache_dir():
    """Point NUMBA_CACHE_DIR at this (system, n, m, b-norm)'s own directory and return it.

    Call it again after changing SYMCLF_CPU_SYSTEM / SYMCLF_CPU_B_NORM and BEFORE importing any
    module that compiles kernels; an inherited NUMBA_CACHE_DIR never wins, because that is how a
    parent process's default-system directory used to reach a child with a different system.
    """
    global CACHE_DIR, TAG
    TAG = cache_tag()
    CACHE_DIR = _os.environ.get("SYMCLF_CPU_NUMBA_CACHE_DIR") or _os.path.join(
        _ROOT, ".numba_cache", TAG
    )
    _os.environ["NUMBA_CACHE_DIR"] = CACHE_DIR
    _os.makedirs(CACHE_DIR, exist_ok=True)
    if "numba" in _sys.modules:   # a mixed process: make the new directory take effect
        _sys.modules["numba"].config.reload_config()
    return CACHE_DIR


TAG = ""
CACHE_DIR = ""
apply_numba_cache_dir()
