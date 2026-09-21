"""Package init -- runs before any submodule, and therefore before numba is imported.

numba's disk cache keys on the function source and the argument types, NOT on module globals,
so a kernel compiled with n = 4 would be reloaded verbatim in an n = 6 process (the Hessian then
arrives as a 4x4 against a 6-vector).  Giving every (system, b-norm) its own cache directory is
what keeps caching safe here; it must happen before `import numba` anywhere.
"""
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_TAG = "{}_{}".format(
    _os.environ.get("SYMCLF6D_SYSTEM", "quad6d") or "quad6d",
    _os.environ.get("SYMCLF6D_B_NORM", "l1") or "l1",
)
_os.environ.setdefault(
    "NUMBA_CACHE_DIR", _os.path.join(_os.path.dirname(_HERE), ".numba_cache", _TAG)
)
_os.makedirs(_os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
