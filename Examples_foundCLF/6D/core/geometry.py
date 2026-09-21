"""The referee's scan lattice, vendored verbatim from src/srcCPU/exact_check_cpu.scan_geometry and
src/srcCPU/det4_check_cpu.sobol_points (dimension-generic already).

Point count for n states:  mesh^n + n * lines^(n-1) * samples + random_lines * random_line_points.
    4-D V8 defaults (21, 9, 801, 200, 401) : 2,610,397
    6-D this folder  (7,  3, 801, 200, 401): 1,365,707
"""
from __future__ import annotations

import numpy as np

_SOBOL_CACHE = {}
_COORDS_CACHE = {}


def sobol_points(bounds, count):
    """Scrambled Sobol starts, fixed seed (the same set every call)."""
    from scipy.stats import qmc

    bounds = np.asarray(bounds, dtype=float)
    if int(count) <= 0:
        return np.empty((0, bounds.shape[0]))
    sampler = qmc.Sobol(d=bounds.shape[0], scramble=True, seed=0)
    return qmc.scale(sampler.random(int(count)), bounds[:, 0], bounds[:, 1])


def sobol_cached(bounds, count):
    key = (np.asarray(bounds, float).tobytes(), int(count))
    if key not in _SOBOL_CACHE:
        _SOBOL_CACHE[key] = sobol_points(bounds, count)
    return _SOBOL_CACHE[key]


def scan_geometry(
    bounds,
    scan_axes=None,
    scan_points=801,
    grid_points_per_axis=9,
    random_lines=200,
    random_line_points=401,
    rng_seed=0,
    mesh_points_per_axis=0,
):
    """Mesh (optional), axis-aligned scan lines and fixed-seed random lines, as one point array."""
    bounds = np.asarray(bounds, dtype=float)
    n = bounds.shape[0]
    scan_axes = tuple(range(n)) if scan_axes is None else tuple(int(a) for a in scan_axes)
    geometry = {"mesh": None, "axes": [], "random": None}
    chunks = []
    offset = 0

    if mesh_points_per_axis and int(mesh_points_per_axis) > 1:
        g = int(mesh_points_per_axis)
        axes_lin = [np.linspace(bounds[i, 0], bounds[i, 1], g) for i in range(n)]
        mesh = np.meshgrid(*axes_lin, indexing="ij")
        points = np.stack([m.ravel() for m in mesh], axis=1)
        geometry["mesh"] = (axes_lin, g, slice(offset, offset + points.shape[0]))
        chunks.append(points)
        offset += points.shape[0]

    for axis in scan_axes:
        others = [i for i in range(n) if i != axis]
        lines = [np.linspace(bounds[i, 0], bounds[i, 1], grid_points_per_axis) for i in others]
        mesh = np.meshgrid(*lines, indexing="ij")
        combos = np.stack([m.ravel() for m in mesh], axis=1)
        line_count = combos.shape[0]
        samples = np.linspace(bounds[axis, 0], bounds[axis, 1], scan_points)
        points = np.empty((line_count * scan_points, n))
        for k, coordinate in enumerate(others):
            points[:, coordinate] = np.repeat(combos[:, k], scan_points)
        points[:, axis] = np.tile(samples, line_count)
        geometry["axes"].append((axis, others, combos, samples, slice(offset, offset + points.shape[0])))
        chunks.append(points)
        offset += points.shape[0]

    if random_lines > 0:
        rng = np.random.default_rng(rng_seed)
        origins = rng.uniform(bounds[:, 0], bounds[:, 1], (random_lines, n))
        directions = rng.normal(size=(random_lines, n))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            ta = (bounds[:, 0][None, :] - origins) / directions
            tb = (bounds[:, 1][None, :] - origins) / directions
        t0 = np.where(np.abs(directions) > 1.0e-12, np.minimum(ta, tb), -np.inf).max(1)
        t1 = np.where(np.abs(directions) > 1.0e-12, np.maximum(ta, tb), np.inf).min(1)
        fraction = np.linspace(0.0, 1.0, random_line_points)
        parameters = t0[:, None] + (t1 - t0)[:, None] * fraction[None, :]
        points = (origins[:, None, :] + parameters[..., None] * directions[:, None, :]).reshape(-1, n)
        geometry["random"] = (origins, directions, parameters, slice(offset, offset + points.shape[0]))
        chunks.append(points)
        offset += points.shape[0]

    geometry["coordinates"] = np.concatenate(chunks, axis=0) if chunks else np.empty((0, n))
    return geometry


def lattice_coords(bounds, **kwargs):
    """Cached (N, n) lattice."""
    key = (np.asarray(bounds, float).tobytes(), tuple(sorted(kwargs.items())))
    if key not in _COORDS_CACHE:
        _COORDS_CACHE[key] = np.ascontiguousarray(
            scan_geometry(np.asarray(bounds, float), **kwargs)["coordinates"], dtype=float
        )
    return _COORDS_CACHE[key]
