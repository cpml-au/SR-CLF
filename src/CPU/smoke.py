#!/usr/bin/env python
"""Self-test of the pipeline: run it before submitting, and after any change.

  A. The cart-pole (cartpole4d): the code must give the stored numbers of two known candidates --
       a numerically verified quadratic              : fitness 5.972398158733, c_v* 0.0216130092063,
                                                       kappa* 0.0519077920849
       the same template at that run's start point   : fitness 1331.0431529522557
  B. quad6d (the 6-state quadrotor): the template evaluates, the quadratic of the Riccati solution is
     numerically verified, and the check (deterministic_search) agrees with the independent
     sympy/scipy audit (src/CPU/audit.py), the 60-digit re-check included.
  C. The template run (src/CPU/cma_run.py) does a few evaluations through its worker pool.
  B and C are skipped when no example folder holds systems/<system>.py (quad6d: examples/quadrotor6d).

Tolerances: A1-A2 fitness within 1e-9, A3-A4 c_v* and kappa* within 1e-10; B4 check vs audit
within 1e-8, B5 mpmath vs kappa* within 1e-9.
Settings read here: system.name (SYMCLF_CPU_SYSTEM: part A sets cartpole4d itself; B and C default to
quad6d).

    python -m src.CPU.smoke            # all parts   (from the repository root)
    python -m src.CPU.smoke --part b   # one part
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))   # src/CPU -> src -> repository root
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
PY = sys.executable


def _example_dir(system):
    """The folder that holds systems/<system>.py (the unchanged `systems.<name>` loader contract):
    examples/<system>, or the one example folder whose systems/ holds it (src/CPU/__init__.py::example_dir,
    e.g. examples/quadrotor6d for quad6d)."""
    from src.CPU import example_dir

    return example_dir(system) or os.path.join(ROOT, "examples", system)


def _has_system(system):
    return os.path.isfile(os.path.join(_example_dir(system), "systems", f"{system}.py"))

SEED2_4D = ("add(add(add(add(add(add(add(add(add(mul(a, mul(x1, x1)), mul(a, mul(x1, x2))), "
            "mul(a, mul(x1, x3))), mul(a, mul(x1, x4))), mul(a, mul(x2, x2))), mul(a, mul(x2, x3))), "
            "mul(a, mul(x2, x4))), mul(a, mul(x3, x3))), mul(a, mul(x3, x4))), mul(a, mul(x4, x4)))")
C_146800 = [0.048198739306794, 0.02242609132683108, -0.6814929779488536, -0.07891171609873011,
            0.06499410224542178, -0.3604573865991704, -0.5090843613379541, 4.770118393296521,
            1.5200301678935837, 1.546968743880691]
X0_146800 = [0.7573693448536968, 1.0177349308151737, 1.0101459968775517, 1.0222389106848166,
             0.5634568947554595, 1.0120257812333866, 1.0193227398080447, 0.8499275584188701,
             1.0113538416787295, 0.5615524075114275]
REF = {"fitness_best": 5.972398158733, "fitness_x0": 1331.0431529522557,
       "cv": 0.0216130092063, "kappa": 0.0519077920849}

_RESULTS = []


def check(name, ok, detail):
    _RESULTS.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)


def part_a():
    """4-D acceptance, in its own process (the system fixes the numba kernels' dimensions)."""
    script = f"""
import json, sys
sys.path.insert(0, {ROOT!r})
sys.path.insert(0, {_example_dir("cartpole4d")!r})
from src.CPU.rule import fitness_fixed_constants
from src.CPU.deterministic_search import deterministic_search
out = {{}}
o = fitness_fixed_constants({SEED2_4D!r}, {C_146800!r})
out['fitness_best'] = o['fitness']; out['numerically_verified'] = o['numerically_verified']
out['cv'] = o['check']['cv']; out['kappa'] = o['check']['kappa']
out['fitness_x0'] = fitness_fixed_constants({SEED2_4D!r}, {X0_146800!r})['fitness']
print('RESULT ' + json.dumps(out))
"""
    env = dict(os.environ, SYMCLF_CPU_SYSTEM="cartpole4d", SYMCLF_CPU_TEMPLATE="full")
    proc = subprocess.run([PY, "-c", script], capture_output=True, text=True, env=env, cwd=ROOT)
    line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    if not line:
        check("A 4-D acceptance", False, f"subprocess failed: {proc.stderr.strip()[-400:]}")
        return
    got = json.loads(line[0][7:])
    check("A1 fitness of the verified quadratic", abs(got["fitness_best"] - REF["fitness_best"]) < 1e-9,
          f"{got['fitness_best']:.12f} vs {REF['fitness_best']} (diff {got['fitness_best'] - REF['fitness_best']:+.2e})")
    check("A2 fitness at the tuner x0", abs(got["fitness_x0"] - REF["fitness_x0"]) < 1e-9,
          f"{got['fitness_x0']:.12f} vs {REF['fitness_x0']} (diff {got['fitness_x0'] - REF['fitness_x0']:+.2e})")
    check("A3 c_v* vs the stored value", abs(got["cv"] - REF["cv"]) < 1e-10,
          f"{got['cv']:.13f} vs {REF['cv']} (diff {got['cv'] - REF['cv']:+.2e})")
    check("A4 kappa* vs the stored value", abs(got["kappa"] - REF["kappa"]) < 1e-10,
          f"{got['kappa']:.13f} vs {REF['kappa']} (diff {got['kappa'] - REF['kappa']:+.2e})")
    check("A5 the 4-D quadratic is numerically verified", bool(got["numerically_verified"]), str(got["numerically_verified"]))


def part_b():
    """quad6d: the template, the ARE reference, and the check against the independent audit."""
    system = os.environ.get("SYMCLF_CPU_SYSTEM", "quad6d") or "quad6d"
    if not _has_system(system):
        print(f"SKIP B: no example folder holds systems/{system}.py", flush=True)
        return
    sys.path.insert(0, _example_dir(system))
    from src.CPU import config
    from src.CPU.deterministic_search import deterministic_search
    from src.CPU.rule import fitness_fixed_constants
    from src.CPU.templates import are_solution, default_template, monomials
    argv, sys.argv = sys.argv, sys.argv[:1]   # src/CPU/audit.py parses the command line at import
    try:
        from src.CPU.audit import audit
    finally:
        sys.argv = argv

    expression, x0, kind = default_template()
    n = config.system().N_STATES
    check("B1 system and box", config.SYSTEM == "quad6d" and len(config.box()) == 6,
          f"{config.SYSTEM}, box +-{config.box()[0]}, grid {config.grid_points()}^{n}, "
          f"lattice {config.lattice_kwargs()['mesh_points_per_axis']}/"
          f"{config.lattice_kwargs()['grid_points_per_axis']}/{config.lattice_kwargs()['scan_points']}")
    P, A, B = are_solution()
    are_c = [(P[i, i] if i == j else 2.0 * P[i, j]) for (i, j) in monomials(n, "full")]
    t0 = time.time()
    start = fitness_fixed_constants(expression, x0)
    t_eval = time.time() - t0
    check("B2 template evaluates at x0", bool(start["fitness"] > 0 and not start["dead"]),
          f"{len(x0)} coefficients, fitness {start['fitness']:.4f}, {t_eval:.1f} s")
    ref = fitness_fixed_constants(expression, are_c)
    check("B3 the ARE quadratic is numerically verified", bool(ref["numerically_verified"]),
          f"fitness {ref['fitness']:.4f}, c_v* {ref['check']['cv']:.6g}, kappa* {ref['check']['kappa']:.6g}")
    fast = deterministic_search(expression, are_c)
    ind = audit(expression, are_c, 8)
    dcv = abs(fast["cv"] - ind["cv"])
    dk = abs(fast["kappa"] - ind["kappa"])
    check("B4 DET4 vs the independent audit", dcv < 1e-8 and dk < 1e-8,
          f"c_v* diff {dcv:.2e}, kappa* diff {dk:.2e} (audit lattice {ind['n_lattice']:,} points)")
    check("B5 60-digit re-check of the worst point",
          ind["mp60_ratio_at_worst"] is not None
          and abs(-ind["mp60_ratio_at_worst"] - ind["kappa"]) < 1e-9,
          f"mpmath {ind['mp60_ratio_at_worst']:.13f} vs -kappa* {-ind['kappa']:.13f}")


def part_c():
    system = os.environ.get("SYMCLF_CPU_SYSTEM", "quad6d") or "quad6d"
    if not _has_system(system):
        print(f"SKIP C: no example folder holds systems/{system}.py", flush=True)
        return
    out = os.path.join(ROOT, "runs", "smoke_cma.jsonl")
    if os.path.exists(out):
        os.remove(out)
    cmd = [PY, "-m", "src.CPU.cma_run", "--budget", "8", "--popsize", "4", "--workers", "2",
           "--threads", "2", "--seed", "1", "--out", out, "--tag", "smoke"]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    finals = [json.loads(l) for l in open(out)] if os.path.exists(out) else []
    final = [r for r in finals if r.get("final")]
    check("C1 the CMA driver runs", bool(final) and final[0]["evals"] >= 8,
          f"{final[0]['evals'] if final else 0} evaluations in {time.time() - t0:.0f} s, "
          f"best {final[0]['best']:.4f}" if final else proc.stderr.strip()[-300:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="abc")
    args = ap.parse_args()
    t0 = time.time()
    if "a" in args.part:
        part_a()
    if "b" in args.part:
        part_b()
    if "c" in args.part:
        part_c()
    ok = sum(_RESULTS)
    print(f"\n{ok}/{len(_RESULTS)} PASS in {time.time() - t0:.0f} s", flush=True)
    sys.exit(0 if ok == len(_RESULTS) else 1)


if __name__ == "__main__":
    main()
