# 4D cart-pole CLF search — GPU5_7

GPU5_7 exists because of one experiment. Run 122340's **generation-7 champion
held the largest certificate any run had produced** — positive definite,
`c_max = 0.042`, every violation sitting above `W = 0.0316` — **and diverges
from inside that certified set** when the loop is actually closed. (Its own
run reported zero exact violations; under this folder's exact settings it
shows 656 with margin +1.42 — see the audit table. Either way the certified
region itself is real and large, which is the point: **a big certified region
is not sufficient**.) Meanwhile the 122304 champion, which fails the Artstein
check at 57 points, converges from every tested start. The old objective and
reality were not just misaligned; on these two candidates they were inverted.

## Why the "certified" CLF diverges

Steering along the gen-7 champion's `V` requires control effort

```
max over the box of  a/|b|  where a > 0:
    122340 gen-7 : 2.1e+05      (needs u ~ 10^5..10^11 near the b=0 manifold)
    96051        : 5.7e+01
    122304 champ : 5.9e+01
```

The Artstein condition only asks that `a < 0` **where b = 0 exactly** — it says
nothing about how hard you must push *near* that manifold. A GP left to
optimise the old objective found a `V` whose descent direction exists
mathematically but demands ~10⁵ force units where any real (or even simulated)
actuator saturates. With `|u| ≤ 10⁴` the trajectory crosses the certified
boundary and blows up. Bounded-control feasibility requires

```
a − u_max·|b| < 0   everywhere,   not only on {b = 0}.
```

## The five changes

All of them are measurements of the candidate itself — **no prior-shaped terms
remain active**. Properness band, `c_star` target, `V_GRAD` targets and
coverage are all weighted **0** by default (knobs still exist).

1. **Saturation term (grid).** `u_required = max a/|b|` over the box;
   penalty `SAT_WEIGHT · log10(u_required / SAT_U_TARGET)` when above target
   (default target `1e3`). Scale-invariant in `V`. This single term separates
   gen-7 (2.1e5) from every candidate that actually works (~60).

2. **ROA term (grid).** `c_max = min(W on box boundary, W at any violating
   point)` where `W = V − V(0)` and "violating" includes saturation-infeasible
   points. Priced entirely on the scale-free ratio `c_ratio = c_max /
   median|W|` (raw `c_max` scales with `V`, so a raw target would be bought by
   inflating `V`): a ramp `ROA_WEIGHT · clip(1 − c_ratio/ROA_C_TARGET, 0, 1)`
   that is continuous through zero and reaches 0 at `ROA_C_TARGET = 0.02`
   (96051 sits at 0.0221 — measured), plus `ROA_NEG_WEIGHT · log1p(−c_ratio)`
   below zero so deep-empty certificates still order. `certified_volume` is
   reported but no longer priced — it is quantised in 1/21⁴ steps and
   hard-zero for `c_max ≤ 0`, which put a 500-point cliff at `c_max = 0`
   (run 122350 stalled at `c_max = −4.8e-5`, 0.27 points of guidance from a
   500-point prize). This is the actual objective — the thing GPU5_3/5_4
   champions optimised *against* (their certified volume was ~0 while their
   fitness improved).

3. **Certificate depth (exact).** An exact violation at `W = 0.0001` empties
   the whole certificate; one at `W = 0.9·boundary_min` costs almost nothing.
   Violations are now priced by `min W over violating points / boundary_min`
   instead of being counted uniformly. PD failures are priced through the same
   curve, at the `W` of the PD minimiser.

4. **Closed-loop rollout gate (exact).** Candidates that pass PD and the exact
   Artstein check are integrated (RK4, saturated Sontag controller,
   `u_max = ROLLOUT_UMAX`, default 1e3) from 4 standard starts. Divergence
   (final ‖x‖ > `ROLLOUT_DIVERGE_NORM`) is charged `ROLLOUT_FAIL_PENALTY`
   per diverged start. Eligibility is strict so it costs CPU seconds only on
   candidates that are about to be declared winners.

5. **PD penalty fixes.** Depth is normalised by the reference quadratic at the
   minimiser (scale invariance) and priced through the same near-zero curve as
   Artstein margins (one currency, not two).

## Audit (offline, exact settings of this folder)

```
                          ---- grid stage ----                    ---- exact stage ----
             candidate  pre_exact   c_max     vol%     u_req    viol   PD   rollout        TOTAL
      96051 (verified)       0.17  0.0212    0.656      57       0   PASS  0/4 diverged      0.17
        122232 (valid)       0.19  0.0204    0.447      82       0   PASS  0/4 diverged      0.19
       340 final champ       0.07  0.0230    0.089     218      27   FAIL  not eligible    371.13
         ARE reference     384.45  0.0042    0.131     602       0   PASS  0/4 diverged    384.45
     304 champ (works)     502.08  0.0000    0.001      60      57   FAIL  not eligible    922.63
      54 champ (degen)     912.99  0.0019    0.027      40      64   PASS  not eligible   1245.44
      53 champ (degen)    1443.72 -0.0002    0.000      28     939   FAIL  not eligible   1673.15
  340 GEN-7 (diverges)     351.58  0.0420    0.593  2.1e+05    656   PASS  not eligible   3848.05
```

The two verified CLFs both sit near 0.2 and the diverging gen-7 is worst by
4.9×. One adjacency to know about: the 340 final champ (371, diverges) edges
below the ARE reference (384, converges but with a genuinely small
certificate, `c_ratio` 0.0044). The 340 final's invalidity lives off-grid, so
only the exact stage prices it — and if a descendant ever fixes its 27
violations and its PD failure, the rollout gate is what catches the
divergence. Forcing the rollout gate on gen-7 (bypassing eligibility):
**4/4 starts diverge**, final ‖x‖ ≈ 1.06; on 96051: 0/4, final ‖x‖ ≈ 10⁻⁴.

## Self-sufficiency

`srcGPU5_7/` contains everything it imports — Pallas interpreter, JAX/exact
candidates, GPU2/3/4 manifold checkers, CPU polish, ray plumbing. No imports
from `srcGPU2..srcGPU5_6`; those trees are untouched. Verified:
`rg "srcGPU[0-9]" srcGPU5_7/` matches only `srcGPU5_7`.

## Running

```bash
cd examples/4DCartPolerGPU5_7
sbatch jobGPU5_7.slurm
```

Key knobs (defaults in `Evaluate.py`, overridable in the slurm file):

```
SYMCLF_GPU5_7_SAT_WEIGHT / SAT_U_TARGET             saturation term      (150 / 1e3)
SYMCLF_GPU5_7_ROA_WEIGHT / ROA_C_TARGET             ROA c_ratio ramp     (500 / 0.02)
SYMCLF_GPU5_7_ROA_NEG_WEIGHT                        c_max < 0 charge     (250)
SYMCLF_GPU5_7_CERT_WEIGHT                           certificate depth    (300)
SYMCLF_GPU5_7_ROLLOUT_ENABLED / _UMAX / _T / _DT    rollout gate         (1 / 1e3 / 10 / 2e-3)
SYMCLF_GPU5_7_ROLLOUT_DIVERGE_NORM / _FAIL_PENALTY  divergence charge    (1.0 / 500)
SYMCLF_GPU5_7_PROPERNESS_WEIGHT, V_GRAD_WEIGHT,
  C_STAR_TAIL_WEIGHT, COVERAGE_WEIGHT               old prior terms      (all 0)
```

Everything structural from GPU5_6 is kept: the ARE reference with its
import-time validity guard, strict `margin > −1e-9` predicate, 21⁴ grid,
tuner-OOM retry, 40 s cheap budget.

## What to watch

1. **`u_required` of the champion per generation.** If it creeps above ~1e3
   the saturation weight is too low; every known-good candidate sits near 60.
2. **`certified_volume`** should grow monotonically-ish; it is the objective.
3. **The rollout gate should almost never fire** — it exists as the last line
   of defence. If it fires often, the saturation term is mis-calibrated.
4. Whether 96051-class candidates (`pre_exact ≈ 0.2`) are reachable: the gap
   between the seed's ~369 and 0.17 is now a descent on real quantities, not
   on priors.
