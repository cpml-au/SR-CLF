# 3DCartPolerV6 = 3DCartPolerV5 + positivity penalties + price-selected Adam elites (2026-09-11)

Decided by the user after the 2026-09-11 analysis (`examples/DET4Fast/tuner_tests/`, summaries
`tuner_variants_summary.txt`, `elite_refiners_summary.txt`). Every switch is off by default in the
engine (`src3DCartPoleV3`), so V3..V5 are unchanged; `job3DCartPoleV6.slurm` turns them on.

## Why
* The DET4 price was flat where the run lived: 500 for any uncertified kappa* >= 0, 0 once
  certified with the level target met. Job 126391 sat at 500.4..500.9 from gen 100 to gen 146.
* The root check's tolerance is absolute (-1e-9 on a), so the verdict depends on V's scale:
  gen 119 x 100 is certified at price 0 with kappa* = 0.0011 unchanged (a 900 s decay time).
* The elite refine never improved a champion (0 of 132 records): its 3 slots went to no-root
  screens first, and the backtracking loop stops after 6-13 evaluations on frontier champions.
* Gen 147 of 126391 is a valid certificate (2M random points, 60-digit checks) with kappa* = 0.074
  but a needle-shaped V (Hessian eigenvalues 0.0156..55, c_v* 0.0076, certified volume 0.0017 vs
  gen 27's 0.0446): the shape, not kappa*, separates it from the robust gen 27 champion.
* On 235 realistic trees (A100): Adam is the lattice tuner (a longer SEA beats it on the lattice
  score at 40% of the cost but its constants are 10-30x worse on the exact kappa*); 12-23% of
  candidates are already at score 0 after the SEA, where Adam has no gradient and ran 40 steps.

## What V6 changes
1. Lattice tuner (`SYMCLF_TUNER_SKIP_ADAM_AT_ZERO`, `SYMCLF_TUNER_ADAM_EXIT_ZERO_GRAD`): Adam is
   skipped when the SEA score is already 0 and exits at the floor or when gradient and momentum are
   both zero. Bit-identical results. `SYMCLF_TUNER_ADAM_PATIENCE` (0 = off) would also stop after
   N steps without improvement (not identical).
2. Elite refine (`SYMCLF_DET4_REFINE_SELECT=price`, `_METHOD=adam`, `_TOP=10`, `_STEPS=100`,
   `_LR=0.02`): the generation's top-10 distinct candidates by the DET4 price are promoted to the
   exact stage whatever their screen ratio (so they also get the root check) and refined there
   with keep-best Adam on the full DET4 rule (DET4Fast numbers, Danskin gradients at the binding
   points, sympy gradient functions cached per expression: 0.3-0.7 s). Each candidate is refined
   once, at its first exact evaluation (cached results are not recomputed). Non-worsening.
3. Positivity penalties (`SYMCLF_DET4_PEN_*`), added to the price in both branches and to the
   kernel's lattice rule (lattice c_v, gated lattice kappa, certified volume), so the tuner sees
   the same objective:
       10 * clip((0.05 - c_v*)/0.05, 0, 1) + 30 * clip((0.05 - kappa*)/0.05, 0, 1)
       + 20 * clip((0.02 - certified volume)/0.02, 0, 1)
   Bounded at 60; 0 for the ARE (c_v* 0.064, kappa* 0.65, volume 0.076). Prices with the recorded
   numbers: ARE 0.18 (unchanged), gen 27 (126376) 20.2, gen 147 (126391) 27.5, gen 119 557,
   all-ones 1456. The certified-volume term has no gradient (a lattice count): it acts on the GP's
   selection, the refine uses the c_v*/kappa* hinges. The GEN line prints `pen=cv/kappa/vol`.

Cost per generation (V5: ~213 s pre-exact, 80 s screen + DET4Fast, 10 s exact): -20..-40 s from
the tuner exits, +33..90 s from the elite refine (10 x 100 x 0.3-0.8 s over 9 actors).

Note (not part of V6): `rollout_check.py` integrates at 2 ms; for stiff V (gen 147) RK4 is
unstable at that step (W rises in 1084/1500 steps) while 0.2 ms shows monotone decay.

# 3DCartPolerV5 = 3DCartPolerV4 + the full DET4 for every screened candidate (2026-09-10)

Only difference from V4 (`job3DCartPoleV5.slurm`: `SYMCLF_DET4_ALL=1`):
* every candidate that reaches the cheap screen gets the full DET4 numbers from det4_fast
  (c_v*, kappa*; option B gate) and is priced on them. No x2 escalation: it compensated the cheap
  kappa's understatement, and these are the full numbers.
* promotion to the exact root check is unchanged (the cheap screen's root ratio <= 0.15, no roots,
  or certified); promoted candidates reuse the numbers computed at the screen.
* GEN lines show `cheap[det4_all]`; the cheap stage now takes ~80 s per 2500 (the 40 s budget
  only prints a warning), the exact stage ~40 s less. Measured estimate: ~277 s -> ~320 s per
  generation.
* switch: `SYMCLF_DET4_ALL` in `src3DCartPoleV3/fitness.py` (`DET4_ALL`), off by default.

# 3DCartPolerV4 = 3DCartPolerV3 + det4_fast (2026-09-10)

Only difference from V3 (`job3DCartPoleV4.slurm`: `SYMCLF_DET4_FAST=1`, `NUMBA_NUM_THREADS=3`):
* the full DET4 numbers (c_v*, kappa*) come from `examples/DET4Fast/det4_fast.py` (the referee's
  det4 algorithm in numba, ~0.14 s on 24 threads, ~0.3 s on 3; W = V - V(0) evaluated from per-node
  increments, no cancellation). Switch: `SYMCLF_DET4_FAST` in `src3DCartPoleV3/fitness.py`
  (`_det4_fast_numbers`; off by default, so V3 runs are unchanged);
* the exact checker runs as in GPU5_7 (no `det4_rho` polish) and gives the root verdict;
* the elites' refine (3 x 40 steps) takes its numbers from det4_fast alone.
* option B (`SYMCLF_DET4_GATE_KAPPA_CAP=1e6`): kappa* gated on PD at all three levels. Not PD
  (full: the referee's pd_valid, c_v* > pd_eps + 1e-6; cheap and lattice: c_v > pd_eps) -> kappa*
  is not computed (det4_fast skips the rate stages) and the rule charges kappa* = -1e6, i.e.
  W (1 + log1p(1e6)) ~ 7400 + the c_v term; no W > 0 point keeps 1e6. Every PD candidate with
  kappa* > -1e6 beats every non-PD one; the arbitrary 1e12..1e54 kappa* noise is gone.
  Promotion to the full check still uses the ungated screened ratio.
Scheduling (GPU5_7), tuner (SEA 35 x 5 + Adam 40 top 4), gates: unchanged.

# 3DCartPolerV3 — DET4 mode (2026-09-10)

`examples/3DCartPolerV3` + engine `src3DCartPoleV3/` (copied from `symclf-main_rnv`, same file
structure as the GPU5_x examples). V is the GP tree itself (no wrap, no anchor); states
x1 = cart velocity, x2 = pole angle, x3 = pole rate; matched box (±0.25, ±0.0494, ±0.1155).
`SYMCLF_EXACT_MODE=roots` is the copied V3 pipeline, untouched; the default here is `det4`.

## What DET4 mode is

The **DET4 condition inside the GPU5_7 pipeline**: the same search (chords, bisection, 40 SLSQP
polishes, the GPU5 PD lattice) and the same scheduling (every gate-1 pass is screened, the full
check goes to the ones near certification, escalated cheap price for the rest), with the DET4
ratios as the polished quantities instead of the absolute margins:

| quantity | GPU5_7 (roots mode) | DET4 mode |
|---|---|---|
| Artstein | max a at the b = 0 roots, absolute, ball 1.1e-3 (the VERDICT, kept) | + max a/W at the same roots and points → κ* = −max, origin excluded at x = 0 only |
| PD | min W − ε·x'Px, absolute | c_v* = min( min W/‖x‖², min (x·∇V)/(2‖x‖²) ), both polished |
| price | margin curves, PD curve, cert_penalty, rollouts | the ROA rule on (c_v*, κ*) below, nothing else |

```
fitness   = pre_exact + extra + price
pre_exact = symbolic (1e6 nested / missing state) + invalid (1e6 non-finite) + ROA_RULE(lattice c_v, lattice κ)
extra     = 50 per nested call + 0.005 * length
price     = ROA_RULE(numbers of the last stage reached) - ROA_RULE(lattice numbers)

ROA_RULE  certified (c_v* > 1e-4, κ* > 0, no root violator)  ->  500 * clip(1 - boundary_min/(w_scale*0.004), 0, 1)
          else                                                ->  500 * (1 + log1p(relu(-κ*))) + 1000 * relu(1 - c_v*/1e-4)
          no W > 0 point at all (constant V)                  ->  1e6
```

## The stages (GPU5_7 scheduling)

| stage | who | what | numbers |
|---|---|---|---|
| grid | everyone, one Pallas call; the tuner's objective (SEA 35×5 + Adam 40 on the lattice, unchanged) | DET4 on the 41³ lattice, finite-difference ∇V, no polish | lattice c_v (min of W/r² and radial), lattice κ = −max (a − ρ\|b\|)/W over W > 0 |
| cheap screen (`artstein_v3`) | every gate-1 pass, batched 32 per launch | GPU5_7's chords → bisection roots; a/W read at the roots (V rides in the Pallas ab kernel) | κ_cheap = −max a/W at the roots (lattice κ if no W > 0 root); c_v from the lattice |
| exact (`b_manifold_check_gpu3` / `_batch_gpu4` + `pd_check_v3`) | GPU5_7's promotion: max a/W ≤ `GPU2_EXACT_A_MAX_GATE` (0.15), no roots, or cheap-certified; capacity `GPU3_EXACT_MAX_PER_GEN` per generation, best cheap ratio first | scans → bisection → 40 SLSQP on a (verdict, unchanged); the referee's rate stage + polish on (a − ρ\|b\|)/W seeded by the scan lattice and the roots; PD lattice → 40 SLSQP + NM on W/‖x‖² + 40 NM on the radial | κ* = the referee's, c_v* = min(cv_w, cv_rad); root verdict |
| elites | best `DET4_REFINE_TOP` (3) by cheap ratio, before their full check | `refine_constants_det4`: descent on the rule with the full numbers, Danskin gradients at the binding points, backtracking, `DET4_REFINE_STEPS` (40) evaluations | refined constants replace the individual's |

Not promoted → GPU5_7's escalated cheap price: the rule with κ_cheap × `CP3D_GATE_PRICE_ESCALATION` (2),
so refusing the check is never cheaper than taking it (the screen understates the max ratio by
~1e-3 relative, measured).

## Accuracy against the referee (src/srcCPU DET4, origin 0, κ over W > 0), CPU, 2026-09-10

The exact stage runs the referee's own rate stage and polish (`rate_stage`, `manifold_seed_set_rate`,
`manifold_ascent_rate`, `rate_polish`: half-space SLSQP + Nelder-Mead on m = (a − ρ|b| + γ₁)/W)
over the exact checker's scan lattice, with the bisected roots added to both seed sets. Two
earlier variants were rejected by this table: "max a/W s.t. b = 0" read 5.7e-5 too high on
all-ones (a point 6e-9 off the manifold at r = 1e-3; the ratio's b-sensitivity grows like 1/r),
and the half-space polish from the roots alone missed the near-origin maxima by 2e-5..1e-3.

| candidate | κ* new | κ* referee | new − ref | c_v* (new = ref) |
|---|---|---|---|---|
| ARE quadratic | 0.649339150 | 0.649339150 | 0 | 0.063594407 |
| all-ones | −5.366563146 | −5.366563146 | −5.0e-11 | 1.0 |
| 126165 gen 1 quadratic | −4.772884034 | −4.772884034 | +4.0e-10 | 8.4639788 |
| 126166 gen 11 near-CLF | −0.056246076 | −0.056246076 | −2.1e-16 | 0.2161906 |
| 126177 gen 6 | −0.728541699 | −0.728541699 | −3.3e-16 | 0.0510371 |
| 126177 gen 8 | −1.199379495 | −1.199379495 | −2.2e-16 | 1.2637950 |

GPU4 batched path (the actors' path, 24 per batch) = GPU3 single path to 0 on the same three
candidates. Pallas rows V / x·∇V vs the numba dual interpreter: 0 / 1.2e-16. The smoke asserts
κ* never higher than the referee (+1e-6) and c_v* within 1e-6, and exits 1 on FAIL (the slurm
preflight aborts the job).

## Found on the way (2026-09-10)

**The exact checker's SLSQP polish was dead in the 3-D V3 engine** (`cpu_polish.py`, a 4-D
leftover `cz[3]` on a 3-vector raised IndexError on every SciPy fun/jac pair; caught, so every
polish failed silently): `polished_accepted: 0` in every record of jobs 126150–126177 and in V2 /
3DCartPole. Fixed the way V7c fixed it (`np.array_equal`). The ARE's root count went 308 → 388.

Earlier findings (jobs 126150, 126164, 126165, 126166, 126177) and the decisions taken are in
the git history of this file; the mechanisms that survived: κ over the W > 0 points whether or
not V is PD, constant V pays 1e6, the c_v term carries 2W, the radial c_v, exp/aq out of the
primitive set. Removed on 2026-09-10 (they were mine, not GPU5_7's): the separate lattice-DET4
"cheap" module, the per-batch caps 8/3, the floor-tie screening and the provisional prices.

## Per-generation output

```
GEN 12 best fitness=0.175 len=35 :: grid=0 [all terms 0] lattice c_v=0.06362 kappa=2.009 | extra=0.175
  | exact[ok] price=0 roots=388 viol=0 | DET4 c_v*=0.06359 kappa*=0.6493 certified=True worst_r=0.25
  | boundary_min=0.00401 w_scale=0.0238
```

Same data as `"breakdown"` in `<JOB_ID>_best_per_generation.jsonl`; `cheap[cheap_det4]` lines
carry `ratio_max`, `kappa`, `kappa_source` (roots | lattice), `certified`. `SYMCLF_CERTIFY_EVERY=N`
adds the certificate line (c*, κ*(Ω_c*)) from `certify.py` every N generations (report only).

## Knobs

`SYMCLF_EXACT_MODE` det4 | roots · `SYMCLF_DET4_RHO` 1000 · `SYMCLF_DET4_PD_EPS` 1e-4 ·
`SYMCLF_DET4_KAPPA_MIN` 0 · `SYMCLF_DET4_KAPPA_SCALE` 1 · `SYMCLF_CP3D_ROA_WEIGHT` 500 ·
`SYMCLF_CP3D_ROA_C_TARGET` 0.004 · `SYMCLF_DET4_REFINE_TOP` 3 · `SYMCLF_DET4_REFINE_STEPS` 40 ·
`SYMCLF_DET4_REFINE_LR` 0.3 · `SYMCLF_CERTIFY_EVERY` 0. Scheduling knobs are GPU5_7's:
`SYMCLF_GPU2_CHEAP_FUSION` 32, `SYMCLF_GPU2_EXACT_A_MAX_GATE` 0.15 (on max a/W here),
`SYMCLF_CP3D_GATE_PRICE_ESCALATION` 2, `SYMCLF_GPU3_EXACT_MAX_PER_GEN` 1200,
`SYMCLF_GPU4_EXACT_BATCH_SIZE` 24. `SYMCLF_CP3D_ROLLOUT_ENABLED` defaults to 0 (no rollouts in
det4 mode).

## Code changes (all marked `2026-09-09` / `2026-09-10`)

| file | change |
|---|---|
| `src3DCartPoleV3/runtime_exact_candidate.py` | Pallas ab kernel outputs 4 rows (a, b, V, x·∇V); `vabr_batch` on both bundles |
| `src3DCartPoleV3/b_manifold_check_gpu.py` | `_vabr_padded`, `_det4_ratio_polish` (the referee's rate stage + polish, seeded by the scan lattice and the roots), `_det4_numbers_from_points` |
| `src3DCartPoleV3/b_manifold_check_gpu3.py`, `_batch_gpu4.py` | `det4_rho=` kwarg: a, b, V on the scan lattice (one more Pallas pass), the referee's rate polish, `det4_ratio_max` / `det4_point` / `det4_info` on the result; the verdict's point set untouched |
| `src3DCartPoleV3/artstein_v3.py` | `det4=` kwarg: a/W at the bisected roots (`det4_ratio_max`, `det4_point`) |
| `src3DCartPoleV3/pd_check_v3.py` | `ratio=True`: W/‖x‖² and the radial ratio, both polished; `cv`, `cv_w`, `cv_rad`, `pd_point`, `pd_radial_point` |
| `src3DCartPoleV3/cpu_polish.py` | the cache fix (above); `v_fn`, `gv_fn`, `v0` |
| `src3DCartPoleV3/grid_fitness.py` | kernel: DET4 lattice ratios (W and radial c_v), the rule, term ledger; details `terms`, `cv_grid`, `kappa_grid`, `cv_rad_grid` |
| `src3DCartPoleV3/fitness.py` | `EXACT_MODE`, `_attach_det4`, `_det4_numbers`, `_det4_cheap_numbers`, `_cheap_entry_det4`, `_det4_gate2_price`, `_roa_rule`, `_det4_price`, `refine_constants_det4` (+ gradients); det4 branches in `_exact_check_kwargs`, `fitness_finish_cheap`, `_second_gate_penalty`, `_compute_exact_result(s_batch)`, `fitness_finish_exact` |
| `src3DCartPoleV3/ray_fitness.py` | `_breakdown`, `_mark_refine`; det4 price replaces (no max with the cheap price) |
| `BaseEvaluate.py`, `Evaluate.py`, `run.py`, `config.yaml`, `job3DCartPoleV3.slurm` | radius 0 + absolute ball, DET4 knobs, per-generation breakdown, exp/aq removed, det4 exports, CPU preflight (`smoke_v3_det4.py`) |
| `certify.py`, `smoke_v3_det4.py`, this file | new |
