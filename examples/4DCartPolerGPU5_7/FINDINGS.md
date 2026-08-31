# GPU5_7 — evidence log

Raw measurements behind the design decisions in this folder. All rollouts:
RK4, dt = 2e-3, Sontag controller `u = −k b λ` with `k = 1e4`, clipped to
`|u| ≤ u_max`, starts at the standard 4 initial conditions.

## 1. The inversion that motivated GPU5_7

Closed-loop rollouts (`u_max = 1e4`, T = 20 s):

```
candidate              certificate status                   rollout result
122340 gen-7 champ     PD ok, c_max 0.042 (largest ever)    DIVERGES (all starts)
122340 final champ     27 exact viol                        diverges
122304 champ           57 exact viol (Artstein INVALID)     CONVERGES (all starts)
96051                  fully verified                       converges, ‖x(T)‖ ~ 1e-4
```

Correction (independent review, confirmed): gen-7 did **not** pass the exact
Artstein check — its own run reported 0 violations, but under 5_7's exact
settings it shows `roots=1204 viol=656 margin_max=+1.42`. All of its
violations sit above `W = 0.0316`, so the certified region (`c_max` truncated
to ~0.032) is still the largest any run produced. The counterexample is
therefore "**a large certified region is not sufficient**", not "it passed
everything" — sharper, and the saturation term is still what separates it.

The objective declared the diverging candidate the best ever produced and the
converging one invalid. Two independent facts confirmed the diagnosis:

**Saturation scan** — max required control `a/|b|` over the 21⁴ box where `a > 0`:

```
122340 gen-7 : 2.11e+05     (and ~1e11 close to the b=0 manifold)
96051        : 5.66e+01
122304 champ : 5.94e+01
```

**Unbounded-control rollout** of gen-7 still fails — the states blow past the
box while `V̇ < 0` numerically, i.e. the level sets are so distorted that
following them is numerically stiff. Saturation is the mechanism; extreme
`u_required` is the observable that flags it on the grid, cheaply, for every
candidate.

## 2. Why the old prior terms are zeroed

GPU5_3/5_4 champions show the failure of prior-shaped fitness directly: their
fitness improved for dozens of generations while their `certified_volume`
stayed ≈ 0 and `c_max ≤ 0` (53 champ: c_max = −1.6e-4 — an empty certificate).
Properness bands, `c_star` ratios and `V_GRAD` targets all measure similarity
to the ARE quadratic, not validity. GPU5_6 fixed the anchor itself; GPU5_7
removes the anchor from the objective and keeps it only as PD reference and
seed structure.

## 3. Scale-invariance checks

- Saturation: `a/|b|` is invariant under `V → cV`.
- ROA: `c_max` scales with `V` but `certified_volume` (a fraction of grid
  points) and `c_max / median|W|` do not.
- Cert depth: `min_W_viol / boundary_min` is a ratio of two `W` values.
- PD depth: normalised by `x'P_ref x` at the minimiser — previously an
  absolute number, so `V → 100V` bought a 100× smaller PD penalty. Fixed.
- Gauge (`V → V + 5`): all terms live on `W = V − V(0)`, unchanged.

## 4. Offline audit of the panel (this folder's exact defaults)

```
             candidate  pre_exact   c_max    vol%     u_req  viol   PD  minW_viol  cert  rollout   TOTAL
      96051 (verified)       0.17  0.0212   0.656      57      0  PASS       inf     0   0/4 div    0.17
        122232 (valid)       0.19  0.0204   0.447      82      0  PASS       inf     0   0/4 div    0.19
       340 final champ       0.07  0.0230   0.089     218     27  FAIL   1.5e-04   302   n/elig   371.13
         ARE reference     384.45  0.0042   0.131     602      0  PASS       inf     0   0/4 div  384.45
     304 champ (works)     502.08  3e-06    0.001      60     57  FAIL   0.0       300   n/elig   922.63
      54 champ (degen)     912.99  0.0019   0.027      40     64  PASS   1.1e-04   282   n/elig  1245.44
      53 champ (degen)    1443.72 -0.0002   0.000      28    939  FAIL  -3.7e-04     0   n/elig  1673.15
  340 GEN-7 (diverges)     351.58  0.0420   0.593  2.1e+05    656  PASS   0.0316     75   n/elig  3848.05
```

(Numbers under the §8 c_ratio ramp. Two notable moves against the earlier
volume-priced audit: 122232 fell 52.85 → 0.19 — its 0.447% volume was under
the arbitrary 0.5% target despite a solid certificate, so the old term
mispriced a *verified* CLF; and the 340 final champ's grid ROA went to ~0
because its invalidity lives entirely off-grid, leaving it at 371, slightly
below the ARE reference at 384. That adjacency is accepted: its 27 exact
violations and PD failure carry the price, and a descendant that repairs
them becomes rollout-eligible, where the divergence is caught.)

Reading:

- The candidates that pass exact checks all also pass the rollout (0/4
  diverged, no penalty added), and both verified CLFs sit near 0.2.
- Gen-7 is worst by 4.9×. Note it now shows **656 exact violations** under the
  5_7 strict predicate/budgets even though run 122340 reported none — the
  original run's exact stage sampled it differently. The saturation term alone
  (u_req = 2.1e5 → log10 penalty) would have flagged it regardless.
- 304's cert penalty is maximal (min W at violations = 0.0: violations touch
  the certificate floor, certified set is empty) — correct even though it
  happens to converge; it is not a certificate.
- 53 champ gets cert = 0 because its `boundary_min < 0` — there is no
  certificate to empty; it is charged through ROA (`c_max < 0`) and its 939
  violations instead.

## 5. Rollout gate, forced (eligibility bypassed)

```
122340 gen-7 : 4/4 diverged, final ‖x‖ = [1.059, 1.006, 1.007, 1.007]  (u_max = 1e3)
96051        : 0/4 diverged, final ‖x‖ = [1.7e-4, 6.5e-4, 5.1e-4, 8.6e-4]
```

The gate does what it is for. In production it only runs on candidates with
zero exact violations + PD pass, so its per-generation cost is ≈ 0 until the
search produces a would-be winner.

## 6. Self-sufficiency verification

`srcGPU5_7/` bundles: `pallas_interpreter`, `jax_candidate`,
`runtime_exact_candidate`, `b_manifold_check_gpu` (GPU2), `b_manifold_check_gpu3`
+ `cpu_polish` (GPU3), `b_manifold_batch_gpu4` (GPU4), `artstein/grad_norm/pd_check`
(renamed from 5_6), `grid_fitness`, `fitness`, `ray_fitness`.
`rg "srcGPU[0-9]"` over the tree matches only `srcGPU5_7`. No file in
`srcGPU2..srcGPU5_6` or their example folders was modified.

## 7. ROA negative side: clip → log1p (fix from run 122349, gen 14)

The first live run exposed a shelf in the original negative-side term
`ROA_NEG_WEIGHT · clip(-c_max/w_scale, 0, 1)`:

```
 c_max/w_scale      vol   vol term   neg term    TOTAL
           0.1    0.003      200.0        0.0    200.0
           0.0      0.0      500.0        0.0    500.0
          -1.0      0.0      500.0      250.0    750.0
        -100.0      0.0      500.0      250.0    750.0     <- flat
      -10000.0      0.0      500.0      250.0    750.0     <- flat
```

Everything with `c_max ≤ −w_scale` scored an identical 750 — the same
saturation defect removed from `c_star` (5e7 shelf), the axis shoulders and
the PD count, relocated. And the early population lives exactly there: at
gen 14 the champion had `certified_volume = 0`, `c_max = −0.0053`,
`u_required = 882` (below target, saturation term = 0), so the entire ROA
objective was a constant offset and the search was being driven by the exact
manifold penalty alone — the failure mode of the earlier versions.

Fixed to `ROA_NEG_WEIGHT · log1p(max(0, −c_max/w_scale))`: identical slope at
shallow negatives (`log1p(x) ≈ x`), monotone and unbounded at depth
(depth 1 → 0.69·w, 100 → 4.6·w, 1e4 → 9.2·w), so deep-empty certificates
always order. Requires restarting any run launched before the fix.

## 8. ROA positive side: volume → c_ratio ramp (fix from run 122350)

The volume half had a dead zone exactly where a winning run arrives. Review
table (measured on the live 122350 champion's V):

```
       c_max   cert vol   HALF A   HALF B  ROA total
  -1.000e-02    0.00000    500.0    50.37      550.4
  -4.763e-05    0.00000    500.0     0.27      500.3   <- champion here
  -1.000e-06    0.00000    500.0     0.01      500.0
   1.000e-06    0.00031    468.6     0.00      468.6   <- cliff: 500 unlocks
   5.000e-03    0.03015      0.0     0.00        0.0
```

`certified_volume = where(c_max > 0, ..., 0.0)` hard-zeroes HALF A across the
whole negative side, so a candidate 5e-5 from crossing zero received 0.27
points of guidance for closing 98% of the gap, with a discontinuous 500-point
prize hidden behind `c_max = 0`. (§7's log1p fix is correct for deep
negatives but is worth almost nothing at −5e-5.)

Fix: price HALF A on `c_ratio = c_max / median|W|` directly —
`ROA_WEIGHT · clip(1 − c_ratio/ROA_C_TARGET, 0, 1)` — continuous through
zero, full slope on `0 < c_ratio < target`. The review proposed a raw-`c_max`
target; that would break scale invariance (`c_max` scales with `V`, every
other 5_7 term is scale-free, so inflating `V` would buy the whole term),
hence the ratio. Target calibrated on the panel, not guessed:

```
96051 (verified)  c_ratio 0.0221      340 final champ  0.0325
122232 (valid)            0.0274      340 gen-7        0.0497
ARE reference             0.0044      122350 champ    -0.0062
```

`ROA_C_TARGET = 0.02` — the term reaches 0 for 96051-class certificates.
Volume is still computed and reported per generation; it is just no longer a
price (quantised in 1/21⁴ steps, constant on half the domain). Requires
restarting any run launched before the fix.

## 9. Still open

- The rollout gate uses 4 fixed starts; a diverging trajectory from an
  untested start would pass. Certified-set sampling of starts is the upgrade.
- `SAT_U_TARGET = 1e3` is a choice, not a measurement of the physical
  cart-pole; if a real actuator bound is known, use it.
- The grid saturation max is taken over 21⁴ samples — a needle-thin
  high-`a/|b|` region between samples can hide, which is exactly what the
  rollout gate is the backstop for.
- Exact-stage sampling differences between runs (gen-7: 0 viol in 122340,
  656 here) deserve a look; budgets/seeds should be pinned per run.
- `u_required` is a max — gen-7's p99.9 is 339 vs a max of 1.76e5, so the
  signal can live in one grid point. If it proves noisy generation to
  generation, switch to p99.9 or a soft-max.
- Through `epsilon_b`, a genuine Artstein violation (a > 0 at b ≈ 0) surfaces
  as `u_required ~ 1e9`, so the saturation term partly re-prices Artstein
  rather than measuring independently. Accepted: a > 0 on {b = 0} is the
  `|u| = ∞` case of bounded-control infeasibility.
