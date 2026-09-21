# 3-D cart-pole V3 — making the correct answer reachable

Fork of [`../3DCartPolerV2`](../3DCartPolerV2) (`src3DCartPoleV3`). V2 was
correct about *ranking* and useless in practice: run 122483 sat on its
generation-1 seed for 33 generations with the constants **bit-identical**,
because the correct answer was not in the search's reach.

## What V2 got wrong (and V3's gate now catches)

V2's acceptance test only checked that the ARE quadratic **ranked first**
among four fixed candidates. It passed. The run then went nowhere.
**Ranking is not reachability.** `gate_v3.py` tests three things, and a run is
not worth starting unless all three pass:

1. **RANKING** — the ARE scores best of the known panel
2. **BASIN** — perturbing the ARE raises the score smoothly, and ±5%/±10% stay
   well below a generic all-ones quadratic
3. **REACHABILITY** — refinement from a spread of starts gets within striking
   distance

## The three findings behind V3

### 1. The constant tuner could not express the answer

`grid_fitness.py` seeded with `uniform(-10, 10)` and mutated by **replacing**
with another `uniform(-10, 10)` draw. Replacement, not perturbation — so no
constant outside that box could ever exist, in any generation.

```
ARE coefficients outside [-10, 10]:
  3-D cart-pole   2/6    (24.84, 20.70)
  4-D cart-pole   4/10   (max 50.82)
  ball & beam     0/10                  <- reachable, which is why it certifies
```

Fixed: `SYMCLF_TUNER_CONST_RANGE` plus a multiplicative-perturbation mutation
that can hill-climb past the box.

### 2. The target sat on the edge of the positive-definite cone

With the inherited `Q = I, R = 1e-4`, the 3-D ARE has `lambda_min(P) = 0.0441`
and `cond(P) = 682`. **56% of ±5% coefficient perturbations lose positive
definiteness.** Those are not Lyapunov functions, so the fitness was *right* to
punish them — the needle-thin basin was correct behaviour, and no amount of
fitness reshaping could widen it. The target itself had to move.

`Q` and `R` only select **which** CLF the Riccati equation returns; every
solution is valid by construction. `Q = diag(1, 10, 100), R = 1e-1` gives
`lambda_min = 1.4065` and 1% PD loss at ±5%:

```
reference               F(ref)  F(+-2%)  F(+-5%)  F(+-10%)   F(all-ones)
Q=I,      R=1e-4          0.00    44.18  1606.97   3312.96        514.66
Q=(1,10,100), R=1e-1      0.00     8.18    33.72     71.17        433.84
```

**48× smaller at ±5%, monotone, and every near perturbation stays far below a
generic quadratic.** Coefficients are then rescaled to `max|c| = 25`; `V` and
`kV` are the same CLF, so scale is pure gauge, chosen to sit inside the tuner's
reach.

### 3. The SEA is random search; it needed a local step

35 individuals × 5 generations cannot resolve a direction in 6–10 dimensions to
a few percent. `refine_constants` adds central-difference Adam on the elite —
one batched evaluator call per step, no autodiff through the Pallas
interpreter (which has no JVP rule).

Two parameterisation bugs found and fixed while building it, both the same
class as the `[-10,10]` clip — *a parameterisation that cannot express the
target*:

- optimising `sign*exp(z)` **freezes every sign**, and the reference needs two
  negative coefficients, so from an all-positive start it was unreachable
- a single global finite-difference step cannot probe a coefficient near 25 and
  one near 0.1 at the same time; the step is now per-coordinate and scale-aware

### Also carried in: positivity as a magnitude

`v_violations` and `flat_violations` were the last **count** terms in the
objective — piecewise-constant, non-monotone, jumping by hundreds. Both are now
magnitudes. Measured honestly: this did **not** move the basin (the PD-cone
edge did), but it removes the last gradient-free terms.

## Gate result

```
1. RANKING        ARE 0.000  <  gen 176  359.031  <  gen 89 358.363 ... PASS
2. BASIN          0 -> 9.07 -> 34.20 -> 70.31   (all-ones 433.84) ..... PASS
3. REACHABILITY   best of 8 refined starts = 81.557  (threshold 108.5)  PASS
```

## What is still not solved — read this before believing a run

The path from a generic start to the ARE is **rugged**, not monotone: a line
scan from all-ones shows spikes at t=0.35 (604.6) and t=0.65 (525.4), with a
clean descent only over the final 25%. A single cold start plateaus around
330; the best of 8 reaches 81.6; **the ARE itself is 0.000.**

So V3 does not "solve" the search. What it does is give it a mechanism it never
had — V1 and V2 could not get below ~370 from anywhere. Whether GP's structural
diversity plus elite refinement is enough to close the last gap is an empirical
question this run answers.

## Running

```bash
python gate_v3.py          # must PASS before starting
sbatch job3DCartPoleV3.slurm
```

| variable | default | what it does |
|---|---|---|
| `SYMCLF_TUNER_CONST_RANGE` | `30` | SEA sampling box for constants |
| `SYMCLF_TUNER_CONST_PERTURB_FRAC` | `0.5` | fraction of mutations that perturb instead of replace |
| `SYMCLF_TUNER_REFINE_STEPS` | `40` | Adam steps on the elite |
| `SYMCLF_TUNER_REFINE_TOP_K` | `4` | how many candidates per group get refined |
| `SYMCLF_V3_POSITIVITY_WEIGHT` | `300` | weight on the `W<=0` magnitude |

Inherited from V2: matched box, `std(log q)` shape term, boundary-pinned `rho`,
`vdot` tautology off.
