# 6-D quadrotor CLF — CMA-ES on the quadratic template with the V8 fitness

Self-contained folder: copy it anywhere (`tar czf clf6d.tgz 6D/`), install `requirements.txt`,
run. It imports **nothing** from the 4-D tree — only numpy, scipy, numba, sympy, pycma
(mpmath optional). No JAX, no Ray, no flex/DEAP.

## 1. The system

Indoor micro quadrotor, angular subsystem (Bouabdallah et al. 2004), as a **control-affine**
system `x' = f(x) + G(x) u` with `u = (U1, U2, U3)` free:

```
f = ( x2,                                   G = 6x3, the coefficients of U1..U3:
      x4 x6 (Iy-Iz)/Ix - (JR/Ix) x4 Omega,        x2' += (l/Ix) U1 = 0.5 U1
      x4,                                         x4' += (l/Iy) U2 = 0.5 U2
      x2 x6 (Iz-Ix)/Iy + (JR/Iy) x2 Omega,        x6' += (l/Iz) U3 = 0.2 U3
      x6,
      x2 x4 (Ix-Iy)/Iz = 0 )                 Omega = sin(x2) cos(x4)
Ix = Iy = 2, Iz = 5, l = 1, JR = 1,  D = {|x_i| <= 3},  Q = I6, R = 1e-4 I3
```

The paper's own controller (`U1 = -(Ix/l)(x1-x1d) - k1 x2`, …) is **not** used: the search looks
for a CLF and the input comes from Sontag's formula, exactly as in the 4-D examples.

`systems/quad6d.py` holds this; `systems/cartpole4d.py` holds the 4-D cart-pole of
`examples/4DCartPolerV8`, which is what the acceptance test runs.

## 2. What is optimised

CMA-ES over the coefficients of the quadratic (LQR) template

```
V(x) = sum_{i<=j} a_ij x_i x_j          21 coefficients   (SYMCLF6D_TEMPLATE=full, default)
V(x) = the 9 monomials the ARE uses     9  coefficients   (SYMCLF6D_TEMPLATE=are)
```

`x0 = ones` — **no LQR/ARE initialisation** (standing rule). The ARE solution is only ever used
as a *reference candidate* in reports and for the `are` sparsity pattern.

Recipe = job 146800's: pycma `fmin2`, BIPOP restarts, popsize 10, sigma0 1.0, per-coordinate
`stds = 0.3 max(|x0|, 0.1)`, fixed seed, budget in evaluations (3000), keep-best.

## 3. The fitness — in sequence

One evaluation of (expression, constants) is `core.rule.fitness_fixed_constants`, the port of
`src4DCartPoleV8/fitness.py:fitness_fixed_constants` (V8 = "one check" mode):

**S1 — lattice stage** (`core/lattice.py`, training grid 7^6 = 117,649 points, numba):
V on the grid → W = V − V(0); gradients by the kernel's finite differences (numpy stencil,
edge_order 2, per-axis spacing); `a = gradV·f`, `b_k = gradV·G[:,k]`, `beta = b'R⁻¹b`,
Sontag `lambda`, `vdot`. From these:
* `cv_grid  = min( min W/|x|², min (x·gradV)/(2|x|²) )` over the grid
* `kappa_grid = −max (a − rho·‖b‖)/W` over the W > 0 points
* `boundary_min` = min W on the box faces, `w_scale` = median |W|,
  `c_max = boundary_min`, `certified volume` = fraction of the grid with W < c_max
* `invalid` = 1e6 per non-finite field, `symbolic` = 1e6 per ignored state / nested exp,aq
* **grid ROA term** = 500·clip(1 − boundary_min/(w_scale·0.004)) if the grid certifies, else
  500·(1 + log1p(relu(−kappa_grid))) + 1000·relu(1 − cv_grid/1e-4), plus the three hinges
  10@0.05 (c_v), 30@0.05 (kappa), 20@0.02 (volume)
* `pre_exact = symbolic + invalid + grid ROA`

**S1b — dead rule** (V8's D4): symbolic ≥ 1e6, invalid ≥ 1e6 or non-finite pre_exact →
`fitness = pre_exact + extra`, no check (a structurally dead candidate is never checked).

**S2 — the DET4 check** (`core/det4.py`, scan lattice 1,365,707 points, numba + scipy SLSQP):
the referee's algorithm —
1. scan the lattice for W, a, b, x·gradV;
2. **PD stage**: `c_v* = min( W/|x|², (x·gradV)/(2|x|²) )`, lattice seeds polished by SLSQP and
   Nelder-Mead;
3. **manifold ascent**: `max a/W` subject to `b = 0` (m equality constraints), each endpoint
   Gauss-Newton projected onto {b = 0} and accepted on the normalised gate
   `rho‖b‖ <= 1e-6 |a|` (V7's PROJECT_ASCENT repair);
4. **origin ray**: the origin's worst manifold direction from the linearisation
   (`max gen-eig of (N'(PA+A'P)N, N'PN)`, `N = null(B'P)`), ±r·v projected — insurance against
   the origin blind spot;
5. **rate stage**: `max (a − rho‖b‖)/W` by SLSQP (split on the sign of each b_k) and NM;
6. `kappa* = −max over every considered point`.
`pd_valid = c_v* > 1e-4 + 1e-6`; a non-PD candidate is charged `kappa* = −1e6` (option B gate).

**S3 — price and fitness** (`core/rule.py`, V8 arithmetic unchanged):
```
certified = c_v* > 1e-4 and kappa* > 0                    (V8 D1: the DET4 certificate alone)
ROA_RULE  = 500*clip(1 - boundary_min/(w_scale*0.004))    if certified
          = 500*(1 + log1p(relu(-kappa*))) + 1000*relu(1 - c_v*/1e-4)   otherwise
            + 10*clip((0.05-c_v*)/0.05) + 30*clip((0.05-kappa*)/0.05)
            + 20*clip((0.02-volume)/0.02)
price     = ROA_RULE - (the grid's ROA term)
extra     = 50 per nested exp/aq/trig + 0.005 * tree nodes      (125 nodes -> 0.625)
fitness   = pre_exact + price + extra
```

### The one change the 3 inputs force
`b = gradV'G` is a **vector**. The bounded-input optimum `min_u (a + b·u)` is
`a − rho·Σ_k |b_k|` for V8's per-input clip `|u_i| <= rho` (**default, `SYMCLF6D_B_NORM=l1`**)
and `a − rho·‖b‖₂` for a round bound (`l2`). For a single input both are the 4-D formula
`a − rho|b|`, so the 4-D acceptance is unaffected. The manifold `{b = 0}` is m equations, hence
the m SLSQP equality constraints, the Gauss-Newton projection and `null(B'P)` above.

## 4. Acceptance (run it first: `python smoke.py`)

The same code in 4-D cart-pole mode against the published 4-D numbers:

| check | this folder | published (V8 / referee) | diff |
|---|---|---|---|
| fitness, job 146800's certified quadratic | 5.972398158733 | 5.972398158733 | +3.2e-13 |
| fitness at the tuner x0 | 1331.043152952256 | 1331.0431529522557 | 0.0 |
| c_v* | 0.0216130092063 | 0.0216130092063 | +3.3e-14 |
| kappa* | 0.0519077920849 | 0.0519077920849 | −2.9e-14 |

and in 6-D, `core/det4.py` against `audit.py` (an independent sympy/scipy implementation):
c_v* 2.0e-14, kappa* 4.0e-15, with a 60-digit mpmath re-check of the worst point.

Reference points in 6-D (box ±3, rho 1000): the **ARE quadratic certifies** —
c_v* 0.019996, kappa* 1.90693, fitness 26.06 (hinges: c_v 6.00 + volume 19.43); the all-ones
start scores 530.6 (kappa* ≈ 0, not certified).

## 5. Running it

```bash
source env.sh                        # every knob, with the V8 defaults
python smoke.py                      # 11 checks, ~75 s
python cma_run.py --budget 3000 --popsize 10 --workers 10 --threads 6 --seed 1
python audit.py --best runs/cma_quad6d_full_s1.jsonl      # referee on the best
sbatch slurm/cma_cpu.slurm           # a CPU node: 3 seeds concurrently + audits
```
Every evaluation is one json line in `runs/*.jsonl` (fitness, certified, kappa, c_v, volume,
coefficients, wall time); the last line holds `best`, `best_c`, `certified_at`.

## 6. Hardware mapping (the new cluster)

| node | cores | layout used | why |
|---|---|---|---|
| `cn-*` (2×EPYC 9654/9655) | 192 | 3 seeds × 10 workers × 6 threads = 180 | one worker per CMA population member; ~6 threads saturates one evaluation |
| `gn-1003/1004` (2×EPYC 9575F) | 128 | 2 seeds × 10 workers × 6 threads = 120 | same, GPUs idle |
| `gn-1001/1002` (2×EPYC 9354) | 64 | 1 seed × 10 workers × 6 threads = 60 | same |

Cost measured on the source cluster's login node (older Xeon): **~2.0 s per evaluation** at 8
threads (scan 0.33 s, seeds 0.22 s, PD 0.40 s, SLSQP 1.03 s), so a 3000-evaluation seed is
~300 CMA iterations ≈ 10-20 min wall with a worker per population member. Expect faster on
EPYC Genoa at 3.7 GHz.

### Why no GPU
The 4-D pipeline used the GPU for one thing only — the training-grid pass. Job 147029's
acceptance measured that GPU lattice at **1.13 s against 0.48 s for the same scan on 9 CPU
threads**, and it was not bit-exact (c_v* differed by 3.2e-5 on one candidate), so V9 shipped
with `SYMCLF_DET4_GPU_LATTICE=0`. Here the grid is 7^6 = 117,649 points (0.6× the 4-D grid) and
the check is CPU-bound SLSQP/Nelder-Mead; a GPU port would add a dependency and lose exactness
for no gain. The `gn` nodes are therefore used as CPU nodes. Say the word and a JAX lattice path
can be added behind a switch.

## 7. Files

| file | what |
|---|---|
| `systems/quad6d.py`, `systems/cartpole4d.py` | f, G, Q, R (sympy) + box |
| `core/config.py` | every knob, from the environment |
| `core/encode.py` | GP expression → bytecode (verbatim from V8) |
| `core/evalnb.py` | numba value/gradient/Hessian interpreters, dynamics compiled from sympy, vector b |
| `core/geometry.py` | the referee's scan lattice (vendored) |
| `core/slsqp_direct.py` | scipy's SLSQP C core driven directly (vendored) |
| `core/det4.py` | the DET4 check for (n states, m inputs) |
| `core/lattice.py` | the training-grid stage (det4-mode terms) |
| `core/rule.py` | the price rule + fitness at fixed constants |
| `core/templates.py` | the quadratic templates and the ARE reference |
| `cma_run.py` | the CMA-ES driver (146800 recipe) with a worker pool |
| `audit.py` | independent sympy/scipy referee + 60-digit re-check |
| `smoke.py` | the 11-check self-test (acceptance + 6-D + plumbing) |
| `env.sh`, `slurm/`, `requirements.txt` | knobs, job scripts, dependencies |

## 8. Open switches (defaults are the ones described above)

* `SYMCLF6D_B_NORM=l1|l2` — which actuator bound defines the DET4 rate (see §3).
* `SYMCLF6D_TEMPLATE=full|are` — 21 or 9 coefficients.
* `SYMCLF6D_GRID_POINTS`, `SYMCLF6D_LATTICE_*` — grid 7^6 and lattice 7/3/801 (1,365,707 points).
* hinge targets `SYMCLF_DET4_PEN_*` — the 4-D values; the 4-D study (job 147053) raised
  kappa to 1.0 and volume to 0.3 to buy rate instead of volume.
