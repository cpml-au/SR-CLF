# 4D Cart-Pole CLF Discovery — Session Handoff

Complete state of the verification/fitness work on this example, written so a
new session (human or AI) can continue without re-deriving anything.

---

## 1. What this project does

Genetic programming (flex/DEAP, `run.py`, config in `config.yaml`) searches for
a **Control Lyapunov Function** V(x) for the 4D cart-pole
(x1 = cart position, x2 = cart velocity, x3 = pole angle from upright,
x4 = angular velocity). Fitness is computed by `Evaluate.py`
(`eval_MSE_breakdown`). Candidates are analyzed in the notebook
`4DCartPolePrime.ipynb`. Cluster jobs write `<jobid>_best_per_generation.jsonl`
(one record per generation: expression string with `a` constant placeholders,
`constants` list, fitness).

**Environments**: local runs use `~/miniforge3/envs/flex/bin/python`
(has flex, pygmo 2.19.7, sympy, scipy). `alpinex`/`alpine` have pygmo but no
flex. System python3 has sympy/scipy only.

**Dynamics conventions** (`SystemDynamics.py` numeric / `SystemDynamicsSR.py`
sympy): g=-10, L=2, d=1 (cart damping), m=1, M=5, with `Sx=-sin(x3)`,
`Cx=-cos(x3)`, `D=mL²(M+m(1-Cx²))`. Only input channel **index 1** is active:
G[1,1]=mL²/D, G[3,1]=-mLCx/D. Q=I, R=1e-4·I (so b'R⁻¹b = 1e4·b²).
Box: ±0.25 per axis. Grid: **21 points/axis (odd — the grid contains x=0 and
the x3=x4=0 plane; this is deliberate, don't change back to 20)** in both
`run.py` and notebook cell 2. The separate grid-Vdot decay rate is
`SHGO_DECAY_RATE = 0.0012`. The Artstein/bounded-input checks instead use the
constant margin `CLF_GAMMA1 = 1e-6` outside
`CLF_ORIGIN_EXCLUDE_RADIUS = 1.1e-3`. Barrier authority `CLF_RHO = 800`.

**Rebuilding a candidate from a jsonl record** (used by every analysis script):
replace `a` tokens in order with the constants, then sympify with
`{"sub":a-b, "div":a/b, "aq":lambda a,b: a/(1+b**2)**0.5, "mul","add","neg","pow"}`.
See `ground_truth_barrier.py` for a working example.

---

## 2. The theory (all derived, nothing hand-waved)

- V̇(x,u) = a(x) + b(x)·u with a = ∇V·f, b = ∇V·g (g = active G column).
- **Unbounded input CLF condition (Artstein/Sontag)**: b(x)=0 ⟹ a(x)<0 for
  all x≠0. Everything lives on the b=0 manifold; off it, control wins.
- **Bounded input, exact**: min_{|u|≤ρ} V̇ = a − ρ|b| (minimizer u* = −ρ·sign b,
  and b·sign(b)=|b| is where the minus sign comes from). So
  **a − ρ|b| < 0 ∀x is EXACTLY the CLF condition for control authority ρ**.
  Quadratic-effort variant: min_u [a + b·u + u'Ru/4ρ] = a − ρ·b'R⁻¹b.
  NEVER use signed b in the barrier (a − ρ·b is meaningless and rejects valid
  CLFs where b<0 — this bug happened once as `b_l11`).
- **Implemented strict numerical version on the punctured box**:
  a − ρ|b| + γ₁ < 0 with γ₁=`CLF_GAMMA1`. The former state-dependent
  +γ‖x‖² term is deliberately disabled in the Artstein/barrier checks; it
  remains only in the separate grid-Vdot performance target.
- **Sontag controller** (as implemented, `src/SymVVdot_Calculations.py`):
  λ = (√(a² + x'Qx·(b'R⁻¹b)²) + a)/(b'R⁻¹b + 1e-6), u = −R⁻¹b·λ. λ ≥ 0 always
  ⟹ **sign(u) = −sign(b)**. The 1e-6 regularizer creates small residual u near
  b≈0 (once caused a spurious coasting equilibrium canceling cart damping).
- **Non-minimum phase sign structure** (the physical must-have): a correct
  cart-pole CLF gives u with OPPOSITE orientations on the two diagonal slices —
  LQR: u goes − → + along the (x1,x2) diagonal and + → − along (x3,x4)
  (cart right ⟹ push right first to lean the pole, then travel left).
  A candidate with the SAME orientation on both planes fights itself and blows
  up. Requires ODD x1 coupling into ∂V/∂x2 or ∂V/∂x4 (like LQR's 2.32·x1·x2
  term); even powers (x1²) satisfy properness but can never steer the cart home.
- **Reference LQR quadratic** (passes the whole pipeline, final_mse ≈ 4.57 —
  the permanent acceptance test):
  `1.965366*x1**2 + 2.862562*x1*x2 - 13.368898*x1*x3 - 5.925124*x1*x4
   + 2.143945*x2**2 - 20.349648*x2*x3 - 8.970854*x2*x4 + 50.818966*x3**2
   + 43.681856*x3*x4 + 9.639298*x4**2`
  Its manifold margins: FD −0.0070, exact −0.0039; origin-probe ratio ≈ 51.

---

## 3. The exploit museum — every loophole GP found (and its detector)

GP will find the cheapest wall. Each family below was actually evolved:

| # | Exploit | Signature | Detector that kills it |
|---|---------|-----------|------------------------|
| 1 | **Flat valley** (positive SEMI-definite V; V ≡ V(0) on the plane x3=x4=0) | v_min ≈ 0 on a whole plane; a=b=0 there; closed loop: pole stabilizes, cart coasts (u never changes sign) | relative positivity `V−V(0) > 1e-4·‖x‖²` (grid + DE/SHGO `pd_eps`); constant γ₁ barrier margin; odd grid |
| 2 | **c\*-collapse** (flat direction touches box boundary ⟹ certified sublevel set empty ⟹ sublevel-gated checks see nothing) | c\* = 0 exactly for whole elite | graded c\* penalty on the RAW (unclamped) boundary min of V−V(0) |
| 3 | **Origin needle v1** (9.2e17 constant; V(0)=0 in a 1e-18-wide spike, V≈1 elsewhere ⟹ fake positive definiteness, V_eff has minimum at box corners ⟹ controller drives state AWAY) | V(0)=0 but V(1e-8)≈1; v_min ≈ box-min of V_eff | origin-consistency probe: ratio (V(x)−V(0))/‖x‖² at ±1e-4, ±1e-2 per axis; legit ≈ 50–60, needle ≈ 1e7–2e7; flag > `ORIGIN_PROBE_K` |
| 4 | **Origin needle v2** (same effect built by COMPOSING aq/exp of O(1–10) constants — constant caps don't help) | probe ratio ~2.4e7; FD-blind (fd_step 1e-5 ≫ needle width) | same probe; also the **exact symbolic** manifold check sees what FD cannot |
| 5 | **Even-power properness** (V proper via x1², sin(x2²) but b gets x1 only as x1² ⟹ u can't flip sign with cart position; wrong NMP orientation) | u same orientation on both diagonal slices; 100+ manifold violations | manifold checks catch the (real) Artstein violations; orientation probe possible (4 evals) but not implemented |
| 6 | **Knife edge** (a = 0 EXACTLY at b=0 points on the box boundary, implemented margin = γ₁ = 1e-6) | both manifold checks see the same positive constant-margin violation | constant γ₁ shift; "clean" requires a+γ₁≤0 on b=0 outside the origin ball |
| 7 | **Corner-hugging b=0 branch** (violation at x3=x4=−0.25 corner varying along x1; scan endpoint ⟹ no sign-change bracket on default x3/x4 scans) | old partial-axis manifold scan missed the true margin **+3.37** | `scan_axes=(0,1,2,3)` at both manifold call sites (NOW IN) |

**Meta-lessons**: (a) any check with a flat/step penalty becomes a wall GP
parks against — every penalty must be margin-graded (smooth in coefficients so
the per-individual constant optimizer `eval_MSE_and_tune_constants` can descend
it); (b) counts are resolution-dependent, margins are not — grade by margin;
(c) if fraud is priced cheaper than honesty, the population collapses into
fraud (gen-228 event in job 68188); (d) FD gradients (step 1e-5) are blind to
structure below ~1e-6 — the exact symbolic check exists for that.

---

## 4. The verification stack (files in `src/`, knobs in `Evaluate.py`)

Order of evaluation in `eval_MSE_breakdown`:

1. **Grid stage** (21⁴ mesh): symbolic structure penalty (nested exp/aq, missing
   vars); **proper band** = count of points with V outside [0.1, 50]×reference
   quadratic (origin excluded — ref(0)=0 makes the band [0,0] there);
   **relative positivity** V−V(0) ≤ 1e-4·‖x‖² count (origin excluded);
   **V̇ decay count** (min-norm V̇ ≥ −γ‖x‖², origin excluded); origin penalty
   1e6·V(0)²; gradient-magnitude targets; **origin probe** (see §3.3–4).
2. **DE/SHGO verifier** (`src/pygmo_counterexample_optimizers.py`,
   `src/numerical_point_verification.py`, and
   `src/shgo_numerical_verification.py`): DE currently uses two runs
   (160 gen × 320 pop) — min of V−V(0)−pd_eps·‖x‖² (positivity) and max of
   barrier margin a − ρ|b| + γ₁ outside the origin ball (FD gradients) — then
   120002 clipped cloud points around the two champions are point-checked with
   the current `local_samples=15000`. SHGO now uses the same positivity and
   barrier formulas with exact symbolic gradients. `pd_eps = 1e-4`. Penalty = root_penalty
   (knob `SHGO_ROOT_PENALTY`) + count + `SHGO_MARGIN_WEIGHT`·(max(margin,0) +
   max(−v_min,0)). **Stochastic**: `NUMERICAL_VERIFIER_RANDOM=True` ⟹ verdict
   is a per-seed lottery for marginal candidates (recovers ~12–50% of true
   margin; see §6 optimizer study). Barrier uses `b_l1` = |b| (NEVER `b_l11`).
3. **c\*/ROA**: c\* = max(boundary_min(V−V(0)), 0); penalty is quadratic in the
   shortfall of the RAW boundary min (graded through zero, capped ×10);
   roa_coverage counts rollout x0's outside {V−V(0) < c\*}.
4. **FD manifold check** (`src/b_manifold_check.py`): axis-parallel line scans
   + vectorized bisection solve {b=0} (FD gradients, batch); exact zeros at
   scan nodes kept (flat valleys); flags roots with a + γ₁ > 0 outside the
   origin ball.
   Deterministic, ~0.2 s (2 axes) / more with 4 axes. Gated by
   `MANIFOLD_MSE_GATE`; unchecked candidates get += GATE·4 (never below gate
   without passing), errored += GATE·5.
5. **Exact symbolic manifold check** (`src/b_manifold_check_exact.py`): same
   scans but a,b from `sympy.diff` — no fd_step error; catches needle-scale
   structure. Gated by `MANIFOLD_EXACT_MSE_GATE`. **Sympy diff/lambdify can
   take 10–60+ s on big trees and has NO internal timeout** — never run it
   ungated in the GP loop.
6. Rollouts exist (`ROLLOUT_ENABLED`) but have been off the whole time.

**Ground-truth adjudicator** (read-only tool, this dir):
`ground_truth_barrier.py` — exact ∇V, barrier margin on 4M random points +
Nelder–Mead polish of top-50, plus all-axes exact manifold cross-check. Used to
settle every shgo-vs-manifold dispute. Note it identifies the candidate by the
origin-probe signature; edit the jsonl list inside for new candidates. It is a
falsifier, not a certificate.

**The sound endgame** (not wired in): `src/formal_verification.py` already
writes SMT2 and calls **dReal** (δ-complete, handles sin/exp). Correct
architecture = falsifiers in the loop, dReal certificate on the final
candidate, counterexamples fed back (CEGIS). Nothing accepted without dReal
should ever be called "valid".

---

## 5. Fitness-shaping principles that ended the plateaus

Runs 67243/67347/67373 flatlined for 100–250 generations. Diagnoses, in order:

- `1000 + count` penalties are **staircases** — piecewise constant in the
  coefficients, zero gradient for the constant optimizer. Fix: add
  margin-graded terms (weight·max(margin,0)) to shgo AND manifold penalties.
- `c* = max(boundary_min, 0)` **clamps away the distance to feasibility** —
  a candidate at boundary_min −10 scored the same as −1e-9. Fix: grade on raw.
- Penalty **interference**: one huge flat penalty (c\* at 1e5, properness at
  1000/pt) can dominate everything and stop selection on other axes.
- **Pricing must rank fraud above honesty**: needle family at 635 outcompeted
  an honest near-CLF at 1100 ⟹ population collapse until re-priced.
- **Noise**: with random DE seeds, the flat shgo root penalty (was 1000) made
  the same individual bounce ±1250 across evaluations. `SHGO_ROOT_PENALTY` is
  now a knob (user set it to 10 — margin terms dominate; cliffs aligned).
- A **smooth margin mapping** was designed and approved:
  `m = margin + 1e-4 (safety); penalty = min(W·m²/(m+0.01), CAP)`, FD cap 6000
  (W=12000), exact cap 1000 (W=2000), zero for margin ≤ −1e-4 — quadratic near
  0 (noise-gentle), linear later, pushes candidates THROUGH zero instead of
  parking on the knife edge. **WARNING: this was lost in a cluster→local file
  overwrite (see §7) — the file currently has the old harsh linear
  clip(1e5·m, 3, 50000) helper. Re-apply if desired.**

The one unpulled lever, recommended repeatedly: **seed the population with the
LQR quadratic** (fitness 4.57 — inside the feasible set, which has open
interior in coefficient space). Every plateau since has been GP trying to
rediscover `-13.4·x1·x3 - 20.3·x2·x3 + ...` by random subtree ops. There is no
guarantee GP finds a valid CLF otherwise (only trivial ergodicity arguments);
seeding converts discovery into polishing.

---

## 6. Falsifier optimizer study (measured, job 67516 candidates)

Ground truth = exact-gradient sup. All at 80×80, box ±0.25:

- Across 15 candidates (paired seeds): zero-false-negative set =
  {**de1220** (best mean, 74% of true margin), de (70%), sade (65%)}.
  **bee_colony/pso/gwo produced false negatives** (reported no violation on
  provably-invalid candidates) — disqualified for gating use.
  **cmaes and xnes are broken here** (don't respect box bounds; exp terms
  explode; reported "margins" of 1e5–1e73) — never use.
- On a single hard candidate (10 seeds): every method recovers only 12–50% of
  the true margin at this budget — the falsifier UNDERESTIMATES; treat "shgo
  valid" as "found nothing this draw".
- `_make_algorithm` supports de/de1220/cmaes; `NUMERICAL_VERIFIER = "de1220"`
  is the best-tested honest upgrade from "de".

---

## 7. Current state of `Evaluate.py` at handoff (2026-07-10) — READ THIS

The working file was **overwritten by an older cluster copy at least twice**
during this work. As of handoff it contains:

- ✅ 2026-07-10 follow-up: Artstein/barrier checks use
  `a - CLF_RHO*|b| + CLF_GAMMA1` with `CLF_GAMMA1=1e-6`; the former
  `+SHGO_DECAY_RATE*||x||²` term is disabled there. DE and SHGO now use the
  same formula and relative-positivity margin. The punctured-domain radius is
  `CLF_ORIGIN_EXCLUDE_RADIUS=1.1e-3`; `SHGO_DECAY_RATE` remains only for the
  separate grid-Vdot performance check.
- ✅ scan_axes=(0,1,2,3) at BOTH manifold call sites (corner-branch fix, verified:
  pinned 68523 gen161 exact check finds 1254 violations, margin 3.3730)
- ✅ origin probe (K=5e4, penalty 1e5), relative positivity, graded raw-min c\*,
  shgo margin terms, `SHGO_ROOT_PENALTY = 10` (user's value), de1220 support,
  proper weight 1, `MANIFOLD_MSE_GATE = 10000`, `MANIFOLD_ROOT_PENALTY = 100`
- ⚠️ OLD linear margin helper `clip(1e5·m, 3, 50000)` — the approved smooth
  mapping (§5 last bullet) is NOT in the file
- ⚠️ `MANIFOLD_EXACT_MSE_GATE = np.inf` — exact symbolic check runs on EVERY
  candidate; measured ~20 s/candidate with all-axes scans ⟹ cluster
  generations will be enormously slower. Recommend a finite gate (e.g. 2000).
- `MANIFOLD_EXACT_PENALTY_MAX = 50000`

**Operational rules learned the hard way**:
1. **Restart the Jupyter kernel after ANY edit to Evaluate.py or src/** — stale
   kernels produced contradictory reports at least five times.
2. **Transfer direction**: local (rnv) → cluster. Never the reverse without
   diffing (`diff` local vs cluster Evaluate.py first).
3. `rnv2` is a stale copy of the repo; it does NOT have the new src modules.
4. jsonl files of running jobs grow — analyses of "the last best" are moving
   targets; pin the generation number when comparing numbers.
5. The acceptance test for ANY fitness change: LQR quadratic must stay < 1000
   with zero manifold/probe/positivity penalties (≈4.57), and the museum
   candidates (68188 needle, 68523 gen161 corner) must stay heavily penalized.

## 8. Key artifacts

- `ground_truth_barrier.py` (this dir) — exact-gradient adjudicator.
- `src/b_manifold_check.py`, `src/b_manifold_check_exact.py` — manifold checks.
- `src/formal_verification.py` — dReal wiring for the final certificate.
- Notable jsonl runs: 63818 (flat-valley families), 67243/67347/67373
  (staircase plateaus), 67516 (optimizer study), 68188 (needle collapse at
  gen 228), 68519 (first fully honest plateau), 68523 (corner monster gen 161).
- The reference quadratic + dynamics conventions in §1–2 are enough to rebuild
  every analysis in a fresh session.
