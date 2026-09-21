# 4DCartPolerV9 = V8 + a faster DET4 check, length 90, 6 threads (2026-09-16, the user's order)

Copy of examples/4DCartPolerV8 (engine copy src4DCartPoleV9, DET4 copy examples/DET4Fast4D_V9). Changes:
* SYMCLF_MAX_TREE_LENGTH 90 (V8: 150); NUMBA_NUM_THREADS 6 per actor (V8: 3), to measure.
* det4_fast (DET4Fast4D_V9), three switches, off by default in the code, on in job4DCartPoleV9.slurm:
  - SYMCLF_DET4_GPU_LATTICE: W, a, b, x.gradV at the referee's 2,610,397 lattice points on the GPU
    (lattice_gpu.py = V7's Pallas ab kernel verbatim, 4-D cart-pole fields inline) and the seed ranking of the
    referee's three stages plus the radial PD seeds on the device (jax.lax.top_k); the lattice extrema used as
    the polishes' initial bests are re-evaluated with the CPU evaluator, so every reported number comes from the
    unchanged numba path. Off automatically without a GPU backend (the login-node smoke).
  - SYMCLF_DET4_TOPK_SEEDS (CPU path): argpartition top-k instead of six full argsorts of 2.6M values; the same
    seed sets except exact ties at rank k (the referee's argsort breaks those arbitrarily too).
  - SYMCLF_DET4_PRECOMPUTED_FIELDS (CPU path): f and g of the dynamics computed once per lattice; bit-identical.
  Measured before (V8, 3 threads, per candidate): scan 0.79-0.92 s, seeds 0.55 s, polishes 0.45 s, total 1.85 s.
  Acceptance: accept/accept_det4.slurm (one GPU): V8 vs V9 on 32 candidates (29 champions, 146800, ARE,
  all-ones), kappa*, c_v*, worst and PD points compared bit for bit, stage times at 3 and 6 threads.
* Acceptance job 147029 (2026-09-16 21:09, accept/accept_result.txt): kappa* and the worst point bit-identical
  on 32/32; c_v* differs on 13/32 with the GPU lattice on: 11 at the rounding level (1e-14..1e-17), 144596 g18
  at 7.7e-11, and 144596 g22 at +3.2e-5 -- the device-ranked radial PD seeds MISSED the minimum the CPU ranking
  finds (W = V - V(0) with cancellation on the GPU vs the cancellation-free _wdelta on the CPU). The GPU scan is
  also slow (1.13 s: the ab kernel materialises its 32x5 stack per chunk) against 0.70 / 0.48 s for the CPU scan
  at 6 / 9 threads. Nine concurrent processes (the actor situation), seconds per candidate: V8 2.32 (3 threads),
  1.58 (6), 1.50 (9); V9-GPU 1.57 / 1.41 / 1.40. DECISION: SYMCLF_DET4_GPU_LATTICE=0 in the V9 slurm; V9 runs the
  CPU path (top-k seeds + precomputed fields, bit-identical on the login test) with 6 threads; second acceptance
  (accept2.slurm, job 147041) verifies the CPU path on the 32 candidates at 6 and 9 threads.
* Acceptance job 147042 (2026-09-16 21:14, accept/accept2_result.txt, the CPU path GPU_LATTICE=0 TOPK=1 PRE=1):
  kappa*, c_v* (W-stage, radial, min) and the worst point bit-identical 32/32 at 6 and at 9 threads; isolated
  1.083 s vs V8 1.639 s (6 threads), 0.798 vs 1.349 (9); under 9 concurrent processes 0.99 s (6) / 0.94 s (9)
  per candidate vs V8 1.58 / 1.50 (and 2.32 at V8's 3 threads). 9 threads buys 5 % over 6 -> 6 threads.
  BUT the seed SETS differed on 5/32 candidates (n_polish_seeds, e.g. 146680_g26 131 vs 141) and one PD-point
  coordinate (144596_g22) at 1e-8 with the same c_v: V8 ranks with numpy's unstable full argsort, V9's
  argpartition breaks exact ties (equal finite scores, e.g. mirror lattice points) differently.
* Tie fallback (det4_fast.py, backup .bak_20260916_ties): _smallest_k_cpu / _largest_k_cpu redo V8's full
  argsort whenever equal FINITE values sit inside the selected k or straddle rank k (non-finite ties do not
  count: the callers drop non-finite seeds). The seeds are then V8's by construction; the partial selection
  only pays off on candidates without ties. _TOPK_FALLBACKS counts the fallbacks (accept_det4.py records it).
  Kernel profile (147042): the Pallas kernel is 0.7 ms per 262144-point chunk (7 ms per lattice); the 0.9 s of
  the GPU scan is the per-chunk retracing of pallas_call on the Python side -- fixable, but the W-cancellation
  inexactness (147029) stands, so the GPU lattice stays off.
* Login-node reruns are NOT a reference for bit-exactness: the login node (Xeon E5-2680 v3, AVX2) and s02n01
  generate different numba code; kappa* differs at 1e-16..5e-13 and mirror seeds are picked on 5/32 even for
  the unchanged V8 ranking. Compare on the same node type (accept3.slurm, job 147047: fallback on / top-k off
  vs the s02n01 V8 reference).
* Acceptance job 147047 (2026-09-16 21:35, accept/accept3_result.txt, s02n01 = AMD EPYC 7302, 6 threads, vs the
  s02n01 V8 reference of 147029): V9 with the tie fallback -> 0 differing non-timing fields on 32/32 (kappa*,
  c_v*, worst and PD points, seed and ascent counts); V9 with TOPK_SEEDS=0 -> also 0. The fallback fires on
  32/32 candidates (323 of ~448 selections: exact ties are structural -- trees that ignore a variable or square
  a coordinate give equal values at many lattice points), so both settings cost the same: 1.570 s per
  candidate vs V8 1.639 s. NET: under bit-exactness the CPU tricks are worth 4 %; the V9 det4 gain is the
  thread doubling (2.32 -> 1.58 s per candidate under 9 processes) plus the length cap 90. The job runs
  TOPK_SEEDS=1 (fallback, proven) + PRECOMPUTED_FIELDS=1 + GPU_LATTICE=0.
* Smoke with the job env sourced (smoke_v9_ties_env_login.log): 16/16 PASS, 248 s.
* SUBMITTED 2026-09-16 21:43 as job 147049 -> died at 1:32 ("SYMCLF_MAX_TREE_LENGTH: unbound variable": the V9
  echo used it 4 lines before its export, under set -u; echo moved) -> job 147050 on s02n01 21:47 (`sbatch -w
  s02n01 job4DCartPoleV9.slurm`; the slurm still
  says s02n02, taken by V8 job 146883): length 90, 6 threads, CMA gate 3000, 3 structures per generation,
  2500 evals each, 18 refine elites.
* 2026-09-17 (the user): job4DCartPoleV9_2node.slurm -- TWO q32g nodes (6 GPUs, 18 actors, node-local env on both
  nodes, Ray head on node 1 + worker on node 2 with a registration wait), CMA on 6 structures per generation
  (popsize 18), and the fitness knobs from the LQR study (~/symclf_tmp/lqr_roa/README.md): kappa hinge 100 @ 1.0
  (LQR's own certified kappa* is 0.72; the CMA quadratics reach 0.8-1.0) and volume hinge 100 @ 0.3. Sontag kept.
  Submitted as job 147113 (15:58); V8 job 146883 cancelled at GEN 68 (7.949, uncertified) to free s02n02.
* 2026-09-18 (the user, after the LQR-vs-CLF study ~/symclf_tmp/lqr_roa/README.md): job4DCartPoleV9_2node.slurm
  (backup .bak_20260918): volume hinge 100 @ 0.3 turned OFF (lines kept, commented) -> the V8 default 20 @ 0.02;
  CMA budget 2500 -> 4000 per structure, count stays 6; SYMCLF_PREFLIGHT=0 by default; kappa hinge 100 @ 1.0 kept.
  New diag_gnorm.py (+ diag_gnorm.slurm, on demand): the |x|^2-referenced numbers (g_norm, gamma, kappa_x,
  lambda_max) of every generation champion -- diagnostics only, NOT priced; run at the end of the 2-node job.
  The controller fix in src/SymVVdot_Calculations.py (the user's lambda_x) committed on main_rnv (20b2cbb).
  Submitted as job 147632 (pending: s02n01 has 2 GPUs held by another user's job for ~43 h).
* Not done yet: the expression-specialized compiled evaluator for the CMA stage and the refine (the polishes'
  share); a cancellation-free (c, d) lattice kernel would make the GPU path exact.

# 4DCartPolerV8 = ONE check (2026-09-15, the user's decisions D1-D5)

Copy of examples/4DCartPolerV7 (logs excluded), engine copy src4DCartPoleV8, DET4 copy examples/DET4Fast4D_V8;
built and smoked 2026-09-15 (18/18, 430 s on the login node) before the cut below.

Per generation (src4DCartPoleV8/ray_fitness.persistent_actor_mapper):
* S0 structure: trees with >= SYMCLF_MAX_TREE_LENGTH nodes (150 in the slurm; engine default 400) -> 1e8.
* S1 constants: the lattice SEA + Adam on the 21^4 grid rule (grid_fitness, byte-identical kernel); the grid
  pass also gives invalid/symbolic, boundary_min, w_scale, certified volume (D2: the level anchors of the price).
* S2 THE CHECK: det4_fast (examples/DET4Fast4D_V8 = the referee's DET4 with the V7 repairs, projection +
  origin ray + normalised gate) on every live tree, once per distinct (expression, constants), cached across
  generations. certified = c_v* > pd_eps and kappa* > kappa_min (D1: no root verdict).
  price = ROA_RULE(c_v*, kappa*, boundary_min, w_scale, volume, certified) - grid ROA term;
  fitness = pre_exact + price + extra  (V7's arithmetic: the grid ROA term cancels).
* S3 refine: refine_constants_det4 (portfolio Adam 0.02 / 0.1 + golden, 100 det4_fast evaluations; D3: the
  level gradient from 2n+1 grid rows only when certified) on the 18 best distinct by fitness (D5), scheduled
  one per work item (2 per actor). Improved -> constants, grid numbers and price replaced.
* D4 (SYMCLF_DET4_SKIP_DEAD=1): a structurally dead tree (symbolic >= 1e6: missing state / nested exp,aq;
  invalid V; non-finite grid score) keeps its grid price and skips the check.
* END: the CPU referee (src/srcCPU det4_check_cpu, sympy) on the champion, report only.
* S2b (2026-09-15, the user's go): CMA-ES on ONE elite per mapper call (ray_fitness.cma_refine_elite): the best
  new structure not yet CMA-refined in the run; pycma fmin2 with BIPOP restarts, sigma0 1, per-coordinate stds
  0.3 max(|c0|, 0.1), seed SYMCLF_CMA_ELITE_SEED (1), popsize SYMCLF_CMA_ELITE_POPSIZE (0 = one evaluation per
  actor per CMA generation), SYMCLF_CMA_ELITE_EVALS evaluations (2500 in the slurm, ~11-12 min of wall time;
  0 = off). Objective = fitness.fitness_fixed_constants on the actors (the V8 fitness at given constants).
  Keep-best: the elite's constants, grid numbers and price are replaced only on improvement. Recipe and
  motivation: job 146800 (examples/4DCartPolerV7Gurobi) certified the quadratic template with this recipe
  after 1910 evaluations, which the lattice tuner and the 100-evaluation refine never achieved.

Removed with their code (src4DCartPoleV8): artstein_v3 (cheap chord screen), b_manifold_check_gpu3 /
b_manifold_check_gpu / b_manifold_batch_gpu4 (GPU root check), pd_check_v3 (GPU5 PD), grad_norm_v3,
jax_candidate, runtime_exact_candidate; in fitness.py the PD attach, normalised margins, cert penalty, rollout,
gate-2/escalation/unscreenable pricing; in ray_fitness.py the exact waves/capacity and the Ray result cache.
Kept byte-identical (verified by AST extraction): _det4_penalty_terms, _det4_gate, _det4_knobs,
_symbolic_with_constants, _rule_and_gradient, _lattice_level, refine_constants_det4, _det4_certified,
_roa_rule, the OOM helpers, the tuner group (_tune_group, _refine_elite, ...), _evaluate_pre_exact_valid_batch.
Changed: _det4_fast_numbers (default folder DET4Fast4D_V8), _det4_numbers (det4_fast only), _exact_cache_key
(expression + constants + box), fitness_pre_exact (needs_det4, no gate-2 fields, D4 dead flag, the length switch).
New: _det4_price_from_entry, _price_entry, fitness_finish_det4, fitness_finish_refine.

Verified (job 146858 on s02n01, unit_v7_vs_v8.slurm -> unit_v7_vs_v8_result.txt): the 29 distinct champions of
jobs 144596 and 146680 get bit-identical prices, fitness, c_v* and kappa* from V7's cheap stage and V8's check;
refine_constants_det4 (12 evaluations, gens 28 and 66 of 144596) gives identical trajectories and constants.
Smoke (smoke_v8b_login.log): 14/14 PASS, 234 s on the login node.
CMA stage unit (unit_cma_v8.slurm -> unit_cma_<job>.log): the fixed-constants fitness of the 146800 quadratic
reproduces 5.972398158733319; cma_refine_elite replays job 146860's first 30 evaluations.

Not bit-identical to V7 by design: (i) a DET4-certified candidate is "certified" without the root verdict (D1);
(ii) dead trees keep the grid price (D4); (iii) the refine of an improved elite reuses its own det4_fast numbers
instead of recomputing them (deterministic). Everything else: the same numbers and the same arithmetic.

## Ray head failures and the node-local environment (2026-09-16)

Symptom (jobs 146795, 146879, 146880, 146896, 146947 attempt 1): the job hangs or dies at "Starting Ray head",
run.py never starts (up to 25 min lost). Cause: /home is an autofs NFS mount and was slow (fe1: `import numba`
21 s; s02n01: 217 s to warm 75 Ray modules). Ray's head start launches six Python helpers; the raylet gives its
agents a short window to report their ports (runtime-env agent port file ~20 s, dashboard agent 100 s), aborts
when they miss it (raylet.err: "Timed out waiting for file .../runtime_env_agent_port_*" or
"WaitForDashboardAgentPorts"), and `ray start` needs ~10 more min to notice. Read the node's Ray logs with
`srun --jobid=<job> --overlap -n1 cat /tmp/ray/session_latest/logs/raylet.err` while the job is alive
(/tmp is per job).

Fixes in job4DCartPoleV8.slurm (backups .bak_20260915_ray, .bak_20260916_retry, .bak_20260916_localenv):
1. the head start gets up to 3 attempts; an attempt ends on an active node in `ray status`, on the head step
   exiting, or at 12 min; failed attempts copy the Ray logs to ray_logs_<job>_<n>/ and force-stop Ray;
2. the Ray helper imports (dashboard head/agent, runtime-env agent, log monitor, client server, monitor) are
   warmed on the head node right before `ray start`; the agent/raylet timeouts are raised to 10 min;
3. the environment runs from the node's local disk: ~/flex.tar.gz = `conda-pack -n flex
   --ignore-editable-packages --ignore-missing-files` (one file, one sequential read), extracted to
   /tmp/flex_<job>, activated, `conda-unpack`; removed by the cleanup trap. The project code and the editable
   flex package stay on /home. Without the tarball the /home env is used. Re-pack after any change to the env
   (`conda-pack -n flex -o ~/flex.tar.gz -f --ignore-editable-packages --ignore-missing-files --n-threads 8`).
4. the preflight smoke is switchable: `sbatch --export=ALL,SYMCLF_PREFLIGHT=0 ...` skips it.

Runs are not bit-reproducible: flex's random generators are unseeded, the tuner seed includes the process id,
and Ray's work stealing decides which actor tunes which batch (the CMA seed is fixed).

# 4DCartPolerV7, second part (2026-09-13): the DET4 search repaired

Found while answering "trend of margin max / smaller sublevel set" on job 144596: for 15 of its 16
champions the worst DET4 point is the origin (kappa* == the max generalised eigenvalue of
(N'(PA+A'P)N, N'PN), N = null(B'P), P = Hessian(V)(0)/2, to 3 digits) -> no sublevel set {W < c}
is valid for any of them; the run shrank one positive eigenvalue by inflating P along it. Two
champions were under-priced: gen 71 (DET4Fast4D -0.063, referee -0.178) and gen 100, champion from
gen 100 on (both codes -0.0036; exact origin ratio +1.0157, 50-digit verified: W 1.0155e-4,
a 1.0314e-4, b 7e-20 at x = (-0.00702, -0.00713, 5.6e-7, 9.2e-8)).
Root cause (const_opt/det4_fixed_test.py): the manifold ascent FINDS the violation (gen 71 seed 0
ends at a/W = +0.1777 in DET4Fast4D too) and DISCARDS it on |b| <= 1e-10 absolute, a razor edge
near the origin (rho|b|/W = 0.018 already at |b| = 1e-10, W ~ 5e-6). The numba SLSQP == scipy
SLSQP on the same evaluator (144/144 seeds); the referee's sympy evaluator rounds differently, so a
different subset passes the edge (17 vs 28 of 144 accepted). Same effect across CPUs (gen 22: job
actor -0.1296, login node -0.1309, referee -0.1301).

Repair, in examples/DET4Fast4D_V7 (a copy; the running V6 job's DET4Fast4D is untouched) and in
the referee src/srcCPU/det4_check_cpu.py (new kwargs, default False = byte-identical behaviour):
* SYMCLF_DET4_PROJECT_ASCENT=1 (ON in V7): every ascent endpoint x is Newton-projected onto b = 0,
      x <- x - b(x) grad b(x) / |grad b(x)|^2   (<= 60 steps, stop at |b| <= 1e-15 max(1,|a|)),
  before the |b| <= b_tol test. 144/144 endpoints accepted on all 16 champions; kappa* <= referee
  and <= the origin ratio on 16/16 (gen 71 -0.17774, gen 100 -1.01565, gen 22 -0.13523).
* SYMCLF_DET4_ORIGIN_SEEDS=1 (OFF in V7 until the user decides): P = Hessian(V)(0)/2 from the
  bytecode's dual numbers, A = Jf(0), B = g(0) from the compiled fields, N = null(B'P), v = the
  eigenvector of the max generalised eigenvalue of (N'(PA+A'P)N, N'PN); the 8 points +-r v,
  r in {1e-3, 1e-2, 3e-2, 1e-1}, projected onto b = 0, are added to the considered points. Exact for
  the origin class (the ratio is scale-free along the ray). Cost: one 4x4 eigenproblem + 8
  projections (~1 ms). The projected ascent alone already recovered gen 100.
The V7 smoke passes the same two flags to the referee (in step).

### Acceptance gate normalised (2026-09-13, the user's call)
The ascent's historical gate |b| <= 1e-10 (absolute) is the wrong quantity: on job 144596's gens 71
and 100 it passed endpoints with rho|b|/|a| up to 289 (the gen-100 winner, a/W = +1.0157, was IN the
accepted set and was then priced with its residual, which made its ratio negative) and rejected 127
harmless ones. With SYMCLF_DET4_PROJECT_ASCENT=1 the gate is now
    rho |b(x)| <= B_REL_TOL |a(x)|,   B_REL_TOL = 1e-6 (SYMCLF_DET4_B_REL_TOL),
i.e. the control term moves the ratio by less than one part in 10^6; scale-free. The referee gates
with the largest rho of its ladder (manifold_ascent_rate(gate_rho=max(rhos))). Defaults (switch off,
gate_rho None) keep the historical absolute test byte-identical. After projection every endpoint
clears the new gate by ~1e6, so no number changes on the 16 champions (unit_gate_v7.log). The ratio
stage prices any residual exactly, (a - rho|b|)/W, so the gate is a selection gate, never a
correctness guard: a wrong gate can lose a point, never accept a false violation.
Origin ray: ON in V7 from 2026-09-13 (SYMCLF_DET4_ORIGIN_SEEDS=1, ~1 ms per candidate).

 Re-pricing of the 16 champions:
examples/DET4Fast/const_opt/reprice_144596.log.

# 4DCartPolerV7 = 4DCartPolerV6 + the constant-tuning study's A, B, C, D (2026-09-13)

Decided by the user after examples/DET4Fast/const_opt (probe, 13-optimizer shootout, long runs,
lattice-vs-exact landscape, 90-candidate elite pool). Engine: src4DCartPoleV7 (copy of
src4DCartPoleV6; the running job 144596 keeps its own engine untouched). Switches off by default.

* A. `SYMCLF_DET4_REFINE_LATTICE_ONLY_CERTIFIED=1`: the elite refine calls the GPU lattice only
  when the candidate is DET4-certified (its level gradient is used only then); uncertified
  candidates keep the screen's certified volume as a constant. Same descent direction; removes
  the 7.5 GiB allocations behind job 144596's 13 out-of-memory refine skips; ~1/5 of a refine.
* B. `SYMCLF_DET4_REFINE_METHOD=portfolio` (`_PORTFOLIO_LRS=0.02,0.1`): Adam 0.02 and Adam 0.1
  for a third of the 100-evaluation budget each, from the same start; the better endpoint; then
  golden-section line searches per constant, most sensitive first, with the rest. Keep-best.
  Evidence at 100 evals: gen 66 champion Adam 0.02 +15.1, Adam 0.1 +24.1, Adam-then-golden
  +21.6; gen 28 champion Adam 0.02 +15.7, Adam 0.1 +0.0 (no single step size is safe).
* C. `SYMCLF_DET4_REFINE_TOP=18`: two refines per actor (+~300 s per generation). The 90-candidate
  pool test: post-refine top-9 at pre-refine ranks 2-23 (9 elites capture 4, 18 capture 8); the
  winner (rank 2) is found by 9, 18, 27, 45 or 90 alike; 90 would cost 3.3x per generation.
* D. budget stays 100 evaluations per elite (Adam identical at 200 and 800).

# 4DCartPolerV6 = 3DCartPolerV6 in 4-D (2026-09-11)

The V6 pipeline unchanged -- DET4 mode, DET4Fast numbers for every screened candidate (V5), option-B
gate, V6 positivity penalties, price-selected Adam elites, tuner exits, GPU5_7 scheduling -- on the
full cart-pole (`examples/4DCartPoler/SystemDynamics.py`: x1 position, x2 velocity, x3 angle,
x4 rate). It undoes the 4-D -> 3-D reduction the way it was made (`symclf-main_rnv`: `srcGPU5_7` ->
`src3DCartPoleV3`); every 4-D line is taken from GPU5_7. No new term, check or knob.
`job4DCartPoleV6.slurm` = `../3DCartPolerV6/job3DCartPoleV6.slurm` except 5 lines (job name, box
default, JAX cache dir, 2 texts). Method and knobs: `../3DCartPolerV6/README_DET4.md`.

## Where (new folders only; nothing existing was edited)

| folder | content |
|---|---|
| `src4DCartPoleV7/` | copy of `src3DCartPoleV3` (the V6 engine). Changed: `runtime_exact_candidate.py` (= srcGPU5_7's file + V6's DET4 hunks; x.gradV with 4 terms), `grid_fitness.py` (kernel 21^4 slicing, 16 probes, 4-D fields / probes / reference grid / radial mesh, x4 in the encoder, scan axes), `fitness.py` (GPU5_7's rollout twin, scan axes, `_BASE_N` 4, `DET4Fast4D` path), `artstein_v3.py` (bounds guard, warm-up), `cpu_polish.py` (`_N` 4), `grad_norm_v3.py`, `pd_check_v3.py`, `ray_fitness.py` (warm-up). Unchanged: the three `b_manifold_*`, `__init__`, `jax_candidate`, `pallas_interpreter`. |
| `examples/DET4Fast4D/` | `det4_fast.py`, `evalnb.py`, `slsqp_nb.py`, `slsqp_direct.py` from `examples/DET4Fast`: `_N` 4; the 3-term sums, Jacobian offsets, NM simplex sort and centroid widened to 4-D. Own numba cache. |
| `examples/4DCartPolerV6/` | `BaseEvaluate.py` (N_STATES 4, box, rollout starts with x1 = 0, x4 in the missing-state rule), `Evaluate.py` (GPU5_7's 4-D ARE block and V_GRAD function), `run.py` (21^4 grid, 4 inputs, GPU5_7's seeds), `smoke_v3_det4.py` (4-D inputs; the 3-D-tree items dropped), `SystemDynamics(SR).py` from `examples/4DCartPoler`; `config.yaml`, `certify.py`, `status.sh`, `EvaluateGPU.py` byte-identical to V6; `gpu_ray.py` package name only. |

Kept as in V6 (not restored from GPU5_7): the legacy x1-axis terms' mask (all zero) and the
properness-reference selector (removed). Neither enters the det4 price.

## Accuracy (login node fe1; box (0.25, 0.25, 0.0494, 0.1155), rho 1000, origin 0)

DET4Fast4D vs the referee (`src/srcCPU` DET4):

| candidate | c_v* | kappa* | diff c_v* | diff kappa* |
|---|---|---|---|---|
| GPU5_7 ARE (Q = I, R = 1e-4) | 0.0440482364011 | 0.723750691792 | -7.6e-17 | -1.1e-16 |
| ARE(Q = diag(1,1,10,100), R = 0.1) x 25/max\|c\| | 0.0242417848196 | 0.167281630805 | +6.9e-18 | +1.7e-16 |
| all-ones | 1 | -5.38516480713 | 0 | -1.8e-15 |
| GPU5_7 seed_expr | 0.5 | -5.47274297398 | 0 | +1.3e-10 |

3-D DET4Fast on the same node: V6 ARE diff 0 / 0, all-ones diff kappa* -1.8e-12.

## Cost (measured on fe1, 3 numba threads)

det4_fast 1.45-1.88 s per candidate (referee lattice 2,610,397 points: scan 0.51-0.80 s, seeds
0.54-0.63 s) vs 0.254-0.260 s in 3-D (284,104 points): x5.6-7.2. Training grid 21^4 = 194,481 vs
41^3 = 68,921 (x2.8). Preflight smoke 341 s vs 201 s.

## Smoke (preflight replica on fe1): 16/16 PASS, 320 s (3-D V6: 22/22, 201 s)

First run 13 PASS, 3 FAIL, all three 3-D expectations: (1) ARE lattice score 17.93 (expected 0) and
(2) ARE fitness 18.549 (expected the length 0.295): the V6 penalties charge the 4-D ARE 1.190
(c_v* 0.04405 < 0.05) + 17.064 (lattice certified volume ~0.0029 < 0.02); the 3-D ARE paid 0
(c_v* 0.064, volume 0.076). (3) all-ones lattice kappa = -0 (expected < 0): on the x1 axis f = 0
and, for V = |x|^2, b = 0, so the ratio is exactly 0 there (the 4-D x1-axis degeneracy); all-ones is
still not certified (545.9 > 500). The user had those 3 checks adapted (below); the re-run passes:
ARE lattice score 17.93 = its V6 lattice terms 17.93; ARE fitness 18.5493 = length 0.295 + V6
18.2543; all-ones lattice kappa -0 <= kappa_min 0.
PASS in both runs: ledger (3), Pallas rows V / x.gradV vs numba (0 / 9.2e-17), cheap screen (ARE certified,
all-ones not), ARE promoted, exact stage certifies the ARE (1794 roots, 0 violations), all-ones'
escalated price exact, ARE and all-ones vs the referee (kappa* -3.3e-16 / -1.8e-15), V6 rule
ranking, elite selection, tuner exits, bad tree.

## The 4-D reference forms on the lattice (21^4, V6 knobs; lattice price = the V6 terms, ROA rule 0)

| box | form | lattice c_v | lattice kappa | certified volume | V6 terms c_v / kappa / volume | lattice price |
|---|---|---|---|---|---|---|
| (c) 0.25, 0.25, 0.0494, 0.1155 (default) | GPU5_7 ARE | 0.04565 | 0.773 | 0.00294 | 0.869 / 0 / 17.064 | 17.93 |
| (c) | V6-recipe ARE | 0.02444 | 1.444 | 0.01582 | 5.113 / 0 / 4.178 | 9.29 |
| (b) 0.25, 0.2394, 0.0492, 0.1129 (V2 rule on the 4-D ARE) | GPU5_7 ARE | 0.04573 | 0.744 | 0.00287 | 0.855 / 0 / 17.126 | 17.98 |
| (b) | V6-recipe ARE | 0.02442 | 0.332 | 0.01671 | 5.116 / 0 / 3.294 | 8.41 |
| (a) uniform 0.25 (GPU5_7) | GPU5_7 ARE | 0.04869 | 0.930 | 0.00131 | 0.263 / 0 / 18.689 | 18.95 |
| (a) | V6-recipe ARE | 0.02607 | 3.103 | 0.00144 | 4.786 / 0 / 18.555 | 23.34 |

The referee's c_v* and kappa* of both forms are the same on (a) and (c) (GPU5_7 ARE 0.04405 / 0.72375,
V6-recipe form 0.02424 / 0.16728); the box moves only the lattice numbers and the certified volume.

## Decided by the user (2026-09-11)

* Box (c) (0.25, 0.25, 0.0494, 0.1155) and grid 21^4: the defaults above, unchanged.
* V6 knobs unchanged (DET4_ALL = 1, elites 10 x 100, penalty targets 0.05 / 0.05 / 0.02 as in 3-D).
* The smoke's 3 checks with 3-D expectations adapted, nothing else: the ARE is certified on the
  lattice with grid score = its V6 lattice terms (ROA rule 0); the ARE's fitness = length + its V6
  terms on the exact numbers (ROA rule 0); all-ones is "not certified" on the lattice when
  kappa <= kappa_min (the rule's own definition) instead of kappa < 0.

## Findings on the way (report only, nothing changed)

* numpy 2.4.6 on fe1 does not order argsort ties like the NM port's stable insertion sort: 7 of 81
  tie patterns at 4 entries (the 3-D port) and 60 of 243 at 5 entries (this port) differ.
* The 4-D -> 3-D reduction dropped `@lru_cache(maxsize=12)` on
  `runtime_exact_candidate._b_interpreter` (the 3-D engines still lack it); the reversal restores
  GPU5_7's decorator here.

# V10 — CMA-ES on a fraction of the population (2026-09-19, the user's order)

New folders only (`examples/4DCartPolerV10`, `src4DCartPoleV10`, `examples/DET4Fast4D_V10` = copies of V9 with the
imports renamed; the det4 check is byte-identical to V9's). Everything new sits behind switches that default to
the V9 behaviour.

## Why
The user wants the CMA stage on at least 10 % of the population (~250 structures per generation instead of 3–6).
Measured (jobs 147719/147720, `~/symclf_tmp/lqr_roa/cma10/`): one det4 evaluation costs **2.5 core-seconds at 1
thread** and 5.5 at 6 threads (only the lattice scan parallelises); the V9 CMA stage — 18 GPU actors × 6 threads
on 64 cores, one population at a time — spends 8.6 core-s per evaluation and reaches 7.4 evals/s, so 250 × 4000
evaluations would take 37 h per generation. A q64 node with 64 single-thread det4 workers gives 17 evals/s, and
det4 is **bit-identical** there (32-candidate acceptance: 0 differing fields at 1 thread on EPYC 7452 vs the
s02n01 6-thread reference).

## What changed
* `src4DCartPoleV10/fitness.py`: `fitness_fixed_constants` = `fitness_fixed_grid` (one lattice call for the
  batch, grid numbers, the dead rule) followed by `fitness_fixed_det4` (det4_fast + price per entry). Same code
  path, same numbers; the halves can run on different machines.
* `src4DCartPoleV10/ray_fitness.py`: `CPUDet4Actor` (num_cpus=1, resource `det4pool`, NUMBA_NUM_THREADS=1, JAX on
  CPU) with `evaluate_fixed_det4`; `GPUFitnessActor.evaluate_fixed_grid`; `create_cpu_pool`;
  `cma_refine_elites_pooled` = the unchanged `cma_refine_elite` loop per structure (same recipe, seed, keep-best)
  with an objective that sends each population's grid half to a GPU actor and its det4 half to the pool, and
  `SYMCLF_CMA_POOL_LOOPS` structures in flight in threads (0 = pool size / popsize). The selection adds
  `SYMCLF_CMA_ELITE_FRACTION` (count = max(COUNT, ceil(fraction × checked structures))).
* `run.py` / `gpu_ray.py`: the pool is created when `SYMCLF_CMA_POOL=1` and handed to the mapper.
* `job4DCartPoleV10_het.slurm`: a Slurm heterogeneous job — het group 0 = one q32g node (3 GPUs, 9 actors, Ray
  head, run.py), het group 1 = 6 q64 nodes (`ray start --num-gpus=0 --resources '{"det4pool": 64}'`). Knobs:
  SYMCLF_CMA_POOL=1, SYMCLF_CMA_ELITE_FRACTION=0.10, SYMCLF_CMA_ELITE_COUNT=3, SYMCLF_CMA_ELITE_POPSIZE=10, CMA
  evals 4000, kappa hinge 100 @ 1.0, volume 20 @ 0.02, preflight off. `job4DCartPoleV10_2node.slurm` is the V9
  2-node job with the V10 code (pool off).

## Expected cost
6 q64 nodes ≈ 100 evals/s → 250 structures × 4000 evals ≈ 2.8 h per generation; 13 nodes ≈ 75 min. The P4
per-structure JIT (not built) would add another 2–3×.

## Tests
* `~/symclf_tmp/lqr_roa/v10test/pool_unit.py` (job 147721, q64, CPU-only with the grid kernel in interpret mode):
  32 candidates through the old in-process path and through the split + pool, every field compared bit for bit;
  the pooled CMA driver vs `cma_refine_elite` on the same seed (2 structures × 40 evals).
* The heterogeneous job itself needs a free GPU node (both are held by run 147711).

### Test results (2026-09-19)
* Unit test (job 147724): split + pool fitness bit-identical (0 differing fields); the pooled CMA driver
  reproduces the sequential `cma_refine_elite` exactly on the same seed — but only with **one process per CMA
  loop** (`_cma_loop_task`, a Ray task): threads in one process diverged because pycma seeds and samples numpy's
  global random stream (job 147721). 2 loops in flight: 195 s vs 357 s sequential on the CPU-only stand-in.
* Node types: det4 is bit-identical on AMD q64 (EPYC 7452) at 1 thread; on Intel q36 (Xeon Gold 6140) 94 of the
  acceptance fields differ (1e-16..1e-12, tie choices) — not numba's code generation (`NUMBA_CPU_NAME=generic`
  changes nothing) and not the BLAS (pinning changes nothing). Throughput: q64 17 evals/s per node, q36 8.3.

### Status and decision (2026-09-19, documented on the user's order; nothing launched)
Exact pool nodes: the two GPU nodes (AMD EPYC 7302) and q64 (AMD EPYC 7452). Intel q36 (Xeon Gold 6140)
reproduces det4 only to 1e-16..1e-12 (94 acceptance fields), independent of numba's CPU target, the BLAS kernels
and numpy's SIMD dispatch (jobs 147725-147729). Time per generation for 10 % of the population (~250 structures),
exact nodes only:

| budget per structure | 2 GPU nodes alone (~17 evals/s) | + 2 q64 (~51) | + 4 q64 (~85) |
|---|---|---|---|
| 4000 evals | ~16 h | ~5.4 h | ~3.3 h |
| 2000 evals | ~8 h | ~2.7 h | ~1.6 h |

With 4 Intel q36 nodes (8.3 evals/s each, 1e-12-level differences): ~2.8 h at 2000 evals. The P4 per-structure JIT
(not built) would add ×2–3 on every row. Options given to the user: (1) V10 on the 2 GPU nodes alone, 10 % × 2000,
~8 h/gen; (2) + q36 (`job4DCartPoleV10_q36.slurm`); (3) build P4 first; (4) wait for q64.

# Why the 4-D runs stop at the quadratic, and what 10 % CMA would cost (2026-09-19, documented on the user's order)

## Diagnosis from job 147711 (V9 2-node, CMA seed 2, 15 generations / 14 h)
| per generation | value |
|---|---|
| live structures (det4-checked) | 1,100–2,000 from GEN 5 |
| certified by the S1 tuner (SEA + Adam on the grid) | 0 (once, GEN 5: one at 55.6) |
| structures given CMA | 6 = 0.3 % of the live ones |
| outcome of the first CMA per generation | 1000–1500 → 600–740, never certified |
| best new structure | ~600: kappa* just below 0, uncertified |

Four causes stack up: (1) the S1 tuner never certifies — every new tree lands on the ~600 plateau; only CMA ever
crossed kappa* = 0 in all studies; (2) CMA reaches 6 of ~2000 structures, picked by the tuner's fitness, not by how
close they are to certification, and none of those certified within 4000 evaluations; (3) offspring do not inherit
their parents' constants — a child of the certified template is re-tuned from random constants and falls back to
~600, so the certified constants live only in the champion individual; (4) the champion at 5.62 is only beaten by a
certified structure with kappa* >= 1 and almost no hinge left, and uncertified ~600 structures lose every
tournament. The earlier non-quadratic champion (147117, x3·sin(x1) in place of x1·x3) came from the FIRST CMA run
on the initial population (a mutated template), not from evolution.
Proposed levers (not built): (a) constant inheritance (warm-start the tuner and the CMA from the parent's constants
when the slots match); (b) the pooled CMA on 10 % of the structures (V10); (c) broader, shallower CMA at equal cost;
(d) choose CMA elites by certification proximity (kappa* close to 0, c_v* > 0) instead of fitness.

## CPU-only 4-D is possible at the same numbers
The 6-D core in `cartpole4d` mode (examples/6DGP engine, SYMCLF6D_SYSTEM=cartpole4d) reproduces the V9 GPU run on the
champions of 147117 (non-quadratic, sin) and 147711 (template): grid numbers identical (0 to 3e-16), det4 c_v*
identical and kappa* to 3.4e-16, price to 1e-14 (job 147765, `~/symclf_tmp/gp6d/cpu4d/`). The one difference is the
6-D core's own `extra_penalty` (it flags a single sin as nested, +50); the 6DGP engine uses the 4-D formula instead.
Costs on q36 (Intel Xeon Gold 6140, 1 thread): lattice stage (21^4) 154–162 ms, full fixed-constants fitness
4.1–4.9 s (job 147764). Before a CPU run: the 32-candidate acceptance against the V9 GPU numbers.

## Cost per generation, 10 % of the population = 250 structures (population 2500, ~2000 live)
| item | CPU only (GenomeDK-type node) | PRIME GPU nodes (tuner on GPU) |
|---|---|---|
| S1 SEA (41 grid evals/tree) | 4.4 core-h | on GPU (V9: 1–5 min) |
| S1 Adam 40 steps (live trees) | ~66 core-h | on GPU |
| S2 det4 on live trees | ~2.5 core-h | same, on the 64 cores |
| CMA 250 × 2000 evals | ~625 core-h | ~515 core-h -> **~8 h** on 64 cores |
| CMA 250 × 1000 evals | ~310 core-h | ~4 h |
| CMA 250 × 500 evals | ~155 core-h | ~2 h |
| **total, 250 × 2000** | **~700 core-h/gen** = 28 min at 1500 cores | ~8 h/gen |
| **total, 250 × 1000** | ~385 core-h/gen = 15 min | ~4 h/gen |
| **total, 250 × 500** | ~230 core-h/gen = 9 min | ~2 h/gen |
With a 10–15k core-h budget: 14–21 generations at 2000 evals, 26–39 at 1000, 43–65 at 500. The P4 per-structure
JIT (not built, ×2–3) would halve every CMA and Adam row. The CPU speed of the other cluster is unknown: calibrate
with one node for one hour (~40 core-h) before the run.

# V10 items 1–3 + P4 JIT + the 6-D lattice on the 4-D + the pilot (2026-09-19 late, the user's order)

Every new behaviour is a switch that is **off in the code** (V9 behaviour); the V10 slurms (`_2node`, `_het`, `_q36`,
`_pilot`) switch items 1–3 on.

| item | switch (slurm default) | what | code |
|---|---|---|---|
| 1 constant inheritance | `SYMCLF_INHERIT_CONSTS=1` | an offspring's `consts` (DEAP clones the parent's attribute; Ray pickles it to the actors) are scored on the grid after the SEA and kept when strictly better, before Adam; only when the constant count matches and all are finite | `fitness._apply_inherited`, `_tune_group`, `fitness_pre_exact` |
| 2 CMA by closeness to certification | `SYMCLF_CMA_SELECT=kappa` | CMA candidates must be PD (kappa* > −1e5 gate value, c_v* > pd_eps), ranked by the highest DET4 kappa* (certified first, then the least negative), then fitness; distinct structures, fitness gate 3000, not CMA'd before | `ray_fitness._select_cma_elites_kappa` |
| 3 P4 per-structure JIT | `SYMCLF_CMA_POOL_JIT=1` (-> `SYMCLF_DET4_JIT=1` in the pool actors only) | det4's lattice scan compiled once per structure (constants a runtime argument); used where a structure is evaluated many times (the CMA pool) | `DET4Fast4D_V10/jit_scan.py`, `det4_fast.py` |
| + no Adam on dead trees | `SYMCLF_SKIP_DEAD_ADAM=1` | trees with a symbolic structure penalty >= 1e6 (missing state, nested exp/aq) skip the Adam refine; the CMA never selects dead trees (both selections exclude them) | `fitness._refine_elite` |

## P4 JIT: exactness, speed, memory
* The generator unrolls the interpreter (`evalnb._dual1` value+gradient, `evalnb._wdelta` W) statement by statement:
  same operands, same order, same functions, no constant folding, no fastmath. What made it hard is **which glibc
  entry point** runs: LLVM fuses sin(v) and cos(v) of one operand in one basic block into ONE `sincos` call, and glibc's
  sincos differs from sin/cos in the last bit on ~0.065 % of arguments (128 / 131 of 200,000 measured). The
  interpreter's dual SIN always computes both from a runtime stack value -> always sincos; its W ops -> always plain
  sin/cos/exp/expm1. Four ways the unrolled code broke that, each found by the acceptance and fixed (jobs 147768–147789):
  | gen. | what broke | fix |
  |---|---|---|
  | v1 | the sin of a top-level `sin(...)` term is unused -> dropped -> plain cos | every dual value stays live (root value reduced into a returned sum) |
  | v2 | W constants shared / folded with the dual part | W reads its own parameter/literal arrays and a runtime zero z |
  | v3 | numba's default error model puts a zero-divisor branch after each division (aq); LLVM sank a sin past it -> pair split | `error_model='numpy'` (every divisor is >= 1 or NaN, never 0: no number changes) |
  | v4 | LLVM rewrote cos(-x) -> cos(x) after a `neg` -> operands differ -> pair split | SIN operand `v - z` (x - (+0.0) == x bit for bit, -0.0/NaN/inf included), opaque to LLVM |
  | v5 | (prevention) any other operand rewrite, reliance on glibc's sin being exactly odd | every libm operand passes through `- z` |
* **Acceptance of v5 (jobs 147789 + 147790 = two fresh random sets, AMD EPYC 7452, 1 thread), each run:** 1,561 candidates = the 305 champions of V6–V10 (+
  perturbations), 56 risky/previously-failing structures and 1,200 random GP trees (600 with add/sub/mul/neg/sin, 600
  also with aq/exp), 2 constant sets each: **raw lattice arrays W, a, b, x.gradV (2,610,397 points) bit-identical on
  1,559/1,561; the other 2 (`neg(exp(exp(exp(exp(exp(x3))))))`, overflow) hold NaN at the same entries with a different
  NaN sign bit only (LLVM may move a negation across a product; no consumer reads a NaN's sign); all 23 det4_fast
  output fields bit-identical on 1,561/1,561.** (Earlier generations: 147775 v2 parallel 929/929; 147780 v2 serial
  907/929; 147784 v3 1525/1529; 147786 v4 1529/1529.) The random trees are fresh every run (DEAP's generation is not
  reproducible across processes); every result file stores the expressions and constants.
* Speed at 1 thread: scan 1.56 -> 0.23 s (**x11**); det4 total 2.81 -> 1.42 s (**x2.0** on the mixed set, x2.7 on
  the champions); compile 0.8 s per structure (max 3 s).
* **Memory:** numba never frees compiled code (job 147779: nothing returned after dropping every reference). A
  prange (parallel) kernel costs 45–67 MB, a serial one 4.2 MB (jobs 147777/147779) -> the kernel is serial by default
  (`SYMCLF_DET4_JIT_PARALLEL=0`; the pool actors run one thread anyway) and the pool replaces an actor whose RSS passed
  `SYMCLF_CMA_POOL_MAX_RSS_MB` (2000) before a CMA stage, in place (list position kept). A fresh pool actor sits at
  ~1 GB. Test 147781: replacement in 2 s, det4 halves identical before/after and to the pool without the JIT.

## Unit tests of items 1, 2 and the dead-tree skip (job 147788, the logic with stand-ins for the grid tuner)
The real tuner on the CPU (Pallas interpret mode) needs > 20 min per call (147769/147776 cancelled), so the logic was
tested with deterministic stand-ins (the V10 changes do not touch the kernels): 16/16 PASS — switch on with nothing
inherited == switch off (same seed); matching parents offered and kept only when strictly better; wrong count / NaN
never offered; members without a parent unchanged; the individual's consts survive clone + pickle and reach the tuner
(switch on) / do not (off); Adam not called on the dead tree, live trees bit-identical; kappa selection = certified
first, then least negative, non-PD / c_v* <= pd_eps / dead / gated / done / duplicate excluded; the default dispatches
to the V9 selection (which, for the record, gives CMA to non-PD structures ranked by fitness).

## The 6-D lattice reduction on the 4-D (job 147770, 305 candidates, 6-D core in cartpole4d mode, q36, 1 thread)
The 6-D folder reduces only the lattice (mesh 7 / lines 3 / samples 801); polish top-k 40, maxiter 60, Sobol 64 are
the 4-D values. Against the full 4-D lattice 21/9/801:

| lattice | s/check | speed-up | kappa* identical | max abs d kappa* | kappa* HIGHER (missed a worse point) | c_v* identical | certificate flips |
|---|---|---|---|---|---|---|---|
| 21/9/801 (V9) | 6.25 | 1.00 | 305/305 | 0 | 0 | 305/305 | 0 |
| 21/7/801 | 4.24 | 1.47 | 220/305 | 7.9e-6 | 13 | 135/305 | 0 |
| 21/5/801 | 3.19 | 1.96 | 242/305 | 2.8e-6 | 3 | 206/305 | 0 |
| 15/9/401 | 4.18 | 1.49 | 254/305 | 4.1e-6 | 6 | 146/305 | 0 |
| 15/5/401 | 2.70 | 2.31 | 230/305 | 4.9e-6 | 8 | 129/305 | 0 |
| 11/3/401 | 2.37 | 2.63 | 223/305 | 4.9e-6 | 6 | 132/305 | 0 |

Every smaller lattice changes the numbers (not bit-exact) and on 3–13 candidates reports a less strict kappa*
(misses the worst point); no certificate flipped on this set. The JIT gives a comparable speed-up (x2.2–2.7) with
bit-identical numbers, so the 4-D keeps the full lattice. (A first run of this test, 147767, hung: a fork-started
pool after numba's thread pool was running; 147770 uses spawn.)

## What runs on the GPU (the user's question)
The S1 tuner (SEA + Adam on the 21^4 training grid, JAX/Pallas) and the grid half of every fitness evaluation
(also inside CMA). det4_fast — its 2.6 M-point lattice scan, the seeds, the SLSQP/NM polishes — runs entirely on the
CPU (the V9 GPU lattice is off because it is not bit-exact).

## Option 4 pilot: `job4DCartPoleV10_pilot.slurm` (prepared, not submitted: 147711 holds both GPU nodes)
= `job4DCartPoleV10_2node.slurm` + the CMA pool on the GPU nodes' own cores (Ray `--num-cpus 32+28`,
`--resources {"det4pool": 28}` per node -> 56 single-thread det4 workers, idle cores during the CMA stage) + CMA on
10 % of the checked structures x 500 evals, popsize 10, seed 2 (as 147711), 12 h, --mem 200G (a pool actor
~1 GB + kernels; V9 uses ~20–23 GB per node). Items 1–3 on.
Estimate per generation (147711 stages: S1 300 s, S2 156 s, refine 165 s): ~195 structures x 500 = ~97,500 det4
evaluations at ~47/s (JIT) -> CMA ~45 min, total **~55–60 min per generation** (V9 now: 62 min with 6 x 4000).

## Finding: the island migration loses individuals when trees repeat (2026-09-19, report only, nothing changed)
With FLEX_MIGRATION_FIX=1 the islands migrate every mig_freq = 10 generations through DEAP's
`migRing(pop, 25, selection=random.sample)`. DEAP finds the slot to overwrite with `list.index(individual)`, i.e. by
TREE EQUALITY: with many copies of one tree (the seeded template with different constants) it overwrites the first
equal tree, which can be an individual that was not sampled — the certified champion. Job 147761 (6-D GP): certified
0.6588 from GEN 1 to 9, **lost at GEN 10 = the first migration**, 600.625 (uncertified) since. Job 147711 (V9) kept its
5.617 champion at GEN 10; the next migration is GEN 20. Simulation (`~/symclf_tmp/migration/mig_sim.py`, job 147791):
a champion that is 1 of 60 template copies in its island is lost by one DEAP migRing in **63.8 %** of 2,000 trials;
a position-based ring migration (the same exchange, slots by index: every moved individual lands one island further,
nothing is lost) in 0 of 2,000. The V10 slurms do not set FLEX_MIGRATION_FIX (no migration unless exported).

# Capped, sliced, RESUMABLE CMA (2026-09-19 night, the user's order after pilot 147839)

## Why the pilot failed (the record)
Pilot 147839 gave 500 evaluations to each of 37 structures over 5 CMA stages (18,537 evaluations) and certified
nothing. The studies of 16.09 had already measured the cost of certification on this template (jobs 147053/147074,
`~/symclf_tmp/lqr_roa/cma_rate|cma_targets`): of 9 CMA runs none certified before eval **1397** (1397, 1571, 1571,
1588, 1844, 2010, 2029, 2422, never), and after 500 evaluations every run was still on the ~1300–1460 plateau. V9's
own certification (job 147711, GEN 1) came at eval **2064** — the earlier report of "eval 206" in this README's
history was a misread of a truncated log line. A 500-evaluation budget therefore cannot certify, and combined with the
V9 rule "one CMA per structure ever" every one of those runs was thrown away at ~1000 fitness.

## What is new (switches, all off in the code; the V10 slurms set them)
| switch (slurm default) | meaning |
|---|---|
| `SYMCLF_CMA_RESUMABLE=1` | the CMA stage runs the capped / sliced / resumable scheme instead of the V9 once-per-structure `fmin2` path |
| `SYMCLF_CMA_MAX_EVALS_PER_STRUCTURE=2500` | **lifetime** cap per structure (P1) |
| `SYMCLF_CMA_DEEP_COUNT=6`, `SYMCLF_CMA_DEEP_EVALS=2500` | the deep tier: 6 structures get the whole cap at once |
| `SYMCLF_CMA_SLICE_COUNT=20`, `SYMCLF_CMA_SLICE_EVALS=500` | the slice tier: 20 structures get 500 evaluations that CONTINUE in later generations |
| `SYMCLF_CMA_MAX_RESTARTS=9` | warm restarts from the best constants when pycma's own criteria stop the search |

* The once-per-structure rule (`_CMA_DONE`) does not apply in this mode (P2): a structure is eligible again until it
  has spent the cap. `SYMCLF_CMA_ELITE_EVALS` still only enables the stage; the tiers set the budgets.
* **A slice resumes the search, it does not restart it.** pycma's state plus numpy's global RNG state pickle to
  ~22 KB; `cma_refine_elite_resumable` returns that blob, the driver keeps it per structure (`_CMA_STATE`) and hands
  it back next generation. Verified: 5 slices of 200 evaluate **the same constants in the same order** and end at the
  same best point, bit for bit, as one continuous 1000-evaluation run (job 147860).
* The resumable driver is a plain ask/tell CMA-ES (same sigma0 = 1, stds 0.3 max(|c0|, 0.1), popsize, seed,
  keep-best) plus warm restarts on stop. `fmin2`'s BIPOP restarts cannot be resumed, so this is **not** the same
  optimiser as V9's — hence the study below.
* Cost per generation at the measured 18.9 evals/s: 6 × 2500 + 20 × 500 = 25,000 evaluations ≈ **22 min**.

## Unit tests (job 147860, deterministic stand-in objective): 13/13 PASS
resumed == continuous (same evaluation sequence, same best, bit for bit) · warm restart fires when pycma stops and the
structure is marked exhausted after the restart limit · tiers 6 × 2500 + 20 × 500 with the cap truncating a structure
that already spent 2300 to 200 · exhausted structures excluded · states resumed only where known · the V9 code path
untouched (the only lines removed from the previous file are the three lines of the CMA stage branch).

## P4 study (job 147859, one GPU node, 4 schemes x 3 seeds x 2500 evals, `~/symclf_tmp/cma_resume/`)
Does the resumable driver certify like the V9 recipe? Same harness and fitness as jobs 147053/147074, so the numbers
are comparable: `bipop2500` (V9's fmin2 + BIPOP, the reference), `askell2500` (the V10 driver in one call),
`sliced5x500` (the same driver in 5 slices with the state pickled between them — must equal `askell2500`),
`warm2x1250` (fmin2 twice, restarted from the best point, no state). Metric: the evaluation at which the run first
certifies, and the best fitness at 500 / 1000 / 1500 / 2000 / 2500.

## The slice must carry the running best (2026-09-20, found by the study itself)
In study 147859 the `sliced5x500` arm drifted from `askell2500` on seed 2 although the two are the same driver. Cause:
when pycma's criteria stop the search the driver warm-restarts from the best point, and a resumed slice only knew the
best point of THAT slice while a continuous run knows the best so far. The running best now travels in the state blob
(`(es, numpy RNG state, restarts, best_x)`). Test 147872 adds a staircase objective that makes pycma stop repeatedly:
with 9 restarts, 3 slices reproduce the continuous run's evaluation sequence and best point bit for bit (13/13 PASS).
The study's `sliced5x500` arm predates this fix; `askell2500` is the arm that represents the shipped driver.

## Production run (2026-09-20)
`job4DCartPoleV10_run.slurm` = the pilot's node layout (both GPU nodes, 28 det4 workers per node) with the capped /
sliced / resumable CMA: 6 deep x 2500 + 20 slices x 500, lifetime cap 2500 per structure, the once-per-structure rule
off, CMA seed 2, popsize 10, inheritance + kappa* selection + the JIT on, migration off (the DEAP migRing defect is
not fixed yet), preflight off, 48 h. Expected ~28 min per generation (25,000 CMA evaluations at the measured
18.9 evals/s, plus ~6 min of the other stages).
