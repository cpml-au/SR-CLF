# Settings

Every numerical setting of a run is a key in the example's `config.yaml`. At the start of a job each key is
turned into one environment variable, and the code reads that variable.

    config.yaml key  ->  environment variable  ->  the code

Two rules:

- **A variable that is already set when the job starts wins over `config.yaml`.** This is how one setting is
  changed for one job without editing a file:
  `sbatch --export=ALL,SYMCLF_CHECK_PEN_KAPPA_TARGET=0.5 job_<example>_cpu.slurm`
- **A key that is missing from `config.yaml`, or set to `null`, falls back to the default written in the code**
  (mostly `src/CPU/config.py` and the top of `src/CPU/gp_engine.py`). Those defaults are not always the values
  of the example configs, so keep the keys in the file.

A key that is not in the table below is an error. Each run writes the variables it actually used to
`runs/<job>/effective_env.txt`.

## Keys of `config.yaml`

This is the table `KEYMAP` of `src/settings.py`: 62 keys with a variable.
`python -m src.settings --table examples/<example>/config.yaml` prints it with the values of one example.

| key in config.yaml | environment variable | meaning |
|---|---|---|
| `system.name` | `SYMCLF_CPU_SYSTEM` | the system file: examples/&lt;example>/systems/&lt;name>.py (f, G, the box) |
| `system.box` | `SYMCLF_BOX` | half-width of the state box on each axis |
| `system.template` | `SYMCLF_CPU_TEMPLATE` | the quadratic seed template: full = all n(n+1)/2 monomials \| are = only those of the Riccati solution |
| `system.template_x0` | `SYMCLF_CPU_TEMPLATE_X0` | start value of every coefficient of the template |
| `backend.threads.cpu` | `NUMBA_NUM_THREADS` | numba threads per worker (one core per worker) |
| `backend.check_batch.cpu` | `SYMCLF_CPU_S1_BATCH` | trees handed to a worker in one call |
| `backend.driver_cpus` | `SYMCLF_CPU_DRIVER_CPUS` | cores kept for the driver; the others each run one worker |
| `grid.points_per_axis` | `SYMCLF_CPU_GRID_POINTS` | points per axis of the training grid (odd, so that the origin is a grid point) |
| `verification.final` | `SYMCLF_CPU_FINAL_AUDIT` | what is run once on the best tree at the end of a run: [audit] = src/CPU/audit.py, [] = nothing |
| `verification.u_max` | `SYMCLF_CHECK_RHO` | the input bound rho in kappa* = -max (a - rho \|b\|) / W |
| `verification.b_norm` | `SYMCLF_CPU_B_NORM` | the norm of b over the inputs: l1 (each input bounded by rho) \| l2 (the input vector bounded by rho) |
| `verification.pd_eps` | `SYMCLF_CHECK_PD_EPS` | positive definite iff c_v* > this |
| `verification.kappa_min` | `SYMCLF_CHECK_KAPPA_MIN` | numerically verified iff kappa* > this (and c_v* > pd_eps) |
| `verification.b_rel_tol` | `SYMCLF_CHECK_B_REL_TOL` | a search end point counts as on b = 0 when rho \|b\| &lt;= this x \|a\| |
| `verification.project_ascent` | `SYMCLF_CHECK_PROJECT_ASCENT` | project the end points of the search on b = 0 onto b = 0 before that test |
| `verification.origin_seeds` | `SYMCLF_CHECK_ORIGIN_SEEDS` | add points along the worst direction at the origin (from the linearisation) to the search |
| `verification.kappa_norm` | `SYMCLF_CHECK_KAPPA_NORM` | what kappa* divides by: w = W = V - V(0) \| xqx = x'Qx with the system's Q |
| `verification.lattice.mesh` | `SYMCLF_CPU_LATTICE_MESH` | scan points of the check: mesh points per axis |
| `verification.lattice.lines` | `SYMCLF_CPU_LATTICE_LINES` | scan points of the check: scan lines per remaining axis |
| `verification.lattice.samples` | `SYMCLF_CPU_LATTICE_SAMPLES` | scan points of the check: samples along each scan line |
| `verification.lattice.random_lines` | `SYMCLF_CPU_LATTICE_RANDOM_LINES` | scan points of the check: number of random lines |
| `verification.lattice.random_points` | `SYMCLF_CPU_LATTICE_RANDOM_POINTS` | scan points of the check: points per random line |
| `verification.lattice.seed` | `SYMCLF_CPU_LATTICE_SEED` | scan points of the check: seed of the random lines |
| `fitness.region_weight` | `SYMCLF_FIT_ROA_WEIGHT` | region term: weight |
| `fitness.region_target` | `SYMCLF_FIT_ROA_C_TARGET` | region term: target of (smallest W on the faces of the training grid) / (median \|W\|) |
| `fitness.hinge.cv_weight` | `SYMCLF_CHECK_PEN_CV` | hinge on c_v*: weight |
| `fitness.hinge.cv_target` | `SYMCLF_CHECK_PEN_CV_TARGET` | hinge on c_v*: target (no charge at or above it) |
| `fitness.hinge.kappa_weight` | `SYMCLF_CHECK_PEN_KAPPA` | hinge on kappa*: weight |
| `fitness.hinge.kappa_target` | `SYMCLF_CHECK_PEN_KAPPA_TARGET` | hinge on kappa*: target (no charge at or above it) |
| `fitness.hinge.volume_weight` | `SYMCLF_CHECK_PEN_VOL` | hinge on the sublevel fraction: weight |
| `fitness.hinge.volume_target` | `SYMCLF_CHECK_PEN_VOL_TARGET` | hinge on the sublevel fraction: target.  The fraction is the share of the training grid points with W below its smallest value on the faces |
| `fitness.non_pd_kappa_cap` | `SYMCLF_CHECK_GATE_KAPPA_CAP` | a candidate that is not positive definite is charged kappa* = -this |
| `fitness.length_reg` | `SYMCLF_LENGTH_REG` | charge per node of the tree |
| `fitness.nested_penalty` | `SYMCLF_FIT_NESTED_PENALTY` | charge for a sin or cos inside a sin or cos, for an exp inside an exp, for an aq inside an aq (once for each of the three) |
| `fitness.non_pd_weight` | `SYMCLF_FIT_NON_PD_WEIGHT` | extra charge for a tree that is not positive definite: this x the share of the check's scan points that fail the c_v* test (0 = off) |
| `stages.gate.max_length` | `SYMCLF_MAX_TREE_LENGTH` | a tree with this many nodes or more is not evaluated (fitness 1e8) |
| `stages.tune.mode` | `SYMCLF_TUNE_MODE` | how the constants are tuned: cma_all = option (b), CMA-ES on every live structure \| cma_vip = option (c), the seed template first, then a draw from the best, then the rest |
| `stages.cma.popsize` | `SYMCLF_CMA_ELITE_POPSIZE` | CMA-ES: evaluations per iteration |
| `stages.cma.seed` | `SYMCLF_CMA_ELITE_SEED` | CMA-ES: seed of every search |
| `stages.cma.max_restarts` | `SYMCLF_CMA_MAX_RESTARTS` | CMA-ES: a search that stops by itself restarts from its best point, at most this many times |
| `stages.cma_all.total_evals` | `SYMCLF_CMA_ALL_TOTAL_EVALS` | option (b): the most CMA evaluations of one generation, all structures together |
| `stages.cma_all.max_evals` | `SYMCLF_CMA_ALL_MAX_EVALS` | option (b): evaluations of the first rank per generation |
| `stages.cma_all.min_evals` | `SYMCLF_CMA_ALL_MIN_EVALS` | option (b): evaluations of the last rank per generation (linear in between) |
| `stages.cma_all.cap_verified` | `SYMCLF_CMA_ALL_CAP_VERIFIED` | options (b) and (c): lifetime evaluations of a structure once it is numerically verified |
| `stages.cma_all.cap_other` | `SYMCLF_CMA_ALL_CAP_OTHER` | options (b) and (c): lifetime evaluations of a structure that is not numerically verified |
| `stages.cma_vip.seed_first` | `SYMCLF_CMA_VIP_SEED_FIRST` | option (c), the seed template: evaluations of each island copy in generation 0 |
| `stages.cma_vip.seed_again` | `SYMCLF_CMA_VIP_SEED_AGAIN` | option (c), the seed template: evaluations each time it appears again |
| `stages.cma_vip.seed_cap` | `SYMCLF_CMA_VIP_SEED_CAP` | option (c), the seed template: lifetime evaluations |
| `stages.cma_vip.vip_pool` | `SYMCLF_CMA_VIP_POOL` | option (c), the draw: it is made among this many best structures (the seed excluded) |
| `stages.cma_vip.vip_count` | `SYMCLF_CMA_VIP_COUNT` | option (c), the draw: structures drawn per generation |
| `stages.cma_vip.vip_first` | `SYMCLF_CMA_VIP_FIRST` | option (c), a drawn structure: evaluations the first time |
| `stages.cma_vip.vip_again` | `SYMCLF_CMA_VIP_AGAIN` | option (c), a drawn structure: evaluations each later time |
| `stages.cma_vip.vip_cap` | `SYMCLF_CMA_VIP_CAP` | option (c), a drawn structure: lifetime evaluations |
| `stages.cma_vip.std_total` | `SYMCLF_CMA_VIP_STD_TOTAL` | option (c), the other structures: evaluations per generation, all together |
| `stages.cma_vip.std_max` | `SYMCLF_CMA_VIP_STD_MAX` | option (c), the other structures: evaluations of their first rank |
| `stages.cma_vip.std_min` | `SYMCLF_CMA_VIP_STD_MIN` | option (c), the other structures: evaluations of their last rank |
| `stages.cma_vip.total_max` | `SYMCLF_CMA_VIP_TOTAL_MAX` | option (c): the most CMA evaluations of one generation, everything together |
| `stages.cma_vip.draw_seed` | `SYMCLF_CMA_VIP_DRAW_SEED` | option (c), the draw: its random seed |
| `gp_runtime.seeds` | `SYMCLF_CPU_SEEDS` | put seed trees into the first population (true) or start from random trees only (false) |
| `gp_runtime.seed_trees` | `SYMCLF_CPU_SEED_TREES` | the seed trees, as a list of expressions; without it: x1^2 + ... + xn^2 + x1 x2, and the quadratic template |
| `gp_runtime.seed_islands` | `FLEX_SEED_ISLANDS` | how many islands get the seed trees |
| `gp_runtime.migration_fix.cpu` | `FLEX_MIGRATION_FIX` | migrate every gp.multi_island.migration.freq generations; an immigrant replaces a tree by its position (docs/flex_patch) |

## Other keys

- `gp:` is read by flex directly: population, generations, islands, migration, crossover and mutation,
  selection, and the list of primitives.
- `template_cma.*` has no variables. The `template_<example>.slurm` scripts, which run `src/CPU/cma_run.py`,
  read it with `python -m src.settings --get <key> <config.yaml>`:

| key in config.yaml | meaning |
|---|---|
| `template_cma.budget` | template run: fitness evaluations per seed (cma_run --budget) |
| `template_cma.popsize` | template run: evaluations per CMA iteration (cma_run --popsize) |
| `template_cma.sigma0` | template run: first step size (cma_run --sigma0) |
| `template_cma.seeds` | template run: the seeds, one run per seed (cma_run --seed) |
| `template_cma.workers` | template run: worker processes per seed (cma_run --workers) |
| `template_cma.threads` | template run: numba threads per worker (cma_run --threads) |

## Variables without a key (set on the `sbatch` line only)

| variable | effect | read at |
|---|---|---|
| `SYMCLF_OPT_NUM_ISLANDS` | number of islands, instead of `gp.multi_island.num_islands` | `src/CPU/run.py` `_apply_env_overrides` |
| `SYMCLF_OPT_POPULATION` | TOTAL population over all islands (trees per island = this / islands) | same |
| `SYMCLF_OPT_GENERATIONS` | number of generations, instead of `gp.generations` | same |
| `SYMCLF_OPT_CROSSOVER_PROB`, `SYMCLF_OPT_MUT_PROB` | crossover and mutation probability | same |
| `SYMCLF_OPT_REG_PARAM` | `gp.penalty.reg_param` | same |
| `SYMCLF_OPT_MIGRATION_FREQ`, `SYMCLF_OPT_MIGRATION_FRAC` | migration frequency and fraction | same |
| `SYMCLF_OPT_CONFIG` | another file for the `gp:` block (for example another list of primitives) | `src/CPU/run.py` `main` |
| `SYMCLF_CONFIG` | another file for all the other blocks | `src/CPU/run.py` `paths` |
| `SYMCLF_RUNS_ROOT` | where the run folders go, instead of `examples/<example>/runs` | `src/CPU/run.py` `paths` |
| `FLEX_TAR` | the packed Python environment, instead of `$HOME/flex.tar.gz` | `tools/cluster.sh` |

## Variables without a key that no reference job set

The code reads these too. None of the 13 reference jobs set one, so every recorded number was computed with
the default.

| variable | default | read at |
|---|---|---|
| `SYMCLF_CHECK_KAPPA_SCALE` | 1.0 | `src/CPU/config.py` |
| `SYMCLF_CLF_GAMMA1` | 0.0 | `src/CPU/config.py` |
| `SYMCLF_CHECK_B_TOL` | 1e-10 | `src/CPU/config.py` |
| `SYMCLF_CHECK_POLISH_TOP_K` | 40 | `src/CPU/config.py` |
| `SYMCLF_CHECK_POLISH_MAXITER` | 60 | `src/CPU/config.py` |
| `SYMCLF_CHECK_SOBOL_SEEDS` | 64 | `src/CPU/config.py` |
| `SYMCLF_CHECK_PD_RATE_TOL` | 1e-6 | `src/CPU/config.py` |
| `SYMCLF_FIT_ROA_RHO_MULTIPLIER` | 1.0 | `src/CPU/config.py` |
| `SYMCLF_CPU_SLSQP_THREADS` | 1 | `src/CPU/deterministic_search.py` |
| `SYMCLF_CMA_POOL_LOOPS` | 0 (= workers / popsize) | `src/CPU/gp_engine.py` |
| `SYMCLF_CPU_NUMBA_CACHE_DIR` | `<repository>/.numba_cache/<tag>` | `src/CPU/__init__.py` |
