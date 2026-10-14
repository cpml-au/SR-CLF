# SR-CLF, CPU pipeline

Symbolic regression of control Lyapunov functions (CLFs). For a system

    x' = f(x) + G(x) u

the code searches for a formula V(x) by genetic programming, checks every candidate numerically, and tunes
the constants of the formula with CMA-ES. Everything runs on CPUs.

## What is where

| Path | What it is |
|---|---|
| `examples/<name>/systems/<name>.py` | the system: f, G, the state box (sympy). The one file you write for a new system. |
| `examples/<name>/config.yaml` | every numerical setting of a run, and at the bottom the GP settings (`gp:` block). |
| `examples/<name>/slurm/job_<name>_cpu.slurm` | the job script, 24 lines: cluster settings only (partition, nodes, time). |
| `examples/<name>/runs/<job>/` | what a run writes (not in git). |
| `examples/<name>/reference/<job>/` | settings and records of the reference jobs, kept for comparison. |
| `src/CPU/run.py` | the runner, the same for every example: `python -m src.CPU.run examples/<name>`. |
| `src/CPU/gp_engine.py` | what happens to each tree: grid stage, check, CMA. |
| `src/CPU/lattice.py`, `deterministic_search.py`, `rule.py` | the grid stage, the check (c_v*, kappa*), the fitness rule. |
| `src/CPU/encode.py`, `evalnb.py` | turn a tree into numbers: V, its gradient, a, b at many points. |
| `src/cma_all.py` | which trees get how many CMA evaluations: options (b) and (c). |
| `src/settings.py`, `src/CPU/config.py` | read `config.yaml` (see `docs/SETTINGS_CPU.md`). |
| `tools/cluster.sh` | the steps of every job: Python environment, settings, Ray on all nodes, the run. |
| `tools/submit_frozen.sh` | submits a job from a fixed copy of the committed code. |
| `tests/gates/` | two unit tests of the CMA budgets. |
| `docs/MAP.md` | where the files of the earlier SymCLF repository are now. |
| `docs/SETTINGS_CPU.md` | every setting: key in `config.yaml`, environment variable, meaning. |

## What a run does

1. The job (`tools/cluster.sh`, the same for every example) unpacks the Python environment on every node,
   reads `config.yaml`, starts Ray on all nodes and starts the runner.
2. The runner builds the trees' building blocks from the `gp:` block: the states x1..xn, the listed
   primitives, and one constant symbol `a` (each `a` in a tree is its own constant). flex evolves the trees on
   several islands.
3. Every tree goes through these stages (`src/CPU/gp_engine.py`):
   - **gate**: a tree with `stages.gate.max_length` nodes or more gets fitness 1e8.
   - **grid stage** (`lattice.py`): V on a grid of `grid.points_per_axis` points per state. A tree that does
     not contain every state, or whose V is not finite, is "dead": it keeps a large score and is not checked.
   - **check** (`deterministic_search.py`): the two margins below, by a scan of many points followed by local
     searches from the worst points.
   - **fitness** (`rule.py`).
4. CMA-ES tunes the constants of the live trees, with the same fitness as objective (`stages.tune.mode`):
   - `cma_all`, option (b): every live structure, more evaluations for the better ranked ones.
   - `cma_vip`, option (c): the seed template first, then a random draw from the best, then the rest.
5. After each generation the best tree is appended to `runs/<job>/<job>_best_per_generation.jsonl`, and every
   numerically verified (expression, constants) pair to `runs/<job>/<job>_numerically_verified.csv`.

### The two margins and the fitness

With W = V - V(0), a = grad V . f, b = grad V . G, and the input bound rho (`verification.u_max`):

    c_v*   = min over the box, x != 0, of  min( W / |x|^2 ,  x . grad V / (2 |x|^2) )
    kappa* = - max over the points with W > 0 of  (a - rho * sum_k |b_k|) / W

A candidate is **numerically verified** when c_v* > `verification.pd_eps` and kappa* > `verification.kappa_min`.
This is the result of a search: no violation was found. It is not a proof.

    fitness = grid score + check cost + complexity penalty

For a live tree this is the rule of `src/CPU/rule.py` on c_v* and kappa* (a region term and three hinges:
`fitness.hinge.*`), plus `fitness.length_reg` per node and `fitness.nested_penalty` per nested call.
Lower is better.

## Running

On a Slurm cluster, from the repository root:

    tools/submit_frozen.sh <name> job_<name>_cpu.slurm -p <partition> --nodes=<N> --cpus-per-task=<C> --mem=<M> --time=<T>

The options after the script name are ordinary `sbatch` options and replace the `#SBATCH` lines of the job
script. The run uses one worker per core (two cores are kept for the driver). Without the helper: `cd examples/<name>/slurm && sbatch job_<name>_cpu.slurm`.

Change a setting for one job without editing a file:

    ... --export=ALL,SYMCLF_TUNE_MODE=cma_vip,SYMCLF_OPT_GENERATIONS=50

## Common tasks

| Task | What to edit |
|---|---|
| Add a system | Copy an example folder. In `systems/<name>.py` set `NAME`, `N_STATES`, `N_INPUTS`, `BOX_HALF_WIDTHS`, `fSR`, `GSR`, `Q_MATRIX`, `R_MATRIX`. In `config.yaml` set `system.name`, `system.box`, `grid.points_per_axis`, `verification.u_max`, `verification.lattice`. Rename the job script. |
| Change a weight or a target of the fitness | `fitness.*` in `config.yaml` (for example `fitness.hinge.kappa_target`). |
| Switch the constant tuning | `stages.tune.mode`: `cma_all` or `cma_vip`; budgets in `stages.cma_all` and `stages.cma_vip`. |
| Change the GP (population, islands, generations, primitives) | the `gp:` block at the bottom of `config.yaml`. |
| More or fewer cores | the `sbatch` options (`--nodes`, `--cpus-per-task`). Nothing else changes. |

## Reference jobs and reproduction

`examples/<name>/reference/<job>/` holds the records of 13 reference jobs: the best tree of every
generation (`<job>_best_per_generation.jsonl`) and every numerically verified (formula, constants) pair
(`<job>_numerically_verified.csv`).

Each example's `config.yaml` gives the settings of its reference job, with three exceptions. Two examples
have a second reference job that took a few settings on the `sbatch` line, and the cartpole4d file was
changed afterwards:

| Example | `config.yaml` describes | The other job, with these settings on the `sbatch` line |
|---|---|---|
| ex2d1 | 148600 (the file also sets `verification.kappa_norm: w`, the code's default) | 148609: `SYMCLF_CHECK_KAPPA_NORM=xqx,SYMCLF_CHECK_PEN_CV_TARGET=1.0,SYMCLF_CHECK_PEN_KAPPA_TARGET=7019.0423484208995,SYMCLF_CHECK_PEN_VOL_TARGET=0.7406305770374777` |
| vanderpol2 | 148603 | 148619: `SYMCLF_OPT_GENERATIONS=200,SYMCLF_CHECK_PEN_CV_TARGET=1.682132389935447,SYMCLF_CHECK_PEN_KAPPA_TARGET=1.0,SYMCLF_CHECK_PEN_VOL_TARGET=0.7406305770374777` |
| cartpole4d | 200 generations and the `exp` primitive switched on | 148669 ran with `generations: 20000`, without `exp`, and `SYMCLF_TUNE_MODE=cma_vip` |

- **Evaluating a given formula is reproducible.** Recomputing every record of the 13 jobs with this code gave
  the stored fitness, c_v* and kappa* digit for digit: 1,457 generation records and 68,521 verified pairs,
  0 differing, on the CPU types the jobs ran on. On another CPU type the last digits can differ.
- **A new run does not grow the same trees.** The random numbers of the genetic programming are not seeded.
  What a new run does repeat is the CMA search on the seed template in option (c): the same path and the
  same result (measured to about 11 digits on another CPU type).

The tag `exact-copy` marks the state that was checked this way.

## Requirements

- Python packages: `environment.yml` or `requirements.txt`. The run also imports `jax` and, through flex,
  `scikit-learn`.
- flex at commit `4f5746e` with the patch in `docs/flex_patch/`, and the deap fork flex uses
  (`github.com/cpml-au/deap`).
- Slurm with `srun`. The job script expects the packed Python environment at `$HOME/flex.tar.gz`
  (`FLEX_TAR` names another path).
