# Where the files of the earlier SymCLF repository are now

For someone who knows the earlier layout (`examples/<Name>/run.py`, `Evaluate.py`, `SystemDynamics.py`,
`SystemDynamicsSR.py`, `config.yaml`, `job.slurm`, and `src/Fitness.py`).

| Earlier | Now | What changed |
|---|---|---|
| `examples/<Name>/run.py` | `src/CPU/run.py`, function `main` | One runner for all examples. It still builds the primitive set from the `gp:` block, adds the constant `a`, calls flex and writes the best tree. It is started as `python -m src.CPU.run examples/<name>` by the job script. |
| `examples/<Name>/SystemDynamicsSR.py` | `examples/<name>/systems/<name>.py` | Same content: `fSR`, `GSR` in sympy, plus `NAME`, `N_STATES`, `N_INPUTS`, `BOX_HALF_WIDTHS`, `Q_MATRIX`, `R_MATRIX`. |
| `examples/<Name>/SystemDynamics.py` (numpy version of f and G) | not needed | The numeric functions are generated from `fSR` and `GSR` (`src/CPU/evalnb.py`, `set_dynamics`). |
| `src/Fitness.py`, function `fitness` (flex calls it for a batch of trees) | `src/CPU/gp_engine.py`, function `persistent_actor_mapper` | Called once per generation with all trees. It sends them to the workers, runs the check once per distinct tree, then runs CMA. |
| `examples/<Name>/Evaluate.py` (the score of one tree, written per example) | `src/CPU/rule.py`, function `fitness_fixed_constants` | The same code for every example; the examples differ only in `config.yaml`. It calls the grid stage (`src/CPU/lattice.py`, `lattice_numbers`) and the check (`src/CPU/deterministic_search.py`, `deterministic_search`). |
| the weights and thresholds written inside `Evaluate.py` | `fitness.*` and `verification.*` in `config.yaml` | See `docs/SETTINGS_CPU.md`. |
| constant tuning inside `Evaluate.py` (`eval_MSE_and_tune_constants`) | `src/cma_all.py` (who gets how many evaluations) and `src/CPU/gp_engine.py`, function `cma_refine_elite_resumable` (the CMA-ES loop) | Options (b) `cma_all` and (c) `cma_vip`, chosen by `stages.tune.mode`. |
| `src/VVdot_Calculations.py` (V, its gradient, a, b on the grid) | `src/CPU/lattice.py` for the grid (finite differences, as before) and `src/CPU/evalnb.py` for the check (exact derivatives) | |
| `config.yaml` (GP settings only) | `examples/<name>/config.yaml` | The earlier file is the `gp:` block at the bottom. The blocks above it hold the settings that were inside `Evaluate.py` and `run.py`. |
| the grid built at the top of `run.py` | `grid.points_per_axis` and `system.box` in `config.yaml` | |
| `job.slurm` | `examples/<name>/slurm/job_<name>_cpu.slurm` | It also starts Ray on every node of the job, so a run can use several nodes. |
| `best_expression.txt`, `output.log` in the example folder | `examples/<name>/runs/<job>/` and `examples/<name>/runs/output_<job>.log` | One folder per job: best tree per generation (`.jsonl`), verified pairs (`.csv`), the settings used (`effective_env.txt`). |
| `src/Functions.py`, `src/PredictScoreFuncs.py`, `src/SymFunctions.py` | same names in `src/` | |
| `src/numerical_verification.py`, `shgo_numerical_verification.py`, `pygmo_counterexample_optimizers.py`, `clf_checks.py` | `src/CPU/deterministic_search.py` | One check, used inside the search for every tree. `src/CPU/audit.py` recomputes it independently for one formula. |
| `src/formal_verification.py`, `src/Sympy2SMT2.py` (dReal), `src/plots.py`, `src/NumSol.py`, `src/CostFunction.py`, the notebooks | not in this repository | This repository holds the CPU search pipeline only. |

## The path of one tree, by function

    src/CPU/run.py            main                       builds the GP, starts the workers
    src/CPU/gp_engine.py      persistent_actor_mapper    one call per generation
      stage_tune                                         length gate, grid stage, dead or alive
        src/CPU/lattice.py    lattice_numbers            V on the grid, grid score
      stage_check                                        once per distinct (expression, constants)
        src/CPU/rule.py       fitness_fixed_constants    grid score + check cost + complexity
          src/CPU/deterministic_search.py  deterministic_search    c_v*, kappa*
            src/CPU/evalnb.py                            V, grad V, a, b at the scan points
      _cma_all_stage or _cma_vip_stage                   CMA-ES on the constants
        src/cma_all.py        select or select_vip       who gets how many evaluations
        cma_refine_elite_resumable                       the CMA-ES loop; every evaluation is fitness_fixed_constants
