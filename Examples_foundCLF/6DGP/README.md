# 6-D GP — CLF symbolic regression for the 6-D quadrotor (2026-09-19)

The user's order: build the 6-D GP "the same as 4D but the dynamics and the settings of the 6D folder".

* **Fitness core**: a copy of `examples/6D/core` + `systems` (the 6-D port of the V8 fitness: lattice stage on a
  7^6 grid, the DET4 check with the 1,365,707-point lattice, the V8 price, b-norm l1 for the three inputs), with
  `examples/6D/env.sh` (box ±3, rho 1000, lattice 7/3/801). Accepted there: 4-D mode bit-identical to V8; 6-D
  check agrees with the independent `audit.py` to 1e-14.
* **Pipeline** (`gp6d_engine.py`), the 4-D V9 stages on CPU Ray actors (one single-thread actor per core):
  S0 length gate; S1 the V9 SEA (35 + ones, 5 steady-state steps) + the V9 central-difference Adam (40 steps,
  lr 0.3) on the lattice stage's pre_exact; S2 the check on every live tree, once per (expression, constants),
  driver cache; S2b CMA (the 146800 recipe) on the best new structures under the gate, one process per structure;
  extra = 50 × nested flags + 0.005 × tree length. Not ported: V9's S3 (the det4-gradient refine of 18 elites).
* **run.py**: Flex GP on 6 variables, the 4-D V9 config.yaml (5 islands × 500, crossover 0.7, mutation 0.3,
  add/sub/mul/neg/sin), seeds = the coefficient-free sum of squares + x1x2 and the 21-monomial template with `a`
  placeholders (no LQR coefficients); GEN line + `<job>_best_per_generation.jsonl`; `audit.py` on the champion.
* **job6DGP.slurm**: q36 nodes (Intel Xeon Gold 6140), CPU only. Knobs: kappa hinge 100 @ 1.0, volume 20 @ 0.02
  (the V9 last version); max tree length **190** (the user, option a: the 125-node template + 50 % headroom, as
  90 is for the 59-node 4-D template); **no Adam on structurally dead trees** (SYMCLF6D_SKIP_DEAD_ADAM=1, the
  user; CMA never selects dead trees); CMA 3 structures × **2000** evals (the 6-D template run 147679 certified
  at eval 28-42 and reached the fitness floor 0.625 at eval 1231 / 1670, nothing gained after 2000), popsize 18,
  seed 2, gate 3000; migration fix on.
* **Costs** (q36, 1 thread): lattice stage 100-170 ms, full fitness 3.9 s per evaluation.
* **Tests**: smoke job 147760 (1 node, 40 individuals, 2 generations): the pipeline runs end to end; it showed the
  4-D cap of 90 rejecting the 125-node template -> the cap 190.
* **Run**: job 147761 on 3 q36 nodes (108 cores), 48 h, submitted 2026-09-19.

# Run 147761 (cancelled 2026-09-20 on the user's order) and the V10 port

## What 147761 produced
13 h, 51 generations. GEN 1-9: the seeded template certified at fitness 0.6588. **GEN 10: the first island
migration lost that champion** (DEAP's `migRing` picks the individual to overwrite with `list.index`, i.e. by tree
equality, so with many copies of one tree it overwrote a structure that had not migrated -- simulation job 147791:
63.8 % of migrations lose such a champion). The run then sat at 600.625 for ~25 generations and recovered by itself:
**GEN 47-51 best = 0.475, certified, c_v* = 0.4702, kappa* = 1.027, volume 2.0 %, 95 nodes** -- better than the
template it had lost, and found by the CMA stage (certified at eval 2 of that run because the structure inherited
good constants). The champion is saved in `147761_FINAL_champion.txt` and `147761_best_per_generation.jsonl`.

## The V10 port (2026-09-20, the user's order "adopt the GPs to the V10")
`gp6d_engine.py` (this folder and every `<system>_GP` folder) now has the 4-D V10 CMA stage behind switches, all set
in the job scripts:
* `SYMCLF_CMA_RESUMABLE=1` -- the budget is a LIFETIME cap per structure spent in slices that RESUME the structure's
  own pycma state (state + numpy RNG state pickled; a resumed run is bit-identical to a continuous one, tests
  147860/147872). The V9 rule "one CMA per structure ever" is off.
* `SYMCLF_CMA_SELECT=kappa` -- candidates ranked by how close the check puts them to certification (PD, highest
  kappa*), not by fitness.
* Tiers `SYMCLF_CMA_DEEP_COUNT=6` x `SYMCLF_CMA_DEEP_EVALS=2500` + `SYMCLF_CMA_SLICE_COUNT=20` x
  `SYMCLF_CMA_SLICE_EVALS=500`, lifetime cap `SYMCLF_CMA_MAX_EVALS_PER_STRUCTURE=2500`, and the FIRST elite of a
  generation may reach `SYMCLF_CMA_FIRST_MAX_EVALS=5000` (certification lands near eval 2000 -- 147876 hit it at
  2064 -- and the fitness falls from ~100 to ~1-6 only in the evaluations after that: 147093 s2 reached 0.99 at
  3000, V9 147711 reached 5.62 at 4015).
* At most `actors / popsize` CMA loops in flight, each its own Ray task (pycma draws from numpy's global stream).
Why it matters here: run 147870 (ducted fan, old engine) ran CMA **twice in 11 generations** because of the
once-per-structure rule, and its champion stalled at kappa* = -0.078, a gap CMA closes.
