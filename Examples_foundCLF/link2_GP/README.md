
# 2-link planar robot balancing -- GP search

Built 2026-09-20 on the user's order, from the dynamics in the user's screenshot ONLY: the paper's neural
Lyapunov function and its neural controllers are **not** used anywhere. The control law is Sontag's formula with
the V this pipeline finds, exactly as in the 4-D and 6-D runs (standing rule). The pipeline, the fitness, the DET4
check and the price rule are the 6-D folders' unchanged copies; only `systems/link2.py`, the box and the lattice
sizes in `env.sh` are new.

## The system (`systems/link2.py`)
4 states `[theta1, theta2, dtheta1, dtheta2]`, 2 inputs (one torque per joint). M, C, tau and the
coefficients a_ij, b_i are the paper's formulas; f = [theta' ; -M^-1 (C theta' + tau)], G = [0 ; M^-1].

## Decisions taken while building (change them in `systems/link2.py` / `env.sh` if you want them different)
* **Link parameters** (the source gives none): every link is a uniform rod, m_i = 1 kg, l_i = 1 m,
  lc_i = 0.5 m, I_i = m l^2/12, g = 9.81.
* **B = identity** (one torque per joint). The source writes `B = [1,...,1]^T` with u in R^n, which only types as
  the identity.
* **Box** +-0.5 per state: it contains the source's domain ||x||_2 <= 0.5 (so the certificate is asked on a larger
  set than the paper's).
* Lattice and training grid: the 4-D values (grid 21^4, lattice 21 / 9 / 801).

## How to run
```
cd examples/link2_GP && sbatch joblink2GP.slurm            # the GP loop (same engine as examples/6DGP)
```
2 q64 node(s), CPU-only Ray, one single-thread actor per core; population 2500 over 5 islands, tree length cap
90, CMA stage 3 structures x 2000 evaluations, migration fix on.


---

# Inherited README of the 6-D folder this was copied from

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

## 2026-09-20: why the first GP runs of this folder never certified (root cause)
`run.py` inherited `N = 6` from the 6-D quadrotor folder. The GP therefore built 6-variable trees and 6-D seeds
(27- and 125-node) for this 4-state system; the actors, running the correct system, gated the template by length
(125 >= 90) and the sum-of-squares seed by encode (x5, x6 do not exist), so the seeds never reached the CMA stage
and the population evolved degenerate structures (jobs 147944-147956). Fixed: `N = C6.system().N_STATES`.
The bicycle folder had the same defect; the 6-state folders were unaffected.
