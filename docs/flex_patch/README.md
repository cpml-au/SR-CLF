# The Flex patch

This repository depends on [Flex](https://github.com/cpml-au/Flex) (the DEAP-based
genetic-programming regressor, `GPSymbolicRegressor`) as an external package rather than
vendoring it. Every result in this repository was produced against Flex commit
`4f5746e26b8aa2a0871bee1e1c05ccd1c5c68c04` plus one local, uncommitted patch to a
single file, `src/flex/gp/regressor.py`. That patch is `regressor_symclf.patch` in this folder,
produced with

```sh
git -C <flex checkout> diff -- src/flex/gp/regressor.py > docs/flex_patch/regressor_symclf.patch
```

against the working checkout the jobs used (its `git remote -v` is `https://github.com/cpml-au/Flex`).

## What the patch does

Two environment-variable switches, both **off by default** — with neither variable set, Flex's
migration step behaves exactly as upstream:

* **`FLEX_SEED_ISLANDS`** — how many of `num_islands` islands receive the trees passed as
  `seed_str`. Upstream Flex seeds island 0 only (`self.__pop[0][:len(self.seed_ind)] = self.seed_ind`).
  The patch seeds the first `N = min(num_islands, int(FLEX_SEED_ISLANDS))` islands instead (default
  `N = 1`, i.e. unchanged), giving each island its own **clone** of the seed trees so they evolve
  independently rather than only diverging after the first migration. Exposed in this repository as
  `gp_runtime.seed_islands` in `config.yaml` (see `docs/SETTINGS_CPU.md`).

* **`FLEX_MIGRATION_FIX`** — two independent fixes to the migration step, both gated by this one
  switch:
  1. **Migration frequency.** Upstream Flex migrates when `cgen % self.mig_frac == 0`, but
     `mig_frac` is the *fraction* of individuals migrated per event (e.g. `0.05`), not a
     generation count — for an integer generation counter that test is essentially never true, so
     islands never migrate under the documented `multi_island.migration.freq` setting. With the
     switch on, the patch tests `cgen % self.mig_freq == 0` instead (the actual frequency field).
  2. **Which individual an immigrant replaces.** Upstream Flex calls DEAP's `migRing`, which finds
     the individual to overwrite with `list.index(individual)` — i.e. by **tree equality**. In a
     population holding many copies of one structure (common once a good structure spreads), this
     can silently overwrite an individual that never migrated. In two runs (6-D quadrotor, 3-link)
     this deleted the numerically verified best tree at generation 10, and in a simulation 63.8 % of
     the upstream migrations lost such a tree, against 0 of 2000 for the replacement. With the
     switch on, the patch instead does a
     **position-based** ring exchange: each island samples `k` individual *positions*, sends those
     individuals to the next island in the ring, and the immigrants land in exactly the `k`
     positions their own island's emigrants vacated — so every individual either stays or moves
     exactly one island along the ring, and none is ever silently dropped.

Both switches are read inside the same migration block of `GPSymbolicRegressor`'s generation loop;
nothing else in Flex is touched, and the patch is additive (upstream's code paths remain, selected
by the environment variable being unset).

This repository sets both switches through `config.yaml` -> `src/settings.py`:
`gp_runtime.seed_islands` -> `FLEX_SEED_ISLANDS` and `gp_runtime.migration_fix.cpu` ->
`FLEX_MIGRATION_FIX` (see the rows in `docs/SETTINGS_CPU.md`). In every example's `config.yaml`,
`seed_islands` is `3` and `migration_fix.cpu` is `true`.

## Applying the patch

```sh
git clone https://github.com/cpml-au/Flex
cd Flex
git checkout 4f5746e26b8aa2a0871bee1e1c05ccd1c5c68c04
git apply /path/to/SR-CLF/docs/flex_patch/regressor_symclf.patch
pip install -e .          # or: pip install .
```

`pip show flex` in the environment these results were produced on reports version `0.2.2`
(editable install from that checkout) — see `environment.yml` / `requirements.txt`.

## What is NOT in this patch

Nothing else about Flex was modified for this repository: the primitive set, the GP operators
(crossover, mutation, selection), the fitness/evaluation interface (`fitness=`, `score_func=`,
`predict_func=`, `callback_func=`, `custom_logger=`), and `flex.gp.util.load_config_data` (which
still reads the `gp:` block of `config.yaml` directly) are all upstream Flex, unpatched.
