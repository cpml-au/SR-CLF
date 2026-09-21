#!/bin/bash -l
#SBATCH --job-name=cp4d_v10q36
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p q32g
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:3
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH hetjob
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH -p q36
#SBATCH --cpus-per-task=36
#SBATCH --mem=120G
#SBATCH --time=48:00:00
# V10 (2026-09-19, the user's order "CMA on >= 10 % of the population"): a HETEROGENEOUS job -- het group 0 =
# one q32g node (3 GPUs: the 9 GPU actors, the Ray head, run.py), het group 1 = q64 CPU nodes (64 cores each:
# single-thread det4 actors, Ray resource det4pool). The CMA stage runs pooled: the grid half of every CMA
# population on the GPU actors, the det4 half on the pool, many structures' loops in flight.
# 2026-09-17 (the user): TWO q32g nodes = 6 GPUs, 18 actors, CMA on 6 structures per generation
# (2026-09-15: pinned to s02n01 on the user's order; unpinned the same evening because another user's
#  2-day jobs took 2 of its GPUs -- q32g = s02n01 or s02n02, the same A100 nodes; add --nodelist to pin)

# ############################################################################
# CP3D -- the certified-region objective. Submit FROM this folder:
#     sbatch job4DCartPoleV8.slurm   (4DCartPolerV8 = 4DCartPolerV7 + max tree length 150)
# ONE node: 32 CPU cores and 3 GPUs (leaves the rest of PRIME free).
# ############################################################################

# Do not enable nounset before activating Conda.
set -eo pipefail

# 2026-09-16 (the user's order): the environment runs from the NODE'S LOCAL DISK. /home is an autofs NFS mount
# and was slow enough (import numba 21 s on fe1; 217 s to warm 75 Ray modules on s02n01) that Ray's helper
# processes missed the raylet's port-file windows (jobs 146795, 146879, 146880, 146896, 146947 attempt 1)
# and the job start took 25 min. ~/flex.tar.gz = `conda-pack -n flex` (one sequential read); it is extracted
# to /tmp/flex_<job> on the node, activated, and `conda-unpack` fixes the prefix paths. Ray, run.py and the
# actors then import from local disk; the project code and the editable flex package stay on /home (small).
# Without the tarball the /home env is used as before.
FLEX_TAR="${FLEX_TAR:-$HOME/flex.tar.gz}"
LOCAL_ENV="/tmp/flex_${SLURM_JOB_ID:-login}"
if [ -f "$FLEX_TAR" ]; then
    _t0=$(date +%s)
    echo "Extracting $FLEX_TAR ($(du -h "$FLEX_TAR" | cut -f1)) to $LOCAL_ENV on $(hostname)..."
    mkdir -p "$LOCAL_ENV"
    tar -xzf "$FLEX_TAR" -C "$LOCAL_ENV"
    source "$LOCAL_ENV/bin/activate"
    conda-unpack
    echo "local env ready in $(( $(date +%s) - _t0 )) s: python=$(which python) CONDA_PREFIX=$CONDA_PREFIX"
else
    echo "no $FLEX_TAR: using the /home environment"
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate flex
fi

# Conda activation has finished, so nounset can now be enabled.
set -u

# 2026-09-13: sbatch exports the submitting shell's environment (--export=ALL is the default), so a
# JAX_PLATFORMS=cpu left over from a login-node test silently forces the whole run onto CPU: job
# 146672 ran its 9 actors on TFRT_CPU_0 with all 3 GPUs at 0%. The preflight smoke below sets these
# inline for itself; the main run must never inherit them.
unset JAX_PLATFORMS SYMCLF_GPU2_ALLOW_CPU SYMCLF_GPU2_PALLAS_INTERPRET

RAY_BIN="$CONDA_PREFIX/bin/ray"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export RAY_DEDUP_LOGS=0

# --- V2: the four changes ----------------------------------------------------
# Every default was calibrated against the 3-D ARE quadratic FIRST -- it is the
# one candidate whose correctness is not in question, and under these settings
# it scores 0.175 (pure length) while the V1 champion scores 314.6.
# Verify with:  python score_v2.py   (the ARE must rank first).
export SYMCLF_V2_BOX="${SYMCLF_V2_BOX:-0.25,0.25,0.0494,0.1155}"   # 4-D: x1 +-0.25 + V6's three
export SYMCLF_V2_SHAPE_WEIGHT="${SYMCLF_V2_SHAPE_WEIGHT:-300}"
export SYMCLF_V2_ROA_RHO_MULTIPLIER="${SYMCLF_V2_ROA_RHO_MULTIPLIER:-1.0}"
export SYMCLF_V2_CERT_WEIGHT="${SYMCLF_V2_CERT_WEIGHT:-500}"
export SYMCLF_V2_VDOT_WEIGHT="${SYMCLF_V2_VDOT_WEIGHT:-0}"

# --- V3: the two changes that made the answer reachable ----------------------
# (a) positivity as a MAGNITUDE not a count; (b) local Adam refinement of the
# constants on the elite, because the SEA is random search and cannot resolve a
# direction in 6-10 dimensions. Verify with:  python gate_v3.py
export SYMCLF_V3_POSITIVITY_WEIGHT="${SYMCLF_V3_POSITIVITY_WEIGHT:-300}"

# --- 2026-09-09 DET4 MODE (see README.md) ---------------------------------------------------
# fitness = symbolic + invalid + length + ROA_RULE(c_v*, kappa*, boundary_min, w_scale);
# every other grid term below is ignored by the kernel in this mode (kept for SYMCLF_EXACT_MODE=roots).
export SYMCLF_EXACT_MODE="${SYMCLF_EXACT_MODE:-det4}"
export SYMCLF_DET4_RHO="${SYMCLF_DET4_RHO:-1000}"          # u_max in (a - rho|b|)/W
export SYMCLF_DET4_PD_EPS="${SYMCLF_DET4_PD_EPS:-1e-4}"    # PD iff c_v* > pd_eps
export SYMCLF_DET4_KAPPA_MIN="${SYMCLF_DET4_KAPPA_MIN:-0}" # certified iff kappa* > kappa_min
# V6: 10 elites by DET4 price, Adam (relative step 0.02) on the full DET4 rule, 100 DET4Fast evaluations each
# V7 (2026-09-13): 18 elites = two refines per actor (+~300 s/gen); the post-refine top-9 sat at pre-refine
# ranks 2-23 in the const_opt pool test (9 elites capture 4 of them, 18 capture 8)
export SYMCLF_DET4_REFINE_TOP="${SYMCLF_DET4_REFINE_TOP:-18}"
export SYMCLF_DET4_REFINE_STEPS="${SYMCLF_DET4_REFINE_STEPS:-100}"
export SYMCLF_DET4_REFINE_LR="${SYMCLF_DET4_REFINE_LR:-0.02}"
# V7: portfolio = Adam 0.02 and Adam 0.1 for a third of the budget each, then golden-section per constant
export SYMCLF_DET4_REFINE_METHOD="${SYMCLF_DET4_REFINE_METHOD:-portfolio}"
export SYMCLF_DET4_REFINE_PORTFOLIO_LRS="${SYMCLF_DET4_REFINE_PORTFOLIO_LRS:-0.02,0.1}"
# V7: the refine calls the GPU lattice only for DET4-certified candidates (no more 7.5 GiB OOM skips)
export SYMCLF_DET4_REFINE_LATTICE_ONLY_CERTIFIED="${SYMCLF_DET4_REFINE_LATTICE_ONLY_CERTIFIED:-1}"
export SYMCLF_DET4_REFINE_SELECT="${SYMCLF_DET4_REFINE_SELECT:-price}"
export SYMCLF_CERTIFY_EVERY="${SYMCLF_CERTIFY_EVERY:-0}"       # certify.py on the champion every N gens (report)
# --- 2026-09-10 V4: the DET4 numbers from examples/DET4Fast (det4_fast: the referee's det4 in numba,
# W = V - V(0) without cancellation). The exact checker runs as GPU5_7 (root verdict only); the
# elites' refine uses det4_fast's numbers. Scheduling, tuner and rule unchanged from V3.
export SYMCLF_DET4_FAST="${SYMCLF_DET4_FAST:-1}"
# V9 (the user's order): det4_fast threads doubled, 3 -> 6 per actor (9 actors x 6 = 54 threads on 32 cores), to measure.
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-6}"
# option B (2026-09-10): kappa* gated on PD at every level; a non-PD candidate is charged kappa* = -cap
export SYMCLF_DET4_GATE_KAPPA_CAP="${SYMCLF_DET4_GATE_KAPPA_CAP:-1e6}"
# V5 (2026-09-10): the full DET4 numbers (det4_fast) for EVERY screened candidate, priced on them
export SYMCLF_DET4_ALL="${SYMCLF_DET4_ALL:-1}"
# --- 2026-09-11 V6 (see README_DET4.md) --------------------------------------------------------
# (1) lattice Adam skipped when the SEA score is already 0 and exited at the floor / zero gradient
#     (bit-identical results); (2) elite refine = top-10 by DET4 price, Adam, 100 evaluations (above);
# (3) positivity penalties on the price, both branches, lattice rule included:
#     10*clip((0.05-c_v*)/0.05) + 30*clip((0.05-kappa*)/0.05) + 20*clip((0.02-certified volume)/0.02)
export SYMCLF_TUNER_SKIP_ADAM_AT_ZERO="${SYMCLF_TUNER_SKIP_ADAM_AT_ZERO:-1}"
export SYMCLF_TUNER_ADAM_EXIT_ZERO_GRAD="${SYMCLF_TUNER_ADAM_EXIT_ZERO_GRAD:-1}"
export SYMCLF_TUNER_ADAM_PATIENCE="${SYMCLF_TUNER_ADAM_PATIENCE:-0}"
export SYMCLF_DET4_PEN_CV="${SYMCLF_DET4_PEN_CV:-10}"
export SYMCLF_DET4_PEN_CV_TARGET="${SYMCLF_DET4_PEN_CV_TARGET:-0.05}"
export SYMCLF_DET4_PEN_KAPPA="${SYMCLF_DET4_PEN_KAPPA:-100}"     # 2026-09-17 study: the rate lever
export SYMCLF_DET4_PEN_KAPPA_TARGET="${SYMCLF_DET4_PEN_KAPPA_TARGET:-1.0}"   # LQR's own kappa* is 0.72
# 2026-09-18 (the user): the volume hinge 100 @ 0.3 turned OFF (kept here, commented); back to the V8 default 20 @ 0.02
# export SYMCLF_DET4_PEN_VOL="${SYMCLF_DET4_PEN_VOL:-100}"         # 2026-09-17: the 147074 setting (Sontag loop converged)
# export SYMCLF_DET4_PEN_VOL_TARGET="${SYMCLF_DET4_PEN_VOL_TARGET:-0.3}"
export SYMCLF_DET4_PEN_VOL="${SYMCLF_DET4_PEN_VOL:-20}"
export SYMCLF_DET4_PEN_VOL_TARGET="${SYMCLF_DET4_PEN_VOL_TARGET:-0.02}"
# V7 (2026-09-13, second part): the DET4 search repaired in a V7 copy of DET4Fast4D (the running V6 job
# keeps examples/DET4Fast4D untouched). PROJECT_ASCENT: manifold-ascent endpoints Newton-projected onto
# b = 0 before the |b| test (the point the search found was being discarded; job 144596 gen 71
# reported -0.063 for a true -0.178, gen 100 reported -0.0036 for a true -1.016). ORIGIN_SEEDS: the
# origin's worst manifold direction from the linearisation, projected, as extra points (OFF until the
# user decides; cost ~1 ms per candidate).
export SYMCLF_DET4_FAST_DIR="$PWD/../DET4Fast4D_V10"
export SYMCLF_DET4_PROJECT_ASCENT="${SYMCLF_DET4_PROJECT_ASCENT:-1}"
export SYMCLF_DET4_ORIGIN_SEEDS="${SYMCLF_DET4_ORIGIN_SEEDS:-1}"   # 2026-09-13: ON (user; ~1 ms per candidate)
# V9 (2026-09-16, the user's order): the DET4 check faster, the same algorithm and polishes: the lattice
# (2.6M points) on the GPU with the seed ranking on the device, top-k seeds instead of full argsorts, the
# dynamics fields precomputed on the lattice (CPU path). Every reported number stays from the CPU evaluator.
export SYMCLF_DET4_GPU_LATTICE="${SYMCLF_DET4_GPU_LATTICE:-0}"   # OFF (acceptance 147029: the GPU-ranked PD seeds missed a radial minimum on 144596 g22, c_v* +3.2e-5; and slower than the 6-thread CPU scan)
export SYMCLF_DET4_TOPK_SEEDS="${SYMCLF_DET4_TOPK_SEEDS:-1}"
export SYMCLF_DET4_PRECOMPUTED_FIELDS="${SYMCLF_DET4_PRECOMPUTED_FIELDS:-1}"
# --- 2026-09-15 V8 (the user's order): a tree with SYMCLF_MAX_TREE_LENGTH nodes or more is not evaluated
# (fitness 1e8); V7 and before: 400. Engine default 400; set here to 150.
# V9 (2026-09-16, the user's order): 90 nodes (V8: 150).
export SYMCLF_MAX_TREE_LENGTH="${SYMCLF_MAX_TREE_LENGTH:-90}"
echo "V9 gpu_lattice=$SYMCLF_DET4_GPU_LATTICE topk_seeds=$SYMCLF_DET4_TOPK_SEEDS precomputed_fields=$SYMCLF_DET4_PRECOMPUTED_FIELDS numba_threads=$NUMBA_NUM_THREADS max_len=$SYMCLF_MAX_TREE_LENGTH"
# V8 (2026-09-15, the user's decisions D1-D5): ONE check = det4_fast on every live tree (the cheap screen, the
# GPU root check and the GPU5 PD check are gone with their code); certified = c_v* > pd_eps and kappa* > kappa_min;
# D4: a structurally dead tree (symbolic >= 1e6, invalid V) keeps its grid price and skips the check.
export SYMCLF_DET4_SKIP_DEAD="${SYMCLF_DET4_SKIP_DEAD:-1}"
export SYMCLF_DET4_BATCH_SIZE="${SYMCLF_DET4_BATCH_SIZE:-32}"   # det4_fast candidates per actor call
echo "V8 max_tree_length=$SYMCLF_MAX_TREE_LENGTH skip_dead=$SYMCLF_DET4_SKIP_DEAD det4_batch=$SYMCLF_DET4_BATCH_SIZE"
# V8 CMA stage (2026-09-15, the user's go): pycma CMA-ES (the 146800 recipe) on ONE elite per mapper call, the
# best new structure not yet CMA-refined; popsize 0 = one evaluation per actor per CMA generation;
# 2500 evaluations ~ 11-12 min of wall time per generation (the user: 2500). 0 = off.
export SYMCLF_CMA_ELITE_EVALS="${SYMCLF_CMA_ELITE_EVALS:-2000}"   # 2026-09-18 (the user): 2500 -> 4000 (kappa* ~ 1 appeared at 2200-3000 evals)
export SYMCLF_CMA_ELITE_POPSIZE="${SYMCLF_CMA_ELITE_POPSIZE:-0}"
export SYMCLF_CMA_ELITE_SEED="${SYMCLF_CMA_ELITE_SEED:-1}"
# V9 (2026-09-16, the user's order): the CMA only when the elite's fitness is below 3000; one structure per call.
export SYMCLF_CMA_ELITE_MAX_FITNESS="${SYMCLF_CMA_ELITE_MAX_FITNESS:-3000}"
export SYMCLF_CMA_ELITE_COUNT="${SYMCLF_CMA_ELITE_COUNT:-3}"   # at least this many structures per generation
# V10: the pooled CMA stage and the fraction of the checked structures that get the CMA (the user: >= 10 %)
export SYMCLF_CMA_POOL="${SYMCLF_CMA_POOL:-1}"
export SYMCLF_CMA_ELITE_FRACTION="${SYMCLF_CMA_ELITE_FRACTION:-0.10}"
export SYMCLF_CMA_POOL_ACTORS="${SYMCLF_CMA_POOL_ACTORS:-0}"    # 0 = every det4pool slot the q64 nodes advertise
export SYMCLF_CMA_POOL_LOOPS="${SYMCLF_CMA_POOL_LOOPS:-0}"      # 0 = pool size / popsize
export SYMCLF_CMA_ELITE_POPSIZE="${SYMCLF_CMA_ELITE_POPSIZE:-10}"   # the 146800 recipe (not one per GPU actor)
echo "V9 cma_elite evals=$SYMCLF_CMA_ELITE_EVALS popsize=$SYMCLF_CMA_ELITE_POPSIZE seed=$SYMCLF_CMA_ELITE_SEED max_fitness=$SYMCLF_CMA_ELITE_MAX_FITNESS count=$SYMCLF_CMA_ELITE_COUNT"
# V10 2026-09-19 (the user's order, items 1-3; the code defaults are off, V9 behaviour):
#  1 constant inheritance: an offspring's parent constants are scored after the SEA and kept when strictly better
#  2 the CMA stage takes the structures closest to certification (PD, highest DET4 kappa*) instead of the best fitness
#  3 the per-structure compiled det4 lattice scan in the CMA pool (bit-identical, ~2.7x per det4 call at 1 thread)
#  + no Adam refine on structurally dead trees (the CMA never takes dead trees)
export SYMCLF_INHERIT_CONSTS="${SYMCLF_INHERIT_CONSTS:-1}"
export SYMCLF_CMA_SELECT="${SYMCLF_CMA_SELECT:-kappa}"
export SYMCLF_CMA_POOL_JIT="${SYMCLF_CMA_POOL_JIT:-1}"
export SYMCLF_SKIP_DEAD_ADAM="${SYMCLF_SKIP_DEAD_ADAM:-1}"
echo "V10 inherit_consts=$SYMCLF_INHERIT_CONSTS cma_select=$SYMCLF_CMA_SELECT cma_pool_jit=$SYMCLF_CMA_POOL_JIT skip_dead_adam=$SYMCLF_SKIP_DEAD_ADAM cma_pool=${SYMCLF_CMA_POOL:-0}"
# V10 2026-09-19 (the user's order after pilot 147839: "P1 max 2500", "do P2", "6 deep, 20 x 500 continuous"):
# the CMA budget is a LIFETIME cap per structure, spent in slices that RESUME the structure's own search (pycma state
# + numpy RNG state pickled; a resumed run is bit-identical to a continuous one). The V9 once-per-structure rule does
# not apply in this mode. Why: pilot 147839 spent 500 evaluations on each of 37 structures and certified none, and the
# studies (147053/147074) show no certification before eval 1397 (V9's own came at 2064).
export SYMCLF_CMA_RESUMABLE="${SYMCLF_CMA_RESUMABLE:-1}"
export SYMCLF_CMA_MAX_EVALS_PER_STRUCTURE="${SYMCLF_CMA_MAX_EVALS_PER_STRUCTURE:-2500}"
export SYMCLF_CMA_DEEP_COUNT="${SYMCLF_CMA_DEEP_COUNT:-6}"
export SYMCLF_CMA_DEEP_EVALS="${SYMCLF_CMA_DEEP_EVALS:-2500}"
export SYMCLF_CMA_SLICE_COUNT="${SYMCLF_CMA_SLICE_COUNT:-20}"
export SYMCLF_CMA_SLICE_EVALS="${SYMCLF_CMA_SLICE_EVALS:-500}"
# 2026-09-20: 9 was fmin2's default and it is spent within one slice at popsize 18; a resumed slice then
# returned 0 evaluations forever (jobs 147921/147927/147924).
export SYMCLF_CMA_MAX_RESTARTS="${SYMCLF_CMA_MAX_RESTARTS:-100}"
echo "V10 cma resumable=$SYMCLF_CMA_RESUMABLE cap=$SYMCLF_CMA_MAX_EVALS_PER_STRUCTURE deep=$SYMCLF_CMA_DEEP_COUNT x $SYMCLF_CMA_DEEP_EVALS slices=$SYMCLF_CMA_SLICE_COUNT x $SYMCLF_CMA_SLICE_EVALS"
python -c "import cma; print('cma', cma.__version__)"
echo "V7 lattice_only_certified=$SYMCLF_DET4_REFINE_LATTICE_ONLY_CERTIFIED portfolio_lrs=$SYMCLF_DET4_REFINE_PORTFOLIO_LRS det4_dir=$SYMCLF_DET4_FAST_DIR project_ascent=$SYMCLF_DET4_PROJECT_ASCENT origin_seeds=$SYMCLF_DET4_ORIGIN_SEEDS"
echo "V6 DET4_FAST=$SYMCLF_DET4_FAST numba_threads=$NUMBA_NUM_THREADS gate_kappa_cap=$SYMCLF_DET4_GATE_KAPPA_CAP det4_all=$SYMCLF_DET4_ALL refine=$SYMCLF_DET4_REFINE_METHOD/$SYMCLF_DET4_REFINE_SELECT top$SYMCLF_DET4_REFINE_TOP x$SYMCLF_DET4_REFINE_STEPS lr$SYMCLF_DET4_REFINE_LR pen cv=$SYMCLF_DET4_PEN_CV@$SYMCLF_DET4_PEN_CV_TARGET kappa=$SYMCLF_DET4_PEN_KAPPA@$SYMCLF_DET4_PEN_KAPPA_TARGET vol=$SYMCLF_DET4_PEN_VOL@$SYMCLF_DET4_PEN_VOL_TARGET tuner_skip=$SYMCLF_TUNER_SKIP_ADAM_AT_ZERO/$SYMCLF_TUNER_ADAM_EXIT_ZERO_GRAD"
export SYMCLF_TUNER_CONST_RANGE="${SYMCLF_TUNER_CONST_RANGE:-30}"
export SYMCLF_TUNER_CONST_PERTURB_FRAC="${SYMCLF_TUNER_CONST_PERTURB_FRAC:-0.5}"
export SYMCLF_TUNER_REFINE_STEPS="${SYMCLF_TUNER_REFINE_STEPS:-40}"
export SYMCLF_TUNER_REFINE_TOP_K="${SYMCLF_TUNER_REFINE_TOP_K:-4}"
export SYMCLF_TUNER_REFINE_LR="${SYMCLF_TUNER_REFINE_LR:-0.30}"
echo "V3 positivity=$SYMCLF_V3_POSITIVITY_WEIGHT const_range=$SYMCLF_TUNER_CONST_RANGE refine=${SYMCLF_TUNER_REFINE_STEPS}x top${SYMCLF_TUNER_REFINE_TOP_K}"
echo "V3 box=$SYMCLF_V2_BOX shape=$SYMCLF_V2_SHAPE_WEIGHT rho_mult=$SYMCLF_V2_ROA_RHO_MULTIPLIER cert=$SYMCLF_V2_CERT_WEIGHT vdot=$SYMCLF_V2_VDOT_WEIGHT"

# --- runtime scaling ---------------------------------------------------------
if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
    GPUS_PER_NODE="$SLURM_GPUS_ON_NODE"
elif [ -n "${SLURM_JOB_GPUS:-}" ]; then
    GPUS_PER_NODE=$(echo "$SLURM_JOB_GPUS" | tr ',' '\n' | grep -c .)
elif command -v nvidia-smi >/dev/null 2>&1; then
    GPUS_PER_NODE=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU" || echo 3)
else
    GPUS_PER_NODE=3
fi
[ "${GPUS_PER_NODE:-0}" -ge 1 ] 2>/dev/null || GPUS_PER_NODE=3
export SYMCLF_GPU2_TOTAL_GPUS="${SYMCLF_GPU2_TOTAL_GPUS:-$GPUS_PER_NODE}"   # het group 0 = one GPU node
export SYMCLF_GPU3_ACTORS_PER_GPU="${SYMCLF_GPU3_ACTORS_PER_GPU:-3}"

# --- GPU5_7: the certified-region objective (SAT / ROA: read by the lattice kernel) -------------
# Saturation: u_required = max a/|b| charged per decade above the target.
# Calibrated: 96051 = 57, 122304 champ = 59, 122340 gen-7 (diverges) = 1.46e5.
export SYMCLF_CP3D_SAT_U_TARGET="${SYMCLF_CP3D_SAT_U_TARGET:-1000}"
export SYMCLF_CP3D_SAT_WEIGHT="${SYMCLF_CP3D_SAT_WEIGHT:-150}"
# ROA: certified level c_max + certified volume vs a 0.5%-of-box target
# (best known certified volume is 96051's 0.657%).
export SYMCLF_CP3D_ROA_WEIGHT="${SYMCLF_CP3D_ROA_WEIGHT:-500}"
export SYMCLF_CP3D_ROA_C_TARGET="${SYMCLF_CP3D_ROA_C_TARGET:-0.004}"
export SYMCLF_CP3D_ROA_NEG_WEIGHT="${SYMCLF_CP3D_ROA_NEG_WEIGHT:-250}"
# Priors OFF (knobs kept for ablation runs).
export SYMCLF_CP3D_PROPERNESS_WEIGHT="${SYMCLF_CP3D_PROPERNESS_WEIGHT:-0}"
export SYMCLF_CP3D_V_GRAD_WEIGHT="${SYMCLF_CP3D_V_GRAD_WEIGHT:-0}"
export SYMCLF_CP3D_C_STAR_PENALTY_MAX="${SYMCLF_CP3D_C_STAR_PENALTY_MAX:-0}"
export SYMCLF_CP3D_C_STAR_PENALTY_CAP="${SYMCLF_CP3D_C_STAR_PENALTY_CAP:-0}"
export SYMCLF_CP3D_C_STAR_TAIL_WEIGHT="${SYMCLF_CP3D_C_STAR_TAIL_WEIGHT:-0}"
export SYMCLF_CP3D_COVERAGE_WEIGHT="${SYMCLF_CP3D_COVERAGE_WEIGHT:-0}"

# --- PD tolerance (the DET4_PD_EPS default) --------------------------------
export SYMCLF_GPU5_PD_EPS="${SYMCLF_GPU5_PD_EPS:-1e-4}"

export SYMCLF_CP3D_NESTED_PENALTY="${SYMCLF_CP3D_NESTED_PENALTY:-50}"
export SYMCLF_CP3D_FLAT_GRAD_FLOOR="${SYMCLF_CP3D_FLAT_GRAD_FLOOR:-1e-3}"
export SYMCLF_CP3D_FLAT_GRAD_WEIGHT="${SYMCLF_CP3D_FLAT_GRAD_WEIGHT:-1.0}"
export SYMCLF_CP3D_AXIS_B_TARGET="${SYMCLF_CP3D_AXIS_B_TARGET:-1e-3}"
export SYMCLF_CP3D_AXIS_B_WEIGHT="${SYMCLF_CP3D_AXIS_B_WEIGHT:-300}"
export SYMCLF_CP3D_AXIS_V_TARGET="${SYMCLF_CP3D_AXIS_V_TARGET:-1e-2}"
export SYMCLF_CP3D_AXIS_V_WEIGHT="${SYMCLF_CP3D_AXIS_V_WEIGHT:-300}"
export SYMCLF_CP3D_AXIS_TAIL_WEIGHT="${SYMCLF_CP3D_AXIS_TAIL_WEIGHT:-0.5}"
export SYMCLF_CP3D_ORIGIN_PENALTY_WEIGHT="${SYMCLF_CP3D_ORIGIN_PENALTY_WEIGHT:-0}"

# --- tuner knobs (the lattice SEA + Adam, unchanged) -------------------------
export SYMCLF_GPU2_BATCH_SIZE="${SYMCLF_GPU2_BATCH_SIZE:-16}"
# 21^4 grid: tuner fusion 2 + eval row chunk 4 keep the device allocation at
# ~3.4 GiB against the 11.2 GiB/actor budget (the 6.9 GiB request killed
# 122252/122299/122339; the fitness-level OOM retry is the second line of
# defence).
export SYMCLF_GPU2_TUNER_FUSION="${SYMCLF_GPU2_TUNER_FUSION:-2}"
export SYMCLF_GPU2_TUNER_RANDOM_POPULATION="${SYMCLF_GPU2_TUNER_RANDOM_POPULATION:-35}"
export SYMCLF_GPU2_TUNER_GENERATIONS="${SYMCLF_GPU2_TUNER_GENERATIONS:-5}"
export SYMCLF_GPU2_PALLAS_BLOCK_SIZE="${SYMCLF_GPU2_PALLAS_BLOCK_SIZE:-128}"
export SYMCLF_GPU2_EVAL_ROW_CHUNK="${SYMCLF_GPU2_EVAL_ROW_CHUNK:-4}"
export SYMCLF_V8_FINAL_AUDIT="${SYMCLF_V8_FINAL_AUDIT:-1}"   # the CPU referee on the champion at the end
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.28}"
export SYMCLF_GPU2_JAX_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/symclf_cp4dv10x2_${SLURM_JOB_ID}"
# ---------------------------------------------------------------------------

cd "${SLURM_SUBMIT_DIR:-$PWD}"

PROJECT_ROOT=$(cd ../.. && pwd)
export PYTHONPATH="$PWD:$PROJECT_ROOT:${PYTHONPATH:-}"

# 2026-09-09: CPU preflight before Ray (as the E25 slurm does): imports, the DET4 grid ledger,
# cheap/full DET4 and the root verdict on the ARE quadratic (~30 s). set -e aborts the job on FAIL.
# 2026-09-15: the preflight is switchable (SYMCLF_PREFLIGHT=0 skips it: ~15-20 min on the node; the same code
# must have passed the smoke on the login node the same day).
# 2026-09-18 (the user): preflight OFF by default in this 2-node slurm (two smoke expectations predate the new hinges)
export SYMCLF_PREFLIGHT="${SYMCLF_PREFLIGHT:-0}"
if [ "${SYMCLF_PREFLIGHT:-1}" = "1" ]; then
JAX_PLATFORMS=cpu SYMCLF_GPU2_ALLOW_CPU=1 SYMCLF_GPU2_PALLAS_INTERPRET=1 \
    SYMCLF_GPU2_TUNER_RANDOM_POPULATION=4 SYMCLF_GPU2_TUNER_GENERATIONS=1 SYMCLF_TUNER_REFINE_STEPS=0 \
    python smoke_v3_det4.py | grep -v "^PASS" || { echo "PREFLIGHT smoke_v3_det4 FAILED"; exit 1; }
else
    echo "PREFLIGHT skipped (SYMCLF_PREFLIGHT=0)"
fi

echo "Project root: $PROJECT_ROOT"
echo "PYTHONPATH: $PYTHONPATH"
echo "Conda environment: $CONDA_PREFIX"
echo "Python: $(which python)"

python --version
python -c "import ray; print('Ray Python package:', ray.__version__, ray.__file__)"
python -c "import jax; print('JAX:', jax.__version__, 'devices:', jax.devices())"
nvidia-smi -L || echo "nvidia-smi not available on submit node"

echo "Ray CLI: $RAY_BIN"
"$RAY_BIN" --version

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0")
readarray -t nodes_array <<< "$nodes"
head_node="${nodes_array[0]}"
readarray -t worker_nodes <<< "$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_1")"
echo "het group 0 (GPU): $head_node | het group 1 (det4 pool): ${worker_nodes[*]}"
CPUS_HEAD="${SLURM_CPUS_PER_TASK_HET_GROUP_0:-32}"
CPUS_WORKER="${SLURM_CPUS_PER_TASK_HET_GROUP_1:-36}"
# 2026-09-17: the node-local environment on every worker node too (same path /tmp/flex_<job>, same job id).
if [ -f "$FLEX_TAR" ]; then
    for _wn in "${worker_nodes[@]}"; do
        _t0=$(date +%s)
        srun --het-group=1 --nodes=1 --ntasks=1 -w "$_wn" bash -c \
            "mkdir -p '$LOCAL_ENV' && tar -xzf '$FLEX_TAR' -C '$LOCAL_ENV' && source '$LOCAL_ENV/bin/activate' && conda-unpack"
        echo "local env ready on $_wn in $(( $(date +%s) - _t0 )) s"
    done
fi
# Unique Ray port per job (hardcoded 6379 collides across concurrent jobs).
# 2026-09-09: the head port must stay outside Ray's worker-port range 10002-19999 (job 126149 got
# 12529 from the old 6380 + job%20000 formula and Ray refused to start); same formula as the E25 slurm.
port=$(( 20000 + (${SLURM_JOB_ID:-0} % 30000) ))

head_ip=$(
    srun --het-group=0 --nodes=1 --ntasks=1 -w "$head_node" \
        bash -c "hostname --ip-address | awk '{print \$1}'"
)

head_address="$head_ip:$port"
export RAY_ADDRESS="$head_address"

ray_pids=()
telemetry_pids=()

cleanup() {
    status=$?

    trap - EXIT INT TERM
    set +e
    set +u

    echo "Stopping Ray Slurm steps..."

    if [ "${#ray_pids[@]}" -gt 0 ]; then
        kill "${ray_pids[@]}" 2>/dev/null
        sleep 5

        for pid in "${ray_pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null
            fi
        done

        wait "${ray_pids[@]}" 2>/dev/null
    fi

    if [ "${#telemetry_pids[@]}" -gt 0 ]; then
        kill "${telemetry_pids[@]}" 2>/dev/null
        wait "${telemetry_pids[@]}" 2>/dev/null
    fi

    # 2026-09-16: the node-local environment is removed at the end (it lives in /tmp/flex_<job>).
    if [ -n "${LOCAL_ENV:-}" ] && [ -d "$LOCAL_ENV" ] && [ -f "${FLEX_TAR:-}" ]; then
        rm -rf "$LOCAL_ENV" && echo "removed $LOCAL_ENV"
        for _wn in "${worker_nodes[@]}"; do srun --het-group=1 --nodes=1 --ntasks=1 --overlap -w "$_wn" rm -rf "$LOCAL_ENV" 2>/dev/null && echo "removed $LOCAL_ENV on $_wn"; done
    fi

    exit "$status"
}

trap cleanup EXIT INT TERM

echo "Allocated nodes: ${nodes_array[*]}"
echo "Head node: $head_node"
echo "Head IP: $head_ip"
echo "Head address: $head_address"

# 2026-09-11 (job 144451): the head died at start ("ray_client_server [exit code=1]") on a node freed
# seconds earlier by the previous job; kill any leftover Ray process of ours on the node first.
srun --het-group=0 --nodes=1 --ntasks=1 --overlap -w "$head_node" "$RAY_BIN" stop --force >/dev/null 2>&1 || true
sleep 5

# 2026-09-15 (jobs 146795, 146879, 146880): the raylet aborted in WaitForDashboardAgentPorts because Ray's
# dashboard agent took > 100 s to import its modules from /home (cold NFS: `import ray` 13 s cold, 1.2 s
# warm; the agent reached module loading 158 s after the head start). Two measures: the raylet/agent
# startup timeouts raised to 10 min, and the dashboard imports warmed on the head node right before
# `ray start` so the agent hits the page cache.
export RAY_agent_register_timeout_ms="${RAY_agent_register_timeout_ms:-600000}"
export RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s:-600}"
echo "Warming Ray's dashboard imports on $head_node..."
srun --het-group=0 --nodes=1 --ntasks=1 --overlap -w "$head_node" python - <<'PYWARM' || echo "warm-up failed (continuing)"
import importlib, pkgutil, time
t = time.time()
import ray, ray.dashboard.agent, ray.dashboard.head  # noqa
# 2026-09-16 (job 146947): the raylet also waits ~20 s for the RUNTIME-ENV agent's port file; warm its
# imports and those of the other helper processes too.
for name in ("ray._private.runtime_env.agent.main", "ray._private.log_monitor",
             "ray.util.client.server", "ray.autoscaler._private.monitor"):
    try:
        importlib.import_module(name)
    except Exception:
        pass
import ray.dashboard.modules as m
n = 0
for _, name, _ in pkgutil.walk_packages(m.__path__, m.__name__ + "."):
    try:
        importlib.import_module(name); n += 1
    except Exception:
        pass
print(f"warmed ray + {n} dashboard modules in {time.time() - t:.1f} s", flush=True)
PYWARM
# 2026-09-16 (job 146896 on s02n01): the head step died after 10.6 min ("Failed to get node info: Deadline
# Exceeded", the raylet never registered; the warm-up alone took 217 s, /home was slow) while the readiness
# loop kept waiting. Now: up to 3 head attempts; an attempt ends when `ray status` shows an active node
# (success), when the head step has exited (failure, detected at once), or after 12 min. On failure the
# Ray session logs are copied to ray_logs_<job>_<attempt>/ (they live in the node's /tmp), Ray is stopped,
# and the head is started again (the page cache is warm by then). Single node: no worker loop.
_ready=0
for _attempt in 1 2 3; do
    echo "Starting Ray head on $head_node (attempt $_attempt)..."
    srun \
        --het-group=0 \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="$CPUS_HEAD" \
        --gres=gpu:"$GPUS_PER_NODE" \
        --exclusive \
        -w "$head_node" \
        "$RAY_BIN" start \
            --head \
            --node-ip-address="$head_ip" \
            --port="$port" \
            --include-dashboard=false \
            --num-cpus="$CPUS_HEAD" \
            --num-gpus="$GPUS_PER_NODE" \
            --block &
    _head_pid=$!
    ray_pids+=("$_head_pid")
    sleep 20
    echo "Ray cluster status:"
    for _try in $(seq 1 72); do
        if "$RAY_BIN" status 2>/dev/null | grep -q "Active:"; then _ready=1; break; fi
        if ! kill -0 "$_head_pid" 2>/dev/null; then echo "  ray head step exited (attempt $_attempt)"; break; fi
        echo "  ray head not ready yet (attempt $_attempt, try $_try, $((_try * 10)) s)"; sleep 10
    done
    if [ "$_ready" = "1" ]; then break; fi
    echo "RAY HEAD attempt $_attempt FAILED -- saving logs, stopping Ray, retrying"
    srun --het-group=0 --nodes=1 --ntasks=1 --overlap -w "$head_node" bash -c \
        "cp -r /tmp/ray/session_latest/logs '$PWD/ray_logs_${SLURM_JOB_ID}_${_attempt}' 2>/dev/null; $RAY_BIN stop --force" \
        >/dev/null 2>&1 || true
    kill "$_head_pid" 2>/dev/null || true
    wait "$_head_pid" 2>/dev/null || true
    sleep 10
done
"$RAY_BIN" status || true
if [ "$_ready" != "1" ]; then echo "RAY HEAD NEVER REGISTERED A NODE (3 attempts) -- aborting"; exit 1; fi

# 2026-09-17: Ray workers on the other nodes (3 GPUs, 32 CPUs each), then wait until every node is registered.
for _wn in "${worker_nodes[@]}"; do
    echo "Starting Ray worker on $_wn..."
    srun \
        --het-group=1 \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="$CPUS_WORKER" \
        --exclusive \
        -w "$_wn" \
        "$RAY_BIN" start \
            --address="$head_address" \
            --num-cpus="$CPUS_WORKER" \
            --num-gpus=0 \
            --resources="{\"det4pool\": $CPUS_WORKER}" \
            --block &
    ray_pids+=("$!")
done
_want=$(( 1 + ${#worker_nodes[@]} ))
for _try in $(seq 1 60); do
    _n=$("$RAY_BIN" status 2>/dev/null | grep -c "node_" || true)
    if [ "$_n" -ge "$_want" ]; then echo "all $_want Ray nodes registered"; break; fi
    echo "  $_n/$_want Ray nodes registered (try $_try)"; sleep 10
done
"$RAY_BIN" status || true
[ "$_n" -ge "$_want" ] || { echo "RAY WORKERS NEVER REGISTERED -- aborting"; exit 1; }

# Five-second per-GPU telemetry on each node (overlapping one-CPU step).
for node in "$head_node"; do
    telemetry_file="gpu_telemetry_${SLURM_JOB_ID}_${node}.csv"
    srun \
        --het-group=0 \
        --overlap \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=1 \
        -w "$node" \
        nvidia-smi \
            --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw \
            --format=csv \
            -l 5 > "$telemetry_file" &
    telemetry_pids+=("$!")
done

echo "Starting run.py (4DCartPolerV10 x 2 nodes: V8 + GPU lattice det4, top-k seeds, precomputed fields, length 90, 6 threads)..."
python run.py
# 2026-09-18 (the user): diagnostics only (NOT priced): g_norm, kappa_x, gamma of every generation champion
python diag_gnorm.py "${SLURM_JOB_ID}_best_per_generation.jsonl" > "${SLURM_JOB_ID}_diagnostics.txt" 2>&1 \
    || echo "diag_gnorm failed (non-fatal)"
