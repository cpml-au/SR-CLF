#!/bin/bash -l
# ############################################################################
#  6-D GP (2026-09-19, the user's order): symbolic regression of a CLF for the 6-D quadrotor with the 4-D V9
#  pipeline (gp6d_engine.py: S1 SEA + Adam tuner on the lattice stage, S2 the DET4 check, S2b CMA, V8 price),
#  CPU only: a Ray cluster over q36 nodes, one single-thread actor per core. No GPU is needed (the 6-D lattice
#  and check are numba/scipy on the CPU).
#      cd examples/6DGP && sbatch job6DGP.slurm
# ############################################################################
#SBATCH --job-name=gp6d
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH -p q36
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=180G
#SBATCH --time=48:00:00

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# --- node-local environment on every node (the /home imports are slow; examples/4DCartPolerV9 lesson) ---
FLEX_TAR="${FLEX_TAR:-$HOME/flex.tar.gz}"
LOCAL_ENV="/tmp/flex_${SLURM_JOB_ID}"
_t0=$(date +%s)
srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 --cpus-per-task=1 bash -c \
    "mkdir -p '$LOCAL_ENV' && tar -xzf '$FLEX_TAR' -C '$LOCAL_ENV' && source '$LOCAL_ENV/bin/activate' && conda-unpack"
source "$LOCAL_ENV/bin/activate"
echo "local env ready on all nodes in $(( $(date +%s) - _t0 )) s: python=$(which python)"
set -u

# --- the 6-D system and fitness (examples/6D/env.sh values) + the latest V9 hinges (2026-09-18) ---
source ./env.sh
export SYMCLF_DET4_PEN_KAPPA="${SYMCLF_DET4_PEN_KAPPA_OVERRIDE:-100}" SYMCLF_DET4_PEN_KAPPA_TARGET="${SYMCLF_DET4_PEN_KAPPA_TARGET_OVERRIDE:-1.0}"
export SYMCLF_DET4_PEN_VOL="${SYMCLF_DET4_PEN_VOL_OVERRIDE:-20}" SYMCLF_DET4_PEN_VOL_TARGET="${SYMCLF_DET4_PEN_VOL_TARGET_OVERRIDE:-0.02}"
# --- the 4-D V9 GP settings ---
# 2026-09-19 (the user, option a): 190 = the 21-monomial template (125 nodes) + 50 % headroom, as 90 is for the
# 4-D template (59 nodes)
export SYMCLF_MAX_TREE_LENGTH="${SYMCLF_MAX_TREE_LENGTH:-190}"
export SYMCLF6D_SKIP_DEAD_ADAM="${SYMCLF6D_SKIP_DEAD_ADAM:-1}"   # the user: no Adam (and no CMA) on dead trees
export SYMCLF_GPU2_TUNER_RANDOM_POPULATION="${SYMCLF_GPU2_TUNER_RANDOM_POPULATION:-35}"
export SYMCLF_GPU2_TUNER_GENERATIONS="${SYMCLF_GPU2_TUNER_GENERATIONS:-5}"
export SYMCLF_TUNER_CONST_RANGE="${SYMCLF_TUNER_CONST_RANGE:-30}"
export SYMCLF_TUNER_CONST_PERTURB_FRAC="${SYMCLF_TUNER_CONST_PERTURB_FRAC:-0.5}"
export SYMCLF_TUNER_REFINE_STEPS="${SYMCLF_TUNER_REFINE_STEPS:-40}"
export SYMCLF_TUNER_REFINE_LR="${SYMCLF_TUNER_REFINE_LR:-0.30}"
export SYMCLF_TUNER_SKIP_ADAM_AT_ZERO="${SYMCLF_TUNER_SKIP_ADAM_AT_ZERO:-1}"
export SYMCLF_TUNER_ADAM_EXIT_ZERO_GRAD="${SYMCLF_TUNER_ADAM_EXIT_ZERO_GRAD:-1}"
export SYMCLF_TUNER_ADAM_PATIENCE="${SYMCLF_TUNER_ADAM_PATIENCE:-0}"
export SYMCLF_CP3D_NESTED_PENALTY="${SYMCLF_CP3D_NESTED_PENALTY:-50}"
# 2026-09-19: CMA on the 6-D template (job 147679, 2 seeds x 20000 evals): certified at eval 28-42, the fitness floor
# 0.625 (all hinges 0, only the length term) reached at eval 1231 / 1670, nothing gained after 2000 -> 2000
export SYMCLF_CMA_ELITE_EVALS="${SYMCLF_CMA_ELITE_EVALS:-2000}"
export SYMCLF_CMA_ELITE_COUNT="${SYMCLF_CMA_ELITE_COUNT:-3}"
export SYMCLF_CMA_ELITE_POPSIZE="${SYMCLF_CMA_ELITE_POPSIZE:-18}"
export SYMCLF_CMA_ELITE_SEED="${SYMCLF_CMA_ELITE_SEED:-2}"
# V10 port (2026-09-20, the user's order "adopt the GPs to the V10"): capped / sliced / RESUMABLE CMA instead of
# one fmin2 run per structure ever -- the structures are ranked by how close the check puts them to certification,
# the budget is a lifetime cap spent in slices that resume the structure's own pycma state, and the first elite of a
# generation may go to 5000 (certification lands near eval 2000 and the fitness falls in the evaluations after it).
export SYMCLF_CMA_RESUMABLE="${SYMCLF_CMA_RESUMABLE:-1}"
export SYMCLF_CMA_SELECT="${SYMCLF_CMA_SELECT:-kappa}"
export SYMCLF_CMA_MAX_EVALS_PER_STRUCTURE="${SYMCLF_CMA_MAX_EVALS_PER_STRUCTURE:-2500}"
export SYMCLF_CMA_FIRST_MAX_EVALS="${SYMCLF_CMA_FIRST_MAX_EVALS:-5000}"
export SYMCLF_CMA_DEEP_COUNT="${SYMCLF_CMA_DEEP_COUNT:-6}"
export SYMCLF_CMA_DEEP_EVALS="${SYMCLF_CMA_DEEP_EVALS:-2500}"
export SYMCLF_CMA_SLICE_COUNT="${SYMCLF_CMA_SLICE_COUNT:-20}"
export SYMCLF_CMA_SLICE_EVALS="${SYMCLF_CMA_SLICE_EVALS:-500}"
# 2026-09-20: 9 was fmin2's default and it is spent within one slice at popsize 18; a resumed slice then
# returned 0 evaluations forever (jobs 147921/147927/147924).
export SYMCLF_CMA_MAX_RESTARTS="${SYMCLF_CMA_MAX_RESTARTS:-100}"
echo "V10 port: cma resumable=$SYMCLF_CMA_RESUMABLE select=$SYMCLF_CMA_SELECT cap=$SYMCLF_CMA_MAX_EVALS_PER_STRUCTURE first_cap=$SYMCLF_CMA_FIRST_MAX_EVALS deep=$SYMCLF_CMA_DEEP_COUNT x $SYMCLF_CMA_DEEP_EVALS slices=$SYMCLF_CMA_SLICE_COUNT x $SYMCLF_CMA_SLICE_EVALS"
export SYMCLF_CMA_ELITE_MAX_FITNESS="${SYMCLF_CMA_ELITE_MAX_FITNESS:-3000}"
export FLEX_MIGRATION_FIX="${FLEX_MIGRATION_FIX:-1}"
# 2026-09-20 (the user): the seeds go into 3 of the 5 islands (Flex FLEX_SEED_ISLANDS, default 1 = island 0 only)
export FLEX_SEED_ISLANDS="${FLEX_SEED_ISLANDS:-3}"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD:$(cd ../.. && pwd):${PYTHONPATH:-}"
env | grep -E "^SYMCLF|^FLEX_MIG" | sort | tr '\n' ' ' | cut -c1-1500; echo

# --- numba cache warm-up once, before 36 x N actors compile concurrently ---
python -c "
import numpy as np
from core.lattice import lattice_numbers
from core.rule import fitness_fixed_constants
from core.templates import default_template
e, x0, _ = default_template()
lattice_numbers(e, x0); fitness_fixed_constants(e, x0); print('numba warm-up done')
" 2>&1 | grep -v "Warning\|warn"

# --- Ray: head on the first node, workers on the rest (CPU only) ---
RAY_BIN="$CONDA_PREFIX/bin/ray"
readarray -t nodes_array <<< "$(scontrol show hostnames "$SLURM_JOB_NODELIST")"
head_node="${nodes_array[0]}"
worker_nodes=("${nodes_array[@]:1}")
port=$(( 20000 + (${SLURM_JOB_ID:-0} % 30000) ))
head_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" bash -c "hostname --ip-address | awk '{print \$1}'")
export RAY_ADDRESS="$head_ip:$port"
ray_pids=()
cleanup() {
    status=$?; trap - EXIT INT TERM; set +e; set +u
    [ "${#ray_pids[@]}" -gt 0 ] && kill "${ray_pids[@]}" 2>/dev/null
    srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 --overlap rm -rf "$LOCAL_ENV" 2>/dev/null
    exit "$status"
}
trap cleanup EXIT INT TERM

_ready=0
for _attempt in 1 2 3; do
    echo "Starting Ray head on $head_node (attempt $_attempt)..."
    srun --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --exclusive -w "$head_node" \
        "$RAY_BIN" start --head --node-ip-address="$head_ip" --port="$port" --include-dashboard=false \
        --num-cpus="$SLURM_CPUS_PER_TASK" --num-gpus=0 --block &
    _head_pid=$!; ray_pids+=("$_head_pid"); sleep 20
    for _try in $(seq 1 72); do
        if "$RAY_BIN" status 2>/dev/null | grep -q "Active:"; then _ready=1; break; fi
        if ! kill -0 "$_head_pid" 2>/dev/null; then echo "  head step exited"; break; fi
        sleep 10
    done
    [ "$_ready" = "1" ] && break
    echo "RAY HEAD attempt $_attempt FAILED -- retrying"
    srun --nodes=1 --ntasks=1 --overlap -w "$head_node" "$RAY_BIN" stop --force >/dev/null 2>&1 || true
    kill "$_head_pid" 2>/dev/null || true; sleep 10
done
[ "$_ready" = "1" ] || { echo "RAY HEAD NEVER CAME UP -- aborting"; exit 1; }
for _wn in "${worker_nodes[@]}"; do
    srun --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --exclusive -w "$_wn" \
        "$RAY_BIN" start --address="$RAY_ADDRESS" --num-cpus="$SLURM_CPUS_PER_TASK" --num-gpus=0 --block &
    ray_pids+=("$!")
done
_want=${#nodes_array[@]}
for _try in $(seq 1 60); do
    _n=$("$RAY_BIN" status 2>/dev/null | grep -c "node_" || true)
    [ "$_n" -ge "$_want" ] && { echo "all $_want Ray nodes registered"; break; }
    sleep 10
done
"$RAY_BIN" status || true

echo "Starting run.py (6-D GP) $(date +%T)"
python run.py
