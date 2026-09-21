#!/bin/bash -l
# ############################################################################
#  6-D quadrotor CLF: CMA-ES on the quadratic template with the exact fitness.
#  GPU node (gn-*): used purely as a CPU node -- 128 cores on gn-[1003-1004],
#  64 on gn-[1001-1002]. The objective has no GPU stage (see README, "Why no GPU").
#
#  EDIT THE TWO LINES MARKED >>> for this cluster's partition/account, then:
#      sbatch slurm/cma_gpu.slurm
# ############################################################################
#SBATCH --job-name=clf6d_cma_gn
#SBATCH --output=runs/cma_gpu_%j.log
#SBATCH --error=runs/cma_gpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=08:00:00
# >>> #SBATCH -p <gpu partition, e.g. gn>
# >>> #SBATCH -A <account>

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}/.."          # the folder root (slurm/ lives one level down)
FOLDER=$PWD
source ./env.sh

# --- python environment: EDIT if this cluster does not use conda ------------
if [ -n "${CLF6D_ENV:-}" ]; then
    source "$CLF6D_ENV"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate "${CLF6D_CONDA:-flex}"
fi
python -c "import numpy, scipy, numba, sympy, cma; print('deps ok', numpy.__version__, scipy.__version__, numba.__version__, cma.__version__)"

CORES="${SLURM_CPUS_PER_TASK:-128}"
SEEDS="${SEEDS:-1 2}"                     # concurrent CMA seeds (independent runs)
POPSIZE="${POPSIZE:-10}"                    # the 146800 recipe
BUDGET="${BUDGET:-3000}"
NSEEDS=$(echo "$SEEDS" | wc -w)
WORKERS="${WORKERS:-$POPSIZE}"              # one worker per population member
THREADS="${THREADS:-$(( CORES / (NSEEDS * WORKERS) ))}"
[ "$THREADS" -lt 1 ] && THREADS=1
echo "node $(hostname) cores $CORES: $NSEEDS seeds x $WORKERS workers x $THREADS numba threads"

# preflight: the acceptance + 6-D self-test on THIS node (~2 min)
if [ "${SMOKE:-1}" = "1" ]; then
    NUMBA_NUM_THREADS=$THREADS python smoke.py || { echo "SMOKE FAILED"; exit 1; }
fi

pids=()
for s in $SEEDS; do
    python cma_run.py --budget "$BUDGET" --popsize "$POPSIZE" --workers "$WORKERS" \
        --threads "$THREADS" --seed "$s" --template "$SYMCLF6D_TEMPLATE" \
        --tag "${SYMCLF6D_SYSTEM}_${SYMCLF6D_TEMPLATE}_s${s}_${SLURM_JOB_ID:-local}" &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

# independent audit of every best (referee role), written next to the logs
for s in $SEEDS; do
    log="runs/cma_${SYMCLF6D_SYSTEM}_${SYMCLF6D_TEMPLATE}_s${s}_${SLURM_JOB_ID:-local}.jsonl"
    [ -f "$log" ] && NUMBA_NUM_THREADS=$THREADS python audit.py --best "$log" \
        --json "${log%.jsonl}_audit.json" | tail -40
done
echo "done: $(date)"
