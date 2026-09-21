#!/bin/bash -l
# ############################################################################
#  6-D quadrotor CLF on PRIME (2026-09-18, the user's order): CMA-ES on the quadratic (LQR) template with
#  the V9 fitness, last version = the V8/V9 price with the kappa hinge 100 @ 1.0 (the 4-D study's lever) and
#  the volume hinge at the default 20 @ 0.02. On the free resources of s02n01 (1 A100 + 24 cores; the
#  objective itself is CPU-only, README "Why no GPU"). Smoke off: its 4-D acceptance numbers are calibrated
#  on the old hinges (30 @ 0.05) and would fail by construction with 100 @ 1.0.
#      cd examples/6D/slurm && sbatch cma_prime_s02n01.slurm
# ############################################################################
#SBATCH --job-name=clf6d_prime
#SBATCH --output=runs/cma_prime_%j.log
#SBATCH --error=runs/cma_prime_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p q32g
#SBATCH --nodelist=s02n01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=60G
#SBATCH --time=48:00:00

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}/.."          # the folder root (slurm/ lives one level down)
FOLDER=$PWD
source ./env.sh
# --- the V9 fitness, last version (2026-09-18) ---
export SYMCLF_DET4_PEN_KAPPA=100 SYMCLF_DET4_PEN_KAPPA_TARGET=1.0
export SYMCLF_DET4_PEN_VOL=20 SYMCLF_DET4_PEN_VOL_TARGET=0.02

source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate "${CLF6D_CONDA:-flex}"
python -c "import numpy, scipy, numba, sympy, cma; print('deps ok', numpy.__version__, scipy.__version__, numba.__version__, cma.__version__)"

CORES="${SLURM_CPUS_PER_TASK:-24}"
SEEDS="${SEEDS:-1 2}"
POPSIZE="${POPSIZE:-10}"
BUDGET="${BUDGET:-20000}"                   # 21 coefficients: ~4x the 4-D budget, with headroom
NSEEDS=$(echo "$SEEDS" | wc -w)
WORKERS="${WORKERS:-$POPSIZE}"
THREADS="${THREADS:-$(( CORES / (NSEEDS * WORKERS) ))}"
[ "$THREADS" -lt 1 ] && THREADS=1
echo "node $(hostname) cores $CORES: $NSEEDS seeds x $WORKERS workers x $THREADS numba threads | budget $BUDGET | kappa hinge $SYMCLF_DET4_PEN_KAPPA@$SYMCLF_DET4_PEN_KAPPA_TARGET vol $SYMCLF_DET4_PEN_VOL@$SYMCLF_DET4_PEN_VOL_TARGET | box $SYMCLF_V2_BOX | template $SYMCLF6D_TEMPLATE | $(date +%T)"

if [ "${SMOKE:-0}" = "1" ]; then
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

for s in $SEEDS; do
    log="runs/cma_${SYMCLF6D_SYSTEM}_${SYMCLF6D_TEMPLATE}_s${s}_${SLURM_JOB_ID:-local}.jsonl"
    [ -f "$log" ] && NUMBA_NUM_THREADS=$THREADS python audit.py --best "$log" \
        --json "${log%.jsonl}_audit.json" | tail -40
done
echo "done: $(date)"
