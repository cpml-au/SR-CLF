#!/bin/bash -l
#SBATCH --job-name=cartpole_clf_gpu5_7
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p q32g
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:3
#SBATCH --mem=120G
#SBATCH --time=48:00:00

# ############################################################################
# GPU5_7 -- the certified-region objective. Submit FROM this folder:
#     sbatch jobGPU5_7.slurm
# ONE node: 32 CPU cores and 3 GPUs (leaves the rest of PRIME free).
# ############################################################################

# Do not enable nounset before activating Conda.
set -eo pipefail

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate flex

# Conda activation has finished, so nounset can now be enabled.
set -u

RAY_BIN="$CONDA_PREFIX/bin/ray"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export RAY_DEDUP_LOGS=0

# --- runtime scaling ---------------------------------------------------------
export SYMCLF_GPU3_EXACT_WAVE_MULT="${SYMCLF_GPU3_EXACT_WAVE_MULT:-6}"
export SYMCLF_GPU3_MESH_POINTS="${SYMCLF_GPU3_MESH_POINTS:-0}"
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
export SYMCLF_GPU2_TOTAL_GPUS="${SYMCLF_GPU2_TOTAL_GPUS:-$GPUS_PER_NODE}"
export SYMCLF_GPU3_ACTORS_PER_GPU="${SYMCLF_GPU3_ACTORS_PER_GPU:-3}"
export SYMCLF_GPU4_EXACT_BATCH_SIZE="${SYMCLF_GPU4_EXACT_BATCH_SIZE:-24}"
export SYMCLF_GPU3_EXACT_MAX_PER_GEN="${SYMCLF_GPU3_EXACT_MAX_PER_GEN:-1200}"

# --- GPU5_7: the certified-region objective ---------------------------------
# Saturation: u_required = max a/|b| charged per decade above the target.
# Calibrated: 96051 = 57, 122304 champ = 59, 122340 gen-7 (diverges) = 1.46e5.
export SYMCLF_GPU5_7_SAT_U_TARGET="${SYMCLF_GPU5_7_SAT_U_TARGET:-1000}"
export SYMCLF_GPU5_7_SAT_WEIGHT="${SYMCLF_GPU5_7_SAT_WEIGHT:-150}"
# ROA: certified level c_max + certified volume vs a 0.5%-of-box target
# (best known certified volume is 96051's 0.657%).
export SYMCLF_GPU5_7_ROA_WEIGHT="${SYMCLF_GPU5_7_ROA_WEIGHT:-500}"
export SYMCLF_GPU5_7_ROA_C_TARGET="${SYMCLF_GPU5_7_ROA_C_TARGET:-0.004}"
export SYMCLF_GPU5_7_ROA_NEG_WEIGHT="${SYMCLF_GPU5_7_ROA_NEG_WEIGHT:-250}"
# Certificate depth: exact violations priced by their W-placement.
export SYMCLF_GPU5_7_CERT_WEIGHT="${SYMCLF_GPU5_7_CERT_WEIGHT:-300}"
# Closed-loop rollout gate for candidates that pass every static check.
export SYMCLF_GPU5_7_ROLLOUT_ENABLED="${SYMCLF_GPU5_7_ROLLOUT_ENABLED:-1}"
export SYMCLF_GPU5_7_ROLLOUT_FAIL_PENALTY="${SYMCLF_GPU5_7_ROLLOUT_FAIL_PENALTY:-500}"
# Priors OFF (knobs kept for ablation runs).
export SYMCLF_GPU5_7_PROPERNESS_WEIGHT="${SYMCLF_GPU5_7_PROPERNESS_WEIGHT:-0}"
export SYMCLF_GPU5_7_V_GRAD_WEIGHT="${SYMCLF_GPU5_7_V_GRAD_WEIGHT:-0}"
export SYMCLF_GPU5_7_C_STAR_PENALTY_MAX="${SYMCLF_GPU5_7_C_STAR_PENALTY_MAX:-0}"
export SYMCLF_GPU5_7_C_STAR_PENALTY_CAP="${SYMCLF_GPU5_7_C_STAR_PENALTY_CAP:-0}"
export SYMCLF_GPU5_7_C_STAR_TAIL_WEIGHT="${SYMCLF_GPU5_7_C_STAR_TAIL_WEIGHT:-0}"
export SYMCLF_GPU5_7_COVERAGE_WEIGHT="${SYMCLF_GPU5_7_COVERAGE_WEIGHT:-0}"

# --- GPU5 positive-definiteness check (search-based, ARE metric) ------------
export SYMCLF_GPU5_PD_EPS="${SYMCLF_GPU5_PD_EPS:-1e-4}"
export SYMCLF_GPU5_PD_TOP_K="${SYMCLF_GPU5_PD_TOP_K:-40}"

# --- inherited GPU5_2..5_6 shapes --------------------------------------------
export SYMCLF_GPU5_7_VIOLATION_WEIGHT="${SYMCLF_GPU5_7_VIOLATION_WEIGHT:-300}"
export SYMCLF_GPU5_7_MEAN_MARGIN_SCALE="${SYMCLF_GPU5_7_MEAN_MARGIN_SCALE:-0.1}"
export SYMCLF_GPU5_7_GATE_MARGIN_ESCALATION="${SYMCLF_GPU5_7_GATE_MARGIN_ESCALATION:-2.0}"
export SYMCLF_GPU5_7_MEAN_MARGIN_WEIGHT="${SYMCLF_GPU5_7_MEAN_MARGIN_WEIGHT:-12000}"
export SYMCLF_GPU5_7_MEAN_MARGIN_PENALTY_MAX="${SYMCLF_GPU5_7_MEAN_MARGIN_PENALTY_MAX:-8000}"
export SYMCLF_GPU5_7_NEAR0_SQRT="${SYMCLF_GPU5_7_NEAR0_SQRT:-0.25}"
export SYMCLF_GPU5_7_NEAR0_QUARTIC="${SYMCLF_GPU5_7_NEAR0_QUARTIC:-4}"
export SYMCLF_GPU5_7_NEAR0_SIXTH="${SYMCLF_GPU5_7_NEAR0_SIXTH:-64}"
export SYMCLF_GPU5_7_GATE_PRICE_ESCALATION="${SYMCLF_GPU5_7_GATE_PRICE_ESCALATION:-2.0}"
# SYMCLF_GPU5_7_GATE_PRICE_CAP is deliberately NOT exported: fitness derives it
# as 2.3 * MANIFOLD_EXACT_MSE_GATE + floor (the price of the gate-1 dodge).
export SYMCLF_GPU5_7_NESTED_PENALTY="${SYMCLF_GPU5_7_NESTED_PENALTY:-50}"
export SYMCLF_GPU5_7_NORMALIZE_MARGIN="${SYMCLF_GPU5_7_NORMALIZE_MARGIN:-1}"
export SYMCLF_GPU5_7_GRAD_NORM_FLOOR="${SYMCLF_GPU5_7_GRAD_NORM_FLOOR:-1e-3}"
export SYMCLF_GPU5_7_FLAT_GRAD_FLOOR="${SYMCLF_GPU5_7_FLAT_GRAD_FLOOR:-1e-3}"
export SYMCLF_GPU5_7_FLAT_GRAD_WEIGHT="${SYMCLF_GPU5_7_FLAT_GRAD_WEIGHT:-1.0}"
export SYMCLF_GPU5_7_GRAD_NORM_MAX_POINTS="${SYMCLF_GPU5_7_GRAD_NORM_MAX_POINTS:-16384}"
export SYMCLF_GPU5_7_AXIS_B_TARGET="${SYMCLF_GPU5_7_AXIS_B_TARGET:-1e-3}"
export SYMCLF_GPU5_7_AXIS_B_WEIGHT="${SYMCLF_GPU5_7_AXIS_B_WEIGHT:-300}"
export SYMCLF_GPU5_7_AXIS_V_TARGET="${SYMCLF_GPU5_7_AXIS_V_TARGET:-1e-2}"
export SYMCLF_GPU5_7_AXIS_V_WEIGHT="${SYMCLF_GPU5_7_AXIS_V_WEIGHT:-300}"
export SYMCLF_GPU5_7_AXIS_TAIL_WEIGHT="${SYMCLF_GPU5_7_AXIS_TAIL_WEIGHT:-0.5}"
export SYMCLF_GPU5_7_ORIGIN_PENALTY_WEIGHT="${SYMCLF_GPU5_7_ORIGIN_PENALTY_WEIGHT:-0}"
export SYMCLF_GPU5_7_STRICT_MARGIN_TOL="${SYMCLF_GPU5_7_STRICT_MARGIN_TOL:-1e-9}"

# --- GPU2 pipeline knobs (unchanged) -----------------------------------------
export SYMCLF_GPU2_BATCH_SIZE="${SYMCLF_GPU2_BATCH_SIZE:-16}"
export SYMCLF_GPU2_EXACT_BATCH_SIZE="${SYMCLF_GPU2_EXACT_BATCH_SIZE:-1}"
# 21^4 grid: tuner fusion 2 + eval row chunk 4 keep the device allocation at
# ~3.4 GiB against the 11.2 GiB/actor budget (the 6.9 GiB request killed
# 122252/122299/122339; the fitness-level OOM retry is the second line of
# defence).
export SYMCLF_GPU2_TUNER_FUSION="${SYMCLF_GPU2_TUNER_FUSION:-2}"
export SYMCLF_GPU2_TUNER_RANDOM_POPULATION="${SYMCLF_GPU2_TUNER_RANDOM_POPULATION:-35}"
export SYMCLF_GPU2_TUNER_GENERATIONS="${SYMCLF_GPU2_TUNER_GENERATIONS:-5}"
export SYMCLF_GPU2_PALLAS_BLOCK_SIZE="${SYMCLF_GPU2_PALLAS_BLOCK_SIZE:-128}"
export SYMCLF_GPU2_EXACT_SCAN_CHUNK="${SYMCLF_GPU2_EXACT_SCAN_CHUNK:-262144}"
export SYMCLF_GPU2_EVAL_ROW_CHUNK="${SYMCLF_GPU2_EVAL_ROW_CHUNK:-4}"
export SYMCLF_GPU2_CHEAP_ENABLED="${SYMCLF_GPU2_CHEAP_ENABLED:-1}"
export SYMCLF_GPU2_FULL_EXACT_ENABLED="${SYMCLF_GPU2_FULL_EXACT_ENABLED:-1}"
export SYMCLF_GPU2_CHEAP_FUSION="${SYMCLF_GPU2_CHEAP_FUSION:-32}"
export SYMCLF_GPU2_CHEAP_LINES_PER_AXIS="${SYMCLF_GPU2_CHEAP_LINES_PER_AXIS:-224}"
export SYMCLF_GPU2_CHEAP_LINE_SAMPLES="${SYMCLF_GPU2_CHEAP_LINE_SAMPLES:-225}"
export SYMCLF_GPU2_CHEAP_OBLIQUE_LINES="${SYMCLF_GPU2_CHEAP_OBLIQUE_LINES:-112}"
export SYMCLF_GPU2_CHEAP_MAX_BRACKETS="${SYMCLF_GPU2_CHEAP_MAX_BRACKETS:-6144}"
export SYMCLF_GPU2_CHEAP_BISECTION_ITERATIONS="${SYMCLF_GPU2_CHEAP_BISECTION_ITERATIONS:-30}"
export SYMCLF_GPU2_CHEAP_B_TOL="${SYMCLF_GPU2_CHEAP_B_TOL:-1e-8}"
export SYMCLF_GPU2_CHEAP_REPORT_POINTS="${SYMCLF_GPU2_CHEAP_REPORT_POINTS:-64}"
export SYMCLF_GPU2_CHEAP_BUDGET_SECONDS="${SYMCLF_GPU2_CHEAP_BUDGET_SECONDS:-40}"
export SYMCLF_GPU2_EXACT_WAVE1_CHEAP_MAX="${SYMCLF_GPU2_EXACT_WAVE1_CHEAP_MAX:-500}"
export SYMCLF_GPU2_EXACT_WAVE2_CHEAP_MAX="${SYMCLF_GPU2_EXACT_WAVE2_CHEAP_MAX:-1500}"
export SYMCLF_GPU2_EXACT_A_MAX_GATE="${SYMCLF_GPU2_EXACT_A_MAX_GATE:-0.15}"
export SYMCLF_GPU2_EXACT_GATE_PENALTY_RATIO="${SYMCLF_GPU2_EXACT_GATE_PENALTY_RATIO:-1.1}"
export SYMCLF_GPU2_ARTSTEIN_ENABLED="${SYMCLF_GPU2_ARTSTEIN_ENABLED:-1}"
export SYMCLF_GPU2_ARTSTEIN_FUSION="${SYMCLF_GPU2_ARTSTEIN_FUSION:-4}"
export SYMCLF_GPU2_ARTSTEIN_LINES_PER_AXIS="${SYMCLF_GPU2_ARTSTEIN_LINES_PER_AXIS:-1024}"
export SYMCLF_GPU2_ARTSTEIN_LINE_SAMPLES="${SYMCLF_GPU2_ARTSTEIN_LINE_SAMPLES:-801}"
export SYMCLF_GPU2_ARTSTEIN_OBLIQUE_LINES="${SYMCLF_GPU2_ARTSTEIN_OBLIQUE_LINES:-512}"
export SYMCLF_GPU2_ARTSTEIN_MAX_BRACKETS="${SYMCLF_GPU2_ARTSTEIN_MAX_BRACKETS:-8192}"
export SYMCLF_GPU2_ARTSTEIN_BISECTION_ITERATIONS="${SYMCLF_GPU2_ARTSTEIN_BISECTION_ITERATIONS:-36}"
export SYMCLF_GPU2_ARTSTEIN_STARTS="${SYMCLF_GPU2_ARTSTEIN_STARTS:-256}"
export SYMCLF_GPU2_ARTSTEIN_ITERATIONS="${SYMCLF_GPU2_ARTSTEIN_ITERATIONS:-24}"
export SYMCLF_GPU2_ARTSTEIN_INITIAL_PROJECTION_STEPS="${SYMCLF_GPU2_ARTSTEIN_INITIAL_PROJECTION_STEPS:-6}"
export SYMCLF_GPU2_ARTSTEIN_PROJECTION_STEPS="${SYMCLF_GPU2_ARTSTEIN_PROJECTION_STEPS:-2}"
export SYMCLF_GPU2_ARTSTEIN_STEP_SIZE="${SYMCLF_GPU2_ARTSTEIN_STEP_SIZE:-0.05}"
export SYMCLF_GPU2_ARTSTEIN_B_TOL="${SYMCLF_GPU2_ARTSTEIN_B_TOL:-1e-8}"
export SYMCLF_GPU2_ARTSTEIN_REPORT_POINTS="${SYMCLF_GPU2_ARTSTEIN_REPORT_POINTS:-64}"
export SYMCLF_GPU2_FINAL_EXACT_AUDIT="${SYMCLF_GPU2_FINAL_EXACT_AUDIT:-1}"
export SYMCLF_GPU2_POLISH_TOP_K="${SYMCLF_GPU2_POLISH_TOP_K:-40}"
export SYMCLF_GPU2_POLISH_ITERATIONS="${SYMCLF_GPU2_POLISH_ITERATIONS:-60}"
export SYMCLF_GPU2_POLISH_B_TOL="${SYMCLF_GPU2_POLISH_B_TOL:-1e-8}"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.28}"
export SYMCLF_GPU2_JAX_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/symclf_gpu5_7_${SLURM_JOB_ID}"
# ---------------------------------------------------------------------------

cd "${SLURM_SUBMIT_DIR:-$PWD}"

PROJECT_ROOT=$(cd ../.. && pwd)
export PYTHONPATH="$PWD:$PROJECT_ROOT:${PYTHONPATH:-}"

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

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
readarray -t nodes_array <<< "$nodes"

head_node="${nodes_array[0]}"
# Unique Ray port per job (hardcoded 6379 collides across concurrent jobs).
port=$(( 6380 + (${SLURM_JOB_ID:-0} % 20000) ))

head_ip=$(
    srun --nodes=1 --ntasks=1 -w "$head_node" \
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

    exit "$status"
}

trap cleanup EXIT INT TERM

echo "Allocated nodes: ${nodes_array[*]}"
echo "Head node: $head_node"
echo "Head IP: $head_ip"
echo "Head address: $head_address"

echo "Starting Ray head on $head_node..."

srun \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    --gres=gpu:"$GPUS_PER_NODE" \
    --exclusive \
    -w "$head_node" \
    "$RAY_BIN" start \
        --head \
        --node-ip-address="$head_ip" \
        --port="$port" \
        --dashboard-host=0.0.0.0 \
        --num-cpus="$SLURM_CPUS_PER_TASK" \
        --num-gpus="$GPUS_PER_NODE" \
        --block &

ray_pids+=("$!")

sleep 20

for node in "${nodes_array[@]:1}"; do
    echo "Starting Ray worker on $node..."

    srun \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="$SLURM_CPUS_PER_TASK" \
        --gres=gpu:"$GPUS_PER_NODE" \
        --exclusive \
        -w "$node" \
        "$RAY_BIN" start \
            --address="$head_address" \
            --num-cpus="$SLURM_CPUS_PER_TASK" \
            --num-gpus="$GPUS_PER_NODE" \
            --block &

    ray_pids+=("$!")
    sleep 5
done

sleep 20

echo "Ray cluster status:"
"$RAY_BIN" status

# Five-second per-GPU telemetry on each node (overlapping one-CPU step).
for node in "${nodes_array[@]}"; do
    telemetry_file="gpu_telemetry_${SLURM_JOB_ID}_${node}.csv"
    srun \
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

echo "Starting run.py (GPU5_7 certified-region objective)..."
python run.py
