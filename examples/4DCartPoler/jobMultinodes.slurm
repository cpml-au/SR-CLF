#!/bin/bash -l
#SBATCH --job-name=cartpole_clf_10n64
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --nodes=13
#SBATCH --ntasks-per-node=1
#SBATCH -p q64
#SBATCH --cpus-per-task=64
#SBATCH --mem=488G
#SBATCH --time=99:00:00

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

cd "${SLURM_SUBMIT_DIR:-$PWD}"

PROJECT_ROOT=$(cd ../.. && pwd)
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo "Project root: $PROJECT_ROOT"
echo "PYTHONPATH: $PYTHONPATH"
echo "Conda environment: $CONDA_PREFIX"
echo "Python: $(which python)"

python --version
python -c "import ray; print('Ray Python package:', ray.__version__, ray.__file__)"

echo "Ray CLI: $RAY_BIN"
"$RAY_BIN" --version

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
readarray -t nodes_array <<< "$nodes"

head_node="${nodes_array[0]}"
port=6379

head_ip=$(
    srun --nodes=1 --ntasks=1 -w "$head_node" \
        bash -c "hostname --ip-address | awk '{print \$1}'"
)

head_address="$head_ip:$port"
export RAY_ADDRESS="$head_address"

ray_pids=()

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
    --exclusive \
    -w "$head_node" \
    "$RAY_BIN" start \
        --head \
        --node-ip-address="$head_ip" \
        --port="$port" \
        --dashboard-host=0.0.0.0 \
        --num-cpus="$SLURM_CPUS_PER_TASK" \
        --block &

ray_pids+=("$!")

sleep 20

for node in "${nodes_array[@]:1}"; do
    echo "Starting Ray worker on $node..."

    srun \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="$SLURM_CPUS_PER_TASK" \
        --exclusive \
        -w "$node" \
        "$RAY_BIN" start \
            --address="$head_address" \
            --num-cpus="$SLURM_CPUS_PER_TASK" \
            --block &

    ray_pids+=("$!")
    sleep 5
done

sleep 20

echo "Ray cluster status:"
"$RAY_BIN" status

echo "Starting run.py..."
python run.py