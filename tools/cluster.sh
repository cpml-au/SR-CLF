# The steps of a CPU job, the same for every example. A job script sources this file after it has set
#   EXAMPLE_DIR   the example folder, examples/<name>
#   REPO_ROOT     the repository root
# (examples/<name>/slurm/job_<name>_cpu.slurm does that). It needs Slurm (srun, scontrol) and the packed
# Python environment $HOME/flex.tar.gz.

cd "$EXAMPLE_DIR"

# 1. The Python environment. Unpack the packed environment on every node of the job and use it
#    (importing Python packages from the shared home is slow). FLEX_TAR names another file.
FLEX_TAR="${FLEX_TAR:-$HOME/flex.tar.gz}"
LOCAL_ENV="/tmp/flex_${SLURM_JOB_ID}"
_t0=$(date +%s)
srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 --cpus-per-task=1 bash -c \
    "mkdir -p '$LOCAL_ENV' && tar -xzf '$FLEX_TAR' -C '$LOCAL_ENV' && source '$LOCAL_ENV/bin/activate' && conda-unpack"
source "$LOCAL_ENV/bin/activate"
echo "local env ready on all nodes in $(( $(date +%s) - _t0 )) s: python=$(which python)"
set -u

# 2. The settings. config.yaml -> environment variables, BEFORE Ray starts, so every worker gets them.
#    A variable that is already set (sbatch --export=ALL,NAME=value) is kept.
#    Print them:  python -m src.settings --table examples/<name>/config.yaml --backend cpu
_SETTINGS="$(cd "$REPO_ROOT" && python -m src.settings --export "$EXAMPLE_DIR/config.yaml" --backend cpu)" \
    || { echo "SETTINGS FAILED -- aborting"; exit 1; }
eval "$_SETTINGS"
echo "settings: $(echo "$_SETTINGS" | wc -l) variables from $EXAMPLE_DIR/config.yaml [cpu]"

# 3. One thread per worker; the example folder and the repository on the Python path.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$EXAMPLE_DIR:$REPO_ROOT:${PYTHONPATH:-}"
env | grep -E "^SYMCLF|^FLEX_" | sort | tr '\n' ' ' | cut -c1-2500; echo

# 4. Compile the numerical kernels once here, so the workers do not all compile them at the same time.
python -c "
from src.CPU.lattice import lattice_numbers
from src.CPU.rule import fitness_fixed_constants
from src.CPU.templates import default_template
e, x0, _ = default_template()
lattice_numbers(e, x0); fitness_fixed_constants(e, x0); print('numba warm-up done')
" 2>&1 | grep -v "Warning\|warn"

# 5. Ray. The head on the first node, one worker process on every other node, all cores, no GPU.
#    The head gets up to three attempts. At the end of the job everything is stopped and the unpacked
#    environment is removed (cleanup).
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

# 6. The run.
echo "Starting src.CPU.run $(date +%T)"
cd "$REPO_ROOT"
python -m src.CPU.run "$EXAMPLE_DIR"
