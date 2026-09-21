#!/bin/bash -l
#SBATCH --job-name=cp3d_v6
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
# CP3D -- the certified-region objective. Submit FROM this folder:
#     sbatch jobCP3D.slurm
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

# --- V2: the four changes ----------------------------------------------------
# Every default was calibrated against the 3-D ARE quadratic FIRST -- it is the
# one candidate whose correctness is not in question, and under these settings
# it scores 0.175 (pure length) while the V1 champion scores 314.6.
# Verify with:  python score_v2.py   (the ARE must rank first).
export SYMCLF_V2_BOX="${SYMCLF_V2_BOX:-0.25,0.0494,0.1155}"
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
export SYMCLF_DET4_REFINE_TOP="${SYMCLF_DET4_REFINE_TOP:-10}"
export SYMCLF_DET4_REFINE_STEPS="${SYMCLF_DET4_REFINE_STEPS:-100}"
export SYMCLF_DET4_REFINE_LR="${SYMCLF_DET4_REFINE_LR:-0.02}"
export SYMCLF_DET4_REFINE_METHOD="${SYMCLF_DET4_REFINE_METHOD:-adam}"
export SYMCLF_DET4_REFINE_SELECT="${SYMCLF_DET4_REFINE_SELECT:-price}"
# scheduling = GPU5_7: every gate-1 pass is screened (SYMCLF_GPU2_CHEAP_FUSION per launch), the full
# check goes to max a/W <= SYMCLF_GPU2_EXACT_A_MAX_GATE or cheap-certified, capacity SYMCLF_GPU3_EXACT_MAX_PER_GEN
export SYMCLF_CERTIFY_EVERY="${SYMCLF_CERTIFY_EVERY:-0}"       # certify.py on the champion every N gens (report)
# --- 2026-09-10 V4: the DET4 numbers from examples/DET4Fast (det4_fast: the referee's det4 in numba,
# W = V - V(0) without cancellation). The exact checker runs as GPU5_7 (root verdict only); the
# elites' refine uses det4_fast's numbers. Scheduling, tuner and rule unchanged from V3.
export SYMCLF_DET4_FAST="${SYMCLF_DET4_FAST:-1}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-3}"   # det4_fast threads per process (9 Ray actors x 3)
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
export SYMCLF_DET4_PEN_KAPPA="${SYMCLF_DET4_PEN_KAPPA:-30}"
export SYMCLF_DET4_PEN_KAPPA_TARGET="${SYMCLF_DET4_PEN_KAPPA_TARGET:-0.05}"
export SYMCLF_DET4_PEN_VOL="${SYMCLF_DET4_PEN_VOL:-20}"
export SYMCLF_DET4_PEN_VOL_TARGET="${SYMCLF_DET4_PEN_VOL_TARGET:-0.02}"
echo "V6 DET4_FAST=$SYMCLF_DET4_FAST numba_threads=$NUMBA_NUM_THREADS gate_kappa_cap=$SYMCLF_DET4_GATE_KAPPA_CAP det4_all=$SYMCLF_DET4_ALL refine=$SYMCLF_DET4_REFINE_METHOD/$SYMCLF_DET4_REFINE_SELECT top$SYMCLF_DET4_REFINE_TOP x$SYMCLF_DET4_REFINE_STEPS lr$SYMCLF_DET4_REFINE_LR pen cv=$SYMCLF_DET4_PEN_CV@$SYMCLF_DET4_PEN_CV_TARGET kappa=$SYMCLF_DET4_PEN_KAPPA@$SYMCLF_DET4_PEN_KAPPA_TARGET vol=$SYMCLF_DET4_PEN_VOL@$SYMCLF_DET4_PEN_VOL_TARGET tuner_skip=$SYMCLF_TUNER_SKIP_ADAM_AT_ZERO/$SYMCLF_TUNER_ADAM_EXIT_ZERO_GRAD"
export SYMCLF_TUNER_CONST_RANGE="${SYMCLF_TUNER_CONST_RANGE:-30}"
export SYMCLF_TUNER_CONST_PERTURB_FRAC="${SYMCLF_TUNER_CONST_PERTURB_FRAC:-0.5}"
export SYMCLF_TUNER_REFINE_STEPS="${SYMCLF_TUNER_REFINE_STEPS:-40}"
export SYMCLF_TUNER_REFINE_TOP_K="${SYMCLF_TUNER_REFINE_TOP_K:-4}"
export SYMCLF_TUNER_REFINE_LR="${SYMCLF_TUNER_REFINE_LR:-0.30}"
echo "V3 positivity=$SYMCLF_V3_POSITIVITY_WEIGHT const_range=$SYMCLF_TUNER_CONST_RANGE refine=${SYMCLF_TUNER_REFINE_STEPS}x top${SYMCLF_TUNER_REFINE_TOP_K}"
echo "V3 box=$SYMCLF_V2_BOX shape=$SYMCLF_V2_SHAPE_WEIGHT rho_mult=$SYMCLF_V2_ROA_RHO_MULTIPLIER cert=$SYMCLF_V2_CERT_WEIGHT vdot=$SYMCLF_V2_VDOT_WEIGHT"

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
export SYMCLF_CP3D_SAT_U_TARGET="${SYMCLF_CP3D_SAT_U_TARGET:-1000}"
export SYMCLF_CP3D_SAT_WEIGHT="${SYMCLF_CP3D_SAT_WEIGHT:-150}"
# ROA: certified level c_max + certified volume vs a 0.5%-of-box target
# (best known certified volume is 96051's 0.657%).
export SYMCLF_CP3D_ROA_WEIGHT="${SYMCLF_CP3D_ROA_WEIGHT:-500}"
export SYMCLF_CP3D_ROA_C_TARGET="${SYMCLF_CP3D_ROA_C_TARGET:-0.004}"
export SYMCLF_CP3D_ROA_NEG_WEIGHT="${SYMCLF_CP3D_ROA_NEG_WEIGHT:-250}"
# Certificate depth: exact violations priced by their W-placement.
export SYMCLF_CP3D_CERT_WEIGHT="${SYMCLF_CP3D_CERT_WEIGHT:-300}"
# Closed-loop rollout gate for candidates that pass every static check.
export SYMCLF_CP3D_ROLLOUT_ENABLED="${SYMCLF_CP3D_ROLLOUT_ENABLED:-0}"   # 2026-09-09: no rollouts in det4 mode
export SYMCLF_CP3D_ROLLOUT_FAIL_PENALTY="${SYMCLF_CP3D_ROLLOUT_FAIL_PENALTY:-500}"
# Priors OFF (knobs kept for ablation runs).
export SYMCLF_CP3D_PROPERNESS_WEIGHT="${SYMCLF_CP3D_PROPERNESS_WEIGHT:-0}"
export SYMCLF_CP3D_V_GRAD_WEIGHT="${SYMCLF_CP3D_V_GRAD_WEIGHT:-0}"
export SYMCLF_CP3D_C_STAR_PENALTY_MAX="${SYMCLF_CP3D_C_STAR_PENALTY_MAX:-0}"
export SYMCLF_CP3D_C_STAR_PENALTY_CAP="${SYMCLF_CP3D_C_STAR_PENALTY_CAP:-0}"
export SYMCLF_CP3D_C_STAR_TAIL_WEIGHT="${SYMCLF_CP3D_C_STAR_TAIL_WEIGHT:-0}"
export SYMCLF_CP3D_COVERAGE_WEIGHT="${SYMCLF_CP3D_COVERAGE_WEIGHT:-0}"

# --- GPU5 positive-definiteness check (search-based, ARE metric) ------------
export SYMCLF_GPU5_PD_EPS="${SYMCLF_GPU5_PD_EPS:-1e-4}"
export SYMCLF_GPU5_PD_TOP_K="${SYMCLF_GPU5_PD_TOP_K:-40}"

# --- inherited GPU5_2..5_6 shapes --------------------------------------------
export SYMCLF_CP3D_VIOLATION_WEIGHT="${SYMCLF_CP3D_VIOLATION_WEIGHT:-300}"
export SYMCLF_CP3D_MEAN_MARGIN_SCALE="${SYMCLF_CP3D_MEAN_MARGIN_SCALE:-0.1}"
export SYMCLF_CP3D_GATE_MARGIN_ESCALATION="${SYMCLF_CP3D_GATE_MARGIN_ESCALATION:-2.0}"
export SYMCLF_CP3D_MEAN_MARGIN_WEIGHT="${SYMCLF_CP3D_MEAN_MARGIN_WEIGHT:-12000}"
export SYMCLF_CP3D_MEAN_MARGIN_PENALTY_MAX="${SYMCLF_CP3D_MEAN_MARGIN_PENALTY_MAX:-8000}"
export SYMCLF_CP3D_NEAR0_SQRT="${SYMCLF_CP3D_NEAR0_SQRT:-0.25}"
export SYMCLF_CP3D_NEAR0_QUARTIC="${SYMCLF_CP3D_NEAR0_QUARTIC:-4}"
export SYMCLF_CP3D_NEAR0_SIXTH="${SYMCLF_CP3D_NEAR0_SIXTH:-64}"
export SYMCLF_CP3D_GATE_PRICE_ESCALATION="${SYMCLF_CP3D_GATE_PRICE_ESCALATION:-2.0}"
# SYMCLF_CP3D_GATE_PRICE_CAP is deliberately NOT exported: fitness derives it
# as 2.3 * MANIFOLD_EXACT_MSE_GATE + floor (the price of the gate-1 dodge).
export SYMCLF_CP3D_NESTED_PENALTY="${SYMCLF_CP3D_NESTED_PENALTY:-50}"
export SYMCLF_CP3D_NORMALIZE_MARGIN="${SYMCLF_CP3D_NORMALIZE_MARGIN:-1}"
export SYMCLF_CP3D_GRAD_NORM_FLOOR="${SYMCLF_CP3D_GRAD_NORM_FLOOR:-1e-3}"
export SYMCLF_CP3D_FLAT_GRAD_FLOOR="${SYMCLF_CP3D_FLAT_GRAD_FLOOR:-1e-3}"
export SYMCLF_CP3D_FLAT_GRAD_WEIGHT="${SYMCLF_CP3D_FLAT_GRAD_WEIGHT:-1.0}"
export SYMCLF_CP3D_GRAD_NORM_MAX_POINTS="${SYMCLF_CP3D_GRAD_NORM_MAX_POINTS:-16384}"
export SYMCLF_CP3D_AXIS_B_TARGET="${SYMCLF_CP3D_AXIS_B_TARGET:-1e-3}"
export SYMCLF_CP3D_AXIS_B_WEIGHT="${SYMCLF_CP3D_AXIS_B_WEIGHT:-300}"
export SYMCLF_CP3D_AXIS_V_TARGET="${SYMCLF_CP3D_AXIS_V_TARGET:-1e-2}"
export SYMCLF_CP3D_AXIS_V_WEIGHT="${SYMCLF_CP3D_AXIS_V_WEIGHT:-300}"
export SYMCLF_CP3D_AXIS_TAIL_WEIGHT="${SYMCLF_CP3D_AXIS_TAIL_WEIGHT:-0.5}"
export SYMCLF_CP3D_ORIGIN_PENALTY_WEIGHT="${SYMCLF_CP3D_ORIGIN_PENALTY_WEIGHT:-0}"
export SYMCLF_CP3D_STRICT_MARGIN_TOL="${SYMCLF_CP3D_STRICT_MARGIN_TOL:-1e-9}"

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
export SYMCLF_GPU2_JAX_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/symclf_cp3dv3_${SLURM_JOB_ID}"
# ---------------------------------------------------------------------------

cd "${SLURM_SUBMIT_DIR:-$PWD}"

PROJECT_ROOT=$(cd ../.. && pwd)
export PYTHONPATH="$PWD:$PROJECT_ROOT:${PYTHONPATH:-}"

# 2026-09-09: CPU preflight before Ray (as the E25 slurm does): imports, the DET4 grid ledger,
# cheap/full DET4 and the root verdict on the ARE quadratic (~30 s). set -e aborts the job on FAIL.
JAX_PLATFORMS=cpu SYMCLF_GPU2_ALLOW_CPU=1 SYMCLF_GPU2_PALLAS_INTERPRET=1 \
    SYMCLF_GPU2_TUNER_RANDOM_POPULATION=4 SYMCLF_GPU2_TUNER_GENERATIONS=1 SYMCLF_TUNER_REFINE_STEPS=0 \
    python smoke_v3_det4.py | grep -v "^PASS" || { echo "PREFLIGHT smoke_v3_det4 FAILED"; exit 1; }

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
# 2026-09-09: the head port must stay outside Ray's worker-port range 10002-19999 (job 126149 got
# 12529 from the old 6380 + job%20000 formula and Ray refused to start); same formula as the E25 slurm.
port=$(( 20000 + (${SLURM_JOB_ID:-0} % 30000) ))

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

# 2026-09-11 (job 144451): the head died at start ("ray_client_server [exit code=1]") on a node freed
# seconds earlier by the previous job; kill any leftover Ray process of ours on the node first.
srun --nodes=1 --ntasks=1 --overlap -w "$head_node" "$RAY_BIN" stop --force >/dev/null 2>&1 || true
sleep 5

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
        --include-dashboard=false \
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

echo "Starting run.py (CP3D V6 penalties + price-selected Adam elites; well-conditioned reference + local constant refinement)..."
python run.py
