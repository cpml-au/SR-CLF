#!/bin/bash -l
# ############################################################################
#  examples/ductedfan6d: CMA-ES on the quadratic template (src/CPU/cma_run.py:
#  BIPOP pycma, x0 = ones, no start from a Riccati solution) with the pipeline's own fitness, one cma_run per
#  seed side by side, then the independent audit (src/CPU/audit.py) of each seed's best.  CPU only.
#  The fitness settings come from ../config.yaml (src/settings.py); budget, popsize, sigma0, seeds,
#  workers and threads are its template_cma block (read with `python -m src.settings --get`; an
#  environment variable BUDGET / POPSIZE / SIGMA0 / SEEDS / WORKERS / THREADS set at sbatch time wins).
#  The #SBATCH lines below ask for q36, 12 cores: 2 seeds x 6 workers x 1 thread.
#      cd examples/ductedfan6d/slurm && mkdir -p ../runs && sbatch template_ductedfan6d.slurm
#  Logs: ../runs/<job>/cma_ductedfan6d_full_s<seed>_<job>.jsonl (+ _audit.json); Slurm logs: ../runs/.
# ############################################################################
#SBATCH --job-name=tpl_ductedfan6d
#SBATCH --output=../runs/template_%j.log
#SBATCH --error=../runs/template_%j.err
#SBATCH -p q36
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=48:00:00

set -eo pipefail
# this file's folder, examples/ductedfan6d/slurm (sbatch runs a spool copy, so ask Slurm for the original)
_SCRIPT="$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | sed -n 's/^ *Command=\([^ ]*\).*/\1/p')"
SLURM_DIR="$(cd "$(dirname "${_SCRIPT:-${BASH_SOURCE[0]}}")" && pwd)"
EXAMPLE_DIR="$(cd "$SLURM_DIR/.." && pwd)"                  # examples/ductedfan6d
REPO_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"               # the repository root
CFG="$EXAMPLE_DIR/config.yaml"
cd "$REPO_ROOT"

source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate "${CLF_CONDA:-flex}"
python -c "import numpy, scipy, numba, sympy, cma; print('deps ok', numpy.__version__, scipy.__version__, numba.__version__, cma.__version__)"

# --- the fitness settings: config.yaml -> environment (the cma_run workers inherit it) ---
_SETTINGS="$(python -m src.settings --export "$CFG" --backend cpu)" || { echo "SETTINGS FAILED -- aborting"; exit 1; }
eval "$_SETTINGS"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$EXAMPLE_DIR:$REPO_ROOT:${PYTHONPATH:-}"
set -u

# --- the run's own flags: the template_cma block of config.yaml ---
get() { python -m src.settings --get "$1" "$CFG"; }
BUDGET="${BUDGET:-$(get template_cma.budget)}"
POPSIZE="${POPSIZE:-$(get template_cma.popsize)}"
SIGMA0="${SIGMA0:-$(get template_cma.sigma0)}"
SEEDS="${SEEDS:-$(get template_cma.seeds | tr ',' ' ')}"
WORKERS="${WORKERS:-$(get template_cma.workers)}"
THREADS="${THREADS:-$(get template_cma.threads)}"
SYS="$SYMCLF_CPU_SYSTEM"; TPL="$SYMCLF_CPU_TEMPLATE"; JOB="${SLURM_JOB_ID:-local}"
RUN_DIR="${SYMCLF_RUNS_ROOT:-$EXAMPLE_DIR/runs}/$JOB"
mkdir -p "$RUN_DIR"
cp "$CFG" "$RUN_DIR/config.yaml"
printf '%s\n' "$_SETTINGS" > "$RUN_DIR/settings_env.sh"
{ echo "commit $(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"; echo "code folder $REPO_ROOT"
  git -C "$REPO_ROOT" status --short --untracked-files=no 2>/dev/null | sed 's/^/modified: /' || true; } \
    > "$RUN_DIR/code_version.txt"
echo "node $(hostname) cores ${SLURM_CPUS_PER_TASK:-?}: seeds [$SEEDS] x $WORKERS workers x $THREADS numba threads | budget $BUDGET popsize $POPSIZE sigma0 $SIGMA0 | box $SYMCLF_BOX | system $SYS | template $TPL | $(date +%T) | run folder $RUN_DIR"

pids=()
for s in $SEEDS; do
    python -m src.CPU.cma_run --system "$SYS" --template "$TPL" --budget "$BUDGET" --popsize "$POPSIZE" \
        --sigma0 "$SIGMA0" --workers "$WORKERS" --threads "$THREADS" --seed "$s" \
        --tag "${SYS}_${TPL}_s${s}_${JOB}" --out "$RUN_DIR/cma_${SYS}_${TPL}_s${s}_${JOB}.jsonl" &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

for s in $SEEDS; do
    log="$RUN_DIR/cma_${SYS}_${TPL}_s${s}_${JOB}.jsonl"
    [ -f "$log" ] && NUMBA_NUM_THREADS=$THREADS python -m src.CPU.audit --system "$SYS" --template "$TPL" \
        --best "$log" --json "${log%.jsonl}_audit.json" | tail -40
done
echo "done: $(date)"
