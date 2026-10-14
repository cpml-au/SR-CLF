#!/bin/bash
# Submit a production run from a FROZEN copy of the current commit.
#
#   tools/submit_frozen.sh <example> <slurm file in examples/<example>/slurm> [extra sbatch arguments...]
#   e.g.  tools/submit_frozen.sh cartpole4d job_cartpole4d_cpu.slurm
#
# Why: a run lasts up to 48 h and its Ray workers can be restarted during the run, so they import the code
# from disk again.  Editing the repository meanwhile would change the code a
# running job uses.  This script checks out the committed code into a separate git worktree
# ($SYMCLF_FROZEN_ROOT/<commit>, default ~/symclf_frozen/<commit>) and submits from there.  The outputs still go
# to the repository's examples/<example>/runs/ (git-ignored): Slurm logs in runs/, everything else in runs/<job>/.
set -euo pipefail
[ $# -ge 2 ] || { sed -n 2,5p "$0"; exit 2; }
EXAMPLE="$1"; SLURM_FILE="$2"; shift 2
REPO="$(cd "$(dirname "$0")/.." && pwd)"
for bad in JAX_PLATFORMS SYMCLF_GPU_ALLOW_CPU SYMCLF_GPU_PALLAS_INTERPRET; do   # sbatch exports this shell
    [ -z "${!bad:-}" ] || { echo "refusing: $bad is set in this shell (it would reach the job)"; exit 1; }
done
if ! git -C "$REPO" diff --quiet || ! git -C "$REPO" diff --cached --quiet; then
    echo "refusing: uncommitted changes to tracked files -- commit first, the job must run a known commit"; exit 1
fi
SHA="$(git -C "$REPO" rev-parse --short=12 HEAD)"
FROZEN="${SYMCLF_FROZEN_ROOT:-$HOME/symclf_frozen}/$SHA"
[ -d "$FROZEN" ] || git -C "$REPO" worktree add --detach "$FROZEN" "$SHA" >/dev/null
RUNS="$REPO/examples/$EXAMPLE/runs"
mkdir -p "$RUNS"
export SYMCLF_RUNS_ROOT="$RUNS" SYMCLF_CODE_COMMIT="$(git -C "$REPO" rev-parse HEAD)"
cd "$FROZEN/examples/$EXAMPLE/slurm"
JOB=$(sbatch --parsable --output="$RUNS/output_%j.log" --error="$RUNS/error_%j.log" "$@" "$SLURM_FILE")
echo "job $JOB | code $SHA frozen in $FROZEN | outputs $RUNS/$JOB/ and $RUNS/output_$JOB.log"
