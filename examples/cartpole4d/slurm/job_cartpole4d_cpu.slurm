#!/bin/bash -l
# The GP search of examples/cartpole4d on CPUs. This file holds the cluster settings only (the #SBATCH lines).
# Every numerical setting is in ../config.yaml; the steps of the job are in tools/cluster.sh.
#     cd examples/cartpole4d/slurm && mkdir -p ../runs && sbatch job_cartpole4d_cpu.slurm
# or, from a fixed copy of the committed code:  tools/submit_frozen.sh cartpole4d job_cartpole4d_cpu.slurm [sbatch options]
# An sbatch option on the command line replaces the #SBATCH line below; --export=ALL,NAME=value changes one
# setting for this job (docs/SETTINGS_CPU.md). Records: ../runs/<job>/, Slurm logs: ../runs/.
#SBATCH --job-name=cp4d_cpu
#SBATCH --output=../runs/output_%j.log
#SBATCH --error=../runs/error_%j.log
#SBATCH -p q36
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=180G
#SBATCH --time=48:00:00

set -eo pipefail
# this file's folder, examples/cartpole4d/slurm (sbatch runs a spool copy, so ask Slurm for the original)
_SCRIPT="$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | sed -n 's/^ *Command=\([^ ]*\).*/\1/p')"
SLURM_DIR="$(cd "$(dirname "${_SCRIPT:-${BASH_SOURCE[0]}}")" && pwd)"   # examples/<name>/slurm
EXAMPLE_DIR="$(cd "$SLURM_DIR/.." && pwd)"                  # examples/cartpole4d
REPO_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"               # the repository root
source "$REPO_ROOT/tools/cluster.sh"
