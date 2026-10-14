#!/bin/bash -l
# The GP search of examples/invpendulum2 on CPUs. This file holds the cluster settings only (the #SBATCH lines).
# Every numerical setting is in ../config.yaml; the steps of the job are in tools/cluster.sh.
#     cd examples/invpendulum2/slurm && mkdir -p ../runs && sbatch job_invpendulum2_cpu.slurm
# or, from a fixed copy of the committed code:  tools/submit_frozen.sh invpendulum2 job_invpendulum2_cpu.slurm [sbatch options]
# An sbatch option on the command line replaces the #SBATCH line below; --export=ALL,NAME=value changes one
# setting for this job (docs/SETTINGS_CPU.md). Records: ../runs/<job>/, Slurm logs: ../runs/.
#SBATCH --job-name=gp_invpendulum2
#SBATCH --output=../runs/output_%j.log
#SBATCH --error=../runs/error_%j.log
#SBATCH -p q32g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=02:00:00

set -eo pipefail
# this file's folder, examples/invpendulum2/slurm (sbatch runs a spool copy, so ask Slurm for the original)
_SCRIPT="$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | sed -n 's/^ *Command=\([^ ]*\).*/\1/p')"
SLURM_DIR="$(cd "$(dirname "${_SCRIPT:-${BASH_SOURCE[0]}}")" && pwd)"
EXAMPLE_DIR="$(cd "$SLURM_DIR/.." && pwd)"                  # examples/invpendulum2
REPO_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"               # the repository root
source "$REPO_ROOT/tools/cluster.sh"
