#!/bin/bash -l
# The GP search of examples/link2 on CPUs. This file holds the cluster settings only (the #SBATCH lines).
# Every numerical setting is in ../config.yaml; the steps of the job are in tools/cluster.sh.
#     cd examples/link2/slurm && mkdir -p ../runs && sbatch job_link2_cpu.slurm
# or, from a fixed copy of the committed code:  tools/submit_frozen.sh link2 job_link2_cpu.slurm [sbatch options]
# An sbatch option on the command line replaces the #SBATCH line below; --export=ALL,NAME=value changes one
# setting for this job (docs/SETTINGS_CPU.md). Records: ../runs/<job>/, Slurm logs: ../runs/.
#SBATCH --job-name=gp_link2
#SBATCH --output=../runs/output_%j.log
#SBATCH --error=../runs/error_%j.log
#SBATCH -p q64
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=180G
#SBATCH --time=48:00:00

set -eo pipefail
# this file's folder, examples/link2/slurm (sbatch runs a spool copy, so ask Slurm for the original)
_SCRIPT="$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | sed -n 's/^ *Command=\([^ ]*\).*/\1/p')"
SLURM_DIR="$(cd "$(dirname "${_SCRIPT:-${BASH_SOURCE[0]}}")" && pwd)"
EXAMPLE_DIR="$(cd "$SLURM_DIR/.." && pwd)"                  # examples/link2
REPO_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"               # the repository root
source "$REPO_ROOT/tools/cluster.sh"
