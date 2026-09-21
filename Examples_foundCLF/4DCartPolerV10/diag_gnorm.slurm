#!/bin/bash -l
#SBATCH --job-name=diag_gnorm
#SBATCH --output=diag_gnorm_%j.log
#SBATCH -p q64
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=24G --time=01:00:00
# On-demand diagnostics of a (running or finished) V9 run:  sbatch --export=ALL,JSONL=<job>_best_per_generation.jsonl diag_gnorm.slurm
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate flex
cd "${SLURM_SUBMIT_DIR:-$PWD}"
python diag_gnorm.py "${JSONL:?set JSONL=<job>_best_per_generation.jsonl}" 2>&1 | grep -v "Warning\|warn"
