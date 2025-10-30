#!/bin/bash
#SBATCH --job-name=cellpose
#SBATCH --account=your_umich_account
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load python/3.10.5          # adjust or remove if using custom stack

# Activate your conda environment
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate cellpose-hpc

mkdir -p logs

# Override via: sbatch --export=CONFIG=path/to/config.yaml slurm/run_cellpose.slurm
CONFIG_PATH="${CONFIG:-config/cellpose_job.yaml}"

echo "[$(date)] Running Cellpose with config: ${CONFIG_PATH}"
python scripts/run_cellpose.py --config "${CONFIG_PATH}"

echo "[$(date)] Job complete."
