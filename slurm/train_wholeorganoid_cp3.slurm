#!/bin/bash
#SBATCH --job-name=cp3_stageA_prepare
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/turbo/umms-parent/cellpose_wholeorganoid_model/logs/%x-%j.out

# Description: Stage A (prepare) — create run_dir, snapshot config, verify dataset.

set -euo pipefail

# Activate environment (adjust to your setup)
source ~/.bashrc
conda activate cellpose_env || true

# Config path (edit for your repo layout)
CONFIG_PATH="/nfs/turbo/umms-parent/cellpose_wholeorganoid_model/configs/cp3_v001.yaml"

# Repo path
REPO_ROOT="${HOME}/Desktop/githubprojects/cellpose_on_umich_hpc"

python "${REPO_ROOT}/scripts/run_experiment.py" \
  --config "${CONFIG_PATH}" \
  --mode prepare
