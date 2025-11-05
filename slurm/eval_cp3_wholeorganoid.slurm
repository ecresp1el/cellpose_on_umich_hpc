#!/bin/bash
# ============================================================================
# eval_cp3_wholeorganoid.slurm
#
# Purpose:
#   Stage C — Evaluate a *previously trained* run (resume mode).
#   - Loads model from RUN_DIR/train/ (CP models dir or weights_final.pt)
#   - Writes eval artifacts to RUN_DIR/eval/ (masks/flows/prob/panels/ROIs/JSON)
#
# Logging:
#   - SLURM logs -> TURBO logs/slurm/
#   - Evaluator prints -> SLURM .out (and per-image JSONs in RUN_DIR/eval/json/)
#
# Usage:
#   sbatch --export=ALL,ENV_NAME=cellpose3,\
# REPO_ROOT=/path/to/cellpose_on_umich_hpc,\
# CONFIG_PATH=/path/to/config.yaml,\
# RUN_DIR=/nfs/turbo/.../results/<model>/run_YYYY-mm-dd_HHMMSS,\
# SPLIT=valid \
#          slurm/eval_cp3_wholeorganoid.slurm
#
# Notes:
#   * RUN_DIR must be a prepared/trained run (contains cfg/config_snapshot.yaml
#     and train/ artifacts). This job does not call prepare() or train().
# ============================================================================

#SBATCH --job-name=cp3_eval_wholeorganoid
#SBATCH --account=parent0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/turbo/umms-parent/cellpose_wholeorganoid_model/logs/slurm/%x_%j.out
#SBATCH --error=/nfs/turbo/umms-parent/cellpose_wholeorganoid_model/logs/slurm/%x_%j.err

set -euo pipefail

# ------------------- OVERRIDES AT SUBMISSION (with --export=) ----------------
: "${REPO_ROOT:=${HOME}/Desktop/githubprojects/cellpose_on_umich_hpc}"
: "${CONFIG_PATH:=${REPO_ROOT}/configs/cp3_v001.yaml}"
: "${ENV_NAME:=cellpose3}"
: "${RUN_DIR:=}"          # REQUIRED: prepared run directory to evaluate
: "${SPLIT:=valid}"       # valid | all
# ----------------------------------------------------------------------------

echo "=== NODE & PATHS ==="
echo "Host        : $(hostname)"
echo "Repo root   : ${REPO_ROOT}"
echo "Config YAML : ${CONFIG_PATH}"
echo "Run dir     : ${RUN_DIR}"
echo "Split       : ${SPLIT}"
echo "Mode        : eval (Stage C, resume)"
echo "Start time  : $(date)"
echo "======================================"

# ---------------------- UMich ARC Anaconda activation -----------------------
module load python3.10-anaconda/2023.03
CONDA_BASE=/sw/pkgs/arc/python3.10-anaconda/2023.03
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  eval "$("$CONDA_BASE/bin/conda" shell.bash hook)"
fi
conda activate "${ENV_NAME}"

# ---------------------------- Guards & sanity --------------------------------
if [[ -z "${RUN_DIR}" ]]; then
  echo "[ERROR] RUN_DIR is required for resume-mode evaluation."
  exit 2
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] CONFIG_PATH not found: ${CONFIG_PATH}"
  exit 2
fi

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "[ERROR] RUN_DIR not found: ${RUN_DIR}"
  exit 2
fi

if [[ ! -f "${RUN_DIR}/cfg/config_snapshot.yaml" ]]; then
  echo "[ERROR] RUN_DIR is not a prepared run (missing cfg/config_snapshot.yaml): ${RUN_DIR}"
  exit 2
fi
# ---------------------- Install dependencies if missing ---------------------

# Install PyYAML (adds YAML parser)
pip install pyyaml

# Verify
# This script runs a Python one-liner to check if the PyYAML library is installed and prints its version.
# Useful for verifying the PyYAML installation and version in the current environment.
python -c "import yaml; print('PyYAML loaded:', yaml.__version__)"


# ------------------------- Stage C: EVALUATION -------------------------------
python -u "${REPO_ROOT}/scripts/run_experiment.py" \
  --config "${CONFIG_PATH}" \
  --mode eval \
  --split "${SPLIT}" \
  --run_dir "${RUN_DIR}"

echo "Finish time : $(date)"
echo "Done."