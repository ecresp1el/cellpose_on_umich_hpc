#!/bin/bash
# ============================================================================
# prepare_cp3_wholeorganoid.slurm
#
# Purpose:
#   Stage A is implied — this job PREPARES a run:
#     - Creates a timestamped run_dir under results/<model>/
#     - Snapshots the YAML config & environment for provenance
#     - Verifies dataset structure and writes a dataset report
#
# Logging (two layers, by design):
#   1) SLURM scheduler logs (stdout/err) -> TURBO logs/slurm/
#      - Node info, module/conda activation, uncaught Python tracebacks.
#   2) Pipeline run artifacts (managed by our Python RunLogger) -> run_dir/cfg/
#      - config_snapshot.yaml, env_and_cfg.json, dataset_report.json, prepare_summary.json
#
# Configuration:
#   * CONFIG_PATH defaults to a YAML inside your repo (version-controlled).
#   * The Python program will snapshot that YAML into run_dir/cfg/.
#   * All knobs/tuning live in YAML — do NOT hardcode hyperparams here.
#
# Usage:
#   # default (uses ENV_NAME=cellpose3, REPO_ROOT in $HOME)
#   sbatch slurm/prepare_cp3_wholeorganoid.slurm
#
#   # override env or paths at submission time:
#   sbatch --export=ALL,ENV_NAME=cellpose3,REPO_ROOT=/path/to/cellpose_on_umich_hpc,CONFIG_PATH=/path/to/config.yaml \
#          slurm/prepare_cp3_wholeorganoid.slurm
#
# CPU-only option:
#   - If GPU queue is busy, switch partition to 'standard' and remove --gres line.
#   (This job does not require a GPU; it only sets up the run and verifies data.)
# ============================================================================

#SBATCH --job-name=cp3_prepare_wholeorganoid
#SBATCH --account=parent0
#SBATCH --partition=standard      # <— use the standard (CPU) partition
# #SBATCH --partition=gpu               # switch to 'standard' if GPUs are busysque
# #SBATCH --gres=gpu:1                  # comment out if using CPU partition
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
# Write scheduler logs to TURBO (separate from run_dir artifacts):
#SBATCH --output=/nfs/turbo/umms-parent/cellpose_wholeorganoid_model/logs/slurm/%x_%j.out
#SBATCH --error=/nfs/turbo/umms-parent/cellpose_wholeorganoid_model/logs/slurm/%x_%j.err

set -euo pipefail

# ------------------- OVERRIDES AT SUBMISSION (with --export=) ----------------
: "${REPO_ROOT:=${HOME}/Desktop/githubprojects/cellpose_on_umich_hpc}"   # repo clone on GL home or scratch
: "${CONFIG_PATH:=${REPO_ROOT}/configs/cyto3_base_cp3_wholeorganoid.yaml}"                    # YAML tracked in repo
: "${ENV_NAME:=cellpose3}"                                                # conda env with PyYAML (Stage A only)
# ----------------------------------------------------------------------------

echo "=== NODE & PATHS ==="
echo "Host        : $(hostname)"
echo "Repo root   : ${REPO_ROOT}"
echo "Config YAML : ${CONFIG_PATH}"
echo "SLURM out   : /nfs/turbo/umms-parent/cellpose_wholeorganoid_model/logs/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.{out,err}"
echo "Start time  : $(date)"
echo "======================================"

# ---------------------- UMich ARC Anaconda activation -----------------------
# Mirrors your compare script style: load module then source conda.sh hook.
module load python3.10-anaconda/2023.03
CONDA_BASE=/sw/pkgs/arc/python3.10-anaconda/2023.03
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  eval "$("$CONDA_BASE/bin/conda" shell.bash hook)"
fi
conda activate "${ENV_NAME}"

# Install PyYAML (adds YAML parser)
pip install pyyaml
# Verify
# This script runs a Python one-liner to check if the PyYAML library is installed and prints its version.
# Useful for verifying the PyYAML installation and version in the current environment.
python -c "import yaml; print('PyYAML loaded:', yaml.__version__)"

# ------------------------- Stage A: PREPARE run dir -------------------------
# This calls the method-first entrypoint. It will:
#   - Create results/<model>/run_YYYY-mm-dd_HHMMSS/
#   - Write cfg/config_snapshot.yaml + cfg/env_and_cfg.json
#   - Write cfg/dataset_report.json + cfg/prepare_summary.json
#   - Create empty train/ and eval/ placeholders for future stages.
python -u "${REPO_ROOT}/scripts/run_experiment.py" \
  --config "${CONFIG_PATH}" \
  --mode prepare

echo "Finish time : $(date)"
echo "Done."
