#!/bin/bash
# explore_images_cpu.slurm — run scripts/exploratory_analysis_images.py
#SBATCH --job-name=cp3_explore_cpu
#SBATCH --account=parent0
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/nfs/turbo/umms-parent/cellpose_wholeorganoid_model/logs/slurm/%x_%j.out
#SBATCH --error=/nfs/turbo/umms-parent/cellpose_wholeorganoid_model/logs/slurm/%x_%j.err

set -euo pipefail

# ------------------- OVERRIDES AT SUBMISSION (with --export=) ----------------
: "${REPO_ROOT:=${HOME}/Desktop/githubprojects/cellpose_on_umich_hpc}"
: "${CONFIG_PATH:=${REPO_ROOT}/configs/cp3_v001.yaml}"   # Use YAML (contract §3) for paths.
: "${STEM:?Set STEM with --export STEM=<file_stem>}"     # The filename *without* extension
: "${ENV_NAME:=cellpose3}"                               # Your conda env
: "${CHANNELS:=1}"                                       # 1=per-channel normalization for RGB; 0=global
# ----------------------------------------------------------------------------

echo "==================== UMICH GREAT LAKES JOB ===================="
echo " Job:           ${SLURM_JOB_NAME}  (ID ${SLURM_JOB_ID:-N/A})"
echo " Node:          $(hostname)"
echo " Repo root:     ${REPO_ROOT}"
echo " Config (YAML): ${CONFIG_PATH}   # paths come from this file"
echo " Image stem:    ${STEM}          # e.g., MyImage_ABC_001"
echo " Conda env:     ${ENV_NAME}"
echo " Channels norm: ${CHANNELS}      # 1=per RGB channel, 0=global"
echo " Start time:    $(date)"
echo "==============================================================="

# ---------------------- UMich ARC Anaconda activation -----------------------
module load python3.10-anaconda/2023.03
CONDA_BASE=/sw/pkgs/arc/python3.10-anaconda/2023.03
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  eval "$("$CONDA_BASE/bin/conda" shell.bash hook)"
fi
conda activate "${ENV_NAME}"

# Make repository code importable (so 'from cp_core....' works)
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# ---------------------- PYTHON ENVIRONMENT DEBUG (for clarity) --------------
echo
echo "---------------- PYTHON / CONDA ENV CHECK ----------------"
python - <<'PY'
import sys, os, platform
print("[DEBUG] Python executable :", sys.executable)
print("[DEBUG] Python version    :", sys.version.split()[0])
print("[DEBUG] Conda prefix       :", os.environ.get("CONDA_PREFIX"))
print("[DEBUG] Active env         :", os.environ.get("CONDA_DEFAULT_ENV"))
print("[DEBUG] First sys.path     :", (sys.path[0] if sys.path else "<empty>"))
print("[DEBUG] Repo in sys.path?  :", any("cellpose_on_umich_hpc" in p for p in sys.path))
print("[DEBUG] Platform           :", platform.platform())
PY
echo "----------------------------------------------------------------"
echo

# Install the minimial scientific Python stack if not already available
pip install numpy tifffile scikit-image pyyaml

# ---------------------- RUN (print-only exploratory script) -----------------
python -u "${REPO_ROOT}/scripts/exploratory_analysis_images.py" \
  --config "${CONFIG_PATH}" \
  --stem "${STEM}" \
  $([ "${CHANNELS}" -eq 1 ] && echo "--channels") \
  --debug

# ---------------------- Finish ------------------------------------------------
echo "Finish time : $(date)"
echo "Done."