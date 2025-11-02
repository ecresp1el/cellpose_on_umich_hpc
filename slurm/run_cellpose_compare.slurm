#!/bin/bash
# ============================================================================
# run_cellpose_compare.slurm  —  Compare Cellpose v3.1.1.2 vs v4.0.7 for whole-organoid masks
#
# This SLURM file lives IN your repo; logs & results live on TURBO (external).
# ============================================================================
#SBATCH --job-name=cp_compare
#SBATCH --account=parent0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
# Write logs to Turbo, NOT the repo:
#SBATCH --output=/nfs/turbo/umms-parent/cellpose_gpu_test/logs/%x_%j.out
#SBATCH --error=/nfs/turbo/umms-parent/cellpose_gpu_test/logs/%x_%j.err

# ------------------- OVERRIDES AT SUBMISSION (with --export=) ----------------
: "${TEST_ROOT:=/nfs/turbo/umms-parent/cellpose_gpu_test}"       # external test dataset (fixed location)
: "${ENV_NAME:=cellpose4}"                                       # cellpose4 or cellpose3
: "${COND:=WT}"                                                  # WT or KO
: "${DIAMETER:=1200}"                                            # expected organoid diameter (px)
: "${FLOW_THRESHOLD:=0.7}"                                       # conservative to avoid over-splitting
: "${MIN_SIZE:=100000}"                                          # filter debris (< this many pixels)
: "${CHAN:=0}"                                                   # grayscale / single-channel TIFFs
# ----------------------------------------------------------------------------

REPO_ROOT="/home/elcrespo/Desktop/githubprojects/cellpose_on_umich_hpc"
PY_RUNNER="${REPO_ROOT}/scripts/cp_compare.py"
IMG_DIR="${TEST_ROOT}/images/${COND}"
OUT_DIR="${TEST_ROOT}/results/${COND}/${ENV_NAME}"
LOG_DIR="${TEST_ROOT}/logs"

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

echo "=== NODE & PATHS ==="
echo "Host        : $(hostname)"
echo "Repo root   : ${REPO_ROOT}"
echo "Python file : ${PY_RUNNER}"
echo "Image dir   : ${IMG_DIR}"
echo "Output dir  : ${OUT_DIR}"
echo "Env         : ${ENV_NAME}"
echo "======================================"

# Load cluster Anaconda & activate requested conda env
module load python3.10-anaconda/2023.03
CONDA_BASE=/sw/pkgs/arc/python3.10-anaconda/2023.03
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  eval "$("$CONDA_BASE/bin/conda" shell.bash hook)"
fi
conda activate "${ENV_NAME}"

# Run the Python wrapper
python "${PY_RUNNER}" \
  --img_dir "${IMG_DIR}" \
  --out_dir "${OUT_DIR}" \
  --diameter "${DIAMETER}" \
  --flow_threshold "${FLOW_THRESHOLD}" \
  --min_size "${MIN_SIZE}" \
  --chan "${CHAN}"