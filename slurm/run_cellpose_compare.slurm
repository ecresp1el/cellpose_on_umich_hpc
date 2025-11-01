#!/bin/bash
# ============================================================================
# run_cellpose_compare.slurm  —  Compare Cellpose v3.1.1.2 vs v4.0.7 for whole-organoid masks
#
# AUDIENCE
#   Non-programmers. This file explains exactly what happens and what to change.
#
# LAYOUT (repo vs data)
#   • This SLURM file lives IN your repo:   cellpose_on_umich_hpc/slurm/run_cellpose_compare.slurm
#   • The Python runner lives IN your repo: cellpose_on_umich_hpc/scripts/cp_compare.py
#   • TEST DATA lives OUTSIDE the repo (not versioned): /nfs/turbo/umms-parent/cellpose_gpu_test
#       images/WT  and  images/KO  are assumed to exist and hold your .tif files.
#
# HOW TO RUN (no code edits; submit from your repo):
#   cd /home/elcrespo/Desktop/githubprojects/cellpose_on_umich_hpc/slurm
#
#   # WT with Cellpose 4.0.7
#   sbatch run_cellpose_compare.slurm --export=ALL,ENV_NAME=cellpose4,COND=WT,DIAMETER=1200,FLOW_THRESHOLD=0.7,MIN_SIZE=100000
#
#   # KO with Cellpose 4.0.7
#   sbatch run_cellpose_compare.slurm --export=ALL,ENV_NAME=cellpose4,COND=KO,DIAMETER=1200,FLOW_THRESHOLD=0.7,MIN_SIZE=100000
#
#   # WT with Cellpose 3.1.1.2
#   sbatch run_cellpose_compare.slurm --export=ALL,ENV_NAME=cellpose3,COND=WT,DIAMETER=1200,FLOW_THRESHOLD=0.7,MIN_SIZE=100000
#
#   # KO with Cellpose 3.1.1.2
#   sbatch run_cellpose_compare.slurm --export=ALL,ENV_NAME=cellpose3,COND=KO,DIAMETER=1200,FLOW_THRESHOLD=0.7,MIN_SIZE=100000
#
# VARIABLE MEANINGS
#   ENV_NAME       : conda environment to use (cellpose3 or cellpose4)
#   COND           : WT or KO — selects images from TEST_ROOT/images/<COND>
#   DIAMETER       : organoid diameter in pixels (~1200 for your data)
#   FLOW_THRESHOLD : 0.6–0.8; higher is more conservative (helps avoid over-splitting)
#   MIN_SIZE       : discard small debris; increase if needed
#   IMG_FILTER     : which files to run on (default '*.tif')
#
# NOTES
#   • Images: merged max-projection single-channel uint16 TIFFs (YX), size ~3840×3840.
#   • We set --chan 0 to use the grayscale plane as-is.
#   • No CSV/metrics — only masks/overlays for visual inspection, per your request.
# ============================================================================
#SBATCH --job-name=cp_compare
#SBATCH --account=parent0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ------------------- OVERRIDES AT SUBMISSION (with --export=) ----------------
: "${TEST_ROOT:=/nfs/turbo/umms-parent/cellpose_gpu_test}"       # external test dataset (fixed location)
: "${ENV_NAME:=cellpose4}"                                       # cellpose4 or cellpose3
: "${COND:=WT}"                                                  # WT or KO
: "${IMG_FILTER:=*.tif}"                                         # which files to include
: "${DIAMETER:=1200}"                                            # expected organoid diameter (px)
: "${FLOW_THRESHOLD:=0.7}"                                       # conservative to avoid over-splitting
: "${MIN_SIZE:=100000}"                                          # filter debris (< this many pixels)
: "${CHAN:=0}"                                                   # grayscale / single-channel TIFFs
# ----------------------------------------------------------------------------

REPO_ROOT="/home/elcrespo/Desktop/githubprojects/cellpose_on_umich_hpc"
PY_RUNNER="${REPO_ROOT}/scripts/cp_compare.py"
IMG_DIR="${TEST_ROOT}/images/${COND}"
OUT_DIR="${TEST_ROOT}/results/${COND}/${ENV_NAME}"

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

# Run the Python wrapper (no inline Python in SLURM)
python "${PY_RUNNER}"   --img_dir "${IMG_DIR}"   --out_dir "${OUT_DIR}"   --img_filter "${IMG_FILTER}"   --diameter "${DIAMETER}"   --flow_threshold "${FLOW_THRESHOLD}"   --min_size "${MIN_SIZE}"   --chan "${CHAN}"
