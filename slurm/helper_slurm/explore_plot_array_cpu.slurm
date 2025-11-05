#!/bin/bash
#SBATCH --job-name=cp3_explore_plot_arr
#SBATCH --account=parent0
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --array=0-99         # adjust after you know list length
#SBATCH --output=/nfs/turbo/umms-parent/cellpose_wholeorganoid_model/logs/slurm/%x_%A_%a.out
#SBATCH --error=/nfs/turbo/umms-parent/cellpose_wholeorganoid_model/logs/slurm/%x_%A_%a.err
set -euo pipefail

: "${REPO_ROOT:=$HOME/Desktop/githubprojects/cellpose_on_umich_hpc}"
: "${CONFIG_PATH:=${REPO_ROOT}/configs/cp3_v001.yaml}"
: "${STEMS_FILE:=${REPO_ROOT}/slurm/helper_slurm/train_stems.txt}"
: "${ENV_NAME:=cellpose3}"
: "${DPI:=150}"

STEM=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$STEMS_FILE")
if [ -z "${STEM:-}" ]; then
  echo "[SKIP] No stem at index $SLURM_ARRAY_TASK_ID"
  exit 0
fi

echo "=== ARRAY JOB ==="
echo "JobID: ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}  Task: ${SLURM_ARRAY_TASK_ID}"
echo "Stem : $STEM"

module load python3.10-anaconda/2023.03
CONDA_BASE=/sw/pkgs/arc/python3.10-anaconda/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# show CLI header so we’re sure we’re running the new script
echo ">>> Python: $(which python)"; python -V
sed -n '1,40p' "${REPO_ROOT}/scripts/exploratory_plot_images.py"

python -u "${REPO_ROOT}/scripts/exploratory_plot_images.py" \
  --config "${CONFIG_PATH}" \
  --stem "${STEM}" \
  --job_id "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
  --dpi "${DPI}" \
  --debug

echo "Finish time : $(date)"
echo "Done."