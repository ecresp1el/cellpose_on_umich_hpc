#!/bin/bash
# explore_plot_cpu.slurm — save exploratory channel & img+mask panels (CPU)
#SBATCH --job-name=cp3_explore_plot
#SBATCH --account=parent0
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/nfs/turbo/umms-parent/cellpose_wholeorganoid_model/logs/slurm/%x_%j.out
#SBATCH --error=/nfs/turbo/umms-parent/cellpose_wholeorganoid_model/logs/slurm/%x_%j.err
set -euo pipefail

: "${REPO_ROOT:=${HOME}/Desktop/githubprojects/cellpose_on_umich_hpc}"
: "${CONFIG_PATH:=${REPO_ROOT}/configs/cp3_v001.yaml}"
: "${STEM:?Set STEM with --export STEM=<file_stem>}"
: "${ENV_NAME:=cellpose3}"
: "${DPI:=150}"

JOB_ID="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"

echo "==================== UMICH GREAT LAKES JOB ===================="
echo " Job:           ${SLURM_JOB_NAME}  (ID ${JOB_ID})"
echo " Repo root:     ${REPO_ROOT}"
echo " Config (YAML): ${CONFIG_PATH}"
echo " Stem:          ${STEM}"
echo " Conda env:     ${ENV_NAME}"
echo " DPI:           ${DPI}"
echo " Start time:    $(date)"
echo "==============================================================="

module load python3.10-anaconda/2023.03
CONDA_BASE=/sw/pkgs/arc/python3.10-anaconda/2023.03
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  eval "$("$CONDA_BASE/bin/conda" shell.bash hook)"
fi
conda activate "${ENV_NAME}"

# repo on PYTHONPATH (for cp_core.*)
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# deps (already present from your logs; keep quiet)
python - <<'PY'
import importlib
mods = ("numpy","tifffile","imageio","matplotlib","yaml")
for m in mods:
    try:
        importlib.import_module(m); print(f"[OK] {m}")
    except Exception as e:
        print(f"[WARN] missing {m}: {e}")
PY

echo ">>> Running exploratory_plot_images.py (saves two panels)…"
python -u "${REPO_ROOT}/scripts/exploratory_plot_images.py" \
  --config "${CONFIG_PATH}" \
  --stem "${STEM}" \
  --job_id "${JOB_ID}" \
  --dpi "${DPI}" \
  --debug

echo "Finish time : $(date)"
echo "Done."