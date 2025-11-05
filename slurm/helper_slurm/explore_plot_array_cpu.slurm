#!/bin/bash
# explore_plot_auto_cpu.slurm — programmatically enumerates stems from YAML and runs plotting
#SBATCH --job-name=cp3_explore_plot_auto
#SBATCH --account=parent0
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/turbo/umms-parent/cellpose_wholeorganoid_model/logs/slurm/%x_%j.out
#SBATCH --error=/nfs/turbo/umms-parent/cellpose_wholeorganoid_model/logs/slurm/%x_%j.err
set -euo pipefail

# ------------------- OVERRIDES AT SUBMISSION (with --export=) ----------------
: "${REPO_ROOT:=$HOME/Desktop/githubprojects/cellpose_on_umich_hpc}"
: "${CONFIG_PATH:=${REPO_ROOT}/configs/cp3_v001.yaml}"
: "${ENV_NAME:=cellpose3}"
: "${DPI:=150}"
# -----------------------------------------------------------------------------

echo "==================== UMICH GREAT LAKES JOB ===================="
echo " Job:           ${SLURM_JOB_NAME}  (ID ${SLURM_JOB_ID:-N/A})"
echo " Repo root:     ${REPO_ROOT}"
echo " Config (YAML): ${CONFIG_PATH}"
echo " Conda env:     ${ENV_NAME}"
echo " DPI:           ${DPI}"
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

# Make repo importable (cp_core.*)
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ---------------------- Build stems programmatically (YAML-driven) ----------
# We output NUL-separated stems so filenames with + or spaces are safe.
STEMS_NUL=$(python - <<'PY'
import sys, yaml, pathlib, json, os
cfg = pathlib.Path(os.environ["CONFIG_PATH"])
y   = yaml.safe_load(cfg.read_text())
p   = y.get("paths", {}) or {}
turbo_root   = p.get("turbo_root", "/nfs/turbo/umms-parent/cellpose_wholeorganoid_model")
images_dir   = pathlib.Path(p.get("data_images_train", "")) if p.get("data_images_train") else pathlib.Path(turbo_root)/"dataset/train/images"
labels_dir   = pathlib.Path(p.get("data_labels_train", "")) if p.get("data_labels_train") else pathlib.Path(turbo_root)/"dataset/train/labels"
mask_suffix  = (y.get("labels", {}) or {}).get("mask_filter")

def has_label(stem:str) -> bool:
    # YAML suffix first, then fallbacks (same as exploratory reader)
    cands = []
    if mask_suffix:
        cands.append(labels_dir / f"{stem}{mask_suffix}")
    cands += [
        labels_dir / f"{stem}_masks.tif",
        labels_dir / f"{stem}.png",
        labels_dir / f"{stem}.npy",
        labels_dir / f"{stem}_seg.npy",
    ]
    return any(p.exists() for p in cands)

stems = []
for ext in ("*.tif","*.tiff"):
    for pth in sorted(images_dir.glob(ext)):
        st = pth.stem
        if has_label(st):
            stems.append(st)

# print NUL-separated list to stdout
sys.stdout.write("\0".join(stems))
PY
)

# Convert NUL-separated to bash array safely
IFS= read -r -d '' || true   # initialize read builtin
STEMS=()
while IFS= read -r -d '' s; do STEMS+=("$s"); done < <(printf '%s' "$STEMS_NUL")

N=${#STEMS[@]}
echo "[INFO] Found $N train stems with matching labels (YAML mask_filter honored)."
if (( N == 0 )); then
  echo "[WARN] No stems found. Exiting."
  exit 0
fi

# ---------------------- Helper to run one stem ------------------------------
run_one() {
  local stem="$1"
  local jid="${2}"
  echo ">>> STEM: ${stem}  (job_tag=${jid})"
  python -u "${REPO_ROOT}/scripts/exploratory_plot_images.py" \
    --config "${CONFIG_PATH}" \
    --stem "${stem}" \
    --job_id "${jid}" \
    --dpi "${DPI}" \
    --debug
}

# ---------------------- Array vs. single-job loop ---------------------------
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  idx="${SLURM_ARRAY_TASK_ID}"
  if (( idx < 0 || idx >= N )); then
    echo "[SKIP] SLURM_ARRAY_TASK_ID=${idx} out of range 0..$((N-1))"
    exit 0
  fi
  run_one "${STEMS[$idx]}" "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${idx}"
else
  echo "[INFO] No array index detected; running all $N stems in a single job."
  i=0
  for s in "${STEMS[@]}"; do
    run_one "$s" "${SLURM_JOB_ID:-manual}_$i"
    i=$((i+1))
  done
fi

echo "Finish time : $(date)"
echo "Done."