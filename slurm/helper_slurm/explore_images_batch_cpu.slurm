#!/bin/bash
# explore_images_batch_cpu.slurm — run a chosen exploratory script for many train stems
#SBATCH --job-name=cp3_explore_batch
#SBATCH --account=parent0
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/turbo/umms-parent/cellpose_wholeorganoid_model/logs/slurm/%x_%j.out
#SBATCH --error=/nfs/turbo/umms-parent/cellpose_wholeorganoid_model/logs/slurm/%x_%j.err
set -euo pipefail

# ------------------- OVERRIDES AT SUBMISSION (with --export=) ----------------
: "${REPO_ROOT:=${HOME}/Desktop/githubprojects/cellpose_on_umich_hpc}"
: "${CONFIG_PATH:=${REPO_ROOT}/configs/cp3_v001.yaml}"   # YAML is source of truth (contract §3)
: "${ENV_NAME:=cellpose3}"
: "${SCRIPT_NAME:=exploratory_analysis_images.py}"       # or exploratory_plot_images.py
: "${CHANNELS:=1}"                                       # used only by *analysis* script
: "${DPI:=150}"                                          # used only by *plot* script
: "${LIMIT:=0}"                                          # 0 = no limit
# ----------------------------------------------------------------------------

echo "==================== UMICH GREAT LAKES JOB ===================="
echo " Job:           ${SLURM_JOB_NAME}  (ID ${SLURM_JOB_ID:-N/A})"
echo " Node:          $(hostname)"
echo " Repo root:     ${REPO_ROOT}"
echo " Config (YAML): ${CONFIG_PATH}"
echo " Conda env:     ${ENV_NAME}"
echo " Script        : ${SCRIPT_NAME}"
echo " Channels norm : ${CHANNELS}  (analysis-only)"
echo " DPI           : ${DPI}        (plotting-only)"
echo " Start time    : $(date)"
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
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ---------------------- Enumerate matching stems (YAML-driven) --------------
STEMS=$(python - <<'PY'
import os, yaml, pathlib, sys
cfg = pathlib.Path(os.environ["CONFIG_PATH"])
y   = yaml.safe_load(cfg.read_text())

p = y.get("paths", {}) or {}
turbo_root = p.get("turbo_root", "/nfs/turbo/umms-parent/cellpose_wholeorganoid_model")
images_dir = pathlib.Path(p.get("data_images_train", "")) if p.get("data_images_train") else pathlib.Path(turbo_root)/"dataset/train/images"
labels_dir = pathlib.Path(p.get("data_labels_train", "")) if p.get("data_labels_train") else pathlib.Path(turbo_root)/"dataset/train/labels"
mask_suf   = (y.get("labels", {}) or {}).get("mask_filter")

# DIAGNOSTICS → STDERR (so they don't pollute the stems list)
print(f"[DIAG] images_dir = {images_dir}", file=sys.stderr)
print(f"[DIAG] labels_dir = {labels_dir}", file=sys.stderr)
print(f"[DIAG] mask_suffix = {mask_suf!r}", file=sys.stderr)

imgs = sorted(list(images_dir.glob("*.tif")) + list(images_dir.glob("*.tiff")))
print(f"[DIAG] #image files: {len(imgs)}", file=sys.stderr)
lbls = sorted(labels_dir.glob("*"))
print(f"[DIAG] #label files: {len(lbls)}", file=sys.stderr)

def has_label(stem:str)->bool:
    cands = []
    if mask_suf:
        cands.append(labels_dir / f"{stem}{mask_suf}")
    cands += [
        labels_dir / f"{stem}_masks.tif",
        labels_dir / f"{stem}.png",
        labels_dir / f"{stem}.npy",
        labels_dir / f"{stem}_seg.npy",
    ]
    return any(pp.exists() for pp in cands)

stems = [p.stem for p in imgs if has_label(p.stem)]
print(f"[DIAG] matched stems (image+label): {len(stems)}", file=sys.stderr)
for s in stems[:5]:
    print("[DIAG] stem example:", s, file=sys.stderr)

# STEMS → STDOUT (one per line)
sys.stdout.write("\n".join(stems))
PY
)
# Turn newline-separated stems into bash array
readarray -t STEM_ARR <<< "$STEMS"
TOTAL=${#STEM_ARR[@]}
echo "[INFO] Found $TOTAL matching stems."

if (( TOTAL == 0 )); then
  echo "[WARN] No stems found. Check [DIAG] above (paths & mask suffix)."
  echo "Finish time : $(date)"; echo "Done."; exit 0
fi

if (( LIMIT > 0 && LIMIT < TOTAL )); then
  echo "[INFO] LIMIT=${LIMIT} → will process first ${LIMIT} stems."
  STEM_ARR=("${STEM_ARR[@]:0:LIMIT}")
  TOTAL=${#STEM_ARR[@]}
fi

# ---------------------- Loop stems (simple, like your 1-image job) ----------
i=0
for STEM in "${STEM_ARR[@]}"; do
  echo "----------------------------------------------------------------"
  echo ">>> [$i/${TOTAL}] STEM: ${STEM}"
  if [[ "${SCRIPT_NAME}" == "exploratory_plot_images.py" ]]; then
    # plotting path -> saves under results/qc_panels/ (contract §0)
    python -u "${REPO_ROOT}/scripts/${SCRIPT_NAME}" \
      --config "${CONFIG_PATH}" \
      --stem "${STEM}" \
      --job_id "${SLURM_JOB_ID:-manual}_${i}" \
      --dpi "${DPI}" \
      --debug
  else
    # analysis/print-only path
    python -u "${REPO_ROOT}/scripts/${SCRIPT_NAME}" \
      --config "${CONFIG_PATH}" \
      --stem "${STEM}" \
      $([ "${CHANNELS}" -eq 1 ] && echo "--channels") \
      --debug
  fi
  ((i=i+1))
done

# ---------------------- Finish ------------------------------------------------
echo "Finish time : $(date)"
echo "Done."