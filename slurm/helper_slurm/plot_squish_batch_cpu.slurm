#!/bin/bash
# plot_squish_batch_cpu.slurm — multi-panel plotting over squished train images (flat, no job subdirs in input)
# Reuses scripts/exploratory_plot_images.py, writing to results/qc_panels/job_<JOBID>_<idx>/<stem>/...
#SBATCH --job-name=cp3_plot_squish_batch
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
: "${CONFIG_PATH:=${REPO_ROOT}/configs/cp3_v001.yaml}"     # base YAML (contract §3)
: "${ENV_NAME:=cellpose3}"
: "${OUT_SUBDIR:=images_squish_max}"                       # e.g. images_squish_max | images_squish_mean | images_squish_sum
: "${DPI:=150}"
: "${LIMIT:=0}"                                            # 0 = all
# -----------------------------------------------------------------------------

echo "==================== UMICH GREAT LAKES JOB ===================="
echo " Job:           ${SLURM_JOB_NAME}  (ID ${SLURM_JOB_ID:-N/A})"
echo " Repo root:     ${REPO_ROOT}"
echo " Base YAML:     ${CONFIG_PATH}"
echo " Squish folder: ${OUT_SUBDIR} (under dataset/train/)"
echo " Conda env:     ${ENV_NAME}"
echo " DPI:           ${DPI}"
echo " Limit:         ${LIMIT}"
echo " Start time:    $(date)"
echo "==============================================================="

# ---------------------- Activate env ----------------------------
module load python3.10-anaconda/2023.03
CONDA_BASE=/sw/pkgs/arc/python3.10-anaconda/2023.03
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  eval "$("$CONDA_BASE/bin/conda" shell.bash hook)"
fi
conda activate "${ENV_NAME}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ---------------------- Build temp YAML (override images->squish) ------------
TMP_CFG="${SLURM_TMPDIR:-/tmp}/cfg_squish_${SLURM_JOB_ID:-manual}.yaml"
python - <<PY
import os, yaml, pathlib, sys
base = pathlib.Path("${CONFIG_PATH}")
cfg  = yaml.safe_load(base.read_text())
paths = cfg.get("paths", {}) or {}
turbo = paths.get("turbo_root", "/nfs/turbo/umms-parent/cellpose_wholeorganoid_model")
squish = pathlib.Path(turbo)/"dataset"/"train"/"${OUT_SUBDIR}"
paths["data_images_train"] = str(squish)
cfg["paths"] = paths
# we do NOT change labels; only images
yaml.safe_dump(cfg, open("${TMP_CFG}","w"))
print("[CFG] wrote temp YAML:", "${TMP_CFG}")
print("[CFG] data_images_train:", paths["data_images_train"])
print("[CFG] data_labels_train:", paths.get("data_labels_train"))
PY

# ---------------------- Enumerate stems from squish dir (paired w/ labels) ---
STEMS_TXT=$(
python - <<'PY'
import os, yaml, pathlib, sys, glob
cfg = pathlib.Path(os.environ["TMP_CFG"])
y   = yaml.safe_load(cfg.read_text())
p   = y.get("paths", {}) or {}
turbo = p.get("turbo_root", "/nfs/turbo/umms-parent/cellpose_wholeorganoid_model")
imgd  = pathlib.Path(p["data_images_train"])
lbld  = pathlib.Path(p.get("data_labels_train", pathlib.Path(turbo)/"dataset/tri
n/labels"))
mask  = (y.get("labels", {}) or {}).get("mask_filter")

imgs = sorted(list(imgd.glob("*.tif")) + list(imgd.glob("*.tiff")))
def has_label(stem):
    c=[]
    if mask: c.append(lbld/f"{stem}{mask}")
    c += [lbld/f"{stem}_masks.tif", lbld/f"{stem}.png", lbld/f"{stem}.npy", lbld/f"{stem}_seg.npy"]
    return any(pp.exists() for pp in c)

stems=[p.stem for p in imgs if has_label(p.stem)]
print(f"[DIAG] squish_dir={imgd}", file=sys.stderr)
print(f"[DIAG] labels_dir={lbld}", file=sys.stderr)
print(f"[DIAG] mask_suffix={mask!r}", file=sys.stderr)
print(f"[DIAG] matched_squish_stems={len(stems)}", file=sys.stderr)
sys.stdout.write("\n".join(stems))
PY
)

# to array; strip blanks/diag (belt & suspenders)
readarray -t STEMS_ARR <<< "$(printf "%s\n" "$STEMS_TXT" | grep -v '^\[DIAG\]' | sed '/^[[:space:]]*$/d')"
TOTAL=${#STEMS_ARR[@]}
echo "[INFO] Found ${TOTAL} squished train stems with labels."
if (( TOTAL == 0 )); then
  echo "[WARN] No squished stems found; nothing to plot."; echo "Done."; exit 0; fi

if (( LIMIT > 0 && LIMIT < TOTAL )); then
  echo "[INFO] LIMIT=${LIMIT} → will plot first ${LIMIT} stems."
  STEMS_ARR=("${STEMS_ARR[@]:0:LIMIT}")
  TOTAL=${#STEMS_ARR[@]}
fi

# ---------------------- Loop and plot ----------------------------------------
i=0
for STEM in "${STEMS_ARR[@]}"; do
  echo "----------------------------------------------------------------"
  echo ">>> [$i/${TOTAL}] STEM: ${STEM}"
  echo ">>> CMD: python -u ${REPO_ROOT}/scripts/exploratory_plot_images.py --config ${TMP_CFG} --stem \"${STEM}\" --job_id ${SLURM_JOB_ID:-manual}_${i} --dpi ${DPI} --debug"
  python -u "${REPO_ROOT}/scripts/exploratory_plot_images.py" \
    --config "${TMP_CFG}" \
    --stem "${STEM}" \
    --job_id "${SLURM_JOB_ID:-manual}_${i}" \
    --dpi "${DPI}" \
    --debug
  ((i=i+1))
done

echo "Finish time : $(date)"
echo "Done."