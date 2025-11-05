#!/bin/bash
# plot_squish_batch_cpu.slurm — validate squished train images by plotting multi-panels (flat squish folder)
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
: "${REPO_ROOT:=$HOME/Desktop/githubprojects/cellpose_on_umich_hpc}"
: "${CONFIG_PATH:=${REPO_ROOT}/configs/cp3_v001.yaml}"  # base YAML (used for labels/results_root/turbo_root)
: "${ENV_NAME:=cellpose3}"
: "${OUT_SUBDIR:=images_squish_max}"                   # flat squish folder under dataset/train/
: "${DPI:=150}"
: "${LIMIT:=0}"                                        # 0 = all
# -----------------------------------------------------------------------------

echo "==================== UMICH GREAT LAKES JOB ===================="
echo " Job:           ${SLURM_JOB_NAME}  (ID ${SLURM_JOB_ID:-N/A})"
echo " Repo root:     ${REPO_ROOT}"
echo " Base YAML:     ${CONFIG_PATH}"
echo " Squish folder: ${OUT_SUBDIR} (dataset/train/${OUT_SUBDIR})"
echo " Conda env:     ${ENV_NAME}"
echo " DPI           : ${DPI}"
echo " Limit         : ${LIMIT}"
echo "==============================================================="

# Env
module load python3.10-anaconda/2023.03
CONDA_BASE=/sw/pkgs/arc/python3.10-anaconda/2023.03
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  eval "$("$CONDA_BASE/bin/conda" shell.bash hook)"
fi
conda activate "${ENV_NAME}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ---------- Enumerate stems from squish dir; require matching labels ----------
STEMS_TXT=$(
python - <<'PY'
import os, sys, yaml, pathlib, glob
cfg = pathlib.Path(os.environ["CONFIG_PATH"])
y   = yaml.safe_load(cfg.read_text())
p   = y.get("paths", {}) or {}
turbo = p.get("turbo_root", "/nfs/turbo/umms-parent/cellpose_wholeorganoid_model")
imgd  = pathlib.Path(turbo) / "dataset" / "train" / os.environ.get("OUT_SUBDIR","images_squish_max")
lbld  = pathlib.Path(p.get("data_labels_train", str(pathlib.Path(turbo)/"dataset"/"train"/"labels")))
mask  = (y.get("labels", {}) or {}).get("mask_filter")

print(f"[DIAG] squish_dir={imgd}", file=sys.stderr)
print(f"[DIAG] labels_dir={lbld}", file=sys.stderr)
print(f"[DIAG] mask_suffix={mask!r}", file=sys.stderr)

imgs = sorted(list(imgd.glob("*.tif")) + list(imgd.glob("*.tiff")))
def has_label(stem):
    c=[]
    if mask: c.append(lbld/f"{stem}{mask}")
    c += [lbld/f"{stem}_masks.tif", lbld/f"{stem}.png", lbld/f"{stem}.npy", lbld/f"{stem}_seg.npy"]
    return any(pp.exists() for pp in c)

stems=[p.stem for p in imgs if has_label(p.stem)]
print(f"[DIAG] matched_squish_stems={len(stems)}", file=sys.stderr)
sys.stdout.write("\n".join(stems))
PY
)

# to array; strip blanks
readarray -t STEMS_ARR <<< "$(printf "%s\n" "$STEMS_TXT" | sed '/^[[:space:]]*$/d')"
TOTAL=${#STEMS_ARR[@]}
echo "[INFO] Found ${TOTAL} squished train stems with labels."
if (( TOTAL == 0 )); then echo "[WARN] No stems to plot."; exit 0; fi
if (( LIMIT > 0 && LIMIT < TOTAL )); then
  echo "[INFO] LIMIT=${LIMIT} → plotting first ${LIMIT} stems."
  STEMS_ARR=("${STEMS_ARR[@]:0:LIMIT}")
  TOTAL=${#STEMS_ARR[@]}
fi

# ---------- Compute override paths once (no temp YAML) ----------
IMAGES_DIR_OVERRIDE=$(python - <<'PY'
import os, yaml, pathlib
cfg = pathlib.Path(os.environ["CONFIG_PATH"])
y   = yaml.safe_load(cfg.read_text())
turbo = (y.get("paths", {}) or {}).get("turbo_root", "/nfs/turbo/umms-parent/cellpose_wholeorganoid_model")
print(str(pathlib.Path(turbo)/"dataset"/"train"/os.environ.get("OUT_SUBDIR","images_squish_max")))
PY
)
LABELS_DIR_OVERRIDE=$(python - <<'PY'
import os, yaml, pathlib
cfg = pathlib.Path(os.environ["CONFIG_PATH"])
y   = yaml.safe_load(cfg.read_text())
p   = y.get("paths", {}) or {}
turbo = p.get("turbo_root", "/nfs/turbo/umms-parent/cellpose_wholeorganoid_model")
print(str(pathlib.Path(p.get("data_labels_train", str(pathlib.Path(turbo)/"dataset"/"train"/"labels")))))
PY
)
RESULTS_ROOT_OVERRIDE=$(python - <<'PY'
import os, yaml, pathlib
cfg = pathlib.Path(os.environ["CONFIG_PATH"])
y   = yaml.safe_load(cfg.read_text())
p   = y.get("paths", {}) or {}
turbo = p.get("turbo_root", "/nfs/turbo/umms-parent/cellpose_wholeorganoid_model")
print(str(pathlib.Path(p.get("results_root", str(pathlib.Path(turbo)/"results")))))
PY
)
MASK_SUFFIX_OVERRIDE=$(python - <<'PY'
import os, yaml
cfg = os.environ["CONFIG_PATH"]
y   = yaml.safe_load(open(cfg))
mask = (y.get("labels", {}) or {}).get("mask_filter")
print("" if mask is None else mask)
PY
)

# ---------- Loop & plot ----------
i=0
for STEM in "${STEMS_ARR[@]}"; do
  echo "----------------------------------------------------------------"
  echo ">>> [$i/${TOTAL}] STEM: ${STEM}"
  python -u "${REPO_ROOT}/scripts/exploratory_plot_images.py" \
    --config "${CONFIG_PATH}" \
    --stem "${STEM}" \
    --job_id "${SLURM_JOB_ID:-manual}_${i}" \
    --dpi "${DPI}" \
    --debug \
    --images_dir_override "${IMAGES_DIR_OVERRIDE}" \
    --labels_dir_override "${LABELS_DIR_OVERRIDE}" \
    --results_root_override "${RESULTS_ROOT_OVERRIDE}" \
    $( [ -n "${MASK_SUFFIX_OVERRIDE}" ] && printf -- '--mask_suffix_override %q' "${MASK_SUFFIX_OVERRIDE}" )
  ((i=i+1))
done

echo "Finish time : $(date)"
echo "Done."