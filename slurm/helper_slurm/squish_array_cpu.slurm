#!/bin/bash
# squish_array_cpu.slurm — collapse train images per array task (flat output)
#SBATCH --job-name=cp3_squish_arr
#SBATCH --account=parent0
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/nfs/turbo/umms-parent/cellpose_wholeorganoid_model/logs/slurm/%x_%A_%a.out
#SBATCH --error=/nfs/turbo/umms-parent/cellpose_wholeorganoid_model/logs/slurm/%x_%A_%a.err
set -euo pipefail

: "${REPO_ROOT:=$HOME/Desktop/githubprojects/cellpose_on_umich_hpc}"
: "${CONFIG_PATH:=${REPO_ROOT}/configs/cp3_v001.yaml}"
: "${ENV_NAME:=cellpose3}"
: "${MODE:=max}"                 # max | mean | sum
: "${OUT_SUBDIR:=}"             # default to images_squish_<mode> if empty

echo "=== SQUISH ARRAY (flat) ==="
echo "Array: ${SLURM_ARRAY_JOB_ID:-N/A}[${SLURM_ARRAY_TASK_ID:-N/A}]  Mode=${MODE}  Out=${OUT_SUBDIR:-<default>}"
echo "Config: ${CONFIG_PATH}"

module load python3.10-anaconda/2023.03
CONDA_BASE=/sw/pkgs/arc/python3.10-anaconda/2023.03
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Build stems (paired with labels) from YAML; stems only to stdout
STEMS=$(
python - <<'PY'
import os, yaml, pathlib, sys
cfg = pathlib.Path(os.environ["CONFIG_PATH"])
y   = yaml.safe_load(cfg.read_text())
p   = y.get("paths", {}) or {}
turbo = pathlib.Path(p.get("turbo_root", "/nfs/turbo/umms-parent/cellpose_wholeorganoid_model"))
imgd  = pathlib.Path(p.get("data_images_train", "")) if p.get("data_images_train") else turbo/"dataset/train/images"
lbld  = pathlib.Path(p.get("data_labels_train", "")) if p.get("data_labels_train") else turbo/"dataset/train/labels"
mask  = (y.get("labels", {}) or {}).get("mask_filter")

def has_label(stem):
    c=[]
    if mask: c.append(lbld/f"{stem}{mask}")
    c += [lbld/f"{stem}_masks.tif", lbld/f"{stem}.png", lbld/f"{stem}.npy", lbld/f"{stem}_seg.npy"]
    return any(pp.exists() for pp in c)

stems=[p.stem for p in sorted(list(imgd.glob("*.tif")) + list(imgd.glob("*.tiff"))) if has_label(p.stem)]
sys.stdout.write("\n".join(stems))
PY
)

readarray -t STEMS_ARR <<< "$(printf "%s\n" "$STEMS" | sed '/^[[:space:]]*$/d')"
N=${#STEMS_ARR[@]}
echo "[INFO] matched stems: $N"
if (( N == 0 )); then echo "[WARN] nothing to do"; exit 0; fi

IDX=${SLURM_ARRAY_TASK_ID:?set --array=0-$((N-1))}
if (( IDX < 0 || IDX >= N )); then echo "[SKIP] out of range ${IDX}"; exit 0; fi

STEM="${STEMS_ARR[$IDX]}"
echo ">>> STEM: ${STEM}"

python -u "${REPO_ROOT}/scripts/squish_one_stem.py" \
  --config "${CONFIG_PATH}" \
  --stem "${STEM}" \z
  --mode "${MODE}" \
  $([ -n "${OUT_SUBDIR}" ] && echo --out_subdir "${OUT_SUBDIR}") \
  --debug

echo "Done."