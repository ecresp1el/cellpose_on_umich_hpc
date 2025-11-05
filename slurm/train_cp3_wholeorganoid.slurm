#!/bin/bash
# ============================================================================
# train_cp3_wholeorganoid.slurm
#
# Purpose:
#   PREPARE + TRAIN in one job (Option A).
#   - Stage A: create run_dir, snapshot YAML/env, verify dataset.
#   - Stage B: train specialist model (rescale=False, diameter=1350, bsize=512).
#
# Notes:
#   - Training is GPU-only. Uses Cellpose v3 in CUDA with your conda env.
#   - No YAML rewriting. If a flat squish folder exists (e.g. dataset/train/images_squish_max/),
#     your prefer_squished_train_images(cfg) hook will auto-swap paths before snapshot and set [0,0].
#
# Logging:
#   - SLURM scheduler logs -> TURBO logs/slurm/
#   - Training console -> run/train/stdout_stderr.log (via RunLogger.tee_stdout)
#
# Usage:
#   sbatch slurm/train_cp3_wholeorganoid.slurm
#   # override env/paths at submit:
#   sbatch --export=ALL,ENV_NAME=cellpose3,REPO_ROOT=/path/to/repo,CONFIG_PATH=/path/to/config.yaml \
#          slurm/train_cp3_wholeorganoid.slurm
# ============================================================================

#SBATCH --job-name=cp3_fulltrain_wholeorganoid
#SBATCH --account=parent0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/turbo/umms-parent/cellpose_wholeorganoid_model/logs/slurm/%x_%j.out
#SBATCH --error=/nfs/turbo/umms-parent/cellpose_wholeorganoid_model/logs/slurm/%x_%j.err


set -euo pipefail

# ------------------- OVERRIDES AT SUBMISSION (with --export=) ----------------
: "${REPO_ROOT:=${HOME}/Desktop/githubprojects/cellpose_on_umich_hpc}"   # repo clone on GL
: "${CONFIG_PATH:=${REPO_ROOT}/configs/cp3_v001.yaml}"                    # YAML tracked in repo
: "${ENV_NAME:=cellpose3}"                                                # conda env with Cellpose v3 + PyYAML
# ----------------------------------------------------------------------------

echo "=== NODE & PATHS ==="
echo "Host        : $(hostname)"
echo "Repo root   : ${REPO_ROOT}"
echo "Config YAML : ${CONFIG_PATH}"
echo "Mode        : full-train (prepare + train)"
echo "Start time  : $(date)"
echo "======================================"

# ---------------------- UMich ARC Anaconda activation -----------------------
module load python3.10-anaconda/2023.03
CONDA_BASE=/sw/pkgs/arc/python3.10-anaconda/2023.03
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  eval "$("$CONDA_BASE/bin/conda" shell.bash hook)"
fi
conda activate "${ENV_NAME}"

# Guard: YAML must exist
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[ERROR] CONFIG_PATH not found: $CONFIG_PATH"
  exit 2
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# --- GPU / env sanity (short, to stdout so it lands in .out) ---
python - <<'PY'
import os, sys, yaml, pathlib
print("[ENV] python     :", sys.executable)
try:
    import torch
    print("[GPU] torch     :", torch.__version__, "cuda?", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[GPU] device    :", torch.cuda.get_device_name(0))
except Exception as e:
    print("[GPU] WARN torch:", e)
try:
    import cellpose
    print("[CP3] cellpose  :", getattr(cellpose, "__version__", "<unknown>"))
except Exception as e:
    print("[CP3] WARN      :", e)
cfg = pathlib.Path(os.environ["CONFIG_PATH"])
y   = yaml.safe_load(cfg.read_text())
p   = y.get("paths",{}) or {}
print("[CFG] turbo_root        :", p.get("turbo_root"))
print("[CFG] data_images_train :", p.get("data_images_train"))
print("[CFG] data_labels_train :", p.get("data_labels_train"))
print("[CFG] results_root      :", p.get("results_root"))
print("[CFG] labels.mask_filter:", (y.get("labels",{}) or {}).get("mask_filter"))
print("[NOTE] If dataset/train/images_squish_* exists, prefer_squished_train_images() will swap before snapshot.")
PY

# ------------------------- Stage A + Stage B (full-train) -------------------
python -u "${REPO_ROOT}/scripts/run_experiment.py" \
  --config "${CONFIG_PATH}" \
  --mode full-train

echo "Finish time : $(date)"
echo "Done."