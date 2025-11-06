#!/bin/bash
# ============================================================================
# eval_cp3_wholeorganoid_cpu_baseline.slurm — Stage C on CPU (fast queue, short walltime)
# Evaluates existing RUN_DIR (resume mode). Writes outputs to $RUN_DIR/ev al/.
# ============================================================================

#SBATCH --job-name=cp3_eval_cpu
#SBATCH --account=parent0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/turbo/umms-parent/cellpose_wholeorganoid_model/logs/slurm/%x_%j.out
#SBATCH --error=/nfs/turbo/umms-parent/cellpose_wholeorganoid_model/logs/slurm/%x_%j.err

set -euo pipefail

# CPU threading hints
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# -------- overrides at submission (via --export=) --------
: "${REPO_ROOT:=${HOME}/Desktop/githubprojects/cellpose_on_umich_hpc}"
: "${CONFIG_PATH:=${REPO_ROOT}/configs/cyto3_base_cp3_wholeorganoid.yaml}"  # base YAML (eval defaults & labels.mask_filter)
: "${RUN_DIR:?Set RUN_DIR=results/<model>/run_YYYY-mm-dd_HHMMSS}"  # REQUIRED: prepared run to evaluate
: "${SPLIT:=all}"
: "${ENV_NAME:=cellpose3}"
# ---------------------------------------------------------

echo "==================== UMICH GREAT LAKES JOB ===================="
echo " Job:        ${SLURM_JOB_NAME}  (ID ${SLURM_JOB_ID:-N/A})"
echo " Node:       $(hostname)"
echo " Repo root:  ${REPO_ROOT}"
echo " Config     : ${CONFIG_PATH}"
echo " Run dir    : ${RUN_DIR}"
echo " Split      : ${SPLIT}"
echo " Env        : ${ENV_NAME}"
echo " Start      : $(date)"
echo "==============================================================="

module load python3.10-anaconda/2023.03
CONDA_BASE=/sw/pkgs/arc/python3.10-anaconda/2023.03
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  eval "$("$CONDA_BASE/bin/conda" shell.bash hook)"
fi
conda activate "${ENV_NAME}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ---------------------- Install dependencies if missing ---------------------

# Install PyYAML (adds YAML parser)
pip install pyyaml

# Verify
# This script runs a Python one-liner to check if the PyYAML library is installed and prints its version.
# Useful for verifying the PyYAML installation and version in the current environment.
python -c "import yaml; print('PyYAML loaded:', yaml.__version__)"

# Install Matplotlib + ImageIO for panel generation
pip install matplotlib imageio
# Verify Matplotlib and ImageIO installation
python -c "import matplotlib, imageio; print('Matplotlib loaded:', matplotlib.__version__); print('ImageIO loaded:', imageio.__version__)"

# --- Sanity: snapshot exists ---
SNAP="${RUN_DIR}/cfg/config_snapshot.yml"
[[ -f "$SNAP" ]] || SNAP="${RUN_DIR}/cfg/config_snapshot.yaml"
if [[ ! -f "$SNAP" ]]; then
  echo "[ERROR] No snapshot at ${RUN_DIR}/cfg/config_snapshot.yaml"; exit 2; fi

# --- Print what we are going to evaluate (paths/knobs read from snapshot) ---
python - <<PY
import yaml, pathlib
snap = pathlib.Path("$SNAP")
cfg  = yaml.safe_load(snap.read_text())
paths = cfg.get("paths",{}) or {}
evalk = cfg.get("eval",{}) or {}
train = cfg.get("train",{}) or {}
print("[SNAP] evaluating with snapshot:", snap)
print("[SNAP] paths.data_images_train :", paths.get("data_images_train"))
print("[SNAP] paths.data_labels_train :", paths.get("data_labels_train"))
print("[SNAP] eval.resample           :", evalk.get("resample"))
print("[SNAP] eval.niter              :", evalk.get("niter"))
print("[SNAP] eval.bsize              :", evalk.get("bsize"))
print("[SNAP] eval.cellprob_threshold :", evalk.get("cellprob_threshold"))
print("[SNAP] eval.flow_threshold     :", evalk.get("flow_threshold"))
print("[SNAP] train.channels          :", train.get("channels"))
print("[SNAP] eval.channels           :", evalk.get("channels"))
PY

# --- Run Stage C on CPU using the run snapshot (no YAML edits) ---
echo
echo ">>> Running Stage C evaluate (CPU) on snapshot (split=${SPLIT})…"
python -u "${REPO_ROOT}/scripts/run_experiment.py" \
  --config "${CONFIG_PATH}" \
  --mode   eval \
  --split  "${SPLIT}" \
  --run_dir "${RUN_DIR}"

echo
echo "Finish time : $(date)"
echo "Done."