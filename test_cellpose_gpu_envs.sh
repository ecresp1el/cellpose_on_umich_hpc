#!/usr/bin/env bash
# ============================================================================
# test_cellpose_gpu_envs.sh  —  One‑click GPU check for Cellpose envs on UMich ARC
#
# AUDIENCE
#   Non‑programmers and new HPC users. Read top‑to‑bottom. You run TWO commands.
#
# WHAT THIS SCRIPT DOES (plain English)
#   1) Assumes you already opened a *GPU* shell with SLURM (see HOW TO RUN).
#   2) Loads the CLUSTER Anaconda module and enables `conda activate` safely.
#   3) Reads the GPU’s CUDA version from `nvidia-smi` (e.g., 12.8).
#   4) Chooses the matching PyTorch “CUDA wheel tag” (e.g., cu128 for CUDA 12.8).
#      - You do NOT need to `module load cuda`; the PyTorch wheel includes CUDA.
#   5) For each environment (cellpose4 and cellpose3), it:
#        • activates the env
#        • REPLACES Torch with a CUDA‑enabled build that matches the GPU
#        • prints a short JSON report: torch version, cuda_available (True/False),
#          CUDA toolkit version used by torch, and the detected GPU name
#        • prints Cellpose version from both CLI and Python
#
# WHAT THIS SCRIPT *DOES NOT* DO
#   • It does not run any segmentation. It only prepares the envs and verifies GPU.
#   • It does not modify any images or files outside your conda envs.
#
# HOW TO RUN (exactly two commands):
#   1) srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:20:00 --pty bash
#   2) bash test_cellpose_gpu_envs.sh
#
# HOW TO READ THE OUTPUT
#   • If you see:   "cuda_available: true"  and a device name (e.g., Tesla V100),
#     then this env can use the GPU and you can run:  cellpose --use_gpu
#   • If installation with the first tag fails, the script automatically tries
#     a small set of fallback tags (cu128, cu126, cu121, cu118) until one works.
#   • Seeing cu126 on a node that reports CUDA 12.8 is OK: drivers are forwards‑compatible.
#
# AFTER THIS SCRIPT (optional smoke test)
#   conda activate cellpose4   # or: conda activate cellpose3
#   cellpose --dir /path/to/images --img_filter your_image.tif \
#           --pretrained_model cyto3 --diameter 30 --save_tif --no_npy --use_gpu
#
# REQUIREMENTS
#   • The two envs already exist from your CPU setup script:
#       - cellpose4  (cellpose==4.0.7)
#       - cellpose3  (cellpose==3.1.1.2)
#
# ============================================================================

set -euo pipefail

ENV_LIST=("cellpose4" "cellpose3")

echo "=== Cellpose GPU checker (UMich ARC) ==="
echo "time: $(date)"
echo "host: $(hostname)"
echo "cwd : $(pwd)"
echo "--------------------------------------------"

# 0) VERIFY: We must already be on a GPU node (interactive shell via srun)
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "❌ You are not on a GPU node."
  echo "   First run this to start a GPU shell:"
  echo "   srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:20:00 --pty bash"
  exit 1
fi

echo "[GPU] nvidia-smi (top lines):"
nvidia-smi | sed -n '1,6p'

# 1) DETECT CUDA VERSION from nvidia-smi reliably
CUDA_LINE="$(nvidia-smi | head -n1)"
# Example: "NVIDIA-SMI 570.124.06 Driver Version: 570.124.06 CUDA Version: 12.8"
CUDA_VER="$(echo "$CUDA_LINE" | sed -n 's/.*CUDA Version:[[:space:]]*\([0-9]\+\.[0-9]\+\).*/\1/p')"

if [[ -z "${CUDA_VER}" ]]; then
  echo "⚠️  Could not parse CUDA version from nvidia-smi; choosing a safe default (12.6)."
  CUDA_VER="12.6"
fi
echo "[GPU] Detected CUDA Version: ${CUDA_VER}"

# 2) MAP CUDA VERSION → PYTORCH WHEEL TAG
TAG=""
case "${CUDA_VER}" in
  12.9*) TAG="cu128" ;;  # use nearest
  12.8*) TAG="cu128" ;;
  12.7*) TAG="cu126" ;;  # PyTorch build provided for 12.6, forward‑compatible
  12.6*) TAG="cu126" ;;
  12.5*) TAG="cu121" ;;  # fallback to closest available
  12.4*) TAG="cu121" ;;
  12.3*) TAG="cu121" ;;
  12.2*) TAG="cu121" ;;
  12.1*) TAG="cu121" ;;
  11.8*) TAG="cu118" ;;
  11.*)  TAG="cu118" ;;
  *)     TAG="cu126" ;;
esac
echo "[GPU] Using PyTorch wheel tag: ${TAG}"
echo "    (PyTorch wheels include CUDA; no 'module load cuda' needed.)"
echo "--------------------------------------------"

# 3) LOAD CLUSTER ANACONDA AND ENABLE `conda activate`
module purge
module load python3.10-anaconda/2023.03
CONDA_BASE=${ANACONDA_ROOT:-$(conda info --base)}
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  eval "$("$CONDA_BASE/bin/conda" shell.bash hook)"
fi
echo "[Conda] base: $(conda info | awk -F': ' '/base environment/ {print $2}')"
echo "--------------------------------------------"

# 4) FUNCTION: Fix & test ONE environment
check_env () {
  local ENV_NAME="$1"
  echo
  echo "=== ENV: ${ENV_NAME} ==="
  if ! conda activate "${ENV_NAME}" 2>/dev/null; then
    echo "❌ Env '${ENV_NAME}' not found. Create it first with your CPU setup script."
    return 1
  fi
  echo "ACTIVE ENV     : ${CONDA_DEFAULT_ENV}"
  echo "python         : $(python -V 2>&1)"
  echo "which python   : $(command -v python)"
  echo "which pip      : $(command -v pip)"
  echo "--------------------------------------------"

  echo "[Torch] Installing a CUDA‑enabled build that matches this node (tag ${TAG}) ..."
  # Remove any existing torch bits to avoid ABI mismatches
  pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true

  # Try selected tag; if it fails, fall back across common tags automatically
  if ! pip install torch torchvision --index-url "https://download.pytorch.org/whl/${TAG}"; then
    echo "… install with ${TAG} failed, trying fallbacks"
    for FBTAG in cu128 cu126 cu121 cu118; do
      if [[ "$FBTAG" == "$TAG" ]]; then continue; fi
      echo "… trying ${FBTAG}"
      if pip install torch torchvision --index-url "https://download.pytorch.org/whl/${FBTAG}"; then
        TAG="$FBTAG"
        echo "✓ installed with ${FBTAG}"
        break
      fi
    done
  else
    echo "✓ installed with ${TAG}"
  fi

  echo "--------------------------------------------"
  echo "[Python probe] GPU availability (if True, this env can use the GPU):"
  python - <<'PY'
import json, torch
out = {
  "torch": torch.__version__,
  "torch_cuda_version": getattr(torch.version, "cuda", None),
  "cuda_available": torch.cuda.is_available(),
  "device_count": torch.cuda.device_count()
}
if torch.cuda.is_available():
  out["device_0"] = torch.cuda.get_device_name(0)
print(json.dumps(out, indent=2))
PY

  echo "--------------------------------------------"
  echo "[Cellpose] versions (CLI and Python):"
  cellpose --version || echo "cellpose CLI not found"
  python - <<'PY'
from importlib.metadata import version, PackageNotFoundError
try:
  print("cellpose (python):", version("cellpose"))
except PackageNotFoundError:
  print("cellpose not installed in this env")
PY

  echo "=== DONE: ${ENV_NAME} ==="
  conda deactivate >/dev/null 2>&1 || true
}

# 5) RUN for each env
for e in "${ENV_LIST[@]}"; do
  check_env "$e"
done

echo
echo "=== SUMMARY ==="
echo "• If you saw 'cuda_available: true', this environment is GPU‑ready."
echo "• Next you can run Cellpose with GPU in that env, e.g.:"
echo "    conda activate cellpose4"
echo "    cellpose --dir /path/to/images --pretrained_model cyto3 --diameter 30 --save_tif --use_gpu"
echo "============================================"
