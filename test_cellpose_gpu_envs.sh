#!/usr/bin/env bash
# ============================================================================
# test_cellpose_gpu_envs_v2.sh — Robust one‑click GPU check for Cellpose envs
#
# WHAT THIS DOES
#   • Assumes you're already on a GPU node (via: srun --partition=gpu ... --pty bash)
#   • Loads the CLUSTER Anaconda module and enables `conda activate`
#   • Detects CUDA version from nvidia-smi and maps it to a PyTorch wheel tag
#   • For each env (cellpose4, cellpose3): installs matching CUDA torch & reports GPU
#
# HOW TO RUN
#   1) srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:20:00 --pty bash
#   2) bash test_cellpose_gpu_envs_v2.sh
#
# READOUT
#   • Look for: "cuda_available: true" and a real GPU name (e.g., Tesla V100)
#   • If true, `cellpose --use_gpu` will use the GPU in that environment
# ============================================================================

set -Eeuo pipefail

ENV_LIST=("cellpose4" "cellpose3")

say() { printf '%s %s\n' "[$(date +%H:%M:%S)]" "$*"; }

say "=== Cellpose GPU checker (robust) ==="
say "Host: $(hostname)"
say "CWD : $(pwd)"
echo "--------------------------------------------"

# 0) Must be on a GPU node
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "❌ Not on a GPU node. First run:"
  echo "   srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:20:00 --pty bash"
  exit 1
fi

say "[GPU] nvidia-smi (top lines):"
nvidia-smi | sed -n '1,6p' || true
echo "--------------------------------------------"

# 1) Detect CUDA version robustly
set +e
CUDA_VER="$(nvidia-smi | awk 'NR==1{for(i=1;i<=NF;i++){if($i=="Version:"&&$(i-1)=="CUDA"){print $(i+1); exit}}}')"
set -e
if [[ -z "${CUDA_VER}" ]]; then
  say "WARN: Could not parse CUDA from nvidia-smi; defaulting to 12.6"
  CUDA_VER="12.6"
fi
say "[GPU] Detected CUDA Version: ${CUDA_VER}"

# Map CUDA version to PyTorch tag
TAG=""
case "${CUDA_VER}" in
  12.9*) TAG="cu128" ;; 12.8*) TAG="cu128" ;;
  12.7*) TAG="cu126" ;; 12.6*) TAG="cu126" ;;
  12.5*) TAG="cu121" ;; 12.4*) TAG="cu121" ;; 12.3*) TAG="cu121" ;;
  12.2*) TAG="cu121" ;; 12.1*) TAG="cu121" ;;
  11.8*) TAG="cu118" ;; 11.*)  TAG="cu118" ;;
  *)     TAG="cu126" ;;
esac
say "[GPU] Using PyTorch wheel tag: ${TAG} (PyTorch wheels include CUDA)"
echo "--------------------------------------------"

# 2) Load cluster Anaconda & init activation
module purge
module load python3.10-anaconda/2023.03
CONDA_BASE=${ANACONDA_ROOT:-$(conda info --base)}
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  eval "$("$CONDA_BASE/bin/conda" shell.bash hook)"
fi
say "[Conda] base: $(conda info | awk -F': ' '/base environment/ {print $2}')"
echo "--------------------------------------------"

# 3) Check one environment
check_env () {
  local ENV_NAME="$1"
  echo
  say "=== ENV: ${ENV_NAME} ==="
  if ! conda activate "${ENV_NAME}" 2>/dev/null; then
    say "ERROR: Env '${ENV_NAME}' not found. Create it first with your CPU script."
    return 1
  fi
  say "ACTIVE ENV     : ${CONDA_DEFAULT_ENV}"
  say "python         : $(python -V 2>&1)"
  say "which python   : $(command -v python)"
  say "which pip      : $(command -v pip)"
  echo "--------------------------------------------"
  say "[Torch] Installing CUDA torch (tag ${TAG}); falling back if needed..."

  # Clean old torch
  pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true

  set +e
  pip install torch torchvision --index-url "https://download.pytorch.org/whl/${TAG}"
  RC=$?
  if [[ $RC -ne 0 ]]; then
    for FBTAG in cu128 cu126 cu121 cu118; do
      [[ "$FBTAG" == "$TAG" ]] && continue
      say "… trying fallback ${FBTAG}"
      pip install torch torchvision --index-url "https://download.pytorch.org/whl/${FBTAG}"
      RC=$?
      if [[ $RC -eq 0 ]]; then TAG="$FBTAG"; break; fi
    done
  fi
  set -e
  if [[ $RC -ne 0 ]]; then
    say "ERROR: Could not install a CUDA torch wheel. Aborting this env."
    conda deactivate >/dev/null 2>&1 || true
    return 1
  fi
  say "✓ torch/vision installed with tag: ${TAG}"
  echo "--------------------------------------------"

  say "[Python probe] GPU availability:"
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
  say "[Cellpose] versions:"
  cellpose --version || echo "cellpose CLI not found"
  python - <<'PY'
from importlib.metadata import version, PackageNotFoundError
try:
  print("cellpose (python):", version("cellpose"))
except PackageNotFoundError:
  print("cellpose not installed in this env")
PY

  say "=== DONE: ${ENV_NAME} ==="
  conda deactivate >/dev/null 2>&1 || true
}

# 4) Run for each env
for e in "${ENV_LIST[@]}"; do
  check_env "$e"
done

echo
say "=== SUMMARY ==="
say "If 'cuda_available: true', that env is GPU‑ready."
say "Run Cellpose with GPU, e.g.:"
echo "  conda activate cellpose4"
echo "  cellpose --dir /path/to/images --pretrained_model cyto3 --diameter 30 --save_tif --use_gpu"
echo "================================================================"
