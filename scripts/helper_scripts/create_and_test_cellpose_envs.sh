#!/usr/bin/env bash
# =============================================================================
# test_cellpose_envs.sh  (HEADLESS, PRINT-ONLY)
#
# PURPOSE (plain English):
#   - Use ONLY the cluster’s Anaconda module (python3.10-anaconda/2023.03).
#   - Create TWO separate environments to keep versions isolated:
#       * cellpose4  -> cellpose==4.0.7
#       * cellpose3  -> cellpose==3.1.1.2
#   - Print exactly what’s active and installed so non-programmers can see:
#       * Which Conda base is used (should be /sw/pkgs/arc/...).
#       * Which environment is active.
#       * Which Python executable/version is used.
#       * Torch version + “is CUDA available?” (will be FALSE on login nodes).
#       * Cellpose version (both CLI and Python import).
#
# HOW TO READ THE OUTPUT:
#   - Lines starting with [Install]/[Paths]/[Python probe]/[Cellpose CLI version]
#     tell you what’s happening now.
#   - If you see cuda_available: false — that’s expected on CPU-only login nodes.
#   - Seeing env paths under ~/.conda/envs/<name>/... is normal (your writable home).
#
# NEXT STEPS (GPU):
#   - Later, in a GPU interactive shell, swap Torch to a CUDA wheel and check that
#     cuda_available becomes TRUE. Until then, this script keeps everything small
#     and headless for quick testing.
# =============================================================================

set -euo pipefail

echo "=== START: Cellpose env creation & verification (headless) ==="
echo "time: $(date)"
echo "host: $(hostname)"
echo "cwd : $(pwd)"
echo "--------------------------------------------------------------"

# 1) Use the cluster’s Anaconda (not personal Miniconda)
echo "[1] module purge && module load python3.10-anaconda/2023.03"
module purge
module load python3.10-anaconda/2023.03

echo "ANACONDA_ROOT = ${ANACONDA_ROOT:-'(unset)'}"
echo "which conda   = $(command -v conda || true)"
echo "type -a conda:"
type -a conda || true
echo "--------------------------------------------------------------"

# 2) Initialize `conda activate` from the module (non-interactive safe)
echo "[2] Initialize Conda activation from the module"
CONDA_BASE="${ANACONDA_ROOT:-$(conda info --base)}"
echo "CONDA_BASE    = ${CONDA_BASE}"
if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  echo "(conda.sh not found; using shell hook)"
  eval "$("${CONDA_BASE}/bin/conda" shell.bash hook)"
fi

conda activate base
echo "ACTIVE ENV    = ${CONDA_DEFAULT_ENV:-'(none)'}"
echo "conda version = $(conda --version 2>&1 || true)"
echo "base env path = $(conda info | awk -F': ' '/base environment/ {print $2}')"
echo "--------------------------------------------------------------"

# Helper: (re)create and verify an env. Forces CPU-only Torch so it’s small & stable.
create_and_check () {
  local ENV_NAME="$1"
  local CP_VER="$2"

  echo
  echo "=== [CREATE/VERIFY] ${ENV_NAME} (cellpose==${CP_VER}, CPU-only Torch) ==="

  # Create env only if it doesn't exist yet
  if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Env '${ENV_NAME}' already exists — reusing."
  else
    echo "Creating env: ${ENV_NAME}"
    conda create -n "${ENV_NAME}" python=3.10 -y
  fi

  echo "[activate] ${ENV_NAME}"
  conda activate "${ENV_NAME}"
  echo "ACTIVE ENV    = ${CONDA_DEFAULT_ENV}"

  echo
  echo "[Paths]"
  echo "which python   = $(command -v python || true)"
  echo "python -V      = $(python -V 2>&1 || true)"
  echo "which pip      = $(command -v pip || true)"

  echo
  echo "[Install] Upgrade pip (inside ${ENV_NAME})"
  python -m pip install --upgrade pip

  echo "[Install] Force CPU-only Torch wheels (keeps env small on login nodes)"
  # If a CUDA wheel is already present, remove it first to avoid mix-ups.
  pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
  # Install CPU builds explicitly:
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

  echo "[Install] cellpose==${CP_VER} (headless, no GUI extras)"
  pip install "cellpose==${CP_VER}"

  echo
  echo "[Cellpose CLI version] (human-readable)"
  cellpose --version || echo "cellpose CLI not found"

  echo
  echo "[Python probe] (machine-readable JSON)"
  python - <<'PY'
import sys, platform, json
out = {"python": platform.python_version(), "exe": sys.executable}

# Torch info
try:
    import torch
    out["torch"] = torch.__version__
    out["cuda_available"] = torch.cuda.is_available()
    out["torch_cuda_version"] = getattr(torch.version, "cuda", None)
    if torch.cuda.is_available():
        out["cuda_device_0"] = torch.cuda.get_device_name(0)
except Exception as e:
    out["torch_error"] = str(e)

# Cellpose version (two ways so it's robust)
cp_ver = None
try:
    import cellpose
    cp_ver = getattr(cellpose, "__version__", None)
except Exception:
    pass
if cp_ver is None:
    try:
        from importlib.metadata import version
        cp_ver = version("cellpose")
    except Exception:
        cp_ver = None

out["cellpose"] = cp_ver if cp_ver is not None else "not_detected"
print(json.dumps(out, indent=2))
PY

  echo "=== [DONE] ${ENV_NAME} ==="
  conda deactivate || true
}

# 3) Create + verify both envs (CPU-first)
create_and_check "cellpose4" "4.0.7"
create_and_check "cellpose3" "3.1.1.2"

echo
echo "=== COMPLETE ==="
echo "On login/CPU nodes:"
echo "  • cuda_available will be FALSE (normal)."
echo "  • Envs live in ~/.conda/envs/<envname> (your writable home)."
echo
echo "GPU test later (interactive GPU shell):"
echo "  srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:20:00 --pty bash"
echo "  module purge && module load python3.10-anaconda/2023.03"
echo "  # re-init conda activation with the same logic as above"
echo "  conda activate cellpose4   # or cellpose3"
echo "  pip uninstall -y torch torchvision"
echo "  # Pick a wheel that matches the GPU driver (nvidia-smi) — e.g. cu126 or cu128:"
echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
echo "  python - <<'PY'"
echo "import torch; print('torch:', torch.__version__, '| cuda_available:', torch.cuda.is_available());"
echo "print('device:' , torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
echo "PY"
echo "================================================================"