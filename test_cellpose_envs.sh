#!/usr/bin/env bash
# =============================================================================
# test_cellpose_envs.sh  (HEADLESS, NO LOG FILES)
#
# GOAL (plain English):
#   1) Use ONLY the cluster’s Anaconda module (not your personal Miniconda).
#   2) Create TWO separate, clean Conda environments:
#        - cellpose4  -> installs cellpose==4.0.7
#        - cellpose3  -> installs cellpose==3.1.1.2
#   3) For each env, print exactly what’s active and installed:
#        - which Conda base is used (cluster path)
#        - which environment is active
#        - which Python executable is used
#        - Python version
#        - Torch version + “is CUDA available?”  (expected FALSE on login nodes)
#        - Cellpose version (from CLI and from Python import)
#
# NOTES:
#   • Environments are created under your HOME (e.g., ~/.conda/envs/...) because
#     the cluster’s Anaconda base under /sw is read-only. This is normal.
#   • We install CPU-only Torch first to keep the test small and stable.
#     Later, on a GPU node, you can swap to a CUDA build of Torch.
#   • Everything runs “headless” (no GUI packages).
# =============================================================================

set -euo pipefail

echo "=== START: Cellpose env creation & verification (headless) ==="
echo "time: $(date)"
echo "host: $(hostname)"
echo "cwd : $(pwd)"
echo "--------------------------------------------------------------"

# 0) Ensure we are using the CLUSTER Anaconda module
echo "[1] Reset modules and load cluster Anaconda (python3.10-anaconda/2023.03)"
module purge
module load python3.10-anaconda/2023.03

echo "ANACONDA_ROOT = ${ANACONDA_ROOT:-'(unset)'}"
echo "which conda   = $(command -v conda || true)"
echo "type -a conda:"
type -a conda || true
echo "--------------------------------------------------------------"

# 1) Initialize Conda activation from the module’s install.
#    This makes `conda activate <env>` work reliably in a non-interactive shell.
echo "[2] Initialize Conda activation from the module"
CONDA_BASE="${ANACONDA_ROOT:-}"
if [[ -z "${CONDA_BASE}" ]]; then
  # Fallback: ask conda where its base is.
  CONDA_BASE="$(conda info --base)"
fi
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

# Small helper to (re)create and verify an env.
# Arguments:
#   $1 = env name (e.g., cellpose4)
#   $2 = cellpose version (e.g., 4.0.7)
create_and_check () {
  local ENV_NAME="$1"
  local CP_VER="$2"

  echo
  echo "=== [CREATE/VERIFY] ${ENV_NAME} (cellpose==${CP_VER}, CPU-first) ==="
  # If the env already exists, reuse it; otherwise create it.
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
  echo "[Paths and versions]"
  echo "which python   = $(command -v python || true)"
  echo "python -V      = $(python -V 2>&1 || true)"
  echo "which pip      = $(command -v pip || true)"

  echo
  echo "[Install] Upgrade pip (inside ${ENV_NAME})"
  python -m pip install --upgrade pip

  echo "[Install] CPU-only Torch (small, stable)"
  # Explicitly use CPU wheels to avoid pulling CUDA builds on login nodes.
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

  echo "[Install] cellpose==${CP_VER} (headless, no GUI extras)"
  pip install "cellpose==${CP_VER}"

  echo
  echo "[Cellpose CLI version] (should print the version and platform info)"
  cellpose --version || echo "cellpose CLI not found"

  echo
  echo "[Python probe] (Torch/CUDA and Cellpose via Python import)"
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

# Cellpose version — try multiple ways so it never shows "unknown"
cp_ver = None
try:
    import cellpose
    cp_ver = getattr(cellpose, "__version__", None)
except Exception:
    pass

if cp_ver is None:
    try:
        # Python 3.8+: robust way to query installed package version
        from importlib.metadata import version
        cp_ver = version("cellpose")
    except Exception:
        cp_ver = None

out["cellpose"] = cp_ver if cp_ver is not None else "not_detected"
print(json.dumps(out, indent=2))
PY

  echo "=== [DONE] ${ENV_NAME} ==="
  # Deactivate so the next env starts clean
  conda deactivate || true
}

# 2) Create + verify both envs (CPU-first)
create_and_check "cellpose4" "4.0.7"
create_and_check "cellpose3" "3.1.1.2"

echo
echo "=== COMPLETE ==="
echo "What to expect on login/CPU nodes:"
echo "  • 'cuda_available' will be FALSE — that's normal here."
echo "  • Envs live under ~/.conda/envs/<envname> (your writable home)."
echo
echo "GPU testing later (interactive GPU shell, not now):"
echo "  srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:20:00 --pty bash"
echo "  module purge && module load python3.10-anaconda/2023.03"
echo "  # re-init conda activate, then:"
echo "  conda activate cellpose4   # or cellpose3"
echo "  pip uninstall -y torch torchvision"
echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX   # choose cu tag that matches 'nvidia-smi' Driver"
echo "  python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
echo "=============================================================="