#!/bin/bash
#SBATCH --job-name=env-probe
#SBATCH --account=parent0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/nfs/turbo/umms-parent/run_env_probe_testfolfder/env_probe_%j.out
#SBATCH --error=/nfs/turbo/umms-parent/run_env_probe_testfolfder/env_probe_%j.err

# ---------- USER SETTING (edit this only if you want a different env) ----------
ENV_NAME="cellpose_env"
# ------------------------------------------------------------------------------

set -euo pipefail

# Ensure the Turbo log directory exists
LOG_DIR="/nfs/turbo/umms-parent/run_env_probe_testfolfder"
mkdir -p "${LOG_DIR}"

echo "=== BASIC NODE INFO ==="
date
echo "Host: $(hostname)"
echo "PWD:  $(pwd)"
echo

echo "=== LOAD ANACONDA MODULE ==="
module load anaconda
echo "module list (filtered):"
module list 2>&1 | grep -E 'anaconda|cuda' || true
echo

echo "=== CONDA SETUP ==="
# robust activation for non-interactive shells
if [ -f "$(dirname $(which conda))/../etc/profile.d/conda.sh" ]; then
  source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
elif [ -f "/sw/anaconda/2023.03/etc/profile.d/conda.sh" ]; then
  source "/sw/anaconda/2023.03/etc/profile.d/conda.sh"
fi

echo "conda version: $(conda --version)"
echo "available envs:"
conda info --envs
echo

echo "=== ACTIVATE TARGET ENV ==="
set +e
conda activate "${ENV_NAME}"
ACT_RC=$?
set -e
if [ $ACT_RC -ne 0 ]; then
  echo "ERROR: failed to activate env '${ENV_NAME}'."
  echo "Tip: create it with: conda create -n ${ENV_NAME} python=3.10 -y"
  exit 1
fi
echo "ACTIVE ENV: ${CONDA_DEFAULT_ENV}"
echo

echo "=== INTERPRETER & PATHS ==="
echo "which python: $(which python)"
echo "python -V:    $(python -V)"
echo "which pip:    $(which pip || true)"
echo "which cellpose: $(which cellpose || true)"
echo

echo "=== PYTHON QUICK CHECKS ==="
python - <<'PY'
import sys, importlib, platform, json
out = {}
out["sys_executable"] = sys.executable
out["python_version"] = platform.python_version()

# torch info
try:
    import torch
    out["torch_version"] = torch.__version__
    out["cuda_available"] = torch.cuda.is_available()
    out["torch_cuda_version"] = getattr(torch.version, "cuda", None)
    if torch.cuda.is_available():
        out["cuda_device_0"] = torch.cuda.get_device_name(0)
except Exception as e:
    out["torch_error"] = str(e)

# cellpose info
try:
    import cellpose
    out["cellpose_version"] = getattr(cellpose, "__version__", "unknown")
except Exception as e:
    out["cellpose_error"] = str(e)

print(json.dumps(out, indent=2))
PY
echo

echo "=== GPU (if present) ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found (likely CPU node or no GPU module found)."
fi
echo

echo "=== DONE ==="