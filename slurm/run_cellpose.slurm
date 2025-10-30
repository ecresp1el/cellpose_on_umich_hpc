#!/bin/bash
#SBATCH --job-name=cellpose
#SBATCH --account=parent0
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elcrespo@umich.edu

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(realpath "${SCRIPT_DIR}/..")"
cd "${REPO_ROOT}"

mkdir -p logs

module load python/3.10.5          # adjust or remove if using custom stack

# Activate your conda environment
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate cellpose-hpc

echo "[$(date)] Repository root: ${REPO_ROOT}"
echo "[$(date)] Hostname: ${HOSTNAME}"
echo "[$(date)] Python executable: $(which python)"
echo "[$(date)] Python version: $(python --version)"

python <<'PYTHON'
import os
import sys
import traceback

print(f"[verify] Working directory: {os.getcwd()}")
print(f"[verify] sys.path[0]: {sys.path[0]}")

errors = []

try:
    import torch
except Exception as exc:
    errors.append(f"Unable to import torch: {exc}")
    torch = None
else:
    print(f"[verify] torch version: {torch.__version__}")
    print(f"[verify] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[verify] CUDA device count: {torch.cuda.device_count()}")
        for idx in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(idx)
            capability = torch.cuda.get_device_capability(idx)
            print(f"[verify] Device {idx}: {name} (compute capability {capability[0]}.{capability[1]})")
    else:
        errors.append("CUDA is not available according to torch.")

try:
    import cellpose
    from cellpose import models
except Exception as exc:
    errors.append(f"Unable to import cellpose: {exc}")
else:
    version = getattr(cellpose, "__version__", None)
    if version is None:
        version = getattr(cellpose, "version_str", "unknown")
    print(f"[verify] cellpose version: {version}")
    location = os.path.dirname(cellpose.__file__)
    print(f"[verify] cellpose package path: {location}")

    if torch is not None and torch.cuda.is_available():
        try:
            model = models.CellposeModel(gpu=True)
        except Exception as exc:
            errors.append(f"Failed to initialize CellposeModel on GPU: {exc}")
            traceback.print_exc()
        else:
            device = getattr(model, "device", None)
            print(f"[verify] CellposeModel initialized on device: {device}")
            if device is not None and str(device).startswith("cuda"):
                print("[verify] ✅ Cellpose is configured to use the GPU.")
            else:
                errors.append("CellposeModel did not bind to a CUDA device.")

if errors:
    print("[verify] Encountered issues during verification:")
    for item in errors:
        print(f"[verify] - {item}")
    sys.exit(1)

print("[verify] ✅ Cellpose environment and GPU check completed successfully.")
PYTHON

echo "[$(date)] GPU verification complete."
