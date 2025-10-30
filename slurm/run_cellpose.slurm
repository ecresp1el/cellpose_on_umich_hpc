#!/bin/bash
#SBATCH --job-name=cellpose
#SBATCH --account=parent0
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elcrespo@umich.edu

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

if command -v git >/dev/null 2>&1 && REPO_ROOT="$(git -C "${SUBMIT_DIR}" rev-parse --show-toplevel 2>/dev/null)"; then
    :
elif [[ -d "${SUBMIT_DIR}/cellpose_on_umich_hpc/.git" ]]; then
    REPO_ROOT="$(realpath "${SUBMIT_DIR}/cellpose_on_umich_hpc")"
elif [[ -d "${SUBMIT_DIR}/.git" ]]; then
    REPO_ROOT="$(realpath "${SUBMIT_DIR}")"
else
    echo "[$(date)] ERROR: Unable to determine repository root from submit directory ${SUBMIT_DIR}" >&2
    exit 1
fi

cd "${REPO_ROOT}"

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

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

if [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_JOB_NAME:-}" ]]; then
    SRC="${SUBMIT_DIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
    if [[ -f "${SRC}" ]]; then
        DEST="${LOG_DIR}/$(basename "${SRC}")"
        if [[ "${SRC}" != "${DEST}" ]]; then
            mv "${SRC}" "${DEST}"
            echo "[$(date)] Relocated job log to ${DEST}"
        fi
    fi
fi
