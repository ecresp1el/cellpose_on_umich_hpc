#!/bin/bash
# ============================================================================
# eval_cp3_wholeorganoid_cpu.slurm — Stage C on CPU (fast queue, short walltime)
# Evaluates existing RUN_DIR (resume mode). Writes outputs to $RUN_DIR/ev al/.
# ============================================================================

#SBATCH --job-name=cp3_eval_cpu
#SBATCH --account=parent0
#SBATCH --partition=standard          # CPU queue → usually schedules faster
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:20:00               # short walltime for quick turn-around
#SBATCH --output=/nfs/turbo/umms-parent/cellpose_wholeorganoid_model/logs/slurm/%x_%j.out
#SBATCH --error=/nfs/turbo/umms-parent/cellpose_wholeorganoid_model/logs/slurm/%x_%j.err

set -euo pipefail

# CPU threading hints
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# -------- overrides at submission (via --export=) --------
: "${REPO_ROOT:=${HOME}/Desktop/githubprojects/cellpose_on_umich_hpc}"
: "${CONFIG_PATH:=${REPO_ROOT}/configs/cp3_v001.yaml}"
: "${RUN_DIR:?Set RUN_DIR with --export}"
: "${SPLIT:=all}"
: "${ENV_NAME:=cellpose3}"
# ---------------------------------------------------------

module load python3.10-anaconda/2023.03
CONDA_BASE=/sw/pkgs/arc/python3.10-anaconda/2023.03
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  eval "$("$CONDA_BASE/bin/conda" shell.bash hook)"
fi
conda activate "${ENV_NAME}"

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

echo "=== CPU EVAL ==="
echo "RUN_DIR   : $RUN_DIR"
echo "SPLIT     : $SPLIT"
echo "Start time: $(date)"

python -u "${REPO_ROOT}/scripts/run_experiment.py" \
  --config "${CONFIG_PATH}" \
  --mode eval \
  --split "${SPLIT}" \
  --run_dir "${RUN_DIR}"

echo "Done at $(date)"