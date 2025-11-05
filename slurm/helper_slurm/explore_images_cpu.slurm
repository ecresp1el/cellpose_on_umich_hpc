#!/bin/bash
# explore_images_cpu.slurm — run scripts/exploratory_analysis_images.py
#SBATCH --job-name=cp3_explore
#SBATCH --account=parent0
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/nfs/turbo/umms-parent/cellpose_wholeorganoid_model/logs/slurm/%x_%j.out
#SBATCH --error=/nfs/turbo/umms-parent/cellpose_wholeorganoid_model/logs/slurm/%x_%j.err
set -euo pipefail

: "${REPO_ROOT:=${HOME}/Desktop/githubprojects/cellpose_on_umich_hpc}"
: "${TURBO_ROOT:=/nfs/turbo/umms-parent/cellpose_wholeorganoid_model}"
: "${STEM:?Set STEM with --export STEM=<file_stem>}"
: "${ENV_NAME:=cellpose3}"
: "${CHANNELS:=1}"   # 1→per-channel norm if X is RGB, 0→global

mkdir -p /nfs/turbo/umms-parent/cellpose_wholeorganoid_model/logs/slurm

module load python3.10-anaconda/2023.03
CONDA_BASE=/sw/pkgs/arc/python3.10-anaconda/2023.03
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  eval "$("$CONDA_BASE/bin/conda" shell.bash hook)"
fi
conda activate "${ENV_NAME}"

echo "=== EXPLORE (print-only) ==="
echo "Host : $(hostname)"
echo "Stem : ${STEM}"
echo "Start: $(date)"

python -u "${REPO_ROOT}/scripts/exploratory_analysis_images.py" \
  --turbo_root "${TURBO_ROOT}" \
  --stem "${STEM}" \
  $([ "${CHANNELS}" -eq 1 ] && echo "--channels") \
  --debug

rc=$?
echo "Exit code: $rc"
echo "End  : $(date)"