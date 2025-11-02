#!/usr/bin/env bash
# rsync_wholeorganoid_to_turbo.sh
# STAGE ONLY (no validation). Run on your Mac. Adjust the first 3 vars and SOURCES.

set -euo pipefail

### ── EDIT THESE VALUES ──────────────────────────────────────────
REMOTE_USER="elcrespo"                                  # your UMich uniqname
REMOTE_HOST="gl-login2.arc-ts.umich.edu"               # Great Lakes login node
REMOTE_ROOT="/nfs/turbo/umms-parent/cellpose_wholeorganoid_model"  # unified Turbo workspace

# Local source directories with .tif/.tiff files to stage (WT/KO mixed is fine)
SOURCES=(
  "/Volumes/Manny4TBUM/10_16_2025/lhx6_pdch19_WTvsKO_projectfolder/cellprofilerandcellpose_folder/cellpose_multichannel_zcyx/PCDHvsLHX6_WTvsKO_IHC/max/KO"
  "/Volumes/Manny4TBUM/10_16_2025/lhx6_pdch19_WTvsKO_projectfolder/cellprofilerandcellpose_folder/cellpose_multichannel_zcyx/PCDHvsLHX6_WTvsKO_IHC/max/WT"
  "/Volumes/Manny4TBUM/10_13_2025/nestin_dcx_pcdh19_kovswt/cellprofilerandcellpose_folder/cellpose_multichannel_zcyx/NestinvsDcx_WTvsKO_IHC/max/KO"
  "/Volumes/Manny4TBUM/10_13_2025/nestin_dcx_pcdh19_kovswt/cellprofilerandcellpose_folder/cellpose_multichannel_zcyx/NestinvsDcx_WTvsKO_IHC/max/WT"

)

# Copy any existing masks that are adjacent to images?
COPY_MASKS="yes"  # set to "no" to skip
### ───────────────────────────────────────────────────────────────

# Enable nullglob so unmatched globs don't cause errors
if [ -n "${BASH_VERSION:-}" ]; then shopt -s nullglob; fi
if [ -n "${ZSH_VERSION:-}" ]; then setopt NULL_GLOB || true; fi

DEST_IMAGES="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/dataset/images/"
DEST_LABELS="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/dataset/labels/"

echo "[REMOTE] Ensuring destination directories exist at ${REMOTE_ROOT} ..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p '${REMOTE_ROOT}/dataset/images' '${REMOTE_ROOT}/dataset/labels' '${REMOTE_ROOT}/results/qc_panels' '${REMOTE_ROOT}/logs'"

echo "[STAGE] Copying .tif/.tiff → ${DEST_IMAGES}"
for SRC in "${SOURCES[@]}"; do
  echo "  • From: $SRC"
  rsync -avh --progress "$SRC/"*.tif  "${DEST_IMAGES}" || true
  rsync -avh --progress "$SRC/"*.tiff "${DEST_IMAGES}" || true
  if [[ "${COPY_MASKS}" == "yes" ]]; then
    rsync -avh --progress "$SRC/"*_masks.*    "${DEST_IMAGES}" 2>/dev/null || true
    rsync -avh --progress "$SRC/"*_cp_masks.* "${DEST_IMAGES}" 2>/dev/null || true
  fi
done

echo "[CHECK] First 20 files on remote:"
ssh "${REMOTE_USER}@${REMOTE_HOST}" "ls -lh '${REMOTE_ROOT}/dataset/images' | head -n 20"

echo "[DONE] Staging complete."
echo "Your pooled images now live at: ${REMOTE_ROOT}/dataset/images/"
