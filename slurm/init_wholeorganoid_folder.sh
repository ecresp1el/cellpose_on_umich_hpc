#!/usr/bin/env bash
set -euo pipefail
DEST="/nfs/turbo/umms-parent/cellpose_wholeorganoid_model"

echo "[INIT] Creating unified whole-organoid workspace at $DEST ..."
mkdir -p "$DEST/dataset/images"          "$DEST/dataset/labels"          "$DEST/dataset/train/images"          "$DEST/dataset/train/labels"          "$DEST/dataset/valid/images"          "$DEST/dataset/valid/labels"          "$DEST/results"          "$DEST/results/qc_panels"          "$DEST/logs"

echo "[OK] Created:"
echo "  $DEST/dataset/images"
echo "  $DEST/dataset/labels"
echo "  $DEST/dataset/train/{images,labels}"
echo "  $DEST/dataset/valid/{images,labels}"
echo "  $DEST/results/"
echo "  $DEST/results/qc_panels/"
echo "  $DEST/logs/"
