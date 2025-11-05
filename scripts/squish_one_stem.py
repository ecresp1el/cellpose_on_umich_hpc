#!/usr/bin/env python3
"""
Squish one training image to single-channel TIFF (flat output; no job subdirs):

  <turbo_root>/dataset/train/<out_subdir>/<stem>.tif

- Reads YAML for train images/labels dirs and labels.mask_filter (contract §3)
- Requires that a matching label exists (paired selection)
- Standardizes to HWC (1..5 chans), drops trivial alpha
- Collapses channels by --mode {max, mean, sum}; no resizing (H×W preserved)
- Saves uint16 (0..65535) grayscale; loud prints; no plotting
"""
from __future__ import annotations
from pathlib import Path
import argparse, yaml, numpy as np
from tifffile import imread, imwrite

from cp_core.helper_functions.dl_helper import (
    ensure_hwc_1to5,
    has_label,
    squish_hwC,   # returns float32 in [0,1], shape (H,W)
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML path (source of truth).")
    ap.add_argument("--stem",   required=True, help="Filename stem (no extension).")
    ap.add_argument("--mode",   default="max", choices=["max","mean","sum"],
                    help="How to collapse channels → single plane.")
    ap.add_argument("--out_subdir", default=None,
                    help="Under dataset/train/. Default: images_squish_<mode>")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    # Load YAML
    y = yaml.safe_load(Path(args.config).read_text())
    p = y.get("paths", {}) or {}
    turbo_root = Path(p.get("turbo_root", "/nfs/turbo/umms-parent/cellpose_wholeorganoid_model"))
    img_dir    = Path(p.get("data_images_train", "")) if p.get("data_images_train") else turbo_root/"dataset/train/images"
    lbl_dir    = Path(p.get("data_labels_train", "")) if p.get("data_labels_train") else turbo_root/"dataset/train/labels"
    mask_suf   = (y.get("labels", {}) or {}).get("mask_filter")

    # Resolve image path
    img_p = next((q for q in (img_dir/f"{args.stem}.tif", img_dir/f"{args.stem}.tiff") if q.exists()), None)
    if img_p is None:
        print(f"[squish-one][SKIP] image not found for stem={args.stem} in {img_dir}")
        return

    # Require paired label
    if not has_label(lbl_dir, args.stem, mask_suf):
        print(f"[squish-one][SKIP] no matching label for stem={args.stem} in {lbl_dir}")
        return

    # Flat output dir
    out_subdir = args.out_subdir or f"images_squish_{args.mode}"
    out_dir = turbo_root/"dataset"/"train"/out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_p = out_dir/f"{args.stem}.tif"

    # Load → standardize → collapse
    X = imread(img_p)
    X = ensure_hwc_1to5(X, debug=args.debug)   # HWC; keep 1..5; drop trivial alpha
    G = squish_hwC(X, mode=args.mode)          # float32 [0,1], shape (H,W)

    # Save as uint16, same H×W (no resizing)
    imwrite(out_p, np.clip(G*65535.0, 0, 65535).astype(np.uint16))
    print(f"[squish-one][OK] stem={args.stem}  in={tuple(X.shape)}  out={G.shape}  → {out_p}")

if __name__ == "__main__":
    main()