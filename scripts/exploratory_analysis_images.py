#!/usr/bin/env python3
"""
Exploratory reader (print-only) driven by the project YAML.
- Loads ONE training image + its label using YAML paths (contract §3 config keys)
- Applies quantile_normalization (TNIA-style)
- Ensures mask is integer-labeled
- Prints detailed stats when --debug is set
- NO writing, NO plotting
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from tifffile import imread
from skimage.measure import label as sk_label
import yaml

from cp_core.helper_functions.dl_helper import quantile_normalization

def ensure_labeled_mask(Y: np.ndarray, debug: bool=False) -> np.ndarray:
    if Y.dtype == bool or (Y.ndim == 2 and np.unique(Y).size <= 3):
        Ylab = sk_label(Y > 0).astype(np.uint16, copy=False)
        if debug:
            print(f"[INFO] label(): unique_labels={int(Ylab.max())}")
        return Ylab
    if Y.dtype != np.uint16:
        Y = Y.astype(np.uint16, copy=False)
    return Y

def find_image_and_label(images_dir: Path, labels_dir: Path, stem: str, debug: bool=False):
    # image
    img_path = None
    for ext in (".tif", ".tiff"):
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            img_path = p; break
    if img_path is None:
        raise FileNotFoundError(f"Image not found for stem '{stem}' in {images_dir}")

    # label candidates (cover common conventions)
    candidates = [
        labels_dir / f"{stem}_masks.tif",
        labels_dir / f"{stem}.png",
        labels_dir / f"{stem}.npy",
        labels_dir / f"{stem}_seg.npy",
    ]
    lbl_path = next((p for p in candidates if p.exists()), None)
    if lbl_path is None:
        raise FileNotFoundError(f"No label found for stem '{stem}' in {labels_dir}")

    if debug:
        print(f"[PATH] X: {img_path}")
        print(f"[PATH] Y: {lbl_path}")
    return img_path, lbl_path

def pstats(name: str, arr: np.ndarray):
    arr = np.asarray(arr)
    nnz = int(np.count_nonzero(arr))
    a_min, a_max, a_mean = float(arr.min()), float(arr.max()), float(arr.mean())
    print(f"[STATS] {name:>3}: shape={arr.shape} dtype={arr.dtype} "
          f"min={a_min:.4g} max={a_max:.4g} mean={a_mean:.4g} nnz={nnz}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config (contract §3)")
    ap.add_argument("--stem", required=True, help="File stem (no extension)")
    ap.add_argument("--channels", action="store_true",
                    help="Per-channel normalization if X has channels (H,W,C)")
    ap.add_argument("--debug", action="store_true", help="Verbose prints")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if args.debug:
        print(f"[INFO] Loading YAML: {cfg_path}")
    y = yaml.safe_load(cfg_path.read_text())

    # Prefer explicit YAML paths; otherwise fall back to contract tree
    # paths.* keys defined in contract §3 (data_images_train, data_labels_train)  [oai_citation:0‡CONTRACT_Methods_Cellpose3_WholeOrganoid_Pipeline.md](file-service://file-BQjntqRVfAzTEQqVdkGoYs)
    p = y.get("paths", {})
    images_dir = Path(p.get("data_images_train", "") or
                      (p.get("turbo_root", "/nfs/turbo/umms-parent/cellpose_wholeorganoid_model") + "/dataset/train/images"))
    labels_dir = Path(p.get("data_labels_train", "") or
                      (p.get("turbo_root", "/nfs/turbo/umms-parent/cellpose_wholeorganoid_model") + "/dataset/train/labels"))

    if args.debug:
        print("[INFO] Train images dir:", images_dir)
        print("[INFO] Train labels dir:", labels_dir)
        print("[INFO] Stem           :", args.stem)
        print("[INFO] Channels norm  :", args.channels)

    img_p, lbl_p = find_image_and_label(images_dir, labels_dir, args.stem, debug=args.debug)

    # read
    X = imread(img_p)
    Y = np.load(lbl_p) if lbl_p.suffix == ".npy" else imread(lbl_p)

    if args.debug:
        pstats("X", X)
        pstats("Yraw", Y)

    # normalize & ensure labeled
    Xn = quantile_normalization(X, channels=args.channels).astype(np.float32, copy=False)
    Yl = ensure_labeled_mask(Y, debug=args.debug)

    # final stats
    pstats("Xn", Xn)
    pstats("Yl", Yl)

if __name__ == "__main__":
    main()