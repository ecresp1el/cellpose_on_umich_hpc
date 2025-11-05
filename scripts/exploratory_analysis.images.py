#!/usr/bin/env python3
"""
Exploratory reader (print-only).
- Loads ONE training image + its label from contract paths
- Applies quantile_normalization (TNIA-style)
- Ensures mask is integer-labeled
- Prints detailed stats when --debug is set
- NO writing, NO plotting

Contract refs:
- Authoritative workspace & dataset layout under /dataset/train/images and /dataset/train/labels
  (we DO NOT change structure or write under dataset)  [see contract §0 Workspace tree].
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from tifffile import imread
from skimage.measure import label as sk_label

# Import your adapted TNIA helpers
from cp_core.helper_functions.dl_helper import quantile_normalization

TURBO_ROOT_DEFAULT = "/nfs/turbo/umms-parent/cellpose_wholeorganoid_model"

def ensure_labeled_mask(Y: np.ndarray, debug: bool=False) -> np.ndarray:
    """Convert binary mask to labeled if needed; keep dtype uint16."""
    if Y.dtype == bool or (Y.ndim == 2 and np.unique(Y).size <= 3):
        Ylab = sk_label(Y > 0).astype(np.uint16, copy=False)
        if debug:
            print(f"[INFO] label(): unique_labels={int(Ylab.max())}")
        return Ylab
    if Y.dtype != np.uint16:
        Y = Y.astype(np.uint16, copy=False)
    return Y

def find_image_and_label(train_images_dir: Path, train_labels_dir: Path, stem: str, debug: bool=False):
    # image
    img_path = None
    for ext in (".tif", ".tiff"):
        p = train_images_dir / f"{stem}{ext}"
        if p.exists():
            img_path = p; break
    if img_path is None:
        raise FileNotFoundError(f"Image not found for stem '{stem}' in {train_images_dir}")

    # label candidates (cover common cases in our data & contract notes)
    candidates = [
        train_labels_dir / f"{stem}_masks.tif",
        train_labels_dir / f"{stem}.png",
        train_labels_dir / f"{stem}.npy",
        train_labels_dir / f"{stem}_seg.npy",
    ]
    lbl_path = next((p for p in candidates if p.exists()), None)
    if lbl_path is None:
        raise FileNotFoundError(f"No label found for stem '{stem}' in {train_labels_dir}")

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
    ap.add_argument("--turbo_root", default=TURBO_ROOT_DEFAULT, help="Contract Turbo root")
    ap.add_argument("--stem", required=True, help="File stem (no extension)")
    ap.add_argument("--channels", action="store_true",
                    help="Per-channel normalization if X has channels (H,W,C)")
    ap.add_argument("--debug", action="store_true", help="Verbose prints")
    args = ap.parse_args()

    turbo_root = Path(args.turbo_root)
    train_images = turbo_root / "dataset" / "train" / "images"
    train_labels = turbo_root / "dataset" / "train" / "labels"

    if args.debug:
        print("[INFO] Turbo root   :", turbo_root)
        print("[INFO] Train images :", train_images)
        print("[INFO] Train labels :", train_labels)
        print("[INFO] Stem         :", args.stem)
        print("[INFO] Channels norm:", args.channels)

    img_p, lbl_p = find_image_and_label(train_images, train_labels, args.stem, debug=args.debug)

    # read
    X = imread(img_p)
    if lbl_p.suffix == ".npy":
        Y = np.load(lbl_p)
    else:
        Y = imread(lbl_p)

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