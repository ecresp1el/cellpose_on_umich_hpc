#!/usr/bin/env python3
"""
Exploratory reader (print-only) driven by the project YAML.

Behavior:
- Load ONE training image + its label using YAML paths (preferred).
- Prefer labels.mask_filter from YAML for the label filename suffix (intended behavior).
- If not found or not provided, fall back to common suffixes.
- Apply quantile_normalization (TNIA-style) to the image.
- Ensure label is integer-labeled (uint16).
- Print detailed stats; NO writing and NO plotting.

Usage (example):
  python scripts/exploratory_analysis_images.py \
    --config /path/to/config.yaml \
    --stem "MyImage_001" \
    --channels \
    --debug
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import yaml
from tifffile import imread
from skimage.measure import label as sk_label

# Your adapted TNIA helper
from cp_core.helper_functions.dl_helper import quantile_normalization


# ---------------------- helpers ----------------------
def ensure_labeled_mask(Y: np.ndarray, debug: bool=False) -> np.ndarray:
    """
    Convert a binary/boolean mask to labeled (uint16) if needed.
    If already labeled, coerce dtype to uint16 for consistency.
    """
    if Y.dtype == bool or (Y.ndim == 2 and np.unique(Y).size <= 3):
        Ylab = sk_label(Y > 0).astype(np.uint16, copy=False)
        if debug:
            print(f"[INFO] label(): unique_labels={int(Ylab.max())}")
        return Ylab
    if Y.dtype != np.uint16:
        Y = Y.astype(np.uint16, copy=False)
    return Y


def find_image_and_label(
    images_dir: Path,
    labels_dir: Path,
    stem: str,
    mask_suffix: str | None,
    debug: bool=False
) -> tuple[Path, Path]:
    """
    Resolve image and label paths given a stem.

    Resolution order for LABEL:
      1) If YAML provided labels.mask_filter (e.g., '_cp_masks.png'), prefer stem + that suffix.
      2) Otherwise, try common fallbacks in this order:
         <stem>_masks.tif, <stem>.png, <stem>.npy, <stem>_seg.npy

    Prints exactly which rule was used when debug=True.
    """
    # Image: try .tif then .tiff
    img_path = None
    for ext in (".tif", ".tiff"):
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            img_path = p
            break
    if img_path is None:
        raise FileNotFoundError(f"Image not found for stem '{stem}' in {images_dir}")

    # Label: YAML-driven suffix first (intended behavior)
    candidates: list[Path] = []
    if mask_suffix:
        candidates.append(labels_dir / f"{stem}{mask_suffix}")

    # Fallbacks (robust to common conventions)
    candidates.extend([
        labels_dir / f"{stem}_masks.tif",
        labels_dir / f"{stem}.png",
        labels_dir / f"{stem}.npy",
        labels_dir / f"{stem}_seg.npy",
    ])

    lbl_path = next((p for p in candidates if p.exists()), None)

    if debug:
        print("[DEBUG] LABEL RESOLUTION:")
        print(f"  - YAML mask_suffix: {mask_suffix!r} (preferred if present)")
        print("  - Candidate order:")
        for c in candidates:
            print(f"      {c}")
        if lbl_path:
            print(f"  -> SELECTED: {lbl_path}")
        else:
            print("  -> No candidate found")

    if lbl_path is None:
        raise FileNotFoundError(
            f"No label found for stem '{stem}' in {labels_dir}. "
            f"(Checked YAML suffix first, then fallbacks.)"
        )

    return img_path, lbl_path


def pstats(name: str, arr: np.ndarray):
    arr = np.asarray(arr)
    nnz = int(np.count_nonzero(arr))
    a_min = float(arr.min()); a_max = float(arr.max()); a_mean = float(arr.mean())
    print(f"[STATS] {name:>3}: shape={arr.shape} dtype={arr.dtype} "
          f"min={a_min:.4g} max={a_max:.4g} mean={a_mean:.4g} nnz={nnz}")


# ---------------------- main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config (source of truth for paths).")
    ap.add_argument("--stem", required=True, help="File stem (no extension).")
    ap.add_argument("--channels", action="store_true",
                    help="Per-channel normalization if X has channels (H,W,C).")
    ap.add_argument("--debug", action="store_true", help="Verbose prints.")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if args.debug:
        print(f"[INFO] Loading YAML: {cfg_path}")

    y = yaml.safe_load(cfg_path.read_text())

    # Prefer explicit YAML paths (contract §3). Fall back to canonical tree if missing.
    p = y.get("paths", {}) or {}
    turbo_root = p.get("turbo_root", "/nfs/turbo/umms-parent/cellpose_wholeorganoid_model")
    images_dir = Path(p.get("data_images_train", "")) if p.get("data_images_train") else Path(turbo_root) / "dataset/train/images"
    labels_dir = Path(p.get("data_labels_train", "")) if p.get("data_labels_train") else Path(turbo_root) / "dataset/train/labels"

    # YAML-defined mask suffix (intended behavior)
    mask_suffix = (y.get("labels", {}) or {}).get("mask_filter")

    if args.debug:
        print("[INFO] Train images dir:", images_dir)
        print("[INFO] Train labels dir:", labels_dir)
        print("[INFO] YAML labels.mask_filter:", repr(mask_suffix), "(preferred)")
        print("[INFO] Stem           :", args.stem)
        print("[INFO] Channels norm  :", args.channels)

    # Resolve paths (YAML suffix preferred; fallbacks if needed)
    img_p, lbl_p = find_image_and_label(images_dir, labels_dir, args.stem, mask_suffix, debug=args.debug)

    # --- read image file ---
    X = imread(img_p)

    # --- normalize channel layout for N-channel images (N = 1..5) ---
    # Goal: ensure channels-last (H,W,C) for any per-channel ops (e.g., quantile normalization).
    if X.ndim == 3:
        HWC_guess = (X.shape[-1] in (1,2,3,4,5)) and (X.shape[0] != X.shape[-1]) and (X.shape[1] != X.shape[-1])
        CHW_guess = (X.shape[0]  in (1,2,3,4,5)) and (X.shape[1] == X.shape[2])

        if CHW_guess and not HWC_guess:
            print(f"[INFO] Detected channels-first layout {X.shape}; converting to channels-last (H,W,C).")
            X = np.moveaxis(X, 0, -1)  # (C,H,W) -> (H,W,C)
        elif HWC_guess:
            print(f"[INFO] Detected channels-last layout {X.shape}.")
        else:
            if X.shape[-1] in (1,2,3,4,5):
                print(f"[WARN] Ambiguous layout {X.shape}; assuming channels-last (H,W,C).")
            elif X.shape[0] in (1,2,3,4,5):
                print(f"[WARN] Ambiguous layout {X.shape}; assuming channels-first and converting to HWC.")
                X = np.moveaxis(X, 0, -1)
            else:
                print(f"[WARN] Unexpected 3D shape {X.shape}; proceeding without axis reordering.")

        if X.ndim == 3 and X.shape[-1] == 4:
            a = X[..., 3]
            near_all_255 = (np.count_nonzero(a > 250) / a.size) > 0.99
            near_all_0   = (np.count_nonzero(a) / a.size) < 0.01
            if near_all_255 or near_all_0:
                print("[INFO] Dropping alpha channel (RGBA → RGB) based on simple heuristic.")
                X = X[..., :3]
            else:
                print("[INFO] 4 channels detected; keeping all 4 (no alpha drop).")

        if X.ndim == 3 and X.shape[-1] > 5:
            print(f"[WARN] {X.shape[-1]} channels > 5; keeping first 5 for normalization.")
            X = X[..., :5]

        print(f"[INFO] Image layout for normalization: {X.shape} (channels-last expected).")
    else:
        # 2D single-channel image; nothing to do.
        pass



    # --- determine how to read the label file (robust and explicit) ---
    suffix = lbl_p.suffix.lower()
    print(f"[INFO] Label file detected: {lbl_p.name}")
    print(f"[INFO] Supported readers available:")
    print("        • NumPy loader (.npy)")
    print("        • ImageIO for PNG/JPEG/BMP")
    print("        • Tifffile for .tif/.tiff")
    print(f"[INFO] Selected reader based on extension: {suffix}")

    try:
        if suffix == ".npy":
            print("[DEBUG] Using NumPy loader...")
            Y = np.load(lbl_p)
        elif suffix in (".png", ".jpg", ".jpeg", ".bmp"):
            print("[DEBUG] Using ImageIO loader for non-TIFF image...")
            import imageio.v3 as iio
            Y = iio.imread(lbl_p)
            if Y.ndim == 3:
                print(f"[DEBUG] PNG mask has {Y.shape[-1]} channels; taking first channel for mask semantics.")
                Y = Y[..., 0]
        else:
            print("[DEBUG] Using Tifffile loader...")
            Y = imread(lbl_p)
    except Exception as e:
        raise RuntimeError(f"Failed to read label file '{lbl_p}' with suffix {suffix}: {e}")

    print(f"[INFO] Label file successfully read → shape={Y.shape}, dtype={Y.dtype}")

    if args.debug:
        print(f"[PATH] X: {img_p}")
        print(f"[PATH] Y: {lbl_p}")
        pstats("X", X)
        pstats("Yraw", Y)

    # Normalize & ensure labeled (print-only)
    Xn = quantile_normalization(X, channels=args.channels).astype(np.float32, copy=False)
    
        # --- per-channel stats on Xn (print-only; helps choose eval/train.channels) ---
    def ch_stats(arr: np.ndarray, name: str = "Xn"):
        arr = np.asarray(arr)
        if arr.ndim == 2:
            # single-channel
            flat = arr.ravel()
            nn = flat.size
            nnz = int(np.count_nonzero(flat))
            p1, med, p99 = np.percentile(flat, [1, 50, 99])
            print(f"[CHSTATS] {name}: single-channel  "
                  f"p1={p1:.4g}  med={med:.4g}  p99={p99:.4g}  nnz_frac={nnz/nn:.4%}")
            return
        if arr.ndim == 3:
            H, W, C = arr.shape
            C_eff = min(C, 5)  # keep it quick & consistent (1..5)
            if C > 5:
                print(f"[WARN] {name}: {C} channels > 5; reporting first 5 only.")
            for c in range(C_eff):
                plane = arr[..., c].ravel()
                nn = plane.size
                nnz = int(np.count_nonzero(plane))
                p1, med, p99 = np.percentile(plane, [1, 50, 99])
                print(f"[CHSTATS] {name}[c{c}]: p1={p1:.4g}  med={med:.4g}  p99={p99:.4g}  nnz_frac={nnz/nn:.4%}")
        else:
            print(f"[WARN] {name}: unexpected ndim={arr.ndim}; skipping per-channel stats.")

    ch_stats(Xn, "Xn")
    
    Yl = ensure_labeled_mask(Y, debug=args.debug)

    # Final stats
    pstats("Xn", Xn)
    pstats("Yl", Yl)


if __name__ == "__main__":
    main()