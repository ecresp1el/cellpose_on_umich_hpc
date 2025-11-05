#!/usr/bin/env python3
"""
Exploratory plotting (TNIA-style) driven by YAML (with optional path overrides):
- Loads 1 image + label using YAML paths and labels.mask_filter (contract §3), unless overridden via CLI.
- Standardizes channel layout to HWC (1..5 channels supported; no resizing).
- Quantile-normalizes per channel for display (0.01..0.99) inside plotter helpers.
- Saves two panels under results/qc_panels/job_<JOBID>/<stem>/:
    1) <stem>_channels+max.png   [C0|C1|...|Max]
    2) <stem>_img+mask_max.png   [Max|Mask]
"""

from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import yaml
from tifffile import imread

from cp_core.plotting.plotter import save_channel_panel, save_img_mask_panel
from cp_core.helper_functions.dl_helper import (
    ensure_hwc_1to5,
    has_label,
    read_label,
)

def find_image_and_label(
    images_dir: Path,
    labels_dir: Path,
    stem: str,
    mask_suffix: str | None,
    debug: bool = False,
) -> tuple[Path, Path]:
    """Find concrete image + label paths (label presence via shared has_label()) and print the same debug trail."""
    # image
    img_path = None
    for ext in (".tif", ".tiff"):
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            img_path = p
            break
    if img_path is None:
        raise FileNotFoundError(f"Image not found for stem '{stem}' in {images_dir}")

    # label (dl_helper.has_label + explicit selection in the same order)
    candidates = []
    if mask_suffix:
        candidates.append(labels_dir / f"{stem}{mask_suffix}")
    candidates += [
        labels_dir / f"{stem}_masks.tif",
        labels_dir / f"{stem}.png",
        labels_dir / f"{stem}.npy",
        labels_dir / f"{stem}_seg.npy",
    ]
    lbl_path = next((p for p in candidates if p.exists()), None) if has_label(labels_dir, stem, mask_suffix) else None

    if debug:
        print("[DEBUG] LABEL RESOLUTION (plot):")
        print(f"  - YAML mask_suffix: {mask_suffix!r}")
        print("  - Candidate order:")
        for c in candidates:
            print(f"      {c}")
        print(f"  -> SELECTED: {lbl_path}" if lbl_path else "  -> No candidate found")

    if lbl_path is None:
        raise FileNotFoundError(f"No label found for stem '{stem}' in {labels_dir}")
    return img_path, lbl_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument("--stem", required=True, help="Image stem (no extension).")
    ap.add_argument("--job_id", required=True, help="SLURM job id (or manual id).")
    ap.add_argument("--dpi", type=int, default=150, help="Figure DPI.")
    ap.add_argument("--debug", action="store_true", help="Verbose prints.")
    # Optional path overrides (no temp YAML; just for this run)
    ap.add_argument("--images_dir_override", default=None, help="Override paths.data_images_train")
    ap.add_argument("--labels_dir_override", default=None, help="Override paths.data_labels_train")
    ap.add_argument("--results_root_override", default=None, help="Override paths.results_root")
    ap.add_argument("--mask_suffix_override", default=None, help="Override labels.mask_filter")
    args = ap.parse_args()

    print("[START] exploratory_plot_images.py")
    print(
        f"[ARGS] config={args.config}  stem={args.stem}  job_id={args.job_id}  "
        f"dpi={args.dpi}  debug={args.debug}"
    )

    cfg_path = Path(args.config)
    y = yaml.safe_load(cfg_path.read_text())

    # paths from YAML (contract §3), with optional CLI overrides
    p = y.get("paths", {}) or {}
    turbo_root = p.get("turbo_root", "/nfs/turbo/umms-parent/cellpose_wholeorganoid_model")
    images_dir = Path(
        args.images_dir_override
        or p.get("data_images_train")
        or (Path(turbo_root) / "dataset" / "train" / "images")
    )
    labels_dir = Path(
        args.labels_dir_override
        or p.get("data_labels_train")
        or (Path(turbo_root) / "dataset" / "train" / "labels")
    )
    results_root = Path(
        args.results_root_override
        or p.get("results_root")
        or (Path(turbo_root) / "results")
    )
    mask_suffix = (
        args.mask_suffix_override
        if args.mask_suffix_override is not None
        else (y.get("labels", {}) or {}).get("mask_filter")
    )

    if args.debug:
        print("[INFO] Train images dir:", images_dir)
        print("[INFO] Train labels dir:", labels_dir)
        print("[INFO] Results root   :", results_root)
        print("[INFO] YAML/override labels.mask_filter:", repr(mask_suffix))

    # find and read
    img_p, lbl_p = find_image_and_label(images_dir, labels_dir, args.stem, mask_suffix, debug=args.debug)
    X = imread(img_p)
    Y = read_label(lbl_p, debug=args.debug)

    print(f"[PATH] X: {img_p}")
    print(f"[PATH] Y: {lbl_p}")
    print(f"[STATS] X: shape={X.shape}, dtype={X.dtype}")
    print(f"[STATS] Y: shape={Y.shape}, dtype={Y.dtype}")

    # standardize HWC (1..5 channels) for plotting (shared helper)
    X = ensure_hwc_1to5(X, debug=args.debug)
    print(f"[INFO] Plotting with HWC layout: {X.shape}")

    # save panels
    p1 = save_channel_panel(X, args.stem, results_root, args.job_id, dpi=args.dpi)
    p2 = save_img_mask_panel(X, Y, args.stem, results_root, args.job_id, dpi=args.dpi)
    print(f"[DONE] Wrote panels:\n  1) {p1}\n  2) {p2}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        print("[FATAL] Uncaught exception in exploratory_plot_images.py:")
        traceback.print_exc()
        raise