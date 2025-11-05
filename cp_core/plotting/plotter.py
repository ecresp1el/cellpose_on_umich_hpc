# cp_core/plotting/plotter.py
# Minimal plotting helpers (TNIA-inspired) adapted for our repo.
# - imshow_multi2d(): simple grid composer (no resampling; loud prints)
# - save_channel_panel(): [C0|C1|...|Max] for 1..5 channels (HWC)
# - save_img_mask_panel(): [Max|Mask] 1×2 panel
#
# Figures are for EXPLORATORY QA only.

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless saving on Great Lakes
import matplotlib.pyplot as plt

# ---------------------------- core plot util ----------------------------

def imshow_multi2d(images, titles, nrows: int, ncols: int,
                   save_path: Path | None = None, dpi: int = 150):
    """Show a 2D grid of images (grayscale or RGB(A)). No resizing. Loud prints."""
    assert len(images) == len(titles), "images/titles length mismatch"
    print(f"[PLOT] imshow_multi2d: n={len(images)} tiles, layout={nrows}x{ncols}")
    fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), squeeze=False)
    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            ax = axs[r, c]
            if idx < len(images):
                img = np.asarray(images[idx])
                t   = titles[idx]
                if img.ndim == 2:
                    ax.imshow(img, interpolation="nearest")
                elif img.ndim == 3 and img.shape[-1] in (3,4):
                    ax.imshow(img, interpolation="nearest")
                else:
                    ax.imshow(np.squeeze(img), interpolation="nearest")
                    print(f"[WARN] unusual image shape at tile {idx}: {img.shape}")
                ax.set_title(t, fontsize=10)
                print(f"[PLOT] tile {idx}: title='{t}', shape={img.shape}, dtype={img.dtype}")
            ax.axis("off")
            idx += 1
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[PLOT] writing → {save_path} (dpi={dpi})")
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[PLOT] saved   → {save_path}")
    plt.close(fig)
    return save_path

# ---------------------------- helpers ----------------------------

def _per_channel_quantile_norm_hwC(X: np.ndarray, q_low=0.01, q_high=0.99) -> np.ndarray:
    """Ensure HWC; quantile normalize each channel to [0,1] for display."""
    X = np.asarray(X)
    if X.ndim == 2:
        lo = np.quantile(X, q_low); hi = np.quantile(X, q_high)
        denom = max(hi - lo, 1e-6)
        return np.clip((X - lo) / denom, 0, 1).astype(np.float32)
    if X.ndim == 3:
        # CHW → HWC if needed
        if X.shape[0] in (1,2,3,4,5) and X.shape[1] == X.shape[2]:
            print(f"[INFO] plot: converting CHW→HWC for display: {X.shape}")
            X = np.moveaxis(X, 0, -1)
        C = X.shape[-1]
        if C > 5:
            print(f"[WARN] plot: {C} channels > 5; using first 5 for display.")
            X = X[..., :5]; C = 5
        Yn = np.empty_like(X, dtype=np.float32)
        for c in range(C):
            plane = X[..., c]
            lo = np.quantile(plane, q_low); hi = np.quantile(plane, q_high)
            denom = max(hi - lo, 1e-6)
            Yn[..., c] = np.clip((plane - lo) / denom, 0, 1).astype(np.float32)
        return Yn
    print(f"[WARN] plot: unexpected ndim={X.ndim}; returning as-is.")
    return X

def _max_contrast_from_hwC(Xn: np.ndarray) -> np.ndarray:
    """Per-pixel max across channels (expects HWC normalized 0..1)."""
    Xn = np.asarray(Xn)
    if Xn.ndim == 2:  return Xn
    if Xn.ndim == 3:  return Xn.max(axis=-1)
    return np.squeeze(Xn)

# ---------------------------- public API ----------------------------

def save_channel_panel(X: np.ndarray, stem: str, results_root: str | Path, job_id: str,
                       dpi: int = 150, q_low: float = 0.01, q_high: float = 0.99) -> Path:
    """
    Save panel of each channel + Max under:
      <results_root>/qc_panels/job_<job_id>/<stem>/<stem>_channels+max.png
    """
    results_root = Path(results_root)
    out_dir = results_root / "qc_panels" / f"job_{job_id}" / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_channels+max.png"

    Xn = _per_channel_quantile_norm_hwC(X, q_low, q_high)
    tiles, titles = [], []
    if Xn.ndim == 2:
        tiles.append(Xn); titles.append("C0")
    elif Xn.ndim == 3:
        C = min(Xn.shape[-1], 5)
        for c in range(C):
            tiles.append(Xn[..., c]); titles.append(f"C{c}")
        if Xn.shape[-1] > 5:
            print(f"[WARN] save_channel_panel: reported only first 5 channels (C={Xn.shape[-1]}).")
    else:
        print(f"[WARN] save_channel_panel: unexpected ndim={Xn.ndim}; attempting squeeze.")
        Xn = np.squeeze(Xn); tiles.append(Xn); titles.append("C0?")

    tiles.append(_max_contrast_from_hwC(Xn)); titles.append("Max")
    print(f"[PLOT] channel panel: tiles={titles}")
    return imshow_multi2d(tiles, titles, nrows=1, ncols=len(tiles), save_path=out_path, dpi=dpi)

def save_img_mask_panel(X: np.ndarray, Y: np.ndarray, stem: str, results_root: str | Path, job_id: str,
                        dpi: int = 150, q_low: float = 0.01, q_high: float = 0.99) -> Path:
    """
    Save 1×2 panel [MaxContrast | Mask] under:
      <results_root>/qc_panels/job_<job_id>/<stem>/<stem>_img+mask_max.png
    """
    results_root = Path(results_root)
    out_dir = results_root / "qc_panels" / f"job_{job_id}" / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_img+mask_max.png"

    Xn = _per_channel_quantile_norm_hwC(X, q_low, q_high)
    Xmax = _max_contrast_from_hwC(Xn)
    tiles = [Xmax, Y]; titles = ["MaxContrast", "Mask"]
    print(f"[PLOT] img+mask panel: tiles={titles}")
    return imshow_multi2d(tiles, titles, nrows=1, ncols=2, save_path=out_path, dpi=dpi)