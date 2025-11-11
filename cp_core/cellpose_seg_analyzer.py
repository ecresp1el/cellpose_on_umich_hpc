# cp_core/cellpose_seg_analyzer.py
from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Optional, List

import numpy as np

try:
    from cellpose import io as cp_io
    from cellpose.models import CellposeModel
except Exception:
    cp_io = None
    CellposeModel = None

# cp_core/cellpose_seg_analyzer.py
from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Optional, List

import numpy as np

try:
    from cellpose import io as cp_io
    from cellpose.models import CellposeModel
except Exception:
    cp_io = None
    CellposeModel = None


# -----------------------
# small print utilities
# -----------------------
def _fmt_shape(x) -> str:
    return str(getattr(x, "shape", None))

def _fmt_dtype(x) -> str:
    return str(getattr(x, "dtype", None))

def _minmax(x) -> Tuple[Optional[float], Optional[float]]:
    try:
        return float(np.nanmin(x)), float(np.nanmax(x))
    except Exception:
        return None, None

def _print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# -----------------------
# channel-axis heuristic
# -----------------------
def detect_channel_axis(arr: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
    """
    Heuristic: smallest of 3 dims is channels; fall back to last if ambiguous.
    Returns (axis, n_channels) or (None, None) if not 3D.
    """
    if arr is None or not isinstance(arr, np.ndarray) or arr.ndim != 3:
        return None, None
    shp = arr.shape
    ch_axis = int(np.argmin(shp))
    n_ch = int(shp[ch_axis])
    # if ambiguous (all big), prefer channels-last when last dim <= 10
    if not (n_ch <= 10 and all(d > 32 for k, d in enumerate(shp) if k != ch_axis)):
        if shp[2] <= 10:
            ch_axis, n_ch = 2, shp[2]
    return ch_axis, n_ch


# -----------------------
# image-level inspection
# -----------------------
def analyze_image_file(img_path: str):
    """Read a single image with cellpose.io and print core stats."""
    if cp_io is None:
        print("[analyze_image_file] cellpose.io not available in this env.")
        return
    p = Path(img_path)
    img = cp_io.imread(p)
    _print_header(f"[IMAGE] {p.name}")
    print(f"shape={_fmt_shape(img)}  dtype={_fmt_dtype(img)}  min/max={_minmax(img)}")
    ax, nc = detect_channel_axis(img)
    if ax is not None:
        print(f"channel_axis={ax}  n_channels={nc}")
        if nc and nc > 3:
            print(f"[NOTE] detected >3 channels; CP-SAM will use first 3.")


# -----------------------
# _seg.npy inspection
# -----------------------
SEG_EXPECTED_KEYS = {
    "filename", "masks", "outlines", "chan_choose", "ismanual", "flows", "zdraw"
}

def analyze_seg_npy(seg_path: str, expect_mask_shape: Optional[Tuple[int,int]] = None):
    """Load a single *_seg.npy and print keys + shapes/dtypes + quick sanity."""
    p = Path(seg_path)
    _print_header(f"[SEG] {p.name}")
    try:
        seg = np.load(p, allow_pickle=True).item()
    except Exception as ex:
        print(f"[ERROR] cannot load {p}: {ex}")
        return

    keys = set(seg.keys())
    print("keys:", sorted(keys))

    missing = [k for k in SEG_EXPECTED_KEYS if k not in keys]
    if missing:
        print("[WARN] missing expected keys:", missing)

    masks    = seg.get("masks")
    outlines = seg.get("outlines")
    flows    = seg.get("flows")
    chan     = seg.get("chan_choose")
    ismanual = seg.get("ismanual")
    zdraw    = seg.get("zdraw")

    print("masks:",   _fmt_shape(masks),   _fmt_dtype(masks))
    print("outlines:",_fmt_shape(outlines),_fmt_dtype(outlines))
    print("chan_choose:", chan)
    print("ismanual:", type(ismanual).__name__, f"len={len(ismanual)}" if hasattr(ismanual,"__len__") else None)
    print("zdraw:",    type(zdraw).__name__, f"len={len(zdraw)}" if hasattr(zdraw,"__len__") else None)

    if isinstance(flows, (list, tuple)):
        print(f"flows: list len={len(flows)}")
        # Common mapping in docs:
        # flows[0]=XY flow in RGB (HSV-like) ; flows[1]=cellprob 0-255 ; flows[2]=Z flow or zeros
        # flows[3]=[dY,dX,cellprob] (or [dZ,dY,dX,cellprob]) ; flows[4]=pixel destinations
        for idx, comp in enumerate(flows):
            print(f"  flows[{idx}]: shape={_fmt_shape(comp)} dtype={_fmt_dtype(comp)}")
    else:
        print("flows: None or unexpected type:", type(flows).__name__)

    if expect_mask_shape and hasattr(masks, "shape"):
        if tuple(masks.shape) != tuple(expect_mask_shape):
            print(f"[WARN] mask shape {masks.shape} != expected {expect_mask_shape}")
        else:
            print(f"[OK] mask shape matches expected {expect_mask_shape}")


# -----------------------
# eval-output inspection (objects already in memory)
# -----------------------
def analyze_eval_outputs(img: np.ndarray, masks: np.ndarray, flows: Any, styles: Any, label: str = ""):
    """Print shapes/types of raw eval outputs (single image)."""
    _print_header(f"[EVAL OUT] {label or 'image'}")
    print("img:",   _fmt_shape(img),   _fmt_dtype(img), "min/max=", _minmax(img))
    ax, nc = detect_channel_axis(img)
    if ax is not None:
        print(f"  channel_axis={ax}  n_channels={nc}")
    print("masks:", _fmt_shape(masks), _fmt_dtype(masks), "n_masks=", int(getattr(masks, "max", lambda:0)()))
    if isinstance(flows, (list, tuple)):
        print(f"flows: list len={len(flows)}")
        for i, comp in enumerate(flows):
            print(f"  flows[{i}]: shape={_fmt_shape(comp)} dtype={_fmt_dtype(comp)}")
    else:
        print("flows:", type(flows).__name__)
    print("styles:", type(styles).__name__)


# -----------------------
# directory helpers
# -----------------------
def summarize_seg_dir(dir_path: str, limit: Optional[int] = 10):
    """List seg files in a directory and print a brief summary for the first N."""
    d = Path(dir_path)
    files = sorted(d.glob("*_seg.npy"))
    _print_header(f"[SEG DIR] {d}  (count={len(files)})")
    for i, p in enumerate(files[:limit]):
        try:
            seg = np.load(p, allow_pickle=True).item()
            masks = seg.get("masks")
            flows = seg.get("flows")
            print(f"{i+1:3d}. {p.name}  masks={_fmt_shape(masks)}  flows={'len='+str(len(flows)) if isinstance(flows,(list,tuple)) else type(flows).__name__}")
        except Exception as ex:
            print(f"{i+1:3d}. {p.name}  [ERROR loading]: {ex}")


# -----------------------
# optional: model/provenance
# -----------------------
def analyze_model(model: Any):
    """Best-effort print of model/device; safe if not a CellposeModel."""
    _print_header("[MODEL]")
    if CellposeModel is not None and isinstance(model, CellposeModel):
        dev = getattr(getattr(model, "device", None), "type", None)
        print("CellposeModel detected; device:", dev)
        print("attrs:", [k for k in dir(model) if not k.startswith("_")][:20], "...")
    else:
        print("Unknown model type:", type(model).__name__)


# -----------------------
# CLI
# -----------------------
def _cli():
    ap = argparse.ArgumentParser("cellpose_seg_analyzer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_img = sub.add_parser("img", help="inspect a raw image file")
    p_img.add_argument("image", type=str)

    p_seg = sub.add_parser("seg", help="inspect a single *_seg.npy")
    p_seg.add_argument("segfile", type=str)

    p_dir = sub.add_parser("segdir", help="summarize *_seg.npy in a directory")
    p_dir.add_argument("dir", type=str)
    p_dir.add_argument("--limit", type=int, default=10)

    args = ap.parse_args()

    if args.cmd == "img":
        analyze_image_file(args.image)
    elif args.cmd == "seg":
        analyze_seg_npy(args.segfile)
    elif args.cmd == "segdir":
        summarize_seg_dir(args.dir, args.limit)


if __name__ == "__main__":
    _cli()
    
