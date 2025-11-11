# cp_core/cellpose_seg_analyzer.py
from __future__ import annotations

import argparse
import json
import os
import sys
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

    # ----- ismanual summary (counts) -----
    if isinstance(ismanual, np.ndarray):
        n_true = int(np.count_nonzero(ismanual))
        n_total = int(ismanual.size)
        print(f"ismanual summary: true={n_true}  false={n_total - n_true}  total={n_total}")
        print("  note: ismanual[k] == True if mask k was drawn manually in GUI; False if computed by Cellpose.")

    # ----- flows mapping with v4 normalization (squeeze batch dim if present) -----
    def _squeeze1(x):
        return x[0] if isinstance(x, np.ndarray) and x.ndim >= 1 and x.shape[0] == 1 else x

    def _mm(x):
        try:
            import numpy as _np
            return f"min={_np.nanmin(x):.3f} max={_np.nanmax(x):.3f}"
        except Exception:
            return "min/max=NA"

    if isinstance(flows, (list, tuple)):
        # v4 typical pack (may vary by build):
        #  flows[0] = XY flow RGB viz (uint8)            (1,H,W,3) or (H,W,3)
        #  flows[1] = cell probability 0..255 (uint8)    (1,H,W)   or (H,W)
        #  flows[2] = Z flow 0..255 (uint8, zeros in 2D) (1,H,W,3) or (H,W,3)
        #  flows[3] = optional pack / None
        #  flows[4] = [dY, dX, cellprob] float32 (or [dZ, dY, dX, cellprob] in 3D)
        hsv_rgb   = _squeeze1(flows[0]) if len(flows) > 0 else None
        prob_255  = _squeeze1(flows[1]) if len(flows) > 1 else None
        zflow_255 = _squeeze1(flows[2]) if len(flows) > 2 else None
        pack_3    = flows[3]            if len(flows) > 3 else None
        dydxprob  = flows[4]            if len(flows) > 4 else None

        print("\n[flows mapping]")
        print("  flows[0] = XY flow (RGB viz):       ", str(getattr(hsv_rgb,'shape',None)),  str(getattr(hsv_rgb,'dtype',None)),  _mm(hsv_rgb))
        print("  flows[1] = cell probability (0-255):", str(getattr(prob_255,'shape',None)), str(getattr(prob_255,'dtype',None)), _mm(prob_255))
        print("  flows[2] = Z flow (0-255):           ", str(getattr(zflow_255,'shape',None)),str(getattr(zflow_255,'dtype',None)),_mm(zflow_255))
        print("  flows[3] = optional pack (legacy):   ", str(getattr(pack_3,'shape',None)),    str(getattr(pack_3,'dtype',None)))
        print("  flows[4] = [dY, dX, cellprob] f32:   ", str(getattr(dydxprob,'shape',None)),  str(getattr(dydxprob,'dtype',None)), _mm(dydxprob))

        # If flows[4] present, break into components
        if isinstance(dydxprob, np.ndarray) and dydxprob.ndim >= 3 and dydxprob.shape[0] in (3, 4):
            names = ["dY", "dX", "cellprob"] if dydxprob.shape[0] == 3 else ["dZ", "dY", "dX", "cellprob"]
            for idx, name in enumerate(names):
                comp = dydxprob[idx]
                print(f"    flows[4][{idx}] = {name}: shape={getattr(comp,'shape',None)} dtype={getattr(comp,'dtype',None)} {_mm(comp)}")

    print("\n[legend]")
    print("  ismanual[k]: True if mask k drawn by hand in GUI, False if computed")
    print("  flows[0]: XY flow field visualization (RGB/HSV-like) used for display")
    print("  flows[1]: cell probability as uint8 (0..255)")
    print("  flows[2]: Z-flow as uint8 (zeros for 2D)")
    print("  flows[4]: raw fields (float32): [dY, dX, cellprob] for 2D  |  [dZ, dY, dX, cellprob] for 3D")
    
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


# ==== SEG PROBE (drop-in) =========================================
from pathlib import Path
import numpy as np

def _arr_info(arr):
    if arr is None:
        return {"type": None}
    info = {
        "type": type(arr).__name__,
        "dtype": str(getattr(arr, "dtype", None)),
        "shape": tuple(getattr(arr, "shape", ())),
    }
    if hasattr(arr, "size") and hasattr(arr, "dtype") and arr.size and np.issubdtype(arr.dtype, np.number):
        try:
            info["min"] = float(np.nanmin(arr))
            info["max"] = float(np.nanmax(arr))
        except Exception:
            pass
    return info

def probe_seg_npy(path):
    """
    Load a Cellpose *_seg.npy and summarize contents.
    Returns a dict with keys: keys, masks, outlines, flows, ismanual, images,
    est_diam, zdraw, colors, filename, channels, checks.
    """
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "error": "file not found"}

    try:
        dat = np.load(p, allow_pickle=True).item()
    except Exception as e:
        return {"path": str(p), "error": f"failed to load as pickled dict: {e}"}

    # tolerate key variations across versions
    masks     = dat.get("masks")
    outlines  = dat.get("outlines")
    flows     = dat.get("flows")
    channels  = dat.get("chan_choose") or dat.get("channels")
    ismanual  = dat.get("ismanual")
    filename  = dat.get("filename")
    images    = dat.get("img") if "img" in dat else dat.get("images")
    est_diam  = dat.get("est_diam") or dat.get("diams")
    zdraw     = dat.get("zdraw")
    colors    = dat.get("colors")

    # basic checks
    checks = {"ok": True, "notes": []}
    if masks is None:
        checks["ok"] = False
        checks["notes"].append("masks missing")
        n_labels = 0
        masks_ndim = None
        masks_shape = None
    else:
        if not np.issubdtype(masks.dtype, np.integer):
            checks["ok"] = False
            checks["notes"].append(f"masks dtype should be integer, got {masks.dtype}")
        if masks.ndim not in (2, 3):
            checks["ok"] = False
            checks["notes"].append(f"masks should be 2D or 3D, got ndim={masks.ndim}")
        masks_shape = masks.shape
        masks_ndim = masks.ndim
        n_labels = int(masks.max()) if masks.size else 0

    if outlines is not None and masks is not None and outlines.shape != masks.shape:
        checks["ok"] = False
        checks["notes"].append(f"outlines shape {outlines.shape} != masks shape {masks.shape}")

    if isinstance(flows, (list, tuple)) and masks is not None and masks_ndim == 2:
        f0 = flows[0] if len(flows) > 0 else None  # XY RGB viz
        f1 = flows[1] if len(flows) > 1 else None  # cellprob
        if f0 is not None and hasattr(f0, "shape") and f0.shape[:2] != masks_shape[:2]:
            checks["ok"] = False
            checks["notes"].append(f"flows[0] spatial {f0.shape[:2]} != masks {masks_shape[:2]}")
        if f1 is not None and hasattr(f1, "shape") and f1.shape[:2] != masks_shape[:2]:
            checks["ok"] = False
            checks["notes"].append(f"flows[1] spatial {f1.shape[:2]} != masks {masks_shape[:2]}")

    summary = {
        "path": str(p),
        "keys": sorted(list(dat.keys())),
        "filename": filename,
        "channels": channels,
        "labels_max": n_labels,
        "masks": _arr_info(masks),
        "outlines": _arr_info(outlines),
        "ismanual": _arr_info(ismanual),
        "images": _arr_info(images),
        "est_diam": (float(est_diam) if np.ndim(est_diam)==0 else np.array(est_diam).tolist()) if est_diam is not None else None,
        "zdraw": _arr_info(zdraw),
        "colors": _arr_info(colors),
        "flows": {
            "len": (len(flows) if isinstance(flows, (list, tuple)) else 0),
            **({f"[{i}]": _arr_info(flows[i]) for i in range(len(flows))} if isinstance(flows, (list, tuple)) else {})
        },
        "checks": checks,
    }
    return summary

def print_seg_summary(summary):
    if "error" in summary:
        print(f"[ERR] {summary['path']}: {summary['error']}")
        return
    print(f"\n=== {summary['path']} ===")
    print("[keys]", summary["keys"])
    print("[meta]")
    fn = summary["filename"]
    if isinstance(fn, (list, tuple, np.ndarray)):
        ex = fn[0] if len(fn) else None
        print(f"  filename: list(len={len(fn)}) example={ex}")
    else:
        print(f"  filename: {fn!r}")
    print(f"  channels: {summary['channels']}")
    print("[arrays]")
    def line(name): 
        a = summary[name]; 
        print(f"  {name}: {a}")
    line("masks"); line("outlines"); line("ismanual"); line("images"); line("zdraw"); line("colors")
    print("[flows]")
    print(f"  len={summary['flows']['len']}")
    for k, v in summary["flows"].items():
        if k != "len":
            print(f"  flows{k}: {v}")
    print("[labels]")
    print(f"  labels_max={summary['labels_max']}")
    print("[checks]")
    print(f"  ok={summary['checks']['ok']}  notes={summary['checks']['notes']}")

# Optional: simple CLI hook you can wire into your analyzer's argparse
def cli_probe(paths):
    rc = 0
    for p in paths:
        s = probe_seg_npy(p)
        print_seg_summary(s)
        rc |= 0 if (("error" not in s) and s["checks"]["ok"]) else 1
    return rc
# ==================================================================


# -----------------------
# CLI
# -----------------------
def _cli():
    import argparse, sys, glob

    ap = argparse.ArgumentParser("cellpose_seg_analyzer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # raw image inspection
    p_img = sub.add_parser("img", help="inspect a raw image file")
    p_img.add_argument("image", type=str)

    # single seg file (kept for backward compat; calls your existing analyzer)
    p_seg = sub.add_parser("seg", help="inspect a single *_seg.npy")
    p_seg.add_argument("segfile", type=str)

    # directory summary of seg files (your existing behavior)
    p_dir = sub.add_parser("segdir", help="summarize *_seg.npy in a directory")
    p_dir.add_argument("dir", type=str)
    p_dir.add_argument("--limit", type=int, default=10)

    # NEW: multi-file probe using the probe_seg_npy/print_seg_summary helpers
    p_probe = sub.add_parser("probe", help="probe one or more *_seg.npy files")
    p_probe.add_argument("paths", nargs="+", help="files or globs (e.g., '/data/*_seg.npy')")

    args = ap.parse_args()

    if args.cmd == "img":
        analyze_image_file(args.image)

    elif args.cmd == "seg":
        # keep existing behavior; if you want to switch seg->probe entirely, replace next line with the probe call
        analyze_seg_npy(args.segfile)

    elif args.cmd == "segdir":
        summarize_seg_dir(args.dir, args.limit)

    elif args.cmd == "probe":
        # expand globs, de-dup, run probe, return nonzero if any file has issues
        files = []
        for p in args.paths:
            expanded = glob.glob(p)
            if expanded:
                files.extend(expanded)
            else:
                files.append(p)  # allow literal path even if no glob match
        if not files:
            print("[ERR] no files matched")
            sys.exit(1)
        rc = 0
        for f in sorted(set(files)):
            summary = probe_seg_npy(f)
            print_seg_summary(summary)
            bad = ("error" in summary) or (not summary.get("checks", {}).get("ok", False))
            rc |= int(bad)
        sys.exit(rc)


if __name__ == "__main__":
    _cli()