# cp_core/cellpose_seg_analyzer.py
from __future__ import annotations

import argparse
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
# channel-axis heuristic (for image-only inspection)
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
# _seg.npy inspection (as-is)
# -----------------------
SEG_EXPECTED_KEYS = {
    "filename", "masks", "outlines", "chan_choose", "ismanual", "flows", "zdraw"
}

def analyze_seg_npy(seg_path: str, expect_mask_shape: Optional[Tuple[int,int]] = None):
    """Load a single *_seg.npy and print keys + shapes/dtypes + quick sanity (AS-IS)."""
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

    print("masks:",    _fmt_shape(masks),    _fmt_dtype(masks))
    print("outlines:", _fmt_shape(outlines), _fmt_dtype(outlines))
    print("chan_choose:", chan)
    print("ismanual:", type(ismanual).__name__, f"len={len(ismanual)}" if hasattr(ismanual,"__len__") else None)
    print("zdraw:",    type(zdraw).__name__,  f"len={len(zdraw)}" if hasattr(zdraw,"__len__") else None)

    # ismanual summary
    if isinstance(ismanual, np.ndarray):
        n_true = int(np.count_nonzero(ismanual))
        n_total = int(ismanual.size)
        print(f"ismanual summary: true={n_true}  false={n_total - n_true}  total={n_total}")
        print("  note: ismanual[k] == True if mask k was drawn manually in GUI; False if computed by Cellpose.")

    # flows mapping (AS-IS)
    def _mm(x):
        try:
            return f"min={np.nanmin(x):.3f} max={np.nanmax(x):.3f}"
        except Exception:
            return "min/max=NA"

    if isinstance(flows, (list, tuple)):
        hsv_rgb   = flows[0] if len(flows) > 0 else None
        prob_255  = flows[1] if len(flows) > 1 else None
        zflow_255 = flows[2] if len(flows) > 2 else None
        pack_3    = flows[3] if len(flows) > 3 else None
        dydxprob  = flows[4] if len(flows) > 4 else None

        print("\n[flows mapping]")
        print("  flows[0] = XY flow (RGB viz):        ", str(getattr(hsv_rgb,'shape',None)),  str(getattr(hsv_rgb,'dtype',None)),  _mm(hsv_rgb))
        print("  flows[1] = cell probability (0-255): ", str(getattr(prob_255,'shape',None)), str(getattr(prob_255,'dtype',None)), _mm(prob_255))
        print("  flows[2] = Z flow (0-255):            ", str(getattr(zflow_255,'shape',None)),str(getattr(zflow_255,'dtype',None)),_mm(zflow_255))
        print("  flows[3] = optional pack (legacy):    ", str(getattr(pack_3,'shape',None)),    str(getattr(pack_3,'dtype',None)))
        print("  flows[4] = [dY, dX, cellprob] f32:    ", str(getattr(dydxprob,'shape',None)),  str(getattr(dydxprob,'dtype',None)), _mm(dydxprob))

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


# ==== PROBE (structured summary + minimal checks, AS-IS) ======================
def _arr_info(arr):
    if arr is None:
        return {"type": None}
    info = {
        "type": type(arr).__name__,
        "dtype": str(getattr(arr, "dtype", None)),
        "shape": tuple(getattr(arr, "shape", ())),
    }
    if hasattr(arr, "size") and hasattr(arr, "dtype") and arr.size and np.issubdtype(getattr(arr, "dtype", None), np.number):
        try:
            info["min"] = float(np.nanmin(arr))
            info["max"] = float(np.nanmax(arr))
        except Exception:
            pass
    return info

def probe_seg_npy(path):
    """Load a *_seg.npy and summarize contents (AS-IS)."""
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "error": "file not found"}
    try:
        dat = np.load(p, allow_pickle=True).item()
    except Exception as e:
        return {"path": str(p), "error": f"failed to load as pickled dict: {e}"}

    masks     = dat.get("masks")
    outlines  = dat.get("outlines")
    flows     = dat.get("flows")
    channels  = dat.get("chan_choose") or dat.get("channels")
    ismanual  = dat.get("ismanual")
    filename  = dat.get("filename")
    images    = dat.get("img") if "img" in dat else dat.get("images")
    est_diam  = dat.get("est_diam") or dat.get("diams") or dat.get("diameter")
    zdraw     = dat.get("zdraw")
    colors    = dat.get("colors")

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

    if outlines is not None and masks is not None and hasattr(outlines, "shape") and outlines.shape != masks.shape:
        checks["ok"] = False
        checks["notes"].append(f"outlines shape {outlines.shape} != masks shape {masks.shape}")

    if isinstance(flows, (list, tuple)) and masks is not None and masks_ndim == 2:
        f0 = flows[0] if len(flows) > 0 else None
        f1 = flows[1] if len(flows) > 1 else None
        if f0 is not None and hasattr(f0, "shape"):
            f0_hw = (f0.shape[-2], f0.shape[-1]) if f0.ndim >= 2 else None
            m_hw  = (masks_shape[-2], masks_shape[-1])
            if f0_hw and f0_hw != m_hw:
                checks["ok"] = False
                checks["notes"].append(f"flows[0] spatial {f0_hw} != masks {m_hw} (raw)")
        if f1 is not None and hasattr(f1, "shape"):
            f1_hw = (f1.shape[-2], f1.shape[-1]) if f1.ndim >= 2 else None
            m_hw  = (masks_shape[-2], masks_shape[-1])
            if f1_hw and f1_hw != m_hw:
                checks["ok"] = False
                checks["notes"].append(f"flows[1] spatial {f1_hw} != masks {m_hw} (raw)")

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
        a = summary[name]
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

def cli_probe(paths: List[str]) -> int:
    rc = 0
    for p in paths:
        s = probe_seg_npy(p)
        print_seg_summary(s)
        rc |= 0 if (("error" not in s) and s["checks"]["ok"]) else 1
    return rc


# ==== SPEC vs OBSERVED AUDIT (AS-IS, NO NORMALIZATION) ========================
from dataclasses import dataclass

@dataclass
class Finding:
    level: str   # "OK","WARN","ERROR","INFO"
    code: str
    msg: str

# Ground-truth spec (loose but explicit) for Cellpose v4 save format
SPEC_REQUIRED_KEYS = {"masks", "outlines", "flows", "filename"}
SPEC_OPTIONAL_KEYS = {"chan_choose", "channels", "ismanual", "est_diam", "diams", "diameter", "zdraw", "img", "colors"}

SPEC_TEXT = [
    "Required keys: masks, outlines, flows, filename.",
    "masks: integer dtype; 2D (H,W) for 2D, or 3D for 3D volumes.",
    "outlines: integer dtype; same shape as masks.",
    "flows: list/tuple; typical v4 has len>=5:",
    "  [0] uint8 RGB-like visualization; spatial HxW should match masks.",
    "  [1] uint8 cell probability; spatial HxW should match masks.",
    "  [2] uint8 Z-flow viz (often zeros in 2D); spatial HxW should match masks.",
    "  [3] optional legacy/None/list.",
    "  [4] float32 raw fields; shape (3,H,W) for 2D or (4,H,W) for 3D; spatial HxW matches masks.",
    "ismanual: length should equal max(masks) (one flag per label id).",
    "chan_choose/channels: typically two ints like [0,0] / [2,1].",
    "filename: string (full path without suffix) or list-like (GUI batches).",
    "img/colors/zdraw may be present; img often omitted to keep file small.",
]

def _raw_spatial_hw(a):
    """Derive spatial (H,W) from raw array WITHOUT modifying it."""
    if a is None or not hasattr(a, "shape"):
        return None
    sh = a.shape
    # RGB(A)-like (..., H, W, C): use last two dims as H,W
    if len(sh) >= 3 and sh[-1] in (3, 4):
        return (sh[-2], sh[-1])
    # Generic: use the last two dims as (H, W)
    if len(sh) >= 2:
        return (sh[-2], sh[-1])
    return None

def audit_raw_seg_dict(dat: dict) -> Tuple[List[Finding], dict]:
    f: List[Finding] = []

    # --- SPEC SECTION ---
    # We print the spec separately in the report printer.

    # --- OBSERVED vs SPEC CHECKS (AS-IS) ---
    keys = set(dat.keys())
    missing = SPEC_REQUIRED_KEYS - keys
    if missing:
        f.append(Finding("ERROR","KEYS.MISSING",f"missing required keys: {sorted(missing)}"))
    else:
        f.append(Finding("OK","KEYS.PRESENT",f"required keys present: {sorted(SPEC_REQUIRED_KEYS)}"))

    extra = keys - (SPEC_REQUIRED_KEYS | SPEC_OPTIONAL_KEYS)
    if extra:
        f.append(Finding("INFO","KEYS.EXTRA",f"extra keys present (not in spec list): {sorted(extra)}"))

    masks    = dat.get("masks")
    outlines = dat.get("outlines")
    flows    = dat.get("flows")
    chans    = dat.get("chan_choose", dat.get("channels"))
    ismanual = dat.get("ismanual")
    est_diam = dat.get("est_diam", dat.get("diams", dat.get("diameter")))
    zdraw    = dat.get("zdraw")
    img      = dat.get("img")
    filename = dat.get("filename")

    # masks
    if masks is None:
        f.append(Finding("ERROR","MASKS.MISSING","'masks' missing"))
        masks_hw = None
    else:
        if not np.issubdtype(getattr(masks,"dtype",np.array([],np.uint8)).type, np.integer):
            f.append(Finding("ERROR","MASKS.DTYPE",f"masks dtype should be integer; observed {getattr(masks,'dtype',None)}"))
        else:
            f.append(Finding("OK","MASKS.DTYPE",f"masks dtype {masks.dtype}"))
        f.append(Finding("INFO","MASKS.SHAPE",f"masks raw shape={getattr(masks,'shape',None)}"))
        f.append(Finding("INFO","MASKS.NDIM",f"masks ndim={getattr(masks,'ndim',None)}"))
        masks_hw = _raw_spatial_hw(masks)

    # outlines
    if outlines is None:
        f.append(Finding("ERROR","OUTLINES.MISSING","'outlines' missing"))
    else:
        f.append(Finding("INFO","OUTLINES.SHAPE",f"outlines raw shape={outlines.shape}"))
        if hasattr(masks,"shape") and outlines.shape != masks.shape:
            f.append(Finding("ERROR","OUTLINES!=MASKS",f"outlines shape {outlines.shape} != masks {getattr(masks,'shape',None)}"))
        if np.issubdtype(getattr(outlines,"dtype",np.array([],np.uint8)).type, np.integer):
            f.append(Finding("OK","OUTLINES.DTYPE",f"outlines dtype {outlines.dtype}"))
        else:
            f.append(Finding("ERROR","OUTLINES.DTYPE",f"outlines dtype not integer; observed {getattr(outlines,'dtype',None)}"))

    # flows (AS-IS)
    if isinstance(flows,(list,tuple)):
        f.append(Finding("INFO","FLOWS.LEN",f"flows length={len(flows)}"))
        for i, fi in enumerate(flows):
            if hasattr(fi, "shape"):
                sh = fi.shape
                f.append(Finding("INFO",f"FLOWS[{i}].SHAPE",f"raw shape={sh}"))
                f.append(Finding("INFO",f"FLOWS[{i}].DTYPE",f"dtype={getattr(fi,'dtype',None)}"))
                if len(sh) >= 1 and sh[0] == 1:
                    f.append(Finding("INFO",f"FLOWS[{i}].LEAD1","leading singleton axis at dim 0 observed (e.g., batch/Z=1)"))
                fi_hw = _raw_spatial_hw(fi)
                if fi_hw and masks_hw:
                    lvl = "OK" if fi_hw == masks_hw else "ERROR"
                    reason = "" if lvl == "OK" else " (raw dims differ; spec expects spatial HxW to match masks)"
                    f.append(Finding(lvl,f"FLOWS[{i}].SPATIAL",f"(H,W)={fi_hw} vs masks (H,W)={masks_hw}{reason}"))
                # dtype expectations for common indices (no coercion; just compare)
                if i == 0 and hasattr(fi,"dtype") and fi.dtype != np.uint8:
                    f.append(Finding("WARN","FLOWS[0].DTYPE.EXPECT","expected uint8 visualization; observed {0}".format(fi.dtype)))
                if i == 1 and hasattr(fi,"dtype") and fi.dtype != np.uint8:
                    f.append(Finding("WARN","FLOWS[1].DTYPE.EXPECT","expected uint8 cellprob; observed {0}".format(fi.dtype)))
                if i == 4 and hasattr(fi,"dtype") and fi.dtype != np.float32:
                    f.append(Finding("WARN","FLOWS[4].DTYPE.EXPECT","expected float32 raw fields; observed {0}".format(fi.dtype)))
                if i == 4 and hasattr(fi,"shape"):
                    if not (fi.ndim == 3 and fi.shape[0] in (3,4)):
                        f.append(Finding("WARN","FLOWS[4].SHAPE.EXPECT","expected (3,H,W) or (4,H,W)"))
            else:
                f.append(Finding("INFO",f"FLOWS[{i}].TYPE",f"type={type(fi).__name__} (no shape)"))
    else:
        f.append(Finding("ERROR","FLOWS.TYPE",f"'flows' expected list/tuple; observed {type(flows).__name__}"))

    # ismanual vs labels
    if ismanual is not None and masks is not None and hasattr(masks,"size"):
        nlabels = int(masks.max()) if masks.size else 0
        try:
            ok_len = (len(ismanual) == nlabels)
            f.append(Finding("OK" if ok_len else "WARN","ISMANUAL.LEN",
                             f"len(ismanual)={len(ismanual)} vs max(masks)={nlabels}"))
        except Exception as e:
            f.append(Finding("WARN","ISMANUAL.TYPE",f"ismanual type={type(ismanual).__name__} not sized: {e}"))
    elif ismanual is None:
        f.append(Finding("INFO","ISMANUAL.MISSING","'ismanual' not present"))

    # channels
    if chans is None:
        f.append(Finding("INFO","CHANNELS.MISSING","channels/chan_choose not present"))
    else:
        try:
            values = list(chans) if not isinstance(chans,str) else chans
        except Exception:
            values = repr(chans)
        f.append(Finding("INFO","CHANNELS.VALUE",f"channels raw value={values}"))
        # light expectation (no coercion)
        try:
            if isinstance(values, list) and len(values) != 2:
                f.append(Finding("WARN","CHANNELS.EXPECT","expected two ints like [0,0] or [2,1]"))
        except Exception:
            pass

    # filename
    if isinstance(filename, str):
        f.append(Finding("OK","FILENAME.STR","filename stored as string"))
    elif isinstance(filename, (list, tuple, np.ndarray)):
        f.append(Finding("INFO","FILENAME.LIST",f"filename stored as list-like (len={len(filename)})"))
    else:
        f.append(Finding("WARN","FILENAME.TYPE",f"filename unexpected type: {type(filename).__name__}"))

    # embedded image
    f.append(Finding("INFO","IMG.PRESENT",f"'img' present={img is not None}"))

    # overall status: ERROR trumps WARN; WARN trumps OK
    status = "OK"
    if any(x.level=="ERROR" for x in f): status = "ERROR"
    elif any(x.level=="WARN" for x in f): status = "WARN"

    meta = {
        "status": status,
        "keys": sorted(keys),
        "spec_lines": SPEC_TEXT,
    }
    return f, meta

def print_audit_raw_report(path: str, findings: List[Finding], meta: dict):
    print(f"\n=== AUDIT-RAW {path} ===")
    # SPEC section
    print("[spec] Cellpose v4 save expectations:")
    for line in meta.get("spec_lines", []):
        print("  -", line)
    # OBSERVED header
    print(f"\n[observed] keys={meta.get('keys', [])}")
    # CHECKS grouped
    print(f"\n[status] {meta['status']}")
    for lvl in ("ERROR","WARN","OK","INFO"):
        group = [x for x in findings if x.level==lvl]
        if group:
            print(f"\n[{lvl}]")
            for g in group:
                print(f"  - {g.code}: {g.msg}")

def cli_audit_raw(paths: list) -> int:
    rc = 0
    for p in paths:
        dat = np.load(p, allow_pickle=True).item()
        findings, meta = audit_raw_seg_dict(dat)
        print_audit_raw_report(p, findings, meta)
        if meta["status"] == "ERROR":
            rc = 2  # ONLY errors cause nonzero exit
    return rc


# -----------------------
# CLI
# -----------------------
def _cli():
    import glob

    ap = argparse.ArgumentParser("cellpose_seg_analyzer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # raw image inspection
    p_img = sub.add_parser("img", help="inspect a raw image file")
    p_img.add_argument("image", type=str)

    # single seg file (as-is viewer)
    p_seg = sub.add_parser("seg", help="inspect a single *_seg.npy")
    p_seg.add_argument("segfile", type=str)

    # directory summary of seg files
    p_dir = sub.add_parser("segdir", help="summarize *_seg.npy in a directory")
    p_dir.add_argument("dir", type=str)
    p_dir.add_argument("--limit", type=int, default=10)

    # structured probe (summary + minimal checks)
    p_probe = sub.add_parser("probe", help="probe one or more *_seg.npy files (AS-IS)")
    p_probe.add_argument("paths", nargs="+", help="files or globs (e.g., '/data/*_seg.npy')")

    # spec vs observed audit (NO NORMALIZATION)
    p_audit = sub.add_parser("audit", help="report spec vs observed for *_seg.npy (no normalization)")
    p_audit.add_argument("paths", nargs="+", help="files or globs (e.g., '/data/*_seg.npy')")

    args = ap.parse_args()

    if args.cmd == "img":
        analyze_image_file(args.image)

    elif args.cmd == "seg":
        analyze_seg_npy(args.segfile)

    elif args.cmd == "segdir":
        summarize_seg_dir(args.dir, args.limit)

    elif args.cmd == "probe":
        files = []
        for p in args.paths:
            expanded = glob.glob(p)
            files.extend(expanded if expanded else [p])
        if not files:
            print("[ERR] no files matched")
            sys.exit(1)
        rc = cli_probe(sorted(set(files)))
        sys.exit(rc)

    elif args.cmd == "audit":
        files = []
        for p in args.paths:
            expanded = glob.glob(p)
            files.extend(expanded if expanded else [p])
        if not files:
            print("[ERR] no files matched")
            sys.exit(3)
        rc = cli_audit_raw(sorted(set(files)))
        sys.exit(rc)


if __name__ == "__main__":
    _cli()