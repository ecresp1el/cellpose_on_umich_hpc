#!/usr/bin/env python3
# Minimal, print-only checker for Stage C artifacts using eval_summary.json
# Works with Cellpose v3 outputs; no refactors, just diagnostics.

import json, os, sys, numpy as np
from pathlib import Path

try:
    from tifffile import imread
except Exception as e:
    print("[ERROR] tifffile not installed. Try: pip install tifffile")
    sys.exit(1)

def fsize(path: Path) -> str:
    try:
        return f"{path.stat().st_size} B"
    except Exception:
        return "N/A"

def load_any(path: Path):
    if path.suffix == ".npy":
        return np.load(path)
    else:
        return imread(path)

def nz_summary(arr: np.ndarray) -> str:
    nnz = int(np.count_nonzero(arr))
    try:
        amin = float(arr.min())
        amax = float(arr.max())
        mean = float(arr.mean())
    except ValueError:
        amin = amax = mean = float("nan")
    return f"shape={tuple(arr.shape)} dtype={arr.dtype} nnz={nnz} min={amin:.4g} max={amax:.4g} mean={mean:.4g}"

def check_path(tag: str, p: str, expect_hw=None):
    pth = Path(p)
    if not pth.exists():
        print(f"  [{tag}] MISSING: {pth}")
        return False, None
    # size sanity
    sz = fsize(pth)
    try:
        arr = load_any(pth)
        msg = nz_summary(arr)
        ok_shape = True
        if expect_hw is not None:
            # Accept (H,W) or (H,W,C)/(C,H,W) for certain artifacts
            H, W = expect_hw
            s = arr.shape
            if s[:2] != (H, W):
                # allow (C,H,W)
                ok_shape = (len(s) == 3 and s[1:] == (H, W))
        shape_note = "" if ok_shape else "  [WARN shape mismatch vs input]"
        print(f"  [{tag}] OK: {pth.name}  ({sz})  {msg}{shape_note}")
        return True, arr
    except Exception as e:
        print(f"  [{tag}] ERROR reading {pth}  ({sz})  -> {e}")
        return False, None

def main(eval_summary_json: str):
    js = json.loads(Path(eval_summary_json).read_text())
    per = js.get("per_image", [])
    print(f"\n[Stage C Artifact Audit]\n- file: {eval_summary_json}\n- images listed: {len(per)}\n")

    totals = {"ok_any":0, "missing_any":0, "empty_prob":0, "empty_masks":0, "empty_flows":0}
    for item in per:
        stem = item.get("stem", "<unknown>")
        H, W = None, None
        shp = item.get("shape")
        if isinstance(shp, list) and len(shp) >= 2:
            H, W = int(shp[0]), int(shp[1])

        print(f"\n=== {stem} ===")
        paths = item.get("paths", {})

        ok_m, m = check_path("masks", paths.get("masks",""), expect_hw=(H,W) if H and W else None)
        ok_p, p = check_path("prob", paths.get("prob_tif",""), expect_hw=(H,W) if H and W else None)
        ok_pv,_ = check_path("prob_view", paths.get("prob_view_tif",""), expect_hw=(H,W) if H and W else None)
        ok_fn, fn = check_path("flows_npy", paths.get("flows_npy",""), expect_hw=(H,W) if H and W else None)
        ok_ft,_ = check_path("flows_tif", paths.get("flows_tif",""), expect_hw=(H,W) if H and W else None)
        ok_panel,_ = check_path("panel_png", paths.get("panel_png",""))

        any_missing = not all([ok_m, ok_p, ok_pv, ok_fn, ok_ft, ok_panel])
        if any_missing:
            totals["missing_any"] += 1
        else:
            totals["ok_any"] += 1

        # emptiness tests (strict)
        if ok_p and isinstance(p, np.ndarray) and np.count_nonzero(p) == 0:
            totals["empty_prob"] += 1
            print("  [NOTE] prob_tif appears all-zero.")

        if ok_m and isinstance(m, np.ndarray) and np.count_nonzero(m) == 0:
            totals["empty_masks"] += 1
            print("  [NOTE] masks appear all-zero.")

        if ok_fn and isinstance(fn, np.ndarray) and np.count_nonzero(fn) == 0:
            totals["empty_flows"] += 1
            print("  [NOTE] flows_npy appear all-zero.")

        # quick consistency: do stems/dims align?
        if ok_m and ok_p and ok_fn and isinstance(m, np.ndarray) and isinstance(p, np.ndarray) and isinstance(fn, np.ndarray):
            mhw = m.shape[:2]
            phw = p.shape[:2]
            # flows may be (2,H,W) or (H,W,2); check both
            fs = fn.shape
            fhw = fs[-2:] if len(fs) == 3 else fs[:2]
            if not (mhw == phw == fhw):
                print(f"  [WARN] Dim mismatch across artifacts: masks{mhw} prob{phw} flows{fs}")

    print("\n--- SUMMARY ---")
    print(f"images with ALL expected artifacts present: {totals['ok_any']}")
    print(f"images with ANY missing artifact:           {totals['missing_any']}")
    print(f"images with ALL-ZERO prob_tif:              {totals['empty_prob']}")
    print(f"images with ALL-ZERO masks:                 {totals['empty_masks']}")
    print(f"images with ALL-ZERO flows_npy:             {totals['empty_flows']}")
    print("\nDone.\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_from_eval_summary.py /path/to/eval_summary.json")
        sys.exit(2)
    main(sys.argv[1])