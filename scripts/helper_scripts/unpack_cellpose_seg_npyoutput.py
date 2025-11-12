#!/usr/bin/env python3
"""
unpack_cellpose_seg_npyoutput.py

Inspect a Cellpose *_seg.npy file and align each field with the Cellpose v4 API:
- masks:       (H, W) uint16, label image (0 = background)
- outlines:    (H, W) uint16, edge map for visualization
- chan_choose: [int, int], channels used (Cellpose convention)
- ismanual:    (N,) bool, per-ROI manual vs model-generated
- filename:    str, source image path (without extension in some versions)
- diameter:    float, used/estimated diameter (may be NaN if native scale)
- flows:       list with:
    [0] flow_rgb        (1, H, W, 3) uint8   — RGB visualization of flow
    [1] cellprob_u8     (1, H, W)    uint8   — cell probability map (viz)
    [2] dP_visual       (1, H, W, 3) uint8   — visualization of flow field
    [3] placeholder     [] or None           — unused in 2D
    [4] dP_raw          (3, H, W)    float32 — raw (dy, dx, cellprob_float32)
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np


def fmt_shape(x):
    return "None" if x is None else str(getattr(x, "shape", None))


def main():
    ap = argparse.ArgumentParser(
        description="Inspect and explain a Cellpose *_seg.npy file."
    )
    ap.add_argument(
        "path",
        nargs="?",
        default="/nfs/turbo/umms-parent/cellpose_wholeorganoid_model/results/cyto3_base_cp3_wholeorganoid/run_2025-11-10_114555/eval/all/Exp68_Slide1_IGI_PCE_VMIX1_DIV24_F_488_LHX6_568_HA_FusionStitcher_F0__max__LHX6+PCDH19+DAPI_seg.npy",
        help="Path to *_seg.npy (defaults to the example file).",
    )
    args = ap.parse_args()

    path = args.path
    if not os.path.exists(path):
        print(f"[ERROR] File not found:\n  {path}", file=sys.stderr)
        sys.exit(1)

    d = np.load(path, allow_pickle=True).item()

    print(f"\nLoaded: {os.path.basename(path)}")
    print(f"Type:   {type(d)}")
    print(f"Keys:   {list(d.keys())}\n", flush=True)

    # ----- masks -----
    masks = d.get("masks", None)
    if isinstance(masks, np.ndarray):
        h, w = masks.shape
        mask_ids = np.unique(masks)
        n_objs = int((mask_ids != 0).sum())
        print("[masks] label image")
        print(f"  shape={masks.shape} dtype={masks.dtype}  H={h} W={w}")
        print(f"  unique IDs: {mask_ids.min()}..{mask_ids.max()}  (excluding 0 => {n_objs} objects)")
    else:
        print("[masks] MISSING or not an ndarray")

    # ----- outlines -----
    outlines = d.get("outlines", None)
    if isinstance(outlines, np.ndarray):
        print("\n[outlines] edge map for visualization")
        print(f"  shape={outlines.shape} dtype={outlines.dtype}")
    else:
        print("\n[outlines] MISSING or not an ndarray")

    # ----- chan_choose -----
    chan_choose = d.get("chan_choose", None)
    print("\n[chan_choose] channels used (Cellpose convention)")
    print(f"  value={chan_choose}  type={type(chan_choose)}")
    if isinstance(chan_choose, (list, tuple)) and len(chan_choose) == 2:
        print("  meaning: [chan, chan2] — e.g., [0,0]=grayscale; [2,0]=use channel 2 only")

    # ----- ismanual -----
    ismanual = d.get("ismanual", None)
    print("\n[ismanual] per-ROI manual flags (True=GUI-drawn, False=model)")
    if isinstance(ismanual, np.ndarray):
        cnt = Counter(ismanual.tolist())
        print(f"  shape={ismanual.shape} dtype={ismanual.dtype}  -> counts: {dict(cnt)}")
        if isinstance(masks, np.ndarray):
            # Compare ROI count to max label
            max_id = int(masks.max())
            if max_id == ismanual.shape[0]:
                print("  sanity: len(ismanual) == max(masks)  ✅")
            else:
                print(
                    f"  sanity: len(ismanual)={ismanual.shape[0]} vs max(masks)={max_id}  ⚠️"
                )
    else:
        print("  MISSING or not an ndarray")

    # ----- filename -----
    fn = d.get("filename", None)
    print("\n[filename] source image path")
    print(f"  {fn}  (type={type(fn)})")

    # ----- diameter -----
    diam = d.get("diameter", None)
    print("\n[diameter] object diameter used/estimated")
    print(f"  {diam}  (type={type(diam)})")
    if isinstance(diam, float) and (np.isnan(diam) or np.isinf(diam)):
        print("  note: NaN is expected when running at native scale (resample=False)")

    # ----- flows -----
    flows = d.get("flows", None)
    print("\n[flows] list of intermediate fields and visualizations")
    if isinstance(flows, (list, tuple)):
        print(f"  len(flows) = {len(flows)}")
        for i, f in enumerate(flows):
            if isinstance(f, np.ndarray):
                print(f"  flows[{i}] shape={f.shape} dtype={f.dtype}")
            else:
                print(f"  flows[{i}] type={type(f)}  (no array)")

        # Try to interpret common slots per Cellpose v4
        # [0] RGB flow viz; [1] prob uint8; [2] dP viz; [3] placeholder; [4] raw (dy, dx, prob_float32)
        def _safe(arr, idx, note):
            if (
                isinstance(arr, (list, tuple))
                and len(arr) > idx
                and isinstance(arr[idx], np.ndarray)
            ):
                a = arr[idx]
                print(f"    ↳ flows[{idx}] = {note} :: shape={a.shape} dtype={a.dtype}")
            else:
                print(f"    ↳ flows[{idx}] = {note} :: (missing)")

        print("\n  Interpretation (Cellpose v4 pattern):")
        _safe(flows, 0, "flow_rgb (RGB visualization)")
        _safe(flows, 1, "cellprob_u8 (grayscale probability, uint8)")
        _safe(flows, 2, "dP_visual (RGB visualization of flow)")
        # index 3 is often unused
        if len(flows) > 3 and flows[3] is not None and not (
            isinstance(flows[3], (list, tuple)) and len(flows[3]) == 0
        ):
            print("    ↳ flows[3] = placeholder/other (non-empty)")
        else:
            print("    ↳ flows[3] = placeholder (empty/unused)")
        if isinstance(flows, (list, tuple)) and len(flows) > 4 and isinstance(flows[4], np.ndarray):
            raw = flows[4]
            print(f"    ↳ flows[4] = dP_raw (dy, dx, cellprob_float32) :: shape={raw.shape} dtype={raw.dtype}")
            if raw.ndim == 3 and raw.shape[0] == 3:
                print("       components: [0]=dy (vertical), [1]=dx (horizontal), [2]=cellprob_float32  ✅")
            else:
                print("       WARNING: expected first dim = 3 for (dy, dx, prob)")
        else:
            print("    ↳ flows[4] = dP_raw (missing)")

        # Quick value stats for key maps (optional, safe for large arrays)
        try:
            if isinstance(flows, (list, tuple)) and len(flows) > 1 and isinstance(flows[1], np.ndarray):
                cp = flows[1]
                cp_u8 = cp[0] if cp.ndim == 3 and cp.shape[0] == 1 else cp
                print(
                    f"\n  cellprob_u8 stats: min={cp_u8.min()} max={cp_u8.max()} mean={cp_u8.mean():.2f}"
                )
            if isinstance(flows, (list, tuple)) and len(flows) > 4 and isinstance(flows[4], np.ndarray):
                dp = flows[4]
                if dp.ndim == 3 and dp.shape[0] == 3:
                    dy, dx, cp32 = dp[0], dp[1], dp[2]
                    print(
                        f"  dP_raw stats: dy(mean={dy.mean():.4f}), dx(mean={dx.mean():.4f}), prob32(min={cp32.min():.4f}, max={cp32.max():.4f})"
                    )
        except Exception as e:
            print(f"  [stats] skipped due to: {e}")
    else:
        print("  MISSING or not a list")

    # ----- final sanity: spatial agreement -----
    print("\n[spatial sanity]")
    def hw(x):
        return getattr(x, "shape", (None, None))[-2:]
    ok = True
    if isinstance(masks, np.ndarray):
        H, W = masks.shape
        if isinstance(outlines, np.ndarray) and outlines.shape != masks.shape:
            ok = False
            print(f"  ⚠️ outlines shape {outlines.shape} != masks shape {masks.shape}")
        if isinstance(flows, (list, tuple)):
            # Check selected slots that should match HxW
            for idx in (0, 1, 2):
                if isinstance(flows, (list, tuple)) and len(flows) > idx and isinstance(flows[idx], np.ndarray):
                    shp = flows[idx].shape
                    # handle leading batch/channel dims
                    if idx in (0, 2) and not (len(shp) == 4 and shp[-3:] == (H, W, 3)):
                        print(f"  ⚠️ flows[{idx}] expected (1,{H},{W},3); got {shp}")
                        ok = False
                    if idx == 1 and not (len(shp) >= 2 and shp[-2:] == (H, W)):
                        print(f"  ⚠️ flows[1] expected (*,{H},{W}); got {shp}")
                        ok = False
            if len(flows) > 4 and isinstance(flows[4], np.ndarray):
                shp4 = flows[4].shape
                if not (len(shp4) == 3 and shp4[0] == 3 and shp4[1:] == (H, W)):
                    print(f"  ⚠️ flows[4] expected (3,{H},{W}); got {shp4}")
                    ok = False
    print("  spatial shapes consistent  ✅" if ok else "  spatial shapes have mismatches  ⚠️")

    print("\nDone.\n", flush=True)


if __name__ == "__main__":
    main()