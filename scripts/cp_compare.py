#!/usr/bin/env python3
"""
cp_compare.py — Run Cellpose from CLI with clear prints (no CSV), for SLURM wrappers.
Moves all output files into the correct results/<COND>/<ENV_NAME> folder.

See header comments in the SLURM file for how this integrates with your repo.
"""
import argparse
import os
import sys
import json
import subprocess
import shutil
import glob
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--diameter", type=float, default=1200.0)
    p.add_argument("--flow_threshold", type=float, default=0.7)
    p.add_argument("--min_size", type=int, default=100000)
    p.add_argument("--chan", type=int, default=0)
    p.add_argument("--model", default="cpsam", help="Pretrained model (cpsam for CP4, cyto3 for CP3).")
    return p.parse_args()

def torch_info():
    try:
        import torch
        d = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
             "torch_cuda_version": getattr(torch.version, "cuda", None)}
        if torch.cuda.is_available(): d["device_0"] = torch.cuda.get_device_name(0)
        return d
    except Exception as e:
        return {"torch": "not importable", "error": str(e)}

def main():
    args = parse_args()
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not img_dir.exists():
        print(f"[ERROR] img_dir does not exist: {img_dir}", file=sys.stderr)
        sys.exit(2)

    print("=== RUNTIME INFO ===")
    print(f"Host          : {os.uname().nodename}")
    print(f"Python        : {sys.version.split()[0]}")
    print(f"Torch info    : {json.dumps(torch_info())}")
    print(f"Images        : {img_dir}")
    print(f"Output        : {out_dir}")
    print(f"Diameter      : {args.diameter}")
    print(f"FlowThreshold : {args.flow_threshold}")
    print(f"MinSize(px)   : {args.min_size}")
    print(f"Channel       : {args.chan}")
    print(f"Model         : {args.model}")
    print("====================", flush=True)

    cmd = [
        "cellpose",
        "--dir", str(img_dir),
        "--pretrained_model", args.model,
        "--diameter", str(args.diameter),
        "--flow_threshold", str(args.flow_threshold),
        "--min_size", str(args.min_size),
        "--chan", str(args.chan),
        "--save_tif", "--save_png",
        "--use_gpu", "--verbose",
        "--savedir", str(out_dir),
    ]

    print("[CMD]", " ".join(cmd), flush=True)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    if proc.returncode != 0:
        print(f"[ERROR] Cellpose exited with code {proc.returncode}", file=sys.stderr)
        sys.exit(proc.returncode)
    
    # --- Post-run gather: move any stray outputs from image folder into results folder ---
    patterns = [
        "*_seg.npy",
        "*_flows.npy",
        "*_cp_masks.tif",
        "*_cp_outlines.png",
    ]
    moved = 0
    for pattern in patterns:
        for f in img_dir.glob(pattern):
            target = out_dir / f.name
            if target.exists():
                stem, suf = f.stem, f.suffix
                i = 1
                while True:
                    new_name = out_dir / f"{stem}__dup{i}{suf}"
                    if not new_name.exists():
                        target = new_name
                        break
                    i += 1
            try:
                shutil.move(str(f), str(target))
                moved += 1
            except Exception as e:
                print(f"[WARN] could not move {f} -> {target}: {e}", flush=True)
    print(f"[INFO] gathered {moved} file(s) into {out_dir}", flush=True)

    print("=== DONE ===", flush=True)

if __name__ == "__main__":
    main()
