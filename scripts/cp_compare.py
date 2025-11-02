#!/usr/bin/env python3
"""
Simple wrapper to run Cellpose from Python with clear prints.
- No metrics/CSV. It only runs segmentation and prints environment info.
- Designed to be called from a SLURM script.
"""
import argparse
import os
import sys
import subprocess
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="Run Cellpose (CP3/CP4) with identical parameters to segment whole organoids.")
    p.add_argument("--img_dir", required=True, help="Folder with .tif images (e.g., /nfs/turbo/.../images/WT)")
    p.add_argument("--out_dir", required=True, help="Output folder (e.g., /nfs/turbo/.../results/WT/cp4)")
    p.add_argument("--diameter", type=float, default=1200.0, help="Expected object diameter in pixels (≈1200)")
    p.add_argument("--flow_threshold", type=float, default=0.7, help="Higher = more conservative boundaries (0.6–0.8)")
    p.add_argument("--min_size", type=int, default=100000, help="Min object size in pixels to keep (filters debris)")
    p.add_argument("--chan", type=int, default=0, help="Primary signal channel (0 for grayscale)")
    p.add_argument("--pretrained_model", default="cyto3", help="Cellpose pretrained model (default: cyto3)")
    args = p.parse_args()

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists():
        print(f"[ERROR] img_dir does not exist: {img_dir}", file=sys.stderr)
        sys.exit(1)

    # Minimal runtime info
    try:
        import torch  # type: ignore
        torch_info = {
            "torch": getattr(torch, "__version__", "?"),
            "cuda_available": torch.cuda.is_available(),
            "torch_cuda_version": getattr(torch.version, "cuda", None),
        }
        if torch.cuda.is_available():
            torch_info["device_0"] = torch.cuda.get_device_name(0)
    except Exception as e:
        torch_info = {"torch": "not importable", "error": str(e)}

    print("=== RUNTIME INFO ===")
    print(f"Host          : {os.uname().nodename}")
    print(f"Python        : {sys.version.split()[0]}")
    print(f"Torch info    : {torch_info}")
    print(f"Images        : {img_dir}  (filter: {args.img_filter})")
    print(f"Output        : {out_dir}")
    print(f"Diameter      : {args.diameter}")
    print(f"FlowThreshold : {args.flow_threshold}")
    print(f"MinSize(px)   : {args.min_size}")
    print(f"Channel       : {args.chan}")
    print(f"Model         : {args.pretrained_model}")
    print("====================")

    # Build Cellpose CLI (identical for CP3 and CP4 to compare fairly)
    cmd = [
        "cellpose",
        "--dir", str(img_dir),
        "--pretrained_model", args.pretrained_model,
        "--diameter", str(args.diameter),
        "--flow_threshold", str(args.flow_threshold),
        "--min_size", str(args.min_size),
        "--chan", str(args.chan),
        "--save_tif", "--save_png", 
        "--use_gpu", "--verbose",
        "--savedir", str(out_dir),   # <- correct flag for output directory
    ]

    print("[CMD]", " ".join(cmd))
    sys.stdout.flush()

    # Run and stream output
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    if proc.returncode != 0:
        print(f"[ERROR] Cellpose exited with code {proc.returncode}", file=sys.stderr)
        sys.exit(proc.returncode)

    print("=== DONE ===")

if __name__ == "__main__":
    main()
