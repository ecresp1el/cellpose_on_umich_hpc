#!/usr/bin/env python3
"""
scripts/run_experiment.py — Stage A+B entrypoint
------------------------------------------------
Top-level CLI wrapper for the Cellpose3 Whole-Organoid pipeline.
This script is intentionally thin — it just:
  1. Parses command-line arguments,
  2. Loads a validated Config object,
  3. Instantiates WholeOrganoidExperiment,
  4. Executes the requested stage:

     - Stage A: `prepare`
     - Stage B (resume mode): `train`  (expects an existing prepared run_dir)
     - Stage B (Option A):    `full-train`  (runs prepare() then run_training() in one call)

Usage examples
--------------
# Prepare a new run (Stage A)
python scripts/run_experiment.py \
    --config configs/cp3_v001.yaml \
    --mode prepare

# Prepare + Train in one job (recommended Option A for SLURM)
python scripts/run_experiment.py \
    --config configs/cp3_v001.yaml \
    --mode full-train

# Train only, resuming into an existing run directory (must contain cfg/)
python scripts/run_experiment.py \
    --config configs/cp3_v001.yaml \
    --mode train --run_dir /nfs/turbo/.../results/cp3_v001/run_YYYY-mm-dd_HHMMSS

Notes
-----
* `full-train` guarantees every training run has a fresh config/env snapshot
  in the same run_dir before training starts.
* `train` requires an existing run_dir that already contains cfg/config_snapshot.yaml.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

# -------------------------------------------------------------
# Ensure repo root is on the import path
# -------------------------------------------------------------
# This allows imports like `from cp_core.config_store import ConfigStore`
# to work when the script is executed directly, rather than as a module.
# e.g., so we can run "python scripts/run_experiment.py" from anywhere.
sys.path.append(str(Path(__file__).resolve().parents[1]))

# -------------------------------------------------------------
# Internal imports — Stage A modules only
# -------------------------------------------------------------
from cp_core.config_store import ConfigStore
from cp_core.experiment import WholeOrganoidExperiment

# -------------------------------------------------------------
# Optional pre-train hook: prefer single-channel “squished” train images
# -------------------------------------------------------------
# -------------------------------------------------------------
# Optional pre-train hook: prefer single-channel “squished” train images
# -------------------------------------------------------------
from pathlib import Path

def prefer_squished_train_images(cfg) -> None:
    """
    If a single-channel 'squished' TRAIN images folder exists (flat, e.g. images_squish_max),
    repoint cfg.paths.data_images_train to it. Labels remain unchanged.
    Handles both dict-based and object-based Configs.
    Prints counts and the exact replacement performed. No resizing, only channels→1.
    """

    # ---- helpers to access cfg as dict or object safely ----
    def _get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _set(obj, key, value):
        if isinstance(obj, dict):
            obj[key] = value
            return True
        if hasattr(obj, key):
            setattr(obj, key, value)
            return True
        return False

    # top-level blocks
    paths  = _get(cfg, "paths")
    labels = _get(cfg, "labels")

    if paths is None:
        print("[squish] cfg.paths not found; no replacement")
        return

    # read current train paths
    train_imgs = Path(_get(paths, "data_images_train", "")) if _get(paths, "data_images_train", "") else None
    train_lbls = Path(_get(paths, "data_labels_train", "")) if _get(paths, "data_labels_train", "") else None

    if not train_imgs or not train_imgs.exists():
        print(f"[squish] TRAIN images dir not found: {train_imgs}  (no replacement)")
        return
    if not train_lbls or not train_lbls.exists():
        print(f"[squish] TRAIN labels dir not found: {train_lbls}  (no replacement)")
        return

    # candidate squished subdirs next to the current train images dir
    prefer = _get(_get(cfg, "preprocess", {}), "squish", {}) or {}
    prefer_subdir = prefer.get("out_subdir")
    candidates = []
    if prefer_subdir:
        candidates.append(train_imgs.parent / prefer_subdir)
    candidates += [
        train_imgs.parent / "images_squish_max",
        train_imgs.parent / "images_squish_mean",
        train_imgs.parent / "images_squish_sum",
    ]

    pick = None
    for cand in candidates:
        if cand.exists():
            n = len(list(cand.glob("*.tif"))) + len(list(cand.glob("*.tiff")))
            if n > 0:
                pick = cand
                break

    if pick is None:
        print("[squish] No squished TRAIN folder found next to", train_imgs,
              "\n          looked for:", ", ".join(str(c) for c in candidates))
        return

    # count pairs respecting YAML labels.mask_filter
    mask_suffix = None
    if isinstance(labels, dict):
        mask_suffix = labels.get("mask_filter")
    else:
        mask_suffix = getattr(labels, "mask_filter", None)

    def _has_label(stem: str) -> bool:
        cands = []
        if mask_suffix:
            cands.append(train_lbls / f"{stem}{mask_suffix}")
        cands += [
            train_lbls / f"{stem}_masks.tif",
            train_lbls / f"{stem}.png",
            train_lbls / f"{stem}.npy",
            train_lbls / f"{stem}_seg.npy",
        ]
        return any(p.exists() for p in cands)

    imgs = sorted(list(pick.glob("*.tif")) + list(pick.glob("*.tiff")))
    n_images = len(imgs)
    n_paired = sum(1 for p in imgs if _has_label(p.stem))

    print(f"[squish] FOUND squished TRAIN folder: {pick}")
    print(f"[squish] images={n_images}  paired_with_labels={n_paired}  labels_dir={train_lbls}")

    # replace path in cfg.paths (dict or object)
    if isinstance(paths, dict):
        prev = paths.get("data_images_train")
        paths["data_images_train"] = str(pick)
        print(f"[squish] REPLACED train images dir:\n          {prev}\n          → {pick}")
    else:
        prev = getattr(paths, "data_images_train", None)
        setattr(paths, "data_images_train", str(pick))
        print(f"[squish] REPLACED train images dir:\n          {prev}\n          → {pick}")

    # enforce single-channel usage for squished data
    train_cfg = _get(cfg, "train") or {}
    eval_cfg  = _get(cfg, "eval")  or {}
    changed_tc = (train_cfg.get("channels") if isinstance(train_cfg, dict) else getattr(train_cfg, "channels", None)) != [0,0]
    changed_ec = (eval_cfg.get("channels")  if isinstance(eval_cfg,  dict) else getattr(eval_cfg,  "channels",  None)) != [0,0]

    if isinstance(train_cfg, dict):
        train_cfg["channels"] = [0,0]
        _set(cfg, "train", train_cfg)
    else:
        setattr(train_cfg, "channels", [0,0])
    if isinstance(eval_cfg, dict):
        eval_cfg["channels"] = [0,0]
        _set(cfg, "eval", eval_cfg)
    else:
        setattr(eval_cfg, "channels", [0,0])

    if changed_tc:
        print("[squish] Set train.channels → [0, 0] (single-channel).")
    if changed_ec:
        print("[squish] Set eval.channels  → [0, 0] (single-channel).")


def parse_args():
    """
    Parse CLI arguments for the pipeline entrypoint.

    Returns
    -------
    argparse.Namespace
        Fields:
        - config : str  -> absolute or relative path to YAML config.
        - mode   : str  -> workflow stage ("prepare" | "train" | "full-train").
        - run_dir: str? -> required when mode == "train" (resume into prepared run).

    Raises
    ------
    SystemExit
        If required args are missing or invalid.
    """
    p = argparse.ArgumentParser(
        description="Whole-Organoid Pipeline Entrypoint (Stage A+B)."
    )
    p.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file (version-controlled).",
    )
    p.add_argument(
        "--mode",
        choices=["prepare", "train", "full-train", "eval"],
        default="prepare",
        help="Workflow stage to execute.",
    )
    p.add_argument(
        "--split",
        default="valid",
        choices=["valid", "all"],
        help="Dataset split to evaluate (Stage C).",
    )
    p.add_argument(
        "--run_dir",
        default=None,
        help="Existing prepared run directory (required for --mode train).",
    )
    return p.parse_args()

def main():
    """
    Main CLI routine — orchestrates Stage A and Stage B.

    Flow
    ----
    1. Parse CLI args and load YAML config using ConfigStore.
    2. Instantiate WholeOrganoidExperiment with the validated config.
    3. Dispatch by mode:
         - prepare:   create run_dir, snapshot config/env, verify dataset.        (Stage A)
         - full-train:prepare() then run_training() in one call.                  (Stage A→B, Option A)
         - train:     resume into an existing run_dir (must contain cfg/).        (Stage B resume)
    4. Print the run directory path at completion.

    Writes
    ------
    - Stage A: results/<model>/run_<ts>/cfg/*
    - Stage B: results/<model>/run_<ts>/train/*

    Notes
    -----
    * Errors propagate so SLURM captures them in .err.
    * Printing is minimal — structured logs live in the run directory.
    """
    
    a = parse_args()

    # Load validated configuration (raises if YAML missing or malformed)
    cfg = ConfigStore.load_from_yaml(a.config)
    
    if a.mode in ("prepare","full-train"):
        prefer_squished_train_images(cfg)
    elif a.mode == "train":
        print("[squish] NOTE: --mode train uses run_dir snapshot; in-memory swap applies to prepare/full-train.")

    # Initialize experiment wrapper with config (creates a *new* run_dir on prepare/full-train)
    exp = WholeOrganoidExperiment(cfg)

    if a.mode == "prepare":
        exp.prepare()
        print(f"[Stage A] Prepared run directory: {exp.get_run_dir()}")
        return

    if a.mode == "full-train":
        # Option A (recommended): always prepare a fresh run, then train into it
        exp.run_full_train()
        print(f"[Stage B] Trained specialist model in run directory: {exp.get_run_dir()}")
        return

    if a.mode == "train":
        # Resume mode: require an existing run_dir with cfg/config_snapshot.yaml
        if a.run_dir is None:
            raise SystemExit("--mode train requires --run_dir (or use --mode full-train).")
        rd = Path(a.run_dir)
        cfg_snapshot = rd / "cfg" / "config_snapshot.yaml"
        if not (rd.exists() and cfg_snapshot.exists()):
            raise SystemExit(f"--run_dir is not a prepared run directory (missing {cfg_snapshot}).")

        # Rebind the experiment to the supplied run_dir (no new prepare)
        exp.run_dir = rd
        exp.cfg_dir = rd / "cfg"
        exp.logger = exp.logger.__class__(exp.run_dir)  # rebind logger to new run_dir

        exp.run_training()
        print(f"[Stage B] Trained specialist model in run directory: {exp.get_run_dir()}")
        return
    
    if a.mode == "eval":
        # If a specific run_dir is given, bind to that prepared run (with train/ artifacts)
        if a.run_dir is not None:
            rd = Path(a.run_dir)
            cfg_snapshot = rd / "cfg" / "config_snapshot.yaml"
            if not (rd.exists() and cfg_snapshot.exists()):
                raise SystemExit(f"--run_dir is not a prepared run directory (missing {cfg_snapshot}).")
            exp.run_dir = rd
            exp.cfg_dir = rd / "cfg"
            exp.logger = exp.logger.__class__(exp.run_dir)
        # Now perform evaluation in exp.run_dir
        exp.run_evaluation(split=a.split)
        print(f"[Stage C] Evaluation complete in run directory: {exp.get_run_dir()}")
        return

# -------------------------------------------------------------
# Standard Python entrypoint guard
# -------------------------------------------------------------
# Ensures the script executes main() only when called directly
# (not when imported as a module inside notebooks or tests).
if __name__ == "__main__":
    main()
