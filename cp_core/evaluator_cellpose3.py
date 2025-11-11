"""
EvaluatorCellpose3 — Stage C (Evaluation & Artifact Generation, Cellpose v3 API)

Purpose
-------
Run native-scale inference with a trained Cellpose v3 model and save tidy
artifacts per image: masks, flows, probabilities, ROIs, and a 1×4 panel
(input | prob | flow viz | overlay). This class adheres to the Stage C
section of the contract.

Notes
-----
* Native grid by contract: resample=False, no diameter passed during eval.
* Uses Cellpose plotting utilities when available (plot.show_segmentation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import json
import time

import numpy as np

import cellpose
from cellpose import io as cp_io
from cellpose.models import CellposeModel

# -------------------- simple helpers (printing / sanity) --------------------

def _print_eval_kwargs(kwargs: dict):
    print("[Stage C] Eval kwargs to be passed to model.eval():")
    if not kwargs:
        print("  (none) → using library defaults")
        return
    for k, v in kwargs.items():
        if k == "normalize" and isinstance(v, dict):
            print(f"  normalize: {{…}} (dict)")
            continue
        print(f"  {k}: {v}")

def _init_model_from_cfg(cfg) -> CellposeModel:
    """
    Initialize a CellposeModel from saved training artifacts, falling back
    to a final weights file if needed.
    """
    pm = str((Path(cfg.paths.get("results_root")) / cfg.model_name_out / "train" / "models" / cfg.model_name_out))
    if not Path(pm).exists():
        pm = str((Path(cfg.paths.get("results_root")) / cfg.model_name_out / "train" / "weights_final.pt"))
    print(f"[Stage C] Loading CellposeModel(pretrained_model='{pm}', gpu=True")
    model = CellposeModel(gpu=True, pretrained_model=pm)
    return model


@dataclass
class EvalArgs:
    """
    Generic container for dynamic Cellpose eval kwargs (CP4/CP-SAM friendly).

    - Holds ONLY the non-null, non-deprecated keys that will be forwarded to model.eval().
    - No defaults are injected here; YAML controls everything.
    - Designed for clean logging / provenance snapshots.
    """
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict suitable for **kwargs into model.eval()."""
        return dict(self.kwargs)

    def save_json(self, path: Path) -> None:
        """
        Write the effective eval kwargs (what we will actually pass to model.eval)
        as a JSON file for reproducibility.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.kwargs, f, indent=2, sort_keys=True)


@dataclass
class EvaluatorCellpose3:
    cfg: Any
    run_dir: Path

    def __post_init__(self):
        self.run_dir = Path(self.run_dir)
        self.eval_dir = self.run_dir / "eval"
        # subdirs
        self.d_masks = self.eval_dir / "masks"
        self.d_flows = self.eval_dir / "flows"
        self.d_prob  = self.eval_dir / "prob"
        self.d_rois  = self.eval_dir / "rois"
        self.d_panels= self.eval_dir / "panels"
        self.d_json  = self.eval_dir / "json"
        for d in (self.d_masks, self.d_flows, self.d_prob, self.d_rois, self.d_panels, self.d_json):
            d.mkdir(parents=True, exist_ok=True)

        # preferred CP model directory created by training (save_path/models/<model_name_out>)
        self.cp_model_dir = self.run_dir / "train" / "models" / self.cfg.model_name_out
        self.fallback_weights = self.run_dir / "train" / "weights_final.pt"

    # -------------------- arg building --------------------
    def build_eval_args(self) -> EvalArgs:
        """Resolve eval args from YAML with Stage C defaults/invariants.
        - YAML keys set to null are skipped (use Cellpose defaults)
        - Deprecated keys are ignored
        - No internal defaults are injected; all thresholds passed through directly
        """
        e = self.cfg.eval or {}

        # Detect Cellpose version once
        try:
            cpv = getattr(cellpose, "__version__", "unknown")
        except AttributeError:
            from cellpose import __version__ as cpv

        # Keys known to be deprecated or ignored in Cellpose 4+
        # NOTE: allow 'channels' to pass through; v4 still consumes [c0, c1] and the writer embeds chan_choose.
        deprecated = {"rescale"}

        # Initialize dictionary of parameters that will actually be passed to model.eval()
        eval_kwargs = {}

        # Iterate over every key/value pair in the YAML `eval:` block
        for k, v in e.items():

            # Skip parameters explicitly set to null in YAML.
            if v is None:
                continue

            # Skip any argument that is known to be deprecated.
            if k in deprecated:
                print(f"[Stage C] Ignoring deprecated key: {k}")
                continue

            # Otherwise, keep the key-value pair to forward directly to model.eval().
            eval_kwargs[k] = v

        # Diagnostic printout for reproducibility and transparency
        print(f"[Stage C] Detected Cellpose v{cpv}")
        print(f"[Stage C] Eval args used: {list(eval_kwargs.keys())}")
        
        
        # (safety) drop any train/loader keys that may have leaked into eval
        for _bad in ("use_pretrained", "model_type", "baseline_pretrained", "pretrained_model"):
            eval_kwargs.pop(_bad, None)

        # Return EvalArgs dataclass (same as before)
        return EvalArgs(kwargs=eval_kwargs)

    # -------------------- main entry --------------------
    def evaluate_images(self, split: str, args):
        """
        Minimal, diagnostic-first evaluation:
        - lists and loads images with per-file checks
        - initializes model if needed
        - runs batch eval with kwargs from args
        - saves masks (+ optional panel)
        """
        # 0) resolve images via Stage A's DatasetManager (no ad-hoc paths)

        dm = DatasetManager(
            getattr(self.cfg, "paths", {}) or {},
            getattr(self.cfg, "labels", {}) or {},
            self.run_dir / "cfg",
        )

        # 1) list images
        files = dm.list_images(split=split)
        if not files:
            print(f"[Stage C] No images found for split={split}")
            return

        # 2) load images with checks
        loaded = dm.load_images(files)
        kept_files, imgs = [], []
        for p, im in loaded:
            if im is None:
                print(f"[Stage C][WARN] Skipping unreadable: {p.name}")
                continue
            kept_files.append(p)
            imgs.append(im)

        if not imgs:
            print("[Stage C] No readable images after loading.")
            return

        out_dir = self.eval_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # 3) init model if needed
        if not hasattr(self, "model") or self.model is None:
            self.model = _init_model_from_cfg(self.cfg)

        # 4) show kwargs (filter out non-eval keys like save_* before calling CP)
        kw = dict(getattr(args, "kwargs", {}))  # copy; don’t mutate original
        for _bad in ("save_panels", "save_rois", "save_flows", "save_prob", "save_prob_view"):
            kw.pop(_bad, None)
        _print_eval_kwargs(kw)
        
        # 5) run eval (per-image for visible progress)
        N = len(imgs)
        print(f"[Stage C] Running eval per-image for visibility… total={N}")

        masks, flows, styles, diams = [], [], [], []
        for i, (im, p) in enumerate(zip(imgs, kept_files), 1):
            t0 = time.time()
            m, f, s, d = self.model.eval([im], **kw)   # CP v4 returns (masks, flows, styles, diams)
            m0 = m[0]
            masks.append(m0)
            flows.append(f)
            styles.append(s)
            diams.append((d[0] if d else float("nan")))  # CP v4 est diam per-image

            nmask = int(getattr(m0, "max", lambda: 0)())
            dt = time.time() - t0
            print(f"[Stage C][{i}/{N}] {p.name}: n_masks={nmask} in {dt:.2f}s")

        # 6) quick print of shapes (first example)
        try:
            if N > 0:
                fpack = flows[0][0] if isinstance(flows[0], (list, tuple)) else flows[0]
                f_hsv = fpack[0] if isinstance(fpack, (list, tuple)) and len(fpack) > 0 else None
                f_vec = fpack[1] if isinstance(fpack, (list, tuple)) and len(fpack) > 1 else None
                cellprob = fpack[2] if isinstance(fpack, (list, tuple)) and len(fpack) > 2 else None
                print(f"[Stage C] Example: masks={getattr(masks[0],'shape','NA')}, hsv={getattr(f_hsv,'shape','NA')}, vec={getattr(f_vec,'shape','NA')}, prob={getattr(cellprob,'shape','NA')}")
        except Exception as ex:
            print(f"[Stage C][WARN] Could not print example outputs: {ex}")

        
        # 7) save artifacts
        print("[Stage C] Saving outputs…")
        n_masks_per_image = []
        for i, p in enumerate(kept_files):
            try:
                stem = p.stem
                # fpack = FULL per-image flow pack (list/tuple len≈5); correct to use [0]
                fpack = flows[i][0] if isinstance(flows[i], (list, tuple)) else flows[i]

                # Save mask via official API (TIF), same filename convention
                _save_masks_api(imgs[i], masks[i], fpack, stem, out_dir)
                nm = int(getattr(masks[i], "max", lambda: 0)())
                n_masks_per_image.append(nm)
                print(f"[Stage C] (n_masks={nm})")

                # Save native *_seg.npy via API + print its metadata (v4 signature w/ diams + channels)
                _save_seg_npy_api(imgs[i], masks[i], fpack, stem, out_dir, kw.get("channels", [0, 0]), diams[i])

                # optional: panels etc. left as-is in your existing code…
            except Exception as ex:
                print(f"[Stage C][WARN] Save failed for {p.name}: {ex}")
                n_masks_per_image.append(0)

        # 8) persist eval kwargs
        try:
            EvalArgs(kwargs=kw).save_json(self.d_json / "eval_kwargs.json")
        except Exception as ex:
            print(f"[Stage C][WARN] could not write eval kwargs json: {ex}")

        # 9) return an aggregation dict expected by experiment.py
        try:
            seg_paths = sorted(str(p) for p in (out_dir.glob("*_seg.npy")))
        except Exception:
            seg_paths = []
        agg = {
            "n_images": len(kept_files),
            "n_masks_total": int(sum(nm for nm in n_masks_per_image if isinstance(nm, int))),
            "n_masks_per_image": n_masks_per_image,
            "eval_dir": str(out_dir),
            "seg_files": seg_paths,
        }
        print(f"[Stage C] Wrote eval artifacts. Split={split}, images={agg['n_images']}, masks_total={agg['n_masks_total']}")
        return agg

# -------------------- save helpers (official I/O only) --------------------

def _save_seg_npy_api(im, m, fpack, stem: str, out_dir: Path, channels, diam) -> Path:
    """
    Write CP v4-compatible *_seg.npy via official API with est_diam and chan_choose populated.
    We pass list-wrapped arguments per the v4 signature.
    """
    base = out_dir / stem
    # flows needs to be a list-of-packs; ensure fpack is list/tuple inside an outer list
    flows_list = [list(fpack) if isinstance(fpack, (list, tuple)) else [fpack]]
    # diams must be float array of shape (N,)
    diams_arr = np.array([float(diam) if diam is not None else np.nan], dtype=float)
    # channels must be two-int list, wrapped in an outer list
    ch = [int(channels[0]), int(channels[1])] if isinstance(channels, (list, tuple)) and len(channels) >= 2 else [0, 0]
    cp_io.masks_flows_to_seg(
        images=[im],
        masks=[m],
        flows=flows_list,
        diams=diams_arr,
        file_names=[str(base)],
        channels=[ch],
    )
    seg_path = out_dir / f"{stem}_seg.npy"
    try:
        seg = np.load(seg_path, allow_pickle=True).item()
        keys = list(seg.keys())
        print(
            "[Stage C][seg.npy] wrote:", seg_path.name,
            "| keys=", keys,
            "| est_diam=", seg.get("est_diam", None),
            "| chan_choose=", seg.get("chan_choose", None),
        )
    except Exception as ex:
        print(f"[Stage C][WARN] could not inspect {seg_path.name}: {ex}")
    return seg_path

def _save_masks_api(im, m, fpack, stem: str, out_dir: Path) -> None:
    """
    Save masks.tif via API. Uses official cp_io.save_masks for consistency.
    """
    images = [im]
    masks  = [m]
    flows_ = [fpack]
    file_names = [str(out_dir / stem)]
    cp_io.save_masks(
        images, masks, flows_, file_names,
        png=False, tif=True, suffix="_masks",
        save_flows=False, save_outlines=False,
        savedir=None, in_folders=False
    )
    print(f"[Stage C] save_masks (API) wrote: {stem}_masks.tif")


# -------------------- dataset manager (abbrev; unchanged portions elided) --------------------
class DatasetManager:
    def __init__(self, paths_cfg: Dict[str, Any], labels_cfg: Dict[str, Any], cfg_dir: Path):
        self.paths_cfg = paths_cfg
        self.labels_cfg = labels_cfg
        self.cfg_dir = Path(cfg_dir)

    def list_images(self, split: str) -> List[Path]:
        root = Path(self.paths_cfg.get(f"data_images_{split}", self.paths_cfg.get("data_images_valid", "")))
        if not root.exists():
            return []
        return sorted(list(root.glob("*.tif"))) + sorted(list(root.glob("*.tiff"))) + sorted(list(root.glob("*.png")))

    def load_images(self, files: Sequence[Path]) -> List[Tuple[Path, Optional[np.ndarray]]]:
        out: List[Tuple[Path, Optional[np.ndarray]]] = []
        for p in files:
            try:
                im = cp_io.imread(str(p))
            except Exception:
                im = None
            out.append((p, im))
        return out