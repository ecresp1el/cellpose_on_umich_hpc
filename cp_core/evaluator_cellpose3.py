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
from typing import Dict, Any, List, Tuple, Optional
import json
import csv

from matplotlib import pyplot as plt
import numpy as np

from cellpose import models, io as cp_io, plot

# Reuse standardized I/O + layout helpers (single source of truth)
from cp_core.helper_functions.dl_helper import ensure_hwc_1to5, has_label, read_label
from cp_core.dataset import DatasetManager

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


class EvaluatorCellpose3:
    """Stage C evaluator using Cellpose v3 native API."""

    def __init__(self, cfg, run_dir: Path):
        """
        Args
        ----
        cfg : Config
            Validated configuration.
        run_dir : Path
            timestamped run directory (results/<model>/run_<ts>/)
        """
        self.cfg = cfg
        self.run_dir = run_dir
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
        - Deprecated keys (channels, rescale) are always ignored
        - No internal defaults are injected; all thresholds passed through directly
        """
        e = self.cfg.eval or {}

        # Detect Cellpose version once
        try:
            cpv = getattr(models, "__version__", "unknown")
        except AttributeError:
            from cellpose import __version__ as cpv

        # Keys known to be deprecated or ignored in Cellpose 4+ (e.g., SAM backend)
        # These arguments should never be passed to model.eval(), as doing so
        # triggers deprecation warnings or silently has no effect.
        deprecated = {"channels", "rescale"}

        # Initialize dictionary of parameters that will actually be passed to model.eval()
        eval_kwargs = {}

        # Iterate over every key/value pair in the YAML `eval:` block
        for k, v in e.items():

            # Skip parameters explicitly set to null in YAML.
            # A null means "use the model's built-in default" (do not override).
            if v is None:
                continue

            # Skip any argument that is known to be deprecated.
            # This prevents deprecation warnings like:
            # "channels deprecated in v4.0.1+. If data contain more than 3 channels, only the first 3 will be used"
            if k in deprecated:
                print(f"[Stage C] Ignoring deprecated key: {k}")
                continue

            # Otherwise, keep the key-value pair to forward directly to model.eval().
            eval_kwargs[k] = v

        # Diagnostic printout for reproducibility and transparency
        # Shows the detected Cellpose version and which parameters will be passed through.
        print(f"[Stage C] Detected Cellpose v{cpv}")
        print(f"[Stage C] Eval args used: {list(eval_kwargs.keys())}")
        
        
        # (safety) drop any train/loader keys that may have leaked into eval
        for _bad in ("use_pretrained", "model_type", "baseline_pretrained", "pretrained_model"):
            eval_kwargs.pop(_bad, None)

        # Return EvalArgs dataclass (same as before)
        return EvalArgs(kwargs=eval_kwargs)


    # -------------------- model loading --------------------
    def load_model(self):
        """Load CPSAM model for evaluation.

        - If a fine-tuned model exists in this run_dir (Stage B output),
          load that checkpoint.
        - Otherwise, use the stock pretrained CPSAM model.

        Only one fine-tuning round is ever supported.
        """
        assert models is not None, "cellpose.models not available."

        # Detect GPU
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except Exception:
            use_gpu = False

        # Check for fine-tuned model directory or checkpoint
        have_cp_dir = self.cp_model_dir.exists()
        have_fallback = self.fallback_weights.exists()

        if have_cp_dir:
            print(f"[Stage C] model_source = fine-tuned CPSAM (from {self.cp_model_dir})")
            return models.CellposeModel(gpu=use_gpu, pretrained_model=str(self.cp_model_dir))

        if have_fallback:
            print(f"[Stage C] model_source = fine-tuned CPSAM (state_dict): {self.fallback_weights}")
            import torch
            m = models.CellposeModel(gpu=use_gpu, pretrained_model=None)
            sd = torch.load(str(self.fallback_weights), map_location="cpu")
            m.net.load_state_dict(sd, strict=False)
            return m

        # Otherwise fall back to pretrained CPSAM
        print("[Stage C] model_source = stock CPSAM (no fine-tuned weights found)")
        return models.CellposeModel(gpu=use_gpu, pretrained_model="cpsam")
        
    # -------------------- evaluation --------------------
    def evaluate_images(self, split: str, args):
        """
        Minimal, diagnostic-first evaluation:
        - lists and loads images with per-file checks
        - initializes model if needed
        - runs batch eval with kwargs from args
        - saves masks (+ optional panel)
        """
        # 0) resolve images via Stage A DatasetManager (no ad-hoc path guessing)
        dm = DatasetManager(self.cfg)
        files = dm.list_images(split=split)   # supports 'valid' and 'all' exactly like Stage A

        # eval output directory
        out_dir = self.run_dir / "eval" / split
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Stage C] Split='{split}'  n_files={len(files)}  out_dir={out_dir}")

        if not files:
            print("[Stage C][ERROR] No images found by DatasetManager.list_images(); check dataset layout or split.")
            return {"n_images": 0}

        # 2) load images (no channel slicing here)
        print("[Stage C] Loading images (no channel manipulation in this step)…")
        imgs, kept_files = _load_all_images(files)
        if not imgs:
            return {"n_images": 0}

        # 3) model init (lazy)
        if not hasattr(self, "model") or self.model is None:
            self.model = _init_model_from_cfg(self.cfg)

        # 4) show kwargs
        kw = getattr(args, "kwargs", {})  # EvalArgs.kwargs
        _print_eval_kwargs(kw)

        # 5) run eval (batch)
        print(f"[Stage C] Running batch eval on {len(imgs)} image(s)…")
        try:
            masks, flows, styles = self.model.eval(imgs, **kw)
        except Exception as ex:
            print(f"[Stage C][ERROR] model.eval failed: {ex}")
            # early exit with zero written
            return {"n_images": 0}

        # 6) validate outputs
        if not _validate_batch_outputs(masks, flows, styles, n_expected=len(imgs)):
            print("[Stage C][ERROR] Output validation failed; stopping to avoid writing inconsistent artifacts.")
            return {"n_images": 0}

        # 7) save artifacts
        print("[Stage C] Saving outputs…")
        n_images = 0
        for i, p in enumerate(kept_files):
            try:
                stem = p.stem

                # save mask
                mpath = out_dir / f"{stem}_masks.tif"
                cp_io.imsave(mpath, masks[i])
                print(f"[Stage C] wrote: {mpath.name} (n_masks={int(getattr(masks[i],'max',lambda:0)())})")

                # optional: 1x4 panel (guarded)
                if (self.cfg.eval or {}).get("save_panels", True):
                    try:
                        import matplotlib.pyplot as plt
                        fig = plt.figure(figsize=(12, 5))
                        # NOTE: use flows[i], not flows[0] (per-image)
                        plot.show_segmentation(fig, imgs[i], masks[i], flows[i])
                        fig.tight_layout()
                        fig.savefig(out_dir / f"{stem}_panel_1x4.png", dpi=150)
                        plt.close(fig)
                        print(f"[Stage C] wrote: {stem}_panel_1x4.png")
                    except Exception as ex:
                        print(f"[Stage C][WARN] panel failed for {p.name}: {ex}")

                n_images += 1

            except Exception as ex:
                print(f"[Stage C][ERROR] saving failed for {p.name}: {ex}")

        print(f"[Stage C] Done. images={n_images}")
        return {"n_images": n_images}

# -----------------------------
# DIAGNOSTIC HELPERS (no channel ops here)
# -----------------------------
def _list_images(img_dir: Path) -> List[Path]:
    print(f"[Stage C] Scanning dir: {img_dir}")
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])
    print(f"[Stage C] Found {len(files)} candidate image(s).")
    if len(files) == 0:
        print("[Stage C][ERROR] No images found. Check your split path and extensions.")
    else:
        print("[Stage C] First few files:", [f.name for f in files[:5]])
    return files

def _safe_read_image(p: Path) -> Optional[np.ndarray]:
    try:
        im = cp_io.imread(p)
        if im is None:
            print(f"[Stage C][WARN] imread returned None for: {p.name}")
            return None
        if not isinstance(im, np.ndarray):
            print(f"[Stage C][WARN] imread did not return ndarray for: {p.name} (type={type(im)})")
            return None
        if np.any(np.isnan(im)) or np.any(np.isinf(im)):
            print(f"[Stage C][WARN] NaN/Inf detected in image: {p.name}")
        # basic stats
        print(f"[Stage C] Loaded {p.name}: shape={im.shape}, dtype={im.dtype}, min={np.min(im)}, max={np.max(im)}")
        if im.ndim >= 3 and im.shape[-1] > 3:
            print(f"[Stage C][NOTE] {p.name} has >3 channels (C={im.shape[-1]}). CP-SAM only uses first 3. (We are NOT slicing here.)")
        return im
    except Exception as ex:
        print(f"[Stage C][ERROR] Failed to read {p.name}: {ex}")
        return None

def _load_all_images(files: List[Path]) -> Tuple[List[np.ndarray], List[Path]]:
    imgs, kept = [], []
    for p in files:
        im = _safe_read_image(p)
        if im is not None:
            imgs.append(im)
            kept.append(p)
    print(f"[Stage C] Loaded {len(imgs)}/{len(files)} image(s) successfully.")
    if len(imgs) == 0:
        print("[Stage C][ERROR] No images could be loaded. Aborting eval soon.")
    return imgs, kept

def _init_model_from_cfg(cfg) -> models.CellposeModel:
    # prefer YAML train.pretrained_model if provided, else cpsam
    pm = (getattr(cfg, "train", None) or {}).get("pretrained_model", "cpsam")
    # version print
    try:
        cpv = getattr(models, "__version__", "unknown")
    except AttributeError:
        from cellpose import __version__ as cpv
    print(f"[Stage C] Initializing Cellpose model (v{cpv}) with pretrained_model='{pm}', gpu=True")
    # init
    model = models.CellposeModel(gpu=True, pretrained_model=pm)
    return model

def _print_eval_kwargs(kwargs: dict):
    print("[Stage C] Eval kwargs to be passed to model.eval():")
    if not kwargs:
        print("  (none) → using library defaults")
        return
    for k, v in kwargs.items():
        if k == "normalize" and isinstance(v, dict):
            print(f"  {k}: dict keys={list(v.keys())}")
        else:
            print(f"  {k}: {v}")

def _validate_batch_outputs(masks, flows, styles, n_expected: int) -> bool:
    ok = True
    # length checks
    try:
        n_masks = len(masks)
    except Exception:
        print("[Stage C][ERROR] masks is not indexable. Got:", type(masks))
        return False
    if n_masks != n_expected:
        print(f"[Stage C][ERROR] Output count mismatch: masks={n_masks} vs images={n_expected}")
        ok = False
    try:
        n_flows = len(flows)
    except Exception:
        print("[Stage C][ERROR] flows is not indexable. Got:", type(flows))
        return False
    if n_flows != n_expected:
        print(f"[Stage C][ERROR] Output count mismatch: flows={n_flows} vs images={n_expected}")
        ok = False
    try:
        _ = styles  # styles can be array or list; don’t over-constrain
    except Exception as ex:
        print(f"[Stage C][WARN] styles access issue: {ex}")
    # sample prints
    try:
        print(f"[Stage C] Example mask shape={getattr(masks[0], 'shape', 'NA')} dtype={getattr(masks[0], 'dtype', 'NA')} max={getattr(masks[0], 'max', lambda: 'NA')() if hasattr(masks[0],'max') else 'NA'}")
        if isinstance(flows, (list, tuple)) and len(flows) > 0:
            f0 = flows[0]
            if isinstance(f0, (list, tuple)) and len(f0) >= 3:
                f_xy, f_vec, cellprob = f0[0], f0[1], f0[2]
                print(f"[Stage C] Example flows[0]: HSV={getattr(f_xy,'shape','NA')}, vec={getattr(f_vec,'shape','NA')}, prob={getattr(cellprob,'shape','NA')}")
    except Exception as ex:
        print(f"[Stage C][WARN] Could not print example outputs: {ex}")
    return ok