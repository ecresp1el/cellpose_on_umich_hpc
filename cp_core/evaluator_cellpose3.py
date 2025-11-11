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
import time

import matplotlib
matplotlib.use("Agg")  # headless backend for SLURM jobs
from matplotlib import pyplot as plt
import numpy as np

import cellpose
from cellpose import io as cp_io, plot
from cellpose.models import CellposeModel

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
            cpv = getattr(cellpose, "__version__", "unknown")
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
        assert CellposeModel is not None, "CellposeModel not available."
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
            return CellposeModel(gpu=use_gpu, pretrained_model=str(self.cp_model_dir))

        if have_fallback:
            print(f"[Stage C] model_source = fine-tuned CPSAM (state_dict): {self.fallback_weights}")
            import torch
            m = CellposeModel(gpu=use_gpu, pretrained_model=None)
            sd = torch.load(str(self.fallback_weights), map_location="cpu")
            m.net.load_state_dict(sd, strict=False)
            return m

        # Otherwise fall back to pretrained CPSAM
        print("[Stage C] model_source = stock CPSAM (no fine-tuned weights found)")
        return CellposeModel(gpu=use_gpu, pretrained_model="cpsam")
        
    # -------------------- evaluation --------------------
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

        files = dm.list_images(split=split)  # 'valid' or 'all' (Stage A union logic)

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

        # 4) show kwargs (filter out non-eval keys like save_* before calling CP)
        kw = dict(getattr(args, "kwargs", {}))  # copy; don’t mutate original
        for _bad in ("save_panels", "save_rois", "save_flows", "save_prob", "save_prob_view"):
            kw.pop(_bad, None)
        _print_eval_kwargs(kw)
        
        # 5) run eval (per-image for visible progress)
        N = len(imgs)
        print(f"[Stage C] Running eval per-image for visibility… total={N}")

        masks, flows, styles = [], [], []
        for i, (im, p) in enumerate(zip(imgs, kept_files), 1):
            t0 = time.time()
            m, f, s = self.model.eval([im], **kw)   # CP-SAM expects list; returns lists
            m0 = m[0]
            masks.append(m0)
            flows.append(f)
            styles.append(s)

            nmask = int(getattr(m0, "max", lambda: 0)())
            dt = time.time() - t0
            print(f"[Stage C] eval {i}/{N}: {p.name}  shape={getattr(im,'shape','?')}  n_masks={nmask}  time={dt:.1f}s", flush=True)

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
                
                # per-image flow pack (HSV, vec, prob, ...)
                fpack = flows[i][0] if isinstance(flows[i], (list, tuple)) else flows[i]

                # Save mask via official API (TIF), same filename convention
                _save_masks_api(imgs[i], masks[i], fpack, stem, out_dir)
                print(f"[Stage C] (n_masks={int(getattr(masks[i],'max',lambda:0)())})")

                # Save native *_seg.npy via API + print its metadata
                _save_seg_npy_api(imgs[i], masks[i], fpack, stem, out_dir)

                # If you later want ROIs, uncomment:
                # _save_rois_api(masks[i], stem, out_dir)
                
                # optional: 1x4 panel (guarded)
                print(f"[Stage C][debug] save_panels flag = {(self.cfg.eval or {}).get('save_panels', True)}")
                if (self.cfg.eval or {}).get("save_panels", True):
                    try:
                        # Prepare image for plotting: CxHxW -> HxWxC (matplotlib wants HWC)
                        im_plot = imgs[i]
                        # Use HxWxC for plotting (your imgs are CxHxW)
                        # Prepare image for panel: smallest dimension = channels → move to last (HxWxC)
                        if im_plot.ndim == 3:
                            
                            # Identify the channel axis as the smallest dimension
                            ch_axis = int(np.argmin(im_plot.shape))          # pick the smallest dim as channels
                            n_ch = int(im_plot.shape[ch_axis])
                            
                            # Move channels to last (HxWxC)
                            if ch_axis != 2:
                                im_plot = np.moveaxis(im_plot, ch_axis, -1)  # -> HxWxC
                            
                            # If more than 3 channels, keep only the first 3 for RGB-style plotting
                            # For plotting, enforce ≤3 channels (CP-SAM/plotting expects RGB/gray)
                            if n_ch > 3:
                                print(f"[Stage C][panel] detected {n_ch} channels; using first 3 for panel")
                                im_plot = im_plot[..., :3]
                            print(f"[Stage C][panel] channel_axis={ch_axis} n_channels={n_ch} → im_plot={im_plot.shape} {im_plot.dtype}")
                        else:
                            # 2D / grayscale case
                            print(f"[Stage C][panel] grayscale image; im_plot={im_plot.shape} {im_plot.dtype}")

                        # Use the per-image flow pack (list/tuple: [HSV, vec, prob, ...])
                        fpack = flows[i][0] if isinstance(flows[i], (list, tuple)) else flows[i]
                        hsv  = fpack[0] if isinstance(fpack, (list, tuple)) and len(fpack) > 0 else None
                        vec  = fpack[1] if isinstance(fpack, (list, tuple)) and len(fpack) > 1 else None
                        prob = fpack[2] if isinstance(fpack, (list, tuple)) and len(fpack) > 2 else None

                        # Diagnostics before plotting
                        print(
                            "[Stage C][panel] "
                            f"im_plot={getattr(im_plot,'shape','?')} {getattr(im_plot,'dtype','?')} | "
                            f"mask={getattr(masks[i],'shape','?')} {getattr(masks[i],'dtype','?')} | "
                            f"hsv={getattr(hsv,'shape','?')} vec={getattr(vec,'shape','?')} prob={getattr(prob,'shape','?')}"
                        )

                        fig = plt.figure(figsize=(12, 5))
                        # IMPORTANT:
                        plot.show_segmentation(fig, im_plot, masks[i], hsv)  # pass ONLY the HSV flow image
                        fig.tight_layout()
                        panel_path = out_dir / f"{stem}_panel_1x4.png"
                        fig.savefig(panel_path, dpi=150)
                        plt.close(fig)
                        print(f"[Stage C] wrote: {panel_path.name}")
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

def _safe_read_image(p: Path) -> Optional[np.ndarray]:
    try:
        im = cp_io.imread(p)
        if im is None:
            print(f"[Stage C][WARN] imread returned None for: {p.name}")
            return None
        if not isinstance(im, np.ndarray):
            print(f"[Stage C][WARN] imread did not return ndarray for: {p.name} (type={type(im)})")
            return None

        # basic stats
        shp = im.shape
        print(f"[Stage C] Loaded {p.name}: shape={shp}, dtype={im.dtype}, min={np.min(im)}, max={np.max(im)}")

        # channel-axis diagnostics (print-only; no mutation)
        if im.ndim == 3:
            # Heuristic: the channel axis is the smallest of the 3 dims
            ch_axis = int(np.argmin(shp))
            n_ch = int(shp[ch_axis])

            # If it looks ambiguous (e.g., all dims large), fall back to channels-last
            if not (n_ch <= 10 and all(d > 32 for k,d in enumerate(shp) if k != ch_axis)):
                # Ambiguity fallback: prefer channels-last if last dim is small
                if shp[2] <= 10:
                    ch_axis, n_ch = 2, shp[2]

            print(f"[Stage C] channels_axis={ch_axis}  n_channels={n_ch}")
            if n_ch > 3:
                print(f"[Stage C][NOTE] {p.name}: detected {n_ch} channels; CP-SAM will internally use the first 3 for inference/plotting.")
        elif im.ndim > 3:
            print(f"[Stage C][NOTE] {p.name}: ndim={im.ndim} (>3). Only the first 3 axes are visual channels/H/W for our logs.")
        # (No slicing occurs here; evaluation uses the array as loaded.)
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

def _init_model_from_cfg(cfg):
    """Initialize a Cellpose model for evaluation (CP-SAM only)."""
    # prefer YAML train.pretrained_model if provided, else cpsam
    pm = (getattr(cfg, "train", None) or {}).get("pretrained_model", "cpsam")

    # version print
    try:
        cpv = getattr(cellpose, "__version__", "unknown")
    except AttributeError:
        from cellpose import __version__ as cpv

    print(f"[Stage C] Initializing Cellpose model (v{cpv}) with pretrained_model='{pm}', gpu=True")

    # init
    model = CellposeModel(gpu=True, pretrained_model=pm)
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

def _save_seg_npy_api(im, m, fpack, stem: str, out_dir: Path) -> Path:
    """
    Save official * _seg.npy via Cellpose API for ONE image.
    API appends '_seg.npy' to the base path we give it.
    """
    base = str(out_dir / stem)
    cp_io.masks_flows_to_seg(im, m, fpack, base, channels=None)  # PACK, not outer list

    seg_path = out_dir / f"{stem}_seg.npy"
    try:
        seg = np.load(seg_path, allow_pickle=True).item()
        keys = list(seg.keys())
        k_masks = seg.get("masks", None)
        k_out   = seg.get("outlines", None)
        k_flow  = seg.get("flows", None)
        k_chan  = seg.get("chan_choose", None)
        print(
            "[Stage C][seg.npy] wrote:", seg_path.name,
            "| keys=", keys,
            "| masks=", getattr(k_masks, "shape", "?"),
            "| outlines=", getattr(k_out, "shape", "?"),
            "| flows_len=", (len(k_flow) if isinstance(k_flow, (list, tuple)) else "NA"),
            "| chan_choose=", k_chan
        )
    except Exception as ex:
        print(f"[Stage C][WARN] could not inspect {seg_path.name}: {ex}")
    return seg_path

def _save_masks_api(im, m, fpack, stem: str, out_dir: Path) -> None:
    """
    Save masks via Cellpose API (TIF). We pass a single image/mask/flow PACK,
    wrapped as 1-element lists (the API accepts lists or singletons).
    """
    images = [im]
    masks  = [m]
    flows_ = [fpack]  # <-- PACK = (HSV, vec, prob, ...)
    file_names = [str(out_dir / stem)]

    cp_io.save_masks(
        images, masks, flows_, file_names,
        png=False, tif=True, suffix="_masks",
        save_flows=False, save_outlines=False,
        savedir=None, in_folders=False
    )
    print(f"[Stage C] save_masks (API) wrote: {stem}_masks.tif")
# Optional (disabled by default)
def _save_rois_api(m, stem: str, out_dir: Path) -> None:
    """
    Save ImageJ ROIs zip via API (disabled unless you call it).
    """
    zip_path = str(out_dir / f"{stem}_rois.zip")
    cp_io.save_rois(m, zip_path)
    print(f"[Stage C] save_rois (API) wrote: {Path(zip_path).name}")