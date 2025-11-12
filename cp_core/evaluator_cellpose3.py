"""
EvaluatorCellpose3 — Stage C (Evaluation & Artifact Generation, Cellpose v3 API)

Purpose
-------
Run native-scale inference with a trained Cellpose v3 model and save tidy
artifacts per image: masks, flows, probabilities, and ROIs. Panel creation
is deferred to a future helper once upstream plotting requirements are
finalized. This class adheres to the Stage C section of the contract.

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
        # `imgs` is a list of numpy arrays exactly as returned by cp_io.imread.
        # `kept_files` mirrors that list with the Path for each successfully loaded image.
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
            # The Cellpose API requires a list of images, even for single-image evaluation.
            # `im` is therefore the numpy array for the current file and `[im]` satisfies the API.
            m, f, s = self.model.eval([im], **kw)   # returns (masks, flows, styles) lists of length 1

            # The public docstring for CellposeModel.eval describes the payload per image:
            #   masks[k] -> labelled array
            #   flows[k] -> [HSV flow, XY vectors, cellprob, Euler integration, ...]
            #   styles[k] -> style vector (retained for CP3 compatibility)
            mask_i = m[0] if isinstance(m, (list, tuple)) else m
            flow_pack_i = f[0] if isinstance(f, (list, tuple)) else f
            style_i = s[0] if isinstance(s, (list, tuple)) else s

            masks.append(mask_i)
            flows.append(flow_pack_i)
            styles.append(style_i)

            nmask = int(getattr(mask_i, "max", lambda: 0)())
            dt = time.time() - t0
            print(
                f"[Stage C] eval {i}/{N}: {p.name}  shape={getattr(im,'shape','?')}  n_masks={nmask}  time={dt:.1f}s",
                flush=True,
            )

        # 6) persist native Cellpose artifacts for each image
        print("[Stage C] Saving native Cellpose outputs…")
        n_images = 0
        for i, p in enumerate(kept_files):
            try:
                stem = p.stem

                # Prepare single-item payloads that adhere to the Cellpose IO API
                images_payload = [imgs[i]]
                masks_payload = [masks[i]]
                flows_payload = [flows[i]]
                file_bases = [str(out_dir / stem)]

                # Save mask via official API (TIF), same filename convention
                cp_io.save_masks(
                    images_payload,
                    masks_payload,
                    flows_payload,
                    file_bases,
                    png=False,
                    tif=True,
                    suffix="_masks",
                    save_flows=False,
                    save_outlines=False,
                    savedir=None,
                    in_folders=False,
                )
                print(f"[Stage C] (n_masks={int(getattr(masks[i],'max',lambda:0)())})")

                # Save native *_seg.npy via API + print its metadata
                cp_io.masks_flows_to_seg(
                    images_payload,
                    masks_payload,
                    flows_payload,
                    file_bases,
                    channels=None,
                )

                seg_path = out_dir / f"{stem}_seg.npy"
                if seg_path.exists():
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
                else:
                    print(f"[Stage C][WARN] Expected seg artifact missing: {seg_path}")

                # If you later want ROIs, uncomment and call Cellpose's API directly:
                # cp_io.save_rois(masks[i], str(out_dir / f"{stem}_rois.zip"))

                # Placeholder hook for future panel plotting against the raw CP outputs.
                _panel_placeholder(imgs[i], masks[i], flows[i], out_dir, stem)

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


def _panel_placeholder(*_args, **_kwargs):
    """Reserved for future CP-native panel plotting once requirements are finalized."""
    return None
