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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import math
import csv

import numpy as np

from cellpose import models, io as cp_io, plot

# Reuse standardized I/O + layout helpers (single source of truth)
from cp_core.helper_functions.dl_helper import ensure_hwc_1to5, has_label, read_label
from cp_core.dataset import DatasetManager

# ---------- small image I/O helpers ----------
def _imsave_tif(path: Path, arr: np.ndarray) -> None:
    """Write TIFF (float32/uint16) strictly using Cellpose I/O.

    If cellpose.io.imsave is unavailable, print an error and abort.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if cp_io is None or not hasattr(cp_io, "imsave"):
        msg = f"[Stage C] ERROR: cellpose.io.imsave unavailable; cannot save {path}"
        print(msg, flush=True)
        raise RuntimeError(msg)

    cp_io.imsave(str(path), arr)
    print(f"[Stage C] Saved via Cellpose API → {path}", flush=True)


@dataclass
class EvalArgs:
    """Resolved evaluation args passed to Cellpose v3 model.eval()."""
    channels: List[int]           # e.g., [0,0]
    normalize: bool               # false by contract unless overridden
    niter: int                    # 2000 by contract
    resample: bool                # false by contract
    bsize: int                    # 512 by contract
    flow_threshold: float         # 0.4 by default
    cellprob_threshold: float     # 0.0 by default

    save_panels: bool
    save_rois: bool
    save_flows: bool
    save_prob: bool
    save_prob_view: bool


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
        """Resolve eval args from YAML with Stage C defaults/invariants."""
        e = self.cfg.eval
        chans = e.get("channels", self.cfg.train.get("channels", [0, 0]))
        return EvalArgs(
            channels=list(chans),
            normalize=bool(e.get("normalize", False)),
            niter=int(e.get("niter", 2000)),
            resample=bool(e.get("resample", False)),   # [CONTRACT] keep native grid
            bsize=int(e.get("bsize", 512)),
            flow_threshold=float(e.get("flow_threshold", 0.4)),
            cellprob_threshold=float(e.get("cellprob_threshold", 0.0)),
            save_panels=bool(e.get("save_panels", True)),
            save_rois=bool(e.get("save_rois", True)),
            save_flows=bool(e.get("save_flows", True)),
            save_prob=bool(e.get("save_prob", True)),
            save_prob_view=bool(e.get("save_prob_view", True)),
        )

    # -------------------- model loading --------------------
    def load_model(self):
        """Load a Cellpose model for evaluation.

        Specialist-first:
        - Prefer the CP-managed model directory created during training
        - Fallback to our saved state_dict if present

        Baseline:
        - If eval.baseline_pretrained: true OR no trained weights/dir found,
            build a vanilla model from eval.model_type (default 'cyto3')
            or eval.pretrained_model if provided.
        """
        assert models is not None, "cellpose.models not available."
        # Auto-detect GPU
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except Exception:
            use_gpu = False

        # -------------------- BASELINE HANDLING (added) --------------------
        # The goal of this block is to decide whether we should run evaluation
        # using a *baseline pretrained* Cellpose model (e.g. vanilla cyto3)
        # instead of a specialist fine-tuned model from this run_dir.
        #
        # This allows the same Stage C evaluation pipeline to handle both:
        #   (1) "baseline" evaluations — using public/pretrained Cellpose weights
        #   (2) "specialist" evaluations — using custom fine-tuned weights saved in run_dir/train/
        #
        # Decision rule:
        #   → If eval.baseline_pretrained is True in the snapshot, OR
        #     if no trained weights are found in cp_model_dir or fallback_weights,
        #     we assume this is a baseline run and use vanilla cyto3 (or user-provided pretrained_model).
        #
        # Otherwise (trained weights exist and baseline flag not set),
        #   → the logic falls through to the normal "specialist" loading section below.
        
        # Extract the eval block from the current run's config/snapshot (self.cfg).
        eval_cfg = getattr(self.cfg, "eval", {}) or {}
       
        # Boolean toggle for explicit baseline mode from YAML:
        # e.g. eval.baseline_pretrained: true
        baseline_flag = bool(getattr(eval_cfg, "baseline_pretrained", False))
        
        # Check for trained model artifacts in the current run directory:
        # - self.cp_model_dir  → main Cellpose-managed model folder (Stage B output)
        # - self.fallback_weights → optional .pt state_dict if cp_model_dir not present
        have_cp_dir = self.cp_model_dir.exists()
        have_fallback = self.fallback_weights.exists()

        
        # If baseline is explicitly requested, OR if there are no trained weights available,
        # then create a *baseline* Cellpose model source.
        if baseline_flag or (not have_cp_dir and not have_fallback):

            # Optional override: user may specify a custom pretrained checkpoint path in YAML
            # via eval.pretrained_model: /path/to/custom_weights.pt
            pretrained_override = getattr(eval_cfg, "pretrained_model", None)
            
            if pretrained_override:
                # Build the model using a specific pretrained weight file path
                print(f"[Stage C] model_source = baseline: pretrained_model={pretrained_override}")
                return models.CellposeModel(gpu=use_gpu, pretrained_model=str(pretrained_override))
            else:
                # Default to Cellpose's built-in 'cyto3' model type (public pretrained weights)
                mtype = getattr(eval_cfg, "model_type", "cyto3")
                print(f"[Stage C] model_source = baseline: {mtype}")
                return models.CellposeModel(gpu=use_gpu, model_type=mtype)
        # -------------------------------------------------------------------
        
        # Primary: CP's own model dir (best compatibility)
        if self.cp_model_dir.exists():
            print(f"[Stage C] model_source = specialist: {self.cp_model_dir}")
            return models.CellposeModel(gpu=use_gpu, pretrained_model=str(self.cp_model_dir))

        # Fallback: load state_dict into a scratch model
        if self.fallback_weights.exists():
            print(f"[Stage C] model_source = specialist (state_dict): {self.fallback_weights}")
            import torch
            m = models.CellposeModel(gpu=use_gpu, model_type=None)
            sd = torch.load(str(self.fallback_weights), map_location="cpu")
            m.net.load_state_dict(sd, strict=False)
            return m

        raise FileNotFoundError(
            f"No model found at {self.cp_model_dir} or {self.fallback_weights}"
        )

        # -------------------- evaluation --------------------
    def evaluate_images(self, split: str, args: EvalArgs) -> Dict[str, Any]:
        """Run inference on a dataset split ('valid' or 'all') and write artifacts.

        Returns
        -------
        dict
            Aggregate stats for eval_summary.json
        """
        # ---------------- discover images from the SNAPSHOT (not source YAML) ----------------
        split = "valid" if split not in ("valid", "all") else split

        def _cfg_get(obj, key):
            return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

        paths = self.cfg.paths
        labels = self.cfg.labels

        img_root = Path(_cfg_get(paths, "data_images_valid" if split == "valid" else "data_images_train"))
        lbl_root = Path(_cfg_get(paths, "data_labels_valid" if split == "valid" else "data_labels_train"))
        mask_sfx = _cfg_get(labels, "mask_filter")

        if not img_root or not img_root.exists():
            raise RuntimeError(f"[Stage C] ERROR: snapshot image root not found for split={split}: {img_root}")

        # Enumerate TIFFs from the snapshot root
        image_paths = sorted(list(img_root.glob("*.tif")) + list(img_root.glob("*.tiff")))

        # NEW: confirm snapshot sources and what's enumerated
        print(f"[EVAL] enumerating images from snapshot: {img_root}", flush=True)
        print(f"[EVAL] labels_dir={lbl_root}  mask_suffix={mask_sfx}", flush=True)
        if len(image_paths) > 0:
            print("[EVAL] first 3 stems:", [p.stem for p in image_paths[:3]], flush=True)

        # ---------------- log eval args ----------------
        print(
            "[Stage C] Eval args:",
            {
                "channels": args.channels,
                "normalize": args.normalize,
                "niter": args.niter,
                "bsize": args.bsize,
                "flow_threshold": args.flow_threshold,
                "cellprob_threshold": args.cellprob_threshold,
                "split": split,
                "n_images_found": len(image_paths),
            },
            flush=True,
        )

        model = self.load_model()

        # capture environment to help reproduce eval
        if cp_io is not None and hasattr(cp_io, "logger_setup"):
            cp_io.logger_setup()

        n_done = 0
        per_image_stats: List[Dict[str, Any]] = []

        for ip in image_paths:
            print(f"[EVAL] reading: {ip}", flush=True)
            img = self._read_image(ip)

            # Effective shape used by CP3 given channels
            H, W = img.shape[:2]
            used_shape = (H, W) if args.channels == [0, 0] else (H, W, img.shape[-1])
            print(
                f"[EVAL][{ip.stem}] img.shape={tuple(img.shape)}  channels={args.channels}  used_shape={used_shape}",
                flush=True
            )

            masks, flows, cellprob = self._eval_single(model, img, args)
            paths_written = self._save_artifacts(ip, img, masks, flows, cellprob, args)

            # per-image diagnostics
            n_masks     = int(masks.max())
            mask_pixels = int((masks > 0).sum())
            prob_mean   = float(np.mean(cellprob))
            prob_max    = float(np.max(cellprob))
            pos_frac    = float((cellprob > args.cellprob_threshold).mean())

            print(
                f"[Stage C][{ip.name}] n_masks={n_masks} "
                f"mask_px={mask_pixels} prob_mean={prob_mean:.3f} "
                f"prob_max={prob_max:.3f} pos_frac@thr={pos_frac:.3f}",
                flush=True,
            )

            per_image_stats.append({
                "stem": ip.stem,
                "paths": paths_written,
                "n_masks": n_masks,
                "mask_pixels": mask_pixels,
                "prob_mean": prob_mean,
                "prob_max": prob_max,
                "pos_frac": pos_frac,
                "shape": list(used_shape),   # record the effective shape actually used
            })
            n_done += 1

        # ---------------- aggregate ----------------
        agg = {
            "n_images": n_done,
            "mean_n_masks": float(np.mean([s["n_masks"] for s in per_image_stats])) if per_image_stats else 0.0,
            "split": split,
        }

        # ---------------- write CSV for quick filtering ----------------
        csv_path = self.eval_dir / "eval_metrics.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["stem", "n_masks", "mask_pixels", "prob_mean", "prob_max", "pos_frac", "shape"]
            )
            writer.writeheader()
            for s in per_image_stats:
                writer.writerow({k: s.get(k) for k in writer.fieldnames})

        # ---------------- console summary of zero-mask images ----------------
        n_zero = sum(1 for s in per_image_stats if s["n_masks"] == 0)
        print(f"[Stage C] zero-mask images: {n_zero}/{n_done}", flush=True)

        # ---------------- write JSON summary ----------------
        (self.eval_dir / "eval_summary.json").write_text(json.dumps({
            "aggregate": agg,
            "per_image": per_image_stats
        }, indent=2))
        print(f"[Stage C] Evaluated {n_done} image(s); wrote artifacts to {self.eval_dir}")
        return agg

    # -------------------- helpers --------------------
    def _read_image(self, path: Path) -> np.ndarray:
        """Read image using the official Cellpose I/O API.

        Behavior:
        - Delegates to cellpose.io.imread() (handles TIFF/PNG/JPG stacks)
        - Returns array ready for model.eval()
        - Still normalized to HWC layout via ensure_hwc_1to5()
        """
        img = cp_io.imread(str(path))              # ← Cellpose official API
        img = ensure_hwc_1to5(img, debug=False)    # keep consistent layout, drop trivial alpha, ≤5 channels
        return img

    def _eval_single(self, model, img: "np.ndarray", args) -> Tuple["np.ndarray", Any, "np.ndarray"]:
        """Run Cellpose v3 eval and return (masks, flows, cellprob).

        Key points:
        - Pass a *list* [img] to eval() to avoid CP treating H as batch.
        - Unwrap the first item from lists returned by CP.
        - Extract cellprob from flows robustly and fallback to zeros.
        """
        out = model.eval(
            [img],                               # <-- IMPORTANT: pass a list
            channels=args.channels,
            normalize=args.normalize,
            rescale=False,                        # keep native grid
            niter=args.niter,
            bsize=args.bsize,
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold
        )

        if not isinstance(out, tuple):
            raise RuntimeError(f"Unexpected eval() return type: {type(out)}")

        if len(out) == 4:
            masks, flows, _styles, _diams = out
        elif len(out) == 3:
            masks, flows, _styles = out
        else:
            raise RuntimeError(f"Unexpected number of values from eval(): {len(out)}")

        # CP returns lists (one per input image) — unwrap index 0
        masks = masks[0]
        flows = flows[0]

        # Extract cellprob / logits robustly
        cellprob = None
        if isinstance(flows, dict):
            for k in ("cellprob", "p", "prob", "P"):
                if k in flows:
                    cellprob = flows[k]
                    break
        else:
            # Some variants return a numpy stack (C,H,W); channel 2 often prob
            if hasattr(flows, "ndim") and flows.ndim >= 3 and flows.shape[0] >= 3:
                cellprob = flows[2].astype(np.float32, copy=False)

        if cellprob is None:
            cellprob = np.zeros(masks.shape, dtype=np.float32)

        return masks, flows, cellprob

    def _save_artifacts(
        self,
        ip: Path,
        image: np.ndarray,
        masks: np.ndarray,
        flows: Any,
        cellprob: np.ndarray,           # kept for compatibility
        args: EvalArgs,
    ) -> Dict[str, str]:
        """
        Save Cellpose-native outputs for a single image using the Cellpose API only.
        - mask TIFF and *_seg.npy via cellpose.io.save_masks()
        - QA segmentation panel via cellpose.plot.show_segmentation()
        - optional ROIs via cellpose.io.save_rois()
        """
        if cp_io is None or not hasattr(cp_io, "save_masks"):
            msg = "[Stage C] ERROR: cellpose.io.save_masks unavailable; cannot write artifacts."
            print(msg, flush=True)
            raise RuntimeError(msg)

        written: Dict[str, str] = {}
        stem = ip.stem

        # ---------- 1) Canonical *_seg.npy + mask TIFF ----------
        try:
            cp_io.save_masks(
                images=[image],
                masks=[masks],
                flows=[flows],
                file_names=[stem],
                savedir=str(self.eval_dir),
                png=False,
                tif=True,                # write _masks.tif
                save_flows=True,
                save_outlines=True,
            )
            f_mask = self.eval_dir / f"{stem}_masks.tif"
            f_seg  = self.eval_dir / f"{stem}_seg.npy"
           
            if f_mask.exists(): written["masks"]   = str(f_mask)
            if f_seg.exists():  written["seg_npy"] = str(f_seg)
           
            print(f"[Stage C][{stem}] saved via Cellpose API → {f_mask.name}, {f_seg.name}", flush=True)
        
        except Exception as e:
            msg = f"[Stage C] ERROR: save_masks() failed for {stem}: {e}"
            print(msg, flush=True)
            raise

        # ---------- 2) 1×4 Segmentation Panel via Cellpose API ----------
        try:
            # Define panel output path
            f_panel = self.d_panels / f"{stem}_panel_1x4.png"

            # Use official Cellpose plotting API (no custom matplotlib logic)
            fig = plot.show_segmentation(
                image=image,
                masks=masks,
                flows=flows if isinstance(flows, dict) else None,
                channels=self.cfg.eval.get("channels", [0, 0]),
                title=f"{stem} (n={int(masks.max())})",
            )

            # If Cellpose returned a matplotlib Figure, save it directly
            if hasattr(fig, "savefig"):
                fig.savefig(str(f_panel), dpi=200, bbox_inches="tight")
                print(f"[Panel][{stem}] saved panel: {f_panel}", flush=True)
                written["panel_png"] = str(f_panel)
            else:
                print(f"[Panel][{stem}] WARN: plot.show_segmentation() did not return a Figure object", flush=True)

        except Exception as e:
            # If plotting fails, log the warning but do not interrupt the run
            print(f"[Panel][{stem}] WARN: plot.show_segmentation failed: {e}", flush=True)

        # ---------- 3) Optional ROIs ----------
        if args.save_rois:
            if hasattr(cp_io, "save_rois"):
                base = (self.d_rois / stem).with_suffix("")  # cp_io appends .zip
                cp_io.save_rois(masks.astype(np.uint16, copy=False), str(base))
                written["rois_zip"] = str(base) + ".zip"
                print(f"[Stage C][{stem}] saved rois: {written['rois_zip']}", flush=True)
            else:
                print(f"[Stage C][{stem}] ERROR: cellpose.io.save_rois unavailable; skipping ROIs.", flush=True)

        # ---------- 4) Minimal JSON summary (optional) ----------
        f_json = self.d_json / f"{stem}_summary.json"
        f_json.write_text(json.dumps({
            "image": str(ip),
            "masks_tif": written.get("masks"),
            "seg_npy": written.get("seg_npy"),
            "panel_png": written.get("panel_png"),
            "rois_zip": written.get("rois_zip"),
        }, indent=2))
        written["summary_json"] = str(f_json)

        return written