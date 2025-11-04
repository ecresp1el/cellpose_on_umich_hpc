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

import numpy as np

try:
    from cellpose import models, io as cp_io, plot as cp_plot
except Exception as e:
    models = None
    cp_io = None
    cp_plot = None

# ---------- small image I/O helpers ----------
def _imsave_tif(path: Path, arr: np.ndarray) -> None:
    """Write TIFF (float32/uint16). Uses cellpose.io if available, else imageio."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if cp_io is not None and hasattr(cp_io, "imsave"):
        cp_io.imsave(str(path), arr)
        return
    try:
        import imageio
        imageio.imwrite(str(path), arr)
    except Exception as e:
        raise RuntimeError(f"Failed to save TIFF: {path}") from e

def _to_uint16_labels(masks: np.ndarray) -> np.ndarray:
    """Ensure label image is uint16 (Cellpose masks are integer-labeled)."""
    if masks.dtype != np.uint16:
        if masks.max() >= 65535:
            raise ValueError("Label IDs exceed uint16 range.")
        masks = masks.astype(np.uint16, copy=False)
    return masks

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically safe sigmoid for logits -> [0,1] view."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


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

        Prefers the CP-managed model directory created during training.
        Falls back to loading our saved state_dict if needed.
        """
        assert models is not None, "cellpose.models not available."
        # Auto-detect GPU
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except Exception:
            use_gpu = False

        # Primary: CP's own model dir (best compatibility)
        if self.cp_model_dir.exists():
            return models.CellposeModel(gpu=use_gpu, pretrained_model=str(self.cp_model_dir))

        # Fallback: load state_dict into a scratch model
        if self.fallback_weights.exists():
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
        # discover images
        from .dataset import DatasetManager
        dm = DatasetManager(self.cfg.paths, self.cfg.labels, self.run_dir / "cfg")
        if split not in ("valid", "all"):
            split = "valid"
        image_paths = dm.list_images(split if split in ("valid",) else "all")

        model = self.load_model()

        # capture environment to help reproduce eval
        if cp_io is not None and hasattr(cp_io, "logger_setup"):
            cp_io.logger_setup()

        n_done = 0
        per_image_stats = []

        # Load images batch-wise; CP eval can take list of arrays
        # For simplicity & memory safety, process one-by-one
        for ip in image_paths:
            img = self._read_image(ip)
            masks, flows, cellprob = self._eval_single(model, img, args)
            paths_written = self._save_artifacts(ip, img, masks, flows, cellprob, args)
            per_image_stats.append({
                "stem": ip.stem,
                "paths": paths_written,
                "n_masks": int(masks.max()),
                "shape": list(img.shape),
            })
            n_done += 1

        # aggregate
        agg = {
            "n_images": n_done,
            "mean_n_masks": float(np.mean([s["n_masks"] for s in per_image_stats])) if per_image_stats else 0.0,
            "split": split,
        }
        (self.eval_dir / "eval_summary.json").write_text(json.dumps({
            "aggregate": agg,
            "per_image": per_image_stats
        }, indent=2))
        print(f"[Stage C] Evaluated {n_done} image(s); wrote artifacts to {self.eval_dir}")
        return agg

    # -------------------- helpers --------------------
    def _read_image(self, path: Path) -> np.ndarray:
        if cp_io is not None and hasattr(cp_io, "imread"):
            arr = cp_io.imread(str(path))
        else:
            try:
                import imageio
                arr = imageio.imread(str(path))
            except Exception as e:
                raise RuntimeError(f"Failed to read image: {path}") from e
        if getattr(arr, "ndim", 2) == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        return arr

    def _eval_single(self, model, img: "np.ndarray", args) -> Tuple["np.ndarray", Any, "np.ndarray"]:
        """Run Cellpose v3 eval on CPU and return (masks, flows, cellprob).

        Compatible with both 3- and 4-return signatures:
        (masks, flows, styles)  or  (masks, flows, styles, diams)
        Extracts the logit map from `flows` regardless of key naming across versions.
        """
        out = model.eval(
            img,                        # can be single array or list
            channels=args.channels,
            normalize=args.normalize,
            rescale=args.resample,      # [CONTRACT] false → native grid
            niter=args.niter,
            bsize=args.bsize,
            flow_threshold=getattr(args, "flow_threshold", None),
            cellprob_threshold=getattr(args, "cellprob_threshold", None),
            gpu=False                   # <--- Force CPU mode
        )

        # Handle 3- or 4-tuple returns
        if not isinstance(out, tuple):
            raise RuntimeError(f"Unexpected eval() return type: {type(out)}")
        if len(out) == 4:
            masks, flows, _styles, _diams = out
        elif len(out) == 3:
            masks, flows, _styles = out
        else:
            raise RuntimeError(f"Unexpected number of values from eval(): {len(out)}")

        # Extract cellprob / logits robustly
        cellprob = None
        if isinstance(flows, dict):
            for k in ("cellprob", "p", "prob", "P"):
                if k in flows:
                    cellprob = flows[k]
                    break
        else:
            # Some builds return an ndarray stack with prob as channel 2
            import numpy as np
            if hasattr(flows, "ndim") and flows.ndim >= 3 and flows.shape[0] >= 3:
                cellprob = flows[2].astype(np.float32, copy=False)

        # Fallback to zeros if nothing found
        if cellprob is None:
            import numpy as np
            cellprob = np.zeros_like(masks, dtype=np.float32)

        return masks, flows, cellprob

    def _save_artifacts(self, ip: Path, image: np.ndarray, masks: np.ndarray,
                        flows: Dict[str, Any], cellprob: np.ndarray, args: EvalArgs) -> Dict[str, str]:
        """Write masks/flows/prob/rois/panel + JSON summary for one image."""
        stem = ip.stem

        # Masks (uint16)
        if args.save_prob or args.save_panels or args.save_rois or True:
            m_u16 = _to_uint16_labels(masks)
            f_mask = self.d_masks / f"{stem}_masks.tif"
            _imsave_tif(f_mask, m_u16)

        # Flows (raw + a simple magnitude viz)
        written = {"masks": str(f_mask)}
        if args.save_flows:
            dP = flows.get("dP", None)   # shape (2,H,W) = (dy,dx)
            if dP is not None and isinstance(dP, np.ndarray):
                f_flow_npy = self.d_flows / f"{stem}_flows.npy"
                f_flow_tif = self.d_flows / f"{stem}_flows.tif"
                np.save(str(f_flow_npy), dP)
                # magnitude visualization
                mag = np.sqrt(dP[0]**2 + dP[1]**2).astype(np.float32)
                _imsave_tif(f_flow_tif, mag)
                written["flows_npy"] = str(f_flow_npy)
                written["flows_tif"] = str(f_flow_tif)

        # Probabilities (logits + view)
        if args.save_prob:
            f_prob = self.d_prob / f"{stem}_prob.tif"
            _imsave_tif(f_prob, cellprob.astype(np.float32))
            written["prob_tif"] = str(f_prob)

        if args.save_prob_view:
            f_probv = self.d_prob / f"{stem}_prob_view.tif"
            prob_view = _sigmoid(cellprob.astype(np.float32))
            _imsave_tif(f_probv, prob_view)
            written["prob_view_tif"] = str(f_probv)

        # ROIs (ImageJ archive)
        if args.save_rois and cp_io is not None and hasattr(cp_io, "save_rois"):
            base = (self.d_rois / stem).with_suffix("")  # CP appends .zip
            cp_io.save_rois(m_u16, str(base))
            written["rois_zip"] = str(base) + ".zip"

        # 1×4 Panel (input | prob | flow viz | overlay) using CP plot when available
        if args.save_panels:
            try:
                if cp_plot is not None and hasattr(cp_plot, "show_segmentation"):
                    f_panel = self.d_panels / f"{stem}_panel_1x4.png"
                    # Build a composite figure
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
                    # 1) input (contrast-limited)
                    axs[0].imshow(self._auto_contrast(image), cmap="gray")
                    axs[0].set_title("input"); axs[0].axis("off")
                    # 2) prob (logit)
                    im1 = axs[1].imshow(cellprob, cmap="gray")
                    axs[1].set_title("prob (logit)"); axs[1].axis("off")
                    # 3) flow magnitude
                    if "dP" in flows and isinstance(flows["dP"], np.ndarray):
                        dP = flows["dP"]
                        mag = np.sqrt(dP[0]**2 + dP[1]**2)
                        axs[2].imshow(mag, cmap="gray")
                    axs[2].set_title("flow mag"); axs[2].axis("off")
                    # 4) overlay (CP utility)
                    try:
                        cp_plot.show_segmentation(image, masks, flows, channels=self.cfg.eval.get("channels",[0,0]), ax=axs[3])
                        axs[3].set_title("overlay"); axs[3].axis("off")
                    except Exception:
                        # fallback: simple boundary overlay
                        bnd = self._boundary_from_labels(m_u16)
                        axs[3].imshow(self._auto_contrast(image), cmap="gray")
                        axs[3].contour(bnd, colors="r", linewidths=0.5)
                        axs[3].set_title("overlay"); axs[3].axis("off")

                    fig.tight_layout()
                    fig.savefig(str(f_panel), dpi=200)
                    plt.close(fig)
                    written["panel_png"] = str(f_panel)
            except Exception:
                # panel is best-effort; keep going
                pass

        # Per-image JSON summary
        f_json = self.d_json / f"{stem}_summary.json"
        f_json.write_text(json.dumps({
            "image": str(ip),
            "masks_tif": written.get("masks"),
            "n_masks": int(m_u16.max()),
            "prob_tif": written.get("prob_tif"),
            "prob_view_tif": written.get("prob_view_tif"),
            "flows_npy": written.get("flows_npy"),
            "flows_tif": written.get("flows_tif"),
            "rois_zip": written.get("rois_zip"),
            "panel_png": written.get("panel_png"),
        }, indent=2))
        written["summary_json"] = str(f_json)
        return written

    # ---------- small image utilities ----------
    @staticmethod
    def _auto_contrast(img: np.ndarray, p_low=2.0, p_high=98.0) -> np.ndarray:
        """Percentile-based contrast stretch for panels."""
        lo, hi = np.percentile(img.astype(np.float32), [p_low, p_high])
        if hi <= lo:
            return img
        x = np.clip((img - lo) / (hi - lo), 0, 1)
        return x

    @staticmethod
    def _boundary_from_labels(lbl: np.ndarray) -> np.ndarray:
        """Return boolean boundary mask from label image."""
        from scipy.ndimage import binary_dilation
        edges = np.zeros_like(lbl, dtype=bool)
        # mark boundaries by comparing with 4-neighborhood shifts
        edges[:-1, :] |= (lbl[:-1, :] != lbl[1:, :])
        edges[:, :-1] |= (lbl[:, :-1] != lbl[:, 1:])
        return binary_dilation(edges, iterations=1)