"""
TrainerCellpose3 — Stage B (Training, Cellpose v3 API, native scale)

Purpose
-------
Train a specialist model for whole-organoid segmentation using the Cellpose v3
Python API (not CLI) so we can explicitly control scale:
  * enforce `rescale=False`  (native pixels)
  * enforce a fixed morphological prior `diameter=1350` px  (configurable)
  * use `bsize=512` for tiling
  * use repo/YAML defaults unless the user overrides (e.g., learning_rate)

This module implements the Stage B section of
CONTRACT_Methods_Cellpose3_WholeOrganoid_Pipeline.md
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import time

# ---------- External dependencies ----------
try:
    import numpy as np
except Exception:
    np = None

try:
    from cellpose import train, models, io as cp_io
except Exception:
    train = None
    models = None
    cp_io = None

# ---------- Local imports ----------
from .dataset import DatasetManager
from .logger import RunLogger
from .config_store import Config


# --------------------------------------------------------------------------- #
# Data structure for resolved training args
# --------------------------------------------------------------------------- #
@dataclass
class TrainArgs:
    """Arguments passed to Cellpose v3 API (train.train_seg).

    Notes
    -----
    * Optional keys (learning_rate, weight_decay) are included only
      if not None, letting Cellpose apply its internal defaults.
    """
    n_epochs: int
    batch_size: int
    rescale: bool           # must be False for specialist training
    diameter: int           # morphological prior (px)
    bsize: int              # tile size
    channels: List[int]     # e.g., [0, 0] for grayscale
    normalize: bool
    min_train_masks: int
    learning_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    save_each: bool = False
    nimg_per_epoch: Optional[int] = None
    extra_kwargs: Dict[str, Any] | None = None


# --------------------------------------------------------------------------- #
# Trainer class
# --------------------------------------------------------------------------- #
class TrainerCellpose3:
    """Cellpose v3 trainer (API-based), enforcing specialist scale policy."""

    def __init__(self, cfg: Config, run_dir: Path, logger: RunLogger):
        self.cfg = cfg
        self.run_dir = run_dir
        self.run_train_dir = self.run_dir / "train"
        self.logger = logger
        self.run_train_dir.mkdir(parents=True, exist_ok=True)

    # -------------------- model init --------------------
    def load_model(self, use_pretrained: bool, model_type: Optional[str]):
        """Initialize a Cellpose model (cyto3 or scratch) for v3 API training."""
        assert models is not None, "cellpose.models not available in environment."
        # Auto-detect CUDA; fall back to CPU cleanly if not available on the node/env
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except Exception:
            use_gpu = False

        if use_pretrained:
            mt = model_type if model_type is not None else "cyto3"
            model = models.CellposeModel(gpu=use_gpu, model_type=mt)
        else:
            model = models.CellposeModel(gpu=use_gpu, model_type=None)
        return model

    # -------------------- arg building --------------------
    def build_train_args(self, cfg: Config) -> TrainArgs:
        """Resolve training args from YAML, enforcing Stage B invariants."""
        t = cfg.train
        diameter = t.get("diameter", 1350)
        bsize = t.get("bsize", 512)
        n_epochs = t.get("n_epochs", 100)
        batch_size = t.get("batch_size", 1)
        channels = t.get("channels", [0, 0])
        normalize = t.get("normalize", False)
        min_train_masks = t.get("min_train_masks", 0)
        lr = t.get("learning_rate", None)
        wd = t.get("weight_decay", None)
        save_each = t.get("save_each", False)
        extra = t.get("extra_kwargs", {}) or {}
        nimg_per_epoch = t.get("nimg_per_epoch", None)   # <-- add this


        # invariants
        if t.get("rescale", False) is not False:
            raise ValueError("Stage B invariant violated: train.rescale must be False (specialist).")
        if diameter is None or int(diameter) <= 0:
            raise ValueError("Stage B requires a positive integer train.diameter (e.g., 1350).")

        return TrainArgs(
            n_epochs=int(n_epochs),
            batch_size=int(batch_size),
            rescale=False,
            diameter=int(diameter),
            bsize=int(bsize),
            channels=list(channels),
            normalize=bool(normalize),
            min_train_masks=int(min_train_masks),
            learning_rate=(float(lr) if lr is not None else None),
            weight_decay=(float(wd) if wd is not None else None),
            save_each=bool(save_each),
            nimg_per_epoch=(int(nimg_per_epoch) if nimg_per_epoch is not None else None),
            extra_kwargs=extra,
        )

    # -------------------- data loading --------------------
    def _load_training_data(self, images: List[Path], labels: List[Path]) -> Tuple[list, list]:
        """Load image/mask arrays into memory."""
        assert np is not None, "numpy required."
        if cp_io is not None and hasattr(cp_io, "imread"):
            imread = cp_io.imread
        else:
            try:
                from skimage.io import imread  # type: ignore
            except Exception as e:
                raise RuntimeError("No valid image reader available.") from e

        X, Y = [], []
        for ip, lp in zip(images, labels):
            img = imread(str(ip))
            if getattr(img, "ndim", 2) == 3 and img.shape[-1] == 1:
                img = img[..., 0]
            X.append(img)

            if lp.suffix.lower() == ".npy":
                mask = np.load(str(lp))
            else:
                mask = imread(str(lp))
            Y.append(mask)
        return X, Y



    # -------------------- training --------------------
    def train(self, model, images: List[Path], labels: List[Path], args: TrainArgs) -> Dict[str, Any]:
        """Run Cellpose v3 training with enforced specialist settings."""
        assert train is not None, "cellpose.train not available in environment."

        X_train, Y_train = self._load_training_data(images, labels)
        X_val, Y_val = None, None  # optional validation split (unused here)

        # base kwargs
        kwargs = dict(
            channels=args.channels,
            save_path=str(self.run_train_dir),
            n_epochs=args.n_epochs,
            min_train_masks=args.min_train_masks,
            rescale=args.rescale,
            model_name=self.cfg.model_name_out,
            normalize=args.normalize,
            bsize=args.bsize,
            batch_size=args.batch_size,
            save_each=args.save_each,  
            **(args.extra_kwargs or {}),
        )
        if args.learning_rate is not None:
            kwargs["learning_rate"] = args.learning_rate
        if args.weight_decay is not None:
            kwargs["weight_decay"] = args.weight_decay
        if args.nimg_per_epoch is not None:       # <-- add this block
            kwargs["nimg_per_epoch"] = args.nimg_per_epoch

        t0 = time.time()
        
        # Enable Cellpose logger so you get [INFO]/[WARNING] lines
        if cp_io is not None and hasattr(cp_io, "logger_setup"):
            cp_io.logger_setup()
            
        print("[Stage B] Starting training with args:",
            {k: (v if k != "extra_kwargs" else "...") for k, v in asdict(args).items()})
        
        train.train_seg(
            model.net,
            X_train, Y_train,
            test_data=X_val, test_labels=Y_val,
            **kwargs
        )
        dt = time.time() - t0

        metrics = {
            "duration_seconds": dt,
            "n_images": len(images),
            "effective_args": {
                k: (v if k != "extra_kwargs" else "...") for k, v in asdict(args).items()
            },
        }
        return metrics

    # -------------------- artifact I/O --------------------
    def save_weights(self, model, dst_dir: Path) -> Path:
        """Save final model weights to run/train/weights_final.pt."""
        dst_dir.mkdir(parents=True, exist_ok=True)
        import torch
        weight_path = dst_dir / "weights_final.pt"
        torch.save(model.net.state_dict(), weight_path)
        return weight_path

    def record_training_metadata(self, dst_dir: Path, metadata: dict) -> None:
        """Append training metadata (args, timings) to metrics.json."""
        dst_dir.mkdir(parents=True, exist_ok=True)
        mpath = dst_dir / "metrics.json"
        if mpath.exists():
            try:
                existing = json.loads(mpath.read_text())
            except Exception:
                existing = {}
            existing.update(metadata)
            mpath.write_text(json.dumps(existing, indent=2))
        else:
            mpath.write_text(json.dumps(metadata, indent=2))