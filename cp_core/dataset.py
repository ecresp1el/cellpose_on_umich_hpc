"""
DatasetManager — Stage A
Dataset discovery and basic validation per contract.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal
import json
import re

IMAGE_EXTS = {".tif", ".tiff"}

class DatasetReport(dict):
    """Lightweight report: counts & warnings, JSON-serializable."""
    pass

class DatasetManager:
    """Discover images/labels and validate structure.

    Methods
    -------
    verify_structure() -> DatasetReport
        Ensure directories exist; count images/labels; return report.
    list_images(split: {"train","valid","all"}) -> list[Path]
        Deterministically list images for a split.
    image_id(image_path: Path) -> str
        Stable, sanitized stem used for artifact filenames.
    label_for(image_path: Path) -> Optional[Path]
        Resolve paired label based on mask_filter convention.
    """
    def __init__(self, paths: Dict[str, Any], labels_cfg: Dict[str, Any], run_cfg_dir: Path):
        self.paths = paths
        self.labels_cfg = labels_cfg
        self.run_cfg_dir = run_cfg_dir
        self.mask_filter: str = labels_cfg.get("mask_filter", "_seg.npy")

    def _list_dir_images(self, d: Path) -> List[Path]:
        return sorted([p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS])

    def list_images(self, split: Literal["train","valid","all"]) -> List[Path]:
        if split not in {"train","valid","all"}:
            raise ValueError(f"Invalid split: {split}")
        train_dir = Path(self.paths["data_images_train"])
        valid_dir = Path(self.paths["data_images_valid"])
        train = self._list_dir_images(train_dir) if train_dir.exists() else []
        valid = self._list_dir_images(valid_dir) if valid_dir.exists() else []
        if split == "train":
            return train
        if split == "valid":
            return valid
        # all
        union = sorted(list({*train, *valid}))
        return union

    def image_id(self, image_path: Path) -> str:
        stem = image_path.stem
        # sanitize: keep alnum, +, -, _, . (for your filenames)
        safe = re.sub(r"[^A-Za-z0-9_\-\+\.\(\)]", "_", stem)
        return safe

    def label_for(self, image_path: Path) -> Optional[Path]:
        # Look in corresponding labels folder for the split
        split = "train" if str(image_path).startswith(self.paths["data_images_train"]) else "valid"
        labels_root = Path(self.paths[f"data_labels_{split}"])
        if not labels_root.exists():
            return None
        stem = image_path.stem
        # Strategy: prefer exact stem + mask_filter; also allow .png fallback
        cand1 = labels_root / f"{stem}{self.mask_filter}"
        if cand1.exists():
            return cand1
        cand2 = labels_root / f"{stem}.png"
        if cand2.exists():
            return cand2
        return None

    def verify_structure(self) -> DatasetReport:
        report = DatasetReport()
        # Required containers
        req = [
            "data_images_train",
            "data_labels_train",
            "data_images_valid",
            "data_labels_valid",
        ]
        missing = [k for k in req if not Path(self.paths[k]).exists()]
        report["missing_dirs"] = missing
        # Counts
        train_images = self._list_dir_images(Path(self.paths["data_images_train"])) if Path(self.paths["data_images_train"]).exists() else []
        valid_images = self._list_dir_images(Path(self.paths["data_images_valid"])) if Path(self.paths["data_images_valid"]).exists() else []
        report["n_train_images"] = len(train_images)
        report["n_valid_images"] = len(valid_images)

        # Mask presence estimate
        has_labels = 0
        for ip in train_images:
            if self.label_for(ip) is not None:
                has_labels += 1
        report["n_train_with_labels"] = has_labels
        report["mask_filter"] = self.mask_filter

        # Warnings
        warnings = []
        if len(train_images) == 0:
            warnings.append("No training images found.")
        if len(valid_images) == 0:
            warnings.append("No validation images found.")
        if has_labels == 0:
            warnings.append("No training labels matching mask_filter.")
        report["warnings"] = warnings

        # Write to run cfg dir
        self.run_cfg_dir.mkdir(parents=True, exist_ok=True)
        (self.run_cfg_dir / "dataset_report.json").write_text(json.dumps(report, indent=2))
        return report
