"""
DatasetManager — Stage A
Dataset discovery and basic validation per contract.

Purpose
-------
Central place to:
  * list images deterministically by split,
  * infer the label path for a given image (based on a suffix pattern),
  * perform a minimal structure/contents check and emit a JSON report.

Notes
-----
* We deliberately keep Stage A checks "shallow but decisive":
  - assert that required directories exist,
  - count images,
  - estimate how many training images have a discoverable label.
* Full content validation (e.g., mask dtype, geometry match) is deferred
  to later stages or dedicated QC utilities.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal
import json
import re

# Recognized image filename extensions for discovery.
IMAGE_EXTS = {".tif", ".tiff"}

class DatasetReport(dict):
    """Lightweight report: counts & warnings, JSON-serializable.

    We inherit from dict to make writing to JSON trivial while still being
    flexible for future keys (Stage B/C may add more fields).
    """
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

    Design Notes
    ------------
    * "mask_filter" is treated as a literal suffix (no globbing). For an image
      stem S and mask_filter F, we first check labels/<S+F>. As a pragmatic
      fallback (historical masks) we also accept labels/<S>.png.
    * This class does not read image data — it only resolves paths and names.
    """
    
    def __init__(self, paths: Dict[str, Any], labels_cfg: Dict[str, Any], run_cfg_dir: Path):
        """
        Args
        ----
        paths : dict
            Expected to contain absolute paths for:
            data_images_train, data_labels_train, data_images_valid, data_labels_valid.
        labels_cfg : dict
            Must include 'mask_filter' (e.g., '_cp_masks.png', '.png', '_seg.npy').
        run_cfg_dir : Path
            Destination where dataset_report.json will be written.

        Notes
        -----
        * We do not create or mutate dataset folders here; only read/inspect.
        """
        self.paths = paths
        self.labels_cfg = labels_cfg
        self.run_cfg_dir = run_cfg_dir
        # If no mask_filter given, prefer the GUI default used historically.
        self.mask_filter: str = labels_cfg.get("mask_filter", "_seg.npy")

    def _list_dir_images(self, d: Path) -> List[Path]:
        """Return a sorted list of image files in directory d.

        Sorting ensures deterministic ordering (useful for reproducible sampling
        or stable eval subsets).
        """
        return sorted([p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS])

    def list_images(self, split: Literal["train","valid","all"]) -> List[Path]:
        """List images for a given split.

        Args
        ----
        split : {"train","valid","all"}
            Which subset to enumerate.

        Returns
        -------
        list[Path]
            Sorted list of image file paths.

        Raises
        ------
        ValueError
            If an invalid split name is provided.

        Notes
        -----
        * 'all' returns the union of train and valid (de-duplicated).
        """
        
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
        """Return a sanitized, stable stem for artifact filenames.

        We keep alnum, '+', '-', '_', '.', and parentheses — this accommodates
        your existing long microscope filenames while avoiding shell surprises.

        Example
        -------
        'Exp68_Slide1...DAPI.tif' -> 'Exp68_Slide1...DAPI'
        """
        stem = image_path.stem
        # sanitize: keep alnum, +, -, _, . (for your filenames)
        safe = re.sub(r"[^A-Za-z0-9_\-\+\.\(\)]", "_", stem)
        return safe

    def label_for(self, image_path: Path) -> Optional[Path]:
        """Resolve the expected label path for a given image.

        Resolution Strategy
        -------------------
        1) Prefer exact: labels/<image_stem + mask_filter>
           e.g., stem='foo', mask_filter='_cp_masks.png' -> 'labels/foo_cp_masks.png'
        2) Fallback (legacy): labels/<image_stem>.png

        Args
        ----
        image_path : Path
            Path to the image in train/ or valid/ images directory.

        Returns
        -------
        Path | None
            The matched label path if it exists, otherwise None.

        Notes
        -----
        * 'mask_filter' is a literal suffix, not a glob pattern.
        * We infer the split by checking whether the image lives under
          data_images_train vs data_images_valid.
        """
        
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
        """Check required dirs, count images, and estimate label coverage.

        Returns
        -------
        DatasetReport
            JSON-serializable summary with:
              - missing_dirs: list[str]
              - n_train_images: int
              - n_valid_images: int
              - n_train_with_labels: int  (how many training images have a label)
              - mask_filter: str
              - warnings: list[str]

        Writes
        ------
        run_cfg_dir/dataset_report.json

        Notes
        -----
        * We only check the TRAIN split for label coverage, because it's the
          supervision set for Stage B. VALID may be empty until Stage C.
        """
        
        report = DatasetReport()

        # Required containers (existence only; contents may be empty in Stage A)
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

        # Mask presence estimate (TRAIN only)
        has_labels = 0
        for ip in train_images:
            if self.label_for(ip) is not None:
                has_labels += 1
        report["n_train_with_labels"] = has_labels
        report["mask_filter"] = self.mask_filter

        # Compose warnings that guide the user to fix common issues quickly.
        warnings = []
        if len(train_images) == 0:
            warnings.append("No training images found.")
        if len(valid_images) == 0:
            warnings.append("No validation images found.")
        if has_labels == 0:
            warnings.append("No training labels matching mask_filter.")
        report["warnings"] = warnings

        # Persist report next to other Stage A metadata for this run.
        self.run_cfg_dir.mkdir(parents=True, exist_ok=True)
        (self.run_cfg_dir / "dataset_report.json").write_text(json.dumps(report, indent=2))
        return report
