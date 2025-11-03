"""
WholeOrganoidExperiment — Stage A
Prepare a timestamped run directory, validate dataset, and log environment.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import json

from .config_store import ConfigStore, Config
from .dataset import DatasetManager
from .logger import RunLogger

class WholeOrganoidExperiment:
    """Stage A orchestrator.

    Methods
    -------
    prepare() -> None
        Create run_dir, save config snapshot, log env, verify dataset.
    get_run_dir() -> Path
        Return the prepared run directory path.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.run_dir = Path(self.cfg.paths["results_root"]) / self.cfg.model_name_out / f"run_{ts}"
        self.cfg_dir = self.run_dir / "cfg"
        self.logger = RunLogger(self.run_dir)

    def prepare(self) -> None:
        """Prepare a new run directory and perform Stage A tasks.

        Writes:
            - run_dir/cfg/config_snapshot.yaml
            - run_dir/cfg/env_and_cfg.json
            - run_dir/cfg/dataset_report.json
            - run_dir/cfg/metrics.json (timings)

        Raises:
            AssertionError if critical directories are missing.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # Snapshot config + env
        ConfigStore.save_snapshot(self.cfg_dir, self.cfg)
        self.logger.log_env_and_cfg({
            "model_name_out": self.cfg.model_name_out,
            "paths": self.cfg.paths,
            "labels": self.cfg.labels,
            "train": self.cfg.train,
            "eval": self.cfg.eval,
            "system": self.cfg.system,
        }, self.cfg_dir)

        # Verify dataset
        dm = DatasetManager(self.cfg.paths, self.cfg.labels, self.cfg_dir)
        report = dm.verify_structure()

        # Create train/ and eval/ placeholders for future stages
        (self.run_dir / "train").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "eval").mkdir(parents=True, exist_ok=True)

        # Human-readable summary
        summary = {
            "run_dir": str(self.run_dir),
            "n_train_images": report.get("n_train_images", 0),
            "n_valid_images": report.get("n_valid_images", 0),
            "n_train_with_labels": report.get("n_train_with_labels", 0),
            "warnings": report.get("warnings", []),
        }
        (self.cfg_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2))

    def get_run_dir(self) -> Path:
        return self.run_dir
