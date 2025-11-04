"""
WholeOrganoidExperiment — Stage A
Prepare a timestamped run directory, validate dataset, and log environment.

Purpose
-------
Stage A orchestration wrapper. This class wires together:
  * Config snapshotting (for provenance),
  * Environment capture (for reproducibility),
  * Dataset discovery/validation (for quick feedback),
  * Creation of the run directory structure used by later stages.

Notes
-----
* Stage A is intentionally "shallow but decisive": we don't touch model code
  or images — we only prepare the ground truth for subsequent stages.
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

    Design Notes
    ------------
    * The timestamped `run_dir` makes every execution self-contained and
      immutable — later stages (training/eval) write under this directory.
    * This class remains thin and declarative; any heavy logic belongs to
      specialized components (ConfigStore, DatasetManager, RunLogger, etc.).
    """

    def __init__(self, cfg: Config):
        """
        Args
        ----
        cfg : Config
            Validated configuration object produced by ConfigStore.

        Notes
        -----
        * We derive a unique timestamped run directory to keep results
          append-only and easy to diff across runs.
        """
        
        self.cfg = cfg
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.run_dir = Path(self.cfg.paths["results_root"]) / self.cfg.model_name_out / f"run_{ts}"
        self.cfg_dir = self.run_dir / "cfg"
        self.logger = RunLogger(self.run_dir)

    def prepare(self) -> None:
        """Prepare a new run directory and perform Stage A tasks.

        Flow
        ----
        1) Create `run_dir` and `cfg/`.
        2) Snapshot the effective configuration and environment.
        3) Verify dataset structure and counts; write a JSON report.
        4) Create empty `train/` and `eval/` folders for later stages.
        5) Emit a human-readable summary (counts + warnings).

        Writes
        ------
        - run_dir/cfg/config_snapshot.yaml
        - run_dir/cfg/env_and_cfg.json
        - run_dir/cfg/dataset_report.json
        - run_dir/cfg/prepare_summary.json
        - run_dir/cfg/metrics.json (timings; via RunLogger.time_block in future)

        Raises
        ------
        AssertionError
            If critical preconditions are violated (not expected in Stage A;
            config validation happens earlier in ConfigStore).

        Notes
        -----
        * We do not fail on empty image folders — instead we record warnings.
        * This method is idempotent with respect to filesystem creation.
        """
        
        # Ensure run_dir exists; keep Stage A idempotent regarding directory creation
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # 1) Snapshot config + env for strong provenance/auditing
        ConfigStore.save_snapshot(self.cfg_dir, self.cfg)
        self.logger.log_env_and_cfg({
            "model_name_out": self.cfg.model_name_out,
            "paths": self.cfg.paths,
            "labels": self.cfg.labels,
            "train": self.cfg.train,
            "eval": self.cfg.eval,
            "system": self.cfg.system,
        }, self.cfg_dir)

        # 2) Verify dataset structure & counts (fast, path-level only)
        dm = DatasetManager(self.cfg.paths, self.cfg.labels, self.cfg_dir)
        report = dm.verify_structure()

        # 3) Create placeholders for future pipeline phases
        (self.run_dir / "train").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "eval").mkdir(parents=True, exist_ok=True)

        # 4) Human-readable summary with the most actionable fields
        summary = {
            "run_dir": str(self.run_dir),
            "n_train_images": report.get("n_train_images", 0),
            "n_valid_images": report.get("n_valid_images", 0),
            "n_train_with_labels": report.get("n_train_with_labels", 0),
            "warnings": report.get("warnings", []),
        }
        (self.cfg_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2))

    def get_run_dir(self) -> Path:
        """Return the absolute Path to the prepared run directory.

        Returns
        -------
        Path
            The timestamped run directory for this execution.
        """
        return self.run_dir
