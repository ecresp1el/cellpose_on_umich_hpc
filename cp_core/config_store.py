"""
ConfigStore — Stage A
Canonical loader/validator for pipeline configuration.
Adheres to the contract in CONTRACT_Methods_Cellpose3_WholeOrganoid_Pipeline.md.

Docstring template conventions:
- Summary, Args, Returns, Writes, Raises, Notes, Checks
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import os
import sys
import json
# PyYAML is the canonical YAML parser we depend on for Stage A config loading.
# We treat it as a hard requirement during runtime (see load_from_yaml guard).
try:
    import yaml  # PyYAML
except Exception as e:
    yaml = None

@dataclass
class Config:
    """Structured configuration (minimal Stage A subset).

    Summary
    -------
    Lightweight container for the YAML-derived configuration used across
    the pipeline. We keep nested sections as dicts for flexibility and to
    avoid premature schema rigidity in Stage A.

    Args
    ----
    model_name_out : str
        Name/version for results folder (e.g., "cp3_v001").
    paths : dict[str, Any]
        Mapping with required absolute directories (see REQUIRED_PATH_KEYS).
    labels : dict[str, Any]
        Label conventions (e.g., `mask_filter: _cp_masks.png`).
    train : dict[str, Any]
        Training knobs (used in Stage B; included here for snapshotting).
    eval : dict[str, Any]
        Evaluation knobs (used in Stage C; included here for snapshotting).
    system : dict[str, Any]
        System options (e.g., seed, use_cuda) for reproducibility.

    Notes
    -----
    * Stage A validates only the presence and existence of key paths.
    * We snapshot the entire dict (incl. train/eval) for provenance even if
      Stage A does not use them yet.
    """
    
    model_name_out: str
    paths: Dict[str, Any]
    labels: Dict[str, Any]
    train: Dict[str, Any]
    eval: Dict[str, Any]
    system: Dict[str, Any]

class ConfigStore:
    """Load and validate YAML config files.

    Methods
    -------
    load_from_yaml(path: str) -> Config
        Load YAML into Config; validate required keys and directories.
    save_snapshot(dst_dir: Path, cfg: Config) -> None
        Write `config_snapshot.yaml` and `env_and_cfg.json` under dst_dir.
    """

    REQUIRED_PATH_KEYS = [
        "turbo_root",
        "data_images_train",
        "data_labels_train",
        "data_images_valid",
        "data_labels_valid",
        "results_root",
        "logs_root",
    ]

    @staticmethod
    def load_from_yaml(path: str) -> Config:
        """Load YAML configuration and validate directory paths.

        Args:
            path: Path to YAML file.

        Returns:
            Config: validated configuration object.

        Writes:
            None

        Raises:
            FileNotFoundError: if YAML path does not exist.
            RuntimeError: if PyYAML is unavailable.
            ValueError: if required keys/dirs are missing.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config YAML not found: {p}")
        if yaml is None:
            raise RuntimeError("PyYAML is required. Please `pip install pyyaml`.")

        with p.open("r") as f:
            data = yaml.safe_load(f)

        # Basic shape checks
        for key in ["model_name_out", "paths", "labels", "train", "eval", "system"]:
            if key not in data:
                raise ValueError(f"Missing top-level key in config YAML: '{key}'")

        # Validate required path keys exist and are absolute directories
        paths = data["paths"]
        for k in ConfigStore.REQUIRED_PATH_KEYS:
            if k not in paths:
                raise ValueError(f"Missing 'paths.{k}' in config YAML.")
            if not Path(paths[k]).is_absolute():
                raise ValueError(f"'paths.{k}' must be absolute: {paths[k]}")
            # For Stage A, only ensure parent directories exist; images may be empty early on
            # We assert existence for the top-level containers.
            container_keys = ["turbo_root", "results_root", "logs_root"]
            if k in container_keys and not Path(paths[k]).exists():
                raise ValueError(f"Directory does not exist: paths.{k} -> {paths[k]}")

        cfg = Config(
            model_name_out=data["model_name_out"],
            paths=data["paths"],
            labels=data.get("labels", {}),
            train=data.get("train", {}),
            eval=data.get("eval", {}),
            system=data.get("system", {}),
        )
        return cfg

    @staticmethod
    def save_snapshot(dst_dir: Path, cfg: Config) -> None:
        """Persist config snapshot and environment metadata under dst_dir.

        Args:
            dst_dir: Directory to write snapshot files.
            cfg: Config object to serialize.

        Writes:
            - `dst_dir/config_snapshot.yaml`
            - `dst_dir/env_and_cfg.json`

        Raises:
            OSError: on IO errors.
        """
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Re-dump YAML using json->yaml if PyYAML present
        cfg_yaml = {
            "model_name_out": cfg.model_name_out,
            "paths": cfg.paths,
            "labels": cfg.labels,
            "train": cfg.train,
            "eval": cfg.eval,
            "system": cfg.system,
        }

        if yaml is not None:
            (dst_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg_yaml, sort_keys=False))
        else:
            # Fallback JSON if PyYAML not present
            (dst_dir / "config_snapshot.yaml").write_text(json.dumps(cfg_yaml, indent=2))

        # Env metadata
        env_meta = {
            "python_version": sys.version,
            "executable": sys.executable,
            "platform": os.name,
            "cwd": str(Path.cwd()),
            "environ_subset": {k: v for k, v in os.environ.items() if k.startswith("SLURM") or k in ("CUDA_VISIBLE_DEVICES","OMP_NUM_THREADS")},
            "config": cfg_yaml,
        }
        (dst_dir / "env_and_cfg.json").write_text(json.dumps(env_meta, indent=2))
