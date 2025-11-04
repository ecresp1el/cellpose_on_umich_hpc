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

    Design Notes
    ------------
    * This class centralizes all config I/O so other components can assume
      they receive a validated `Config`.
    * Validation is intentionally “shallow but decisive” in Stage A:
      we assert directory existence for containers and absoluteness for paths.
    """
    
    # Paths that must exist in the YAML (under `paths:`) and (for containers)
    # must exist on disk. Images themselves may be empty early on.
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

        Summary
        -------
        Reads the YAML file, checks presence of top-level keys and required
        path entries, ensures absolute paths, and verifies the existence of
        container directories (turbo_root, results_root, logs_root).

        Args
        ----
        path : str
            Path to YAML file.

        Returns
        -------
        Config
            Validated configuration object.

        Writes
        ------
        None

        Raises
        ------
        FileNotFoundError
            If the YAML path does not exist.
        RuntimeError
            If PyYAML is unavailable in the active environment.
        ValueError
            If required keys or required directories are missing.

        Checks
        ------
        * Top-level keys exist: model_name_out, paths, labels, train, eval, system
        * All REQUIRED_PATH_KEYS present under `paths:`
        * All `paths.*` are absolute; containers exist for turbo_root/results_root/logs_root
        """
        p = Path(path)
        if not p.exists():
            # Fail fast with a clear error so SLURM logs point to the actual cause.
            raise FileNotFoundError(f"Config YAML not found: {p}")
        if yaml is None:
            # Stage A requires PyYAML; installing in the job is acceptable,
            # but we prefer having it present in the env beforehand.
            raise RuntimeError("PyYAML is required. Please `pip install pyyaml`.")

        with p.open("r") as f:
            data = yaml.safe_load(f)

        # Basic shape checks: keep these consistent with the contract.
        for key in ["model_name_out", "paths", "labels", "train", "eval", "system"]:
            if key not in data:
                raise ValueError(f"Missing top-level key in config YAML: '{key}'")

        # Validate required path keys exist and are absolute directories.
        paths = data["paths"]
        for k in ConfigStore.REQUIRED_PATH_KEYS:
            if k not in paths:
                raise ValueError(f"Missing 'paths.{k}' in config YAML.")
            
            # Enforce absolute paths so jobs are location-agnostic and safer on HPC.
            if not Path(paths[k]).is_absolute():
                raise ValueError(f"'paths.{k}' must be absolute: {paths[k]}")
            
            # For Stage A, only ensure parent directories exist; images may be empty early on
            # We assert existence for the top-level containers.
            container_keys = ["turbo_root", "results_root", "logs_root"]
            if k in container_keys and not Path(paths[k]).exists():
                raise ValueError(f"Directory does not exist: paths.{k} -> {paths[k]}")

        # Construct the Config; keep nested dicts intact for snapshotting.
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

        Summary
        -------
        Writes two files into `dst_dir` for strong provenance:
        1) `config_snapshot.yaml` — the effective configuration used.
        2) `env_and_cfg.json`   — Python/SLURM environment + same config dict.

        Args
        ----
        dst_dir : pathlib.Path
            Directory to write snapshot files into (created if missing).
        cfg : Config
            Config object to serialize.

        Writes
        ------
        dst_dir/config_snapshot.yaml
        dst_dir/env_and_cfg.json

        Raises
        ------
        OSError
            On I/O errors (e.g., permission issues).

        Notes
        -----
        * If PyYAML is unavailable, we still write the snapshot as JSON text
          (so runs are never blocked from recording provenance).
        * We include an `environ_subset` focused on SLURM and core HPC vars
          to keep the JSON readable yet useful for auditing.
        """
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Prepare a serializable view of the config.
        cfg_yaml = {
            "model_name_out": cfg.model_name_out,
            "paths": cfg.paths,
            "labels": cfg.labels,
            "train": cfg.train,
            "eval": cfg.eval,
            "system": cfg.system,
        }

        # Prefer YAML for human readability, fall back to JSON if PyYAML is missing.
        if yaml is not None:
            (dst_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg_yaml, sort_keys=False))
        else:
            # Fallback JSON if PyYAML not present
            (dst_dir / "config_snapshot.yaml").write_text(json.dumps(cfg_yaml, indent=2))
        
        # Environment + config echo for reproducibility audits.    
        env_meta = {
            "python_version": sys.version,
            "executable": sys.executable,
            "platform": os.name,
            "cwd": str(Path.cwd()),
            "environ_subset": {k: v for k, v in os.environ.items() if k.startswith("SLURM") or k in ("CUDA_VISIBLE_DEVICES","OMP_NUM_THREADS")},
            "config": cfg_yaml,
        }
        (dst_dir / "env_and_cfg.json").write_text(json.dumps(env_meta, indent=2))
