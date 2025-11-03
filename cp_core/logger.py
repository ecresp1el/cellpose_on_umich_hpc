"""
RunLogger — Stage A
Structured logging, env capture, and timing contexts.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import os
import sys
import json
import time
from contextlib import contextmanager

class RunLogger:
    """Structured run logging utilities.

    Methods
    -------
    log_env_and_cfg(cfg: Dict[str, Any], dst_dir: Path) -> None
        Write environment + cfg snapshot JSON.
    tee_stdout(to_file: Path)
        Context manager to mirror stdout/stderr to file.
    time_block(name: str)
        Context manager to record block durations.
    """
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self._metrics: Dict[str, float] = {}

    def log_env_and_cfg(self, cfg: Dict[str, Any], dst_dir: Path) -> None:
        dst_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "python_version": sys.version,
            "executable": sys.executable,
            "platform": os.name,
            "cwd": str(Path.cwd()),
            "environ_subset": {k: v for k, v in os.environ.items() if k.startswith("SLURM") or k in ("CUDA_VISIBLE_DEVICES","OMP_NUM_THREADS")},
            "config": cfg,
        }
        (dst_dir / "env_and_cfg.json").write_text(json.dumps(data, indent=2))

    @contextmanager
    def tee_stdout(self, to_file: Path):
        to_file.parent.mkdir(parents=True, exist_ok=True)
        old_out, old_err = sys.stdout, sys.stderr
        with open(to_file, "a") as f:
            class Tee:
                def write(self, s):
                    f.write(s)
                    old_out.write(s)
                def flush(self):
                    f.flush()
                    old_out.flush()
            sys.stdout = sys.stderr = Tee()
            try:
                yield
            finally:
                sys.stdout, sys.stderr = old_out, old_err

    @contextmanager
    def time_block(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self._metrics[name] = dt
            # persist under train/metrics.json or cfg/metrics.json depending on phase
            # Stage A: write under cfg/
            metrics_path = self.run_dir / "cfg" / "metrics.json"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            existing = {}
            if metrics_path.exists():
                try:
                    existing = json.loads(metrics_path.read_text())
                except Exception:
                    existing = {}
            existing[name] = dt
            metrics_path.write_text(json.dumps(existing, indent=2))
