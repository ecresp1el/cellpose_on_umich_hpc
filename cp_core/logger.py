"""
RunLogger — Stage A
Structured logging, env capture, and timing contexts.

Purpose
-------
Provide small, reliable primitives for:
  * capturing the runtime environment + effective config (JSON),
  * mirroring stdout/stderr to a file (without changing calling code),
  * measuring elapsed time for named blocks and persisting metrics.

Notes
-----
* Stage A writes metrics under run_dir/cfg/metrics.json.
  Later stages (B/C) can write to train/ or eval/ metrics as needed.
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

    Design Notes
    ------------
    * We keep this class free of pipeline logic — it's a generic utility.
    * `tee_stdout` is a non-invasive way to persist prints; callers don't
      need to change their `print()` calls.
    * `time_block` accumulates timings and persists them in a simple JSON
      that can be appended to across phases/runs.
    """
    def __init__(self, run_dir: Path):
        """
        Args
        ----
        run_dir : Path
            Root directory for the current pipeline run
            (e.g., results/<model>/run_YYYY-mm-dd_HHMMSS/).
        """
        self.run_dir = run_dir
        self._metrics: Dict[str, float] = {}

    def log_env_and_cfg(self, cfg: Dict[str, Any], dst_dir: Path) -> None:
        """Write environment snapshot + effective config to JSON.

        Args
        ----
        cfg : dict
            Already-validated configuration mapping (serializable).
        dst_dir : Path
            Destination directory for `env_and_cfg.json`.

        Writes
        ------
        dst_dir/env_and_cfg.json

        Notes
        -----
        * We record only a focused subset of environment variables:
          - SLURM* (scheduler context)
          - CUDA_VISIBLE_DEVICES, OMP_NUM_THREADS (compute context)
        * Keeping the JSON small makes it easier to diff across runs.
        """
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
        """Mirror stdout/stderr to `to_file` while preserving normal console prints.

        Args
        ----
        to_file : Path
            File path to append a live copy of stdout/stderr.

        Writes
        ------
        Appends to `to_file`.

        Notes
        -----
        * Implemented with a tiny file-backed Tee object.
        * Safe for nested contexts, but best used at a coarse scope
          (e.g., around an entire training/eval step).
        """
        to_file.parent.mkdir(parents=True, exist_ok=True)
        old_out, old_err = sys.stdout, sys.stderr
        with open(to_file, "a") as f:
            class Tee:
                def write(self, s):
                     # Write to file first, then to the original stream.
                    f.write(s)
                    old_out.write(s)
                def flush(self):
                    f.flush()
                    old_out.flush()
        
            # Mirror both stdout and stderr through the Tee.
            sys.stdout = sys.stderr = Tee()
            try:
                yield
            finally:
                # Always restore original streams.
                sys.stdout, sys.stderr = old_out, old_err

    @contextmanager
    def time_block(self, name: str):
        """Measure elapsed wall time for a named block and persist it.

        Args
        ----
        name : str
            A unique label for the measured block (e.g., "prepare",
            "list_images", "save_snapshot").

        Writes
        ------
        run_dir/cfg/metrics.json   (Stage A convention)

        Notes
        -----
        * We store metrics as a flat mapping {name: seconds}.
        * If the metrics file already exists, we load/merge to avoid losing
          earlier entries (best-effort; errors fall back to a fresh dict).
        """
        
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self._metrics[name] = dt
            # Stage A: write under cfg/. Later stages may override the path.
            metrics_path = self.run_dir / "cfg" / "metrics.json"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            existing = {}
            if metrics_path.exists():
                try:
                    existing = json.loads(metrics_path.read_text())
                except Exception:
                    # If the file is corrupt or unparsable, start fresh
                    existing = {}
            existing[name] = dt
            metrics_path.write_text(json.dumps(existing, indent=2))
