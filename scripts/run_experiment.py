#!/usr/bin/env python3
"""
scripts/run_experiment.py — Stage A entrypoint
------------------------------------------------
Top-level CLI wrapper for the Cellpose3 Whole-Organoid pipeline.
This script is intentionally thin — it just:
  1. Parses command-line arguments,
  2. Loads a validated Config object,
  3. Instantiates WholeOrganoidExperiment,
  4. Executes the requested stage (Stage A: prepare).

Usage examples
--------------
# Prepare a new run (Stage A)
python scripts/run_experiment.py \
    --config configs/cp3_v001.yaml \
    --mode prepare

Notes
-----
* The same entrypoint will later dispatch to training (Stage B)
  and evaluation (Stage C) once implemented.
* `--mode prepare` is currently the only valid option.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

# -------------------------------------------------------------
# Ensure repo root is on the import path
# -------------------------------------------------------------
# This allows imports like `from cp_core.config_store import ConfigStore`
# to work when the script is executed directly, rather than as a module.
# e.g., so we can run "python scripts/run_experiment.py" from anywhere.
sys.path.append(str(Path(__file__).resolve().parents[1]))

# -------------------------------------------------------------
# Internal imports — Stage A modules only
# -------------------------------------------------------------
from cp_core.config_store import ConfigStore
from cp_core.experiment import WholeOrganoidExperiment

def parse_args():
    """
    Parse CLI arguments for the pipeline entrypoint.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with fields:
        - config : str  -> absolute or relative path to YAML config.
        - mode   : str  -> workflow stage to run ("prepare" only for Stage A).

    Raises
    ------
    SystemExit
        If required args are missing or invalid.
    """
    p = argparse.ArgumentParser(
        description="Whole-Organoid Pipeline Entrypoint (Stage A)."
    )
    p.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file (version-controlled).",
    )
    p.add_argument(
        "--mode",
        choices=["prepare"],  # future: ["prepare","train","eval"]
        default="prepare",
        help="Stage of the workflow to execute. Currently only 'prepare'.",
    )
    return p.parse_args()

def main():
    """
    Main CLI routine — orchestrates Stage A preparation.

    Flow
    ----
    1. Parse CLI args and load YAML config using ConfigStore.
    2. Instantiate WholeOrganoidExperiment with the validated config.
    3. Run .prepare() to:
         - create a timestamped run_dir,
         - snapshot config/env,
         - verify dataset structure,
         - write reports under run_dir/cfg/.
    4. Print the final run directory path to stdout.

    Writes
    ------
    - results/<model_name>/run_YYYY-mm-dd_HHMMSS/cfg/*
      (see contract for exact list)

    Notes
    -----
    * All errors from config validation or dataset structure
      propagate upward so that SLURM captures them in the .err file.
    * Printing here is minimal by design — structured logs
      are written inside WholeOrganoidExperiment.prepare().
    """
    # Parse command-line arguments
    a = parse_args()
    
    # Load validated configuration (raises if YAML missing or malformed)
    cfg = ConfigStore.load_from_yaml(a.config)
    
    # Initialize experiment wrapper with config
    exp = WholeOrganoidExperiment(cfg)
    
    # Execute Stage A preparation workflow
    exp.prepare()

    # Minimal console feedback (everything else logged in cfg/)
    print(f"[Stage A] Prepared run directory: {exp.get_run_dir()}")

# -------------------------------------------------------------
# Standard Python entrypoint guard
# -------------------------------------------------------------
# Ensures the script executes main() only when called directly
# (not when imported as a module inside notebooks or tests).
if __name__ == "__main__":
    main()
