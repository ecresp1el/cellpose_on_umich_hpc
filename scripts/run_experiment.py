#!/usr/bin/env python3
"""
scripts/run_experiment.py — Stage A entrypoint

Usage:
    python scripts/run_experiment.py --config configs/cp3_v001.yaml --mode prepare
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

# Allow running from repo root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from cp_core.config_store import ConfigStore
from cp_core.experiment import WholeOrganoidExperiment

def parse_args():
    p = argparse.ArgumentParser(description="Whole-Organoid Pipeline Entrypoint (Stage A).")
    p.add_argument("--config", required=True, help="Path to YAML config.")
    p.add_argument("--mode", choices=["prepare"], default="prepare", help="Stage to run (Stage A).")
    return p.parse_args()

def main():
    a = parse_args()
    cfg = ConfigStore.load_from_yaml(a.config)
    exp = WholeOrganoidExperiment(cfg)
    exp.prepare()
    print(f"[Stage A] Prepared run directory: {exp.get_run_dir()}")

if __name__ == "__main__":
    main()
