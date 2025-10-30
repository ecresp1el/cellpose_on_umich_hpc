#!/usr/bin/env python3
"""Helper to launch Cellpose from a YAML configuration."""
import argparse
import os
import shlex
import subprocess
from pathlib import Path
from typing import List, Union

import yaml


def _as_list(value: Union[str, List[str], None]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration at {path} must be a mapping/dict.")
    return data


def expand_path(value: str, context: dict) -> Path:
    expanded = os.path.expanduser(os.path.expandvars(value))
    expanded = expanded.format(**context)
    return Path(expanded).resolve()


def build_command(cfg: dict, context: dict) -> List[str]:
    cmd: List[str] = ["python", "-m", "cellpose"]

    input_dir = expand_path(cfg["input_dir"], context)
    cmd.extend(["--dir", str(input_dir)])

    img_filters = _as_list(cfg.get("img_filter", []))
    for token in img_filters:
        if token:
            cmd.extend(["--img_filter", token])

    if cfg.get("look_one_level_down"):
        cmd.append("--look_one_level_down")

    if cfg.get("use_gpu"):
        cmd.append("--use_gpu")
    gpu_device = cfg.get("gpu_device")
    if gpu_device not in (None, "", "auto"):
        cmd.extend(["--gpu_device", str(gpu_device)])

    pretrained_model = cfg.get("pretrained_model")
    if pretrained_model:
        cmd.extend(["--pretrained_model", str(pretrained_model)])

    channel_axis = cfg.get("channel_axis")
    if channel_axis is not None:
        cmd.extend(["--channel_axis", str(channel_axis)])

    z_axis = cfg.get("z_axis")
    if z_axis is not None:
        cmd.extend(["--z_axis", str(z_axis)])

    diameter = cfg.get("diameter")
    if diameter is not None:
        cmd.extend(["--diameter", str(diameter)])

    output_dir = expand_path(cfg["output_dir"], context)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd.extend(["--savedir", str(output_dir)])

    for extra in _as_list(cfg.get("extra_args", [])):
        cmd.append(str(extra))

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Cellpose from a YAML config.")
    parser.add_argument(
        "--config",
        default="config/cellpose_job.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without executing Cellpose.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    context = {"run_name": cfg.get("run_name", "cellpose_run")}

    command = build_command(cfg, context)
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"[cellpose] Command: {printable}")

    if args.dry_run:
        print("[cellpose] Dry run: not executing command.")
        return

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
