"""
Generate per-trial GLIM configs by patching the default config tree.

GLIM resolves its parameters from a directory of JSON files:

    config.json                  # master, points at the others
    config_ros.json
    config_sensors.json
    config_odometry_gpu.json     # frontend
    config_sub_mapping_gpu.json  # sub-mapping
    config_global_mapping_gpu.json   # global mapping

The orchestrator copies this whole directory to a per-run location, mutates
specified leaves, and mounts the result as `/config` inside the container.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping


# -----------------------------------------------------------------------------
# Parameter mapping
# -----------------------------------------------------------------------------
# Maps flat W&B parameter names (e.g. "frontend.voxel_resolution") to a
# (filename, dotted-JSON-path) tuple inside the config directory.
#
# Update this map to expose more parameters to the sweep. Verify each key
# against the actual JSON files in your default config — GLIM is silent
# about unknown keys, which makes this a common silent-failure mode (the
# whole reason the previous bash sweep "did nothing").
#
# Filenames here assume the GPU presets (CUDA build), as configured by
# `config.json` -> `"config_odometry": "config_odometry_gpu.json"` etc.
# Adjust if you use the CPU presets.

PARAM_MAP: dict[str, tuple[str, str]] = {
    # Frontend / odometry
    "frontend.voxel_resolution":            ("config_odometry_gpu.json", "voxel_resolution"),
    "frontend.voxelmap_levels":             ("config_odometry_gpu.json", "voxelmap_levels"),
    "frontend.max_correspondence_distance": ("config_odometry_gpu.json", "max_correspondence_distance"),
    "frontend.max_num_keyframes":           ("config_odometry_gpu.json", "max_num_keyframes"),
    "frontend.keyframe_update_strategy":    ("config_odometry_gpu.json", "keyframe_update_strategy"),
    "frontend.keyframe_max_overlap":        ("config_odometry_gpu.json", "keyframe_max_overlap"),
    "frontend.keyframe_delta_trans":        ("config_odometry_gpu.json", "keyframe_delta_trans"),
    "frontend.keyframe_delta_rot":          ("config_odometry_gpu.json", "keyframe_delta_rot"),
    "frontend.registration_type":           ("config_odometry_gpu.json", "registration_type"),

    # Sub-mapping
    "sub_mapping.min_implicit_loop_overlap": ("config_sub_mapping_gpu.json", "min_implicit_loop_overlap"),
    "sub_mapping.submap_voxel_resolution":   ("config_sub_mapping_gpu.json", "submap_voxel_resolution"),
    "sub_mapping.keyframe_voxel_resolution": ("config_sub_mapping_gpu.json", "keyframe_voxel_resolution"),

    # Global mapping
    "global_mapping.submap_voxel_resolution":   ("config_global_mapping_gpu.json", "submap_voxel_resolution"),
    "global_mapping.enable_optimization":       ("config_global_mapping_gpu.json", "enable_optimization"),
    "global_mapping.registration_error_factor_type": ("config_global_mapping_gpu.json", "registration_error_factor_type"),
}


# -----------------------------------------------------------------------------
# Implementation
# -----------------------------------------------------------------------------

def _set_dotted(d: dict, dotted_key: str, value: Any) -> None:
    """Write `value` at the dotted JSON path `dotted_key`, creating dicts as needed."""
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def materialize_config(
    default_dir: Path,
    target_dir: Path,
    params: Mapping[str, Any],
    *,
    strict: bool = False,
) -> Path:
    """
    Copy `default_dir` -> `target_dir` and patch leaf values in JSON files
    according to `params` (resolved through PARAM_MAP).

    Args:
        default_dir: directory containing GLIM's default config JSONs.
        target_dir:  destination; will be removed and recreated.
        params:      flat dict like {"frontend.voxel_resolution": 0.5, ...}.
                     Keys not in PARAM_MAP are silently skipped (W&B adds
                     internal metadata keys like _wandb).
        strict:      if True, raise on any unknown key instead of skipping.

    Returns:
        The materialized target_dir (Path).
    """
    default_dir = Path(default_dir)
    target_dir = Path(target_dir)

    if not default_dir.is_dir():
        raise FileNotFoundError(f"Default config directory not found: {default_dir}")
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(default_dir, target_dir)

    # Group patches by file so each JSON is loaded and written exactly once.
    by_file: dict[str, dict[str, Any]] = {}
    for flat_key, value in params.items():
        if flat_key not in PARAM_MAP:
            if strict and not flat_key.startswith("_"):
                raise KeyError(f"Unknown sweep parameter: {flat_key}")
            continue
        fname, dotted = PARAM_MAP[flat_key]
        by_file.setdefault(fname, {})[dotted] = value

    for fname, leafs in by_file.items():
        path = target_dir / fname
        if not path.is_file():
            raise FileNotFoundError(
                f"Cannot patch {fname!r}: not found in {default_dir}. "
                f"Either your default config tree is missing this file, or "
                f"PARAM_MAP refers to a file that doesn't exist for your "
                f"GLIM build (e.g. CPU vs GPU preset)."
            )
        with path.open("r") as f:
            cfg = json.load(f)
        for dotted, value in leafs.items():
            _set_dotted(cfg, dotted, value)
        with path.open("w") as f:
            json.dump(cfg, f, indent=2)

    return target_dir
