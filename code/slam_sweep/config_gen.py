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
import warnings
import re
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
    # All params live under the "odometry_estimation" top-level key in the JSON.
    "frontend.voxel_resolution":            ("config_odometry_gpu.json", "odometry_estimation.voxel_resolution"),
    "frontend.voxelmap_levels":             ("config_odometry_gpu.json", "odometry_estimation.voxelmap_levels"),
    "frontend.max_correspondence_distance": ("config_odometry_gpu.json", "odometry_estimation.max_correspondence_distance"),
    "frontend.max_num_keyframes":           ("config_odometry_gpu.json", "odometry_estimation.max_num_keyframes"),
    "frontend.smoother_lag":               ("config_odometry_gpu.json", "odometry_estimation.smoother_lag"),
    "frontend.use_isam2_dogleg":           ("config_odometry_gpu.json", "odometry_estimation.use_isam2_dogleg"),
    "frontend.full_connection_window_size": ("config_odometry_gpu.json", "odometry_estimation.full_connection_window_size"),
    "frontend.keyframe_update_strategy":    ("config_odometry_gpu.json", "odometry_estimation.keyframe_update_strategy"),
    "frontend.keyframe_max_overlap":        ("config_odometry_gpu.json", "odometry_estimation.keyframe_max_overlap"),
    "frontend.keyframe_delta_trans":        ("config_odometry_gpu.json", "odometry_estimation.keyframe_delta_trans"),
    "frontend.keyframe_delta_rot":          ("config_odometry_gpu.json", "odometry_estimation.keyframe_delta_rot"),
    "frontend.registration_type":           ("config_odometry_gpu.json", "odometry_estimation.registration_type"),

    # Sub-mapping
    # All params live under the "sub_mapping" top-level key in the JSON.
    "sub_mapping.submap_voxel_resolution":   ("config_sub_mapping_gpu.json", "sub_mapping.submap_voxel_resolution"),
    "sub_mapping.keyframe_voxel_resolution": ("config_sub_mapping_gpu.json", "sub_mapping.keyframe_voxel_resolution"),
    "sub_mapping.enable_optimization":       ("config_sub_mapping_gpu.json", "sub_mapping.enable_optimization"),
    "sub_mapping.max_num_keyframes":         ("config_sub_mapping_gpu.json", "sub_mapping.max_num_keyframes"),

    # Global mapping
    # All params live under the "global_mapping" top-level key in the JSON.
    "global_mapping.submap_voxel_resolution":        ("config_global_mapping_gpu.json", "global_mapping.submap_voxel_resolution"),
    "global_mapping.enable_optimization":            ("config_global_mapping_gpu.json", "global_mapping.enable_optimization"),
    "global_mapping.registration_error_factor_type": ("config_global_mapping_gpu.json", "global_mapping.registration_error_factor_type"),
    "global_mapping.min_implicit_loop_overlap":      ("config_global_mapping_gpu.json", "global_mapping.min_implicit_loop_overlap"),
    "global_mapping.max_implicit_loop_distance":     ("config_global_mapping_gpu.json", "global_mapping.max_implicit_loop_distance"),
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


def flatten_params(obj: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """
    Recursively flatten a possibly-nested mapping into dotted keys.

    W&B treats dots in sweep parameter names as nesting separators, so a
    sweep declaring `frontend.voxel_resolution` delivers it back via
    `wandb.config` as `{"frontend": {"voxel_resolution": 0.5}}`. This
    helper restores the flat dotted form that PARAM_MAP uses as its keys.

    Both forms are handled (so we tolerate any future W&B SDK change):
      {"frontend": {"voxel_resolution": 0.5}}    -> {"frontend.voxel_resolution": 0.5}
      {"frontend.voxel_resolution": 0.5}         -> {"frontend.voxel_resolution": 0.5}

    Keys that already contain dots are treated as fully-qualified leaves
    and are NOT recursed into, so existing flat keys round-trip cleanly.
    """
    out: dict[str, Any] = {}
    for k, v in obj.items():
        # Skip internal W&B metadata.
        if k.startswith("_"):
            continue
        full = f"{prefix}{k}"
        if isinstance(v, dict) and "." not in k:
            out.update(flatten_params(v, prefix=f"{full}."))
        else:
            out[full] = v
    return out


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
        params:      mapping of parameter names to values. Either flat
                     dotted form (`{"frontend.voxel_resolution": 0.5}`)
                     or nested form (`{"frontend": {"voxel_resolution": 0.5}}`)
                     is accepted — the latter is what W&B delivers via
                     `wandb.config` because it interprets dots as nesting.
                     Internal keys starting with `_` are skipped.
        strict:      if True, raise on any unknown key. Recommended in
                     production: a silent skip means a sweep can spend
                     hours patching nothing.

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

    flat = flatten_params(params)

    # Group patches by file so each JSON is loaded and written exactly once.
    by_file: dict[str, dict[str, Any]] = {}
    skipped: list[str] = []
    for flat_key, value in flat.items():
        if flat_key not in PARAM_MAP:
            skipped.append(flat_key)
            continue
        fname, dotted = PARAM_MAP[flat_key]
        by_file.setdefault(fname, {})[dotted] = value

    if skipped:
        msg = (
            f"materialize_config: {len(skipped)} parameter(s) not in PARAM_MAP "
            f"and were not applied: {skipped}. Known keys: {sorted(PARAM_MAP.keys())}"
        )
        if strict:
            raise KeyError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    for fname, leafs in by_file.items():
        path = target_dir / fname
        if not path.is_file():
            raise FileNotFoundError(
                f"Cannot patch {fname!r}: not found in {default_dir}. "
                f"Either your default config tree is missing this file, or "
                f"PARAM_MAP refers to a file that doesn't exist for your "
                f"GLIM build (e.g. CPU vs GPU preset)."
            )
        #with path.open("r") as f:
            #cfg = json.load(f)
        #new version to delete all comments in the json files
        with path.open("r") as f:
            content = f.read()
            # Step 1: Remove multi-line blocks (/* ... */)
            # We use re.S here so the dot matches newlines within the block
            content = re.sub(r'/\*.*?\*/', '', content, flags=re.S)
            
            # Step 2: Remove single-line comments (// ...)
            # We do NOT use re.S here so that // only eats until the end of its line
            content = re.sub(r'//.*', '', content)

            cfg = json.loads(content)
        for dotted, value in leafs.items():
            _set_dotted(cfg, dotted, value)
        with path.open("w") as f:
            json.dump(cfg, f, indent=2)

    return target_dir
