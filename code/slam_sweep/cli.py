"""
slam-sweep CLI:

    slam-sweep run-once   --bag PATH --default-config DIR ...
        Run GLIM once with the unmodified default config and report the
        loop-closure error. Useful as a baseline before launching a sweep.

    slam-sweep export-ply --rep-dir runs/<run_id>/rep_NN --out map.ply
        Reconstruct a point-cloud PLY from a kept GLIM dump. Runs on the
        host using Open3D — the GLIM container is not involved.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import click

DEFAULT_IMAGE = "koide3/glim_ros2:jazzy_cuda13.1"


# -----------------------------------------------------------------------------
# slam-sweep run-once
# -----------------------------------------------------------------------------

@click.group()
def main() -> None:
    """slam-sweep: orchestration helpers for GLIM parameter sweeps."""


@main.command("run-once")
@click.option("--bag", "bag_str", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Absolute path to the .mcap rosbag.")
@click.option("--default-config", "default_config_str", type=click.Path(exists=True, file_okay=False), required=True,
              help="Path to the default GLIM config directory.")
@click.option("--runs-root", type=click.Path(file_okay=False), default="./runs", show_default=True,
              help="Where to write the per-run directory.")
@click.option("--image", default=DEFAULT_IMAGE, show_default=True,
              help="Docker image tag.")
@click.option("--reps", type=int, default=1, show_default=True,
              help="Number of repetitions.")
@click.option("--timeout-s", type=int, default=None,
              help="Per-rep timeout, seconds. Omit for no limit.")
@click.option("--failure-penalty-m", type=float, default=100.0, show_default=True)
@click.option("--runaway-threshold-m", type=float, default=500.0, show_default=True)
@click.option("--keep-dumps/--no-keep-dumps", default=False, show_default=True,
              help="Keep the full GLIM dump (large) per rep.")
@click.option("--display/--no-display", default=False, show_default=True,
              help="Forward $DISPLAY + X11 socket so the GLFW viewer can open. "
                   "Requires libstandard_viewer.so to be enabled in config.json. "
                   "Use only for interactive debugging — leave off for sweeps.")
@click.option("--params", "params_json", default="{}", show_default=True,
              help="JSON dict of parameter overrides (same keys as PARAM_MAP).")
def run_once(bag_str, default_config_str, runs_root, image, reps,
             timeout_s, failure_penalty_m, runaway_threshold_m, keep_dumps,
             display, params_json):
    """Run one trial without W&B. Prints the aggregate metrics."""
    import argparse
    from .agent import evaluate_trial

    params = json.loads(params_json)
    args = argparse.Namespace(
        bag=bag_str,
        default_config=default_config_str,
        runs_root=runs_root,
        image=image,
        reps=reps,
        timeout_s=timeout_s,
        failure_penalty_m=failure_penalty_m,
        runaway_threshold_m=runaway_threshold_m,
        keep_dumps=keep_dumps,
        use_display=display,
    )
    run_id = time.strftime("local-%Y%m%d-%H%M%S")
    summary = evaluate_trial(args, params, run_id)
    click.echo(json.dumps(summary, indent=2))


# -----------------------------------------------------------------------------
# slam-sweep optuna-run
# -----------------------------------------------------------------------------

@main.command("optuna-run")
@click.option("--bag", "bag_str", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Absolute path to the .mcap rosbag.")
@click.option("--default-config", "default_config_str", type=click.Path(exists=True, file_okay=False), required=True,
              help="Path to the default GLIM config directory.")
@click.option("--wandb-project", required=True, help="W&B project name.")
@click.option("--wandb-entity", default=None, help="W&B entity (team or user). Defaults to your W&B default.")
@click.option("--runs-root", type=click.Path(file_okay=False), default="./runs", show_default=True,
              help="Where to write per-run directories and the SQLite study file.")
@click.option("--image", default=DEFAULT_IMAGE, show_default=True, help="Docker image tag.")
@click.option("--reps", type=int, default=3, show_default=True, help="Repetitions per trial.")
@click.option("--timeout-s", type=int, default=None, help="Per-rep timeout, seconds. Omit for no limit.")
@click.option("--n-trials", type=int, default=100, show_default=True, help="Total Optuna trials to run.")
@click.option("--study-name", default="glim_sweep", show_default=True,
              help="Optuna study name; also used as the SQLite filename (<runs-root>/<study-name>.db).")
@click.option("--failure-penalty-m", type=float, default=100.0, show_default=True)
@click.option("--runaway-threshold-m", type=float, default=500.0, show_default=True)
@click.option("--keep-dumps/--no-keep-dumps", default=False, show_default=True,
              help="Keep the full GLIM dump (large) per rep.")
@click.option("--display/--no-display", default=False, show_default=True,
              help="Forward $DISPLAY + X11 socket into the container.")
@click.option("--tags", default="", show_default=False,
              help="Extra W&B tags, comma-separated (e.g. 'outdoor,building_A'). "
                   "The bag filename stem is always added automatically.")
def optuna_run(bag_str, default_config_str, wandb_project, wandb_entity, runs_root,
               image, reps, timeout_s, n_trials, study_name,
               failure_penalty_m, runaway_threshold_m, keep_dumps, display, tags):
    """Run a conditional Bayesian sweep with Optuna (TPE sampler) + W&B logging."""
    import argparse
    from .optuna_agent import run_study

    args = argparse.Namespace(
        bag=bag_str,
        default_config=default_config_str,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        runs_root=runs_root,
        image=image,
        reps=reps,
        timeout_s=timeout_s,
        n_trials=n_trials,
        study_name=study_name,
        failure_penalty_m=failure_penalty_m,
        runaway_threshold_m=runaway_threshold_m,
        keep_dumps=keep_dumps,
        use_display=display,
        tags=[t.strip() for t in tags.split(",") if t.strip()],
    )
    run_study(args)


# -----------------------------------------------------------------------------
# slam-sweep export-ply
# -----------------------------------------------------------------------------

def _parse_data_txt(file_path):
    """Extract T_world_origin matrix from a GLIM submap data.txt."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "T_world_origin:" in line:
            matrix_lines = lines[i + 1:i + 5]
            values = [float(x) for row in matrix_lines for x in row.split()]
            return np.array(values).reshape(4, 4)
    return np.eye(4)


def _save_ply(filename, points, intensities):
    """Write a binary little-endian PLY with x, y, z, intensity."""
    n = len(points)
    header = (
        f"ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        f"property float x\nproperty float y\nproperty float z\n"
        f"property float intensity\nend_header\n"
    ).encode()
    data = np.column_stack((points, intensities)).astype(np.float32)
    with open(filename, 'wb') as f:
        f.write(header)
        f.write(data.tobytes())


@main.command("export-ply")
@click.option("--rep-dir", type=click.Path(exists=True, file_okay=False), required=True,
              help="A `runs/<run_id>/rep_NN` directory containing an `output/` dump.")
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=True,
              help="Output PLY path.")
def export_ply(rep_dir, out_path):
    """Build a PLY point cloud from a kept GLIM dump."""
    rep_dir = Path(rep_dir).resolve()
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output_dir = rep_dir / "output"
    if not output_dir.is_dir():
        sys.exit(f"No GLIM dump found at {output_dir}. "
                 f"Re-run with --keep-dumps to preserve it.")

    submap_dirs = sorted(glob.glob(str(output_dir / ("[0-9]" * 6))))
    if not submap_dirs:
        sys.exit(f"No submap directories found under {output_dir}.")

    click.echo(f"Found {len(submap_dirs)} submap(s), reading...")

    all_points = []
    all_intensities = []

    for subdir in submap_dirs:
        data_path   = os.path.join(subdir, "data.txt")
        points_path = os.path.join(subdir, "points_compact.bin")
        inten_path  = os.path.join(subdir, "intensities_compact.bin")

        if not (os.path.exists(data_path) and os.path.exists(points_path)):
            continue

        T = _parse_data_txt(data_path)
        points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 3)
        intensities = (np.fromfile(inten_path, dtype=np.float32)
                       if os.path.exists(inten_path)
                       else np.zeros(len(points), dtype=np.float32))

        homog = np.hstack([points, np.ones((len(points), 1))])
        transformed = (T @ homog.T).T[:, :3]

        all_points.append(transformed)
        all_intensities.append(intensities)

    if not all_points:
        sys.exit(f"No valid submap data found in {output_dir}.")

    final_points = np.concatenate(all_points, axis=0)
    final_intensities = np.concatenate(all_intensities, axis=0)

    click.echo(f"Writing {len(final_points):,} points to {out_path}...")
    _save_ply(str(out_path), final_points, final_intensities)
    click.echo(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
