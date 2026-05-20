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
import sys
import time
from pathlib import Path

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
def optuna_run(bag_str, default_config_str, wandb_project, wandb_entity, runs_root,
               image, reps, timeout_s, n_trials, study_name,
               failure_penalty_m, runaway_threshold_m, keep_dumps, display):
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
    )
    run_study(args)


# -----------------------------------------------------------------------------
# slam-sweep export-ply
# -----------------------------------------------------------------------------

@main.command("export-ply")
@click.option("--rep-dir", type=click.Path(exists=True, file_okay=False), required=True,
              help="A `runs/<run_id>/rep_NN` directory containing an `output/` dump.")
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=True,
              help="Output PLY path.")
@click.option("--voxel-size", type=float, default=None,
              help="Optional voxel-grid downsample size (m). Skip for full resolution.")
def export_ply(rep_dir, out_path, voxel_size):
    """
    Build a single PLY point cloud from a kept GLIM dump.

    GLIM's `dump_path` stores per-submap point clouds (`*.ply`). The
    optimized poses are already baked into each submap cloud at dump
    time, so concatenation is sufficient — no graph re-application
    needed for a static map output.

    Runs on the host. Open3D must be installed in the orchestrator's
    Python environment (`pip install open3d`).
    """
    try:
        import open3d as o3d
    except ImportError:
        sys.exit("Open3D not installed. `pip install open3d` in the orchestrator's venv.")

    rep_dir = Path(rep_dir).resolve()
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output_dir = rep_dir / "output"
    if not output_dir.is_dir():
        sys.exit(f"No GLIM dump found at {output_dir}. "
                 f"Re-run the trial with --keep-dumps to preserve it.")

    ply_files = sorted(glob.glob(str(output_dir / "**" / "*.ply"), recursive=True))
    if not ply_files:
        sys.exit(f"No .ply files found under {output_dir}.")

    click.echo(f"Reading {len(ply_files)} submap cloud(s)...")
    merged = o3d.geometry.PointCloud()
    total = 0
    for p in ply_files:
        pcd = o3d.io.read_point_cloud(p)
        n = len(pcd.points)
        if n == 0:
            continue
        merged += pcd
        total += n

    if total == 0:
        sys.exit("All dump PLYs were empty.")

    click.echo(f"Concatenated: {total:,} points")

    if voxel_size is not None and voxel_size > 0:
        before = len(merged.points)
        merged = merged.voxel_down_sample(voxel_size)
        click.echo(f"Voxel-downsampled @ {voxel_size} m: {before:,} -> {len(merged.points):,}")

    o3d.io.write_point_cloud(str(out_path), merged, write_ascii=False)
    click.echo(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
