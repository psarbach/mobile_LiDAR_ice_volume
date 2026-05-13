"""
Parse GLIM's TUM-format trajectory dump and compute a loop-closure error.

GLIM writes (per the official quickstart):

    odom_lidar.txt   : LiDAR-frame trajectory, no loop closure
    traj_lidar.txt   : LiDAR-frame trajectory, with loop closure (preferred)
    odom_imu.txt     : IMU-frame trajectory, no loop closure
    traj_imu.txt     : IMU-frame trajectory, with loop closure

All are TUM format: each row is `t tx ty tz qx qy qz qw`, whitespace-separated.

For closed-loop datasets (the scanner returns to its start), the Euclidean
distance between the first and last positions in `traj_lidar.txt` is a
useful proxy for trajectory quality. For open trajectories, swap in your
own metric (APE/RPE against ground truth, etc.).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# Order matters: prefer the loop-closed LiDAR trajectory.
TRAJECTORY_CANDIDATES = (
    "traj_lidar.txt",
    "traj_imu.txt",
    "odom_lidar.txt",
    "odom_imu.txt",
)


def find_trajectory(dump_dir: Path) -> Path | None:
    """Return the first non-empty TUM trajectory file in `dump_dir`, or None."""
    dump_dir = Path(dump_dir)
    for name in TRAJECTORY_CANDIDATES:
        p = dump_dir / name
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def load_tum_trajectory(path: Path) -> np.ndarray:
    """Load a TUM-format trajectory as an Nx8 array `[t x y z qx qy qz qw]`."""
    arr = np.loadtxt(path, comments=["#"])
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 4:
        raise ValueError(
            f"Trajectory {path} has only {arr.shape[1]} columns; expected ≥4."
        )
    return arr


def loop_closure_error(traj: np.ndarray) -> float:
    """Euclidean distance between the start and end positions, in meters."""
    if traj.shape[0] < 2:
        return float("nan")
    start = traj[0, 1:4]
    end = traj[-1, 1:4]
    return float(np.linalg.norm(end - start))


def trajectory_length(traj: np.ndarray) -> float:
    """Total path length, useful as a sanity check (not the objective)."""
    if traj.shape[0] < 2:
        return 0.0
    deltas = np.diff(traj[:, 1:4], axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())
