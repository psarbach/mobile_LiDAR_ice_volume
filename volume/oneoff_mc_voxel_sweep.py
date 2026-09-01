"""
ONE-OFF diagnostic — NOT part of the pipeline.

Task 1: does marching cubes stop LEAKING if we use bigger voxels?

The clouds are sampled at ~1 cm, but the Livox FOV bands (and sparse Leica
patches) leave metre-scale wall holes. MC seals the voxelised shell by dilating
it `seal_iterations` voxels; a hole only closes once the *physical* seal
thickness (seal_iterations x h) approaches the hole width. So coarsening h (or
raising the seal count) should eventually bridge the holes — at the cost of
resolution and an inward bias that no longer cancels in the h->0 extrapolation.

This script sweeps h and seal_iterations on the REAL leica + livox clouds, using
the SAME centreline + domain caps the pipeline builds, and reports for each
(h, seal) whether the enclosed cavity survived (a volume) or leaked / over-filled.

Run:
  ~/.venvs/slam_sweep/bin/python oneoff_mc_voxel_sweep.py            # both clouds
  ~/.venvs/slam_sweep/bin/python oneoff_mc_voxel_sweep.py leica      # one cloud
"""

import logging
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

from config import Config
from io_utils import (load_ply, load_and_transform_trajectory, load_trajectory,
                      load_targets, load_registration, check_rigid_registration)
from spine import fit_centreline, to_cylindrical
from marching_cubes import run_marching_cubes

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger("mc_sweep")
log.setLevel(logging.INFO)

# The experiment matrix. Coarser h and thicker seals bridge wider holes.
VOXEL_SIZES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75]
SEAL_ITERS = [1, 2, 3]


def build_centreline_and_domain(cfg):
    """Reproduce the pipeline's centreline + [s_start, s_end] domain (cheap:
    trajectory only, no 22M-point cylindrical transform)."""
    reg = None
    if cfg.registration_path.exists():
        reg = load_registration(cfg.registration_path)
        check_rigid_registration(reg, tol=cfg.registration_rigid_tol)
    if reg is not None and cfg.trajectory_path.suffix.lower() != ".ply":
        traj = load_and_transform_trajectory(cfg.trajectory_path, reg)
    else:
        traj = load_trajectory(cfg.trajectory_path)

    targets = load_targets(cfg.targets_path)
    t_start = targets[cfg.domain_start_target_idx]
    t_end = targets[cfg.domain_end_target_idx]
    cl = fit_centreline(traj, cfg, orient_toward=t_end)
    s_targets, _, _ = to_cylindrical(np.stack([t_start, t_end]), cl)
    domain = (float(min(s_targets)), float(max(s_targets)))
    log.info("domain s=[%.3f, %.3f]  L=%.3f m", domain[0], domain[1],
             domain[1] - domain[0])
    return cl, domain


def sweep_cloud(name, path, cl, domain):
    log.info("=== %s : loading %s ===", name.upper(), path)
    pts = load_ply(path)
    log.info("  %d points", len(pts))

    # results[seal][h] = (leaked, V_voxel, n_air)
    results = {}
    for seal in SEAL_ITERS:
        results[seal] = {}
        for h in VOXEL_SIZES:
            cfg = Config()
            cfg.marching_cubes_voxel_sizes_m = [h]
            cfg.marching_cubes_seal_iterations = seal
            # allow the coarse grids through (they are all far under budget anyway)
            cfg.marching_cubes_max_voxels = 1.0e9
            res = run_marching_cubes(pts, cfg, cl=cl, domain=domain,
                                     cloud_name=name, export_path=None)
            e = res.per_h[0] if res.per_h else {"leaked": True, "V_voxel": np.nan,
                                                "n_air": 0}
            results[seal][h] = (e["leaked"], e.get("V_voxel", np.nan), e["n_air"])
    return results


def print_table(name, results):
    print("\n" + "=" * 78)
    print(f"  {name.upper()} — MC enclosed-cavity volume [m³]  "
          "(LEAK = merged with outside / over-filled)")
    print("=" * 78)
    header = "  seal\\h " + "".join(f"{h:>8.2f}" for h in VOXEL_SIZES)
    print(header)
    for seal in SEAL_ITERS:
        cells = []
        for h in VOXEL_SIZES:
            leaked, v, n_air = results[seal][h]
            cells.append("   LEAK " if leaked else f"{v:8.1f}")
        print(f"  {seal:>4d}   " + "".join(cells))
    print("=" * 78)
    print("  seal x h = physical seal thickness [m] (what a hole must be under to close).")


def main():
    which = [a for a in sys.argv[1:] if a in ("leica", "livox")]
    clouds = which or ["leica", "livox"]

    cfg = Config()
    cfg.targets_csv = "targets_leica.txt"
    cfg.targets_livox_csv = "targets_livox.txt"
    cl, domain = build_centreline_and_domain(cfg)

    paths = {"leica": cfg.leica_path, "livox": cfg.livox_path}
    all_results = {}
    for name in clouds:
        all_results[name] = sweep_cloud(name, paths[name], cl, domain)

    for name in clouds:
        print_table(name, all_results[name])


if __name__ == "__main__":
    main()
