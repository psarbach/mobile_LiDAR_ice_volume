"""
Coverage diagnostics behind the day-3 golden-segment finding.

Reports, for the real target-capped domain:
  1. per-cloud gap-map % missing and per-slab coverage stats
  2. longest contiguous BOTH-clouds window vs coverage threshold
  3. longest contiguous LEICA-ONLY window vs coverage threshold
  4. Livox coverage as a function of theta  <- shows the fixed FOV notch

Conclusion this produced: a both-clouds golden segment does not exist at any
threshold, because the Livox gap is a fixed-azimuth FOV notch (not localized
holes) and its median slab coverage is only ~70.6%. See day3_findings.md §3.

Reads the cached (s, r, theta) npz — run the pipeline first so the cache exists
and was built with the CURRENT centreline config:

    ~/.venvs/slam_sweep/bin/python run_pipeline.py --run-real --no-cache \
        --targets-file targets_leica.txt --targets-livox-file targets_livox.txt
    ~/.venvs/slam_sweep/bin/python diagnostics/coverage_report.py
"""

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.WARNING)

from config import Config
from io_utils import (load_and_transform_trajectory, load_targets,
                      load_registration, check_rigid_registration)
from spine import fit_centreline, build_gap_map, to_cylindrical, _longest_window

THRESHOLDS = (0.98, 0.95, 0.92, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50)


def build_centreline(cfg):
    # The registration matrix is pre-applied to the point clouds but NOT to the
    # trajectory. Skipping this step silently puts the centreline in the wrong
    # frame and every number below becomes wrong without erroring.
    reg = load_registration(cfg.registration_path)
    check_rigid_registration(reg, tol=cfg.registration_rigid_tol)
    traj = load_and_transform_trajectory(cfg.trajectory_path, reg)
    targets = load_targets(cfg.targets_path)
    cl = fit_centreline(traj, cfg, orient_toward=targets[cfg.domain_end_target_idx])
    s_t, _, _ = to_cylindrical(targets, cl)
    domain = (float(s_t[cfg.domain_start_target_idx]),
              float(s_t[cfg.domain_end_target_idx]))
    return cl, domain


def runs_vs_threshold(cov, s_edges, label):
    ds = float(s_edges[1] - s_edges[0])
    print(f"\n{label}\n thresh | longest run | s-range")
    for th in THRESHOLDS:
        start, n = _longest_window(cov >= th, target_n=10 ** 9)
        hi = s_edges[min(start + n, len(s_edges) - 1)]
        print(f"  {th * 100:4.0f}% | {n * ds:6.1f} m | s=[{s_edges[start]:6.2f}, {hi:6.2f}]")


def main():
    cfg = Config(targets_csv="targets_leica.txt")
    cl, domain = build_centreline(cfg)
    print(f"centreline: length={cl.total_length_m:.2f} m  fit RMS={cl.fit_rms_m:.4f} m")
    print(f"domain = {domain[0]:.2f} .. {domain[1]:.2f} m  (L={domain[1]-domain[0]:.3f} m)")

    gms = {}
    for name in ("leica", "livox"):
        path = cfg.cache_path(f"{name}_cyl")
        if not path.exists():
            raise SystemExit(f"missing cache {path} — run the pipeline first")
        d = np.load(path)
        gms[name] = build_gap_map(d["s"], d["theta"], d["r"], cfg, domain=domain)

    print()
    for name, gm in gms.items():
        c = gm.coverage_per_slab
        print(f"{name:6s}: {gm.frac_missing*100:5.1f}% missing | per-slab coverage "
              f"median={np.median(c)*100:5.1f}%  p10={np.percentile(c,10)*100:5.1f}%  "
              f"min={c.min()*100:5.1f}%")

    cov_both = np.minimum(gms["leica"].coverage_per_slab,
                          gms["livox"].coverage_per_slab)
    print(f"{'both':6s}: per-slab coverage median={np.median(cov_both)*100:5.1f}%  "
          f"p10={np.percentile(cov_both,10)*100:5.1f}%  min={cov_both.min()*100:5.1f}%")

    runs_vs_threshold(cov_both, gms["leica"].s_edges, "BOTH-CLOUDS golden segment")
    runs_vs_threshold(gms["leica"].coverage_per_slab, gms["leica"].s_edges,
                      "LEICA-ONLY golden segment")

    # Fixed-azimuth notch: coverage per theta bin, averaged over all slabs.
    # A dip here that is constant along s is a sensor blind zone, not geometry.
    gl = gms["livox"]
    cov_t = (~np.isnan(gl.r_grid)).mean(axis=0)
    tc = 0.5 * (gl.theta_edges[:-1] + gl.theta_edges[1:])
    print("\nLIVOX coverage vs theta (fraction of slabs with data)")
    if cfg.theta_reference == "up":
        print("theta=0 = CEILING, +-180 = FLOOR, +-90 = side walls.")
    else:
        print(f"NOTE: theta_reference={cfg.theta_reference!r} — theta=0 is an "
              "ARBITRARY seed direction, NOT physical. Set theta_reference='up' "
              "to read these as ceiling/floor.")
    for i in range(0, len(tc), 6):
        label = {0: " <- ceiling", 90: " <- side wall", -90: " <- side wall",
                 180: " <- floor", -180: " <- floor"}
        near = min(label, key=lambda k: abs(((tc[i] - k + 180) % 360) - 180))
        tag = label[near] if abs(((tc[i] - near + 180) % 360) - 180) < 8 else ""
        print(f"  theta={tc[i]:7.1f}°  {'#' * int(cov_t[i] * 50):50s} "
              f"{cov_t[i]*100:5.1f}%{tag}")


if __name__ == "__main__":
    main()
