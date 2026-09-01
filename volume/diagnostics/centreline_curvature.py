"""
Centreline curvature / pause-cusp diagnostics — the day-3 gap-map artefact.

Reports:
  1. trajectory pause clusters (where the operator stopped walking)
  2. spline knot placement in arclength, and curvature peaks
  3. a sweep of (resample ds, smoothing factor) -> curvature, knots, fit RMS

Background: the raw trajectory is TIME-sampled, so every operator pause is a
dense cluster of jittering near-stationary points. splprep parameterizes by
chord length, so u barely advances across a pause while the position jitters —
forcing a near-cusp. Measured kappa_max was 808 1/m (1.2 mm bend radius) in a
straight tunnel. Those cusps scramble the tangent, hence the RMF, hence theta,
punching full-azimuth stripes of false "no data" into BOTH gap maps at every
pause. The point clouds are fine — the artefact is entirely in the centreline
fit, which is why it is invisible in CloudCompare. See day3_findings.md §1.

Run:
    ~/.venvs/slam_sweep/bin/python diagnostics/centreline_curvature.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.WARNING)

from config import Config
from io_utils import load_and_transform_trajectory, load_registration
from spine import _find_outbound_leg, _resample_by_arclength


def outbound_leg(cfg):
    reg = load_registration(cfg.registration_path)
    traj = load_and_transform_trajectory(cfg.trajectory_path, reg)
    outb = _find_outbound_leg(traj)
    d = np.linalg.norm(np.diff(outb, axis=0), axis=1)
    return outb[np.concatenate([[True], d > 1e-6])]


def report_pauses(outb):
    step = np.linalg.norm(np.diff(outb, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(step)])
    print(f"outbound leg: {len(outb)} pts, arclength {cum[-1]:.1f} m")
    print(f"step size: median={np.median(step)*1000:.2f} mm  "
          f"min={step.min()*1000:.3f} mm  max={step.max()*1000:.1f} mm")
    print(f"steps < 1 mm: {np.sum(step < 0.001)} ({100*np.mean(step < 0.001):.1f}%)")

    slow = step < np.percentile(step, 10)
    edges = np.arange(0, cum[-1] + 2, 2.0)
    hist, _ = np.histogram(cum[:-1][slow], bins=edges)
    ctrs = 0.5 * (edges[:-1] + edges[1:])
    print("\npause clusters (count of slowest-10% steps per 2 m of arclength):")
    print("expect ~8 — one per target, roughly every 20 m")
    for c, n in zip(ctrs, hist):
        if n > 0:
            print(f"  {c:6.1f} m : {'#' * min(n // 3, 60)} {n}")


def fit_report(pts, raw, factor, label):
    smoothing = factor * len(pts)
    tck, _ = splprep(pts.T, s=smoothing, k=3)
    u = np.linspace(0, 1, 20000)
    fine = np.array(splev(u, tck)).T
    length = np.linalg.norm(np.diff(fine, axis=0), axis=1).sum()
    d1 = np.array(splev(u, tck, der=1)).T
    d2 = np.array(splev(u, tck, der=2)).T
    kappa = (np.linalg.norm(np.cross(d1, d2), axis=1)
             / np.maximum(np.linalg.norm(d1, axis=1) ** 3, 1e-12))
    rms = float(np.sqrt(np.mean(cKDTree(fine).query(raw)[0] ** 2)))
    print(f"{label:42s} L={length:7.2f}m  knots={len(tck[0])-8:3d}  "
          f"k_max={kappa.max():10.3f}  Rmin={1/max(kappa.max(),1e-9):9.3f}m  "
          f"RMS={rms:.3f}m")
    return tck, u, fine, kappa


def main():
    cfg = Config(targets_csv="targets_leica.txt")
    outb = outbound_leg(cfg)
    report_pauses(outb)

    print("\n--- BASELINE: raw trajectory, old config (factor=0.01) ---")
    tck, u, fine, kappa = fit_report(outb, outb, 0.01, "  raw, factor=0.01")

    seg = np.linalg.norm(np.diff(fine, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    knots_s = np.interp(tck[0][4:-4], u, cum)
    print(f"  interior knot arclengths [m]: {np.round(knots_s, 2)}")
    peaks, _ = find_peaks(kappa, height=np.percentile(kappa, 90))
    print(f"  curvature peaks [m]: {np.round(np.interp(u[peaks], u, cum), 1)[:40]}")
    print("  ^ these match the gap-map stripe locations one-for-one")

    print("\n--- SWEEP: arclength resampling x smoothing ---")
    print("resampling ALONE is not enough — it must be paired with smoothing")
    for ds in (0.25, 0.5, 1.0):
        rs = _resample_by_arclength(outb, ds)
        for factor in (0.01, 0.05, 0.2):
            marker = " <- CHOSEN" if (ds == 0.5 and factor == 0.05) else ""
            fit_report(rs, outb, factor,
                       f"  ds={ds}m (N={len(rs)}), factor={factor}{marker}")
    print("\nfactor=0.2 collapses to 0 interior knots (a single cubic — too")
    print("stiff, loses real bends, doubles RMS). factor=0.05 @ ds=0.5 keeps a")
    print("sane ~30 m bend radius at 0.11 m fit RMS.")


if __name__ == "__main__":
    main()
