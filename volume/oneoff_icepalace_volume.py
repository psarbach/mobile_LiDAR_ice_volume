"""
ONE-OFF — NOT part of the pipeline. Do not import from run_pipeline.

Marching-cubes volume of the ice-palace tunnel network
(data/12_55_33_full_1cm_rep00_sor20.ply), whose FLOOR is entirely missing and
which has 7 open mouths. Both are sealed here by adding synthetic points so the
existing marching_cubes.run_marching_cubes (unchanged) sees a closed shell:

  - floor: a dense grid of points at z = FLOOR_Z across the network footprint.
  - mouths: for each data/icepalace_opening/mouth{1..7}.txt (rim points picked
    in CloudCompare), fit a plane and fill its rim convex hull with points.

This lives outside the pipeline on purpose: it patches bad data for this one
cloud only. Run:  ~/.venvs/slam_sweep/bin/python oneoff_icepalace_volume.py
"""

import glob
import logging
import os
import sys

import numpy as np
from scipy.spatial import Delaunay

from config import Config
from io_utils import load_ply
from marching_cubes import run_marching_cubes

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("icepalace")

MOUTH_DIR = "data/icepalace_opening"
FILL_SPACING = 0.04          # < smallest voxel (0.05) so the patches are watertight

# Cloud path: pass any *.ply as an argument, else the default. The cleaned
# variant has the same coordinate frame, so the picked mouths + floor still apply.
_plys = [a for a in sys.argv[1:] if a.endswith(".ply")]
CLOUD = _plys[0] if _plys else "data/12_55_33_full_1cm_rep00_sor20.ply"

# CLI:  [mc|poisson]  [floor_z]
#   mc      (default) — flood-fill marching cubes; leaks on this incomplete scan.
#   poisson           — Poisson surface reconstruction; fills the missing walls,
#                       so it never leaks, but INVENTS geometry where data is
#                       absent → a filled-surface ballpark, not a measurement.
# floor_z: actual floor is -1.45; another value TRUNCATES the network there
#          (a diagnostic — see the -0.5 test).
MODE = "poisson" if "poisson" in sys.argv else "mc"
_nums = [a for a in sys.argv[1:] if a.replace("-", "").replace(".", "").isdigit()]
FLOOR_Z = float(_nums[0]) if _nums else -1.45


def make_floor(pts: np.ndarray, z: float, spacing: float) -> np.ndarray:
    """Dense horizontal grid of points at height z over the cloud's XY footprint."""
    lo, hi = pts.min(0), pts.max(0)
    xs = np.arange(lo[0], hi[0] + spacing, spacing)
    ys = np.arange(lo[1], hi[1] + spacing, spacing)
    X, Y = np.meshgrid(xs, ys)
    return np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])


def make_cap(rim: np.ndarray, spacing: float) -> np.ndarray:
    """Fill a mouth: fit a plane to its rim points, tile its convex hull."""
    c = rim.mean(0)
    _, _, Vt = np.linalg.svd(rim - c, full_matrices=False)
    u, v, n = Vt[0], Vt[1], Vt[2]            # in-plane basis + plane normal
    q = np.column_stack([(rim - c) @ u, (rim - c) @ v])   # rim in 2-D
    hull = Delaunay(q)
    a = np.arange(q[:, 0].min(), q[:, 0].max() + spacing, spacing)
    b = np.arange(q[:, 1].min(), q[:, 1].max() + spacing, spacing)
    A, B = np.meshgrid(a, b)
    grid2d = np.column_stack([A.ravel(), B.ravel()])
    inside = hull.find_simplex(grid2d) >= 0
    g = grid2d[inside]
    return c + g[:, 0:1] * u + g[:, 1:2] * v


def build_augmented():
    """Network cloud + floor plane + 7 mouth caps, clipped to z >= FLOOR_Z."""
    wall = load_ply(CLOUD)
    log.info("network cloud: %d pts, bbox z=[%.2f, %.2f]",
             len(wall), wall[:, 2].min(), wall[:, 2].max())
    floor = make_floor(wall, FLOOR_Z, FILL_SPACING)
    log.info("added floor: %d pts at z=%.2f", len(floor), FLOOR_Z)
    caps = []
    for f in sorted(glob.glob(os.path.join(MOUTH_DIR, "mouth*.txt"))):
        rim = np.loadtxt(f, delimiter=",")[:, -3:]
        cap = make_cap(rim, FILL_SPACING)
        caps.append(cap)
        log.info("  %-11s %2d rim pts -> %5d cap pts", os.path.basename(f),
                 len(rim), len(cap))
    aug = np.vstack([wall, floor] + caps)
    aug = aug[aug[:, 2] >= FLOOR_Z - 1e-6]
    log.info("augmented cloud: %d pts (+%d floor +%d caps, z>=%.2f)",
             len(aug), len(floor), sum(len(c) for c in caps), FLOOR_Z)
    return aug


def run_mc(aug):
    cfg = Config()
    res = run_marching_cubes(
        aug, cfg, cl=None, domain=None, cloud_name="icepalace",
        export_path="output/marching_cubes_icepalace.ply",
    )
    print("\n" + "=" * 56)
    if res.leaked:
        print("  RESULT: LEAKED — no reliable enclosed volume.")
        print("  Floor + 7 mouth caps did not seal the shell; large")
        print("  wall/ceiling scan-holes remain open. Try 'poisson'.")
    else:
        print("  ICE-PALACE VOLUME (marching cubes)")
        print(f"    V_extrapolated (h→0) = {res.V_extrapolated:.1f} m³")
        print("  Mesh: output/marching_cubes_icepalace.ply")
    print("=" * 56)


def run_poisson(aug):
    """
    Poisson surface reconstruction — fills the missing walls, so it never leaks,
    but INVENTS geometry where data is absent. The volume is a filled-surface
    BALLPARK, not a measurement, and (given how much wall is missing here) it
    likely balloons across the gaps.
    """
    import open3d as o3d
    from surface_mesh import mesh_volume, check_closed

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(aug)
    # Downsample hard to 0.20 m: orient_normals_consistent_tangent_plane is
    # single-threaded and was intractable at 0.10 m (>16 min on 227k pts, killed)
    # — the dense flat floor makes a degenerate propagation graph. ~0.20 m leaves
    # a few 10k points, plenty for a ballpark volume on a 31 m network.
    pcd = pcd.voxel_down_sample(0.20)
    log.info("Poisson: %d pts after 0.20 m downsample; estimating normals…",
             len(pcd.points))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.6, max_nn=30))
    log.info("Poisson: orienting normals (consistent tangent plane, k=8)…")
    pcd.orient_normals_consistent_tangent_plane(8)

    log.info("Poisson: reconstructing (depth=9)…")
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9)
    dens = np.asarray(dens)

    # Poisson extrapolates a watertight surface into unsupported regions; those
    # vertices have low sampling density. Trim them to pull the surface back to
    # the data, then clip to the cloud's bbox to drop any remaining balloon.
    q = 0.04
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, q))
    mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
    mesh.remove_unreferenced_vertices()

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    V = mesh_volume(verts, faces)
    is_closed, is_oriented = check_closed(faces)

    out = "output/poisson_icepalace.ply"
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(out, mesh)

    print("\n" + "=" * 56)
    print("  ICE-PALACE VOLUME (Poisson — filled-surface BALLPARK)")
    print(f"    enclosed volume ≈ {V:.1f} m³")
    print(f"    mesh: {len(verts)} verts / {len(faces)} faces  "
          f"(density-trimmed q={q}, bbox-cropped)")
    print(f"    watertight after trim: {is_closed and is_oriented}")
    print("    ⚠ NOT a measurement — Poisson invents the missing walls/floor.")
    print(f"  Mesh exported for CloudCompare: {out}")
    print("=" * 56)


def main():
    aug = build_augmented()
    if MODE == "poisson":
        run_poisson(aug)
    else:
        run_mc(aug)


if __name__ == "__main__":
    main()
