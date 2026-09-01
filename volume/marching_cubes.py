"""
Marching-cubes volume  (the trajectory-INDEPENDENT estimator)

The third volume estimate, and the only genuinely independent one: it never
builds the r(s, θ) grid that profiles and surface_mesh share, so it can catch an
error in that extraction which those two would agree on and hide. It also works
where a centreline is meaningless — a cave/chamber with a wandering path — since
the volume computation touches only the raw points and a voxel grid.

Idea
----
The cloud is the WALL surface (a shell). "Air" and "rock/outside" are BOTH empty
of points, so occupancy alone cannot tell them apart — the shell separates them.
So:

  1. Voxelise the raw points into an occupancy grid at voxel size h.
  2. Morphological CLOSING seals pinholes so the shell is a watertight barrier.
  3. Tube mode only: wall off the two open ends with the domain cap planes (the
     SAME planes profiles/surface_mesh use — from the centreline at s_start /
     s_end — so all three methods measure a bit-identical domain). Cave mode:
     no caps, the surface is assumed closed.
  4. Flood-fill the exterior: label the free-space components; any component
     touching the padded grid boundary is "outside". The AIR CAVITY is the free
     space the shell hides from the outside — no seed point, no trajectory.
  5. Volume two ways: voxel count x h³ (swept over h, Richardson-extrapolated to
     h->0), and marching cubes on the air field -> watertight mesh -> divergence
     theorem (reuses surface_mesh.mesh_volume), exported for Cloud-to-Mesh in CC.

Only the CAPS (step 3, tube mode) use the centreline; the volume itself does
not. Cave mode is fully trajectory-free.

Known limitation — wall holes
-----------------------------
The flood-fill assumes the shell fully separates air from outside. A hole wider
than the dilation seal (marching_cubes_seal_iterations voxels) lets the exterior
LEAK into the cavity: the two merge and the air volume collapses. Expected on Livox,
whose FOV bands are metre-scale holes — MC is honest about it (leak detected and
reported) rather than silently wrong. That failure is itself informative: it is
the price the other two methods pay via their (s, θ) hole-filling, made visible.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage

from config import Config
from surface_mesh import mesh_volume, check_closed

log = logging.getLogger(__name__)


@dataclass
class MarchingCubesResult:
    V_mc: float                 # m³ — finest-grid marching-cubes volume (primary)
    V_extrapolated: float       # m³ — voxel-count volume extrapolated to h->0
    mean_area_m2: float         # m² — V_mc / L (tube) or nan (cave)
    per_h: List[dict] = field(default_factory=list)  # one dict per voxel size
    leaked: bool = False        # exterior flood-fill leaked into the cavity
    cloud_name: str = ""
    vertices: np.ndarray = None
    faces: np.ndarray = None


# --------------------------------------------------------------------------- #
#  Voxelisation + shell sealing                                                #
# --------------------------------------------------------------------------- #

def voxelize(pts: np.ndarray, h: float, pad: int = 3
             ) -> Tuple[np.ndarray, np.ndarray]:
    """Occupancy grid: True where a voxel contains >=1 point. Returns (O, lo)."""
    lo = pts.min(axis=0) - pad * h
    hi = pts.max(axis=0) + pad * h
    dims = (np.ceil((hi - lo) / h).astype(int) + 1)
    idx = np.floor((pts - lo) / h).astype(int)
    idx = np.clip(idx, 0, dims - 1)
    O = np.zeros(tuple(dims), dtype=bool)
    O[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return O, lo


def seal_shell(O: np.ndarray, iterations: int) -> np.ndarray:
    """
    Dilate the wall shell to a watertight >=2-voxel barrier.

    NOT a closing (dilate-then-erode): the erode step re-thins the wall to one
    voxel and reopens the diagonal pinholes a face-connected flood-fill leaks
    through. A pure dilation keeps the wall thick. The cost is an inward bias
    (the cavity shrinks by ~iterations voxels) that is O(h) with FIXED
    iterations, so it cancels in the h->0 extrapolation.
    """
    structure = ndimage.generate_binary_structure(3, 1)
    return ndimage.binary_dilation(O, structure=structure, iterations=max(1, iterations))


def _centreline_at(cl, s: float) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate centreline point and unit tangent at arclength s."""
    p = np.array([np.interp(s, cl.s_samples, cl.samples_xyz[:, k]) for k in range(3)])
    t = np.array([np.interp(s, cl.s_samples, cl.tangent[:, k]) for k in range(3)])
    return p, t / np.linalg.norm(t)


def add_end_caps(O: np.ndarray, lo: np.ndarray, h: float, cl,
                 domain: Tuple[float, float]) -> None:
    """
    Wall off everything axially outside [s_start, s_end], in place.

    The two half-spaces before the start cap and after the end cap are marked
    occupied, which seals the open tunnel ends with a full planar wall at each
    cap — the SAME planes profiles/surface_mesh cut on. Done in a loop over the
    first axis so no full-grid float array is ever materialised.
    """
    s0, s1 = domain
    p0, n0 = _centreline_at(cl, s0)   # n0 points toward +s (into the domain)
    p1, n1 = _centreline_at(cl, s1)
    nx, ny, nz = O.shape
    gy = lo[1] + (np.arange(ny) + 0.5) * h
    gz = lo[2] + (np.arange(nz) + 0.5) * h
    # per-axis contributions to (center - p)·n
    y0 = (gy - p0[1]) * n0[1]; z0 = (gz - p0[2]) * n0[2]
    y1 = (gy - p1[1]) * n1[1]; z1 = (gz - p1[2]) * n1[2]
    for i in range(nx):
        cx = lo[0] + (i + 0.5) * h
        f0 = (cx - p0[0]) * n0[0] + y0[:, None] + z0[None, :]   # >0 inside from start
        f1 = (cx - p1[0]) * n1[0] + y1[:, None] + z1[None, :]   # <0 inside from end
        O[i] |= (f0 < 0) | (f1 > 0)


# --------------------------------------------------------------------------- #
#  Cavity extraction                                                           #
# --------------------------------------------------------------------------- #

def enclosed_air(O: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    The single largest cavity the shell hides from the outside.

    Labels the connected components of the free space (~O); any component that
    touches the grid boundary is exterior. The cavity is the LARGEST interior
    component — the tunnel/chamber is one connected air region, so returning the
    union of all interior components would fold in spurious little pockets
    trapped in wall crevices. Returns (air_mask, n_air_voxels). If the shell
    leaks, the main cavity connects to the outside and only tiny pockets remain,
    so n_air_voxels collapses — the driver flags that against a physical floor.
    """
    free = ~O
    labels, n = ndimage.label(free)
    if n == 0:
        return np.zeros_like(O), 0

    boundary = np.concatenate([
        labels[0, :, :].ravel(), labels[-1, :, :].ravel(),
        labels[:, 0, :].ravel(), labels[:, -1, :].ravel(),
        labels[:, :, 0].ravel(), labels[:, :, -1].ravel(),
    ])
    exterior = set(np.unique(boundary)) - {0}

    counts = np.bincount(labels.ravel())
    best_lab, best_count = 0, 0
    for lab in range(1, n + 1):
        if lab in exterior:
            continue
        if counts[lab] > best_count:
            best_lab, best_count = lab, counts[lab]
    if best_lab == 0:
        return np.zeros_like(O), 0
    return labels == best_lab, int(best_count)


def unseal_air(air: np.ndarray, iterations: int) -> np.ndarray:
    """
    Undo the seal dilation's inward bias by growing the air mask back by the
    same amount. The seal thickened the wall inward by `iterations` voxels,
    shrinking the cavity; dilating the (now isolated) air region back by
    `iterations` restores it to the wall's inner face. Leaves only the ~0.5·h
    wall-half-thickness bias, which the h->0 extrapolation removes. Makes each
    single-resolution volume interpretable, not just the extrapolated one.
    """
    if iterations <= 0:
        return air
    structure = ndimage.generate_binary_structure(3, 1)
    return ndimage.binary_dilation(air, structure=structure, iterations=iterations)


def marching_cubes_mesh(air: np.ndarray, lo: np.ndarray, h: float
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Marching cubes on the air field -> (vertices_world, faces).

    The field is padded with a non-air border so the surface closes even if the
    cavity abuts the grid edge. Vertices are returned in world (metre) coords.
    """
    from skimage.measure import marching_cubes
    padded = np.pad(air.astype(np.float32), 1)
    verts, faces, _, _ = marching_cubes(padded, level=0.5)
    # padded index -> unpadded index (-1) -> world (voxel centres: +0.5, then -1
    # for the pad cancels to -0.5). Origin offset does not affect the volume.
    verts_world = lo + (verts - 0.5) * h
    return verts_world, faces


# --------------------------------------------------------------------------- #
#  Driver                                                                      #
# --------------------------------------------------------------------------- #

def run_marching_cubes(
    pts: np.ndarray,
    cfg: Config,
    cl=None,
    domain: Optional[Tuple[float, float]] = None,
    cloud_name: str = "",
    export_path: Optional[str] = None,
) -> MarchingCubesResult:
    """
    Trajectory-independent marching-cubes volume, swept over voxel size.

    Parameters
    ----------
    pts    : (N, 3) raw cloud points (Leica datum).
    cl     : Centreline — used ONLY to place the tube end caps. Pass None (or set
             geometry_mode="volumetric") for a cave: no caps, no trajectory.
    domain : (s_start, s_end) for the tube caps; ignored in cave mode.
    export_path : write the finest-grid mesh to PLY for Cloud-to-Mesh in CC.
    """
    tube = cfg.geometry_mode == "tube" and cl is not None and domain is not None
    L = (domain[1] - domain[0]) if domain is not None else float("nan")

    per_h = []
    finest_mesh = (None, None)
    finest_h = None
    leaked_any = False

    for h in sorted(cfg.marching_cubes_voxel_sizes_m, reverse=True):  # coarse->fine
        lo = pts.min(axis=0) - 3 * h
        hi = pts.max(axis=0) + 3 * h
        n_vox = float(np.prod(np.ceil((hi - lo) / h) + 1))
        if n_vox > cfg.marching_cubes_max_voxels:
            log.warning(
                "Marching cubes (%s): h=%.3f m -> %.2e voxels exceeds budget "
                "%.2e — skipped.", cloud_name, h, n_vox, cfg.marching_cubes_max_voxels,
            )
            continue

        O, lo = voxelize(pts, h)
        O = seal_shell(O, cfg.marching_cubes_seal_iterations)
        if tube:
            add_end_caps(O, lo, h, cl, domain)

        air, n_air = enclosed_air(O)
        V_cavity = n_air * h ** 3
        leaked = V_cavity < cfg.marching_cubes_leak_min_m3
        leaked_any = leaked_any or leaked

        if not leaked:
            air = unseal_air(air, cfg.marching_cubes_seal_iterations)
            V_vox = float(air.sum()) * h ** 3
        else:
            V_vox = float("nan")

        entry = {"h": h, "V_voxel": V_vox, "n_air": int(n_air), "leaked": leaked}

        if not leaked:
            verts, faces = marching_cubes_mesh(air, lo, h)
            V_mc = mesh_volume(verts, faces)
            is_closed, is_oriented = check_closed(faces)
            entry.update(V_mc=V_mc, closed=(is_closed and is_oriented),
                         n_faces=len(faces))
            if finest_h is None or h < finest_h:
                finest_h, finest_mesh = h, (verts, faces)
        else:
            entry.update(V_mc=float("nan"), closed=False, n_faces=0)
            log.warning(
                "Marching cubes (%s): h=%.3f m LEAKED — largest enclosed cavity "
                "only %.2f m³ (< %.1f m³ floor). The exterior reached through a "
                "real wall hole bigger than the seal; no reliable volume.",
                cloud_name, h, V_cavity, cfg.marching_cubes_leak_min_m3,
            )

        per_h.append(entry)
        if not leaked:
            log.info(
                "Marching cubes (%s): h=%.3f m  V_voxel=%.2f m³  V_mc=%.2f m³",
                cloud_name, h, V_vox, entry["V_mc"],
            )

    # Richardson extrapolation of the voxel-count volume to h->0 (first-order in
    # h: the stair-stepped surface over/under-counts by O(h) per unit area).
    good = [e for e in per_h if not e["leaked"]]
    V_extrap = float("nan")
    if len(good) >= 2:
        hs = np.array([e["h"] for e in good])
        vs = np.array([e["V_voxel"] for e in good])
        # V(h) ≈ V0 + a·h  -> fit and read off V0
        a, V0 = np.polyfit(hs, vs, 1)[0], np.polyfit(hs, vs, 1)[1]
        V_extrap = float(V0)

    verts, faces = finest_mesh
    V_mc = mesh_volume(verts, faces) if verts is not None else float("nan")
    mean_area = V_mc / L if (tube and np.isfinite(V_mc) and L > 0) else float("nan")

    if export_path and verts is not None:
        try:
            import open3d as o3d
            m = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(verts),
                o3d.utility.Vector3iVector(faces),
            )
            m.compute_vertex_normals()
            o3d.io.write_triangle_mesh(str(export_path), m)
            log.info("Marching cubes (%s): exported %s (for Cloud-to-Mesh in CC)",
                     cloud_name, export_path)
        except ImportError:
            log.warning("open3d unavailable — skipping mesh export")

    log.info(
        "Marching cubes (%s): V_mc(finest h=%s)=%.2f m³  V_extrap(h->0)=%.2f m³%s",
        cloud_name, f"{finest_h:.3f}" if finest_h else "—",
        V_mc if np.isfinite(V_mc) else float("nan"),
        V_extrap if np.isfinite(V_extrap) else float("nan"),
        "  [LEAKED at one or more resolutions]" if leaked_any else "",
    )

    return MarchingCubesResult(
        V_mc=V_mc,
        V_extrapolated=V_extrap,
        mean_area_m2=mean_area,
        per_h=per_h,
        leaked=leaked_any,
        cloud_name=cloud_name,
        vertices=verts,
        faces=faces,
    )
