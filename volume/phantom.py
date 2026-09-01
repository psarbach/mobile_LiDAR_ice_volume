"""
Phantom cylinder validation.

Run the WHOLE spine + profiles pipeline on a synthetic dataset of known
geometry BEFORE touching the real ice data.  Every method must pass here
first.

Phantom geometry
----------------
- Cylinder:  radius R,  length L,  axis along +Z
- True volume: π R² L  (default ≈ 1 256.6 m³)
- Trajectory: straight line along the Z axis with small Gaussian wander
  (to test that PCA recovers the true axis despite noise)
- Targets: two points at z = 0  and z = L  (= domain end caps)

Hole test
---------
Punch a ceiling-like angular wedge out of the cloud, rerun, confirm that
interpolation recovers the volume within a given tolerance.
"""

import logging
from dataclasses import dataclass, replace
from typing import Tuple

import numpy as np

from config import Config
from spine import fit_centreline, to_cylindrical
from cross_sections import run_profiles, run_hull_bound
from surface_mesh import run_surface_mesh
from marching_cubes import run_marching_cubes

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Cloud generators                                                            #
# --------------------------------------------------------------------------- #

def make_cylinder_cloud(
    radius: float,
    length: float,
    n_theta: int,
    n_z: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Return (N, 3) points uniformly on a closed cylinder surface (no end caps).

    Axis is along +Z, centred at origin.
    """
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    z = np.linspace(-length / 2, length / 2, n_z)
    TH, Z = np.meshgrid(theta, z)
    x = radius * np.cos(TH.ravel())
    y = radius * np.sin(TH.ravel())
    z = Z.ravel()
    pts = np.column_stack([x, y, z])
    return pts


def make_cylinder_trajectory(
    length: float,
    n_pts: int = 200,
    noise_sigma: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Straight-line trajectory along +Z with Gaussian wander.

    Simulates an out-and-back handheld walk: goes from z=-L/2 to z=+L/2
    then back, so the average is the true axis.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n_half = n_pts // 2
    z_fwd = np.linspace(-length / 2, length / 2, n_half)
    z_back = z_fwd[::-1]
    z_all = np.concatenate([z_fwd, z_back])
    xy = rng.normal(0.0, noise_sigma, (len(z_all), 2))
    return np.column_stack([xy, z_all])


def make_cylinder_targets(length: float) -> np.ndarray:
    """Two targets: one at each end of the cylinder on the axis."""
    return np.array([
        [0.0, 0.0, -length / 2],   # target 0  (domain start)
        [0.0, 0.0,  length / 2],   # target 1  (domain end)
    ])


# --------------------------------------------------------------------------- #
#  Punch holes                                                                 #
# --------------------------------------------------------------------------- #

def punch_hole(
    pts: np.ndarray,
    theta_lo_deg: float = 80.0,
    theta_hi_deg: float = 130.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove points in a ceiling angular wedge.

    Returns (reduced_cloud, hole_mask) where hole_mask is True for removed
    points.
    """
    theta = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))  # −180 … +180
    in_hole = (theta >= theta_lo_deg) & (theta <= theta_hi_deg)
    return pts[~in_hole], in_hole


# --------------------------------------------------------------------------- #
#  Validation runner                                                           #
# --------------------------------------------------------------------------- #

@dataclass
class PhantomResult:
    true_volume_m3: float
    profiles_V_trap: float
    profiles_V_simp: float
    hull_V: float
    mesh_V: float
    axis_angle_deg: float       # angle between fitted axis and true axis [0°]
    centreline_rms_m: float
    profiles_error_pct: float
    hull_error_pct: float
    mesh_error_pct: float
    mesh_watertight: bool
    mc_V: float
    mc_error_pct: float
    mc_leaked: bool


def run_phantom_test(cfg: Config, with_hole: bool = False) -> PhantomResult:
    """
    Build synthetic cylinder, run spine + profiles + hull bound, compare to truth.

    Parameters
    ----------
    with_hole : if True, punch a 50° ceiling wedge before running (hole test).
    """
    # The phantom cylinder's axis is +Z — i.e. a VERTICAL shaft, which is
    # precisely the case where theta_reference="up" degenerates (the component
    # of up perpendicular to the tangent shrinks to zero and its direction
    # becomes arbitrary). Force "rmf" here, which is exactly what that option
    # exists for. Safe: the phantom asserts on VOLUME and axis direction, both
    # invariant to where theta=0 points.
    #
    # Consequence: the phantom does NOT exercise the "up" frame that real runs
    # use. That path is verified instead by plot_theta_reference() against real
    # world z (figure 01b). Making the phantom cylinder horizontal would let it
    # cover both — worth doing, not done yet.
    if cfg.theta_reference != "rmf":
        cfg = replace(cfg, theta_reference="rmf")
        log.info(
            "Phantom: forcing theta_reference='rmf' (the synthetic cylinder's "
            "axis is vertical, where 'up' degenerates). Volume results are "
            "unaffected — they do not depend on where theta=0 points."
        )

    R = cfg.phantom_radius_m
    L = cfg.phantom_length_m
    true_V = np.pi * R**2 * L

    rng = np.random.default_rng(0)
    cloud = make_cylinder_cloud(R, L, cfg.phantom_n_theta, cfg.phantom_n_z)
    traj = make_cylinder_trajectory(L, noise_sigma=cfg.phantom_traj_noise_m, rng=rng)
    targets = make_cylinder_targets(L)

    if with_hole:
        cloud, _ = punch_hole(cloud, theta_lo_deg=80.0, theta_hi_deg=130.0)
        hole_pct = (130 - 80) / 360 * 100
        log.info("Phantom hole test: %.0f° wedge removed (%.1f%%)", 50, hole_pct)

    # ---- spine
    cl = fit_centreline(traj, cfg, orient_toward=targets[1])
    s, r, theta = to_cylindrical(cloud, cl)

    # ---- domain (from synthetic targets, projected onto the fitted spline)
    s_targets, _, _ = to_cylindrical(targets, cl)
    domain = (float(min(s_targets)), float(max(s_targets)))

    # ---- methods
    prof = run_profiles(s, r, theta, domain, cfg, cloud_name="phantom")
    hull = run_hull_bound(s, r, theta, domain, cfg, cloud_name="phantom")
    mesh = run_surface_mesh(s, r, theta, domain, cfg, cl, cloud_name="phantom")
    mc = run_marching_cubes(cloud, cfg, cl=cl, domain=domain, cloud_name="phantom")

    # ---- diagnostics: mean tangent vs true (+Z) axis
    true_axis = np.array([0.0, 0.0, 1.0])
    mean_tangent = cl.tangent.mean(axis=0)
    mean_tangent /= np.linalg.norm(mean_tangent)
    cos_a = float(np.clip(abs(np.dot(mean_tangent, true_axis)), -1, 1))
    angle_deg = np.degrees(np.arccos(cos_a))

    result = PhantomResult(
        true_volume_m3=true_V,
        profiles_V_trap=prof.V_trap,
        profiles_V_simp=prof.V_simp,
        hull_V=hull.V_hull,
        mesh_V=mesh.V_mesh,
        axis_angle_deg=angle_deg,
        centreline_rms_m=cl.fit_rms_m,
        profiles_error_pct=(prof.V_trap - true_V) / true_V * 100,
        hull_error_pct=(hull.V_hull - true_V) / true_V * 100,
        mesh_error_pct=(mesh.V_mesh - true_V) / true_V * 100,
        mesh_watertight=mesh.is_watertight,
        mc_V=mc.V_extrapolated,
        mc_error_pct=(mc.V_extrapolated - true_V) / true_V * 100
                     if np.isfinite(mc.V_extrapolated) else float("nan"),
        mc_leaked=mc.leaked,
    )

    tag = "hole" if with_hole else "full"
    log.info("=== Phantom (%s) ===", tag)
    log.info("  True V (domain L=%.2f m) = %.2f m³", L, true_V)
    log.info("  profiles V (trap) = %.2f m³  (%+.3f %%)",
             prof.V_trap, result.profiles_error_pct)
    log.info("  profiles V (simp) = %.2f m³", prof.V_simp)
    log.info("  surface mesh V    = %.2f m³  (%+.3f %%)  closed=%s",
             mesh.V_mesh, result.mesh_error_pct, mesh.is_watertight)
    log.info("  marching cubes V  = %.2f m³  (%+.3f %%, h→0 extrapolated)  leaked=%s",
             result.mc_V, result.mc_error_pct, mc.leaked)
    log.info("  hull bound V      = %.2f m³  (%+.3f %%)",
             hull.V_hull, result.hull_error_pct)
    log.info("  Axis angle      = %.4f °  (want ≈0)", angle_deg)
    log.info("  Centreline fit RMS = %.4f m", cl.fit_rms_m)

    # Strict checks.
    # Tolerance is 0.5%, not the original 2%: once the ds/2 end truncation was
    # fixed both estimators land at ~0.01% on this phantom, so 2% could not
    # catch even a gross regression (the truncation itself only ever cost
    # 0.25%). 0.5% still leaves ~50x headroom over the observed error.
    tol_pct = 0.5
    assert abs(result.profiles_error_pct) < tol_pct, (
        f"PHANTOM FAIL: profiles volume error "
        f"{result.profiles_error_pct:.3f}% exceeds ±{tol_pct}%"
    )
    assert abs(result.mesh_error_pct) < tol_pct, (
        f"PHANTOM FAIL: surface-mesh volume error "
        f"{result.mesh_error_pct:.3f}% exceeds ±{tol_pct}%"
    )
    assert result.mesh_watertight is not False, (
        "PHANTOM FAIL: surface mesh is not watertight — the enclosed volume "
        "is not well defined"
    )
    # Marching cubes. It is a voxel method extrapolated to h->0, so it is
    # inherently coarser than profiles/mesh — 1.5% tolerance, still ~6x the
    # observed ~0.25% error. On the CLOSED phantom it must not leak; on the
    # HOLED phantom (a real ceiling slot) it MUST leak — that validates the
    # leak detection, without which MC could report a silently-wrong number on
    # the FOV-limited real clouds.
    if not with_hole:
        assert not result.mc_leaked, (
            "PHANTOM FAIL: marching cubes leaked on the CLOSED phantom — the "
            "wall seal is not watertight"
        )
        assert abs(result.mc_error_pct) < 1.5, (
            f"PHANTOM FAIL: marching-cubes volume error "
            f"{result.mc_error_pct:.3f}% exceeds ±1.5%"
        )
    else:
        assert result.mc_leaked, (
            "PHANTOM FAIL: marching cubes did NOT leak on the HOLED phantom — "
            "leak detection is broken; MC could report wrong volumes on clouds "
            "with real wall holes"
        )
    assert angle_deg < 1.0, (
        f"PHANTOM FAIL: centreline axis off by {angle_deg:.2f}°"
    )
    log.info("  ✓ Phantom PASSED")
    return result
