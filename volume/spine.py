"""
Spine computation: centreline → cylindrical coordinates → gap map.

Step 1 — Centreline (general smooth spline + rotation-minimizing frame)
    Only ONE trajectory exists (the mobile Livox/GLIM path, already in the
    Leica datum) and it is used for BOTH clouds. It is handheld (wanders) and
    out-and-back (loop closure) — an ordering, not a clean line, so:
      a. Reduce to a single monotonic traverse (the outbound leg) to avoid
         doubling artefacts from the return leg.
      b. Fit a smoothing 3-D B-spline (scipy splprep) through it; the
         smoothing factor absorbs handheld wander.
      c. Resample at uniform arclength spacing.
      d. Build the reference frame spanning θ (see Step 2) — never Frenet,
         which flips/twists on low-curvature sections and would scramble θ.
    For a near-straight tunnel this degenerates to nearly a line (no separate
    code path needed); it also works for a curved cave path.

Step 2 — Cylindrical coordinates
    For every point p: find the nearest centreline sample, refine s by
    projecting onto that sample's local tangent, then:
        r     = ‖p_⊥‖                              (p_⊥ = perpendicular part)
        θ     = atan2(p_⊥ · ref2, p_⊥ · ref1)      (−180° … +180°)
    Result saved as (s, r, theta) per cloud.

    What θ=0 MEANS depends on cfg.theta_reference:
      "up"  (default) — ref1 is the world up-vector projected perpendicular to
              the tangent, so **θ=0 is the ceiling, θ=±180 the floor, θ=±90 the
              two side walls** (+90 along t × up). Physically interpretable, and
              cannot drift: each sample's frame is computed independently from
              its own tangent, with no propagation. Degenerates if the tangent
              approaches vertical.
      "rmf" — rotation-minimizing frame from an arbitrary seed, propagated by
              double reflection. θ=0 is meaningless but it survives any path
              geometry, including vertical shafts. Kept for the future
              cave/volumetric case.

Step 3 — Gap map
    Rasterise (s, θ) into cells; empty = hole.
    Reports % missing per cloud, hole size stats, and locates a "golden
    segment" where BOTH clouds are near-complete (for method validation).
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree

from config import Config

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Data structures                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class Centreline:
    samples_xyz: np.ndarray     # (M, 3) resampled spline points, uniform arclength
    s_samples: np.ndarray       # (M,) arclength of each sample [m], s_samples[0] = 0
    tangent: np.ndarray         # (M, 3) unit tangent at each sample
    ref1: np.ndarray            # (M, 3) unit vector spanning θ=0 (ceiling, if theta_reference="up")
    ref2: np.ndarray            # (M, 3) unit vector spanning θ=+90° (= tangent × ref1)
    total_length_m: float       # arclength of the fitted spline [m]
    fit_rms_m: float            # RMS distance, outbound trajectory -> spline
    kdtree: object = field(repr=False, compare=False)  # cKDTree over samples_xyz
    # Max curvature of the fit [1/m]. Kept on the object (not just logged) so a
    # run's summary.json records it — across-run statistics are only comparable
    # if every run's centreline was cusp-free (see cfg.centreline_max_curvature_warn).
    kappa_max_1pm: float = float("nan")


@dataclass
class GapMapResult:
    # Rasterised r values (NaN = empty)
    r_grid: np.ndarray          # (n_s, n_theta)
    s_edges: np.ndarray         # (n_s + 1,) bin edges [m]
    theta_edges: np.ndarray     # (n_theta + 1,) bin edges [deg]
    frac_missing: float         # fraction of grid cells with no data
    # Golden-segment search results (along-s coverage fraction per slab)
    coverage_per_slab: np.ndarray  # (n_s,)


# --------------------------------------------------------------------------- #
#  Step 1 — Centreline (general spline + RMF)                                  #
# --------------------------------------------------------------------------- #

def _find_outbound_leg(traj_pts: np.ndarray) -> np.ndarray:
    """
    Reduce a handheld, out-and-back trajectory to a single monotonic leg.

    The path is time-ordered (raw file order) and returns close to where it
    started (loop closure), so a rough-axis projection is roughly V-shaped
    (or inverted-V, depending on arbitrary SVD sign) with the turnaround at
    the extremum — NOT necessarily a max, and not near either endpoint since
    start ≈ end. Find the turnaround as the point furthest (along the rough
    axis) from the trajectory's own start, which is robust to both the SVD
    sign ambiguity and to a partial/uneven return leg.
    """
    centroid = traj_pts.mean(axis=0)
    X = traj_pts - centroid
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    rough_axis = Vt[0]
    proj = X @ rough_axis
    turn_idx = int(np.argmax(np.abs(proj - proj[0])))
    turn_idx = max(turn_idx, min(len(traj_pts) - 1, 10))
    return traj_pts[: turn_idx + 1]


def _resample_by_arclength(pts: np.ndarray, ds: float) -> np.ndarray:
    """
    Resample a polyline to uniform arclength spacing.

    Must run before splprep. The raw trajectory is time-sampled, so wherever
    the operator paused (once per target here) it holds a dense cluster of
    near-stationary but jittering points. splprep parameterizes by chord
    length, so across such a cluster u barely advances while the position
    jitters — forcing a near-cusp (observed: kappa 808 1/m, i.e. a 1.2 mm bend
    radius in a straight tunnel). Those cusps scramble the tangent and hence
    the RMF, spraying nearby points to wrong (s, theta) and punching a
    full-azimuth stripe of false "no data" into the gap map at every pause.
    Uniform arclength sampling removes the pause weighting entirely.
    """
    step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(step)])
    total = float(cum[-1])
    if total <= 0:
        return pts
    n = max(int(np.floor(total / ds)) + 1, 4)
    s_new = np.linspace(0.0, total, n)
    return np.column_stack([np.interp(s_new, cum, pts[:, k]) for k in range(3)])


def _double_reflection_step(
    p0: np.ndarray, t0: np.ndarray, r0: np.ndarray,
    p1: np.ndarray, t1: np.ndarray,
) -> np.ndarray:
    """One step of the double-reflection RMF transport (Wang et al. 2008)."""
    v1 = p1 - p0
    c1 = float(np.dot(v1, v1))
    if c1 < 1e-14:
        return r0
    rL = r0 - (2.0 / c1) * np.dot(v1, r0) * v1
    tL = t0 - (2.0 / c1) * np.dot(v1, t0) * v1
    v2 = t1 - tL
    c2 = float(np.dot(v2, v2))
    r1 = rL if c2 < 1e-14 else rL - (2.0 / c2) * np.dot(v2, rL) * v2
    r1 = r1 - np.dot(r1, t1) * t1   # re-orthogonalize against the new tangent
    r1 /= np.linalg.norm(r1)
    return r1


def _frame_rmf(samples_xyz: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    """
    Rotation-minimizing frame, seeded from an arbitrary helper vector and
    propagated by double reflection.

    theta=0 has NO physical meaning here — it is wherever the seed pointed.
    Works for ANY path including near-vertical, which is why it is kept as an
    option (see Config.theta_reference).
    """
    ref1 = np.zeros_like(tangent)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(tangent[0], helper)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    r0 = np.cross(tangent[0], helper)
    r0 /= np.linalg.norm(r0)
    ref1[0] = r0
    for i in range(len(tangent) - 1):
        ref1[i + 1] = _double_reflection_step(
            samples_xyz[i], tangent[i], ref1[i],
            samples_xyz[i + 1], tangent[i + 1],
        )
    return ref1


def _frame_gravity_aligned(tangent: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Gravity-aligned frame: ref1 = the world up-vector projected perpendicular
    to the tangent, so **theta=0 points UP (the ceiling)**.

    This is what makes theta physically interpretable: theta=0 ceiling,
    theta=+-180 floor, theta=+-90 the side walls. It needs no propagation
    (each sample is computed independently from its own tangent), so unlike a
    Frenet or RMF frame it cannot drift or twist along the path.

    Degenerates as the tangent approaches vertical — the perpendicular
    component of up shrinks to zero and its direction becomes arbitrary. Guarded
    by theta_up_max_tangent_tilt_deg.
    """
    up = np.asarray(cfg.theta_up_vector, dtype=float)
    up /= np.linalg.norm(up)

    dot = tangent @ up                       # cos(angle between tangent and up)
    tilt_deg = np.degrees(np.arcsin(np.clip(np.abs(dot), 0.0, 1.0)))
    max_tilt = float(tilt_deg.max())
    if max_tilt > cfg.theta_up_max_tangent_tilt_deg:
        log.warning(
            "Centreline tangent tilts up to %.1f deg from horizontal (limit "
            "%.1f) — theta_reference='up' is degenerating toward vertical and "
            "theta will be unstable there. Switch theta_reference to 'rmf'.",
            max_tilt, cfg.theta_up_max_tangent_tilt_deg,
        )

    ref1 = up[np.newaxis, :] - dot[:, np.newaxis] * tangent
    norms = np.linalg.norm(ref1, axis=1, keepdims=True)
    ref1 /= norms
    log.info(
        "Theta frame: 'up' (theta=0 = ceiling, +-180 = floor, +-90 = side "
        "walls). Tangent tilt from horizontal: median=%.2f deg, max=%.2f deg.",
        float(np.median(tilt_deg)), max_tilt,
    )
    return ref1


def fit_centreline(
    traj_pts: np.ndarray,
    cfg: Config,
    orient_toward: np.ndarray | None = None,
) -> Centreline:
    """
    Fit a general smooth-spline centreline with a rotation-minimizing frame.

    Parameters
    ----------
    traj_pts      : (N, 3) trajectory positions (already in Leica frame).
    cfg           : Config (centreline_smoothing_factor, centreline_resample_ds_m).
    orient_toward : optional (3,) point; if given, the centreline is reversed
                    (if needed) so this point has larger s than the start —
                    same role as before, just applied to the general spline.
    """
    if traj_pts.ndim != 2 or traj_pts.shape[1] != 3:
        raise ValueError(f"traj_pts must be (N, 3), got {traj_pts.shape}")

    outbound = _find_outbound_leg(traj_pts)

    # Drop consecutive near-duplicate points (splprep chokes on them)
    d = np.linalg.norm(np.diff(outbound, axis=0), axis=1)
    keep = np.concatenate([[True], d > 1e-6])
    outbound = outbound[keep]
    if len(outbound) < 4:
        raise ValueError(
            f"Trajectory outbound leg too short/degenerate to fit a spline "
            f"({len(outbound)} distinct points)"
        )

    # Residuals are always reported against the raw outbound leg, never the
    # resampled one, so the fit quality stays comparable across resample settings.
    raw_outbound = outbound
    outbound = _resample_by_arclength(outbound, cfg.centreline_traj_resample_ds_m)

    k = min(3, len(outbound) - 1)
    smoothing = cfg.centreline_smoothing_factor * len(outbound)
    tck, _ = splprep(outbound.T, s=smoothing, k=k)

    # Fine evaluation to get a good arclength estimate
    n_fine = max(2000, len(outbound) * 5)
    u_fine = np.linspace(0.0, 1.0, n_fine)
    fine_pts = np.array(splev(u_fine, tck)).T
    seg_len = np.linalg.norm(np.diff(fine_pts, axis=0), axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_length = float(cum_len[-1])
    if total_length <= 0:
        raise ValueError("Fitted centreline has zero length")

    # Resample at uniform arclength spacing
    ds = cfg.centreline_resample_ds_m
    n_samples = max(int(np.floor(total_length / ds)) + 1, 2)
    s_samples = np.linspace(0.0, total_length, n_samples)
    u_at_s = np.interp(s_samples, cum_len, u_fine)
    samples_xyz = np.array(splev(u_at_s, tck)).T

    dpts = np.array(splev(u_at_s, tck, der=1)).T
    tangent = dpts / np.linalg.norm(dpts, axis=1, keepdims=True)

    # Reference frame spanning theta. See Config.theta_reference.
    if cfg.theta_reference == "up":
        ref1 = _frame_gravity_aligned(tangent, cfg)
    elif cfg.theta_reference == "rmf":
        log.info(
            "Theta frame: 'rmf' — theta=0 is an ARBITRARY seed direction with "
            "no physical meaning. Results cannot be stated as ceiling/floor."
        )
        ref1 = _frame_rmf(samples_xyz, tangent)
    else:
        raise ValueError(
            f"theta_reference must be 'up' or 'rmf', got {cfg.theta_reference!r}"
        )
    ref2 = np.cross(tangent, ref1)
    ref2 /= np.linalg.norm(ref2, axis=1, keepdims=True)

    # Curvature guard. A cusp (huge kappa) means the fit is chasing trajectory
    # noise rather than the tunnel axis; it corrupts the RMF and therefore
    # every (s, theta) downstream, so surface it here rather than let it show
    # up as unexplained stripes in the gap map.
    d1 = np.array(splev(u_at_s, tck, der=1)).T
    d2 = np.array(splev(u_at_s, tck, der=2)).T
    kappa = np.linalg.norm(np.cross(d1, d2), axis=1) / np.maximum(
        np.linalg.norm(d1, axis=1) ** 3, 1e-12
    )
    kappa_max = float(kappa.max())

    fit_tree = cKDTree(samples_xyz)
    fit_dists, _ = fit_tree.query(raw_outbound)
    fit_rms = float(np.sqrt(np.mean(fit_dists**2)))
    log.info(
        "Centreline: spline fit RMS=%.4f m, length=%.3f m, %d samples "
        "(ds=%.2f m), max curvature=%.4f 1/m (min bend radius %.1f m)",
        fit_rms, total_length, n_samples, ds,
        kappa_max, 1.0 / max(kappa_max, 1e-12),
    )
    if kappa_max > cfg.centreline_max_curvature_warn:
        log.warning(
            "Centreline max curvature %.3f 1/m (bend radius %.3f m) exceeds "
            "threshold %.3f 1/m — the spline has a cusp/kink and the (s, theta) "
            "mapping near it is unreliable. Increase "
            "centreline_smoothing_factor or centreline_traj_resample_ds_m.",
            kappa_max, 1.0 / max(kappa_max, 1e-12),
            cfg.centreline_max_curvature_warn,
        )
    if fit_rms > cfg.centreline_fit_rms_warn_m:
        log.warning(
            "Centreline spline fit RMS %.4f m > warn threshold %.4f m — "
            "consider more smoothing, or check for trajectory noise/outliers",
            fit_rms, cfg.centreline_fit_rms_warn_m,
        )

    cl = Centreline(
        samples_xyz=samples_xyz, s_samples=s_samples, tangent=tangent,
        ref1=ref1, ref2=ref2, total_length_m=total_length,
        fit_rms_m=fit_rms, kdtree=fit_tree, kappa_max_1pm=kappa_max,
    )

    # Orient so orient_toward has larger s than the start
    if orient_toward is not None:
        s_target, _, _ = to_cylindrical(orient_toward[np.newaxis, :], cl)
        if float(s_target[0]) < total_length / 2.0:
            cl = _reverse_centreline(cl)

    return cl


def _reverse_centreline(cl: Centreline) -> Centreline:
    """Flip traversal direction: reverses sample order/tangent, keeps ref1."""
    samples_xyz = cl.samples_xyz[::-1]
    s_samples = cl.total_length_m - cl.s_samples[::-1]
    tangent = -cl.tangent[::-1]
    ref1 = cl.ref1[::-1]
    ref2 = np.cross(tangent, ref1)
    ref2 /= np.linalg.norm(ref2, axis=1, keepdims=True)
    return Centreline(
        samples_xyz=samples_xyz, s_samples=s_samples, tangent=tangent,
        ref1=ref1, ref2=ref2, total_length_m=cl.total_length_m,
        fit_rms_m=cl.fit_rms_m, kdtree=cKDTree(samples_xyz),
        kappa_max_1pm=cl.kappa_max_1pm,
    )


# --------------------------------------------------------------------------- #
#  Step 2 — Cylindrical coordinates                                            #
# --------------------------------------------------------------------------- #

def to_cylindrical(
    pts: np.ndarray,
    cl: Centreline,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project points into cylindrical coordinates (s, r, θ) against the spline
    centreline: nearest sample -> refine s via local tangent projection ->
    r/θ in that sample's rotation-minimizing frame.

    Returns
    -------
    s     : (N,) chainage (arclength) along the centreline [m]
    r     : (N,) perpendicular distance from the centreline [m]
    theta : (N,) azimuth angle in degrees, −180 … +180
    """
    _, idx = cl.kdtree.query(pts)
    base = cl.samples_xyz[idx]
    tang = cl.tangent[idx]
    v = pts - base
    ds_local = np.einsum("ij,ij->i", v, tang)
    s = cl.s_samples[idx] + ds_local
    v_perp = v - ds_local[:, np.newaxis] * tang
    r = np.linalg.norm(v_perp, axis=1)
    ref1_i = cl.ref1[idx]
    ref2_i = cl.ref2[idx]
    theta_rad = np.arctan2(
        np.einsum("ij,ij->i", v_perp, ref2_i),
        np.einsum("ij,ij->i", v_perp, ref1_i),
    )
    theta_deg = np.degrees(theta_rad)
    return s, r, theta_deg


def from_cylindrical(
    s: np.ndarray,
    r: np.ndarray,
    theta_deg: np.ndarray,
    cl: Centreline,
) -> np.ndarray:
    """Inverse of `to_cylindrical`: (s, r, θ) → (N, 3) xyz in the cloud's frame.

    The frame is interpolated to each s from the centreline samples, exactly as
    `surface_mesh.build_mesh` does when it wraps the r(s, θ) grid to 3-D — the
    two must agree, or an exported cloud would not sit on its own mesh.

    Componentwise interpolation of unit vectors shortens them slightly between
    samples, so ref1/ref2 are renormalised; at the 0.10 m sample spacing the
    correction is ~1e-6 but it costs nothing to be exact.

    Only valid for r below the local radius of curvature — beyond that the
    frames of neighbouring s overlap and the mapping is not injective. The
    centreline curvature guard (κ_max ≈ 0.03 1/m → 30 m radius) keeps a ~1.5 m
    tunnel far from that limit.
    """
    s = np.asarray(s, dtype=float)
    base = np.column_stack([np.interp(s, cl.s_samples, cl.samples_xyz[:, k])
                            for k in range(3)])
    ref1 = np.column_stack([np.interp(s, cl.s_samples, cl.ref1[:, k])
                            for k in range(3)])
    ref2 = np.column_stack([np.interp(s, cl.s_samples, cl.ref2[:, k])
                            for k in range(3)])
    ref1 /= np.linalg.norm(ref1, axis=1, keepdims=True)
    ref2 /= np.linalg.norm(ref2, axis=1, keepdims=True)

    th = np.radians(np.asarray(theta_deg, dtype=float))
    dirs = np.cos(th)[:, np.newaxis] * ref1 + np.sin(th)[:, np.newaxis] * ref2
    return base + np.asarray(r, dtype=float)[:, np.newaxis] * dirs


# --------------------------------------------------------------------------- #
#  Step 3 — Gap map                                                            #
# --------------------------------------------------------------------------- #

def build_gap_map(
    s: np.ndarray,
    theta_deg: np.ndarray,
    r: np.ndarray,
    cfg: Config,
    domain: Tuple[float, float] | None = None,
) -> GapMapResult:
    """
    Rasterise (s, θ) space; report coverage.

    Parameters
    ----------
    domain : (s_start, s_end) limits; if None, uses data extent.
    """
    s_lo, s_hi = domain if domain else (s.min(), s.max())

    # Build bin edges
    s_edges = np.arange(s_lo, s_hi + cfg.gap_map_ds_m, cfg.gap_map_ds_m)
    t_edges = np.arange(-180.0, 180.0 + cfg.gap_map_dtheta_deg, cfg.gap_map_dtheta_deg)
    n_s = len(s_edges) - 1
    n_t = len(t_edges) - 1

    r_grid = np.full((n_s, n_t), np.nan)

    if n_s == 0 or n_t == 0:
        log.warning("Gap map: no bins in domain %.2f … %.2f", s_lo, s_hi)
        return GapMapResult(r_grid, s_edges, t_edges, 1.0, np.array([]))

    si = np.digitize(s, s_edges) - 1
    ti = np.digitize(theta_deg, t_edges) - 1
    in_domain = (si >= 0) & (si < n_s) & (ti >= 0) & (ti < n_t)
    si, ti, r_in = si[in_domain], ti[in_domain], r[in_domain]

    # Median r per cell. Sort by flat cell index once, then slice each cell's
    # contiguous run — the run boundaries come free from np.unique(return_index).
    # (The previous `inv == j` mask per cell rescanned all N points for every
    # cell: ~190k cells x 8M points, which cost ~5 min per cloud.)
    flat_idx = si * n_t + ti
    order = np.argsort(flat_idx, kind="stable")
    flat_sorted = flat_idx[order]
    r_sorted = r_in[order]
    unique_cells, starts = np.unique(flat_sorted, return_index=True)
    bounds = np.append(starts, len(flat_sorted))
    medians = np.array([
        np.median(r_sorted[bounds[j]:bounds[j + 1]])
        for j in range(len(unique_cells))
    ])
    cell_si = unique_cells // n_t
    cell_ti = unique_cells % n_t
    r_grid[cell_si, cell_ti] = medians

    n_cells = n_s * n_t
    n_missing = int(np.sum(np.isnan(r_grid)))
    frac_missing = n_missing / n_cells

    coverage_per_slab = np.array([
        np.sum(~np.isnan(r_grid[i, :])) / n_t for i in range(n_s)
    ])

    return GapMapResult(
        r_grid=r_grid,
        s_edges=s_edges,
        theta_edges=t_edges,
        frac_missing=frac_missing,
        coverage_per_slab=coverage_per_slab,
    )


def check_cloud_aligned(
    pts: np.ndarray,
    cl: Centreline,
    cloud_name: str,
    max_median_r_m: float = 15.0,
) -> None:
    """
    Fail loudly if a cloud is not in the centreline's datum.

    An unregistered cloud does not error anywhere downstream — to_cylindrical
    happily returns huge r, the gap map reports ~96% "missing", and the profiles
    fills the void by interpolation and emits a plausible-looking but
    meaningless volume. Catch it at load time instead. A real tunnel point sits
    within a few metres of the axis, so a median r of tens of metres means the
    wrong file, not a wide tunnel.
    """
    sub = pts[:: max(1, len(pts) // 200_000)]
    _, r, _ = to_cylindrical(sub, cl)
    median_r = float(np.median(r))
    if median_r > max_median_r_m:
        raise ValueError(
            f"Cloud '{cloud_name}' has median r={median_r:.1f} m from the "
            f"centreline (limit {max_median_r_m} m) — it is almost certainly "
            f"not registered into the trajectory/Leica datum. Check which PLY "
            f"is configured."
        )
    log.info("Cloud '%s' alignment OK (median r=%.2f m)", cloud_name, median_r)


def find_golden_segment(
    gm_a: GapMapResult,
    gm_b: GapMapResult,
    target_length_m: float = 25.0,
    min_coverage: float = 0.90,
) -> Tuple[float, float]:
    """
    Find a contiguous s-range of ~target_length_m where BOTH gap maps exceed
    min_coverage in every slab.

    Falls back to the best contiguous window if the threshold is not met.
    """
    # Both maps must share the same s_edges (they are built on the same domain)
    cov = np.minimum(gm_a.coverage_per_slab, gm_b.coverage_per_slab)
    good = cov >= min_coverage
    ds = float(gm_a.s_edges[1] - gm_a.s_edges[0])
    target_n = int(np.ceil(target_length_m / ds))

    best_start, best_count = _longest_window(good, target_n)
    s_lo = float(gm_a.s_edges[best_start])
    s_hi = float(gm_a.s_edges[min(best_start + best_count, len(gm_a.s_edges) - 1)])
    log.info(
        "Golden segment: s=[%.2f, %.2f] m  (%.1f m,  coverage ≥%.0f%%)",
        s_lo, s_hi, s_hi - s_lo, min_coverage * 100,
    )
    return s_lo, s_hi


def _longest_window(good: np.ndarray, target_n: int) -> Tuple[int, int]:
    """Return (start_idx, length) of the longest run of True in good[]."""
    best_start, best_len = 0, 0
    run_start, run_len = 0, 0
    for i, g in enumerate(good):
        if g:
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len > best_len:
                best_len, best_start = run_len, run_start
        else:
            run_len = 0
    return best_start, min(best_len, target_n)


# --------------------------------------------------------------------------- #
#  Figures                                                                     #
# --------------------------------------------------------------------------- #

def plot_centreline(
    traj_pts: np.ndarray,
    cl: Centreline,
    cloud_pts: np.ndarray | None,
    save_path: str | None = None,
) -> None:
    """3-panel figure: XY, XZ, YZ projections of trajectory + fitted spline centreline."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [("X [m]", "Y [m]", 0, 1),
              ("X [m]", "Z [m]", 0, 2),
              ("Y [m]", "Z [m]", 1, 2)]

    line_pts = cl.samples_xyz

    for ax, (xl, yl, xi, yi) in zip(axes, panels):
        if cloud_pts is not None:
            ax.scatter(cloud_pts[::20, xi], cloud_pts[::20, yi],
                       s=0.5, c="lightgray", rasterized=True, label="cloud (1:20)")
        ax.scatter(traj_pts[:, xi], traj_pts[:, yi],
                   s=4, c="steelblue", label="trajectory")
        ax.plot(line_pts[:, xi], line_pts[:, yi], "r-", lw=2, label="centreline")
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_aspect("equal"); ax.legend(markerscale=4, fontsize=7)
    fig.suptitle(
        f"Centreline  length={cl.total_length_m:.2f} m  fit RMS={cl.fit_rms_m:.4f} m"
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_gap_map(
    gm: GapMapResult,
    cloud_name: str,
    save_path: str | None = None,
) -> None:
    """Heatmap of coverage (1=data present, 0=hole). Coverage is strictly
    binary per cell, so the colorbar uses two discrete colors/ticks rather
    than a continuous scale that would imply in-between values exist."""
    coverage = (~np.isnan(gm.r_grid)).astype(float)
    s_ctrs = 0.5 * (gm.s_edges[:-1] + gm.s_edges[1:])
    t_ctrs = 0.5 * (gm.theta_edges[:-1] + gm.theta_edges[1:])

    cmap = ListedColormap(["firebrick", "forestgreen"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    fig, ax = plt.subplots(figsize=(14, 4))
    img = ax.pcolormesh(s_ctrs, t_ctrs, coverage.T, cmap=cmap, norm=norm)
    cbar = plt.colorbar(img, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(["0 (no data)", "1 (data)"])
    cbar.set_label("Coverage")
    ax.set_xlabel("s [m]"); ax.set_ylabel("θ [°]")
    ax.set_title(f"Gap map — {cloud_name}  ({gm.frac_missing * 100:.1f}% missing)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


def _fit_dz_phase(
    ctrs_deg: np.ndarray,
    mean_dz: np.ndarray,
    counts: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Where does the measured vertical offset actually peak, in θ?

    Fits the first harmonic `dz(θ) = A·cos θ + B·sin θ + C` by weighted least
    squares and returns `(phase_deg, amplitude_m, offset_m, r2)`, where
    `phase = atan2(B, A)` is the θ at which dz is maximal — i.e. the direction
    the point cloud says is UP. If θ=0 really is the ceiling, phase ≈ 0.

    Why a harmonic fit and not `argmax(mean_dz)`, which is the obvious thing:

    - **argmax is ill-conditioned exactly where it looks.** Near its maximum a
      cosine is flat (`cos θ ≈ 1 − θ²/2`), so over ±25° the expected signal moves
      by only 9% of its amplitude — ~14 cm at r̄≈1.5 m. That is the same size as
      real bin-to-bin variation from the arch shape (an ice tunnel is not a
      circle) and from uneven point density, so the single tallest 5° bin wanders
      freely across a broad plateau. Measured on five reps of ONE tunnel with a
      provably identical frame, argmax returned −18°, −8°, +22° … while the
      harmonic phase held within ~1°.
    - **The fit reads the whole curve**, and takes most of its phase information
      from θ≈±90° where the cosine is steepest — the region argmax ignores.
    - **Weighting by point count** stops a sparsely covered azimuth (Livox's FOV
      notch is precisely that) from tilting the answer, which is how a coverage
      artefact used to masquerade as a frame error.

    `C` is fitted, not assumed zero: the centreline does not sit exactly at the
    cross-section's vertical centroid, and forcing C=0 would bleed that offset
    into the phase. `r2` says whether dz is a cosine in θ at all — if it is not,
    this test cannot confirm the frame either way, and that is the case worth
    complaining about.
    """
    ok = (counts > 0) & np.isfinite(mean_dz)
    if ok.sum() < 4:
        return float("nan"), float("nan"), float("nan"), float("nan")

    th = np.radians(ctrs_deg[ok])
    y = mean_dz[ok]
    w = counts[ok] / counts[ok].sum()
    sw = np.sqrt(w)

    M = np.column_stack([np.cos(th), np.sin(th), np.ones_like(th)])
    coef, *_ = np.linalg.lstsq(M * sw[:, np.newaxis], y * sw, rcond=None)
    A, B, C = (float(c) for c in coef)

    resid = y - M @ coef
    y_mean = float(np.sum(w * y))
    ss_tot = float(np.sum(w * (y - y_mean) ** 2))
    r2 = 1.0 - float(np.sum(w * resid ** 2)) / ss_tot if ss_tot > 0 else float("nan")

    return float(np.degrees(np.arctan2(B, A))), float(np.hypot(A, B)), C, r2


def plot_theta_reference(
    pts: np.ndarray,
    cl: Centreline,
    cfg: Config,
    cloud_name: str,
    domain: Tuple[float, float] | None = None,
    save_path: str | None = None,
) -> dict:
    """
    Verification figure for the θ reference frame: is θ=0 really UP?

    Deliberately checks θ against **world coordinates**, not against the
    formula that produced it — re-plotting r·cos θ against the definition of
    θ would be circular and would "confirm" any bug. Instead every panel
    derives the vertical direction from the raw point z values.

    Panels
    ------
    A  ref1·ẑ along s. ref1 is meant to be "up, projected ⊥ to the tangent",
       so this should sit at cos(tangent tilt) ≈ 1 for a near-horizontal
       tunnel. Flat ≈1 = correct; drifting/oscillating = wrong frame.
    B  Measured mean vertical offset of points, (p − base)·ẑ, binned by θ.
       Uses raw world z only, so it tests the claim "θ=0 is the ceiling"
       independently of the formula that produced θ. The claim is checked by
       fitting the FIRST HARMONIC, dz(θ) = A·cos θ + B·sin θ + C, and reading
       its phase φ = atan2(B, A): if θ=0 is up, φ ≈ 0.
       Not by the argmax of the curve — see the phase note below.
    C  One cross-section drawn in a WORLD-referenced basis: horizontal axis
       ĥ = normalize(t × ẑ), vertical axis ẑ. Up is genuinely up on the page.
       The θ=0 ray is overlaid — it must point straight up.

    Returns a dict of the numeric checks so a caller can assert on them.
    """
    up = np.asarray(cfg.theta_up_vector, dtype=float)
    up /= np.linalg.norm(up)

    # ---- Panel A: is ref1 as close to "up" as the tangent allows? ----
    ref1_dot_up = cl.ref1 @ up
    tangent_tilt = np.degrees(np.arcsin(np.clip(np.abs(cl.tangent @ up), 0, 1)))
    expected = np.cos(np.radians(tangent_tilt))   # |ref1·up| if ref1 = up ⊥ t

    # ---- Project a subsample of points, keeping their raw world offsets ----
    step = max(1, len(pts) // 400_000)
    sub = pts[::step]
    s_sub, r_sub, th_sub = to_cylindrical(sub, cl)
    if domain is not None:
        in_dom = (s_sub >= domain[0]) & (s_sub <= domain[1])
        sub, s_sub, r_sub, th_sub = sub[in_dom], s_sub[in_dom], r_sub[in_dom], th_sub[in_dom]

    _, idx = cl.kdtree.query(sub)
    base = cl.samples_xyz[idx]
    tang = cl.tangent[idx]
    v = sub - base
    v_perp = v - np.einsum("ij,ij->i", v, tang)[:, np.newaxis] * tang
    dz = v_perp @ up                      # raw world vertical offset

    # ---- Panel B: mean measured dz per θ bin ----
    edges = np.arange(-180.0, 181.0, 5.0)
    ctrs = 0.5 * (edges[:-1] + edges[1:])
    bi = np.clip(np.digitize(th_sub, edges) - 1, 0, len(ctrs) - 1)
    counts = np.bincount(bi, minlength=len(ctrs)).astype(float)
    mean_dz = np.array([
        dz[bi == k].mean() if np.any(bi == k) else np.nan for k in range(len(ctrs))
    ])
    mean_r = np.array([
        r_sub[bi == k].mean() if np.any(bi == k) else np.nan for k in range(len(ctrs))
    ])
    phase, amp, offset, fit_r2 = _fit_dz_phase(ctrs, mean_dz, counts)
    dz_fit = amp * np.cos(np.radians(ctrs - phase)) + offset
    # Kept for continuity with older notes/figures, and reported as secondary:
    # this is the estimator that used to drive the warning, and it is noisy by
    # construction (see _fit_dz_phase).
    peak_theta = float(ctrs[np.nanargmax(mean_dz)])
    trough_theta = float(ctrs[np.nanargmin(mean_dz)])

    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.4])

    ax = fig.add_subplot(gs[0, :])
    ax.plot(cl.s_samples, ref1_dot_up, "steelblue", lw=1.5, label="ref1 · ẑ  (measured)")
    ax.plot(cl.s_samples, expected, "k--", lw=1,
            label="cos(tangent tilt)  (expected upper bound)")
    ax.set_ylim(0.9, 1.005)
    ax.set_xlabel("s [m]"); ax.set_ylabel("ref1 · ẑ")
    ax.set_title(f"A — is θ=0 pointing up along the whole centreline?  "
                 f"(mode='{cfg.theta_reference}')  flat ≈1 is correct")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(ctrs, mean_dz, "steelblue", lw=2, label="measured mean (p−base)·ẑ")
    ax.plot(ctrs, mean_r * np.cos(np.radians(ctrs)), "salmon", ls="--", lw=1.2,
            label="r̄·cos θ  (expected if θ=0 is up)")
    ax.plot(ctrs, dz_fit, "black", ls="-", lw=1.2, alpha=0.8,
            label=f"1st-harmonic fit (R²={fit_r2:.3f})")
    ax.axvline(0, color="green", ls=":", lw=1.5)
    ax.axvline(phase, color="red", ls="-", lw=1.5, alpha=0.8,
               label=f"fitted phase θ={phase:+.1f}°  ← the test")
    ax.axvline(peak_theta, color="grey", ls=":", lw=1, alpha=0.6,
               label=f"argmax bin θ={peak_theta:.0f}° (noisy, not used)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("θ [°]"); ax.set_ylabel("vertical offset [m]")
    ax.set_title("B — vertical offset vs θ (raw world z)\n"
                 "fitted phase ≈0 = ceiling; the flat-topped\n"
                 "argmax wanders, so it is not the test", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xticks(np.arange(-180, 181, 60))

    # ---- Panel C: cross-section in a world-referenced basis ----
    ax = fig.add_subplot(gs[1, 1])
    s_mid = float(np.median(s_sub))
    sel = np.abs(s_sub - s_mid) < 0.25
    i_mid = int(np.argmin(np.abs(cl.s_samples - s_mid)))
    t_mid = cl.tangent[i_mid]
    h = np.cross(t_mid, up)                # horizontal, ⊥ to tunnel axis
    h /= np.linalg.norm(h)
    px, py = v_perp[sel] @ h, v_perp[sel] @ up
    ax.scatter(px, py, s=1.5, c="lightgray", rasterized=True)

    # Scale arrows to the LOCAL cross-section, not the global max r (which is an
    # outlier metres away and throws the arrows off the axes).
    L = float(np.percentile(r_sub[sel], 90)) if sel.any() else 1.0
    for ang, col, lab in ((0, "green", "θ=0"), (90, "orange", "θ=+90"),
                          (180, "purple", "θ=±180"), (-90, "brown", "θ=−90")):
        d3 = (np.cos(np.radians(ang)) * cl.ref1[i_mid]
              + np.sin(np.radians(ang)) * cl.ref2[i_mid])
        dx, dy = L * (d3 @ h), L * (d3 @ up)
        ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5))
        ax.text(dx * 1.18, dy * 1.18, lab, color=col, fontsize=10,
                fontweight="bold", ha="center", va="center")
    ax.plot(0, 0, "k+", ms=10, mew=2)
    lim = L * 1.45
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("horizontal (t × ẑ) [m]"); ax.set_ylabel("world up ẑ [m]")
    ax.set_title(f"C — cross-section at s={s_mid:.1f} m, drawn world-up\n"
                 "green θ=0 arrow must point straight up")
    ax.grid(alpha=0.3)

    fig.suptitle(f"θ reference verification — {cloud_name}  "
                 f"(theta_reference='{cfg.theta_reference}')", fontsize=13)
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)

    checks = {
        "ref1_dot_up_min": float(ref1_dot_up.min()),
        "ref1_dot_up_median": float(np.median(ref1_dot_up)),
        "tangent_tilt_max_deg": float(tangent_tilt.max()),
        # The test: phase of the first harmonic of dz(θ) (see _fit_dz_phase).
        "dz_phase_theta_deg": phase,
        "dz_amplitude_m": amp,
        "dz_offset_m": offset,
        "dz_fit_r2": fit_r2,
        # Secondary/legacy, noisy — do NOT assert on these.
        "dz_peak_theta_deg": peak_theta,
        "dz_trough_theta_deg": trough_theta,
    }
    log.info(
        "Theta check (%s): ref1·ẑ median=%.4f min=%.4f (expect cos(tilt)=%.4f) "
        "| vertical-offset phase θ=%+.1f° — want ≈0 = ceiling "
        "(amplitude %.2f m, fit R²=%.3f; raw argmax bin θ=%.0f°, flat-topped so "
        "not the test)",
        cloud_name, checks["ref1_dot_up_median"], checks["ref1_dot_up_min"],
        float(np.cos(np.radians(np.median(tangent_tilt)))),
        phase, amp, fit_r2, peak_theta,
    )
    if cfg.theta_reference == "up":
        if np.isfinite(fit_r2) and fit_r2 < 0.90:
            log.warning(
                "Theta check (%s): the vertical offset is not a cosine in θ "
                "(fit R²=%.3f) — this test cannot confirm or refute the frame. "
                "Check the centreline (cusps scramble θ) before reading any "
                "ceiling/floor statement off the gap map.", cloud_name, fit_r2,
            )
        elif np.isfinite(phase) and abs(phase) > 15.0:
            log.warning(
                "Theta check (%s): vertical offset peaks at θ=%+.1f°, not 0° — "
                "θ=0 is NOT the ceiling. The frame is wrong.", cloud_name, phase,
            )
    return checks


def plot_radius_histogram(
    r: np.ndarray,
    cloud_name: str,
    save_path: str | None = None,
) -> None:
    """r histogram: expect single hump, nothing near 0."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(r, bins=200, color="steelblue", edgecolor="none")
    ax.set_xlabel("r [m]"); ax.set_ylabel("count")
    ax.set_title(f"Radius distribution — {cloud_name}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)
