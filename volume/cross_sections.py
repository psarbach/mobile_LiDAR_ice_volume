"""
Cross-section estimators. Two methods, both consuming the (s, r, θ) arrays
produced by spine.py:

  run_profiles()    — per-slab cross-section area.   THE CORE ESTIMATOR.
  run_hull_bound()  — per-slab convex-hull area.     AN UPPER BOUND ONLY.

profiles
--------
For each slab [s_c − Δs/2, s_c + Δs/2]:
  1. Bin θ into 1° bins; take MEDIAN r per bin (robust to outliers).
  2. Star-shape guard: flag θ bins whose r values are bimodal (a radial gap
     > profile_cluster_gap_r_m means two separated clusters along one ray, so
     "the radius here" is ambiguous); take the OUTERMOST cluster, and warn only
     if the fraction in a slab exceeds profile_cluster_abort_frac.
     NB: `n_bimodal_slabs` counts slabs with >=1 flagged bin OUT OF 360, which
     is near-guaranteed and hence a misleading metric — the meaningful figure is
     the bin-level fraction (~4.7% on Leica).
  3. Fill missing θ bins:
       a. Circular interpolation along θ (within the slab).
       b. Along-s interpolation from neighbouring slabs for cells still NaN.
  4. Area = ½ Σᵢ rᵢ rᵢ₊₁ sin(Δθ)  (shoelace formula for a polar polygon).
  5. Volume via trapezoidal rule AND scipy.integrate.simpson (difference =
     discretisation indicator).

hull bound
----------
Same slab loop; 2-D convex-hull area of the slab's points. OVER-estimates any
concave or deformed cross-section, so it is only ever an upper bound — never
quote it as an estimate. (It also UNDER-reads when data is missing, since the
hull spans the hole: on the phantom hole test it lands at −2.15%. So it is only
a bound when coverage is good.)

Known bias shared with the surface-mesh method
----------------------------------------------
Integration runs between the FIRST and LAST SLAB CENTRE, so it misses ds/2 at
each end of the domain — a systematic −ds/L (≈ −0.19% at ds=0.25 m over 133 m;
−0.25% on the phantom). This is most of the phantom's headline "−0.26% error":
against the span actually integrated the error is −0.008%. Ā is computed as
V / L_domain, so it carries the same bias.
"""

import logging
from dataclasses import dataclass, field
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson as scipy_simpson
from scipy.spatial import ConvexHull

from config import Config

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Result containers                                                           #
# --------------------------------------------------------------------------- #

@dataclass
class ProfileResult:
    V_trap: float               # m³ — trapezoidal integration
    V_simp: float               # m³ — Simpson's rule integration
    mean_area_m2: float         # m² — V_trap / L
    length_m: float             # m — domain length
    s_centers: np.ndarray       # (n_slabs,) slab centres [m]
    A_s: np.ndarray             # (n_slabs,) cross-section areas [m²]
    frac_interp: np.ndarray     # (n_slabs,) fraction interpolated per slab
    n_bimodal_slabs: int        # slabs flagged for star-shape issue
    cloud_name: str
    # The wall this method actually integrated, kept so it can be exported and
    # looked at (profile_cloud.py) instead of only existing inside the integral.
    # Nothing downstream of the volume reads these — they are a record, not an
    # input, so the numbers above are unaffected by their presence.
    r_grid: np.ndarray = None      # (n_slabs, n_theta) filled radii [m], NaN = unfilled
    theta_centers: np.ndarray = None  # (n_theta,) bin centres [deg]
    was_interp: np.ndarray = None  # (n_slabs, n_theta) bool — True = interpolated


@dataclass
class HullBoundResult:
    V_hull: float               # m³ — convex-hull integration (upper bound)
    mean_area_m2: float
    length_m: float
    s_centers: np.ndarray
    A_hull_s: np.ndarray        # (n_slabs,) convex-hull areas [m²]
    cloud_name: str


# --------------------------------------------------------------------------- #
#  Internal helpers                                                            #
# --------------------------------------------------------------------------- #

def _fill_circular(row: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fill NaN values in a 1-D array via linear interpolation with circular
    (periodic) wrap-around.

    Returns
    -------
    filled   : array with no NaN (same length as row)
    was_nan  : boolean mask, True where original value was NaN
    """
    n = len(row)
    was_nan = np.isnan(row)
    if not was_nan.any():
        return row.copy(), was_nan
    if was_nan.all():
        # No data in this row — signal caller to use along-s fill
        return row.copy(), was_nan

    valid_idx = np.where(~was_nan)[0]
    valid_val = row[valid_idx]

    # Extend valid indices one period in each direction for circular wrap
    idx_ext = np.concatenate([valid_idx - n, valid_idx, valid_idx + n])
    val_ext = np.concatenate([valid_val, valid_val, valid_val])

    all_idx = np.arange(n)
    filled = np.interp(all_idx, idx_ext, val_ext)
    return filled, was_nan


def _polar_area(r: np.ndarray, dtheta_rad: float) -> float:
    """
    Shoelace area of a closed polar polygon with equal angular spacing.

    A = ½ Σᵢ rᵢ rᵢ₊₁ sin(Δθ)   (exact for polygon; matches π r² for circle)
    """
    r_next = np.roll(r, -1)
    return 0.5 * float(np.sum(r * r_next * np.sin(dtheta_rad)))


def _detect_bimodal(r_in_bin: np.ndarray, gap_threshold: float) -> bool:
    """True if the sorted r values in a single theta-bin have a large gap."""
    if len(r_in_bin) < 2:
        return False
    rs = np.sort(r_in_bin)
    return bool(np.max(np.diff(rs)) > gap_threshold)


# --------------------------------------------------------------------------- #
#  Profiles — cross-section areas                                              #
# --------------------------------------------------------------------------- #

def run_profiles(
    s: np.ndarray,
    r: np.ndarray,
    theta_deg: np.ndarray,
    domain: Tuple[float, float],
    cfg: Config,
    cloud_name: str = "",
) -> ProfileResult:
    """
    Cross-section profile method.

    Parameters
    ----------
    s, r, theta_deg : cylindrical coordinates from spine.to_cylindrical()
    domain          : (s_start, s_end) in metres; same for all methods
    """
    s_start, s_end = domain
    L = s_end - s_start
    dtheta_deg = cfg.profile_dtheta_deg
    dtheta_rad = np.radians(dtheta_deg)
    ds = cfg.profile_ds_m
    gap_m = cfg.profile_cluster_gap_r_m
    abort_frac = cfg.profile_cluster_abort_frac
    min_cov = cfg.profile_min_theta_coverage

    # Theta bin edges: 0°…360° (closed ring)
    t_edges = np.arange(-180.0, 180.0 + dtheta_deg, dtheta_deg)
    n_theta = len(t_edges) - 1
    t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])

    # Slab centres
    s_centers = np.arange(s_start + ds / 2, s_end, ds)
    n_slabs = len(s_centers)

    if n_slabs == 0:
        log.error("Profiles: no slabs in domain [%.2f, %.2f]", s_start, s_end)
        empty = np.array([])
        return ProfileResult(0, 0, 0, L, empty, empty, empty, 0, cloud_name)

    # ---- Build raw median-r grid  (n_slabs × n_theta) ----
    r_grid = np.full((n_slabs, n_theta), np.nan)
    n_bimodal = 0

    for i, sc in enumerate(s_centers):
        slab_mask = (s >= sc - ds / 2) & (s < sc + ds / 2)
        if not slab_mask.any():
            continue

        s_r = r[slab_mask]
        s_theta = theta_deg[slab_mask]

        # Bin into theta cells
        ti = np.digitize(s_theta, t_edges) - 1
        valid_ti = (ti >= 0) & (ti < n_theta)
        ti, s_r = ti[valid_ti], s_r[valid_ti]

        bimodal_count = 0
        for j in range(n_theta):
            bin_mask = ti == j
            if not bin_mask.any():
                continue
            r_in_bin = s_r[bin_mask]
            if _detect_bimodal(r_in_bin, gap_m):
                bimodal_count += 1
                # Take outermost cluster: max r in the bin
                r_grid[i, j] = float(np.max(r_in_bin))
            else:
                r_grid[i, j] = float(np.median(r_in_bin))

        if bimodal_count > 0:
            n_bimodal += 1
            frac = bimodal_count / n_theta
            if frac > abort_frac:
                log.warning(
                    "Slab s=%.2f: %.0f%% of theta bins are bimodal — "
                    "star-shape assumption may be violated",
                    sc, frac * 100,
                )

    if n_bimodal > 0:
        log.info(
            "Profiles (%s): %d/%d slabs had bimodal theta bins (took outermost r)",
            cloud_name, n_bimodal, n_slabs,
        )

    # ---- Pass 1: circular theta fill per slab ----
    r_grid_filled = r_grid.copy()
    was_interp = np.zeros_like(r_grid, dtype=bool)

    fully_empty_slabs = []
    for i in range(n_slabs):
        row = r_grid[i]
        coverage = np.sum(~np.isnan(row)) / n_theta
        if coverage < min_cov:
            fully_empty_slabs.append(i)
            continue
        filled_row, nan_mask = _fill_circular(row)
        r_grid_filled[i] = filled_row
        was_interp[i] = nan_mask

    # ---- Pass 2: along-s fill for fully-empty slabs ----
    # For each NaN cell in a fully-empty slab, interpolate from along-s neighbors
    if fully_empty_slabs:
        log.info(
            "Profiles (%s): %d/%d slabs have <%.0f%% theta coverage → "
            "along-s fill",
            cloud_name, len(fully_empty_slabs), n_slabs, min_cov * 100,
        )
        for j in range(n_theta):          # for each theta column
            col = r_grid_filled[:, j].copy()
            valid_idx = np.where(~np.isnan(col))[0]
            if len(valid_idx) < 2:
                continue
            all_idx = np.arange(n_slabs)
            col_filled = np.interp(all_idx, valid_idx, col[valid_idx],
                                   left=col[valid_idx[0]],
                                   right=col[valid_idx[-1]])
            for i in fully_empty_slabs:
                if np.isnan(r_grid_filled[i, j]):
                    r_grid_filled[i, j] = col_filled[i]
                    was_interp[i, j] = True

    # ---- Compute area per slab ----
    A_s = np.zeros(n_slabs)
    frac_interp = np.zeros(n_slabs)
    for i in range(n_slabs):
        row = r_grid_filled[i]
        if np.isnan(row).any():
            # Still some NaN → treat whole slab as missing (zero area, excluded)
            A_s[i] = np.nan
            frac_interp[i] = 1.0
        else:
            A_s[i] = _polar_area(row, dtheta_rad)
            frac_interp[i] = float(np.sum(was_interp[i]) / n_theta)

    # Drop NaN slabs from integration
    valid = ~np.isnan(A_s)
    if not valid.any():
        log.error("Profiles (%s): all slabs are empty", cloud_name)
        return ProfileResult(0, 0, 0, L, s_centers, A_s, frac_interp, n_bimodal,
                             cloud_name, r_grid_filled, t_centers, was_interp)

    s_v = s_centers[valid]
    A_v = A_s[valid]

    # ---- Integrate over the FULL domain [s_start, s_end] ----
    # trapezoid/Simpson over slab CENTRES span only [s_v[0], s_v[-1]], leaving
    # the outer half-slab at each end unmeasured — a systematic -ds/L bias
    # (-0.19% at ds=0.25 m over 133 m; -0.25% on the phantom, which was most of
    # its apparent "-0.26% error"). Add the two end pieces explicitly.
    #
    # A is held CONSTANT across each end piece rather than linearly
    # extrapolated: the pieces are only ~ds/2 wide, so the difference is second
    # order (<0.001% of V), while holding constant cannot be destabilised by a
    # noisy or sparsely-sampled end slab.
    if len(s_v) >= 2:
        core_trap = float(np.trapezoid(A_v, x=s_v))
        core_simp = float(scipy_simpson(A_v, x=s_v))
    else:
        core_trap = core_simp = 0.0
    head = float(A_v[0]) * max(0.0, float(s_v[0]) - s_start)
    tail = float(A_v[-1]) * max(0.0, s_end - float(s_v[-1]))
    V_trap = core_trap + head + tail
    V_simp = core_simp + head + tail
    mean_area = V_trap / L
    log.debug(
        "Profiles (%s): end caps head=%.3f m³ tail=%.3f m³ (%.2f%% of V) — "
        "closes the ds/2 truncation at each domain edge",
        cloud_name, head, tail, (head + tail) / V_trap * 100 if V_trap else 0.0,
    )

    log.info(
        "Profiles (%s): V_trap=%.2f m³  V_simp=%.2f m³  "
        "Ā=%.3f m²  L=%.2f m  interp_frac_mean=%.1f%%",
        cloud_name, V_trap, V_simp, mean_area, L,
        float(np.nanmean(frac_interp)) * 100,
    )

    return ProfileResult(
        V_trap=V_trap,
        V_simp=V_simp,
        mean_area_m2=mean_area,
        length_m=L,
        s_centers=s_centers,
        A_s=A_s,
        frac_interp=frac_interp,
        n_bimodal_slabs=n_bimodal,
        cloud_name=cloud_name,
        r_grid=r_grid_filled,
        theta_centers=t_centers,
        was_interp=was_interp,
    )


# --------------------------------------------------------------------------- #
#  Hull bound — convex hull per slice (UPPER BOUND)                            #
# --------------------------------------------------------------------------- #

def run_hull_bound(
    s: np.ndarray,
    r: np.ndarray,
    theta_deg: np.ndarray,
    domain: Tuple[float, float],
    cfg: Config,
    cloud_name: str = "",
) -> HullBoundResult:
    """
    Convex-hull upper bound.  Uses the SAME slab loop as the profiles method.

    For each slab, project points to the 2-D perpendicular plane and compute
    the convex-hull area.  Meaningful only when the slab has ≥ 3 non-collinear
    points; otherwise NaN.

    The result OVER-estimates on a concave or deformed shape.
    """
    s_start, s_end = domain
    L = s_end - s_start
    ds = cfg.profile_ds_m

    s_centers = np.arange(s_start + ds / 2, s_end, ds)
    n_slabs = len(s_centers)
    A_hull = np.full(n_slabs, np.nan)

    theta_rad = np.radians(theta_deg)
    x2d = r * np.cos(theta_rad)
    y2d = r * np.sin(theta_rad)

    for i, sc in enumerate(s_centers):
        mask = (s >= sc - ds / 2) & (s < sc + ds / 2)
        if mask.sum() < 3:
            continue
        pts2d = np.column_stack([x2d[mask], y2d[mask]])
        try:
            hull = ConvexHull(pts2d)
            A_hull[i] = hull.volume   # scipy ConvexHull: .volume = area in 2-D
        except Exception:
            pass

    valid = ~np.isnan(A_hull)
    if not valid.any():
        log.warning("Hull bound (%s): no valid slabs for convex hull", cloud_name)
        return HullBoundResult(0, 0, L, s_centers, A_hull, cloud_name)

    # Same full-domain end capping as run_profiles — otherwise the hull carries
    # the -ds/L truncation that the profiles method no longer does, and the
    # hull-vs-profiles comparison silently conflates that 0.25% with the real
    # concavity gap the bound is meant to expose.
    s_v = s_centers[valid]
    A_v = A_hull[valid]
    core = float(np.trapezoid(A_v, x=s_v)) if len(s_v) >= 2 else 0.0
    V_hull = (core
              + float(A_v[0]) * max(0.0, float(s_v[0]) - s_start)
              + float(A_v[-1]) * max(0.0, s_end - float(s_v[-1])))
    mean_area = V_hull / L

    log.info(
        "Hull bound (%s): V_hull=%.2f m³  Ā=%.3f m²  "
        "[upper bound — over-estimates concave sections]",
        cloud_name, V_hull, mean_area,
    )

    return HullBoundResult(
        V_hull=V_hull,
        mean_area_m2=mean_area,
        length_m=L,
        s_centers=s_centers,
        A_hull_s=A_hull,
        cloud_name=cloud_name,
    )


# --------------------------------------------------------------------------- #
#  Figures                                                                     #
# --------------------------------------------------------------------------- #

def plot_area_profile(
    prof: ProfileResult,
    hull: HullBoundResult | None,
    save_path: str | None = None,
) -> None:
    """A(s) curve from the profiles and hull-bound methods on the same axes."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax = axes[0]
    ax.fill_between(prof.s_centers, prof.A_s, alpha=0.3, color="steelblue")
    ax.plot(prof.s_centers, prof.A_s, "steelblue", lw=1.5, label=f"profiles ({prof.cloud_name})")
    if hull is not None:
        ax.plot(hull.s_centers, hull.A_hull_s, "salmon", lw=1, ls="--",
                label=f"hull bound ({hull.cloud_name})")
    ax.set_ylabel("Area [m²]")
    ax.legend()
    ax.set_title(
        f"Cross-section profile — {prof.cloud_name}\n"
        f"V(trap)={prof.V_trap:.2f} m³  V(simp)={prof.V_simp:.2f} m³  "
        f"Ā={prof.mean_area_m2:.3f} m²"
    )

    ax2 = axes[1]
    ax2.plot(prof.s_centers, prof.frac_interp * 100, "darkorange", lw=1)
    ax2.axhline(50, ls=":", color="red", alpha=0.5)
    ax2.set_ylabel("Interpolated [%]")
    ax2.set_xlabel("s [m]")
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_cross_sections(
    s: np.ndarray,
    r: np.ndarray,
    theta_deg: np.ndarray,
    prof: ProfileResult,
    domain: Tuple[float, float],
    n_panels: int = 6,
    save_path: str | None = None,
) -> None:
    """Plot n_panels cross-sections evenly spaced along the domain."""
    if len(prof.s_centers) == 0:
        log.warning(
            "plot_cross_sections (%s): domain has no slabs — skipping figure",
            prof.cloud_name,
        )
        return

    s_start, s_end = domain
    ds = prof.length_m / (n_panels + 1)
    s_show = [s_start + ds * (k + 1) for k in range(n_panels)]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    dtheta_deg = (prof.s_centers[1] - prof.s_centers[0]) if len(prof.s_centers) > 1 else 0.25
    dtheta_rad = np.radians(1.0)  # profiles use 1° bins

    t_edges = np.arange(-180.0, 180.0 + 1.0, 1.0)
    n_theta = len(t_edges) - 1
    t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])

    for ax, sc in zip(axes.ravel(), s_show):
        mask = (s >= sc - prof.length_m / (2 * len(prof.s_centers))) & \
               (s < sc + prof.length_m / (2 * len(prof.s_centers)))
        theta_rad = np.radians(theta_deg[mask])
        x2d = r[mask] * np.cos(theta_rad)
        y2d = r[mask] * np.sin(theta_rad)
        ax.scatter(x2d, y2d, s=0.5, c="lightgray", rasterized=True)

        ax.set_aspect("equal")
        ax.set_title(f"s={sc:.1f} m")
        ax.set_xlabel("x⊥ [m]"); ax.set_ylabel("y⊥ [m]")

    fig.suptitle(f"Cross-sections — {prof.cloud_name}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_cross_sections_combined(
    data_by_cloud: dict,
    domain: Tuple[float, float],
    n_panels: int = 6,
    window_m: float = 0.15,
    save_path: str | None = None,
) -> None:
    """
    Overlay cross-sections from multiple clouds at the SAME s-locations (one
    color per cloud in every subplot), for direct shape comparison.

    Parameters
    ----------
    data_by_cloud : cloud_name -> (s, r, theta_deg) cylindrical-coordinate
                    arrays (same arrays passed to run_profiles/run_hull_bound).
    domain        : (s_start, s_end) — same domain used for the per-cloud
                    plots, so panels line up at the same chainage.
    window_m      : half-width of the along-s window of points shown per
                    panel (independent of any per-cloud slab count).
    """
    s_start, s_end = domain
    ds = (s_end - s_start) / (n_panels + 1)
    s_show = [s_start + ds * (k + 1) for k in range(n_panels)]

    palette = ["steelblue", "salmon", "seagreen", "darkorange"]
    colors = {name: palette[i % len(palette)] for i, name in enumerate(data_by_cloud)}

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, sc in zip(axes.ravel(), s_show):
        for name, (s, r, theta_deg) in data_by_cloud.items():
            mask = (s >= sc - window_m) & (s < sc + window_m)
            theta_rad = np.radians(theta_deg[mask])
            x2d = r[mask] * np.cos(theta_rad)
            y2d = r[mask] * np.sin(theta_rad)
            ax.scatter(x2d, y2d, s=0.5, c=colors[name], rasterized=True, label=name)

        ax.set_aspect("equal")
        ax.set_title(f"s={sc:.1f} m")
        ax.set_xlabel("x⊥ [m]"); ax.set_ylabel("y⊥ [m]")

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", markerscale=10)
    fig.suptitle("Cross-sections — cloud comparison")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)
