"""
Lateral (azimuthal) comparison, Leica vs Livox.

The per-segment module (slice_compare.py, fig 07) answers *where along s* the two
clouds disagree. This answers the orthogonal question: *where around the
cross-section* (θ) do they disagree, and where does each cloud actually have
data. It rasterises both clouds onto the SAME r(s, θ) grid the profiles/mesh
methods use (build_r_grid, before any hole fill) and compares them cell-by-cell:

  - Δr(s, θ) = r_other − r_ref, only where BOTH observed a wall (a like-for-like
    radius difference — no interpolation, no integration).
  - per-θ coverage: the fraction of slabs each cloud observed at that azimuth,
    which makes the Livox vertical-FOV notch (a fixed θ band, the ceiling) visible
    directly, and shows it is a *lateral* deficit, not a longitudinal one.
  - per-θ mean Δr: the azimuthal signature of the shape difference.

θ = 0° is the ceiling (gravity-up), ±180° the floor, by the physical θ frame.
"""

import logging
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import Config
from surface_mesh import build_r_grid

log = logging.getLogger(__name__)


def run_lateral_compare(
    cyl_by_cloud: Dict[str, dict],
    domain: Tuple[float, float],
    cfg: Config,
    ref_cloud: str = "leica",
    other_cloud: str = "livox",
    save_path: str | None = None,
) -> dict:
    """
    Compare two clouds azimuthally on the shared r(s, θ) grid.

    Parameters
    ----------
    cyl_by_cloud : {cloud_name: {"s":.., "r":.., "theta":..}} — raw cylindrical
                   coords (the pipeline's cyl_data / *_cyl.npz cache).
    domain       : (s_start, s_end); both clouds rasterised on the same grid.

    Returns a dict of grids + per-θ marginals + summary stats.
    """
    if ref_cloud not in cyl_by_cloud or other_cloud not in cyl_by_cloud:
        raise ValueError(f"Need '{ref_cloud}' and '{other_cloud}'; got {list(cyl_by_cloud)}")

    grids, s_c, t_c = {}, None, None
    for name in (ref_cloud, other_cloud):
        d = cyl_by_cloud[name]
        g, s_c, t_c = build_r_grid(d["s"], d["r"], d["theta"], domain, cfg)
        grids[name] = g

    r_ref, r_oth = grids[ref_cloud], grids[other_cloud]
    both = ~np.isnan(r_ref) & ~np.isnan(r_oth)
    dr = np.where(both, r_oth - r_ref, np.nan)          # (n_s, n_theta)

    # per-θ marginals (average across s)
    cov_ref = np.mean(~np.isnan(r_ref), axis=0) * 100.0   # % of slabs observed
    cov_oth = np.mean(~np.isnan(r_oth), axis=0) * 100.0
    with np.errstate(invalid="ignore"):
        dr_theta = np.nanmean(dr, axis=0)                 # mean Δr per azimuth
    both_theta = np.mean(both, axis=0) * 100.0

    med_abs = float(np.nanmedian(np.abs(dr)))
    med_signed = float(np.nanmedian(dr))
    frac_both = float(np.mean(both))

    log.info(
        "Lateral comparison (%s−%s): overlap %.1f%% of (s,θ) cells; "
        "median Δr=%+.3f m (|Δr| median %.3f m).",
        other_cloud, ref_cloud, frac_both * 100.0, med_signed, med_abs,
    )
    # azimuth of worst coverage gap for the other cloud
    worst = int(np.argmin(cov_oth))
    log.info(
        "Lateral comparison: %s coverage minimum %.0f%% at θ≈%+.0f° "
        "(vs %s %.0f%% there) — the FOV notch is a lateral band.",
        other_cloud, cov_oth[worst], t_c[worst], ref_cloud, cov_ref[worst],
    )

    if save_path:
        _plot(r_ref, r_oth, dr, s_c, t_c, cov_ref, cov_oth, dr_theta, both_theta,
              ref_cloud, other_cloud, save_path)

    return {
        "s_centers": s_c, "theta_centers": t_c,
        "r_ref": r_ref, "r_oth": r_oth, "dr": dr,
        "cov_ref": cov_ref, "cov_oth": cov_oth,
        "dr_theta": dr_theta, "both_theta": both_theta,
        "median_abs_dr": med_abs, "median_signed_dr": med_signed,
        "frac_overlap": frac_both,
        "ref_cloud": ref_cloud, "other_cloud": other_cloud,
    }


def _plot(r_ref, r_oth, dr, s_c, t_c, cov_ref, cov_oth, dr_theta, both_theta,
          ref_cloud, other_cloud, save_path):
    extent = [s_c[0], s_c[-1], t_c[0], t_c[-1]]   # x=s, y=θ
    kw = dict(aspect="auto", origin="lower", extent=extent, interpolation="nearest")

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 2, width_ratios=[3, 1], hspace=0.32, wspace=0.22)

    # radius cap so a few large-radius entrance cells don't wash out the scale
    rmax = np.nanpercentile(np.concatenate([r_ref.ravel(), r_oth.ravel()]), 98)

    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(r_ref.T, vmin=0, vmax=rmax, cmap="viridis", **kw)
    ax0.set_title(f"A — r(s,θ) {ref_cloud} [m]"); ax0.set_ylabel("θ [deg] (0=ceiling)")
    fig.colorbar(im0, ax=ax0, pad=0.01)

    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0, sharey=ax0)
    im1 = ax1.imshow(r_oth.T, vmin=0, vmax=rmax, cmap="viridis", **kw)
    ax1.set_title(f"B — r(s,θ) {other_cloud} [m]  (white = not observed)")
    ax1.set_ylabel("θ [deg]")
    fig.colorbar(im1, ax=ax1, pad=0.01)

    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0, sharey=ax0)
    dlim = np.nanpercentile(np.abs(dr), 98) or 0.5
    im2 = ax2.imshow(dr.T, vmin=-dlim, vmax=dlim, cmap="RdBu_r", **kw)
    ax2.set_title(f"C — Δr = {other_cloud} − {ref_cloud} [m]  (both observed only)")
    ax2.set_xlabel("s [m]"); ax2.set_ylabel("θ [deg]")
    fig.colorbar(im2, ax=ax2, pad=0.01)

    # -- right column: azimuthal marginals (y=θ to line up with the heatmaps)
    axc = fig.add_subplot(gs[0:2, 1])
    axc.plot(cov_ref, t_c, color="steelblue", lw=1.4, label=ref_cloud)
    axc.plot(cov_oth, t_c, color="salmon", lw=1.4, label=other_cloud)
    axc.set_xlim(0, 100); axc.set_ylim(t_c[0], t_c[-1])
    axc.set_xlabel("slabs observed [%]"); axc.set_ylabel("θ [deg]")
    axc.set_title("coverage vs azimuth")
    axc.legend(loc="lower left", fontsize=8); axc.grid(alpha=0.3)

    axd = fig.add_subplot(gs[2, 1])
    axd.plot(dr_theta, t_c, color="k", lw=1.4)
    axd.axvline(0, color="gray", lw=0.6)
    axd.set_ylim(t_c[0], t_c[-1]); axd.set_xlabel("mean Δr [m]"); axd.set_ylabel("θ [deg]")
    axd.set_title("mean Δr vs azimuth"); axd.grid(alpha=0.3)

    fig.suptitle(f"Lateral (azimuthal) comparison — {other_cloud} vs {ref_cloud}",
                 fontsize=13)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Standalone: load the *_cyl.npz caches and produce the figure directly.       #
# --------------------------------------------------------------------------- #

def _main():
    import numpy as np
    from io_utils import (load_and_transform_trajectory, load_trajectory,
                          load_targets, load_registration, check_rigid_registration)
    from spine import fit_centreline, to_cylindrical

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = Config()
    cfg.targets_csv = "targets_leica.txt"

    # domain from the two end targets (needs the centreline)
    reg = load_registration(cfg.registration_path)
    check_rigid_registration(reg, tol=cfg.registration_rigid_tol)
    traj = load_and_transform_trajectory(cfg.trajectory_path, reg)
    targets = load_targets(cfg.targets_path)
    cl = fit_centreline(traj, cfg, orient_toward=targets[cfg.domain_end_target_idx])
    s_t, _, _ = to_cylindrical(
        np.stack([targets[cfg.domain_start_target_idx],
                  targets[cfg.domain_end_target_idx]]), cl)
    domain = (float(min(s_t)), float(max(s_t)))

    cyl = {}
    for name in ("leica", "livox"):
        d = np.load(cfg.cache_dir / f"{name}_cyl.npz")
        cyl[name] = {"s": d["s"], "r": d["r"], "theta": d["theta"]}
        log.info("loaded %s cyl cache: %d points", name, len(d["s"]))

    run_lateral_compare(cyl, domain, cfg, save_path=str(
        cfg.figures_dir / "08_lateral_compare.png"))
    print("wrote figures/08_lateral_compare.png")


if __name__ == "__main__":
    _main()
