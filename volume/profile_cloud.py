"""
Export the wall the profiles method actually integrated, as a point cloud.

The profiles estimator (`cross_sections.run_profiles`) does not integrate the
raw scan. It integrates a filled r(s, θ) grid: one median radius per 0.25 m ×
1° cell, with the empty cells interpolated — circularly around θ first, then
along s for slabs too sparse to close on their own. That grid is the thing the
volume is a property of, and until now it existed only inside the integral.

This module wraps it back to 3-D through the same centreline frame that
produced the coordinates (`spine.from_cylindrical`, the inverse of the
`to_cylindrical` that made them) and writes it as a binary PLY with the
per-point flag that matters:

    interpolated = 1   the wall here was invented by the hole fill
    interpolated = 0   the wall here is the median of real returns

so that in CloudCompare you can colour by that field and see exactly which
metres of tunnel the volume is trusting the interpolation for. On the April
Livox walks that is about a third of the cross-sectional area; on July, 3%.

Nothing here feeds back into any volume: `run_profiles` computes its result
first and this reads what it kept.

Size: one point per grid cell, so ~540 slabs × 360 bins ≈ 195k points ≈ 4 MB —
three orders below the 20-40 M point input clouds, and it opens instantly.
"""

import logging
from pathlib import Path

import numpy as np

from io_utils import save_ply_scalars
from spine import from_cylindrical

log = logging.getLogger(__name__)


def build_profile_cloud(prof, cl):
    """(xyz, fields) for the filled r(s, θ) grid of one ProfileResult.

    Cells the fill could not close (NaN — the slabs `run_profiles` drops from
    the integral) are omitted rather than placed at some invented radius: a
    point there would claim a wall position the method never had.
    """
    if prof.r_grid is None or prof.theta_centers is None:
        raise ValueError(
            "ProfileResult carries no r_grid — it came from a degenerate run "
            "(no slabs in the domain, or every slab empty)."
        )

    n_s, n_t = prof.r_grid.shape
    s_col = np.repeat(prof.s_centers[:n_s], n_t)
    t_col = np.tile(prof.theta_centers, n_s)
    r_col = prof.r_grid.reshape(-1)
    interp_col = (prof.was_interp.reshape(-1) if prof.was_interp is not None
                  else np.zeros(r_col.shape, dtype=bool))

    keep = np.isfinite(r_col)
    s_col, t_col, r_col, interp_col = (s_col[keep], t_col[keep], r_col[keep],
                                       interp_col[keep])

    xyz = from_cylindrical(s_col, r_col, t_col, cl)
    fields = {
        "interpolated": interp_col.astype(np.uint8),
        "r": r_col.astype(np.float32),
        "s": s_col.astype(np.float32),
        "theta_deg": t_col.astype(np.float32),
    }
    return xyz, fields


def export_profile_cloud(prof, cl, path: Path | str, domain=None) -> dict:
    """Write the profiles method's interpolated wall to `path` (binary PLY).

    Returns a small dict for summary.json — how many points, and how many of
    them are interpolated, which is the honesty number for that run.
    """
    xyz, fields = build_profile_cloud(prof, cl)
    n_interp = int(fields["interpolated"].sum())
    n = len(xyz)

    comments = [f"profiles-method wall, cloud={prof.cloud_name}",
                "scalar 'interpolated': 1 = hole-filled, 0 = measured median"]
    if domain is not None:
        comments.append(f"domain s=[{domain[0]:.3f}, {domain[1]:.3f}] m")

    save_ply_scalars(path, xyz, fields, comments=tuple(comments))
    log.info(
        "Profile cloud (%s): exported %d points (%.1f%% interpolated) → %s",
        prof.cloud_name, n, n_interp / n * 100 if n else 0.0, path,
    )
    return {"path": Path(path).name, "n_points": n,
            "n_interpolated": n_interp,
            "frac_interpolated": (n_interp / n) if n else None}
