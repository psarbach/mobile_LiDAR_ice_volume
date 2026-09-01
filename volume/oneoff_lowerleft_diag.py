"""
ONE-OFF diagnostic — the θ≈-120° (lower-left) Livox radius offset (day3 §14).

(a) Radius distribution at θ∈[-130,-110]° for both clouds: a spurious NEAR
    cluster (~body scale) only in Livox → self-returns; a uniform inward shift
    of the whole distribution → registration/tilt.
(b) Antisymmetry test: a lateral centreline OFFSET makes Δr(θ) ≈ A·cos(θ-θ0)
    — a dip at θ0 and an equal BUMP 180° away. If the -120° dip is matched by a
    +Δr bump near +60° and a cosine fits well, it is a global offset; if the dip
    is narrow and isolated, it is a real/local feature (self-returns or geometry).
"""

import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from io_utils import (load_and_transform_trajectory, load_targets,
                      load_registration, check_rigid_registration)
from spine import fit_centreline, to_cylindrical
from surface_mesh import build_r_grid

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("lowerleft")

BAND = (-130.0, -110.0)     # the lower-left offset band
OPP = (50.0, 70.0)          # the +60° azimuth where a lateral offset would compensate


def main():
    cfg = Config()
    cfg.targets_csv = "targets_leica.txt"

    reg = load_registration(cfg.registration_path)
    check_rigid_registration(reg, tol=cfg.registration_rigid_tol)
    traj = load_and_transform_trajectory(cfg.trajectory_path, reg)
    targets = load_targets(cfg.targets_path)
    cl = fit_centreline(traj, cfg, orient_toward=targets[cfg.domain_end_target_idx])
    s_t, _, _ = to_cylindrical(np.stack([targets[cfg.domain_start_target_idx],
                                         targets[cfg.domain_end_target_idx]]), cl)
    domain = (float(min(s_t)), float(max(s_t)))

    cyl = {}
    for name in ("leica", "livox"):
        d = np.load(cfg.cache_dir / f"{name}_cyl.npz")
        s, r, th = d["s"], d["r"], d["theta"]
        m = (s >= domain[0]) & (s <= domain[1])
        cyl[name] = {"s": s[m], "r": r[m], "theta": th[m]}

    # ---- (a) radius distribution in the band (raw points, per cloud) ----
    print("\n(a) RAW radius distribution at θ ∈ [%.0f, %.0f]°" % BAND)
    print("    cloud   n_pts   median_r   p10    p50    p90   frac(r<0.8m)")
    band_r = {}
    for name in ("leica", "livox"):
        th = cyl[name]["theta"]; r = cyl[name]["r"]
        inb = (th >= BAND[0]) & (th <= BAND[1])
        rb = r[inb]
        band_r[name] = rb
        p10, p50, p90 = np.percentile(rb, [10, 50, 90])
        near = float(np.mean(rb < 0.8))
        print(f"    {name:6s} {len(rb):8d}  {np.median(rb):7.3f}  "
              f"{p10:5.2f} {p50:5.2f} {p90:5.2f}   {near*100:5.1f}%")

    # ---- (b) Δr(θ) shape: dip at -120 vs bump at +60, cosine fit ----
    grids = {}
    for name in ("leica", "livox"):
        g, s_c, t_c = build_r_grid(cyl[name]["s"], cyl[name]["r"],
                                   cyl[name]["theta"], domain, cfg)
        grids[name] = g
    both = ~np.isnan(grids["leica"]) & ~np.isnan(grids["livox"])
    dr = np.where(both, grids["livox"] - grids["leica"], np.nan)
    with np.errstate(invalid="ignore"):
        dr_theta = np.nanmean(dr, axis=0)
    cov = np.mean(both, axis=0)

    def at(lo, hi):
        sel = (t_c >= lo) & (t_c <= hi)
        return float(np.nanmean(dr_theta[sel]))

    print("\n(b) mean Δr(θ) = livox − leica")
    print(f"    at θ∈[-130,-110] (dip):   {at(*BAND):+.3f} m")
    print(f"    at θ∈[+50,+70]  (opp):    {at(*OPP):+.3f} m")

    # cosine fit dr_theta ≈ A cos(θ-θ0) + c on well-covered azimuths
    ok = cov > 0.3
    th_r = np.radians(t_c[ok]); y = dr_theta[ok]
    A = np.column_stack([np.cos(th_r), np.sin(th_r), np.ones_like(th_r)])
    (a1, b1, c1), *_ = np.linalg.lstsq(A, y, rcond=None)
    amp = float(np.hypot(a1, b1)); phi = float(np.degrees(np.arctan2(-b1, a1)))
    resid = y - A @ np.array([a1, b1, c1])
    ss = 1 - np.var(resid) / np.var(y)
    print(f"    cosine fit: amplitude {amp:.3f} m at θ0={phi:+.0f}°, "
          f"offset {c1:+.3f} m, R²={ss:.2f}")
    print("    (a lateral centreline offset => R²≈1, |dip|≈|bump|; "
          "a local feature => low R², dip>>bump)")

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    # per-cloud median r vs θ
    for name, col in (("leica", "steelblue"), ("livox", "salmon")):
        med = np.nanmedian(grids[name], axis=0)
        ax[0].plot(med, t_c, color=col, lw=1.4, label=name)
    ax[0].axhspan(*BAND, color="k", alpha=0.08)
    ax[0].set_xlabel("median r [m]"); ax[0].set_ylabel("θ [deg]")
    ax[0].set_title("median wall radius vs azimuth"); ax[0].legend(); ax[0].grid(alpha=.3)

    ax[1].plot(dr_theta, t_c, color="k", lw=1.4, label="Δr data")
    fit_full = a1*np.cos(np.radians(t_c)) + b1*np.sin(np.radians(t_c)) + c1
    ax[1].plot(fit_full, t_c, color="crimson", lw=1.0, ls="--", label="cosine fit")
    ax[1].axhspan(*BAND, color="k", alpha=0.08); ax[1].axhspan(*OPP, color="green", alpha=0.08)
    ax[1].axvline(0, color="gray", lw=.6)
    ax[1].set_xlabel("mean Δr [m]"); ax[1].set_title("Δr(θ): dip(grey) vs opp(green)")
    ax[1].legend(); ax[1].grid(alpha=.3)

    bins = np.linspace(0, 3, 80)
    for name, col in (("leica", "steelblue"), ("livox", "salmon")):
        ax[2].hist(band_r[name], bins=bins, density=True, alpha=0.55,
                   color=col, label=name)
    ax[2].set_xlabel("r [m]"); ax[2].set_ylabel("density")
    ax[2].set_title(f"radius dist. at θ∈[{BAND[0]:.0f},{BAND[1]:.0f}]°")
    ax[2].legend(); ax[2].grid(alpha=.3)

    fig.tight_layout()
    out = str(cfg.figures_dir / "09_lowerleft_diag.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
