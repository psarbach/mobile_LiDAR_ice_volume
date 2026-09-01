"""
Per-segment volume comparison, Leica vs Livox.

Answers: does the Leica–Livox volume difference grow *gradually* along the
tunnel (a uniform scale/coverage effect — cumulative ΔV(s) is a straight line),
or is it concentrated in specific stretches (kinks/steps)?

Source of per-slice volume: the profiles method's cross-section area A(s), one
value per Δs slab, on the SAME s-grid for both clouds (same centreline, same
domain, same Δs). A(s)·Δs integrated over a segment is that segment's volume —
this is the profiles volume sliced up, and profiles ≈ surface mesh to <1% (they
share the r(s, θ) extraction and differ only in how they integrate), so it is
also the mesh volume for this purpose, without having to clip the mesh.

Missing slabs (A_s = NaN, i.e. no data even after the profiles hole-fill) are
linearly interpolated along s here so the cumulative curve is continuous; this
matches how the profiles integral already bridges them.
"""

import logging
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import Config

log = logging.getLogger(__name__)


def _clean_area(prof) -> tuple:
    """(s_centers, A_s) with any NaN areas linearly interpolated along s."""
    s = np.asarray(prof.s_centers, dtype=float)
    A = np.asarray(prof.A_s, dtype=float).copy()
    nan = np.isnan(A)
    if nan.all():
        raise ValueError(f"Profiles for '{prof.cloud_name}' are entirely empty")
    if nan.any():
        A[nan] = np.interp(s[nan], s[~nan], A[~nan])
    return s, A


def _cumulative_volume(s: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Cumulative ∫A ds along s (trapezoid), same length as s, starting at 0."""
    seg = 0.5 * (A[1:] + A[:-1]) * np.diff(s)
    return np.concatenate([[0.0], np.cumsum(seg)])


def compare_volume_slices(
    prof_by_cloud: Dict[str, object],
    cfg: Config,
    ref_cloud: str = "leica",
    other_cloud: str = "livox",
    save_path: str | None = None,
) -> dict:
    """
    Compare per-segment volume of two clouds along the tunnel.

    Parameters
    ----------
    prof_by_cloud : {cloud_name: ProfileResult} — must contain ref_cloud and
                    other_cloud, both from the same domain/Δs.
    ref_cloud     : the reference the difference is taken against (Leica).

    Returns a dict with per-segment edges, volumes, ΔV, ΔV%, and totals.
    """
    if ref_cloud not in prof_by_cloud or other_cloud not in prof_by_cloud:
        raise ValueError(
            f"Need both '{ref_cloud}' and '{other_cloud}' in prof_by_cloud; "
            f"got {list(prof_by_cloud)}"
        )

    s_ref, A_ref = _clean_area(prof_by_cloud[ref_cloud])
    s_oth, A_oth = _clean_area(prof_by_cloud[other_cloud])

    # The two share the s-grid (same domain/Δs); guard, and put the other cloud
    # onto the reference grid if a rounding difference ever creeps in.
    if s_ref.shape != s_oth.shape or not np.allclose(s_ref, s_oth):
        A_oth = np.interp(s_ref, s_oth, A_oth)
    s = s_ref

    cum_ref = _cumulative_volume(s, A_ref)
    cum_oth = _cumulative_volume(s, A_oth)

    # Segment edges every slice_segment_m across the domain
    seg_len = cfg.slice_segment_m
    edges = np.arange(s[0], s[-1] + seg_len, seg_len)
    if edges[-1] < s[-1]:
        edges = np.append(edges, s[-1])
    edges = np.clip(edges, s[0], s[-1])

    cum_ref_e = np.interp(edges, s, cum_ref)
    cum_oth_e = np.interp(edges, s, cum_oth)
    V_ref_seg = np.diff(cum_ref_e)
    V_oth_seg = np.diff(cum_oth_e)
    dV_seg = V_oth_seg - V_ref_seg
    with np.errstate(divide="ignore", invalid="ignore"):
        dV_pct = np.where(V_ref_seg != 0, dV_seg / V_ref_seg * 100.0, np.nan)
    seg_mid = 0.5 * (edges[:-1] + edges[1:])

    V_ref_tot = float(cum_ref[-1])
    V_oth_tot = float(cum_oth[-1])
    dV_tot = V_oth_tot - V_ref_tot

    log.info(
        "Slice comparison (%s vs %s), %g m segments: total V_%s=%.1f m³, "
        "V_%s=%.1f m³, ΔV=%.1f m³ (%.2f%%)",
        other_cloud, ref_cloud, seg_len, ref_cloud, V_ref_tot, other_cloud,
        V_oth_tot, dV_tot, dV_tot / V_ref_tot * 100.0,
    )
    worst = int(np.nanargmax(np.abs(dV_seg)))
    log.info(
        "Slice comparison: largest single-segment difference at s≈%.0f–%.0f m: "
        "ΔV=%.1f m³ (%.1f%%). |ΔV| segment mean=%.1f m³.",
        edges[worst], edges[worst + 1], dV_seg[worst], dV_pct[worst],
        float(np.mean(np.abs(dV_seg))),
    )

    if save_path:
        _plot(s, A_ref, A_oth, cum_ref, cum_oth, edges, seg_mid, dV_seg, dV_pct,
              prof_by_cloud, ref_cloud, other_cloud, seg_len, save_path)

    return {
        "s": s, "edges": edges, "seg_mid": seg_mid,
        "V_ref_seg": V_ref_seg, "V_oth_seg": V_oth_seg,
        "dV_seg": dV_seg, "dV_pct": dV_pct,
        "V_ref_total": V_ref_tot, "V_oth_total": V_oth_tot, "dV_total": dV_tot,
        "ref_cloud": ref_cloud, "other_cloud": other_cloud,
    }


def _plot(s, A_ref, A_oth, cum_ref, cum_oth, edges, seg_mid, dV_seg, dV_pct,
          prof_by_cloud, ref_cloud, other_cloud, seg_len, save_path):
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

    # -- Panel A: cross-section area A(s), both clouds, + Livox interp fraction
    ax = axes[0]
    ax.plot(s, A_ref, color="steelblue", lw=1.3, label=f"{ref_cloud}")
    ax.plot(s, A_oth, color="salmon", lw=1.3, label=f"{other_cloud}")
    ax.set_ylabel("cross-section area A(s) [m²]")
    ax.legend(loc="upper right")
    ax.set_title("A — cross-section area along the tunnel")
    ax.grid(alpha=0.3)
    fr = np.asarray(prof_by_cloud[other_cloud].frac_interp, dtype=float)
    axf = ax.twinx()
    axf.fill_between(prof_by_cloud[other_cloud].s_centers, fr * 100, 0,
                     color="salmon", alpha=0.15, step="mid")
    axf.set_ylabel(f"{other_cloud} interpolated [%]", color="salmon")
    axf.set_ylim(0, 100); axf.tick_params(axis="y", colors="salmon")

    # -- Panel B: cumulative volume + cumulative ΔV  (THE gradual-vs-localised plot)
    ax = axes[1]
    ax.plot(s, cum_ref, color="steelblue", lw=1.3, label=f"cum V {ref_cloud}")
    ax.plot(s, cum_oth, color="salmon", lw=1.3, label=f"cum V {other_cloud}")
    ax.set_ylabel("cumulative volume [m³]")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    axd = ax.twinx()
    axd.plot(s, cum_oth - cum_ref, color="k", lw=1.6,
             label=f"cumulative ΔV ({other_cloud}−{ref_cloud})")
    axd.axhline(0, color="gray", lw=0.6)
    axd.set_ylabel(f"cumulative ΔV [m³]")
    axd.legend(loc="lower left")
    ax.set_title("B — cumulative volume; black ΔV straight = gradual, kinked = localised")

    # -- Panel C: per-segment ΔV bars
    ax = axes[2]
    colors = ["salmon" if d > 0 else "steelblue" for d in dV_seg]
    ax.bar(seg_mid, dV_seg, width=seg_len * 0.9, color=colors, edgecolor="k", lw=0.4)
    ax.axhline(0, color="gray", lw=0.6)
    for x, d, p in zip(seg_mid, dV_seg, dV_pct):
        ax.annotate(f"{p:+.0f}%", (x, d), ha="center",
                    va="bottom" if d >= 0 else "top", fontsize=7)
    ax.set_xlabel("s [m]")
    ax.set_ylabel(f"ΔV per {seg_len:g} m  ({other_cloud}−{ref_cloud}) [m³]")
    ax.set_title(f"C — per-segment volume difference (label = % of {ref_cloud})")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"Per-segment volume comparison — {other_cloud} vs {ref_cloud}",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
