"""
Volume decomposition: V = Ā × L.

The Leica and Livox scans disagree on tunnel length (~0.15% over ~130 m —
the longitudinal SLAM scale error). That must be REPORTED, not hidden by
comparing a single V number. Decomposing V = Ā × L separates:

  - L ratio  : longitudinal SCALE error (target-to-target distance per cloud)
  - Ā ratio  : SHAPE/radial error, measured over the SAME domain (the target
               end-cap planes) for both clouds, so length is held fixed and
               any Ā difference is pure shape
  - V ratio  : total = Ā ratio × L ratio, now attributable to shape vs length

Two independent length measurements are needed to see a real scale error.
If Config.targets_livox_csv is unset, both clouds fall back to the same
targets.csv and L ratio is trivially 1.0 — expected until per-cloud target
picks exist, not a bug.
"""

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class LengthResult:
    L_m: float                          # target[end] - target[start] distance
    cumulative_from_first_m: np.ndarray  # (n_targets,) distance of target i from target 0
    segment_m: np.ndarray = None         # (n_targets-1,) target i -> i+1 distance
    measure: str = "straight-line 3-D"   # how those distances were measured
    chainage_m: np.ndarray = None        # (n_targets,) arclength s, when measured on a centreline


@dataclass
class DecompositionResult:
    cloud_a: str
    cloud_b: str
    L: Dict[str, LengthResult]
    area_bar_m2: Dict[str, float]       # Ā over the shared domain, per cloud
    L_ratio: float                      # L[b] / L[a]
    area_ratio: float                   # Ā[b] / Ā[a]
    V_ratio: float                      # area_ratio * L_ratio
    same_targets_for_both: bool          # True => L_ratio is trivially 1.0


def compute_length(
    targets: np.ndarray, row_start: int, row_end: int, cl=None
) -> LengthResult:
    """
    Length between the two domain end-cap targets, plus cumulative and
    per-segment series.

    cl : optional Centreline (spine.fit_centreline). **Pass it.** Each target is
    then projected onto the fitted centreline and every distance is measured as
    ARCLENGTH ALONG THAT CURVE, so the numbers follow the tunnel's bends. This is
    the same ruler the volume domain uses, so the length quoted here and the L
    the volume integrates over are by construction the same quantity.

    Without `cl` the distances are straight-line 3-D point-to-point. That is a
    lower bound on the path length and it degrades with curvature: a meandering
    conduit's chord can be far shorter than the conduit. On this near-straight
    tunnel the two differ by only ~48 mm in 135 m, but the chord version must not
    be carried forward to a curved conduit, which is why the centreline is the
    default path here.

    Note what does NOT need the centreline: a *scale* factor. Stretching
    multiplies chords and arcs by the same k, so `scale_from_targets` below
    (pairwise 3-D distances) measures stretch correctly at any curvature. The
    curve matters for "how long is the tunnel", not for "is one cloud stretched".
    """
    if cl is not None:
        from spine import to_cylindrical
        s, _, _ = to_cylindrical(np.asarray(targets, dtype=float), cl)
        s = np.asarray(s, dtype=float)
        return LengthResult(
            L_m=float(abs(s[row_end] - s[row_start])),
            cumulative_from_first_m=s - s[0],
            segment_m=np.diff(s),
            measure="arclength along centreline",
            chainage_m=s,
        )

    L_m = float(np.linalg.norm(targets[row_end] - targets[row_start]))
    cumulative = np.linalg.norm(targets - targets[0], axis=1)
    segment = np.linalg.norm(np.diff(targets, axis=0), axis=1)
    return LengthResult(L_m=L_m, cumulative_from_first_m=cumulative,
                        segment_m=segment, measure="straight-line 3-D")


def scale_from_targets(t_ref: np.ndarray, t_oth: np.ndarray) -> dict:
    """
    Longitudinal scale of one cloud against the other, from the target network.

    Fits `d_oth = k · d_ref` (through-origin least squares) over ALL target
    pairs, where d is the plain 3-D distance between two targets. k−1 is the
    SLAM/registration scale error (drift) and `sigma_k` its standard error.

    Pairwise distances are used deliberately: they need no origin, no axis and
    no ordering, so the measurement holds for a meandering conduit exactly as it
    does for a straight one. Never project target offsets onto a single global
    axis — that silently assumes the targets are collinear.
    """
    dl, dv = [], []
    for i, j in combinations(range(len(t_ref)), 2):
        dl.append(np.linalg.norm(t_ref[i] - t_ref[j]))
        dv.append(np.linalg.norm(t_oth[i] - t_oth[j]))
    dl, dv = np.array(dl), np.array(dv)
    k = float(np.sum(dl * dv) / np.sum(dl * dl))       # through-origin LS
    resid = dv - k * dl
    # stderr of the slope for through-origin regression
    dof = max(len(dl) - 1, 1)
    sigma_k = float(np.sqrt(np.sum(resid ** 2) / dof / np.sum(dl ** 2)))
    return {"k": k, "sigma_k": sigma_k,
            "resid_rms": float(np.sqrt(np.mean(resid ** 2))),
            "L_ref": float(np.linalg.norm(t_ref[0] - t_ref[-1])),
            "L_oth": float(np.linalg.norm(t_oth[0] - t_oth[-1]))}


def target_consistency(t_ref: np.ndarray, t_oth: np.ndarray) -> dict:
    """
    Which targets are mutually consistent between the two clouds?

    Rotation- and translation-invariant: works only on the pairwise distances
    between targets, so it assumes nothing about where the targets sit or whether
    they line up. For each target, drop it and refit the single scale k over the
    remaining pairs; a target whose removal collapses the residual spread is
    inconsistent with the rest.

    Returns per-target leave-one-out residual RMS, the per-target mean pairwise
    residual, and the indices flagged as outliers (removal improves the fit by
    more than `improve_factor`).
    """
    n = len(t_ref)
    base = scale_from_targets(t_ref, t_oth)
    loo = np.empty(n)
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        loo[i] = scale_from_targets(t_ref[keep], t_oth[keep])["resid_rms"]

    improve = base["resid_rms"] / loo
    flagged = [int(i) for i in np.where(improve > 2.0)[0]]

    # Each target's mean residual is measured against the CLEAN partners only,
    # under the scale fitted on those same partners. Averaging a target's pairs
    # with a known-bad target would smear that one target's error across all the
    # others and make every target look slightly off.
    clean = [i for i in range(n) if i not in flagged] or list(range(n))
    k = (scale_from_targets(t_ref[clean], t_oth[clean])["k"] if flagged
         else base["k"])
    mean_resid = np.zeros(n)
    for i in range(n):
        r = [np.linalg.norm(t_oth[i] - t_oth[j]) - k * np.linalg.norm(t_ref[i] - t_ref[j])
             for j in clean if j != i]
        mean_resid[i] = float(np.mean(r)) if r else 0.0
    res = {"base_resid_rms": base["resid_rms"], "loo_resid_rms": loo,
           "improve_factor": improve, "mean_pair_resid_m": mean_resid,
           "flagged": flagged, "k_base": base["k"], "k_clean": k}
    log.info("Target consistency (pairwise distances, rotation-invariant): "
             "base residual RMS %.1f mm", base["resid_rms"] * 1000)
    for i in range(n):
        log.info("  target %d: leave-one-out RMS %6.1f mm (x%.2f better), "
                 "mean pair residual %+7.1f mm%s", i, loo[i] * 1000, improve[i],
                 mean_resid[i] * 1000, "   <-- OUTLIER" if i in flagged else "")
    if flagged:
        res["k_excl_flagged"] = k
        log.warning(
            "Targets %s are inconsistent with the rest. Scale WITH them: "
            "k=%.5f (%+.3f%%, residual RMS %.1f mm); WITHOUT: k=%.5f (%+.3f%%). "
            "Quote the latter.",
            flagged, base["k"], (base["k"] - 1) * 100, base["resid_rms"] * 1000,
            k, (k - 1) * 100)
    return res


def compute_decomposition(
    cfg,
    targets_by_cloud: Dict[str, np.ndarray],
    mean_area_m2_by_cloud: Dict[str, float],
    cl=None,
) -> DecompositionResult:
    """
    Parameters
    ----------
    targets_by_cloud     : cloud_name -> (N, 3) target coordinates used to
                            measure THAT cloud's own length. If the same
                            array object/values are passed for both clouds,
                            L_ratio is trivially 1.0.
    mean_area_m2_by_cloud : cloud_name -> Ā (mean cross-sectional area) measured
                            over the SAME domain (target end-cap planes) for
                            both clouds — the shared domain is what makes this
                            a pure shape comparison.
    """
    names = list(targets_by_cloud)
    if len(names) != 2:
        raise ValueError("Decomposition needs exactly two clouds")
    a, b = names

    L = {
        name: compute_length(
            targets_by_cloud[name],
            cfg.domain_start_target_idx,
            cfg.domain_end_target_idx,
            cl=cl,
        )
        for name in names
    }
    same_targets = bool(np.allclose(targets_by_cloud[a], targets_by_cloud[b]))

    L_ratio = L[b].L_m / L[a].L_m
    area_ratio = mean_area_m2_by_cloud[b] / mean_area_m2_by_cloud[a]
    V_ratio = area_ratio * L_ratio

    log.info("=== Volume decomposition: V = Ā x L  (cap_mode=%s) ===", cfg.cap_mode)
    for name in names:
        log.info(
            "  %-6s  L=%.4f m   Ā(common domain)=%.4f m²",
            name, L[name].L_m, mean_area_m2_by_cloud[name],
        )
    log.info("  L ratio (%s/%s) = %.5f  <- scale error", b, a, L_ratio)
    log.info("  Ā ratio (%s/%s) = %.5f  <- shape error", b, a, area_ratio)
    log.info("  V ratio (%s/%s) = %.5f  = Ā_ratio x L_ratio", b, a, V_ratio)

    if same_targets:
        log.warning(
            "Both clouds used the SAME target coordinates -> L ratio is "
            "trivially 1.0. A genuine per-cloud scale-error measurement "
            "needs independently picked target coordinates in each cloud "
            "(set Config.targets_livox_csv once you have them)."
        )

    return DecompositionResult(
        cloud_a=a, cloud_b=b, L=L, area_bar_m2=mean_area_m2_by_cloud,
        L_ratio=L_ratio, area_ratio=area_ratio, V_ratio=V_ratio,
        same_targets_for_both=same_targets,
    )


def plot_target_3d_distances(
    targets_by_cloud: Dict[str, np.ndarray],
    save_path: str | None = None,
    ref_cloud: str = "leica",
    sigma_delta_m: float | None = None,
    flagged_targets: list | None = None,
) -> None:
    """
    Straight-line 3-D distance between each pair of NEIGHBOURING targets, one bar
    per cloud, side by side.

    The companion to figure 06, and deliberately a different quantity. Figure 06
    asks "how long is the tunnel", which has to follow the conduit's bends, so it
    measures arclength along the centreline. This asks "how far apart are these
    two targets", for which the straight line between them is exactly right and
    the shape of the tunnel in between is irrelevant. It also needs no centreline,
    no origin and no ordering assumption, so it stays valid however the conduit
    meanders.

    Bars start at zero and the two clouds agree to <1%%, so the pair of bars will
    look identical by eye — that is the honest picture. The number that matters is
    printed above each pair: Δ = other − reference in mm, coloured by whether it
    exceeds the picking-repeatability band.
    """
    names = list(targets_by_cloud)
    if ref_cloud not in names:
        ref_cloud = names[0]
    order = [ref_cloud] + [n for n in names if n != ref_cloud]
    flagged = list(flagged_targets or [])

    T0 = targets_by_cloud[order[0]]
    n_seg = len(T0) - 1
    x = np.arange(n_seg)
    width = 0.8 / len(order)
    cols = {"leica": "steelblue", "livox": "salmon"}

    seg = {name: np.linalg.norm(np.diff(targets_by_cloud[name], axis=0), axis=1)
           for name in order}

    fig, ax = plt.subplots(figsize=(11, 6))
    for k, name in enumerate(order):
        pos = x + (k - (len(order) - 1) / 2) * width
        bars = ax.bar(pos, seg[name], width=width, color=cols.get(name),
                      edgecolor="k", lw=.5, label=name, zorder=3)
        for b, v in zip(bars, seg[name]):
            ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=7.5, rotation=90,
                        xytext=(0, 3), textcoords="offset points")

    # Δ above each group — the only place the difference is legible.
    other = order[1] if len(order) > 1 else order[0]
    d_mm = (seg[other] - seg[order[0]]) * 1000
    top = max(max(seg[n]) for n in order)
    for i, d in enumerate(d_mm):
        big = sigma_delta_m is not None and abs(d) > sigma_delta_m * 1000
        ax.annotate(f"Δ {d:+.0f} mm", (x[i], top * 1.10), ha="center",
                    fontsize=9, fontweight="bold" if big else "normal",
                    color="firebrick" if big else "dimgrey")

    labels = [f"{i+1}–{i+2}" for i in range(n_seg)]
    for i in range(n_seg):
        if i in flagged or i + 1 in flagged:
            labels[i] += "\n(target flagged)"
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("target pair")
    ax.set_ylabel("3-D distance between the two targets [m]")
    band = (f"   ·   Δ in bold red exceeds the ±{sigma_delta_m*1000:.0f} mm "
            "picking-repeatability band" if sigma_delta_m else "")
    ax.set_title(f"Straight-line 3-D distance between neighbouring targets — "
                 f"{' vs '.join(order)}\nΔ = {other} − {order[0]}{band}")
    ax.set_ylim(0, top * 1.20)
    ax.legend(loc="lower right")
    ax.grid(alpha=.3, axis="y", zorder=0)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_cumulative_distance(
    L_by_cloud: Dict[str, LengthResult],
    save_path: str | None = None,
    ref_cloud: str = "leica",
    sigma_delta_m: float | None = None,
    sigma_label: str = "pick accuracy (1σ, 4 picks)",
    flagged_targets: list | None = None,
) -> None:
    """
    Target-to-target distance comparison — the length/scale-error diagnostic.

    Panel A alone (the original figure) is useless on these datasets: the two
    clouds agree on total length to <0.05%, so the curves plot on top of each
    other and the eye can resolve nothing. The signal lives in the DIFFERENCE,
    which is 4 orders of magnitude smaller than the distance itself and needs
    its own axis:

      A  absolute cumulative distance per cloud — the context (how long, how
         the targets are spaced). Deliberately kept, so the figure still stands
         alone in a report.
      B  Δ = other − reference, in mm (left) AND as % of distance (right, its
         own series — the two are not a fixed rescale of each other since %
         divides by a distance that grows). A uniform scale error is a straight
         line through the origin in mm and a FLAT line in %; the dashed line is
         the best-fit uniform scale, so departures from it are what is not a
         scale error.
      C  the same difference per SEGMENT (target i → i+1) — the derivative of
         B. A single localized drift or a mis-picked target shows up as one
         outlier bar here, where in B it is smeared into every later point.

    All distances come from the LengthResult, so whatever ruler `compute_length`
    used (arclength along the centreline, or straight-line 3-D) is what is drawn;
    the axis labels say which. Nothing here assumes the targets are aligned — the
    per-segment panel is a distance between two neighbouring targets, which is
    the same number however the tunnel bends.

    sigma_delta_m: 1σ band on the plotted difference, propagated from the
    picking repeatability (4 picks → 2·σ_pick). Differences inside it are
    picking noise, not geometry.
    flagged_targets: indices found inconsistent with the rest (target_consistency).
    They are marked on the figure rather than dropped: an unexplained spike is
    what makes a plot confusing, and silently deleting a real measurement is
    worse than labelling it.
    """
    flagged = list(flagged_targets or [])
    names = list(L_by_cloud)
    if ref_cloud not in names:
        ref_cloud = names[0]
    others = [n for n in names if n != ref_cloud]

    ref = L_by_cloud[ref_cloud]
    cum_ref = ref.cumulative_from_first_m
    n = len(cum_ref)
    idx = np.arange(n)

    fig, ax = plt.subplots(3, 1, figsize=(9, 11), sharex=True,
                           gridspec_kw={"height_ratios": [1.1, 1.0, 0.9]})
    cols = {"leica": "steelblue", "livox": "salmon"}

    # ---------------------------------------------------------------- A
    # Distinct linestyles, not just colours: the two curves differ by <0.05% and
    # WILL overplot at this scale — the reader must still see there are two.
    styles = [dict(ls="-", lw=2.4, marker="o", ms=7, alpha=.9),
              dict(ls="--", lw=1.4, marker="x", ms=9, mew=1.8)]
    for (name, res), st in zip(L_by_cloud.items(), styles):
        ax[0].plot(idx, res.cumulative_from_first_m, color=cols.get(name),
                   label=f"{name}  (L = {res.L_m:.3f} m)", **st)
    measure = ref.measure
    ax[0].set_ylabel(f"Distance from target 0\n[m]  ({measure})")
    ax[0].set_title(f"Target-to-target distance, measured as {measure}\n"
                    "absolute (A), difference (B), per-segment difference (C)")
    ax[0].legend(loc="upper left")
    ax[0].grid(alpha=.3)
    # Second x-axis: where each target sits along the tunnel, in metres.
    top = ax[0].secondary_xaxis("top")
    top.set_xticks(idx)
    top.set_xticklabels([f"{c:.1f}" for c in cum_ref], fontsize=8)
    top.set_xlabel(f"chainage of each target in {ref_cloud} [m]", fontsize=9)
    # Every cumulative curve is measured FROM target 0, so if target 0 is itself
    # inconsistent, the whole curve inherits its error. Say so on the figure.
    if 0 in flagged:
        ax[0].text(0.985, 0.06,
                   "⚠ target 0 is flagged inconsistent, and it is the anchor of\n"
                   "   panels B and C — the whole Δ curve carries its error.\n"
                   "   Panel C localises it; the fit in B excludes it.",
                   transform=ax[0].transAxes, fontsize=8, va="bottom", ha="right",
                   bbox=dict(boxstyle="round,pad=0.35", fc="mistyrose",
                             ec="firebrick", alpha=.9))

    # ---------------------------------------------------------------- B
    axr = ax[1].twinx()
    for name in others:
        d_mm = (L_by_cloud[name].cumulative_from_first_m - cum_ref) * 1000
        ax[1].plot(idx, d_mm, "o-", ms=6, color=cols.get(name, "salmon"),
                   label=f"Δ {name} − {ref_cloud}  [mm, left]")
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(cum_ref > 0, d_mm / 1000 / np.where(cum_ref > 0, cum_ref, 1) * 100, np.nan)
        axr.plot(idx, rel, "s-", ms=5, lw=1.4, color="#7b3294", alpha=.9,
                 label="same, relative [%, right axis]")
        # Uniform-scale model, fitted WITH AN INTERCEPT on the unflagged targets.
        # The intercept matters: the plotted Δ is measured from target 0, so if
        # target 0 is itself displaced the whole curve is shifted by that amount.
        # A through-origin line would then float away from every data point and
        # look like a bad model when the model is fine and the anchor is not.
        # slope → the scale error; intercept → how far the anchor target is off.
        fit_idx = [i for i in idx if i not in flagged]
        if len(fit_idx) < 2:
            fit_idx = list(idx)
        slope, intercept = np.polyfit(cum_ref[fit_idx], d_mm[fit_idx], 1)
        k = 1.0 + slope / 1000.0            # mm per m → dimensionless
        resid = d_mm[fit_idx] - (slope * cum_ref[fit_idx] + intercept)
        fit_note = ("all targets" if len(fit_idx) == len(idx)
                    else f"excl. {flagged}")
        ax[1].plot(idx, slope * cum_ref + intercept, ":", lw=1.5, color="k",
                   alpha=.85,
                   label=f"uniform-scale fit ({fit_note}):  k−1 = {(k-1)*100:+.3f}%"
                         f",  RMS {np.sqrt((resid**2).mean()):.0f} mm")
        if flagged:
            off = d_mm[flagged] - (slope * cum_ref[flagged] + intercept)
            ax[1].plot(np.array(idx)[flagged], d_mm[flagged], "o", ms=13,
                       mfc="none", mec="firebrick", mew=2.0,
                       label=f"flagged target{'s' if len(flagged) > 1 else ''} "
                             f"{flagged}: {', '.join(f'{o:+.0f}' for o in off)} mm "
                             "off the fit")
        # Cross-check: a genuine uniform scale gives the same k two ways — from
        # the fit and from the two end targets. Disagreement means
        # the difference is not proportional to distance, so neither number may
        # be quoted as "the scale error" on its own.
        k_end = L_by_cloud[name].L_m / ref.L_m
        agree = abs((k - 1) - (k_end - 1)) < 5e-4
        ends_flagged = [i for i in flagged
                        if i in (0, len(cum_ref) - 1)]
        if agree:
            verdict = "consistent → read as a uniform scale error"
        elif ends_flagged:
            verdict = (f"they disagree, but the end-target value is built on "
                       f"flagged target{'s' if len(ends_flagged) > 1 else ''} "
                       f"{ends_flagged} — quote the fit")
        else:
            verdict = ("they disagree → not a uniform scale; the residuals in C "
                       "are the story")
        ax[1].text(
            0.015, 0.04,
            f"scale from fit {(k-1)*100:+.3f}%   vs   from the two end targets "
            f"{(k_end-1)*100:+.3f}%\n" + verdict,
            transform=ax[1].transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round,pad=0.35",
                      fc="honeydew" if agree else "lightyellow",
                      ec="grey", alpha=.9),
        )
    if sigma_delta_m:
        ax[1].axhspan(-sigma_delta_m * 1000, sigma_delta_m * 1000,
                      color="grey", alpha=.18, zorder=0,
                      label=f"±{sigma_delta_m*1000:.0f} mm {sigma_label}")
    ax[1].axhline(0, color="k", lw=.8)
    ax[1].set_ylabel(f"Δ distance from target 0\n[mm]  ({others[0] if others else ''} − {ref_cloud})")
    axr.set_ylabel("Δ relative to distance [%]", fontsize=9)
    # Put both zeros at the same height. Two independent scales sharing a frame
    # with offset origins is a classic way to mislead: a series can look
    # positive against the other's zero line.
    l0, l1 = ax[1].get_ylim()
    frac = (0.0 - l0) / (l1 - l0) if l1 > l0 else 0.5
    r0, r1 = axr.get_ylim()
    if 0.0 < frac < 1.0 and r1 > r0:
        span = max(-r0 / frac if frac > 0 else 0.0,
                   r1 / (1 - frac) if frac < 1 else 0.0)
        axr.set_ylim(-span * frac, span * (1 - frac))
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    ax[1].grid(alpha=.3)

    # ---------------------------------------------------------------- C
    seg_ref = ref.segment_m
    xs = idx[:-1] + 0.5
    width = 0.6 / max(len(others), 1)
    for k, name in enumerate(others):
        seg = L_by_cloud[name].segment_m
        d_mm = (seg - seg_ref) * 1000
        # A segment touching a flagged target inherits that target's error, so
        # hatch it: the bar is real, but it is not evidence about the tunnel.
        touches = np.array([(i in flagged) or (i + 1 in flagged) for i in range(len(xs))])
        bars = ax[2].bar(xs + (k - (len(others) - 1) / 2) * width, d_mm, width=width,
                         color=cols.get(name, "salmon"), edgecolor="k", lw=.5,
                         label=f"Δ segment  {name} − {ref_cloud}")
        for b, t in zip(bars, touches):
            if t:
                b.set_hatch("///")
                b.set_edgecolor("firebrick")
        if touches.any():
            ax[2].bar(np.nan, np.nan, color=cols.get(name, "salmon"),
                      edgecolor="firebrick", hatch="///", lw=.5,
                      label=f"segment touching flagged target {flagged}")
        for x, v in zip(xs, d_mm):
            ax[2].annotate(f"{v:+.0f}", (x, v), ha="center", fontsize=7,
                           va="bottom" if v >= 0 else "top",
                           xytext=(0, 2 if v >= 0 else -2), textcoords="offset points")
    if sigma_delta_m:
        ax[2].axhspan(-sigma_delta_m * 1000, sigma_delta_m * 1000,
                      color="grey", alpha=.18, zorder=0)
    ax[2].axhline(0, color="k", lw=.8)
    ax[2].set_ylabel("Δ segment length\n[mm]")
    ax[2].set_xlabel("Target index (segments plotted between their two targets)")
    ax[2].set_xticks(idx)
    ax[2].legend(loc="best", fontsize=8)
    ax[2].grid(alpha=.3, axis="y")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
