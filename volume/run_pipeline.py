"""
Command 1 of 2 — compute the volume of one run (or of every rep of a dataset).

Each run writes an immutable results directory and never overwrites a previous
one; `run_statistics.py` (command 2) turns a set of those directories into the
across-run volume statistics.

Usage
-----
# One rep:
    python run_pipeline.py --run-real --dataset April_12_05_05 --rep rep00

# Every rep of a dataset (sequential; the Leica reference is processed once):
    python run_pipeline.py --run-real --dataset April_12_05_05 --rep all

# The same, but ALSO re-measuring the Leica reference in every rep. The Leica
# cloud is identical each time and only the trajectory differs, so the spread of
# those five volumes is what the centreline fit and the processing contribute on
# their own — the part of the Livox scatter that is not the walk or the sensor:
    python run_pipeline.py --run-real --dataset April_12_05_05 --rep all --cloud both

# Phantom validation only (no real data needed) — run this before trusting
# anything above; it checks the methods against a cylinder of known volume:
    python run_pipeline.py --phantom
"""

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SUMMARY_SCHEMA = 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phantom", action="store_true",
                   help="Run phantom cylinder validation first")
    p.add_argument("--phantom-hole", action="store_true",
                   help="Also run phantom with a punched hole")
    p.add_argument("--run-real", action="store_true",
                   help="Run on real ice data (requires data/ files)")

    # ---- which run(s) ----
    p.add_argument("--dataset", default=None,
                   help="Dataset directory under data/, e.g. April_12_05_05")
    p.add_argument("--rep", default=None,
                   help="Rep inside the dataset: 'rep00', a comma-separated "
                        "list, or 'all' for every rep (run sequentially)")
    p.add_argument("--cloud", choices=["auto", "leica", "livox", "both"],
                   default="auto",
                   help="Which cloud(s) to process. 'auto' (default) processes "
                        "the Livox cloud of every rep plus the shared Leica "
                        "reference ONCE per dataset — the Leica scan is the same "
                        "cloud for all reps, so re-processing it per rep only "
                        "re-measures the centreline, not the tunnel")
    p.add_argument("--leica-rep", default=None,
                   help="With --cloud auto: the rep whose centreline is used "
                        "for the Leica reference volume (default: the first rep "
                        "processed, if the dataset has no Leica result yet)")

    # ---- methods / cost ----
    p.add_argument("--no-mesh", action="store_true",
                   help="Skip the surface-mesh method")
    p.add_argument("--no-hull", action="store_true",
                   help="Skip the convex-hull bound (the slowest method)")
    p.add_argument("--mc", action="store_true",
                   help="Also run marching cubes. OFF by default: it leaks on "
                        "these clouds (wall holes wider than the shell seal) "
                        "and its voxel grids are the memory-heavy step")
    p.add_argument("--golden-only", action="store_true",
                   help="Run methods on the golden segment only (for validation)")
    p.add_argument("--no-profile-cloud", action="store_true",
                   help="Skip exporting the profiles method's interpolated "
                        "wall as a PLY (~4 MB per cloud, opens in CloudCompare "
                        "with an 'interpolated' scalar field)")

    # ---- domain end caps ----
    p.add_argument("--cap-mode", choices=["auto", "target_planes",
                                          "feature_planes"], default="auto",
                   help="Where the two domain end caps come from. 'auto' "
                        "(default) uses the dataset's caps.txt if it exists and "
                        "the surveyed targets otherwise. 'feature_planes' "
                        "requires a cap file and is what makes two campaigns "
                        "comparable; 'target_planes' forces the surveyed "
                        "targets even when a cap file is present")
    p.add_argument("--caps-file", default=None,
                   help="Filename in data/ for the two shared-feature cap "
                        "points (id,x,y,z; exactly two rows)")

    # ---- caching ----
    p.add_argument("--no-cache", action="store_true",
                   help="Recompute cylindrical coords even if cached")
    p.add_argument("--no-cache-save", action="store_true",
                   help="Do not write the (s, r, theta) npz cache (~0.5-1 GB "
                        "per cloud; a full 10-run sweep is ~9 GB)")

    # ---- data file overrides (filenames resolved inside data_dir/) ----
    p.add_argument("--leica-file", default=None,
                   help="Filename in data/ for the reference (Leica) cloud")
    p.add_argument("--livox-file", default=None,
                   help="Filename in data/ for the mobile (Livox) cloud")
    p.add_argument("--trajectory-file", default=None,
                   help="Filename in data/ for the trajectory (.ply or TUM .txt)")
    p.add_argument("--registration-file", default=None,
                   help="Filename in data/ for the registration matrix")
    p.add_argument("--targets-file", default=None,
                   help="Filename in data/ for the Leica targets")
    p.add_argument("--targets-livox-file", default=None,
                   help="Filename in data/ for the Livox cloud's own picked "
                        "target coordinates")
    p.add_argument("--tag", default=None,
                   help="Free-text label stored in summary.json (e.g. why this "
                        "run was made)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  Per-run result directory                                                   #
# --------------------------------------------------------------------------- #

def make_run_dir(cfg, dataset: str, rep: str) -> Path:
    """`results/<dataset>/<rep>/<UTC timestamp>/` — a fresh directory per run.

    The timestamp is what makes results immutable: running the same rep again
    never touches the earlier result, so the statistics command can see every
    run that was ever made (and, by default, use the newest per rep).
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    run_dir = cfg.results_dir / dataset / rep / stamp
    if run_dir.exists():                      # same-second re-run
        stamp += "_b"
        run_dir = cfg.results_dir / dataset / rep / stamp
    (run_dir / "figures").mkdir(parents=True)
    return run_dir


def append_run_index(cfg, summary: dict) -> None:
    """Append-only one-line-per-run index, for grepping without reading JSON."""
    idx = cfg.results_dir / "runs_index.csv"
    header = ("timestamp_utc,dataset,rep,clouds,L_m,"
              "livox_profiles_m3,livox_surface_mesh_m3,livox_hull_bound_m3,"
              "leica_profiles_m3,leica_surface_mesh_m3,leica_hull_bound_m3,"
              "result_dir\n")
    if not idx.exists():
        idx.write_text(header)

    def v(cloud, method):
        c = summary["results"].get(cloud)
        if not c:
            return ""
        x = c["volumes"].get(method)
        return "" if x is None else f"{x:.3f}"

    row = ",".join([
        summary["timestamp_utc"], summary["dataset"], summary["rep"],
        "+".join(summary["clouds"]), f"{summary['domain']['L_m']:.4f}",
        v("livox", "profiles"), v("livox", "surface_mesh"), v("livox", "hull_bound"),
        v("leica", "profiles"), v("leica", "surface_mesh"), v("leica", "hull_bound"),
        summary["result_dir"],
    ])
    with idx.open("a") as fh:
        fh.write(row + "\n")


def resolve_domain(cfg, args, targets, cl):
    """Where the two end caps go, and why. Returns ((s_start, s_end), info).

    Two modes, and the difference matters more than it looks:

    "target_planes" caps at two surveyed targets. Correct within one campaign —
    every rep of that campaign then measures the same stretch of tunnel — but
    the targets of April and July are NOT the same physical points (they were
    re-placed between campaigns), so two target-capped volumes are volumes of
    different stretches and their difference is not a change in the tunnel.

    "feature_planes" caps at two points picked on a feature identifiable in both
    campaigns' reference scans. Each campaign projects its own picks onto its
    own centreline, so no cross-campaign transform is needed and the result
    stays correct for a curved tunnel: the domain is an arclength interval
    between two physical places, which is exactly what has to be held fixed for
    a difference of volumes to mean something.
    """
    from io_utils import load_targets
    from spine import to_cylindrical

    def project(pts):
        s, r, _ = to_cylindrical(np.asarray(pts, dtype=float), cl)
        return s, r

    # Target-derived domain: always computed, so the log can state what the
    # feature caps changed rather than quietly replacing it.
    t_start = targets[cfg.domain_start_target_idx]
    t_end = targets[cfg.domain_end_target_idx]
    s_tgt, _ = project(np.stack([t_start, t_end]))
    tgt_domain = (float(min(s_tgt)), float(max(s_tgt)))

    caps_path = cfg.caps_path
    have_caps = caps_path is not None and caps_path.exists()
    mode = args.cap_mode
    if mode == "auto":
        mode = "feature_planes" if have_caps else "target_planes"

    info = {"cap_mode": mode,
            "caps_file": cfg.caps_csv if have_caps else None,
            "target_domain": {"s_start_m": tgt_domain[0],
                              "s_end_m": tgt_domain[1],
                              "L_m": tgt_domain[1] - tgt_domain[0]}}

    if mode == "target_planes":
        log.info("Domain caps: surveyed targets #%d and #%d (cap_mode=%s). "
                 "Comparable across the reps of THIS campaign only.",
                 cfg.domain_start_target_idx, cfg.domain_end_target_idx, mode)
        if have_caps:
            log.warning("A cap file exists (%s) but --cap-mode target_planes "
                        "was forced — this run is NOT comparable to another "
                        "campaign.", cfg.caps_csv)
        return tgt_domain, info

    if not have_caps:
        where = (str(caps_path) if caps_path is not None
                 else f"no cap file is configured for this dataset "
                      f"(expected {cfg.data_dir}/<dataset>/caps.txt)")
        raise SystemExit(
            "--cap-mode feature_planes needs two cap points, and none were "
            f"found: {where}. Pick one physical feature near "
            "each end of the tunnel in this campaign's Leica cloud — the SAME "
            "two features in every campaign — and save them as caps.txt "
            "(id,x,y,z, two rows) in the dataset directory."
        )

    caps = load_targets(caps_path)
    if len(caps) != 2:
        raise SystemExit(
            f"{caps_path} holds {len(caps)} points; a domain has exactly two "
            "end caps. One point near each end of the tunnel, nothing else."
        )

    s_cap, r_cap = project(caps)
    too_far = r_cap > cfg.cap_max_offset_m
    if too_far.any():
        raise SystemExit(
            f"Cap point(s) {np.flatnonzero(too_far).tolist()} in {caps_path} "
            f"lie {r_cap[too_far].round(2).tolist()} m off the centreline "
            f"(limit {cfg.cap_max_offset_m} m). A cap is a feature on the "
            "tunnel wall, so this is a mis-pick or a file from the wrong "
            "campaign's datum."
        )

    domain = (float(min(s_cap)), float(max(s_cap)))
    info["caps"] = [{"xyz": [float(v) for v in caps[i]],
                     "s_m": float(s_cap[i]), "r_offset_m": float(r_cap[i])}
                    for i in range(2)]
    info["shift_vs_targets_m"] = {"start": domain[0] - tgt_domain[0],
                                  "end": domain[1] - tgt_domain[1]}
    log.info(
        "Domain caps: shared features from %s (cap_mode=feature_planes). "
        "s=[%.3f, %.3f] L=%.3f m — target-capped domain would have been "
        "s=[%.3f, %.3f] L=%.3f m (start %+.3f m, end %+.3f m).",
        cfg.caps_csv, domain[0], domain[1], domain[1] - domain[0],
        tgt_domain[0], tgt_domain[1], tgt_domain[1] - tgt_domain[0],
        domain[0] - tgt_domain[0], domain[1] - tgt_domain[1],
    )
    log.info("Cap points sit %.3f m and %.3f m off the centreline "
             "(wall features, as expected).", r_cap[0], r_cap[1])
    return domain, info


def _f(x):
    """float for JSON, or None if not finite (never a silent NaN in a result)."""
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


# --------------------------------------------------------------------------- #
#  One run                                                                    #
# --------------------------------------------------------------------------- #

def run_real(cfg, args, dataset="_custom", rep="run", clouds=("leica", "livox"),
             provenance=None) -> dict:
    """Run the volume methods on `clouds` and write one result directory.

    Returns the summary dict that was written to summary.json.
    """
    if cfg.geometry_mode == "volumetric":
        log.info(
            "geometry_mode=volumetric: centreline-based methods (profiles, "
            "surface mesh, hull bound) do not apply (no single cross-section "
            "per chainage). Marching cubes on an SDF is geometry-agnostic and "
            "required here, but the volumetric driver is not implemented yet. "
            "Nothing to run in this mode."
        )
        return {}

    for attr, what in (("trajectory_file", "trajectory"),
                       ("targets_csv", "Leica target file")):
        if not getattr(cfg, attr):
            raise SystemExit(
                f"No {what} configured. Name a run with --dataset/--rep, or "
                f"pass every file explicitly (--trajectory-file, --targets-file, "
                f"--leica-file, --livox-file, --registration-file)."
            )

    t_run0 = time.time()
    run_dir = make_run_dir(cfg, dataset, rep)
    fig_dir = run_dir / "figures"

    # Every log line of this run also lands in the result directory, so a number
    # can always be traced back to the warnings that were raised while making it.
    fh = logging.FileHandler(run_dir / "run.log")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s"))
    logging.getLogger().addHandler(fh)
    try:
        summary = _run_real_inner(cfg, args, dataset, rep, clouds, provenance,
                                  run_dir, fig_dir)
        summary["runtime_s"] = round(time.time() - t_run0, 1)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        append_run_index(cfg, summary)
        log.info("Result written: %s", run_dir)
        return summary
    finally:
        logging.getLogger().removeHandler(fh)
        fh.close()


def _run_real_inner(cfg, args, dataset, rep, clouds, provenance, run_dir,
                    fig_dir) -> dict:
    from io_utils import (load_ply, load_trajectory, load_and_transform_trajectory,
                          load_targets, load_cache, save_cache, cache_exists,
                          load_registration, check_rigid_registration)
    from spine import (fit_centreline, to_cylindrical, check_cloud_aligned,
                       build_gap_map, find_golden_segment,
                       plot_centreline, plot_gap_map, plot_radius_histogram,
                       plot_theta_reference)
    from cross_sections import (run_profiles, run_hull_bound, plot_area_profile,
                                plot_cross_sections, plot_cross_sections_combined)

    log.info("=" * 70)
    log.info("  RUN  dataset=%s  rep=%s  clouds=%s", dataset, rep, "+".join(clouds))
    log.info("=" * 70)

    # ------------------------------------------------------------ Registration
    reg = None
    if cfg.registration_txt and cfg.registration_path.exists():
        log.info("Checking registration matrix: %s", cfg.registration_path)
        reg = load_registration(cfg.registration_path)
        check_rigid_registration(reg, tol=cfg.registration_rigid_tol)

    # ------------------------------------------------------------------ I/O
    log.info("Loading trajectory: %s", cfg.trajectory_path)
    if reg is not None and cfg.trajectory_path.suffix.lower() != ".ply":
        log.info(
            "Applying registration matrix to trajectory (GLIM/Livox-native "
            "SLAM output — unlike the point clouds it is not pre-transformed)"
        )
        traj = load_and_transform_trajectory(cfg.trajectory_path, reg)
    else:
        traj = load_trajectory(cfg.trajectory_path)

    log.info("Loading targets: %s", cfg.targets_path)
    targets = load_targets(cfg.targets_path)
    if cfg.domain_end_target_idx >= len(targets):
        raise ValueError(
            f"cfg.domain_end_target_idx={cfg.domain_end_target_idx} is out of "
            f"range for {cfg.targets_path} ({len(targets)} targets, valid "
            f"rows 0..{len(targets) - 1})"
        )

    targets_livox = None
    if cfg.targets_livox_path is not None and cfg.targets_livox_path.exists():
        targets_livox = load_targets(cfg.targets_livox_path)

    t_start = targets[cfg.domain_start_target_idx]
    t_end   = targets[cfg.domain_end_target_idx]
    log.info("Domain targets:  #%d %s  →  #%d %s",
             cfg.domain_start_target_idx, t_start.round(3),
             cfg.domain_end_target_idx,   t_end.round(3))

    # -------------------------------------------------------- Centreline
    log.info("Fitting general spline centreline on trajectory …")
    # Orientation stays target-derived in every cap mode: it only fixes which
    # way s counts, and keeping it independent of the caps means adding a cap
    # file cannot flip the sign of s and invalidate every cached coordinate.
    cl = fit_centreline(traj, cfg, orient_toward=t_end)

    # Domain in s-coordinates — from the surveyed targets, or from the shared
    # feature caps if this dataset has them (see resolve_domain).
    domain, cap_info = resolve_domain(cfg, args, targets, cl)
    cfg.cap_mode = cap_info["cap_mode"]
    s_start, s_end = domain
    log.info("Domain: s=[%.3f, %.3f] m  L=%.3f m  (cap_mode=%s)",
             s_start, s_end, s_end - s_start, cfg.cap_mode)

    # ----------------------------------------- Cylindrical coords (with cache)
    cloud_paths = {"leica": cfg.leica_path, "livox": cfg.livox_path}
    cyl_data, n_points, loaded_pts = {}, {}, {}
    for name in clouds:
        ply_path = cloud_paths[name]
        cache_file = cfg.cache_path(f"{name}_cyl")
        # Signature covers every setting that changes (s, r, theta), so a stale
        # cache is detected rather than silently reused (load_cache -> None).
        sig = cfg.coord_signature(ply_name=Path(ply_path).name)
        cached = None
        if not args.no_cache and cache_exists(cache_file):
            cached = load_cache(cache_file, signature=sig)
            if cached is not None:
                log.info("Loading cached cylindrical coords for %s (%s)",
                         name, cache_file)

        if cached is not None:
            cyl_data[name] = cached
            n_points[name] = int(len(cached["s"]))
        else:
            log.info("Loading cloud: %s", ply_path)
            pts = load_ply(ply_path)
            log.info("  %d points", len(pts))
            n_points[name] = int(len(pts))
            check_cloud_aligned(pts, cl, name)
            s, r, theta = to_cylindrical(pts, cl)
            if not args.no_cache_save:
                save_cache(cache_file, signature=sig, s=s, r=r, theta=theta)
            cyl_data[name] = {"s": s, "r": r, "theta": theta}
            loaded_pts[name] = pts

    # Centreline + θ-verification figures need the raw xyz. Only draw them if a
    # cloud was loaded anyway — re-reading a 1 GB PLY just for a figure on an
    # otherwise cached run is not worth the ten minutes.
    if loaded_pts:
        fig_cloud = next(iter(loaded_pts))
        plot_centreline(traj, cl, loaded_pts[fig_cloud],
                        save_path=str(fig_dir / "01_centreline.png"))
        plot_theta_reference(loaded_pts[fig_cloud], cl, cfg, fig_cloud,
                             domain=domain,
                             save_path=str(fig_dir / "01b_theta_reference.png"))
    else:
        log.info("All clouds came from cache — skipping figures 01/01b "
                 "(they need the raw points; re-run with --no-cache for them)")
    for name, d in cyl_data.items():
        plot_radius_histogram(
            d["r"], name,
            save_path=str(fig_dir / f"02_radius_hist_{name}.png"))

    # ------------------------------------------------------------ Gap maps
    gap_maps = {}
    for name, d in cyl_data.items():
        gm = build_gap_map(d["s"], d["theta"], d["r"], cfg, domain=domain)
        gap_maps[name] = gm
        log.info("Gap map %s: %.1f%% missing (median slab coverage %.1f%%)",
                 name, gm.frac_missing * 100,
                 float(np.median(gm.coverage_per_slab)) * 100)
        plot_gap_map(gm, name, save_path=str(fig_dir / f"03_gap_map_{name}.png"))

    if len(gap_maps) == 2:
        gm_list = list(gap_maps.values())
        s_gold_lo, s_gold_hi = find_golden_segment(gm_list[0], gm_list[1])
    else:
        s_gold_lo, s_gold_hi = domain

    # ------------------------- profiles / hull bound / surface mesh / (MC) ----
    all_profiles, all_hull, all_mesh, all_mc = {}, {}, {}, {}
    all_profile_clouds = {}
    domain_to_run = (s_gold_lo, s_gold_hi) if args.golden_only else domain

    for name, d in cyl_data.items():
        s, r, theta = d["s"], d["r"], d["theta"]
        prof = run_profiles(s, r, theta, domain_to_run, cfg, cloud_name=name)
        all_profiles[name] = prof

        # The wall the profiles method integrated, as a cloud you can open next
        # to the raw scan and colour by what was measured vs what was filled in.
        if not args.no_profile_cloud and prof.r_grid is not None:
            from profile_cloud import export_profile_cloud
            try:
                all_profile_clouds[name] = export_profile_cloud(
                    prof, cl, run_dir / f"profile_cloud_{name}.ply",
                    domain=domain_to_run)
            except Exception as exc:      # a failed export must not cost a run
                log.warning("Profile cloud (%s) not exported: %s", name, exc)

        hull = None
        if not args.no_hull:
            hull = run_hull_bound(s, r, theta, domain_to_run, cfg, cloud_name=name)
            all_hull[name] = hull

        if not args.no_mesh:
            from surface_mesh import run_surface_mesh
            all_mesh[name] = run_surface_mesh(
                s, r, theta, domain_to_run, cfg, cl, cloud_name=name,
                export_path=str(run_dir / f"surface_mesh_{name}.ply"),
            )

        if args.mc:
            from marching_cubes import run_marching_cubes
            # MC needs the RAW points, not (s, r, θ).
            mc_pts = loaded_pts.get(name)
            if mc_pts is None:
                mc_pts = load_ply(cloud_paths[name])
            all_mc[name] = run_marching_cubes(
                mc_pts, cfg, cl=cl, domain=domain_to_run, cloud_name=name,
                export_path=str(run_dir / f"marching_cubes_{name}.ply"),
            )
            del mc_pts

        plot_area_profile(
            prof, hull, save_path=str(fig_dir / f"04_area_profile_{name}.png"))
        plot_cross_sections(
            s, r, theta, prof, domain_to_run,
            save_path=str(fig_dir / f"05_cross_sections_{name}.png"))

    loaded_pts.clear()

    if len(cyl_data) == 2:
        data_by_cloud = {n: (d["s"], d["r"], d["theta"]) for n, d in cyl_data.items()}
        plot_cross_sections_combined(
            data_by_cloud, domain_to_run,
            save_path=str(fig_dir / "05b_cross_sections_combined.png"))

    # ---------------------------------------------------------- Summary log
    log.info("\n%s", "=" * 60)
    log.info("  VOLUME SUMMARY  (%s %s)", dataset, rep)
    log.info("  Domain: s=[%.3f, %.3f] m  L=%.3f m", *domain_to_run,
             domain_to_run[1] - domain_to_run[0])
    log.info("  %-6s %10s %10s %10s %10s %8s %8s", "cloud", "profiles",
             "surf.mesh", "hull(UB)", "MC(h→0)", "Ā [m²]", "interp")
    for name in cyl_data:
        prof = all_profiles[name]
        mesh = all_mesh.get(name)
        hull = all_hull.get(name)
        mc = all_mc.get(name)
        mc_str = "—"
        if mc is not None:
            mc_str = (f"{mc.V_extrapolated:.2f}"
                      if np.isfinite(mc.V_extrapolated) else "LEAK")
        log.info("  %-6s %10.2f %10s %10s %10s %8.3f %7.1f%%",
                 name, prof.V_trap,
                 f"{mesh.V_mesh:.2f}" if mesh else "—",
                 f"{hull.V_hull:.2f}" if hull else "—",
                 mc_str, prof.mean_area_m2,
                 float(np.mean(prof.frac_interp)) * 100)

    # profiles vs surface mesh: these SHARE the (s,θ)->r extraction, so this
    # compares integration + hole-filling only. It is not an independent check.
    for name, mesh in all_mesh.items():
        v_p = all_profiles[name].V_trap
        if v_p > 0:
            d_pct = (mesh.V_mesh - v_p) / v_p * 100
            log.info("  %-6s  profiles vs surface mesh: %+.2f%%  "
                     "(shared extraction — tests integration + fill only)",
                     name, d_pct)
            if abs(d_pct) > 2.0:
                log.warning(
                    "  %s: profiles and surface mesh differ by %.2f%% (>2%%). "
                    "They share the r(s,θ) extraction, so this gap is in the "
                    "integration or the hole-filling — investigate before "
                    "trusting either.", name, d_pct,
                )

    # profiles vs marching cubes: MC is INDEPENDENT (no r(s,θ) grid), so this is
    # the real cross-check — it can catch an extraction error the two above share.
    for name, mc in all_mc.items():
        v_p = all_profiles[name].V_trap
        if mc.leaked or not np.isfinite(mc.V_extrapolated):
            log.info("  %-6s  marching cubes LEAKED (wall holes > seal) — no "
                     "independent volume; expected on these clouds.", name)
        elif v_p > 0:
            log.info("  %-6s  profiles vs marching cubes: %+.2f%%  "
                     "(INDEPENDENT — different extraction entirely)",
                     name, (mc.V_extrapolated - v_p) / v_p * 100)

    if len(all_profiles) == 2:
        names = list(all_profiles)
        v0, v1 = (all_profiles[n].V_trap for n in names)
        diff_pct = abs(v0 - v1) / ((v0 + v1) / 2) * 100
        log.info("  %s vs %s (profiles): %.2f%%", names[0], names[1], diff_pct)
    log.info("%s\n", "=" * 60)

    # ---------------------------------------- target geometry / scale / Ā×L
    from decomposition import (compute_decomposition, compute_length,
                               plot_cumulative_distance, plot_target_3d_distances,
                               target_consistency, scale_from_targets)
    from config import SIGMA_TARGET_PICK

    scale = None
    consistency = None
    decomp_out = None
    if targets_livox is not None and len(targets_livox) == len(targets):
        # Needs only the two target files, so it is recorded for EVERY run —
        # including Livox-only ones, which is what gives the statistics a
        # per-rep SLAM scale number alongside the per-rep volume.
        scale = scale_from_targets(targets, targets_livox)
        log.info("Target scale (Livox/Leica): k=%.5f ± %.5f (drift %+.3f%%), "
                 "pairwise resid RMS %.4f m",
                 scale["k"], scale["sigma_k"], (scale["k"] - 1) * 100,
                 scale["resid_rms"])
        consistency = target_consistency(targets, targets_livox)

        sigma_delta = 2.0 * SIGMA_TARGET_PICK
        log.info("Picking repeatability σ=%.0f mm/pick → ±%.0f mm on a "
                 "cloud-to-cloud distance difference.",
                 SIGMA_TARGET_PICK * 1000, sigma_delta * 1000)
        targets_by_cloud = {"leica": targets, "livox": targets_livox}
        L_by_cloud = {
            n: compute_length(t, cfg.domain_start_target_idx,
                              cfg.domain_end_target_idx, cl=cl)
            for n, t in targets_by_cloud.items()
        }
        plot_cumulative_distance(
            L_by_cloud,
            save_path=str(fig_dir / "06_cumulative_distance.png"),
            ref_cloud="leica", sigma_delta_m=sigma_delta,
            flagged_targets=consistency["flagged"],
        )
        plot_target_3d_distances(
            targets_by_cloud,
            save_path=str(fig_dir / "12_target_3d_distances.png"),
            ref_cloud="leica", sigma_delta_m=sigma_delta,
            flagged_targets=consistency["flagged"],
        )

        # V = Ā × L needs BOTH clouds' Ā over the shared domain.
        if len(all_profiles) == 2 and not args.golden_only:
            decomp = compute_decomposition(
                cfg, targets_by_cloud=targets_by_cloud,
                mean_area_m2_by_cloud={n: all_profiles[n].mean_area_m2
                                       for n in all_profiles},
                cl=cl,   # distances measured ALONG the centreline, not as chords
            )
            decomp_out = {"L_ratio": _f(decomp.L_ratio),
                          "area_ratio": _f(decomp.area_ratio),
                          "V_ratio": _f(decomp.V_ratio),
                          "L_m": {n: _f(v.L_m) for n, v in decomp.L.items()},
                          "same_targets_for_both": bool(decomp.same_targets_for_both)}

    # ----------------------------- where the two clouds differ (both-cloud run)
    if len(all_profiles) == 2 and not args.golden_only:
        from slice_compare import compare_volume_slices
        compare_volume_slices(
            all_profiles, cfg, ref_cloud="leica", other_cloud="livox",
            save_path=str(fig_dir / "07_volume_slices.png"))

        from lateral_compare import run_lateral_compare
        run_lateral_compare(
            cyl_data, domain, cfg, ref_cloud="leica", other_cloud="livox",
            save_path=str(fig_dir / "08_lateral_compare.png"))

    # ------------------------------------------------------------- summary.json
    results_out = {}
    for name in cyl_data:
        prof = all_profiles[name]
        mesh = all_mesh.get(name)
        hull = all_hull.get(name)
        mc = all_mc.get(name)
        gm = gap_maps[name]
        results_out[name] = {
            "n_points": n_points[name],
            "gap_frac_missing": _f(gm.frac_missing),
            "median_slab_coverage": _f(np.median(gm.coverage_per_slab)),
            "volumes": {
                # The primary number per method. profiles -> trapezoidal (Simpson
                # is kept alongside as an integration cross-check, not a
                # separate method).
                "profiles": _f(prof.V_trap),
                "profiles_simpson": _f(prof.V_simp),
                "surface_mesh": _f(mesh.V_mesh) if mesh else None,
                "hull_bound": _f(hull.V_hull) if hull else None,
                "marching_cubes": (_f(mc.V_extrapolated)
                                   if mc and not mc.leaked else None),
            },
            "mean_area_m2": _f(prof.mean_area_m2),
            "frac_interp_mean": _f(np.mean(prof.frac_interp)),
            "n_bimodal_slabs": int(prof.n_bimodal_slabs),
            "mesh_watertight": bool(mesh.is_watertight) if mesh else None,
            "marching_cubes_leaked": bool(mc.leaked) if mc else None,
            "profile_cloud": all_profile_clouds.get(name),
        }

    summary = {
        "schema": SUMMARY_SCHEMA,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": dataset,
        "rep": rep,
        "clouds": list(cyl_data),
        "result_dir": str(run_dir.relative_to(Path(__file__).parent)),
        "tag": args.tag,
        "inputs": provenance,
        "config": {
            "coord_signature": cfg.coord_signature(
                ply_name=Path(cfg.livox_path).name),
            "cap_mode": cfg.cap_mode,
            "geometry_mode": cfg.geometry_mode,
            "domain_start_target_idx": cfg.domain_start_target_idx,
            "domain_end_target_idx": cfg.domain_end_target_idx,
            "theta_reference": cfg.theta_reference,
            "profile_ds_m": cfg.profile_ds_m,
            "profile_dtheta_deg": cfg.profile_dtheta_deg,
            "centreline_traj_resample_ds_m": cfg.centreline_traj_resample_ds_m,
            "centreline_smoothing_factor": cfg.centreline_smoothing_factor,
            "centreline_resample_ds_m": cfg.centreline_resample_ds_m,
            "golden_only": bool(args.golden_only),
        },
        "domain": {"s_start_m": _f(domain_to_run[0]),
                   "s_end_m": _f(domain_to_run[1]),
                   "L_m": _f(domain_to_run[1] - domain_to_run[0]),
                   # How the caps were placed, and what they moved relative to
                   # the surveyed targets — a volume is only ever comparable to
                   # another volume capped the same way.
                   "caps": cap_info},
        "centreline": {"total_length_m": _f(cl.total_length_m),
                       "fit_rms_m": _f(cl.fit_rms_m),
                       "kappa_max_1pm": _f(cl.kappa_max_1pm),
                       "n_traj_poses": int(len(traj))},
        "targets": {
            "n": int(len(targets)),
            "have_livox_picks": targets_livox is not None,
            "scale_livox_over_leica": (
                {k: _f(v) for k, v in scale.items()} if scale else None),
            "flagged_targets": (consistency["flagged"] if consistency else None),
        },
        "decomposition": decomp_out,
        # Per-cloud volumes [m³] over the domain above. "profiles" is the
        # trapezoidal integration; "profiles_simpson" is the same method with a
        # different integration rule, kept as a cross-check, NOT a 4th method.
        "results": results_out,
    }
    return summary


# --------------------------------------------------------------------------- #
#  Driver                                                                     #
# --------------------------------------------------------------------------- #

def _dataset_has_leica_result(cfg, dataset: str) -> bool:
    """Has any previous run of this dataset produced a Leica volume?"""
    root = cfg.results_dir / dataset
    if not root.is_dir():
        return False
    for p in root.glob("*/*/summary.json"):
        try:
            s = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "leica" in s.get("clouds", []):
            return True
    return False


def main() -> None:
    args = parse_args()

    from config import Config
    import dataset as ds_mod
    cfg = Config()

    if args.phantom or not (args.run_real or args.dataset):
        log.info("=== PHANTOM VALIDATION ===")
        from phantom import run_phantom_test
        run_phantom_test(cfg, with_hole=False)
        if args.phantom_hole:
            run_phantom_test(cfg, with_hole=True)

    if not (args.run_real or args.dataset):
        log.info("Nothing else to do. Typical commands:")
        log.info("  python run_pipeline.py --phantom")
        log.info("  python run_pipeline.py --run-real --dataset April_12_05_05 --rep all")
        log.info("  python run_statistics.py --dataset April_12_05_05")
        return

    # ---- which reps ----
    if args.dataset:
        reps = (ds_mod.list_reps(cfg.data_dir, args.dataset)
                if args.rep in (None, "all")
                else [r.strip() for r in args.rep.split(",") if r.strip()])
    else:
        reps = [None]           # single run from explicit file flags

    # ---- which cloud(s) per rep ----
    leica_rep = None
    if args.cloud == "auto" and args.dataset:
        leica_rep = args.leica_rep or reps[0]
        if leica_rep not in reps:
            raise SystemExit(f"--leica-rep {leica_rep} is not among the reps to "
                             f"run ({', '.join(reps)})")
        if _dataset_has_leica_result(cfg, args.dataset):
            log.info("Dataset %s already has a Leica reference result — "
                     "processing Livox only. Force it with --cloud both.",
                     args.dataset)
            leica_rep = None
        else:
            log.info("Leica reference volume will be computed once, on %s "
                     "(the Leica scan is identical for every rep).", leica_rep)

    if args.cloud == "both" and len(reps) > 1:
        log.info(
            "--cloud both over %d reps: the Leica reference is re-measured in "
            "every rep. Its cloud is identical each time and only the "
            "trajectory differs, so the spread of those %d volumes isolates "
            "what the centreline fit and the processing contribute.",
            len(reps), len(reps),
        )

    log.info("Runs queued: %s", ", ".join(str(r) for r in reps))

    failures = []
    for rep in reps:
        if args.dataset:
            run = ds_mod.resolve_run(cfg.data_dir, args.dataset, rep)
            cfg = Config()
            ds_mod.apply_to_config(cfg, run)
            provenance = ds_mod.file_provenance(cfg.data_dir, run)
            name, rep_name = args.dataset, rep
        else:
            cfg = Config()
            provenance = None
            name, rep_name = "_custom", "run"

        # explicit flags always win over dataset discovery
        for attr, val in (("leica_ply", args.leica_file),
                          ("livox_ply", args.livox_file),
                          ("trajectory_file", args.trajectory_file),
                          ("registration_txt", args.registration_file),
                          ("targets_csv", args.targets_file),
                          ("targets_livox_csv", args.targets_livox_file),
                          ("caps_csv", args.caps_file)):
            if val:
                setattr(cfg, attr, val)

        if args.cloud == "auto":
            clouds = ("leica", "livox") if (rep == leica_rep and args.dataset) \
                else (("livox",) if args.dataset else ("leica", "livox"))
        elif args.cloud == "both":
            clouds = ("leica", "livox")
        else:
            clouds = (args.cloud,)

        try:
            run_real(cfg, args, dataset=name, rep=rep_name, clouds=clouds,
                     provenance=provenance)
        except Exception as exc:                       # one bad rep must not
            failures.append((rep_name, exc))           # abort a 10-run sweep
            log.error("RUN FAILED  %s %s: %s", name, rep_name, exc)
            log.error("%s", traceback.format_exc())

    if failures:
        log.error("%d of %d run(s) failed:", len(failures), len(reps))
        for rep_name, exc in failures:
            log.error("  %s: %s", rep_name, exc)
        sys.exit(1)

    if args.dataset:
        log.info("Done. Now aggregate:  python run_statistics.py --dataset %s",
                 args.dataset)


if __name__ == "__main__":
    main()
