"""Single source of truth for all pipeline parameters and paths."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# Manual target-picking repeatability, 1σ per pick [m]. A DISTANCE between two
# targets carries √2·σ; the cloud-to-cloud DIFFERENCE of such a distance carries
# four picks (2 targets × 2 clouds) → 2σ = 20 mm, which is the band drawn on
# figures 06 and 12. Used by decomposition.py / run_pipeline.py.
SIGMA_TARGET_PICK: float = 0.010


@dataclass
class Config:
    # ------------------------------------------------------------------ paths
    # All the *_ply / *_file / *_csv names below are resolved INSIDE data_dir and
    # are normally filled in by dataset.resolve_run() from `--dataset/--rep`
    # (run_pipeline.py), so they are left empty here rather than pointing at one
    # particular acquisition. Pass them explicitly (--leica-file, …) only for a
    # dataset that does not follow the <dataset>/<rep>/ layout.
    data_dir: Path = Path(__file__).parent / "data"
    leica_ply: str = ""
    # MUST be the cloud already registered into the Leica datum (the
    # "…_clean_and_regist.ply" in each rep folder). An unregistered cloud does
    # not overlap the Leica bbox at all; it would silently yield ~96% gap-map
    # "missing" and a nonsense Ā, so spine.check_cloud_aligned rejects it.
    livox_ply: str = ""
    # Trajectory file: TUM format (t x y z qx qy qz qw), whitespace-separated.
    # Loaded via load_trajectory() in io_utils, NOT load_ply.
    # RAW GLIM/Livox-frame poses — do NOT pre-transform into the Leica datum.
    # Unlike the clouds (transformed once, externally), the trajectory is
    # registered by the pipeline at load (load_and_transform_trajectory), so a
    # pre-transformed file gets the matrix applied twice.
    trajectory_file: str = ""
    registration_txt: str = ""
    targets_csv: str = ""
    # Second target-pick file: the Livox cloud's own picked coordinates for the
    # same physical targets, in the Leica frame. Gives a real per-cloud length L
    # for the scale-error decomposition; if unset, both clouds use targets_csv
    # and the L ratio is trivially 1.0.
    targets_livox_csv: Optional[str] = None
    # Optional dataset-level cap file: EXACTLY TWO points, same id,x,y,z format
    # as the targets, picked on a physical feature that is identifiable in more
    # than one campaign's reference scan (a bolt, a bracket, a door-frame
    # corner). Its reason to exist is cross-campaign comparability: the surveyed
    # targets were not placed at the same physical spots in April and July, so
    # target-capped domains cover different stretches of tunnel and their
    # volumes are not differences of the same thing. Caps picked on a shared
    # feature are. When set, cap_mode becomes "feature_planes" and these two
    # points replace the targets as the domain ends — see the cap_mode note.
    caps_csv: Optional[str] = None
    output_dir: Path = Path(__file__).parent / "output"
    cache_dir: Path = Path(__file__).parent / "cache"
    figures_dir: Path = Path(__file__).parent / "figures"
    # Root for per-run results. Every volume run writes an immutable
    # results/<dataset>/<rep>/<timestamp>/ directory (summary.json + figures);
    # nothing is ever overwritten, which is what makes the across-run statistics
    # in run_statistics.py possible.
    results_dir: Path = Path(__file__).parent / "results"
    # Sub-directory of cache_dir for this run's (s, r, θ) npz files, e.g.
    # "April_12_05_05/rep00". Set by run_pipeline from --dataset/--rep. Without
    # it every rep would write cache/livox_cyl.npz and evict the previous rep's
    # (each npz is ~0.5–1 GB, so a full 10-run sweep is ~9 GB — use
    # --no-cache-save if that matters more than re-running a rep quickly).
    cache_key: str = ""

    # -------------------------------------------------- domain end-cap targets
    # 0-based row indices in targets.csv that define the two volume end caps.
    # Row 0 = first target in tunnel, row 7 = last target (8 targets total,
    # ids 1-8 -> 0-based rows 0-7).
    # SAME two planes are used for every method (profiles, hull bound,
    # surface mesh, and Ā in the decomposition report) so all comparisons
    # share one measurement domain.
    domain_start_target_idx: int = 0
    domain_end_target_idx: int = 7
    # How the two domain end caps were derived. Recorded in summary.json and in
    # run_statistics' MUST_MATCH_CONFIG, so a target-capped run can never be
    # averaged in with a feature-capped one — they measure different stretches
    # of tunnel and their spread would be a domain change, not a measurement.
    #   "target_planes"  : caps at surveyed targets domain_start/end_target_idx.
    #                      Valid WITHIN one campaign; the targets of two
    #                      campaigns are not the same physical points.
    #   "feature_planes" : caps at the two points in caps_csv, picked on a
    #                      physical feature visible in both campaigns' reference
    #                      scans. THE ONLY MODE IN WHICH TWO CAMPAIGNS' VOLUMES
    #                      ARE DIFFERENCES OF THE SAME THING.
    # Set automatically to "feature_planes" when a cap file is present; force
    # either way with run_pipeline's --cap-mode.
    cap_mode: str = "target_planes"
    # A cap point further than this from the centreline [m] is refused. A cap is
    # a feature ON the tunnel wall, so a few metres is the most it can honestly
    # be; a larger offset means the pick landed on the wrong thing or the file
    # is in the wrong campaign's datum — either way the domain would be silently
    # wrong rather than obviously wrong, which is what this prevents.
    cap_max_offset_m: float = 5.0

    # ---------------------------------------------------- geometry regime
    # "tube": centreline-based methods apply (profiles, surface mesh,
    #         hull bound). Requires a single traversable centreline.
    # "volumetric": no single cross-section per chainage (cave/chamber/branching).
    #         Only marching cubes (on an SDF) applies; it needs no centreline.
    geometry_mode: str = "tube"

    # --------------------------------------------------- centreline (spline)
    # General smooth-spline + rotation-minimizing-frame centreline. Degenerates
    # to a near-straight line for a tunnel; also works for a curved cave path.
    # Uniform arclength spacing to resample the RAW trajectory to before
    # fitting [m]. The trajectory is time-sampled, so each operator pause (one
    # per target) is a dense cluster of jittering near-stationary points that
    # chord-length parameterization turns into a spline cusp. Resampling by
    # arclength removes that weighting. Must be paired with enough smoothing —
    # resampling alone still left kappa_max ~ 84 1/m.
    centreline_traj_resample_ds_m: float = 0.50
    # scipy.splprep smoothing factor, scaled by point count (s = factor * N).
    # 0.05 with the 0.5 m resample gives kappa_max ~ 0.03 1/m (30 m bend
    # radius) at 0.11 m fit RMS. 0.2 collapses the fit to a single cubic (zero
    # interior knots) and doubles the residual; 0.01 leaves cusps in.
    centreline_smoothing_factor: float = 0.05
    # Arclength spacing to resample the fitted spline at [m].
    centreline_resample_ds_m: float = 0.10
    # Warn if trajectory-to-spline RMS fit residual exceeds this [m] — signals
    # under-smoothing (noise) or a fit that isn't tracking the path well.
    # Measured against the raw outbound leg, not the resampled one.
    centreline_fit_rms_warn_m: float = 0.30
    # Warn if the fitted centreline's max curvature exceeds this [1/m]. A real
    # tunnel bends on a scale of tens of metres (~0.03 1/m); anything far above
    # this is a fit artefact, not geology, and corrupts theta via the RMF.
    centreline_max_curvature_warn: float = 0.50

    # ------------------------------------------------ theta reference frame
    # Defines what theta=0 MEANS. Without this, theta=0 is wherever an
    # arbitrary seed vector happened to point, so no result can be stated as
    # "the gap is in the ceiling" — only as "the gap is at theta=+37deg",
    # which is unfalsifiable and useless in a writeup.
    #   "up"  : theta=0 points UP (against gravity) -> the ceiling.
    #           theta=+-180 = floor, theta=+-90 = the two side walls.
    #           Built by projecting the world up-vector perpendicular to the
    #           tangent. Well-defined for any path that is not near-vertical.
    #           USE THIS for a tunnel.
    #   "rmf" : rotation-minimizing frame seeded from an arbitrary helper
    #           vector (the original behaviour). theta=0 has NO physical
    #           meaning. Needed only where "up" degenerates — i.e. a path that
    #           goes near-vertical (a shaft), or a future cave/volumetric case.
    theta_reference: str = "up"
    # World up direction. The Leica datum is Z-up.
    theta_up_vector: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    # Warn if the tangent tilts more than this from horizontal [deg]. As the
    # tangent approaches vertical, "up projected perpendicular to the tangent"
    # shrinks to zero length and theta becomes numerically unstable, then
    # meaningless. If this fires, switch theta_reference to "rmf".
    theta_up_max_tangent_tilt_deg: float = 75.0

    # ---------------------------------------------------------- gap-map grid
    gap_map_ds_m: float = 0.25         # along-tunnel bin width [m]
    gap_map_dtheta_deg: float = 2.0    # azimuth bin width [deg]

    # --------------------------------------------- profiles (cross sections)
    profile_ds_m: float = 0.25             # slab thickness [m]
    profile_dtheta_deg: float = 1.0        # azimuth bin resolution [deg]
    # A theta bin needs ≥ this many points to be counted as measured (not NaN)
    profile_min_pts_per_bin: int = 1
    # Gap in sorted r values that triggers the star-shape bimodal flag [m]
    profile_cluster_gap_r_m: float = 0.30
    # Fraction of bimodal bins above which we abort (vs. caveat-and-continue)
    profile_cluster_abort_frac: float = 0.20
    # Slabs with < this fraction of theta bins filled are treated as fully
    # missing and filled via along-s interpolation.
    profile_min_theta_coverage: float = 0.10

    # -------------------------------------------- per-segment slice comparison
    # Segment length for the along-s Leica-vs-Livox volume comparison [m]. The
    # cumulative ΔV(s) curve shows whether the gap is gradual (straight line) or
    # localised (kinks); the per-segment bars localise it.
    slice_segment_m: float = 10.0

    # ------------------------------------------------------- marching cubes
    # Trajectory-INDEPENDENT volume: voxelise the raw cloud, seal the wall shell,
    # flood-fill the exterior, and the enclosed free space is the air. The only
    # place the centreline enters (tube mode) is capping the two open ends; a
    # cave ("volumetric" mode) needs no caps and no trajectory at all.
    # Voxel sizes to sweep, for a discretisation series extrapolated to h->0 [m].
    marching_cubes_voxel_sizes_m: List[float] = field(
        default_factory=lambda: [0.05, 0.10, 0.15]
    )
    # Voxels to DILATE the wall shell by before flood-filling [integer voxels].
    # A 1-voxel-thick voxelised surface has diagonal pinholes that a
    # face-connected flood-fill leaks through; dilating by >=1 makes the wall a
    # watertight >=2-voxel barrier. Must be a FIXED voxel count (not derived
    # from h): then the inward bias it introduces is O(h) and cancels in the
    # h->0 extrapolation. 1 suffices for a cloud sampled finer than h; raise to
    # 2 only if a sparse-but-real surface still leaks (at the cost of more bias,
    # still O(h)). It does NOT bridge the metre-scale Livox FOV bands — MC is
    # meant to leak there (an honest, informative failure — see docs).
    marching_cubes_seal_iterations: int = 1
    # A run "leaked" if the largest enclosed cavity is smaller than this [m³] —
    # the shell had a hole bigger than the seal, the cavity merged with the
    # outside, and only tiny noise pockets survive. Well below any real
    # tunnel/cave (>1000 m³ here) and well above pocket noise (<3 m³), so the
    # split is unambiguous. Lower it for a genuinely small chamber.
    marching_cubes_leak_min_m3: float = 10.0
    # Skip any voxel size whose grid would exceed this many cells (labeling +
    # marching cubes are the memory hogs). At 0.05 m the real tunnel bbox is
    # ~800M cells, so it is skipped on real data but still runs on the phantom.
    marching_cubes_max_voxels: float = 2.0e8

    # ---------------------------------------------------------- phantom test
    phantom_radius_m: float = 2.0
    phantom_length_m: float = 100.0
    phantom_n_theta: int = 360          # points per cross-section ring
    # Rings along the axis. 2500 -> 0.04 m ring spacing, finer than the smallest
    # marching-cubes voxel (0.05 m), so the voxelised wall has no axial gaps and
    # the 1-voxel seal makes it watertight. A real ~1 cm cloud is far denser than
    # any voxel, so this just makes the phantom representative of that.
    phantom_n_z: int = 2500             # rings along the axis
    # Trajectory wander noise (sigma) to simulate hand-held motion [m]
    phantom_traj_noise_m: float = 0.05

    # ----------------------------------------- synthetic-hole validation
    # Fraction of (s, θ) cells to randomly black out in the hole test
    synth_hole_fraction: float = 0.20
    synth_hole_n_trials: int = 50

    # ------------------------------------------------- registration sanity
    # Registration must be RIGID (6-DOF), never scaled (7-DOF Helmert) — a
    # scaled alignment would absorb the SLAM scale error and erase the
    # length-difference signal the decomposition below is trying to measure.
    # Tolerance on |det(R) - 1| and orthonormality of the rotation block.
    registration_rigid_tol: float = 1e-3

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.cache_dir = Path(self.cache_dir)
        self.figures_dir = Path(self.figures_dir)
        self.results_dir = Path(self.results_dir)
        for d in (self.output_dir, self.cache_dir, self.figures_dir,
                  self.results_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------- path helpers
    @property
    def leica_path(self) -> Path:
        return self.data_dir / self.leica_ply

    @property
    def livox_path(self) -> Path:
        return self.data_dir / self.livox_ply

    @property
    def trajectory_path(self) -> Path:
        return self.data_dir / self.trajectory_file

    @property
    def registration_path(self) -> Path:
        return self.data_dir / self.registration_txt

    @property
    def targets_path(self) -> Path:
        return self.data_dir / self.targets_csv

    @property
    def targets_livox_path(self) -> Optional[Path]:
        return self.data_dir / self.targets_livox_csv if self.targets_livox_csv else None

    @property
    def caps_path(self) -> Optional[Path]:
        return self.data_dir / self.caps_csv if self.caps_csv else None

    def cache_path(self, name: str) -> Path:
        d = self.cache_dir / self.cache_key if self.cache_key else self.cache_dir
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{name}.npz"

    def coord_signature(self, ply_name: str = "") -> str:
        """
        Fingerprint of every setting that changes the cached (s, r, theta).

        The cache is keyed by cloud name alone, so without this a config change
        silently reuses coordinates computed under the OLD settings — which has
        already caused two wrong-number incidents (an unregistered PLY reused
        under the 'livox' key, and a centreline change reused across runs).
        Stored in the npz and verified on load; a mismatch forces a recompute
        instead of quietly returning stale arrays.
        """
        parts = (
            f"ply={ply_name}",
            f"traj={self.trajectory_file}",
            f"reg={self.registration_txt}",
            f"targets={self.targets_csv}",
            f"orient_idx={self.domain_end_target_idx}",
            f"traj_resample={self.centreline_traj_resample_ds_m}",
            f"smoothing={self.centreline_smoothing_factor}",
            f"cl_resample={self.centreline_resample_ds_m}",
            f"theta_ref={self.theta_reference}",
            f"up={tuple(float(v) for v in self.theta_up_vector)}",
        )
        return "|".join(parts)
