# Ice tunnel volume estimation — `volume/`

Estimates the air volume of an englacial ice tunnel from two point clouds — a
Leica RTC360 reference scan and a Livox Mid-360 SLAM (GLIM) mobile scan — by
three independent-ish methods, and states the uncertainty of the result from the
**spread of repeated runs** rather than from any error model.

Everything runs through two commands.

```bash
PY=~/.venvs/slam_sweep/bin/python

# 1 — compute volumes.  One result directory per run; nothing is overwritten.
$PY run_pipeline.py --run-real --dataset April_12_05_05 --rep all

# 2 — turn those runs into the error estimate.
$PY run_statistics.py --dataset April_12_05_05
```

---

## 1. `run_pipeline.py` — compute the volume of one run

```bash
# every rep of a campaign (sequential; the Leica reference is processed once)
$PY run_pipeline.py --run-real --dataset April_12_05_05 --rep all

# a single rep
$PY run_pipeline.py --run-real --dataset July_20_04_29 --rep rep02

# validate the methods against a cylinder of known volume — no data needed
$PY run_pipeline.py --phantom
```

`--dataset/--rep` discovers all six input files (see **Data layout**), so no path
flags are needed. Each run writes

```
results/<dataset>/<rep>/<UTC timestamp>/
    summary.json          volumes + domain + coverage + config + input provenance
    run.log               every log line of that run, warnings included
    figures/*.png         the diagnostic figures for that run
    surface_mesh_*.ply    watertight mesh, for a Cloud-to-Mesh check in CloudCompare
    profile_cloud_*.ply   the wall the profiles method integrated (see below)
```

### `profile_cloud_*.ply` — what the profiles method actually integrated

The profiles estimator does not integrate the raw scan. It integrates a filled
`r(s, θ)` grid: one median radius per 0.25 m × 1° cell, with empty cells
interpolated — circularly around θ first, then along `s` for slabs too sparse to
close on their own. That grid *is* the surface the volume is a property of, and
it used to exist only inside the integral.

Each run now writes it as a binary PLY, one point per grid cell (~192k points,
4.6 MB — three orders below the input clouds, so it opens instantly next to
them). Four per-point scalar fields travel with it:

| field | meaning |
|---|---|
| `interpolated` | **1 = the wall here was invented by the hole fill, 0 = median of real returns** |
| `r` | the radius that cell contributed [m] |
| `s` | chainage along the centreline [m] |
| `theta_deg` | azimuth, 0 = ceiling, ±180 = floor |

In CloudCompare: open the file, and when asked map the extra properties to
scalar fields. Colour by `interpolated` and you are looking directly at which
metres of tunnel the volume is trusting the interpolation for — on the April
Livox walks that is 34.0% of the cells, on the Leica reference 12.5%, and those
two numbers are the `frac_interp_mean` in `summary.json`, not an approximation
of it. `--no-profile-cloud` skips the export.

The file is written with a hand-rolled PLY writer (`io_utils.save_ply_scalars`)
rather than through open3d, because open3d's writer keeps only x/y/z and
silently drops every extra property — which is the entire point of this export.
Coordinates are float32: ~1e-5 m of quantisation at 100 m from the origin, four
orders below the measurement.

The timestamp is what makes runs accumulate instead of overwrite: run the same
rep again and you get a second directory, and the statistics command decides
which to use (newest per rep, by default).

**Key flags**

| flag | effect |
|---|---|
| `--rep all` \| `rep00` \| `rep00,rep03` | which reps to run; a failing rep is logged and the sweep continues |
| `--cloud auto` (default) | Livox for every rep, plus the Leica reference **once per dataset** — the Leica scan is the same cloud for all reps, so re-processing it per rep only re-measures the centreline |
| `--cloud both\|livox\|leica` | force it |
| `--no-hull`, `--no-mesh` | skip a method (hull is the slow one) |
| `--no-profile-cloud` | skip the interpolated-wall PLY export |
| `--cap-mode auto\|target_planes\|feature_planes` | where the domain end caps come from (see **Domain end caps**) |
| `--caps-file …` | explicit cap file, instead of the dataset's `caps.txt` |
| `--mc` | also run marching cubes — **off by default**, it leaks on these clouds (see below) |
| `--no-cache` / `--no-cache-save` | ignore / don't write the `(s, r, θ)` cache (~0.3–1 GB per cloud) |
| `--tag "why I ran this"` | free text stored in `summary.json` |
| `--leica-file …` etc. | explicit paths, for data that doesn't follow the `<dataset>/<rep>/` layout |

### Re-measuring the Leica reference in every rep

```bash
$PY run_pipeline.py --run-real --dataset April_12_05_05 --rep all --cloud both
```

By default (`--cloud auto`) the Leica reference is processed **once** per
dataset, because it is the same cloud for every rep and re-processing it only
re-measures the centreline. `--cloud both` does exactly that re-measuring on
purpose, and it answers a question the Livox reps cannot: **the reference cloud
is identical in all five runs, and only the trajectory — hence the centreline
fitted from it — differs, so the spread of those five Leica volumes is what the
processing contributes on its own.**

Read it as a **floor under the Livox SD, not as a rival estimate**:

- Livox SD ≫ Leica SD → the scatter is the walk and the coverage, which is what
  the April/July split already says (35.2% vs 3.0% interpolated area).
- Livox SD ≈ Leica SD → you are measuring the pipeline, not the survey.

It is *not* a determinism test. The pipeline is deterministic: feed the same
cloud and the same trajectory twice and the volumes agree to the last digit.
Five runs that vary nothing would produce an SD of exactly zero and tell you
nothing about the measurement.

The cost is real — each rep must compute the Leica `(s, r, θ)` coordinates
against its own centreline (~20 M points), so a five-rep `--cloud both` sweep is
substantially slower than the Livox-only one and roughly doubles the cache.

## 2. `run_statistics.py` — the error estimate

```bash
$PY run_statistics.py                            # every dataset under results/
$PY run_statistics.py --dataset April_12_05_05   # one campaign
$PY run_statistics.py --all                      # every saved run, not newest-per-rep
```

Writes `results/<dataset>/statistics/<UTC timestamp>/` containing
`statistics.md` (the report, also printed), `summary.csv`, `runs.csv`,
`statistics.json` and `volume_statistics.png`.

**What it reports, and why in that shape**

- **Repeatability (random, 1σ)** — the SD across reps, per cloud and per method:
  `n`, mean, SD, CV%, the 95% CI of the mean (Student-t), min–max, and the 95%
  CI **of the SD itself** (χ²). That last column matters: a 5-sample SD is only
  known to about ±35%, so the SD is worth two significant figures and no more.
  A Grubbs test flags a single gross outlier (a rep processed from the wrong
  file) — it is reported, never used to silently drop a rep.
- **Bias vs the Leica reference (systematic)** — the mean Livox−Leica offset per
  method. It does **not** shrink with more reps: walking the tunnel again cannot
  recover a wall the Mid-360 never saw. It is printed next to the SD and
  **never added in quadrature with it** — one is scatter you can average down,
  the other is a fixed offset that averaging cannot touch.
- **Leica repeatability**, when the campaign was run with `--cloud both` — its
  own row in the table, its own headline line, and a hatched bar next to each
  Livox bar in the figure. Plus a **paired Livox−Leica table**: both volumes in
  each of those differences came from the same run and therefore the same
  centreline, so the centreline's contribution cancels and what is left is the
  difference between the two clouds. If the paired SD is smaller than the
  unpaired one, the centreline was a shared source of scatter; if they agree, it
  was not.
- **Between-method spread within a run** — surface mesh vs profiles. Reported
  separately again, because those two share the `r(s, θ)` extraction and differ
  only in integration and hole-filling, so their gap measures hole-fill
  sensitivity, not independent agreement.
- **Run-to-run stability of the inputs** — domain length `L`, centreline fit RMS
  and max curvature, target scale `k`, Livox coverage and interpolated fraction.
  Volume scales with `L`, so if `L`'s CV is comparable to the volume's CV, the
  scatter is length, not shape.

Datasets are never pooled: April and July are different tunnel states, so
merging them would report seasonal change as measurement noise.

Runs that were processed with different settings (different `cap_mode`, binning,
centreline parameters, or a different Leica scan) are **refused**, not averaged;
pass `--allow-mixed-config` to override and the warning is carried into the
report.

### Why the spread of runs, and not an error budget

An earlier version modelled four error terms per cloud per method (sensor noise,
discretisation, length scale, coverage) and ran a Monte-Carlo over them. It has
been removed. A modelled budget can only contain the errors someone thought of,
and it was structurally blind to anything the methods get wrong in common —
including an error in the shared `r(s, θ)` extraction, which profiles and
surface mesh would both inherit silently. Repeated real runs contain every error
that varies from run to run, whether or not anyone modelled it. What repetition
*cannot* see is an error common to every rep — which is exactly what the Leica
bias column and the between-method spread are there for.

---

## Current results (2026-08-21, 5 reps per campaign, **feature-capped domain**)

Livox volume ± the measured run-to-run repeatability, and — separately — the
systematic offset from that campaign's Leica reference. Both campaigns are
capped on the same two shared features, so **these two rows are finally volumes
of the same stretch of tunnel** (April L = 135.322 m, July L = 135.292 m,
agreeing to 30 mm):

| campaign | method | Livox V ± 1σ [m³] | CV | Leica V ± 1σ [m³] | CV | offset |
|---|---|---|---|---|---|---|
| **April_12_05_05** (L = 135.322 m) | profiles | 1014.8 ± 13.7 | 1.35% | 1132.11 ± 0.75 | 0.067% | −10.36% |
| | surface mesh | 1027.7 ± 11.8 | 1.15% | 1126.68 ± 0.65 | 0.058% | −8.79% |
| | convex hull (UB) | 1118.9 ± 33.3 | 2.98% | 1245.11 ± 0.25 | 0.020% | −10.14% |
| **July_20_04_29** (L = 135.292 m) | profiles | 1148.3 ± 1.6 | 0.14% | 1117.20 ± 0.36 | 0.032% | +2.78% |
| | surface mesh | 1150.9 ± 1.6 | 0.14% | 1115.46 ± 0.49 | 0.044% | +3.17% |
| | convex hull (UB) | 1319.5 ± 4.5 | 0.34% | 1226.24 ± 0.49 | 0.040% | +7.61% |

**n = 5 for every cell** (2026-08-26 sweep, `--rep all --cloud both`): five Livox
walks, and the one Leica scan re-measured through each of those five reps'
centrelines. The offset column is the *paired* Livox−Leica difference, both
volumes from the same run and therefore the same centreline.

**The Leica campaign comparison is the one that changed.** On the target-capped
domain July read **+2.81% larger** than April; on the common feature-capped
domain it reads **−1.25% smaller** (profiles; −0.95% mesh, −1.54% hull). The old
comparison was dominated by the domain mismatch, not by the tunnel: April's
target-capped domain was 2.33 m shorter and started 2.15 m further inside the
wide entrance chamber. April's volume rose 38.9 m³ when the domain grew 1.95 m —
20 m² per metre, which is the ~24 m² entrance chamber and not the ~8 m² main
tunnel, and is why normalising by length (Ā) could never have rescued it.

**The processing floor** is the Leica column above: one fixed cloud through five
different centrelines, CV 0.02–0.07%. That is 20× below April's Livox scatter
and 3–4× below July's, so the Livox repeatability is the walk and the coverage,
not the centreline fit — measured, not assumed.

### July − April, Leica vs Leica

| method | April | July | Δ [m³] | Δ [%] | σ_proc | Δ/σ_proc |
|---|---|---|---|---|---|---|
| profiles | 1132.11 ± 0.75 | 1117.20 ± 0.36 | **−14.91** | **−1.32%** | 0.83 | 17.9 |
| surface mesh | 1126.68 ± 0.65 | 1115.46 ± 0.49 | −11.22 | −1.00% | 0.82 | 13.7 |
| convex hull (UB) | 1245.11 ± 0.25 | 1226.24 ± 0.49 | −18.87 | −1.52% | 0.55 | 34.2 |

**Read σ_proc correctly.** It is the two campaigns' processing scatter added in
quadrature, so Δ/σ_proc says only this: *the processing cannot have produced the
−14.9 m³.* It is **not** the uncertainty of the difference as a measurement of
the tunnel. Each campaign's Leica scan carries its own coverage and hole-fill
bias, which is identical in all five of its runs and therefore invisible to this
SD — and the two scans do differ there (13.0% vs 10.8% interpolated area), in
the direction that inflates April. Quoting −14.91 ± 0.83 m³ as the volume change
would overstate what has been established. Domain length contributes 30 mm of
135 m, i.e. ~0.25 m³, so it is not length either.

### Where the April→July change sits (`oneoff_campaign_compare.py`)

Both feature-capped domains start at the same physical feature, so
`s' = s − s_start` is a common chainage and the two campaigns' `A(s')` curves
can be differenced slab for slab even though their xyz frames cannot be
compared at all. Leica vs Leica:

- **The change is distributed, not localised.** All 14 ten-metre segments are
  negative (−0.27% to −3.57%), mean area falls 8.367 → 8.260 m² (−1.28%), and
  the cumulative ΔV curve is a near-straight ramp to −14.5 m³ rather than a
  step. A domain slip, a bad cap or a registration error would concentrate
  somewhere — most likely at one end or in the entrance chamber. This does not.
- **The entrance-chamber peak (~25 m² at s' ≈ 3 m) lines up in both campaigns**,
  which is independent confirmation that the caps are on the same feature —
  stronger evidence than the 30 mm agreement of the two domain lengths, because
  it matches a *shape* and not just a distance.

Figure: `figures/13_campaign_compare.png`.

### Livox vs Livox, for contrast (`--cloud both`, figure 14)

The same difference taken on the mobile scans, and why it must not be quoted as
a tunnel change. Livox reports **+133.5 m³ (+13.2%)** where Leica reports −14.9.
The identity that explains it — exact to the last digit, since offset is defined
as V_livox − V_leica within a campaign:

    ΔV_livox  =  ΔV_leica  +  Δ(offset)
     +133.47   =    −14.91   +   +148.38     (profiles; mesh and hull the same
                                              shape, tunnel share 8% each)

**Only 9% of the magnitude is the tunnel; 91% is the change in how much wall the
Mid-360 failed to see**, and the two terms have opposite signs, so Livox reports
a rise where the tunnel fell. Localised: **74% of that sensor term sits in the
first 10 m** (+92.5 of +125.8 m³ on the rep04 pair), the entrance chamber, where
April's Livox interpolates 60–75% of its area and July's under 10%. Panel C of
figure 14 plots exactly that.

This is the strongest available argument for keeping a tripod reference in a
repeat survey: five walks reproduce each campaign's Livox volume to 1.35% and
0.14%, and the campaign-to-campaign change is still wrong by an order of
magnitude and the wrong sign. Repeatability was never the problem.

### The cache does not enter any of this

Audited 2026-08-21. The centreline is **never cached** — `fit_centreline` runs
unconditionally on every run, before the cache block — and only the `(s, r, θ)`
arrays are stored, under a signature covering the trajectory, the registration
matrix and every centreline parameter. Verified by recomputing a rep per
campaign with `--no-cache`: April rep02 and July rep01 reproduce the cached
run's domain length, centreline length, fit RMS, κ_max and all three volumes to
**every printed digit** (difference exactly 0.0). Caps are deliberately absent
from the signature: they change the integration domain, not the coordinates, so
reusing coordinates across a cap change is correct.

**Repeatability tracks coverage, and nothing else here comes close.** July's
Livox walk interpolates 3.0% of its cross-sectional area; April's interpolates
35.2% — an 11× difference, and the repeatability differs 10× the same way
(0.13% vs 1.35%). Meanwhile the domain length repeats to 0.01% in both
campaigns and the target scale `k` to 0.05%, so the scatter is not length and
not registration: it is how much of the wall each walk happened to see. That is
a conclusion the old modelled budget could assert only by assumption; here it
falls out of the runs.

The sign of the Leica offset flips between campaigns (April −8%, July +3%),
which is why it is reported as a per-campaign systematic and never folded into
a single σ. April's Livox walk misses a third of its wall area (the Mid-360 FOV
bands) and under-reads; July's sees almost everything and reads slightly *large*
— consistent with the ~+27 mm radial bias diagnosed in `day4_findings.md`.

Reproduce with (the cap files must be in place — see **Domain end caps**):

```bash
$PY run_pipeline.py --run-real --dataset April_12_05_05 --rep all
$PY run_pipeline.py --run-real --dataset July_20_04_29  --rep all
$PY run_statistics.py
```

## Data layout

A dataset is one acquisition campaign = one Leica reference scan + N repeated
Livox walks. `data/` is not in git.

```
data/April_12_05_05/
    rtc_clean_1cm.ply        Leica reference cloud, ~1 cm, cleaned/cropped   (shared)
    targets_leica.txt        targets picked in the Leica cloud               (shared)
    caps.txt                 OPTIONAL: two shared-feature domain caps        (shared)
    rep00/ … rep04/
        *_clean_and_regist.ply    Livox/GLIM cloud, ~1 cm, cleaned, ALREADY registered
                                  into the Leica datum (one PLY per rep folder)
        trajectory.txt            GLIM trajectory, TUM (t x y z qx qy qz qw)
        transformation_matrix.txt 4×4 Livox → Leica
        targets_livox.txt         the same physical targets, picked in THIS Livox cloud
```

Notes on the formats, each of which has cost someone an afternoon:

- **The trajectory must NOT be pre-transformed.** Feed in whatever GLIM wrote.
  The clouds are registered once, externally (hundreds of millions of points);
  the trajectory is registered by the pipeline on every run (a few thousand
  poses). Pre-transforming it applies the matrix twice and
  `spine.check_cloud_aligned` then raises about the *cloud* — read that error as
  "the centreline and the cloud disagree", not "the cloud is wrong". (The legacy
  name `trajectory_transformed.txt` is a misnomer; its contents are raw poses.)
- **PLY may be ASCII or binary** — `open3d.io.read_point_cloud` handles both
  identically, verified against CloudCompare's property layout. Prefer binary:
  ~2.5× smaller and faster to parse. A `_ascii` suffix is a naming convention
  here, not a requirement. The genuinely text-only inputs are the trajectory,
  the targets and the matrix (all `np.loadtxt`).
- Targets are `id, x, y, z`, comma- or whitespace-separated, header optional.
  `data/`'s **top-level** `targets_*.txt` are retired — some coordinates there
  are placeholders, so no distance, length or scale statement may come from
  them. Only the per-dataset files above are valid.

## Domain end caps — and why April and July are not comparable without them

A volume here is the air between two cap planes. Which two, is `cap_mode`.

**`target_planes` (default).** Caps at the surveyed targets `#0` and `#7`.
Correct *within* one campaign: every rep is then measuring the same stretch of
tunnel, which is exactly what the repeatability statistics require.

**It does not survive the jump to another campaign.** The targets were re-placed
between April and July, so `#0` and `#7` are not the same physical points in the
two datasets, and the two volumes are volumes of *different stretches of
tunnel*. Their difference is then part tunnel change and part domain change,
with no way to separate the two. The datasets do not even share a coordinate
frame — the April target line runs along ≈(0.248, −0.962, 0.120) and the July
one along ≈(0.888, −0.451, 0.125), the same inclined tunnel in two datums
rotated ~48° apart about vertical — so nothing cross-campaign is computable
until a shared physical anchor exists.

**`feature_planes` — the fix.** Pick one physical feature near each end of the
tunnel that is identifiable in **both** campaigns' reference scans, and cap
there instead:

```
data/April_12_05_05/caps.txt        data/July_20_04_29/caps.txt
1, 1.837, -14.894, 0.359            1, 28.114, -14.002, 3.9
2, 25.259, -106.515, 12.140         2, 111.902, -57.201, 15.1
```

Same format as the targets (`id, x, y, z`, comma or whitespace, header
optional), **exactly two rows**, each in its own campaign's datum. Drop the file
into the dataset directory and it is picked up automatically; `cap_mode` flips
to `feature_planes` and every run records that in `summary.json`.

Why this is enough, and why it needs no cross-campaign transform: each campaign
projects **its own** picks onto **its own** centreline and takes the arclength
interval between them. The domain is then "the tunnel between these two physical
places" in both datasets — which is the thing that has to be held fixed for a
difference of volumes to mean anything. Being an arclength interval, it also
stays correct for a curved tunnel; a straight-line chord would not.

**Picking the features (CloudCompare).** Use the *Leica* cloud of each campaign,
not the Livox one — it is the cleaner scan and it is the shared reference.
Choose something small, hard and man-made where possible: a bolt head, a cable
bracket, a door-frame corner, the lip of a niche. Avoid anything on the ice
itself, which moves and melts between campaigns. Pick as near the two ends of
the tunnel as a *shared* feature allows — the domain can only ever be the part
both campaigns can name. Point-picking repeatability is ~10 mm, negligible
against a ~100 m domain, so precision of the pick is not the constraint;
picking the *same* feature is.

Two guards, because a silently wrong domain is worse than a crash: a cap file
that does not hold exactly two points is refused, and a cap lying more than
`cfg.cap_max_offset_m` (5 m) off the centreline is refused as a mis-pick or a
file in the wrong campaign's datum. Every run logs what the feature caps moved
relative to the target-capped domain:

```
Domain caps: shared features from caps.txt (cap_mode=feature_planes).
s=[30.949, 126.251] L=95.302 m — target-capped domain would have been
s=[9.382, 142.751] L=133.369 m (start +21.566 m, end -16.500 m).
```

`cap_mode` is in `run_statistics`' `MUST_MATCH_CONFIG`, so a target-capped run
can never be silently averaged in with a feature-capped one. **Check that both
campaigns report the same feature-capped `L`** — that is the test that the caps
landed on the same features, and it is worth doing before believing any
April-vs-July difference.

Force either mode with `--cap-mode`; forcing `target_planes` while a cap file
exists is allowed and warns that the run is not cross-campaign comparable.

## Methods

| method | module | what it is |
|---|---|---|
| **profiles** | `cross_sections.py` | core estimator: slab the domain by Δs, bin azimuth 1°, median `r` per bin, fill gaps (circular then along-`s`), polar shoelace area, integrate (trapezoid + Simpson) |
| **surface mesh** | `surface_mesh.py` | rasterise `(s, θ)` → median-`r` grid, fill holes by 2-D periodic interpolation, wrap onto the centreline frame, cap both ends, volume by the divergence theorem |
| **convex hull** | `cross_sections.py` | same slab loop, 2-D convex-hull area — an **upper bound**, since it spans concavities (and *under*-reads where data is missing, so it bounds only where coverage is good) |
| marching cubes | `marching_cubes.py` | trajectory-independent voxel volume. **Off by default** (`--mc`): it needs a sealed wall shell, and both clouds' holes (Leica scan shadows, Livox FOV bands) are wider than the 1-voxel seal, so the flood-fill escapes and it reports `LEAK` rather than a wrong number. It validated the other three on the phantom (+0.26%) and its leaking is itself a measure of how much the real data depends on hole-filling |

Everything shares one **spine** (`spine.py`): a smooth-spline + rotation-minimizing-frame
centreline fitted to the trajectory, and cylindrical coordinates `(s, r, θ)` per
point. `θ` is gravity-referenced by default — **θ=0 is the ceiling, ±180 the
floor, ±90 the side walls** — so a coverage gap can be named physically instead
of as "θ≈+37°".

⚠️ **profiles and surface mesh are not independent.** They share `build_r_grid`,
so their agreement validates the *integration* and the *hole-filling*, not the
extraction; an extraction error is common-mode and cancels silently. The
independent check is marching cubes, which never builds an `r(s, θ)` grid.

**Validate before trusting:** `run_pipeline.py --phantom` synthesises a cylinder
of known volume with a handheld-like trajectory and asserts every method
recovers it — profiles and surface mesh land at −0.008%, hull at −0.001%,
marching cubes at +0.26%, centreline axis within 0.01°.

## Files

| file | role |
|---|---|
| `run_pipeline.py` | **command 1** — one volume run, one result directory |
| `run_statistics.py` | **command 2** — across-run statistics, the error estimate |
| `dataset.py` | `<dataset>/<rep>/` layout → the six input files; strict discovery |
| `config.py` | single `Config` dataclass: every path, bin size, threshold, mode switch |
| `io_utils.py` | PLY / TUM / targets / matrix loaders, rigid-registration check, `(s,r,θ)` npz cache |
| `spine.py` | centreline, cylindrical coordinates, θ frame, gap map, alignment guard |
| `cross_sections.py` | profiles + hull bound |
| `surface_mesh.py` | surface-mesh volume |
| `marching_cubes.py` | voxel volume (opt-in) |
| `phantom.py` | known-volume validation harness |
| `profile_cloud.py` | exports the filled `r(s, θ)` wall the profiles method integrated |
| `decomposition.py` | V = Ā×L split (shape vs length), target scale `k`, target consistency |
| `slice_compare.py` | where along `s` the two clouds differ |
| `lateral_compare.py` | where *around* the cross-section they differ |
| `diagnostics/` | standalone reproductions of the day-3 findings |
| `oneoff_campaign_compare.py` | April vs July, Leica vs Leica, on the shared feature-capped domain |
| `oneoff_*.py` | one-off scripts, not wired into the pipeline (see day3_findings §11) |

Per-run figures (in each result directory): `01_centreline`,
`01b_theta_reference`, `02_radius_hist_*`, `03_gap_map_*`, `04_area_profile_*`,
`05_cross_sections_*`, `05b_cross_sections_combined`, `06_cumulative_distance`,
`07_volume_slices`, `08_lateral_compare`, `12_target_3d_distances`. The
statistics command adds `volume_statistics.png`.

`cache/<dataset>/<rep>/*.npz` holds the `(s, r, θ)` arrays, stamped with a
signature of every setting that affects them; a stale cache is recomputed rather
than silently reused.

## Known limitations

- **Livox coverage is FOV-limited for a scratched sensor.** The Mid-360's vertical
  FOV (−7°…+52°) leaves two lower flanks and a ceiling band systematically thin,
  at fixed azimuth, along the whole domain — median slab coverage ~71% vs ~97%
  for Leica. So ~a third of the Livox area is interpolated, no both-clouds
  "golden segment" of near-complete data exists at any threshold, and the
  Livox−Leica gap is mostly **shape, not length** (the target scale error is only
  ~0.2%). 
- Marching cubes leaks on real clouds (above), so the fully independent
  cross-check is available on the phantom only.
- `geometry_mode="volumetric"` (cave/chamber, no single centreline) is supported
  by marching cubes but has no driver in `run_pipeline.py` yet.

Dependencies: numpy 2.x, scipy, open3d, matplotlib, scikit-image (marching cubes
only). Install into the venv with `~/.venvs/slam_sweep/bin/pip install`.