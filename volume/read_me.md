# Ice Tunnel Volume Estimation — `volume/`

A modular pipeline for calculating the air volume of englacial ice tunnels from 3D point clouds. The package supports both high-density static terrestrial laser scanning (TLS, e.g., Leica RTC360) and mobile LiDAR SLAM point clouds (e.g., handheld Livox Mid-360 with GLIM).

Uncertainty is evaluated empirically from the **spread across repeated survey runs** and direct comparison against a static reference scan, rather than through an idealized synthetic error model.

---

## 1. Overview & Workflow

The pipeline operates in two sequential stages:

```bash
PY=~/.venvs/slam_sweep/bin/python

# Step 1 — Compute volumes for individual runs (writes isolated timestamped run directories)
$PY run_pipeline.py --run-real --dataset <dataset_name> --rep all

# Step 2 — Aggregate runs into statistical metrics and repeatability estimates
$PY run_statistics.py --dataset <dataset_name>
```

```
           ┌────────────────────────────────────────┐
           │ Input Point Clouds + Trajectory Trait  │
           │  • Static Reference (e.g., Leica TLS)  │
           │  • Mobile Scan (Livox + GLIM Traj)     │
           └───────────────────┬────────────────────┘
                               │
                               ▼
           ┌────────────────────────────────────────┐
           │           `run_pipeline.py`            │
           │  1. Spine & Cylindrical Frame (s,r,θ)  │
           │  2. Domain Capping (Target / Feature)  │
           │  3. Estimation (Profiles, Mesh, Hull)  │
           └───────────────────┬────────────────────┘
                               │
                               ▼
           ┌────────────────────────────────────────┐
           │           `run_statistics.py`          │
           │  • Run-to-run repeatability (1σ)       │
           │  • Systematic bias vs. Reference scan  │
           │  • Cross-method hole-fill sensitivity  │
           └────────────────────────────────────────┘
```

---

## 2. Running the Volume Pipeline (`run_pipeline.py`)

### Usage Examples

```bash
# Process all repetitions in a campaign (Reference processed once by default)
$PY run_pipeline.py --run-real --dataset <dataset_name> --rep all

# Process a single mobile repetition
$PY run_pipeline.py --run-real --dataset <dataset_name> --rep rep01

# Re-measure the reference cloud against each individual repetition trajectory
$PY run_pipeline.py --run-real --dataset <dataset_name> --rep all --cloud both

# Validation mode: test estimators against an analytical synthetic cylinder phantom
$PY run_pipeline.py --phantom
```

### Run Output Structure

Each execution creates an isolated, non-overwriting timestamped directory:

```
results/<dataset>/<rep>/<UTC timestamp>/
    summary.json          Calculated volumes, domain length, coverage stats, input provenance
    run.log               Execution log and runtime warnings
    figures/*.png         Per-run cross-section, gap map, and diagnostic plots
    surface_mesh_*.ply    Watertight reconstructed 3D surface mesh (for CloudCompare C2M checks)
    profile_cloud_*.ply   The interpolated cylindrical grid integrated by the profile estimator
```

### Visualizing Integrated Profiles (`profile_cloud_*.ply`)

The profile integration method does not integrate raw point coordinates directly; it integrates an interpolated cylindrical grid r(s, θ) containing median radial returns per chainage (s) and azimuth (θ) cell.

The output `profile_cloud_*.ply` exports this surface as a binary point cloud containing embedded per-point scalar fields:

| Scalar Field | Description |
|---|---|
| `interpolated` | Binary flag: `0` = median of raw returns; `1` = synthetic geometry filled across coverage voids |
| `r` | Radial distance from the tunnel spine/centreline (meters) |
| `s` | Chainage along the fitted centreline spline (meters) |
| `theta_deg` | Gravity-referenced azimuth angle (0° = ceiling, ±90° = walls, ±180° = floor) |

> **Inspection Tip:** Open `profile_cloud_*.ply` in CloudCompare, map the scalar fields, and color by `interpolated` to visually audit where data gaps forced radial interpolation.

### CLI Flag Reference

| Flag | Default | Description |
|---|---|---|
| `--rep <val>` | `all` | Specific repetition folder (`rep00`, `rep01,rep02`, or `all`). |
| `--cloud <mode>` | `auto` | `auto`: Processes mobile SLAM per rep and reference scan once per dataset.<br>`both`: Forces re-evaluation of reference cloud against each rep's spine.<br>`livox` / `leica`: Process mobile or reference cloud only. |
| `--cap-mode <mode>` | `auto` | Domain boundary strategy: `auto`, `target_planes`, or `feature_planes`. |
| `--caps-file <path>` | Auto-detected | Explicit path to a domain cap coordinate file. |
| `--no-hull`, `--no-mesh` | `False` | Skips convex hull or watertight meshing calculations. |
| `--no-profile-cloud` | `False` | Disables export of the `profile_cloud_*.ply` artifact. |
| `--mc` | `False` | Enables volumetric Marching Cubes (requires sealed surface point data). |
| `--no-cache` | `False` | Ignores cached (s, r, θ) spatial transformation arrays. |
| `--tag "<text>"` | None | User metadata string attached to `summary.json`. |

---

## 3. Statistical Analysis & Error Estimation (`run_statistics.py`)

Rather than relying on theoretical error propagation, `run_statistics.py` calculates empirical random and systematic error across multiple survey passes:

```bash
# Analyze all datasets found in results/
$PY run_statistics.py

# Analyze a specific acquisition campaign
$PY run_statistics.py --dataset <dataset_name>

# Aggregate all historical runs instead of only the newest timestamp per rep
$PY run_statistics.py --all
```

Outputs are written to `results/<dataset>/statistics/<UTC timestamp>/` containing `statistics.md`, `summary.csv`, `statistics.json`, and `volume_statistics.png`.

### Reported Metrics

* **Repeatability (Random Error, 1σ):** Standard deviation across mobile repetitions for each method (Profiles, Mesh, Convex Hull), along with 95% confidence intervals for the mean and the standard deviation (χ²). Single gross outlier identification is checked via Grubbs' test without silent dropping.
* **Systematic Bias vs. Reference:** Mean offset (ΔV = V_mobile - V_ref). Evaluated separately from random standard deviation because systematic coverage deficits cannot be averaged out by repeated passes.
* **Processing Uncertainty Floor (`--cloud both`):** Measures the standard deviation of the *static reference cloud* re-evaluated across varying mobile SLAM centrelines. 
  * If σ_mobile ≫ σ_ref_floor, run-to-run scatter is driven by walking path and sensor coverage.
  * If σ_mobile ≈ σ_ref_floor, run-to-run scatter is limited by trajectory/spine variation.
* **Hole-Fill Sensitivity:** Spread between Profile and Surface Mesh methods, which share extraction grids but differ in spatial interpolation mechanics.
* **Geometric Stability Metrics:** Variation in domain length (L), spline curvature (κ_max), and target scale calibration (k).

---

## 4. Domain End Caps & Multi-Campaign Comparisons

Volume computation is strictly bounded between two transverse planes intersecting the centreline. The pipeline supports two capping mechanisms:

```
[Entrance Cap Plane] =================== Tunnel Domain (L) =================== [Terminal Cap Plane]
         │                                                                             │
         ├─ Mode: target_planes  --> Anchored to surveyed targets (e.g., #0 to #7)    ┤
         └─ Mode: feature_planes --> Anchored to persistent natural/structural features ┘
```

### Cap Modes

1. **`target_planes` (Default):**
   * Projects surveyed target markers onto the centreline spline.
   * Ensures identical domain bounds for all mobile runs within a single measurement session.
   * **Limitation:** Inapplicable for direct cross-campaign comparisons if physical targets were moved or surveyed in independent local coordinates.
2. **`feature_planes` (Cross-Campaign Longitudinal Tracking):**
   * Uses two physically identifiable, immutable features picked at opposite ends of the tunnel (e.g., rock bolts, structural frames).
   * Enabled automatically when `caps.txt` is present in the dataset directory.
   * Each campaign defines the identical physical feature in its own local coordinate system. The pipeline projects these picks to the respective centreline arclength (s), creating an invariant arclength interval domain across surveys without requiring rigid coordinate registration between campaigns.

### Cap File Format (`caps.txt`)

Place a 2-line ASCII file in the campaign root directory:

```
id, x, y, z
1,  1.837, -14.894,  0.359
2, 25.259, -106.515, 12.140
```

---

## 5. Volume Estimation Methods

```
                                  Trajectory & Cloud
                                          │
                                          ▼
                                ┌───────────────────┐
                                │ Spine Extraction  │
                                │   (s, r, θ frame) │
                                └─────────┬─────────┘
                                          │
         ┌────────────────────────────────┼────────────────────────────────┐
         ▼                                ▼                                ▼
┌───────────────────┐            ┌───────────────────┐            ┌───────────────────┐
│     Profiles      │            │   Surface Mesh    │            │    Convex Hull    │
│ (cross_sections)  │            │  (surface_mesh)   │            │ (cross_sections)  │
├───────────────────┤            ├───────────────────┤            ├───────────────────┤
│ • Sliced by Δs    │            │ • 2D periodic     │            │ • Sliced by Δs    │
│ • Azimuth binned  │            │   r(s,θ) grid     │            │ • Convex boundary │
│ • 1D hole fill    │            │ • 3D mesh wrapped │            │   per slice       │
│ • Shoelace area + │            │ • Divergence      │            │ • Theoretical     │
│   integration     │            │   theorem volume  │            │   upper bound     │
└───────────────────┘            └───────────────────┘            └───────────────────┘
```

* **Profiles (`cross_sections.py`):** Slices the tunnel along chainage s into cross-sectional slabs (Δs), bins azimuth (1°) to compute median radii r, fills polar gaps circularly and longitudinally, calculates closed polygon area via the polar shoelace formula, and integrates along s (Trapezoidal / Simpson rules).
* **Surface Mesh (`surface_mesh.py`):** Constructs a 2D periodic (s, θ) grid with 2D hole-filling interpolation, wraps the reconstructed surface to the 3D curvilinear spine frame, closes planar end caps, and integrates volume via the divergence theorem.
* **Convex Hull (`cross_sections.py`):** Calculates the 2D convex hull cross-sectional area per slice. Functions as an upper bound where point coverage is continuous.
* **Marching Cubes (`marching_cubes.py`, Opt-in `--mc`):** Discretizes point coordinates into an occupancy voxel grid. Independent of the centreline spine; requires dense, watertight surface returns to prevent flood-fill leakage.

### Coordinate Spine Frame (`spine.py`)

Points are converted from Cartesian (x, y, z) to curvilinear cylindrical coordinates (s, r, θ):
* s: Arclength along the spline centreline fitted from the trajectory.
* r: Orthogonal distance from the centreline.
* θ: Gravity-referenced azimuthal angle (0° = ceiling, ± 90° = side walls, ± 180° = floor).

---

## 6. Sensor Characteristics: Reference Scan vs. Mobile SLAM

Understanding the systematic variance between scanning systems is critical for interpretation:

```
Static Reference (e.g., Leica RTC360)         Mobile SLAM (e.g., Livox Mid-360)
┌───────────────────────────────────────┐    ┌───────────────────────────────────────┐
│ • High point density, stationary setups│    │ • Continuous handheld/mobile walk     │
│ • High azimuthal coverage (low shadow)│    │ • Prone to sensor FOV blind spots     │
│ • Minimal interpolation required      │    │ • Trajectory-dependent coverage gaps  │
│ • Processing benchmark (baseline)     │    │ • Requires hole-filling interpolation │
└───────────────────────────────────────┘    └───────────────────────────────────────┘
```

1. **Coverage Discrepancies:** Limited vertical field-of-view (FOV) on mobile scanners can leave longitudinal bands along ceilings or lower side walls sparsely sampled, requiring higher fractions of geometric interpolation than static multi-station reference scans.
2. **Spatial Drift vs. Shape:** Trajectory registration drift affects total domain arclength (L), whereas sensor occlusion affects cross-sectional shape and median area (Ā). The pipeline decomposes volume changes (V ≈ Ā × L) to verify whether variations are driven by path length or wall occlusion.

---

## 7. Data Layout & Requirements

Input data must follow this hierarchy under `data/`:

```
data/<dataset_name>/
    rtc_clean_1cm.ply             # Reference point cloud (clean, subsampled ~1cm)
    targets_leica.txt             # Target picks in reference datum
    caps.txt                      # (Optional) Persistent feature domain picks
    rep00/ ... repNN/             # Repetition folders
        *_clean_and_regist.ply    # Mobile point cloud, pre-registered to reference datum
        trajectory.txt            # Raw SLAM trajectory in TUM format (t x y z qx qy qz qw)
        transformation_matrix.txt # 4x4 rigid registration matrix (Mobile -> Reference)
        targets_livox.txt         # Target picks in mobile SLAM coordinates
```

### Input Format Rules

* **Trajectories (`trajectory.txt`):** Must contain raw poses from the SLAM estimator. **Do not pre-transform the trajectory coordinates**. The pipeline internally applies `transformation_matrix.txt` during spine extraction.
* **Point Clouds (`.ply`):** Standard ASCII or binary format. Mobile point clouds must be pre-registered into the reference coordinate system.
* **Coordinate Matrices & Targets:** Clean whitespace- or comma-delimited text files readable via `numpy.loadtxt`. Target files require `id, x, y, z` per row.

---

## 8. Repository Structure

```
├── run_pipeline.py            # Primary CLI: single-run execution pipeline
├── run_statistics.py          # Statistics CLI: multi-run aggregation and reporting
├── dataset.py                 # File discovery and directory hierarchy resolver
├── config.py                  # Global parameters, thresholds, and execution flags
├── spine.py                   # Spline centreline fitting, (s, r, θ) coordinate extraction
├── cross_sections.py          # Profile slicing, azimuthal binning, convex hull estimation
├── surface_mesh.py            # 2D grid hole-filling, 3D mesh synthesis, divergence integration
├── marching_cubes.py          # Voxel occupancy volume estimator
├── profile_cloud.py           # Custom binary PLY exporter with custom scalar fields
├── io_utils.py                # Point cloud, trajectory, and matrix I/O routines
├── decomposition.py           # V = A_mean * L decomposition and target scale checks
├── phantom.py                 # Synthetic cylindrical validation geometry generator
└── diagnostics/               # Diagnostic inspection and verification tools
```

---

## 9. Installation & Dependencies

Requires **Python 3.10+**.

```bash
# Clone the repository
git clone <repo-url>
cd volume

# Create and activate a virtual environment
python3 -m venv ~/.venvs/slam_sweep
source ~/.venvs/slam_sweep/bin/activate

# Install required dependencies
pip install numpy scipy open3d matplotlib scikit-image
```

Run the synthetic phantom validation to verify numerical accuracy across all estimators:

```bash
python run_pipeline.py --phantom
```