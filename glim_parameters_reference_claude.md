# GLIM Configuration Parameter Reference

Based on `koide3/glim` master branch (config/ directory + docs/parameters.md + source verification).

GLIM's pipeline is **preprocess → odometry → sub-mapping → global-mapping**, with all four stages configured independently. The top-level `config.json` is just a dispatcher that points to the active config file for each stage, so you can swap entire estimation backends (GPU / CPU / CT-only / pose-graph) by changing one line.

Defaults below are the ones shipped in the repo. "Recommended ranges" come from the official `docs/parameters.md`, the Sensor Setup wiki guidance, and source-code inspection.

---

## 1. `config.json` — top-level dispatcher

Just selects which sub-config file to load. The only one you'll usually touch:

- **config_odometry** — pick `config_odometry_gpu.json` (default, fastest+best), `config_odometry_cpu.json` (no GPU available), or `config_odometry_ct.json` (LiDAR-only, no IMU, useful when IMU is bad or absent).
- **config_sub_mapping** / **config_global_mapping** — pick the `_gpu` / `_cpu` / `_passthrough` / `_pose_graph` variant matching your hardware and chosen odometry.

The CPU/GPU choice has to be **consistent across all three** (odometry, sub-mapping, global). Don't mix.

---

## 2. `config_preprocess.json` — input cleanup and downsampling

This is the front-end filter that every scan goes through before anything else. Tuning here directly affects both quality and speed of everything downstream.

- **distance_near_thresh** (default `0.5` m) — Drop points closer than this. Removes self-hits, mount, person carrying the scanner. **Low (0.1 m):** keeps near clutter (cables, body). **High (1.5 m):** safer indoor, but you start losing valid wall/floor returns in tight rooms.

- **distance_far_thresh** (default `100.0` m) — Drop points beyond this. **Low (30 m):** less noise, less RAM, but throws away outdoor structure. **High (200 m):** more reach, but far-range points are noisy and degrade covariance estimation; useful for highway/open-area scans where you need long-range tie features.

- **use_random_grid_downsampling** (default `true`) — `true` = voxel-based random sampling (preserves density variation, recommended). `false` = classic voxel-grid centroid downsampling (loses density info, slightly worse for GICP covariances).

- **downsample_resolution** (default `1.0` m) — Voxel size for the downsampling stage. **Low (0.25 m):** denser cloud, slower. **High (2 m):** very sparse, fast, but you lose geometric detail — bad for tight indoor / structural-deformation work.

- **random_downsample_target** (default `10000` points) — Hard cap on output points per scan. **This is the main speed knob.** Set to `-1` to disable the cap (then `random_downsample_rate` is used). **Low (3000–5000):** fast, but undersamples in tunnels/featureless scenes → divergence risk (see the highway-rosbag issue: target=`-1` worked everywhere except tunnels). **High (20k–50k):** more robust but linearly slower. For geomonitoring you may want this high to retain the surface fidelity that matters downstream.

- **random_downsample_rate** (default `0.1`) — Only used when `random_downsample_target ≤ 0`. Fraction of points kept (0.1 = 10%).

- **enable_outlier_removal** (default `false`) — Statistical outlier removal (k-NN-based). Off by default because GICP is already fairly robust and SOR is slow. Worth turning **on** for noisy outdoor LiDAR with fog/rain/dust.

- **outlier_removal_k** (default `10`) — k-NN size for SOR. Standard SOR knob; bigger k = smoother decision but slower.

- **outlier_std_mul_factor** (default `1.0`) — A point is rejected if its mean-neighbor-distance is > μ + factor·σ. **Low (0.5):** aggressive, kills real geometry. **High (3):** lenient, lets noise through.

- **enable_cropbox_filter** (default `false`) — Hard box-crop. Useful to remove the vehicle/robot body or to limit the workspace.

- **crop_bbox_frame** (default `"lidar"`) — Reference frame of the crop box (`"lidar"` or `"imu"`).

- **crop_bbox_min** / **crop_bbox_max** — AABB corners in the chosen frame.

- **k_correspondences** (default `10`) — Number of neighbors used to compute the per-point covariance (the "G" in GICP). **Critical for sparse LiDARs.** Docs explicitly say to raise to **15–30 for Velodyne VLP16-class sparse scanners** to avoid degenerate covariances. For dense Livox MID-360 / OS1-64, 10 is fine. Diagnostic: set viewer color mode to `NORMAL` — if flat planes look uniform, covariances are healthy. **Too low:** noisy/degenerate covariances → registration drift, plane-on-plane slipping. **Too high:** smooths real geometry, slower.

- **num_threads** (default `2`) — Preprocessing CPU threads. Cheap to bump on a modern CPU (4–8).

---

## 3. `config_odometry_gpu.json` — GPU LiDAR-IMU odometry (default backend)

The core real-time estimator. Uses keyframe-based VGICP on a fixed-lag factor graph with ISAM2.

### Initialization
- **initialization_mode** (default `"LOOSE"`) — `"LOOSE"` runs a loose-coupled IMU initialization over a window (takes a few seconds at startup, robust). `"NAIVE"` just uses the first acceleration vector as gravity (instant, works if the sensor starts stationary and the IMU is well-aligned). Use LOOSE unless you have a reason not to.
- **initialization_window_size** (default `1.0` s, CPU default `3.0`) — Duration of the LOOSE-init data window. **Low (<1 s):** less data, bias estimate noisier. **High (5–10 s):** more accurate gravity/bias init, but you must hold the sensor still that long before motion.
- **init_pose_damping_scale** (default `1e10`) — Damping (= precision) of the prior on the very first pose. This pins the gauge of the optimization (otherwise the whole graph can slide in SE(3) freely). `1e10` is "essentially rigid." You rarely change this; lowering it is only meaningful when you want to softly anchor the first pose to an external reference.

### Optimization (ISAM2 fixed-lag smoother)
- **smoother_lag** (default `5.0` s) — Length of the sliding optimization window. **All variables older than this are marginalized out.** This is the trade-off between local consistency and compute. **Low (1–2 s):** cheap, but no chance to correct recent drift via re-linearization. **High (10–20 s):** much better local consistency, slower, more RAM. Note: anything in `extension_modules` that adds factors must reference variables still inside this window.
- **use_isam2_dogleg** (default `false`) — `true` = use Powell's dogleg in ISAM2 instead of Gauss-Newton. More robust to bad linearizations, **noticeably slower**. Turn on for very degenerate environments (tunnels, long corridors) if you see crashes/divergences.
- **isam2_relinearize_skip** (default `1`) — Relinearize every N updates. **`1`:** every step (most accurate, default). Larger = re-linearize less often = faster but lazier.
- **isam2_relinearize_thresh** (default `0.1`) — Relinearize a variable only if its linear delta exceeds this. **Lower (0.01):** more re-linearizations, more accurate, slower. **Higher (0.5):** fewer re-linearizations, faster, may miss corrections.
- **fix_imu_bias** (default `false`) — `true` = don't estimate IMU bias online; use the value from init. Only useful if you've pre-calibrated bias and trust it for the run length.

### VGICP voxel parameters (registration core)
- **voxel_resolution** (default `0.25` m) — Base voxel size for the VGICP cost. **Indoor:** `0.1–0.25 m`. **Outdoor / large landslides / glaciers:** `0.5–1.0 m`. **Too low:** voxels under-populated → noisy local Gaussians → registration jitter. **Too high:** geometry blurred, fine features lost, lower accuracy on planar structures.
- **voxel_resolution_max** (default `0.5` m) — Adaptive upper bound. If `> voxel_resolution`, GLIM **automatically grows voxel size with the median point distance**, clamped between `voxel_resolution_dmin` and `voxel_resolution_dmax`. Net effect: tight voxels indoors, big voxels at long range. This is helpful for mixed-scale scenes.
- **voxel_resolution_dmin** (default `5.0` m) — Distance at which the adaptive sizing starts ramping up.
- **voxel_resolution_dmax** (default `20.0` m) — Distance at which voxel size saturates at `voxel_resolution_max`.
  Formula (from source): `base = res + clamp((dist_median − dmin) / (dmax − dmin), 0, 1) · (res_max − res)`.
- **voxelmap_levels** (default `2`) — Number of multi-resolution voxelmap pyramid levels. Each higher level is `scaling_factor` times coarser. **More levels → larger basin of convergence → robust to large inter-frame motion.** Docs: set to 2–3 for better convergence. **1:** single scale, fast but fragile to fast motion. **3+:** very robust, slower.
- **voxelmap_scaling_factor** (default `2.0`) — Ratio between successive levels. 2.0 is standard (each level is 2× coarser).
- **full_connection_window_size** (default `2`) — Latest sensor pose is connected to the last *N* poses via registration factors. **`1`:** chain-only (frame-to-previous). **`3–5`:** dense graph, recommended for aggressive motion — gives multiple constraints per pose, much more drift-resistant in fast-yaw or vibrating settings.

### Keyframe management
- **keyframe_update_strategy** (default `"OVERLAP"`) — Three options:
  - **`"OVERLAP"`** (recommended default): inserts a keyframe when overlap with the current keyframe set drops below `keyframe_max_overlap`. Self-adapts to indoor and outdoor; **`keyframe_max_overlap` is the tuning knob.**
  - **`"DISPLACEMENT"`**: inserts based on translation/rotation thresholds. Intuitive, easy to reason about — tune with `keyframe_delta_trans` / `keyframe_delta_rot`.
  - **`"ENTROPY"`**: covariance-entropy-based. Docs say *"often difficult to tune and is not recommended."* Avoid.
- **max_num_keyframes** (default `15`) — Cap on keyframes held in the sliding window. **Increasing this reduces odometry drift** (docs). **5:** very light, drifts more. **30:** much more stable, RAM/compute up. For long traverses (your highway-style or glacier transects) 20–30 helps.
- **keyframe_min_overlap** (default `0.01`) — Keyframes with overlap below this with the latest are *dropped*. Very small by default (don't drop unless completely separated).
- **keyframe_max_overlap** (default `0.7`) — If the latest frame's overlap with all keyframes is below this, insert a new keyframe. **Higher (0.8–0.9):** more frequent keyframes, denser, more robust in dynamic scenes. **Lower (0.5):** fewer keyframes, lighter.
- **keyframe_delta_trans** (default `2.0` m) — DISPLACEMENT-mode translation threshold.
- **keyframe_delta_rot** (default `0.5` rad ≈ 28.6°) — DISPLACEMENT-mode rotation threshold.
- **keyframe_entropy_thresh** (default `0.99`) — ENTROPY-mode threshold (relative to running entropy average). Insert a keyframe when the new frame's information content is below 99% of the running mean. Not recommended to tune; just don't use ENTROPY mode.

### Misc
- **validate_imu** (default `true`) — Logs IMU validation diagnostics. Keep on; cheap.
- **save_imu_rate_trajectory** (default `true`) — Saves the high-rate (IMU-integrated) trajectory in the dump. Useful for your post-processing.
- **num_threads** (default `2`) — Odometry CPU threads. 4 is reasonable on a beefy machine.

---

## 4. `config_odometry_cpu.json` — CPU LiDAR-IMU odometry

Same factor-graph structure as the GPU version but uses CPU-based registration. Most parameters are identical; the registration core differs:

- **registration_type** (default `"GICP"`) — `"GICP"` uses iVox-based GICP (accurate, robust). `"VGICP"` uses voxelized GICP (faster, needs tuning indoors).
- **max_iterations** (default `8`) — LM iterations per scan-matching call. **Low (3):** fast, may not converge on fast motion. **High (20):** more iterations of polishing, slower.
- **lru_thresh** (default `100`) — iVox cache LRU threshold (number of voxels kept in the cache).
- **target_downsampling_rate** (default `0.1`) — Fraction of target-cloud points used per match. **Lower:** faster, more registration noise. **Higher:** more robust, slower.
- **ivox_resolution** (default `1.0` m) — iVox cell size for GICP. **Also controls max correspondence distance.** Docs: use ~`1.0` for outdoor; `0.5` is OK indoor.
- **ivox_min_dist** (default `0.1` m) — Minimum point spacing inside an iVox cell (keeps the local map from getting redundantly dense).
- **vgicp_resolution** (default `0.5` m) — VGICP voxel size if you switch to VGICP. **Indoor:** `0.25–0.5 m`; **outdoor:** `0.5–2.0 m` (per docs).
- **vgicp_voxelmap_levels** (default `1`) — Multi-resolution levels for VGICP path.
- **vgicp_voxelmap_scaling_factor** (default `2.0`) — Same role as GPU version.

The CPU backend is genuinely slower than GPU; if you have CUDA, prefer GPU.

---

## 5. `config_odometry_ct.json` — CT-ICP (LiDAR-only, no IMU)

Use this when you don't have IMU or it's unreliable. **Note:** `X(i)` here represents the LiDAR pose (not IMU pose), and there are no velocity/bias variables. Inside, each scan is treated as a continuous-time trajectory between two endpoints `X(i)` (start) and `Y(i)` (end), letting de-skewing and registration happen jointly.

- **ivox_resolution** (default `1.0` m), **ivox_min_points_dist** (default `0.1` m), **ivox_lru_thresh** (default `200`) — Same role as in the CPU GICP backend.
- **max_correspondence_distance** (default `2.0` m) — Maximum point-to-point search distance for CT-ICP correspondences. **Low (0.5 m):** restrictive, fast motion will fail. **High (5 m):** robust to large motion, but more bad correspondences.
- **location_consistency_inf_scale** (default `1e-3`) — Weight of the soft prior that the *new* scan endpoint stays near the *previous* scan endpoint (continuity prior on consecutive endpoints). **Low (1e-5):** very loose continuity → can drift between scans. **High (1):** glues endpoints together — bad for fast motion, but stabilizes static/slow scenes.
- **constant_velocity_inf_scale** (default `1e3`) — Weight of the "scan starts and ends with the same delta-pose as previous scan" constraint (constant-velocity prior on within-scan motion). **High value by default = strong CV assumption.** Lower this for jerky motion (handheld, drone with thrust pulses); raise for smooth motion (vehicle on a road).
- **lm_max_iterations** (default `8`) — LM iterations.
- **smoother_lag** (default `1.0` s) — Note this is **much shorter** than the IMU variants (5 s). Because there's no IMU pre-integration giving smooth predictions, holding many past poses in the smoother is more expensive per unit benefit. Increase cautiously.
- **use_isam2_dogleg / isam2_relinearize_skip / isam2_relinearize_thresh** — Same as IMU variants.
- **num_threads** (default `4`).

---

## 6. `config_sub_mapping_gpu.json` — sub-mapping (groups keyframes into submaps)

Sub-mapping bundles N consecutive keyframes into a "submap" (a small bundle-adjusted local map) that becomes a single node in global mapping. This is where the multi-resolution structure of the global map is born.

### General
- **enable_imu** (default `true`) — Must be `false` if you're using `odometry_ct` (no IMU variables exist).
- **enable_optimization** (default `false`) — `false` = keyframes are placed using odometry poses as-is (cheap). `true` = re-optimize each submap as it's being built. Docs: turn `true` only if odometry is unstable enough that submaps need internal re-bundle-adjustment.

### Keyframe management (inside one submap)
- **max_num_keyframes** (default `15`) — Keyframes per submap. Larger submaps = fewer, bigger nodes in global graph = lower global compute but coarser correction granularity.
- **keyframe_update_strategy** (default `"OVERLAP"`) — Same OVERLAP / DISPLACEMENT logic as in odometry.
- **keyframe_update_min_points** (default `500`) — Reject candidates with fewer than this many points (likely degraded or short scans).
- **keyframe_update_interval_rot** (default `3.14` rad ≈ 180°), **keyframe_update_interval_trans** (default `1.0` m) — DISPLACEMENT thresholds for keyframe insertion within a submap. Note rot is essentially disabled (π).
- **max_keyframe_overlap** (default `0.6`) — OVERLAP threshold for insertion within a submap.

### Relative-pose factors (between consecutive keyframes inside a submap)
- **create_between_factors** (default `false` GPU, `true` CPU) — If `true`, add SE(3) between-factors derived from odometry between consecutive keyframes. Adds soft odometry constraints; the GPU path doesn't need them because VGICP factors alone are strong enough.
- **between_registration_type** (default `"GICP"`) — How the information matrix of those between-factors is computed. `"NONE"` = use a fixed isotropic noise model.

### Registration error factors (the meat of sub-mapping)
- **registration_error_factor_type** (default `"VGICP_GPU"`, CPU: `"VGICP"`) — Sets the cost function. Must match your hardware/build.
- **keyframe_randomsampling_rate** (default `1.0`) — Fraction of keyframe points used. `1.0` = all of them. Docs explicitly say the GPU implementation can handle `1.0` for full global registration error minimization. Lower this if VRAM-limited.
- **keyframe_voxel_resolution** (default `0.25` m) — Base VGICP voxel resolution. **Indoor: 0.15–0.25 m. Outdoor: 0.5–1.0 m.**
- **keyframe_voxelmap_levels** (default `2`) — Multi-res levels. Docs recommend 2–3.
- **keyframe_voxelmap_scaling_factor** (default `2.0`).

### Post-processing (the submap product)
- **submap_downsample_resolution** (default `0.1` m GPU, `0.3` m CPU) — Voxel size for downsampling the final accumulated submap point cloud. Smaller = denser submap, more RAM/disk.
- **submap_voxel_resolution** (default `0.5` m) — [deprecated label] Voxel size used in global mapping for this submap.
- **submap_target_num_points** (default `50000`) — Hard point cap per submap (`-1` disables). For your TLS-comparison work where you eventually pull dense submaps offline, raise this if RAM allows.

---

## 7. `config_sub_mapping_passthrough.json` — minimal pass-through sub-mapping

Lighter alternative: just accumulates keyframes into a voxel-binned cloud without optimization. Used as the default companion to the pose-graph global-mapping backend.

- **keyframe_update_interval_rot** (default `0.01` rad), **keyframe_update_interval_trans** (default `0.1` m) — Very fine thresholds; means **every reasonable odometry frame becomes a keyframe**.
- **max_num_keyframes** (default `50`) — Submap-issue criterion: when this many keyframes accumulated, close the submap. Set to `-1` to disable.
- **max_num_voxels** (default `-1`) — Submap closes when voxel count exceeds this. `-1` = disabled.
- **adaptive_max_num_voxels** (default `2.5`) — Submap closes when voxel count exceeds `2.5 ×` the voxel count after the first 3 keyframes. Adaptive criterion that adjusts to environment density (closes earlier indoors with dense walls, later outdoors). Set `-1` to disable. **Any one of the three above triggers a close.**
- **submap_voxel_resolution** (default `0.5` m), **min_dist_in_voxel** (default `0.2` m), **max_num_points_in_voxel** (default `100`) — Submap-construction voxel parameters; cap points per voxel and minimum spacing.
- **submap_target_num_points** (default `50000`) — Final downsampling cap.

---

## 8. `config_global_mapping_gpu.json` — global mapping (VGICP-based, implicit loop closure)

Builds and optimizes the full submap-level factor graph. Uses **implicit loop closure**: instead of a separate detector, any pair of submaps within `max_implicit_loop_distance` whose overlap exceeds `min_implicit_loop_overlap` gets a registration-error factor — i.e., loops are detected and constrained in one shot.

### General
- **enable_imu** (default `true`) — Disable if using LiDAR-only.
- **enable_optimization** (default `true`) — Setting `false` disables global optimization entirely (you just stitch by odometry).
- **init_pose_damping_scale** (default `1e10`) — Gauge prior on the first submap. Same role as in odometry.

### Between-factors
- **create_between_factors** (default `false` GPU, `true` CPU) — Add SE(3) odom-between factors between consecutive submaps.
- **between_registration_type** (default `"GICP"`).

### Registration error factors (the loop-closure mechanism)
- **registration_error_factor_type** (default `"VGICP_GPU"`).
- **randomsampling_rate** (default `1.0` GPU, `0.2` CPU) — Point subsampling for factor evaluation. GPU can handle 1.0.
- **submap_voxel_resolution** (default `0.5` m) — Base voxel resolution for inter-submap VGICP. **Indoor: 0.15–0.25 m, outdoor: 0.5–1.0 m.**
- **submap_voxel_resolution_max / _dmin / _dmax** (defaults `1.0 / 5.0 / 20.0`) — Same adaptive sizing logic as odometry (only present in GPU variant).
- **submap_voxelmap_levels** (default `2`) — Multi-res levels.
- **submap_voxelmap_scaling_factor** (default `2.0`).
- **max_implicit_loop_distance** (default `100.0` m) — **Two submaps farther apart than this are never paired for loop closure.** Larger = more loop candidates (more compute, but catches big loops). For city-scale traverses, raise to 200–500. For room-scale, 30–50 is plenty.
- **min_implicit_loop_overlap** (default `0.2`) — Minimum geometric overlap (computed via `gtsam_points::overlap_auto` against the voxelmap) to create a loop factor. **Lower (0.1):** more factors, including weak ones — can introduce bad loops. **Higher (0.4):** only well-overlapping pairs — safer, may miss real loops with grazing overlap.

### Optimizer
- **use_isam2_dogleg / isam2_relinearize_skip / isam2_relinearize_thresh** — Same as odometry. For global mapping, dogleg is more often worth it because the graph is bigger and bad linearizations cost more.

---

## 9. `config_global_mapping_pose_graph.json` — pose-graph global mapping (explicit loops)

Alternative backend. Uses **explicit loop detection** (geometric distance + registration validation) and a classic pose-graph instead of dense submap-to-submap VGICP factors. Lighter, simpler, but lower-accuracy than the implicit/VGICP path.

- **enable_optimization** (default `true`).
- **init_pose_damping_scale** (default `1e6`) — Note this is lower than the VGICP backend (`1e10`); pose-graph BA is less stiff overall.

### Loop detection
- **registration_type** (default `"VGICP"`) — Type used to verify loop candidates. `"GICP"` or `"VGICP"`.
- **min_travel_dist** (default `50.0` m) — A submap pair is only considered for loops if the trajectory between them is at least this long. **Prevents trivial "loops" between adjacent submaps.** Lower for tight indoor; raise for outdoor.
- **max_neighbor_dist** (default `5.0` m) — Pair must be spatially within this distance. **Higher = catches looser loops at cost of more false candidates.**
- **min_inliear_fraction** (default `0.5`, sic: typo in source as `min_inliear_fraction`) — Minimum inlier fraction (overlap with target) to accept a loop candidate. **Lower (0.3):** lenient, accepts shaky loops. **Higher (0.7):** strict, may miss real ones.
- **subsample_target** (default `10000`) — Point cap for loop-candidate validation. `-1` disables subsampling.
- **subsample_rate** (default `0.1`) — Used when `subsample_target < 0`.
- **gicp_max_correspondence_dist** (default `2.0` m) — Max correspondence distance during GICP-based loop validation.
- **vgicp_voxel_resolution** (default `2.0` m) — Voxel resolution if using VGICP for loop validation. Coarser than odometry — loop matching should tolerate larger initial error.

### Factor settings
- **odom_factor_stddev** (default `1e-3`) — Sigma (SE(3) isotropic) of between-factors derived from odometry. Tighter = trusts odometry more. `1e-3` says "odometry is essentially correct locally."
- **loop_factor_stddev** (default `0.1`) — Sigma of loop-closure factors. Looser than odom factors because loop matches are less reliable per-axis.
- **loop_factor_robust_width** (default `1.0`) — Width parameter of the Huber robust kernel wrapped around loop factors. **Smaller (0.5):** aggressively down-weights outliers — safer against bad loops. **Larger (2):** more linear behavior, accepts more loop information.
- **loop_candidate_buffer_size** (default `100`) — How many pending candidates the detector buffers before evaluation.
- **loop_candidate_eval_per_thread** (default `2`) — Throughput knob for the loop validator.

### Optimizer
- **use_isam2_dogleg / isam2_relinearize_skip / isam2_relinearize_thresh** — Same as elsewhere.
- **num_threads** (default `2`).

---

## 10. `config_viewer.json` — UI only (no impact on mapping)

Two viewer profiles (`standard_viewer`, `interactive_viewer`) with identical fields:

- **viewer_width / viewer_height** (default `2560 × 1440`).
- **default_z_range** (default `[-2.0, 4.0]`) — Z-color-mapping range in meters.
- **enable_partial_rendering** (default `false`) — Renders submaps in chunks across frames. Useful for weak GPUs / very large maps, at the cost of visual glitches.
- **partial_rendering_budget** (default `1024`) — Points per frame budget when partial rendering is on.
- **point_shape_circle** (default `true`) — Circle vs square sprites.
- **point_size_metric** (default `true`) — `true` = point size in meters; `false` = pixels.
- **point_size** (default `0.025`) — Point size value (units depend on `point_size_metric`).
- **points_alpha** / **factors_alpha** (default `0.5`) — Transparency.

---

## 11. `config_logging.json` — logging only

- **log_dir** (default `/tmp`).
- **save_logs** (default `true`).
- **rotate_logs** (default `true`).
- **max_file_size_kb** (default `8192` = 8 MB).
- **max_files** (default `10`).

---

# Sensor-specific parameters (documentation reference)

These are correctness-critical but they're **calibration values, not tuning knobs** — set them once per sensor and don't touch.

## `config_sensors.json`

### IMU noise model (used in GTSAM IMU preintegration)
- **imu_acc_noise** (default `0.05`) — Accelerometer white-noise standard deviation [m/s²/√Hz]. Datasheet value.
- **imu_gyro_noise** (default `0.02`) — Gyro white-noise std-dev [rad/s/√Hz].
- **imu_int_noise** (default `1e-3`) — Integration noise. Generally a numerical-stability term in GTSAM; leave at default unless you know what you're doing.
- **imu_bias_noise** (default `1e-5`) — Bias random-walk std-dev. Higher = bias is allowed to wander more.

These directly affect the IMU-factor information matrix. **If they're too tight relative to the actual IMU**, the optimizer will overtrust IMU and reject good visual corrections; **too loose**, the IMU contributes nothing.

### LiDAR
- **global_shutter_lidar** (default `false`) — `true` skips per-point timestamp processing and disables motion-deskew. Set `true` for true global-shutter sensors (e.g. flash LiDAR, ToF cameras). For all rotating/scanning LiDARs leave `false`.
- **T_lidar_imu** (default `[0.006, -0.012, 0.008, 0, 0, 0, 1]`) — SE(3) extrinsic from IMU frame to LiDAR frame in TUM `[x, y, z, qx, qy, qz, qw]`. Sanity check: with the IMU stationary and z-axis up, gravity vector should read ~`[0, 0, +9.81]` (ROS REP-145 convention). Reference values are commented in the file for Ouster OS0, Livox Avia, Realsense L515, Azure Kinect, ZED2i, Newer College datasets — **do not trust those blindly**, calibrate your own.
- **intensity_field** (default `"intensity"`) — Field name in the `PointCloud2` message containing intensity.
- **ring_field** (default `""`) — Field name for the ring (laser ID). Blank = auto.

### LiDAR per-point timing (you debugged this for Livox before)
- **autoconf_perpoint_times** (default `true`) — Auto-detect whether per-point times are absolute or relative.
- **autoconf_prefer_frame_time** (default `false`) — When `true`, always use the frame timestamp regardless of per-point times.
- **perpoint_relative_time** (default `true`) — `true` = per-point times are offsets from the frame timestamp; `false` = absolute. (For Livox SDK2 they're absolute ns — the exact issue you hit before.)
- **perpoint_time_scale** (default `1.0`) — Multiplier. `1.0` = seconds; `1e-9` = nanoseconds; `1e-6` = microseconds. Set this correctly for your driver.

### Camera (only used by `glim_ext` modules)
- **global_shutter_camera** (default `true`).
- **image_size** (`[752, 480]`).
- **T_lidar_camera** — SE(3) extrinsic from camera frame to LiDAR frame.
- **intrinsics** — `[fx, fy, cx, cy]`.
- **distortion_model** (default `"plumb_bob"`) — pinhole+radial-tangential.
- **distortion_coeffs** — `[k1, k2, p1, p2, k3]`.

---

## `config_ros.json`

Mostly transport / topic plumbing. The mapping-relevant ones:

- **enable_local_mapping** / **enable_global_mapping** — Disable backend stages independently. With both off, GLIM runs odometry-only.
- **keep_raw_points** (default `false`) — Pass raw (un-downsampled) points through the pipeline. Required only by extension modules (e.g. visual). Costs RAM.
- **imu_time_offset** (default `0.0` s), **points_time_offset** (default `0.0` s) — Time offsets to compensate driver latency. **A wrong offset of even 10 ms is visible as drift on fast motion.** Validate with `libimu_validator.so`.
- **acc_scale** (default `0.0` = auto-detect; set to `9.80665` for Livox where accel is in `g`) — Pre-multiplier on accel values.
- **imu_frame_id / lidar_frame_id / base_frame_id / odom_frame_id / map_frame_id** — Standard TF chain.
- **publish_imu2lidar** (default `true`).
- **tf_time_offset** (default `1e-6` s).
- **extension_modules** — List of `.so` modules to dynamically load. Defaults: `memory_monitor`, `standard_viewer`, `rviz_viewer`. Add `imu_validator` to debug IMU/extrinsic issues.
- **imu_topic / points_topic / image_topic** — ROS topic names.
- **imu_qos / points_qos / image_qos** — ROS2 QoS profiles. For high-rate IMU at 200–1000 Hz, the `depth: 1000` setting matters — too small and the subscriber will drop samples.

---

# Tuning workflow suggestions (for your geomonitoring use cases)

A practical order in which to touch knobs:

1. **First get the sensor block right** (`T_lidar_imu`, IMU noise, per-point time settings). If this is wrong, nothing downstream will save you. Run with `libimu_validator.so` loaded to verify.
2. **Set `distance_far_thresh`** to match your real working range (TLS may not need 100 m; landslide aerial scans may need 200 m+).
3. **Tune `voxel_resolution` and `k_correspondences`** to your scene scale. Indoor structural monitoring → 0.15–0.25 m and k≈10; outdoor landslide → 0.5–1.0 m and k≈10–15; sparse VLP16-like → k=15–30.
4. **For long traverses** (highway-style): raise `max_num_keyframes` (20–30), raise `smoother_lag` (8–10 s), raise `max_implicit_loop_distance` to your loop scale.
5. **Speed knobs:** `random_downsample_target` for the front-end; `num_threads` everywhere; `voxelmap_levels` down to 1 if desperate.
6. **Stability knobs for degenerate scenes** (tunnels, long corridors, flat snow/ice): raise `voxelmap_levels` to 3, raise `full_connection_window_size` to 3–5, consider `use_isam2_dogleg: true`.

# When in doubt, the docs/parameters.md note says:

> *To see if estimated covariances are fine, change `color_mode` in the standard viewer to `NORMAL`. If point colors are uniform on flat planes, covariances should be ok.*

That's the cheapest diagnostic — visual normals — and it will catch most bad `k_correspondences` / `voxel_resolution` / sensor-extrinsic configurations.
