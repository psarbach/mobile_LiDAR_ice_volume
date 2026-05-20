# GLIM Parameter Dependency Tree

Companion to `glim_parameters_reference_claude.md`. The earlier doc explained *what each parameter does*; this one explains *which parameters are active under which conditions*, so you can build a sane conditional search space in Optuna and avoid wasting trials on dead variables.

Conventions used below:

- `[GATE]`  parameter whose value activates/deactivates one or more children
- `└──`     child active only under the parent's gate condition
- `[XOR]`   children are mutually exclusive (one branch active at a time)
- `[ANY]`   any combination of children can be active (independent siblings)
- `★`       genuinely worth tuning
- `⚙`       calibration / one-time setup — do not put in search space
- `🚫`       documented as "not recommended" — exclude from search
- `🐢`       speed/throughput knob — tune only if also optimizing latency

---

## 0. Top-level: the three-way backend triplet

The single most important conditional is which `config_*` files you load — this gates entire blocks of downstream parameters. **The three backends must be chosen consistently**; you cannot mix GPU and CPU variants across stages.

```
config.json (dispatcher)
│
├── config_odometry  [XOR — pick exactly one]
│   ├── config_odometry_gpu.json      ── GPU LiDAR-IMU (default, best)
│   ├── config_odometry_cpu.json      ── CPU LiDAR-IMU (no CUDA available)
│   └── config_odometry_ct.json       ── LiDAR-only CT-ICP (no IMU)
│
├── config_sub_mapping  [XOR — must match odometry's CPU/GPU choice]
│   ├── config_sub_mapping_gpu.json
│   ├── config_sub_mapping_cpu.json
│   └── config_sub_mapping_passthrough.json   ── lightest; pairs with pose_graph below
│
└── config_global_mapping  [XOR — must match odometry's CPU/GPU choice]
    ├── config_global_mapping_gpu.json        ── implicit loop closure (VGICP)
    ├── config_global_mapping_cpu.json
    └── config_global_mapping_pose_graph.json ── explicit loop closure (lighter, less accurate)
```

### Cross-config dependencies (these are real and easy to miss)

```
IF config_odometry == "config_odometry_ct.json":
    REQUIRED  config_sub_mapping_gpu.json   :: enable_imu  = false
    REQUIRED  config_global_mapping_gpu.json :: enable_imu = false
    (because odometry_ct never creates V(i) and B(i) variables)
```

For your sweep this means **the backend triplet is a single categorical choice with ~4 sensible combinations**, not three independent ones. Practical menu:

| Triplet name | odometry | sub-mapping | global-mapping | When |
|---|---|---|---|---|
| `gpu_full`     | `_gpu` | `_gpu`          | `_gpu`        | CUDA box, default and best accuracy |
| `gpu_light`    | `_gpu` | `_passthrough`  | `_pose_graph` | CUDA box but want speed |
| `cpu_full`     | `_cpu` | `_cpu`          | `_cpu`        | No CUDA, want accuracy |
| `cpu_light`    | `_cpu` | `_passthrough`  | `_pose_graph` | No CUDA, want speed |
| `ct_only`      | `_ct`  | `_gpu` + enable_imu=false | `_gpu` + enable_imu=false | LiDAR-only |

For an Optuna sweep I'd **fix the triplet** (probably `gpu_full`, since that's what you're running with the Unitree L1) rather than searching over it. Tune within the chosen triplet.

---

## 1. `config_preprocess.json` (active in every triplet)

```
config_preprocess
│
├── distance_near_thresh         ★  (independent)
├── distance_far_thresh          ★
├── k_correspondences            ★  ← critical for sparse LiDARs; for Unitree L1 try 10–20
├── num_threads                  🐢
│
├── use_random_grid_downsampling [GATE bool, mostly leave true]
│   └── (no children — picks algorithm, not a parameter switch)
│
├── downsample_resolution        ★
│
├── random_downsample_target [GATE int]  ★
│   │  IF target > 0  →  target is used, random_downsample_rate IGNORED
│   │  IF target ≤ 0  →  random_downsample_rate IS USED
│   └── random_downsample_rate          ★ (only when target ≤ 0)
│
├── enable_outlier_removal [GATE bool]
│   │  IF false  →  the two below are IGNORED
│   ├── outlier_removal_k               ★ (only if enabled)
│   └── outlier_std_mul_factor          ★ (only if enabled)
│
└── enable_cropbox_filter [GATE bool]
    │  IF false  →  the three below are IGNORED
    ├── crop_bbox_frame                 ⚙ (application-specific, not for search)
    ├── crop_bbox_min                   ⚙
    └── crop_bbox_max                   ⚙
```

For your Optuna search, the active set under `enable_outlier_removal=False` and `enable_cropbox_filter=False` (the defaults) is just the eight top-level `★` params plus `random_downsample_rate` conditionally.

---

## 2. Odometry — three disjoint trees

### 2.1 `config_odometry_gpu.json`

```
odometry_gpu
│
├── INITIALIZATION
│   ├── initialization_mode  [GATE str, "LOOSE" | "NAIVE"]
│   │   │  IF "NAIVE"  →  initialization_window_size IGNORED
│   │   └── initialization_window_size    ★ (only when "LOOSE")
│   └── init_pose_damping_scale           ⚙ (gauge prior, ~never tuned)
│
├── OPTIMIZER (ISAM2)
│   ├── smoother_lag                      ★
│   ├── use_isam2_dogleg [GATE bool]
│   │   └── (no children — pure optimizer swap)
│   ├── isam2_relinearize_skip            ★
│   ├── isam2_relinearize_thresh          ★
│   └── fix_imu_bias [GATE bool]          ⚙ (only useful with pre-calibrated bias)
│       └── (no children)
│
├── VGICP REGISTRATION
│   ├── voxel_resolution                  ★ (single most impactful registration knob)
│   │
│   ├── voxel_resolution_max [GATE float]
│   │   │  IF voxel_resolution_max ≤ voxel_resolution  →  no adaptive sizing;
│   │   │                                              dmin/dmax are IGNORED
│   │   │  IF voxel_resolution_max  > voxel_resolution  →  adaptive on
│   │   ├── voxel_resolution_dmin         ★ (only when adaptive active)
│   │   └── voxel_resolution_dmax         ★ (only when adaptive active)
│   │
│   ├── voxelmap_levels [GATE int ≥ 1]    ★
│   │   │  IF levels == 1  →  scaling_factor IGNORED (only one level exists)
│   │   └── voxelmap_scaling_factor       ★ (only when levels ≥ 2)
│   │
│   └── full_connection_window_size       ★
│
├── KEYFRAME MGMT
│   ├── max_num_keyframes                 ★ (unconditional cap)
│   │
│   └── keyframe_update_strategy [GATE str, "OVERLAP" | "DISPLACEMENT" | "ENTROPY"]
│       │
│       ├─[if "OVERLAP"]──┐
│       │   ├── keyframe_max_overlap          ★
│       │   └── keyframe_min_overlap          (rarely tuned; safety floor)
│       │
│       ├─[if "DISPLACEMENT"]──┐
│       │   ├── keyframe_delta_trans          ★
│       │   └── keyframe_delta_rot            ★
│       │
│       └─[if "ENTROPY"]──┐  🚫  docs explicitly say "not recommended"
│           └── keyframe_entropy_thresh
│
└── MISC
    ├── validate_imu                      ⚙ (logging only)
    ├── save_imu_rate_trajectory          ⚙ (logging only)
    └── num_threads                       🐢
```

### 2.2 `config_odometry_cpu.json`

Same INITIALIZATION / OPTIMIZER / KEYFRAME MGMT / MISC subtrees as GPU (above). The registration block differs and itself has a sub-gate:

```
odometry_cpu  (registration block only — rest same as GPU)
│
├── max_iterations                        ★
├── lru_thresh                            🐢 (cache size)
├── target_downsampling_rate              ★
│
└── registration_type  [GATE str, "GICP" | "VGICP"]
    │
    ├─[if "GICP"]──┐         ← default, uses iVox-based GICP
    │   ├── ivox_resolution                ★ (most impactful)
    │   └── ivox_min_dist                  ★
    │
    └─[if "VGICP"]──┐
        ├── vgicp_resolution               ★
        │
        └── vgicp_voxelmap_levels [GATE int ≥ 1]
            │  IF levels == 1  →  scaling_factor IGNORED
            └── vgicp_voxelmap_scaling_factor   ★ (only when levels ≥ 2)
```

### 2.3 `config_odometry_ct.json` (LiDAR-only, CT-ICP)

```
odometry_ct
│
├── REGISTRATION  (iVox + CT-ICP — no GICP/VGICP toggle here)
│   ├── ivox_resolution                   ★
│   ├── ivox_min_points_dist              ★
│   ├── ivox_lru_thresh                   🐢
│   └── max_correspondence_distance       ★
│
├── PRIORS (CT-specific)
│   ├── location_consistency_inf_scale    ★ (continuity between scan endpoints)
│   └── constant_velocity_inf_scale       ★ (within-scan CV prior)
│
├── OPTIMIZER (same ISAM2 family as IMU variants, but shorter lag)
│   ├── smoother_lag                      ★ (default 1.0 s, not 5.0 s)
│   ├── lm_max_iterations                 ★
│   ├── use_isam2_dogleg                  [GATE bool]
│   ├── isam2_relinearize_skip            ★
│   └── isam2_relinearize_thresh          ★
│
└── num_threads                           🐢
```

**Note:** no IMU init parameters, no VGICP block, no keyframe-management strategy choice — CT-ICP uses continuous-time within-scan optimization rather than a keyframe graph.

---

## 3. Sub-mapping — three disjoint trees

### 3.1 `config_sub_mapping_gpu.json` (and `_cpu` with `enable_imu` swap)

```
sub_mapping_gpu
│
├── enable_imu                            [GATE bool, MUST match odometry choice]
│   └── (gates whether IMU variables are referenced; no other children)
│
├── enable_optimization                   [GATE bool]   ★
│   │  IF false  →  keyframes placed using odometry poses as-is;
│   │               the entire REGISTRATION ERROR FACTORS block below
│   │               is built but its content is mostly irrelevant
│   │  IF true   →  full block active
│   │
│   ├─ REGISTRATION ERROR FACTORS  (the bundle-adjustment cost)
│   │   ├── registration_error_factor_type      ⚙ (must match build, not search)
│   │   ├── keyframe_randomsampling_rate        ★
│   │   ├── keyframe_voxel_resolution           ★
│   │   ├── keyframe_voxelmap_levels [GATE]     ★
│   │   │   │  IF levels == 1  →  scaling_factor IGNORED
│   │   │   └── keyframe_voxelmap_scaling_factor   ★ (only when ≥ 2)
│   │   └── (factor type is fixed by your build, not a runtime knob)
│
├── KEYFRAME MGMT (inside one submap)
│   ├── max_num_keyframes                 ★ (keyframes per submap)
│   ├── keyframe_update_min_points        ★ (unconditional reject threshold)
│   │
│   └── keyframe_update_strategy  [GATE str, "OVERLAP" | "DISPLACEMENT"]
│       │
│       ├─[if "OVERLAP"]──┐
│       │   └── max_keyframe_overlap            ★
│       │
│       └─[if "DISPLACEMENT"]──┐
│           ├── keyframe_update_interval_trans  ★
│           └── keyframe_update_interval_rot    (often disabled, default π)
│
├── BETWEEN-FACTORS
│   └── create_between_factors  [GATE bool]
│       │  IF false  →  between_registration_type IGNORED
│       └── between_registration_type           (only when creating between-factors)
│
└── SUBMAP POSTPROCESSING
    ├── submap_downsample_resolution      ★ (storage knob)
    ├── submap_voxel_resolution           ⚙ (deprecated label; for global-mapping use)
    └── submap_target_num_points          🐢 (RAM knob; -1 disables)
```

### 3.2 `config_sub_mapping_passthrough.json`

```
sub_mapping_passthrough
│
├── KEYFRAME MGMT (very fine — every odometry frame is essentially a kf)
│   ├── keyframe_update_interval_trans    ★ (default 0.1 m)
│   └── keyframe_update_interval_rot      ★ (default 0.01 rad)
│
├── SUBMAP-CLOSE TRIGGERS  [ANY — first hit closes the submap]
│   ├── max_num_keyframes                 ★ (-1 disables this trigger)
│   ├── max_num_voxels                    ★ (-1 disables)
│   └── adaptive_max_num_voxels           ★ (-1 disables; default 2.5 × baseline)
│
├── SUBMAP CONSTRUCTION VOXELS
│   ├── submap_voxel_resolution           ★
│   ├── min_dist_in_voxel                 ★
│   └── max_num_points_in_voxel           🐢
│
└── submap_target_num_points              🐢
```

**Note on the close-trigger block:** since all three are ORed (any one triggers a close), if you tune them you should treat them as a 3-way joint search, not independently — otherwise one knob will dominate the others and the rest become inert.

---

## 4. Global mapping — two disjoint trees

### 4.1 `config_global_mapping_gpu.json` (implicit loop closure, default)

```
global_mapping_gpu
│
├── enable_imu                            [GATE bool, matches odometry]
│
├── enable_optimization                   [GATE bool]   ★
│   │  IF false  →  no global optimization; stitching by odometry only;
│   │               EVERYTHING below this gate is inert
│   │  IF true   →  full block active
│   │
│   ├── init_pose_damping_scale           ⚙
│   │
│   ├── REGISTRATION ERROR FACTORS  (the implicit-loop mechanism)
│   │   ├── registration_error_factor_type      ⚙ (build-dependent)
│   │   ├── randomsampling_rate                 ★
│   │   ├── submap_voxel_resolution             ★
│   │   │
│   │   ├── submap_voxel_resolution_max [GATE float]
│   │   │   │  IF max ≤ submap_voxel_resolution  →  dmin/dmax IGNORED
│   │   │   │  IF max  > submap_voxel_resolution  →  adaptive on
│   │   │   ├── submap_voxel_resolution_dmin    ★ (only when adaptive active)
│   │   │   └── submap_voxel_resolution_dmax    ★ (only when adaptive active)
│   │   │
│   │   └── submap_voxelmap_levels [GATE int ≥ 1]   ★
│   │       │  IF levels == 1  →  scaling_factor IGNORED
│   │       └── submap_voxelmap_scaling_factor      ★ (only when ≥ 2)
│   │
│   ├── IMPLICIT LOOP CONSTRAINTS
│   │   ├── max_implicit_loop_distance          ★ (huge effect at scale)
│   │   └── min_implicit_loop_overlap           ★
│   │
│   ├── BETWEEN-FACTORS
│   │   └── create_between_factors  [GATE bool]
│   │       └── between_registration_type        (only when true)
│   │
│   └── OPTIMIZER (ISAM2)
│       ├── use_isam2_dogleg               [GATE bool]
│       ├── isam2_relinearize_skip         ★
│       └── isam2_relinearize_thresh       ★
```

### 4.2 `config_global_mapping_pose_graph.json` (explicit loop closure)

```
global_mapping_pose_graph
│
├── enable_optimization                   [GATE bool]   ★
│   │  IF false  →  everything below inert
│   │
│   ├── init_pose_damping_scale           ⚙ (1e6 default — softer than VGICP backend)
│   │
│   ├── LOOP DETECTION (geometric pre-filter)
│   │   ├── min_travel_dist               ★ (lower bound on loop arc length)
│   │   └── max_neighbor_dist             ★ (upper bound on spatial distance)
│   │
│   ├── LOOP VALIDATION  (the geometric registration test)
│   │   ├── min_inliear_fraction          ★ (note: typo in source, but real name)
│   │   ├── subsample_target [GATE int]
│   │   │   │  IF subsample_target > 0  →  subsample_rate IGNORED
│   │   │   │  IF subsample_target ≤ 0  →  subsample_rate IS USED
│   │   │   └── subsample_rate            ★ (only when target ≤ 0)
│   │   │
│   │   └── registration_type [GATE str, "GICP" | "VGICP"]
│   │       │
│   │       ├─[if "GICP"]──┐
│   │       │   └── gicp_max_correspondence_dist   ★
│   │       │
│   │       └─[if "VGICP"]──┐
│   │           └── vgicp_voxel_resolution         ★
│   │
│   ├── FACTOR SETTINGS
│   │   ├── odom_factor_stddev            ★
│   │   ├── loop_factor_stddev            ★
│   │   └── loop_factor_robust_width      ★
│   │
│   ├── THROUGHPUT
│   │   ├── loop_candidate_buffer_size    🐢
│   │   ├── loop_candidate_eval_per_thread 🐢
│   │   └── num_threads                   🐢
│   │
│   └── OPTIMIZER
│       ├── use_isam2_dogleg              [GATE bool]
│       ├── isam2_relinearize_skip        ★
│       └── isam2_relinearize_thresh      ★
```

---

## 5. Sensors / I/O (`config_sensors.json`, `config_ros.json`)

These are calibration and plumbing — **none of them belong in an Optuna search space**. They go into a fixed "sensor profile" outside the search. Showing the conditional structure anyway because a few have it:

```
config_sensors
│
├── IMU NOISE MODEL  (all ⚙ — set from datasheet/Allan variance, not from search)
│   ├── imu_acc_noise / imu_gyro_noise / imu_int_noise / imu_bias_noise
│
├── LIDAR
│   ├── T_lidar_imu                       ⚙ (extrinsic — calibrate, don't search)
│   ├── intensity_field / ring_field      ⚙
│   │
│   ├── global_shutter_lidar  [GATE bool]
│   │   │  IF true  →  per-point timestamps not used; deskew disabled;
│   │   │              the per-point timing block below becomes inert
│   │   │
│   │   └─ PER-POINT TIMING (only meaningful when global_shutter_lidar == false)
│   │       ├── autoconf_perpoint_times  [GATE bool]
│   │       │   │  IF true  →  perpoint_relative_time AUTO-DETECTED
│   │       │   │              (your manual setting overridden)
│   │       │   │  IF false →  perpoint_relative_time IS USED
│   │       │   └── perpoint_relative_time        ⚙ (only honored when autoconf=false)
│   │       │
│   │       ├── autoconf_prefer_frame_time  [GATE bool]
│   │       │   │  IF true  →  per-point times are IGNORED entirely; frame time used
│   │       │   │  IF false →  per-point times respected per autoconf logic above
│   │       │
│   │       └── perpoint_time_scale       ⚙ (only relevant when per-point times honored)
│   │
└── CAMERA  (only consumed by glim_ext modules, ignore unless using them)

config_ros
│
├── PIPELINE STAGE TOGGLES
│   ├── enable_local_mapping     [GATE bool]   if false → sub_mapping config inert
│   └── enable_global_mapping    [GATE bool]   if false → global_mapping config inert
│
├── acc_scale  [GATE float, 0.0 means auto-detect]
│   │  ★ correctness-critical for the Unitree L1 (you confirmed ~15.69 m/s²
│   │    actual, set explicitly rather than relying on autodetect)
│
├── imu_time_offset / points_time_offset    ⚙ (calibration; validate via libimu_validator.so)
├── tf_time_offset                          ⚙
├── *_frame_id / *_topic / *_qos            ⚙ (plumbing)
└── keep_raw_points                         ⚙ (only true if you need raw cloud passthrough)
```

---

## 6. Optuna implementation pattern

The conditional structure above maps naturally onto Optuna's nested-`suggest_*` pattern. Sketch (Python pseudo-code, adapt to your `slam_sweep` orchestration):

```python
def suggest_glim_params(trial):
    cfg = {}

    # ---- Fix the backend triplet up-front (don't search over this) ----
    # If you DO want to compare triplets, run a separate Optuna study per triplet
    # and compare the best-of-each — this is far cheaper than a nested categorical.
    triplet = "gpu_full"   # or use trial.suggest_categorical at the outer level

    # ---- Preprocess (always active) ----
    cfg["preprocess"] = {
        "distance_near_thresh":   trial.suggest_float("pp_near", 0.2, 1.5),
        "distance_far_thresh":    trial.suggest_float("pp_far", 30.0, 200.0, log=True),
        "downsample_resolution":  trial.suggest_float("pp_dsr", 0.1, 1.5, log=True),
        "k_correspondences":      trial.suggest_int("pp_k", 8, 25),
    }
    target = trial.suggest_int("pp_target", -1, 30000)  # -1 sentinel + range
    cfg["preprocess"]["random_downsample_target"] = target
    if target <= 0:
        cfg["preprocess"]["random_downsample_rate"] = trial.suggest_float(
            "pp_rate", 0.05, 0.5
        )
    # outlier removal & cropbox: leave off, not tuning

    # ---- Odometry (GPU branch — change function if triplet uses cpu/ct) ----
    odo = {
        "smoother_lag":           trial.suggest_float("odo_lag", 2.0, 12.0),
        "voxel_resolution":       trial.suggest_float("odo_vres", 0.1, 1.5, log=True),
        "full_connection_window_size": trial.suggest_int("odo_fcw", 1, 5),
        "max_num_keyframes":      trial.suggest_int("odo_maxkf", 8, 30),
        "isam2_relinearize_thresh": trial.suggest_float("odo_relth", 0.01, 0.5, log=True),
    }

    # Voxel resolution adaptive sub-tree
    vres = odo["voxel_resolution"]
    vres_max = trial.suggest_float("odo_vres_max", vres, 2.0)  # bounded ≥ vres
    odo["voxel_resolution_max"] = vres_max
    if vres_max > vres + 1e-6:   # strict inequality = adaptive on
        odo["voxel_resolution_dmin"] = trial.suggest_float("odo_dmin", 2.0, 15.0)
        odo["voxel_resolution_dmax"] = trial.suggest_float(
            "odo_dmax", odo["voxel_resolution_dmin"] + 1.0, 50.0
        )
    # else: dmin/dmax intentionally NOT suggested → not part of trial parameters

    # voxelmap_levels sub-tree
    levels = trial.suggest_int("odo_levels", 1, 3)
    odo["voxelmap_levels"] = levels
    if levels > 1:
        odo["voxelmap_scaling_factor"] = trial.suggest_float("odo_sf", 1.5, 3.0)

    # Initialization sub-tree
    init_mode = trial.suggest_categorical("odo_init_mode", ["LOOSE", "NAIVE"])
    odo["initialization_mode"] = init_mode
    if init_mode == "LOOSE":
        odo["initialization_window_size"] = trial.suggest_float(
            "odo_init_win", 0.5, 5.0
        )

    # Keyframe strategy sub-tree (exclude ENTROPY — docs say don't)
    strat = trial.suggest_categorical(
        "odo_kf_strat", ["OVERLAP", "DISPLACEMENT"]
    )
    odo["keyframe_update_strategy"] = strat
    if strat == "OVERLAP":
        odo["keyframe_max_overlap"] = trial.suggest_float("odo_kfmax", 0.5, 0.95)
    else:  # DISPLACEMENT
        odo["keyframe_delta_trans"] = trial.suggest_float("odo_dtrans", 0.5, 5.0)
        odo["keyframe_delta_rot"]   = trial.suggest_float("odo_drot",   0.1, 1.0)

    cfg["odometry_gpu"] = odo

    # ---- Sub-mapping, global mapping: same pattern as above ----
    # ... see the trees in sections 3-4 for which knobs to gate on which conditions

    return cfg
```

### Why this pattern (and not a flat search space + masking)

Three reasons it matters that you only call `suggest_*` *inside* the active branch:

1. **TPE / GP samplers learn per-parameter posteriors.** If a parameter is suggested but then ignored 50% of the time, its observed effect on the objective becomes noise. The sampler will under-explore the half where it actually matters.
2. **Trial budget.** Suggesting 60 params and only ever using 30 of them per trial means Optuna spends 2× the trials needed to converge each one.
3. **W&B logging.** If you log all params each run, the "unused" ones look like real settings in your sweep dashboard and pollute parallel-coordinate plots. The pattern above logs only the active subset, which makes the W&B view honest.

### Piping to W&B

Two clean options:

- **`optuna-integration[wandb]`**: `WeightsAndBiasesCallback` auto-logs each trial as a W&B run. Easiest if you want trial-by-trial parity with your existing `slam_sweep` dashboards.
- **Manual**: start a W&B run in your objective function, log `cfg` + the eval metric (your loop-closure RMSE), end the run. Gives you more control over `wandb.config` semantics — useful since the active parameter set differs per trial.

If you go manual, write `wandb.config` with **only the keys that were actually suggested in that trial** (not the full schema). Then parallel-coordinate views in W&B will correctly show missing axes for runs where a branch was inactive, which is exactly the visualization you want for a conditional space.

---

## 7. Suggested first-pass search budget (for the `gpu_full` triplet, with your Unitree L1)

Ranked by expected effect on loop-closure RMSE, this is what I'd actually search:

**Tier 1 — high impact, always tune (≈ 8 params):**
- `preprocess.k_correspondences`, `preprocess.downsample_resolution`, `preprocess.random_downsample_target`
- `odometry.voxel_resolution`, `odometry.voxelmap_levels`, `odometry.full_connection_window_size`, `odometry.smoother_lag`
- `global_mapping.min_implicit_loop_overlap`

**Tier 2 — medium impact, tune if Tier 1 budget allows (≈ 6 params):**
- `odometry.max_num_keyframes`, `odometry.keyframe_max_overlap` (under OVERLAP strategy)
- `global_mapping.submap_voxel_resolution`, `global_mapping.max_implicit_loop_distance`
- `sub_mapping.keyframe_voxel_resolution`, `sub_mapping.max_num_keyframes`

**Tier 3 — only if you've seen specific failure modes:**
- Adaptive voxel sub-tree (`*_max`/`_dmin`/`_dmax`) — only if your dataset has very mixed range scales
- `isam2_relinearize_thresh`, `use_isam2_dogleg` — only if you're seeing divergences
- Initialization (`LOOSE` vs `NAIVE` and the window size) — only if you suspect startup is bad

**Never in the search space:**
- All ⚙-marked items above (sensor model, extrinsics, calibration)
- Pure throughput knobs (🐢) unless you're explicitly optimizing for compute, with a multi-objective study
- Anything under `enable_*=False` if you've also fixed the enable flag to False
