# slam_sweep — GLIM parameter sweeps with W&B Bayesian optimization

A small Python orchestrator that drives [GLIM](https://github.com/koide3/glim)
through Weights & Biases sweeps. Each W&B trial mutates the default GLIM
config, runs GLIM in a non-interactive Docker container N times, and reports
an aggregate trajectory error back to the optimizer.

## Architecture: where everything runs

The container is **the upstream image, unmodified**:
`koide3/glim_ros2:jazzy_cuda13.1`. Nothing is installed at runtime, nothing
is baked into a derived image. The container's only job is to run
`ros2 run glim_ros glim_rosbag`.

Everything else lives on the **host** (your WSL shell) in a Python venv:

| Component | Location |
|---|---|
| GLIM binary, CUDA, ROS 2 | upstream container |
| Config materialization (default → patched JSON tree) | host Python |
| `docker run` invocation per trial | host shell |
| Trajectory parsing (TUM format) | host Python (numpy) |
| Metric aggregation | host Python |
| W&B agent loop & Bayesian optimizer | host Python |
| Optional PLY export from a kept dump | host Python (open3d) |

The container is treated as a frozen execution sandbox. The orchestrator
mounts in:

* `/bag`   — the rosbag's parent directory, read-only
* `/config` — your patched config tree, read-only
* `/output` — where GLIM dumps trajectories and submap clouds, writable

At the end of each run a `chmod -R a+rwX /output` (coreutils, already
present in the base image) makes the dump readable from the host without
sudo.

## What changed vs. the bash pipeline

| Concern | Before (bash) | Now |
|---|---|---|
| Docker | Long-lived interactive container with `apt`/`pip` installed at runtime | `docker run --rm` per trial against the upstream image |
| Config patching | Tried (and failed) to mutate parameters from the shell | Python copies the default config tree per run and patches JSON leaves declaratively |
| Search strategy | Manual grid sweep | W&B Bayesian optimizer |
| Per-run output | Mixed in `/output` between runs | Self-contained: `runs/<run_id>/{config,rep_NN/{output,trajectory.txt,glim.log,summary.json},summary.json,params.json}` |
| Failure handling | Bash exit code only | Exit code, timeout, missing trajectory, runaway distance — all penalized in the objective |
| Repetitions | Loop in shell, no aggregation | First-class: N reps per trial, aggregated to a variance-aware objective |

## Repository layout

```
slam_sweep/
├── pyproject.toml              # Python deps: wandb, numpy, click; open3d optional
├── sweep.yaml                  # Example W&B sweep spec (Bayesian)
├── README.md
└── slam_sweep/
    ├── __init__.py
    ├── agent.py                # W&B agent entry point (one trial = one config × N reps)
    ├── cli.py                  # `slam-sweep` CLI (run-once, export-ply)
    ├── config_gen.py           # Default-config copy + JSON leaf patcher; PARAM_MAP
    ├── docker_runner.py        # Non-interactive `docker run` for one GLIM execution
    ├── trajectory.py           # TUM-format trajectory parsing + loop-closure error
    └── metrics.py              # Multi-rep aggregation (RMSE-with-failure-penalty)
```

## One-time setup

```bash
# 1. Pull the upstream GLIM image (no build, no derived image):
docker pull koide3/glim_ros2:jazzy_cuda13.1

# 2. Install the orchestrator in your WSL Python:
python3 -m venv ~/.venvs/slam_sweep
source ~/.venvs/slam_sweep/bin/activate
pip install -e .                 # core
pip install -e .[ply]            # also installs open3d for export-ply
wandb login                       # one-time

# 3. Make a writeable copy of GLIM's default config to use as the baseline:
cp -r ~/glim/config_default ~/glim/config_baseline
```

## Running a sweep

```bash
# 1. Edit sweep.yaml — pick the parameters you actually want to search.
#    Every key must exist in slam_sweep/config_gen.py PARAM_MAP.

# 2. Register the sweep with W&B:
wandb sweep sweep.yaml
# This prints a sweep ID like 'user/glim-sweep/abc123'.

# 3. Launch one or more agents. One worker per GPU — GLIM saturates the GPU.
export SLAM_SWEEP_BAG=/abs/path/to/your.mcap
export SLAM_SWEEP_DEFAULT_CONFIG=/home/user/glim/config_baseline
export SLAM_SWEEP_RUNS_ROOT=/home/user/slam_sweep_runs
export SLAM_SWEEP_REPS=3                 # repetitions per trial
export SLAM_SWEEP_TIMEOUT=2400           # seconds per repetition
wandb agent <your-sweep-id>
```

Stop the agent with Ctrl-C. To launch a worker on a second machine, copy
the env vars and rerun `wandb agent` with the same sweep ID.

## Single-run sanity check (no W&B)

Useful before launching a sweep — confirms the docker plumbing works and
the default config produces a parseable trajectory.

```bash
slam-sweep run-once \
  --bag /abs/path/to/your.mcap \
  --default-config /home/user/glim/config_baseline \
  --runs-root ./runs --reps 1
```

## Exporting a PLY from a finished run

The orchestrator strips dumps by default to save disk. Re-run the trial
of interest with `--keep-dumps` (or set `SLAM_SWEEP_KEEP_DUMPS=1`), then:

```bash
slam-sweep export-ply --rep-dir runs/<run_id>/rep_00 --out map.ply
slam-sweep export-ply --rep-dir runs/<run_id>/rep_00 --out map.ply --voxel-size 0.05
```

This reads the dumped submap PLYs on the host and concatenates them with
Open3D. The container is not involved.

## On the metric

For each repetition we compute the loop-closure error
`||t_end - t_start||` from `traj_lidar.txt`. The aggregate objective is

```
errors = [e if not failed else PENALTY for e in repetitions]
objective = sqrt(mean(errors^2))      # RMSE with failure substitution
```

Properties:

- **Penalizes mean error.** Obvious.
- **Penalizes variance.** RMSE > mean whenever there is spread, so a
  parameter setting that is sometimes good and sometimes bad scores worse
  than one that is consistently mediocre with the same mean.
- **Heavily penalizes failures.** A failed rep contributes `PENALTY^2` to
  the squared sum. With `PENALTY=100 m`: 1 success at 0.5 m + 4 crashes →
  RMSE ≈ 89 m; 5 consistent runs at 5 m → RMSE = 5 m. Consistent
  mediocrity beats lucky-but-crashy by a wide margin.
- **Tunable.** `--failure-penalty-m` sets the substitution value. Pick it
  ~10–100× the worst error you'd consider acceptable.

The full per-rep error list, `num_failures`, and a failure breakdown
(`timeout` / `exit_<N>` / `no_trajectory` / `runaway`) are also logged to
W&B.

## On multiple datasets

Each sweep optimizes for one dataset. Pooling errors across datasets with
very different scales (an indoor courtyard vs. a glacier outline) gives
the optimizer noisy gradients.

Recommended workflow:

1. Run a sweep on dataset A.
2. Take A's best config and run a *short* sweep on dataset B with a
   tightened range around it.
3. If the per-dataset optima diverge, that divergence is the most useful
   diagnostic — it tells you which params are environment-dependent.

A single multi-dataset objective (averaging per-dataset RMSEs inside one
trial) is possible but multiplies cost by the number of datasets. Skip it
until per-dataset baselines exist.

## Things this intentionally does not do

- It does not auto-merge MCAP chunks. Run `merge_chunks_ps.sh` once,
  ahead of time. The orchestrator assumes a single `.mcap`.
- It does not produce ground-truth-relative evaluation (APE/RPE). If you
  later have GT trajectories, swap the metric in
  `slam_sweep/trajectory.py`. Loop closure is a stand-in.
- It does not ship a viewer. Use GLIM's `offline_viewer` against any
  `runs/*/rep_*/output/` dump (run `--keep-dumps` first).
