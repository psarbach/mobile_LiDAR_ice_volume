"""
W&B sweep agent entry point.

One trial = one parameter configuration evaluated over N repetitions.

Invocation:
    wandb agent <sweep-id>
which spawns:
    python -m slam_sweep.agent
under the hood. The trial parameters are read from `wandb.config`, NOT
from argv (we deliberately drop `${args}` from sweep.yaml's command).

Environment variables (or matching CLI flags) configure the rest:
    SLAM_SWEEP_BAG               required, absolute path to the .mcap
    SLAM_SWEEP_DEFAULT_CONFIG    required, abs path to the GLIM config dir
    SLAM_SWEEP_RUNS_ROOT         where to write per-run dirs (default ./runs)
    SLAM_SWEEP_IMAGE             docker image tag (default koide3/glim_ros2:jazzy_cuda13.1)
    SLAM_SWEEP_REPS              repetitions per trial (default 3)
    SLAM_SWEEP_TIMEOUT           seconds per repetition (default 1800)
    SLAM_SWEEP_PENALTY           failure penalty in meters (default 100.0)
    SLAM_SWEEP_RUNAWAY           runaway distance threshold (default 500.0)
    SLAM_SWEEP_KEEP_DUMPS        "1" to keep full GLIM dumps per rep
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import wandb

from . import config_gen, docker_runner, metrics, trajectory


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="slam_sweep.agent")
    p.add_argument("--bag", default=os.environ.get("SLAM_SWEEP_BAG"))
    p.add_argument("--default-config", default=os.environ.get("SLAM_SWEEP_DEFAULT_CONFIG"))
    p.add_argument("--runs-root", default=os.environ.get("SLAM_SWEEP_RUNS_ROOT", "./runs"))
    p.add_argument("--image", default=os.environ.get("SLAM_SWEEP_IMAGE", "koide3/glim_ros2:jazzy_cuda13.1"))
    p.add_argument("--reps", type=int, default=int(os.environ.get("SLAM_SWEEP_REPS", "3")))
    p.add_argument("--timeout-s", type=int, default=int(os.environ.get("SLAM_SWEEP_TIMEOUT", "1800")))
    p.add_argument("--failure-penalty-m", type=float,
                   default=float(os.environ.get("SLAM_SWEEP_PENALTY", "100.0")))
    p.add_argument("--runaway-threshold-m", type=float,
                   default=float(os.environ.get("SLAM_SWEEP_RUNAWAY", "500.0")))
    p.add_argument("--keep-dumps", action="store_true",
                   default=_bool_env("SLAM_SWEEP_KEEP_DUMPS", False))
    
    # Display forwarding defaults ON. Many GLIM builds segfault when the
    # GLFW viewer can't open a window, so we need an X server reachable
    # from inside the container. Set SLAM_SWEEP_NO_DISPLAY=1 (or pass
    # --no-display) only if your GLIM build supports headless operation
    # AND you've disabled libstandard_viewer.so in config.json.
    display_group = p.add_mutually_exclusive_group()
    display_group.add_argument("--display", dest="use_display", action="store_true",
                               default=not _bool_env("SLAM_SWEEP_NO_DISPLAY", False),
                               help="Enable display forwarding (default)")
    display_group.add_argument("--no-display", dest="use_display", action="store_false",
                               help="Disable display forwarding (requires headless-capable GLIM)")

    args = p.parse_args()


    missing = [k for k, v in [("--bag", args.bag), ("--default-config", args.default_config)] if not v]
    if missing:
        sys.exit(f"Missing required argument(s): {', '.join(missing)}")
    return args


def evaluate_trial(args: argparse.Namespace, params: dict, run_id: str) -> dict:
    """Run N reps with the given params, return a flat metrics dict."""
    runs_root = Path(args.runs_root).resolve()
    run_root = runs_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    # Log display configuration for debugging
    print(f"[slam_sweep] trial {run_id}: use_display={args.use_display}, DISPLAY={os.environ.get('DISPLAY', 'not set')}")

    # Flatten W&B's nested config into PARAM_MAP-shaped dotted keys, then
    # patch. `strict=True` turns the previously-silent "unknown key, skip"
    # path into a hard error — a sweep that patches nothing is the worst
    # kind of bug because it burns hours producing identical-looking output.
    flat_params = config_gen.flatten_params(params)
    config_dir = config_gen.materialize_config(
        default_dir=Path(args.default_config),
        target_dir=run_root / "config",
        params=flat_params,
        strict=True,
    )
    # Audit-only sidecars next to (NOT inside) the config tree. GLIM never
    # reads these; they exist purely so you can see at a glance which
    # parameters this trial received, without diffing the full JSON tree.
    # The actual config GLIM reads is the complete copy of `default_config/`
    # in `<run_root>/config/`, with the specific leaves overwritten in place.
    (run_root / "_wandb_params.json").write_text(
        json.dumps(params, indent=2, sort_keys=True, default=_json_default))
    (run_root / "_applied_patches.json").write_text(
        json.dumps(flat_params, indent=2, sort_keys=True, default=_json_default))
    print(f"[slam_sweep] trial {run_id}: patching {len(flat_params)} parameter(s):")
    for k, v in sorted(flat_params.items()):
        print(f"  {k} = {v}")

    bag_path = Path(args.bag).resolve()

    rep_results: list[metrics.RepResult] = []
    for i in range(args.reps):
        rep_dir = run_root / f"rep_{i:02d}"
        output_dir = rep_dir / "output"
        log_path = rep_dir / "glim.log"

        run = docker_runner.run_glim(
            image=args.image,
            bag_path=bag_path,
            config_dir=config_dir,
            output_dir=output_dir,
            log_path=log_path,
            timeout_s=args.timeout_s,
            use_display=getattr(args, "use_display", True),
        )

        # ---- Failure path: container crashed or timed out ------------------
        if not run.succeeded:
            reason = "timeout" if run.timed_out else f"exit_{run.exit_code}"
            rep_results.append(metrics.RepResult(error_m=None, failed=True, failure_reason=reason))
            _save_rep_summary(rep_dir, error_m=None, failed=True, reason=reason)
            continue

        # ---- No trajectory dumped ------------------------------------------
        traj_path = trajectory.find_trajectory(output_dir)
        if traj_path is None:
            rep_results.append(metrics.RepResult(error_m=None, failed=True, failure_reason="no_trajectory"))
            _save_rep_summary(rep_dir, error_m=None, failed=True, reason="no_trajectory")
            continue

        traj = trajectory.load_tum_trajectory(traj_path)
        err = trajectory.loop_closure_error(traj)
        path_len = trajectory.trajectory_length(traj)

        # ---- Runaway sanity check ------------------------------------------
        if err > args.runaway_threshold_m:
            rep_results.append(metrics.RepResult(error_m=err, failed=True, failure_reason="runaway"))
            _save_rep_summary(rep_dir, error_m=err, failed=True, reason="runaway",
                              path_length_m=path_len)
        else:
            rep_results.append(metrics.RepResult(error_m=err, failed=False))
            _save_rep_summary(rep_dir, error_m=err, failed=False, reason="",
                              path_length_m=path_len)

        # ---- Disk hygiene --------------------------------------------------
        # Copy the trajectory out, then drop the bulky dump unless asked to keep.
        shutil.copy2(traj_path, rep_dir / "trajectory.txt")
        if not args.keep_dumps:
            shutil.rmtree(output_dir, ignore_errors=True)

    agg = metrics.aggregate(rep_results, failure_penalty_m=args.failure_penalty_m)

    summary = {
        "run_id": run_id,
        "params": params,
        "num_runs": agg.num_runs,
        "num_failures": agg.num_failures,
        "failure_breakdown": agg.failure_breakdown,
        "objective": agg.objective,
        "rmse": agg.rmse,
        "mean_error": agg.mean_error,
        "std_error": agg.std_error,
        "max_error": agg.max_error,
        "min_error": agg.min_error,
        "rep_errors": [r.error_m for r in rep_results],
        "rep_failures": [r.failure_reason for r in rep_results],
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))
    return summary


def _save_rep_summary(rep_dir: Path, **fields) -> None:
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "summary.json").write_text(json.dumps(fields, indent=2, default=_json_default))


def _json_default(o):
    # NaN/Inf -> string, so the JSON stays valid.
    if isinstance(o, float):
        if o != o or o in (float("inf"), -float("inf")):
            return str(o)
    raise TypeError(f"Cannot serialize {type(o)}")


def _check_prerequisites(use_display: bool) -> None:
    """Verify Docker and (optionally) an X server are reachable; exit on failure."""
    errors: list[str] = []

    # Docker: `docker info` exits non-zero when the daemon is not running.
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
        if r.returncode != 0:
            errors.append("Docker daemon not responding — is Docker Desktop / dockerd started?")
    except FileNotFoundError:
        errors.append("'docker' not found on PATH — is Docker installed?")
    except subprocess.TimeoutExpired:
        errors.append("'docker info' timed out — Docker daemon may be stalled.")

    # X server: try Unix sockets (WSLg / local X) then TCP port 6000 (XWin32).
    if use_display:
        x_ok = any(Path(f"/tmp/.X11-unix/X{n}").exists() for n in range(4))
        if not x_ok:
            for port in [6000, 6001, 6002]:
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=2):
                        x_ok = True
                        break
                except OSError:
                    pass
        if not x_ok:
            errors.append("No X server found — start XWin32 (or VcXsrv / WSLg) before launching the sweep.")

    if errors:
        for msg in errors:
            print(f"[slam_sweep] FATAL: {msg}", file=sys.stderr)
        sys.exit(1)
    print("[slam_sweep] Pre-flight OK: Docker running" + (", X server reachable." if use_display else "."))


def main() -> None:
    args = _parse_args()

    # When running through `wandb agent`, force display enabled for GLIM.
    # This ensures the container gets proper X11/display forwarding.
    args.use_display = True

    _check_prerequisites(args.use_display)

    # `wandb.init()` reads the trial's parameter assignment from the agent.
    wandb.init()
    params = {k: v for k, v in dict(wandb.config).items() if not k.startswith("_")}
    run_id = wandb.run.id if wandb.run is not None else time.strftime("local-%Y%m%d-%H%M%S")

    summary = evaluate_trial(args, params, run_id)

    # Log scalars to W&B for the optimizer + dashboards.
    wandb.log({
        "objective":     summary["objective"],
        "rmse":          summary["rmse"],
        "mean_error":    summary["mean_error"],
        "std_error":     summary["std_error"],
        "max_error":     summary["max_error"],
        "min_error":     summary["min_error"],
        "num_failures":  summary["num_failures"],
        "num_runs":      summary["num_runs"],
    })

    # Stash the per-rep error array as a summary metric so it's visible in
    # the run page without scrolling through history.
    wandb.run.summary["rep_errors"] = summary["rep_errors"]
    wandb.run.summary["rep_failures"] = summary["rep_failures"]
    wandb.run.summary["failure_breakdown"] = summary["failure_breakdown"]

    wandb.finish()


if __name__ == "__main__":
    main()
