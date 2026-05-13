"""Run a single GLIM execution in a non-interactive Docker container."""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GlimRunResult:
    exit_code: int
    timed_out: bool
    log_path: Path

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


def run_glim(
    *,
    image: str,
    bag_path: Path,
    config_dir: Path,
    output_dir: Path,
    log_path: Path,
    timeout_s: int = 1800,
    auto_quit: bool = True,
    use_gpu: bool = True,
    use_display: bool = False,
    extra_docker_args: list[str] | None = None,
) -> GlimRunResult:
    """
    Spawn a fresh container, run `glim_rosbag` against `bag_path`, and exit.

    The container is started with `--rm`, so nothing persists. The bag,
    config, and output directories are bind-mounted; everything GLIM
    produces ends up in `output_dir` on the host.

    Args:
        image:       Docker image tag (e.g. "koide3/glim_ros2:jazzy_cuda13.1").
        bag_path:    Absolute path to the .mcap rosbag on the host.
        config_dir:  Absolute path to the (already materialized) config dir.
        output_dir:  Absolute path where GLIM should dump trajectories etc.
                     Created if missing.
        log_path:    Where to write the container's combined stdout/stderr.
        timeout_s:   Wall-clock cap. On expiry, the container is killed and
                     `timed_out=True` is returned.
        auto_quit:   Pass `auto_quit:=true` to GLIM so it exits after the bag.
        use_gpu:     Pass `--gpus all` to docker.
        use_display: Forward $DISPLAY and the X11 socket into the container.
                     Set True only when you want GLIM's standard_viewer to
                     pop up — i.e. an interactive sanity check. For sweeps,
                     leave False and disable libstandard_viewer.so in your
                     config.json instead.
        extra_docker_args: Optional extra args inserted before the image
                           name (e.g. ["--cpus", "8"]).

    Returns:
        GlimRunResult with exit code, timeout flag, and log path.
    """
    bag_path = Path(bag_path).resolve()
    config_dir = Path(config_dir).resolve()
    output_dir = Path(output_dir).resolve()
    log_path = Path(log_path).resolve()

    if not bag_path.is_file():
        raise FileNotFoundError(f"Bag not found: {bag_path}")
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Config dir not found: {config_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # The bag is bind-mounted by its parent directory so the path inside
    # the container is /bag/<basename>. This avoids leaking absolute host
    # paths into the container.
    #
    # NOTE on shell flags: ROS 2's setup.bash references unbound variables
    # (e.g. AMENT_TRACE_SETUP_FILES), so `set -u` makes sourcing it fail
    # immediately. We also don't want `set -e` here because we need to
    # reach the final `chmod` regardless of GLIM's exit status — we capture
    # GLIM's exit code explicitly and propagate it as the script's exit code.
    inner_cmd = (
        "source /opt/ros/jazzy/setup.bash; "
        f"ros2 run glim_ros glim_rosbag /bag/{shlex.quote(bag_path.name)} "
        "--ros-args "
        "-p config_path:=/config "
        "-p dump_path:=/output "
        f"-p auto_quit:={'true' if auto_quit else 'false'}; "
        "rc=$?; "
        # Make outputs host-readable without sudo, even if GLIM crashed.
        "chmod -R a+rwX /output 2>/dev/null || true; "
        "exit $rc"
    )

    docker_cmd: list[str] = ["docker", "run", "--rm", "--network", "host"]
    if use_gpu:
        docker_cmd += ["--gpus", "all"]
    if use_display:
        # X11 forwarding for the GLFW-based standard_viewer. On WSL with
        # X-Win32 (or VcXsrv/Xming), DISPLAY is typically set in the host
        # shell already. If DISPLAY isn't set on the host, the viewer
        # still won't open — but the rest of the run will, provided the
        # viewer plugin is disabled in config.json.
        display = os.environ.get("DISPLAY")
        if display:
            docker_cmd += ["-e", f"DISPLAY={display}"]
        # Mount the X11 socket if it exists (it does under WSLg and under
        # X-Win32 setups that follow the usual convention).
        if Path("/tmp/.X11-unix").exists():
            docker_cmd += ["-v", "/tmp/.X11-unix:/tmp/.X11-unix"]
    docker_cmd += [
        "-v", f"{bag_path.parent}:/bag:ro",
        "-v", f"{config_dir}:/config:ro",
        "-v", f"{output_dir}:/output",
    ]
    if extra_docker_args:
        docker_cmd += list(extra_docker_args)
    docker_cmd += [image, "bash", "-lc", inner_cmd]

    timed_out = False
    try:
        with log_path.open("wb") as logf:
            logf.write(b"# docker command:\n# " + " ".join(shlex.quote(a) for a in docker_cmd).encode() + b"\n\n")
            logf.flush()
            proc = subprocess.run(
                docker_cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
            )
            exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        exit_code = 124  # GNU `timeout`'s convention for timeouts.
    except FileNotFoundError as e:
        # `docker` not on PATH. Surface this clearly rather than as a
        # generic crash inside the orchestrator.
        raise RuntimeError(
            "`docker` command not found. The orchestrator runs on the host "
            "(WSL), not inside the GLIM container — make sure docker CLI is "
            "available on PATH."
        ) from e

    return GlimRunResult(exit_code=exit_code, timed_out=timed_out, log_path=log_path)
