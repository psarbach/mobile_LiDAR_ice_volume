#!/usr/bin/env python3
"""
Minimal Optuna test — 3 parameters only.

Tests the full Optuna + W&B pipeline:
  - Conditional parameter logic (keyframe_delta_rot is None/NaN in W&B
    when keyframe_update_strategy = OVERLAP)
  - SQLite study persistence (stop and re-run to resume)
  - One W&B run created per trial with correct config schema

Run from the repo's code/ directory:

    python run_optuna_test.py \\
      --bag ~/glim_sweep/data/rosbag2_2026_03_30-12_05_05_merged.mcap \\
      --default-config ~/glim_sweep/config_default/ \\
      --wandb-project slam-sweep-test \\
      --wandb-entity spiegelburg-eth-z-rich

The study is saved to <runs-root>/optuna_test.db. Re-running the same
command resumes from completed trials.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make slam_sweep importable when running from code/
sys.path.insert(0, str(Path(__file__).parent))

import optuna
import wandb

from slam_sweep.agent import evaluate_trial


def suggest_params(trial: optuna.Trial) -> tuple[dict, dict]:
    """
    Returns (glim_params, wandb_params).
    glim_params  — active values only, written to GLIM config files.
    wandb_params — full schema; inactive params set to None so W&B
                   parallel-coordinate axes show them as missing/NaN.
    """
    g: dict = {}
    w: dict = {}

    def _set(key, value):
        g[key] = value
        w[key] = value

    def _inactive(key):
        w[key] = None  # null in W&B; displays as NaN on parallel-coord axes

    # --- Always active -------------------------------------------------------
    _set("global_mapping.min_implicit_loop_overlap",
         trial.suggest_float("global_mapping.min_implicit_loop_overlap", 0.05, 0.25))

    # --- Conditional: keyframe_delta_rot only active under DISPLACEMENT ------
    strategy = trial.suggest_categorical(
        "frontend.keyframe_update_strategy", ["OVERLAP", "DISPLACEMENT"]
    )
    _set("frontend.keyframe_update_strategy", strategy)

    if strategy == "DISPLACEMENT":
        _set("frontend.keyframe_delta_rot",
             trial.suggest_float("frontend.keyframe_delta_rot", 0.15, 0.5))
    else:
        _inactive("frontend.keyframe_delta_rot")

    return g, w


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    glim_params, wandb_params = suggest_params(trial)

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        config=wandb_params,
        reinit=True,
        tags=["optuna-test"],
    )
    run_id = wandb.run.id

    try:
        summary = evaluate_trial(args, glim_params, run_id)
    except Exception as exc:
        wandb.finish(exit_code=1)
        raise optuna.exceptions.TrialPruned() from exc

    wandb.log({
        "objective":    summary["objective"],
        "rmse":         summary["rmse"],
        "mean_error":   summary["mean_error"],
        "std_error":    summary["std_error"],
        "max_error":    summary["max_error"],
        "min_error":    summary["min_error"],
        "num_failures": summary["num_failures"],
        "num_runs":     summary["num_runs"],
    })
    wandb.run.summary["rep_errors"]        = summary["rep_errors"]
    wandb.run.summary["rep_failures"]      = summary["rep_failures"]
    wandb.run.summary["failure_breakdown"] = summary["failure_breakdown"]
    wandb.finish()

    return summary["objective"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bag",            required=True, help="Absolute path to .mcap rosbag.")
    p.add_argument("--default-config", required=True, help="Path to GLIM config directory.")
    p.add_argument("--wandb-project",  required=True, help="W&B project name.")
    p.add_argument("--wandb-entity",   default=None,  help="W&B entity (team/user).")
    p.add_argument("--runs-root",      default="./runs")
    p.add_argument("--image",          default="koide3/glim_ros2:jazzy_cuda13.1")
    p.add_argument("--reps",           type=int, default=2)
    p.add_argument("--n-trials",       type=int, default=5)
    p.add_argument("--timeout-s",      type=int, default=None)
    p.add_argument("--failure-penalty-m",   type=float, default=100.0)
    p.add_argument("--runaway-threshold-m", type=float, default=500.0)
    p.add_argument("--keep-dumps",     action="store_true", default=False)
    p.add_argument("--no-display",     action="store_true", default=False,
                   help="Disable X11 forwarding (requires headless GLIM build).")
    args = p.parse_args()
    args.use_display = not args.no_display

    runs_root = Path(args.runs_root).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{runs_root / 'optuna_test.db'}"

    print(f"[test] W&B project : {args.wandb_project}")
    print(f"[test] reps/trial  : {args.reps}")
    print(f"[test] n_trials    : {args.n_trials}")
    print(f"[test] storage     : {runs_root / 'optuna_test.db'}")

    study = optuna.create_study(
        study_name="glim_test",
        storage=storage_url,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    if done:
        print(f"[test] resuming — {done} completed trial(s) already in study")

    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\n[test] best trial #{best.number}: objective = {best.value:.4f} m")
    for k, v in sorted(best.params.items()):
        print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
