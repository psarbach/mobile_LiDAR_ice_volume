"""
Optuna-driven sweep agent.

Replaces the W&B-native Bayesian optimizer (wandb agent / sweep.yaml) with
Optuna's TPE sampler, which correctly models conditional parameter trees:
parameters that are inactive for a given trial are never suggested, so the
sampler never wastes budget on dead variables.

Two parameter dicts are built per trial:
  glim_params  — only active values; passed to config_gen.materialize_config()
  wandb_params — full schema; inactive params set to None (displays as
                 NaN/missing on W&B parallel-coordinate axes)

Invoked via:
    slam-sweep optuna-run --bag ... --default-config ... --wandb-project ...
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import optuna
import wandb

from .agent import _json_default, evaluate_trial  # reuse unchanged trial runner


# ---------------------------------------------------------------------------
# Parameter suggestion
# ---------------------------------------------------------------------------

def suggest_params(trial: optuna.Trial) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Sample one trial's parameter configuration.

    Returns
    -------
    glim_params  : flat dict of {param_key: value} for every ACTIVE parameter.
                   Passed directly to config_gen.materialize_config().
    wandb_params : same dict plus None for every INACTIVE conditional parameter.
                   Passed to wandb.init(config=...) so every run has the full
                   schema — inactive axes show as missing in W&B plots.
    """
    g: dict[str, Any] = {}   # GLIM config values (active only)
    w: dict[str, Any] = {}   # W&B log values (active + None for inactive)

    def _set(key: str, value: Any) -> None:
        g[key] = value
        w[key] = value

    def _inactive(key: str) -> None:
        # None serialises to JSON null; W&B renders it as a missing point on
        # the parallel-coordinate axis — visually equivalent to NaN.
        w[key] = None

    def _r2(v: float) -> float:
        """Round to 2 decimal places — limits precision written to GLIM configs and W&B."""
        return round(v, 2)

    # ------------------------------------------------------------------
    # Unconditional parameters
    # ------------------------------------------------------------------
    _set("frontend.use_isam2_dogleg",
         trial.suggest_categorical("frontend.use_isam2_dogleg", [True, False]))
    _set("frontend.smoother_lag",
         trial.suggest_int("frontend.smoother_lag", 5, 15))              # int seconds
    _set("frontend.max_num_keyframes",
         trial.suggest_int("frontend.max_num_keyframes", 15, 30))
    _set("frontend.full_connection_window_size",
         trial.suggest_int("frontend.full_connection_window_size", 2, 15))
    _set("frontend.voxel_resolution",
         _r2(trial.suggest_float("frontend.voxel_resolution", 0.08, 0.25, log=True)))
    _set("frontend.voxelmap_levels",
         trial.suggest_int("frontend.voxelmap_levels", 2, 3))

    _set("sub_mapping.max_num_keyframes",
         trial.suggest_int("sub_mapping.max_num_keyframes", 15, 50))

    _set("global_mapping.use_isam2_dogleg",
         trial.suggest_categorical("global_mapping.use_isam2_dogleg", [True, False]))
    _set("global_mapping.submap_voxel_resolution",
         _r2(trial.suggest_float("global_mapping.submap_voxel_resolution", 0.05, 0.5, log=True)))
    _set("global_mapping.min_implicit_loop_overlap",
         _r2(trial.suggest_float("global_mapping.min_implicit_loop_overlap", 0.05, 0.25)))
    _set("global_mapping.max_implicit_loop_distance",
         trial.suggest_int("global_mapping.max_implicit_loop_distance", 80, 200))  # int metres

    # ------------------------------------------------------------------
    # Branch 1: odometry keyframe strategy
    # Only one set of children is active; the other receives None in wandb_params.
    # ------------------------------------------------------------------
    strategy = trial.suggest_categorical(
        "frontend.keyframe_update_strategy", ["OVERLAP", "DISPLACEMENT"]
    )
    _set("frontend.keyframe_update_strategy", strategy)

    if strategy == "OVERLAP":
        _set("frontend.keyframe_max_overlap",
             _r2(trial.suggest_float("frontend.keyframe_max_overlap", 0.5, 0.95)))
        _inactive("frontend.keyframe_delta_trans")
        _inactive("frontend.keyframe_delta_rot")
    else:  # DISPLACEMENT
        _set("frontend.keyframe_delta_trans",
             _r2(trial.suggest_float("frontend.keyframe_delta_trans", 0.1, 2.0, log=True)))
        _set("frontend.keyframe_delta_rot",
             _r2(trial.suggest_float("frontend.keyframe_delta_rot", 0.15, 0.5)))
        _inactive("frontend.keyframe_max_overlap")

    # ------------------------------------------------------------------
    # Branch 2: sub-mapping bundle-adjustment gate
    # keyframe_voxel_resolution is only meaningful when enable_optimization=True.
    # ------------------------------------------------------------------
    enable_opt = trial.suggest_categorical(
        "sub_mapping.enable_optimization", [True, False]
    )
    _set("sub_mapping.enable_optimization", enable_opt)

    if enable_opt:
        _set("sub_mapping.keyframe_voxel_resolution",
             _r2(trial.suggest_float("sub_mapping.keyframe_voxel_resolution", 0.1, 0.3, log=True)))
    else:
        _inactive("sub_mapping.keyframe_voxel_resolution")

    return g, w


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """Run one GLIM trial and return the loop-closure RMSE."""
    import json

    glim_params, wandb_params = suggest_params(trial)

    bag_stem = Path(args.bag).stem
    all_tags = ["optuna", args.study_name, bag_stem] + list(getattr(args, "tags", []))
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        config={"dataset": bag_stem, **wandb_params},
        reinit=True,
        tags=all_tags,
    )
    run_id = wandb.run.id if wandb.run is not None else time.strftime("optuna-%Y%m%d-%H%M%S")

    try:
        summary = evaluate_trial(args, glim_params, run_id)
    except Exception as exc:
        # Surface the error as a W&B run failure so it's visible in the dashboard,
        # then let Optuna mark the trial as failed (by re-raising).
        wandb.log({"objective": float("inf"), "error": str(exc)})
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
    wandb.run.summary["rep_errors"]       = summary["rep_errors"]
    wandb.run.summary["rep_failures"]     = summary["rep_failures"]
    wandb.run.summary["failure_breakdown"] = summary["failure_breakdown"]
    wandb.finish()

    return summary["objective"]


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------

def run_study(args: argparse.Namespace) -> None:
    runs_root = Path(args.runs_root).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    storage_path = runs_root / f"{args.study_name}.db"
    storage_url = f"sqlite:///{storage_path}"

    print(f"[optuna] study name : {args.study_name}")
    print(f"[optuna] storage    : {storage_path}")
    print(f"[optuna] n_trials   : {args.n_trials}")
    print(f"[optuna] W&B project: {args.wandb_project}")

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        load_if_exists=True,   # resume transparently after crash/stop
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    already_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if already_done:
        print(f"[optuna] resuming — {already_done} completed trial(s) already in study")

    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\n[optuna] best trial : #{best.number}  objective = {best.value:.4f} m")
    print(f"[optuna] best params:")
    for k, v in sorted(best.params.items()):
        print(f"  {k} = {v}")
