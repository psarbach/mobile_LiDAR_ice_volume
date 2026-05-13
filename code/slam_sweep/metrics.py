"""
Aggregate per-repetition errors into a single scalar objective for the
Bayesian optimizer, plus diagnostic stats for forensics.

Design choice: the objective is the RMSE over all repetitions, where any
failed run is substituted with a configurable penalty value. This:

  * Penalizes mean error (obvious).
  * Penalizes variance — RMSE > mean whenever there is spread, so a setting
    that is sometimes good and sometimes bad scores worse than one that is
    consistently mediocre with the same mean.
  * Heavily penalizes failures via the squared penalty term.

Why not "just sum"? The user explicitly noted that 1 lucky run + 4 crashes
is worse than 5 consistent mediocre runs. RMSE-with-substitution encodes
that: 1×0.5 + 4×100 → RMSE ≈ 89; 5×5 → RMSE = 5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class RepResult:
    """Outcome of one repetition (one full GLIM execution)."""
    error_m: float | None             # None if the run did not produce a usable trajectory.
    failed: bool
    failure_reason: str = ""          # "timeout" | "crash" | "no_trajectory" | "runaway" | ""


@dataclass
class AggregateMetrics:
    objective: float                  # The single number passed to the optimizer.
    rmse: float                       # Same as objective, named explicitly.
    mean_error: float                 # NaN if all reps failed.
    std_error: float                  # Population std (not sample); NaN if <2 successes.
    max_error: float                  # NaN if all reps failed.
    min_error: float                  # NaN if all reps failed.
    num_failures: int
    num_runs: int
    failure_breakdown: dict[str, int] = field(default_factory=dict)


def aggregate(reps: list[RepResult], failure_penalty_m: float) -> AggregateMetrics:
    n = len(reps)
    if n == 0:
        raise ValueError("aggregate() called with zero repetitions.")

    # Substituted error vector (failures replaced by the penalty value).
    substituted = [
        (failure_penalty_m if r.failed else r.error_m)
        for r in reps
    ]
    rmse = math.sqrt(sum(e * e for e in substituted) / n)

    # Stats over successful runs only (NaN-safe for the all-fail case).
    successes = [r.error_m for r in reps if not r.failed and r.error_m is not None]
    if successes:
        mean = sum(successes) / len(successes)
        var = sum((e - mean) ** 2 for e in successes) / len(successes)
        std = math.sqrt(var) if len(successes) > 1 else 0.0
        mx, mn = max(successes), min(successes)
    else:
        mean = std = mx = mn = float("nan")

    breakdown: dict[str, int] = {}
    for r in reps:
        if r.failed:
            breakdown[r.failure_reason or "unknown"] = breakdown.get(r.failure_reason or "unknown", 0) + 1

    return AggregateMetrics(
        objective=rmse,
        rmse=rmse,
        mean_error=mean,
        std_error=std,
        max_error=mx,
        min_error=mn,
        num_failures=sum(1 for r in reps if r.failed),
        num_runs=n,
        failure_breakdown=breakdown,
    )
