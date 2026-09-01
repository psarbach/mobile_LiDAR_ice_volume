#!/usr/bin/env python3
"""
Plot objective vs. parameter value for every swept parameter of an Optuna
GLIM sweep, to reveal whether a search range was too restrictive.

For each parameter one panel is produced:

  * Numeric params  -> scatter of (param value, objective). The search
    bounds are drawn as dashed walls and the out-of-range area is shaded.
    If the good (low-objective) points pile up against a wall, the range
    was too tight on that side -> a "[!] best near LOW/HIGH bound" note is
    printed in the panel title.
  * Categorical / boolean params -> one strip column per category.

Points are coloured by objective (dark = good = low). Failed trials
(objective == FAILURE_PENALTY) are drawn as red x markers at the top so you
can also see *where in parameter space* failures happened.

Usage
-----
    python plot_param_ranges.py [DB_PATH] [OUT_DIR]

    DB_PATH  defaults to "glim_sweep_ice_tunnel.db"
    OUT_DIR  defaults to "./param_plots"

Writes one PNG per parameter into OUT_DIR plus a combined overview
"_overview.png". Pools every COMPLETE trial across all studies found in the
DB (the two ice_tunnel studies share an identical search space).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # headless-safe; remove for interactive
import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.trial import TrialState

optuna.logging.set_verbosity(optuna.logging.WARNING)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DB_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("glim_sweep_ice_tunnel.db")
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("param_plots")

FAILURE_PENALTY = 100.0       # objective value metrics.py substitutes for a failed rep
NEAR_BOUND_FRAC = 0.15        # best value within this frac of a bound -> flag "too tight"
RNG_SEED = 0                  # for integer-jitter reproducibility

# Search space, mirrored from optuna_agent.suggest_params().
#   name: (low, high, kind)   kind in {"int","lin","log","cat"}
# For "cat", the third entry is the ordered list of categories instead of a kind.
PARAM_SPEC: dict[str, tuple] = {
    # --- unconditional numeric ---
    "frontend.smoother_lag":                     (5, 15, "int"),
    "frontend.max_num_keyframes":                (15, 30, "int"),
    "frontend.full_connection_window_size":      (2, 15, "int"),
    "frontend.voxel_resolution":                 (0.08, 0.25, "log"),
    "sub_mapping.max_num_keyframes":             (15, 50, "int"),
    "global_mapping.submap_voxel_resolution":    (0.05, 0.5, "log"),
    "global_mapping.min_implicit_loop_overlap":  (0.05, 0.25, "lin"),
    "global_mapping.max_implicit_loop_distance": (80, 200, "int"),
    # --- conditional numeric (OVERLAP branch) ---
    "frontend.keyframe_max_overlap":             (0.5, 0.95, "lin"),
    # --- conditional numeric (DISPLACEMENT branch) ---
    "frontend.keyframe_delta_trans":             (0.1, 2.0, "log"),
    "frontend.keyframe_delta_rot":               (0.15, 0.5, "lin"),
    # --- conditional numeric (sub-map optimisation gate) ---
    "sub_mapping.keyframe_voxel_resolution":     (0.1, 0.3, "log"),
    # --- categorical / boolean ---
    "frontend.voxelmap_levels":                  (None, None, [2, 3]),
    "frontend.keyframe_update_strategy":         (None, None, ["OVERLAP", "DISPLACEMENT"]),
    "frontend.use_isam2_dogleg":                 (None, None, [False, True]),
    "global_mapping.use_isam2_dogleg":           (None, None, [False, True]),
    "sub_mapping.enable_optimization":           (None, None, [False, True]),
}

CMAP = plt.get_cmap("viridis_r")            # reversed: dark = low objective = good


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_trials(db_path: Path) -> list[dict]:
    """Pool every COMPLETE trial across all studies in the DB."""
    storage = f"sqlite:///{db_path}"
    rows: list[dict] = []
    for summary in optuna.get_all_study_summaries(storage=storage):
        study = optuna.load_study(study_name=summary.study_name, storage=storage)
        for t in study.trials:
            if t.state != TrialState.COMPLETE or t.value is None:
                continue
            rows.append({"value": float(t.value), **t.params})
    return rows


def col(rows, key):
    """Extract a parameter column (with objective) dropping trials where it is absent."""
    xs, ys = [], []
    for r in rows:
        if key in r and r[key] is not None:
            xs.append(r[key])
            ys.append(r["value"])
    return xs, ys


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #
def _split_fail(xs, ys):
    """Separate successful from penalty (failed) trials."""
    ok_x, ok_y, bad_x = [], [], []
    for x, y in zip(xs, ys):
        if y >= FAILURE_PENALTY - 1e-9:
            bad_x.append(x)
        else:
            ok_x.append(x)
            ok_y.append(y)
    return ok_x, ok_y, bad_x


def plot_numeric(ax, name, low, high, kind, xs, ys, vmin, vmax):
    rng = np.random.default_rng(RNG_SEED)
    ok_x, ok_y, bad_x = _split_fail(xs, ys)

    # Integer jitter to reduce overplotting on discrete axes.
    def jit(arr):
        if kind == "int" and arr:
            return np.array(arr, float) + rng.uniform(-0.18, 0.18, size=len(arr))
        return np.array(arr, float)

    top = max(ok_y) if ok_y else FAILURE_PENALTY
    ymax = top * 1.12

    # Shade the out-of-range regions so the "wall" is unmistakable.
    pad = (high - low) * 0.12 if kind != "log" else 0
    if kind == "log":
        lo_view, hi_view = low * 0.6, high * 1.6
    else:
        lo_view, hi_view = low - pad, high + pad
    ax.axvspan(lo_view, low, color="0.85", zorder=0)
    ax.axvspan(high, hi_view, color="0.85", zorder=0)
    ax.axvline(low, ls="--", lw=1.2, color="#A32D2D")
    ax.axvline(high, ls="--", lw=1.2, color="#A32D2D")

    sc = ax.scatter(jit(ok_x), ok_y, c=ok_y, cmap=CMAP, vmin=vmin, vmax=vmax,
                    s=42, edgecolor="white", linewidth=0.5, zorder=3)
    if bad_x:
        ax.scatter(jit(bad_x), [top * 1.05] * len(bad_x), marker="x",
                   color="#A32D2D", s=45, linewidth=1.6, zorder=4,
                   label=f"{len(bad_x)} failed")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.7)

    if kind == "log":
        ax.set_xscale("log")
    ax.set_xlim(lo_view, hi_view)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("objective (m)", fontsize=8)
    ax.tick_params(labelsize=7)

    # "Too restrictive?" verdict based on where the BEST trial sits.
    flag = ""
    if ok_y:
        bx = ok_x[int(np.argmin(ok_y))]
        if kind == "log":
            posn = (np.log(bx) - np.log(low)) / (np.log(high) - np.log(low))
        else:
            posn = (bx - low) / (high - low)
        if posn <= NEAR_BOUND_FRAC:
            flag = "  [!] best near LOW bound"
        elif posn >= 1 - NEAR_BOUND_FRAC:
            flag = "  [!] best near HIGH bound"
    ax.set_title(f"{name}\n[{low}, {high}] {kind}{flag}", fontsize=8.5,
                 color=("#A32D2D" if flag else "black"))
    return sc


def plot_categorical(ax, name, cats, xs, ys, vmin, vmax):
    rng = np.random.default_rng(RNG_SEED)
    ok_x, ok_y, bad_x = _split_fail(xs, ys)
    pos = {c: i for i, c in enumerate(cats)}
    top = max(ok_y) if ok_y else FAILURE_PENALTY

    def jx(vals):
        return [pos[v] + rng.uniform(-0.18, 0.18) for v in vals]

    ax.scatter(jx(ok_x), ok_y, c=ok_y, cmap=CMAP, vmin=vmin, vmax=vmax,
               s=42, edgecolor="white", linewidth=0.5, zorder=3)
    if bad_x:
        ax.scatter(jx(bad_x), [top * 1.05] * len(bad_x), marker="x",
                   color="#A32D2D", s=45, linewidth=1.6, zorder=4)
    # median marker per category
    for c in cats:
        cy = [y for x, y in zip(ok_x, ok_y) if x == c]
        if cy:
            ax.plot([pos[c] - 0.25, pos[c] + 0.25], [np.median(cy)] * 2,
                    color="#185FA5", lw=2, zorder=5)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([str(c) for c in cats], fontsize=8)
    ax.set_xlim(-0.5, len(cats) - 0.5)
    ax.set_ylim(0, top * 1.12)
    ax.set_ylabel("objective (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(f"{name}\ncategorical (blue bar = median)", fontsize=8.5)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    if not DB_PATH.exists():
        sys.exit(f"DB not found: {DB_PATH}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_trials(DB_PATH)
    real = [r["value"] for r in rows if r["value"] < FAILURE_PENALTY - 1e-9]
    vmin, vmax = (min(real), max(real)) if real else (0, FAILURE_PENALTY)
    print(f"loaded {len(rows)} completed trials "
          f"({len(rows) - len(real)} failed at penalty={FAILURE_PENALTY})")
    print(f"objective range (successful): {vmin:.2f} - {vmax:.2f} m")

    # --- combined overview grid ---
    n = len(PARAM_SPEC)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.8 * nrow))
    fig.subplots_adjust(hspace=0.55, wspace=0.32)
    axes = axes.ravel()
    last_sc = None

    for ax, (name, spec) in zip(axes, PARAM_SPEC.items()):
        xs, ys = col(rows, name)
        low, high, kind = spec
        if not xs:
            ax.set_visible(False)
            continue
        if isinstance(kind, list):
            plot_categorical(ax, name, kind, xs, ys, vmin, vmax)
        else:
            last_sc = plot_numeric(ax, name, low, high, kind, xs, ys, vmin, vmax)
        ax.set_xlabel(name, fontsize=7.5)

        # also save the panel on its own
        single, sax = plt.subplots(figsize=(5.2, 4))
        if isinstance(kind, list):
            plot_categorical(sax, name, kind, xs, ys, vmin, vmax)
        else:
            plot_numeric(sax, name, low, high, kind, xs, ys, vmin, vmax)
        sax.set_xlabel(name, fontsize=9)
        single.colorbar(plt.cm.ScalarMappable(
            norm=plt.Normalize(vmin, vmax), cmap=CMAP), ax=sax,
            label="objective (m)", fraction=0.046, pad=0.04)
        single.tight_layout()
        single.savefig(OUT_DIR / f"{name.replace('.', '__')}.png", dpi=130)
        plt.close(single)

    for ax in axes[n:]:
        ax.set_visible(False)

    if last_sc is not None:
        cbar = fig.colorbar(plt.cm.ScalarMappable(
            norm=plt.Normalize(vmin, vmax), cmap=CMAP),
            ax=axes.tolist(), fraction=0.015, pad=0.01)
        cbar.set_label("objective (m) — darker = better")
    fig.suptitle(f"Objective vs. parameter value  ({DB_PATH.name})\n"
                 "red dashed = search bound · grey = out of range · "
                 "[!] = best trial sits against a wall",
                 fontsize=13, y=0.995)
    fig.savefig(OUT_DIR / "_overview.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {n} per-parameter PNGs + _overview.png to {OUT_DIR}/")


if __name__ == "__main__":
    main()
