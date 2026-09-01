"""
Command 2 of 2 — the volume error estimate, from the spread of repeated runs.

The uncertainty here is MEASURED, not modelled. The same tunnel is walked N
times; each walk is processed independently by `run_pipeline.py` into its own
result directory; the scatter of those N volumes is the uncertainty of one
volume measurement. Per method, so each method carries its own repeatability,
and per dataset, so the two campaigns are never pooled (different tunnel,
different reference scan — pooling them would report seasonal change as noise).

Two numbers are reported, side by side and never added in quadrature:

  REPEATABILITY (random, 1σ) — the SD across reps. This is what changes when you
      walk the tunnel again with the same sensor and the same processing.
  BIAS vs the Leica reference (systematic) — the mean Livox−Leica offset. It does
      not shrink with more reps: walking again cannot fix a wall the Mid-360
      never saw. Adding it to the SD would misreport a fixed offset as scatter.

If the campaign was run with `--cloud both`, the Leica reference was re-measured
in every rep and gets a repeatability row of its own. Read it as the PROCESSING
FLOOR, not as a second measurement: the reference cloud is identical across
those runs and only the trajectory — hence the fitted centreline — differs, so
its spread is what the pipeline contributes to a volume whose input never
changed. The report then also gains a paired Livox−Leica table, in which both
volumes come from the same run and the centreline's contribution cancels.

Deliberately absent: any modelled error term. An earlier version enumerated
sensor noise, discretisation, scale and coverage terms and ran a Monte-Carlo over
them. It was removed: the model can only ever contain the errors we thought of,
and it silently missed anything the methods get wrong in common. Repeated runs
contain every error that varies run to run, whether or not anyone modelled it.
What repeats CANNOT see is an error common to all reps — that is precisely what
the Leica bias column, and the between-method spread, are for.

Usage
-----
    python run_statistics.py                              # every dataset in results/
    python run_statistics.py --dataset April_12_05_05     # one campaign
    python run_statistics.py --all                        # every saved run, not
                                                          # just the newest per rep
"""

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps

log = logging.getLogger(__name__)

# Method key in summary.json -> label for tables and figures. "profiles_simpson"
# is deliberately NOT here: it is the profiles method under a second integration
# rule, so treating it as an independent estimator would double-count it.
METHODS = {
    "profiles": "profiles",
    "surface_mesh": "surface mesh",
    "hull_bound": "convex hull (UB)",
    "marching_cubes": "marching cubes",
}
CLOUDS = ("livox", "leica")

# Config/input fields that must agree across the runs being compared. A volume
# is only comparable to another volume measured over the same domain, with the
# same binning, from the same reference scan — cap_mode exists precisely so a
# differently-capped run is never silently averaged in with this one.
MUST_MATCH_CONFIG = (
    "cap_mode", "geometry_mode", "domain_start_target_idx",
    "domain_end_target_idx", "theta_reference", "profile_ds_m",
    "profile_dtheta_deg", "centreline_traj_resample_ds_m",
    "centreline_smoothing_factor", "centreline_resample_ds_m", "golden_only",
)
MUST_MATCH_INPUTS = ("leica_ply", "targets_leica")


# --------------------------------------------------------------------------- #
#  Collecting runs                                                            #
# --------------------------------------------------------------------------- #

@dataclass(eq=False)           # identity hash: runs are de-duplicated by object
class Run:
    path: Path                 # …/<dataset>/<rep>/<stamp>/summary.json
    dataset: str
    rep: str
    stamp: str                 # directory name = UTC timestamp, so sortable
    summary: dict

    @property
    def clouds(self) -> List[str]:
        return list(self.summary.get("clouds", []))

    def volume(self, cloud: str, method: str) -> Optional[float]:
        c = self.summary.get("results", {}).get(cloud)
        return None if c is None else c["volumes"].get(method)


def load_runs(results_dir: Path, dataset: str) -> List[Run]:
    runs = []
    for p in sorted((results_dir / dataset).glob("*/*/summary.json")):
        try:
            summary = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Skipping unreadable %s (%s)", p, exc)
            continue
        if not summary.get("results"):
            log.warning("Skipping %s — no volumes in it (run failed?)", p)
            continue
        runs.append(Run(path=p, dataset=dataset, rep=p.parent.parent.name,
                        stamp=p.parent.name, summary=summary))
    return runs


def select_runs(runs: List[Run], use_all: bool) -> List[Run]:
    """Newest run per (rep, cloud) — or every run, with --all.

    Re-running one rep is not a new sample of the tunnel: the inputs are
    identical and the pipeline is deterministic, so counting it twice would pull
    the SD down and inflate n. Newest-per-rep also means a rep re-run after a
    config change replaces its predecessor instead of being mixed with it.
    """
    if use_all:
        return sorted(runs, key=lambda r: (r.rep, r.stamp))
    keep: Dict[str, Run] = {}
    for r in sorted(runs, key=lambda r: r.stamp):
        for cloud in r.clouds:
            keep[f"{r.rep}/{cloud}"] = r       # later stamp wins
    return sorted(set(keep.values()), key=lambda r: (r.rep, r.stamp))


def check_comparable(runs: List[Run], strict: bool) -> List[str]:
    """Every selected run must share the settings that define the measurement.

    Keyed by rep AND stamp, never by rep alone. One rep routinely contributes
    two selected runs — `select_runs` keeps the newest per (rep, cloud), so a
    rep whose newest Livox run and newest Leica run are different directories
    appears twice, as does every rep under --all. Keyed by rep, the second run
    silently overwrote the first in this dict and the comparison was made
    against a set of one value per rep, so a difference BETWEEN two runs of the
    same rep could not be seen. That is the exact case this guard exists for: it
    let a target-capped Leica volume be averaged with a feature-capped one, and
    reported the 2 m domain change as a 2.5% measurement scatter.
    """
    problems = []
    for key in MUST_MATCH_CONFIG:
        vals = {f"{r.rep}/{r.stamp}": r.summary.get("config", {}).get(key)
                for r in runs}
        if len(set(map(repr, vals.values()))) > 1:
            problems.append(f"config.{key} differs across runs: {vals}")

    # The CAP POINTS themselves, not just cap_mode. Two runs can both be
    # feature_planes and still be capped in different places: the file is always
    # called caps.txt, so re-picking a feature and re-running only some reps
    # would leave cap_mode agreeing while the domain silently moved. Compared as
    # coordinates rounded to 0.1 mm, which is far below the ~10 mm picking
    # repeatability and so cannot fire on formatting alone.
    fingerprints = {}
    for r in runs:
        caps = (r.summary.get("domain") or {}).get("caps") or {}
        pts = caps.get("caps") or []
        fingerprints[f"{r.rep}/{r.stamp}"] = (
            caps.get("caps_file"),
            tuple(tuple(round(float(v), 4) for v in p["xyz"]) for p in pts),
        )
    if len(set(map(repr, fingerprints.values()))) > 1:
        problems.append(f"the domain end caps differ across runs: {fingerprints}")
    for key in MUST_MATCH_INPUTS:
        vals = {}
        for r in runs:
            item = (r.summary.get("inputs") or {}).get(key)
            vals[f"{r.rep}/{r.stamp}"] = (
                None if item is None else (item["path"], item["bytes"]))
        if len(set(map(repr, vals.values()))) > 1:
            problems.append(f"inputs.{key} differs across runs: {vals}")
    if problems:
        msg = ("These runs were not made the same way, so their spread would "
               "mix processing changes with real run-to-run variation:\n  - "
               + "\n  - ".join(problems))
        if strict:
            raise SystemExit(
                msg + "\n\nEither re-run the odd ones out so every rep is "
                "processed identically, or pass --allow-mixed-config to compute "
                "the statistics anyway (the report will carry the warning)."
            )
        log.warning("%s", msg)
    return problems


# --------------------------------------------------------------------------- #
#  Statistics                                                                 #
# --------------------------------------------------------------------------- #

def grubbs(values: np.ndarray, alpha: float = 0.05) -> dict:
    """Grubbs' test for a single outlier — the honest check available at n=5.

    G = max|x−x̄|/s against the two-sided critical value. With n=5 the test only
    catches a gross outlier (a rep processed from the wrong file, say); it cannot
    police the third digit, and it is reported as a flag, never used to silently
    drop a rep.
    """
    n = len(values)
    if n < 3:
        return {"n": n, "applicable": False}
    sd = float(np.std(values, ddof=1))
    if sd == 0:
        return {"n": n, "applicable": True, "G": 0.0, "G_crit": None,
                "outlier_index": None, "flagged": False}
    g = float(np.max(np.abs(values - values.mean())) / sd)
    t = sps.t.ppf(1 - alpha / (2 * n), n - 2)
    g_crit = float((n - 1) / np.sqrt(n) * np.sqrt(t ** 2 / (n - 2 + t ** 2)))
    return {"n": n, "applicable": True, "G": g, "G_crit": g_crit,
            "outlier_index": int(np.argmax(np.abs(values - values.mean()))),
            "flagged": bool(g > g_crit)}


def describe(values: List[float], alpha: float = 0.05) -> dict:
    """Sample statistics of one method's repeated volumes."""
    v = np.asarray([x for x in values if x is not None], dtype=float)
    n = len(v)
    out = {"n": n, "mean": float(v.mean()) if n else None,
           "min": float(v.min()) if n else None,
           "max": float(v.max()) if n else None}
    if n < 2:
        out.update(sd=None, cv_pct=None, sem=None, ci95_lo=None, ci95_hi=None,
                   range=None, sd_ci95_lo=None, sd_ci95_hi=None,
                   grubbs=grubbs(v))
        return out
    sd = float(v.std(ddof=1))
    sem = sd / np.sqrt(n)
    t95 = float(sps.t.ppf(1 - alpha / 2, n - 1))
    # A 5-sample SD is itself uncertain by ~35% (1/sqrt(2(n-1))); the chi-square
    # interval below is the honest way to say so, and it is why the SD is quoted
    # to 2 significant figures and not more.
    chi_lo = sps.chi2.ppf(1 - alpha / 2, n - 1)
    chi_hi = sps.chi2.ppf(alpha / 2, n - 1)
    out.update(
        sd=sd,
        cv_pct=sd / out["mean"] * 100 if out["mean"] else None,
        sem=float(sem),
        ci95_lo=out["mean"] - t95 * sem, ci95_hi=out["mean"] + t95 * sem,
        range=float(v.max() - v.min()),
        sd_ci95_lo=float(sd * np.sqrt((n - 1) / chi_lo)),
        sd_ci95_hi=float(sd * np.sqrt((n - 1) / chi_hi)),
        grubbs=grubbs(v),
    )
    return out


def analyse(runs: List[Run]) -> dict:
    """Per-cloud, per-method statistics + the Livox−Leica bias, for one dataset."""
    # One x-axis slot per RUN, not per rep: with --all the same rep can appear
    # twice, and collapsing those two onto one slot would hide the very samples
    # --all was asked for.
    n_per_rep: Dict[str, int] = {}
    for r in runs:
        n_per_rep[r.rep] = n_per_rep.get(r.rep, 0) + 1
    key_of = {id(r): (r.rep if n_per_rep[r.rep] == 1 else f"{r.rep}@{r.stamp}")
              for r in runs}
    labels = [key_of[id(r)] for r in runs]

    # ---- per-run values, per cloud/method ----
    samples: Dict[str, Dict[str, List[dict]]] = {}
    for cloud in CLOUDS:
        samples[cloud] = {}
        for method in METHODS:
            rows = []
            for r in runs:
                v = r.volume(cloud, method)
                if v is not None:
                    rows.append({"rep": r.rep, "stamp": r.stamp,
                                 "key": key_of[id(r)], "value": v})
            if rows:
                samples[cloud][method] = rows

    stats_out: Dict[str, Dict[str, dict]] = {}
    for cloud, by_method in samples.items():
        stats_out[cloud] = {m: describe([row["value"] for row in rows])
                            for m, rows in by_method.items()}

    # ---- bias: Livox minus the Leica reference, per method ----
    # Against the MEAN Leica volume. When the Leica scan was processed once its
    # volume is a single value and the bias inherits the Livox scatter alone;
    # when it was re-processed per rep (--cloud both) the reference is itself a
    # mean of n and the paired comparison below is the sharper statement.
    bias = {}
    for method in METHODS:
        liv = samples.get("livox", {}).get(method)
        lei = stats_out.get("leica", {}).get(method)
        if not liv or not lei or lei["mean"] is None:
            continue
        ref = lei["mean"]
        d = np.array([row["value"] - ref for row in liv], dtype=float)

        # Paired: both clouds out of the SAME run, so they shared a centreline.
        # Whatever the centreline fit contributed cancels in the difference,
        # which the unpaired comparison above cannot do. Only available when
        # Leica was re-measured per rep.
        pairs = []
        for r in runs:
            v_liv, v_lei = r.volume("livox", method), r.volume("leica", method)
            if v_liv is not None and v_lei is not None:
                pairs.append({"rep": r.rep, "key": key_of[id(r)],
                              "livox_m3": v_liv, "leica_m3": v_lei,
                              "delta_m3": v_liv - v_lei,
                              "delta_pct": (v_liv - v_lei) / v_lei * 100})
        paired = None
        if len(pairs) >= 2:
            dp = np.array([p["delta_m3"] for p in pairs], dtype=float)
            paired = {
                "n": len(pairs),
                "mean_delta_m3": float(dp.mean()),
                "mean_delta_pct": float(np.mean([p["delta_pct"] for p in pairs])),
                "sd_delta_m3": float(dp.std(ddof=1)),
                "per_rep": pairs,
            }

        bias[method] = {
            "leica_reference_m3": ref,
            "leica_n": lei["n"],
            "mean_delta_m3": float(d.mean()),
            "mean_delta_pct": float(d.mean() / ref * 100),
            "sd_delta_m3": float(d.std(ddof=1)) if len(d) > 1 else None,
            "paired": paired,
            "per_rep": [{"rep": row["rep"], "key": row["key"],
                         "delta_m3": float(row["value"] - ref),
                         "delta_pct": float((row["value"] - ref) / ref * 100)}
                        for row in liv],
        }

    # ---- run-level quantities that must also be stable across reps ----
    def per_run(getter):
        vals = []
        for r in runs:
            try:
                x = getter(r.summary)
            except (KeyError, TypeError):
                x = None
            if x is not None:
                vals.append({"rep": r.rep, "value": float(x)})
        return vals

    aux_series = {
        "domain_L_m": per_run(lambda s: s["domain"]["L_m"]),
        "centreline_fit_rms_m": per_run(lambda s: s["centreline"]["fit_rms_m"]),
        "centreline_kappa_max_1pm": per_run(lambda s: s["centreline"]["kappa_max_1pm"]),
        "target_scale_k": per_run(
            lambda s: s["targets"]["scale_livox_over_leica"]["k"]),
        "livox_gap_frac_missing": per_run(
            lambda s: s["results"]["livox"]["gap_frac_missing"]),
        "livox_frac_interp_mean": per_run(
            lambda s: s["results"]["livox"]["frac_interp_mean"]),
    }
    aux_stats = {k: describe([row["value"] for row in v])
                 for k, v in aux_series.items() if v}

    # ---- between-method spread WITHIN each rep -------------------------------
    # profiles and surface-mesh share the r(s,θ) extraction and differ only in
    # integration + hole-filling, so this is not an independent check — it is a
    # sensitivity, and it is reported separately from the repeatability for
    # exactly that reason (the hull bound is excluded: it is a bound, not an
    # estimator, so its offset is not an error).
    within = []
    for r in runs:
        vp = r.volume("livox", "profiles")
        vm = r.volume("livox", "surface_mesh")
        if vp and vm:
            within.append({"rep": r.rep, "delta_pct": (vm - vp) / vp * 100})
    within_stats = describe([w["delta_pct"] for w in within]) if within else None

    return {"reps": sorted({r.rep for r in runs}), "labels": labels,
            "samples": samples, "stats": stats_out, "bias": bias,
            "aux_series": aux_series, "aux_stats": aux_stats,
            "within_run_mesh_vs_profiles_pct": {"per_rep": within,
                                                "stats": within_stats}}


# --------------------------------------------------------------------------- #
#  Reporting                                                                  #
# --------------------------------------------------------------------------- #

def _fmt(x, nd=2, dash="—"):
    return dash if x is None else f"{x:.{nd}f}"


def print_report(dataset: str, res: dict, runs: List[Run], problems: List[str]) -> str:
    """Console/markdown report. Returns the text so it can also be written out."""
    L = []
    add = L.append
    add(f"# Volume statistics — {dataset}")
    add("")
    add(f"Runs used: {len(runs)}  (reps: {', '.join(res['reps'])})")
    for r in runs:
        add(f"  - {r.rep}  {r.stamp}  clouds={'+'.join(r.clouds)}  "
            f"({r.path.parent.name})")
    if problems:
        add("")
        add("**WARNING — runs are not identically configured:**")
        for p in problems:
            add(f"  - {p}")

    add("")
    add("## Repeatability across reps (random error, 1σ)")
    add("")
    add(f"| cloud | method | n | mean [m³] | SD [m³] | CV [%] | "
        f"95% CI of mean [m³] | min–max [m³] | SD 95% CI [m³] | Grubbs |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for cloud in CLOUDS:
        for method, label in METHODS.items():
            st = res["stats"].get(cloud, {}).get(method)
            if not st:
                continue
            g = st.get("grubbs") or {}
            g_str = ("n/a" if not g.get("applicable")
                     else ("OUTLIER" if g.get("flagged")
                           else f"ok (G={_fmt(g.get('G'))}<{_fmt(g.get('G_crit'))})"))
            ci = (f"[{_fmt(st['ci95_lo'])}, {_fmt(st['ci95_hi'])}]"
                  if st["ci95_lo"] is not None else "—")
            sdci = (f"[{_fmt(st['sd_ci95_lo'])}, {_fmt(st['sd_ci95_hi'])}]"
                    if st.get("sd_ci95_lo") is not None else "—")
            add(f"| {cloud} | {label} | {st['n']} | {_fmt(st['mean'])} | "
                f"{_fmt(st['sd'])} | {_fmt(st['cv_pct'], 3)} | {ci} | "
                f"{_fmt(st['min'])}–{_fmt(st['max'])} | {sdci} | {g_str} |")

    n_leica = max((st["n"] for st in res["stats"].get("leica", {}).values()),
                  default=0)
    if n_leica >= 2:
        add("")
        add(f"**The Leica rows have n = {n_leica}, and they are not repeated "
            "measurements of the tunnel.** The reference cloud is one scan, "
            "used unchanged in every rep; what differs between those runs is "
            "only the rep's trajectory, hence the centreline fitted from it. "
            "So the Leica SD is the processing's own contribution — how much "
            "the volume of a *fixed* cloud moves when the centreline is "
            "re-fitted — and it is a floor under the Livox SD in the same "
            "table, not a rival to it. Livox SD ≫ Leica SD means the scatter "
            "is the walk and the coverage; Livox SD ≈ Leica SD would mean the "
            "processing is what you are measuring.")

    add("")
    add("## Bias vs the Leica reference (systematic — does NOT shrink with n)")
    add("")
    if res["bias"]:
        add("| method | Leica ref [m³] | mean Livox−Leica [m³] | [%] | SD of Δ [m³] |")
        add("|---|---|---|---|---|")
        for method, b in res["bias"].items():
            add(f"| {METHODS[method]} | {_fmt(b['leica_reference_m3'])} | "
                f"{b['mean_delta_m3']:+.2f} | {b['mean_delta_pct']:+.2f} | "
                f"{_fmt(b['sd_delta_m3'])} |")
        add("")
        if n_leica < 2:
            add("The SD of Δ equals the Livox repeatability above (the "
                "reference is a single value), so it is not new information — "
                "it is listed to make clear that the offset is fixed and the "
                "scatter is Livox's.")
        else:
            add(f"The reference is itself a mean of n = {n_leica} runs here, so "
                "the SD of Δ is no longer just Livox's. The paired table below "
                "is the sharper comparison.")

        paired_rows = [(m, b["paired"]) for m, b in res["bias"].items()
                       if b.get("paired")]
        if paired_rows:
            add("")
            add("### Paired within a run (both clouds through the same centreline)")
            add("")
            add("| method | n | mean Livox−Leica [m³] | [%] | SD of Δ [m³] |")
            add("|---|---|---|---|---|")
            for method, p in paired_rows:
                add(f"| {METHODS[method]} | {p['n']} | "
                    f"{p['mean_delta_m3']:+.2f} | {p['mean_delta_pct']:+.2f} | "
                    f"{_fmt(p['sd_delta_m3'])} |")
            add("")
            add("Both volumes in each of these differences were computed on the "
                "same rep's centreline, so the centreline's own contribution "
                "cancels and what is left is the difference between the two "
                "clouds. If this SD is smaller than the unpaired one, the "
                "centreline was a shared source of scatter; if they agree, it "
                "was not.")
    else:
        add("No Leica reference volume in these runs — run one rep with "
            "`--cloud both` to get the bias column.")

    add("")
    add("## Headline")
    add("")
    for method, label in METHODS.items():
        st = res["stats"].get("livox", {}).get(method)
        if not st or st["n"] < 2:
            continue
        b = res["bias"].get(method)
        line = (f"- **{label}**: V = {st['mean']:.1f} ± {st['sd']:.1f} m³ "
                f"({st['cv_pct']:.2f}% 1σ, n={st['n']})")
        if b:
            line += (f"; offset from Leica {b['mean_delta_m3']:+.1f} m³ "
                     f"({b['mean_delta_pct']:+.2f}%, systematic)")
        add(line)
    if n_leica >= 2:
        add("")
        for method, label in METHODS.items():
            st = res["stats"].get("leica", {}).get(method)
            if not st or st["n"] < 2:
                continue
            add(f"- **{label}, Leica reference re-processed {st['n']}×**: "
                f"V = {st['mean']:.1f} ± {st['sd']:.2f} m³ "
                f"({st['cv_pct']:.3f}% 1σ) — one fixed cloud through "
                f"{st['n']} different centrelines, i.e. the processing's own "
                "scatter")
    add("")
    add("Random and systematic are quoted separately and must not be added in "
        "quadrature: the first is scatter you can average down with more walks, "
        "the second is a fixed offset that more walks cannot touch.")

    wr = res["within_run_mesh_vs_profiles_pct"]["stats"]
    if wr and wr["n"] >= 1:
        add("")
        add("## Between-method spread within a run (Livox: surface mesh vs profiles)")
        add("")
        add(f"mean {wr['mean']:+.2f}%"
            + (f", SD {wr['sd']:.2f}%, n={wr['n']}" if wr["sd"] is not None
               else f", n={wr['n']}"))
        add("")
        add("Reported separately from everything above. These two methods share "
            "the r(s,θ) extraction and differ only in integration and "
            "hole-filling, so their gap measures hole-fill sensitivity, not "
            "independent agreement.")

    if res["aux_stats"]:
        add("")
        add("## Run-to-run stability of the inputs to the volume")
        add("")
        add("| quantity | n | mean | SD | CV [%] | min–max |")
        add("|---|---|---|---|---|---|")
        for key, st in res["aux_stats"].items():
            if st["n"] == 0:
                continue
            add(f"| {key} | {st['n']} | {_fmt(st['mean'], 4)} | "
                f"{_fmt(st['sd'], 4)} | {_fmt(st['cv_pct'], 3)} | "
                f"{_fmt(st['min'], 4)}–{_fmt(st['max'], 4)} |")
        add("")
        add("`domain_L_m` is the length each rep's own centreline gives between "
            "the two end caps (surveyed targets, or the shared features of "
            "caps.txt — see `cap_mode`). Volume scales with it, so a CV here "
            "that is comparable to the volume CV above says the scatter is "
            "length, not shape.")

    text = "\n".join(L) + "\n"
    print("\n" + text)
    return text


def write_csvs(out_dir: Path, dataset: str, res: dict, runs: List[Run]) -> None:
    # Empty cell, not the em-dash the tables use: these files get read by a
    # spreadsheet or pandas, where "—" turns a numeric column into text.
    def c(x, nd=4):
        return _fmt(x, nd, dash="")

    with (out_dir / "runs.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "rep", "stamp", "cloud", "method", "volume_m3",
                    "domain_L_m", "result_dir"])
        for r in runs:
            for cloud in r.clouds:
                for method in METHODS:
                    v = r.volume(cloud, method)
                    if v is None:
                        continue
                    w.writerow([dataset, r.rep, r.stamp, cloud, method,
                                f"{v:.4f}",
                                f"{r.summary['domain']['L_m']:.4f}",
                                r.summary.get("result_dir", "")])

    with (out_dir / "summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "cloud", "method", "n", "mean_m3", "sd_m3",
                    "cv_pct", "sem_m3", "ci95_lo_m3", "ci95_hi_m3", "min_m3",
                    "max_m3", "bias_vs_leica_m3", "bias_vs_leica_pct"])
        for cloud in CLOUDS:
            for method in METHODS:
                st = res["stats"].get(cloud, {}).get(method)
                if not st:
                    continue
                b = res["bias"].get(method) if cloud == "livox" else None
                w.writerow([
                    dataset, cloud, method, st["n"],
                    c(st["mean"]), c(st["sd"]), c(st["cv_pct"]), c(st["sem"]),
                    c(st["ci95_lo"]), c(st["ci95_hi"]), c(st["min"]), c(st["max"]),
                    c(b["mean_delta_m3"]) if b else "",
                    c(b["mean_delta_pct"]) if b else "",
                ])


# --------------------------------------------------------------------------- #
#  Figure                                                                     #
# --------------------------------------------------------------------------- #

METHOD_COLORS = {"profiles": "steelblue", "surface_mesh": "seagreen",
                 "hull_bound": "salmon", "marching_cubes": "purple"}


def plot_statistics(dataset: str, res: dict, save_path: Path) -> None:
    """Three panels: the per-rep volumes, their spread, and the Leica offset."""
    labels = res["labels"]
    x = np.arange(len(labels))
    methods = [m for m in METHODS if res["stats"].get("livox", {}).get(m)]
    n_leica = max((st["n"] for st in res["stats"].get("leica", {}).values()),
                  default=0)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    # -- A: every rep's volume, per method, with the mean ± SD band ----------
    ax = axes[0]
    for m in methods:
        st = res["stats"]["livox"][m]
        by_key = {row["key"]: row["value"] for row in res["samples"]["livox"][m]}
        y = [by_key.get(k, np.nan) for k in labels]
        c = METHOD_COLORS.get(m, "grey")
        ax.plot(x, y, "o-", color=c, lw=1.2, ms=6, label=f"Livox {METHODS[m]}")
        if st["sd"] is not None:
            ax.axhspan(st["mean"] - st["sd"], st["mean"] + st["sd"],
                       color=c, alpha=0.12)
            ax.axhline(st["mean"], color=c, lw=0.8, ls="-", alpha=0.6)
        lei = res["stats"].get("leica", {}).get(m)
        if lei and lei["mean"] is not None:
            ax.axhline(lei["mean"], color=c, lw=1.4, ls="--", alpha=0.9)
        # With the reference re-processed per rep, draw those runs too: a flat
        # line of open markers against a scattered filled one is the whole
        # "processing vs walk" result in a single glance.
        lei_rows = res["samples"].get("leica", {}).get(m) if n_leica >= 2 else None
        if lei_rows:
            by_key = {row["key"]: row["value"] for row in lei_rows}
            ax.plot(x, [by_key.get(k, np.nan) for k in labels], "o:",
                    color=c, lw=1.0, ms=6, mfc="none", alpha=0.9,
                    label=f"Leica {METHODS[m]} (n={len(lei_rows)})")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Volume [m³]")
    ax.set_title("Per-rep volume\nband = mean ± 1σ, dashed = Leica reference")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.25)

    # -- B: the repeatability itself, with how well 5 samples pin it ---------
    ax = axes[1]
    paired_bars = n_leica >= 2
    w = 0.34 if paired_bars else 0.6
    for i, m in enumerate(methods):
        c = METHOD_COLORS.get(m, "grey")
        for k, cloud in enumerate(("livox", "leica") if paired_bars
                                  else ("livox",)):
            st = res["stats"].get(cloud, {}).get(m)
            if not st or st["cv_pct"] is None:
                continue
            pos = i + (k - 0.5) * w if paired_bars else i
            # Same method colour for both clouds; the reference is hatched and
            # hollow so it reads as a floor line rather than a second estimate.
            ax.bar(pos, st["cv_pct"], color=c, width=w,
                   alpha=0.75 if cloud == "livox" else 0.30,
                   edgecolor=c, hatch=None if cloud == "livox" else "//")
            if st.get("sd_ci95_lo") is not None and st["mean"]:
                lo = st["sd_ci95_lo"] / st["mean"] * 100
                hi = st["sd_ci95_hi"] / st["mean"] * 100
                ax.errorbar(pos, st["cv_pct"], yerr=[[st["cv_pct"] - lo],
                                                     [hi - st["cv_pct"]]],
                            fmt="none", ecolor="black", capsize=4, lw=1)
            ax.text(pos, st["cv_pct"], f" {st['cv_pct']:.2f}%", ha="center",
                    va="bottom", fontsize=7)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([METHODS[m] for m in methods], rotation=20, ha="right",
                       fontsize=8)
    ax.set_ylabel("Repeatability, 1σ [% of mean]")
    n_used = max((res["stats"]["livox"][m]["n"] for m in methods), default=0)
    title = (f"Measured repeatability (n={n_used})\n"
             "whisker = 95% CI of σ itself (χ²)")
    if paired_bars:
        title = (f"Repeatability: Livox (solid, n={n_used}) vs Leica "
                 f"reference re-processed (hatched, n={n_leica})\n"
                 "the hatched bar is the processing floor, not a rival estimate")
    ax.set_title(title, fontsize=9 if paired_bars else 10)
    ax.grid(alpha=0.25, axis="y")

    # -- C: the systematic offset, which repeating cannot remove ------------
    ax = axes[2]
    if res["bias"]:
        for m in methods:
            b = res["bias"].get(m)
            if not b:
                continue
            by_key = {p["key"]: p["delta_pct"] for p in b["per_rep"]}
            ax.plot(x, [by_key.get(k, np.nan) for k in labels], "o-",
                    color=METHOD_COLORS.get(m, "grey"), lw=1.2, ms=6,
                    label=f"{METHODS[m]}  ({b['mean_delta_pct']:+.2f}% mean)")
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("(Livox − Leica) / Leica [%]")
        ax.set_title("Offset from the Leica reference\n"
                     "systematic — not part of the σ on the left")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    else:
        ax.text(0.5, 0.5, "no Leica reference volume\n(run one rep with "
                          "--cloud both)", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="grey")
        ax.set_axis_off()

    fig.suptitle(f"Volume error from repeated runs — {dataset}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Driver                                                                     #
# --------------------------------------------------------------------------- #

def run_dataset(cfg, dataset: str, use_all: bool, strict: bool) -> Optional[dict]:
    runs = load_runs(cfg.results_dir, dataset)
    if not runs:
        log.warning("No results for dataset %s under %s — run run_pipeline.py "
                    "first.", dataset, cfg.results_dir / dataset)
        return None
    selected = select_runs(runs, use_all)
    log.info("%s: %d result dir(s) found, %d selected (%s)", dataset, len(runs),
             len(selected), "all runs" if use_all else "newest per rep/cloud")
    problems = check_comparable(selected, strict=strict)

    res = analyse(selected)
    n_livox = max((st["n"] for st in res["stats"].get("livox", {}).values()),
                  default=0)
    if n_livox < 2:
        log.warning("%s: only %d Livox run(s) — a spread needs at least 2, and "
                    "5 is what makes the SD worth quoting.", dataset, n_livox)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out_dir = cfg.results_dir / dataset / "statistics" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    text = print_report(dataset, res, selected, problems)
    (out_dir / "statistics.md").write_text(text)
    write_csvs(out_dir, dataset, res, selected)
    (out_dir / "statistics.json").write_text(json.dumps(
        {"dataset": dataset,
         "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "selection": "all" if use_all else "newest_per_rep_cloud",
         "config_problems": problems,
         "runs": [{"rep": r.rep, "stamp": r.stamp, "clouds": r.clouds,
                   "result_dir": r.summary.get("result_dir")} for r in selected],
         **{k: v for k, v in res.items() if k != "samples"},
         "samples": res["samples"]},
        indent=2, default=float))
    plot_statistics(dataset, res, out_dir / "volume_statistics.png")

    log.info("%s: wrote %s", dataset, out_dir)
    return res


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-7s %(message)s")
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", action="append", default=None,
                   help="Dataset to analyse; repeat the flag for several. "
                        "Default: every dataset found under results/. Datasets "
                        "are never pooled — one table each.")
    p.add_argument("--all", action="store_true",
                   help="Include every saved run, not just the newest per rep")
    p.add_argument("--allow-mixed-config", action="store_true",
                   help="Compute statistics even if the runs were processed "
                        "with different settings (warns instead of refusing)")
    args = p.parse_args()

    from config import Config
    cfg = Config()

    datasets = args.dataset
    if not datasets:
        datasets = sorted(
            d.name for d in cfg.results_dir.iterdir()
            if d.is_dir() and any(d.glob("*/*/summary.json"))
        )
        if not datasets:
            raise SystemExit(
                f"No run results under {cfg.results_dir}. Compute some volumes "
                "first:\n  python run_pipeline.py --run-real "
                "--dataset April_12_05_05 --rep all")
        log.info("Datasets found: %s", ", ".join(datasets))

    for ds in datasets:
        run_dataset(cfg, ds, use_all=args.all,
                    strict=not args.allow_mixed_config)


if __name__ == "__main__":
    main()
