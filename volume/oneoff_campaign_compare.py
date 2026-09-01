"""
April vs July, Leica against Leica — where along the tunnel the volume changed.

The two campaigns are in different datums and have no common registration, so
nothing can be compared point to point. What they DO share, once both are
feature-capped, is the physical place their domain starts: caps.txt names the
same wall feature in both scans. So

    s' = s - s_start

is a common chainage — "metres of tunnel past the entrance cap" — in both
campaigns, and the two A(s') curves are comparable slab for slab even though
their xyz frames are not. That is the whole reason the caps had to come first.

This compares the two REFERENCE scans only. The Leica pair is the campaign-
comparable measurement: the Livox walks differ by 11x in coverage between the
two campaigns (35% vs 3% of area interpolated), so a Livox-to-Livox difference
would be mostly a difference in what the sensor happened to see.

Usage
-----
    python oneoff_campaign_compare.py
    python oneoff_campaign_compare.py --a April_12_05_05 --b July_20_04_29

Not wired into the pipeline (an oneoff_*, per day3_findings section 11): it reads
existing run results and the coordinate cache, and computes no new volume that
run_pipeline does not already write.
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger(__name__)


def find_feature_capped_run(cfg, dataset: str) -> dict:
    """Newest run of `dataset` that is feature-capped AND has a Leica volume.

    Refuses a target-capped run outright: its domain is bounded by that
    campaign's own surveyed targets, which are not the other campaign's, so the
    comparison this module exists to make would be invalid.
    """
    best = None
    for p in sorted((cfg.results_dir / dataset).glob("*/*/summary.json")):
        s = json.loads(p.read_text())
        if s.get("config", {}).get("cap_mode") != "feature_planes":
            continue
        if "leica" not in s.get("clouds", []):
            continue
        s["_rep"] = p.parent.parent.name
        best = s                                   # glob is sorted, newest wins
    if best is None:
        raise SystemExit(
            f"No feature-capped run with a Leica volume for {dataset}. Run:\n"
            f"  python run_pipeline.py --run-real --dataset {dataset} "
            f"--rep rep00 --cloud both\n"
            f"with caps.txt in data/{dataset}/."
        )
    return best


def area_profile(cfg, dataset: str, summary: dict, cloud: str = "leica"):
    """A(s') for one cloud of one campaign, s' measured from the entry cap.

    Both clouds are read out of the SAME run, so within a campaign they share a
    centreline and a domain: any centreline effect is common to them and cancels
    when their two Delta curves are compared against each other.
    """
    import dataset as ds_mod
    from io_utils import load_cache
    from cross_sections import run_profiles

    rep = summary["_rep"]
    run = ds_mod.resolve_run(cfg.data_dir, dataset, rep)
    ds_mod.apply_to_config(cfg, run)

    ply = cfg.leica_ply if cloud == "leica" else cfg.livox_ply
    cache_file = cfg.cache_path(f"{cloud}_cyl")
    sig = cfg.coord_signature(ply_name=Path(ply).name)
    cached = load_cache(cache_file, signature=sig) if cache_file.exists() else None
    if cached is None:
        raise SystemExit(
            f"No valid cached {cloud} coordinates for {dataset}/{rep} "
            f"({cache_file}). Re-run that rep with --cloud both to create them."
        )

    dom = (summary["domain"]["s_start_m"], summary["domain"]["s_end_m"])
    prof = run_profiles(cached["s"], cached["r"], cached["theta"], dom, cfg,
                        cloud_name=f"{cloud}/{dataset}")
    return prof, dom


def resample_pair(pa, doma, pb, domb):
    """Both campaigns' A(s') on one common s' grid. Returns (grid, Aa, Ab)."""
    sa, sb = pa.s_centers - doma[0], pb.s_centers - domb[0]
    Aa, Ab = pa.A_s, pb.A_s
    ka, kb = np.isfinite(Aa), np.isfinite(Ab)
    s_hi = min(sa[ka].max(), sb[kb].max())
    grid = sa[(sa <= s_hi) & ka]
    return (grid,
            np.interp(grid, sa[ka], Aa[ka]),
            np.interp(grid, sb[kb], Ab[kb]))


def compare_both_clouds(args, summaries, leica_profs, doms) -> None:
    """Livox-vs-Livox July-April, set against the Leica-vs-Leica one.

    The identity that makes this readable:

        dV_livox  =  dV_leica  +  d(offset)

    where offset = V_livox - V_leica within a campaign. The left side is what a
    Livox-only monitoring campaign would have reported; the first term on the
    right is the tunnel; the second is the change in how much wall the Mid-360
    failed to see between the two campaigns. Splitting the Livox temporal signal
    into those two pieces is the point of this mode — it says how much of what
    Livox reports is tunnel and how much is sensor.
    """
    from config import Config

    livox_profs = {}
    for name in (args.a, args.b):
        cfg = Config()
        livox_profs[name], _ = area_profile(cfg, name, summaries[name], "livox")

    grid, Lea, Leb = resample_pair(leica_profs[args.a], doms[args.a],
                                   leica_profs[args.b], doms[args.b])
    _,    Lia, Lib = resample_pair(livox_profs[args.a], doms[args.a],
                                   livox_profs[args.b], doms[args.b])
    dA_lei, dA_liv = Leb - Lea, Lib - Lia

    def integ(y):
        return float(np.trapezoid(y, x=grid))

    V = {("leica", args.a): integ(Lea), ("leica", args.b): integ(Leb),
         ("livox", args.a): integ(Lia), ("livox", args.b): integ(Lib)}
    d_lei, d_liv = integ(dA_lei), integ(dA_liv)
    off_a = V[("livox", args.a)] - V[("leica", args.a)]
    off_b = V[("livox", args.b)] - V[("leica", args.b)]

    print("\n" + "=" * 78)
    print(f"  {args.b} MINUS {args.a} — Livox vs Leica, over the shared domain")
    print("=" * 78)
    print(f"  (single rep per campaign — {summaries[args.a]['_rep']} / "
          f"{summaries[args.b]['_rep']} — both clouds out of the same run, so "
          f"they share a centreline)\n")
    print(f"  {'':10} {args.a[:12]:>12} {args.b[:12]:>12} {'Delta':>10} {'Delta %':>9}")
    print("  " + "-" * 58)
    for cloud in ("leica", "livox"):
        va, vb = V[(cloud, args.a)], V[(cloud, args.b)]
        print(f"  {cloud:10} {va:12.2f} {vb:12.2f} {vb - va:+10.2f} "
              f"{(vb - va) / va * 100:+8.2f}%")
    print(f"  {'offset':10} {off_a:12.2f} {off_b:12.2f} {off_b - off_a:+10.2f}")
    print("   (offset = Livox - Leica within a campaign)")

    print(f"\n  DECOMPOSITION of what Livox reports as the temporal change:")
    print(f"    dV_livox            = {d_liv:+9.2f} m3")
    print(f"      of which tunnel   = {d_lei:+9.2f} m3   (the Leica difference)")
    print(f"      of which sensor   = {off_b - off_a:+9.2f} m3   (change in the "
          f"Livox-Leica offset)")
    share = abs(off_b - off_a) / (abs(d_lei) + abs(off_b - off_a)) * 100
    print(f"\n  {share:.0f}% of the magnitude is the offset change, not the tunnel."
          f"\n  The two terms have OPPOSITE signs, so Livox reports a "
          f"{'rise' if d_liv > 0 else 'fall'} where the tunnel {'fell' if d_lei < 0 else 'rose'}.")

    print(f"\n  Per-{args.segment_m:.0f} m segment:")
    print(f"  {'s range [m]':>16} {'dV leica':>10} {'dV livox':>10} "
          f"{'livox-leica':>12}")
    edges = np.arange(0.0, grid.max() + args.segment_m, args.segment_m)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (grid >= lo) & (grid < hi)
        if m.sum() < 2:
            continue
        a_, b_ = float(np.trapezoid(dA_lei[m], x=grid[m])), float(np.trapezoid(dA_liv[m], x=grid[m]))
        print(f"  {lo:7.0f}-{hi:<7.0f} {a_:10.2f} {b_:10.2f} {b_ - a_:+12.2f}")

    cum_lei = np.concatenate([[0.], np.cumsum(0.5 * (dA_lei[1:] + dA_lei[:-1]) * np.diff(grid))])
    cum_liv = np.concatenate([[0.], np.cumsum(0.5 * (dA_liv[1:] + dA_liv[:-1]) * np.diff(grid))])

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    ax = axes[0]
    ax.plot(grid, dA_lei, lw=1.0, color="steelblue", label="Leica − Leica")
    ax.plot(grid, dA_liv, lw=1.0, color="salmon", alpha=0.8, label="Livox − Livox")
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("ΔA [m²]")
    ax.legend()
    ax.set_title(f"Δ cross-section area, {args.b} − {args.a}\n"
                 "reference scans vs mobile scans, same shared domain")
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(grid, cum_lei, lw=1.8, color="steelblue",
            label=f"Leica: {cum_lei[-1]:+.1f} m³ (the tunnel)")
    ax.plot(grid, cum_liv, lw=1.8, color="salmon",
            label=f"Livox: {cum_liv[-1]:+.1f} m³ (tunnel + sensor)")
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("cumulative ΔV [m³]")
    ax.legend()
    ax.set_title("Cumulative Δ volume — the gap between the curves is the "
                 "change in Livox coverage, not the tunnel")
    ax.grid(alpha=0.25)

    ax = axes[2]
    for name, c in ((args.a, "steelblue"), (args.b, "salmon")):
        p = livox_profs[name]
        ax.plot(p.s_centers - doms[name][0], p.frac_interp * 100, lw=1.0,
                color=c, label=f"Livox {name}")
    ax.set_ylabel("interpolated [%]")
    ax.set_xlabel("s' — chainage from the shared entry cap [m]")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.set_title("Livox interpolated fraction — the reason the two Δ curves differ")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out = Path(args.save).with_name("14_campaign_compare_both_clouds.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  Figure: {out}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="April_12_05_05", help="first campaign")
    ap.add_argument("--b", default="July_20_04_29", help="second campaign")
    ap.add_argument("--segment-m", type=float, default=10.0,
                    help="segment length for the localisation table [m]")
    ap.add_argument("--save", default="figures/13_campaign_compare.png")
    ap.add_argument("--cloud", choices=["leica", "both"], default="leica",
                    help="'leica' (default): the campaign comparison. 'both': "
                         "also do the Livox-vs-Livox difference and set it "
                         "against the Leica one — see the warning it prints")
    args = ap.parse_args()

    from config import Config

    profs, doms, summaries = {}, {}, {}
    for name in (args.a, args.b):
        cfg = Config()
        summaries[name] = find_feature_capped_run(cfg, name)
        profs[name], doms[name] = area_profile(cfg, name, summaries[name])

    if args.cloud == "both":
        return compare_both_clouds(args, summaries, profs, doms)

    pa, pb = profs[args.a], profs[args.b]
    La, Lb = pa.length_m, pb.length_m

    print("\n" + "=" * 72)
    print(f"  LEICA REFERENCE, {args.a}  vs  {args.b}")
    print("=" * 72)
    print(f"  domain length      {La:10.3f} m   {Lb:10.3f} m   "
          f"Delta = {Lb - La:+.3f} m ({(Lb - La) / La * 100:+.3f}%)")
    for name in (args.a, args.b):
        s = summaries[name]
        print(f"  {name:16} rep={s['_rep']}  run={Path(s['result_dir']).name}  "
              f"interp={s['results']['leica']['frac_interp_mean'] * 100:.1f}%")

    print("\n  method              %-12s %-12s      Delta         Delta%%" % (args.a[:11], args.b[:11]))
    for key, label in (("profiles", "profiles"), ("surface_mesh", "surface mesh"),
                       ("hull_bound", "convex hull (UB)")):
        va = summaries[args.a]["results"]["leica"]["volumes"].get(key)
        vb = summaries[args.b]["results"]["leica"]["volumes"].get(key)
        if va is None or vb is None:
            continue
        print(f"  {label:18} {va:12.2f} {vb:12.2f}  {vb - va:+10.2f} m3  "
              f"{(vb - va) / va * 100:+7.2f}%")

    # ---- align on s' = s - s_start and difference the area profiles ----------
    sa = pa.s_centers - doms[args.a][0]
    sb = pb.s_centers - doms[args.b][0]
    Aa, Ab = pa.A_s, pb.A_s

    # Compare on the SHORTER of the two, so no slab is compared against
    # extrapolated area. The two lengths agree to a few cm, so this discards
    # almost nothing.
    s_hi = min(sa[np.isfinite(Aa)].max(), sb[np.isfinite(Ab)].max())
    grid = sa[(sa <= s_hi) & np.isfinite(Aa)]
    Aa_g = np.interp(grid, sa[np.isfinite(Aa)], Aa[np.isfinite(Aa)])
    Ab_g = np.interp(grid, sb[np.isfinite(Ab)], Ab[np.isfinite(Ab)])
    dA = Ab_g - Aa_g

    print(f"\n  Common chainage compared: s' = 0 .. {grid.max():.2f} m "
          f"(from the shared entry cap)")
    print(f"  mean area   {np.mean(Aa_g):8.3f} m2  {np.mean(Ab_g):8.3f} m2   "
          f"Delta = {np.mean(dA):+.3f} m2 ({np.mean(dA) / np.mean(Aa_g) * 100:+.2f}%)")

    # ---- per-segment localisation -------------------------------------------
    print(f"\n  Per-{args.segment_m:.0f} m segment (Delta = {args.b} - {args.a}):")
    print(f"  {'s'' range [m]':>16} {'V_a [m3]':>10} {'V_b [m3]':>10} "
          f"{'dV [m3]':>10} {'dV [%]':>8}")
    edges = np.arange(0.0, grid.max() + args.segment_m, args.segment_m)
    seg_mid, seg_dv = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (grid >= lo) & (grid < hi)
        if m.sum() < 2:
            continue
        va = float(np.trapezoid(Aa_g[m], x=grid[m]))
        vb = float(np.trapezoid(Ab_g[m], x=grid[m]))
        flag = "  <<<" if va > 0 and abs((vb - va) / va) > 0.05 else ""
        print(f"  {lo:7.0f}-{hi:<7.0f} {va:10.2f} {vb:10.2f} {vb - va:+10.2f} "
              f"{(vb - va) / va * 100:+7.2f}%{flag}")
        seg_mid.append(0.5 * (lo + hi))
        seg_dv.append(vb - va)

    cum = np.concatenate([[0.0], np.cumsum(0.5 * (dA[1:] + dA[:-1]) * np.diff(grid))])
    print(f"\n  Total over the common chainage: {cum[-1]:+.2f} m3")
    print("  A cumulative curve that drops in one place localises the change; a "
          "straight ramp\n  means it is distributed along the whole tunnel.")

    # ---- figure --------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    ax = axes[0]
    ax.plot(grid, Aa_g, lw=1.2, color="steelblue", label=f"{args.a} (Leica)")
    ax.plot(grid, Ab_g, lw=1.2, color="salmon", label=f"{args.b} (Leica)")
    ax.set_ylabel("A(s') [m²]")
    ax.legend()
    ax.set_title(f"Leica cross-section area, both campaigns on the shared "
                 f"feature-capped domain\n"
                 f"s' = 0 at the common entry cap")
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.axhline(0, color="black", lw=1)
    ax.fill_between(grid, dA, color="purple", alpha=0.35)
    ax.bar(seg_mid, np.array(seg_dv) / args.segment_m, width=args.segment_m * 0.85,
           color="grey", alpha=0.45, label=f"{args.segment_m:.0f} m segment mean")
    ax.set_ylabel("ΔA [m²]")
    ax.legend()
    ax.set_title(f"Δ area  ({args.b} − {args.a})")
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.axhline(0, color="black", lw=1)
    ax.plot(grid, cum, lw=1.5, color="darkgreen")
    ax.set_ylabel("cumulative ΔV [m³]")
    ax.set_xlabel("s' — chainage from the shared entry cap [m]")
    ax.set_title("Cumulative Δ volume — a step localises the change, a ramp "
                 "distributes it")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  Figure: {out}\n")


if __name__ == "__main__":
    main()
