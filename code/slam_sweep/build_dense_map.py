"""
Build a dense, deskewed global point cloud from a ROS2 bag + GLIM traj_lidar.txt.

Pipeline:
  1. Read & parse all messages  (sequential, I/O-bound)
  2. Filter OOB & stationary    (sequential, cheap)
  3. Range filter + deskew       (parallel across --workers)
  4. Voxel downsample + SOR      (sequential)

"""

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.core as o3c
from rosbags.highlevel import AnyReader
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp


# ---- optional tqdm --------------------------------------------------------
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # minimal shim so the script still runs without tqdm
        def __init__(self, iterable=None, total=None, **kw):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable)
        def set_postfix(self, **kw):
            pass
        def update(self, n=1):
            pass
        def close(self):
            pass
        @staticmethod
        def write(msg):
            print(msg)


# ---- trajectory -----------------------------------------------------------
def load_trajectory(path):
    data = np.loadtxt(path)
    t = data[:, 0]
    trans_interp = interp1d(
        t, data[:, 1:4], axis=0, kind="linear",
        bounds_error=False, fill_value=np.nan, assume_sorted=True,
    )
    rot_slerp = Slerp(t, R.from_quat(data[:, 4:8]))
    return t[0], t[-1], trans_interp, rot_slerp


# ---- PointCloud2 parser ---------------------------------------------------
_ROS_DT = {1: "i1", 2: "u1", 3: "i2", 4: "u2",
           5: "i4", 6: "u4", 7: "<f4", 8: "<f8"}

_PC2_TIME_MODE = None


def _interpret_per_point_time(t_raw, header_t):
    t_raw = t_raw.astype(np.float64)
    candidates = [
        ("abs_s",  t_raw),
        ("abs_ns", t_raw * 1e-9),
        ("rel_s",  t_raw + header_t),
        ("rel_ns", t_raw * 1e-9 + header_t),
    ]
    best = min(candidates,
               key=lambda c: abs(float(np.median(c[1])) - header_t))
    return best[1], best[0]


def parse_pointcloud2(msg):
    offsets, dtypes, t_name = {}, {}, None
    for f in msg.fields:
        dt = _ROS_DT.get(f.datatype, "<f4")
        if f.name == "x":          offsets["x"] = f.offset; dtypes["x"] = dt
        elif f.name == "y":        offsets["y"] = f.offset; dtypes["y"] = dt
        elif f.name == "z":        offsets["z"] = f.offset; dtypes["z"] = dt
        elif f.name in ("intensity", "reflectivity"):
            offsets["i"] = f.offset; dtypes["i"] = dt
        elif f.name in ("timestamp", "time", "offset_time", "t"):
            offsets["t"] = f.offset; dtypes["t"] = dt; t_name = f.name
    if "x" not in offsets:
        return None

    names, formats, offs = [], [], []
    for k in ("x", "y", "z", "i", "t"):
        if k in offsets:
            names.append(k); formats.append(dtypes[k]); offs.append(offsets[k])
    dt = np.dtype({"names": names, "formats": formats,
                   "offsets": offs, "itemsize": msg.point_step})
    raw = np.frombuffer(msg.data, dtype=dt)
    n = len(raw)

    xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float32)
    intensity = (raw["i"].astype(np.float32) if "i" in offsets
                 else np.zeros(n, dtype=np.float32))

    header_t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    if "t" not in offsets:
        t_abs = np.full(n, header_t, dtype=np.float64)
    else:
        t_abs, mode = _interpret_per_point_time(raw["t"], header_t)
        global _PC2_TIME_MODE
        if _PC2_TIME_MODE != mode:
            tqdm.write(f"[time mode] per-point {t_name!r} interpreted as {mode}")
            _PC2_TIME_MODE = mode
    return xyz, intensity, t_abs


# ---- Livox CustomMsg parser ----------------------------------------------
def parse_customsg(msg):
    n = msg.point_num
    if n == 0:
        return None
    pts = msg.points

    if hasattr(pts, "dtype") and pts.dtype.fields is not None:
        xyz = np.stack([pts["x"], pts["y"], pts["z"]], axis=1).astype(np.float32)
        intensity = pts["reflectivity"].astype(np.float32)
        offset = pts["offset_time"].astype(np.int64)
    else:
        xyz = np.empty((n, 3), dtype=np.float32)
        intensity = np.empty(n, dtype=np.float32)
        offset = np.empty(n, dtype=np.int64)
        for i, p in enumerate(pts):
            xyz[i, 0] = p.x; xyz[i, 1] = p.y; xyz[i, 2] = p.z
            intensity[i] = p.reflectivity
            offset[i] = p.offset_time
    t_abs = (int(msg.timebase) + offset) * 1e-9
    return xyz, intensity, t_abs


def parse_message(msg, msgtype):
    if "PointCloud2" in msgtype:
        return parse_pointcloud2(msg)
    if "CustomMsg" in msgtype:
        return parse_customsg(msg)
    return None


# ---- deskew + transform --------------------------------------------------
def deskew_transform(xyz_local, t_abs, trans_interp, rot_slerp, t_start, t_end):
    t_clip = np.clip(t_abs, t_start, t_end)
    in_range = (t_abs >= t_start) & (t_abs <= t_end)
    trans = trans_interp(t_clip)
    rots = rot_slerp(t_clip).as_matrix()
    xyz_w = np.einsum("nij,nj->ni", rots, xyz_local) + trans
    return xyz_w.astype(np.float32), in_range & ~np.isnan(xyz_w).any(axis=1)


def range_filter(xyz, r_min, r_max):
    r2 = np.einsum('ij,ij->i', xyz, xyz)
    return (r2 > r_min**2) & (r2 < r_max**2) & ~np.isnan(xyz).any(axis=1)


def voxel_downsample(points, intensities, voxel_size):
    pcd = o3d.t.geometry.PointCloud()
    pcd.point.positions = o3c.Tensor(points, o3c.float32)
    pcd.point.intensity = o3c.Tensor(intensities.reshape(-1, 1), o3c.float32)
    ds = pcd.voxel_down_sample(voxel_size)
    return ds.point.positions.numpy(), ds.point.intensity.numpy().reshape(-1)


# ---- parallel worker -----------------------------------------------------
_W_TRAJ = None


def _init_worker(traj_path):
    """Pool initializer: each worker loads its own trajectory interpolators."""
    global _W_TRAJ
    _W_TRAJ = load_trajectory(traj_path)


def _process_chunk(chunk_and_args):
    """Process a chunk of scans: range filter + deskew. Returns merged arrays."""
    chunk, min_range, max_range = chunk_and_args
    t_start, t_end, trans_interp, rot_slerp = _W_TRAJ
    acc_pts, acc_int = [], []
    c_after_range = c_kept = 0
    for xyz, intensity, t_abs in chunk:
        m = range_filter(xyz, min_range, max_range)
        if not m.any():
            continue
        c_after_range += 1
        xyz_f, int_f, t_f = xyz[m], intensity[m], t_abs[m]
        xyz_w, valid = deskew_transform(
            xyz_f, t_f, trans_interp, rot_slerp, t_start, t_end)
        if valid.any():
            acc_pts.append(xyz_w[valid])
            acc_int.append(int_f[valid].astype(np.float32))
            c_kept += 1
    if acc_pts:
        return np.vstack(acc_pts), np.concatenate(acc_int), c_after_range, c_kept
    return None, None, c_after_range, c_kept


# ---- main ----------------------------------------------------------------
def main(args):
    n_workers = args.workers
    t_start, t_end, trans_interp, rot_slerp = load_trajectory(args.traj)
    print(f"Trajectory: {t_start:.3f} .. {t_end:.3f} ({t_end-t_start:.1f} s)")

    # -- Phase 1: read & parse all messages (sequential, I/O-bound) ---------
    scans = []          # list of (xyz, intensity, t_abs, scan_t)
    c_read = c_parsed = c_parse_fail = 0

    with AnyReader([Path(args.bag)]) as reader:
        conns = [c for c in reader.connections if c.topic == args.topic]
        if not conns:
            raise RuntimeError(
                f"Topic {args.topic!r} not in bag. "
                f"Available: {[c.topic for c in reader.connections]}")
        msgtype = conns[0].msgtype
        total = sum(c.msgcount for c in conns)
        print(f"Topic : {args.topic}  type={msgtype}  msgs={total}")

        pbar = tqdm(total=total, unit="msg", desc="reading",
                    dynamic_ncols=True, smoothing=0.1)
        try:
            for c, ts_ns, raw in reader.messages(connections=conns):
                c_read += 1
                msg = reader.deserialize(raw, c.msgtype)
                parsed = parse_message(msg, c.msgtype)
                if parsed is None:
                    c_parse_fail += 1
                else:
                    xyz, intensity, t_abs = parsed
                    c_parsed += 1
                    scan_t = float(np.median(t_abs))
                    scans.append((xyz, intensity, t_abs, scan_t))
                pbar.update(1)
        finally:
            pbar.close()

    # -- Phase 2: filter out-of-range & stationary (sequential, cheap) ------
    c_oob = c_stationary = 0
    filtered = []       # list of (xyz, intensity, t_abs)
    last_xyz, last_t = None, None

    for xyz, intensity, t_abs, scan_t in scans:
        if scan_t < t_start - 0.2 or scan_t > t_end + 0.2:
            c_oob += 1
            continue
        if args.skip_stationary > 0:
            trans_now = trans_interp(np.clip(scan_t, t_start, t_end))
            if last_xyz is not None and last_t is not None:
                dt = scan_t - last_t
                if dt > 1e-6:
                    v = float(np.linalg.norm(trans_now - last_xyz) / dt)
                    if v < args.skip_stationary:
                        c_stationary += 1
                        last_xyz, last_t = trans_now, scan_t
                        continue
            last_xyz, last_t = trans_now, scan_t
        filtered.append((xyz, intensity, t_abs))
    c_in_range = len(filtered) + c_stationary

    print(f"\n--- frame stats ---")
    print(f"  read              : {c_read}")
    print(f"  parsed            : {c_parsed}   (parse failures: {c_parse_fail})")
    print(f"  inside traj range : {c_in_range}   (out-of-range: {c_oob})")
    print(f"  stationary skipped: {c_stationary}")
    print(f"  to process        : {len(filtered)}")

    if not filtered:
        hint = ""
        if c_parsed == 0 and c_read > 0:
            hint = f"\nAll {c_read} messages failed to parse. Type = {msgtype!r}."
        elif c_in_range == 0 and c_parsed > 0:
            hint = (f"\nAll parsed messages fell outside the trajectory "
                    f"range [{t_start:.3f}, {t_end:.3f}]. "
                    f"Check traj_lidar.txt matches this bag.")
        print(f"No scans survived filtering.{hint}")
        return

    # -- Phase 3: range filter + deskew (parallel) --------------------------
    # Split scans into roughly equal chunks, one per worker.
    chunk_size = max(1, len(filtered) // n_workers)
    chunks = [filtered[i:i + chunk_size]
              for i in range(0, len(filtered), chunk_size)]
    del filtered   # free memory before forking

    print(f"Processing {sum(len(ch) for ch in chunks)} scans "
          f"across {len(chunks)} chunk(s) ({n_workers} worker(s))...")

    c_after_range = c_kept = 0
    acc_pts, acc_int = [], []

    if n_workers <= 1:
        # Single-process fast path — no IPC overhead.
        _init_worker(args.traj)
        for chunk in chunks:
            P, I, ar, k = _process_chunk((chunk, args.min_range, args.max_range))
            c_after_range += ar; c_kept += k
            if P is not None:
                acc_pts.append(P); acc_int.append(I)
    else:
        with mp.Pool(n_workers, initializer=_init_worker,
                      initargs=(args.traj,)) as pool:
            work = [(ch, args.min_range, args.max_range) for ch in chunks]
            for P, I, ar, k in pool.imap_unordered(_process_chunk, work):
                c_after_range += ar; c_kept += k
                if P is not None:
                    acc_pts.append(P); acc_int.append(I)

    print(f"  passed range filt : {c_after_range}")
    print(f"  kept (deskewed)   : {c_kept}")

    if not acc_pts:
        print("No points survived range filter + deskew. "
              "Try --min-range 0.1 --max-range 100.")
        return

    # -- Phase 4: merge, downsample, SOR, write -----------------------------
    P = np.vstack(acc_pts); I = np.concatenate(acc_int)
    del acc_pts, acc_int
    print(f"Total points before downsample: {len(P):,}")

    if args.voxel > 0:
        P, I = voxel_downsample(P, I, args.voxel)
        print(f"After voxel ({args.voxel} m): {len(P):,}")

    if args.sor_k > 0:
        print(f"Running statistical outlier removal on {len(P):,} points...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(P.astype(np.float64))
        _, idx = pcd.remove_statistical_outlier(
            nb_neighbors=args.sor_k, std_ratio=args.sor_std)
        P, I = P[idx], I[idx]
        print(f"After SOR: {len(P):,}")

    out = o3d.t.geometry.PointCloud()
    out.point.positions = o3c.Tensor(P, o3c.float32)
    out.point.intensity = o3c.Tensor(I.reshape(-1, 1), o3c.float32)
    o3d.t.io.write_point_cloud(args.out, out)
    print(f"Wrote {len(P):,} points -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--topic", default="/livox/lidar")
    ap.add_argument("--out", default="dense_map.pcd")
    ap.add_argument("--voxel", type=float, default=0.03)
    ap.add_argument("--min-range", dest="min_range", type=float, default=0.8)
    ap.add_argument("--max-range", dest="max_range", type=float, default=60.0)
    ap.add_argument("--skip-stationary", dest="skip_stationary",
                    type=float, default=0.03)
    ap.add_argument("--sor-k", dest="sor_k", type=int, default=20)
    ap.add_argument("--sor-std", dest="sor_std", type=float, default=2.0)
    ap.add_argument("--workers", type=int,
                    default=max(1, os.cpu_count() - 1),
                    help="parallel workers for deskew (default: cpu_count-1)")
    main(ap.parse_args())