"""I/O helpers: load PLY clouds, targets CSV, 4×4 registration matrix, npz cache."""

import io
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Point cloud                                                                 #
# --------------------------------------------------------------------------- #

def load_ply(path: Path | str) -> np.ndarray:
    """Return (N, 3) float64 xyz array from a PLY file."""
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.size == 0:
        raise ValueError(f"No points loaded from {path}")
    return pts


def load_trajectory(path: Path | str) -> np.ndarray:
    """
    Return (N, 3) float64 xyz array from a trajectory file.

    Dispatches on extension:
      .ply         -> read as a point cloud (load_ply)
      .txt / other -> TUM format: t x y z qx qy qz qw (whitespace-separated,
                      '#' comment lines allowed). Only columns 1-3 (xyz) are
                      kept; timestamp and quaternion are discarded.
    """
    path = Path(path)
    if path.suffix.lower() == ".ply":
        return load_ply(path)

    raw = np.loadtxt(str(path), comments="#")
    if raw.ndim == 1:
        raw = raw[np.newaxis, :]
    if raw.shape[1] < 4:
        raise ValueError(
            f"Expected TUM format (t x y z qx qy qz qw) with >=4 columns, "
            f"got {raw.shape[1]} in {path}"
        )
    pts = raw[:, 1:4].astype(np.float64)
    if pts.size == 0:
        raise ValueError(f"No points loaded from {path}")
    return pts


def load_and_transform_trajectory(
    path: Path | str, registration: np.ndarray
) -> np.ndarray:
    """
    Load a TUM-format trajectory (t x y z qx qy qz qw) and apply a 4x4 RIGID
    registration transform to both position and orientation, returning the
    transformed (N, 3) xyz positions.

    Unlike the point clouds (which are pre-transformed externally, once, and
    re-used across runs since re-transforming hundreds of millions of points
    is expensive), the GLIM/Livox trajectory is a native SLAM output living in
    the Livox sensor's own frame -- cheap (one array multiply) to correct at
    load time every run, so it is done here rather than requiring an external
    re-export step.

    Quaternions (scalar-last, x y z w -- SciPy convention) are composed with
    the registration rotation (rot_new = rot_registration * rot_pose) even
    though this pipeline currently only consumes positions downstream, so a
    caller needing full transformed poses later doesn't have to redo this.
    """
    path = Path(path)
    if path.suffix.lower() == ".ply":
        raise ValueError(
            f"{path} is a .ply trajectory (xyz only, no orientation) — "
            "cannot apply a full pose transform to it. load_trajectory() "
            "already assumes .ply trajectories arrive pre-transformed."
        )

    raw = np.loadtxt(str(path), comments="#")
    if raw.ndim == 1:
        raw = raw[np.newaxis, :]
    if raw.shape[1] < 8:
        raise ValueError(
            f"Expected TUM format (t x y z qx qy qz qw) with 8 columns, "
            f"got {raw.shape[1]} in {path}"
        )
    xyz = raw[:, 1:4].astype(np.float64)
    quat_xyzw = raw[:, 4:8].astype(np.float64)

    R_reg = registration[:3, :3]
    t_reg = registration[:3, 3]

    xyz_new = xyz @ R_reg.T + t_reg

    rot_new = Rotation.from_matrix(R_reg) * Rotation.from_quat(quat_xyzw)
    _ = rot_new.as_quat()   # transformed orientation; not consumed downstream yet

    if xyz_new.size == 0:
        raise ValueError(f"No points loaded from {path}")
    return xyz_new


def save_ply(path: Path | str, pts: np.ndarray, colours: np.ndarray | None = None) -> None:
    """Write (N, 3) xyz array (and optional (N, 3) float32 RGB 0–1) to PLY."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    if colours is not None:
        pcd.colors = o3d.utility.Vector3dVector(colours.astype(np.float64))
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)


# PLY property type per numpy dtype. Only the few CloudCompare reads as scalar
# fields without complaint; anything else must be cast by the caller.
_PLY_TYPE = {np.dtype("float32"): "float", np.dtype("float64"): "double",
             np.dtype("uint8"): "uchar", np.dtype("int32"): "int"}


def save_ply_scalars(path: Path | str, xyz: np.ndarray,
                     fields: Dict[str, np.ndarray] | None = None,
                     comments: tuple = ()) -> int:
    """Write (N, 3) xyz plus named per-point scalars to a binary PLY.

    Written by hand rather than through open3d because open3d's writer keeps
    only x/y/z (+ normals/colours) and silently drops everything else — and the
    whole point of this export is the per-point flags that travel with it.
    CloudCompare maps any extra PLY property onto a scalar field, so the flags
    arrive as something you can colour by and threshold on.

    Coordinates are float32: at ~1e2 m from the origin that is ~1e-5 m of
    quantisation, four orders below the measurement, and it halves the file.

    Returns the number of points written.
    """
    path = Path(path)
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N, 3), got {xyz.shape}")
    fields = dict(fields or {})

    dtypes = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    for name, arr in fields.items():
        arr = np.asarray(arr)
        if len(arr) != len(xyz):
            raise ValueError(f"field '{name}' has {len(arr)} values for "
                             f"{len(xyz)} points")
        if arr.dtype not in _PLY_TYPE:
            raise ValueError(f"field '{name}': unsupported dtype {arr.dtype}")
        dtypes.append((name, arr.dtype.newbyteorder("<")))

    rec = np.empty(len(xyz), dtype=dtypes)
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    for name, arr in fields.items():
        rec[name] = np.asarray(arr)

    header = ["ply", "format binary_little_endian 1.0"]
    header += [f"comment {c}" for c in comments]
    header.append(f"element vertex {len(xyz)}")
    header += ["property float x", "property float y", "property float z"]
    header += [f"property {_PLY_TYPE[np.dtype(dt)]} {name}"
               for name, dt in dtypes[3:]]
    header.append("end_header")

    with path.open("wb") as fh:
        fh.write(("\n".join(header) + "\n").encode("ascii"))
        rec.tofile(fh)
    return len(xyz)


# --------------------------------------------------------------------------- #
#  Targets CSV                                                                 #
# --------------------------------------------------------------------------- #

def load_targets(path: Path | str) -> np.ndarray:
    """
    Load target coordinates from CSV.

    Expected format (NO header):
        id  x  y  z    (whitespace or comma separated)

    The id column is discarded; returns (N, 3) float64 xyz.

    Falls back gracefully if the file has only 3 columns (plain x y z),
    or if the first line is a text header (auto-skipped).
    """
    path = Path(path)
    with open(path) as f:
        lines = f.readlines()

    first_line = lines[0].strip()
    tokens = first_line.replace(",", " ").split()
    has_header = any(not _is_numeric(t) for t in tokens)
    skiprows = 1 if has_header else 0

    # Normalize commas to whitespace so comma-, tab-, and space-separated
    # files all parse identically (np.loadtxt's default delimiter=None only
    # splits on whitespace, so "1, -3.38, ..." would otherwise fail to parse
    # as floats because of the trailing commas).
    normalized = "\n".join(line.replace(",", " ") for line in lines[skiprows:])
    raw = np.loadtxt(io.StringIO(normalized))
    if raw.ndim == 1:
        raw = raw[np.newaxis, :]          # single-row file

    if raw.shape[1] == 3:
        pts = raw[:, :3]                  # plain x y z
    elif raw.shape[1] >= 4:
        pts = raw[:, 1:4]                 # id  x  y  z  → take columns 1–3
    else:
        raise ValueError(
            f"targets.csv must have 3 or 4 columns (id,x,y,z), "
            f"got {raw.shape[1]} in {path}"
        )

    return pts.astype(np.float64)


def _is_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------- #
#  Registration matrix                                                         #
# --------------------------------------------------------------------------- #

def load_registration(path: Path | str) -> np.ndarray:
    """Load a 4×4 transformation matrix from a text file (rows of 4 numbers)."""
    mat = np.loadtxt(str(path))
    if mat.shape != (4, 4):
        raise ValueError(
            f"Expected a 4×4 matrix in {path}, got shape {mat.shape}"
        )
    return mat.astype(np.float64)


def check_rigid_registration(matrix: np.ndarray, tol: float = 1e-3) -> Dict[str, float | bool]:
    """
    Verify a 4x4 transform is RIGID (6-DOF): rotation block orthonormal,
    det(R) ~= +1, no scale.

    A scaled (7-DOF Helmert) alignment would absorb the longitudinal SLAM
    scale error this pipeline is trying to measure and quantify, and must
    never be used for Leica<->Livox registration. Warns loudly (does not
    raise) if the check fails, since the matrix has already been applied.
    """
    R = matrix[:3, :3]
    orthonormal_err = float(np.max(np.abs(R.T @ R - np.eye(3))))
    det = float(np.linalg.det(R))
    is_rigid = orthonormal_err < tol and abs(det - 1.0) < tol

    if not is_rigid:
        log.warning(
            "Registration matrix does NOT look rigid "
            "(det(R)=%.6f, orthonormality max-err=%.2e, tol=%.1e). "
            "A scaled/Helmert transform absorbs the SLAM scale error and "
            "erases the length-difference signal the Ā x L decomposition "
            "is meant to measure. Re-derive the registration RIGID-ONLY.",
            det, orthonormal_err, tol,
        )
    else:
        log.info(
            "Registration matrix is rigid (det(R)=%.6f, orthonormality "
            "max-err=%.2e).", det, orthonormal_err,
        )

    return {"det": det, "orthonormal_err": orthonormal_err, "is_rigid": is_rigid}


# --------------------------------------------------------------------------- #
#  NPZ cache                                                                   #
# --------------------------------------------------------------------------- #

def save_cache(path: Path | str, signature: str = "", **arrays: np.ndarray) -> None:
    """Write arrays to npz, stamped with the config signature that produced them."""
    np.savez_compressed(str(path), _signature=np.array(signature), **arrays)


def load_cache(
    path: Path | str, signature: str | None = None
) -> Dict[str, np.ndarray] | None:
    """
    Load a cached npz, returning None if it is STALE.

    Pass the signature the caller expects (Config.coord_signature). A cache
    written under different settings returns None so the caller recomputes
    rather than silently using coordinates that no longer match the config.
    Caches written before signatures existed have none, and are treated as
    stale — a one-off recompute is far cheaper than a wrong volume.
    """
    data = np.load(str(path))
    d = dict(data)
    stored = d.pop("_signature", None)

    if signature is None:
        return d

    if stored is None:
        log.warning("Cache %s predates signature checking — recomputing.", path)
        return None

    stored_str = str(stored)
    if stored_str != signature:
        log.warning(
            "Cache %s is STALE — recomputing.\n  cached: %s\n  wanted: %s",
            path, stored_str, signature,
        )
        return None
    return d


def cache_exists(path: Path | str) -> bool:
    return Path(str(path)).exists()
