"""
Dataset layout → input files for one run.

A dataset (one acquisition campaign, one Leica reference scan) is a directory
under `data/` laid out like this:

    data/April_12_05_05/
        rtc_clean_1cm.ply        <- Leica reference cloud (shared by all reps)
        targets_leica.txt        <- targets picked in the Leica cloud  (shared)
        caps.txt                 <- OPTIONAL: two shared-feature domain caps
        rep00/
            *_clean_and_regist.ply   <- Livox cloud, registered into the Leica datum
            trajectory.txt           <- GLIM trajectory, RAW Livox frame
            transformation_matrix.txt<- 4x4 Livox -> Leica
            targets_livox.txt        <- the same targets, picked in THIS Livox cloud
        rep01/ … rep04/

Every file is discovered rather than configured, so a run is named by
`--dataset April_12_05_05 --rep rep00` instead of six path flags. Discovery is
strict: a missing or ambiguous file raises here, at resolve time, rather than
surfacing later as a confusing load error or — worse — a plausible wrong number
from the wrong file.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Shared, dataset-level files (one Leica reference scan per campaign).
LEICA_PLY_NAME = "rtc_clean_1cm.ply"
TARGETS_LEICA_NAME = "targets_leica.txt"
# Optional, dataset-level: two points picked on a physical feature that is also
# identifiable in the OTHER campaign's reference scan, used as the domain end
# caps instead of the surveyed targets. Present => the run is feature-capped.
CAPS_NAME = "caps.txt"

# Per-rep files. Several names are accepted for the same thing because the
# acquisitions were exported at different times; the first one found wins.
TRAJECTORY_NAMES = ("trajectory.txt", "trajectory_transformed.txt")
REGISTRATION_NAMES = (
    "transformation_matrix.txt",
    "tranformation_matrix.txt",          # typo present in an earlier export
    "registration_livox_to_leica.txt",
)
TARGETS_LIVOX_NAME = "targets_livox.txt"


@dataclass
class RunInputs:
    """The input files of one run, as paths relative to `data_dir`."""
    dataset: str
    rep: str
    leica_ply: str
    livox_ply: str
    trajectory: str
    registration: str
    targets_leica: str
    targets_livox: Optional[str]
    caps: Optional[str] = None

    @property
    def run_key(self) -> str:
        """`<dataset>/<rep>` — the cache sub-directory and results sub-path."""
        return f"{self.dataset}/{self.rep}"


def list_datasets(data_dir: Path) -> List[str]:
    """Dataset directories under `data_dir` that hold a Leica reference scan."""
    data_dir = Path(data_dir)
    return sorted(
        d.name for d in data_dir.iterdir()
        if d.is_dir() and (d / LEICA_PLY_NAME).exists()
    )


def list_reps(data_dir: Path, dataset: str) -> List[str]:
    """Rep sub-directories of a dataset, sorted (rep00, rep01, …)."""
    root = Path(data_dir) / dataset
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}\n"
            f"  available: {', '.join(list_datasets(data_dir)) or '(none)'}"
        )
    reps = sorted(d.name for d in root.iterdir()
                  if d.is_dir() and not d.name.startswith("."))
    if not reps:
        raise FileNotFoundError(f"No rep sub-directories in {root}")
    return reps


def _pick_one(directory: Path, names, what: str) -> str:
    for n in names:
        if (directory / n).exists():
            return n
    raise FileNotFoundError(
        f"No {what} in {directory} — looked for {', '.join(names)}"
    )


def _find_livox_ply(rep_dir: Path) -> str:
    """The one PLY in the rep folder: the registered Livox cloud.

    Ambiguity is an error, not a guess: picking the wrong PLY (e.g. an
    unregistered export left in the folder) yields a plausible-looking wrong
    volume rather than a crash.
    """
    plys = sorted(p.name for p in rep_dir.glob("*.ply"))
    if not plys:
        raise FileNotFoundError(f"No *.ply (Livox cloud) in {rep_dir}")
    if len(plys) > 1:
        raise FileNotFoundError(
            f"{len(plys)} PLY files in {rep_dir}: {', '.join(plys)}. "
            "Leave only the registered Livox cloud there, or name it explicitly "
            "with --livox-file."
        )
    return plys[0]


def resolve_run(data_dir: Path, dataset: str, rep: str) -> RunInputs:
    """Discover the input files for `dataset/rep`."""
    data_dir = Path(data_dir)
    ds_dir = data_dir / dataset
    rep_dir = ds_dir / rep

    if not rep_dir.is_dir():
        raise FileNotFoundError(
            f"Rep directory not found: {rep_dir}\n"
            f"  available reps: {', '.join(list_reps(data_dir, dataset))}"
        )
    for name, what in ((LEICA_PLY_NAME, "Leica reference cloud"),
                       (TARGETS_LEICA_NAME, "Leica target file")):
        if not (ds_dir / name).exists():
            raise FileNotFoundError(f"No {what} at {ds_dir / name}")

    targets_livox = (f"{dataset}/{rep}/{TARGETS_LIVOX_NAME}"
                     if (rep_dir / TARGETS_LIVOX_NAME).exists() else None)
    return RunInputs(
        dataset=dataset,
        rep=rep,
        leica_ply=f"{dataset}/{LEICA_PLY_NAME}",
        livox_ply=f"{dataset}/{rep}/{_find_livox_ply(rep_dir)}",
        trajectory=f"{dataset}/{rep}/"
                   f"{_pick_one(rep_dir, TRAJECTORY_NAMES, 'trajectory')}",
        registration=f"{dataset}/{rep}/"
                     f"{_pick_one(rep_dir, REGISTRATION_NAMES, 'registration matrix')}",
        targets_leica=f"{dataset}/{TARGETS_LEICA_NAME}",
        targets_livox=targets_livox,
        caps=(f"{dataset}/{CAPS_NAME}" if (ds_dir / CAPS_NAME).exists()
              else None),
    )


def apply_to_config(cfg, run: RunInputs) -> None:
    """Point a Config at one run's files (and at its own cache sub-directory)."""
    cfg.leica_ply = run.leica_ply
    cfg.livox_ply = run.livox_ply
    cfg.trajectory_file = run.trajectory
    cfg.registration_txt = run.registration
    cfg.targets_csv = run.targets_leica
    cfg.targets_livox_csv = run.targets_livox
    cfg.caps_csv = run.caps
    cfg.cache_key = run.run_key


def file_provenance(data_dir: Path, run: RunInputs) -> Dict[str, dict]:
    """Per-input path + size + mtime, recorded in every run's summary.json.

    Two runs are only comparable if they were fed the files they claim; this is
    what lets the statistics detect a silently swapped or re-exported input.
    """
    out: Dict[str, dict] = {}
    for field_name in ("leica_ply", "livox_ply", "trajectory", "registration",
                       "targets_leica", "targets_livox", "caps"):
        rel = getattr(run, field_name)
        if rel is None:
            out[field_name] = None
            continue
        p = Path(data_dir) / rel
        st = p.stat()
        out[field_name] = {"path": rel, "bytes": st.st_size,
                           "mtime": int(st.st_mtime)}
    return out
