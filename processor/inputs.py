"""
Finding and reading pose input.

Pose files (port `pose`) are found by name and content anywhere under INPUT_DIR:
  *.nwb with an ndx-pose PoseEstimation       (pose-estimate output, DANDI files)
  *.h5 / *.hdf5 DeepLabCut (pandas table)     or SLEAP analysis HDF5 ("tracks")
  *.slp SLEAP labels/predictions
  *.csv DeepLabCut or Lightning Pose (scorer / bodyparts / coords header rows)
Session NWB files (port `session`, optional) give frame timestamps, the arena geometry, the
px -> cm scale and the session identity (session_start_time, subject, ...). Every NWB file is a
session candidate, including the pose NWB itself.

NWB pose is read with pynwb directly, not with movement's NWB loader: movement 0.17 rebuilds time as
frame / fps and ignores the stored timestamps (movement issue #1169), which would break alignment
to neural data on the session clock.
"""
from __future__ import annotations

import csv
import logging
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from processor.common import Sessions, file_stem, read_sheet, walk_files

log = logging.getLogger("pose-kinematics")

POSE_PRIORITY = {"nwb": 0, "dlc-h5": 1, "sleap": 2, "sleap-h5": 2, "dlc-csv": 3, "lp-csv": 3}
SCALE_RE_XY = re.compile(r"([0-9.]+)\s*px\s*per\s*cm\s*along\s*x\D+?([0-9.]+)\s*px\s*per\s*cm\s*along\s*y", re.I)
SCALE_RE = re.compile(r"([0-9.]+)\s*(?:px|pixels?)\s*(?:per|/)\s*cm", re.I)


@dataclass
class PoseFile:
    path: Path
    kind: str
    session_id: str


@dataclass
class Pose:
    positions: np.ndarray            # (T, K, 2) float64, pixels (x right, y down, as in the video)
    confidence: np.ndarray | None    # (T, K) or None
    keypoints: list[str]
    timestamps: np.ndarray           # (T,) seconds on the session clock
    time_source: str
    source_software: str
    scorer: str = ""
    individual: str = ""
    n_individuals: int = 1
    edges: list = field(default_factory=list)
    confidence_definition: str = ""
    nominal_fps: float | None = None


@dataclass
class SessionInfo:
    path: Path | None = None
    identity: dict = field(default_factory=dict)       # session_start_time, identifier, ... from the NWB
    arena_outline: np.ndarray | None = None            # (N, 2) px, the arena border
    arena_kind: str | None = None                      # e.g. "Circle" when the geometry says so
    px_per_cm: tuple | None = None                     # (sx, sy) from the geometry's description
    px_per_cm_source: str | None = None
    position_cm: tuple | None = None                   # (timestamps, (T, 2) cm) a Position series, for a scale fit
    video_timestamps: np.ndarray | None = None
    pose_timestamps: np.ndarray | None = None


# ----------------------------------------------------------------------------- discovery

def _h5_kind(path: Path) -> str | None:
    try:
        with h5py.File(path, "r") as f:
            if "tracks" in f and "node_names" in f:
                return "sleap-h5"
            if "df_with_missing" in f or any(k.startswith("df") for k in f.keys()):
                return "dlc-h5"
            if f.attrs.get("neurodata_type") == "NWBFile" or "nwb_version" in f.attrs:
                return "nwb-h5"
    except (OSError, ValueError):
        return None
    return None


def _nwb_has_pose(path: Path) -> bool:
    found = []

    def visit(name, obj):
        if found:
            return
        nt = obj.attrs.get("neurodata_type")
        if isinstance(nt, bytes):
            nt = nt.decode()
        if nt == "PoseEstimation":
            found.append(name)

    try:
        with h5py.File(path, "r") as f:
            if "processing" in f:
                f["processing"].visititems(visit)
    except (OSError, ValueError):
        return False
    return bool(found)


def _csv_kind(path: Path) -> str | None:
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows = [next(csv.reader(fh), []) for _ in range(3)]
    except (OSError, UnicodeDecodeError, StopIteration):
        return None
    if len(rows) == 3 and rows[0] and rows[0][0].strip().lower() == "scorer" and rows[1] and \
            rows[1][0].strip().lower() == "bodyparts":
        scorer = " ".join(rows[0][1:2]).lower()
        return "lp-csv" if "heatmap" in scorer or "lightning" in scorer else "dlc-csv"
    return None


@dataclass
class Found:
    pose: list[PoseFile]
    nwbs: dict                       # session_id -> [Path] (every NWB, pose or not)
    sheet: Sessions
    sheet_path: Path | None
    skipped: dict
    timestamp_csvs: dict             # stem -> Path


def discover(input_dir: Path) -> Found:
    files = list(walk_files(input_dir))
    sheet_path = None
    for p in files:
        if p.name == "sessions.csv":
            try:
                rows = read_sheet(p)
            except (OSError, csv.Error):
                continue
            if rows and any(r.get("path") for r in rows):
                sheet_path = p
                break
    sessions = Sessions(read_sheet(sheet_path) if sheet_path else None)
    candidates: dict[str, list[PoseFile]] = {}
    nwbs: dict[str, list[Path]] = {}
    skipped = {}
    ts_csvs = {}
    for p in files:
        low = p.name.lower()
        rel = p.relative_to(input_dir).as_posix()
        in_session_port = rel.split("/", 1)[0] == "session"
        kind = None
        if low.endswith(".nwb"):
            sid = sessions.session_id(p)
            nwbs.setdefault(sid, []).append(p)
            if not in_session_port and _nwb_has_pose(p):
                kind = "nwb"
        elif low.endswith((".h5", ".hdf5")) and not in_session_port:
            kind = _h5_kind(p)
            if kind == "nwb-h5":
                kind = None
        elif low.endswith(".slp") and not in_session_port:
            kind = "sleap"
        elif low.endswith("_timestamps.csv"):
            ts_csvs[p.name[: -len("_timestamps.csv")]] = p
        elif low.endswith(".csv") and not in_session_port:
            kind = _csv_kind(p)
        if kind:
            sid = sessions.session_id(p)
            candidates.setdefault(sid, []).append(PoseFile(p, kind, sid))
    pose = []
    for sid, cands in candidates.items():
        cands.sort(key=lambda c: (0 if file_stem(c.path).endswith("_pose") else 1, POSE_PRIORITY.get(c.kind, 9),
                                  len(c.path.as_posix())))
        pose.append(cands[0])
        for c in cands[1:]:
            skipped[c.path.relative_to(input_dir).as_posix()] = f"another pose file of session {sid} is used ({cands[0].path.name})"
    pose.sort(key=lambda c: c.session_id)
    return Found(pose, nwbs, sessions, sheet_path, skipped, ts_csvs)


# ----------------------------------------------------------------------------- NWB reading

def _open_nwb(path: Path):
    import pynwb

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            import ndx_pose  # noqa: F401  (registers the extension for files without a cached spec)
        except ImportError:
            pass
        io = pynwb.NWBHDF5IO(str(path), "r")
        return io, io.read()


def _find(nwb, type_name: str):
    out = []
    for mod in nwb.processing.values():
        for name, obj in mod.data_interfaces.items():
            if type(obj).__name__ == type_name:
                out.append((name, obj))
    return out


def nwb_identity(nwb) -> dict:
    subj = None
    if nwb.subject is not None:
        s = nwb.subject
        subj = {k: getattr(s, k, None) for k in ("subject_id", "species", "sex", "genotype", "strain", "age",
                                                 "description", "date_of_birth", "weight")}
        subj = {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in subj.items() if v is not None}
    return {
        "session_start_time": nwb.session_start_time.isoformat() if nwb.session_start_time else None,
        "timestamps_reference_time": nwb.timestamps_reference_time.isoformat() if nwb.timestamps_reference_time else None,
        "identifier": nwb.identifier,
        "session_description": nwb.session_description,
        "subject": subj,
    }


def _series_times(s) -> np.ndarray:
    n = s.data.shape[0]
    if s.timestamps is not None:
        ts = s.timestamps
        ts = ts.timestamps if hasattr(ts, "timestamps") else ts   # a linked TimeSeries
        return np.asarray(ts[:], dtype=float)
    return float(s.starting_time or 0.0) + np.arange(n) / float(s.rate)


def read_nwb_pose(path: Path) -> tuple[Pose, dict]:
    io, nwb = _open_nwb(path)
    try:
        found = _find(nwb, "PoseEstimation")
        if not found:
            raise ValueError("no PoseEstimation in processing/")
        name, pe = found[0]
        series = list(pe.pose_estimation_series.values())
        if not series:
            raise ValueError(f"PoseEstimation {name} has no series")
        keypoints = [s.name for s in series]
        ts = _series_times(series[0])
        n = len(ts)
        pos = np.full((n, len(series), 2), np.nan)
        conf = np.full((n, len(series)), np.nan)
        has_conf = False
        for k, s in enumerate(series):
            d = np.asarray(s.data[:], dtype=float)
            m = min(n, d.shape[0])
            pos[:m, k, :] = d[:m, :2]
            if s.confidence is not None:
                c = np.asarray(s.confidence[:], dtype=float)
                conf[: min(n, c.shape[0]), k] = c[:n]
                has_conf = True
        edges = []
        skel = getattr(pe, "skeleton", None)
        if skel is not None and skel.edges is not None:
            nodes = list(skel.nodes[:])
            for a, b in np.asarray(skel.edges[:]).tolist():
                if a < len(nodes) and b < len(nodes) and nodes[a] in keypoints and nodes[b] in keypoints:
                    edges.append((nodes[a], nodes[b]))
        conf_def = getattr(series[0], "confidence__definition", None) or getattr(series[0], "confidence_definition", "") or ""
        software = getattr(pe, "source_software", None) or "unknown"
        rate = getattr(series[0], "rate", None)
        pose = Pose(pos, conf if has_conf else None, keypoints, ts,
                    "nwb PoseEstimation timestamps" if series[0].timestamps is not None else "nwb starting_time + rate",
                    software, getattr(pe, "scorer", "") or "", individual=name, n_individuals=len(found), edges=edges,
                    confidence_definition=conf_def, nominal_fps=float(rate) if rate else None)
        if len(found) > 1:
            log.warning("%s: %d PoseEstimation objects; using %s", path.name, len(found), name)
        return pose, nwb_identity(nwb)
    finally:
        io.close()


def read_session_nwb(path: Path) -> SessionInfo:
    """What a session NWB says about timing, arena and identity. Everything optional."""
    io, nwb = _open_nwb(path)
    info = SessionInfo(path=path)
    try:
        info.identity = nwb_identity(nwb)
        pe = _find(nwb, "PoseEstimation")
        if pe:
            s = list(pe[0][1].pose_estimation_series.values())
            if s:
                info.pose_timestamps = _series_times(s[0])
        for obj in nwb.acquisition.values():
            if type(obj).__name__ in ("ImageSeries", "OpticalSeries") and (obj.timestamps is not None):
                ts = obj.timestamps
                ts = ts.timestamps if hasattr(ts, "timestamps") else ts
                info.video_timestamps = np.asarray(ts[:], dtype=float)
                break
        # Arena geometry: a DynamicTable with x_px / y_px outlines (DANDI 002047: "arena_geometry").
        for mod in nwb.processing.values():
            for name, obj in mod.data_interfaces.items():
                cols = getattr(obj, "colnames", None) or ()
                if type(obj).__name__ == "DynamicTable" and "x_px" in cols and "y_px" in cols:
                    _read_arena(info, obj)
                    break
        # A Position series in cm/m, for a px -> cm fit when nothing else gives the scale.
        for _, posn in _find(nwb, "Position"):
            for ss in posn.spatial_series.values():
                unit = (ss.unit or "").lower()
                if ss.data.ndim == 2 and ss.data.shape[1] >= 2 and ("cm" in unit or "centimet" in unit or unit in ("m", "meter", "meters", "metre", "metres")):
                    factor = 100.0 if unit in ("m", "meter", "meters", "metre", "metres") else 1.0
                    data = np.asarray(ss.data[:, :2], dtype=float) * float(ss.conversion or 1.0) * factor
                    info.position_cm = (_series_times(ss), data, ss.name)
                    break
            if info.position_cm:
                break
    finally:
        io.close()
    return info


def _read_arena(info: SessionInfo, table) -> None:
    df = table.to_dataframe()
    desc = table.description or ""
    m = SCALE_RE_XY.search(desc)
    if m:
        info.px_per_cm = (float(m.group(1)), float(m.group(2)))
        info.px_per_cm_source = f"{table.name} description ('{m.group(0)}')"
    else:
        m = SCALE_RE.search(desc)
        if m:
            info.px_per_cm = (float(m.group(1)), float(m.group(1)))
            info.px_per_cm_source = f"{table.name} description ('{m.group(0)}')"
    row = None
    for col in ("element", "name", "kind"):
        if col in df.columns:
            hit = df[df[col].astype(str).str.lower().isin(["arena", "border"])]
            if len(hit):
                row = hit.iloc[0]
                break
    if row is None and len(df):
        row = df.iloc[0]
    if row is not None:
        x, y = np.asarray(row["x_px"], dtype=float), np.asarray(row["y_px"], dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= 3:
            info.arena_outline = np.column_stack([x[ok], y[ok]])
            info.arena_kind = str(row.get("geometry", "")) or None


# ----------------------------------------------------------------------------- other pose formats

def read_movement_pose(pf: PoseFile, fps: float | None) -> Pose:
    """DeepLabCut / SLEAP / Lightning Pose through movement; time is filled in later."""
    from movement.io import load_poses

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if pf.kind in ("dlc-h5", "dlc-csv"):
            ds, software = load_poses.from_dlc_file(pf.path, fps=fps), "DeepLabCut"
        elif pf.kind == "lp-csv":
            ds, software = load_poses.from_lp_file(pf.path, fps=fps), "LightningPose"
        else:
            ds, software = load_poses.from_sleap_file(pf.path, fps=fps), "SLEAP"
    pos = ds["position"]           # (time, space, keypoints, individuals)
    conf = ds["confidence"] if "confidence" in ds else None
    individuals = [str(i) for i in ds["individual"].values]
    # The individual with the most tracked points (single-animal apps: the only one).
    counts = [int(np.isfinite(pos.isel(individual=i).values).sum()) for i in range(len(individuals))]
    i = int(np.argmax(counts)) if counts else 0
    p = np.transpose(pos.isel(individual=i).values, (0, 2, 1)).astype(float)   # (T, K, space)
    c = conf.isel(individual=i).values.astype(float) if conf is not None else None
    if c is not None and np.isnan(c).all():
        c = None
    scorer = ""
    if pf.kind in ("dlc-csv", "lp-csv"):
        with open(pf.path, newline="", encoding="utf-8-sig") as fh:
            row = next(csv.reader(fh), [])
            scorer = row[1] if len(row) > 1 else ""
    return Pose(p[:, :, :2], c, [str(k) for k in ds["keypoint"].values], np.arange(p.shape[0], dtype=float),
                "frame index", software, scorer, individual=individuals[i], n_individuals=len(individuals),
                confidence_definition="likelihood" if software != "SLEAP" else "SLEAP point score")


def timestamps_from_csv(path: Path) -> np.ndarray:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or "time" not in rows[0]:
        raise ValueError(f"{path.name}: needs columns frame, time")
    if "frame" in rows[0]:
        rows.sort(key=lambda r: int(float(r["frame"])))
    return np.array([float(r["time"]) for r in rows])
