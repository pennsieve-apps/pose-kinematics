"""
Helpers shared by the processor: threads from the cgroup quota, JSON that never
holds NaN, peak memory, and the session contract of conventions.md (session_id
from sessions.csv, else sub-<a>_ses-<b>, else the file stem).

Nothing here imports numpy at module level, so main() can set the thread
variables before numpy loads.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from pathlib import Path

SUB_SES_RE = re.compile(r"sub-([A-Za-z0-9-]+?)_ses-([A-Za-z0-9-]+)")
UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
# Suffixes pose tools add to the video stem: DeepLabCut "<video>DLC_resnet50_...shuffle1_100000",
# SLEAP "<video>.predictions" / ".analysis", our own "<session>_pose".
POSE_SUFFIX_RE = re.compile(r"(DLC_.*|\.predictions.*|\.analysis.*|_pose|_predictions)$")


def writable_caches() -> str:
    """Point every library cache at $TMPDIR (Lambda: only INPUT_DIR, OUTPUT_DIR, HOME and TMPDIR are
    writable, and nothing the image put in /tmp or HOME survives). Called before matplotlib, numba,
    movement or JAX are imported."""
    import tempfile

    tmp = os.environ.get("TMPDIR") or tempfile.gettempdir()
    try:   # the runtime may name HOME / TMPDIR without creating them
        os.makedirs(tmp, exist_ok=True)
    except OSError:
        tmp = tempfile.gettempdir()
    os.environ["TMPDIR"] = tmp
    tempfile.tempdir = None
    if os.environ.get("HOME"):
        try:
            os.makedirs(os.environ["HOME"], exist_ok=True)
        except OSError:
            pass
    for var, sub in (("MPLCONFIGDIR", "mpl"), ("NUMBA_CACHE_DIR", "numba"), ("XDG_CACHE_HOME", "cache")):
        if not os.environ.get(var) or not os.access(os.environ[var], os.W_OK):
            os.environ[var] = os.path.join(tmp, sub)
        os.makedirs(os.environ[var], exist_ok=True)
    if not os.environ.get("HOME") or not os.access(os.environ["HOME"], os.W_OK):
        os.environ["HOME"] = os.path.join(tmp, "home")
        os.makedirs(os.environ["HOME"], exist_ok=True)
    os.environ.setdefault("USER", "app")
    return tmp


def cgroup_cpus() -> int:
    """CPUs from the cgroup quota (v2 cpu.max, v1 cfs), else the affinity mask; never the host's count alone."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            return max(1, math.floor(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        per = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0 and per > 0:
            return max(1, q // per)
    except (OSError, ValueError):
        pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def set_threads(threads: int) -> int:
    n = threads if threads > 0 else cgroup_cpus()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = str(n)
    return n


def peak_rss_mb() -> float | None:
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)
    except (ImportError, OSError):
        return None


def clean_json(obj):
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist") and getattr(obj, "ndim", 0) > 0:   # numpy / JAX arrays: nested lists
        return clean_json(obj.tolist())
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return clean_json(obj.item())
        except (ValueError, AttributeError):
            return str(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def write_json(path: Path, data) -> None:
    with open(path, "w") as fh:
        json.dump(clean_json(data), fh, indent=2)
        fh.write("\n")


def sanitise(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "-", name).strip("-") or "session"


def sub_ses_id(text: str) -> str | None:
    m = SUB_SES_RE.search(text)
    if not m:
        return None
    sub, ses = m.group(1), m.group(2)
    # A UUID session label is truncated to its first 8 characters (sub-NR-0031_ses-caa5dddc).
    uuid = re.match(r"^([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})", ses)
    if uuid:
        ses = ses[:8]
    return f"sub-{sub}_ses-{ses}"


def file_stem(path: Path) -> str:
    name = path.name
    for ext in (".nwb", ".h5", ".hdf5", ".csv", ".slp", ".parquet"):
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
            break
    return name


def read_sheet(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(fh)]
    return [r for r in rows if r.get("session_id")]


class Sessions:
    """session_id resolution per conventions.md, "Sessions"."""

    def __init__(self, sheet_rows: list[dict] | None = None):
        self.rows = {}
        self._paths = []
        for r in sheet_rows or []:
            sid = sanitise(r["session_id"])
            self.rows[sid] = dict(r, session_id=sid)
            if r.get("path"):
                self._paths.append((r["path"].strip().strip("/"), sid))
        # Longest paths first, so "a/b.nwb" wins over "a".
        self._paths.sort(key=lambda t: -len(t[0]))

    def from_sheet(self, path: Path) -> str | None:
        parts = path.as_posix()
        for p, sid in self._paths:
            name = Path(p).name
            if path.name == name or path.name.startswith(file_stem(Path(name)) + ".") or f"/{p}" in f"/{parts}" \
                    or any(par.name == name for par in path.parents):
                return sid
        return None

    def session_id(self, path: Path) -> str:
        sid = self.from_sheet(path)
        if sid:
            return sid
        sid = sub_ses_id(path.name)
        if sid:
            return sid
        for parent in path.parents:
            sid = sub_ses_id(parent.name)
            if sid:
                return sid
        stem = POSE_SUFFIX_RE.sub("", file_stem(path)) or file_stem(path)
        return sanitise(stem)

    def sheet_columns(self, sid: str) -> dict:
        row = self.rows.get(sid, {})
        return {k: v for k, v in row.items() if k not in ("session_id", "path")}


def walk_files(root: Path):
    """Every file under root, following symlinks (INPUT_DIR entries are symlinks), without loops."""
    seen = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        try:
            real = os.path.realpath(dirpath)
        except OSError:
            continue
        if real in seen:
            dirnames[:] = []
            continue
        seen.add(real)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for f in sorted(filenames):
            if not f.startswith("."):
                yield Path(dirpath) / f


IDENTITY_FIELDS = ("session_id", "experimenter", "institution", "lab", "keywords", "experiment_description")


def nwbfile_kwargs(identity: dict, sid: str, app: str, version: str, what: str) -> dict:
    """NWBFile arguments per conventions.md "Time": session fields copied from the source NWB, a new
    UUID identifier, the source file and identifier recorded in notes."""
    import uuid
    from datetime import datetime, timezone

    def dt(text):
        try:
            return datetime.fromisoformat(text) if text else None
        except ValueError:
            return None

    start = dt(identity.get("session_start_time")) or datetime(1970, 1, 1, tzinfo=timezone.utc)
    src = identity.get("source_file")
    derived = (f"Derived from {src}" + (f" (identifier {identity['identifier']})" if identity.get("identifier") else "")
               if src or identity.get("identifier") else "No source NWB")
    kw = {"session_description": identity.get("session_description") or f"session {sid}",
          "identifier": str(uuid.uuid4()), "session_start_time": start,
          "timestamps_reference_time": dt(identity.get("timestamps_reference_time")) or start,
          "session_id": identity.get("session_id") or sid,
          "source_script": f"{app} {version}", "source_script_file_name": "processor/main.py",
          "notes": f"{what} {derived} by {app} {version}."}
    for k in ("experimenter", "keywords"):
        v = identity.get(k)
        if v:
            kw[k] = list(v) if isinstance(v, (list, tuple)) else [v]
    for k in ("institution", "lab", "experiment_description"):
        if identity.get(k):
            kw[k] = identity[k]
    return kw
