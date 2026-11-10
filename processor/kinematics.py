"""
Cleaning, unit conversion, kinematics and arena regions for one session.

Cleaning uses movement (filter_by_confidence, interpolate_over_time, savgol_filter /
rolling_filter) on the pose with its real timestamps as the time coordinate, so derivatives
(movement's compute_speed, xarray differentiate) use the true frame intervals.

Coordinates stay in the video's orientation (x right, y down). Heading is the angle of the
tail -> head vector, counter-clockwise as seen on screen, 0 = pointing right, in radians (-pi, pi].
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("pose-kinematics")

CENTROID_NAMES = ["bodycenter", "body_center", "centroid", "center", "centre", "spine2", "spine_2", "mid_back",
                  "midback", "mid_backend", "back", "body", "torso", "thorax", "spine_mid", "midbody"]
HEAD_NAMES = ["headcenter", "head_center", "head", "head_midpoint", "neck"]
NOSE_NAMES = ["nose", "snout", "nose_tip"]
TAIL_NAMES = ["tailbase", "tail_base", "tail_root", "tailroot", "tail1", "tail_1", "tailstart", "tail_start", "tail"]
EAR_PAIRS = [("leftear", "rightear"), ("left_ear", "right_ear"), ("ear_left", "ear_right"), ("l_ear", "r_ear"),
             ("earl", "earr")]


def _norm(name: str) -> str:
    return name.lower().replace(" ", "").replace("-", "_")


def _match(keypoints: list[str], names: list[str]) -> str | None:
    norm = {_norm(k): k for k in keypoints}
    for n in names:
        if n in norm:
            return norm[n]
        if n.replace("_", "") in {k.replace("_", ""): v for k, v in norm.items()}:
            return {k.replace("_", ""): v for k, v in norm.items()}[n.replace("_", "")]
    return None


@dataclass
class Anatomy:
    centroid: list[str]
    head: list[str]             # one keypoint, or two ears (midpoint)
    tail: str | None
    nose: str | None
    notes: list = field(default_factory=list)


def resolve_anatomy(keypoints: list[str], body: str, head: str, tail: str) -> Anatomy:
    notes = []

    def pick(param, names, what):
        if param and param != "auto":
            want = [p.strip() for p in param.split(",") if p.strip()]
            missing = [w for w in want if w not in keypoints]
            if missing:
                raise ValueError(f"{what}: keypoints {missing} not in this file ({', '.join(keypoints)})")
            return want
        m = _match(keypoints, names)
        return [m] if m else []

    centroid = pick(body, CENTROID_NAMES, "BODY_KEYPOINTS")
    tail_kp = pick(tail, TAIL_NAMES, "TAIL_KEYPOINT")
    nose = _match(keypoints, NOSE_NAMES)
    head_kp = pick(head, HEAD_NAMES, "HEAD_KEYPOINT")
    if not head_kp:
        for a, b in EAR_PAIRS:
            ma, mb = _match(keypoints, [a]), _match(keypoints, [b])
            if ma and mb:
                head_kp = [ma, mb]
                notes.append(f"head = midpoint of {ma} and {mb}")
                break
    if not head_kp and nose:
        head_kp = [nose]
    if not centroid:
        centroid = [k for k in keypoints if "tail" not in _norm(k) or (tail_kp and k == tail_kp[0])]
        centroid = centroid or list(keypoints)
        notes.append(f"centroid = mean of {len(centroid)} keypoints (no body-centre keypoint)")
    return Anatomy(centroid, head_kp, tail_kp[0] if tail_kp else None, nose, notes)


def odd_window(seconds: float, fps: float, minimum: int) -> int:
    w = max(minimum, int(round(seconds * fps)))
    return w if w % 2 else w + 1


def jump_filter(xy, ts, max_speed):
    """Drop points whose step from the last accepted point of the same keypoint implies a speed above
    `max_speed` (position units per second). Walking forward, a point is accepted when
    |p_i - p_last| <= max_speed * (t_i - t_last); otherwise it is set to NaN and p_last stays. A
    confidently tracked excursion (the keypoint jumps away for several frames and back) is removed
    as a whole, and the allowed distance grows with the time since the last accepted point, so
    tracking recovers after a gap. Returns the filtered copy and the number of points dropped per
    keypoint."""
    out = xy.copy()
    T, K, _ = xy.shape
    dropped = np.zeros(K, int)
    for k in range(K):
        x, y = out[:, k, 0], out[:, k, 1]
        ok = np.where(np.isfinite(x) & np.isfinite(y))[0]
        if len(ok) < 2:
            continue
        # Start from the point closest to the median of the first 30 tracked points (not a jump), then
        # walk forward and backward from it.
        first = ok[:30]
        start = first[np.argmin(np.hypot(x[first] - np.median(x[first]), y[first] - np.median(y[first])))]
        for order in (ok[ok > start], ok[ok < start][::-1]):
            last = start
            for i in order:
                dt = abs(ts[i] - ts[last])
                if np.hypot(x[i] - x[last], y[i] - y[last]) <= max_speed * max(dt, 1e-6):
                    last = i
                else:
                    x[i] = y[i] = np.nan
                    dropped[k] += 1
    return out, dropped


def clean_pose(pose, threshold: float, max_gap_s: float, smoothing: str, window_s: float, max_speed_cm: float = 0.0,
               scale=None):
    """movement cleaning (confidence filter, jump filter, interpolation, smoothing).
    Returns (positions (T,K,2), report)."""
    import xarray as xr
    from movement import filtering

    ts = pose.timestamps
    dts = np.diff(ts)
    fps = 1.0 / float(np.median(dts[dts > 0])) if np.any(dts > 0) else 30.0
    T, K, _ = pose.positions.shape
    pos = xr.DataArray(np.transpose(pose.positions, (0, 2, 1))[:, :, :, None],
                       dims=("time", "space", "keypoint", "individual"),
                       coords={"time": ts, "space": ["x", "y"], "keypoint": pose.keypoints, "individual": ["id_0"]})
    report = {"fps_effective": fps, "n_frames": int(T)}
    missing0 = np.isnan(pose.positions).any(axis=2)
    report["missing_in_source"] = {k: float(missing0[:, i].mean()) for i, k in enumerate(pose.keypoints)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if pose.confidence is not None and threshold > 0:
            conf = xr.DataArray(pose.confidence[:, :, None], dims=("time", "keypoint", "individual"),
                                coords={"time": ts, "keypoint": pose.keypoints, "individual": ["id_0"]})
            pos = filtering.filter_by_confidence(pos, conf, threshold=threshold)
        low = np.isnan(pos.values[:, 0, :, 0]) & ~missing0
        report["below_confidence"] = {k: float(low[:, i].mean()) for i, k in enumerate(pose.keypoints)}
        report["jumps_dropped"] = {k: 0 for k in pose.keypoints}
        if max_speed_cm > 0 and scale:
            arr = np.transpose(pos.values[:, :, :, 0], (0, 2, 1)) / np.array(scale)   # in cm
            arr, dropped = jump_filter(arr, ts, max_speed_cm)
            pos = pos.copy(data=np.transpose(arr * np.array(scale), (0, 2, 1))[:, :, :, None])
            report["jumps_dropped"] = {k: int(d) for k, d in zip(pose.keypoints, dropped)}
        gap = max(1, int(round(max_gap_s * fps)))
        if max_gap_s > 0:
            pos = filtering.interpolate_over_time(pos, method="linear", max_gap=gap)
        report["max_gap_frames"] = gap if max_gap_s > 0 else 0
        if smoothing == "savgol":
            w = odd_window(window_s, fps, 5)
            pos = filtering.savgol_filter(pos, w, polyorder=2, mode="nearest")
            report["smoothing"] = f"Savitzky-Golay, {w} frames, polyorder 2"
        elif smoothing == "median":
            w = odd_window(window_s, fps, 3)
            pos = filtering.rolling_filter(pos, w, statistic="median")
            report["smoothing"] = f"rolling median, {w} frames"
        else:
            report["smoothing"] = "none"
    out = np.transpose(pos.values[:, :, :, 0], (0, 2, 1))
    missing1 = np.isnan(out).any(axis=2)
    report["missing_after_cleaning"] = {k: float(missing1[:, i].mean()) for i, k in enumerate(pose.keypoints)}
    return out, report


# ----------------------------------------------------------------------------- scale and arena

def parse_arena_size(text: str) -> tuple | None:
    text = (text or "").strip().lower().replace(" ", "")
    if text in ("", "0", "0.0", "auto"):
        return None
    for sep in ("x", "*", ","):
        if sep in text:
            a, b = text.split(sep, 1)
            return float(a), float(b)
    v = float(text)
    return v, v


def fit_position_scale(ts, centroid_px, position):
    """px -> cm from a Position series in cm on the same clock: per-axis least squares."""
    pts, pdata = position[0], position[1]
    if len(pts) < 50:
        return None
    idx = np.searchsorted(pts, ts)
    idx = np.clip(idx, 0, len(pts) - 1)
    close = np.abs(pts[idx] - ts) < 0.002
    ok = close & np.isfinite(centroid_px).all(axis=1) & np.isfinite(pdata[idx]).all(axis=1)
    if ok.sum() < 50:
        return None
    scale, r2s = [], []
    for ax in range(2):
        x, y = centroid_px[ok, ax], pdata[idx[ok], ax]
        a, b = np.polyfit(x, y, 1)
        resid = y - (a * x + b)
        r2 = 1 - resid.var() / max(y.var(), 1e-12)
        if a <= 0 or r2 < 0.98:
            return None
        scale.append(1.0 / a)
        r2s.append(r2)
    return (scale[0], scale[1]), min(r2s)


@dataclass
class Arena:
    shape: str                 # circle | rectangle
    cx: float
    cy: float
    rx: float                  # half extent along x (output units)
    ry: float
    source: str
    outline: np.ndarray | None = None

    def to_dict(self):
        return {"shape": self.shape, "center": [self.cx, self.cy], "half_extent": [self.rx, self.ry],
                "source": self.source}


def _shape_from_outline(outline, kind) -> str:
    if kind and "circ" in kind.lower():
        return "circle"
    if kind and ("rect" in kind.lower() or "square" in kind.lower()):
        return "rectangle"
    x, y = outline[:, 0], outline[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    box = (x.max() - x.min()) * (y.max() - y.min())
    return "rectangle" if box > 0 and area / box > 0.9 else "circle"


def arena_from_outline(outline, kind, source) -> Arena:
    x, y = outline[:, 0], outline[:, 1]
    return Arena(_shape_from_outline(outline, kind), (x.max() + x.min()) / 2, (y.max() + y.min()) / 2,
                 (x.max() - x.min()) / 2, (y.max() - y.min()) / 2, source, outline)


def arena_from_trajectory(points, shape: str) -> Arena:
    """The extent of every tracked keypoint (0.5-99.5 percentiles): the nose reaches the walls."""
    p = points.reshape(-1, 2)
    p = p[np.isfinite(p).all(axis=1)]
    lo, hi = np.percentile(p, 0.5, axis=0), np.percentile(p, 99.5, axis=0)
    cx, cy = (lo + hi) / 2
    rx, ry = (hi - lo) / 2
    if shape == "auto":
        # Time in the corners of the bounding box, outside its inscribed ellipse: a circular arena has none.
        u, v = (p[:, 0] - cx) / max(rx, 1e-9), (p[:, 1] - cy) / max(ry, 1e-9)
        corner = (np.hypot(u, v) > 1.1) & (np.abs(u) <= 1) & (np.abs(v) <= 1)
        shape = "rectangle" if corner.mean() > 0.01 else "circle"
    return Arena(shape, cx, cy, rx, ry, "extent of the tracked keypoints (0.5-99.5 percentile)")


def scale_arena(a: Arena, sx: float, sy: float) -> Arena:
    outline = None if a.outline is None else a.outline / np.array([sx, sy])
    return Arena(a.shape, a.cx / sx, a.cy / sy, a.rx / sx, a.ry / sy, a.source, outline)


def regions(arena: Arena, xy, center_fraction: float, wall_fraction: float):
    """Per frame: in_arena, in_center, in_periphery, in_wall_zone, distance_to_wall."""
    u = (xy[:, 0] - arena.cx) / arena.rx
    v = (xy[:, 1] - arena.cy) / arena.ry
    valid = np.isfinite(u) & np.isfinite(v)
    if arena.shape == "circle":
        r = np.hypot(u, v)
        in_center = r <= center_fraction
        # Distance to the wall along the radius, in output units (exact for a circle, approximate for an ellipse).
        dist = (1 - r) * np.hypot(u * arena.rx, v * arena.ry) / np.maximum(r, 1e-9)
        dist = np.where(r > 1e-9, dist, min(arena.rx, arena.ry))
        wall_band = wall_fraction * 2 * min(arena.rx, arena.ry)
    else:
        in_center = (np.abs(u) <= center_fraction) & (np.abs(v) <= center_fraction)
        dist = np.minimum(arena.rx - np.abs(xy[:, 0] - arena.cx), arena.ry - np.abs(xy[:, 1] - arena.cy))
        wall_band = wall_fraction * 2 * min(arena.rx, arena.ry)
    in_wall = dist <= wall_band
    return {"in_center": in_center & valid, "in_periphery": ~in_center & valid, "in_wall_zone": in_wall & valid,
            "distance_to_wall": np.where(valid, dist, np.nan)}, wall_band


# ----------------------------------------------------------------------------- ROIs

def parse_rois(text: str) -> list[dict]:
    """ROIS: JSON list of shapes, or a GeoJSON FeatureCollection (napari / movement export)."""
    import json

    text = (text or "").strip()
    if not text:
        return []
    data = json.loads(text)
    out = []
    if isinstance(data, dict) and data.get("type") == "FeatureCollection":
        for i, f in enumerate(data.get("features", [])):
            g = f.get("geometry") or {}
            name = (f.get("properties") or {}).get("name") or f"roi{i + 1}"
            if g.get("type") == "Polygon":
                out.append({"name": name, "type": "polygon", "points": g["coordinates"][0], "units": "px"})
            elif g.get("type") == "Point":
                raise ValueError("ROIS: GeoJSON points have no area; use polygons")
        return _check_rois(out)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("ROIS must be a JSON list of shapes")
    return _check_rois(data)


def _check_rois(rois):
    import re

    seen = set()
    for i, r in enumerate(rois):
        r.setdefault("name", f"roi{i + 1}")
        r["name"] = re.sub(r"[^A-Za-z0-9_-]", "_", str(r["name"]))
        if r["name"] in seen:
            raise ValueError(f"ROIS: duplicate name {r['name']}")
        seen.add(r["name"])
        r["units"] = str(r.get("units", "px")).lower()
        if r["units"] not in ("px", "cm"):
            raise ValueError(f"ROIS {r['name']}: units must be px or cm")
        t = r.get("type")
        if t == "circle":
            if "center" not in r or "radius" not in r:
                raise ValueError(f"ROIS {r['name']}: circle needs center [x, y] and radius")
        elif t == "rectangle":
            if not all(k in r for k in ("x0", "y0", "x1", "y1")):
                raise ValueError(f"ROIS {r['name']}: rectangle needs x0, y0, x1, y1")
        elif t == "polygon":
            if len(r.get("points", [])) < 3:
                raise ValueError(f"ROIS {r['name']}: polygon needs at least 3 points")
        else:
            raise ValueError(f"ROIS {r['name']}: type must be circle, rectangle or polygon")
    return rois


def roi_mask(roi: dict, xy, to_out) -> np.ndarray:
    """xy in output units; to_out converts a px point to output units (identity when output is px)."""
    from matplotlib.path import Path as MPath

    conv = (lambda p: np.asarray(p, float)) if roi["units"] == "cm" else to_out
    if roi["type"] == "circle":
        c = conv(roi["center"])
        edge = conv(np.asarray(roi["center"], float) + np.array([roi["radius"], 0.0]))
        rad = float(np.linalg.norm(edge - c)) if roi["units"] == "px" else float(roi["radius"])
        return np.hypot(xy[:, 0] - c[0], xy[:, 1] - c[1]) <= rad
    if roi["type"] == "rectangle":
        a, b = conv([roi["x0"], roi["y0"]]), conv([roi["x1"], roi["y1"]])
        return (xy[:, 0] >= min(a[0], b[0])) & (xy[:, 0] <= max(a[0], b[0])) & \
               (xy[:, 1] >= min(a[1], b[1])) & (xy[:, 1] <= max(a[1], b[1]))
    pts = np.array([conv(p) for p in roi["points"]])
    ok = np.isfinite(xy).all(axis=1)
    m = np.zeros(len(xy), bool)
    m[ok] = MPath(pts).contains_points(xy[ok])
    return m


def roi_outline(roi: dict, to_out):
    conv = (lambda p: np.asarray(p, float)) if roi["units"] == "cm" else to_out
    if roi["type"] == "circle":
        t = np.linspace(0, 2 * np.pi, 100)
        c = np.asarray(roi["center"], float)
        pts = np.column_stack([c[0] + roi["radius"] * np.cos(t), c[1] + roi["radius"] * np.sin(t)])
        return np.array([conv(p) for p in pts])
    if roi["type"] == "rectangle":
        pts = [[roi["x0"], roi["y0"]], [roi["x1"], roi["y0"]], [roi["x1"], roi["y1"]], [roi["x0"], roi["y1"]],
               [roi["x0"], roi["y0"]]]
        return np.array([conv(p) for p in pts])
    pts = list(roi["points"]) + [roi["points"][0]]
    return np.array([conv(p) for p in pts])


# ----------------------------------------------------------------------------- kinematics

def _point(xy, keypoints, names):
    idx = [keypoints.index(n) for n in names]
    pts = xy[:, idx, :]
    if len(idx) == 1:
        return pts[:, 0, :]
    n_ok = np.isfinite(pts).all(axis=2).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean = np.nanmean(pts, axis=1)
    mean[n_ok < max(1, int(np.ceil(len(idx) / 2)))] = np.nan
    return mean


def _speed(ts, xy):
    """Speed with movement (xarray differentiate on the real timestamps)."""
    import xarray as xr
    from movement.kinematics import compute_speed

    da = xr.DataArray(xy, dims=("time", "space"), coords={"time": ts, "space": ["x", "y"]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(compute_speed(da).values, float)


def compute(ts, xy, keypoints, anat: Anatomy):
    """Per-frame features. xy (T, K, 2) cleaned, in output units."""
    out = {}
    centroid = _point(xy, keypoints, anat.centroid)
    out["centroid_x"], out["centroid_y"] = centroid[:, 0], centroid[:, 1]
    out["speed"] = _speed(ts, centroid)
    step = np.r_[0.0, np.hypot(*np.diff(centroid, axis=0).T)]
    out["step_distance"] = np.where(np.isfinite(step), step, 0.0)
    head = _point(xy, keypoints, anat.head) if anat.head else None
    tail = xy[:, keypoints.index(anat.tail), :] if anat.tail else None
    back = tail if tail is not None else (centroid if anat.head and anat.head != anat.centroid else None)
    if head is not None and back is not None:
        vec = head - back
        heading = np.arctan2(-vec[:, 1], vec[:, 0])          # y down: counter-clockwise on screen is positive
        out["heading"] = heading
        ok = np.isfinite(heading)
        ang = np.full(len(ts), np.nan)
        if ok.sum() > 2:
            unwrapped = np.interp(ts, ts[ok], np.unwrap(heading[ok]))
            ang = np.gradient(unwrapped, ts)
            ang[~ok] = np.nan
        out["angular_speed"] = ang
    else:
        out["heading"] = np.full(len(ts), np.nan)
        out["angular_speed"] = np.full(len(ts), np.nan)
    front = xy[:, keypoints.index(anat.nose), :] if anat.nose else head
    if front is not None and tail is not None:
        out["body_length"] = np.hypot(*(front - tail).T)
    else:
        out["body_length"] = np.full(len(ts), np.nan)
    for i, k in enumerate(keypoints):
        out[f"kp_{k}_x"] = xy[:, i, 0]
        out[f"kp_{k}_y"] = xy[:, i, 1]
        out[f"kp_{k}_speed"] = _speed(ts, xy[:, i, :])
    return out


def frame_durations(ts) -> np.ndarray:
    d = np.diff(ts)
    med = float(np.median(d)) if len(d) else 0.0
    return np.r_[d, med]


def episodes(mask, ts):
    """[(start_time, stop_time)] of runs of True; stop = time of the first frame after the run."""
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    d = np.diff(np.r_[0, m.astype(int), 0])
    starts, stops = np.where(d == 1)[0], np.where(d == -1)[0]
    end = ts[-1] + (frame_durations(ts)[-1] if len(ts) else 0)
    return [(float(ts[a]), float(ts[b]) if b < len(ts) else float(end)) for a, b in zip(starts, stops)]
