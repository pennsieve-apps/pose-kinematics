"""
pose-kinematics: from pose tracks to cleaned trajectories, kinematics and arena regions.

Reads pose files (port `pose`: ndx-pose NWB, DeepLabCut .h5/.csv, SLEAP .slp/.h5, Lightning Pose
.csv) and optional session NWB files (port `session`: timestamps, arena geometry, identity), and
writes, under OUTPUT_DIR/pose-kinematics/ (the single output port `kinematics`):

  <session_id>_behavior.nwb       cleaned PoseEstimation, BehavioralTimeSeries, Position, arena_regions
  <session_id>_kinematics.parquet time x features (the input of behavior-syllables)
  occupancy.csv                   time per region per session
  metrics.csv                     per-session headline metrics + sessions.csv columns
  group_stats.csv                 between-condition tests of the metrics (when sessions.csv has conditions)
  sessions.csv                    the sessions processed (session_id, path, ..., status, n_frames)
  <session_id>_trajectory.png, <session_id>_occupancy.png, <session_id>_pose_qc.png,
  speed_distribution.png, group_comparison.png
  summary.json, provenance.json
"""
from __future__ import annotations

import csv
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from processor.common import peak_rss_mb, set_threads, writable_caches, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("pose-kinematics")
log.setLevel(logging.INFO)
for noisy in ("matplotlib", "PIL", "fontTools", "h5py", "hdmf", "pynwb", "movement"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

APP_NAME = "pose-kinematics"
REPO_ROOT = Path(__file__).resolve().parent.parent
CITATIONS = ["10.5281/zenodo.12755724"]
TOOL_CITATIONS = {"DeepLabCut": ["10.1038/s41593-018-0209-y", "10.1038/s41596-019-0176-0"],
                  "SLEAP": ["10.1038/s41592-022-01426-1"]}

DEFAULTS = {
    "CONFIDENCE_THRESHOLD": "0.6",
    "MAX_GAP_S": "0.5",
    "SMOOTHING": "savgol",
    "SMOOTH_WINDOW_S": "0.2",
    "ARENA_SIZE_CM": "0",
    "PX_PER_CM": "0",
    "ARENA_SHAPE": "auto",
    "CENTER_FRACTION": "0.5",
    "WALL_ZONE_FRACTION": "0.1",
    "ROIS": "",
    "BODY_KEYPOINTS": "auto",
    "HEAD_KEYPOINT": "auto",
    "TAIL_KEYPOINT": "auto",
    "FPS": "0",
    "IMMOBILE_SPEED": "2.0",
    "MAX_SPEED_CM_S": "150",
    "GROUP_BY": "condition",
    "THREADS": "0",
}
METRICS = [("distance_travelled", "Distance travelled"), ("mean_speed", "Mean speed"),
           ("center_fraction", "Time in centre (fraction)"), ("thigmotaxis", "Thigmotaxis (wall-zone fraction)"),
           ("immobile_fraction", "Immobile (fraction)")]


class ParameterError(ValueError):
    pass


def read_parameters(env=os.environ) -> dict:
    raw = {k: (env.get(k) if env.get(k) not in (None, "") else v) for k, v in DEFAULTS.items()}
    p = {}
    try:
        for k in ("CONFIDENCE_THRESHOLD", "MAX_GAP_S", "SMOOTH_WINDOW_S", "PX_PER_CM", "CENTER_FRACTION",
                  "WALL_ZONE_FRACTION", "FPS", "IMMOBILE_SPEED", "MAX_SPEED_CM_S"):
            p[k] = float(raw[k])
        p["THREADS"] = int(raw["THREADS"])
    except ValueError as exc:
        raise ParameterError(f"bad numeric parameter: {exc}") from exc
    from processor.kinematics import parse_arena_size, parse_rois

    try:
        p["ARENA_SIZE_CM"] = parse_arena_size(raw["ARENA_SIZE_CM"])
    except ValueError as exc:
        raise ParameterError(f"ARENA_SIZE_CM must be a number (diameter or side) or WxH, got {raw['ARENA_SIZE_CM']!r}") from exc
    if p["ARENA_SIZE_CM"] and min(p["ARENA_SIZE_CM"]) <= 0:
        raise ParameterError("ARENA_SIZE_CM must be > 0")
    try:
        p["ROIS"] = parse_rois(raw["ROIS"])
    except (ValueError, KeyError, TypeError) as exc:
        raise ParameterError(f"ROIS: {exc}") from exc
    p["SMOOTHING"] = raw["SMOOTHING"].strip().lower()
    if p["SMOOTHING"] not in ("savgol", "median", "none"):
        raise ParameterError("SMOOTHING must be savgol, median or none")
    p["ARENA_SHAPE"] = raw["ARENA_SHAPE"].strip().lower()
    if p["ARENA_SHAPE"] not in ("auto", "circle", "rectangle"):
        raise ParameterError("ARENA_SHAPE must be auto, circle or rectangle")
    for k in ("BODY_KEYPOINTS", "HEAD_KEYPOINT", "TAIL_KEYPOINT", "GROUP_BY"):
        p[k] = raw[k].strip()
    if p["CONFIDENCE_THRESHOLD"] < 0:
        raise ParameterError("CONFIDENCE_THRESHOLD must be >= 0 (0 = no confidence filter)")
    if p["MAX_GAP_S"] < 0 or p["SMOOTH_WINDOW_S"] < 0 or p["PX_PER_CM"] < 0 or p["FPS"] < 0 or p["THREADS"] < 0 \
            or p["IMMOBILE_SPEED"] < 0 or p["MAX_SPEED_CM_S"] < 0:
        raise ParameterError("MAX_GAP_S, SMOOTH_WINDOW_S, PX_PER_CM, FPS, IMMOBILE_SPEED, THREADS must be >= 0")
    if not 0 < p["CENTER_FRACTION"] < 1:
        raise ParameterError("CENTER_FRACTION must be in (0, 1)")
    if not 0 < p["WALL_ZONE_FRACTION"] < 0.5:
        raise ParameterError("WALL_ZONE_FRACTION must be in (0, 0.5)")
    return p


def app_version() -> str:
    try:
        return (REPO_ROOT / "VERSION").read_text().strip()
    except OSError:
        return "unknown"


def tool_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": platform.python_version()}
    for pkg in ("movement", "pynwb", "hdmf", "ndx-pose", "numpy", "scipy", "pandas", "xarray", "pyarrow",
                "matplotlib", "h5py", "sleap-io"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return out


# ----------------------------------------------------------------------------- one session

def choose_session_nwb(sid, pose_path, found, n_pose_sessions):
    """The session NWB for a pose file: same session_id, not the pose file itself; or the only NWB."""
    cands = [p for p in found.nwbs.get(sid, []) if p != pose_path]
    if not cands and n_pose_sessions == 1:
        others = [p for ps in found.nwbs.values() for p in ps if p != pose_path]
        if len(others) == 1:
            log.warning("%s: pairing with the only session NWB %s (session ids differ)", sid, others[0].name)
            cands = others
    # Prefer files in the session port, then the shortest path.
    cands.sort(key=lambda p: (0 if "session" in p.parts else 1, len(str(p))))
    return cands[0] if cands else None


def resolve_times(pose, pf, sess, found, params, warnings_):
    import numpy as np

    from processor.inputs import timestamps_from_csv

    n = len(pose.timestamps)
    if pf.kind == "nwb":
        return
    for attr, label in (("pose_timestamps", "session NWB PoseEstimation timestamps"),
                        ("video_timestamps", "session NWB video (ImageSeries) timestamps")):
        ts = getattr(sess, attr, None) if sess else None
        if ts is not None and len(ts) >= n:
            if len(ts) != n:
                warnings_.append(f"{pf.session_id}: {label} have {len(ts)} entries for {n} pose frames; the first {n} are used")
            pose.timestamps, pose.time_source = np.asarray(ts[:n], float), f"{label} ({sess.path.name})"
            return
    from processor.common import POSE_SUFFIX_RE, file_stem

    stems = {pf.session_id, POSE_SUFFIX_RE.sub("", file_stem(pf.path))}
    for st in stems:
        p = found.timestamp_csvs.get(st)
        if p is not None:
            ts = timestamps_from_csv(p)
            if len(ts) >= n:
                pose.timestamps, pose.time_source = ts[:n], f"{p.name}"
                return
            warnings_.append(f"{pf.session_id}: {p.name} has {len(ts)} rows for {n} frames; not used")
    fps = params["FPS"] or pose.nominal_fps
    if not fps:
        fps = 30.0
        warnings_.append(f"{pf.session_id}: no timestamps and no frame rate; assumed 30 fps (set FPS)")
    pose.timestamps, pose.time_source = np.arange(n) / fps, f"frame / {fps:g} fps"


def process_session(pf, found, params, out_dir, n_pose, input_dir, warnings_):
    import numpy as np
    import pandas as pd

    from processor import figures as F
    from processor import kinematics as K
    from processor.inputs import SessionInfo, read_movement_pose, read_nwb_pose, read_session_nwb
    from processor.nwbout import write_behavior_nwb

    sid = pf.session_id
    t0 = time.monotonic()
    identity = {}
    if pf.kind == "nwb":
        pose, identity = read_nwb_pose(pf.path)
    else:
        pose = read_movement_pose(pf, params["FPS"] or None)
    if pose.n_individuals > 1:
        warnings_.append(f"{sid}: {pose.n_individuals} individuals in the pose file; only {pose.individual} is analysed")
    sess_path = choose_session_nwb(sid, pf.path, found, n_pose)
    sess = SessionInfo()
    for p in ([sess_path] if sess_path else []) + ([pf.path] if pf.kind == "nwb" else []):
        try:
            info = read_session_nwb(p)
        except Exception as exc:
            warnings_.append(f"{sid}: could not read {p.name}: {exc}")
            continue
        if sess.path is None:
            sess = info
        else:   # fill what the session NWB lacks from the pose NWB
            for attr in ("arena_outline", "arena_kind", "px_per_cm", "px_per_cm_source", "position_cm"):
                if getattr(sess, attr) is None:
                    setattr(sess, attr, getattr(info, attr))
    if sess.identity:
        identity = dict(sess.identity, source_file=sess.path.name)
    elif identity:
        identity = dict(identity, source_file=pf.path.name)
    resolve_times(pose, pf, sess if sess.path else None, found, params, warnings_)
    ts = pose.timestamps
    if len(ts) < 10:
        raise ValueError(f"only {len(ts)} frames")
    if np.any(np.diff(ts) <= 0):
        order = np.argsort(ts, kind="stable")
        keep = np.r_[True, np.diff(ts[order]) > 0]
        warnings_.append(f"{sid}: timestamps not strictly increasing; {int((~keep).sum())} duplicate frames dropped")
        idx = order[keep]
        pose.positions, pose.timestamps = pose.positions[idx], ts[idx]
        if pose.confidence is not None:
            pose.confidence = pose.confidence[idx]
        ts = pose.timestamps
    nominal = len(ts) / (ts[-1] - ts[0]) if ts[-1] > ts[0] else None
    anat = K.resolve_anatomy(pose.keypoints, params["BODY_KEYPOINTS"], params["HEAD_KEYPOINT"], params["TAIL_KEYPOINT"])
    xy_px, clean_report = K.clean_pose(pose, params["CONFIDENCE_THRESHOLD"], params["MAX_GAP_S"], params["SMOOTHING"],
                                       params["SMOOTH_WINDOW_S"])
    fps = clean_report["fps_effective"]

    # Arena (pixels), from the session's geometry or the extent of the tracks.
    if sess.arena_outline is not None:
        arena_px = K.arena_from_outline(sess.arena_outline, sess.arena_kind, f"arena outline in {sess.path.name}")
        if params["ARENA_SHAPE"] != "auto":
            arena_px.shape = params["ARENA_SHAPE"]
    else:
        arena_px = K.arena_from_trajectory(xy_px, params["ARENA_SHAPE"])

    # Scale px -> cm.
    scale, scale_source = None, None
    if params["PX_PER_CM"] > 0:
        scale, scale_source = (params["PX_PER_CM"],) * 2, "PX_PER_CM parameter"
    elif params["ARENA_SIZE_CM"]:
        w, h = params["ARENA_SIZE_CM"]
        scale = (2 * arena_px.rx / w, 2 * arena_px.ry / h)
        scale_source = f"ARENA_SIZE_CM {w:g} x {h:g} over the {arena_px.source}"
        if sess.arena_outline is None:
            warnings_.append(f"{sid}: arena size from the extent of the tracks (no arena outline); the scale is approximate")
    elif sess.px_per_cm:
        scale, scale_source = sess.px_per_cm, sess.px_per_cm_source
    elif sess.position_cm is not None:
        centroid_px = K._point(xy_px, pose.keypoints, anat.centroid)
        fit = K.fit_position_scale(ts, centroid_px, sess.position_cm)
        if fit:
            scale = fit[0]
            scale_source = f"linear fit of the centroid to Position/{sess.position_cm[2]} in cm (R2 {fit[1]:.4f})"
    units = "cm" if scale else "px"
    # Jump filter (needs the scale): clean again, dropping confidently tracked points whose step implies
    # more than MAX_SPEED_CM_S, before interpolation, so the jumps become gaps that get interpolated.
    if scale and params["MAX_SPEED_CM_S"] > 0:
        xy_px, clean_report = K.clean_pose(pose, params["CONFIDENCE_THRESHOLD"], params["MAX_GAP_S"],
                                           params["SMOOTHING"], params["SMOOTH_WINDOW_S"],
                                           max_speed_cm=params["MAX_SPEED_CM_S"], scale=scale)
        n_jump = sum(clean_report["jumps_dropped"].values())
        if n_jump:
            worst = sorted(clean_report["jumps_dropped"].items(), key=lambda kv: -kv[1])[:3]
            log.info("%s: jump filter dropped %d points faster than %g cm/s (%s)", sid, n_jump,
                     params["MAX_SPEED_CM_S"], ", ".join(f"{k} {v}" for k, v in worst))
    elif params["MAX_SPEED_CM_S"] > 0:
        warnings_.append(f"{sid}: no px -> cm scale, so the jump filter (MAX_SPEED_CM_S) was not applied")
    if scale:
        xy = xy_px / np.array(scale)
        arena = K.scale_arena(arena_px, *scale)
        to_out = lambda p: np.asarray(p, float) / np.array(scale)  # noqa: E731
    else:
        xy, arena, to_out = xy_px, arena_px, (lambda p: np.asarray(p, float))
        if any(r["units"] == "cm" for r in params["ROIS"]):
            raise ValueError("ROIS in cm but the px -> cm scale is unknown (set ARENA_SIZE_CM or PX_PER_CM)")

    feats = K.compute(ts, xy, pose.keypoints, anat)
    cxy = np.column_stack([feats["centroid_x"], feats["centroid_y"]])
    reg, wall_band = K.regions(arena, cxy, params["CENTER_FRACTION"], params["WALL_ZONE_FRACTION"])
    feats.update(reg)
    roi_outlines = []
    for roi in params["ROIS"]:
        feats[f"roi_{roi['name']}"] = K.roi_mask(roi, cxy, to_out) & np.isfinite(cxy).all(axis=1)
        roi_outlines.append((roi["name"], K.roi_outline(roi, to_out)))
    valid = np.isfinite(cxy).all(axis=1)
    feats["valid"] = valid
    dur = K.frame_durations(ts)

    # Tables.
    df = pd.DataFrame({"time": ts, "frame": np.arange(len(ts))})
    for k, v in feats.items():
        df[k] = v
    meta = {"app": APP_NAME, "version": app_version(), "session_id": sid, "units": units,
            "px_per_cm": list(scale) if scale else None, "scale_source": scale_source,
            "keypoints": pose.keypoints, "edges": [list(e) for e in pose.edges], "centroid_keypoints": anat.centroid, "head_keypoints": anat.head,
            "tail_keypoint": anat.tail, "nose_keypoint": anat.nose, "fps_effective": fps, "time_source": pose.time_source,
            "arena": arena.to_dict(), "source": str(pf.path.relative_to(input_dir)), "source_software": pose.source_software,
            "scorer": pose.scorer, "sheet": found.sheet.sheet_columns(sid), "nwb": identity,
            "session_nwb": str(sess.path.relative_to(input_dir)) if sess.path and sess.path.is_relative_to(input_dir) else None}
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}),
                                           b"pennsieve": json.dumps(meta, default=str).encode()})
    pq.write_table(table, out_dir / f"{sid}_kinematics.parquet", compression="zstd")

    region_names = ["in_center", "in_periphery", "in_wall_zone"] + [f"roi_{r['name']}" for r in params["ROIS"]]
    occ_rows, region_eps = [], {}
    total = float(dur[valid].sum())
    for rn in region_names:
        m = np.asarray(feats[rn], bool)
        name = rn[3:] if rn.startswith("in_") else rn
        eps = K.episodes(m, ts)
        region_eps[name] = eps
        sec = float(dur[m].sum())
        first = float(ts[m][0] - ts[0]) if m.any() else None
        occ_rows.append({"session_id": sid, "region": name, "seconds": sec,
                         "fraction": sec / total if total else None, "entries": len(eps),
                         "latency_s": first, "mean_speed": float(np.nanmean(feats["speed"][m])) if m.any() else None})

    nan_speed = feats["speed"][valid]
    distance = float(np.nansum(feats["step_distance"][valid]))
    immobile = (feats["speed"] < params["IMMOBILE_SPEED"]) & valid if units == "cm" else None
    # Distance while moving: steps of frames at or above IMMOBILE_SPEED, so pose jitter while still adds nothing.
    immobile_not = (feats["speed"] >= params["IMMOBILE_SPEED"]) & valid if units == "cm" else None
    metrics = {
        "session_id": sid, "status": "ok", "units": units, "n_frames": int(len(ts)),
        "duration_s": float(ts[-1] - ts[0] + dur[-1]), "fps_effective": round(fps, 4), "time_source": pose.time_source,
        "valid_fraction": float(valid.mean()), "jumps_dropped": int(sum(clean_report["jumps_dropped"].values())),
        "distance_travelled": distance,
        "distance_moving": float(np.nansum(feats["step_distance"][immobile_not])) if units == "cm" else None,
        "mean_speed": float(np.nanmean(nan_speed)) if nan_speed.size else None,
        "median_speed": float(np.nanmedian(nan_speed)) if nan_speed.size else None,
        "p95_speed": float(np.nanpercentile(nan_speed, 95)) if np.isfinite(nan_speed).any() else None,
        "immobile_fraction": float(dur[immobile].sum() / total) if immobile is not None and total else None,
        "center_fraction": occ_rows[0]["fraction"], "center_entries": occ_rows[0]["entries"],
        "periphery_fraction": occ_rows[1]["fraction"], "thigmotaxis": occ_rows[2]["fraction"],
        "mean_distance_to_wall": float(np.nanmean(feats["distance_to_wall"])),
        "mean_body_length": float(np.nanmean(feats["body_length"])) if np.isfinite(feats["body_length"]).any() else None,
        "mean_abs_angular_speed": float(np.nanmean(np.abs(feats["angular_speed"]))) if np.isfinite(feats["angular_speed"]).any() else None,
        "px_per_cm_x": scale[0] if scale else None, "px_per_cm_y": scale[1] if scale else None,
    }
    if nominal and pose.nominal_fps and abs(nominal - pose.nominal_fps) / pose.nominal_fps > 0.001:
        metrics["nominal_fps"] = pose.nominal_fps
        metrics["fps_drift_percent"] = 100 * (nominal - pose.nominal_fps) / pose.nominal_fps

    write_behavior_nwb(out_dir / f"{sid}_behavior.nwb", sid, identity, ts, xy, pose.confidence, pose.keypoints,
                       pose.edges, units, feats, region_eps, [r["name"] for r in params["ROIS"]], pose, app_version(),
                       f"Coordinates in {units}" + (f" (scale from {scale_source})." if scale else " (no scale known)."),
                       f"Confidence < {params['CONFIDENCE_THRESHOLD']} removed, gaps up to {params['MAX_GAP_S']} s "
                       f"interpolated, smoothing {clean_report['smoothing']}.")
    figs = []
    unit_label = units
    try:
        figs.append((F.trajectory(sid, ts, cxy[:, 0], cxy[:, 1], feats["speed"], arena, params["CENTER_FRACTION"],
                                  roi_outlines, unit_label, out_dir / f"{F.safe_name(sid)}_trajectory.png"),
                     f"{sid}: path of the body centroid over the session (lighter = earlier, darker = later), "
                     f"arena outline in grey and the centre zone dashed. A path hugging the wall with few "
                     f"crossings of the centre is thigmotaxis, a sign of anxiety-like behaviour; distance "
                     f"travelled {distance:.0f} {units}, {100 * (metrics['center_fraction'] or 0):.0f}% of the time in the centre."))
        figs.append((F.occupancy(sid, cxy[:, 0], cxy[:, 1], dur, arena, params["CENTER_FRACTION"], roi_outlines,
                                 unit_label, out_dir / f"{F.safe_name(sid)}_occupancy.png"),
                     f"{sid}: where the animal spent its time (seconds per spatial bin; white = never visited). "
                     f"Dark bins along the edge and an empty middle mean the animal avoided the open centre."))
        figs.append((F.confidence_qc(sid, pose.keypoints, clean_report["below_confidence"],
                                     clean_report["missing_after_cleaning"], out_dir / f"{F.safe_name(sid)}_pose_qc.png"),
                     f"{sid}: pose quality. Orange: share of frames where a keypoint's confidence was below "
                     f"{params['CONFIDENCE_THRESHOLD']} (removed). Blue: share still missing after gaps up to "
                     f"{params['MAX_GAP_S']} s were interpolated. Keypoints with large bars (occluded ears, a "
                     f"tail under a cable) should not be trusted for kinematics."))
    except Exception as exc:
        log.exception("%s: figure failed: %s", sid, exc)
    info = {"metrics": metrics, "occupancy": occ_rows, "figures": figs, "speed": feats["speed"][valid],
            "clean": clean_report, "anatomy": anat, "scale_source": scale_source, "arena": arena.to_dict(),
            "wall_band": wall_band, "session_nwb": sess.path.name if sess.path else None,
            "session_nwb_rel": meta["session_nwb"], "source_software": pose.source_software,
            "seconds": round(time.monotonic() - t0, 2)}
    log.info("%s: %d frames (%.1f s, %.2f fps, time from %s), %s, %.0f%% valid, distance %.1f %s, centre %.1f%%, "
             "thigmotaxis %.1f%%, %.1fs", sid, len(ts), metrics["duration_s"], fps, pose.time_source,
             f"{units} ({scale_source})" if scale else "pixels (no scale)", 100 * valid.mean(), distance, units,
             100 * (metrics["center_fraction"] or 0), 100 * (metrics["thigmotaxis"] or 0), info["seconds"])
    return info


def _write_csv(path, rows, columns=None):
    if columns is None:
        columns = []
        for r in rows:
            for k in r:
                if k not in columns:
                    columns.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else (f"{v:.6g}" if isinstance(v, float) else v)) for k, v in r.items()})


def main() -> int:
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    input_dir, output_dir = os.environ.get("INPUT_DIR"), os.environ.get("OUTPUT_DIR")
    if not input_dir or not output_dir:
        log.error("INPUT_DIR and OUTPUT_DIR must be set")
        return 2
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    writable_caches()
    try:
        params = read_parameters()
    except ParameterError as exc:
        log.error("%s", exc)
        return 2
    threads = set_threads(params["THREADS"])
    if not input_dir.is_dir():
        log.error("INPUT_DIR %s is not a folder", input_dir)
        return 1

    from processor import figures as F
    from processor.inputs import discover
    from processor.stats import compare_groups

    found = discover(input_dir)
    for name, reason in found.skipped.items():
        log.warning("%s: skipped, %s", name, reason)
    if not found.pose:
        log.error("no pose file (ndx-pose NWB, DeepLabCut .h5/.csv, SLEAP .slp/.h5, Lightning Pose .csv) under %s", input_dir)
        return 1
    log.info("%d sessions; sheet %s; confidence >= %g, gaps <= %g s, smoothing %s; %d threads", len(found.pose),
             found.sheet_path.name if found.sheet_path else "none", params["CONFIDENCE_THRESHOLD"], params["MAX_GAP_S"],
             params["SMOOTHING"], threads)
    top = output_dir / APP_NAME
    top.mkdir(parents=True, exist_ok=True)
    warnings_, results, statuses, figures = [], {}, {}, []
    for pf in found.pose:
        t = time.monotonic()
        try:
            results[pf.session_id] = process_session(pf, found, params, top, len(found.pose), input_dir, warnings_)
            statuses[pf.session_id] = {"status": "ok", "seconds": round(time.monotonic() - t, 2)}
        except Exception as exc:
            log.exception("%s: failed: %s", pf.session_id, exc)
            statuses[pf.session_id] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                       "seconds": round(time.monotonic() - t, 2)}
    for w in warnings_:
        log.warning("%s", w)
    ok = [s for s in statuses if statuses[s]["status"] == "ok"]

    sheet_rows = []
    for pf in found.pose:
        sid = pf.session_id
        r = {"session_id": sid, "path": str(pf.path.relative_to(input_dir))}
        r.update(found.sheet.sheet_columns(sid))
        r["status"] = statuses[sid]["status"]
        r["n_frames"] = results[sid]["metrics"]["n_frames"] if sid in results else None
        sheet_rows.append(r)
    cols = ["session_id", "path", "subject_id", "condition", "group"]
    _write_csv(top / "sessions.csv", sheet_rows, cols + [k for r in sheet_rows for k in r if k not in cols][::1])
    if not ok:
        log.error("every session failed")
        return 1

    group_col = params["GROUP_BY"] or "condition"
    metric_rows = []
    for sid in ok:
        m = dict(results[sid]["metrics"])
        m.update({k: v for k, v in found.sheet.sheet_columns(sid).items() if k not in m})
        metric_rows.append(m)
    if not any(r.get(group_col) for r in metric_rows) and group_col == "condition" and any(r.get("group") for r in metric_rows):
        group_col = "group"
    _write_csv(top / "metrics.csv", metric_rows)
    _write_csv(top / "occupancy.csv", [dict(o, **{k: v for k, v in found.sheet.sheet_columns(sid).items()
                                                  if k in ("subject_id", "condition", "group")})
                                       for sid in ok for o in results[sid]["occupancy"]])
    units = {results[s]["metrics"]["units"] for s in ok}
    unit = units.pop() if len(units) == 1 else "mixed"
    if unit == "mixed":
        warnings_.append("sessions differ in units (some cm, some px): group statistics are skipped")
    stats_rows = compare_groups(metric_rows, METRICS, group_col) if unit != "mixed" else []
    if stats_rows:
        _write_csv(top / "group_stats.csv", [{k: v for k, v in r.items() if k != "groups"} for r in stats_rows])

    for sid in ok:
        figures += [{"path": f"{APP_NAME}/{p.name}", "caption": c} for p, c in results[sid]["figures"][:2]]
    try:
        groups = {sid: found.sheet.sheet_columns(sid).get(group_col) for sid in ok}
        p = F.speed_distribution({sid: results[sid]["speed"] for sid in ok}, groups, unit, top / "speed_distribution.png")
        figures.insert(0, {"path": f"{APP_NAME}/{p.name}", "caption":
                           f"Distribution of body-centroid speed, one line per session"
                           f"{', coloured by ' + group_col if any(groups.values()) else ''}. A peak near zero is "
                           f"time spent still (resting, grooming, freezing); the long right tail is running. "
                           f"Lines of one condition shifted left of the other mean that condition moved less."})
        if stats_rows:
            p = F.group_comparison(metric_rows, stats_rows, group_col, top / "group_comparison.png")
            if p:
                g = stats_rows[0]["groups"]
                ns = ", ".join(f"{x} n={stats_rows[0].get(f'n_{x}')}" for x in g)
                figures.insert(1, {"path": f"{APP_NAME}/{p.name}", "caption":
                                   f"Per-session open-field metrics by {group_col} ({ns} sessions; each dot is one "
                                   f"session, the bar the mean). Test: {stats_rows[0]['test']} per metric, two-sided, "
                                   f"uncorrected for the {len(stats_rows)} metrics"
                                   f"{'; g = Hedges g effect size (' + g[0] + ' minus ' + g[1] + ', in pooled SDs; Mann-Whitney U p-values in group_stats.csv)' if len(g) == 2 else ''}. "
                                   f"With this few sessions per group the comparison is underpowered: a real "
                                   f"difference can easily miss p < 0.05, so read the effect sizes, not only p."})
    except Exception as exc:
        log.exception("cohort figures failed: %s", exc)
    for sid in ok:
        figures += [{"path": f"{APP_NAME}/{p.name}", "caption": c} for p, c in results[sid]["figures"][2:]]

    summary = {
        "app": APP_NAME, "version": app_version(),
        "sessions": {sid: {k: v for k, v in results[sid]["metrics"].items() if k not in ("session_id", "status")}
                     | {k: v for k, v in found.sheet.sheet_columns(sid).items() if k in ("subject_id", "condition", "group")}
                     | {"scale_source": results[sid]["scale_source"], "arena": results[sid]["arena"],
                        "session_nwb": results[sid]["session_nwb"]}
                     for sid in ok},
        "group_column": group_col if stats_rows else None,
        "group_stats": [{k: v for k, v in r.items() if k != "groups"} for r in stats_rows],
        "figures": figures,
        "tables": [f"{APP_NAME}/{n}" for n in ("metrics.csv", "occupancy.csv", "sessions.csv")]
                  + ([f"{APP_NAME}/group_stats.csv"] if stats_rows else []),
        "warnings": warnings_,
    }
    write_json(top / "summary.json", summary)
    write_json(top / "provenance.json", {
        "app": APP_NAME, "version": app_version(), "variant": "default", "tools": tool_versions(),
        "parameters": params,
        "inputs": sorted({str(pf.path.relative_to(input_dir)) for pf in found.pose}
                         | {results[sid]["session_nwb_rel"] for sid in ok if results[sid]["session_nwb_rel"]}),
        "started": started.isoformat(), "finished": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": round(time.monotonic() - t0, 2), "peak_rss_mb": peak_rss_mb(), "gpu": None,
        "sessions": statuses,
        "citations": CITATIONS + sorted({d for sid in ok for sw, ds in TOOL_CITATIONS.items()
                                         if sw.lower() in (results[sid]["source_software"] or "").lower() for d in ds}),
        "method": {sid: {"cleaning": results[sid]["clean"], "anatomy": vars(results[sid]["anatomy"]),
                         "scale_source": results[sid]["scale_source"], "wall_band": results[sid]["wall_band"]}
                   for sid in ok},
        "skipped": found.skipped,
    })
    log.info("done: %d sessions ok, %d failed, %.1fs, peak RSS %s MB -> %s", len(ok), len(statuses) - len(ok),
             time.monotonic() - t0, peak_rss_mb(), top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
