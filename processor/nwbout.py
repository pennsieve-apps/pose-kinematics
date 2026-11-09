"""<session_id>_behavior.nwb: cleaned PoseEstimation, BehavioralTimeSeries, Position and arena regions."""
from __future__ import annotations

import uuid
import warnings
from datetime import datetime, timezone

import numpy as np


def _dt(text):
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def write_behavior_nwb(path, sid, identity, ts, xy, conf, keypoints, edges, units, feats, region_eps, roi_names,
                       pose, app_version, scale_note, cleaning_note):
    import pynwb
    from hdmf.backends.hdf5.h5_utils import H5DataIO
    from ndx_pose import PoseEstimation, PoseEstimationSeries, Skeleton, Skeletons
    from pynwb.behavior import BehavioralTimeSeries, Position, SpatialSeries
    from pynwb.epoch import TimeIntervals
    from pynwb.file import Subject

    unit = "cm" if units == "cm" else "pixels"
    rate_unit = f"{unit}/s"
    start = _dt(identity.get("session_start_time")) or datetime(1970, 1, 1, tzinfo=timezone.utc)
    ref = _dt(identity.get("timestamps_reference_time")) or start
    subj = None
    if identity.get("subject"):
        s = dict(identity["subject"])
        if s.get("date_of_birth"):
            s["date_of_birth"] = _dt(s["date_of_birth"])
        s = {k: v for k, v in s.items() if k in ("subject_id", "species", "sex", "genotype", "strain", "age",
                                                  "description", "date_of_birth", "weight") and v is not None}
        try:
            subj = Subject(**s)
        except Exception:
            subj = None
    nwb = pynwb.NWBFile(
        session_description=identity.get("session_description") or f"session {sid}",
        identifier=identity.get("identifier") or str(uuid.uuid4()),
        session_start_time=start, timestamps_reference_time=ref, subject=subj,
        source_script=f"pose-kinematics {app_version}", source_script_file_name="processor/main.py",
        notes=f"Cleaned pose and kinematics for session {sid}. {cleaning_note} {scale_note}".strip())
    mod = nwb.create_processing_module("behavior", "Cleaned pose, kinematics and arena regions (pose-kinematics)")
    camera = nwb.create_device(name="camera", description="behaviour camera of the source pose")
    node_index = {k: i for i, k in enumerate(keypoints)}
    edge_arr = np.array([[node_index[a], node_index[b]] for a, b in edges], dtype="uint8").reshape(-1, 2)
    skeleton = Skeleton(name="skeleton", nodes=list(keypoints), edges=edge_arr, subject=subj)
    mod.add(Skeletons(skeletons=[skeleton]))
    comp = dict(compression="gzip", compression_opts=4)
    series = []
    first = None
    for i, k in enumerate(keypoints):
        kw = dict(name=k, description=f"{k}, cleaned (confidence filter, interpolation, smoothing), {unit}",
                  data=H5DataIO(xy[:, i, :].astype("float32"), **comp), unit=unit,
                  reference_frame="(0, 0) is the top-left corner of the video; x right, y down")
        if conf is not None:
            kw["confidence"] = conf[:, i].astype("float32")
            kw["confidence_definition"] = pose.confidence_definition or "confidence of the source pose estimate"
        kw["timestamps"] = H5DataIO(ts, **comp) if first is None else first
        s = PoseEstimationSeries(**kw)
        first = first or s
        series.append(s)
    mod.add(PoseEstimation(name="PoseEstimation", pose_estimation_series=series,
                           description=f"Cleaned by pose-kinematics from {pose.source_software} pose. {cleaning_note}",
                           scorer=pose.scorer or None, source_software=pose.source_software or None,
                           skeleton=skeleton, devices=[camera]))
    bts = BehavioralTimeSeries(name="BehavioralTimeSeries")
    named = [("speed", rate_unit, "centroid speed"), ("heading", "radians", "tail -> head angle, counter-clockwise on screen, 0 = right"),
             ("angular_speed", "radians/s", "time derivative of the unwrapped heading"),
             ("body_length", unit, "nose (or head) to tail base distance"),
             ("distance_to_wall", unit, "centroid distance to the arena wall")]
    for name, u, desc in named:
        if name in feats and np.isfinite(feats[name]).any():
            bts.create_timeseries(name=name, data=H5DataIO(np.asarray(feats[name], "float32"), **comp), unit=u,
                                  timestamps=first, description=desc)
    kp_speed = np.column_stack([feats[f"kp_{k}_speed"] for k in keypoints]).astype("float32")
    bts.create_timeseries(name="keypoint_speed", data=H5DataIO(kp_speed, **comp), unit=rate_unit, timestamps=first,
                          description="speed of each keypoint; columns in the order: " + ", ".join(keypoints))
    mod.add(bts)
    centroid = np.column_stack([feats["centroid_x"], feats["centroid_y"]]).astype("float32")
    mod.add(Position(spatial_series=SpatialSeries(
        name="centroid", data=H5DataIO(centroid, **comp), timestamps=first, unit=unit,
        reference_frame="(0, 0) is the top-left corner of the video; x right, y down",
        description="body centroid (see summary.json for the keypoints used)")))
    ti = TimeIntervals(name="arena_regions", description="Episodes with the centroid in each arena region "
                       "(center, periphery, wall_zone, user ROIs as roi_<name>); stop = time of the first frame after.")
    ti.add_column(name="region", description="region name")
    rows = []
    for region, eps in region_eps.items():
        rows += [(a, b, region) for a, b in eps]
    for a, b, region in sorted(rows):
        ti.add_interval(start_time=a, stop_time=b, region=region)
    if rows:
        mod.add(ti)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pynwb.NWBHDF5IO(str(path), "w") as io:
            io.write(nwb)
