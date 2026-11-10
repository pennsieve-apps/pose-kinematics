"""
Synthetic input for pose-kinematics: four simulated open-field sessions (tests/simulate.py),
two "WT" (more walking, drawn to the centre) and two "KO", as DeepLabCut CSV pose with planted
low-confidence outliers and a 0.3 s tracking gap, a sessions.csv sheet, a frame timestamps CSV,
and (when pynwb + ndx-pose are installed) an ndx-pose NWB pose file and a session NWB with an
arena_geometry table, video frame timestamps and labelled behaviour intervals.

  python -m tests.make_fixture OUT [--layout linked|flat|both] [--seconds 60] [--no-nwb]

writes OUT/linked/{pose,session}/ (one folder per input port) and/or OUT/flat/ (everything side by
side), plus OUT/truth/ (the simulated truth, outside the input).
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tests.simulate import KEYPOINTS, simulate, to_pixels

PX_PER_CM = 8.0
RADIUS_CM = 30.0
CENTER_PX = (300.0, 260.0)
FPS = 30.0
SESSIONS = [  # session_id, file stem, subject, condition, center_bias, walk_bias, seed
    ("wt-01", "mouse01_day1", "m01", "WT", 1.0, 1.6, 1),
    ("wt-02", "mouse02_day1", "m02", "WT", 1.0, 1.6, 2),
    ("ko-01", "mouse03_day1", "m03", "KO", 0.0, 0.7, 3),
    ("ko-02", "mouse04_day1", "m04", "KO", 0.0, 0.7, 4),
]
SCORER = "DLC_resnet50_fixtureNov9shuffle1_1000"


def corrupt(px, rng, gap=None):
    """Confidence per keypoint; 2% outliers (jumps with low confidence) and a tracking gap."""
    n, k, _ = px.shape
    conf = rng.uniform(0.9, 1.0, (n, k))
    bad = rng.random((n, k)) < 0.02
    conf[bad] = rng.uniform(0.0, 0.2, bad.sum())
    px = px.copy()
    px[bad] += rng.uniform(-40, 40, (bad.sum(), 2))
    a, b = gap or (n // 3, n // 3 + 9)
    conf[a:b] = 0.05
    px[a:b] += rng.uniform(-60, 60, (b - a, k, 2))
    return px, conf


def write_dlc_csv(path, px, conf):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scorer"] + [SCORER] * (3 * len(KEYPOINTS)))
        w.writerow(["bodyparts"] + [k for k in KEYPOINTS for _ in range(3)])
        w.writerow(["coords"] + ["x", "y", "likelihood"] * len(KEYPOINTS))
        for i in range(px.shape[0]):
            row = [i]
            for j in range(len(KEYPOINTS)):
                row += [f"{px[i, j, 0]:.3f}", f"{px[i, j, 1]:.3f}", f"{conf[i, j]:.4f}"]
            w.writerow(row)


def arena_outline_px():
    t = np.linspace(0, 2 * np.pi, 400)
    return CENTER_PX[0] + RADIUS_CM * PX_PER_CM * np.cos(t), CENTER_PX[1] + RADIUS_CM * PX_PER_CM * np.sin(t)


def write_nwb(path, sid, subject, ts, px=None, conf=None, states=None, arena=False, video_ts=None):
    import pynwb
    from pynwb.file import Subject

    nwb = pynwb.NWBFile(session_description=f"synthetic open field {sid}", identifier=f"fixture-{sid}",
                        session_start_time=datetime(2026, 11, 9, 10, 0, tzinfo=timezone.utc),
                        subject=Subject(subject_id=subject, species="Mus musculus", sex="M"))
    mod = nwb.create_processing_module("behavior", "behaviour")
    if px is not None:
        from ndx_pose import PoseEstimation, PoseEstimationSeries, Skeleton, Skeletons

        skel = Skeleton(name="skeleton", nodes=KEYPOINTS, edges=np.array([[0, 3], [1, 3], [2, 3], [3, 4], [4, 5]], "uint8"))
        mod.add(Skeletons(skeletons=[skel]))
        series, first = [], None
        for j, k in enumerate(KEYPOINTS):
            s = PoseEstimationSeries(name=k, data=px[:, j, :], unit="pixels", reference_frame="top-left",
                                     confidence=conf[:, j], confidence_definition="likelihood",
                                     timestamps=ts if first is None else first)
            first = first or s
            series.append(s)
        cam = nwb.create_device(name="camera")
        mod.add(PoseEstimation(name="PoseEstimation", pose_estimation_series=series, source_software="DeepLabCut",
                               scorer=SCORER, skeleton=skel, devices=[cam]))
    if video_ts is not None:
        nwb.add_acquisition(pynwb.image.ImageSeries(name="BehaviorVideo", external_file=[f"{sid}.mp4"],
                                                    format="external", starting_frame=[0], timestamps=video_ts))
    if arena:
        x, y = arena_outline_px()
        mod.add(_arena_table(x, y))
    if states is not None:
        from pynwb.epoch import TimeIntervals

        for st in sorted(set(states)):
            m = np.r_[0, (states == st).astype(int), 0]
            d = np.diff(m)
            a, b = np.where(d == 1)[0], np.where(d == -1)[0]
            ti = TimeIntervals(name=st, description=f"Behavioral state. Episodes of {st} (simulated truth).")
            end = ts[-1] + (ts[1] - ts[0])
            for i0, i1 in zip(a, b):
                ti.add_interval(start_time=float(ts[i0]), stop_time=float(ts[i1]) if i1 < len(ts) else float(end))
            mod.add(ti)
    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)


def _arena_table(x, y):
    from hdmf.common import DynamicTable

    t = DynamicTable(name="arena_geometry",
                     description=f"Arena outline in video pixels. Scale: {PX_PER_CM} px per cm along x, "
                                 f"{PX_PER_CM} px per cm along y.")
    t.add_column(name="element", description="element name")
    t.add_column(name="geometry", description="geometry")
    t.add_column(name="x_px", description="x outline, pixels", index=True)
    t.add_column(name="y_px", description="y outline, pixels", index=True)
    t.add_row(element="Arena", geometry="Circle", x_px=list(x), y_px=list(y))
    return t


def have_nwb() -> bool:
    try:
        import ndx_pose  # noqa: F401
        import pynwb  # noqa: F401

        return True
    except ImportError:
        return False


def build(out: Path, layout: str, seconds: float, nwb: bool):
    nwb = nwb and have_nwb()
    staging = out / "_staging"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "pose").mkdir(parents=True)
    (staging / "session").mkdir(parents=True)
    truth_dir = out / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)
    truth = {"px_per_cm": PX_PER_CM, "radius_cm": RADIUS_CM, "center_px": CENTER_PX, "fps": FPS, "sessions": {}}
    sheet = []
    for sid, stem, subject, cond, cbias, wbias, seed in SESSIONS:
        sim = simulate(seconds, FPS, RADIUS_CM, seed, center_bias=cbias, walk_bias=wbias)
        rng = np.random.default_rng(100 + seed)
        px = to_pixels(sim["keypoints"], PX_PER_CM, CENTER_PX)
        px, conf = corrupt(px, rng)
        ts = sim["t"].copy()
        if sid == "wt-02":   # a frame clock with jitter, given as <video stem>_timestamps.csv
            ts = 12.5 + np.cumsum(np.r_[0, rng.uniform(0.9, 1.1, len(ts) - 1) / FPS])
            with open(staging / "pose" / f"{stem}_timestamps.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["frame", "time"])
                w.writerows([[i, f"{t:.6f}"] for i, t in enumerate(ts)])
        if sid == "wt-01":   # session clock starts at 3.25 s; times come from the session NWB
            ts = ts + 3.25
        c = sim["centroid"]
        steps = np.hypot(*np.diff(c, axis=0).T)
        r = np.hypot(c[:, 0], c[:, 1])
        truth["sessions"][sid] = {
            "distance_cm": float(steps.sum()), "center_fraction": float((r <= 0.5 * RADIUS_CM).mean()),
            "state_fraction": {s: float((sim["state"] == s).mean()) for s in ("rest", "walk", "rear")},
            "t0": float(ts[0]), "n_frames": int(len(ts)), "condition": cond,
        }
        with open(truth_dir / f"{sid}_states.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["time", "state"])
            w.writerows([[f"{t:.6f}", s] for t, s in zip(ts, sim["state"])])
        if sid == "ko-02" and nwb:
            name = f"{stem}_pose.nwb"
            write_nwb(staging / "pose" / name, sid, subject, ts, px=px, conf=conf)
        else:
            name = f"{stem}{SCORER}.csv"
            write_dlc_csv(staging / "pose" / name, px, conf)
        if sid == "wt-01" and nwb:
            write_nwb(staging / "session" / f"{stem}_session.nwb", sid, subject, ts, states=sim["state"], arena=True,
                      video_ts=ts)
        sheet.append({"session_id": sid, "path": name, "subject_id": subject, "condition": cond})
    # A session can have several rows, one per file: the session NWB is matched to wt-01 by its own row.
    if nwb:
        sheet.append({"session_id": "wt-01", "path": "mouse01_day1_session.nwb", "subject_id": "m01", "condition": "WT"})
    with open(staging / "pose" / "sessions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["session_id", "path", "subject_id", "condition"])
        w.writeheader()
        w.writerows(sheet)
    with open(truth_dir / "truth.json", "w") as fh:
        json.dump(truth, fh, indent=2)
    if layout in ("linked", "both"):
        dest = out / "linked"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(staging, dest)
        if not any((dest / "session").iterdir()):
            (dest / "session").rmdir()
    if layout in ("flat", "both"):
        dest = out / "flat"
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        for p in staging.rglob("*"):
            if p.is_file():
                shutil.copy2(p, dest / p.name)
    shutil.rmtree(staging)
    return truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--layout", choices=["linked", "flat", "both"], default="both")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--no-nwb", action="store_true", help="DeepLabCut CSV only (no pynwb needed)")
    a = ap.parse_args()
    build(a.out, a.layout, a.seconds, not a.no_nwb)
    print(f"fixture written to {a.out}")


if __name__ == "__main__":
    main()
