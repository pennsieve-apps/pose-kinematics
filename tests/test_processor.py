"""The synthetic fixture through the processor (no Docker)."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests import make_fixture

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("fixture")
    make_fixture.build(out, "both", 60.0, nwb=True)
    return out


def run(input_dir, output_dir, **params):
    env = dict(os.environ, INPUT_DIR=str(input_dir), OUTPUT_DIR=str(output_dir), MPLBACKEND="Agg")
    env.update({k: str(v) for k, v in params.items()})
    return subprocess.run([sys.executable, "-m", "processor.main"], cwd=REPO, env=env, capture_output=True, text=True)


def metrics(out):
    with open(out / "pose-kinematics" / "metrics.csv") as fh:
        return {r["session_id"]: r for r in csv.DictReader(fh)}


@pytest.fixture(scope="session")
def linked_out(fixture_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("out-linked")
    r = run(fixture_dir / "linked", out, PX_PER_CM=8)
    assert r.returncode == 0, r.stdout + r.stderr
    return out


def test_all_sessions_and_outputs(linked_out):
    top = linked_out / "pose-kinematics"
    m = metrics(linked_out)
    assert set(m) == {"wt-01", "wt-02", "ko-01", "ko-02"}
    for sid in m:
        for suffix in ("_behavior.nwb", "_kinematics.parquet", "_trajectory.png", "_occupancy.png"):
            assert (top / f"{sid}{suffix}").exists(), suffix
    for name in ("occupancy.csv", "metrics.csv", "sessions.csv", "group_stats.csv", "summary.json",
                 "provenance.json", "speed_distribution.png", "group_comparison.png"):
        assert (top / name).exists(), name
    summary = json.loads((top / "summary.json").read_text())
    assert summary["app"] == "pose-kinematics"
    assert all(f["caption"] and (linked_out / f["path"]).exists() for f in summary["figures"])
    prov = json.loads((top / "provenance.json").read_text())
    assert set(prov) >= {"app", "version", "variant", "tools", "parameters", "inputs", "started", "finished",
                         "wall_seconds", "peak_rss_mb", "gpu", "sessions", "citations"}


def test_distance_and_centre_against_truth(linked_out, fixture_dir):
    truth = json.loads((fixture_dir / "truth" / "truth.json").read_text())["sessions"]
    m = metrics(linked_out)
    for sid, t in truth.items():
        assert float(m[sid]["distance_moving"]) == pytest.approx(t["distance_cm"], rel=0.1), sid
    # wt-01 has the session NWB's arena outline, so the centre zone is exact.
    assert float(m["wt-01"]["center_fraction"]) == pytest.approx(truth["wt-01"]["center_fraction"], abs=0.03)


def test_time_sources(linked_out, fixture_dir):
    m = metrics(linked_out)
    assert m["wt-01"]["time_source"].startswith("session NWB")
    assert m["wt-02"]["time_source"] == "mouse02_day1_timestamps.csv"
    assert m["ko-02"]["time_source"].startswith("nwb PoseEstimation")
    k = pd.read_parquet(linked_out / "pose-kinematics" / "wt-01_kinematics.parquet")
    assert k["time"].iloc[0] == pytest.approx(3.25)   # the session clock is kept, not re-zeroed
    k2 = pd.read_parquet(linked_out / "pose-kinematics" / "wt-02_kinematics.parquet")
    assert k2["time"].iloc[0] == pytest.approx(12.5)


def test_planted_behaviours_visible_in_kinematics(linked_out, fixture_dir):
    k = pd.read_parquet(linked_out / "pose-kinematics" / "wt-01_kinematics.parquet")
    states = pd.read_csv(fixture_dir / "truth" / "wt-01_states.csv")["state"].to_numpy()
    med = k.groupby(states)[["speed", "body_length"]].median()
    assert med.loc["walk", "speed"] > 10 and med.loc["rest", "speed"] < 2
    assert med.loc["walk", "body_length"] > med.loc["rest", "body_length"] > med.loc["rear", "body_length"]


def test_nwb_output(linked_out):
    import pynwb

    with pynwb.NWBHDF5IO(str(linked_out / "pose-kinematics" / "wt-01_behavior.nwb"), "r") as io:
        nwb = io.read()
        assert nwb.identifier != "fixture-wt-01" and "fixture-wt-01" in nwb.notes   # new UUID, source noted
        assert nwb.source_script_file_name
        assert nwb.subject.subject_id == "m01"
        beh = nwb.processing["behavior"]
        pe = beh["PoseEstimation"]
        s = pe.pose_estimation_series["nose"]
        assert s.unit == "centimeters"
        assert s.timestamps[0] == pytest.approx(3.25)
        assert "speed" in beh["BehavioralTimeSeries"].time_series
        regions = beh["arena_regions"].to_dataframe()
        assert set(regions["region"]) >= {"center", "periphery"}


def test_group_stats(linked_out):
    with open(linked_out / "pose-kinematics" / "group_stats.csv") as fh:
        rows = {r["metric"]: r for r in csv.DictReader(fh)}
    c = rows["center_fraction"]
    assert c["test"] == "Welch t-test" and c["n_WT"] == "2" and c["n_KO"] == "2"
    assert float(c["mean_WT"]) > float(c["mean_KO"])


def test_flat_layout_and_arena_size(fixture_dir, tmp_path):
    r = run(fixture_dir / "flat", tmp_path, ARENA_SIZE_CM=60)
    assert r.returncode == 0, r.stdout + r.stderr
    m = metrics(tmp_path)
    assert len(m) == 4 and all(v["units"] == "cm" for v in m.values())
    assert float(m["wt-01"]["px_per_cm_x"]) == pytest.approx(8.0, rel=0.01)   # from the arena outline


def test_session_nwb_scale_without_parameters(fixture_dir, tmp_path):
    r = run(fixture_dir / "linked", tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    m = metrics(tmp_path)
    assert m["wt-01"]["units"] == "cm" and float(m["wt-01"]["px_per_cm_x"]) == pytest.approx(8.0)
    assert m["ko-01"]["units"] == "px"   # nothing gives its scale


def test_rois(fixture_dir, tmp_path):
    rois = json.dumps([{"name": "left", "type": "rectangle", "x0": -40, "y0": -40, "x1": 0, "y1": 40, "units": "cm"},
                       {"name": "spot", "type": "circle", "center": [300, 260], "radius": 80}])
    r = run(fixture_dir / "linked", tmp_path, PX_PER_CM=8, ROIS=rois)
    assert r.returncode == 0, r.stdout + r.stderr
    occ = pd.read_csv(tmp_path / "pose-kinematics" / "occupancy.csv")
    assert {"roi_left", "roi_spot"} <= set(occ["region"])


def test_dlc_csv_only_fixture(tmp_path):
    make_fixture.build(tmp_path / "fx", "linked", 20.0, nwb=False)
    r = run(tmp_path / "fx" / "linked", tmp_path / "out", PX_PER_CM=8)
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(metrics(tmp_path / "out")) == 4


def test_bad_parameters_and_no_input(fixture_dir, tmp_path):
    assert run(fixture_dir / "linked", tmp_path / "a", SMOOTHING="kalman").returncode == 2
    assert run(fixture_dir / "linked", tmp_path / "b", ROIS="[{\"type\": \"star\"}]").returncode == 2
    assert run(fixture_dir / "linked", tmp_path / "c", CENTER_FRACTION=1.5).returncode == 2
    (tmp_path / "empty").mkdir()
    assert run(tmp_path / "empty", tmp_path / "d").returncode == 1


def test_sleap_and_lightning_pose_files(tmp_path):
    """The same simulated session as a SLEAP analysis HDF5 (read through movement / sleap-io) and a
    Lightning Pose CSV gives the same kinematics as the DeepLabCut CSV."""
    import h5py

    from tests.simulate import KEYPOINTS, simulate, to_pixels

    sim = simulate(20.0, 30.0, 30.0, seed=7)
    px = to_pixels(sim["keypoints"], 8.0, (300.0, 260.0))
    n, k = px.shape[:2]
    conf = np.full((n, k), 0.95)
    for name in ("dlc", "lp", "sleap"):
        (tmp_path / name).mkdir()
    make_fixture.write_dlc_csv(tmp_path / "dlc" / "mouse.csv", px, conf)
    text = (tmp_path / "dlc" / "mouse.csv").read_text().replace(make_fixture.SCORER, "heatmap_tracker")
    (tmp_path / "lp" / "mouse.csv").write_text(text)
    with h5py.File(tmp_path / "sleap" / "mouse.analysis.h5", "w") as f:
        f["tracks"] = np.transpose(px, (2, 1, 0))[None].astype("float64")          # (tracks, 2, nodes, frames)
        f["point_scores"] = np.transpose(conf)[None]                                  # (tracks, nodes, frames)
        f["instance_scores"] = np.ones((1, n))
        f["tracking_scores"] = np.ones((1, n))
        f["track_occupancy"] = np.ones((n, 1), dtype=np.uint8)
        f["node_names"] = [s.encode() for s in KEYPOINTS]
        f["track_names"] = [b"mouse"]
    dist = {}
    for name in ("dlc", "lp", "sleap"):
        out = tmp_path / f"out-{name}"
        r = run(tmp_path / name, out, PX_PER_CM=8, FPS=30)
        assert r.returncode == 0, name + r.stdout + r.stderr
        m = metrics(out)
        assert list(m) == ["mouse"], m
        dist[name] = float(m["mouse"]["distance_travelled"])
    assert dist["lp"] == pytest.approx(dist["dlc"], rel=1e-6)
    assert dist["sleap"] == pytest.approx(dist["dlc"], rel=1e-3)
