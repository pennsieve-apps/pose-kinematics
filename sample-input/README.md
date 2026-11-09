# Sample input

## Real data: DANDI:002047 (open field, WT vs 5xFAD)

Each NWB file of DANDI:002047 (v0.261106.0903, CC-BY-4.0) holds the authors' DeepLabCut pose
(`processing/behavior/PoseEstimation`, ndx-pose 0.4, 14 keypoints, frame timestamps), an
`arena_geometry` table with the px/cm scale, and the expert behaviour labels. Link the NWB files to
the `pose` port; nothing else is needed (the same file serves as its own session NWB).

Download single files by asset id (the whole dandiset is large), e.g. `sub-F01_ses-OF-F01-S1_behavior+image+ophys.nwb` (asset
`52efa19d-15cf-45b6-93ae-4e16a4f4a056`, 45.7 MB):
`https://api.dandiarchive.org/api/assets/52efa19d-15cf-45b6-93ae-4e16a4f4a056/download/`.

Add a `sessions.csv` (`session_id, path, subject_id, condition`) to compare genotypes. The Pennsieve
demo uses six day-1 open-field sessions (WT F35, F40, F48; 5xFAD F07, F14, F20), staged by
`workflow-systems-neuroscience/scripts/stage-demo-data.py openfield-5xfad`.

## From pose-estimate

Link pose-estimate's `pose` output (`pose-estimate/<session_id>_pose.nwb`) to `pose`, and, for the
px/cm scale and the session identity, the original DANDI NWB files to `session`.

## Your own DeepLabCut / SLEAP / Lightning Pose files

Put the `.h5` / `.csv` / `.slp` files in `pose`, plus optionally `<video stem>_timestamps.csv`
(`frame, time`) per video and a `sessions.csv`. Set `ARENA_SIZE_CM` (or `PX_PER_CM`) to get cm, and
`FPS` when there are no timestamps.

## Synthetic

`python -m tests.make_fixture data/fixture` writes four simulated sessions (see `tests/simulate.py`).
