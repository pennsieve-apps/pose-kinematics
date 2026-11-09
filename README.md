# pose-kinematics

This app is built on **[movement](https://github.com/neuroinformatics-unit/movement)**, the
Python toolbox for analysing animal body movements developed by Nikoloz Sirmpilatze, Chang Huan
Lo, Sofía Miñano, Brandon D. Peri, Dhruv Sharma, Laura Porta, Iván Varela, Adam L. Tyson and
contributors at the **Neuroinformatics Unit of the Sainsbury Wellcome Centre and Gatsby
Computational Neuroscience Unit, University College London**
([documentation](https://movement.neuroinformatics.dev), Zenodo
[10.5281/zenodo.12755724](https://doi.org/10.5281/zenodo.12755724)). It reads pose files made with
[DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) (Mathis lab, EPFL; Mathis et al. 2018),
[SLEAP](https://github.com/talmolab/sleap) (Talmo Pereira, Salk Institute, and the Murthy and
Shaevitz labs, Princeton; Pereira et al. 2022; read through [sleap-io](https://github.com/talmolab/sleap-io)
by Divya Murali, Liezl Maree, David Samy and Talmo Pereira) and [Lightning Pose](https://github.com/paninski-lab/lightning-pose)
(Paninski lab, Columbia), and NWB files with the [ndx-pose](https://github.com/rly/ndx-pose)
extension (Ryan Ly, Ben Dichter, Alexander Mathis, Talmo Pereira and colleagues) through
[PyNWB](https://github.com/NeurodataWithoutBorders/pynwb) (Neurodata Without Borders; Rübel et al.
2022). **Pennsieve packages these tools as a workflow step; it did not develop them.** The demo
data are DANDI:002047 by Viktor Plusnin, Nikita Pospelov, Olga Rogozhnikova, Vladimir Sotskov, Anna
Ivanova, Nikita Savelev, Ksenia Toropova, Olga Ivashkina and Konstantin Anokhin ([10.48324/dandi.002047/0.261106.0903](https://doi.org/10.48324/dandi.002047/0.261106.0903), CC-BY-4.0).

**What it does.** Pose estimation gives, for every video frame, the pixel position of each body
part (nose, ears, body centre, tail base...) and a confidence. Those raw tracks jitter, drop out
when a body part is hidden, and are in pixels. This app turns them into clean trajectories and the
standard open-field measures a neuroscientist reports: how far and how fast the animal moved, how
it turned, how much time it spent in the centre of the arena versus along the walls
(thigmotaxis, a classic index of anxiety-like behaviour; Simon et al. 1994, Prut & Belzung 2003),
and in any regions you define. It then compares conditions from `sessions.csv` (e.g. wild type vs
5xFAD). It is the second step of the **Markerless Behaviour** workflow:
`pose-estimate → pose-kinematics → behavior-syllables → neuro-report`, and its NWB output feeds
`neuro-align` in the decoding workflow. It also runs on its own on any pose file or DANDI NWB with
`PoseEstimation`.

For a computational analyst: movement's `filter_by_confidence`, `interpolate_over_time`
(`max_gap` in frames) and `savgol_filter` / `rolling_filter` run on an `xarray` whose time
coordinate is the session clock; speeds come from movement's `compute_speed` (xarray
`differentiate` on the real, uneven frame times). NWB pose is read with pynwb directly, because
movement 0.17's NWB loader rebuilds time as `frame / fps` and ignores stored timestamps (movement
issue #1169); in DANDI 002047 frame intervals vary from 7 to 59 ms, so that would misplace frames
by up to seconds relative to the calcium imaging.

## Method

Per session, streaming (one session in memory at a time):

1. **Read.** ndx-pose NWB (`PoseEstimation` in any processing module) with its `timestamps`;
   DeepLabCut `.h5`/`.csv`, SLEAP `.slp`/analysis `.h5` (via sleap-io, not the GPL-ambiguous
   sleap-nn), Lightning Pose `.csv` through movement's loaders (tested on the same simulated
   session written in all three formats).
   Several individuals: the most completely tracked one is analysed (logged). Several pose files
   for one session (pose-estimate writes both `_pose.nwb` and the DLC `_pose.h5`): the NWB wins.
2. **Time** (conventions.md, "Time"): NWB timestamps; for other formats, the session NWB's
   `PoseEstimation` or video `ImageSeries` timestamps, else `<video stem>_timestamps.csv`
   (`frame, time`), else `frame / FPS`. `summary.json` says which (`time_source`), and the times are
   never re-zeroed.
3. **Clean.** Points with confidence `< CONFIDENCE_THRESHOLD` are removed; then, once the px → cm
   scale is known, a **jump filter** removes confidently tracked points whose step from the last
   accepted point of the same keypoint implies more than `MAX_SPEED_CM_S` (default 150 cm/s:
   about three times the fastest real centroid step in the demo data, 1.8 cm per 33-ms frame ≈
   54 cm/s, and above the running speeds of mice in an open field, so only tracking errors exceed
   it; a whole excursion away and back is removed, and the count is logged and in `metrics.csv`
   `jumps_dropped`); gaps up to `MAX_GAP_S`
   are linearly interpolated (a step across a gap that stays missing counts as 0 distance, so long
   tracking losses shorten the distance travelled; `valid_fraction` says how much is missing); then a Savitzky–Golay filter (order 2, `SMOOTH_WINDOW_S`; Savitzky &
   Golay 1964) or a rolling median smooths every keypoint.
4. **Scale (px → cm)**, first that applies: `PX_PER_CM`; `ARENA_SIZE_CM` over the arena outline
   in the session NWB (else over the extent of all tracked keypoints, approximate); the scale
   written in the session NWB's `arena_geometry` description (DANDI 002047: "9.2147 px per cm
   along x, 8.4293 px per cm along y", anisotropic); a linear fit of the body centroid to a
   `Position` series in cm on the same clock (R² ≥ 0.98). Otherwise everything stays in pixels and
   the outputs say so.
5. **Arena.** The outline from the session NWB's `arena_geometry` table (`x_px`, `y_px`, the row
   named `Arena`), else the extent of the tracks with `ARENA_SHAPE` (auto: rectangle when the
   animal visits the corners of its bounding box). Centre zone = the arena shrunk to
   `CENTER_FRACTION` of its diameter; periphery = the rest; wall zone = within
   `WALL_ZONE_FRACTION` × diameter of the wall (thigmotaxis = time there); user `ROIS`.
6. **Kinematics.** Body centroid (a body-centre keypoint, else the mean of the non-tail
   keypoints), its speed, distance travelled (sum of steps) and distance while moving (steps at
   ≥ `IMMOBILE_SPEED`), heading (tail base → head, counter-clockwise on screen, radians), angular
   speed, body length (nose → tail base), every keypoint's speed, immobile time.
7. **Groups.** Per-session metrics compared between the `GROUP_BY` groups: Welch's t-test (Welch
   1947), with the Mann–Whitney U test (Mann & Whitney 1947) and Hedges' g effect size (Hedges
   1981) alongside; one-way ANOVA and Kruskal–Wallis for 3+ groups. The session is the unit; no
   correction across the five metrics (exploratory).

## Ports

| Port | Kind | Files | Notes |
|---|---|---|---|
| `pose` | input (subfolder) | `*.nwb` with `PoseEstimation`, DLC `.h5`/`.csv`, SLEAP `.slp`/`.h5`, LP `.csv`; optional `sessions.csv`, `<video stem>_timestamps.csv` | pose-estimate's output, or any lab's pose files, or DANDI NWB files |
| `session` | input (subfolder, optional) | `*.nwb` | timestamps, `arena_geometry`, `Position` in cm, session identity; matched by session_id (sessions.csv rows may list several files per session) |
| `kinematics` | output | `pose-kinematics/...` (below) | the whole OUTPUT_DIR |

Without links, files are found by name and content anywhere under INPUT_DIR (symlinks followed).
session_id comes from `sessions.csv`, else `sub-<a>_ses-<b>`, else the file stem without the
pose tool's suffix (`<video>DLC_resnet50_...`, `.predictions`, `_pose`).

## Parameters

| Parameter | Default | What it does |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.6` | Points below this confidence are removed (DeepLabCut's usual p-cutoff); 0 = none. |
| `MAX_GAP_S` | `0.5` | Longest gap (s) filled by linear interpolation. |
| `SMOOTHING` | `savgol` | `savgol`, `median` (robust to single-frame jumps) or `none`. |
| `SMOOTH_WINDOW_S` | `0.2` | Smoothing window (s), rounded to an odd number of frames. |
| `ARENA_SIZE_CM` | `0` | Arena diameter or side (cm), or `WxH`; 0 = unknown. |
| `PX_PER_CM` | `0` | Pixels per cm; overrides everything; 0 = automatic. |
| `ARENA_SHAPE` | `auto` | `auto`, `circle`, `rectangle` (when there is no outline). |
| `CENTER_FRACTION` | `0.5` | Centre zone = arena shrunk to this fraction of its diameter (a quarter of the area). |
| `WALL_ZONE_FRACTION` | `0.1` | Wall zone width as a fraction of the diameter (9.8 cm in the 98 cm demo arena). |
| `ROIS` | empty | JSON list of `{"name", "type": "circle"/"rectangle"/"polygon", "center"+"radius" / "x0","y0","x1","y1" / "points", "units": "px"/"cm"}`, or a GeoJSON FeatureCollection of polygons (movement / napari). |
| `BODY_KEYPOINTS` | `auto` | Keypoint(s) averaged into the centroid. |
| `HEAD_KEYPOINT` | `auto` | Head end of the heading (head centre / neck, else midpoint of the ears, else nose). |
| `TAIL_KEYPOINT` | `auto` | Tail base. |
| `FPS` | `0` | Frame rate for files without timestamps; 0 = from the file, else 30 (warning). |
| `IMMOBILE_SPEED` | `2.0` | cm/s below which the animal counts as immobile. |
| `MAX_SPEED_CM_S` | `150` | Jump filter: points implying a faster step are removed before interpolation (needs a cm scale); 0 = off. |
| `GROUP_BY` | `condition` | sessions.csv column compared (falls back to `group`). |
| `THREADS` | `0` | numpy/BLAS threads; 0 = cgroup CPU quota. |

Bad parameters exit 2; no pose file or every session failed exit 1; a failing session is logged
and skipped.

## Outputs (`OUTPUT_DIR/pose-kinematics/`)

| File | Contents |
|---|---|
| `<session_id>_behavior.nwb` | Session fields copied from the source NWB (`session_start_time`, `timestamps_reference_time`, `session_id`, `subject`, `experimenter`, `institution`, `lab`, `keywords`, `experiment_description`); a new UUID `identifier`, with the source file and its identifier in `notes`; units spelled the NWB way (`centimeters`, `centimeters/second`, `radians`). Passes nwb-inspect (schema, NWB Inspector, DANDI) on all six demo sessions. `processing/behavior`: `PoseEstimation` (cleaned keypoints, cm or pixels, original confidence, timestamps on the session clock; `Skeletons`), `BehavioralTimeSeries` (`speed`, `heading`, `angular_speed`, `body_length`, `distance_to_wall`, `keypoint_speed` (T × K)), `Position/centroid`, `arena_regions` (`TimeIntervals`: start_time, stop_time, region = center / periphery / wall_zone / roi_<name>). |
| `<session_id>_kinematics.parquet` | One row per frame: `time, frame, centroid_x, centroid_y, speed, step_distance, heading, angular_speed, body_length, kp_<k>_x, kp_<k>_y, kp_<k>_speed` (every keypoint), `in_center, in_periphery, in_wall_zone, distance_to_wall, roi_<name>, valid`. Parquet metadata `pennsieve` (JSON): units, px_per_cm, scale source, keypoints, skeleton edges, centroid/head/tail keypoints, fps, time source, arena, sheet columns and the source NWB identity. The input of behavior-syllables. |
| `occupancy.csv` | `session_id, region, seconds, fraction, entries, latency_s, mean_speed, subject_id, condition`. |
| `metrics.csv` | Per session: `units, n_frames, duration_s, fps_effective, time_source, valid_fraction, jumps_dropped, distance_travelled, distance_moving, mean_speed, median_speed, p95_speed, immobile_fraction, center_fraction, center_entries, periphery_fraction, thigmotaxis, mean_distance_to_wall, mean_body_length, mean_abs_angular_speed, px_per_cm_x, px_per_cm_y` + sessions.csv columns. |
| `group_stats.csv` | Per metric: `n_`, `mean_`, `sd_` per group, `test, statistic, p_value, test_secondary, p_value_secondary, difference, hedges_g`. |
| `sessions.csv` | `session_id, path, subject_id, condition, group, ..., status, n_frames`. |
| `speed_distribution.png`, `group_comparison.png`, `<session_id>_trajectory.png`, `<session_id>_occupancy.png`, `<session_id>_pose_qc.png` | Figures; captions in `summary.json`. |
| `summary.json`, `provenance.json` | Headline numbers per session, group tests, figures with captions; versions, parameters, cleaning report per keypoint, timings, peak memory. |

## Results on the demo data

DANDI 002047, six 12-min day-1 open-field sessions (98 cm circular arena), the authors' DeepLabCut
pose (14 keypoints) from the NWB files, defaults. Output in
`workflow-systems-neuroscience/data/work/openfield-5xfad/pose-kinematics/`.

| Session | Genotype | Valid frames | Distance (m) | Mean speed (cm/s) | Centre | Thigmotaxis | Immobile |
|---|---|---|---|---|---|---|---|
| F35 | WT | 97 % | 50.0 | 7.1 | 4.2 % | 80.7 % | 37.6 % |
| F40 | WT | 92 % | 40.2 | 5.9 | 2.2 % | 90.8 % | 45.0 % |
| F48 | WT | 100 % | 12.6 | 1.7 | 2.2 % | 92.6 % | 82.0 % |
| F07 | 5xFAD | 91 % | 48.3 | 7.3 | 4.8 % | 78.8 % | 45.8 % |
| F14 | 5xFAD | 97 % | 39.3 | 5.6 | 11.9 % | 68.1 % | 44.9 % |
| F20 | 5xFAD | 98 % | 51.7 | 7.2 | 18.1 % | 49.9 % | 33.0 % |

**Check against the authors' own numbers**: distance travelled is within 1–3 % of the path length
of the authors' `Position` series in five sessions (e.g. F35 4 996 vs 5 047 cm, F48 1 263 vs
1 272 cm), mean speed within 0.2 cm/s of their `speed` series, and our per-frame speed
correlates with theirs at Pearson r = 0.96–0.99 (Spearman 0.92–0.98) in all six. **F07** is the
exception the jump filter exists for: its DeepLabCut body centre jumps ~80 cm away and back within
0.3 s at t ≈ 46 s and 103 s with high confidence (speeds up to 943 cm/s). Without the filter these
excursions added 4.8 m (9 %) to F07's distance and, as single huge speeds, dropped the speed
correlation to Pearson r 0.38 (Spearman 0.95, measured by the independent review). With `MAX_SPEED_CM_S=150` (173 points dropped, on 11 of the 14 keypoints) F07
travels 48.3 m (was 53.0 m) and r = 0.99; the authors' own `Position` has a similar excursion at
t ≈ 53 s (6 steps > 10 cm, 3.3 m), and without it their path is 50.4 m. The filter drops 11–105
points in the other sessions (nose, head and tail base mostly) without changing their distances
by more than 0.1 %. The px → cm scale came from the `arena_geometry` description in every file.

**WT vs 5xFAD** (n = 3 sessions per group; Welch t-test, Hedges g = 5xFAD − WT in pooled SDs):
centre time 11.6 ± 6.6 % vs 2.9 ± 1.1 % (p = 0.15, g = 1.5); thigmotaxis 66 ± 15 % vs 88 ± 6 %
(p = 0.10, g = −1.6); distance 46.4 ± 6.4 m vs 34.3 ± 19.4 m (p = 0.39, g = 0.67). The 5xFAD mice
explored the centre more and hugged the wall less, the direction reported for disinhibited
behaviour in 5xFAD mice, but **with three animals per group the comparison is badly underpowered**
(Mann–Whitney p cannot go below 0.1 at n = 3 + 3) and one WT animal (F48) barely moved. Treat it
as a demonstration, not a finding.

On the synthetic fixture (four simulated 60-s sessions with planted outliers and a tracking gap,
known truth): distance while moving within 2–5 % of the true path, centre time exact (0.374 vs
0.374) where the arena outline is known, speed and body length per planted behaviour as
simulated (walk 16 cm/s, rest/rear < 1 cm/s; body 9.0 / 6.5 / 4.1 cm).

## Runtime

`computeTypes: [standard]`, 1 vCPU / 2048 MiB, 15 min timeout: within the platform's Lambda limits,
so the platform can run it as a Lambda step (it decides from `runtime` and the image size). Measured in the
linux/amd64 image under emulation on an M-series Mac (Docker `--memory 2g --cpus 1 --network
none`), six 12-min sessions (129 484 frames, 14 keypoints), with the jump filter (two cleaning
passes): **54 s** processing, 78 s including container start (up to 327 s when sharing the
machine with other builders' jobs), peak RSS **0.65 GB**, cgroup `memory.peak` **1.0 GB**. Image: 1.1 GB unpacked (`/usr/local`), well under
Lambda's 10 GB. Memory grows with one session; a one-hour
session at 30 fps (~110 k frames) should stay under 2 GB; much longer recordings or many
keypoints: use the standard compute type with more memory.

It is Lambda-safe: the demo run above is the brief's Lambda test (`docker run --read-only --user
65534:65534 --tmpfs /tmp:mode=1777`, HOME and TMPDIR under `/tmp/w`, INPUT_DIR and OUTPUT_DIR under
`/tmp/s`):
caches (matplotlib, numba, XDG, movement's log) go to `$TMPDIR` / `$HOME`, nothing is read from the
image's `/tmp` or HOME.

## Running it locally

```sh
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
make test PYTHON=.venv/bin/python    # pytest on the synthetic fixture, no Docker
make run PYTHON=.venv/bin/python     # Docker over data/fixture/linked -> data/output/
```

## Limitations

- **Garbage in, garbage out.** The cleaning removes low-confidence points and jitter; it cannot
  fix confident mistakes (a swapped left/right ear, a cable tracked as the head). Check
  `<session_id>_pose_qc.png`. With a head-mounted miniscope, as in the demo data, the nose and
  head points are often hidden (the authors' nose confidence has a median of 0.83).
- **Units need a scale.** Without `PX_PER_CM`, `ARENA_SIZE_CM` or a session NWB that gives one,
  everything is in pixels and between-session comparisons are only valid with the same camera
  set-up. The arena estimated from the tracks (no outline) is approximate, and wrong when the
  animal did not reach every wall (short sessions): centre and wall-zone times then shift.
- **Zone definitions differ between labs.** The centre zone here is half the arena diameter;
  the 002047 authors' own `center` intervals use their own zone mask. Compare numbers only within
  one definition.
- **Distance travelled includes pose jitter** while the animal is still; `distance_moving` (steps
  at ≥ `IMMOBILE_SPEED`) is less sensitive to it.
- Single animal only (the best-tracked individual of a multi-animal file).
- Group statistics are per session, uncorrected across metrics, and underpowered with a few
  animals per group.

## Licences

Apache-2.0 for this app (see `LICENSE`). Bundled components and their licences:
`THIRD_PARTY.md`. No GPL code is imported.

## Citations

Please cite these works if you use this app. Verified on Crossref/DataCite and PubMed; full
author lists in `CITATION.cff`. The entries below are generated from the registry records.

movement, as its README asks ("please cite the following Zenodo DOI"; its CITATION.CFF names Nikoloz
Sirmpilatze, Chang Huan Lo, Sofía Miñano, Brandon D. Peri, Dhruv Sharma, Laura Porta, Iván Varela and
Adam L. Tyson; DataCite lists the v0.17.0 GitHub contributors as creators):

> Niko Sirmpilatze, Chang Huan Lo, sfmig, et al. **neuroinformatics-unit/movement: v0.17.0.** Zenodo (2026).
> https://doi.org/10.5281/zenodo.12755724.

Pose estimation tools whose files this app reads (DeepLabCut asks for both papers):

> Mathis A, Mamidanna P, Cury KM, Abe T, Murthy VN, Mathis MW, Bethge M. **DeepLabCut: markerless pose estimation of user-defined body parts with deep learning.** Nature Neuroscience 21(9):1281-1289 (2018).
> https://doi.org/10.1038/s41593-018-0209-y, PMID 30127430.

> Nath T, Mathis A, Chen AC, Patel A, Bethge M, Mathis MW. **Using DeepLabCut for 3D markerless pose estimation across species and behaviors.** Nature Protocols 14(7):2152-2176 (2019).
> https://doi.org/10.1038/s41596-019-0176-0, PMID 31227823.

> Pereira TD, Tabris N, Matsliah A, Turner DM, Li J, Ravindranath S, Papadoyannis ES, Normand E, Deutsch DS, Wang ZY, McKenzie-Smith GC, Mitelut CC, Castro MD, D’Uva J, Kislin M, Sanes DH, Kocher SD, Wang SSH, Falkner AL, Shaevitz JW, Murthy M. **SLEAP: A deep learning system for multi-animal pose tracking.** Nature Methods 19(4):486-495 (2022).
> https://doi.org/10.1038/s41592-022-01426-1, PMID 35379947.

SLEAP files are read with sleap-io (no paper; https://github.com/talmolab/sleap-io, by Divya Murali,
Liezl Maree, David Samy and Talmo Pereira, Talmo Lab, Salk Institute; listed in `CITATION.cff`).

Neurodata Without Borders (file format; PyNWB and ndx-pose, https://github.com/rly/ndx-pose):

> Rübel O, Tritt A, Ly R, Dichter BK, Ghosh S, Niu L, Baker P, Soltesz I, Ng L, Svoboda K, Frank L, Bouchard KE. **The Neurodata Without Borders ecosystem for neurophysiological data science.** eLife 11:e78362 (2022).
> https://doi.org/10.7554/eLife.78362, PMID 36193886.

Methods:

> Savitzky A, Golay MJE. **Smoothing and Differentiation of Data by Simplified Least Squares Procedures.** Analytical Chemistry 36(8):1627-1639 (1964).
> https://doi.org/10.1021/ac60214a047.

> Simon P, Dupuis R, Costentin J. **Thigmotaxis as an index of anxiety in mice. Influence of dopaminergic transmissions.** Behavioural Brain Research 61(1):59-64 (1994).
> https://doi.org/10.1016/0166-4328(94)90008-6, PMID 7913324.

> Prut L, Belzung C. **The open field as a paradigm to measure the effects of drugs on anxiety-like behaviors: a review.** European Journal of Pharmacology 463(1-3):3-33 (2003).
> https://doi.org/10.1016/s0014-2999(03)01272-x, PMID 12600700.

> WELCH BL. **THE GENERALIZATION OF ‘STUDENT'S’ PROBLEM WHEN SEVERAL DIFFERENT POPULATION VARLANCES ARE INVOLVED.** Biometrika 34(1-2):28-35 (1947).
> https://doi.org/10.1093/biomet/34.1-2.28, PMID 20287819.

> Mann HB, Whitney DR. **On a Test of Whether one of Two Random Variables is Stochastically Larger than the Other.** The Annals of Mathematical Statistics 18(1):50-60 (1947).
> https://doi.org/10.1214/aoms/1177730491.

> Hedges LV. **Distribution Theory for Glass's Estimator of Effect size and Related Estimators.** Journal of Educational Statistics 6(2):107-128 (1981).
> https://doi.org/10.3102/10769986006002107.

Demo data (DataCite creators; DataCite lists as contributors Viktor Plusnin, Olga Rogozhnikova, Vladimir
Sotskov, Nikita Pospelov, Anna Ivanova, Nikita Savelev, Ksenia Toropova, Olga Ivashkina and Konstantin
Anokhin; CC-BY-4.0):

> Plusnin V, Pospelov N. **Hippocampal CA1 one-photon calcium imaging and pose-tracked behavior in wild-type and 5xFAD mice across four behavioral paradigms.** DANDI Archive (2026).
> https://doi.org/10.48324/dandi.002047/0.261106.0903.

Lightning Pose (Biderman et al. 2024, https://github.com/paninski-lab/lightning-pose) `.csv` files are read
too; its paper is not listed in `CITATION.cff` because its Crossref author list (the International
Brain Laboratory's members, some of them twice) cannot be written as CFF, which requires unique authors.

## Acknowledgements

movement is developed by the Neuroinformatics Unit (Sainsbury Wellcome Centre and Gatsby
Computational Neuroscience Unit, UCL) and its contributors. NWB and ndx-pose are developed by the Neurodata Without Borders community; the demo data are
shared on the DANDI Archive (BRAIN Initiative). Thanks to the DANDI:002047 authors for releasing
pose, expert behaviour labels and imaging together under CC-BY.
