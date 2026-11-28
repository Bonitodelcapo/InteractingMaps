# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase (TU Berlin thesis, supervisor Prof. Gallego) implementing the
**Interacting Maps** model: from an event-camera stream it jointly infers image
intensity `I`, gradient `G`, optical flow `F`, and the headline output — camera
angular velocity `R`/`ω`. Pure Python/NumPy, no deep learning.

**Read [architecture.md](architecture.md) first** — it is the authoritative
end-to-end reference (the three model variants, the `V+F·G=0` / `G=∇I` / `F=C·R`
constraints, the β-scale ambiguity, data→metric pipeline, module-by-module). This
file only adds the operational layer (commands, knobs, workflow) and flags where
the code has moved past architecture.md.

The CMax work (event-based, IMU-free angular-velocity anchor) has its own
reference: [cmax.md](cmax.md).

## Environment & commands

```bash
pip install -r requirements.txt
```

Everything runs from the repo root (scripts rely on relative `data/` and
`results/` paths). Datasets live in `data/<name>/` as `events.txt calib.txt
imu.txt groundtruth.txt images.txt` (+ `omega_gt.txt` for synthetic); they are
git-ignored and must be present locally — `download_gdrive.sh` fetches them.

**Single-run harness** (`evaluation.py`, argparse, `--exp` selects experiment
2, 5, 6 or 9; see §6.6 of architecture.md). Every run behind the report is a
command of `report/run_experiments.py` instead (see the report workflow below):

```bash
python evaluation.py --exp 2 --dataset boxes_rotation --model thesis_imu --segment seg_A
```

- `--model`: `cook`, `thesis`, `thesis_imu`, `thesis_cmax`, `thesis_cmax_v2`, `all`;
  plus the persistence network (Martel §6.5/§6.8.3, `interacting_maps/network_persistence.py`):
  `thesis_persist`, `thesis_bm`, `thesis_persist_bm` (`config.PERSIST_PARAMS`).
- `--exp 2` = multi-frame tracking (the core experiment → `tracking.csv`,
  `run.json`, `tracking_plot.png`, per-frame panels unless `--no-frames`);
  `5` = MP4s from saved frames; `6` = basin of attraction; `9` = distortion ablation.
- `--distortion-mode {undistort_events,C_full}` overrides the global default.

Each tracking run writes one self-describing `run.json`
(`{provenance, config, summary}`, with git commit + dirty flag) and, for reported
runs (`out_root='results'`), mirrors to a local MLflow store — see
[experiment_tracking.py](experiment_tracking.py), which also holds `find_run`,
the one lookup of a run folder by its stored config. `MLFLOW_TRACKING_URI` points it at a running
server; default is a local SQLite backend (`mlflow ui --backend-store-uri
sqlite:///mlflow.db`). `IM_MLFLOW=0` disables; `IM_MLFLOW_ALL=1` also logs sweeps.

**Live demo:** `demo.ipynb` (matplotlib animation of all maps).

**Discover new segments** from a dataset's IMU and print a `config.py` block:
`python interacting_maps/find_segments.py data/<name>/imu.txt`.

**Convert ECRot/ESIM ROSbags** into the pipeline format:
`python convert_ecrot.py --bag <path>`. ECRot datasets are named
`ecrot_<sequence>` (`ecrot_city`, `ecrot_street`, `ecrot_bicycle`); `--name`
defaults to `ecrot_<bag stem>`.

## Tests

Tests live in `tests/` and are **standalone scripts, not pytest** — run them
from the repo root with the root on the path, and read stdout (each prints
`[ok]`/`[FAIL]` lines; they are not collected by a runner):

```bash
PYTHONPATH=. python tests/test_golden_tracking.py    # tracking numbers vs golden values
PYTHONPATH=. python tests/test_poisson_solvers.py    # Poisson/Richardson solver recovery checks
PYTHONPATH=. python tests/test_viz.py                # plot conventions (style.py, viz.py)
PYTHONPATH=. python tests/test_cmax_frontend.py      # CMax step-1a validation vs GT & IMU
PYTHONPATH=. python tests/test_distortion_cmax.py
PYTHONPATH=. python tests/test_ecrot_convert.py      # ROSbag converter on a tiny synthetic bag
PYTHONPATH=. python tests/test_cmax_v1_vs_v2.py
PYTHONPATH=. python tests/test_cmax_gradient.py      # analytic contrast gradient vs finite differences
```

## Configuration lives in config.py

`config.py` is pure data + helpers and is the single place to tune the model.
Hyper-parameters (`THESIS_PARAMS`, `COOK_PARAMS`, `ITERS_PER_FRAME`) are the
per-relation relaxation step sizes `delta_*`. **The in-code defaults are the
values every number in the report uses** — changing them silently changes
published results, so treat them as load-bearing and check the inline comments
before editing.

Two global switches control the model (both documented in architecture.md §6.2/§6.5):

- `DISTORTION_MODE = 'C_full'` (default) — lens distortion carried inside the
  kinematic matrix `C` (distortion-aware coords + Brown–Conrady Jacobian), not by
  `undistort_events` before binning. `--exp 9` compares the two modes.
- `POISSON_MODE = 'iterative' | 'fft' | 'dct'` — how the thesis network recovers
  `I` from `G` (Richardson step vs exact frequency-domain solve); thesis network only.

`DATASET_SEGMENTS` holds hand-picked constant-ω segments (generated by
`interacting_maps/find_segments.py`); the synthetic ECRot sequences are
`ecrot_bicycle` and `ecrot_street`; `initial_R: None` means warm-start `R` from the IMU gyro at
`t_start`.

## Report / paper workflow

The LaTeX report (`report/bare_jrnl.tex`) is driven by `report/run_all.sh`
(guide: [report/RUN_ALL.md](report/RUN_ALL.md)), run from the repo root. It runs
every job in parallel, skips the ones whose code, data and command are unchanged,
then draws and tabulates. Four scripts, one job each:

```bash
python report/run_experiments.py --what main     # runs (one --what per experiment)
python report/make_figures.py                     # draws every report figure
python report/make_tables.py                      # writes every table, table_<name>.tex
python report/text_numbers.py                     # every number in the text no table holds
```

`report/report_runs.py` holds what the four must agree on: the Table II
segments, where each run's folder is, the APS lookup and the image-settings
search. Figures use the report's font (Times, `interacting_maps/style.py`).

Key rule enforced by `run_experiments.py`: **every reported number is recomputed,
never read back from `results/`** (which mixes stale configurations). Runs for the
cited tables land in `results/`; exploratory searches go to
`experiments/<name>/`. All report runs use `distortion_mode='C_full'` and
the segments in `report_runs.SEGMENTS` (= the report's Table II).

## Conventions that bite

- **Units:** `R` is rad/**frame**; `ω = R/dt` is rad/s; `frame_duration` is both
  the event-accumulation window and `dt`. Warm start `R_init = ω_gyro·dt`.
- **Two ground truths, different jobs:** `imu.txt` gyro is a *model input* (and the
  frame-0 warm start); `groundtruth.txt` (Vicon, differenced as `R1ᵀR2`, body
  frame) is the *independent scoring reference*. Scoring an IMU-fed model against
  the gyro is circular — use Vicon.
- **Pixel indexing:** `[...,0]=x` (cols), `[...,1]=y` (rows). Quaternions ordered
  `(qx,qy,qz,qw)`.
- `R` is set **once** before the frame loop and is fully recurrent thereafter —
  there is no per-frame GT oracle/reseed on this branch (see architecture.md §6.6).
