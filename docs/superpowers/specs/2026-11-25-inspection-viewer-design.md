# Inspection viewer: link a curve back to the data and the model state

Status: design, approved approach A (2026-11-25). Not built.

## Why

A tracking curve (error, β, |r|, …) shows a spike at, say, frame 6. Today there is no way to see
what happened there: was the input itself odd (event burst, fast motion, blurred APS frame), or did
the model go wrong (F or G breaking down)? A run keeps only `maps_final.npz`, so the model's state
at frame 6 is gone, and the raw data has to be dug out by hand.

The goal is to go from "a point on a curve" to "the data and the model at that moment" in one step,
for one model or for up to five models on the same segment, and to see how the model evolves over
frames and over the iterations inside one frame.

## Goals

1. **Per-frame checkpoints.** Every tracking run saves the full recurrent state at the end of each
   frame's last iteration (the todo item in `docs/notes/todo.md`).
2. **Iteration replay.** A script re-runs one frame from checkpoint k−1 and writes every iteration
   to disk. Runs on the cluster or locally; the viewer never computes.
3. **Viewer** (notebook): pick 1–5 runs on the same segment, a frame slider and an iteration slider,
   dropdowns for which curves and which map panels to show. Clicking a curve jumps to that frame.
4. **Offline export:** the same figure, looped over frames (or iterations), written as GIF/MP4,
   headless, so it can run on the cluster.

## Non-goals

- No compute inside the viewer (no model stepping, no CMax solves).
- No web app, no server, no new visualisation framework.
- Not a replacement for the report figures; the report pipeline is untouched.

## Anti-goals (a failure even if it works)

- A second rendering path that looks different from the report (other colour scales, other units).
- A viewer that only works for one model, or hard-codes the current set of curves.
- Checkpoints that look right but don't replay exactly: then the iteration view shows a model state
  that never existed.
- Changing any reported number. Saving state must not change the computation.

## Constraints

- Runs are produced on the cluster; inspection happens on the laptop. Everything goes through files.
- `results/` is load-bearing: no scratch writes there. Exploratory checkpoints go to `experiments/<name>/`.
- Sizes: ≈ 1 MB/frame (float32, 180×240), ≈ 75–100 MB per 75-frame run before compression.
  Start with a few segments, then all models on all segments.
- Code rules: small atomic functions, reuse `viz.py` (`draw_map`, `map_scale`, `net_maps`, `draw_omega`,
  `draw_series`), `eval_io` (APS/IMU/GT by window), `data_loader` (events); import from core modules,
  not from scripts. Scripts ship as `.py` + `.ipynb`.
- Same quantity ⇒ same y-limits across panels (and same map scale across models in one view).
- Units as in the code: frames index `tracking.csv`'s `frame`; the time axis is seconds since segment start.

## Approaches considered

- **A. Notebook, matplotlib + ipywidgets (chosen).** One function draws the whole inspector figure;
  the widget wraps it for live use, a loop over it makes the GIF/MP4. One rendering path, report look,
  reuses `viz.py`. Cost: redraws lag a little with 5 models × several panels.
- **B. Rerun (rerun.io).** Smooth scrubbing and linked time cursors for free, but a new dependency with
  its own rendering, layout outside our code, and still needs A's figure function for GIFs. Can be
  added later on top of the same loader if the lag becomes a problem.
- **C. Web app (Panel/HoloViews, Dash).** Real linked brushing, most code to write and maintain.
  Against "simple".

## Design

Three layers, each usable on its own.

### 1. State (in the model code)

- `get_state() -> dict[str, np.ndarray]` / `set_state(dict)` on each network class:
  - Cook (`interacting_maps/network.py`): `I, G, F, R`.
  - Thesis / thesis_imu (`network_dissertation.py`): `I, G, F, R` (`Quantity.value`). Checked in the
    code on 2026-11-26: `gradient_accumulator` is reset every iteration, so it is not recurrent.
  - CMax V2: nothing extra — `Cost_CMax.set_frame` is rebuilt from each frame's events.
  - CMax V1: `omega_cmax_prev` lives in `experiment_tracking`'s loop, not the network; it is saved with
    the checkpoint as loop state.
- In `experiment_tracking`, after `net.step(...)` and only when `rc.save_states`: write checkpoint k. One file per frame or one file
  per run is the builder's call (see below).
- **Acceptance test:** for each model, `set_state(checkpoint k−1)` + `step(V_k)` reproduces
  checkpoint k bit-exactly (or to float tolerance if not achievable; state which). A standalone test
  script in `tests/`, same `[ok]/[FAIL]` style.
- Golden tracking numbers (`tests/test_golden_tracking.py`) must not change.

### 2. Replay (script)

- Input: a run folder and a frame k. Loads checkpoint k−1 (or the run's initial state for k = 0),
  rebuilds `V_k` and the frame's side inputs exactly as `experiment_tracking` does (gyro for
  thesis_imu, CMax anchor for V1 with the saved `omega_cmax_prev`, the event window for V2), steps
  one iteration at a time and saves every iteration's state inside the inspect run folder
  (e.g. `experiments/inspect/<run>/replay/frame_0006.npz`); never under `results/`.
- Check: the last replayed iteration equals checkpoint k.
- This supersedes how `run_experiments --what converge` builds its state: today it iterates frame k
  from a **fresh** network (zero maps, `initial_R`, CMax `omega_init = 0`), not from the state the
  real run had after frames 0…k−1. Whether `fig_converge` moves onto replay is a separate decision
  (it would change a report figure).

### 3. Viewer (notebook + module)

- **Loader** `load_run(folder)` → config (`run.json`), rows (`tracking.csv`), `state(k)`,
  `iterations(k)` (if a replay exists), and raw data for frame k's window: event frame V / event
  count, nearest APS image, IMU and Vicon ω around the window. Lazy: nothing heavy is read until asked.
- **Figure** `draw_inspector(runs, k, it=None, panels=[...], curves=[...]) -> Figure`:
  - one row per run: the selected map panels (I, G, F, V, I read-out, …) via `draw_map`, shared
    scale per quantity across runs;
  - one raw-data row: V (events) and APS;
  - the selected curves stacked below, every run overlaid, a vertical marker at frame k.
- **Curves** are any column of `tracking.csv` (so new columns appear automatically), plus derived
  raw-data curves: events per frame, |ω| of Vicon and IMU at full rate.
- **Widget**: frame slider, iteration slider (enabled when a replay exists for that frame), multi-select
  for curves and panels, click on a curve axis → jump to the nearest frame.
- **Export** `export_animation(runs, frames | iterations, path, fps)` loops `draw_inspector` and writes
  GIF/MP4 with imageio; no display needed (Agg backend).
- **Compare 1–5 runs:** they must share dataset, segment, `t_start` and `frame_duration`; the loader
  checks this and refuses otherwise.

## Decided by the user

- Save the full state at every frame, final iteration; warm-start replay to see the iterations.
- All models, all segments; start with a few.
- Both views: over frames and over iterations.
- Both uses: live interactive and offline GIF/MP4.
- Viewer reads only from disk.
- 1 model or 2–5 models side by side.
- Approach A.
- **Saving is opt-in:** a `save_states` switch on `RunConfig`, off by default, not part of the stored
  config `find_run` matches on. Sweeps, the golden test and the report runs are unchanged.
- **Nothing of this goes into `results/`** — neither checkpoints nor replays. Inspectable runs and
  their replays live under `experiments/inspect/` (mirror tree of the run's config).
- **Cluster entry point:** a new `run_experiments --what inspect --sequences …` that reruns the
  report's main configuration with `save_states` on, into `experiments/inspect/`. Run folders are
  copied back to the laptop; the datasets being inspected must be present locally under `data/`.
- **Map scale (decided after the 2026-11-25 spike, `experiments/scale_spike/`): one fixed scale per
  quantity for the whole sequence, set separately for each model.** A switch shares it across the
  compared models when the size difference itself is the question. The spike showed: the per-frame
  scale hides breakdowns (thesis_imu's I at poster seg_C frame 14 looks only slightly worse); the EMA
  makes identical data look different depending on how many frames came before (V scale still
  sliding at frame 30); sharing across models whitens cook's G (≈ 4× smaller than thesis_imu's).
  The fixed scale needs one pass over the run's checkpoints when it is loaded.
- **Still open:** whether `draw_inspector` replaces `_save_3col_frame` / `--exp 5`. Until decided,
  leave both untouched.

## Calls made in the design (not the user's; check these)

- **Iteration slider = absolute iteration.** Runs with fewer iterations hold their last iteration;
  runs without a replay at that frame show their checkpoint and are marked "no replay".
- **Verified from the code:** `step(V, n_iters=1)` called N times equals one `step(V, n_iters=N)`
  for both networks (only per-frame inputs are re-set; gradients are cleared every iteration), so
  replay can step one iteration at a time. Cook's initial state is seeded (`default_rng(42)` in
  `run_config.make_network`); the initial state is still saved explicitly as checkpoint −1.

## Left to the builder

- Checkpoint file layout (one `.npz` per frame vs one per run) and float32 vs float64 on disk —
  float64 if needed for exact replay.
- Exact default panel and curve sets.
- Where the module lives (proposed: `interacting_maps/inspect.py` for loader + figure,
  `inspect_viewer.ipynb` + `.py` at the root).
- Adding `ipywidgets` and `ipympl` to `requirements.txt` (not installed now).
