# Inspection app (Qt): the viewer, fast

Status: design, approach A chosen by the user (2026-11-26). Not built.
Predecessor: [2026-11-25-inspection-viewer-design.md](2026-11-25-inspection-viewer-design.md)
(the notebook viewer, `inspect_viewer.py` / `.ipynb`, on branch `inspection-viewer`).

## Why

The notebook viewer has every feature the user wants, but it lags so much that scrubbing and
playing are unusable. Each slider step rebuilds a whole matplotlib figure, ipympl ships it to the
browser as an image, and the data layer re-reads and decompresses `.npz` files on every step
(states per frame, replays for the residuals, all frames again when a pixel probe is set,
`omega_fullrate` recomputed each time). The features, the behaviour and the layout are right;
the drawing medium and the data access are wrong.

Goal: the same tool, on the laptop, that reacts instantly.

## Goals

1. A desktop app, `python inspect_app.py`, opening a window. No notebook, no browser.
2. **Instant interaction:** play at ≥ 25 frames/s and scrub without visible lag with 10 runs loaded, 5 with maps shown ×
   I, G, F + events + APS + 3 curves loaded (poster seg_C, 150 frames, 180×240).
3. **Every feature of the notebook viewer**, same behaviour:
   - Runs: folder to scan, data folder, dataset → segment → run folders (≤ 10, one segment, any
     models, same model allowed — filterable by model) → load;
     run info table (model, iterations, deltas, git commit/dirty, mean error, β, replayed frames).
   - View: play/pause + speed, ◀ ▶, frame slider, iteration slider (enabled only where a loaded
     run has a replay; 0 = end-of-frame state); curves (any `tracking.csv` column, `n_events`,
     `omega_fullrate`); maps I/G/F; scale mode fixed per model / shared / per frame; spike curve +
     threshold + ◀ spike / worst frame / spike ▶; click a curve → jump to frame.
   - Pixel probe: click a map pixel → I, |G|, |F| at that pixel over frames, and over the
     iterations of a replayed frame; marker on every map; clear button.
   - Residuals per iteration on a replayed frame: mean |V+F·G|, mean |G−∇I|, |ω|.
   - Export: frame range, fps, gif/mp4/png, output folder; export frames / iterations / current view;
     names carry dataset, segment, models, frames.
4. **New, cheap in Qt and wanted for the same purpose:** drag the frame marker on any curve to
   scrub; zoom/pan linked across all map panels; curves share one x-axis (time).

## Non-goals

- No hosting on the cluster / in a browser (approach B was the alternative for that).
- No new analysis: same quantities as the notebook viewer.
- No change to how runs are made (`--what inspect`, `replay_frame.py`, `states_io`).

## Anti-goals

- A second look that disagrees with the report: map colours must come from the same rules
  (`viz.render_*`, `viz.map_scale`, the fixed per-model scale of the scale spike).
- Exports that differ from the report style: GIF/MP4/PNG export keeps using
  `inspect_fig.export_animation` / `draw_inspector` (matplotlib, report fonts).
- Fast but wrong: a frame shown must be exactly that frame's saved state (or that iteration).
- A rewrite of the logic: loader, catalog, navigation helpers and scale rules are reused, not copied.

## Constraints

- Laptop: Intel Core Ultra 7 165U, 64 GB RAM (~32 GB free), Windows 11. Memory is not a constraint
  at this scale (5 runs × 150 frames of decompressed states ≈ 1.3 GB).
- Python 3.12 (the user's plain install). New dependencies: `PySide6`, `pyqtgraph` (pip).
- Same repo rules: core logic in `interacting_maps/`, scripts import from it; nothing written under
  `results/`; standalone `tests/test_*.py` scripts printing `[ok]/[FAIL]`; local runs only as smoke
  tests (the real runs come from the cluster).

## Approaches considered

- **A. Qt desktop app with pyqtgraph (chosen).** Images and plots update in place
  (`ImageItem.setImage`, `PlotDataItem.setData`), built for real-time display; native window;
  all features port; reuses the Python modules. Cost: Qt layout code, two new dependencies.
- **B. Local browser app (Panel/Bokeh).** In-place updates over a websocket, ~10–20 fps; could later
  be served from JupyterHub. Cost: web stack, slower than A.
- **C. Rerun.** Instant scrubbing in a generic viewer, but the run picker, spike tools, scale modes
  and probe do not port.
- Rejected: keep the notebook and only optimise — ipympl still ships a full image per step.

## Design

### Data layer: load once, then only index

`RunCache` (new, `interacting_maps/inspect_cache.py`) wraps a `RunView` (`inspect_io`) and, when a
run is opened, loads everything the app reads per step into memory once:

- states of all frames, cropped to H×W (I, G, F, R);
- event frames V of all frames (one pass of `EventFrameSequence`), events per frame, APS index;
- curves (`RunView.curves()`), `omega_fullrate` (computed once);
- replays: listed at open, each loaded on first use and kept.

Derived data are computed once and kept: per-frame `map_scale` values (for the three scale modes),
pixel series are read from the in-memory states (no file access). Loading runs off the UI thread
with a progress indication; the window stays responsive.

### Rendering

- One `ImageItem` per map panel, created when runs are loaded; changing frame/iteration/scale only
  calls `setImage` with the RGB from the existing `viz.render_V / render_I / render_vec` at the
  chosen scale (so colours match the report).
- If rendering on the fly cannot reach the frame-rate goal, cache rendered RGB per
  (run, kind, frame, scale mode) — memory allows it.
- Curves: one `PlotWidget` per selected curve, x linked (time since segment start), one line per run
  in the model colours of `style.MODELS`, a movable `InfiniteLine` as the frame marker.
- Probe and residual plots: `PlotWidget`s in a side/bottom dock, updated with `setData`.
- Map panels: `ViewBox`es with linked x/y (zoom/pan together), aspect locked, y pointing down as in
  the notebook (`[...,0]=x` columns, `[...,1]=y` rows).

### Layout (proposal)

```
┌ Runs ─────────────┐┌───────────── maps: one row per run (I | G | F) ─────────────┐
│ folder, scan      ││ cook        [I] [G] [F]                                    │
│ data folder       ││ thesis+IMU  [I] [G] [F]                                    │
│ dataset/segment   ││ raw         [events V] [APS]                               │
│ models, load      │├─────────────────────────────────────────────────────────────┤
│ run info table    ││ curves (stacked, shared time axis, draggable marker)       │
├ View ─────────────┤├─────────────────────────────────────────────────────────────┤
│ ▷ ◀ ▶ frame, it.  ││ probe over frames / iterations · residuals per iteration    │
│ scale, spikes     │└─────────────────────────────────────────────────────────────┘
│ curves, maps      │
├ Export ───────────┤
└───────────────────┘
```

Controls in a left dock with three tabs (Runs, View, Export), as in the notebook.

### Reuse

`inspect_io` (RunView, scan_runs, check_comparable, segment_label), `inspect_nav` (worst_frame,
next_spike, pixel series and residual formulas — adapted to read the in-memory cache),
`inspect_fig` (fixed_scales rules, export_animation, draw_inspector for exports),
`viz` (renderers, map_scale), `style` (model colours/names).

### Tests

Headless with `QT_QPA_PLATFORM=offscreen`: build the window on the fake runs of
`tests/_inspect_fake.py`, drive it through its methods (load, set frame, click, spike, scale mode,
probe, export) and check state and what the image items hold; plus a timing check that stepping
through all frames of 5 fake runs averages under 40 ms per frame.

## Decided by the user

- Laptop, ease of use first; another host only if the laptop were not enough (it is).
- Approach A (Qt + pyqtgraph).
- Port all features of the notebook viewer.
- **Loading: load everything, then go.** One progress bar while the runs are loaded; the UI is
  enabled when all frames, event frames, curves and scales are in memory. A decompressed cache on
  disk (`<run>/.inspect_cache/`, float32 maps, written on first open, reused when the run's
  `states/` are unchanged — mtime/size check) makes reopening the same runs take seconds. Replays
  are loaded in the same pass (they are part of "everything").
- **Up to 10 runs loaded (one constant, `MAX_RUNS = 10`), with two independent checklists:**
  *curves for* (which runs draw a line in every curve plot — all overlaid; toggling only shows/hides
  lines) and *maps for* (which runs get an I | G | F row — e.g. 3 of the 10; toggling adds/removes
  rows from in-memory data, no reload). The info table always lists all loaded runs. The map area
  scrolls when many rows are ticked. The pixel probe and the residual plots follow *curves for*.
  The limit of 10 is not the laptop's (≈ 2.6 GB of states for 10 × 150 frames) but screen space.
- **Several runs of the same model in one view — required.** Up to 10 runs of one segment, any mix:
  e.g. 10 runs of `thesis_imu` from a parameter sweep. The run picker lists **run folders**, not
  models (filter by model). Each run gets a **label** = model display name + the config fields that
  differ between the loaded runs (deltas, `n_iters`, `poisson`, …), e.g.
  `thesis+IMU dFR=0.10` / `thesis+IMU dFR=0.50`; when nothing differs, the run folder name.
  Each run gets its **own colour** (model colours from `style.MODELS` when the models are all
  different, a distinct-colour palette otherwise). Labels are used in the map rows, legends, info
  table and export file names.

## Calls made in the design (not the user's)

- The notebook viewer stays in the repo, unchanged, until the app has proved itself; removing it is a
  later decision.
- Exports keep the matplotlib path (report look) rather than pyqtgraph's exporters.
- Drag-to-scrub and linked zoom are added (cheap in Qt, same purpose).
- *shared* scale pools over **all loaded** runs (not only those shown), so colours do not shift when
  a run is ticked on or off.
- Exports (GIF/MP4/PNG via `draw_inspector`) always show the full frame in the report style,
  independent of the on-screen zoom, without the probe/residual plots — as in the notebook.
  A separate **"screenshot"** button saves exactly what the window shows (Qt `grab()`), zoom,
  probe and residual plots included.
- No `.ipynb` twin for the app (a Qt window cannot live in a notebook): `inspect_viewer.ipynb`
  gets one extra cell that launches `inspect_app.py` as a separate process. Noted exception to the
  `.py` + `.ipynb` rule.

## Left to the builder

- Whether on-the-fly rendering meets the frame-rate goal or the RGB cache is needed.
- Exact widget arrangement within the docks; window size/state persistence.
- Threading details of the loader (QThread vs. concurrent.futures).
