# Inspection App (Qt) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A fast desktop app (`python inspect_app.py`) with every feature of the notebook viewer, up to 10 runs, independent curve/map checklists, built on a quantity registry and an in-memory run cache.

**Architecture:** Data layer (`inspect_cache.RunCache`: everything decompressed once, disk cache) → quantity registry (`inspect_registry`) → Qt widgets (`interacting_maps/inspect_app/`: runs panel, map area, curve area, side plots, export) that only swap data on a step. Exports reuse the matplotlib `draw_inspector` / `export_animation`.

**Tech Stack:** Python 3.12, NumPy, PySide6, pyqtgraph, matplotlib (exports only).

**Spec:** `docs/superpowers/specs/2026-11-26-inspect-app-qt-design.md`

## Global Constraints

- Work in the worktree `.claude/worktrees/inspection-viewer`, branch `inspection-viewer`; commits local, never pushed; no attribution lines.
- `MAX_RUNS = 10` loaded runs of **one segment** (dataset, segment_id, t_start, frame_duration, n_frames); any mix of models, same model allowed.
- Map colours from `viz.render_V / render_I / render_vec` + `viz.map_scale`; scale modes `fixed` (per run, 95th pct of per-frame scales; I: 5th pct lows / 95th pct highs — `inspect_fig._pool` rules), `shared` (pooled over **all loaded** runs), `per frame`.
- Performance: stepping frames with 10 runs loaded and 5 with maps shown averages **< 40 ms per frame** (≥ 25 fps) in the offscreen timing test.
- Load everything, then go: one progress indication; UI enabled after load. Disk cache `<run>/.inspect_cache/` (float32 maps), reused when `states/` and `replay/` file list + mtimes + sizes are unchanged.
- Run labels: model display name (`style.MODELS`) + config fields that differ between loaded runs; else the run folder name. Colours: `style.MODELS` colours when all models differ, otherwise `tab10`.
- Exports via `inspect_fig.export_animation` / `draw_inspector` (full frame, report style); "screenshot" = Qt `grab()` of the window.
- Nothing written under `results/`; tests are standalone `tests/test_*.py` scripts printing `[ok]/[FAIL]`, run with `PYTHONPATH=.` and `QT_QPA_PLATFORM=offscreen`, using `tests/_inspect_fake.py` (no `data/`).
- Notebook viewer (`inspect_viewer.py/.ipynb`) stays; the notebook only gains a launcher cell.

## Review Focus

1. **Stale disk cache:** a run re-made in the same folder (new states) → cache rebuilt, not reused (test in Task 2).
2. **Loading fails half-way** (corrupt/missing state file in one run) → error shown, other runs not half-loaded into the UI, app stays usable (test in Task 5).
3. **10 runs of the same model, identical config except one delta** → 10 distinct labels and colours (test in Task 3).
4. **Toggling all maps off / all curves off** → no crash, empty area, curves/raw still work (test in Task 6).
5. **Probe pixel outside the image after zooming** (click on the border) → clamped to the image, not an IndexError (test in Task 8).

## File structure

| File | Responsibility |
|---|---|
| `interacting_maps/inspect_cache.py` | `RunCache`: all data of one run in memory; disk cache |
| `interacting_maps/inspect_registry.py` | `Quantity` entries; built-in registry |
| `interacting_maps/inspect_labels.py` | `run_labels`, `run_colors` |
| `interacting_maps/inspect_app/__init__.py` | `InspectApp(QMainWindow)`, wiring |
| `interacting_maps/inspect_app/runs_panel.py` | Runs tab (scan, pick, load thread, info table) |
| `interacting_maps/inspect_app/map_area.py` | map rows + raw row, image items, scales, probe marker |
| `interacting_maps/inspect_app/curve_area.py` | curve plots, visibility, draggable marker |
| `interacting_maps/inspect_app/side_plots.py` | probe and residual plots |
| `interacting_maps/inspect_app/view_panel.py`, `export_panel.py` | View and Export tabs |
| `inspect_app.py` | CLI entry: `python inspect_app.py [--root R] [--data D] [run dirs…]` |
| `interacting_maps/inspect_fig.py` (modify) | use `run.display` / `run.color` when set |

---

### Task 1: Dependencies and the quantity registry

**Files:** Create `interacting_maps/inspect_registry.py`; Modify `requirements.txt`; Test `tests/test_inspect_registry.py`

**Interfaces:**
- Produces: `@dataclass Quantity(name: str, kind: str, source: str, compute: Callable, render: str|None, unit: str = '')` with `kind ∈ {'map','curve','iter_curve','fullrate'}`, `source ∈ {'state','replay','csv','raw','derived'}`; `compute(cache, k, it=None)` returns the array for frame k (maps, iteration maps) or the whole series (curves). `REGISTRY: dict[str, Quantity]`; `register(q)`; `of_kind(kind) -> list[Quantity]`; `curve_names(cache) -> list[str]` (csv columns + registered curves present for that run).
- Built-ins: maps `I`,`G`,`F` (state/replay), `V`,`aps` (raw); curves = every numeric csv column (added per run by `curve_names`) + `n_events`; fullrate `omega_fullrate`; iter_curves `|V+F·G|`, `|G−∇I|`, `|R|`; probe series reuse `inspect_nav` formulas.

- [ ] **Step 1:** `pip install PySide6 pyqtgraph`; add both to `requirements.txt` with a comment `# inspect_app (Qt viewer)`.
- [ ] **Step 2: Failing test** `tests/test_inspect_registry.py`: `test_builtins` (`{'I','G','F','V','aps'} <= {q.name for q in of_kind('map')}`; `of_kind('iter_curve')` names == `['|V+F·G|','|G−∇I|','|R|']`); `test_register` (a dummy derived map registered appears in `of_kind('map')`, then removed); `test_curve_names` on a fake `RunCache` (Task 2 not yet there: use a `SimpleNamespace(curves={'err_deg_s':…, 'n_events':…})`) → contains `'err_deg_s'`, `'n_events'`, `'omega_fullrate'`.
- [ ] **Step 3:** run → FAIL (module missing). **Step 4:** implement. **Step 5:** run → all `[ok]`. **Step 6:** commit `"inspect_registry: quantity registry with the built-in quantities"`.

### Task 2: RunCache — load once, disk cache

**Files:** Create `interacting_maps/inspect_cache.py`; Test `tests/test_inspect_cache.py`

**Interfaces:**
- Consumes: `inspect_io.load_run`, `states_io`, `inspect_nav` formulas.
- Produces: `RunCache.open(run_dir, data_root=None, progress=None) -> RunCache` (`progress(frac: float)` callback). Attributes: `run` (RunView), `n_frames, n_iters, H, W, times, dt`, `maps: dict[str, np.ndarray]` (`'I'` (n,H,W) float32, `'G'`,`'F'` (n,H,W,2) float32, `'R'` (n,3) float64), `V: np.ndarray|None` (n,H,W), `aps: list[(t, img)]`, `aps_index: np.ndarray` (nearest APS per frame), `curves: dict[str, np.ndarray]`, `fullrate: dict`, `replays: dict[int, dict]` (all loaded), `frame_scales: dict[str, list]` (per-frame `map_scale` of I, G, F, V). Methods: `map_at(kind, k, it=None) -> np.ndarray` (replay iteration when `it` and replay exist, else state), `has_replay(k) -> bool`, `pixel_series(x, y) -> dict`, `pixel_iterations(k, x, y) -> dict|None`, `iteration_residuals(k) -> dict|None`.
- Disk cache: `<run>/.inspect_cache/maps.npz` (+ `V.npz`, `replay_XXXX.npz`) and `stamp.json` = sorted `[name, size, mtime_ns]` of `states/*` and `replay/*`; rebuilt when the stamp differs.

- [ ] **Step 1: Failing test** `tests/test_inspect_cache.py` (fake runs + `make_dataset`): `test_values` (`c.map_at('I', 1)` ≈ `load_run(...).state(1)['I']` to float32 tolerance; `c.maps['G'].dtype == float32`); `test_replay` (fake replay at frame 2: `c.has_replay(2)`, `c.map_at('I', 2, it=3)` equals replay `I[2]`); `test_disk_cache_reused` (second `open` reads `.inspect_cache` — assert stamp file exists and second open is ≥ 3× faster, or monkeypatch `states_io.load_state` to raise and still open); `test_stale_cache` (rewrite `states/frame_0001.npz` with new values → reopen returns the new values); `test_progress` (callback reaches 1.0, monotone); `test_probe_residuals` (match `inspect_nav.pixel_series` / `iteration_residuals` on the same run).
- [ ] **Step 2:** FAIL. **Step 3:** implement (reuse `inspect_nav` formulas by passing in-memory arrays; do not duplicate them — refactor `inspect_nav` to accept arrays if needed, keeping its current functions working). **Step 4:** `[ok]`, and `tests/test_inspect_tools.py` still `[ok]`. **Step 5:** commit `"inspect_cache: all of a run in memory, disk cache with stamp"`.

### Task 3: Labels and colours

**Files:** Create `interacting_maps/inspect_labels.py`; Modify `interacting_maps/inspect_fig.py`; Test `tests/test_inspect_labels.py`

**Interfaces:**
- Produces: `run_labels(configs: list[dict]) -> list[str]`; `run_colors(configs: list[dict]) -> list[str]` (hex). Differing fields considered: `model`, every key of `params`, `n_iters`, `poisson`, `distortion_mode`; formatting `delta_FR` → `dFR=0.10` (`{:g}` style, `delta_` stripped, first letters upper as in folder names: use the folder-name abbreviations `dFR`, `dimu`, else the stripped key).
- `inspect_fig.draw_inspector` / `_draw_curve`: when a run has attributes `display` / `color`, use them for the row label and line colour (fallback: current behaviour).

- [ ] **Step 1: Failing test:** `test_different_models` (cook, thesis_imu → `['cook', 'thesis+IMU']`, colours = `MODELS` colours); `test_sweep` (10 × thesis_imu, `delta_FR` 0.1…1.0 → 10 distinct labels all starting `thesis+IMU dFR=`, 10 distinct colours); `test_two_fields` (differ in `delta_FR` and `n_iters` → both appear); `test_identical` (two identical configs → labels are the run folder names, distinct).
- [ ] **Step 2:** FAIL. **Step 3:** implement (`test_identical` needs the folder name: signature takes `configs` with an added `'_dir'` key). **Step 4:** `[ok]`; `tests/test_inspect_fig.py` still `[ok]`. **Step 5:** commit `"inspect_labels: run labels from differing config, distinct colours"`.

### Task 4: App shell and entry point

**Files:** Create `interacting_maps/inspect_app/__init__.py`, `inspect_app.py`; Test `tests/test_inspect_app.py`

**Interfaces:**
- Produces: `InspectApp(root='experiments/inspect', data_root=None)` (QMainWindow): left dock `QTabWidget` with tabs `Runs`, `View`, `Export`; central `QSplitter` (vertical) holding map area, curve area, side plots (placeholders here). State: `caches: list[RunCache]`, `frame: int`, `iteration: int` (0 = end of frame), `maps_for: set[int]`, `curves_for: set[int]`, `probe: tuple|None`. Signals/methods used by later tasks: `set_runs(caches)`, `set_frame(k)`, `set_iteration(it)`, `refresh()`. `main(argv)` in `inspect_app.py`: `--root`, `--data`, positional run dirs (loaded directly).
- [ ] **Step 1: Failing test** `tests/test_inspect_app.py` (offscreen, one `QApplication`): `test_shell` (window builds; tab titles `['Runs','View','Export']`); `test_set_frame_clamps` (with two fake caches loaded via `set_runs`, `set_frame(99)` → `frame == n_frames-1`).
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** `[ok]`. **Step 5:** commit `"inspect_app: window shell and entry point"`.

### Task 5: Runs tab

**Files:** Create `interacting_maps/inspect_app/runs_panel.py`; Test `tests/test_inspect_app.py` (extend)

**Interfaces:**
- Consumes: `inspect_io.scan_runs`, `segment_label`, `RunCache.open`, `run_labels`, `run_colors`.
- Produces: `RunsPanel(app)`: folder `QLineEdit` + *scan*, data folder `QLineEdit`, dataset `QComboBox`, segment `QComboBox`, model filter `QComboBox` (`all` + models), run list `QListWidget` with checkboxes (item text = folder name + model; ≤ `MAX_RUNS` checkable, further checks refused with a status message), *load* button, `QProgressBar`, info `QTableWidget` (label, model, iterations, deltas, git commit (+dirty), mean err, β, replayed frames). Loading runs in a `QThread` (or `concurrent.futures` + Qt signal); on success `app.set_runs(caches)` with `display`/`color` set; on any exception: message box/status line, previous runs kept.
- [ ] **Step 1: Failing tests:** `test_pick_and_load` (scan fake root with 3 runs on segment A + 1 on segment B: segment combo has 2 entries; checking 3 runs and calling `panel.load(blocking=True)` → `len(app.caches) == 3`, info table 3 rows); `test_filter_model`; `test_max_runs` (11 fake runs on one segment → only 10 checkable); `test_load_failure_keeps_previous` (corrupt one `states/frame_0001.npz` of a second selection → error reported, `app.caches` unchanged).
- [ ] **Step 2:** FAIL. **Step 3:** implement (`load(blocking=False)`; `blocking=True` for tests). **Step 4:** `[ok]`. **Step 5:** commit `"inspect_app: runs tab (scan, pick run folders, threaded load, info)"`.

### Task 6: Map area

**Files:** Create `interacting_maps/inspect_app/map_area.py`; Test extend `tests/test_inspect_app.py`

**Interfaces:**
- Consumes: `RunCache.map_at`, `frame_scales`, `viz.render_*`, `inspect_fig._pool` (scale rules), registry map entries.
- Produces: `MapArea(app)`: inside a `QScrollArea`, a `pg.GraphicsLayoutWidget` with one row per run in `app.maps_for` (label + one `ImageItem` per map kind in `app.map_kinds`, default `('I','G','F')`) and a raw row (V, APS of the first loaded run). All map `ViewBox`es linked in x and y, `invertY(True)`, aspect locked. `update()` swaps images only; `rebuild()` (on `maps_for`/kinds change) recreates rows. Scale: `scales(mode) -> list[dict]` computed once per mode from `frame_scales` (shared = pooled over all loaded runs; per frame = `{}`). RGB cache dict keyed `(run_i, kind, k, it, mode)`, filled on demand, cleared on runs change. Probe marker: one `ScatterPlotItem` per view at `app.probe`. Click on an image → `app.set_probe(x, y)` clamped to `[0,W-1]×[0,H-1]`.
- [ ] **Step 1: Failing tests:** `test_rows_follow_checklist` (10 fake runs loaded, `maps_for={0,3,5}` → 3 run rows + raw row); `test_image_is_frame` (after `set_frame(2)`, the `ImageItem` of run 0 / I holds `render_I(cache.map_at('I',2), scale)` as uint8, equal to recomputing it); `test_scale_modes` (shared → identical scale dicts for all runs; per frame → `{}`); `test_no_maps` (`maps_for=set()` → no run rows, raw row present, `set_frame` works); `test_click_clamped` (simulated click at (-5, 999) → probe `(0, H-1)`); `test_speed` (10 runs × 30 frames loaded, 5 with maps: stepping all 30 frames twice averages < 40 ms/frame).
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** `[ok]`. **Step 5:** commit `"inspect_app: map area (rows per checklist, linked zoom, scales, probe)"`.

### Task 7: Curve area

**Files:** Create `interacting_maps/inspect_app/curve_area.py`; Test extend `tests/test_inspect_app.py`

**Interfaces:**
- Produces: `CurveArea(app)`: one `pg.PlotWidget` per curve in `app.curve_names_shown` (default `('err_deg_s','beta')`), x linked (seconds since segment start), one `PlotDataItem` per loaded run (`pen=run.color`, name = label), visibility from `app.curves_for`; `'omega_fullrate'` plots reference and gyro of the first run; a movable `pg.InfiniteLine` per plot at `times[frame]`, dragging calls `app.set_frame(nearest)`; clicking a plot jumps to the nearest frame. `update()` moves the lines only; `set_visible(run_i, bool)` toggles lines.
- [ ] **Step 1: Failing tests:** `test_lines_per_run` (10 runs → 10 lines per plot); `test_curves_for` (`curves_for={1}` → only run 1's line visible); `test_drag_marker` (set line value to `times[2]+0.001`, emit `sigPositionChangeFinished` → `app.frame == 2`); `test_no_curves` (`curves_for=set()` → plots exist, no visible line, no crash).
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** `[ok]`. **Step 5:** commit `"inspect_app: curve area (per-run lines, checklist, draggable marker)"`.

### Task 8: View tab and side plots

**Files:** Create `interacting_maps/inspect_app/view_panel.py`, `interacting_maps/inspect_app/side_plots.py`; Test extend `tests/test_inspect_app.py`

**Interfaces:**
- View tab: ▷ play/pause (`QTimer`, interval from *ms/frame* spin box, wraps at the end), ◀ ▶, frame `QSlider`, iteration `QSlider` (enabled when any loaded run `has_replay(frame)`), *curves for* and *maps for* checkable lists (labels in run colours), curve choice (checkable list from `inspect_registry.curve_names`), map kinds (I, G, F checkboxes), scale mode radio (`fixed`/`shared`/`per frame`), spike curve combo + threshold + *◀ spike* / *worst frame* / *spike ▶* (signal = max over runs in `curves_for`, `inspect_nav.worst_frame/next_spike`), *clear pixel probe*.
- Side plots: probe over frames (I, |G|, |F|; runs in `curves_for`; marker at frame), probe over iterations and residuals per iteration (only on replayed frames; marker at iteration).
- [ ] **Step 1: Failing tests:** `test_play` (timer tick advances frame, wraps to 0); `test_iteration_enabled_only_with_replay`; `test_worst_and_spikes` (as `tests/test_inspect_tools.py`, over `curves_for`); `test_probe_plots` (probe set → 3 probe plots with one line per run in `curves_for`); `test_residuals_on_replay` (frame with replay → residual plots present, absent elsewhere).
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** `[ok]`. **Step 5:** commit `"inspect_app: view tab (play, spikes, checklists, scale) and side plots"`.

### Task 9: Export tab, launcher, docs

**Files:** Create `interacting_maps/inspect_app/export_panel.py`; Modify `inspect_viewer.ipynb` (launcher cell), `docs/inspection_viewer.md`, `CLAUDE.md`; Test extend `tests/test_inspect_app.py`

**Interfaces:**
- Export tab: frame range (two spin boxes), fps, format (gif/mp4/png), folder; *export frames*, *export iterations*, *save current view* (matplotlib `draw_inspector` of the runs in `maps_for`, curves of `curves_for`), *screenshot* (`window.grab().save(...)`). File names: `{dataset}_{segment}_{labels joined by '+', spaces→'_'}_{tail}.{ext}`, ≤ 150 chars (truncate labels, keep tail).
- Notebook launcher cell: `subprocess.Popen([sys.executable, 'inspect_app.py', '--root', INSPECT_ROOT])`.
- Docs: guide section "Desktop app" (launch, the two checklists, drag-to-scrub, screenshot); `CLAUDE.md` command + new tests.
- [ ] **Step 1: Failing tests:** `test_export_gif_range` (frames 0–1 → GIF with 2 frames, name contains `f000-001`); `test_screenshot` (PNG written, non-empty); `test_long_names` (10 sweep runs → file name ≤ 150 chars).
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** all viewer/app tests `[ok]`: `test_inspect_registry`, `test_inspect_cache`, `test_inspect_labels`, `test_inspect_app`, `test_inspect_tools`, `test_inspect_fig`. **Step 5:** smoke: `python inspect_app.py <the 3-frame cook inspect run>` opens (manual, user). **Step 6:** commit `"inspect_app: export tab, notebook launcher, docs"`.
