# Run Comparison Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Jupyter/ipywidgets viewer that puts up to 20 finished tracking runs side by side, output by output, so the effect of a parameter is seen along one row.

**Architecture:** A small package `interacting_maps/compare_runs/`: `catalog` (scan run folders, cascading choices), `outputs` (registry of every output kind + lazy loaders), `bands` (draw one band of the grid from loaded runs, pure functions returning figures/HTML), `app` (the widgets). A thin `compare_runs.ipynb` at the repo root builds and shows the app. Nothing is computed except display (Poisson solve of a saved G for "I from G").

**Tech Stack:** Python 3.12, ipywidgets 8, matplotlib 3.10, pandas, numpy; core modules `experiment_tracking`, `interacting_maps.style`, `interacting_maps.viz`, `interacting_maps.eval_io`, `interacting_maps.network_dissertation`.

**Spec:** `docs/superpowers/specs/2026-11-27-run-compare-design.md`

## Global Constraints

- Viewer only: never run the network, never write into run folders, `results/`, `report/`.
- Every ω / ω-error in deg/s via `style.to_deg_s`; never print or plot rad/s.
- Same quantity ⇒ shared y-limits across its axes (data-driven, `sharey`), shared colour scale across runs for the same map kind (`viz.map_scale` on the stacked arrays).
- Model colours / names from `style.MODELS`; plots inside `style.use_style('screen')`.
- Import only from core modules (`experiment_tracking`, `interacting_maps/*`, `config`); never from `walkthrough/`, `report/`, `evaluation.py`.
- Max 20 run slots, min 1.
- Curves x-axis: time since each run's `t_start` = `style.frame_times(dt, n)` indexed by the csv `frame` column.
- More than one segment (dataset + segment_id) among chosen runs ⇒ a warning line naming the segments: comparing runs of different segments does not make sense.
- Dropped: exp11 outputs (`exp11_*`, `iter_frames/`, `iteration_evolution.gif`).
- Tests: standalone scripts, `PYTHONPATH=. python tests/<file>.py`, `[ok]`/`[FAIL]` lines, exit 1 on failure (pattern of `tests/test_viz.py`). Locally only smoke checks.
- Commits: local only, no `--author`, no attribution lines; never push.

## Review Focus

1. A run whose `tracking.csv` lacks a column (old runs: no `recon_r`, `I_std`) — it is left out of that plot and the plot says `not in: #2, #5`, never silently.
2. Legacy folders with only `params.json`/`summary.json` (no `segment_id`, no `poisson`) — segment_id parsed from the folder name (`seg_[^_]+`), poisson `'iterative'`, a missing delta shown as `'—'`; they appear in the cascade like new runs.
3. Frame slider beyond a run's sequence length — that cell shows `no frame k`, the others still draw.
4. Two slots picking the same folder — allowed; both columns show it (labels `#1`, `#3` keep them apart).
5. Removing a slot while the comparison shows it — the grid redraws without it; removing the last remaining slot is refused (button disabled at 1).

---

### Task 1: Catalog — scan and cascading choices

**Files:**
- Create: `interacting_maps/compare_runs/__init__.py` (empty for now)
- Create: `interacting_maps/compare_runs/catalog.py`
- Test: `tests/test_compare_runs.py`

**Interfaces:**
- Produces:
  - `MISSING = '—'`
  - `BASE_KEYS = ['model', 'dataset', 'segment_id', 't_start', 'dt', 'n_frames', 'n_iters', 'poisson', 'distortion_mode', 'root']`
  - `scan(roots: list[str]) -> pd.DataFrame` — one row per folder under any root (recursive) holding `run.json` or `params.json` (via `experiment_tracking.read_run`); columns `folder`, every `BASE_KEYS` (`dt` = config `frame_duration`; `root` = the scanned root the folder is under), every `delta_*` / other key found in any `config['params']` (missing → `MISSING`), `err_deg_s` (= summary `mean_err_deg_s`, NaN if absent). Unreadable folders skipped.
  - `choice_keys(cat) -> list[str]` — `BASE_KEYS` + the param columns, in that order.
  - `options(cat, chosen: dict) -> dict[str, list]` — per key, the sorted distinct values of rows matching every *other* chosen key (`'any'` = no filter); sort by `str`.
  - `matches(cat, chosen) -> pd.DataFrame` — rows matching all choices, sorted by `err_deg_s` ascending.

- [ ] **Step 1: Write the failing tests** — a helper `make_run(root, rel, config, summary, legacy=False)` writes `run.json` (or `params.json`+`summary.json`) into a temp dir. Tests:
  - `test_scan`: 3 runs (2 new: thesis/boxes_rotation/seg_A dt 0.02 & 0.01 with `delta_FR` 0.3; 1 legacy cook folder named `seg_B_t7.3_dt20ms…` without segment_id/poisson/`delta_FR`) → `len(cat)==3`; legacy row `segment_id=='seg_B'`, `poisson=='iterative'`, `delta_FR==MISSING`; `set(cat.dt)=={0.02,0.01}`.
  - `test_options_cascade`: `options(cat, {'model':'thesis'})['dt'] == [0.01, 0.02]`; `options(cat, {'model':'thesis','dt':0.01})['model']` still contains `'cook'` (a chosen key keeps its alternatives).
  - `test_matches_order`: two matching rows come back lowest `err_deg_s` first.
- [ ] **Step 2: Run** `PYTHONPATH=. python tests/test_compare_runs.py` — Expected: ImportError / `[FAIL]`.
- [ ] **Step 3: Implement `catalog.py`** with the interfaces above (segment_id fallback regex `r'(seg_[^_]+)'` on the folder basename, as `walkthrough` does).
- [ ] **Step 4: Run** the test — Expected: all `[ok]`, exit 0.
- [ ] **Step 5: Commit** `git add interacting_maps/compare_runs tests/test_compare_runs.py && git commit -m "compare_runs: catalog scan + cascading choices"`

### Task 2: Output registry and lazy loaders

**Files:**
- Create: `interacting_maps/compare_runs/outputs.py`
- Test: `tests/test_compare_runs.py` (add)

**Interfaces:**
- Consumes: nothing from Task 1 beyond a folder path.
- Produces:
  - `Output = namedtuple('Output', 'key label band')`, `band ∈ {'curves','table','figures','maps','frames'}`.
  - `OUTPUTS: list[Output]` — exactly, in this order:
    curves: `omega` (ω per axis, from `est_w*`), `err` (`err_deg_s`), `dir_err` (`dir_err_deg`), `beta`, `recon_r` (`recon_r` + `recon_r_inloop`), `I_std`, `t_frame` (`t_frame_s`), `contrast` (`iwe/iwe_log.csv`);
    table: `table`;
    figures: `tracking_plot.png`, `recon_final.png`, `recon_plot.png`, `iwe/contrast_curve.png`, `exp1_convergence.png`, `exp3_parameters.png`, `exp6_basin.png` (key = the relative path);
    maps: `final:V`, `final:I`, `final:I_from_G`, `final:G`, `final:F`, `final:aps`, and the same six with `center:`;
    frames: `video_frames`, `iwe_frames` (`iwe/frame_*.png`), `iwe_sequence.gif`, `states` (`states/frame_*.npz` → V, I, G, F), `comparison.mp4`.
  - `available(folder) -> set[str]` — keys whose file/column exists (curve keys check the csv header; `final:*` keys check the npz members; `*:aps` needs `t_mid` resolvable (Task 3 rule) and `data/<dataset>/images.txt`; `I_from_G` needs `G`). Reads only headers / npz member lists.
  - `read_tracking(folder) -> pd.DataFrame` (cached by folder, `functools.lru_cache`).
  - `read_maps(folder, which: str) -> dict` (`which` ∈ `'final','center'`; npz → dict of arrays).
  - `frame_paths(folder, seq: str) -> list[str]` — sorted by frame number (`video_frames`, `iwe_frames`, `states`).

- [ ] **Step 1: Write the failing tests** — `test_available`: a temp run with `tracking.csv` holding `frame,time,est_wx,est_wy,est_wz,gt_wx,gt_wy,gt_wz,err_deg_s,ref_source` and `maps_final.npz` with only I,G,F and `video_frames/frame_0000.png`,`frame_0002.png` → `available` contains `omega`, `err`, `final:I`, `final:I_from_G`, `video_frames`, `table`; does not contain `recon_r`, `final:V`, `center:I`, `beta`. `test_frame_paths_sorted`: frames `frame_10.png`, `frame_2.png` → order 2, 10. `test_outputs_keys_unique`: `len({o.key for o in OUTPUTS}) == len(OUTPUTS)` and no key contains `exp11` or `iter_frames`.
- [ ] **Step 2: Run** — Expected `[FAIL]`.
- [ ] **Step 3: Implement `outputs.py`.**
- [ ] **Step 4: Run** — Expected all `[ok]`.
- [ ] **Step 5: Commit** `"compare_runs: output registry + lazy loaders"`

### Task 3: Bands — labels, warning, curves, table, images, maps

**Files:**
- Create: `interacting_maps/compare_runs/bands.py`
- Test: `tests/test_compare_runs.py` (add)

**Interfaces:**
- Consumes: `catalog` rows (pd.Series, Task 1), `outputs.read_tracking / read_maps / frame_paths / OUTPUTS` (Task 2).
- A **run** passed to every band is `Run = namedtuple('Run', 'slot color row ticked')` (`slot` 1-based int, `color` hex, `row` the catalog row, `ticked` set of output keys). Defined in `bands.py`.
- Produces:
  - `SLOT_COLORS` — 20 distinct colours (`matplotlib` `tab20`, as hex).
  - `labels(runs) -> dict[int, str]` — `'#<slot> ' + ' '.join(k=v)` over `choice_keys` whose values differ between the runs (`model` shown by `style.MODELS` name, `dt` in ms, `delta_X` as `δ_X`); all same → `'#<slot> <model name>'`.
  - `segment_warning(runs) -> str | None` — `None` if one segment; else `'Runs span N segments (boxes_rotation seg_A, …): comparing runs of different segments does not make sense.'`
  - `curve_figure(runs, key) -> (Figure, list[int])` — one figure, all runs ticked for `key` overlaid in slot colours, x = `frame_times(dt, n)[frame]`, ω/err in deg/s; `omega`: 3 stacked axes (`viz.draw_omega`-style labels), reference from `gt_w*` and IMU from `imu_w*` drawn once per distinct segment (black/grey, `ref_source` label); `recon_r`: |recon_r| solid, |recon_r_inloop| dashed, `ylim (0,1)`; `contrast`: `iwe_log.csv` vs frame time. Returns the slots left out (column missing) — the caller prints `not in: #…`. y shared across axes of the same quantity.
  - `table_html(runs, only_diff: bool) -> str` — one row per run (label first): config (`choice_keys`), provenance `git_commit_short` + `git_dirty`, every scalar summary key; differing columns get `style="background:#fff3c4"`; `only_diff=True` drops the constant columns; ω quantities already deg/s in summary.
  - `map_figure(runs, key) -> Figure` — `key` like `final:I`; one axis per run (columns in slot order, title = label), `viz.draw_map` with a shared `scale` from `viz.map_scale(kind, np.stack(...))` (same shape required; differing shapes → per-run scale and a note in the title); `I_from_G` = `solve_poisson_exact(G, 'fft')` drawn as kind `'I'`; `aps` = `eval_io.nearest_aps(load_aps_list('data/<dataset>'), t_mid)` where `t_mid` = npz `t_mid`, else for `final` the last `tracking.csv` `time` (title adds `t_mid from tracking.csv`), else for `center` an empty axis titled `no t_mid saved`.
  - `image_cells(runs, key, frame=None) -> list[tuple[int, str | None]]` — per run `(slot, path)`; for frames bands the path of frame `frame` or `None` (→ cell text `no frame k`).
  - `state_figure(runs, frame) -> Figure` — `states/frame_k.npz` → V, I, G, F rows × runs columns, shared scales per kind.

- [ ] **Step 1: Write the failing tests** (Agg backend, synthetic runs from Task 1/2 helpers):
  - `test_labels_differ`: two thesis runs dt 0.02/0.01 → labels `'#1 dt=20ms'`, `'#2 dt=10ms'`.
  - `test_segment_warning`: same segment → `None`; seg_A + seg_B → string containing `'2 segments'` and `'does not make sense'`.
  - `test_curve_missing_column`: run 1 has `beta`, run 2 not → `curve_figure(runs,'beta')[1] == [2]`; axes have 1 line.
  - `test_curve_deg_s`: `est_wz = 1.0` rad/s → the plotted y ≈ 57.2958.
  - `test_curve_x_since_start`: dt 0.01, frames 0..3 → x = `[0.005,0.015,0.025,0.035]`.
  - `test_shared_sharey`: `omega` figure's 3 axes each overlay 2 runs; two `err` runs share one axis (same ylim by construction).
  - `test_table_highlight`: dt differs → `'background:#fff3c4'` appears in the dt cell; `only_diff=True` drops `model` when all models equal.
  - `test_map_shared_scale`: two `final:I` maps (one ×2) drawn → both images use one scale (check `viz.map_scale` called once: compare imshow data range of the smaller map < 1).
  - `test_aps_center_no_tmid`: legacy `maps_center.npz` without `t_mid` → axis title `'no t_mid saved'`.
  - `test_frame_beyond`: run with 2 frames, `frame=5` → `image_cells(...)` gives `(slot, None)`.
- [ ] **Step 2: Run** — Expected `[FAIL]`.
- [ ] **Step 3: Implement `bands.py`.** Reuse `viz.draw_map`, `viz.map_scale`, `style.frame_times`, `style.to_deg_s`, `style.omega_label`, `style.ref_label`, `style.REF_STYLE`, `style.IMU_STYLE`, `style.LABEL_*`.
- [ ] **Step 4: Run** — Expected all `[ok]`.
- [ ] **Step 5: Commit** `"compare_runs: bands (labels, segment warning, curves, table, maps, frames)"`

### Task 4: The widget app and the notebook

**Files:**
- Create: `interacting_maps/compare_runs/app.py`
- Modify: `interacting_maps/compare_runs/__init__.py` (`from .app import CompareApp`)
- Create: `compare_runs.ipynb` (repo root, 2 cells: markdown how-to; `from interacting_maps.compare_runs import CompareApp; app = CompareApp(); app.show()`)
- Test: `tests/test_compare_runs.py` (add, headless — ipywidgets objects work without a kernel)

**Interfaces:**
- Consumes: `catalog.scan/options/matches/choice_keys`, `outputs.OUTPUTS/available`, `bands.*`.
- Produces:
  - `RunSlot(app, slot: int)` — `.box` (VBox: coloured header, one `Dropdown` per `choice_keys` (+`'any'`), `run` Dropdown of `matches` folders labelled `"<basename>  err=<x.x>°/s"`, one `Checkbox` per `OUTPUTS` grouped by band, disabled if not in `available(folder)`), `.folder -> str | None` (the single match or the `run` choice), `.ticked -> set[str]`.
  - `CompareApp(roots=('results', 'experiments'), max_runs=20)` — widgets: roots `Text` + `rescan` Button, `＋ add run` / `− remove run` Buttons (add disabled at 20, remove at 1; a new slot copies the previous slot's choices and ticks), `only differing columns` Checkbox, `frame` IntSlider (max = longest ticked sequence − 1), `compare` Button, `out` Output. `.slots: list[RunSlot]`, `.runs() -> list[bands.Run]` (slots with a folder), `.render()` draws into `out`: segment warning → curves (one figure per ticked curve key, then `not in: #…`) → table → figures (HBox of `wd.Image` per run, empty cell `—`) → maps (one figure per ticked map key) → frames (HBox per sequence at the slider frame; `states` via `state_figure`; gif/mp4 via `wd.Image`/`wd.Video`). The slider redraws only the frames band (own Output).
  - `.show()` → `display(self.box)`.
- Dropdown cascade: on any change, refill every dropdown from `options(cat, chosen)` keeping a still-valid value (guard flag against re-entrant observe, as in the walkthrough's `refresh`), then refill `run` and the checkbox enabled-states.

- [ ] **Step 1: Write the failing tests** — `test_app_slots`: `CompareApp(roots=[tmp])`; click add 25× → `len(slots)==20`, add disabled; click remove 25× → `len(slots)==1`, remove disabled. `test_app_cascade`: set slot 1 `model='thesis'` → its `dt` options are `['any',0.01,0.02]`; choose `dt=0.01` → `slot.folder` is that run's folder. `test_app_render_smoke`: two slots, tick `omega`, `table`, `tracking_plot.png` (write a 2×2 png), call `render()` → no exception, `out.outputs` non-empty.
- [ ] **Step 2: Run** — Expected `[FAIL]`.
- [ ] **Step 3: Implement `app.py`, `__init__.py`, `compare_runs.ipynb`.**
- [ ] **Step 4: Run** — Expected all `[ok]`.
- [ ] **Step 5: Local smoke on real runs** — `PYTHONPATH=. python -c "from interacting_maps.compare_runs import catalog; c=catalog.scan(['results','experiments']); print(len(c), c.segment_id.nunique())"` → about 360 rows, several segments. Then open `compare_runs.ipynb`, pick 3 runs of `boxes_rotation seg_A` differing in dt/model, tick ω, table, tracking_plot, final:I, video_frames; check visually; pick a seg_B run → warning appears.
- [ ] **Step 6: Add** a line for the tool to `CLAUDE.md` § Environment & commands (`compare_runs.ipynb` — compare up to 20 saved runs side by side, viewer only) and the test command to § Tests.
- [ ] **Step 7: Commit** `"compare_runs: ipywidgets app + notebook"`
