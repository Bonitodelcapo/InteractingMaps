# Inspection Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Save the model's state at every frame (opt-in), replay one frame iteration by iteration, and inspect 1–5 runs in a notebook viewer whose curves link back to the maps and the raw data, with GIF/MP4 export.

**Architecture:** Three layers. (1) State: `get_state`/`set_state` on both networks, a `FrameDriver` that owns the per-frame inputs and the loop state (shared by the tracking run and the replay), and a states I/O module. (2) Replay: re-runs frame k from checkpoint k−1 and stores every iteration. (3) Viewer: a lazy run loader, one figure function `draw_inspector`, an export loop over it, and a thin ipywidgets wrapper.

**Tech Stack:** Python 3.12, NumPy, matplotlib, imageio (exists), ipywidgets + ipympl (new, viewer only).

**Spec:** `docs/superpowers/specs/2026-11-25-inspection-viewer-design.md`

## Global Constraints

- Work on a local branch `inspection-viewer` (create from `main` before Task 1). Commits use the local git identity; no `--author`, no attribution lines; never push.
- Nothing is ever written under `results/` — no states, no replays, no test output. Inspect runs go to `experiments/inspect/`, test runs to `experiments/<test_name>/`.
- Saving is opt-in: `RunConfig.save_states = False` by default and **not** in `RunConfig.to_dict()` (so `find_run` and `run.json` are unchanged).
- Saving states must not change any computation: `PYTHONPATH=. python tests/test_golden_tracking.py` passes unchanged after every task.
- Replay exactness: replaying frame k from checkpoint k−1 reproduces checkpoint k with `np.array_equal` (states stored float64).
- Map scale in the viewer: one fixed scale per quantity for the whole sequence, per model (95th percentile of the per-frame `viz.map_scale` values; for `I` the 5th pct of lows and 95th pct of highs); `share_scale=True` pools all compared runs.
- Iteration slider = absolute iteration; a run with fewer iterations holds its last; a run with no replay at that frame shows its checkpoint and is titled "no replay".
- Compared runs must share dataset, segment_id, t_start, frame_duration, n_frames.
- Curves x-axis: seconds since segment start (`viz`/`style` conventions, `frame_times`).
- Leave `_save_3col_frame`, `--exp 5` and `cmd_converge` untouched (open decisions, see spec).
- Tests are standalone scripts in `tests/` printing `[ok]`/`[FAIL]`, run as `PYTHONPATH=. python tests/<file>.py`, exit non-zero on failure, skip (exit 0) when `data/poster_rotation` is absent.
- Reuse `viz.draw_map`, `viz.map_scale`, `viz.net_maps`, `viz.draw_series`, `eval_io` loaders, `data_loader` event loaders; core code lives in `interacting_maps/`, not in scripts.

## Review Focus

1. **Comparing runs from different segments or dt** → `check_comparable` raises `ValueError` naming the differing field (test in Task 6).
2. **Dataset not present locally** (only the run folder copied back) → maps and curves still draw; raw panels show "n/a" (test in Task 7).
3. **Run made without `save_states`** → `load_run` raises `FileNotFoundError` saying to rerun with `save_states` / `--what inspect` (test in Task 6).
4. **A run that blew up** (NaN/inf in maps or curves) → scales ignore non-finite values, the figure still draws (test in Task 7).
5. **Replay of frame 0** → starts from the saved initial state `init.npz`, not from a fresh network (test in Task 4).

## Correction to the spec, verified in the code

`Quantity.gradient_accumulator` is reset at the start of every iteration (`InteractingMapsThesis.step`, Phase 1), and `Cost_IMU.R_imu` / `Cost_CMax.set_frame` are set per frame. The thesis network's recurrent state is therefore `I, G, F, R` only — same as Cook. CMax V1's only loop state is `omega_cmax_prev`. The exact-replay test (Task 3) is what proves this.

## File structure

| File | Responsibility |
|---|---|
| `interacting_maps/network.py`, `network_dissertation.py` (modify) | `get_state` / `set_state` |
| `interacting_maps/frame_driver.py` (create) | per-frame inputs (gyro, CMax anchor, event window) + loop state; `step` |
| `interacting_maps/states_io.py` (create) | where states/replays live on disk; save/load |
| `run_config.py` (modify) | `save_states` attribute; `rc_from_run(run_dir)` |
| `evaluation.py` (modify) | `experiment_tracking` uses `FrameDriver`, writes states when `rc.save_states` |
| `interacting_maps/replay.py` (create) + `replay_frame.py` (create, CLI) | iteration replay |
| `report/run_experiments.py` (modify) | `--what inspect` |
| `interacting_maps/inspect_io.py` (create) | `RunView`, `load_run`, `check_comparable`, curves |
| `interacting_maps/inspect_fig.py` (create) | scales, `draw_inspector`, `export_animation` |
| `inspect_viewer.py` + `inspect_viewer.ipynb` (create) | widgets wrapper + walkthrough |

---

### Task 1: State API on both networks

**Files:**
- Modify: `interacting_maps/network.py` (class `InteractingMaps`)
- Modify: `interacting_maps/network_dissertation.py` (class `InteractingMapsThesis`)
- Test: `tests/test_network_state.py`

**Interfaces:**
- Produces: `net.get_state() -> dict[str, np.ndarray]` with keys `'I','G','F','R'`, full internal shapes (Cook `I` is `(H+1, W+1)`), **copies**, float64. `net.set_state(state: dict) -> None` copies the arrays back in (thesis: into `q_*.value`). Missing key → `KeyError`.

- [ ] **Step 1: Write the failing test** — for `cook` and `thesis_imu`, build a network with `run_config.make_network` on a synthetic 20×30 grid (fake `rc` via `RunConfig(dataset='poster_rotation', model=m, segment='seg_C', n_frames=2)`; skip if data absent), run `step(V, n_iters=3)` on random `V` (seed 0):
  - `test_roundtrip`: `s = net.get_state(); net2.set_state(s); all(np.array_equal(net2.get_state()[k], s[k]))`.
  - `test_copy`: mutating `net.get_state()['I']` does not change `net.I`.
  - `test_step_equal`: net A steps frame V1 then V2; net B `set_state(A_after_V1)` then steps V2 (same kwargs) → `np.array_equal` on all four keys.
- [ ] **Step 2: Run** `PYTHONPATH=. python tests/test_network_state.py` → `[FAIL]` (`AttributeError: get_state`).
- [ ] **Step 3: Implement** `get_state` / `set_state` on both classes.
- [ ] **Step 4: Run** the test → all `[ok]`; run `tests/test_golden_tracking.py` → `[ok]`.
- [ ] **Step 5: Commit** `git commit -m "Networks: get_state/set_state for I, G, F, R"`

### Task 2: FrameDriver — one place that builds a frame's inputs

**Files:**
- Create: `interacting_maps/frame_driver.py`
- Modify: `evaluation.py` (`experiment_tracking` frame loop, lines ~118–197)
- Test: `tests/test_golden_tracking.py` (unchanged, is the guard)

**Interfaces:**
- Produces: `class FrameDriver(rc, net, H, W, calib)`; loads gyro, and for `rc.use_cmax` the undistorted events of the segment, and builds `CMaxAngularVelocity` for V1 — moved verbatim from `experiment_tracking`.
  - `step(k: int, V: np.ndarray, n_iters: int) -> None` — the four-way branch now in the loop (V2 / V1 anchor / thesis_imu / plain).
  - `frame_kwargs(k: int) -> dict` — the kwargs `step` passes to `net.step` (for V1 it runs the CMax solve and advances `omega_cmax_prev`); used by the replay.
  - `loop_state() -> dict` → `{'omega_cmax_prev': (3,)}` (empty dict when not V1); `set_loop_state(d) -> None`.
  - Attributes read by the IWE saver: `last_window`, `last_anchor`, `cmax_est`.
- Consumes: `net.step` signatures (unchanged).

- [ ] **Step 1: Run** `tests/test_golden_tracking.py` → `[ok]` (baseline before the move).
- [ ] **Step 2: Create `FrameDriver`** by moving the CMax/gyro setup and the per-frame branch out of `experiment_tracking`; `step` = `net.step(V, n_iters=n_iters, **self.frame_kwargs(k))`, with the V2 branch passing `events=` and plain models passing nothing. The IWE block in `experiment_tracking` reads `driver.last_window` / `driver.last_anchor` / `driver.cmax_est`.
- [ ] **Step 3: Run** `tests/test_golden_tracking.py` → `[ok]`, identical numbers; `tests/test_cmax_v1_vs_v2.py` → `[ok]` (covers the two CMax branches the golden test does not).
- [ ] **Step 4: Commit** `git commit -m "experiment_tracking: per-frame inputs in FrameDriver"`

### Task 3: Save states during a run (opt-in) + exact-replay test

**Files:**
- Create: `interacting_maps/states_io.py`
- Modify: `run_config.py` (`RunConfig.__init__`: `self.save_states = False`; add `rc_from_run`)
- Modify: `evaluation.py` (`experiment_tracking`)
- Test: `tests/test_states_replay.py`

**Interfaces:**
- Produces (`states_io`):
  - `states_dir(run_dir) -> str` = `<run_dir>/states`; `state_path(run_dir, k) -> str` = `states/frame_{k:04d}.npz`, `k = -1` → `states/init.npz`.
  - `save_state(run_dir, k, state: dict, loop: dict) -> None` (`np.savez_compressed`, float64, loop keys prefixed `loop_`).
  - `load_state(run_dir, k) -> tuple[dict, dict]` (state, loop); missing → `FileNotFoundError` whose message names `save_states` and `--what inspect`.
  - `has_states(run_dir) -> bool`.
  - `assert_not_results(path) -> None` — raises `ValueError` if the normalised path has a `results` component at the repo root.
- Produces (`run_config`): `rc_from_run(run_dir) -> RunConfig` rebuilt from `run.json` config (`dataset, model, segment_id, t_start, frame_duration, n_frames, n_iters, distortion_mode, poisson, params` → `deltas=`), `out_root` = the run's root, and `rc.initial_R` set from the stored value.
- In `experiment_tracking`: if `rc.save_states`: `assert_not_results(rc.output_dir)`, save `init` (k = −1) right after the network is built, and frame k right after `driver.step`.

- [ ] **Step 1: Write the failing test** `tests/test_states_replay.py`: for each of `cook, thesis, thesis_imu, thesis_cmax, thesis_cmax_v2` run `experiment_tracking(rc, save_frames=False)` with `poster_rotation seg_C`, `n_frames=3`, `n_iters=10`, `out_root='experiments/states_test'`, `rc.save_states=True`, `IM_MLFLOW=0`. Then:
  - `test_files`: `init.npz` and `frame_0000..0002.npz` exist.
  - `test_exact_replay`: for k in 1, 2: `rc2 = rc_from_run(dir)`; fresh net + driver; `net.set_state(load k−1)`, `driver.set_loop_state(loop k−1)`; rebuild `V_k` with `EventFrameSequence` (same args as the run); `driver.step(k, V_k, rc2.n_iters)` → `np.array_equal` with saved frame k for `I, G, F, R`.
  - `test_off_by_default`: same run with `save_states` left default → no `states/` folder; `'save_states' not in rc.to_dict()`.
  - `test_refuses_results`: `assert_not_results('results/x')` raises `ValueError`.
  - Clean `experiments/states_test` at the end.
- [ ] **Step 2: Run** → `[FAIL]` (module missing).
- [ ] **Step 3: Implement** `states_io`, `rc_from_run`, the two save calls.
- [ ] **Step 4: Run** the test → all `[ok]`; `tests/test_golden_tracking.py` → `[ok]`.
- [ ] **Step 5: Commit** `git commit -m "Opt-in per-frame states (save_states) with exact-replay test"`

### Task 4: Iteration replay

**Files:**
- Create: `interacting_maps/replay.py`, `replay_frame.py` (root CLI)
- Test: `tests/test_states_replay.py` (extend)

**Interfaces:**
- Produces: `replay_frame(run_dir: str, k: int) -> str` — loads state k−1 (`init` for k = 0), rebuilds `V_k` and `FrameDriver` from `rc_from_run`, calls `net.step(V_k, n_iters=1, **kw)` `n_iters` times with `kw = driver.frame_kwargs(k)` computed **once**, stacks `I, G, F` (cropped with `net_maps`) and `R` per iteration, writes `replay_path(run_dir, k)` = `<run_dir>/replay/frame_{k:04d}.npz` with arrays `I (n_it,H,W)`, `G (n_it,H,W,2)`, `F (n_it,H,W,2)`, `R (n_it,3)`. Calls `assert_not_results` first. Returns the path.
- `replay_path` lives in `states_io`.
- CLI: `python replay_frame.py --run <dir> --frame K [--frame K2 ...]`.

- [ ] **Step 1: Extend the test**: `test_replay_last_equals_checkpoint` (k = 2: replay's last `I/G/F/R` equal the cropped checkpoint 2, `np.array_equal`); `test_replay_frame0_uses_init` (k = 0: replay's last equals checkpoint 0); `test_replay_refuses_results` (a run_dir under `results/` → `ValueError`, nothing written).
- [ ] **Step 2: Run** → `[FAIL]`.
- [ ] **Step 3: Implement** `replay_frame` and the CLI.
- [ ] **Step 4: Run** → `[ok]`.
- [ ] **Step 5: Commit** `git commit -m "Replay one frame iteration by iteration from its checkpoint"`

### Task 5: `run_experiments --what inspect`

**Files:**
- Modify: `report/run_experiments.py` (new `cmd_inspect`, entry in `COMMANDS`, docstring line)

**Interfaces:**
- `cmd_inspect(E, args)`: for the `SEGMENTS` selected by `--sequences` and the models of `--models` (default all `MODELS`), `rc = run_cfg(ds, sid, t0, model, args.dt, nf, out_root='experiments/inspect')`, `rc.save_states = True`, `_run(E, rc, not args.no_frames, args.frame_stride)`. Then call `replay_frame` for each frame in the new arg `--replay-frames` (comma-separated, e.g. `6,14`; default empty = no replay). Validates sequences/models before running, like `cmd_main`.

- [ ] **Step 1: Run** `python report/run_experiments.py --what inspect --sequences poster --models cook --duration 0.06 --no-frames --replay-frames 1` → `experiments/inspect/poster_rotation/cook/<run>/states/frame_0000..0002.npz` and `replay/frame_0001.npz` exist; nothing new under `results/` (`git status results/` clean).
- [ ] **Step 2: Commit** `git commit -m "run_experiments: --what inspect (states + replays into experiments/inspect)"`

### Task 6: Run loader

**Files:**
- Create: `interacting_maps/inspect_io.py`
- Test: `tests/test_inspect_io.py`

**Interfaces:**
- `load_run(run_dir) -> RunView` — reads `run.json` and `tracking.csv` eagerly, everything else lazily. Raises `FileNotFoundError` (message from `states_io`) when `has_states` is false.
- `class RunView`: `.dir`, `.config` (dict), `.label` (model name), `.n_frames`, `.n_iters`, `.H, .W`, `.times` (s since segment start, `frame_times`), `.rows: dict[str, np.ndarray]` (numeric csv columns only).
  - `.state(k) -> dict` cropped `I,G,F` + `R` (LRU cache).
  - `.iterations(k) -> dict | None` (replay arrays, or None).
  - `.raw(k) -> dict` with `V` (rebuilt event frame, `None` if dataset absent), `aps` (`(t, image)` via `eval_io.nearest_aps`, or `None`).
  - `.curves() -> dict[str, np.ndarray]` = `.rows` + `'n_events'` (events per frame, from `load_events_fast` + histogram; NaN array if dataset absent).
  - `.omega_fullrate() -> dict` `{'t_imu','imu','t_gt','gt'}` (|ω| in deg/s, time since segment start), `{}` if absent.
- `check_comparable(runs: list[RunView]) -> None` — 1 ≤ len ≤ 5, same `dataset, segment_id, t_start, frame_duration, n_frames`; `ValueError` naming the first differing field.

- [ ] **Step 1: Write the failing test** using a fake run folder built in `experiments/inspect_io_test/` (run.json with poster seg_C config, 3-row tracking.csv, states from random arrays saved with `save_state`):
  - `test_rows`: `rv.rows['err_deg_s'].shape == (3,)`, `ref_source` not in rows.
  - `test_state_crop`: `rv.state(1)['I'].shape == (H, W)`.
  - `test_no_states`: removing `states/` → `load_run` raises `FileNotFoundError` with `'save_states'` in the message.
  - `test_not_comparable`: two views, second with `t_start` changed → `ValueError` containing `'t_start'`; six views → `ValueError`.
  - `test_missing_dataset`: config dataset `'nope'` → `rv.raw(0)['V'] is None`, `np.isnan(rv.curves()['n_events']).all()`.
- [ ] **Step 2: Run** `PYTHONPATH=. python tests/test_inspect_io.py` → `[FAIL]`.
- [ ] **Step 3: Implement** `inspect_io`.
- [ ] **Step 4: Run** → `[ok]`.
- [ ] **Step 5: Commit** `git commit -m "inspect_io: lazy run loader for the viewer"`

### Task 7: The inspector figure and export

**Files:**
- Create: `interacting_maps/inspect_fig.py`
- Test: `tests/test_inspect_fig.py`

**Interfaces:**
- `fixed_scales(runs: list[RunView], kinds, share=False) -> list[dict]` — one dict `{kind: scale}` per run (identical dicts when `share`); non-finite map values and non-finite per-frame scales are dropped before the percentiles. `V` scales come from `raw(k)['V']` when available.
- `draw_inspector(runs, k, it=None, panels=('V','I','G','F','aps'), curves=('err_deg_s','beta'), share_scale=False, scales=None) -> matplotlib.figure.Figure`:
  - one row per run with the map panels (`I/G/F` from `state(k)` or, when `it` is given and a replay exists, `iterations(k)[...][min(it, n_it)-1]`; title suffix `"no replay"` otherwise); `V` and `aps` come from the first run only and are drawn once in an extra raw row (`"n/a"` axes when missing);
  - one axis per curve, all runs overlaid with their model colour (`viz.draw_series`), x = `run.times`, a vertical line at `times[k]`, same y-limits across runs of one curve;
  - suptitle: dataset, segment, frame k, t, iteration.
  - the special curve name `'omega_fullrate'` draws IMU and Vicon |ω| at full rate from `omega_fullrate()` (first run only, they are the same data for all compared runs) with the same frame marker;
  - Each curve axis gets attribute `ax._inspect_curve = name` (the widget uses it).
- `export_animation(runs, path, frames=None, it_frame=None, fps=5, **draw_kw) -> str` — if `it_frame` is given, loops `it` over that frame's iterations, else loops `k` over `frames` (default all); Agg backend; `.gif` via `imageio.mimsave`, `.mp4` via `imageio` ffmpeg writer.
- `frame_from_click(x: float, times: np.ndarray) -> int` — nearest frame index.

- [ ] **Step 1: Write the failing test** (reuse the fake run folder helper from Task 6, two runs, models `cook` and `thesis_imu`):
  - `test_layout`: `draw_inspector(runs, 1, panels=('I','G'), curves=('err_deg_s',))` → 2 map rows × 2 + raw row + 1 curve axis; the curve axis has a vertical line at `times[1]`.
  - `test_scale_per_model`: `fixed_scales(runs, ['I'])` differ between runs whose I differ by ×4; with `share=True` they are equal.
  - `test_nan_run`: one state with NaNs in G and a NaN in `err_deg_s` → figure draws, scales finite.
  - `test_missing_dataset_draws`: dataset `'nope'` → figure draws, raw axes titled `"n/a"`.
  - `test_export_gif`: `export_animation(runs, 'experiments/inspect_fig_test/a.gif', frames=[0,1,2])` → GIF with 3 frames (`imageio.mimread`).
  - `test_click`: `frame_from_click(0.031, np.array([0.01,0.03,0.05])) == 1`.
  - `test_fullrate_missing`: `curves=('omega_fullrate',)` with dataset `'nope'` → axis drawn, titled `"n/a"`.
- [ ] **Step 2: Run** → `[FAIL]`.
- [ ] **Step 3: Implement** `inspect_fig`.
- [ ] **Step 4: Run** → `[ok]`; also `tests/test_viz.py` → `[ok]`.
- [ ] **Step 5: Commit** `git commit -m "inspect_fig: draw_inspector, fixed per-model scales, GIF/MP4 export"`

### Task 8: Notebook viewer

**Files:**
- Create: `inspect_viewer.py`, `inspect_viewer.ipynb`
- Modify: `requirements.txt` (add `ipywidgets`, `ipympl`)

**Interfaces:**
- `class Inspector(run_dirs: list[str])` in `inspect_viewer.py`: loads with `load_run`, `check_comparable`; widgets: frame `IntSlider` (0…n_frames−1), iteration `IntSlider` (0 = checkpoint, 1…max n_iters; disabled when no compared run has a replay at the current frame), `SelectMultiple` curves (from `runs[0].curves()` keys), `SelectMultiple` panels, `Checkbox` share scale, `Button` "export GIF" (frames) and "export iterations GIF". Redraw = `draw_inspector` into one ipympl figure; `button_press_event` on an axis with `_inspect_curve` sets the frame slider via `frame_from_click`.
- Notebook: `%matplotlib widget`, one cell per use: pick run folders under `experiments/inspect/`, `Inspector([...]).show()`, export cell. Mirrors `inspect_viewer.py` (dual .py + .ipynb).

- [ ] **Step 1: Install** `pip install ipywidgets ipympl`; add both to `requirements.txt`.
- [ ] **Step 2: Implement** `Inspector` and the notebook.
- [ ] **Step 3: Verify by hand** on the Task 5 run plus a `thesis_imu` inspect run of poster seg_C (`--duration 0.6`, `--replay-frames 14`): slider moves all panels and the curve marker; clicking the error curve near t = 0.29 s jumps to frame 14; the iteration slider at frame 14 shows the replay; export writes a GIF. Run the notebook top to bottom with `jupyter nbconvert --to notebook --execute inspect_viewer.ipynb --output experiments/inspect/_nbcheck.ipynb` → exit 0.
- [ ] **Step 4: Commit** `git commit -m "Inspection viewer notebook (ipywidgets)"`

### Task 9: Docs

**Files:**
- Modify: `CLAUDE.md` (commands: `--what inspect`, `replay_frame.py`, `inspect_viewer.ipynb`, the new tests), `architecture.md` §6.6 (states + replay, one paragraph), `docs/notes/todo.md` (tick the checkpoint item; leave the `fig_converge` item open).

- [ ] **Step 1: Edit** the three files.
- [ ] **Step 2: Run** every test listed in `CLAUDE.md` → all `[ok]`.
- [ ] **Step 3: Commit** `git commit -m "Docs: inspection viewer, per-frame states, replay"`
