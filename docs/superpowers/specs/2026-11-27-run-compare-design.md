# Run comparison tool (Jupyter)

## Intent and why
The user wants to **see what a parameter does**: pick several finished tracking runs
(different models, segments, step sizes, dt, iterations, …) and look at their outputs side
by side. It is a viewer only: every output already exists on disk on the cluster; the tool
never runs the network or writes run outputs.

## Goals
1. **Run slots, 1–20.** A `＋ add run` / `− remove run` pair adds a slot or removes the last one.
   Each slot has a colour (its colour in every plot and as its column header) and a row of
   **cascading dropdowns**, the same behaviour as the walkthrough's saved-run lookup: choosing a
   value narrows the others to what saved runs actually have. Keys:
   `model → dataset → segment_id → t_start → dt → n_frames → n_iters → poisson →
   distortion_mode → delta_*` (+ `root`, so `results/` and `experiments/<name>/` runs can be
   told apart). When the choices leave more than one folder, a last `run` dropdown lists the
   matching folders (best ω error first); when exactly one is left it is picked.
2. **Segments are free.** Dataset and segment come from what is on disk, not the report's
   fixed five (no `poster`/`dynamic` labels as the only choices).
3. **Per-run output checkboxes.** Each slot ticks which of its outputs enter the comparison.
   Only outputs that run actually has are offered; the others are shown greyed (so the user
   sees what is missing). A slot gets the previous slot's ticks when added.
4. **Comparison grid by output** (approach 1 below): one band per output type, runs as
   columns in slot order; a run that did not tick an output leaves its cell empty.
5. **Rescan** button: rereads the run folders (new runs arrived on disk).

## Outputs (all of them; the user selects)
Everything below is read from the run folder. Each line is one checkbox; the band it lands in
is in brackets.

**Curves** — overlaid, one plot per quantity, all runs, deg/s, shared y between same-quantity
axes (data-driven), from `tracking.csv` (column missing → that run absent from that plot):
- ω estimate per axis x/y/z (3 stacked axes), with the reference (Vicon / ω_gt) and the IMU
  gyro drawn once per distinct segment (dashed/dotted, as in `viz.draw_omega`) [curves]
- |ω| error `err_deg_s` [curves]
- direction error `dir_err_deg` [curves]
- scale β `beta` [curves]
- reconstruction |r| vs APS: `recon_r` (I from G, exact) and `recon_r_inloop` [curves]
- std of in-loop I `I_std` [curves]
- frame time `t_frame_s` [curves]
- CMax contrast over frames, `iwe/iwe_log.csv` [curves]

**Table** — one row per run [table]:
- config: model, dataset, segment, t_start, dt (ms), n_frames, n_iters, poisson,
  distortion_mode, every delta_*; provenance: git commit + dirty flag
- summary: mean ω error, mean direction error, mean β, recon_r_mean / min / p10 (exact and
  in-loop), and any other scalar the summary holds
- columns whose value differs between the runs are highlighted; a toggle hides the columns
  that are the same in all runs

**Saved figures** — the run's own PNG, one cell per run [figures]:
- `tracking_plot.png`
- `recon_final.png` (final state: in-loop I, I from G, G, F, nearest APS)
- `recon_plot.png` (|r| and std I over the track)
- `iwe/contrast_curve.png`
- `exp1_convergence.png`, `exp3_parameters.png`, `exp6_basin.png`

**Maps** — drawn from the saved arrays with `viz.draw_map` (same colour scale across runs for
the same map) [maps]:
- from `maps_final.npz` (last frame) and `maps_center.npz` (middle frame), each map its own
  checkbox: V, I, G, F
- I from G (exact Poisson solve of the saved G — display only, as `recon_final.png` does)
- nearest APS frame to the map's `t_mid` (when the dataset is under `data/`). *Builder-check
  call:* 358/468 `maps_final.npz` are legacy (only I, G, F; no V, `t_mid`). For
  `maps_final` without `t_mid`, `t_mid` = `time` of the last `tracking.csv` row (it is the
  last frame by definition), and the cell says `t_mid from tracking.csv`. For a legacy
  `maps_center.npz` without `t_mid` (no code writes it any more; which frame it holds is not
  recorded) the APS cell stays empty with `no t_mid saved` — never a guessed frame. Missing
  V → greyed.
- APS loader: `interacting_maps/eval_io.py` (`load_aps_list`, `nearest_aps`).

**Frame sequences** — one shared frame slider for the whole band (frame k of every run) [frames]:
- `video_frames/` (per-frame panels)
- `iwe/frame_*.png` (CMax IWE) — and `iwe_sequence.gif`
- `states/frame_*.npz` (per-frame saved state → V, I, G, F maps at frame k)
- `comparison.mp4` (exp 5 video), played in the cell

## Non-goals / anti-goals
- No runs, no reruns, no network, no writing into run folders, `results/` or `report/`.
- Not the walkthrough: the walkthrough's lookup stays where it is, untouched.
- Not the Qt inspect app (playback of states with linked curves stays its job).
- Anti-goal: a plot that looks comparable but is not — different y-limits for the same
  quantity, rad/s anywhere, or a run silently missing from a plot without saying so.

## Constraints
- Jupyter + ipywidgets, run on the cluster where the outputs are. No Qt.
- deg/s for every ω/error (`style.to_deg_s`); report style and model colours from `style.py`.
- Reuse core modules: `experiment_tracking.read_run` (handles legacy `params.json` /
  `summary.json`), `style`, `viz.draw_omega` / `draw_map` / `render_*`,
  `network_dissertation.solve_poisson_exact`, the APS loader in `eval_io`. Import only
  from core modules, never from `walkthrough/`, `report/` or `evaluation.py`.
- Up to 20 runs × hundreds of frames: read lazily (only ticked outputs, only the current frame
  for sequences); the scan reads only `run.json`/`params.json`.
- Local machine: smoke checks only; it is used for real on the cluster.

## Approaches
1. **Grid by output, runs side by side (chosen).** Slots on top, then bands per output type,
   runs as columns, curves overlaid in one plot. The same output is always on the same row,
   so a parameter's effect is read along one row.
2. One card per run with its ticked outputs stacked. Simpler, but at 10–20 runs the cards are
   narrow and need horizontal scrolling, curves cannot be overlaid, and the same output sits at
   different heights. Rejected.
3. Qt window like `inspect_app`. Needs a display on the cluster or local copies. Rejected by
   the user.

## Decided parts of the how
- **Curves x-axis: time since each run's `t_start`** (s). Runs with different dt line up in
  real time. *User decision:* comparing runs of different segments does not make sense; the
  tool allows it but says so — when the chosen runs span more than one segment
  (dataset + segment_id), a warning line above the comparison names the segments, and each
  curve plot draws one reference per segment.
- **Dropped (user decision):** the six legacy exp11 folders (`exp11_iter_history.csv`,
  `exp11_iter_convergence.png`, `iter_frames/`, `iteration_evolution.gif`): they have no
  `run.json`/`params.json` and no current code writes them.
- Discovery: every folder under the scanned roots (default `results/` and `experiments/`, a
  text box to change them) that holds `run.json` or `params.json`.
- Run label in headers/legends: slot number + the config values that differ between the
  chosen runs (e.g. `#3 thesis_imu dt=10ms δ_FR=0.3`).
- Frame slider bounds: 0 … the longest sequence; a run without frame k shows an empty cell.

## Left to the builder
- File layout (a small `interacting_maps/compare_runs/` package or one module + a thin
  `compare_runs.ipynb` that builds and displays the tool); a CLI `.py` is not needed since it
  is an interactive viewer.
- Grid rendering details (HTML grid of image widgets vs matplotlib subplots per band), image
  sizing, caching of loaded PNGs/npz.
- A smoke test in `tests/` (standalone script, `[ok]`/`[FAIL]`) that builds the tool headless
  on a couple of local runs and checks scan, cascade and band assembly.
