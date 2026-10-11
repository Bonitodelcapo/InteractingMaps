# ECRot clamped-ω maps: do I and F relax well when ω is right?

Date: 2026-10-11 · Branch: `modularity-notebooks` · Source: "Interacting maps 2" notes,
items *ECRot debug for I and F* and *GIF on ECRot* (both marked "!").

## Goal

ECRot (ESIM) provides exact ω per event (`omega_gt.txt`). Clamp the network's R to
that ω every frame, let I, G, F relax, and check whether I and F come out good.
Show it as two GIFs (over frames; over iterations at one frame), one static grid
figure and one flow-error plot, compared against the existing ladder rungs.

Out of scope: the CMax items (standalone ω vs IMU/GT, warped-event images) — they
get their own spec.

## Rows compared (one row each in every figure)

| row | rung | what is clamped | free |
|---|---|---|---|
| 1 | `1c_R` (new) | R = ω_gt·dt each frame | I, G, F |
| 2 | `1b_free` (existing) | F = C·R_gt each frame | I, G, R |
| 3 | `0` (existing) | nothing | I, G, F, R |

No ω anchor (`omega_imu=None`) in any row. Columns: V | I | G | F | F_gt | F − F_gt,
with `F_gt = C·R_gt` (`gt_flow`). I has no reference on ECRot city (no
`images.txt`), so it is judged visually; F is scored by endpoint error.

## Reused (core modules / existing ladder code)

- R freeze: `InteractingMapsThesis.cost_kin.update_r = False` (existing V2 switch);
  with no IMU anchor nothing sends gradient to `q_R`.
- `viz.net_maps`, `viz.draw_map`, `viz.render_vec` (error map = `F − F_gt` at the
  flow scale), `viz.draw_series`.
- `style.frame_times`, `style.use_style`, `style.save`, `style.MAP_TITLES`.
- `iwe_io.make_gif` (folder of `frame_*.png` → GIF).
- Ladder: `prep_segment_context`, `build_net`, `gt_flow`, `clamp_flow`,
  `trace_relaxation`, `score_frames`, `ref_omega`.

## Changes

### Core modules
- `metrics.py`: `flow_epe(F, F_ref)` → mean endpoint error (px/frame),
  `mean(‖F − F_ref‖₂)` over pixels.
- `iwe_io.make_gif(out_dir, fps=15, name='iwe_sequence.gif')`: optional output
  name; default keeps current callers unchanged.

### `diag_omega_ladder.py` (compute, cluster-ready)
- `clamp_R(net, R_star)`: sets `net.cost_kin.update_r = False` and
  `net.q_R.value = R_star.copy()`. Called every frame.
- `trace_relaxation(..., on_iter=None)`: if given, `on_iter(net)` after each
  single-iteration step. Unchanged otherwise.
- `rung0_baseline`, `rung1b_clamped_flow`: optional `on_frame=None`
  (`on_frame(k, net, V, F_gt)` after frame k) and `iter_frame=None` (at that frame
  index use `trace_relaxation(..., on_iter=...)` instead of `net.step`). Default
  path byte-for-byte as today.
- `rung1c_clamped_R(ctx, on_frame=None, iter_frame=None)`: same shape as rung 1b;
  warm-start from gyro at `t_start` (as the other rungs), per frame
  `clamp_R(net, w*dt)` then step with `config.ITERS_PER_FRAME`; scores ω (trivially
  ~0, kept for uniform columns) and `flow_epe(net.F, gt_flow(net, w*dt))`.
- `RESULTS_HEADER` gains `F_epe` (filled for `0`, `1b_*`, `1c_R`; blank for `1a`).
  `run_all_segments` adds the `1c_R` row.
- `--mode maps --segment <name> --frame K` (defaults: `city_short`, `K = n//2`):
  runs rows `1c_R`, `1b_free`, `0` on that segment, collects per frame
  `viz.net_maps` + V + F_gt, and per iteration at frame K the I, G, F maps.
  Writes `diag_maps_<segment>.npz` (`savez_compressed`, float32):
  `V (n,H,W)`, `F_gt (n,H,W,2)`, `I (3,n,H,W)`, `G (3,n,H,W,2)`, `F (3,n,H,W,2)`,
  `I_it (3,T,H,W)`, `G_it (3,T,H,W,2)`, `F_it (3,T,H,W,2)` (T = iterations),
  `rungs`, `dt`, `t_start`, `K`. `--limit` applies.

### `diag_omega_ladder.py` (plots, local, `--mode plots`)
Each map plot runs only if `diag_maps_<segment>.npz` exists.
- `_map_scales(d)`: fixed scales computed once — V: p99 |V| over all frames;
  I: p1–p99 of row 1c; G: p99 |G| of row 1c; F, F_gt, error: p99 |F_gt|. Same
  scales for every row and every GIF frame (so the free row may look washed
  out/saturated — that is the β-ambiguity being shown).
- `draw_maps_grid(axes, maps_t, scales)`: fills the 3×6 grid via `viz.draw_map`;
  row name as y-label of the first column. Shared by GIFs and the static figure.
- `gif_frames.gif`: one figure per frame, suptitle frame index + time (s) from
  `style.frame_times`; PNGs to `diag_maps_<seg>/frames/`, assembled with
  `make_gif(..., name='gif_frames.gif')`.
- `gif_iters.gif`: one figure per iteration at frame K, suptitle
  "frame K, iter i/T"; V and F_gt columns constant; PNGs to
  `diag_maps_<seg>/iters/`.
- `diag_P4_maps_frameK.png`: static grid at frame K (end of relaxation).
- `diag_P5_F_epe.png`: per-frame `flow_epe` vs time, one line per row
  (`viz.draw_series`), computed from the npz at plot time.

### `diag_omega_ladder.ipynb` (authored walkthrough)
New section "Clamp R to ω_gt: do I and F relax well?": what-to-look-for markdown,
small local run (`LIMIT=8`, `K=4`) of the maps collection, static grid + EPE plot,
both GIFs inline via `IPython.display.Image`.

## Testing (script style, `[ok]`/`[FAIL]`)
- `--mode setup` adds: R unchanged (exact) after `clamp_R` + `net.step(V, 75)`;
  `trace_relaxation(on_iter=...)` final maps equal `net.step`; `flow_epe(F, F) == 0`.
- Local smoke: `--mode maps --limit 6 --frame 3` then `--mode plots` produces both
  GIFs and both PNGs.
- `tests/test_viz.py`: `make_gif(..., name=...)` writes the named file.
- Full run (`city_short`, 75 frames) on the cluster.

## Data note
Only `ecrot_city` is local; its `omega_gt` ends at 1.5 s, so `city_short`
(t_start 0.05 s, 20 ms) is capped by `cap_to_coverage` to ~72 frames. `ecrot_street` segments work identically on the cluster.
