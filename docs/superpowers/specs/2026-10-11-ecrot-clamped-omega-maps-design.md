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
  (`on_frame(k, net)` after frame k), `on_iter=None` and `iter_frame=None` (at that frame
  index use `trace_relaxation(..., on_iter=...)` instead of `net.step`). Default
  path byte-for-byte as today.
- `rung1c_clamped_R(ctx, on_frame=None, on_iter=None, iter_frame=None)`: same shape as rung 1b;
  warm-start from gyro at `t_start` (as the other rungs), per frame
  `clamp_R(net, w*dt)` then step with `config.ITERS_PER_FRAME`; scores ω (trivially
  ~0, kept for uniform columns) and `flow_epe(net.F, gt_flow(net, w*dt))`.
- `RESULTS_HEADER` gains `F_epe` (filled for `0`, `1b_*`, `1c_R`; blank for `1a`).
  `run_all_segments` adds the `1c_R` row.
- `--mode maps --segment <name> --frame K` (defaults: `city_short`, `K = n//2`):
  runs rows `1c_R`, `1b_free`, `0` on that segment, collects per frame
  `viz.net_maps` + V + F_gt, and per iteration at frame K the I, G, F maps.
  Writes `diag_maps_<segment>.npz` (`savez_compressed`, float16 — visual use, halves the size):
  `V (n,H,W)`, `F_gt (n,H,W,2)`, `I (3,n,H,W)`, `G (3,n,H,W,2)`, `F (3,n,H,W,2)`,
  `I_it (3,T,H,W)`, `G_it (3,T,H,W,2)`, `F_it (3,T,H,W,2)` (T = iterations),
  `rungs`, `dt`, `t_start`, `K`. `--limit` applies.

### `diag_omega_ladder.py` (plots, local, `--mode plots`)
Each map plot runs only if `diag_maps_<segment>.npz` exists.
- `_map_scales(d)`: fixed scales computed once — V: p99 |V| over all frames;
  I: p1–p99 of row 1c; G: p99 |G| of row 1c; F, F_gt, error: p99 |F_gt|. Same
  scales for every row and every GIF frame (so the free row may look washed
  out/saturated — that is the β-ambiguity being shown).
- `draw_maps_grid(axes, maps_t, scales, rungs)`: fills the 3×6 grid via `viz.draw_map`;
  row name as y-label of the first column. Shared by GIFs and the static figure.
- `gif_frames.gif`: one figure per frame, suptitle frame index + time (s) from
  `style.frame_times`; PNGs to `diag_maps_<seg>/frames/`, assembled with
  `make_gif(..., name='gif_frames.gif')`.
- `gif_iters.gif`: one figure per iteration at frame K, suptitle
  "frame K, iter i/T"; V and F_gt columns constant; PNGs to
  `diag_maps_<seg>/iters/`.
- `diag_P4_maps_<seg>.png`: static grid at frame K (end of relaxation).
- `diag_P5_F_epe_<seg>.png`: per-frame `flow_epe` vs time, one line per row
  (`viz.draw_series`), computed from the npz at plot time.

### `diag_omega_ladder.ipynb` (authored walkthrough)
New section "Clamp R to ω_gt: do I and F relax well?": what-to-look-for markdown,
small local run (the notebook slice, `LIMIT=10`, `K = n//2`) of the maps collection, static grid + EPE plot,
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

## Findings (2026-10-12)

**Striped I with the correct ω is an OFCE step instability, not a plotting bug**
(measured in float64 on city_short, last of 8 frames, 20 ms).

- `Cost_OFCE` updates G with an explicit gradient step. Along F the residual is
  multiplied per iteration by `1 − 2·δ_VFG·|F|²`, so the step is unstable once
  `|F| > 1/√δ_VFG ≈ 2.9 px/frame` (`δ_VFG = 0.12`).
- There G flips sign almost every iteration; only the per-pixel `max_grad=5`
  clip keeps it bounded. The Poisson step integrates the oscillating G into the
  stripes seen in I.

| row | pixels with \|F\| > 2.9 px/frame | G sign flips per iteration there |
|---|---|---|
| 0 (free) | 18 % | 0.23 (0.00 below the bound) |
| 1c (R clamped) | ~100 % | 0.81–0.99 |
| 1b (F clamped) | ~100 % | 0.80–0.98 |

The free row is barely striped only because its β ≈ 2.9 scale error shrinks F
about 3× into the stable range. Related: in the cluster's 75-frame run rung 1b
reported clamp drift 0.61, i.e. `F_gt` exceeds `CLIP_F = 10` px/frame there and
is clipped.

**Proposed remedies** (not applied — `THESIS_PARAMS` drive the reported numbers):

1. Shorter `dt` (F scales with dt); 5–10 ms keeps |F| under the bound here.
2. Smaller `δ_VFG` (the bound `1/√δ_VFG` grows).
3. Scale the G step by `1/|F|²` per pixel (Newton-like / implicit step), which
   is stable for any |F|.

Open check: count pixels with |F| > 2.9 px/frame on the report segments, to see
whether reported results are affected.

**Derivation of the bound.** Per pixel, along F (g = F̂·G): g_new = (1 − 2·δ_VFG·|F|² − δ_IG)·g + const (OFCE step + Cost_Spatial pull toward ∇I). Stable iff |λ| < 1 ⇒ |F| < √((2−δ_IG)/(2δ_VFG)) ≈ 2.8 px/frame; sign flips (λ < 0) from |F| > √((1−δ_IG)/(2δ_VFG)) ≈ 1.9 px/frame (δ_VFG=0.12, δ_IG=0.10). Ignoring δ_IG gives 1/√δ_VFG ≈ 2.9. Single-pixel, F-fixed, clip-free linearisation: indicative, not proven — the δ_VFG sweep (onset should scale as 1/√δ_VFG) is the test.

## Findings: G flipping measured (2026-10-14)

Cluster run of `python diag_omega_ladder.py --mode ofce` (all seven variants,
`city_short`, 72 frames at 20 ms = 1.44 s; 145 / 290 frames at 10 / 5 ms) and
`--mode ofce_summary`. Plan: `docs/superpowers/plans/2026-10-14-ofce-g-flip-investigation.md`.
x = |F| / b with b = √((2−δ_IG)/(2δ_VFG)). Per row, at the middle frame K:
λ̂ = median per-iteration ratio of successive changes of G along F over the
first 12 iterations; flip rate = sign changes of G along F over the last 20
iterations; occupancy = share of pixels per x bin. Figures: `diag_ofce/ofce_lambda.png`,
`ofce_flips.png`, `ofce_beta.png`. Criteria were fixed in the plan before the run.

**RQ1 (mechanism) — refuted below x ≈ 0.9, holds near and above the bound.**
With F fixed at the truth (row 1b), λ̂ is −1.00 above x = 1 in every clipped
variant (G pinned by the clip, flipping every iteration) and matches λ(x) in the
last bin below the bound. Below that it is 0.77–0.88 whatever x is.

| row 1b, x bin | 0–0.5 | 0.5–0.69 | 0.69–0.9 | 0.9–1.0 | > 1 |
|---|---|---|---|---|---|
| λ(x) over the bin | 0.90 … 0.43 | 0.43 … 0.00 | 0.00 … −0.64 | −0.64 … −1.00 | < −1 |
| E4_dt05 λ̂ | 0.77 | 0.84 | 0.88 | – | – |
| E4_dt10 λ̂ | – | – | −0.52 | −0.86 | −1.00 |
| E0 / E1 λ̂ | – | – | – | – | −1.00 |

Reading (hypothesis, not tested): λ̂ reports the slowest mode still moving. The
one-pixel OFCE mode decays within a few iterations where |λ| is small, leaving a
slower mode (G coupled to I through G = ∇I) at ≈ 0.8; the OFCE mode is only
visible where |λ| ≳ 0.8. E4_dt10's −0.52 at 0.69–0.9 sits next to unstable
pixels, E4_dt05's 0.88 next to stable ones. E2_noclip is also −1.00: removing
`max_grad` leaves G bounded by `CLIP_G`, which it hits on > 99 % of pixels, so
"below −1 without the clip" could not be tested.

**RQ2 (scaling) — the curves collapse in x; sustained flips start at x ≈ 1, not at x₀ = 0.69.**
Flip rate by x bin, all clipped variants:

| x bin | < 0.9 | 0.9–1.0 | 1.0–1.1 | > 1.1 |
|---|---|---|---|---|
| row 1b (F fixed) | ≤ 0.002 | 0.12 (E4_dt10) | 1.00 (E4_dt10) | 0.91–1.00 |
| row 0 (free) | ≤ 0.002 | 0.07–0.12 | 0.21–0.56 | – (no pixels) |

The onset in px moves with b (≈ 2 px for δ_VFG = 0.24, ≈ 4 px for 0.06), so the
refutation condition (same px onset) is not met. The pre-set "> 0.5 above x ≈ 0.8"
is not met either. For −1 < λ < 0 the flips decay and are gone before the last 20
iterations, so this measure only counts flips that are sustained, which the
theory puts at x ≥ 1.

**RQ3 (consequence) — supported, with F capped near 0.8·b rather than at b.**

| variant | b px | mean\|F\*\|/b | row-0 occupancy at x 0.69–1.0 | above x = 1 | mean x | β | max(1, mean\|F\*\|/b) | mean\|F\*\|/(b·mean x) | dir ° |
|---|---|---|---|---|---|---|---|---|---|
| E1_dVFG006 | 3.98 | 1.54 | 84 % | 0.6 % | 0.78 | 2.01 | 1.54 | 1.97 | 12.5 |
| E0_base | 2.81 | 2.18 | 98 % | 1.9 % | 0.82 | 2.83 | 2.18 | 2.66 | 13.5 |
| E1_dVFG024 | 1.99 | 3.08 | 96 % | 3.9 % | 0.83 | 3.87 | 3.08 | 3.70 | 17.9 |
| E4_dt10 | 2.81 | 1.09 | 99 % | 1.1 % | 0.81 | 1.46 | 1.09 | 1.34 | 15.7 |
| E4_dt05 | 2.81 | 0.54 | 73 % | 0.3 % | 0.75 | 0.75 | 1.00 | 0.72 | 13.8 |
| E3_normalize | (none) | 2.18 | – | – | > 2 | 0.72 | 1.00 | – | 14.5 |
| E2_noclip | 2.81 | 2.18 | 0 % | 99.6 % | > 2 | 13.2 | 2.18 | – | 89.2 |

(mean x uses bin centres, 0.795 for 0.69–0.9.) In every clipped variant the free
network's |F| sits just below the bound: 0.3–3.9 % of pixels above x = 1, mean
x 0.75–0.83, independent of the true flow (mean|F\*|/b from 0.54 to 3.08). The
pre-set "β within ±25 % of max(1, mean|F\*|/b)" fails narrowly (+26 to +34 %),
because the cap sits at ≈ 0.8·b, not b. Using the measured mean x instead,
β ≈ mean|F\*|/(b·mean x) to within 2–9 % in the four capped variants. E3_normalize
and E4_dt05 give β < 1.3 as required.

**Not explained by the bound (open):**
- Where nothing caps F (E4_dt05, E3_normalize), β ≈ 0.73: |F| and ω are
  overestimated by ≈ 1.37×. In E4_dt05 the free |F| still sits at mean x 0.75,
  above the true 0.54.
- The direction error is 12.5–17.9° in every converged variant, whatever b is.
- E2_noclip: G hits `CLIP_G` on > 99 % of pixels, F runs above 2·b, β 13 and
  dir 89°. That is close to the real-data failure (dir ≈ 85°); not tested further.

**What follows.** The rung-0 scale error β ≈ 3 at dt = 20 ms comes from the
explicit OFCE step on G: the free network cannot hold |F| above ≈ 0.8·b, so
ω is underestimated by mean|F\*|/(0.8·b). It is a numerical property of the
update, not of the data or the kinematics. The remedies (normalised step,
shorter dt, smaller δ_VFG) remove that cap, but in the two runs where the cap is
gone, ω is overestimated by ≈ 1.37× instead, and the ≈ 14° direction error stays.
Neither fixes ω on its own. The earlier reading "β shrinks F into the stable
range" has the cause reversed: the instability holds F down, and β follows.
