# Unified plotting module — design

Date: 2026-10-11
Status: approved in brainstorming, pending spec review

## Problem

An inventory of every figure in the repo (evaluation.py, report/*, debug/diag
scripts, demo, notebooks, interacting_maps/iwe_io.py, find_segments) found that
the same quantity is drawn differently depending on the script:

- ω in rad/s, deg/s or rad/frame; time as absolute seconds, relative seconds or
  frame index.
- V in three colour schemes, I with three normalisations (the 1–99 percentile
  one re-implemented 7 times), G as |G| `hot`, |G| grey or direction hue, F with
  three different HSV encodings.
- `thesis_imu` under four display names; red meaning "thesis" in one paper
  figure and "CMax-anchor" in another; the ω-axis palette hard-coded 6 times.
- 7 different DPIs, fonts set only in `make_figures.py`, some figures never
  saved.
- Plotting bugs: omega_gt reference labelled "GT (IMU)" (`_plot_tracking`),
  y=x line added after `legend()` (`analyze_grid`), demo notebook stepping
  frame 0 twice.

## Goal

The same quantity always looks the same, everywhere. Every drawing convention
lives in exactly one place, so changing a colour, name, unit or line style is a
one-line edit.

## Scope

In scope: every figure in the repo (paper, evaluation, debug, diag, demo,
notebooks, IWE output, find_segments) and the model display names used by
`report/make_tables.py`.

Out of scope (separate follow-up): output file naming, CSV/JSON column names
and schemas, metric key names, the reference-ω choice in Exp 1/3/6.

## Decisions

| Topic | Decision |
|---|---|
| ω unit in plots | deg/s everywhere. Data files stay rad/s; conversion happens only when plotting. R (rad/frame) is never plotted, always converted to ω = R/dt. |
| Time axis | Seconds since segment start, at frame midpoints: `t = (k + 0.5)·dt`. Titles carry the absolute `t_start`. |
| Non-time axes | Exempt: iterations (exp1, convergence), parameter values (exp3, grids). `find_segments` plots a whole dataset before any segment exists and keeps absolute dataset seconds. |
| V | `bwr` (red +, blue −, exactly white at 0), symmetric ±p99 of \|V\|. |
| I | Grey, 1–99 percentile (`normalise_robust`). |
| APS | log(APS+1), then the same encoding as I (I is a log-intensity estimate and is scored against log APS). |
| G and F | One encoding for any 2-vector field: hue = direction, saturation = \|v\|/p99 clipped to [0,1], value = 1 (white base). Square colour key inset at the bottom-left of the image. |
| IWE | Grey, \|IWE\|/p99, white = high. |
| Scaling rule | A single image is scaled per image (p99). Panels meant to be compared share one `scale=` so magnitude differences (e.g. β-scale errors) stay visible. Videos pass a temporally steady scale. |
| Model names | Cook, Thesis, IMU-anchor, CMax-anchor, CMax-inloop. |
| Model vs axis colour | A figure colours lines by axis when it shows one model, by model when it compares models, never both. |
| Style | One rcParams style, two size presets (`paper`, `screen`). One `save()` helper. Every figure is saved; notebooks display and save. |

## Architecture

Two core modules. Matplotlib is the only plotting dependency.

### `interacting_maps/style.py` — conventions only, no drawing

- `MODELS`: `{key: (display name, colour)}`, Okabe–Ito colour-blind-safe palette.

  | key | name | colour |
  |---|---|---|
  | `cook` | Cook | `#E69F00` |
  | `thesis` | Thesis | `#D55E00` |
  | `thesis_imu` | IMU-anchor | `#009E73` |
  | `thesis_cmax` | CMax-anchor | `#0072B2` |
  | `thesis_cmax_v2` | CMax-inloop | `#CC79A7` |

- `OMEGA_AXIS_COLORS = ('#4e79a7', '#f28e2b', '#e15759')` for x/y/z.
- `REF_STYLE`: solid black, lw 1.2. `IMU_STYLE`: grey, dotted, label
  "IMU gyro (input)"; drawn only when the IMU is not the reference.
- `ref_label(ref_src)` → "reference (ω_gt)" / "reference (Vicon)" /
  "reference (IMU)" for `'omega_gt'` / `'groundtruth'` / `'imu'`.
- `to_deg_s(omega_rad_s)` and `frame_times(dt, n)` → `(arange(n) + 0.5)·dt`.
- Label constants (mathtext): `"time (s)"`, `r"$\omega_x$ (°/s)"` (per axis),
  `"error (°/s)"`, `"direction error (°)"`, `r"scale $\beta$"`, and default map
  titles ("events $V$", "intensity $I$", "gradient $G$", "flow $F$",
  "APS (log)", "IWE").
- `use_style('paper' | 'screen')` returns a context manager (`with use_style('screen'):`) applying rcParams only inside the block. Both presets: same font family,
  grid α 0.3, top and right spines hidden. `paper`: 8 pt, widths 3.4 in
  (single column) / 7.05 in (double column). `screen`: 10 pt, width 12 in.
- `save(fig, path, show=False)`: PNG at 200 dpi, `bbox_inches='tight'`; additionally a PDF when `path` is in `report/figures/`; then closes the figure, or calls `plt.show()` when `show=True` (notebooks).

### `interacting_maps/viz.py` — drawing

Maps have two layers; line plots have one.

**Renderers** (array → RGB `(H, W, 3)` float in [0, 1]). They use matplotlib
colour utilities (`matplotlib.colors.hsv_to_rgb`, `matplotlib.colormaps`) but
never create figures or axes and never save.

| function | encoding |
|---|---|
| `render_V(V, scale=None)` | `bwr`, symmetric ±scale (default p99 of \|V\|) |
| `render_I(I, scale=None)` | grey via `normalise_robust` (default 1–99 percentile); `scale=(lo, hi)` overrides |
| `render_vec(v, scale=None)` | hue = direction in image coordinates (`[...,0]` = x right, `[...,1]` = y down), saturation = \|v\|/scale clipped, value = 1 (default scale p99 of \|v\|) |
| `render_IWE(iwe, scale=None)` | grey, \|IWE\|/scale clipped (default p99) |
| `vec_key(n)` | square key: grid of vectors from −1 to +1 in x and y, rendered by `render_vec` with scale 1, so key and image always agree |

Degenerate input (all zero or constant) returns the neutral colour: white for
V/G/F, mid-grey for I, black for IWE (no events).

Percentile scales fall back to the maximum when the 99th percentile is ~0 (maps with under 1 % non-zero pixels).

**Map drawer**: `draw_map(ax, kind, arr, title=None, scale=None)` with `kind`
in `'V' | 'I' | 'aps' | 'G' | 'F' | 'IWE'`. `'aps'` applies log(APS+1) then
`render_I`. Calls `imshow(interpolation='nearest')`, hides ticks, uses the
default title from `style.py`, and adds the `vec_key` inset (bottom-left) for
`'G'`/`'F'`.

**Cropping**: `net_maps(net, H, W)` → `{'I', 'G', 'F'}` cropped to `[:H, :W]`.
Cook's I is (H+1)×(W+1) (staggered grid); the thesis network's I is H×W; G and
F are H×W×2 in both. This is the only place maps are cropped for display.

**Line-plot drawers** (x always seconds from `frame_times`; ω converted to
deg/s inside; axis labels from `style.py`):

- `draw_omega(ax, t, est, ref=None, imu=None, axis=0, model=None, ref_src=None)`
  — one ω component; estimate coloured by axis if `model` is None, else by
  model; reference and IMU lines with the shared styles and `ref_label`.
- `draw_error(ax, t, err, model=None)` — error in deg/s.
- `draw_series(ax, t, y, label, model=None)` — any other time series
  (residuals, IWE contrast, I_std).

**Removed**: `normalise` (min-max; flickers, outlier-driven), `flow_to_rgb`,
`grad_to_rgb`. Callers move to the functions above; no compatibility shims.
`normalise_robust` stays as the building block of `render_I`.

## Migration

Each step leaves the repo runnable. Plotting bugs listed in the inventory are
fixed in the step that touches that code.

1. **Core**: add `style.py`; extend `viz.py`; `iwe_io` uses `render_IWE` and
   `frame_times`, and no longer forces the Agg backend at import. `plot_contrast_curve(out_dir, t_start)` gains `t_start` for its seconds axis.
2. **`evaluation.py`**: `_plot_tracking` (fixes the omega_gt label),
   `_save_3col_frame` (video frame; keeps its steady display range by passing
   `scale=`), exp1, exp3, exp6.
3. **`report/*`**: `make_figures.py`, figures in `run_experiments.py`,
   `analyze_grid.py` (fixes legend order), display names in `make_tables.py`.
   Paper figures change visibly (deg/s, map style, model colours); this is
   intended.
4. **Other scripts**: `debug_visual.py`, `debug_pipeline.py`,
   `diag_omega_ladder.py`, `demo.py`, `eval_pipeline.py`, `find_segments.py`;
   matching notebooks regenerated (`demo.ipynb` also fixes the double first
   step).

## Testing

`tests/test_viz.py`, standalone script in the repo's `[ok]` / `[FAIL]` style,
checking conventions numerically:

- `render_vec` of a zero vector is white.
- A +x vector has the same hue as the right-edge centre of `vec_key`.
- `render_V` of zero is white; `render_I` of a constant is mid-grey.
- `net_maps` returns H×W maps for both networks.
- `frame_times(dt, n)` returns midpoints; `to_deg_s` converts correctly.
- `ref_label` returns the right label for all three reference sources.
- `save` writes a PNG, plus a PDF when the path is under `report/figures/`.

After each migration step, that step's scripts are run on `poster_rotation`
(output to `experiments/`, never `results/`) and the figures are reviewed by
eye before the next step.
