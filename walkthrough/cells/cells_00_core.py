CORE = ('core',)

CELLS = [
('md', CORE, r'''
# Experiments walkthrough

Every experiment behind the report, and the research diagnostics around it,
unrolled step by step: how each run is made, and how each figure, table and
quoted number comes out of it. One section per ID of
[experiments.md](experiments.md), in the same order.

**Two modes** (the toggle in the panel below):

- **Redraw** reads the runs that already exist (`results/`, `experiments/`,
  `report/experiments/`, synced from the cluster). It follows the report's own
  choices of segment, model and parameters, so with it every section reproduces
  the report. A section whose runs are missing says which files and which
  `report/run_all.sh` job make them, and skips its remaining cells.
- **Live** makes a short run now with the panel's values. Its runs go only to
  `experiments/walkthrough/<ID>/`, never to `results/`. `n_frames` and the sweep
  extent set how big a live run is: small here, longer on the cluster.

**Same-code check.** Each section ends by calling the report's own function on
the same input and comparing with the unrolled result, loosely (`rtol=1e-2`).
It prints `[ok]` or `[DIFF]` and never stops the notebook.

The tracking run itself (`evaluation.experiment_tracking`) is not unrolled
here; section A0 links the per-model walkthroughs that show the inside of one.

Run from the repository root (the setup cell finds it). Every angular velocity
is shown in deg/s (`style.to_deg_s`); the code itself computes in rad/s.
'''),

('md', CORE, r'''
### What you see, cell by cell

| Section · cell | Shows |
|---|---|
| **setup** | nothing printed; imports and paths |
| **panel** | the widgets, at the report's defaults; *reset* restores them |
| **helpers** | nothing printed; `check`, `need`, `redirect`, `%%section` |
| **A0** | links to the five per-model walkthroughs |
| **A1 · 1** inputs | the window (20 ms in Redraw, panel dt in Live) and the sequences with data |
| **A1 · 2** reference | the reference ω per sequence: frame count and first 3 frames, deg/s |
| **A1 · 3** rows | window_choice rows per sequence and duration: \|ω\| deg/s, rel_std, sign flips |
| **A1 · 4** scene motion | rotation (deg) and centre-pixel travel (px) in one window of 10/20/30/50 ms |
| **A1 · 5** segment speeds | gyro speed of every candidate segment over 3 s; median-speed vs reported segment |
| **A1 · 6** text lines | the image-travel line (`intensity`) and the sign-changes line (`track_length`) |
| **A1 · check** | rows vs `cmd_window` and window_choice.csv; lines vs `segment_speeds`, `intensity`, `track_length` |
| **A2 · 1** the runs | one folder per model (Redraw) or the folder of the new run (Live); in Redraw, a line naming panel values no saved run has |
| **A2 · 2** reference and gyro | their shapes and the first 4 frames, deg/s |
| **A2 · 3** gyro floor | gyro error against the reference: deg/s and deg |
| **A2 · 4** rows | one `cmd_main` csv row per model: err, median, dir, β, \|r\| |
| **A2 · 5** text numbers | the lines of `poster_segments` and of the 3 s `inloop_identical` (Redraw) |
| **A2 · 6** track length | `track_length`: error and \|r\| over the first N frames of the 3 s runs |
| **A2 · check** | `[ok]`/`[DIFF]`: rows vs `main_base_*.csv`, text vs `text_numbers` (Redraw); config and row vs `cmd_main` itself (Live) |
| **A3 · 1** the runs | as A2 · 1, at 75 frames |
| **A3 · 2** rows | gyro floor and the 1.5 s rows of every model |
| **A3 · 5** tab:main | every sequence's cells as a table, then the LaTeX of Table III |
| **A3 · 6** fig:tracking | ω per axis over 1.5 s, poster seg_C, CMax-anchor and thesis vs reference |
| **A3 · 7** text numbers | the lines of `unanchored` and `reversal` (Redraw) |
| **A3 · 8** reference | `reference`: gyro against a one-frame and a ±50 ms reference, deg/s |
| **A3 · 9** intensity | first part of `intensity`: I std and \|r\| along the poster track |
| **A3 · 10** fig:dissociation | read-out of thesis / CMax-anchor / thesis+IMU on boxes, bicycle, street, with \|r\| and err |
| **A3 · check** | rows vs `dissoc_1p5s_*.csv`, Table III vs `table_main`, figure lines vs `fig_tracking`, images vs `fig_dissociation`, text vs `text_numbers` |
| **A4 · 1** the runs | the δ_curl 0.6 run of each model (Redraw) or the new run (Live) |
| **A4 · 2** rows | gyro floor and one `main_curl` row per model |
| **A4 · 5** tab:curl | err and \|r\| with and without curl, then the LaTeX |
| **A4 · 6** text numbers | the 1.5 s half of `inloop_identical` |
| **A4 · check** | rows vs `main_curl_*.csv`, table vs `table_curl`, text; Live: config and row vs `cmd_main` |
| **A5 · 1–4** baseline | inputs, events and CMax, the per-frame loop (ref / CMax / gyro, deg/s), metrics and csv rows |
| **A5 · 5** tab:baseline | CMax alone / gyro alone / network per sequence, then the LaTeX |
| **A5 · 6** fig_cmax_baseline | ω per axis of CMax and gyro vs reference, three IWEs (not in the tex) |
| **A5 · 7** text numbers | `maps_add` |
| **A5 · check** | rows vs `baseline_dt20ms.csv`, `table_baseline`, figure vs `fig_cmax_baseline`, text; Live: vs `cmd_baseline` |
| **A6 · 1–3** β runs | the 21 β runs; speed (deg/s) and mean β per run |
| **A6 · 4** fig:beta_speed | β against speed, three series with Pearson r and fits |
| **A6 · 5** text numbers | `beta_speed`, `synthetic_beta_original` |
| **A6 · check** | β vs `read_summary`, figure vs `fig_beta_speed`, text; Live: `cmd_beta` config |
| **A7 · 1–3** tab:anchor | runs per poster segment and model, gyro floor, the table (not in the tex) |
| **A7 · check** | `gyro_floor`, `table_anchor` |
| **A8 · defs** | nothing printed; the sweep jobs and helpers (sweep-defs) |
| **A8 · 1–2** sens runs | csv rows (Redraw) or the Live job with its run and frame totals; sweep points per axis |
| **A8 · 3** tab:sweep | err per block, value and sequence, then the LaTeX |
| **A8 · 4** fig_sensitivity | mean err and \|r\| per panel (not in the tex) |
| **A8 · 5** text numbers | `sensitivity_both`, `sweep_ends`, `anchor_ablation`, `anchors_across_weights`, `kappa_rows` |
| **A8 · check** | table, figure data, 5 text functions; Live: `cmd_sweep` config and rows |
| **A9 · 1–5** grid | err pivot δ_FR × δ_anchor, tab:grid (not in the tex), tab:params rows vs grid optimum, `grid` |
| **A9 · check** | `table_grid`, `table_params` rows and tex, text |
| **A10 · 1–3** old sweep | the 80 sweep_* jobs (2 read, 78 unread: flag), the base-point rows |
| **A10 · check** | `synthetic_beta_original` |
| **A11 · 1–4** ablation | tab:regularisers, fig:regularisers, `regularisers` |
| **A11 · check** | table, figure images, text |
| **A12 · 1–5** best image | `image_search`, tab:image, fig:image_settings, `image_window` |
| **A12 · check** | search, table, figure images, text |
| **A13 · 1–4** maps | `compute_maps` stopped at frame 44, fig:maps and appendix, `polarity` |
| **A13 · check** | stop-at-44 equals Table III, figure images, text |
| **image-defs** | nothing printed; helpers of the image-figure checks |
| **A14 · 1–3** convergence | npz snapshots (Redraw) or `cmd_converge`'s loop (Live); std of I per checkpoint; fig:converge |
| **A14 · check** | figure pixels vs `fig_converge`; Live: snapshots vs `cmd_converge` |
| **A15 · 1–4** reconstruction | `recon_cfg` runs at 20/10 ms (with commit and date), \|r\| per panel, fig:recon |
| **A15 · check** | figure pixels and \|r\| vs `fig_recon`; Live: config vs `cmd_recon` |
| **A16 · 1–4** read-out | `cmd_readout` scoring, tab:recon, the 10-vs-20 ms lines of `intensity` |
| **A16 · check** | rows vs readout.csv, `table_recon`, text; Live: vs `cmd_readout` |
| **A17 · 1–4** in-loop Poisson | iterative/fft/dct runs, five panels, `intensity_update` |
| **A17 · check** | text, stats and pixels vs `cmd_poisson` |
| **A18 · 1–4** distortion | calibration, `distortion_numbers`, events raw and undistorted, fig:distortion |
| **A18 · check** | numbers vs csv and `distortion_numbers`, pixels vs `fig_distortion` |
| **A19 · 1–3** parameters | `config.py` values, tab:params, table vs the panel's defaults |
| **A19 · check** | table vs `table_params` and the tex; panel defaults = table values |

A section whose runs are missing prints `missing: <file>` and the `run_all.sh`
job that makes it, then each of its cells prints `(skipped: <ID> inputs missing)`.
'''),

('code', CORE, r'''
# setup -- repo root, paths, imports
import os, sys, contextlib
ROOT = os.getcwd()
while not os.path.exists(os.path.join(ROOT, 'config.py')):
    ROOT = os.path.dirname(ROOT)
os.chdir(ROOT)
os.environ['IM_MLFLOW'] = '0'                  # walkthrough runs are not logged
for p in (ROOT, os.path.join(ROOT, 'report')):
    if p not in sys.path:
        sys.path.insert(0, p)
sys.argv = sys.argv[:1]                         # evaluation.py parses argv

import csv
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as wd
from IPython.display import display
from IPython.core.magic import register_cell_magic

import evaluation as E                          # the tracking run (Live mode)
import experiment_tracking
from config import THESIS_PARAMS, ITERS_PER_FRAME, POISSON_MODE
from interacting_maps.style import to_deg_s     # every omega is shown in deg/s
from report_runs import SEGMENTS, MODELS, run_cfg

# the report scripts the sections unroll and check against
import run_experiments, make_figures, make_tables, text_numbers
get_ipython().run_line_magic('matplotlib', 'inline')   # make_figures selected Agg
'''),

('code', CORE + ('widgets',), r'''
# the panel -- defaults are the report's settings
DELTAS = [k for k in THESIS_PARAMS if k.startswith('delta_')]
DEFAULTS = dict(mode='Redraw', model='thesis_cmax', segment='poster', dt=0.02,
                n_iters=ITERS_PER_FRAME, poisson=POISSON_MODE, n_frames=10,
                sweep='one point', **{k: THESIS_PARAMS[k] for k in DELTAS})

W = dict(mode=wd.ToggleButtons(options=['Redraw', 'Live']),
         model=wd.Dropdown(options=MODELS),
         segment=wd.Dropdown(options=[s[3] for s in SEGMENTS]),
         dt=wd.FloatText(step=0.005),
         n_iters=wd.IntText(),
         poisson=wd.Dropdown(options=['iterative', 'fft', 'dct']),
         n_frames=wd.IntText(),
         sweep=wd.ToggleButtons(options=['one point', 'one axis', 'full']),
         **{k: wd.FloatText(step=0.01) for k in DELTAS})


def reset_panel(_=None):
    for k, v in DEFAULTS.items():
        W[k].value = v


for k, w in W.items():
    w.description = k
reset_panel()
reset = wd.Button(description='reset to report defaults')
reset.on_click(reset_panel)
display(wd.VBox([W['mode'], wd.HBox([W['model'], W['segment']]),
                 wd.HBox([W['dt'], W['n_iters'], W['poisson']]),
                 wd.HBox([W['n_frames'], W['sweep']]),
                 wd.GridBox([W[k] for k in DELTAS],
                            layout=wd.Layout(grid_template_columns='repeat(3, 1fr)')),
                 reset]))
'''),

('code', CORE + ('helpers',), r'''
# helpers shared by every section
def params():
    """The panel's current values; read inside each cell, so a changed widget
    takes effect when the section is re-run."""
    p = {k: w.value for k, w in W.items()}
    p['seg'] = next(s for s in SEGMENTS if s[3] == p['segment'])   # (ds, sid, t0, label)
    p['deltas'] = {k: p[k] for k in DELTAS}
    return p


def walk_dir(ID):
    """experiments/walkthrough/<ID>: where Live runs and check outputs go."""
    d = os.path.join('experiments', 'walkthrough', ID)
    os.makedirs(d, exist_ok=True)
    return d


def need(paths, jobs=()):
    """True if every path exists; else print what is missing and what makes it."""
    miss = [p for p in paths if not os.path.exists(p)]
    for p in miss:
        print(f'missing: {p}')
    if miss and jobs:
        print('make with run_all.sh job(s): ' + ', '.join(jobs))
    return not miss


def check(what, got, want, rtol=1e-2):
    """Loose same-code check: numbers, arrays or dicts of numbers. Never raises."""
    keys = None
    try:
        if isinstance(want, dict):
            odd = sorted(set(want) ^ set(got))
            if odd:
                raise KeyError(f'keys only on one side: {odd}')
            keys = sorted(want)
            got, want = [got[k] for k in keys], [want[k] for k in keys]
        g, w = np.asarray(got, float), np.asarray(want, float)
        if g.size == 0 and w.size == 0:
            print(f'[skip] {what}: nothing to compare')
            return None
        if g.shape != w.shape:
            raise ValueError(f'shape {g.shape} vs {w.shape}')
        ok = bool(np.allclose(g, w, rtol=rtol, atol=1e-9, equal_nan=True))
        rel = np.nanmax(np.abs(g - w) / np.maximum(np.abs(w), 1e-9)) if g.size else 0.0
        detail = f'max rel diff {rel:.3g}'
        if keys and not ok:                     # name the entries that differ
            bad = ~np.isclose(g, w, rtol=rtol, atol=1e-9, equal_nan=True)
            detail += ' in ' + ', '.join(f'{k} {x:g} vs {y:g}'
                                         for k, x, y, b in zip(keys, g, w, bad) if b)
    except Exception as e:                      # a failed comparison is a DIFF
        ok, detail = False, f'{type(e).__name__}: {e}'
    print(f'[ok] {what}' if ok else f'[DIFF] {what}: {detail}')
    return ok


def nums(lines):
    """Every number in some printed lines, in order -- to compare text output."""
    import re
    return [float(x) for l in lines
            for x in re.findall(r'[-+]?\d+\.?\d*(?:e[-+]?\d+)?', l)]


def note_redraw(p, nf):
    """Redraw shows the report's saved runs. Say so, and name the panel values
    no saved run has: those need a rerun in Live mode."""
    off = [f'{k} {p[k]}' for k in DEFAULTS
           if k not in ('mode', 'model', 'segment', 'n_frames', 'sweep') and p[k] != DEFAULTS[k]]
    if p['n_frames'] != nf:
        off.append(f"n_frames {p['n_frames']} (saved: {nf})")
    print(f"Redraw: the report's saved runs of every model, {nf} frames, report settings.")
    if off:
        print(f"No saved run has the panel's {', '.join(off)}: rerun it in Live mode.")


@contextlib.contextmanager
def redirect(module, **attrs):
    """Set module attributes (output folders, save) and restore them, even on error."""
    old = {k: getattr(module, k) for k in attrs}
    for k, v in attrs.items():
        setattr(module, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(module, k, v)


READY = {}                                      # section ID -> its inputs exist


@register_cell_magic
def section(line, cell):
    """%%section A2 -- run the cell only if section A2's inputs were found."""
    ID = line.strip()
    if READY.get(ID, True):
        get_ipython().run_cell(cell).raise_error()   # a failing cell fails "Run all"
    else:
        print(f'(skipped: {ID} inputs missing)')
'''),
]
