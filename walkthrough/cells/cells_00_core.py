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

('md', CORE + ('overview',), r'''
(replaced at build time by experiments.md, see walkthrough/build_nb.py::overview_md)
'''),

('md', CORE, r'''
### What you see, cell by cell

| Section · cell | Shows |
|---|---|
| **setup** | nothing printed; imports and paths |
| **overview** | experiments.md; each ID links to its section |
| **panel** | the old global panel, read only by sections without a form yet (A1, A5, A8–A13, A18, A19) |
| **registry** | the saved-runs lookup (model → available values → matching runs, best ω error first) |
| **forms** | nothing printed; the per-experiment form machinery |
| **helpers** | nothing printed; `check`, `need`, `redirect`, `%%section` |
| **A0** | links to the five per-model walkthroughs |
| **A1 · 1** inputs | the window (20 ms in Redraw, panel dt in Live) and the sequences with data |
| **A1 · 2** reference | the reference ω per sequence: frame count and first 3 frames, deg/s |
| **A1 · 3** rows | window_choice rows per sequence and duration: \|ω\| deg/s, rel_std, sign flips |
| **A1 · 4** scene motion | rotation (deg) and centre-pixel travel (px) in one window of 10/20/30/50 ms |
| **A1 · 5** segment speeds | gyro speed of every candidate segment over 3 s; median-speed vs reported segment |
| **A1 · 6** text lines | the image-travel line (`intensity`) and the sign-changes line (`track_length`) |
| **A1 · check** | rows vs `cmd_window` and window_choice.csv; lines vs `segment_speeds`, `intensity`, `track_length` |
| **A2 · form** | models, segment, n_frames, dt, n_iters, poisson, δ_FR; Redraw: saved values only · Live: free |
| **A2 · 1** the runs | one folder per model of the form; Live: `SAVED RUN (not rerun)` when it exists |
| **A2 · 2** reference and gyro | their shapes and the first 4 frames, deg/s |
| **A2 · 3** gyro floor | gyro error against the reference: deg/s and deg |
| **A2 · 4** rows | one `cmd_main` csv row per model: err, median, dir, β, \|r\| |
| **A2 · 5** text numbers | the lines of `poster_segments` and of the 3 s `inloop_identical` (Redraw) |
| **A2 · 6** track length | `track_length`: error and \|r\| over the first N frames of the 3 s runs |
| **A2 · check** | `[ok]`/`[DIFF]`: rows vs `main_base_*.csv`, text vs `text_numbers` (Redraw); config and row vs `cmd_main` itself (Live) |
| **A3 · form, 1** | as A2, at 75 frames; report-only steps skip away from the report configuration |
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
# the Live panel; shown with the Redraw lookup by the registry cell below
PANEL = wd.VBox([W['model'],
                 wd.HBox([W['dt'], W['n_iters'], W['poisson']]),
                 wd.HBox([W['n_frames'], W['sweep']]),
                 wd.GridBox([W[k] for k in DELTAS],
                            layout=wd.Layout(grid_template_columns='repeat(3, 1fr)')),
                 reset])
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
    """Redraw shows the report's saved runs; other saved runs are in the lookup."""
    print(f"Redraw: the report's saved runs of every model, {nf} frames, report settings "
          "(other saved runs: the lookup in the panel).")


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

('code', CORE + ('registry',), r'''
# run registry -- every saved run.json, the Redraw lookup and the Live guard
import html, io, json, re

LOOKUP_KEYS = ['segment', 'dt', 'n_frames', 'n_iters', 'poisson', 'distortion_mode', *DELTAS]
REG_COLS = ['folder', 'root', 'model', 'dataset', 'segment_id', 'segment', 't_start',
            *LOOKUP_KEYS[1:6], *DELTAS, 'err_deg_s', 'dir_err_deg', 'beta']


def scan_runs(roots=('results', 'experiments')):
    """One row per readable <root>/**/run.json. Old configs are filled like
    experiment_tracking.find_run fills them: segment_id from the folder name,
    poisson 'iterative', a step size the run did not store = 0.0."""
    rows = []
    for top in roots:
        for f in glob.glob(os.path.join(top, '**', 'run.json'), recursive=True):
            d = os.path.dirname(f)
            try:
                r = experiment_tracking.read_run(d)
                c, s = r['config'], r.get('summary') or {}
            except (OSError, ValueError, KeyError, TypeError):
                continue
            sid = c.get('segment_id') or (re.match(r'(seg_[^_]+)', os.path.basename(d)) or [None, '?'])[1]
            pr = c.get('params') or {}
            rows.append(dict(
                folder=d, root=os.path.dirname(os.path.dirname(os.path.dirname(d))),
                model=c.get('model'), dataset=c.get('dataset'), segment_id=sid,
                segment=f"{c.get('dataset')} {sid}", t_start=c.get('t_start'),
                dt=c.get('frame_duration'), n_frames=c.get('n_frames'),
                n_iters=c.get('n_iters'),
                poisson=c.get('poisson', 'iterative'), distortion_mode=c.get('distortion_mode'),
                **{k: float(pr.get(k) or 0.0) for k in DELTAS},
                err_deg_s=s.get('mean_err_deg_s', np.nan),
                dir_err_deg=s.get('mean_dir_err_deg', np.nan), beta=s.get('mean_beta', np.nan)))
    return pd.DataFrame(rows, columns=REG_COLS)


def matching(reg, chosen, skip=None):
    """Rows of reg that match every choice ('any' or absent = no filter), except `skip`."""
    m = pd.Series(True, index=reg.index)
    for k, v in chosen.items():
        if k != skip and v != 'any':
            m &= reg[k] == v
    return reg[m]


def available(reg, chosen):
    """For each lookup key, the values saved runs have given every other choice
    (a chosen dropdown keeps its alternatives)."""
    return {k: sorted(matching(reg, chosen, skip=k)[k].dropna().unique().tolist())
            for k in LOOKUP_KEYS}


def show_matches(chosen):
    """The matching runs, best mean omega error first, into the MATCH widget."""
    hit = matching(REG, chosen).sort_values('err_deg_s')
    cols = ['folder', 'segment', 'n_frames', 'err_deg_s', 'dir_err_deg', 'beta']
    MATCH.value = ('<b>no saved run matches</b>' if hit.empty else
                   f'<b>{len(hit)} run(s)</b> -- play them back in the inspection viewer'
                   + hit[cols].to_html(index=False, float_format='%.2f'))
    return 'no saved run matches' if hit.empty else hit[cols].to_string(index=False)


# --- the Redraw lookup: model first, then each choice narrows the others
LOOK = {'model': wd.Dropdown(options=MODELS, description='model'),
        **{k: wd.Dropdown(options=['any'], description=k) for k in LOOKUP_KEYS}}
INFO, MATCH, HEAD = wd.HTML(), wd.HTML(), wd.HTML()
_busy = [False]


def chosen_now():
    return {k: w.value for k, w in LOOK.items()}


def refresh(change=None, keep=False):
    """Refill every dropdown from REG; a model change resets the others and
    prints the values that model's saved runs have (keep=True: a rescan keeps them)."""
    if _busy[0]:
        return
    _busy[0] = True
    try:
        HEAD.value = f'<b>Saved runs</b> ({len(REG)} run.json under results/ and experiments/)'
        if not keep and (change is None or change['owner'] is LOOK['model']):
            for k in LOOKUP_KEYS:
                LOOK[k].options, LOOK[k].value = ['any'], 'any'
            vals = available(REG, {'model': LOOK['model'].value})
            INFO.value = '<pre>' + html.escape(repr(vals), quote=False) + '</pre>'
        av = available(REG, chosen_now())
        for k in LOOKUP_KEYS:
            keep = LOOK[k].value
            LOOK[k].options = ['any', *av[k]]
            LOOK[k].value = keep if keep in LOOK[k].options else 'any'
        show_matches(chosen_now())
    finally:
        _busy[0] = False


def rescan(_=None):
    global REG
    REG = scan_runs()
    refresh(keep=True)


for w in LOOK.values():
    w.observe(refresh, names='value')
rescan_btn = wd.Button(description='rescan runs')
rescan_btn.on_click(rescan)
REG = scan_runs()
refresh()
LOOKUP = wd.VBox([HEAD, LOOK['model'], INFO,
                  wd.GridBox([LOOK[k] for k in LOOKUP_KEYS],
                             layout=wd.Layout(grid_template_columns='repeat(3, 1fr)')),
                  rescan_btn, MATCH])


def show_mode(change=None):
    LOOKUP.layout.display = '' if W['mode'].value == 'Redraw' else 'none'
    PANEL.layout.display = 'none' if W['mode'].value == 'Redraw' else ''


W['mode'].observe(show_mode, names='value')
show_mode()
TOP = wd.HBox([W['mode'], W['segment']])          # the sections' segment, in both modes
display(wd.VBox([TOP, LOOKUP, PANEL]))


# --- saved runs of one configuration (used by saved_or_run in the forms cell)
def saved_runs(rc):
    """Complete saved runs (with a summary) of rc's configuration, by
    experiment_tracking.find_run, in every scanned root and in rc.out_root."""
    roots = sorted(set(REG.root) | {rc.out_root})
    with contextlib.redirect_stdout(io.StringIO()):      # find_run's AMBIGUOUS lines
        hits = [experiment_tracking.find_run(
                    root, rc.dataset, rc.model, rc.segment_id, rc.n_frames, rc.params,
                    dt=rc.frame_duration, t_start=rc.t_start, n_iters=rc.n_iters,
                    poisson=rc.poisson, distortion_mode=rc.distortion_mode) for root in roots]
    return [h for h in hits
            if h and 'mean_err_deg_s' in (experiment_tracking.read_summary(h) or {})]
'''),

('code', CORE + ('forms',), r'''
# forms -- one per experiment: its own mode and knobs; never a global value
from collections import namedtuple
from IPython.display import HTML

# kind: 'one' | 'many'. choices: a fixed list (configs, cases, solvers ...) offered in
# both modes, not looked up in the registry; the section says which have no output.
Knob = namedtuple('Knob', 'name report kind choices', defaults=('one', None))


def _pairs(choices):
    """Widget options for a choices knob: (label, value), so tuples stay values."""
    return [(str(c), c) for c in choices]
FORMS = {}                                      # ID -> {'mode', 'knobs', 'redraw', 'live', 'hints', 'box'}
LABEL = {f'{ds} {sid}': lab for ds, sid, _t0, lab in SEGMENTS}      # REG segment -> panel label
COL = {'models': 'model'}                       # knob name -> REG column
LIVE_FRAMES = 10                                # Live n_frames default: smoke size here
PRESET = globals().get('PRESET', {})            # starting values for new forms (tests set it)


def saved_view(fixed=None):
    """REG rows a form can redraw: report segments, C_full, the form's fixed values."""
    v = REG[REG.distortion_mode == 'C_full'].copy()
    v['segment'] = v['segment'].map(LABEL)
    for k, val in (fixed or {}).items():
        v = v[v[k] == val]
    return v.dropna(subset=['segment'])


def form_options(ID):
    """Redraw options of each knob: values of saved runs that fit the form's other knobs."""
    F = FORMS[ID]
    v = saved_view(F['fixed'])
    opts = {}
    for k in F['knobs']:
        if k.choices is not None:
            continue
        rows = v
        for o in F['knobs']:
            val = F['redraw'][o.name].value
            if o.name == k.name or o.choices is not None or val in (None, ()):
                continue
            col = COL.get(o.name, o.name)
            rows = rows[rows[col].isin(val)] if o.kind == 'many' else rows[rows[col] == val]
        opts[k.name] = sorted(rows[COL.get(k.name, k.name)].dropna().unique().tolist())
    return opts


def _refresh_form(ID):
    """Refill the Redraw options until they agree with the values (a value forced to
    another option changes the options of the others)."""
    F = FORMS[ID]
    if F['busy']:
        return
    F['busy'] = True
    try:
        for _ in range(len(F['knobs']) + 1):
            changed = False
            for name, vals in form_options(ID).items():
                w = F['redraw'][name]
                keep = w.value
                if list(w.options) != vals:
                    w.options = vals
                if isinstance(keep, tuple):
                    new = tuple(x for x in keep if x in vals) or tuple(vals)
                else:
                    new = keep if keep in vals else (vals[0] if vals else None)
                w.value = new
                changed |= new != keep
            if not changed:
                break
    finally:
        F['busy'] = False


def _choice_widget(k):
    w = (wd.SelectMultiple if k.kind == 'many' else wd.Dropdown)(options=_pairs(k.choices),
                                                                  description=k.name)
    w.value = tuple(k.report) if k.kind == 'many' else k.report
    return w


def _live_widget(k):
    if k.choices is not None:
        return _choice_widget(k)
    if k.name == 'models':
        return wd.SelectMultiple(options=MODELS, value=tuple(k.report), description='models')
    if k.name == 'segment':
        return wd.Dropdown(options=list(LABEL.values()), value=k.report, description='segment')
    if k.name == 'poisson':
        return wd.Dropdown(options=['iterative', 'fft', 'dct'], value=k.report, description='poisson')
    if k.name in ('n_frames', 'n_iters'):
        return wd.IntText(value=LIVE_FRAMES if k.name == 'n_frames' else k.report, description=k.name)
    return wd.FloatText(value=k.report, description=k.name, step=0.01)


def form(ID, knobs, fixed=None):
    """Show experiment ID's form: mode, one widget per knob, reset. Redraw offers only
    saved values (cascading) among runs with the `fixed` values (e.g. no curl);
    Live is free (n_frames from 10, report value as hint). Re-running keeps the values."""
    if ID not in FORMS:
        F = FORMS[ID] = dict(knobs=list(knobs), busy=False, fixed=dict(fixed or {}),
                             mode=wd.ToggleButtons(options=['Redraw', 'Live'], description=ID))
        F['redraw'] = {k.name: (_choice_widget(k) if k.choices is not None else
                                wd.SelectMultiple(description=k.name, value=()) if k.kind == 'many'
                                else wd.Dropdown(description=k.name)) for k in knobs}
        F['live'] = {k.name: _live_widget(k) for k in knobs}
        F['hints'] = {k.name: wd.HTML(f'<i>report: {k.report}</i>') for k in knobs}
        for k in knobs:                         # the report value first, where saved
            if k.choices is not None:
                continue
            F['redraw'][k.name].options = [k.report] if k.kind == 'one' else list(k.report)
            F['redraw'][k.name].value = k.report if k.kind == 'one' else tuple(k.report)
        _refresh_form(ID)
        _preset(F)
        _refresh_form(ID)
        for w in F['redraw'].values():
            w.observe(lambda _c, ID=ID: _refresh_form(ID), names='value')
        reset = wd.Button(description='reset to report')
        reset.on_click(lambda _b, ID=ID: reset_form(ID))
        red = wd.GridBox(list(F['redraw'].values()),
                         layout=wd.Layout(grid_template_columns='repeat(3, 1fr)'))
        live = wd.GridBox([wd.VBox([F['live'][k.name], F['hints'][k.name]]) for k in knobs],
                          layout=wd.Layout(grid_template_columns='repeat(3, 1fr)'))
        note = wd.HTML()
        def show(_c=None, red=red, live=live, note=note, F=F):
            r = F['mode'].value == 'Redraw'
            red.layout.display, live.layout.display = ('', 'none') if r else ('none', '')
            note.value = ('<i>Redraw: only values with saved runs</i>' if r else
                          '<i>Live: free values; a saved configuration is shown, not rerun</i>')
        F['mode'].observe(show, names='value')
        show()
        F['box'] = wd.VBox([wd.HBox([F['mode'], reset]), red, live, note],
                           layout=wd.Layout(border='1px solid #999', padding='4px'))
    display(FORMS[ID]['box'])


def _preset(F):
    """Start a new form from PRESET (mode, and knob values; 'model' sets 'models')."""
    pre = dict(PRESET)
    if 'model' in pre:
        pre.setdefault('models', (pre['model'],))
    F['mode'].value = pre.get('mode', F['mode'].value)
    for k in F['knobs']:
        if k.name not in pre:
            continue
        v = tuple(pre[k.name]) if k.kind == 'many' else pre[k.name]
        F['live'][k.name].value = v
        r = F['redraw'][k.name]
        opts = k.choices if k.choices is not None else r.options
        if (set(v) <= set(opts)) if k.kind == 'many' else (v in opts):
            r.value = v


def reset_form(ID):
    """Every knob back to the report's value (Redraw: where saved runs have it)."""
    F = FORMS[ID]
    F['busy'] = True                            # no cascade while the values are set
    try:
        for k in F['knobs']:
            rep = tuple(k.report) if k.kind == 'many' else k.report
            F['live'][k.name].value = LIVE_FRAMES if k.name == 'n_frames' else rep
            w = F['redraw'][k.name]
            if k.choices is None:
                w.options = list(rep) if k.kind == 'many' else [rep]
            w.value = rep
    finally:
        F['busy'] = False
    _refresh_form(ID)


panel_params = globals().get('panel_params', params)   # the old global panel; kept on a re-run


def params(ID=None):
    """Experiment ID's form values, with the keys of the old panel params();
    params() without an ID reads the old global panel."""
    if ID is None:
        return panel_params()
    F = FORMS[ID]
    q = {**DEFAULTS, 'mode': F['mode'].value, 'models': list(MODELS)}
    ws = F['redraw'] if q['mode'] == 'Redraw' else F['live']
    for k in F['knobs']:
        v = ws[k.name].value
        q[k.name] = list(v) if k.kind == 'many' else v
    for k in ('n_frames', 'n_iters'):           # REG columns can come back as floats
        q[k] = int(q[k]) if q[k] is not None else None
    q['form'] = ID
    if not any(k.name == 'model' for k in F['knobs']):     # a form's own model knob wins
        q['model'] = q['models'][0] if q['models'] else None
    q['seg'] = next((s for s in SEGMENTS if s[3] == q['segment']), None)
    q['deltas'] = {k: q[k] for k in DELTAS}
    return q


def is_report(ID):
    """Every knob at the report's value. The choice of models and of the segment does
    not count: the report has runs of every model on every Table II segment."""
    q = params(ID)
    same = lambda k: (set(q[k.name]) == set(k.report)) if k.kind == 'many' else q[k.name] == k.report
    return all(same(k) for k in FORMS[ID]['knobs'] if k.name not in ('models', 'segment'))


def report_skip(ID):
    """True (and says so) when the form is away from the report's configuration."""
    if is_report(ID):
        return False
    print('[skip] not the report configuration')
    return True


def roots():
    """Where a form looks for saved runs: results/ first, then every other scanned root."""
    return ['results'] + sorted(set(REG.root) - {'results'})


def report_or_saved(rc, root, need='maps_final.npz'):
    """Redraw: rc's run in the report job's root (the folder the job names, as the
    report reads it), else any complete saved run with that configuration; it must
    hold `need`. None if there is none."""
    old, rc.out_root = rc.out_root, root
    try:
        d = rc.output_dir
    finally:
        rc.out_root = old
    if os.path.exists(os.path.join(d, need)):
        return d
    hits = [h for h in saved_runs(rc) if os.path.exists(os.path.join(h, need))]
    return hits[0] if hits else None


def saved_or_run(rc):
    """The folder of rc's run: a complete saved run is used, not rerun (banner);
    otherwise run it now and save it."""
    hits = saved_runs(rc)
    if hits:
        print(f'>>> SAVED RUN (not rerun): {hits[0]}')
        display(HTML(f'<b style="color:#c00">SAVED RUN (not rerun):</b> <code>{hits[0]}</code>'))
        return hits[0]
    E.experiment_tracking(rc, save_frames=False)
    rescan()
    return rc.output_dir
'''),
]
