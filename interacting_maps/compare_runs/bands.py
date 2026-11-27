"""The comparison grid, one band at a time: pure functions from runs to figures,
HTML or image paths. Runs keep their slot colour everywhere; the same quantity
shares its axis (curves overlaid) or its colour scale (maps).
"""
import math
import os
from collections import namedtuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import experiment_tracking
from interacting_maps import style, viz
from interacting_maps.eval_io import load_aps_list, nearest_aps
from interacting_maps.network_dissertation import solve_poisson_exact
from . import outputs
from .catalog import MISSING, choice_keys

Run = namedtuple('Run', 'slot color row ticked')
SLOT_COLORS = [matplotlib.colors.to_hex(c) for c in plt.get_cmap('tab20').colors]
DIFF_STYLE = 'background:#fff3c4'
MAP_COLS = 5                      # map panels per figure row
_SEG_LS = ['-', '--', ':', '-.']  # reference line style per segment
_NOT_CONFIG = ('folder', 'err_deg_s')


# ------------------------------------------------------------ labels, warning
def _keys(runs):
    return [k for k in choice_keys(pd.DataFrame([r.row for r in runs]))]


def _fmt(k, v):
    if k == 'model':
        return style.MODELS.get(v, (v,))[0]
    if k == 'dt' and v != MISSING:
        return f'dt={1000 * v:g}ms'
    if k.startswith('delta_'):
        return f'δ_{k[6:]}={v}'
    return f'{k}={v}'


def differing(runs):
    """The choice keys whose value is not the same in all runs."""
    return [k for k in _keys(runs) if len({str(r.row[k]) for r in runs}) > 1]


def labels(runs):
    """slot -> '#<slot> ' + the config values that differ between the runs."""
    diff = differing(runs)
    return {r.slot: f'#{r.slot} ' + (' '.join(_fmt(k, r.row[k]) for k in diff)
                                     or _fmt('model', r.row['model']))
            for r in runs}


def _segment(r):
    return f"{r.row['dataset']} {r.row['segment_id']}"


def segment_warning(runs):
    segs = list(dict.fromkeys(_segment(r) for r in runs))
    if len(segs) < 2:
        return None
    return (f'Runs span {len(segs)} segments ({", ".join(segs)}): comparing runs of '
            'different segments does not make sense.')


# ------------------------------------------------------------ curves
def _t(df, dt):
    """Seconds since the run's t_start at each row (frame midpoints)."""
    return (df['frame'].to_numpy() + 0.5) * dt


def _references(ax, runs, axis):
    """Reference and IMU of ω_axis, once per distinct segment."""
    first = {}
    for r in runs:
        first.setdefault(_segment(r), r)
    for i, (seg, r) in enumerate(first.items()):
        df = outputs.read_tracking(r.row.folder)
        t = _t(df, r.row['dt'])
        ls, tag = _SEG_LS[i % len(_SEG_LS)], (f' {seg}' if len(first) > 1 else '')
        src = df['ref_source'].iloc[0] if 'ref_source' in df else 'groundtruth'
        ref = f'gt_w{"xyz"[axis]}'
        if ref in df:
            ax.plot(t, style.to_deg_s(df[ref].to_numpy()), **{**style.REF_STYLE, 'ls': ls},
                    label=style.ref_label(src) + tag)
        imu = f'imu_w{"xyz"[axis]}'
        if imu in df and src != 'imu':
            ax.plot(t, style.to_deg_s(df[imu].to_numpy()), **{**style.IMU_STYLE, 'ls': ls,
                                                              'label': 'IMU gyro' + tag})


def _draw_curve(ax, key, r, lab, df, t):
    if key == 'recon_r':
        ax.plot(t, np.abs(df['recon_r']), color=r.color, label=lab, **style.EST_STYLE)
        if 'recon_r_inloop' in df:
            ax.plot(t, np.abs(df['recon_r_inloop']), color=r.color, ls='--', lw=1.0)
        return
    col = {'err': 'err_deg_s', 'dir_err': 'dir_err_deg', 'beta': 'beta', 'I_std': 'I_std',
           't_frame': 't_frame_s', 'contrast': 'contrast'}[key]
    ax.plot(t, df[col].to_numpy(), color=r.color, label=lab, **style.EST_STYLE)


_YLABEL = {'err': style.LABEL_ERROR, 'dir_err': style.LABEL_DIR_ERROR, 'beta': style.LABEL_BETA,
           'recon_r': '|r| vs APS (solid: I from G, dashed: in-loop)',
           'I_std': 'std of in-loop I', 't_frame': 'frame time (s)', 'contrast': 'IWE contrast'}


def _curve_data(r, key):
    """(DataFrame, t) of one run for a curve key, or None if the run lacks it."""
    if key == 'contrast':
        p = os.path.join(r.row.folder, 'iwe', 'iwe_log.csv')
        df = pd.read_csv(p) if os.path.exists(p) else None
    elif key in outputs.available(r.row.folder):
        df = outputs.read_tracking(r.row.folder)
    else:
        df = None
    return None if df is None else (df, _t(df, r.row['dt']))


def curve_figure(runs, key, lab=None):
    """All runs of one curve key overlaid on one figure; returns (fig, slots left out)."""
    lab = lab or labels(runs)
    data = {r.slot: _curve_data(r, key) for r in runs}
    missing = [s for s, d in data.items() if d is None]
    drawn = [r for r in runs if data[r.slot] is not None]
    with style.use_style('screen'):
        n = 3 if key == 'omega' else 1
        fig, axes = plt.subplots(n, 1, figsize=(style.SCREEN_WIDTH, 2.6 * n), sharex=True,
                                 squeeze=False)
        axes = axes[:, 0]
        for a, ax in enumerate(axes):
            if key == 'omega':
                _references(ax, drawn, a)
            for r in drawn:
                df, t = data[r.slot]
                if key == 'omega':
                    ax.plot(t, style.to_deg_s(df[f'est_w{"xyz"[a]}'].to_numpy()), color=r.color,
                            label=lab[r.slot], **style.EST_STYLE)
                else:
                    _draw_curve(ax, key, r, lab[r.slot], df, t)
            ax.set_ylabel(style.omega_label(a) if key == 'omega' else _YLABEL[key])
            if key == 'recon_r':
                ax.set_ylim(0, 1)
        axes[-1].set_xlabel(style.LABEL_TIME + ' since t_start')
        axes[0].legend(loc='lower left', bbox_to_anchor=(0, 1), ncol=4, frameon=False,
                       fontsize=8)
    return fig, missing


# ------------------------------------------------------------ table
def _table_rows(runs, lab):
    rows = []
    for r in runs:
        rec = experiment_tracking.read_run(r.row.folder)
        row = {'run': lab[r.slot], 'folder': os.path.basename(r.row.folder)}
        for k in _keys(runs):
            v = r.row[k]
            row['dt (ms)' if k == 'dt' else k] = 1000 * v if k == 'dt' and v != MISSING else v
        prov = rec.get('provenance') or {}
        row['commit'] = prov.get('git_commit_short', MISSING)
        row['dirty'] = prov.get('git_dirty', MISSING)
        row.update({k: v for k, v in (rec.get('summary') or {}).items()
                    if isinstance(v, (int, float, str, bool))})
        rows.append(row)
    return pd.DataFrame(rows)


def table_html(runs, only_diff=False, lab=None):
    """One row per run: config, provenance, summary; differing columns highlighted."""
    df = _table_rows(runs, lab or labels(runs))
    same = [c for c in df.columns if c != 'run' and df[c].astype(str).nunique() <= 1]
    if only_diff:
        df = df.drop(columns=same)
    fmt = lambda v: f'{v:.4g}' if isinstance(v, float) else str(v)
    head = ''.join(f'<th>{c}</th>' for c in df.columns)
    body = ''.join(
        '<tr>' + ''.join(f'<td style="{"" if c in same or c == "run" else DIFF_STYLE}">'
                         f'{fmt(v)}</td>' for c, v in row.items()) + '</tr>'
        for _, row in df.iterrows())
    return f'<table><tr>{head}</tr>{body}</table>'


# ------------------------------------------------------------ maps
def _map_array(r, which, kind):
    """(array, draw kind, title note) of one run's map, or (None, None, note)."""
    m = outputs.read_maps(r.row.folder, which)
    if kind == 'I_from_G':
        return solve_poisson_exact(m['G'], 'fft'), 'I', 'I from G (exact)'
    if kind == 'aps':
        t, note = outputs.t_mid(r.row.folder, which)
        d = outputs.aps_dir(r.row.folder)
        if t is None or d is None:
            return None, None, note if t is None else f'no APS for {r.row["dataset"]}'
        img = nearest_aps(load_aps_list(d), t)
        return img, 'aps', '' if note == 'saved' else note
    return m.get(kind), kind, ''


def _grid(n):
    cols = min(n, MAP_COLS)
    return math.ceil(n / cols), cols


def map_figure(runs, key, lab=None):
    """One panel per run of one map (key 'final:I', 'center:aps', ...), shared colour scale."""
    lab = lab or labels(runs)
    which, kind = key.split(':')
    got = [(r, *_map_array(r, which, kind)) for r in runs]
    arrs = [a for _, a, _, _ in got if a is not None]
    draw_kind = next((k for _, a, k, _ in got if a is not None), None)
    same = len({a.shape for a in arrs}) == 1
    scale = viz.map_scale(draw_kind, np.stack(arrs)) if arrs and same else None
    nr, nc = _grid(len(runs))
    with style.use_style('screen'):
        fig, axes = plt.subplots(nr, nc, figsize=(2.6 * nc, 2.2 * nr), squeeze=False)
        for ax in axes.flat[len(runs):]:
            ax.set_axis_off()
        for ax, (r, a, k, note) in zip(axes.flat, got):
            title = lab[r.slot] + (f'\n{note}' if note else '') + ('' if same else '\nown scale')
            if a is None:
                ax.set_axis_off()
                ax.set_title(title + ('' if note else f'\nno {kind}'), color=r.color)
                continue
            viz.draw_map(ax, k, a, title=title, scale=scale)
            ax.title.set_color(r.color)
        fig.suptitle(outputs.BY_KEY[key].label)
    return fig


def state_figure(runs, frame, lab=None):
    """Saved states/frame_<k>.npz: rows V, I, G, F; one column per run; shared scales."""
    lab = lab or labels(runs)
    states = []
    for r in runs:
        paths = outputs.frame_paths(r.row.folder, 'states')
        states.append(dict(np.load(paths[frame])) if frame < len(paths) else None)
    kinds = ['V', 'I', 'G', 'F']
    with style.use_style('screen'):
        fig, axes = plt.subplots(len(kinds), len(runs), figsize=(2.6 * len(runs), 8.4),
                                 squeeze=False)
        for i, kind in enumerate(kinds):
            arrs = [s[kind] for s in states if s is not None and kind in s]
            same = arrs and len({a.shape for a in arrs}) == 1
            scale = viz.map_scale(kind, np.stack(arrs)) if same else None
            for j, (r, s) in enumerate(zip(runs, states)):
                ax = axes[i, j]
                if s is None or kind not in s:
                    ax.set_axis_off()
                    ax.set_title(f'{lab[r.slot]}\nno frame {frame}' if i == 0 else '')
                    continue
                viz.draw_map(ax, kind, s[kind], title=lab[r.slot] if i == 0 else None,
                             scale=scale)
    return fig


# ------------------------------------------------------------ images
def image_cells(runs, key, frame=None):
    """[(slot, path or None)]: a saved figure/movie, or frame `frame` of a sequence."""
    cells = []
    for r in runs:
        if key in outputs.SEQUENCES:
            paths = outputs.frame_paths(r.row.folder, key)
            p = paths[frame] if frame is not None and frame < len(paths) else None
        else:
            p = os.path.join(r.row.folder, outputs.MOVIES.get(key, key))
            p = p if os.path.exists(p) else None
        cells.append((r.slot, p))
    return cells
