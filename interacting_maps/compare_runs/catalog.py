"""The saved runs a comparison can pick from, and the cascading choices over them.

scan() reads only run.json (or the legacy params.json / summary.json) of every run
folder under the given roots; options() / matches() narrow them like the
walkthrough's saved-run lookup: each choice narrows the others, a chosen key keeps
its alternatives, 'any' is no filter.
"""
import os
import re

import numpy as np
import pandas as pd

import experiment_tracking

MISSING = '—'
BASE_KEYS = ['model', 'dataset', 'segment_id', 't_start', 'dt', 'n_frames', 'n_iters',
             'poisson', 'distortion_mode', 'r_init', 'r_sigma', 'r_seed', 'root']
_RECORDS = ('run.json', 'params.json')


_SKIP_DIRS = {'video_frames', 'iter_frames', 'iwe', 'states', 'replay', '.ipynb_checkpoints'}


def _run_folders(root):
    """Folders under root holding a run record; frame folders are not descended into."""
    found = []
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in _SKIP_DIRS)
        if any(n in files for n in _RECORDS):
            found.append(os.path.normpath(d))
    return found


def _segment_id(config, folder):
    m = re.match(r'(seg_[^_]+)', os.path.basename(folder))
    return config.get('segment_id') or (m.group(1) if m else MISSING)


def _row(folder, root):
    """One catalog row, or None if the folder's record is unreadable."""
    try:
        rec = experiment_tracking.read_run(folder)
        c, s = rec['config'], rec.get('summary') or {}
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not c:
        return None
    row = dict(folder=folder, root=root, model=c.get('model'), dataset=c.get('dataset'),
               segment_id=_segment_id(c, folder), t_start=c.get('t_start'),
               dt=c.get('frame_duration'), n_frames=c.get('n_frames'),
               n_iters=c.get('n_iters'), poisson=c.get('poisson', 'iterative'),
               distortion_mode=c.get('distortion_mode'),
               # frame-0 R: earlier runs stored none, they are the gyro warm start
               r_init=c.get('r_init', 'gyro'), r_sigma=c.get('r_sigma'),
               r_seed=c.get('r_seed'),
               err_deg_s=s.get('mean_err_deg_s', np.nan))
    row = {k: (MISSING if v is None else v) for k, v in row.items()}
    row.update(c.get('params') or {})
    return row


def scan(roots):
    """One row per run folder (holding run.json or params.json) under any root."""
    rows = [r for root in roots for f in _run_folders(root) if (r := _row(f, root))]
    cat = pd.DataFrame(rows, columns=['folder', *BASE_KEYS, 'err_deg_s'] if not rows else None)
    params = sorted(set(cat.columns) - {'folder', 'err_deg_s', *BASE_KEYS})
    cat[params] = cat[params].astype(object).where(cat[params].notna(), MISSING)
    return cat[['folder', *BASE_KEYS, *params, 'err_deg_s']]


def choice_keys(cat):
    """The dropdown keys: BASE_KEYS, then every run parameter (delta_*, ...)."""
    return BASE_KEYS + [k for k in cat.columns if k not in ('folder', 'err_deg_s', *BASE_KEYS)]


def _mask(cat, chosen, skip=None):
    m = pd.Series(True, index=cat.index)
    for k, v in chosen.items():
        if k != skip and v != 'any':
            m &= cat[k] == v
    return m


def _order(v):
    return (isinstance(v, str), v if not isinstance(v, str) else 0, str(v))


def options(cat, chosen):
    """Per key, the values of the runs matching every other choice, sorted."""
    return {k: sorted(cat.loc[_mask(cat, chosen, skip=k), k].unique().tolist(), key=_order)
            for k in choice_keys(cat)}


def matches(cat, chosen):
    """The runs matching every choice, lowest mean omega error first."""
    return cat[_mask(cat, chosen)].sort_values('err_deg_s', na_position='last')
