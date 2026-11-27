"""Every output a run folder can hold, which of them a run has, and lazy loaders.

Output keys are what the comparison's checkboxes tick. available() reads only
csv headers, npz member lists and file existence.
"""
import functools
import glob
import os
import re
from collections import namedtuple

import numpy as np
import pandas as pd

import experiment_tracking

Output = namedtuple('Output', 'key label band')

# curve key -> tracking.csv columns it needs
CURVE_COLUMNS = {'omega': ['est_wx', 'est_wy', 'est_wz'], 'err': ['err_deg_s'],
                 'dir_err': ['dir_err_deg'], 'beta': ['beta'], 'recon_r': ['recon_r'],
                 'I_std': ['I_std'], 't_frame': ['t_frame_s']}
FIGURES = ['tracking_plot.png', 'recon_final.png', 'recon_plot.png', 'iwe/contrast_curve.png',
           'exp1_convergence.png', 'exp3_parameters.png', 'exp6_basin.png']
MAP_KINDS = ['V', 'I', 'I_from_G', 'G', 'F', 'aps']
# frame sequence key -> glob of its frames, relative to the run folder
SEQUENCES = {'video_frames': 'video_frames/frame_*.png', 'iwe_frames': 'iwe/frame_*.png',
             'states': 'states/frame_*.npz'}
MOVIES = {'iwe_sequence.gif': 'iwe/iwe_sequence.gif', 'comparison.mp4': 'comparison.mp4'}

OUTPUTS = [
    Output('omega', 'ω estimate (x, y, z)', 'curves'),
    Output('err', '|ω| error', 'curves'),
    Output('dir_err', 'direction error', 'curves'),
    Output('beta', 'scale β', 'curves'),
    Output('recon_r', 'reconstruction |r| vs APS', 'curves'),
    Output('I_std', 'std of in-loop I', 'curves'),
    Output('t_frame', 'frame time', 'curves'),
    Output('contrast', 'CMax contrast (iwe_log.csv)', 'curves'),
    Output('table', 'config + summary table', 'table'),
    *[Output(f, f, 'figures') for f in FIGURES],
    *[Output(f'{w}:{k}', f'{k} ({w})', 'maps') for w in ('final', 'center') for k in MAP_KINDS],
    Output('video_frames', 'per-frame panels', 'frames'),
    Output('iwe_frames', 'CMax IWE frames', 'frames'),
    Output('states', 'saved states (V, I, G, F)', 'frames'),
    *[Output(k, k, 'frames') for k in MOVIES],
]
BY_KEY = {o.key: o for o in OUTPUTS}


@functools.lru_cache(maxsize=256)
def read_tracking(folder):
    return pd.read_csv(os.path.join(folder, 'tracking.csv'))


@functools.lru_cache(maxsize=64)
def read_maps(folder, which):
    """maps_<which>.npz as a dict of arrays (which: 'final' or 'center')."""
    with np.load(os.path.join(folder, f'maps_{which}.npz')) as z:
        return {k: z[k] for k in z.files}


def clear_cache():
    read_tracking.cache_clear()
    read_maps.cache_clear()


def _frame_number(path):
    m = re.search(r'(\d+)\D*$', os.path.basename(path))
    return int(m.group(1)) if m else -1


def frame_paths(folder, seq):
    """The frames of a sequence, ordered by frame number."""
    return sorted(glob.glob(os.path.join(folder, SEQUENCES[seq])), key=_frame_number)


def t_mid(folder, which):
    """(time of the maps in maps_<which>.npz, where it came from) or (None, reason).
    Final maps without a stored t_mid are the last tracking.csv frame; center maps
    without one are not guessed."""
    m = read_maps(folder, which)
    if 't_mid' in m:
        return float(m['t_mid']), 'saved'
    if which == 'final' and os.path.exists(os.path.join(folder, 'tracking.csv')):
        return float(read_tracking(folder)['time'].iloc[-1]), 't_mid from tracking.csv'
    return None, 'no t_mid saved'


def aps_dir(folder, data_root='data'):
    """data/<dataset> of the run if it lists APS frames, else None."""
    ds = experiment_tracking.read_config(folder).get('dataset')
    d = os.path.join(data_root, ds or '')
    return d if ds and os.path.exists(os.path.join(d, 'images.txt')) else None


def _csv_columns(path):
    return set(pd.read_csv(path, nrows=0).columns) if os.path.exists(path) else set()


def _npz_members(path):
    if not os.path.exists(path):
        return set()
    with np.load(path) as z:
        return set(z.files)


def _map_keys(folder, which, data_root):
    have = _npz_members(os.path.join(folder, f'maps_{which}.npz'))
    keys = {k for k in ('V', 'I', 'G', 'F') if k in have}
    if 'G' in have:
        keys.add('I_from_G')
    if have and aps_dir(folder, data_root) and t_mid(folder, which)[0] is not None:
        keys.add('aps')
    return {f'{which}:{k}' for k in keys}


def available(folder, data_root='data'):
    """The output keys this run folder has."""
    cols = _csv_columns(os.path.join(folder, 'tracking.csv'))
    keys = {k for k, need in CURVE_COLUMNS.items() if set(need) <= cols}
    if os.path.exists(os.path.join(folder, 'iwe', 'iwe_log.csv')):
        keys.add('contrast')
    keys.add('table')
    keys |= {f for f in FIGURES if os.path.exists(os.path.join(folder, f))}
    for which in ('final', 'center'):
        keys |= _map_keys(folder, which, data_root)
    keys |= {s for s in SEQUENCES if frame_paths(folder, s)}
    keys |= {k for k, p in MOVIES.items() if os.path.exists(os.path.join(folder, p))}
    return keys
