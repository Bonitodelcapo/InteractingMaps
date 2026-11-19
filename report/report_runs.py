"""
report_runs.py — what the report's runs are and where they live.

Shared by run_experiments.py (which makes the runs), make_figures.py (which
draws them) and make_tables.py (which tabulates them), so the three agree on
the segments, the run folders and the image-settings search by construction.
"""

import csv
import os

import numpy as np

import experiment_tracking
from config import THESIS_PARAMS, get_dataset_paths
from interacting_maps.eval_io import load_aps_list, nearest_aps
from run_config import RunConfig

#: the segments reported in Table II -- do not substitute others
SEGMENTS = [
    ('boxes_rotation',    'seg_B', 7.322,  'boxes'),
    ('poster_rotation',   'seg_C', 8.816,  'poster'),
    ('dynamic_rotation',  'seg_D', 1.619,  'dynamic'),
    ('ecrot_bicycle', 'seg_A', 1.001,  'bicycle'),
    ('ecrot_street',  'seg_A', 1.001,  'street'),
]
MODELS = ['cook', 'thesis', 'thesis_imu', 'thesis_cmax', 'thesis_cmax_v2']
#: labels of the synthetic sequences
SYNTHETIC = ('bicycle', 'street')
#: the adopted step sizes of config.py, for the commands that fix them
ADOPTED = {k: THESIS_PARAMS[k] for k in ('delta_FR', 'delta_IMU')}

#: every segment of the two real sequences -- the beta-vs-speed population
BETA_SEGMENTS = [
    ('boxes_rotation', 'seg_A', 9.972), ('boxes_rotation', 'seg_B', 7.322),
    ('boxes_rotation', 'seg_C', 4.422), ('boxes_rotation', 'seg_D', 1.922),
    ('poster_rotation', 'seg_A', 14.166), ('poster_rotation', 'seg_C', 8.816),
    ('poster_rotation', 'seg_D', 1.866),
]
#: (model, delta_FR, delta_anchor) -- the three series of fig_beta_speed
BETA_CONFIGS = [('thesis_cmax', 0.5, 0.3), ('thesis_cmax', 0.1, 0.5),
                ('thesis_cmax_v2', 0.5, None)]


def run_cfg(ds, sid, t0, model, dt, nf, out_root=None, **kw):
    """The RunConfig of one report run; its output_dir is where the run lives."""
    seg = {'id': sid, 't_start': t0, 'frame_duration': dt, 'n_frames': nf,
           'initial_R': None, 'sensor_size': (180, 240)}
    return RunConfig(dataset=ds, model=model, segment=seg, n_frames=nf,
                     distortion_mode='C_full', out_root=out_root, **kw)


def recon_cfg(ds, sid, t0, model, dt, duration, root):
    """A run of the reconstruction figure: iterative update, adopted step sizes."""
    return run_cfg(ds, sid, t0, model, dt, int(round(duration / dt)), out_root=root,
                   poisson='iterative', deltas=dict(ADOPTED))


def converge_path(label, frame, name='converge'):
    """Where run_experiments --what converge stores one sequence's I snapshots."""
    return os.path.join('experiments', name, f'{label}_frame{frame}.npz')


def aps_frame(ds, t):
    """The APS frame nearest time t, for visual reference."""
    return nearest_aps(load_aps_list(get_dataset_paths(ds)['data_dir']), t)


def track_stats(d):
    """(track mean |r| of the read-out, mean omega error) of one run."""
    rows = list(csv.DictReader(open(os.path.join(d, 'tracking.csv'))))
    r = np.abs([float(x['recon_r']) for x in rows])
    return float(np.nanmean(r)), experiment_tracking.read_summary(d)['mean_err_deg_s']


#: (model, dt, delta_curl) searched for image quality. The 20 ms points are
#: the runs of Table III and of the curl table; the 10 ms ones are made by
#:   --what sweep --vary 'dt=0.01;delta_curl=0,0.6' --mode grid --name best_image
IMG_SEARCH = [(m, dt, c) for m in ('thesis', 'thesis_cmax')
              for dt in (0.02, 0.01) for c in (0.0, 0.6)]


def image_search(duration, name=''):
    """{(kind, cfg): (mean |r|, mean err)} over IMG_SEARCH, and the best cfg per kind.

    kind is 'real' or 'synthetic'; the best configuration of a kind is the
    IMG_SEARCH point with the highest mean |r|. Nothing is recomputed.
    """
    img_root = os.path.join('experiments', name or 'best_image')

    def run_dir(ds, sid, t0, model, dt, curl):
        return run_cfg(ds, sid, t0, model, dt, int(round(duration / dt)),
                       out_root=None if dt == 0.02 else img_root,
                       deltas={'delta_curl': curl} if curl else None).output_dir

    kinds = {k: [s for s in SEGMENTS if (s[3] in SYNTHETIC) == (k == 'synthetic')]
             for k in ('real', 'synthetic')}
    score, best = {}, {}
    for kind, seqs in kinds.items():
        for cfg in IMG_SEARCH:
            try:
                score[kind, cfg] = np.mean([track_stats(run_dir(ds, sid, t0, *cfg))
                                            for ds, sid, t0, _ in seqs], axis=0)
            except OSError:
                print(f'    missing runs for {kind} {cfg}')
        best[kind] = max((c for c in IMG_SEARCH if (kind, c) in score),
                         key=lambda c: score[kind, c][0])
    return score, best, run_dir
