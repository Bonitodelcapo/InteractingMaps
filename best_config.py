"""
best_config.py — Best hyper-parameters per (dataset, segment, model).

These are derived from the Exp-8 parameter-grid sweeps
(results/parameter_grid_<dataset>.csv). For each (dataset, segment_id, model)
we keep the single grid row with the lowest ``mean_err_deg_s``.

Usage
-----
Look one up in code / from evaluation.py::

    from best_config import get_best_config
    cfg = get_best_config('dynamic_rotation', 'seg_A', 'thesis_imu')
    # -> {'frame_duration': 0.01, 'n_frames': 50, 'n_iters': 75,
    #     'delta_FR': 0.1, 'delta_IMU': 0.5, 'mean_err_deg_s': 2.145}

Regenerate the table after running new grid sweeps (see update_best_config.py)::

    python update_best_config.py --update     # rescan all results/parameter_grid_*.csv
    python update_best_config.py --show       # print current table

The generated block below is rewritten in place by ``--update``; edit the code
outside the BEGIN/END markers freely.
"""

import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

# Hyper-parameter keys we carry over from a grid row into a config.
PARAM_KEYS = ('frame_duration', 'n_frames', 'n_iters', 'delta_FR', 'delta_IMU')

# BEGIN GENERATED  (python update_best_config.py --update)
# ===========================================================================
BEST_CONFIGS = {
    'dynamic_rotation': {
        'seg_A': {
            'cook':       {'frame_duration': 0.01, 'n_frames': 25, 'n_iters': 75, 'delta_FR': 0.1, 'delta_IMU': 0.0, 'mean_err_deg_s': 16.964},
            'thesis':     {'frame_duration': 0.01, 'n_frames': 25, 'n_iters': 75, 'delta_FR': 0.1, 'delta_IMU': 0.0, 'mean_err_deg_s': 40.071},
            'thesis_imu': {'frame_duration': 0.01, 'n_frames': 50, 'n_iters': 75, 'delta_FR': 0.1, 'delta_IMU': 0.5, 'mean_err_deg_s': 2.145},
        },
        'seg_B': {
            'cook':       {'frame_duration': 0.01, 'n_frames': 50, 'n_iters': 100, 'delta_FR': 0.5, 'delta_IMU': 0.0, 'mean_err_deg_s': 71.55},
            'thesis':     {'frame_duration': 0.01, 'n_frames': 50, 'n_iters': 100, 'delta_FR': 0.5, 'delta_IMU': 0.0, 'mean_err_deg_s': 70.753},
            'thesis_imu': {'frame_duration': 0.01, 'n_frames': 50, 'n_iters': 75, 'delta_FR': 0.1, 'delta_IMU': 0.5, 'mean_err_deg_s': 2.957},
        },
        'seg_C': {
            'cook':       {'frame_duration': 0.01, 'n_frames': 50, 'n_iters': 100, 'delta_FR': 0.5, 'delta_IMU': 0.0, 'mean_err_deg_s': 24.427},
            'thesis':     {'frame_duration': 0.01, 'n_frames': 50, 'n_iters': 75, 'delta_FR': 0.1, 'delta_IMU': 0.0, 'mean_err_deg_s': 32.978},
            'thesis_imu': {'frame_duration': 0.01, 'n_frames': 150, 'n_iters': 100, 'delta_FR': 0.1, 'delta_IMU': 0.5, 'mean_err_deg_s': 2.237},
        },
        'seg_D': {
            'cook':       {'frame_duration': 0.01, 'n_frames': 25, 'n_iters': 75, 'delta_FR': 0.5, 'delta_IMU': 0.0, 'mean_err_deg_s': 20.043},
            'thesis':     {'frame_duration': 0.01, 'n_frames': 25, 'n_iters': 100, 'delta_FR': 0.5, 'delta_IMU': 0.0, 'mean_err_deg_s': 19.993},
            'thesis_imu': {'frame_duration': 0.01, 'n_frames': 25, 'n_iters': 100, 'delta_FR': 0.1, 'delta_IMU': 0.5, 'mean_err_deg_s': 0.018},
        },
    },
}
# ===========================================================================
# END GENERATED


# ---------------------------------------------------------------------------
# Lookup API
# ---------------------------------------------------------------------------

def get_best_config(dataset, segment_id, model):
    """Return the best-params dict for a (dataset, segment_id, model), or None.

    The returned dict contains PARAM_KEYS plus 'mean_err_deg_s' (the grid error
    that this config achieved). Returns None if no grid entry exists.
    """
    try:
        return dict(BEST_CONFIGS[dataset][segment_id][model])
    except KeyError:
        return None


def best_params(dataset, segment_id, model):
    """Like get_best_config but only the tunable params (no 'mean_err_deg_s')."""
    cfg = get_best_config(dataset, segment_id, model)
    if cfg is None:
        return None
    return {k: cfg[k] for k in PARAM_KEYS if k in cfg}
