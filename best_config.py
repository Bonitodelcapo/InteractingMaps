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

Regenerate the table after running new grid sweeps::

    python best_config.py --update            # rescan all results/parameter_grid_*.csv
    python best_config.py --show              # print current table

The generated block below is rewritten in place by ``--update``; edit the code
outside the BEGIN/END markers freely.
"""

import os
import re
import csv
import glob
import math

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

# Hyper-parameter keys we carry over from a grid row into a config.
PARAM_KEYS = ('frame_duration', 'n_frames', 'n_iters', 'delta_FR', 'delta_IMU')

# BEGIN GENERATED  (python best_config.py --update)
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


# ---------------------------------------------------------------------------
# Derivation from grid CSVs
# ---------------------------------------------------------------------------

def derive_from_grids(results_dir=RESULTS_DIR, objective='mean_err_deg_s'):
    """Scan results/parameter_grid_*.csv and return (nested best dict, csv_paths).

    For each (dataset, segment_id, model) keep the row minimizing `objective`.
    The dataset name is taken from the CSV's 'dataset' column (not the filename,
    which may be misspelled).
    """
    best = {}  # (dataset, seg, model) -> (score, row)
    csv_paths = sorted(glob.glob(os.path.join(results_dir, 'parameter_grid_*.csv')))
    for path in csv_paths:
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                try:
                    score = float(row[objective])
                except (KeyError, ValueError, TypeError):
                    continue
                if math.isnan(score):
                    continue
                key = (row['dataset'], row['segment_id'], row['model'])
                if key not in best or score < best[key][0]:
                    best[key] = (score, row)

    out = {}
    for (ds, seg, model), (score, row) in best.items():
        entry = {
            'frame_duration': float(row['frame_duration']),
            'n_frames': int(float(row['n_frames'])),
            'n_iters': int(float(row['n_iters'])),
            'delta_FR': float(row['delta_FR']),
            'delta_IMU': float(row['delta_IMU']),
            'mean_err_deg_s': round(float(row['mean_err_deg_s']), 3),
        }
        out.setdefault(ds, {}).setdefault(seg, {})[model] = entry
    return out, csv_paths


def _format_block(configs):
    """Render a nested best dict as the BEST_CONFIGS python literal."""
    lines = ['BEST_CONFIGS = {']
    for ds in sorted(configs):
        lines.append(f"    {ds!r}: {{")
        for seg in sorted(configs[ds]):
            lines.append(f"        {seg!r}: {{")
            for model in sorted(configs[ds][seg]):
                e = configs[ds][seg][model]
                key = f"{model!r}:"
                lines.append(
                    f"            {key:<13} "
                    f"{{'frame_duration': {e['frame_duration']}, "
                    f"'n_frames': {e['n_frames']}, "
                    f"'n_iters': {e['n_iters']}, "
                    f"'delta_FR': {e['delta_FR']}, "
                    f"'delta_IMU': {e['delta_IMU']}, "
                    f"'mean_err_deg_s': {e['mean_err_deg_s']}}},"
                )
            lines.append("        },")
        lines.append("    },")
    lines.append("}")
    return '\n'.join(lines)


_BEGIN = '# BEGIN GENERATED'
_END = '# END GENERATED'


def update_file(configs, path=__file__):
    """Rewrite the BEST_CONFIGS literal in the whole BEGIN..END region in place."""
    with open(path, encoding='utf-8') as f:
        src = f.read()
    banner = '# ' + '=' * 75
    replacement = (
        _BEGIN + '  (python best_config.py --update)\n'
        + banner + '\n'
        + _format_block(configs) + '\n'
        + banner + '\n'
        + _END
    )
    # Anchor markers to column 0 so the _BEGIN/_END string literals in this
    # module's own source don't count as a second region.
    pattern = re.compile('^' + re.escape(_BEGIN) + r'.*?^' + re.escape(_END),
                         re.DOTALL | re.MULTILINE)
    new_src, n = pattern.subn(lambda m: replacement, src)
    if n != 1:
        raise RuntimeError(
            f"Expected exactly one BEGIN..END GENERATED region in {path}, found {n}."
        )
    with open(path, 'w', encoding='utf-8') as f:
        f.write(new_src)


def print_table(configs=None):
    configs = configs if configs is not None else BEST_CONFIGS
    print(f"{'dataset':<18} {'seg':<6} {'model':<12} {'dt_ms':>5} {'n_fr':>5} "
          f"{'iters':>5} {'dFR':>5} {'dIMU':>5} {'err/s':>7}")
    print('-' * 74)
    for ds in sorted(configs):
        for seg in sorted(configs[ds]):
            for model in sorted(configs[ds][seg]):
                e = configs[ds][seg][model]
                print(f"{ds:<18} {seg:<6} {model:<12} "
                      f"{e['frame_duration']*1000:>5.0f} {e['n_frames']:>5} "
                      f"{e['n_iters']:>5} {e['delta_FR']:>5.2f} {e['delta_IMU']:>5.2f} "
                      f"{e.get('mean_err_deg_s', float('nan')):>7.2f}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Best-config table utilities')
    ap.add_argument('--update', action='store_true',
                    help='Rescan results/parameter_grid_*.csv and rewrite BEST_CONFIGS')
    ap.add_argument('--show', action='store_true', help='Print the current table')
    args = ap.parse_args()

    if args.update:
        configs, csv_paths = derive_from_grids()
        if not configs:
            print("No parameter_grid_*.csv files found under results/ — nothing to do.")
        else:
            update_file(configs)
            print(f"Updated BEST_CONFIGS from {len(csv_paths)} CSV(s):")
            for p in csv_paths:
                print(f"  - {os.path.relpath(p)}")
            print()
            print_table(configs)
    elif args.show:
        print_table()
    else:
        ap.print_help()
