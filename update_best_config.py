"""
update_best_config.py — Rescan parameter-grid sweeps and rewrite best_config.py.

For each (dataset, segment_id, model), keeps the results/parameter_grid_*.csv
row with the lowest mean_err_deg_s and writes it into the BEST_CONFIGS literal
in best_config.py (the BEGIN/END GENERATED region), which evaluation.py reads
via best_config.get_best_config / best_params.

Usage:
    python update_best_config.py --update     # rescan all results/parameter_grid_*.csv
    python update_best_config.py --show       # print current table
"""

import os
import re
import csv
import glob
import math
import argparse

from best_config import RESULTS_DIR, BEST_CONFIGS

_BEST_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'best_config.py')
_BEGIN = '# BEGIN GENERATED'
_END = '# END GENERATED'


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


def update_file(configs, path=_BEST_CONFIG_PATH):
    """Rewrite the BEST_CONFIGS literal in best_config.py's BEGIN..END region."""
    with open(path, encoding='utf-8') as f:
        src = f.read()
    banner = '# ' + '=' * 75
    replacement = (
        _BEGIN + '  (python update_best_config.py --update)\n'
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


def main():
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


if __name__ == '__main__':
    main()
