"""
backfill_run_figures.py — recon_final.png + recon_plot.png for tracking runs
written before experiment_tracking drew them (viz.run_figures).

Reads each run folder's maps_final.npz, tracking.csv and run.json and writes the
two figures into that same folder, next to tracking_plot.png. Plotting only: no
run is repeated and no existing file is changed. Run from the repo root:

    python backfill_run_figures.py                      # results/ and experiments/
    python backfill_run_figures.py experiments/sens --force
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import numpy as np

from config import get_dataset_paths
from experiment_tracking import read_config
from interacting_maps.eval_io import load_aps_list
from interacting_maps.viz import run_figures


def run_dirs(roots):
    """Every folder under roots that holds a tracking run's final maps and csv."""
    for root in roots:
        for d, _, files in os.walk(root):
            if 'maps_final.npz' in files and 'tracking.csv' in files:
                yield d


def backfill(d):
    """Draw the two figures of the run in folder d from what it saved."""
    cfg = read_config(d)
    with open(os.path.join(d, 'tracking.csv'), newline='') as f:
        rows = list(csv.DictReader(f))
    t = float(rows[-1]['time'])
    aps = load_aps_list(get_dataset_paths(cfg['dataset'])['data_dir'], t - 0.5, t + 0.5)
    run_figures(d, dict(np.load(os.path.join(d, 'maps_final.npz'))), rows, aps, cfg)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('roots', nargs='*', default=['results', 'experiments'])
    ap.add_argument('--force', action='store_true', help='redraw where recon_final.png exists')
    args = ap.parse_args()
    n = {'ok': 0, 'skip': 0, 'FAIL': 0}
    for d in run_dirs(args.roots):
        if not args.force and os.path.exists(os.path.join(d, 'recon_final.png')):
            n['skip'] += 1
            continue
        try:
            backfill(d)
            n['ok'] += 1
            print(f'[ok] {d}', flush=True)
        except Exception as e:                      # one bad folder must not stop the rest
            n['FAIL'] += 1
            print(f'[FAIL] {d}: {type(e).__name__}: {e}', flush=True)
    print(f"\n{n['ok']} drawn, {n['skip']} already had figures, {n['FAIL']} failed")


if __name__ == '__main__':
    main()
