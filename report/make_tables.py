"""make_tables.py -- emit the LaTeX body of the main results table from csv.

    python report/make_tables.py                  # baseline configuration
    python report/make_tables.py --tag _curl0.6   # a second configuration

Reads the csv files written by `run_experiments.py --what main`, merging the
per-sequence pieces it may have been run in, and prints rows ready to paste
into tab:main. Generated rather than typed so the table cannot drift from the
runs behind it, which it had done once already.
"""
import argparse
import csv
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, 'report', 'experiments')

# display name, csv model key
MODELS = [(r'\texttt{cook}', 'cook'),
          (r'\texttt{thesis}', 'thesis'),
          (r'\texttt{thesis+IMU}', 'thesis_imu'),
          (r'\texttt{CMax-anchor}', 'thesis_cmax'),
          (r'\texttt{CMax-inloop}', 'thesis_cmax_v2')]
# csv sequence key, display label, extra lines under the label
SEQS = [('boxes_rotation', r'\texttt{boxes}', []),
        ('poster_rotation', r'\texttt{poster}', []),
        ('dynamic_rotation', r'\texttt{dynamic}', ['(person)']),
        ('bycicle_sinthetic', r'\texttt{bicycle}', ['(synthetic)']),
        ('street_sinthetic', r'\texttt{street}', ['(synthetic)'])]


def load(patterns):
    """All rows from every csv matching the patterns; later files win."""
    rows = {}
    for pat in patterns:
        for path in sorted(glob.glob(os.path.join(OUTDIR, pat))):
            for r in csv.DictReader(open(path)):
                rows[(r['sequence'], r['model'])] = r
    return rows


def fmt(v, nd=1):
    try:
        return f'{float(v):.{nd}f}'
    except (TypeError, ValueError):
        return '---'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='', help="e.g. '_curl0.6'")
    ap.add_argument('--speeds', default='',
                    help='optional "seq=0.88,seq=0.91" rad/s labels')
    args = ap.parse_args()

    pats = [f'main_dt20ms{args.tag}.csv', f'main_*{args.tag.strip("_") or "base"}*.csv']
    if not args.tag:
        pats = ['main_dt20ms.csv', 'main_base_*.csv']
    rows = load(pats)
    if not rows:
        sys.exit(f'no csv found for tag {args.tag!r} in {OUTDIR}')

    speeds = dict(p.split('=') for p in args.speeds.split(',') if '=' in p)
    missing = []
    for ds, label, extra in SEQS:
        head = [label] + ([f'${speeds[ds]}$\\,rad/s'] if ds in speeds else []) + extra
        print(r'\midrule')
        print(r'\multirow{6}{*}{\shortstack[l]{' + r'\\'.join(head) + '}}')
        floor = None
        for disp, key in MODELS:
            r = rows.get((ds, key))
            if r is None:
                missing.append(f'{ds}/{key}')
                print(f' & {disp:<22}& --- & --- & --- & --- \\\\')
                continue
            floor = (r.get('floor'), r.get('floor_dir'))
            print(f' & {disp:<22}& {fmt(r["err"])} & {fmt(r["dir"])} & '
                  f'{fmt(r["beta"], 2)} & {fmt(r.get("recon"), 2)} \\\\')
        if floor and floor[0] is not None:
            print(f' & \\emph{{gyro vs Vicon}}  & \\emph{{{fmt(floor[0])}}} & '
                  f'\\emph{{{fmt(floor[1])}}} & --- & --- \\\\')
    if missing:
        print('\n% MISSING RUNS: ' + ', '.join(missing), file=sys.stderr)
        print(f'% {len(missing)} cell(s) incomplete', file=sys.stderr)


if __name__ == '__main__':
    main()
