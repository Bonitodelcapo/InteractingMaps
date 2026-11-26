"""Inspection app (Qt): up to 10 runs of one segment, linked to their curves.

    python inspect_app.py                                  # pick runs in the Runs tab
    python inspect_app.py --root D:/cluster_runs/inspect   # scan another folder
    python inspect_app.py <run dir> <run dir> ...          # open runs directly

Runs come from `python report/run_experiments.py --what inspect` (cluster).
Guide: docs/inspection_viewer.md
"""
import argparse
import sys

from interacting_maps.inspect_app import InspectApp, qt_app


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='experiments/inspect', help='folder with inspect runs')
    ap.add_argument('--data', default=None, help='folder holding <dataset>/ (default ./data)')
    ap.add_argument('runs', nargs='*', help='run folders to open directly')
    args = ap.parse_args(argv)
    qa = qt_app()
    win = InspectApp(root=args.root, data_root=args.data)
    win.runs_panel.scan()
    if args.runs:
        win.runs_panel.load_dirs(args.runs)
    win.show()
    return qa.exec()


if __name__ == '__main__':
    sys.exit(main())
