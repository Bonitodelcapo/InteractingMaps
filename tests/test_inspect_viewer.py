"""The notebook viewer's logic, headless (Task 8): widgets exist without a
browser, so the slider, the click-to-frame jump and export can be driven here.

Standalone script:  PYTHONPATH=. python tests/test_inspect_viewer.py
Uses the tiny generated dataset of tests/_inspect_fake.py; needs nothing in data/.
"""
import os
import sys
import shutil
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(__file__))
from _inspect_fake import make_run, make_dataset
from interacting_maps import states_io
from inspect_viewer import Inspector

ROOT = 'experiments/inspect_viewer_test'
FAILED = []


def check(name, ok):
    print(f'[{"ok" if ok else "FAIL"}] {name}', flush=True)
    if not ok:
        FAILED.append(name)


shutil.rmtree(ROOT, ignore_errors=True)
DATA = make_dataset(os.path.join(ROOT, 'data'))
try:
    d1 = make_run(ROOT, 'cook', seed=1)
    d2 = make_run(ROOT, 'thesis_imu', seed=2)
    # a replay of frame 2 for the second run only
    rng = np.random.default_rng(5)
    os.makedirs(os.path.dirname(states_io.replay_path(d2, 2)), exist_ok=True)
    np.savez_compressed(states_io.replay_path(d2, 2), I=rng.standard_normal((10, 180, 240)),
                        G=rng.standard_normal((10, 180, 240, 2)),
                        F=rng.standard_normal((10, 180, 240, 2)), R=np.zeros((10, 3)))

    v = Inspector([d1, d2], data_root=DATA)
    check('frame slider spans the run', (v.frame.min, v.frame.max) == (0, 2))
    check('iteration slider off at frame 0 (no replay)', v.iteration.disabled)
    v.frame.value = 2
    check('iteration slider on at frame 2 (replay exists)', not v.iteration.disabled)
    check('figure title follows the frame', 'frame 2' in v.fig._suptitle.get_text())

    curve = [a for a in v.fig.axes if getattr(a, '_inspect_curve', None)][0]
    v.on_click(SimpleNamespace(inaxes=curve, xdata=0.031))
    check('click on a curve jumps to the nearest frame', v.frame.value == 1)
    v.on_click(SimpleNamespace(inaxes=None, xdata=None))
    check('click outside curves ignored', v.frame.value == 1)

    v.curves.value = ('err_deg_s', 'n_events')
    n_curve = len([a for a in v.fig.axes if getattr(a, '_inspect_curve', None)])
    check('curve dropdown redraws', n_curve == 2)

    path = v.export(os.path.join(ROOT, 'frames.gif'))
    check('export over frames writes a file', os.path.exists(path))

    try:
        Inspector([d1, make_run(ROOT, 'thesis', t_start=1.02)], data_root=DATA)
        check('mismatched runs refused', False)
    except ValueError:
        check('mismatched runs refused', True)
finally:
    shutil.rmtree(ROOT, ignore_errors=True)

sys.exit(1 if FAILED else 0)
