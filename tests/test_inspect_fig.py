"""The inspector figure and export (Task 7), on fake run folders.

Standalone script:  PYTHONPATH=. python tests/test_inspect_fig.py
Uses a tiny generated dataset (tests/_inspect_fake.py); needs nothing in data/.
"""
import os
import sys
import shutil

import numpy as np
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(__file__))
import imageio.v2 as imageio
from _inspect_fake import make_run, make_dataset
from interacting_maps.inspect_io import load_run
from interacting_maps import states_io
from interacting_maps.inspect_fig import (fixed_scales, draw_inspector, export_animation,
                                          frame_from_click)

ROOT = 'experiments/inspect_fig_test'
FAILED = []


def check(name, ok):
    print(f'[{"ok" if ok else "FAIL"}] {name}', flush=True)
    if not ok:
        FAILED.append(name)


def role(fig, r):
    return [a for a in fig.axes if getattr(a, '_inspect_role', None) == r]


shutil.rmtree(ROOT, ignore_errors=True)
DATA = make_dataset(os.path.join(ROOT, 'data'))
try:
    runs = [load_run(data_root=DATA, run_dir=make_run(ROOT, 'cook', scale=1.0, seed=1)),
            load_run(data_root=DATA, run_dir=make_run(ROOT, 'thesis_imu', scale=4.0, seed=2))]

    fig = draw_inspector(runs, 1, panels=('I', 'G'), curves=('err_deg_s',))
    check('layout: 2 runs x 2 map panels', len(role(fig, 'map')) == 4)
    check('layout: raw row V + APS', len(role(fig, 'raw')) == 2)
    curve = role(fig, 'curve')
    check('layout: one curve axis', len(curve) == 1)
    xs = [l.get_xdata()[0] for l in curve[0].lines if len(set(np.ravel(l.get_xdata()))) == 1]
    check('curve marker at times[1]', any(np.isclose(x, runs[0].times[1]) for x in xs))
    check('curve axis tagged with its name', getattr(curve[0], '_inspect_curve', '') == 'err_deg_s')

    sc = fixed_scales(runs, ['I', 'G'])
    check('scales per model differ (x4 maps)', sc[0]['G'] < sc[1]['G'] / 2)
    sh = fixed_scales(runs, ['I', 'G'], share=True)
    check('share=True: equal scales', sh[0] == sh[1])

    # a run that blew up: NaN in G of frame 1 and in its error curve
    d = make_run(ROOT, 'thesis', scale=1.0, seed=3)
    st, loop = states_io.load_state(d, 1)
    st['G'][:10] = np.nan
    states_io.save_state(d, 1, st, loop)
    bad = load_run(d, DATA)
    bad.rows['err_deg_s'][1] = np.nan
    sb = fixed_scales([bad], ['G'])[0]
    check('NaN run: scale finite', np.isfinite(sb['G']))
    try:
        draw_inspector([bad], 1, curves=('err_deg_s',))
        check('NaN run: figure draws', True)
    except Exception as e:
        check(f'NaN run: figure draws ({e!r})', False)

    nope = load_run(data_root=DATA, run_dir=make_run(ROOT, 'cook', dataset='nope'))
    f2 = draw_inspector([nope], 0, curves=('err_deg_s', 'omega_fullrate'))
    check('missing dataset: raw axes n/a', all(a.get_title() == 'n/a' for a in role(f2, 'raw')))
    full = [a for a in role(f2, 'curve') if a._inspect_curve == 'omega_fullrate']
    check('missing dataset: omega_fullrate axis n/a', full and full[0].get_title() == 'n/a')

    gif = export_animation(runs, os.path.join(ROOT, 'a.gif'), frames=[0, 1, 2],
                           panels=('I',), curves=('err_deg_s',))
    check('export: GIF with 3 frames', len(imageio.mimread(gif)) == 3)

    check('click -> nearest frame', frame_from_click(0.031, np.array([0.01, 0.03, 0.05])) == 1)
finally:
    shutil.rmtree(ROOT, ignore_errors=True)

sys.exit(1 if FAILED else 0)
