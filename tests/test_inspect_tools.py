"""Viewer tools (run picker, navigation, scale modes, pixel probe, iteration
residuals, export panel), headless, on fake runs and a tiny generated dataset.

Standalone script:  PYTHONPATH=. python tests/test_inspect_tools.py
Needs nothing in data/.
"""
import os
import sys
import shutil
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(__file__))
from _inspect_fake import make_run, make_dataset, NF
from interacting_maps import states_io
from interacting_maps.inspect_io import scan_runs, load_run
from interacting_maps.inspect_nav import (worst_frame, next_spike, pixel_series,
                                          pixel_iterations, iteration_residuals)
from inspect_viewer import Inspector

ROOT = 'experiments/inspect_tools_test'
FAILED = []


def check(name, ok):
    print(f'[{"ok" if ok else "FAIL"}] {name}', flush=True)
    if not ok:
        FAILED.append(name)


def fake_replay(d, k, n_it=10, seed=5):
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.dirname(states_io.replay_path(d, k)), exist_ok=True)
    np.savez_compressed(states_io.replay_path(d, k), I=rng.standard_normal((n_it, 180, 240)),
                        G=rng.standard_normal((n_it, 180, 240, 2)),
                        F=rng.standard_normal((n_it, 180, 240, 2)), R=rng.standard_normal((n_it, 3)))


shutil.rmtree(ROOT, ignore_errors=True)
DATA = make_dataset(os.path.join(ROOT, 'data'))
try:
    d_cook = make_run(ROOT, 'cook', seed=1)
    d_imu = make_run(ROOT, 'thesis_imu', seed=2)
    d_other = make_run(ROOT, 'thesis', t_start=1.02, seed=3)     # another segment
    fake_replay(d_imu, 2)

    # ---- catalog
    cat = scan_runs(ROOT)
    check('scan finds the three runs', len(cat) == 3)
    e = [c for c in cat if c['dir'] == d_imu][0]
    check('entry knows model, dataset, replays', (e['model'], e['dataset'], e['replays'])
          == ('thesis_imu', 'tinyset', [2]))
    check('two segments told apart', len({c['segment'] for c in cat}) == 2)

    # ---- navigation helpers
    y = np.array([1.0, 5.0, np.nan, 2.0, 9.0, 1.0])
    check('worst frame ignores NaN', worst_frame(y) == 4)
    check('next spike forward', next_spike(y, 1, 4.0, +1) == 4)
    check('next spike backward', next_spike(y, 4, 4.0, -1) == 1)
    check('no spike -> None', next_spike(y, 4, 4.0, +1) is None)

    # ---- pixel probe and iteration residuals
    rv = load_run(d_imu, DATA)
    ps = pixel_series(rv, 10, 20)
    check('pixel series over frames', set(ps) == {'I', '|G|', '|F|'} and len(ps['I']) == NF
          and np.isclose(ps['I'][1], rv.state(1)['I'][20, 10]))
    pi = pixel_iterations(rv, 2, 10, 20)
    check('pixel series over iterations', pi is not None and len(pi['I']) == 10)
    check('no replay -> pixel_iterations None', pixel_iterations(rv, 1, 10, 20) is None)
    res = iteration_residuals(rv, 2)
    check('iteration residuals per iteration',
          res is not None and all(len(v) == 10 for v in res.values())
          and {'|V+F·G|', '|G−∇I|', '|R|'} <= set(res))
    check('no replay -> residuals None', iteration_residuals(rv, 1) is None)

    # ---- viewer: pick runs with the cascading dropdowns
    v = Inspector(root=ROOT, data_root=DATA)
    check('dataset dropdown filled', v.ds.options == ('tinyset',))
    check('segment dropdown has both segments', len(v.seg.options) == 2)
    v.seg.value = [s for s in v.seg.options if 't0 1.000' in s][0]
    check('models of the chosen segment only', set(v.models.options) == {'cook', 'thesis_imu'})
    v.models.value = ('cook', 'thesis_imu')
    v.load()
    check('load opens the runs', [r.label for r in v.runs] == ['cook', 'thesis_imu'])
    check('info card lists models and replays', 'thesis_imu' in v.info.value and 'replays' in v.info.value)
    check('play spans the frames', (v.play.min, v.play.max) == (0, NF - 1))

    # ---- navigation in the viewer
    v.frame.value = 0
    v.step(+1)
    check('step forward', v.frame.value == 1)
    v.step(-5)
    check('step clamps at 0', v.frame.value == 0)
    v.spike_curve.value = 'err_deg_s'          # 5, 6, 7 -> worst is the last frame
    v.goto_worst()
    check('jump to worst frame', v.frame.value == NF - 1)
    v.threshold.value = 5.5
    v.frame.value = 0
    v.goto_spike(+1)
    check('jump to next spike above threshold', v.frame.value == 1)

    # ---- scale modes
    v.scale_mode.value = 'shared'
    s = v.scales()
    check('shared scale equal across runs', s[0] == s[1])
    v.scale_mode.value = 'per frame'
    check('per-frame scale = no fixed scale', all(sc == {} for sc in v.scales()))

    # ---- pixel probe: click a map pixel
    ax = [a for a in v.fig.axes if getattr(a, '_inspect_role', None) == 'map'][0]
    v.on_click(SimpleNamespace(inaxes=ax, xdata=10.4, ydata=20.2))
    check('click on a map sets the probe pixel', v.probe == (10, 20))
    check('probe drawn in the side figure', any(getattr(a, '_inspect_aux', '') == 'probe'
                                                for a in v.aux.axes))
    v.frame.value = 2
    check('residuals drawn at a replayed frame', any(getattr(a, '_inspect_aux', '') == 'residuals'
                                                    for a in v.aux.axes))

    # ---- export panel
    v.fmt.value = 'png'
    p = v.export_current()
    check('PNG of the current view', p.endswith('.png') and os.path.exists(p))
    v.fmt.value = 'gif'
    v.frames_range.value = (0, 1)
    p = v.export_frames()
    check('GIF over a frame range', os.path.exists(p))
    check('auto file name names segment and frames', 'f000-001' in os.path.basename(p))
finally:
    shutil.rmtree(ROOT, ignore_errors=True)

sys.exit(1 if FAILED else 0)
