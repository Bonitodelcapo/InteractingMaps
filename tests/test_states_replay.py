"""Per-frame states and the iteration replay (inspection viewer, Tasks 3-4).

Runs every model for 3 frames x 10 iterations on poster_rotation seg_C with
save_states on (into experiments/states_test, deleted at the end), then checks
that frame k replayed from the saved state k-1 is bit-identical to state k.

Standalone script:  PYTHONPATH=. python tests/test_states_replay.py
Skips (exit 0) if data/poster_rotation is absent.
"""
import os
import sys
import shutil

import numpy as np

os.environ.setdefault('IM_MLFLOW', '0')

if not os.path.isdir('data/poster_rotation'):
    print('[skip] data/poster_rotation not present')
    sys.exit(0)

import evaluation as E
from run_config import RunConfig, make_network, rc_from_run
from interacting_maps.data_loader import EventFrameSequence
from interacting_maps.frame_driver import FrameDriver
from interacting_maps import states_io

OUT = 'experiments/states_test'
MODELS = ['cook', 'thesis', 'thesis_imu', 'thesis_cmax', 'thesis_cmax_v2']
KEYS = ('I', 'G', 'F', 'R')
FAILED = []


def check(name, ok):
    print(f'[{"ok" if ok else "FAIL"}] {name}', flush=True)
    if not ok:
        FAILED.append(name)


def make_rc(model, save, n_iters=10):
    rc = RunConfig(dataset='poster_rotation', model=model, segment='seg_C',
                   n_frames=3, n_iters=n_iters, out_root=OUT)
    if save:
        rc.save_states = True
    return rc


def rebuild(run_dir):
    """Network, driver and the frames' V of a saved run, from run.json alone."""
    rc = rc_from_run(run_dir)
    seq = EventFrameSequence(rc.paths['events'], rc.paths['calib'],
                             frame_duration=rc.frame_duration, t_start=rc.t_start,
                             n_frames=rc.n_frames, clip_value=10.0,
                             undistort=rc.undistort_at_event_level,
                             sensor_size=rc.sensor_size)
    c = seq.calib
    net = make_network(rc, seq.H, seq.W, c.fx, c.fy, c.cx, c.cy)
    return rc, net, FrameDriver(rc, net, seq.H, seq.W, c), [V for V, _ in seq]


def same(a, b):
    return all(np.array_equal(a[k], b[k]) for k in KEYS)


shutil.rmtree(OUT, ignore_errors=True)
try:
    for model in MODELS:
        rc = make_rc(model, save=True)
        E.experiment_tracking(rc, save_frames=False)
        d = rc.output_dir
        check(f'{model} files', os.path.exists(states_io.state_path(d, -1)) and all(
            os.path.exists(states_io.state_path(d, k)) for k in range(3)))

        rc2, net, drv, Vs = rebuild(d)
        ok = True
        for k in (1, 2):
            st, loop = states_io.load_state(d, k - 1)
            net.set_state(st)
            drv.set_loop_state(loop)
            drv.step(k, Vs[k], rc2.n_iters)
            ok &= same(net.get_state(), states_io.load_state(d, k)[0])
        check(f'{model} exact replay of frames 1, 2', ok)

    rc = make_rc('cook', save=False, n_iters=9)     # own folder, not the run above
    E.experiment_tracking(rc, save_frames=False)
    check('off by default: no states/', not os.path.exists(states_io.states_dir(rc.output_dir)))
    check('save_states not in to_dict', 'save_states' not in make_rc('cook', True).to_dict())

    try:
        states_io.assert_not_results('results/poster_rotation/cook/x')
        check('assert_not_results refuses results/', False)
    except ValueError:
        check('assert_not_results refuses results/', True)

    try:
        states_io.load_state(rc.output_dir, 0)
        check('missing states: FileNotFoundError', False)
    except FileNotFoundError as e:
        check('missing states: FileNotFoundError names save_states', 'save_states' in str(e))
finally:
    shutil.rmtree(OUT, ignore_errors=True)

sys.exit(1 if FAILED else 0)
