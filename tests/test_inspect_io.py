"""Run loader of the inspection viewer (Task 6), on fake run folders.

Standalone script:  PYTHONPATH=. python tests/test_inspect_io.py
Uses a tiny generated dataset (tests/_inspect_fake.py); needs nothing in data/.
"""
import os
import sys
import json
import shutil

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _inspect_fake import make_run, make_dataset, H, W, NF
from interacting_maps.inspect_io import load_run, check_comparable
from interacting_maps import states_io

ROOT = 'experiments/inspect_io_test'
FAILED = []


def check(name, ok):
    print(f'[{"ok" if ok else "FAIL"}] {name}', flush=True)
    if not ok:
        FAILED.append(name)


def raises(exc, fn, text=''):
    try:
        fn()
    except exc as e:
        return text in str(e)
    return False


shutil.rmtree(ROOT, ignore_errors=True)
DATA = make_dataset(os.path.join(ROOT, 'data'))
try:
    rv = load_run(data_root=DATA, run_dir=make_run(ROOT, 'cook'))
    check('rows: err_deg_s has one value per frame', rv.rows['err_deg_s'].shape == (NF,))
    check('rows: text column ref_source dropped', 'ref_source' not in rv.rows)
    check('times: seconds since segment start', np.allclose(rv.times, [0.01, 0.03, 0.05]))
    check('state cropped to H x W (cook I is H+1 x W+1)', rv.state(1)['I'].shape == (H, W))
    check('no replay -> iterations None', rv.iterations(1) is None)
    raw = rv.raw(1)
    check('raw V is the event frame', raw['V'] is not None and raw['V'].shape == (H, W))
    check('n_events curve finite', np.isfinite(rv.curves()['n_events']).all())

    d2 = make_run(ROOT, 'thesis_imu')
    shutil.rmtree(states_io.states_dir(d2))
    check('no states -> FileNotFoundError naming save_states',
          raises(FileNotFoundError, lambda: load_run(d2, DATA), 'save_states'))

    other = load_run(data_root=DATA, run_dir=make_run(ROOT, 'thesis', t_start=8.836))
    check('different t_start -> ValueError naming t_start',
          raises(ValueError, lambda: check_comparable([rv, other]), 't_start'))
    check('six runs -> ValueError', raises(ValueError, lambda: check_comparable([rv] * 6)))
    check('same segment passes', check_comparable([rv, rv]) is None)

    d3 = make_run(ROOT, 'cook', dataset='nope')
    rn = load_run(d3, DATA)
    check('missing dataset: raw V None', rn.raw(0)['V'] is None and rn.raw(0)['aps'] is None)
    check('missing dataset: n_events all NaN', np.isnan(rn.curves()['n_events']).all())
    check('missing dataset: omega_fullrate empty', rn.omega_fullrate() == {})
finally:
    shutil.rmtree(ROOT, ignore_errors=True)

sys.exit(1 if FAILED else 0)
