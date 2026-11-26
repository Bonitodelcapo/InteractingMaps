"""RunCache: all of a run in memory, disk cache with a stamp (Task 2).

Standalone script:  PYTHONPATH=. python tests/test_inspect_cache.py
Uses the tiny generated dataset of tests/_inspect_fake.py; needs nothing in data/.
"""
import os
import sys
import shutil

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _inspect_fake import make_run, make_dataset, NF
from interacting_maps import states_io, inspect_nav
from interacting_maps.inspect_io import load_run
from interacting_maps.inspect_cache import RunCache

ROOT = 'experiments/inspect_cache_test'
FAILED = []


def check(name, ok):
    print(f'[{"ok" if ok else "FAIL"}] {name}', flush=True)
    if not ok:
        FAILED.append(name)


shutil.rmtree(ROOT, ignore_errors=True)
DATA = make_dataset(os.path.join(ROOT, 'data'))
try:
    d = make_run(ROOT, 'thesis_imu', seed=2)
    rng = np.random.default_rng(5)
    os.makedirs(os.path.dirname(states_io.replay_path(d, 2)), exist_ok=True)
    rep = {'I': rng.standard_normal((10, 180, 240)), 'G': rng.standard_normal((10, 180, 240, 2)),
           'F': rng.standard_normal((10, 180, 240, 2)), 'R': rng.standard_normal((10, 3))}
    np.savez_compressed(states_io.replay_path(d, 2), **rep)

    seen = []
    c = RunCache.open(d, DATA, progress=seen.append)
    rv = load_run(d, DATA)
    check('maps equal the saved states (float32)',
          np.allclose(c.map_at('I', 1), rv.state(1)['I'], atol=1e-5)
          and c.maps['G'].dtype == np.float32 and c.maps['I'].shape == (NF, 180, 240))
    check('replay iteration', c.has_replay(2) and not c.has_replay(1)
          and np.allclose(c.map_at('I', 2, it=3), rep['I'][2], atol=1e-5))
    check('iteration past the end holds the last', np.allclose(c.map_at('I', 2, it=99), rep['I'][-1], atol=1e-5))
    check('progress reaches 1, monotone', seen and seen[-1] == 1.0 and seen == sorted(seen))
    check('raw V and APS in memory', c.V is not None and c.V.shape == (NF, 180, 240)
          and c.aps_at(0) is not None)
    check('per-frame scales for I, G, F, V', all(len(c.frame_scales[k]) == NF for k in 'IGFV'))

    ps = c.pixel_series(10, 20)
    ref = inspect_nav.pixel_series(rv, 10, 20)
    check('pixel series = inspect_nav', all(np.allclose(ps[k], ref[k], atol=1e-4) for k in ref))
    r1, r2 = c.iteration_residuals(2), inspect_nav.iteration_residuals(rv, 2)
    check('residuals = inspect_nav', all(np.allclose(r1[k], r2[k], rtol=1e-4, atol=1e-5) for k in r2))

    # second open uses the disk cache: the state files are not read
    orig = states_io.load_state
    states_io.load_state = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('read states'))
    try:
        c2 = RunCache.open(d, DATA)
        check('reopen reads the disk cache, not states/', np.allclose(c2.map_at('I', 1), c.map_at('I', 1)))
    except RuntimeError:
        check('reopen reads the disk cache, not states/', False)
    finally:
        states_io.load_state = orig

    # a re-made run (new states) must not reuse the old cache
    st, loop = states_io.load_state(d, 1)
    st['I'] = st['I'] + 100.0
    states_io.save_state(d, 1, st, loop)
    c3 = RunCache.open(d, DATA)
    check('stale cache rebuilt after new states', np.allclose(c3.map_at('I', 1), st['I'][:180, :240], atol=1e-3))
    # review fixes
    import interacting_maps.inspect_cache as ic
    calls = []
    orig_res = ic.residuals_of_stack
    ic.residuals_of_stack = lambda *a: calls.append(1) or orig_res(*a)
    c4 = RunCache.open(d, DATA)
    c4.iteration_residuals(2); c4.iteration_residuals(2)
    ic.residuals_of_stack = orig_res
    check('residuals computed once per frame (memoised)', len(calls) == 1)

    orig_w = RunCache._write_cache
    RunCache._write_cache = lambda self, *a: (_ for _ in ()).throw(OSError('read-only'))
    try:
        shutil.rmtree(os.path.join(d, ic.CACHE_DIR), ignore_errors=True)
        c5 = RunCache.open(d, DATA)
        check('read-only folder: loads without the disk cache', np.allclose(c5.map_at('I', 1), c3.map_at('I', 1)))
    except OSError:
        check('read-only folder: loads without the disk cache', False)
    finally:
        RunCache._write_cache = orig_w
    check('no disk cache under results/', not ic.cache_allowed('results/poster_rotation/cook/x')
          and ic.cache_allowed(d))
finally:
    shutil.rmtree(ROOT, ignore_errors=True)

sys.exit(1 if FAILED else 0)
