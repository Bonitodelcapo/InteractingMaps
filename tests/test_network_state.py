"""get_state / set_state on both networks (inspection viewer, Task 1).

Standalone script:  PYTHONPATH=. python tests/test_network_state.py
Skips (exit 0) if data/poster_rotation is absent (RunConfig reads its calib/IMU).
"""
import os
import sys

import numpy as np

os.environ.setdefault('IM_MLFLOW', '0')

if not os.path.isdir('data/poster_rotation'):
    print('[skip] data/poster_rotation not present')
    sys.exit(0)

from run_config import RunConfig, make_network

H, W = 20, 30
KEYS = ('I', 'G', 'F', 'R')
FAILED = []


def check(name, ok):
    print(f'[{"ok" if ok else "FAIL"}] {name}')
    if not ok:
        FAILED.append(name)


def build(model):
    rc = RunConfig(dataset='poster_rotation', model=model, segment='seg_C', n_frames=2)
    return rc, make_network(rc, H, W, 200.0, 200.0, W / 2, H / 2)


def kwargs(rc):
    return {'omega_imu': np.array([0.1, -0.2, 0.3])} if rc.use_imu else {}


rng = np.random.default_rng(0)
V1 = rng.standard_normal((H, W)) * 0.1
V2 = rng.standard_normal((H, W)) * 0.1

for model in ('cook', 'thesis_imu'):
    rc, a = build(model)
    a.step(V1, n_iters=3, **kwargs(rc))
    s = a.get_state()
    _, b = build(model)
    b.set_state(s)
    check(f'{model} roundtrip', all(np.array_equal(b.get_state()[k], s[k]) for k in KEYS))

    s['I'][...] = 123.0
    check(f'{model} get_state copies', not np.any(a.I == 123.0))

    s1 = a.get_state()
    a.step(V2, n_iters=3, **kwargs(rc))
    _, c = build(model)
    c.set_state(s1)
    c.step(V2, n_iters=3, **kwargs(rc))
    check(f'{model} step after set_state equal',
          all(np.array_equal(a.get_state()[k], c.get_state()[k]) for k in KEYS))

    try:
        c.set_state({'I': s1['I']})
        check(f'{model} missing key raises', False)
    except KeyError:
        check(f'{model} missing key raises', True)

sys.exit(1 if FAILED else 0)
