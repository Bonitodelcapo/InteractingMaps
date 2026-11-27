"""Checks of the run comparison tool (interacting_maps/compare_runs), on synthetic
run folders in a temp dir -- no real runs needed.

    PYTHONPATH=. python tests/test_compare_runs.py
"""
import json
import os
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use('Agg')

FAILURES = []


def check(name, ok, detail=''):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ''))
    if not ok:
        FAILURES.append(name)


def make_run(root, rel, config, summary=None, legacy=False):
    """A run folder root/rel with run.json (or legacy params.json + summary.json)."""
    d = os.path.normpath(os.path.join(root, rel))
    os.makedirs(d, exist_ok=True)
    summary = summary or {}
    if legacy:
        json.dump(config, open(os.path.join(d, 'params.json'), 'w'))
        json.dump(summary, open(os.path.join(d, 'summary.json'), 'w'))
    else:
        json.dump({'provenance': {'git_commit_short': 'abc1234', 'git_dirty': False},
                   'config': config, 'summary': summary},
                  open(os.path.join(d, 'run.json'), 'w'))
    return d


def cfg(model='thesis', dataset='boxes_rotation', seg='seg_A', dt=0.02, n=4, **params):
    c = dict(model=model, dataset=dataset, segment_id=seg, t_start=1.0, frame_duration=dt,
             n_frames=n, n_iters=75, poisson='iterative', distortion_mode='C_full',
             params=params or {'delta_FR': 0.3})
    if seg is None:
        del c['segment_id']
    return c


def three_runs(root):
    a = make_run(root, 'boxes_rotation/thesis/seg_A_dt20', cfg(dt=0.02), {'mean_err_deg_s': 5.0})
    b = make_run(root, 'boxes_rotation/thesis/seg_A_dt10', cfg(dt=0.01), {'mean_err_deg_s': 3.0})
    legacy = cfg(model='cook', seg=None, dt=0.02, delta_IG=0.1)
    del legacy['poisson']
    c = make_run(root, 'boxes_rotation/cook/seg_B_t7.3_dt20ms_n4', legacy,
                 {'mean_err_deg_s': 9.0}, legacy=True)
    return a, b, c


# ---------------------------------------------------------------- Task 1: catalog
def test_catalog():
    from interacting_maps.compare_runs import catalog
    print('catalog')
    with tempfile.TemporaryDirectory() as root:
        a, b, c = three_runs(root)
        cat = catalog.scan([root])
        check('scan finds 3 runs', len(cat) == 3, str(len(cat)))
        leg = cat[cat.folder == c].iloc[0]
        check('legacy segment_id from folder name', leg.segment_id == 'seg_B', leg.segment_id)
        check('legacy poisson iterative', leg.poisson == 'iterative')
        check('missing delta is MISSING', leg.delta_FR == catalog.MISSING, str(leg.delta_FR))
        check('dt values', set(cat.dt) == {0.02, 0.01})
        check('root column', set(cat.root) == {root})
        keys = catalog.choice_keys(cat)
        check('choice keys: base then params',
              keys[:len(catalog.BASE_KEYS)] == catalog.BASE_KEYS
              and {'delta_FR', 'delta_IG'} <= set(keys), str(keys))

        opt = catalog.options(cat, {'model': 'thesis'})
        check('cascade narrows dt', opt['dt'] == [0.01, 0.02], str(opt['dt']))
        opt = catalog.options(cat, {'model': 'thesis', 'dt': 0.02})
        check('chosen key keeps alternatives', 'cook' in opt['model'], str(opt['model']))
        hit = catalog.matches(cat, {'model': 'thesis'})
        check('matches best error first', list(hit.folder) == [b, a])
        check('any = no filter', len(catalog.matches(cat, {'model': 'any'})) == 3)


TESTS = [test_catalog]

if __name__ == '__main__':
    only = sys.argv[1:]
    for t in TESTS:
        if not only or any(o in t.__name__ for o in only):
            t()
    print('FAILURES:', FAILURES if FAILURES else 'none')
    sys.exit(1 if FAILURES else 0)
