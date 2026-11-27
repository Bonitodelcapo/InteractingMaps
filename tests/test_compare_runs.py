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


TRACK_COLS = ['frame', 'time', 'est_wx', 'est_wy', 'est_wz', 'gt_wx', 'gt_wy', 'gt_wz',
              'err_deg_s', 'ref_source']


def write_tracking(d, n=4, extra=None, est_wz=1.0):
    """tracking.csv with n frames; extra = {column: value} added to every row."""
    import pandas as pd
    df = pd.DataFrame({'frame': np.arange(n), 'time': 1.0 + 0.02 * (np.arange(n) + 0.5),
                       'est_wx': 0.0, 'est_wy': 0.0, 'est_wz': est_wz, 'gt_wx': 0.0,
                       'gt_wy': 0.0, 'gt_wz': 1.0, 'err_deg_s': 2.0,
                       'ref_source': 'groundtruth', **(extra or {})})
    df.to_csv(os.path.join(d, 'tracking.csv'), index=False)


def write_png(path, value=0.5):
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.imsave(path, np.full((2, 2), value), cmap='gray', vmin=0, vmax=1)


def write_maps(d, which='final', scale=1.0, **extra):
    H, W = 6, 8
    rng = np.random.default_rng(0)
    np.savez(os.path.join(d, f'maps_{which}.npz'), I=scale * rng.random((H, W)),
             G=rng.random((H, W, 2)), F=rng.random((H, W, 2)), **extra)


# ---------------------------------------------------------------- Task 2: outputs
def test_outputs():
    from interacting_maps.compare_runs import outputs
    print('outputs')
    keys = [o.key for o in outputs.OUTPUTS]
    check('output keys unique', len(set(keys)) == len(keys))
    check('no exp11 / iter_frames', not any('exp11' in k or 'iter_frames' in k for k in keys))
    check('bands known', {o.band for o in outputs.OUTPUTS}
          == {'curves', 'table', 'figures', 'maps', 'frames'})
    with tempfile.TemporaryDirectory() as root:
        d = make_run(root, 'ds/thesis/seg_A', cfg())
        write_tracking(d)
        write_maps(d)
        write_png(os.path.join(d, 'video_frames', 'frame_0000.png'))
        write_png(os.path.join(d, 'video_frames', 'frame_0002.png'))
        av = outputs.available(d)
        want = {'omega', 'err', 'final:I', 'final:I_from_G', 'video_frames', 'table'}
        check('available has what is there', want <= av, str(want - av))
        nope = {'recon_r', 'final:V', 'center:I', 'beta', 'tracking_plot.png', 'final:aps'}
        check('available lacks what is not', not (nope & av), str(nope & av))
        for name in ('frame_10.png', 'frame_2.png'):
            write_png(os.path.join(d, 'iwe', name))
        paths = outputs.frame_paths(d, 'iwe_frames')
        check('frame paths sorted by number',
              [os.path.basename(p) for p in paths] == ['frame_2.png', 'frame_10.png'])
        check('read_tracking', len(outputs.read_tracking(d)) == 4)
        check('read_maps', set(outputs.read_maps(d, 'final')) == {'I', 'G', 'F'})


# ---------------------------------------------------------------- Task 3: bands
def make_runs(root, specs):
    """specs: [(rel, cfg kwargs, tracking kwargs, maps kwargs or None)] -> list of bands.Run."""
    from interacting_maps.compare_runs import bands, catalog
    for rel, ck, tk, mk in specs:
        d = make_run(root, rel, cfg(**ck), {'mean_err_deg_s': 1.0, 'mean_beta': 1.0})
        write_tracking(d, **tk)
        if mk is not None:
            write_maps(d, **mk)
    cat = catalog.scan([root])
    rows = [cat[cat.folder == os.path.normpath(os.path.join(root, s[0]))].iloc[0] for s in specs]
    return [bands.Run(i + 1, bands.SLOT_COLORS[i], r, set()) for i, r in enumerate(rows)]


def test_bands():
    from interacting_maps.compare_runs import bands
    print('bands')
    check('20 slot colours, distinct', len(set(bands.SLOT_COLORS)) == 20)
    with tempfile.TemporaryDirectory() as root:
        runs = make_runs(root, [('a', dict(dt=0.02), dict(extra={'beta': 1.0}), {}),
                                ('b', dict(dt=0.01), dict(est_wz=2.0), dict(scale=2.0))])
        lab = bands.labels(runs)
        check('labels show what differs', lab == {1: '#1 dt=20ms', 2: '#2 dt=10ms'}, str(lab))
        check('one segment: no warning', bands.segment_warning(runs) is None)

        fig, missing = bands.curve_figure(runs, 'beta')
        check('run without the column is reported', missing == [2], str(missing))
        check('it is not drawn', len(fig.axes[0].lines) == 1)

        fig, missing = bands.curve_figure(runs, 'omega')
        ax_z = fig.axes[2]
        est = [l for l in ax_z.lines if l.get_color() == runs[0].color][0]
        check('omega in deg/s', np.allclose(est.get_ydata(), np.degrees(1.0)))
        check('x = time since start', np.allclose(est.get_xdata(), [0.01, 0.03, 0.05, 0.07]))
        est2 = [l for l in ax_z.lines if l.get_color() == runs[1].color][0]
        check('x of dt=10ms run', np.allclose(est2.get_xdata(), [0.005, 0.015, 0.025, 0.035]))
        check('one reference for one segment',
              sum(l.get_color() == 'k' for l in ax_z.lines) == 1)

        html = bands.table_html(runs, only_diff=False)
        check('table highlights differing dt', bands.DIFF_STYLE in html and '20' in html)
        check('only_diff drops constant model column',
              'model' in html and 'model' not in bands.table_html(runs, only_diff=True))

        fig = bands.map_figure(runs, 'final:I')
        a, b = (ax.images[0].get_array() for ax in fig.axes[:2])
        check('shared map scale (×2 map brighter than ×1)', float(np.mean(a)) < float(np.mean(b)))

        write_maps(runs[0].row.folder, which='center')
        fig = bands.map_figure(runs[:1], 'center:aps')
        check('center without t_mid: no guessed APS',
              'no t_mid saved' in fig.axes[0].get_title())

        write_png(os.path.join(runs[0].row.folder, 'video_frames', 'frame_0000.png'))
        write_png(os.path.join(runs[0].row.folder, 'video_frames', 'frame_0001.png'))
        cells = bands.image_cells(runs, 'video_frames', frame=5)
        check('frame beyond the run: empty cell', cells == [(1, None), (2, None)], str(cells))
        cells = bands.image_cells(runs, 'video_frames', frame=1)
        check('frame k path', cells[0][1].endswith('frame_0001.png') and cells[1][1] is None)
        cells = bands.image_cells(runs, 'tracking_plot.png')
        check('figure missing: None', cells == [(1, None), (2, None)])

    with tempfile.TemporaryDirectory() as root:
        runs = make_runs(root, [('a', dict(seg='seg_A'), {}, None),
                                ('b', dict(seg='seg_B'), {}, None)])
        w = bands.segment_warning(runs) or ''
        check('two segments: warning', '2 segments' in w and 'does not make sense' in w, w)
        fig, _ = bands.curve_figure(runs, 'omega')
        check('one reference per segment',
              sum(l.get_color() == 'k' for l in fig.axes[2].lines) == 2)


TESTS = [test_catalog, test_outputs, test_bands]

if __name__ == '__main__':
    only = sys.argv[1:]
    for t in TESTS:
        if not only or any(o in t.__name__ for o in only):
            t()
    print('FAILURES:', FAILURES if FAILURES else 'none')
    sys.exit(1 if FAILURES else 0)
