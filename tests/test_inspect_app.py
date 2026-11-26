"""The Qt inspection app, headless (Tasks 4-9): shell, runs tab, map area,
curve area, view tab, side plots, export. Drives the window through its methods.

Standalone script:  PYTHONPATH=. python tests/test_inspect_app.py
Runs offscreen (QT_QPA_PLATFORM=offscreen); uses tests/_inspect_fake.py, needs nothing in data/.
"""
import os
import sys
import time
import shutil

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _inspect_fake import make_run, make_dataset, NF
from interacting_maps import states_io
from interacting_maps.inspect_cache import RunCache
from interacting_maps.inspect_app import InspectApp, qt_app
from interacting_maps.viz import render_I

ROOT = 'experiments/inspect_app_test'
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


qt_app()
shutil.rmtree(ROOT, ignore_errors=True)
DATA = make_dataset(os.path.join(ROOT, 'data'))
try:
    # three runs on segment A (two of the same model, a sweep), one on segment B
    dA = [make_run(ROOT, 'cook', seed=1, folder='cook_a'),
          make_run(ROOT, 'thesis_imu', seed=2, folder='imu_dfr01', deltas={'delta_FR': 0.1}),
          make_run(ROOT, 'thesis_imu', seed=3, folder='imu_dfr05', deltas={'delta_FR': 0.5})]
    make_run(ROOT, 'thesis', t_start=1.02, seed=4, folder='other_seg')
    fake_replay(dA[1], 2)

    # ---------------- Task 4: shell
    app = InspectApp(root=ROOT, data_root=DATA)
    check('tabs Runs / View / Export',
          [app.tabs.tabText(i) for i in range(app.tabs.count())] == ['Runs', 'View', 'Export'])

    # ---------------- Task 5: runs tab
    rp = app.runs_panel
    rp.scan()
    check('two segments offered', rp.seg.count() == 2)
    rp.seg.setCurrentIndex([i for i in range(rp.seg.count()) if 't0 1.000' in rp.seg.itemText(i)][0])
    check('run list = folders of that segment', rp.run_list.count() == 3)
    rp.model_filter.setCurrentText('thesis_imu')
    check('model filter', rp.visible_runs() == 2)
    rp.model_filter.setCurrentText('all')
    rp.check_all()
    rp.load(blocking=True)
    check('load opens 3 runs', len(app.caches) == 3)
    check('info table one row per run', rp.info.rowCount() == 3)
    check('labels tell the sweep apart',
          [c.display for c in app.caches] == ['cook', 'thesis+IMU dFR=0.1', 'thesis+IMU dFR=0.5'])
    check('distinct colours', len({c.color for c in app.caches}) == 3)

    app.set_frame(99)
    check('set_frame clamps', app.frame == NF - 1)

    # load failure keeps the previous runs
    bad = make_run(ROOT, 'thesis', seed=9, folder='broken')
    with open(states_io.state_path(bad, 1), 'wb') as f:
        f.write(b'not an npz')
    before = list(app.caches)
    rp.load_dirs([bad], blocking=True)
    check('failed load keeps the previous runs', app.caches == before and rp.last_error)
    shutil.rmtree(bad)

    # ---------------- Task 6: map area
    ma = app.map_area
    app.set_maps_for({0, 2})
    check('map rows follow "maps for"', ma.run_rows == [0, 2])
    app.set_frame(2)
    img = ma.images[(0, 'I')].image
    sc = ma.scales()[0]['I']
    want = (render_I(app.caches[0].map_at('I', 2), scale=sc) * 255).astype(np.uint8)
    check('image holds that frame', img is not None and np.abs(img.astype(int) - want.astype(int)).max() <= 1)
    app.set_scale_mode('shared')
    s = ma.scales()
    check('shared scale equal across all loaded runs', s[0] == s[1] == s[2])
    app.set_scale_mode('per frame')
    check('per-frame scale = no fixed scale', all(x == {} for x in ma.scales()))
    app.set_scale_mode('fixed')
    app.set_maps_for(set())
    check('no maps: no run rows, raw row kept', ma.run_rows == [] and ma.raw_items)
    app.set_frame(1)
    app.set_maps_for({0, 1, 2})
    ma.on_image_click(-5, 999)
    check('probe click clamped to the image', app.probe == (0, 179))

    # ---------------- Task 7: curve area
    ca = app.curve_area
    check('one line per run in every curve plot',
          all(len([k for k in ca.lines if k[0] == n]) == 3 for n in ca.plots))
    app.set_curves_for({1})
    vis = [ca.lines[('err_deg_s', i)].isVisible() for i in range(3)]
    check('"curves for" toggles lines', vis == [False, True, False])
    m = ca.markers['err_deg_s']
    m.setValue(app.caches[0].times[2] + 0.001)
    m.sigPositionChangeFinished.emit(m)
    check('dragging the marker sets the frame', app.frame == 2)
    app.set_curves_for(set())
    check('no curves: plots kept, no crash', bool(ca.plots))
    app.set_curves_for({0, 1, 2})

    # ---------------- Task 8: view tab and side plots
    vp, side = app.view_panel, app.side_plots
    app.set_frame(NF - 1)
    vp.tick()
    check('play wraps to frame 0', app.frame == 0)
    app.set_frame(1)
    check('iteration slider off without a replay', not vp.iteration.isEnabled())
    app.set_frame(2)
    check('iteration slider on with a replay', vp.iteration.isEnabled())
    vp.spike_curve.setCurrentText('err_deg_s')          # 5, 6, 7 per frame
    app.set_frame(0)
    vp.goto_worst()
    check('worst frame', app.frame == NF - 1)
    vp.threshold.setValue(5.5)
    app.set_frame(0)
    vp.goto_spike(+1)
    check('next spike', app.frame == 1)
    app.set_probe(10, 20)
    check('probe plots, one line per run', len(side.plots.get('probe:I').listDataItems()) == 3)
    app.set_frame(2)
    check('residual plots on a replayed frame', 'res:|R|' in side.plots and side.plots['res:|R|'].isVisible())
    app.set_frame(1)
    check('no residual plots without replay', not side.plots['res:|R|'].isVisible())

    # ---------------- Task 9: export
    ep = app.export_panel
    ep.frame_from.setValue(0)
    ep.frame_to.setValue(1)
    ep.fmt.setCurrentText('gif')
    ep.folder.setText(os.path.join(ROOT, 'exports'))
    p = ep.export_frames()
    import imageio.v2 as imageio
    check('GIF over the frame range', len(imageio.mimread(p)) == 2 and 'f000-001' in os.path.basename(p))
    p = ep.screenshot()
    check('screenshot written', os.path.getsize(p) > 0)
    check('file names stay short', len(os.path.basename(ep._name('f000-001', 'gif'))) <= 150)

    # ---------------- speed: 10 runs, 5 with maps
    dirs = [make_run(ROOT, 'thesis_imu', seed=20 + i, n_frames=12, folder=f'sweep{i}', t_start=1.1,
                     deltas={'delta_FR': 0.1 * (i + 1)}) for i in range(10)]
    caches = [RunCache.open(d, DATA) for d in dirs]
    app.set_runs(caches)
    app.set_maps_for({0, 1, 2, 3, 4})
    app.map_area.prefill(blocking=True)     # what the background pass does after loading
    check('prefill renders every frame of the shown runs',
          all((i, kind, k, None, 'fixed') in app.map_area._rgb
              for i in range(5) for kind in ('I', 'G', 'F') for k in range(12)))
    t0 = time.perf_counter()
    for _ in range(2):
        for k in range(12):
            app.set_frame(k)
    ms = 1000 * (time.perf_counter() - t0) / 24
    check(f'stepping 10 runs / 5 with maps: {ms:.1f} ms per frame (< 40)', ms < 40)
finally:
    shutil.rmtree(ROOT, ignore_errors=True)

sys.exit(1 if FAILED else 0)
