A14 = ('A14',)
IMG = ('A14', 'image-defs')

CELLS = [
('md', A14, r'''
## A14 · Convergence within a frame

**Question.** How does the intensity map I sharpen over the message-passing
iterations of one frame, model by model?

**Made by** `report/run_experiments.py::cmd_converge` (`--what converge`),
`run_all.sh` jobs `converge_dynamic` (`--sequences dynamic --frame 45`) and
`converge_bicycle` (`--sequences bicycle --frame 40`) → one npz per sequence at
`report_runs.converge_path(label, frame)` = `experiments/converge/<label>_frame<frame>.npz`,
holding I of every model at the checkpoints `1,3,10,25,50,75`.

**Read by** `report/make_figures.py::fig_converge` → `fig_converge_dynamic`
(fig:converge_dynamic) and `fig_converge_bicycle` (appendix).

**Steps** (those of `cmd_converge`, then `fig_converge`): the frame's events →
a network per model, warm-started like a run → `step(V, n_iters=1)` repeated,
I kept at every checkpoint → the grid of images → same-code check.

Redraw reads the npz (it needs the synthetic `ecrot_bicycle` and the `dynamic`
runs of the cluster; without them the section skips). Live iterates now on the
panel's segment, at frame `n_frames - 1`, with the checkpoints up to `n_iters`.
'''),

('code', IMG, r'''
# image-section definitions (A14-A18): capture a report figure instead of saving
# it, and read what it shows -- the figure-data checks compare these
import io


def fig_pixels(fig):
    """Every image drawn on a figure, in axes order, as one flat array."""
    arrs = [np.asarray(im.get_array(), float).ravel() for ax in fig.axes for im in ax.get_images()]
    return np.concatenate(arrs) if arrs else np.array([])


def fig_text(fig):
    """Every title and axis label of a figure, in axes order."""
    return [s for ax in fig.axes for s in (ax.get_title(), ax.get_xlabel(), ax.get_ylabel()) if s]


@contextlib.contextmanager
def caught_figures(module, **attrs):
    """Run a report drawing function with its save() replaced by a collector and
    its output globals redirected (**attrs). Yields {'pixels': {file: array},
    'text': {file: [labels]}, 'out': its stdout}; the figures are closed."""
    got = {'pixels': {}, 'text': {}, 'out': io.StringIO()}
    def keep(fig, path, show=False):
        name = os.path.basename(path)
        got['pixels'][name], got['text'][name] = fig_pixels(fig), fig_text(fig)
        plt.close(fig)
    with redirect(module, save=keep, **attrs), contextlib.redirect_stdout(got['out']):
        yield got
'''),

('code', A14, r'''
# step 1 -- the snapshots: found (Redraw) or set up to be made now (Live)
from report_runs import converge_path
p = params()
if p['mode'] == 'Redraw':
    # the report's (sequence, frame) pairs; n_frames of its runs = frame + 1
    cases_A14 = list(make_figures.CONVERGE)
    note_redraw(p, cases_A14[0][1] + 1)
    paths_A14 = {lab: converge_path(lab, fr) for lab, fr in cases_A14
                 if need([converge_path(lab, fr)], [f'converge_{lab}'])}
    ckpts_A14 = None                            # read from the npz
    READY['A14'] = bool(paths_A14)
else:
    ds_A14, sid_A14, t0_A14, lab_A14 = p['seg']
    cases_A14 = [(lab_A14, p['n_frames'] - 1)]  # the last frame of an n_frames track
    ckpts_A14 = sorted({c for c in (1, 3, 10, 25, 50, 75) if c <= p['n_iters']} | {p['n_iters']})
    paths_A14 = {lab_A14: os.path.join(walk_dir('A14'), f'{lab_A14}_frame{cases_A14[0][1]}.npz')}
    READY['A14'] = need([os.path.join('data', ds_A14)])
    print(f"Live: {lab_A14} frame {cases_A14[0][1]}, dt {p['dt']*1000:.0f} ms, "
          f"checkpoints {ckpts_A14}; cmd_converge uses the adopted step sizes (panel deltas ignored)")
for lab, path in paths_A14.items():
    print(f'{lab:8s} {path}')
'''),

('code', A14, r'''
%%section A14
# step 2 -- run_experiments.py::cmd_converge, unrolled (Live), or its npz read (Redraw)
from interacting_maps.data_loader import (EventFrameSequence, CameraCalibration,
                                          load_events_fast, undistort_events)
from interacting_maps.angular_velocity import CMaxAngularVelocity
from interacting_maps.viz import net_maps


def converge_one_A14(seg, model, dt, frame, checkpoints):
    """cmd_converge for one (segment, model): {checkpoint: I}."""
    ds, sid, t0, lab = seg
    pth = E.get_dataset_paths(ds)
    calib = CameraCalibration(pth['calib'])
    nf = frame + 1
    rc = run_cfg(ds, sid, t0, model, dt, nf)
    seq = EventFrameSequence(pth['events'], pth['calib'], frame_duration=dt, t_start=t0,
                             n_frames=nf, clip_value=10.0,
                             undistort=rc.undistort_at_event_level, sensor_size=(180, 240))
    H, W = seq.H, seq.W
    V, _tm = list(seq)[frame]                   # the event frame iterated on
    net = E.make_network(rc, H, W, calib.fx, calib.fy, calib.cx, calib.cy)
    if hasattr(net, 'initialize_from_rotation'):  # warm start from the gyro, as a run
        net.initialize_from_rotation(rc.initial_R)
    else:
        net.R = rc.initial_R.copy()
    lo, hi = t0 + frame * dt, t0 + (frame + 1) * dt
    kw = {}                                     # the anchor target of this frame
    if getattr(rc, 'use_cmax', False) or getattr(rc, 'use_imu', False):
        cev = undistort_events(load_events_fast(pth['events'], t_start=lo, duration=dt + 0.05), calib)
        win = cev[(cev[:, 0] >= lo) & (cev[:, 0] < hi)]
        kw['events'] = win
        if getattr(rc, 'use_cmax', False) and len(win) > 10:
            est = CMaxAngularVelocity(H, W, calib.fx, calib.fy, calib.cx, calib.cy, use_polarity=True)
            kw['omega_imu'] = est.estimate(win, t_ref=0.5*(lo+hi), omega_init=np.zeros(3))
        elif getattr(rc, 'use_imu', False):
            kw['omega_imu'] = E.get_gyro_for_frame(E.load_imu(pth['imu']), lo, hi)
    snaps = {}
    for it in range(1, max(checkpoints) + 1):   # the network's own step(), one iteration
        try:
            net.step(V, n_iters=1, **kw)
        except TypeError:                       # cook's step takes no anchor
            net.step(V, n_iters=1)
        if it in checkpoints:
            snaps[it] = net_maps(net, H, W)['I'].copy()
    return snaps


snaps_A14 = {}                                  # {label: {model: {checkpoint: I}}}
if p['mode'] == 'Live':
    lab, frame = cases_A14[0]
    snaps_A14[lab] = {m: converge_one_A14(p['seg'], m, p['dt'], frame, ckpts_A14) for m in MODELS}
    np.savez_compressed(paths_A14[lab], checkpoints=ckpts_A14, segment=p['seg'][1],
                        **{f'{m}_{it}': snaps_A14[lab][m][it] for m in MODELS for it in ckpts_A14})
    print(f'-> {paths_A14[lab]}')
else:
    for lab, path in paths_A14.items():
        z = np.load(path)
        ck = [int(c) for c in z['checkpoints']]
        snaps_A14[lab] = {m: {it: z[f'{m}_{it}'] for it in ck} for m in MODELS}
# how I sharpens: its spatial std at every checkpoint, per model
for lab, by_model in snaps_A14.items():
    print(f'== {lab}: std of I at each checkpoint (rows: models, columns: iterations)')
    display(pd.DataFrame({m: {it: float(np.std(I)) for it, I in s.items()}
                          for m, s in by_model.items()}).T.round(4))
'''),

('code', A14, r'''
%%section A14
# step 3 -- make_figures.py::fig_converge, unrolled: rows models, columns checkpoints
from interacting_maps.style import use_style, save
from interacting_maps.viz import draw_map
pix_A14 = {}
for lab, by_model in snaps_A14.items():
    ck = sorted(next(iter(by_model.values())))
    with use_style('paper'):
        fig_A14, axes_A14 = plt.subplots(len(MODELS), len(ck),
                                 figsize=make_figures.grid_size(len(MODELS), len(ck), label_lines=1))
        axes_A14 = np.atleast_2d(axes_A14)
        for r, model in enumerate(MODELS):
            for c, it in enumerate(ck):
                draw_map(axes_A14[r, c], 'I', by_model[model][it], title=f'iter {it}' if r == 0 else '')
            axes_A14[r, 0].set_ylabel(make_figures.MODELS[model][0])
        pix_A14[f'fig_converge_{lab}.png'] = fig_pixels(fig_A14)
        save(fig_A14, os.path.join(walk_dir('A14'), f'fig_converge_{lab}.png'), show=True)
'''),

('code', A14, r'''
%%section A14
# same-code check -- fig_converge itself on the same npz; in Live also cmd_converge
import argparse
if p['mode'] == 'Live':
    lab, frame = cases_A14[0]
    chk_A14 = os.path.join(walk_dir('A14'), 'check', f'{lab}_frame{frame}.npz')
    os.makedirs(os.path.dirname(chk_A14), exist_ok=True)
    args_A14 = argparse.Namespace(checkpoints=','.join(map(str, ckpts_A14)), sequences=lab,
                              frame=frame, dt=p['dt'])
    with redirect(run_experiments, converge_path=lambda l, f, name='converge': chk_A14), \
         contextlib.redirect_stdout(io.StringIO()):
        run_experiments.cmd_converge(E, args_A14)
    z = np.load(chk_A14)
    check('cmd_converge snapshots',
          np.concatenate([z[f'{m}_{it}'].ravel() for m in MODELS for it in ckpts_A14]),
          np.concatenate([snaps_A14[lab][m][it].ravel() for m in MODELS for it in ckpts_A14]))
    attrs_A14 = dict(CONVERGE=cases_A14, converge_path=lambda l, f, name='converge': paths_A14[l])
else:
    attrs_A14 = {}
with caught_figures(make_figures, FIG=os.path.join(walk_dir('A14'), 'check'), **attrs_A14) as got_A14:
    make_figures.fig_converge(E)
for name, mine in pix_A14.items():
    if name in got_A14['pixels']:
        check(f'fig_converge {name}', mine, got_A14['pixels'][name])
    else:
        print(f'[DIFF] fig_converge {name}: the report function drew no such figure')
'''),
]
