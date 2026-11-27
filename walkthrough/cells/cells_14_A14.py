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

The form picks the cases (sequence, frame), dt and the iteration count. Redraw
reads the saved npz of each case (the report's need the synthetic `ecrot_bicycle`
and `dynamic` runs of the cluster); Live iterates now on the cases without one, into
`experiments/walkthrough/converge*/` (converge_path's file names).
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
# the A14 form: which (sequence, frame) cases, at which dt and iteration count
from report_runs import converge_path
CASES_A14 = list(make_figures.CONVERGE) + [(s[3], 74) for s in SEGMENTS
                                           if (s[3], 74) not in make_figures.CONVERGE]
form('A14', [Knob('cases', list(make_figures.CONVERGE), 'many', choices=CASES_A14),
             Knob('dt', 0.02, choices=[0.02, 0.01]), Knob('n_iters', 75, choices=[75, 150])])
'''),

('code', A14, r'''
# step 1 -- the snapshots per case: a saved npz (shown, not recomputed) or made now (Live)
p = params('A14')
rep_A14 = is_report('A14')                      # the report-only steps run only then
ckpts_A14 = sorted({c for c in (1, 3, 10, 25, 50, 75) if c <= p['n_iters']} | {p['n_iters']})
same_as_report = p['dt'] == 0.02 and p['n_iters'] == 75
# Live npz: converge_path's file name, in a folder per (dt, n_iters)
NAME_A14 = 'walkthrough/converge' + ('' if same_as_report else f"_dt{p['dt']*1000:g}ms_i{p['n_iters']}")
cases_A14, paths_A14, make_A14 = list(p['cases']), {}, []
for lab, fr in cases_A14:
    saved = ([converge_path(lab, fr)] if same_as_report else []) + [converge_path(lab, fr, NAME_A14)]
    hit = next((x for x in saved if os.path.exists(x)), None)
    if hit:
        paths_A14[lab, fr] = hit
        if p['mode'] == 'Live':
            print(f'>>> SAVED RUN (not rerun): {hit}')
    elif p['mode'] == 'Live':
        seg = next(s for s in SEGMENTS if s[3] == lab)
        if need([os.path.join('data', seg[0])]):
            paths_A14[lab, fr] = converge_path(lab, fr, NAME_A14)
            make_A14.append((lab, fr))
    else:
        print(f'missing: {saved[0]} -- no saved run for {lab} frame {fr}'
              + (f' (run_all.sh job converge_{lab})' if rep_A14 else ''))
READY['A14'] = bool(paths_A14)
for (lab, fr), path in paths_A14.items():
    print(f'{lab:8s} frame {fr}: {path}' + ('  (made now)' if (lab, fr) in make_A14 else ''))
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


snaps_A14 = {}                                  # {(label, frame): {model: {checkpoint: I}}}
for (lab, fr), path in paths_A14.items():
    if (lab, fr) in make_A14:                   # cmd_converge, unrolled: made now
        seg = next(s for s in SEGMENTS if s[3] == lab)
        snaps_A14[lab, fr] = {m: converge_one_A14(seg, m, p['dt'], fr, ckpts_A14) for m in MODELS}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, checkpoints=ckpts_A14, segment=seg[1],
                            **{f'{m}_{it}': snaps_A14[lab, fr][m][it] for m in MODELS for it in ckpts_A14})
        print(f'-> {path}')
    else:                                       # its npz read
        z = np.load(path)
        ck = [int(c) for c in z['checkpoints']]
        snaps_A14[lab, fr] = {m: {it: z[f'{m}_{it}'] for it in ck} for m in MODELS}
# how I sharpens: its spatial std at every checkpoint, per model
for (lab, fr), by_model in snaps_A14.items():
    print(f'== {lab} frame {fr}: std of I at each checkpoint (rows: models, columns: iterations)')
    display(pd.DataFrame({m: {it: float(np.std(I)) for it, I in s.items()}
                          for m, s in by_model.items()}).T.round(4))
'''),

('code', A14, r'''
%%section A14
# step 3 -- make_figures.py::fig_converge, unrolled: rows models, columns checkpoints
from interacting_maps.style import use_style, save
from interacting_maps.viz import draw_map
pix_A14 = {}
for (lab, fr), by_model in snaps_A14.items():
    one_per_lab = len({l for l, _ in snaps_A14}) == len(snaps_A14)
    name = f'fig_converge_{lab}' + ('' if one_per_lab else f'_frame{fr}')   # fig_converge's names
    ck = sorted(next(iter(by_model.values())))
    with use_style('paper'):
        fig_A14, axes_A14 = plt.subplots(len(MODELS), len(ck),
                                 figsize=make_figures.grid_size(len(MODELS), len(ck), label_lines=1))
        axes_A14 = np.atleast_2d(axes_A14)
        for r, model in enumerate(MODELS):
            for c, it in enumerate(ck):
                draw_map(axes_A14[r, c], 'I', by_model[model][it], title=f'iter {it}' if r == 0 else '')
            axes_A14[r, 0].set_ylabel(make_figures.MODELS[model][0])
        pix_A14[f'{name}.png'] = fig_pixels(fig_A14)
        save(fig_A14, os.path.join(walk_dir('A14'), f'{name}.png'), show=True)
'''),

('code', A14, r'''
%%section A14
# same-code check -- cmd_converge on the cases made now; fig_converge itself on
# the report's npz (report configuration only)
import argparse
for lab, frame in make_A14:
    chk_A14 = os.path.join(walk_dir('A14'), 'check', f'{lab}_frame{frame}.npz')
    os.makedirs(os.path.dirname(chk_A14), exist_ok=True)
    args_A14 = argparse.Namespace(checkpoints=','.join(map(str, ckpts_A14)), sequences=lab,
                                  frame=frame, dt=p['dt'])
    with redirect(run_experiments, converge_path=lambda l, f, name='converge': chk_A14), \
         contextlib.redirect_stdout(io.StringIO()):
        run_experiments.cmd_converge(E, args_A14)
    z = np.load(chk_A14)
    check(f'cmd_converge snapshots {lab} frame {frame}',
          np.concatenate([z[f'{m}_{it}'].ravel() for m in MODELS for it in ckpts_A14]),
          np.concatenate([snaps_A14[lab, frame][m][it].ravel() for m in MODELS for it in ckpts_A14]))
# fig_converge on the form's own npz: its CONVERGE and converge_path pointed at them
# (the report's own when the form is the report's)
if len({l for l, _ in paths_A14}) < len(paths_A14):
    print('[skip] fig_converge: one figure per sequence, and two cases share one')
else:
    report_npz = rep_A14 and all(paths_A14[c] == converge_path(*c) for c in paths_A14)
    attrs_A14 = {} if report_npz else dict(
        CONVERGE=list(paths_A14), converge_path=lambda l, f, name='converge': paths_A14[l, f])
    with caught_figures(make_figures, FIG=os.path.join(walk_dir('A14'), 'check'), **attrs_A14) as got_A14:
        make_figures.fig_converge(E)
    for name, mine in pix_A14.items():
        if name in got_A14['pixels']:
            check(f'fig_converge {name}', mine, got_A14['pixels'][name])
        else:
            print(f'[DIFF] fig_converge {name}: the report function drew no such figure')
'''),
]
