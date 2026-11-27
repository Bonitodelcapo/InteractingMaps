A15 = ('A15',)

CELLS = [
('md', A15, r'''
## A15 · Reconstruction: 20 against 10 ms, iterative against exact

**Question.** What does the intensity map depend on? Halving the accumulation
window cleans up G; solving the Poisson equation exactly recovers the low
frequencies the iterative update leaves out.

**Made by** `report/run_experiments.py::cmd_recon` (`--what recon`), `run_all.sh`
job `recon` (`--sequences boxes,poster --duration 0.8`) → for each sequence two
CMax-anchor runs, `report_runs.recon_cfg` at dt = 20 ms (40 frames) and 10 ms
(80 frames), iterative update, adopted step sizes, in `experiments/recon_figure/`.

**Read by** `report/make_figures.py::fig_recon` → fig:recon (APS, iterative at 20
and 10 ms, the exact fft read-out at 10 ms; |r| under each panel).

**Steps**: the two runs per sequence (found or made) → their final maps →
the |r| of each panel against log(1+APS) → the figure → same-code check.

**Flag.** The local `recon_figure` runs predate the cluster's csvs (stage-1
finding): their numbers may differ from what the cluster's runs give now.
Step 2 prints each run's commit and date. The same-code check compares the
figure against `fig_recon` drawn from the *same* runs, so it is unaffected.
'''),

('code', A15, r'''
# the A15 form: which sequences, which model, how long
from report_runs import recon_cfg
RECON_A15 = make_figures.RECON                  # the report's sequences, duration, model, root
form('A15', [Knob('segments', list(RECON_A15['sequences']), 'many', choices=[s[3] for s in SEGMENTS]),
             Knob('model', RECON_A15['model'], choices=MODELS[1:]),
             Knob('duration', RECON_A15['duration'], choices=[0.06, 0.2, 0.8, 1.5])])
'''),

('code', A15, r'''
# step 1 -- the runs, recon_cfg at 20 and 10 ms: found (Redraw) or made now (Live;
# a saved run is shown, not rerun). Live: choose a short duration here (0.06 s).
p = params('A15')
rep_A15 = is_report('A15')                      # the report-only steps run only then
seqs_A15 = [s for s in SEGMENTS if s[3] in p['segments']]
model_A15, dur_A15 = p['model'], p['duration']
runs_A15, cfgs_A15 = {}, {}                     # {label: {dt: run folder}}, {label: {dt: RunConfig}}
for ds, sid, t0, lab in seqs_A15:
    if not need([os.path.join('data', ds)]):    # recon_cfg reads the dataset's gyro
        continue
    cfgs_A15[lab] = {dt: recon_cfg(ds, sid, t0, model_A15, dt, dur_A15, 'experiments/walkthrough')
                     for dt in (0.02, 0.01)}
    got = {}
    for dt, rc in cfgs_A15[lab].items():
        if p['mode'] == 'Live':
            got[dt] = saved_or_run(rc)
        else:                                   # the report's root first
            got[dt] = report_or_saved(rc, RECON_A15['root'])
            if got[dt] is None:
                print(f'no saved run for {lab} {dt*1000:.0f} ms' + (' (run_all.sh job recon)' if rep_A15 else ''))
    if all(got.values()):
        runs_A15[lab] = got
READY['A15'] = bool(runs_A15)
for lab, d in runs_A15.items():
    for dt, f in d.items():
        print(f'{lab:8s} {dt*1000:.0f} ms  {f}')
'''),

('code', A15, r'''
%%section A15
# step 2 -- what each run is, and which code made it (the local runs may be stale)
info_A15 = []
for lab, d in runs_A15.items():
    for dt, f in d.items():
        r = experiment_tracking.read_run(f)
        c, s, pv = r['config'], r['summary'] or {}, r.get('provenance', {})
        info_A15.append([lab, f'{dt*1000:.0f} ms', c['n_frames'], c['poisson'],
                     round(s.get('mean_err_deg_s', np.nan), 2), round(s.get('recon_r_mean', np.nan), 3),
                     round(s.get('recon_r_mean_inloop', np.nan), 3),
                     str(pv.get('git_commit'))[:7], str(pv.get('timestamp'))[:10]])
display(pd.DataFrame(info_A15, columns=['sequence', 'dt', 'n_frames', 'poisson', 'err deg/s',
                                    '|r| read-out', '|r| in-loop', 'commit', 'date']))
'''),

('code', A15, r'''
%%section A15
# step 3 -- the four panels of each row, and |r| = abs_corr on crop_interior against log(1+APS)
from report_runs import aps_frame
from interacting_maps.metrics import abs_corr, crop_interior
from interacting_maps.network_dissertation import solve_poisson_exact
panels_A15, r_A15 = {}, []
for ds, sid, t0, lab in seqs_A15:
    if lab not in runs_A15:
        continue
    got_A15 = {dt: np.load(os.path.join(f, 'maps_final.npz')) for dt, f in runs_A15[lab].items()}
    aps = aps_frame(ds, t0 + dur_A15)           # the APS frame at the end of the track
    ref = np.log(aps.astype(np.float64) + 1.0) if aps is not None else None
    panels_A15[lab] = [('APS', aps, 'aps'),
                       (r'iterative, $\Delta t = 20$ ms', got_A15[0.02]['I'], 'I'),
                       (r'iterative, $\Delta t = 10$ ms', got_A15[0.01]['I'], 'I'),
                       (r'exact read-out, $\Delta t = 10$ ms', solve_poisson_exact(got_A15[0.01]['G'], 'fft'), 'I')]
    r_A15.append([lab] + [abs_corr(crop_interior(img), crop_interior(ref)) if ref is not None else np.nan
                          for _t, img, _k in panels_A15[lab][1:]])
    if aps is None:
        print(f'{lab}: no APS frame in data/{ds} -- no APS panel and no |r|')
display(pd.DataFrame(r_A15, columns=['sequence', 'iterative 20 ms', 'iterative 10 ms',
                                     'exact fft 10 ms']).round(3))
'''),

('code', A15, r'''
%%section A15
# step 4 -- make_figures.py::fig_recon, unrolled
from interacting_maps.style import use_style, save
from interacting_maps.viz import draw_map
with use_style('paper'):
    fig_A15, axes_A15 = plt.subplots(len(panels_A15), 4,
                                 figsize=make_figures.grid_size(len(panels_A15), 4, below=1, label_lines=1))
    axes_A15 = np.atleast_2d(axes_A15)
    for row, (lab, panels) in enumerate(panels_A15.items()):
        aps = panels[0][1]
        ref = np.log(aps.astype(np.float64) + 1.0) if aps is not None else None
        for col, (ttl, img, kind) in enumerate(panels):
            a = axes_A15[row, col]
            if img is None:
                a.set_axis_off()
                continue
            draw_map(a, kind, img, title=ttl if row == 0 else '')
            if col == 0:
                a.set_ylabel(lab)
            if col > 0 and ref is not None:
                a.set_xlabel(f'$|r|={abs_corr(crop_interior(img), crop_interior(ref)):.2f}$')
    pix_A15, text_A15 = fig_pixels(fig_A15), fig_text(fig_A15)
    save(fig_A15, os.path.join(walk_dir('A15'), 'fig_recon.png'), show=True)
'''),

('code', A15, r'''
%%section A15
# same-code check -- in Live, cmd_recon's RunConfigs; fig_recon itself on the
# report's runs (report configuration only: its RECON is the report's)
import argparse
if p['mode'] == 'Live':
    for lab in runs_A15:
        seen_A15 = []
        args_A15 = argparse.Namespace(name='walkthrough/A15', sequences=lab, model=model_A15,
                                      duration=dur_A15, no_frames=True, frame_stride=1)
        with redirect(run_experiments, _run=lambda E_, rc, sf, st: seen_A15.append(rc)), \
             contextlib.redirect_stdout(io.StringIO()):
            run_experiments.cmd_recon(E, args_A15)
        strip = lambda rc: {k: v for k, v in rc.to_dict().items() if k != 'out_root'}
        mine_A15 = [strip(rc) for rc in cfgs_A15[lab].values()]
        theirs_A15 = [strip(rc) for rc in seen_A15]
        print(f'[ok] config {lab}' if mine_A15 == theirs_A15 else f'[DIFF] config {lab}')
# fig_recon on the form's own runs: its RECON pointed at them (sequences, duration,
# model, and the root they lie in -- recon_cfg names the folders the same way)
roots_A15 = {os.path.dirname(os.path.dirname(os.path.dirname(os.path.normpath(f))))
             for d in runs_A15.values() for f in d.values()}
if len(roots_A15) != 1:
    print('[skip] fig_recon: the runs lie in several roots' if roots_A15 else '[skip] fig_recon: no runs')
else:
    labs_A15 = tuple(runs_A15)
    if len(labs_A15) < len(seqs_A15):
        print(f'fig_recon is checked on {labs_A15} only (the other rows are missing above)')
    attrs_A15 = dict(RECON=dict(RECON_A15, sequences=labs_A15, duration=dur_A15, model=model_A15,
                                root=roots_A15.pop()))
    with caught_figures(make_figures, FIG=os.path.join(walk_dir('A15'), 'check'), **attrs_A15) as got_A15:
        make_figures.fig_recon(E)
    check('fig_recon pixels', pix_A15, got_A15['pixels'].get('fig_recon.png', []))
    check('fig_recon |r|', nums(text_A15), nums(got_A15['text'].get('fig_recon.png', [])))
'''),
]
