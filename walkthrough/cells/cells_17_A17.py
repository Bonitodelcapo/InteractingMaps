A17 = ('A17',)

CELLS = [
('md', A17, r'''
## A17 · In-loop Poisson

**Question.** Where should the exact Poisson solve be applied: inside the loop,
on every iteration (`poisson=fft/dct`), or once at the end as a read-out of the
G the iterative network produced? In the loop, the exact solve's 1/|k|² gain
integrates every low-frequency error of G into blobs that feed back into G.

**Made by** `report/run_experiments.py::cmd_poisson` (`--what poisson`),
`run_all.sh` job `poisson` (its defaults: `--sequences poster,bicycle`,
3 s of 20 ms, CMax-anchor, adopted step sizes) → one run per in-loop mode
(iterative, fft, dct) in `experiments/poisson_solvers/`, and a five-panel
figure per sequence `experiments/poisson_solvers/poisson_<label>.png` (APS, the
three in-loop I, the fft and dct read-outs of the iterative run's G).

**Read by** `report/text_numbers.py::intensity_update` (text only, no figure or
table in the report).

**Steps**: the three runs (found or made) → their summaries → the five panels
and their low-frequency share and contrast (`_intensity_stats`) → the text
numbers → same-code check.
'''),

('code', A17, r'''
# the A17 form: which sequences, which in-loop modes, which model, how long
MODES_A17 = ('iterative', 'fft', 'dct')
form('A17', [Knob('segments', ['poster', 'bicycle'], 'many', choices=[s[3] for s in SEGMENTS]),
             Knob('modes', list(MODES_A17), 'many', choices=list(MODES_A17)),
             Knob('model', 'thesis_cmax', choices=MODELS[1:]), Knob('n_frames', 150)])
'''),

('code', A17, r'''
# step 1 -- the runs: found (Redraw, as intensity_update finds them: by stored config,
# the poisson job's folder first) or made now (Live; a saved run is shown, not rerun)
from report_runs import ADOPTED
p = params('A17')
rep_A17 = is_report('A17')                      # the report-only steps run only then
ROOT_A17 = os.path.join('experiments', 'poisson_solvers')
model_A17, nf_A17 = p['model'], p['n_frames']
runs_A17 = {}                                   # {segment: {mode: run folder}}
for s in [s for s in SEGMENTS if s[3] in p['segments']]:
    ds, sid, t0, lab = s
    for mode in [m for m in MODES_A17 if m in p['modes']]:
        d = None
        if p['mode'] == 'Redraw':
            # the poisson job's folder as intensity_update reads it (the last match) ...
            for f in sorted(glob.glob(os.path.join(ROOT_A17, ds, model_A17, f'{sid}_*'))):
                try:
                    c = experiment_tracking.read_config(f)
                except (OSError, ValueError):
                    continue
                if c.get('n_frames') == nf_A17 and c.get('poisson', 'iterative') == mode:
                    d = f
            if d is None:                       # ... else cmd_poisson's config in any saved root
                need([os.path.join(ROOT_A17, ds, model_A17)], ['poisson'] if rep_A17 else [])
                prm = {**make_tables.expected_params(model_A17), **ADOPTED}   # every step size
                with contextlib.redirect_stdout(io.StringIO()):
                    d = next((h for h in (experiment_tracking.find_run(
                             r, ds, model_A17, sid, nf_A17, prm, t_start=t0, poisson=mode)
                             for r in roots()) if h), None)
        elif need([os.path.join('data', ds)]):  # cmd_poisson's run_cfg: adopted step sizes
            d = saved_or_run(run_cfg(ds, sid, t0, model_A17, 0.02, nf_A17, out_root='experiments/walkthrough',
                                     poisson=mode, deltas=dict(ADOPTED)))
        if d is None:
            print(f'no saved run for {lab} {mode}' + (' (run_all.sh job poisson)' if rep_A17 else ''))
        else:
            runs_A17.setdefault(s, {})[mode] = d
READY['A17'] = bool(runs_A17)
for s, by in runs_A17.items():
    for mode, d in by.items():
        print(f'{s[3]:8s} {mode:9s} {d}')
'''),

('code', A17, r'''
%%section A17
# step 2 -- the summaries of the runs, the fields intensity_update quotes
info_A17 = []
for s, by in runs_A17.items():
    for mode in MODES_A17:
        if mode in by:
            c, sm = experiment_tracking.read_config(by[mode]), experiment_tracking.read_summary(by[mode])
            info_A17.append([s[3], mode, round(c['n_frames'] * c['frame_duration'], 2),
                         sm['mean_err_deg_s'], sm['low_freq_share'], sm['recon_r_mean_inloop'],
                         sm['recon_r_mean'], sm['I_std_median'], sm['blowup_frac']])
display(pd.DataFrame(info_A17, columns=['sequence', 'in-loop', 'secs', 'err deg/s', 'low-freq share',
                                    '|r| in-loop', '|r| read-out', 'median I std', 'blow-up']).round(3))
'''),

('code', A17, r'''
%%section A17
# step 3 -- cmd_poisson's five panels, unrolled: APS, the in-loop I of each mode,
# and the exact fft / dct read-outs of the iterative run's G (solve_poisson_exact)
from report_runs import aps_frame
from interacting_maps.network_dissertation import solve_poisson_exact
from interacting_maps.style import use_style, save, PAPER_WIDE
from interacting_maps.viz import draw_map
full_A17 = [s for s, by in runs_A17.items() if 'iterative' in by]
seg_A17 = full_A17[0] if full_A17 else None       # the first chosen sequence with an iterative run
stats_A17, pix_A17 = [], None
if seg_A17 is None:
    print('no iterative run: cmd_poisson draws nothing')
else:
    ds, sid, t0, lab = seg_A17
    c = experiment_tracking.read_config(runs_A17[seg_A17]['iterative'])
    maps_A17 = {m: np.load(os.path.join(runs_A17[seg_A17][m], 'maps_final.npz'))
            for m in MODES_A17 if m in runs_A17[seg_A17]}
    panels_A17 = [('APS reference', aps_frame(ds, t0 + c['n_frames'] * c['frame_duration']), 'aps')]
    for m in maps_A17:
        panels_A17.append((f'{m} in-loop' if m != 'iterative' else 'iterative (Eq. 6.61)', maps_A17[m]['I'], 'I'))
    for m in ('fft', 'dct'):
        if 'iterative' not in maps_A17:
            break
        panels_A17.append((f'{m} read-out', solve_poisson_exact(maps_A17['iterative']['G'], m), 'I'))
    print(f"{lab}: {'low_freq':>10} {'contrast':>10}")
    with use_style('paper'):
        fig_A17, ax_A17 = plt.subplots(1, len(panels_A17), figsize=(PAPER_WIDE, 1.9))
        for a, (ttl, img, kind) in zip(ax_A17, panels_A17):
            if img is None:
                a.set_axis_off()
                continue
            st = E._intensity_stats(img)
            stats_A17.append(f"  {ttl:<22} {st['low_freq_share']:10.3f} {st['contrast']:10.3f}")
            print(stats_A17[-1])
            draw_map(a, kind, img, title=ttl)
        fig_A17.suptitle(f'{lab}: the same G, five ways of turning it into I')
        pix_A17 = fig_pixels(fig_A17)
        save(fig_A17, os.path.join(walk_dir('A17'), f'poisson_{lab}.png'), show=True)
'''),

('code', A17, r'''
%%section A17
# step 4 -- text_numbers.py::intensity_update, unrolled (report configuration only:
# it reads the poisson job)
lines_A17 = []
if rep_A17:
    for s in SEGMENTS:
        ds, sid, t0, lab = s
        by = runs_A17.get(s, {})
        for mode in MODES_A17:
            if mode not in by:
                continue
            c, sm = experiment_tracking.read_config(by[mode]), experiment_tracking.read_summary(by[mode])
            lines_A17.append(f'{lab:8s} {mode:9s} ({c["n_frames"] * c["frame_duration"]:.1f} s): '
                             f'err {sm["mean_err_deg_s"]:.2f}, low-freq share of the final I '
                             f'{sm["low_freq_share"]:.3f}, track-mean |r| in-loop {sm["recon_r_mean_inloop"]:.2f} '
                             f'read-out {sm["recon_r_mean"]:.2f}, median I std {sm["I_std_median"]:.2f}, '
                             f'blow-up {sm["blowup_frac"]:.2f}')
        if 'iterative' in by and lab == 'poster':   # the APS's own low-frequency share
            c = experiment_tracking.read_config(by['iterative'])
            aps = aps_frame(ds, t0 + c['n_frames'] * c['frame_duration'])
            if aps is not None:
                lines_A17.append(f'{lab:8s} APS frame at the end of the track: low-freq share '
                                 f'{E._intensity_stats(np.log(aps + 1.0))["low_freq_share"]:.3f}')
    print('\n'.join(lines_A17) or '(no runs)')
else:
    print('[skip] report-only step')
'''),

('code', A17, r'''
%%section A17
# same-code check -- intensity_update itself (Redraw); cmd_poisson with its run step
# replaced by a no-op on the same runs: same stats, same figure, same RunConfigs
import argparse
from_report = all(os.path.normpath(d).startswith(os.path.normpath(ROOT_A17))
                  for by in runs_A17.values() for d in by.values())
under_A17 = lambda root: all(os.path.normpath(d).startswith(os.path.normpath(root))
                             for by in runs_A17.values() for d in by.values())
if not rep_A17:
    print('[skip] not the report configuration')
elif not from_report:
    need([ROOT_A17], ['poisson'])
    print("[skip] intensity_update: these runs are not the poisson job's")
else:
    with redirect(text_numbers, OUT=[]), contextlib.redirect_stdout(io.StringIO()):
        text_numbers.intensity_update()
        theirs_A17 = [l for l in text_numbers.OUT if l and not l.startswith(('==', '   ('))]
    check('text_numbers.intensity_update', nums(lines_A17), nums(theirs_A17))
# cmd_poisson names its runs under experiments/<name>: the report job's, or the walkthrough's
name_A17 = '' if from_report else ('walkthrough' if under_A17(os.path.join('experiments', 'walkthrough'))
                                   else None)
if name_A17 is None:
    print('[skip] cmd_poisson: the runs come from several roots')
elif seg_A17 is not None and need([os.path.join('data', seg_A17[0])]):
    c = experiment_tracking.read_config(runs_A17[seg_A17]['iterative'])
    base_A17 = {k: v for k, v in c['params'].items() if k in ('delta_FR', 'delta_IMU')
            or v != THESIS_PARAMS.get(k)}
    args_A17 = argparse.Namespace(name=name_A17,
                              duration=c['n_frames'] * c['frame_duration'], dt=c['frame_duration'],
                              sequences=seg_A17[3], model=model_A17,
                              modes=','.join(m for m in MODES_A17 if m in runs_A17[seg_A17]),
                              base=','.join(f'{k}={v}' for k, v in base_A17.items()),
                              tag='', no_frames=True, frame_stride=1)
    seen_A17 = []
    try:
        with caught_figures(run_experiments, _run=lambda E_, rc, sf, st: seen_A17.append(rc)) as got_A17:
            run_experiments.cmd_poisson(E, args_A17)
        KEYS_A17 = ('dataset', 'model', 'segment_id', 'n_frames', 'frame_duration', 'poisson', 'params')
        strip = lambda c_: {k: c_.get(k) for k in KEYS_A17}
        mine_A17 = [strip(experiment_tracking.read_config(runs_A17[seg_A17][m]))
                    for m in MODES_A17 if m in runs_A17[seg_A17]]
        same_A17 = [strip(rc.to_dict()) for rc in seen_A17] == mine_A17
        print('[ok] config' if same_A17 else f'[DIFF] config: {[rc.output_dir for rc in seen_A17]}')
        out_A17 = [l for l in got_A17['out'].getvalue().splitlines()
               if l.startswith('  ') and not l.startswith(('  running', '  ->'))]
        check('cmd_poisson stats', nums(stats_A17), nums(out_A17))
        check('cmd_poisson pixels', pix_A17, next(iter(got_A17['pixels'].values()), []))
    except Exception as e:
        print(f'[DIFF] cmd_poisson stats: {type(e).__name__}: {e}')
    finally:
        get_ipython().run_line_magic('matplotlib', 'inline')   # cmd_poisson selected Agg
'''),
]
