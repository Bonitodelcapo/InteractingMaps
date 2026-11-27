A6 = ('A6',)

CELLS = [
('md', A6, r'''
## A6 · β vs speed

**Question.** Does the scale β = |ω_ref| / |ω_est| depend on the rotation speed?
Seven segments of the two real sequences (boxes, poster), 3 s each, under three
anchor settings: CMax-anchor at δ_FR 0.5 / anchor 0.3 (the original step sizes),
CMax-anchor at δ_FR 0.1 / anchor 0.5 (the adopted ones), CMax-inloop at δ_FR 0.5.

**Made by** `report/run_experiments.py::cmd_beta`, one run per
(`report_runs.BETA_SEGMENTS` × `BETA_CONFIGS`) pair, `run_all.sh` jobs
`beta_<i>` (i = 3·segment + config) → runs in `results/` (150 frames). Jobs 4
and 16 are skipped there while config.py adopts δ_FR 0.1 / anchor 0.5: those runs
are the `main_<seq>_thesis_cmax` runs of A2.

**Read by** `report/make_figures.py::fig_beta_speed` (fig:beta_speed) and
`report/text_numbers.py::beta_speed`. `text_numbers.py::synthetic_beta_original`
(the synthetic β at the original step sizes, from the A10 sweep base point) is
quoted next to it. The 3 s runs are also read by A2's `poster_segments`.

**Steps:** the runs (found, or one made now) → per run: speed and mean β →
the speed as the figure takes it → the figure → the text numbers → same-code check.
Live makes the panel's segment only, if it is one of the seven: `one point` with
the adopted config, `one axis` / `full` with the three `BETA_CONFIGS`.
'''),

('code', A6, r'''
# the A6 form: which anchor configurations on which beta segments
from report_runs import BETA_CONFIGS
form('A6', [Knob('configs', BETA_CONFIGS, 'many', choices=BETA_CONFIGS),
            Knob('segments', BETA_SEGMENTS, 'many', choices=BETA_SEGMENTS),
            Knob('n_frames', 150), Knob('dt', 0.02)])
'''),

('code', A6, r'''
# step 1 -- the runs of the form's (segment x config) pairs: found (Redraw) or made (Live)
p = params('A6')
rep_A6 = is_report('A6')                        # the report-only steps run only then
PAIRS_A6 = [(s, c) for s in BETA_SEGMENTS for c in BETA_CONFIGS]   # cmd_beta's order: index i
nf_A6, runs_A6, miss = p['n_frames'], {}, []


def beta_run_dir_A6(seg, model, d_fr, d_a):
    """The saved run of one (segment, config): every step size as cmd_beta's run_cfg sets
    it (config.py's, with the config's delta_FR / anchor), so a sweep run with another
    rate never stands in; results/ first."""
    prm = {**make_tables.expected_params(model), 'delta_FR': d_fr,
           **({'delta_IMU': d_a} if d_a is not None else {})}
    with contextlib.redirect_stdout(io.StringIO()):
        return next((h for h in (experiment_tracking.find_run(r, seg[0], model, seg[1], nf_A6, prm,
                                                               dt=p['dt'], t_start=seg[2])
                                 for r in roots()) if h), None)


for seg in p['segments']:
    for model, d_fr, d_a in p['configs']:
        if p['mode'] == 'Redraw':
            d = beta_run_dir_A6(seg, model, d_fr, d_a)
        elif need([os.path.join('data', seg[0])]):      # cmd_beta's run_cfg call
            d = saved_or_run(run_cfg(*seg, model, p['dt'], nf_A6, out_root='experiments/walkthrough',
                                     delta_FR=d_fr, delta_IMU=d_a))
        else:
            d = None
        if d is None:
            miss.append((seg, (model, d_fr, d_a)))
        else:
            runs_A6[seg, (model, d_fr, d_a)] = d
print(f'{len(runs_A6)} of {len(runs_A6) + len(miss)} runs found')
for seg, cfg in miss:
    i = PAIRS_A6.index((seg, cfg)) if (seg, cfg) in PAIRS_A6 else None
    print(f'no saved run for {seg[0][:6]} {seg[1]} {cfg}' + (f' (run_all.sh job beta_{i})' if i is not None else ''))
READY['A6'] = bool(runs_A6)
for (seg, cfg), d in runs_A6.items():
    print(f'{seg[0][:6]} {seg[1]} {cfg[0]:15s} dFR {cfg[1]} anchor {cfg[2]}: {d}')
'''),

('code', A6, r'''
%%section A6
# step 2 -- per run: speed |w_ref| and mean beta, from its tracking.csv (text_numbers.run)
pts_A6 = []
for (seg, (model, d_fr, d_a)), d in runs_A6.items():
    tr = tracking(d)
    pts_A6.append(dict(segment=f'{seg[0][:6]} {seg[1]}', model=model, dFR=d_fr, anchor=d_a,
                       speed_deg_s=float(mag(tr['gt']).mean() * DEG), beta=float(tr['beta'].mean()),
                       beta_summary=experiment_tracking.read_summary(d)['mean_beta'],
                       err_deg_s=float(tr['err'].mean()), dir_deg=float(tr['dir'].mean())))
display(pd.DataFrame(pts_A6).round(3))
'''),

('code', A6, r'''
%%section A6
# step 3 -- the speed as fig_beta_speed takes it: mean |reference| over the
# segment's 150 frames, from the dataset (make_figures.reference_omega); where
# the dataset is not on this machine, the same reference from the runs' tracking.csv
speed_A6, speed_src_A6 = {}, {}
for seg in BETA_SEGMENTS:
    have = [d for (s, _c), d in runs_A6.items() if s == seg]
    if not have:
        continue
    if os.path.isdir(os.path.join('data', seg[0])):
        ref, _ = reference_and_gyro(seg + ('',), p['dt'], nf_A6)
        speed_A6[seg], speed_src_A6[seg] = float(to_deg_s(mag(ref).mean())), 'dataset'
    elif have:
        speed_A6[seg] = float(to_deg_s(mag(tracking(have[0])['gt']).mean()))
        speed_src_A6[seg] = 'tracking.csv'
for seg, w in speed_A6.items():
    print(f'{seg[0][:6]} {seg[1]}: |w| = {w:6.1f} deg/s ({speed_src_A6[seg]})')
'''),

('code', A6, r'''
%%section A6
# step 4 -- make_figures.py::fig_beta_speed, unrolled: beta against speed, one
# series per BETA_CONFIGS entry, Pearson r and a straight-line fit
from interacting_maps.style import (use_style, save, PAPER_WIDTH, NEUTRAL_COLOR, LABEL_BETA,
                                    MODELS as STYLE_MODELS_A6)
series_A6 = {'default': ('thesis_cmax', 0.5, 0.3, False),     # fixed in fig_beta_speed
             'tuned':   ('thesis_cmax', 0.1, 0.5, True),
             'inloop':  ('thesis_cmax_v2', 0.5, None, True)}
data_A6 = {k: ([], []) for k in series_A6}
for seg in BETA_SEGMENTS:
    for key, (model, d_fr, d_a, _) in series_A6.items():
        d = runs_A6.get((seg, (model, d_fr, d_a)))
        if d is None:
            continue
        data_A6[key][0].append(speed_A6[seg])
        data_A6[key][1].append(experiment_tracking.read_summary(d)['mean_beta'])
with use_style('paper'):
    fig_A6, ax = plt.subplots(figsize=(PAPER_WIDTH, 2.5))
    for key, (model, d_fr, _, filled) in series_A6.items():
        x, y = np.array(data_A6[key][0]), np.array(data_A6[key][1])
        name, col = STYLE_MODELS_A6[model]
        if len(x) < 2:                                  # Live: a point, no fit
            if len(x):
                ax.scatter(x, y, s=26, zorder=3, color=col if filled else 'none', edgecolors=col,
                           linewidths=0.8, label=rf'{name}, $\delta_{{FR}}={d_fr}$ (no fit: 1 point)')
            continue
        r = np.corrcoef(x, y)[0, 1]
        ax.scatter(x, y, s=26, zorder=3, color=col if filled else 'none', edgecolors=col,
                   linewidths=0.8, label=rf'{name}, $\delta_{{FR}}={d_fr}$  ($r={r:+.2f}$)')
        xs = np.linspace(x.min(), x.max(), 20)
        ax.plot(xs, np.polyval(np.polyfit(x, y, 1), xs), color=col, lw=1.0, alpha=0.55,
                ls='-' if filled else '--', zorder=2)
        print(f'{key:8s} {name} dFR {d_fr}: n = {len(x)}, r = {r:+.2f}')
    ax.axhline(1.0, color=NEUTRAL_COLOR, lw=0.8, ls=':', zorder=1)
    ax.set_xlabel(r'rotation speed $\|\omega\|$ (°/s)')
    ax.set_ylabel(LABEL_BETA)
    ax.legend(loc='upper left')
    pts_fig_A6 = [np.asarray(c.get_offsets(), float) for c in ax.collections]
    lines_fig_A6 = [l.get_ydata() for l in ax.lines]
    save(fig_A6, os.path.join(walk_dir('A6'), 'fig_beta_speed.png'), show=True)
'''),

('code', A6, r'''
%%section A6
# step 5 -- the text numbers (report configuration only: they read the report's runs)
from scipy import stats
from text_numbers import rows_of, f as tn_f
lines_A6 = {'beta_speed': [], 'synthetic_beta_original': []}
if rep_A6:
    L = lines_A6['beta_speed']          # text_numbers.beta_speed
    tn_data = {}
    for cfg in BETA_CONFIGS:
        model, d_fr, d_a = cfg
        pts = []
        for seg in BETA_SEGMENTS:
            d = runs_A6.get((seg, cfg))
            if d is not None:
                t_ = tracking(d)
                pts.append((mag(t_['gt']).mean(), t_['beta'].mean(), t_['err'].mean(),
                            t_['dir'].mean(), t_['est']))
        tn_data[cfg] = pts
        if len(pts) < 3:
            continue
        w, b = np.array([q[0] for q in pts]), np.array([q[1] for q in pts])
        r, pv = stats.pearsonr(w, b)
        L.append(f'{model} dFR {d_fr} anchor {d_a}: r = {r:+.2f} (p = {pv:.3g}, n = {len(w)}), '
                 f'beta {b.min():.2f}-{b.max():.2f}, beta at slowest {b[np.argmin(w)]:.2f}, '
                 f'at fastest {b[np.argmax(w)]:.2f}; speeds {w.min()*DEG:.1f}-{w.max()*DEG:.1f} deg/s')
    orig, adopt, inl = (tn_data.get(c, []) for c in BETA_CONFIGS)
    if len(orig) >= 2:                  # the original fit, extrapolated to the synthetic speed
        fit = np.polyfit([q[0] for q in orig], [q[1] for q in orig], 1)
        syn = [run_dir(s, 'thesis_cmax', 75) for s in SEGMENTS if s[3] in ('bicycle', 'street')]
        ws = np.mean([mag(tracking(d)['gt']).mean() for d in syn if d is not None])
        L.append(f'original fit predicts beta {np.polyval(fit, ws):.2f} at the synthetic {ws*DEG:.1f} deg/s')
    if len(inl) == len(orig) == len(adopt):
        for (o, a, v, seg) in zip(orig, adopt, inl, BETA_SEGMENTS):
            L.append(f'{seg[0][:6]} {seg[1]}: dir CMax-anchor {o[3]:.1f} / {a[3]:.1f}, CMax-inloop {v[3]:.1f}; '
                     f'err CMax-anchor {o[2]:.1f} / {a[2]:.1f}, CMax-inloop {v[2]:.1f} (original / adopted)')
    L = lines_A6['synthetic_beta_original']   # text_numbers.synthetic_beta_original (A10's sweep csv)
    for s in [s for s in SEGMENTS if s[3] in ('bicycle', 'street')]:
        path = os.path.join('experiments', 'sweep', f'sweep_{s[3]}_cmax_delta_FR=0.5.csv')
        need([path], [f'sweep_{s[3]}_cmax_delta_FR=0.5'])
        for r in rows_of(path):
            L.append(f'{s[3]:8s} dFR {r["delta_FR"]} anchor {r["delta_IMU"]}: beta {tn_f(r, "beta"):.3f}, '
                     f'err {tn_f(r):.2f}')
    for k, ls in lines_A6.items():
        print(f'== {k}'); print('\n'.join(ls) or '(no runs)')
else:
    print('[skip] report-only step')
'''),

('code', A6, r'''
%%section A6
# same-code check -- beta per run vs read_summary; cmd_beta's config (Live);
# fig_beta_speed and text_numbers (Redraw)
import io, argparse
check('beta per run vs read_summary', {f"{q['segment']} {q['model']} {q['dFR']}": q['beta'] for q in pts_A6},
      {f"{q['segment']} {q['model']} {q['dFR']}": q['beta_summary'] for q in pts_A6})
both = [s for s in speed_A6 if speed_src_A6[s] == 'dataset' and any(k[0] == s for k in runs_A6)]
if both:                                # the run's stored reference is the dataset's
    check('speed: dataset vs tracking.csv',
          [speed_A6[s] for s in both],
          [float(to_deg_s(mag(tracking(next(d for (s_, _c), d in runs_A6.items() if s_ == s))['gt']).mean()))
           for s in both])
if p['mode'] == 'Live':
    # cmd_beta with its run step replaced by a recorder: same RunConfig per index?
    for (seg, cfg), d in runs_A6.items():
        if (seg, cfg) not in PAIRS_A6:
            print(f'[skip] cmd_beta config: {cfg} is not a BETA_CONFIGS entry')
            continue
        seen = []
        def record(E_, rc, save_frames, stride):
            seen.append(rc)
            return experiment_tracking.read_summary(d), 0
        args = argparse.Namespace(dt=p['dt'], duration=nf_A6 * p['dt'],
                                  index=PAIRS_A6.index((seg, cfg)), no_frames=True, frame_stride=1)
        with redirect(run_experiments, _run=record), contextlib.redirect_stdout(io.StringIO()):
            run_experiments.cmd_beta(E, args)
        mine = run_cfg(*seg, cfg[0], p['dt'], nf_A6, delta_FR=cfg[1], delta_IMU=cfg[2]).to_dict()
        a = seen[0].to_dict() if seen else {}
        differ = sorted(k for k in mine if k != 'out_root' and a.get(k) != mine[k])
        print(f'[ok] cmd_beta config {cfg}' if seen and not differ
              else f'[DIFF] cmd_beta config {cfg}: {differ or "no run"}')
if not rep_A6:
    print('[skip] not the report configuration')
else:
    # fig_beta_speed reads the reference from the dataset (make_figures.reference_omega).
    # A dataset not on this machine: its reference is taken from the segment's run
    # (tracking.csv stores the same reference, see the speed check above).
    ref_fn = make_figures.reference_omega
    def ref_or_run(E_, ds, t0, n=150):
        if os.path.isdir(os.path.join('data', ds)):
            return ref_fn(E_, ds, t0, n)
        d = next((d for (s, _c), d in runs_A6.items() if s[0] == ds and abs(s[2] - t0) < 1e-9), None)
        return (tracking(d)['gt'][:n] if d else np.full((n, 3), np.nan)), 'groundtruth'
    gone = sorted({s[0] for s in BETA_SEGMENTS if not os.path.isdir(os.path.join('data', s[0]))})
    if gone:
        print(f'fig_beta_speed check: {", ".join(gone)} not here, its reference read from the runs')
    captured = []
    def keep(fig, path, show=False):
        captured.append(fig)
    with redirect(make_figures, FIG=os.path.join(walk_dir('A6'), 'check'), save=keep,
                  reference_omega=ref_or_run), contextlib.redirect_stdout(io.StringIO()):
        make_figures.fig_beta_speed(E)
    ax_ = captured[0].axes[0]
    flat = lambda ls: np.concatenate([np.ravel(x) for x in ls]) if ls else np.array([])
    check('fig_beta_speed points', flat(pts_fig_A6),
          flat([np.asarray(c.get_offsets(), float) for c in ax_.collections]))
    check('fig_beta_speed lines', flat(lines_fig_A6), flat([l.get_ydata() for l in ax_.lines]))
    for f_ in captured:
        plt.close(f_)
    for name in lines_A6:
        with redirect(text_numbers, OUT=[]), contextlib.redirect_stdout(io.StringIO()):
            getattr(text_numbers, name)()
            theirs = [l for l in text_numbers.OUT if l and not l.startswith(('==', '   (missing'))]
        check(f'text_numbers.{name}', nums(lines_A6[name]), nums(theirs))
'''),
]
