A16 = ('A16',)

CELLS = [
('md', A16, r'''
## A16 · Read-out: fft against dct

**Question.** Which exact Poisson solver gives the better image when applied
once, as a read-out, to the G a run produced: fft (periodic border) or dct
(mirrored border)? And does halving the window to 10 ms help every read-out?

**Made by** `report/run_experiments.py::cmd_readout` (`--what readout`),
`run_all.sh` jobs `readout_20ms` (`--sequences boxes,poster,bicycle,street`) and
`readout_10ms` (same, `--dt 0.01 --name readout_dt10ms`) → CMax-anchor runs
(iterative update, adopted step sizes) at track lengths 0.4, 0.8, 1.2 s, and one
csv each: `experiments/readout_fft_vs_dct/readout.csv`,
`experiments/readout_dt10ms/readout.csv`.

**Scoring** (paired, same run, same G): |Pearson r| (`metrics.abs_corr`) of
each read-out against log(1+APS), on the full frame and on `crop_interior`
(10 % border cut). Correlation is blind to the scale and offset the
reconstruction is free in.

**Read by** `report/make_tables.py::table_recon` (tab:recon) and the second
heading of `report/text_numbers.py::intensity`, *"Intensity: 10 against 20 ms
on the same read-outs"* -- that is where the read-out and 10-vs-20 ms lines are
printed (`intensity_update` prints the in-loop poisson job, A17).
'''),

('code', A16, r'''
# step 1 -- the readout csvs (Redraw), or one readout run made now (Live)
from report_runs import ADOPTED
p = params()
CSV_A16 = {20: os.path.join('experiments', 'readout_fft_vs_dct', 'readout.csv'),
           10: os.path.join('experiments', 'readout_dt10ms', 'readout.csv')}
ds_A16, sid_A16, t0_A16, lab_A16 = p['seg']
if p['mode'] == 'Redraw':
    note_redraw(p, 60)                          # the longest track: 1.2 s of 20 ms
    model_A16, dt_A16 = 'thesis_cmax', 0.02     # the job's --model and --dt (20 ms csv)
    READY['A16'] = need(list(CSV_A16.values()), ['readout_20ms', 'readout_10ms'])
else:
    model_A16 = p['model'] if p['model'] != 'cook' else 'thesis_cmax'
    if model_A16 != p['model']:
        print("cook has no delta_IMU for cmd_readout's adopted step sizes; using thesis_cmax")
    dt_A16, nf_A16 = p['dt'], p['n_frames']
    READY['A16'] = False
    if need([os.path.join('data', ds_A16)]):
        rc_A16 = run_cfg(ds_A16, sid_A16, t0_A16, model_A16, dt_A16, nf_A16, out_root=walk_dir('A16'),
                         poisson='iterative', deltas=dict(ADOPTED))
        E.experiment_tracking(rc_A16, save_frames=False)
        print(f"Live: {lab_A16}, {nf_A16} frames of {dt_A16*1000:.0f} ms -> {rc_A16.output_dir}")
        READY['A16'] = True
'''),

('code', A16, r'''
%%section A16
# step 2 -- run_experiments.py::cmd_readout's scoring, unrolled on the panel's segment
from report_runs import aps_frame
from interacting_maps.metrics import abs_corr, crop_interior
from interacting_maps.network_dissertation import solve_poisson_exact
READOUT_COLS_A16 = ['sequence', 'duration_s', 'n_frames', 'variant', 'r_full', 'r_crop',
                    'low_freq', 'contrast']


def score_A16(ds, t0, dur, nf, dt, folder):
    """cmd_readout's rows for one run: the three read-outs of its final G."""
    m = np.load(os.path.join(folder, 'maps_final.npz'))
    aps = aps_frame(ds, t0 + nf * dt)
    if aps is None:
        print(f'  {ds}: no APS, skipped')
        return []
    ref = np.log(aps.astype(np.float64) + 1.0)
    variants = {'iterative': m['I'],
                'fft': solve_poisson_exact(m['G'], 'fft'),
                'dct': solve_poisson_exact(m['G'], 'dct')}
    rows = []
    for name, I in variants.items():
        st = E._intensity_stats(I)
        rows.append([ds, dur, nf, name, round(abs_corr(I, ref), 4),
                     round(abs_corr(crop_interior(I), crop_interior(ref)), 4),
                     round(st['low_freq_share'], 4), round(st['contrast'], 4)])
    return rows


rows_A16 = []
if p['mode'] == 'Live':
    rows_A16 = score_A16(ds_A16, t0_A16, round(nf_A16 * dt_A16, 4), nf_A16, dt_A16, rc_A16.output_dir)
elif need([os.path.join('data', ds_A16)]):      # the runs behind the 20 ms csv, this segment
    durs_A16 = sorted({float(r['duration_s']) for r in csv.DictReader(open(CSV_A16[20]))
                   if r['sequence'] == ds_A16})
    for dur in durs_A16:
        nf = int(round(dur / dt_A16))
        f = run_cfg(ds_A16, sid_A16, t0_A16, model_A16, dt_A16, nf,
                    out_root=os.path.dirname(CSV_A16[20]), poisson='iterative',
                    deltas=dict(ADOPTED)).output_dir
        if need([os.path.join(f, 'maps_final.npz')], ['readout_20ms']):
            rows_A16 += score_A16(ds_A16, t0_A16, dur, nf, dt_A16, f)
display(pd.DataFrame(rows_A16, columns=READOUT_COLS_A16))
if rows_A16:
    wins_A16 = {}
    for r in rows_A16:
        wins_A16.setdefault(r[1], {})[r[3]] = r[5]
    print(f"paired wins on r_crop: fft {sum(w['fft'] > w['dct'] for w in wins_A16.values())}, "
          f"dct {sum(w['fft'] <= w['dct'] for w in wins_A16.values())}")
'''),

('code', A16, r'''
%%section A16
# step 3 -- make_tables.py::table_recon, unrolled: mean |r_crop| per read-out, fft-over-dct wins
from make_tables import tabular
if p['mode'] == 'Redraw':
    body_A16, means_A16 = [], {}
    for dt, path in CSV_A16.items():
        r, runs = {}, {}
        for row in csv.DictReader(open(path)):
            v = abs(float(row['r_crop']))
            r.setdefault(row['variant'], []).append(v)
            runs.setdefault((row['sequence'], row['duration_s']), {})[row['variant']] = v
        mean = {k: float(np.mean(v)) for k, v in r.items()}
        best = max(mean.values())
        cell = lambda k: (rf'$\mathbf{{{mean[k]:.2f}}}$' if dt == 10 and mean[k] == best
                          else f'${mean[k]:.2f}$')
        wins_A16 = sum(v['fft'] > v['dct'] for v in runs.values())
        body_A16.append(f'${dt}$\\,ms & {cell("iterative")} & {cell("fft")} & {cell("dct")} & '
                    f'${wins_A16}/{len(runs)}$ \\\\')
        means_A16[f'{dt} ms'] = {**{k: round(v, 3) for k, v in mean.items()},
                                 'fft > dct': f'{wins_A16}/{len(runs)}'}
    head__A16 = [r'$\Delta t$ & iterative~\eqref{eq:cook_poisson}',
             r'           & exact, FFT & exact, DCT & FFT $>$ DCT \\', r'\midrule']
    table_A16 = tabular('@{}lcccc@{}', head__A16, body_A16)
    display(pd.DataFrame(means_A16).T)
    print(table_A16)
else:
    print('Live: table_recon reads both readout csvs of the report; the mean |r_crop| of this run:')
    print({v: np.mean([r[5] for r in rows_A16 if r[3] == v]).round(3)
           for v in ('iterative', 'fft', 'dct')} if rows_A16 else '(no rows)')
'''),

('code', A16, r'''
%%section A16
# step 4 -- text_numbers.py::intensity, its heading "10 against 20 ms on the same read-outs"
lines_A16 = []
if p['mode'] == 'Redraw':
    rd = text_numbers.rows_of
    r20, r10 = rd(CSV_A16[20]), rd(CSV_A16[10])
    key = lambda r: (r['sequence'], r['duration_s'], r['variant'])
    a = {key(r): abs(float(r['r_crop'])) for r in r20}
    b = {key(r): abs(float(r['r_crop'])) for r in r10}
    for v in ('iterative', 'fft', 'dct'):
        ks = [k for k in a if k[2] == v and k in b]
        if ks:
            lines_A16.append(f'{v:9s}: 10 ms better on {sum(b[k] > a[k] for k in ks)}/{len(ks)} runs; '
                             f'mean {np.mean([a[k] for k in ks]):.2f} -> {np.mean([b[k] for k in ks]):.2f}')
    for name, tab in (('20 ms', a), ('10 ms', b)):
        pairs = {(k[0], k[1]) for k in tab}
        for x, y in (('fft', 'iterative'), ('dct', 'iterative'), ('fft', 'dct')):
            n = [q for q in pairs if (q + (x,)) in tab and (q + (y,)) in tab]
            if n:
                lines_A16.append(f'{name}: {x} beats {y} on '
                                 f'{sum(tab[q + (x,)] > tab[q + (y,)] for q in n)}/{len(n)}')
    print('\n'.join(lines_A16) or '(no rows)')
else:
    print('Live: the text numbers read the report runs; switch to Redraw to see them.')
'''),

('code', A16, r'''
%%section A16
# same-code check -- rows vs the csv / cmd_readout, table vs table_recon, text vs text_numbers
import io, argparse
if p['mode'] == 'Redraw':
    want_A16 = {(r['sequence'], float(r['duration_s']), r['variant']): r
            for r in csv.DictReader(open(CSV_A16[20]))}
    mine_A16 = {(r[0], float(r[1]), r[3]): r for r in rows_A16}
    if mine_A16:
        check('readout.csv rows', [x for k in sorted(mine_A16) for x in mine_A16[k][4:]],
              [float(want_A16[k][c]) if k in want_A16 else np.nan
               for k in sorted(mine_A16) for c in READOUT_COLS_A16[4:]])
    check('table_recon', nums(table_A16.splitlines()), nums(make_tables.table_recon().splitlines()))
    try:
        with redirect(text_numbers, OUT=[]), contextlib.redirect_stdout(io.StringIO()):
            text_numbers.intensity()
            out_A16 = list(text_numbers.OUT)
        i = next(k for k, l in enumerate(out_A16) if l.startswith('== Intensity: 10 against 20 ms'))
        theirs_A16 = [l for l in out_A16[i + 1:] if l and not l.startswith('   (missing')]
        check('text_numbers.intensity 10 vs 20 ms', nums(lines_A16), nums(theirs_A16))
    except Exception as e:
        print(f'[DIFF] text_numbers.intensity 10 vs 20 ms: {type(e).__name__}: {e}')
else:
    seen_A16 = []
    args_A16 = argparse.Namespace(name='walkthrough/A16', sequences=lab_A16, durations=str(round(nf_A16 * dt_A16, 4)),
                              dt=dt_A16, model=model_A16, no_frames=True, frame_stride=1)
    with redirect(run_experiments, _run=lambda E_, rc_, sf, st: seen_A16.append(rc_)), \
         contextlib.redirect_stdout(io.StringIO()):
        run_experiments.cmd_readout(E, args_A16)
    same_A16 = [os.path.normpath(r.output_dir) for r in seen_A16] == [os.path.normpath(rc_A16.output_dir)]
    print('[ok] config' if same_A16 else f'[DIFF] config: {[r.output_dir for r in seen_A16]}')
    theirs_A16 = list(csv.DictReader(open(os.path.join(walk_dir('A16'), 'readout.csv'))))
    check('cmd_readout rows', [x for r in rows_A16 for x in r[4:]],
          [float(r[c]) for r in theirs_A16 for c in READOUT_COLS_A16[4:]])
'''),
]
