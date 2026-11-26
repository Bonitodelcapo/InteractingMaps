A10 = ('A10',)

CELLS = [
('md', A10, r'''
## A10 · Old coordinate sweep (mostly unread)

**Question.** One axis at a time around the step sizes the work started from
(δ_FR 0.5, δ_anchor 0.3), 3 s: δ_anchor, δ_FR, iterations, window (CMax-anchor)
and δ_anchor (thesis+IMU). The sens sweep (A8) replaced it.

**Made by** `report/run_experiments.py::cmd_sweep`
(`--what sweep --name sweep --mode coord --base delta_FR=0.5,delta_IMU=0.3`,
3 s), `run_all.sh` jobs `sweep_<seq>_cmax_<v>` (13 per sequence) and
`sweep_<seq>_imu_<v>` (3 per sequence) → runs and
`sweep_<seq>_{cmax,imu}_<v>.csv` in `experiments/sweep/`.

**Read by** only `report/text_numbers.py::synthetic_beta_original`, which
reads the base point of the two synthetic sequences,
`sweep_{bicycle,street}_cmax_delta_FR=0.5.csv` (that function is unrolled in
A6; here it is only checked). **⚠ Flag:** the other 78 of the 80 `sweep_*`
jobs are read by no table, figure or text number: candidates for removal from
`run_all.sh`.
'''),

('code', A10, r'''
# step 1 -- the runs: the saved sweep rows (Redraw) or the panel's job (Live; 3 s capped by n_frames)
p = params()
rows_A10 = run_sweep(p, 'A10', ['sweep_cmax', 'sweep_imu'])
if READY['A10']:
    have_cols(rows_A10, SWEEP_COLS[:-1], 'A10')
'''),

('code', A10, r'''
%%section A10
# step 2 -- run_all.sh's sweep_* jobs (one per axis value) and the ones anything reads
READ_A10 = [f'sweep_{lab}_cmax_delta_FR=0.5.csv' for lab in ('bicycle', 'street')]  # synthetic_beta_original
jobs_A10 = [f'sweep_{s[3]}_{kind}_{k}={v:g}' for s in SEGMENTS
            for kind, job in (('cmax', 'sweep_cmax'), ('imu', 'sweep_imu'))
            for k, vals in job_axes(job) for v in vals]
on_disk_A10 = {os.path.basename(q) for q in glob.glob(os.path.join('experiments', 'sweep', 'sweep_*.csv'))}
unread_A10 = [j for j in jobs_A10 if f'{j}.csv' not in READ_A10]
print(f'{len(jobs_A10)} sweep_* jobs in run_all.sh; {len(on_disk_A10)} of their csvs on disk here; '
      f'read: {len(READ_A10)} ({", ".join(READ_A10)})')
print(f'unread, candidates for removal from run_all.sh ({len(unread_A10)}):')
for lab in [s[3] for s in SEGMENTS]:
    print(f'   {lab:8s} ' + ' '.join(j.split(f'sweep_{lab}_')[1] for j in unread_A10
                                  if j.startswith(f'sweep_{lab}_')))
'''),

('code', A10, r'''
%%section A10
# step 3 -- the base point (delta_FR 0.5, anchor 0.3, 75 iterations, 20 ms) of every sequence
is_base_A10 = lambda r: (r['model'] == 'thesis_cmax' and abs(float(r['delta_FR']) - 0.5) < 1e-9
                         and abs(float(r['delta_IMU']) - 0.3) < 1e-9 and float(r['dt_ms']) == 20.0
                         and int(float(r['n_iters'])) == ITERS_PER_FRAME)
base_A10 = {r['sequence']: r for r in rows_A10 if is_base_A10(r)}
display(pd.DataFrame([{k: r[k] for k in ('sequence', 'n_frames', 'delta_FR', 'delta_IMU', 'err',
                                         'median', 'dir', 'beta', 'recon_mean', 'blowup')}
                      for r in base_A10.values()]))
print('err, median in deg/s, dir in deg; read by the text: the synthetic rows only')
'''),

('code', A10, r'''
%%section A10
# same-code check -- cmd_sweep (Live); text_numbers.synthetic_beta_original (Redraw)
if p['mode'] == 'Live':
    check_live_sweep(p, 'A10', LIVE_JOB['A10'], rows_A10)
else:
    from text_numbers import SYN as TN_SYN
    mine = []
    for s in TN_SYN:                      # its rows: the whole base-point file of each synthetic sequence
        path = os.path.join('experiments', 'sweep', f'sweep_{s[3]}_cmax_delta_FR=0.5.csv')
        if need([path], [f'sweep_{s[3]}_cmax_delta_FR=0.5']):
            for r in sweep_rows('sweep', (os.path.basename(path),)):
                mine.append(f'{s[3]:8s} dFR {r["delta_FR"]} anchor {r["delta_IMU"]}: '
                            f'beta {float(r["beta"]):.3f}, err {float(r["err"]):.2f}')
    print('\n'.join(mine))
    with redirect(text_numbers, OUT=[]), contextlib.redirect_stdout(io.StringIO()):
        text_numbers.synthetic_beta_original()
        theirs = [l for l in text_numbers.OUT if l and not l.startswith(('==', '   (missing'))]
    check('text_numbers.synthetic_beta_original', nums(mine), nums(theirs))
'''),
]
