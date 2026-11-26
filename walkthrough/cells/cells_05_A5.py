A5 = ('A5',)

CELLS = [
('md', A5, r'''
## A5 · Anchor alone (baseline)

**Question.** What do the two anchor signals score on their own, with no
network? CMax-anchor feeds the network a per-frame CMax ω, thesis+IMU feeds it
the gyro; if the network cannot beat its own anchor, the maps add nothing to ω.

**Made by** `report/run_experiments.py::cmd_baseline` (`--duration 1.5`),
`run_all.sh` job `baseline` → `report/experiments/baseline_dt20ms.csv` (one row
per sequence and source, cmax / gyro) and `baseline_dt20ms_<seq>.npz` (the
per-frame ω series of CMax, gyro and reference, and the CMax IWE at the first,
middle and last frame).

**Read by** `report/make_tables.py::table_baseline` (tab:baseline; its
*network* column is the CMax-anchor row of the A3 csvs `dissoc_1p5s_*`),
`report/make_figures.py::fig_cmax_baseline` (**not in the tex**: drawn, but no
`\includegraphics` uses it) and `report/text_numbers.py::maps_add`. The npz
reference also gives the speed under each sequence name of tab:main (A3).

**Steps** (those of `cmd_baseline`'s per-frame loop, then of the readers): the
segment's events and the CMax estimator → per frame: reference, CMax with a warm
start from the previous frame, gyro → `compute_metrics` per frame and source →
the csv rows → table, figure, text → same-code check. Redraw shows the stored
series (it does not rerun CMax); Live runs the loop now on the panel's segment.
The panel's model and step sizes do not enter: there is no network here.
'''),

('code', A5, r'''
# step 1 -- the inputs: the stored baseline (Redraw) or the segment's data (Live)
import time
p = params()
ds_A5, sid_A5, t0_A5, lab_A5 = p['seg']
BASE_A5 = os.path.join('report', 'experiments')
if p['mode'] == 'Redraw':
    note_redraw(p, 75)
    nf_A5, dt_A5, src_A5 = 75, 0.02, BASE_A5
    READY['A5'] = need([os.path.join(BASE_A5, 'baseline_dt20ms.csv'),
                        os.path.join(BASE_A5, f'baseline_dt20ms_{lab_A5}.npz')], ['baseline'])
else:
    nf_A5, dt_A5, src_A5 = min(p['n_frames'], 75), p['dt'], walk_dir('A5')
    print(f"Live: cmd_baseline's loop on {lab_A5}, {nf_A5} frames of {dt_A5*1000:g} ms "
          f"(no network: the panel's model and step sizes are not used)")
    READY['A5'] = need([os.path.join('data', ds_A5)])
csv_A5 = os.path.join(src_A5, f'baseline_dt{int(dt_A5*1000)}ms.csv')
npz_A5 = csv_A5.replace('.csv', f'_{lab_A5}.npz')
print(f'series: {npz_A5}')
'''),

('code', A5, r'''
%%section A5
# step 2 -- run_experiments.py::cmd_baseline: the segment's events and the CMax estimator
from interacting_maps.data_loader import load_events_fast, undistort_events, CameraCalibration
from interacting_maps.angular_velocity import CMaxAngularVelocity
if p['mode'] == 'Live':
    pp_A5 = E.get_dataset_paths(ds_A5)
    calib_A5 = CameraCalibration(pp_A5['calib'])
    gt_A5, imu_A5 = E.load_groundtruth(pp_A5['groundtruth']), E.load_imu(pp_A5['imu'])
    om_A5 = E.load_omega_gt(pp_A5.get('omega_gt'))
    # the same undistorted events experiment_tracking gives CMax
    ev_A5 = undistort_events(load_events_fast(pp_A5['events'], t_start=t0_A5,
                                              duration=nf_A5 * dt_A5 + 0.1), calib_A5)
    est_A5 = CMaxAngularVelocity(180, 240, calib_A5.fx, calib_A5.fy,
                                 calib_A5.cx, calib_A5.cy, use_polarity=True)
    print(f'{len(ev_A5)} undistorted events from t = {t0_A5} s over {nf_A5*dt_A5 + 0.1:.2f} s; '
          f'CMax on a 180 x 240 sensor, polarity used')
else:
    print(f'Redraw: the events were read when the baseline job ran; '
          f'the series it stored are in {npz_A5}')
'''),

('code', A5, r'''
%%section A5
# step 3 -- cmd_baseline's per-frame loop: reference, CMax (warm start), gyro; the npz
if p['mode'] == 'Live':
    w_prev = np.zeros(3)                                # CMax starts from zero at frame 0
    keep_A5 = (0, nf_A5 // 2, nf_A5 - 1)                # frames whose IWE is saved
    ser_A5, iwes = {'cmax': [], 'gyro': [], 'ref': []}, []
    t_wall = time.time()
    for k in range(nf_A5):
        t_lo, t_hi = t0_A5 + k * dt_A5, t0_A5 + (k + 1) * dt_A5
        w_ref, ref_src_A5 = E.get_reference_omega(gt_A5, imu_A5, t_lo, t_hi, om_A5)
        win = ev_A5[(ev_A5[:, 0] >= t_lo) & (ev_A5[:, 0] < t_hi)]
        if len(win) > 10:                               # warm start: last frame's CMax
            w_prev = est_A5.estimate(win, t_ref=0.5 * (t_lo + t_hi), omega_init=w_prev)
        w_gyro = E.get_gyro_for_frame(imu_A5, t_lo, t_hi)
        for src, w in (('cmax', w_prev), ('gyro', w_gyro), ('ref', w_ref)):
            ser_A5[src].append(w)
        if k in keep_A5:
            iwes.append(est_A5.last_iwe if len(win) > 10 else np.zeros((180, 240)))
    secs_A5 = time.time() - t_wall
    np.savez_compressed(npz_A5, frames=keep_A5, iwe=np.array(iwes), ref_src=ref_src_A5,
                        **{s: np.array(v) for s, v in ser_A5.items()})
    print(f'{nf_A5} frames in {secs_A5:.0f} s -> {npz_A5}')
else:
    secs_A5 = None
d_A5 = np.load(npz_A5)
series_A5 = {s: d_A5[s] for s in ('cmax', 'gyro', 'ref')}
print(f"series (frames x xyz): {series_A5['ref'].shape}, reference from {d_A5['ref_src']}; "
      f"IWE kept at frames {d_A5['frames'].tolist()}. First 4 frames, deg/s:")
print(pd.DataFrame(to_deg_s(np.hstack([series_A5[s][:4] for s in ('ref', 'cmax', 'gyro')])),
                   columns=[f'{s}_{a}' for s in ('ref', 'cmax', 'gyro') for a in 'xyz']).round(2))
'''),

('code', A5, r'''
%%section A5
# step 4 -- compute_metrics per frame and source (evaluation.compute_metrics), then the csv rows
BL_COLS_A5 = ['err', 'median', 'dir', 'beta']
acc_A5 = {src: np.array([E.compute_metrics(w, r) for w, r in zip(series_A5[src], series_A5['ref'])])
          for src in ('cmax', 'gyro')}                  # (n, 3): err deg/s, dir deg, beta
rows_A5 = {src: dict(zip(BL_COLS_A5, (round(a[:, 0].mean(), 2), round(np.median(a[:, 0]), 2),
                                      round(a[:, 1].mean(), 2), round(a[:, 2].mean(), 3))))
           for src, a in acc_A5.items()}
print('per frame, first 5 (err deg/s, dir deg, beta):')
print(pd.DataFrame(np.hstack([acc_A5['cmax'][:5], acc_A5['gyro'][:5]]),
                   columns=['cmax_err', 'cmax_dir', 'cmax_beta',
                            'gyro_err', 'gyro_dir', 'gyro_beta']).round(3))
display(pd.DataFrame([{'sequence': ds_A5, 'segment': sid_A5, 'source': s, **r}
                      for s, r in rows_A5.items()]))
if p['mode'] == 'Live':                                  # cmd_baseline's csv, this segment only
    with open(csv_A5, 'w', newline='', encoding='utf-8') as fh:
        wr = csv.writer(fh)
        wr.writerow(['sequence', 'segment', 'source', 'dt_ms', 'n_frames',
                     'err', 'median', 'dir', 'beta', 'secs'])
        for src, r in rows_A5.items():
            wr.writerow([ds_A5, sid_A5, src, int(dt_A5 * 1000), nf_A5,
                         *[r[c] for c in BL_COLS_A5], round(secs_A5)])
    print(f'-> {csv_A5}')
'''),

('md', A5, r'''
### tab:baseline -- `make_tables.py::table_baseline`

Per sequence: the mean ω error of CMax alone and of the gyro alone (this
section's csv), and of the CMax-anchored network (the `thesis_cmax` row of A3's
`dissoc_1p5s_*.csv`). Bold marks the lower of CMax alone and network. In Live
the CMax and gyro columns hold only the panel's segment, from the Live csv.
'''),

('code', A5, r'''
%%section A5
# step 5 -- make_tables.py::table_baseline, unrolled (fmt / tabular are its formatters)
base_A5 = {}
if os.path.exists(csv_A5):
    for r in csv.DictReader(open(csv_A5)):
        base_A5[(r['sequence'], r['source'])] = float(r['err'])
net_A5 = make_tables.load(['dissoc_1p5s_*.csv'])        # the network column: A3's csvs
body_A5, cells_A5 = [], []
for i, (ds, disp, _lab, _extra) in enumerate(make_tables.SEQS):
    if i and ds in make_tables.SYNTHETIC and make_tables.SEQS[i - 1][0] not in make_tables.SYNTHETIC:
        body_A5.append(r'\midrule')
    cm, gy = base_A5.get((ds, 'cmax')), base_A5.get((ds, 'gyro'))
    nw = net_A5.get((ds, 'thesis_cmax'))
    nw = float(nw['err']) if nw else None
    best = min(v for v in (cm, nw) if v is not None) if (cm or nw) else None
    cell = lambda v: (rf'\textbf{{{make_tables.fmt(v, 2)}}}' if v is not None and v == best
                      else make_tables.fmt(v, 2))
    body_A5.append(f'{disp:<18}& {cell(cm)} & {make_tables.fmt(gy, 2)} & {cell(nw)} \\\\')
    cells_A5.append([_lab, cm, gy, nw])
head_A5 = [r'sequence & CMax alone & gyro alone & network \\',
        r' & [$^\circ$/s] & [$^\circ$/s] & [$^\circ$/s] \\', r'\midrule']
table_A5 = make_tables.tabular('@{}lccc@{}', head_A5, body_A5)
display(pd.DataFrame(cells_A5, columns=['sequence', 'CMax alone (deg/s)', 'gyro alone (deg/s)',
                                     'network (deg/s)']))
print(table_A5)
'''),

('md', A5, r'''
### fig_cmax_baseline -- `make_figures.py::fig_cmax_baseline` (not in the tex)

CMax alone and the gyro against the reference, ω per axis, with the CMax IWE at
the three stored frames (dashed lines). The report draws boxes and bicycle;
Live draws the panel's segment from the Live npz.
'''),

('code', A5, r'''
%%section A5
# step 6 -- make_figures.py::fig_cmax_baseline, unrolled
from interacting_maps.style import use_style, save, frame_times, PAPER_WIDE, NEUTRAL_COLOR
from interacting_maps.viz import draw_map, draw_omega
labels_A5 = ('boxes', 'bicycle') if p['mode'] == 'Redraw' else (lab_A5,)
with use_style('paper'):
    fig_A5 = plt.figure(figsize=(PAPER_WIDE, 5.2))
    for sub, lab in zip(fig_A5.subfigures(1, len(labels_A5)), labels_A5):
        path = os.path.join(src_A5, f'baseline_dt{int(dt_A5*1000)}ms_{lab}.npz')
        if not need([path], ['baseline']):
            continue
        d = np.load(path)
        t = frame_times(dt_A5, len(d['ref']))
        gs = sub.add_gridspec(4, 3, height_ratios=(1, 1, 1, 1.3))
        for i in range(3):
            ax = sub.add_subplot(gs[i, :])
            draw_omega(ax, t, d['cmax'][:, i], ref=d['ref'][:, i], imu=d['gyro'][:, i],
                       axis=i, model='thesis_cmax', ref_src=str(d['ref_src']),
                       label='CMax alone')
            for k in d['frames']:
                ax.axvline(t[k], color=NEUTRAL_COLOR, lw=0.6, ls='--')
            if i < 2:
                ax.tick_params(labelbottom=False)
                ax.set_xlabel('')
        # one IWE per stored image (a Live run of < 3 frames stores fewer than 3 frames)
        for j, k in enumerate(d['frames'][:len(d['iwe'])]):
            draw_map(sub.add_subplot(gs[3, j]), 'IWE', d['iwe'][j],
                     title=f'IWE  t = {t[k]:.2f} s')
        sub.suptitle(lab)
        sub.legend(*sub.axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=3)
    lines_fig_A5 = [l.get_ydata() for ax in fig_A5.axes for l in ax.lines]
    imgs_fig_A5 = [np.asarray(im.get_array(), float) for ax in fig_A5.axes for im in ax.images]
    save(fig_A5, os.path.join(walk_dir('A5'), 'fig_cmax_baseline.png'), show=True)
'''),

('code', A5, r'''
%%section A5
# step 7 -- text_numbers.py::maps_add (Redraw only: it reads the report's csvs)
from text_numbers import rows_of, sens_rows, axis as tn_axis, f as tn_f
lines_A5 = []
if p['mode'] == 'Redraw':
    L = lines_A5
    bl_A5 = {(r['sequence'], r['source']): r for r in rows_of(os.path.join(BASE_A5, 'baseline_dt20ms.csv'))}
    nt_A5 = {r['sequence']: r for r in rows_of(os.path.join(BASE_A5, 'dissoc_1p5s_*_thesis_cmax.csv'))}
    for seg in SEGMENTS:                       # network against its own anchor, mean and median
        b, n = bl_A5.get((seg[0], 'cmax')), nt_A5.get(seg[0])
        if b and n:
            L.append(f'{seg[3]:8s} CMax alone {tn_f(b):.2f} (median {tn_f(b, "median"):.2f}), '
                     f'network {tn_f(n):.2f} (median {tn_f(n, "median"):.2f}): '
                     f'mean {100*(1-tn_f(n)/tn_f(b)):+.0f}%, '
                     f'median {100*(1-tn_f(n, "median")/tn_f(b, "median")):+.0f}%')
    sens_A5 = sens_rows()                         # the delta_VFG axis of the A8 sweep
    L.append('delta_VFG (CMax-anchor):')
    for seg in SEGMENTS:
        a, b = tn_axis(sens_A5, 'delta_VFG', seg[0]), bl_A5.get((seg[0], 'cmax'))
        if not a or not b:
            continue
        lo, hi = a[min(a)], a[max(a)]
        dirs = [tn_f(r, 'dir') for r in a.values()]
        L.append(f'  {seg[3]:8s} at {min(a):g}: err {tn_f(lo):.2f} vs CMax alone {tn_f(b):.2f} '
                 f'(diff {tn_f(lo)-tn_f(b):+.2f}); dir range {max(dirs)-min(dirs):.2f} deg; '
                 f'beta {tn_f(lo, "beta"):.2f} -> {tn_f(hi, "beta"):.2f}; at {max(a):g} '
                 f'{100*(1-tn_f(hi)/tn_f(b)):+.0f}% vs CMax alone')
    print('== maps_add (err in deg/s)')
    print('\n'.join(lines_A5) or '(no rows)')
else:
    print('Live: the text numbers read the report csvs; switch to Redraw to see them.')
'''),

('code', A5, r'''
%%section A5
# same-code check -- cmd_baseline itself (Live); the csv, table_baseline,
# fig_cmax_baseline and text_numbers.maps_add (Redraw)
import io, argparse
if p['mode'] == 'Live':
    # cmd_baseline on the panel's segment only (SEGMENTS redirected), same frames,
    # its csv and npz into walk_dir('A5')/check
    out = os.path.join(walk_dir('A5'), 'check')
    os.makedirs(out, exist_ok=True)
    args = argparse.Namespace(dt=dt_A5, duration=nf_A5 * dt_A5)
    with redirect(run_experiments, SEGMENTS=[p['seg']], OUTDIR=out), \
         contextlib.redirect_stdout(io.StringIO()):
        run_experiments.cmd_baseline(E, args)
    theirs = {r['source']: r for r in csv.DictReader(open(os.path.join(out, os.path.basename(csv_A5))))}
    for src, r in rows_A5.items():
        check(f'cmd_baseline row {src}', r, {c: float(theirs[src][c]) for c in BL_COLS_A5})
    d2 = np.load(os.path.join(out, os.path.basename(npz_A5)))
    for s in ('cmax', 'gyro', 'ref'):
        check(f'cmd_baseline series {s}', series_A5[s], d2[s])
else:
    # the rows recomputed from the stored series against the csv the job wrote
    rep = {r['source']: r for r in csv.DictReader(open(csv_A5)) if r['sequence'] == ds_A5}
    for src, r in rows_A5.items():
        check(f'baseline row {src}', r, {c: float(rep[src][c]) for c in BL_COLS_A5})
    check('table_baseline', nums(table_A5.splitlines()),
          nums(make_tables.table_baseline().splitlines()))
    captured = []
    def keep(fig, path, show=False):
        captured.append(fig)
    with redirect(make_figures, FIG=os.path.join(walk_dir('A5'), 'check'), save=keep), \
         contextlib.redirect_stdout(io.StringIO()):
        make_figures.fig_cmax_baseline(E)
    flat = lambda ls: np.concatenate([np.ravel(x) for x in ls]) if ls else np.array([])
    th_lines = [l.get_ydata() for f_ in captured for ax in f_.axes for l in ax.lines]
    th_imgs = [np.asarray(im.get_array(), float) for f_ in captured for ax in f_.axes for im in ax.images]
    for f_ in captured:
        plt.close(f_)
    check('fig_cmax_baseline lines', flat(lines_fig_A5), flat(th_lines))
    check('fig_cmax_baseline IWE images', flat(imgs_fig_A5), flat(th_imgs))
    with redirect(text_numbers, OUT=[]), contextlib.redirect_stdout(io.StringIO()):
        text_numbers.maps_add()
        theirs = [l for l in text_numbers.OUT if l and not l.startswith(('==', '   (missing'))]
    check('text_numbers.maps_add', nums(lines_A5), nums(theirs))
'''),
]
