A3 = ('A3',)

CELLS = [
('md', A3, r'''
## A3 · Main table (Table III), 1.5 s

**Question.** Every model on every Table II segment over 1.5 s (75 frames of
20 ms): ω error, direction error, β and |r|. This is the report's main result.

**Made by** `report/run_experiments.py::cmd_main` with `--duration 1.5`,
`run_all.sh` jobs `dissoc_<seq>_<model>` → runs in `results/`, one row each in
`report/experiments/dissoc_1p5s_<seq>_<model>.csv`. The steps up to the row are
those of A2 (its definitions cell is reused).

**Read by** `report/make_tables.py::table_main` (tab:main),
`report/make_figures.py::fig_tracking` (fig:tracking), and
`report/text_numbers.py::unanchored` / `::reversal`. Also read, but unrolled
where their second input is: `fig_dissociation`, `fig_maps`/`fig_appendix`
(A13) and `fig_image_settings` (A12); the network column of `table_baseline`
(A5).
'''),

('code', A3, r'''
# steps 1-4 of A2 at 1.5 s: the run(s), reference and gyro, floor, the rows
p = params()
runs_A3, nf_A3 = find_runs(p, 'A3', nf=75, duration=1.5)
'''),

('code', A3, r'''
%%section A3
ref_A3, gy_A3 = reference_and_gyro(p['seg'], ref_dt(p), nf_A3)
floor_A3, fdir_A3 = gyro_floor(gy_A3, ref_A3)
rows_A3 = {m: main_row(*p['seg'][:2], m, experiment_tracking.read_summary(d), floor_A3, fdir_A3)
           for m, d in runs_A3.items()}
print(f'gyro floor {floor_A3:.2f} deg/s, direction {fdir_A3:.2f} deg')
display(pd.DataFrame(rows_A3.values(), columns=MAIN_COLS).drop(columns='secs'))
'''),

('md', A3, r'''
### tab:main -- `make_tables.py::table_main`

Reads every `dissoc_1p5s_*.csv` (all sequences, all models), keeps four
columns, marks the best model per sequence in bold (lowest err and dir, β
closest to 1, highest |r|) and adds the gyro floor as a last row. The speed
under each sequence name comes from the A5 baseline npz.
'''),

('code', A3, r'''
%%section A3
# step 5 -- table_main, unrolled (fmt / mark_best / tabular are its formatters)
from make_tables import fmt, mark_best, tabular, floor_label, SEQS, MODELS as TAB_MODELS
csv_rows = {}
for path in sorted(glob.glob(os.path.join('report', 'experiments', 'dissoc_1p5s_*.csv'))):
    with open(path) as f:
        for r in csv.DictReader(f):
            csv_rows[(r['sequence'], r['model'])] = r
print(f'{len(csv_rows)} rows from dissoc_1p5s_*.csv')

def seq_speed(lab):
    """mean |omega| of the reference over the segment, deg/s, from the A5 npz"""
    path = os.path.join('report', 'experiments', f'baseline_dt20ms_{lab}.npz')
    if not need([path], ['baseline']):
        return None
    return float(to_deg_s(np.linalg.norm(np.load(path)['ref'], axis=1).mean()))

body, cells = [], []
for seq, disp, lab, extra in SEQS:
    sp = seq_speed(lab)
    head_ = [disp] + ([f'${sp:.0f}$\\,$^\\circ$/s'] if sp else []) + extra
    body += [r'\midrule', r'\multirow{6}{*}{\shortstack[l]{' + r'\\'.join(head_) + '}}']
    floor = None
    cols = {c: {} for c in ('err', 'dir', 'beta', 'recon')}
    for mdisp, key in TAB_MODELS:
        r = csv_rows.get((seq, key))
        if r is None:
            continue
        floor = (r['floor'], r['floor_dir'])
        cols['err'][key], cols['dir'][key] = fmt(r['err']), fmt(r['dir'])
        cols['beta'][key] = fmt(r['beta'], 0 if float(r['beta']) >= 100 else 2)
        cols['recon'][key] = fmt(r['recon'], 2)
        cells.append([lab, key, r['err'], r['dir'], r['beta'], r['recon']])
    cols = {'err': mark_best(cols['err'], lambda v: v),
            'dir': mark_best(cols['dir'], lambda v: v),
            'beta': mark_best(cols['beta'], lambda v: abs(v - 1.0)),
            'recon': mark_best(cols['recon'], lambda v: -v)}
    for mdisp, key in TAB_MODELS:
        if key not in cols['err']:
            body.append(f' & {mdisp:<22}& --- & --- & --- & --- \\\\')
            continue
        body.append(f' & {mdisp:<22}& {cols["err"][key]} & {cols["dir"][key]} & '
                    f'{cols["beta"][key]} & {cols["recon"][key]} \\\\')
    if floor:
        nd = 2 if float(floor[0]) < 1 else 1
        body.append(f' & {floor_label(seq):<22}& \\emph{{{fmt(floor[0], nd)}}} & '
                    f'\\emph{{{fmt(floor[1])}}} & --- & --- \\\\')
table_A3 = tabular('@{}llrrrr@{}',
                   [r'Sequence & Model & err ($^\circ$/s) & dir ($^\circ$) & $\beta$ & $|r|$ \\'],
                   body)
display(pd.DataFrame(cells, columns=['sequence', 'model', 'err', 'dir', 'beta', 'recon']))
print(table_A3)
'''),

('md', A3, r'''
### fig:tracking -- `make_figures.py::fig_tracking`

ω per axis over the 1.5 s of poster seg_C: CMax-anchor and thesis against the
reference. Redraw draws exactly the report's figure (its segment and models are
fixed in the code); Live draws the panel's run against its own reference.
'''),

('code', A3, r'''
%%section A3
# step 6 -- fig_tracking, unrolled
from interacting_maps.style import use_style, save, frame_times, PAPER_WIDTH
from interacting_maps.viz import draw_omega
if p['mode'] == 'Redraw':
    seg_fig = ('poster_rotation', 'seg_C', 8.816)
    pp = get_dataset_paths(seg_fig[0])
    ref_fig, _ = reference_and_gyro(seg_fig + ('poster',), 0.02, 75)
    ref_src = get_reference_omega(load_groundtruth(pp['groundtruth']), load_imu(pp['imu']),
                                  8.816, 8.836, load_omega_gt(pp.get('omega_gt')))[1]
    fig_runs = [(m, run_dir(seg_fig, m, 75)) for m in ('thesis_cmax', 'thesis')]
else:
    fig_runs = list(runs_A3.items())
    ref_fig, ref_src = tracking(fig_runs[0][1])['gt'], 'groundtruth'
t = frame_times(0.02 if p['mode'] == 'Redraw' else p['dt'], len(ref_fig))

with use_style('paper'):
    fig_A3, axes = plt.subplots(3, 1, figsize=(PAPER_WIDTH, 3.6), sharex=True)
    first = True
    for model, d in fig_runs:
        if d is None:
            continue
        est = tracking(d)['est'][:len(ref_fig)]
        for i in range(3):
            draw_omega(axes[i], t[:len(est)], est[:, i], axis=i, model=model,
                       ref=ref_fig[:len(est), i] if first else None, ref_src=ref_src)
        first = False
    for ax in axes:
        ax.label_outer()
    fig_A3.legend(*axes[0].get_legend_handles_labels(), loc='outside upper center', ncol=3)
    lines_fig = [l.get_ydata() for ax in fig_A3.axes for l in ax.lines]
    if first:                                   # no run was found: nothing drawn
        plt.close(fig_A3)
        print('fig_tracking: its runs are missing (jobs dissoc_poster_thesis, dissoc_poster_thesis_cmax)')
    else:
        save(fig_A3, os.path.join(walk_dir('A3'), 'fig_tracking.png'), show=True)
'''),

('code', A3, r'''
%%section A3
# step 7 -- the text numbers (Redraw only: they read the report's runs)
from report_runs import SYNTHETIC
REAL = [s for s in SEGMENTS if s[3] not in SYNTHETIC]
SYN = [s for s in SEGMENTS if s[3] in SYNTHETIC]
lines_A3 = {'unanchored': [], 'reversal': []}
if p['mode'] == 'Redraw':
    L = lines_A3['unanchored']          # text_numbers.unanchored: cook and thesis
    for seg in REAL + SYN:
        for m in ('cook', 'thesis'):
            d = run_dir(seg, m, 75)
            if d is None:
                continue
            tr = tracking(d)
            e = tr['est']
            m0 = np.linalg.norm(e[0])
            L.append(f'{seg[3]:8s} {m:7s} |w| within {100*np.max(np.abs(mag(e)/m0 - 1)):5.1f}% '
                     f'of frame 0, direction within {max(deg(x, e[0]) for x in e):5.1f} deg; '
                     f'frame 0 |w|/gyro = {m0/np.linalg.norm(tr["imu"][0]):.3f}; '
                     f'last/first |w| = {mag(e)[-1]/m0:.3f}')
        d = run_dir(seg, 'thesis', 75)
        if d is not None:
            g = tracking(d)
            L.append(f'{seg[3]:8s} reference: axis turns up to {max(deg(x, g["gt"][0]) for x in g["gt"]):5.1f} '
                     f'deg, speed {mag(g["gt"]).min()*DEG:.1f}-{mag(g["gt"]).max()*DEG:.1f} deg/s')
    L = lines_A3['reversal']            # text_numbers.reversal: worst CMax-anchor frame
    for seg in REAL:
        dc, di = run_dir(seg, 'thesis_cmax', 75), run_dir(seg, 'thesis_imu', 75)
        if dc is None or di is None:
            continue
        c, i = tracking(dc), tracking(di)
        k = int(np.argmax(c['err']))
        keep = np.arange(len(c['err'])) != k
        dom = int(np.argmax(np.abs(c['gt']).mean(0)))     # dominant rotation axis
        near = c['gt'][max(k - 3, 0):k + 4, dom]
        flip = bool((np.diff(np.sign(near)) != 0).any())
        L.append(f'{seg[3]:8s} frame {k}: |w_ref| = {mag(c["gt"])[k]*DEG:.1f} deg/s '
                 f'({"dominant axis reverses within 3 frames" if flip else "no reversal within 3 frames"}; '
                 f'{int((mag(c["gt"]) <= 0.2).sum())} frames of the track at <= {0.2*DEG:.1f} deg/s); '
                 f'err CMax-anchor {c["err"][k]:6.1f}, thesis+IMU {i["err"][k]:4.1f} deg/s; '
                 f'mean without it: CMax-anchor {c["err"][keep].mean():5.2f}, '
                 f'thesis+IMU {i["err"][keep].mean():5.2f}, gap {c["err"][keep].mean()-i["err"][keep].mean():5.2f} '
                 f'(all frames: gap {c["err"].mean()-i["err"].mean():5.2f})')
    for k_, ls in lines_A3.items():
        print(f'== {k_}'); print('\n'.join(ls) or '(no runs)')
else:
    print('Live: the text numbers read the report runs; switch to Redraw to see them.')
'''),

('code', A3, r'''
%%section A3
# step 8 -- text_numbers.reference: the gyro against a one-frame and a +-50 ms reference.
# No runs: ground truth and IMU only, 75 frames of 20 ms on every Table II and beta segment.
from interacting_maps import eval_io as eio_A3
L = lines_A3['reference'] = []
if p['mode'] == 'Redraw':
    segs = REAL + [s + ('',) for s in BETA_SEGMENTS if s not in [r[:3] for r in REAL]] + SYN
    for seg in segs:
        ds_, sid_, t0_ = seg[:3]
        pp = get_dataset_paths(ds_)
        if not os.path.exists(pp['imu']):
            print(f'missing: {pp["imu"]}')
            continue
        imu, gt, om = (eio_A3.load_imu(pp['imu']), eio_A3.load_groundtruth(pp['groundtruth']),
                       eio_A3.load_omega_gt(pp.get('omega_gt')))
        one, wide = [], []
        for k in range(75):
            lo, hi = t0_ + k * 0.02, t0_ + (k + 1) * 0.02
            g = eio_A3.get_gyro_for_frame(imu, lo, hi)
            if om is not None:                      # exact omega over the frame itself
                msk = (om[:, 0] >= lo) & (om[:, 0] < hi)
                ref1 = om[msk, 1:4].mean(0) if msk.any() else eio_A3.omega_gt_at(om, lo, hi)
            else:
                ref1 = eio_A3.gt_omega_body(gt, lo, hi)
            one.append(np.linalg.norm(g - ref1) * DEG)
            wide.append(np.linalg.norm(g - eio_A3.get_reference_omega(gt, imu, lo, hi, om)[0]) * DEG)
        L.append(f'{ds_[:14]:14s} {sid_}: gyro vs one-frame reference {np.mean(one):5.2f} deg/s, '
                 f'vs +-50 ms {np.mean(wide):5.2f}')
    print('== reference'); [print(l) for l in L]
'''),

('code', A3, r'''
%%section A3
# step 9 -- text_numbers.intensity, its first part: I std and |r| along the poster track
# of the anchored and the unanchored network (the rest of that function reads A8, A1, A16)
L = lines_A3['intensity'] = []
if p['mode'] == 'Redraw':
    FIVE = ('cook', 'thesis', 'thesis_imu', 'thesis_cmax', 'thesis_cmax_v2')
    poster = [s for s in SEGMENTS if s[3] == 'poster'][0]
    c, t_ = run_tn(poster, 'thesis_cmax'), run_tn(poster, 'thesis')
    if c is not None and t_ is not None:
        k = int(np.argmax(c['r']))
        L.append(f'CMax-anchor I std {c["I_std"].min():.2f}-{c["I_std"].max():.2f}; '
                 f'thesis (no anchor) {t_["I_std"].min():.2f}-{t_["I_std"].max():.2f}')
        b = c['I_std'] > 5 * np.median(c['I_std'])
        L.append(f'CMax-anchor omega error on the {b.sum()} frames with I std above 5x its median '
                 f'{c["err"][b].mean():.1f} deg/s, on the other {(~b).sum()} frames {c["err"][~b].mean():.1f} deg/s')
        L.append(f'CMax-anchor I std: median {np.median(c["I_std"]):.2f}, above 5x the median at frames '
                 f'{np.where(b)[0].min()}-{np.where(b)[0].max()}, peak {c["I_std"].max():.2f} at frame {int(np.argmax(c["I_std"]))}')
        L.append(f'CMax-anchor |r|: highest at frame {k} ({c["r"][k]:.2f}), track mean {c["r"].mean():.2f}')
    L.append('frames with I std above 5x its median, per model (Table main runs, 1.5 s):')
    for s in SEGMENTS:
        n = {m: run_tn(s, m) for m in FIVE}
        L.append(f'  {s[3]:8s} ' + ', '.join(f'{m} {int((r["I_std"] > 5 * np.median(r["I_std"])).sum())}/{len(r["I_std"])}'
                                            for m, r in n.items() if r is not None))
    allr = [run_tn(poster, m) for m in FIVE]
    if all(a is not None for a in allr):
        mr = np.mean([a['r'] for a in allr], axis=0)
        L.append(f'mean |r| over the five models: highest at frame {int(np.argmax(mr))}')
    print('== intensity (first part)'); [print(l) for l in L]
'''),

('md', A3, r'''
### fig:dissociation -- `make_figures.py::fig_dissociation`

The read-out (fft solve of the final G) of the Table III runs of thesis,
CMax-anchor and thesis+IMU on boxes, bicycle and street, next to the APS frame;
each labelled with its track-mean |r| and ω error. The run folder is the one
`run_cfg` names (not a search).
'''),

('code', A3, r'''
%%section A3
# step 10 -- fig_dissociation, unrolled (Redraw: the report's runs)
from interacting_maps.network_dissertation import solve_poisson_exact
from interacting_maps.viz import draw_map
from report_runs import aps_frame, track_stats
fig_dis_A3 = None
if p['mode'] == 'Redraw':
    D = make_figures.DISSOC
    nf_ = int(round(D['duration'] / 0.02))
    seqs = [s for s in SEGMENTS if s[3] in D['sequences']]
    with use_style('paper'):
        fig_dis_A3, axes = plt.subplots(len(seqs), len(D['models']) + 1,
                                        figsize=make_figures.grid_size(len(seqs), len(D['models']) + 1,
                                                                       below=1, label_lines=1))
        axes = np.atleast_2d(axes)
        for r_i, (ds_, sid_, t0_, lab) in enumerate(seqs):
            if not need([os.path.join('data', ds_)]):     # run_cfg reads the dataset's IMU
                for a in axes[r_i]:
                    a.set_axis_off()
                continue
            panels = [('APS', aps_frame(ds_, t0_ + nf_ * 0.02), None, 'aps')]
            for m in D['models']:
                d = run_cfg(ds_, sid_, t0_, m, 0.02, nf_).output_dir
                npz = os.path.join(d, 'maps_final.npz')
                img = st = None
                if need([npz], [f'dissoc_{lab}_{m}']):
                    img = solve_poisson_exact(np.load(npz)['G'], 'fft')   # the fft read-out
                    st = track_stats(d)                                    # (track mean |r|, err deg/s)
                panels.append((make_figures.MODELS[m][0], img, st, 'I'))
            for c_i, (title, img, st, kind) in enumerate(panels):
                a = axes[r_i, c_i]
                if img is None:
                    a.set_axis_off()
                    continue
                draw_map(a, kind, img, title=title if r_i == 0 else '')
                if c_i == 0:
                    a.set_ylabel(lab)
                if st is not None:
                    a.set_xlabel(f"$|r|={st[0]:.2f}$,  $\\omega$ err $={st[1]:.1f}$ °/s")
        save(fig_dis_A3, os.path.join(walk_dir('A3'), 'fig_dissociation.png'), show=True)
else:
    print("Live: fig_dissociation reads the report's runs; switch to Redraw to see it.")
'''),

('code', A3, r'''
%%section A3
# same-code check
import io
if p['mode'] == 'Live':
    check_live_main(p, 'A3', nf_A3, runs_A3, rows_A3)
else:
    for m, row in rows_A3.items():      # the rows against dissoc_1p5s_<seq>_<model>.csv
        path = os.path.join('report', 'experiments', f'dissoc_1p5s_{p["segment"]}_{m}.csv')
        if need([path], [f'dissoc_{p["segment"]}_{m}']):
            with open(path) as f:
                rep = list(csv.reader(f))[1]
            if not check(f'dissoc row {m}', dict(zip(MAIN_COLS[3:12], row[3:12])),
                         dict(zip(MAIN_COLS[3:12], map(float, rep[3:12])))):
                stale_note(runs_A3[m])
    check('table_main', nums(table_A3.splitlines()),
          nums(make_tables.table_main().splitlines()))
    captured = []
    def keep(fig, path, show=False):
        captured.append(fig)
    with redirect(make_figures, FIG=os.path.join(walk_dir('A3'), 'check'), save=keep), \
         contextlib.redirect_stdout(io.StringIO()):
        make_figures.fig_tracking(E)
    theirs = [l.get_ydata() for ax in captured[0].axes for l in ax.lines] if captured else []
    for f_ in captured:
        plt.close(f_)
    flat = lambda ls: np.concatenate(ls) if ls else np.array([])
    check('fig_tracking lines', flat(lines_fig), flat(theirs))
    captured = []
    dis_data = [os.path.join('data', s[0]) for s in SEGMENTS if s[3] in make_figures.DISSOC['sequences']]
    if all(os.path.exists(d) for d in dis_data):     # fig_dissociation needs every dataset
        with redirect(make_figures, FIG=os.path.join(walk_dir('A3'), 'check'), save=keep), \
             contextlib.redirect_stdout(io.StringIO()):
            make_figures.fig_dissociation(E)
        imgs = lambda fig: [np.asarray(im.get_array(), float).ravel()
                            for ax in fig.axes for im in ax.images] if fig else []
        check('fig_dissociation images', flat(imgs(fig_dis_A3)), flat(imgs(captured[0] if captured else None)))
    else:
        print('[skip] fig_dissociation: datasets missing here: '
              + ', '.join(d for d in dis_data if not os.path.exists(d)))
    for f_ in captured:
        plt.close(f_)
    for name in lines_A3:
        try:
            with redirect(text_numbers, OUT=[]), contextlib.redirect_stdout(io.StringIO()):
                getattr(text_numbers, name)()
                theirs = [l for l in text_numbers.OUT if l and not l.startswith(('==', '   (missing'))]
        except FileNotFoundError as e:          # the report's function needs a dataset absent here
            print(f'[skip] text_numbers.{name}: {e}')
            continue
        if name == 'intensity':                 # only its first part is unrolled here
            cut = [i for i, l in enumerate(theirs) if l.startswith('curl share')]
            theirs = theirs[:cut[0]] if cut else theirs
        check(f'text_numbers.{name}', nums(lines_A3[name]), nums(theirs))
'''),
]
