A11 = ('A11',)

CELLS = [
('md', A11, r'''
## A11 · Regulariser ablation

**Question.** What does each regulariser do on its own — shrink-I
(δ_shrinkI = 1), the MAP prior (δ_map 0.01, 0.03) and the curl term (δ_curl 0.3,
0.6) — to the ω error, the read-out and in-loop |r|, and the excursions?
CMax-anchor, 1.5 s, four sequences (street repeats bicycle's trajectory).

**Made by** `report/run_experiments.py::cmd_sweep`
(`--what sweep --name ablation --mode coord --duration 1.5`), `run_all.sh`
jobs `ablation_<seq>_<v>` (one value per job; `delta_shrinkI=0` is the "none"
point) → runs and `sweep_<seq>_<v>.csv` in `experiments/ablation/`.

**Read by** `report/make_tables.py::table_regularisers` (tab:regularisers),
`report/make_figures.py::fig_regularisers` (fig:regularisers),
`report/text_numbers.py::regularisers` (its curl-share half reads the Table III
and curl runs in `results/`, A3/A4).
'''),

('code', A11, r'''
# step 1 -- the runs: the saved ablation rows (Redraw) or the panel's ablation points (Live)
p = params()
rows_A11 = run_sweep(p, 'A11', 'ablation')
if READY['A11']:
    have_cols(rows_A11, SWEEP_COLS[:-1], 'A11')
'''),

('code', A11, r'''
%%section A11
# step 2 -- table_regularisers, unrolled: each term alone, mean over the four sequences
from make_tables import REGULARISERS as TAB_REG, ABLATION_SEQS, fmt, tabular
def table_regularisers_A11(rows):
    terms = ('delta_shrinkI', 'delta_map', 'delta_curl')
    stats = []
    for label, on in TAB_REG:
        want = {k: on.get(k, 0.0) for k in terms}
        hit = {r['sequence']: r for r in rows
               if r['model'] == 'thesis_cmax' and int(float(r['n_iters'])) == ITERS_PER_FRAME
               and all(abs(float(r[k]) - v) < 1e-9 for k, v in want.items())}
        got = [hit[ds] for ds in ABLATION_SEQS if ds in hit]
        mean = (lambda k: float(np.mean([abs(float(r[k])) for r in got]))
                if len(got) == len(ABLATION_SEQS) else None)     # all four, or '---'
        stats.append((label, [mean('err'), mean('recon_mean'), mean('recon_inloop'), mean('blowup')],
                      len(got)))
    pick = [min, max, max, min]                 # bold: lowest err and blow-up, highest |r|
    best = [f([s[1][c] for s in stats if s[1][c] is not None], default=None)
            for c, f in enumerate(pick)]
    body = []
    for label, vals, _n in stats:
        cells = []
        for c, (v, nd) in enumerate(zip(vals, (2, 3, 3, 3))):
            t = fmt(v, nd)
            cells.append(rf'\textbf{{{t}}}' if v is not None and v == best[c] else t)
        body.append(f'{label:<38} & ' + ' & '.join(cells) + r' \\')
    head = [r' & $\wvec$ err & $|r|$ read-out & $|r|$ in-loop & blow-up \\',
            r' & [$^\circ$/s] & & & \\', r'\midrule']
    return stats, tabular('@{}lcccc@{}', head, body)

stats_A11, table_A11 = table_regularisers_A11(rows_A11)
display(pd.DataFrame([[lab, n, *v] for lab, v, n in stats_A11],
                     columns=['term', 'n seq', 'err deg/s', '|r| read-out', '|r| in-loop', 'blow-up']))
print(table_A11)
'''),

('md', A11, r'''
### fig:regularisers -- `make_figures.py::fig_regularisers`

The read-out of each ablation run's final G (Poisson solve, fft), one row per
sequence, next to the APS frame; |r| under each panel. Redraw draws the
sequences whose data are here; Live draws the panel's segment from the Live runs.
'''),

('code', A11, r'''
%%section A11
# step 3 -- fig_regularisers, unrolled
from interacting_maps.network_dissertation import solve_poisson_exact
from interacting_maps.viz import draw_map
from interacting_maps.metrics import abs_corr, crop_interior
from report_runs import aps_frame
from make_figures import REGULARISERS as FIG_REG, grid_size
cols_A11 = [('APS', None), ('none', {}),
            (r'$\delta_{\mathrm{shrink}I}{=}1$', {'delta_shrinkI': 1.0}),
            (r'$\delta_{\mathrm{map}}{=}0.03$', {'delta_map': 0.03}),
            (r'$\delta_{\mathrm{curl}}{=}0.6$', {'delta_curl': 0.6})]
if p['mode'] == 'Redraw':
    seqs_A11 = [s for s in SEGMENTS if s[3] in FIG_REG['sequences']]
    seqs_A11 = [s for s in seqs_A11 if need([os.path.join('data', s[0])], [])]   # aps + run_cfg need them
    root_A11, nf_A11 = FIG_REG['root'], int(round(FIG_REG['duration'] / 0.02))
else:
    seqs_A11 = [sweep_seg(p, SWEEP_JOBS['ablation'])]
    root_A11 = walk_dir('A11')
    nf_A11 = int(round(sweep_duration(p, SWEEP_JOBS['ablation']) / 0.02))
img_A11 = []
if seqs_A11:
    with use_style('paper'):
        fig_A11, axes = plt.subplots(len(seqs_A11), len(cols_A11),
                                     figsize=grid_size(len(seqs_A11), len(cols_A11), below=1, label_lines=1))
        axes = np.atleast_2d(axes)
        for row, (ds, sid, t0, lab) in enumerate(seqs_A11):
            aps = aps_frame(ds, t0 + nf_A11 * 0.02)
            ref = np.log(aps.astype(np.float64) + 1.0) if aps is not None else None
            for col, (title, deltas) in enumerate(cols_A11):
                a = axes[row, col]
                kind, img = 'aps', aps
                if col > 0:
                    kind, img = 'I', None
                    rc = run_cfg(ds, sid, t0, FIG_REG['model'], 0.02, nf_A11, out_root=root_A11,
                                 deltas=dict(deltas))
                    npz = os.path.join(rc.output_dir, 'maps_final.npz')
                    if os.path.exists(npz):
                        img = solve_poisson_exact(np.load(npz)['G'], 'fft')
                    else:
                        print(f'   (no run: {rc.output_dir})')
                if img is None:
                    a.set_axis_off()
                    continue
                draw_map(a, kind, img, title=title if row == 0 else '')
                if col == 0:
                    a.set_ylabel(lab)
                elif ref is not None and ref.shape == img.shape:
                    a.set_xlabel(f'$|r|={abs_corr(crop_interior(img), crop_interior(ref)):.2f}$')
        img_A11 = fig_images([fig_A11])
        style_save(fig_A11, os.path.join(walk_dir('A11'), 'fig_regularisers.png'), show=True)
else:
    print('fig_regularisers: no sequence has its data here')
'''),

('code', A11, r'''
%%section A11
# step 4 -- text_numbers.regularisers, unrolled (Redraw only: it reads the report's runs)
def text_regularisers_A11(rows):
    L = []
    terms = (('none', {}), ('shrinkI=1', {'delta_shrinkI': 1.0}),
             ('map=0.01', {'delta_map': 0.01}), ('map=0.03', {'delta_map': 0.03}),
             ('curl=0.3', {'delta_curl': 0.3}), ('curl=0.6', {'delta_curl': 0.6}))
    reg = ('delta_shrinkI', 'delta_map', 'delta_curl')
    for s in SEGMENTS:
        mine = [r for r in rows if r['sequence'] == s[0]]
        for name, on in terms:
            hit = [r for r in mine if all(abs(float(r[k]) - on.get(k, 0.0)) < 1e-9 for k in reg)]
            if hit:
                r = hit[0]
                L.append(f'{s[3]:8s} {name:10s} err {float(r["err"]):6.2f}  |r| read-out '
                         f'{float(r["recon_mean"]):.3f}  in-loop {float(r["recon_inloop"]):.3f}  '
                         f'blow-up {float(r["blowup"]):.2f}  curl share {float(r["curl"]):.2f}')
    L.append('curl share of the final G, CMax-anchor, Table curl runs (none -> curl 0.6):')
    for s in SEGMENTS:                  # text_numbers.run_dir = A2's run_dir at 75 frames
        a, b = run_dir(s, 'thesis_cmax', 75, curl=0.0), run_dir(s, 'thesis_cmax', 75, curl=0.6)
        if a and b:
            L.append(f'  {s[3]:8s} {experiment_tracking.read_summary(a)["curl_share"]:.2f} -> '
                     f'{experiment_tracking.read_summary(b)["curl_share"]:.2f}')
    return L

lines_A11 = []
if p['mode'] == 'Redraw':
    # text_numbers reads experiments/ablation/sweep_*.csv (the table: *.csv)
    lines_A11 = text_regularisers_A11(sweep_rows('ablation', ('sweep_*.csv',)))
    print('== regularisers'); print('\n'.join(lines_A11))
else:
    print('Live: the text numbers read the report runs; switch to Redraw to see them.')
'''),

('code', A11, r'''
%%section A11
# same-code check -- cmd_sweep (Live); table_regularisers, fig_regularisers, text (Redraw)
if p['mode'] == 'Live':
    check_live_sweep(p, 'A11', LIVE_JOB['A11'], rows_A11)
else:
    check('table_regularisers', nums(table_A11.splitlines()),
          nums(make_tables.table_regularisers().splitlines()))
    if seqs_A11:                        # the report's figure on the same sequences
        figs = report_figs(make_figures.fig_regularisers, 'A11',
                           REGULARISERS={**FIG_REG, 'sequences': tuple(s[3] for s in seqs_A11)})
        check(f'fig_regularisers images ({", ".join(s[3] for s in seqs_A11)})',
              img_A11, fig_images(figs))
    else:
        need([os.path.join('data', s[0]) for s in SEGMENTS if s[3] in FIG_REG['sequences']],
             ['(raw data: download_gdrive.sh)'])
    with redirect(text_numbers, OUT=[]), contextlib.redirect_stdout(io.StringIO()):
        text_numbers.regularisers()
        theirs = [l for l in text_numbers.OUT if l and not l.startswith(('==', '   (missing'))]
    check('text_numbers.regularisers', nums(lines_A11), nums(theirs))
'''),
]
