A12 = ('A12',)

CELLS = [
('md', A12, r'''
## A12 · Settings for the best image

**Question.** Set for the image instead of for ω: which of
{thesis (no anchor), CMax-anchor} × Δt {20, 10} ms × δ_curl {0, 0.6} gives the
highest mean read-out |r|, separately for real and synthetic data, and what
does it cost in ω error? 1.5 s runs.

**Made by** `report/run_experiments.py::cmd_sweep`
(`--what sweep --name best_image --mode grid --vary 'dt=0.01;delta_curl=0,0.6'
--duration 1.5`), `run_all.sh` jobs `best_image_<seq>_<model>` (thesis,
thesis_cmax) → the 10 ms runs and `sweep_<model>_<seq>.csv` in
`experiments/best_image/`. The 20 ms points are not run here: they are the
Table III runs (A3, jobs `dissoc_<seq>_<model>`) and the curl runs (A4, jobs
`curl_<seq>_<model>`, which also write `main_curl_<seq>_<model>.csv`), in `results/`.

**Read by** `report/report_runs.py::image_search` (the search, from the run
folders, not the csvs), used by `report/make_tables.py::table_image`
(tab:image), `report/make_figures.py::fig_image_settings` (fig:image_settings;
its ω column is the Table III CMax-anchor run, A3) and
`report/text_numbers.py::image_window`.
'''),

('code', A12, r'''
# step 1 -- the runs: the saved 10 ms rows (Redraw) or the panel's best_image points (Live)
p = params()
rows_A12 = run_sweep(p, 'A12', ['best_image_thesis', 'best_image_thesis_cmax'])
if READY['A12']:
    have_cols(rows_A12, SWEEP_COLS[:-1], 'A12')
if p['mode'] == 'Redraw' and READY['A12']:       # the 20 ms half: A3's and A4's runs
    need([os.path.join('report', 'experiments', f'{pre}_{s[3]}_{m}.csv') for s in SEGMENTS
          for m in ('thesis', 'thesis_cmax') for pre in ('dissoc_1p5s', 'main_curl')],
         ['dissoc_<seq>_<model>', 'curl_<seq>_<model>'])
'''),

('code', A12, r'''
%%section A12
# step 2 -- report_runs.py::image_search, unrolled: mean (|r|, err) per kind and configuration
from report_runs import IMG_SEARCH, SYNTHETIC as RR_SYN, track_stats
def image_search_A12(duration, img_root):
    def rdir(ds, sid, t0, model, dt, curl):   # the folder of one search point (run_cfg's output_dir)
        return run_cfg(ds, sid, t0, model, dt, int(round(duration / dt)),
                       out_root=None if dt == 0.02 else img_root,
                       deltas={'delta_curl': curl} if curl else None).output_dir
    kinds = {k: [s for s in SEGMENTS if (s[3] in RR_SYN) == (k == 'synthetic')]
             for k in ('real', 'synthetic')}
    score, best = {}, {}
    for kind, seqs in kinds.items():
        for cfg in IMG_SEARCH:
            try:
                score[kind, cfg] = np.mean([track_stats(rdir(ds, sid, t0, *cfg))
                                            for ds, sid, t0, _ in seqs], axis=0)
            except OSError:
                print(f'    missing runs for {kind} {cfg}')
        # the report's max() raises when a kind has no point at all; here None
        best[kind] = max((c for c in IMG_SEARCH if (kind, c) in score),
                         key=lambda c: score[kind, c][0], default=None)
    return score, best, rdir

if p['mode'] == 'Redraw':
    score_A12, best_A12, rdir_A12 = image_search_A12(1.5, os.path.join('experiments', 'best_image'))
    display(pd.DataFrame([[kind, *cfg[:2], cfg[2], *sc] for (kind, cfg), sc in score_A12.items()],
                         columns=['kind', 'model', 'dt s', 'delta_curl', 'mean |r|', 'mean err deg/s']))
    print('best per kind (highest mean |r|):', best_A12)
else:
    print('Live: the 10 ms runs just made (the 20 ms half and the search need the report runs: Redraw)')
    display(pd.DataFrame([[os.path.basename(d), *track_stats(d)] for d in LIVE_RUNS['A12'].values()],
                         columns=['run', 'mean |r|', 'mean err deg/s']))
'''),

('code', A12, r'''
%%section A12
# step 3 -- table_image, unrolled (Redraw)
from make_tables import tabular
table_A12 = ''
if p['mode'] == 'Redraw':
    body_A12 = []
    for model, title in (('thesis', r'none (\texttt{thesis})'), ('thesis_cmax', r'\texttt{CMax-anchor}')):
        if body_A12:
            body_A12.append(r'\midrule')
        for i, cfg in enumerate(c for c in IMG_SEARCH if c[0] == model):
            cells = []
            for kind in ('real', 'synthetic'):
                if (kind, cfg) not in score_A12:
                    cells.append('--- & ---')
                    continue
                r, err = score_A12[kind, cfg]
                rr = rf'\textbf{{{r:.2f}}}' if cfg == best_A12[kind] else f'{r:.2f}'
                cells.append(f'{rr} & {err:.1f}')
            body_A12.append(f'{title if i == 0 else "":<22} & {cfg[1] * 1000:.0f} & {cfg[2]:g} & '
                        + ' & '.join(cells) + r' \\')
    head_A12 = [r' & & & \multicolumn{2}{c}{real} & \multicolumn{2}{c}{synthetic} \\',
            r'\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
            r'Anchor & $\Delta t$ & $\delta_{\text{curl}}$ & $|r|$ & err & $|r|$ & err \\', r'\midrule']
    table_A12 = tabular('@{}lccrrrr@{}', head_A12, body_A12)
    print(table_A12)
else:
    print('Live: tab:image needs the report runs; switch to Redraw.')
'''),

('md', A12, r'''
### fig:image_settings -- `make_figures.py::fig_image_settings`

Every sequence: the APS frame, the read-out at the ω configuration (the Table
III CMax-anchor run, A3) and at the image configuration of its kind (step 2).
'''),

('code', A12, r'''
%%section A12
# step 4 -- fig_image_settings, unrolled (Redraw; it needs every sequence's data and runs)
from interacting_maps.network_dissertation import solve_poisson_exact
from interacting_maps.viz import draw_map
from report_runs import aps_frame
from make_figures import grid_size, PAPER_WIDTH, IMAGE_DURATION
img_A12 = None
if p['mode'] == 'Live':
    print('Live: fig:image_settings needs the report runs; switch to Redraw.')
elif need([os.path.join('data', s[0]) for s in SEGMENTS], ['(raw data: download_gdrive.sh)']) \
        and None not in best_A12.values():
    omega_cfg_A12 = ('thesis_cmax', 0.02, 0.0)            # tab:params
    with use_style('paper'):
        fig_A12, axes = plt.subplots(len(SEGMENTS), 3, figsize=grid_size(
            len(SEGMENTS), 3, width=PAPER_WIDTH, below=2, title_lines=2))
        for row, (ds, sid, t0, lab) in enumerate(SEGMENTS):
            kind = 'synthetic' if lab in RR_SYN else 'real'
            panels = [('APS', aps_frame(ds, t0 + IMAGE_DURATION), None, 'aps')]
            for ttl, cfg in (('configured\n' r'for $\omega$', omega_cfg_A12),
                             ('configured\nfor the image', best_A12[kind])):
                d = rdir_A12(ds, sid, t0, *cfg)
                img = solve_poisson_exact(np.load(os.path.join(d, 'maps_final.npz'))['G'], 'fft')
                panels.append((ttl, img, track_stats(d) + cfg, 'I'))
            for c, (ttl, img, st, kind_) in enumerate(panels):
                a = axes[row, c]
                if img is None:
                    a.set_axis_off()
                    continue
                draw_map(a, kind_, img, title=ttl if row == 0 else '')
                if c == 0:
                    a.set_ylabel(f'{lab}\n({kind})')
                if st is not None:
                    r, err = st[:2]
                    a.set_xlabel(f'$|r|={r:.2f}$\n$\\omega$ err ${err:.1f}$ °/s')
        img_A12 = fig_images([fig_A12])
        style_save(fig_A12, os.path.join(walk_dir('A12'), 'fig_image_settings.png'), show=True)
'''),

('code', A12, r'''
%%section A12
# step 5 -- text_numbers.image_window, unrolled (Redraw): |r| at 20 -> 10 ms per sequence
def text_image_window_A12():
    L = []
    try:                                    # text_numbers.main reports a failed section so
        for ds, sid, t0, lab in SEGMENTS:
            parts = []
            for model in ('thesis', 'thesis_cmax'):
                for curl in (0.0, 0.6):
                    r20 = track_stats(rdir_A12(ds, sid, t0, model, 0.02, curl))[0]
                    r10 = track_stats(rdir_A12(ds, sid, t0, model, 0.01, curl))[0]
                    parts.append(f'{model} curl {curl:g}: {r20:.2f} -> {r10:.2f}')
            L.append(f'  {lab:8s} ' + ' | '.join(parts))
    except Exception as e:
        L.append(f'   (image_window failed: {type(e).__name__}: {e})')
    return L

lines_A12 = []
if p['mode'] == 'Redraw':
    lines_A12 = text_image_window_A12()
    print('== image_window'); print('\n'.join(lines_A12))
else:
    print('Live: the text numbers read the report runs; switch to Redraw to see them.')
'''),

('code', A12, r'''
%%section A12
# same-code check -- cmd_sweep (Live); image_search, table_image, fig_image_settings, text (Redraw)
import report_runs
def report_call_A12(fn, *a):
    """A report function; a failure is returned as text_numbers.main would print it."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*a), None
    except Exception as e:
        return None, f'{type(e).__name__}: {e}'

if p['mode'] == 'Live':
    check_live_sweep(p, 'A12', LIVE_JOB['A12'], rows_A12)
else:
    got, err = report_call_A12(report_runs.image_search, 1.5)
    if err:
        print(f'[DIFF] image_search: the report function failed: {err}')
    else:
        flat = lambda sc, b: {**{f'{k}|{c}|{j}': x for (k, c), v in sc.items() for j, x in enumerate(v)},
                              **{f'best {k}': IMG_SEARCH.index(c) for k, c in b.items() if c}}
        check('image_search', flat(score_A12, best_A12), flat(got[0], got[1]))
    got, err = report_call_A12(make_tables.table_image)
    if err:
        print(f'[DIFF] table_image: the report function failed: {err}')
    else:
        check('table_image', nums(table_A12.splitlines()), nums(got.splitlines()))
    if img_A12 is not None:
        check('fig_image_settings images', img_A12,
              fig_images(report_figs(make_figures.fig_image_settings, 'A12')))
    else:
        need([os.path.join('data', s[0]) for s in SEGMENTS], ['(fig_image_settings: raw data)'])
    with redirect(text_numbers, OUT=[]), contextlib.redirect_stdout(io.StringIO()):
        try:
            text_numbers.image_window()
            theirs = list(text_numbers.OUT)
        except Exception as e:
            theirs = text_numbers.OUT + [f'   (image_window failed: {type(e).__name__}: {e})']
    theirs = [l for l in theirs if l and not l.startswith(('==', '   (missing'))]
    check('text_numbers.image_window', nums(lines_A12), nums(theirs))
'''),
]
