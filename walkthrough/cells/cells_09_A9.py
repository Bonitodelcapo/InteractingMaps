A9 = ('A9',)

CELLS = [
('md', A9, r'''
## A9 · The (δ_FR, δ_anchor) grid

**Question.** δ_FR against the anchor weight on a full 5×5 grid
(δ_FR ∈ {0.02, 0.05, 0.1, 0.3, 0.5} × δ_anchor ∈ {0.1, 0.3, 0.5, 0.7, 0.9}),
CMax-anchor, 1.5 s, every sequence: where is each sequence best, and does the
error depend on the two only through κ = δ_FR/δ_anchor?

**Made by** `report/run_experiments.py::cmd_sweep`
(`--what sweep --name grid --mode grid --duration 1.5`), `run_all.sh` jobs
`grid_<seq>_FR<x>` (one δ_FR row per job; here as one job, the union) → runs and
`grid_<seq>_FR<x>.csv` in `experiments/grid/`.

**Read by** `report/make_tables.py::table_grid` (**not in the tex**: written
to `table_grid.tex`, never pasted), `::table_params` (tab:params: its δ_FR and
δ_anchor rows are the values "set after a parameter sweep"; the table is typed
into the tex by hand, so it is compared with the tex here),
`report/text_numbers.py::grid`. Also the κ panel of `fig_sensitivity` and the
grid half of `text_numbers.sensitivity_both`, unrolled in A8.
'''),

('code', A9, r'''
# step 1 -- the runs: the saved grid rows (Redraw) or the panel's grid points (Live)
p = params()
rows_A9 = run_sweep(p, 'A9', 'grid')
if READY['A9']:
    have_cols(rows_A9, SWEEP_COLS[:-1], 'A9')
'''),

('code', A9, r'''
%%section A9
# step 2 -- the grid per sequence: err (deg/s) over delta_FR (rows) x delta_anchor (columns)
grid_A9 = {}
for r in rows_A9:
    grid_A9.setdefault(r['sequence'], {})[(float(r['delta_FR']), float(r['delta_IMU']))] = float(r['err'])
for ds_, g in grid_A9.items():
    print(f'{ds_}: err deg/s, {len(g)} cells')
    display(pd.Series(g).unstack().rename_axis(index='delta_FR', columns='delta_anchor').round(2))
'''),

('md', A9, r'''
### tab:grid -- `make_tables.py::table_grid` (not in the tex)

Each sequence reduced to its adopted cell and its best cell, with κ.
'''),

('code', A9, r'''
%%section A9
# step 3 -- table_grid, unrolled (every csv row, any model, keyed by sequence)
from make_tables import fmt, tabular, SEQS as TAB_SEQS, ADOPTED as TAB_ADOPTED
body_A9, opt_A9 = [], {}
for ds_, disp, *_ in TAB_SEQS:
    g = grid_A9.get(ds_)
    if not g:
        continue
    (fr, an), e = min(g.items(), key=lambda kv: kv[1])
    ad = g.get(TAB_ADOPTED)
    opt_A9[ds_] = (e, fr, an)
    body_A9.append(f'{disp:<18}& {fmt(ad, 2)} & {fmt(TAB_ADOPTED[0] / TAB_ADOPTED[1], 2)} & '
                   f'{fmt(e, 2)} & {fr:g} & {an:g} & {fmt(fr / an, 2)} \\\\')
head_A9 = [r' & \multicolumn{2}{c}{adopted} & \multicolumn{4}{c}{grid optimum} \\',
           r'\cmidrule(lr){2-3}\cmidrule(lr){4-7}',
           r'sequence & err & $\kappa$ & err & $\delta_{F\!R}$ & $\delta_{\text{anchor}}$ & $\kappa$ \\',
           r'\midrule']
table_A9 = tabular('@{}lrrrrrr@{}', head_A9, body_A9)
print(table_A9)
'''),

('md', A9, r'''
### tab:params -- the grid's part of `make_tables.py::table_params`

table_params prints config.py's step sizes; two of them, δ_FR and δ_anchor
(`delta_IMU`), are the ones the report says were set after a parameter sweep.
Below: those two rows, next to the grid's optimum per sequence and per kind of
data (the cell with the lowest mean error over the kind's sequences).
'''),

('code', A9, r'''
%%section A9
# step 4 -- table_params' delta_FR and delta_anchor rows, against the grid optimum
from make_tables import COOK_PARAMS as TAB_COOK, THESIS_PARAMS as TAB_THESIS
from report_runs import SYNTHETIC as RR_SYN
pcell_A9 = lambda d, k: f'{d[k]:.2f}' if k in d else '---'
rows_params_A9 = {k: [pcell_A9(TAB_COOK, k), pcell_A9(TAB_THESIS, k)] for k in ('delta_FR', 'delta_IMU')}
for k, (c, m) in rows_params_A9.items():
    print(f'table_params {k:9s}: Cook {c}, Martel {m}')
def grid_optimum_A9(grid):
    """The cell of lowest mean error over each kind's sequences (cells all of them have)."""
    out = {}
    for kind in ('real', 'synthetic'):
        seqs = [s[0] for s in SEGMENTS if (s[3] in RR_SYN) == (kind == 'synthetic') and s[0] in grid]
        cells = set.intersection(*[set(grid[s]) for s in seqs]) if seqs else set()
        if cells:
            mean = {c: np.mean([grid[s][c] for s in seqs]) for c in cells}
            best = min(mean, key=mean.get)
            out[kind] = best
            print(f'grid optimum, {kind} mean over {len(seqs)}: dFR {best[0]:g}, anchor {best[1]:g} '
                  f'(kappa {best[0]/best[1]:.2f}), {mean[best]:.2f} deg/s; adopted '
                  f'({TAB_ADOPTED[0]:g}, {TAB_ADOPTED[1]:g}): {mean.get(TAB_ADOPTED, np.nan):.2f} deg/s')
    return out

kinds_A9 = grid_optimum_A9(grid_A9)
for ds_, (e, fr, an) in opt_A9.items():
    print(f'   {ds_:18s} optimum dFR {fr:g}, anchor {an:g}: {e:.2f} deg/s')
'''),

('code', A9, r'''
%%section A9
# step 5 -- text_numbers.grid, unrolled (Redraw only: it reads the report's rows)
def text_grid_A9(rows):
    from text_numbers import REAL as TN_REAL, SYN as TN_SYN
    from report_runs import ADOPTED as RR_ADOPTED
    L = []
    if not rows:
        return ['   (no grid csv)']
    tab = {}
    for r in rows:
        an = r.get('delta_IMU', r.get('delta_anchor'))     # older grid csv: delta_anchor
        tab[(r['sequence'], round(float(r['delta_FR']), 4), round(float(an), 4))] = float(r['err'])
    seqs = [s for s in SEGMENTS if any(k[0] == s[0] for k in tab)]
    spread = 0.0
    for s in seqs:
        by_k = {}
        for (q, fr, an), e in tab.items():
            if q == s[0]:
                by_k.setdefault(round(fr / an, 3), []).append(e)
        own = max([0.0] + [max(v) - min(v) for v in by_k.values() if len(v) > 1])
        spread = max(spread, own)
        ks = sorted(by_k)
        mean = {k: np.mean(by_k[k]) for k in ks}
        best = min(ks, key=mean.get)
        errs = [mean[k] for k in ks]
        mono = all(x <= y for x, y in zip(errs, errs[1:]))
        ad = tab.get((s[0], RR_ADOPTED['delta_FR'], RR_ADOPTED['delta_IMU']))
        L.append(f'{s[3]:8s} best kappa {best:g} ({mean[best]:.2f}); smallest kappa {ks[0]:g} '
                 f'({mean[ks[0]]:.2f}){" error rises with kappa throughout" if mono else ""}; '
                 f'adopted {ad:.2f}; kappa 0.6 (0.3/0.5) {tab.get((s[0], 0.3, 0.5), np.nan):.2f}; '
                 f'start (0.5/0.3) {tab.get((s[0], 0.5, 0.3), np.nan):.2f}; '
                 f'equal-kappa spread {own:.2f}')
    L.append(f'largest spread between settings of equal kappa: {spread:.2f} deg/s')
    for name, group in (('real', TN_REAL), ('synthetic', TN_SYN)):
        m = lambda fr, an: np.mean([tab[(s[0], fr, an)] for s in group if (s[0], fr, an) in tab])
        L.append(f'{name} mean: adopted {m(RR_ADOPTED["delta_FR"], RR_ADOPTED["delta_IMU"]):.2f}, '
                 f'start {m(0.5, 0.3):.2f}, kappa 0.6 {m(0.3, 0.5):.2f}')
    return L

lines_A9 = []
if p['mode'] == 'Redraw':
    lines_A9 = text_grid_A9(rows_A9)
    print('== grid'); print('\n'.join(lines_A9))
else:
    print('Live: the text numbers read the report rows; switch to Redraw to see them.')
'''),

('code', A9, r'''
%%section A9
# same-code check -- cmd_sweep (Live); table_grid, table_params, text_numbers.grid (Redraw)
if p['mode'] == 'Live':
    check_live_sweep(p, 'A9', LIVE_JOB['A9'], rows_A9)
else:
    check('table_grid', nums(table_A9.splitlines()), nums(make_tables.table_grid().splitlines()))
    tp_A9 = make_tables.table_params().splitlines()
    theirs = {k: nums([l for l in tp_A9 if tag in l][0].split('&')[2:]) for k, tag in
              (('delta_FR', r'from $\Fmap$'), ('delta_IMU', 'anchor~'))}
    mine = {k: nums([v for v in vals if v != '---']) for k, vals in rows_params_A9.items()}
    check('table_params rows delta_FR, delta_anchor', np.concatenate(list(mine.values())),
          np.concatenate(list(theirs.values())))
    tex_A9 = open(os.path.join('report', 'bare_jrnl.tex'), encoding='utf-8').read()
    blk = tex_A9[tex_A9.index(r'\label{tab:params}'):]
    blk = blk[blk.index(r'\begin{tabular}'):blk.index(r'\end{tabular}')]
    check('table_params vs tab:params typed in the tex', nums(blk.splitlines()), nums(tp_A9[:-2]))
    with redirect(text_numbers, OUT=[]), contextlib.redirect_stdout(io.StringIO()):
        text_numbers.grid()
        theirs = [l for l in text_numbers.OUT if l and not l.startswith(('==', '   (missing'))]
    check('text_numbers.grid', nums(lines_A9), nums(theirs))
'''),
]
