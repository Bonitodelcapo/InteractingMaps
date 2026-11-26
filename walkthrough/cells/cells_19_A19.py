A19 = ('A19',)

CELLS = [
('md', A19, r'''
## A19 · Adopted parameters (tab:params)

**Question.** Which step sizes does the report use, per relation, for Cook's
network and for the thesis network?

**Made by** nothing run: the values are `config.py`'s `COOK_PARAMS`,
`THESIS_PARAMS` and `ITERS_PER_FRAME` (and the 20 ms window).

**Read by** `report/make_tables.py::table_params` → `report/experiments/table_params.tex`
→ tab:params. The table is written from the code, so it cannot disagree with it.

**Steps**: the six relaxation rates and the loop settings → the LaTeX table →
the panel's defaults against the table → same-code check. Redraw and Live are
the same here.
'''),

('code', A19, r'''
# step 1 -- the values: config.py, nothing to find or run
from config import COOK_PARAMS
p = params()
READY['A19'] = True
print(f"config.py: ITERS_PER_FRAME {ITERS_PER_FRAME}; window 20 ms ({p['mode']} reads the same values)")
display(pd.DataFrame({'cook': COOK_PARAMS, 'thesis': THESIS_PARAMS}))
'''),

('code', A19, r'''
%%section A19
# step 2 -- make_tables.py::table_params, unrolled: one row per relation
from make_tables import tabular
ROWS_A19 = [(r'$\delta_{\Vmap\!\Fmap\!\Gmap}$', r'flow constraint~\eqref{eq:ofce}', 'delta_VFG'),
            (r'$\delta_{I\!G}$', r'$\Gmap$ from $\Imap$', 'delta_IG'),
            (r'$\delta_{G\!I}$', r'$\Imap$ from $\Gmap$', 'delta_GI'),
            (r'$\delta_{R\!F}$', r'$\Fmap$ from $\wvec$', 'delta_RF'),
            (r'$\delta_{F\!R}$', r'$\wvec$ from $\Fmap$', 'delta_FR'),
            (r'$\delta_{\text{anchor}}$', r'external $\wvec$ anchor~\eqref{eq:anchorcost}', 'delta_IMU')]
cell_A19 = lambda d, k: f'{d[k]:.2f}' if k in d else '---'
body_A19 = [f'{s} & {rel} & {cell_A19(COOK_PARAMS, k)} & {cell_A19(THESIS_PARAMS, k)} \\\\' for s, rel, k in ROWS_A19]
body_A19 += [r'\midrule',
         rf'\multicolumn{{2}}{{@{{}}l}}{{iterations per frame}} & \multicolumn{{2}}{{c}}{{{ITERS_PER_FRAME}}} \\',
         r'\multicolumn{2}{@{}l}{window $\Delta t$} & \multicolumn{2}{c}{20\,ms} \\']
table_A19 = tabular('@{}llcc@{}', [r'Symbol & Relation & Cook & Martel \\', r'\midrule'], body_A19)
display(pd.DataFrame([[k, cell_A19(COOK_PARAMS, k), cell_A19(THESIS_PARAMS, k)] for _s, _r, k in ROWS_A19],
                     columns=['parameter', 'Cook', 'thesis (Martel)']))
print(table_A19)
'''),

('code', A19, r'''
%%section A19
# step 3 -- the panel against the table: its defaults are the table's thesis column
want_A19 = {**{k: float(cell_A19(THESIS_PARAMS, k)) for _s, _r, k in ROWS_A19},
            'n_iters': ITERS_PER_FRAME, 'dt': 0.02}
display(pd.DataFrame({'table': want_A19,
                      'panel default': {k: DEFAULTS[k] for k in want_A19},
                      'panel now': {k: p[k] for k in want_A19}}))
changed_A19 = [k for k in want_A19 if p[k] != DEFAULTS[k]]
if changed_A19:
    print(f'the panel differs from the report in {changed_A19}: reset it to get the report back')
'''),

('code', A19, r'''
%%section A19
# same-code check -- table_params itself, the tex the report job wrote, and the panel defaults
theirs_A19 = make_tables.table_params()
check('table_params tex', nums(table_A19.splitlines()), nums(theirs_A19.splitlines()))
if table_A19 != theirs_A19:
    print('   the numbers agree but the LaTeX text differs')
tex_A19 = os.path.join('report', 'experiments', 'table_params.tex')
if need([tex_A19], ['make_tables (python report/make_tables.py)']):
    with open(tex_A19, encoding='utf-8') as f:
        rep_A19 = [l for l in f.read().splitlines() if not l.startswith('%')]
    check('table_params vs report tex', nums(table_A19.splitlines()), nums(rep_A19))
check('panel defaults = table_params', {k: round(float(DEFAULTS[k]), 2) for k in want_A19}, want_A19)
'''),
]
