A4 = ('A4',)

CELLS = [
('md', A4, r'''
## A4 · Curl regulariser (tab:curl), 1.5 s

**Question.** What does the integrability prior on G, `delta_curl = 0.6`, change
in every model over 1.5 s: ω error and read-out |r|, against the runs of
Table III (A3) without it?

**Made by** `report/run_experiments.py::cmd_main` with `--duration 1.5 --vary
delta_curl=0.6`, `run_all.sh` jobs `curl_<seq>_<model>` → runs in `results/`,
one row each in `report/experiments/main_curl_<seq>_<model>.csv`. The steps up
to the row are those of A2 (its definitions cell is reused) with `delta_curl`
at 0.6.

**Read by** `report/make_tables.py::table_curl` (tab:curl; its *none* half is
the A3 csvs) and the 1.5 s half of `report/text_numbers.py::inloop_identical`
(quoted in the text; its 3 s half is A2). Also read, but unrolled where their
other input is: `make_tables.table_image` (A12) and the curl-share lines of
`text_numbers.regularisers` (A11).
'''),

('code', A4, r'''
# the A4 form: as A3, with the curl step size (report: 0.6)
form('A4', main_knobs(75) + [Knob('delta_curl', 0.6)],
     fixed={'delta_shrinkI': 0.0, 'delta_map': 0.0})
'''),

('code', A4, r'''
# step 1 -- the runs with the form's delta_curl: found (Redraw) or made now (Live)
p = params('A4')
runs_A4, nf_A4 = find_runs(p, 'A4', duration=1.5)
q_A4, rep_A4 = p, is_report('A4')               # the report-only steps run only then
'''),

('code', A4, r'''
%%section A4
# steps 2-4 of A2 (cmd_main): reference and gyro, gyro floor, one row per run
ref_A4, gy_A4 = reference_and_gyro(q_A4['seg'], ref_dt(q_A4), nf_A4)
floor_A4, fdir_A4 = gyro_floor(gy_A4, ref_A4)
rows_A4 = {m: main_row(*q_A4['seg'][:2], m, experiment_tracking.read_summary(d), floor_A4, fdir_A4)
           for m, d in runs_A4.items()}
print(f'gyro floor {floor_A4:.2f} deg/s, direction {fdir_A4:.2f} deg')
display(pd.DataFrame(rows_A4.values(), columns=MAIN_COLS).drop(columns='secs'))
'''),

('md', A4, r'''
### tab:curl -- `make_tables.py::table_curl`

Two inputs: the `dissoc_1p5s_*.csv` rows (A3, *none*) and the `main_curl_*_*.csv`
rows (this section, *curl*), keyed by (sequence, model). Per sequence it shows
err and |r| of every model with and without the curl term, and marks in bold the
lowest err and the highest |r| among the ten runs of that sequence.
'''),

('code', A4, r'''
%%section A4
# step 5 -- table_curl, unrolled (load / fmt / mark_best / tabular are its helpers)
# It reads the report's csv rows of every segment: report configuration only.
from make_tables import load, fmt, mark_best, tabular, SEQS, MODELS as TAB_MODELS
none_A4, curl_A4 = (load(['dissoc_1p5s_*.csv']), load(['main_curl_*_*.csv'])) if rep_A4 else ({}, {})
if not rep_A4:
    print('[skip] report-only step')
print(f'{len(none_A4)} rows without curl (dissoc_1p5s_*.csv), {len(curl_A4)} with (main_curl_*_*.csv)')


def table_curl_A4(none, curl):
    """make_tables.table_curl on the two row sets: (LaTeX, the numbers shown)."""
    body, cells = [], []
    for ds, disp, lab, _extra in SEQS:
        body.append(r'\midrule')
        g = lambda r, k, nd: fmt(r[k], nd) if r else '---'
        # bold: the lowest err and the highest |r| of the ten runs of this sequence
        err = mark_best({(key, w): g(src.get((ds, key)), 'err', 1) for _, key in TAB_MODELS
                         for w, src in (('none', none), ('curl', curl))}, lambda v: v)
        rec = mark_best({(key, w): g(src.get((ds, key)), 'recon', 2) for _, key in TAB_MODELS
                         for w, src in (('none', none), ('curl', curl))}, lambda v: -v)
        for j, (mdisp, key) in enumerate(TAB_MODELS):
            body.append(f'{disp if j == 0 else "":<22}& {mdisp:<22}& '
                        f'{err[key, "none"]} & {err[key, "curl"]} & '
                        f'{rec[key, "none"]} & {rec[key, "curl"]} \\\\')
            a, b = none.get((ds, key)), curl.get((ds, key))
            cells.append([lab, key, a and a['err'], b and b['err'], a and a['recon'], b and b['recon']])
    head = [r' & & \multicolumn{2}{c}{$\wvec$ err [$^\circ$/s]} & \multicolumn{2}{c}{$|r|$} \\',
            r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}',
            r'Sequence & Model & none & curl & none & curl \\']
    return tabular('@{}llrrrr@{}', head, body), cells


table_A4, cells_A4 = table_curl_A4(none_A4, curl_A4)
if rep_A4:
    display(pd.DataFrame(cells_A4, columns=['sequence', 'model', 'err none (deg/s)', 'err curl (deg/s)',
                                            '|r| none', '|r| curl']))
    print(table_A4)
'''),

('code', A4, r'''
%%section A4
# step 6 -- text_numbers.py::inloop_identical, 1.5 s half (report configuration only:
# it reads the report's runs). CMax-inloop omega with and without the curl term: the maps
# change, the anchor's omega should not.
lines_A4 = []
if rep_A4:
    for seg in SEGMENTS:
        da = run_dir(seg, 'thesis_cmax_v2', 75)
        db = run_dir(seg, 'thesis_cmax_v2', 75, curl=0.6)
        if da and db:
            lines_A4.append(f'{seg[3]:8s} curl 0 vs 0.6 (1.5 s): max |dw| = '
                            f'{np.abs(tracking(da)["est"]-tracking(db)["est"]).max()*DEG:.2e} deg/s')
    print('== inloop_identical (1.5 s)'); print('\n'.join(lines_A4) or '(no runs)')
else:
    print('[skip] report-only step')
'''),

('code', A4, r'''
%%section A4
# same-code check
import io
if p['mode'] == 'Live':                         # cmd_main --vary delta_curl=...
    for m_ in runs_A4:
        check_live_main({**q_A4, 'model': m_}, 'A4', nf_A4, runs_A4, rows_A4)
if not rep_A4:
    print('[skip] not the report configuration')
else:
    for m, row in rows_A4.items():      # the rows against main_curl_<seq>_<model>.csv
        path = os.path.join('report', 'experiments', f'main_curl_{p["segment"]}_{m}.csv')
        if need([path], [f'curl_{p["segment"]}_{m}']):
            with open(path) as f:
                rep = list(csv.reader(f))[1]
            if not check(f'main_curl row {m}', dict(zip(MAIN_COLS[3:12], row[3:12])),
                         dict(zip(MAIN_COLS[3:12], map(float, rep[3:12])))):
                stale_note(runs_A4[m])
    with redirect(text_numbers, OUT=[]), contextlib.redirect_stdout(io.StringIO()):
        text_numbers.inloop_identical()
        theirs = [l for l in text_numbers.OUT if '(1.5 s)' in l]
    check('text_numbers.inloop_identical', nums(lines_A4), nums(theirs))
    check('table_curl', nums(table_A4.splitlines()), nums(make_tables.table_curl().splitlines()))
'''),
]
