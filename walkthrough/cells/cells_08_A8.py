A8 = ('A8',)
DEFS = ('A8', 'sweep-defs')

CELLS = [
('md', A8, r'''
## A8 · Sensitivity sweep (one parameter at a time)

**Question.** Around the adopted configuration, how do the ω error, the
direction error, β and the read-out |r| move when one parameter changes: the
anchor weight (CMax-anchor and thesis+IMU), δ_FR, the four relaxation rates,
the iterations per frame and the window? CMax-anchor, 1.5 s, every sequence.

**Made by** `report/run_experiments.py::cmd_sweep`
(`--what sweep --name sens --mode coord --duration 1.5`), `run_all.sh` jobs
`sens_cmax_<seq>`, `sens_imu_<seq>`, `sens_rates_<seq>` → runs and
`sens_*.csv` in `experiments/sens/`. `table_sweep` and `text_numbers` fall back
to `report/experiments/sens_*.csv` when that folder is empty;
`fig_sensitivity` does not (flagged below).

**Read by** `report/make_tables.py::table_sweep` (tab:sweep),
`report/make_figures.py::fig_sensitivity` (**not in the tex**: drawn, never
`\includegraphics`-ed), and `report/text_numbers.py::sensitivity_both`,
`::sweep_ends`, `::anchor_ablation`, `::anchors_across_weights`,
`::kappa_rows`. The κ panel of `fig_sensitivity` and the grid half of
`sensitivity_both` read A9's grid rows.

**Sweep extent** (panel, Live only; sections A8–A13 share it): `one point`
runs the job's base point, `one axis` every value of the axis the panel
changed (else the job's first axis), `full` the job's whole `--vary`; always on
the panel's segment, and `n_frames` caps the track (n_frames × 20 ms, the
duration `cmd_sweep` holds fixed, so a 5 ms point runs 4× the frames). The
panel's model picks the job (`thesis_imu` → `sens_imu`; a changed rate →
`sens_rates`).
'''),

('code', DEFS, r'''
# sweep definitions -- the unrolled plumbing of run_experiments.py::cmd_sweep,
# shared by A8-A13 (the counterpart of A2-defs)
import io, json, argparse
from run_experiments import ALL_DELTAS, _parse_vary, _points, _split
from interacting_maps.style import use_style, save as style_save

#: cmd_sweep's csv columns, in its order
SWEEP_COLS = (['sequence', 'segment', 'model', 'dt_ms', 'n_frames', 'n_iters', 'poisson']
              + ALL_DELTAS + ['err', 'median', 'dir', 'beta', 'low_freq', 'contrast', 'curl',
                              'recon_mean', 'recon_min', 'recon_inloop', 'recon_inloop_min',
                              'blowup', 'secs'])
_ALL = 'boxes,poster,dynamic,bicycle,street'
_S = lambda m, seqs: tuple(f'sweep_{m}_{s}.csv' for s in seqs.split(','))


def _job(name, mode, model, duration, vary, csv_, run_all, base='', seqs=_ALL):
    return dict(name=name, mode=mode, model=model, duration=duration, vary=vary, base=base,
                csv=csv_, run_all=run_all, seqs=seqs)


#: run_all.sh's sweep jobs, copied. Where run_all.sh splits one sweep into one
#: job per value (grid: one delta_FR row per job; sweep_*: one value per job),
#: the entry is their union, which _points turns into the same set of points.
SWEEP_JOBS = {
    'sens_cmax': _job('sens', 'coord', 'thesis_cmax', 1.5,
                      'delta_IMU=0,0.1,0.3,0.5,0.7,0.9;delta_FR=0.02,0.05,0.1,0.3,0.5,0.7,0.9;'
                      'n_iters=25,50,75,150;dt=0.005,0.01,0.02,0.04',
                      ('sens_cmax_*.csv',), 'sens_cmax_<seq>'),
    'sens_imu': _job('sens', 'coord', 'thesis_imu', 1.5, 'delta_IMU=0.1,0.3,0.5,0.7,0.9',
                     ('sens_imu_*.csv',), 'sens_imu_<seq>'),
    'sens_rates': _job('sens', 'coord', 'thesis_cmax', 1.5,
                       'delta_VFG=0.03,0.06,0.12,0.24,0.48;delta_IG=0.03,0.1,0.2,0.4;'
                       'delta_GI=0.02,0.05,0.1,0.2;delta_RF=0.01,0.02,0.05,0.1,0.2',
                       ('sens_rates_*.csv',), 'sens_rates_<seq>'),
    'grid': _job('grid', 'grid', 'thesis_cmax', 1.5,
                 'delta_FR=0.02,0.05,0.10,0.30,0.50;delta_IMU=0.1,0.3,0.5,0.7,0.9',
                 ('grid_*.csv',), 'grid_<seq>_FR<x>'),
    'sweep_cmax': _job('sweep', 'coord', 'thesis_cmax', 3.0,
                       'delta_IMU=0,0.1,0.5;delta_FR=0.1,0.3,0.5,0.7,0.9;n_iters=25,50,150;dt=0.01,0.05',
                       ('sweep_*_cmax_*.csv',), 'sweep_<seq>_cmax_<v>',
                       base='delta_FR=0.5,delta_IMU=0.3'),
    'sweep_imu': _job('sweep', 'coord', 'thesis_imu', 3.0, 'delta_IMU=0.1,0.3,0.5',
                      ('sweep_*_imu_*.csv',), 'sweep_<seq>_imu_<v>',
                      base='delta_FR=0.5,delta_IMU=0.3'),
    'ablation': _job('ablation', 'coord', 'thesis_cmax', 1.5,
                     'delta_shrinkI=0,1;delta_map=0.01,0.03;delta_curl=0.3,0.6',
                     ('*.csv',), 'ablation_<seq>_<v>', seqs='boxes,poster,dynamic,bicycle'),
    **{f'best_image_{m}': _job('best_image', 'grid', m, 1.5, 'dt=0.01;delta_curl=0,0.6',
                               _S(m, _ALL), f'best_image_<seq>_{m}')
       for m in ('thesis', 'thesis_cmax')},
    **{f'maps_poster_{m}': _job('maps_poster', 'coord', m, 0.88, 'n_iters=75',
                                (f'sweep_{m}.csv',), f'maps_poster_{m}', seqs='poster')
       for m in MODELS},
}
LIVE_JOB, LIVE_RUNS = {}, {}                    # section ID -> the job / runs Live made


def job_axes(job):
    return _parse_vary(SWEEP_JOBS[job]['vary'])


def job_base(job):
    """cmd_sweep's parsing of --base."""
    base = {}
    for part in filter(None, (s.strip() for s in SWEEP_JOBS[job]['base'].split(','))):
        k, _, v = part.partition('=')
        base[k.strip()] = v.strip() if k.strip() == 'poisson' else float(v)
    return base


def panel_changes(p):
    """The panel's sweepable values that differ from the report's defaults."""
    return [k for k in DELTAS + ['n_iters', 'dt'] if p[k] != DEFAULTS[k]]


def pick_job(p, jobs):
    """Live: the job of the panel's model whose axes hold a changed value, else its first."""
    mine = [j for j in jobs if SWEEP_JOBS[j]['model'] == p['model']] or list(jobs)
    ch = panel_changes(p)
    return next((j for j in mine if any(a in ch for a, _ in job_axes(j))), mine[0])


def _held(name, vals, base):
    """grid mode: an axis not being varied sits at the base, else at the adopted
    value if the axis has it, else at its first value."""
    if name in base:
        return base[name]
    adopted = {**THESIS_PARAMS, 'n_iters': ITERS_PER_FRAME, 'dt': 0.02}.get(name)
    return adopted if adopted in vals else vals[0]


def sweep_points(job, p):
    """cmd_sweep's points (run_experiments._points) cut to the panel's sweep extent."""
    J, axes, base = SWEEP_JOBS[job], job_axes(job), job_base(job)
    if p['sweep'] == 'full':
        return _points(axes, J['mode'], base)
    held = {n: _held(n, v, base) for n, v in axes} if J['mode'] == 'grid' else {}
    if p['sweep'] == 'one point':
        return [{**base, **held}]
    ch = panel_changes(p)
    name, vals = next(((n, v) for n, v in axes if n in ch), axes[0])
    return [{**base, **held, name: v} for v in vals]


def sweep_duration(p, J):
    """Live track length: the job's duration, capped at n_frames frames of 20 ms."""
    return min(J['duration'], p['n_frames'] * 0.02)


def sweep_seg(p, J):
    labels = J['seqs'].split(',')
    if p['segment'] in labels:
        return p['seg']
    print(f"the job runs {J['seqs']} only; using {labels[0]} instead of {p['segment']}")
    return next(s for s in SEGMENTS if s[3] == labels[0])


def sweep_rows(name, patterns=('*.csv',), fallback=True):
    """The job csvs in experiments/<name>/ (for sens, failing that in
    report/experiments/, as make_tables._sens_rows and text_numbers.sens_rows do),
    read with csv.DictReader. A file missing a cmd_sweep column is a [DIFF]."""
    find = lambda top: sorted(q for pat in patterns for q in glob.glob(os.path.join(top, pat)))
    paths = find(os.path.join('experiments', name))
    if not paths and name == 'sens' and fallback:
        paths = find(os.path.join('report', 'experiments'))
    rows = []
    for path in paths:
        with open(path, newline='', encoding='utf-8') as f:
            rd = csv.DictReader(f)
            gone = [c for c in SWEEP_COLS if c not in (rd.fieldnames or [])]
            if gone:
                print(f'[DIFF] {os.path.basename(path)}: missing column(s) {", ".join(gone)} '
                      '(an older csv; rerun its job)')
            rows += list(rd)
    if paths:
        print(f'   {len(rows)} rows from {len(paths)} csv(s) in {os.path.dirname(paths[0])}')
    return rows


def have_cols(rows, cols, ID):
    """Gate a section on the columns it reads; a missing one skips it (the
    [DIFF] naming it was printed by sweep_rows)."""
    gone = sorted({c for r in rows for c in cols if c not in r})
    if gone:
        print(f'({ID}: rows lack column(s) {", ".join(gone)}; the section is skipped)')
        READY[ID] = False
    return not gone


def sweep_row(rc, s, secs=0):
    """cmd_sweep's csv row for one run summary: same columns, same rounding."""
    g = lambda k: s.get(k, float('nan'))
    vals = ([rc.dataset, rc.segment_id, rc.model, round(rc.frame_duration * 1000, 1),
             rc.n_frames, rc.n_iters, rc.poisson] + [rc.params.get(d, '') for d in ALL_DELTAS]
            + [round(s['mean_err_deg_s'], 2), round(s['median_err_deg_s'], 2),
               round(s['mean_dir_err_deg'], 2), round(s['mean_beta'], 3),
               round(g('low_freq_share'), 4), round(g('contrast'), 4), round(g('curl_share'), 4),
               round(g('recon_r_mean'), 4), round(g('recon_r_min'), 4),
               round(g('recon_r_mean_inloop'), 4), round(g('recon_r_min_inloop'), 4),
               round(g('blowup_frac'), 4), round(secs)])
    return dict(zip(SWEEP_COLS, vals))


def _cfg_key(rc):
    return json.dumps({k: v for k, v in rc.to_dict().items() if k != 'out_root'},
                      sort_keys=True, default=str)


def note_sweep(p, jobs, nf):
    """Redraw: say that the saved rows are shown, and which panel values need Live."""
    off = [f'{k} {p[k]}' for k in panel_changes(p) + ['poisson'] if p[k] != DEFAULTS[k]]
    if p['n_frames'] != nf:
        off.append(f"n_frames {p['n_frames']} (saved: {nf})")
    if p['sweep'] != DEFAULTS['sweep']:
        off.append(f"sweep {p['sweep']!r}")
    print(f"Redraw: the saved rows of {', '.join(jobs)}: every point, every sequence, "
          f"report settings.")
    if off:
        print(f"No saved run has the panel's {', '.join(off)}: rerun it in Live mode.")


def run_sweep(p, ID, jobs):
    """Step 'the runs' of a sweep section. Redraw: the saved rows of the jobs.
    Live: the panel's job (pick_job), its points cut to the sweep extent, run on
    the panel's segment into walk_dir(ID), one cmd_sweep row each. Sets READY[ID]."""
    jobs = [jobs] if isinstance(jobs, str) else list(jobs)
    if p['mode'] == 'Redraw':
        J0 = SWEEP_JOBS[jobs[0]]
        note_sweep(p, jobs, int(round(J0['duration'] / 0.02)))
        rows = []
        for j in jobs:
            J = SWEEP_JOBS[j]
            got = sweep_rows(J['name'], J['csv'])
            if not got:
                need([os.path.join('experiments', J['name'], J['csv'][0])], [J['run_all']])
            rows += got
        READY[ID] = bool(rows)
        return rows
    job = pick_job(p, jobs)
    J = SWEEP_JOBS[job]
    ds, sid, t0, lab = sweep_seg(p, J)
    READY[ID], LIVE_JOB[ID], LIVE_RUNS[ID] = False, job, {}
    if not need([os.path.join('data', ds)]):
        return []
    pts, dur = sweep_points(job, p), sweep_duration(p, J)
    nfs = [int(round(dur / (pt.get('dt') or 0.02))) for pt in pts]
    print(f"Live: job {job} ({J['model']}) on {lab}, extent {p['sweep']!r}: "
          f"{len(pts)} runs, {sum(nfs)} frames in total ({dur:g} s each)")
    if p['sweep'] == 'one point' and panel_changes(p):
        print(f"   'one point' runs the base point; the panel's {', '.join(panel_changes(p))} "
              "pick the axis of 'one axis'")
    root = walk_dir(ID)
    path = os.path.join(root, f'{job}.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(SWEEP_COLS)
        for pt, nf in zip(pts, nfs):
            poisson, dt, n_iters, deltas = _split(pt)
            rc = run_cfg(ds, sid, t0, J['model'], dt or 0.02, nf, out_root=root,
                         poisson=poisson, deltas=deltas, n_iters=n_iters)
            E.experiment_tracking(rc, save_frames=False)
            row = sweep_row(rc, experiment_tracking.read_summary(rc.output_dir))
            LIVE_RUNS[ID][_cfg_key(rc)] = rc.output_dir
            wr.writerow([row[c] for c in SWEEP_COLS])
            f.flush()
            print('   ' + ' '.join(f'{k}={v}' for k, v in sorted(pt.items()))
                  + f"  err {row['err']} deg/s, beta {row['beta']}")
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    READY[ID] = True
    return rows


def check_live_sweep(p, ID, job, rows):
    """Live same-code check: run cmd_sweep itself on the job, with its run step
    replaced by a recorder that hands back the Live runs' summaries (a point
    the Live run did not make is refused). Same configs? Same csv rows?"""
    J = SWEEP_JOBS[job]
    runs = LIVE_RUNS.get(ID, {})
    seen = []
    def record(E_, rc, save_frames, stride):
        k = _cfg_key(rc)
        if k not in runs:
            raise LookupError('not a point of the Live run')
        seen.append(k)
        return experiment_tracking.read_summary(runs[k]), 0
    name = os.path.join('walkthrough', ID, 'check')
    args = argparse.Namespace(vary=J['vary'], base=J['base'], mode=J['mode'], model=J['model'],
                              duration=sweep_duration(p, J), dt=0.02,
                              sequences=sweep_seg(p, J)[3], name=name, out=f'{job}_check.csv',
                              no_frames=True, frame_stride=1)
    said = io.StringIO()
    with redirect(run_experiments, _run=record), contextlib.redirect_stdout(said), \
         contextlib.redirect_stderr(said):
        run_experiments.cmd_sweep(E, args)
    miss = len(set(runs) - set(seen))
    print(f'[ok] config ({len(runs)} of {len(runs)} Live points are cmd_sweep points)' if not miss
          else f'[DIFF] config: {miss} of {len(runs)} Live points were not made by cmd_sweep')
    with open(os.path.join('experiments', name, f'{job}_check.csv'), newline='') as f:
        theirs = {tuple(r[c] for c in SWEEP_COLS[3:16]): r for r in csv.DictReader(f)}
    num = SWEEP_COLS[16:-1]
    for r in rows:
        t = theirs.get(tuple(r[c] for c in SWEEP_COLS[3:16]))
        what = 'cmd_sweep row ' + ' '.join(f'{c}={r[c]}' for c in SWEEP_COLS[3:16]
                                           if c in ('dt_ms', 'n_iters') or r[c] != str(THESIS_PARAMS.get(c)))
        if t is None:
            print(f'[DIFF] {what}: no such row in the cmd_sweep csv')
        else:
            check(what, {c: float(r[c]) for c in num}, {c: float(t[c]) for c in num})


def report_figs(fn, ID, show=False, **attrs):
    """Call a make_figures function with FIG/save redirected (never report/figures/);
    every figure it saves is kept, and shown + saved to walk_dir(ID) if show."""
    figs = []
    def keep(fig, path, show_=False):
        figs.append(fig)
        if show:
            style_save(fig, path, show=True)
    out = walk_dir(ID) if show else os.path.join(walk_dir(ID), 'check')
    quiet = contextlib.nullcontext() if show else contextlib.redirect_stdout(io.StringIO())
    with redirect(make_figures, FIG=out, save=keep, **attrs), quiet:
        fn(E)
    if not show:
        for fg in figs:
            plt.close(fg)
    return figs


def fig_images(figs):
    """Every image drawn in the figures, flattened in axes order (loose image check)."""
    arrs = [np.ma.filled(np.ma.asarray(im.get_array(), float), np.nan).ravel()
            for fg in figs for ax in fg.axes for im in ax.images]
    return np.concatenate(arrs) if arrs else np.array([])
'''),

('code', A8, r'''
# step 1 -- the runs: the saved sens rows (Redraw) or the panel's sens job (Live)
p = params()
JOBS_A8 = ['sens_cmax', 'sens_imu', 'sens_rates']
rows_A8 = run_sweep(p, 'A8', JOBS_A8)
if READY['A8']:
    have_cols(rows_A8, SWEEP_COLS[:-1], 'A8')
'''),

('code', A8, r'''
%%section A8
# step 2 -- the rows as cells: make_tables.py::_sens_rows keys every row by
# (model, the 8 SENS_KEYS); a point varies one key away from the adopted base
from make_tables import SENS_KEYS, SEQS as TAB_SEQS, fmt, mark_best, tabular
from text_numbers import BASE as TN_BASE, axis as tn_axis   # one-at-a-time selection
cell_A8 = {}
for r in rows_A8:
    key = (r['model'],) + tuple(round(float(r[k]), 6) for k in SENS_KEYS)
    cell_A8.setdefault(key, {})[r['sequence']] = r
base_A8 = dict({k: THESIS_PARAMS[k] for k in SENS_KEYS[:6]}, n_iters=ITERS_PER_FRAME, dt_ms=20.0)
pts_A8 = []
for key, by_seq in cell_A8.items():
    off = [k for k, v in zip(SENS_KEYS, key[1:]) if abs(v - base_A8[k]) > 1e-9]
    pts_A8.append([key[0], ','.join(off) or '(base)',
                   ','.join(f'{v:g}' for k, v in zip(SENS_KEYS, key[1:]) if k in off) or '-',
                   len(by_seq), ' '.join(sorted(s.split('_')[0] for s in by_seq))])
print(f'{len(cell_A8)} points; adopted base: {base_A8}')
display(pd.DataFrame(pts_A8, columns=['model', 'varied', 'value', 'n seq', 'sequences'])
        .sort_values(['model', 'varied']).reset_index(drop=True))
'''),

('md', A8, r'''
### tab:sweep -- `make_tables.py::table_sweep`

One block per parameter. The adopted value of each block is bold; the lowest
error of each block on each sequence is underlined (the δ_anchor = 0 ablation
row excluded). err in deg/s, dir in degrees.
'''),

('code', A8, r'''
%%section A8
# step 3 -- table_sweep, unrolled (its block() is copied; fmt/mark_best/tabular reused)
seen_A8 = []
def block_A8(title, model, name, values, fmt_label=lambda v: f'{v:g}'):
    out = [r'\midrule', rf'\multicolumn{{12}}{{@{{}}l}}{{{title}}}\\']
    rows = []
    for v in values:
        pt = dict(base_A8, **{name: v})
        rows.append((v, cell_A8.get((model,) + tuple(round(float(pt[k]), 6) for k in SENS_KEYS), {})))
    err = {}
    for ds, *_ in TAB_SEQS:
        err[ds] = mark_best({v: fmt(rs[ds]['err']) if ds in rs else '---'
                             for v, rs in rows if not (name == 'delta_IMU' and v == 0)},
                            lambda x: x, mark=r'\underline')
    for v, rs in rows:
        label = fmt_label(v)
        lab = rf'$\mathbf{{{label}}}$' if abs(v - base_A8[name]) < 1e-9 else f'${label}$'
        vals = ' & '.join((f'{err[ds].get(v, fmt(rs[ds]["err"]))} & {fmt(rs[ds]["dir"])}')
                          if ds in rs else '--- & ---' for ds, *_ in TAB_SEQS)
        out.append(f'& {lab:<20}& {vals} \\\\')
        seen_A8.append([model, name, v] + [float(rs[ds]['err']) if ds in rs else np.nan
                                           for ds, *_ in TAB_SEQS])
    return out

ms_A8 = lambda v: rf'{v:g}\,\mathrm{{ms}}'
body_A8 = (block_A8(r'\emph{anchor weight} $\delta_{\text{anchor}}$, contrast anchor', 'thesis_cmax',
                    'delta_IMU', [0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
                    lambda v: r'0\ \text{(abl.)}' if v == 0 else f'{v:g}')
           + block_A8(r'\emph{anchor weight} $\delta_{\text{anchor}}$, gyroscope', 'thesis_imu',
                      'delta_IMU', [0.1, 0.3, 0.5, 0.7, 0.9])
           + block_A8(r'\emph{kinematic weight} $\delta_{F\!R}$', 'thesis_cmax',
                      'delta_FR', [0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9])
           + block_A8(r'\emph{flow constraint} $\delta_{V\!F\!G}$', 'thesis_cmax',
                      'delta_VFG', [0.03, 0.06, 0.12, 0.24, 0.48])
           + block_A8(r'\emph{flow from} $\wvec$, $\delta_{R\!F}$', 'thesis_cmax',
                      'delta_RF', [0.01, 0.02, 0.05, 0.1, 0.2])
           + block_A8(r'\emph{gradient from intensity} $\delta_{I\!G}$', 'thesis_cmax',
                      'delta_IG', [0.03, 0.1, 0.2, 0.4])
           + block_A8(r'\emph{intensity from gradient} $\delta_{G\!I}$', 'thesis_cmax',
                      'delta_GI', [0.02, 0.05, 0.1, 0.2])
           + block_A8(r'\emph{iterations per frame}', 'thesis_cmax', 'n_iters', [25, 50, 75, 150])
           + block_A8(r'\emph{accumulation window} $\Delta t$', 'thesis_cmax', 'dt_ms',
                      [5.0, 10.0, 20.0, 40.0], ms_A8))
head_A8 = ['& & ' + '\n& '.join(rf'\multicolumn{{2}}{{c}}{{{d}}}' for _, d, *_ in TAB_SEQS) + r' \\',
           ''.join(rf'\cmidrule(lr){{{3 + 2*i}-{4 + 2*i}}}' for i in range(len(TAB_SEQS))),
           '& & ' + ' & '.join(['err & dir'] * len(TAB_SEQS)) + r' \\']
table_A8 = tabular('@{}ll' + 'rr' * len(TAB_SEQS) + '@{}', head_A8, body_A8)
print('err (deg/s) per block, value and sequence:')
display(pd.DataFrame(seen_A8, columns=['model', 'parameter', 'value'] + [s for _, _, s, _ in TAB_SEQS]))
print(table_A8)
'''),

('md', A8, r'''
### fig_sensitivity -- `make_figures.py::fig_sensitivity` (not in the tex)

`sensitivity_data` averages err and read-out |r| over the three real and the
two synthetic sequences per swept value; its κ panel pools A9's grid by
κ = δ_FR/δ_anchor, and the δ_anchor = 0 row is the "no anchor" point. **Flag:**
it reads `experiments/sens/` only, with no fallback to `report/experiments/`
(unlike table_sweep and text_numbers), so on a machine holding only the
fallback csvs its one-at-a-time panels are empty while tab:sweep is full.
The data are unrolled; the drawing is the report's own code fed these data.
'''),

('code', A8, r'''
%%section A8
# step 4 -- make_figures.py::sensitivity_data, unrolled; then fig_sensitivity draws it
from make_figures import SENS_REAL, SENS_SYN, SENS_BASE, SENS_AXES
if p['mode'] == 'Redraw':
    grid_rows_A8 = sweep_rows('grid', ('*.csv',))                 # A9's grid: the kappa panel
    fig_rows_A8 = sweep_rows('sens', ('sens_*.csv',), fallback=False)   # experiments/sens only
    if not fig_rows_A8:
        print('   fig_sensitivity: no csv in experiments/sens/ (it has no fallback): '
              'its one-at-a-time panels stay empty')
else:
    grid_rows_A8, fig_rows_A8 = [], rows_A8

def sensitivity_data_A8(grid_rows, sens_rows):
    """make_figures.sensitivity_data on the given rows: {panel: {value: {kind: (err, |r|)}}}."""
    def mean(rows):
        out = {}
        for kind, seqs in (('real', SENS_REAL), ('syn', SENS_SYN)):
            sel = [r for r in rows if r['sequence'] in seqs]
            if sel:
                out[kind] = (np.mean([float(r['err']) for r in sel]),
                             np.mean([float(r['recon_mean']) for r in sel]))
        return out
    data, by_k = {}, {}
    for r in (r for r in grid_rows if r['model'] == 'thesis_cmax'):
        by_k.setdefault(round(float(r['delta_FR']) / float(r['delta_IMU']), 4), []).append(r)
    data['kappa'] = {k: mean(v) for k, v in sorted(by_k.items())}
    sens = [r for r in sens_rows if r['model'] == 'thesis_cmax']
    for key, _ in SENS_AXES + [('delta_IMU', '')]:
        by_v = {}
        for r in sens:
            if all(abs(float(r[k]) - v) < 1e-9 for k, v in SENS_BASE.items() if k != key):
                by_v.setdefault(float(r[key]), []).append(r)
        data[key] = {v: mean(rs) for v, rs in sorted(by_v.items())}
    return data, data.pop('delta_IMU').get(0.0)          # delta_anchor = 0 is thesis

data_A8, no_anchor_A8 = sensitivity_data_A8(grid_rows_A8, fig_rows_A8)
display(pd.DataFrame([[pn, v, kind, e, rr] for pn, d in data_A8.items() for v, kk in d.items()
                      for kind, (e, rr) in kk.items()],
                     columns=['panel', 'value', 'kind', 'err deg/s', '|r| read-out']))
print('no anchor (delta_anchor = 0):', {k: tuple(round(x, 2) for x in v)
                                        for k, v in (no_anchor_A8 or {}).items()})
figs_A8 = report_figs(make_figures.fig_sensitivity, 'A8', show=True,
                      sensitivity_data=lambda: (data_A8, no_anchor_A8))
'''),

('code', A8, r'''
%%section A8
# step 5 -- the text numbers (Redraw only: they read the report's rows)
from text_numbers import REAL as TN_REAL, SYN as TN_SYN
def text_numbers_A8(rows, grid_rows):
    """The five text_numbers functions on the sens rows (= text_numbers.sens_rows())."""
    fl = lambda r, k='err': float(r[k])
    lines = {k: [] for k in ('sensitivity_both', 'sweep_ends', 'anchor_ablation',
                             'anchors_across_weights', 'kappa_rows')}
    # text_numbers.sensitivity_both: real/synthetic means per swept value, grid by kappa
    L = lines['sensitivity_both']
    L.append('value: err real/synthetic (deg/s) | read-out |r| real/synthetic | in-loop |r| '
             'real/synthetic | fraction of frames in an excursion real/synthetic')
    def line_A8(label, rs):
        cells = []
        for k, fm in (('err', '5.1f'), ('recon_mean', '.2f'), ('recon_inloop', '.2f'), ('blowup', '.3f')):
            m = [np.mean([fl(r, k) for r in rs if r['sequence'] in [s[0] for s in g]])
                 for g in (TN_REAL, TN_SYN)]
            cells.append(f'{m[0]:{fm}} {m[1]:{fm}}')
        L.append(f'  {label:<12s} ' + ' | '.join(cells))
    bk = {}
    for r in (r for r in grid_rows if r['model'] == 'thesis_cmax'):
        an = r.get('delta_IMU', r.get('delta_anchor'))
        bk.setdefault(round(fl(r, 'delta_FR') / float(an), 4), []).append(r)
    if bk:
        L.append('grid, by kappa = delta_FR / delta_anchor:')
        for k in sorted(bk):
            line_A8(f'kappa {k:g}', bk[k])
        spread = 0.0
        for rs in bk.values():
            for s in SEGMENTS:
                v = [fl(r, 'recon_mean') for r in rs if r['sequence'] == s[0]]
                if len(v) > 1:
                    spread = max(spread, max(v) - min(v))
        L.append(f'  largest |r| spread between settings of equal kappa: {spread:.3f}')
    for model, keys in (('thesis_cmax', ('delta_IMU', 'delta_FR', 'delta_VFG', 'delta_RF', 'delta_IG',
                                         'delta_GI', 'n_iters', 'dt_ms')), ('thesis_imu', ('delta_IMU',))):
        for key in keys:
            by_v = {}
            for s in SEGMENTS:
                for v, r in tn_axis(rows, key, s[0], model).items():
                    by_v.setdefault(v, []).append(r)
            if by_v:
                L.append(f'{model} {key}:')
                for v in sorted(by_v):
                    line_A8(f'{v:g}', by_v[v])
    lo, hi = 0.01, TN_BASE['delta_RF']
    cells = []
    for s in TN_REAL:
        a = tn_axis(rows, 'delta_RF', s[0])
        if lo in a and hi in a:
            cells.append(f'{s[3]} err {fl(a[lo]):.1f} vs {fl(a[hi]):.1f}, |r| '
                         f'{fl(a[lo], "recon_mean"):.2f} vs {fl(a[hi], "recon_mean"):.2f}')
    if cells:
        L.append(f'delta_RF {lo:g} vs {hi:g} per real sequence: ' + '; '.join(cells))
    # text_numbers.sweep_ends: the best value of each axis per sequence, and its range
    L = lines['sweep_ends']
    for key in ('delta_FR', 'delta_IMU', 'delta_VFG', 'delta_RF', 'delta_IG', 'delta_GI', 'n_iters', 'dt_ms'):
        cells = []
        for s in SEGMENTS:
            a = {k: v for k, v in tn_axis(rows, key, s[0]).items() if not (key == 'delta_IMU' and k == 0)}
            if a:
                e = {k: fl(v) for k, v in a.items()}
                cells.append(f'{s[3]} best {min(e, key=e.get):g} (range {max(e.values())-min(e.values()):.2f})')
        L.append(f'{key:10s} ' + '; '.join(cells))
    # text_numbers.anchor_ablation: err over the anchor weight, both anchors
    L = lines['anchor_ablation']
    for model in ('thesis_cmax', 'thesis_imu'):
        for s in SEGMENTS:
            a = tn_axis(rows, 'delta_IMU', s[0], model)
            L.append(f'{model:11s} {s[3]:8s} ' + ', '.join(f'{k:g}: {fl(v):.1f}' for k, v in a.items()))
    for model in ('thesis_cmax', 'thesis_imu'):
        syn = [tn_axis(rows, 'delta_IMU', s[0], model) for s in TN_SYN]
        ws = sorted(set.intersection(*[set(a) for a in syn])) if syn else []
        if ws:
            L.append(f'{model} synthetic mean: ' + ', '.join(
                f'{w:g}: {np.mean([fl(a[w]) for a in syn]):.1f}' for w in ws))
    # text_numbers.anchors_across_weights: CMax minus gyro anchor at each weight
    L = lines['anchors_across_weights']
    worst = 0.0
    for seg in SEGMENTS:
        c, i = tn_axis(rows, 'delta_IMU', seg[0]), tn_axis(rows, 'delta_IMU', seg[0], 'thesis_imu')
        diffs = {w: fl(c[w]) - fl(i[w]) for w in c if w in i and w > 0}
        if seg in TN_REAL:
            worst = max([worst] + [abs(v) for v in diffs.values()])
        L.append(f'{seg[3]:8s} CMax minus gyro: ' + ', '.join(f'{w:g}: {v:+.1f}' for w, v in diffs.items())
                 + (f';  no anchor: {fl(c[0.0]):.1f}' if 0.0 in c else ''))
    L.append(f'largest |CMax - gyro| on the real sequences: {worst:.1f} deg/s')
    # text_numbers.kappa_rows: anchor 0.1 vs dFR 0.5 (both kappa 1); beta over dFR
    L = lines['kappa_rows']
    for seg in SEGMENTS:
        an, fr = tn_axis(rows, 'delta_IMU', seg[0]), tn_axis(rows, 'delta_FR', seg[0])
        if 0.1 in an and 0.5 in fr:
            b = [fl(r, 'beta') for r in fr.values()]
            mono = all(x < y for x, y in zip(b, b[1:]))
            L.append(f'{seg[3]:8s} anchor 0.1 vs dFR 0.5 (kappa 1): {fl(an[0.1]):.2f} vs {fl(fr[0.5]):.2f}; '
                     f'beta over dFR {min(fr):g}-{max(fr):g}: {b[0]:.2f} -> {b[-1]:.2f}, '
                     f'{"rises at every step" if mono else "NOT monotonic"}')
    return lines


lines_A8 = {}
if p['mode'] == 'Redraw':
    lines_A8 = text_numbers_A8(rows_A8, grid_rows_A8)
    for k, ls in lines_A8.items():
        print(f'== {k}'); print('\n'.join(ls) or '(no rows)')
else:
    print('Live: the text numbers read the report rows; switch to Redraw to see them.')
'''),

('code', A8, r'''
%%section A8
# same-code check -- cmd_sweep (Live); table_sweep, sensitivity_data, text_numbers (Redraw)
if p['mode'] == 'Live':
    check_live_sweep(p, 'A8', LIVE_JOB['A8'], rows_A8)
else:
    check('table_sweep', nums(table_A8.splitlines()), nums(make_tables.table_sweep().splitlines()))
    flat_A8 = lambda d, na: {**{f'{pn}|{v:g}|{kind}|{j}': x for pn, dd in d.items() for v, kk in dd.items()
                                for kind, xs in kk.items() for j, x in enumerate(xs)},
                             **{f'no_anchor|{kind}|{j}': x for kind, xs in (na or {}).items()
                                for j, x in enumerate(xs)}}
    theirs_A8 = make_figures.sensitivity_data()
    check('fig_sensitivity data', flat_A8(data_A8, no_anchor_A8), flat_A8(*theirs_A8))
    for name in lines_A8:
        with redirect(text_numbers, OUT=[]), contextlib.redirect_stdout(io.StringIO()):
            getattr(text_numbers, name)()
            theirs = [l for l in text_numbers.OUT if l and not l.startswith(('==', '   (missing'))]
        check(f'text_numbers.{name}', nums(lines_A8[name]), nums(theirs))
'''),
]
