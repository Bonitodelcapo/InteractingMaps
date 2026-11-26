A7 = ('A7',)

CELLS = [
('md', A7, r'''
## A7 · Anchor across poster segments

**Question.** Does the ranking of the models hold on the other poster segments?
Every model on poster seg_A (fastest), seg_C and seg_D (slowest), 3 s each,
with the gyro floor of each segment.

**Made by** `evaluation.py --exp 2 --dataset poster_rotation --segment <seg>
--model <model> --no-frames`, `run_all.sh` jobs `anchor_<seg>_<model>` (seg_A /
seg_D × cook, thesis, thesis_imu, thesis_cmax_v2) → runs in
`results/poster_rotation/<model>/` (150 frames, config.py's step sizes). The
other cells are runs made elsewhere: seg_C is the 3 s main run of A2
(`main_poster_<model>`), thesis_cmax on seg_A / seg_D the adopted β run of A6
(`beta_<i>`).

**Read by** `report/make_tables.py::table_anchor` (tab:anchor), including
`make_tables.py::gyro_floor`. **Not in the tex**: `make_tables.py` writes
`table_anchor.tex`, but `bare_jrnl.tex` does not input it.

**Steps:** the runs (found, or made now) → the gyro floor per segment → the
table → same-code check. Live runs the panel's model on poster seg_A
(`one point`) or on all three segments (`one axis` / `full`).
'''),

('code', A7, r'''
# step 1 -- the runs: found by table_anchor's lookup (Redraw) or made now (Live)
from report_runs import BETA_CONFIGS
p = params()
SEGS_A7 = make_tables.ANCHOR_SEGMENTS                      # seg_A, seg_C, seg_D
T0_A7 = {s['id']: s['t_start'] for s in E.DATASET_SEGMENTS['poster_rotation']}


def anchor_job(sid, model):
    """The run_all.sh job that makes table_anchor's run of (sid, model)."""
    if sid == 'seg_C':
        return f'main_poster_{model}'
    if model == 'thesis_cmax':                              # the adopted beta run
        seg = ('poster_rotation', sid, T0_A7[sid])
        cfg = ('thesis_cmax', ADOPTED['delta_FR'], ADOPTED['delta_IMU'])
        return f'beta_{3 * BETA_SEGMENTS.index(seg) + BETA_CONFIGS.index(cfg)}'
    return f'anchor_{sid}_{model}'


runs_A7 = {}
if p['mode'] == 'Redraw':
    note_redraw(p, 150)
    miss = []
    for sid in SEGS_A7:
        for _disp, k in make_tables.MODELS:
            # table_anchor: find_run on results/, 150 frames, every step size as in config.py
            d = experiment_tracking.find_run('results', 'poster_rotation', k, sid, 150,
                                             make_tables.expected_params(k))
            if d is None:
                print(f'missing: results/poster_rotation/{k}/{sid}_* (150 frames, config.py step sizes)')
                miss.append(anchor_job(sid, k))
            else:
                runs_A7[sid, k] = d
    if miss:
        print('make with run_all.sh job(s): ' + ', '.join(miss))
    READY['A7'] = bool(runs_A7)
else:
    nf_A7 = min(p['n_frames'], 150)
    sids_A7 = ['seg_A'] if p['sweep'] == 'one point' else list(SEGS_A7)
    print(f"{len(sids_A7)} run(s) of {nf_A7} frames = {len(sids_A7) * nf_A7} frames: "
          f"{p['model']} on poster {', '.join(sids_A7)}")
    READY['A7'] = False
    if need([os.path.join('data', 'poster_rotation')]):
        deltas = {k: v for k, v in p['deltas'].items() if v != THESIS_PARAMS[k]}
        if p['model'] == 'cook' and deltas:
            print(f'cook has its own step sizes; ignoring {sorted(deltas)}')
            deltas = {}
        for sid in sids_A7:
            # evaluation.py --exp 2's RunConfig, plus the panel's values and out_root
            rc = E.RunConfig(dataset='poster_rotation', model=p['model'], segment=sid,
                             n_frames=nf_A7, frame_duration=p['dt'], n_iters=p['n_iters'],
                             poisson=p['poisson'], deltas=deltas or None,
                             out_root=walk_dir('A7'))
            E.experiment_tracking(rc, save_frames=False)
            runs_A7[sid, p['model']] = rc.output_dir
        READY['A7'] = True
for (sid, k), d in runs_A7.items():
    print(f'{sid} {k:16s} {d}')
'''),

('code', A7, r'''
%%section A7
# step 2 -- make_tables.py::gyro_floor: the gyro against the reference, from the
# thesis+IMU run's tracking.csv (Live: from the panel's run; every run stores both)
floors_A7 = {}
for sid in SEGS_A7:
    d = runs_A7.get((sid, 'thesis_imu')) or next((d for (s, _k), d in runs_A7.items() if s == sid), None)
    if d is None:
        continue
    tr = tracking(d)
    g, t_ = tr['imu'], tr['gt']                             # gyro, reference (code unit)
    cos = (g * t_).sum(1) / (np.linalg.norm(g, axis=1) * np.linalg.norm(t_, axis=1))
    floors_A7[sid] = (np.degrees(np.linalg.norm(g - t_, axis=1)).mean(),
                      np.degrees(np.arccos(np.clip(cos, -1, 1))).mean(),
                      np.linalg.norm(t_, axis=1).mean())
    print(f'{sid}: gyro floor {floors_A7[sid][0]:.2f} deg/s, direction {floors_A7[sid][1]:.2f} deg; '
          f'mean |w_ref| {to_deg_s(floors_A7[sid][2]):.1f} deg/s  ({os.path.basename(d)})')
'''),

('md', A7, r'''
### tab:anchor -- `make_tables.py::table_anchor` (not in the tex)

One block per segment, slowest last: err, dir and β of every model from its
run summary, and the gyro floor in italics. The row label is the segment's mean
|ω| in deg/s.
'''),

('code', A7, r'''
%%section A7
# step 3 -- table_anchor, unrolled (fmt / tabular / floor_label are its formatters)
body_A7, cells_A7 = [], []
for sid in SEGS_A7:
    fl = floors_A7.get(sid) if (sid, 'thesis_imu') in runs_A7 or p['mode'] == 'Live' else None
    body_A7 += [r'\midrule',
                rf'\multirow{{6}}{{*}}{{${to_deg_s(fl[2]):.0f}$\,$^\circ$/s}}' if fl else r'\multirow{6}{*}{---}']
    for mdisp, key in make_tables.MODELS:
        d = runs_A7.get((sid, key))
        if d is None:
            body_A7.append(f' & {mdisp:<22}& --- & --- & --- \\\\')
            continue
        s = experiment_tracking.read_summary(d)
        body_A7.append(f' & {mdisp:<22}& {make_tables.fmt(s["mean_err_deg_s"])} & '
                       f'{make_tables.fmt(s["mean_dir_err_deg"])} & '
                       f'{make_tables.fmt(s["mean_beta"], 1 if s["mean_beta"] >= 100 else 2)} \\\\')
        cells_A7.append([sid, key, s['mean_err_deg_s'], s['mean_dir_err_deg'], s['mean_beta']])
    if fl:
        body_A7.append(rf' & {make_tables.floor_label("poster_rotation")}  & '
                       rf'\emph{{{make_tables.fmt(fl[0])}}} & \emph{{{make_tables.fmt(fl[1])}}} & --- \\')
        cells_A7.append([sid, 'gyro vs ref.', fl[0], fl[1], np.nan])
table_A7 = make_tables.tabular('@{}llrrr@{}',
                               [r'$\|\wvec\|$ & Model & err ($^\circ$/s) & dir ($^\circ$) & $\beta$ \\'],
                               body_A7)
df_A7 = pd.DataFrame(cells_A7, columns=['segment', 'model', 'err (deg/s)', 'dir (deg)', 'beta'])
df_A7.insert(1, '|w_ref| (deg/s)', [to_deg_s(floors_A7[s][2]) if s in floors_A7 else np.nan
                                    for s in df_A7['segment']])
display(df_A7.round(2))
print(table_A7)
'''),

('code', A7, r'''
%%section A7
# same-code check -- gyro_floor and table_anchor themselves (Redraw);
# the run summaries against their tracking.csv and gyro_floor (Live)
import io
for sid, fl in floors_A7.items():
    d = runs_A7.get((sid, 'thesis_imu')) or next(d for (s, _k), d in runs_A7.items() if s == sid)
    check(f'gyro_floor {sid}', fl, make_tables.gyro_floor(d))
if p['mode'] == 'Live':
    for (sid, k), d in runs_A7.items():
        s, tr = experiment_tracking.read_summary(d), tracking(d)
        check(f'summary vs tracking.csv {sid} {k}',
              [s['mean_err_deg_s'], s['mean_dir_err_deg'], s['mean_beta']],
              [tr['err'].mean(), tr['dir'].mean(), tr['beta'].mean()])
else:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        theirs = make_tables.table_anchor()
    check('table_anchor', nums(table_A7.splitlines()), nums(theirs.splitlines()))
'''),
]
