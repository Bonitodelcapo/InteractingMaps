A2 = ('A2',)
DEFS = ('A2', 'A2-defs')

CELLS = [
('md', A2, r'''
## A2 · Every model, 3 s

**Question.** How does every model track ω over a full 3 s segment (150 frames
of 20 ms)?

**Made by** `report/run_experiments.py::cmd_main` with `--duration 3.0`,
`run_all.sh` jobs `main_<seq>_<model>` → runs in `results/`, one row each in
`report/experiments/main_base_<seq>_<model>.csv`.

**Read by** `report/text_numbers.py::poster_segments` and the 3 s half of
`::inloop_identical` (its 1.5 s half is A4). These numbers are quoted in the
report text; no figure or table uses the 3 s runs.

**Steps** (those of `cmd_main`, then of the two readers): reference ω and gyro
→ gyro floor → the run (found or made) → its summary row → the text numbers →
same-code check.
'''),

('code', DEFS, r'''
# A2 definitions -- the unrolled pieces of cmd_main and text_numbers.run,
# reused by A3 (the same command at 1.5 s)
from interacting_maps.eval_io import (load_groundtruth, load_imu, load_omega_gt,
                                      get_reference_omega, get_gyro_for_frame)
from config import get_dataset_paths
from report_runs import BETA_SEGMENTS, ADOPTED

DEG = np.degrees(1.0)
MAIN_COLS = ['sequence', 'segment', 'model', 'err', 'median', 'dir', 'beta',
             'recon', 'recon_inloop', 'blowup', 'floor', 'floor_dir', 'secs']


def mag(v):
    return np.linalg.norm(v, axis=-1)


def deg(a, b):
    """Angle between two vectors, degrees (text_numbers.deg)."""
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def reference_and_gyro(seg, dt, nf):
    """cmd_main: reference omega (scoring) and gyro (model input) per frame.
    In the code's unit, rad/s; every print converts with to_deg_s."""
    ds, _sid, t0, _lab = seg
    p = get_dataset_paths(ds)
    gt, imu = load_groundtruth(p['groundtruth']), load_imu(p['imu'])
    om = load_omega_gt(p.get('omega_gt'))
    ref = np.array([get_reference_omega(gt, imu, t0+k*dt, t0+(k+1)*dt, om)[0]
                    for k in range(nf)])
    gy = np.array([get_gyro_for_frame(imu, t0+k*dt, t0+(k+1)*dt) for k in range(nf)])
    return ref, gy


def ref_dt(p):
    """The frame length the reference is taken over: the report's 20 ms in
    Redraw (its runs are all 20 ms), the panel's dt in Live."""
    return 0.02 if p['mode'] == 'Redraw' else p['dt']


def stale_note(d):
    """After a row DIFF: which code made the run on disk."""
    pv = experiment_tracking.read_run(d).get('provenance', {})
    print(f"   run on disk: commit {str(pv.get('git_commit'))[:7]} of {pv.get('timestamp')}; "
          f"the csv was written by the run_all.sh job -- re-sync results/ if they disagree")


def gyro_floor(gy, ref):
    """cmd_main: mean |gyro - reference| (deg/s) and mean direction error (deg)."""
    floor = np.degrees(np.linalg.norm(gy-ref, axis=1)).mean()
    fdir = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', gy, ref) /
           (np.linalg.norm(gy, axis=1)*np.linalg.norm(ref, axis=1)), -1, 1))).mean()
    return floor, fdir


def changed_deltas(p):
    """The panel's deltas that differ from the report's: cmd_main's --vary."""
    return {k: v for k, v in p['deltas'].items() if v != THESIS_PARAMS[k]}


def cook_deltas(p):
    """cmd_main passes --vary deltas to cook too; cook has only the relations of
    COOK_PARAMS, so a changed delta it lacks is dropped (and named)."""
    from config import COOK_PARAMS
    d = changed_deltas(p)
    lost = sorted(k for k in d if k not in COOK_PARAMS)
    if lost:
        print(f'cook has no {lost}; ignoring them')
    return {k: v for k, v in d.items() if k in COOK_PARAMS}


def main_cfg(p, nf, out_root):
    """cmd_main's run_cfg call, with the panel's values. cmd_main passes the
    --vary deltas and poisson only; deltas a model does not have are dropped."""
    ds, sid, t0, _lab = p['seg']
    deltas = cook_deltas(p) if p['model'] == 'cook' else changed_deltas(p)
    if p['n_iters'] != ITERS_PER_FRAME:
        print(f'cmd_main does not vary n_iters; the run uses {ITERS_PER_FRAME}')
    return run_cfg(ds, sid, t0, p['model'], p['dt'], nf, out_root=out_root,
                   poisson=p['poisson'], deltas=deltas)


def main_row(ds, sid, model, s, floor, fdir, secs=0):
    """cmd_main's csv row for one run summary, same columns and rounding."""
    return [ds, sid, model, round(s['mean_err_deg_s'], 2),
            round(s['median_err_deg_s'], 2), round(s['mean_dir_err_deg'], 2),
            round(s['mean_beta'], 3), round(s.get('recon_r_mean', float('nan')), 4),
            round(s.get('recon_r_mean_inloop', float('nan')), 4),
            round(s.get('blowup_frac', float('nan')), 4), round(floor, 2),
            round(fdir, 2), round(secs)]


def run_dir(seg, model, nf, d_fr=None, d_anchor='adopted', curl=0.0, root='results'):
    """text_numbers.run's lookup: the run folder whose stored config matches."""
    ds, sid, t0 = seg[:3]
    prm = {'delta_FR': ADOPTED['delta_FR'] if d_fr is None else d_fr,
           'delta_shrinkI': 0.0, 'delta_map': 0.0, 'delta_curl': curl}
    if model in ('thesis_imu', 'thesis_cmax') and d_anchor is not None:
        prm['delta_IMU'] = ADOPTED['delta_IMU'] if d_anchor == 'adopted' else d_anchor
    d = experiment_tracking.find_run(root, ds, model, sid, nf, prm, dt=0.02, t_start=t0)
    if d is None:
        print(f'   (missing run: {ds}/{sid} {model} n{nf} {prm})')
    return d


def tracking(d):
    """text_numbers.run's read: tracking.csv of one run folder as arrays."""
    with open(os.path.join(d, 'tracking.csv')) as f:
        rows = list(csv.DictReader(f))
    col = lambda *k: np.array([[float(r[c]) for c in k] for r in rows])
    return {'est': col('est_wx', 'est_wy', 'est_wz'), 'gt': col('gt_wx', 'gt_wy', 'gt_wz'),
            'imu': col('imu_wx', 'imu_wy', 'imu_wz'), 'err': col('err_deg_s')[:, 0],
            'dir': col('dir_err_deg')[:, 0], 'beta': col('beta')[:, 0],
            'r': np.abs(col('recon_r')[:, 0]), 'I_std': col('I_std')[:, 0]}


def run_tn(seg, model, nf=75, **kw):
    """text_numbers.run: the arrays of the matching run's tracking.csv, or None."""
    d = run_dir(seg, model, nf, **kw)
    return tracking(d) if d else None


from report_runs import SYNTHETIC
REAL = [s for s in SEGMENTS if s[3] not in SYNTHETIC]          # text_numbers.REAL / SYN
SYN = [s for s in SEGMENTS if s[3] in SYNTHETIC]


def find_runs(p, ID, nf, duration):
    """Step 'the run': Redraw finds the report's run of every model; Live makes
    one run of the panel's model. Sets READY[ID]; returns {model: run folder}."""
    ds, sid, t0, lab = p['seg']
    if p['mode'] == 'Redraw':
        note_redraw(p, nf)
        dirs = {m: run_dir(p['seg'], m, nf) for m in MODELS}
        dirs = {m: d for m, d in dirs.items() if d is not None}
        READY[ID] = need([os.path.join('data', ds)] if dirs else [f'results/{ds}/<a {nf}-frame run>'],
                         [f'{"main" if duration == 3.0 else "dissoc"}_{lab}_<model>'])
    else:
        nf = min(p['n_frames'], nf)
        READY[ID], dirs = False, {}             # until the run has succeeded
        if need([os.path.join('data', ds)]):
            rc = main_cfg(p, nf, walk_dir(ID))
            E.experiment_tracking(rc, save_frames=False)
            dirs, READY[ID] = {p['model']: rc.output_dir}, True
    for m, d in dirs.items():
        print(f'{m:16s} {d}')
    return dirs, nf


def check_live_main(p, ID, nf, runs, rows):
    """Live same-code check: run cmd_main itself with its run step replaced by a
    recorder. Same RunConfig as main_cfg? Same csv row as main_row?"""
    import io, argparse
    seen = []
    def record(E_, rc, save_frames, stride):
        seen.append(rc)
        return experiment_tracking.read_summary(runs[p['model']]), 0
    mc = main_cfg(p, nf, None)
    vary = [f'{k}={v}' for k, v in changed_deltas(p).items()]   # the panel's, as typed
    if p['poisson'] != POISSON_MODE:
        vary.append(f"poisson={p['poisson']}")
    args = argparse.Namespace(dt=p['dt'], duration=nf*p['dt'], sequences=p['segment'],
                              models=p['model'], vary=';'.join(vary), out=f'{ID}_check.csv',
                              no_frames=True, frame_stride=1)
    out = os.path.join(walk_dir(ID), 'check')
    os.makedirs(out, exist_ok=True)
    said = io.StringIO()
    with redirect(run_experiments, _run=record, OUTDIR=out), contextlib.redirect_stdout(said):
        run_experiments.cmd_main(E, args)
    if not seen:                                # cmd_main catches its own errors
        print('[DIFF] config: cmd_main made no run; its last output:')
        for line in said.getvalue().strip().splitlines()[-5:]:
            print('   ' + line)
        return
    a, b = seen[0].to_dict(), mc.to_dict()
    differ = sorted(k for k in a if k != 'out_root' and a[k] != b[k])
    print('[ok] config' if not differ else f'[DIFF] config: {differ}')
    with open(os.path.join(out, f'{ID}_check.csv')) as f:
        got = list(csv.reader(f))[1]
    check('cmd_main row', dict(zip(MAIN_COLS[3:12], map(float, got[3:12]))),
          dict(zip(MAIN_COLS[3:12], rows[p['model']][3:12])))
'''),

('code', A2, r'''
# step 1 -- the run(s): found (Redraw) or made now (Live)
p = params()
runs_A2, nf_A2 = find_runs(p, 'A2', nf=150, duration=3.0)
'''),

('code', A2, r'''
%%section A2
# step 2 -- reference omega (Vicon / omega_gt) and the gyro, per 20 ms frame
ref_A2, gy_A2 = reference_and_gyro(p['seg'], ref_dt(p), nf_A2)
print('reference', ref_A2.shape, ' gyro', gy_A2.shape, ' (frames x xyz); first 4 frames, deg/s:')
print(pd.DataFrame(to_deg_s(np.hstack([ref_A2[:4], gy_A2[:4]])),
                   columns=['ref_x', 'ref_y', 'ref_z', 'gyro_x', 'gyro_y', 'gyro_z']).round(2))
'''),

('code', A2, r'''
%%section A2
# step 3 -- the gyro floor: what the gyro scores against the reference
floor_A2, fdir_A2 = gyro_floor(gy_A2, ref_A2)
print(f'gyro floor {floor_A2:.2f} deg/s, direction {fdir_A2:.2f} deg')
'''),

('code', A2, r'''
%%section A2
# step 4 -- one cmd_main row per run, from its run.json summary
rows_A2 = {m: main_row(*p['seg'][:2], m, experiment_tracking.read_summary(d), floor_A2, fdir_A2)
           for m, d in runs_A2.items()}
display(pd.DataFrame(rows_A2.values(), columns=MAIN_COLS).drop(columns='secs'))
'''),

('code', A2, r'''
%%section A2
# step 5 -- the text numbers (Redraw only: they read the report's runs)
lines_A2 = {'poster_segments': [], 'inloop_identical': []}
if p['mode'] == 'Redraw':
    # text_numbers.poster_segments: fastest-segment gap, error relative to speed
    for seg in BETA_SEGMENTS:
        dc = run_dir(seg, 'thesis_cmax', 150, d_fr=0.1, d_anchor=0.5)   # BETA_CONFIGS series
        di = run_dir(seg, 'thesis_imu', 150) if seg[0] == 'poster_rotation' else None
        if dc is None:
            continue
        c, i = tracking(dc), tracking(di) if di else None
        w = mag(c['gt']).mean()
        floor = mag(c['imu'] - c['gt']).mean() * DEG
        line = (f'{seg[0][:6]} {seg[1]}: |w| {w*DEG:.1f} deg/s; CMax-anchor {c["err"].mean():5.2f} deg/s '
                f'= {100*c["err"].mean()/(w*DEG):4.1f}% of |w|; gyro floor {floor:4.2f} '
                f'= {100*floor/(w*DEG):4.1f}%')
        if i is not None:
            line += f'; thesis+IMU {i["err"].mean():5.2f}, gap {c["err"].mean()-i["err"].mean():4.2f}'
        lines_A2['poster_segments'].append(line)
    # text_numbers.inloop_identical, 3 s half: CMax-inloop omega at two delta_FR
    for seg in [b for b in BETA_SEGMENTS if b in [s[:3] for s in SEGMENTS]]:
        da = run_dir(seg, 'thesis_cmax_v2', 150, d_fr=0.5, d_anchor=None)
        db = run_dir(seg, 'thesis_cmax_v2', 150)
        if da and db:
            lines_A2['inloop_identical'].append(
                f'{seg[0][:6]} {seg[1]}: dFR 0.5 vs {ADOPTED["delta_FR"]} (3 s): '
                f'max |dw| = {np.abs(tracking(da)["est"]-tracking(db)["est"]).max()*DEG:.2e} deg/s')
    for k, ls in lines_A2.items():
        print(f'== {k}'); print('\n'.join(ls) or '(no runs)')
else:
    print('Live: the text numbers read the report runs; switch to Redraw to see them.')
'''),

('code', A2, r'''
%%section A2
# step 6 -- text_numbers.track_length: the 3 s runs cut after N frames (Redraw only).
# The network is causal, so their first 75 frames are the 1.5 s runs of A3.
L = lines_A2['track_length'] = []
if p['mode'] == 'Redraw':
    Ns = (25, 50, 75, 100, 150)
    FIVE = ('cook', 'thesis', 'thesis_imu', 'thesis_cmax', 'thesis_cmax_v2')
    same = []
    for model in FIVE:
        L.append(model)
        for s in SEGMENTS:
            long_, short = run_tn(s, model, nf=150), run_tn(s, model)
            if long_ is None:
                continue
            if short is not None:
                same.append(np.array_equal(long_['err'][:75], short['err'])
                            and np.array_equal(long_['r'][:75], short['r']))
            L.append(f'  {s[3]:8s} ' + '  '.join(
                f'{n * 0.02:.1f}s {long_["err"][:n].mean():6.1f} {long_["r"][:n].mean():.2f}' for n in Ns))
    L.append(f'first 75 frames of the 3 s runs identical to the 1.5 s runs: {sum(same)} of {len(same)}')
    T_ = {(m, s[3]): run_tn(s, m, nf=150) for m in FIVE for s in SEGMENTS}
    if all(v is not None for v in T_.values()):
        Em = lambda m, lab, n: T_[m, lab]['err'][:n].mean()
        Rr = lambda m, lab, n: T_[m, lab]['r'][:n].mean()
        real, labs, anch = [s[3] for s in REAL], [s[3] for s in SEGMENTS], FIVE[2:]
        L.append('anchored (thesis_imu, thesis_cmax), largest change of the error from 1.5 to 3 s: '
                 f'{max(abs(Em(m, l, 150) - Em(m, l, 75)) for m in anch[:2] for l in labs):.1f} deg/s')
        order = lambda l, n: sorted(anch, key=lambda m: Em(m, l, n))
        L.append('order of the three anchored models the same at 0.5, 1.5 and 3 s on every sequence: '
                 + str(all(order(s[3], n) == order(s[3], 75) for s in SEGMENTS for n in (25, 150))))
        for n in (25, 75, 150):
            v = [Em(m, l, n) for m in FIVE[:2] for l in real]
            L.append(f'unanchored on the real sequences over {n * 0.02:g} s: {min(v):.1f}-{max(v):.1f} deg/s')
        for n in (25, 75):
            v = [T_[m, l]['dir'][:n].mean() for m in FIVE[:2] for l in real]
            L.append(f'unanchored direction error on the real sequences over {n * 0.02:g} s: {min(v):.0f}-{max(v):.0f} deg')
        L.append('unanchored error over 1.5 s as a share of the 3 s value, real sequences: at least '
                 f'{min(Em(m, l, 75) / Em(m, l, 150) for m in FIVE[:2] for l in real):.0%}')
        L.append('anchored read-out lowest over 0.5 s on every real sequence: '
                 + str(all(Rr(m, l, 25) < min(Rr(m, l, 75), Rr(m, l, 150)) for m in anch for l in real))
                 + f'; largest change from 1.5 to 3 s {max(abs(Rr(m, l, 150) - Rr(m, l, 75)) for m in anch for l in labs):.2f}')
        L.append('best read-out over 3 s per sequence: ' + ', '.join(
            f'{s[3]} {max(T_, key=lambda k: Rr(*k, 150) if k[1] == s[3] else -1)[0]} '
            f'{max(Rr(m, s[3], 150) for m in FIVE):.2f}' for s in SEGMENTS))
        wc = os.path.join('report', 'experiments', 'window_choice.csv')
        if os.path.exists(wc):
            with open(wc) as f:
                wrows = list(csv.DictReader(f))
            L.append('sign changes of the dominant axis of the reference (run_experiments window): ' + '; '.join(
                f'{lab} ' + ', '.join(f'{r["duration_s"]} s {r["sign_flips"]}' for r in wrows
                                      if r['sequence'] == lab and float(r['duration_s']) in (0.5, 1.5, 3.0))
                for lab in real))
    print('== track_length')
    for line in L:
        print(line)
'''),

('code', A2, r'''
%%section A2
# same-code check -- against cmd_main (Live, Redraw) and text_numbers (Redraw)
import io
if p['mode'] == 'Live':
    check_live_main(p, 'A2', nf_A2, runs_A2, rows_A2)
else:
    # the rows against report/experiments/main_base_<seq>_<model>.csv
    for m, row in rows_A2.items():
        path = os.path.join('report', 'experiments', f'main_base_{p["segment"]}_{m}.csv')
        if need([path], [f'main_{p["segment"]}_{m}']):
            with open(path) as f:
                rep = list(csv.reader(f))[1]
            if not check(f'main_base row {m}', dict(zip(MAIN_COLS[3:12], row[3:12])),
                         dict(zip(MAIN_COLS[3:12], map(float, rep[3:12])))):
                stale_note(runs_A2[m])
    # the text numbers against text_numbers' own functions
    for name in lines_A2:
        with redirect(text_numbers, OUT=[]), contextlib.redirect_stdout(io.StringIO()):
            getattr(text_numbers, name)()
            theirs = [l for l in text_numbers.OUT if not l.startswith(('==', '   (missing'))
                      and l and (name != 'inloop_identical' or '(3 s)' in l)]
        check(f'text_numbers.{name}', nums(lines_A2[name]), nums(theirs))
'''),
]
