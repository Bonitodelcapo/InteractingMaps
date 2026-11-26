A1 = ('A1',)

CELLS = [
('md', A1, r'''
## A1 · Window choice

**Question.** How constant is ω over a segment of 0.5–3 s, and how far does the
scene move inside one accumulation window? This chose the 20 ms window and the
segment lengths before any model was run. Ground truth only: no network runs,
so both modes compute everything here (Redraw at the report's 20 ms, Live at
the panel's `dt`).

**Made by** `report/run_experiments.py::cmd_window` (`--what window`),
`run_all.sh` job `window` → `report/experiments/window_choice.csv` (one row per
sequence and duration; its `omega_mean` column stays in the code's unit, rad/s,
as stored data; every speed shown here is deg/s).

**Read by** `report/text_numbers.py`, quoted in the report text (no figure or
table): the line *image travel per 20 ms window* is in `::intensity`, the line
*sign changes of the dominant axis* is in `::track_length`. Both read the csv.
Also unrolled here, because it is the same kind of ground-truth-only question:
`::segment_speeds` (is each reported segment the one of median gyro speed?).

**Steps**: reference ω per frame → per duration |ω|, relative std, sign flips
(the csv rows) → scene motion inside one window → segment speeds → the text
lines → same-code check.
'''),

('code', A1, r'''
# step 1 -- the inputs: ground truth of every Table II sequence (no runs needed)
p = params()
dt_A1 = 0.02 if p['mode'] == 'Redraw' else p['dt']     # cmd_window's window is 20 ms
if p['mode'] == 'Redraw':
    print("Redraw: the report's 20 ms window (cmd_window fixes it); nothing here needs a saved run.")
    if p['dt'] != 0.02:
        print(f"The panel's dt {p['dt']} is used in Live mode.")
else:
    print(f'Live: window dt {dt_A1*1000:g} ms from the panel')
DURS_A1 = (0.5, 1.0, 1.5, 2.0, 3.0)                     # cmd_window's durations
segs_A1 = [s for s in SEGMENTS if need([os.path.join('data', s[0])])]
if len(segs_A1) < len(SEGMENTS):
    print('   (datasets: download_gdrive.sh; every step below uses the sequences with data)')
READY['A1'] = bool(segs_A1)
print('sequences with data:', [s[3] for s in segs_A1])
'''),

('code', A1, r'''
%%section A1
# step 2 -- the reference omega per frame, run_experiments.py::cmd_window
# cmd_window recomputes it for each duration from the same t0; the first n frames
# of the 3 s series are the same numbers, so one series per sequence is sliced.
def reference_series_A1(seg, dt, n):
    """cmd_window: get_reference_omega over frames [t0+k*dt, t0+(k+1)*dt), rad/s."""
    ds, _sid, t0, _lab = seg
    pp = E.get_dataset_paths(ds)
    gt, imu = E.load_groundtruth(pp['groundtruth']), E.load_imu(pp['imu'])
    om = E.load_omega_gt(pp.get('omega_gt'))
    return np.array([E.get_reference_omega(gt, imu, t0+k*dt, t0+(k+1)*dt, om)[0]
                     for k in range(n)])

w_A1 = {s[3]: reference_series_A1(s, dt_A1, int(round(max(DURS_A1) / dt_A1))) for s in segs_A1}
for lab, w in w_A1.items():
    print(f'{lab:8s} {w.shape[0]} frames of {dt_A1*1000:g} ms; first 3 frames, deg/s:',
          np.round(to_deg_s(w[:3]), 1).tolist())
'''),

('code', A1, r'''
%%section A1
# step 3 -- per duration: mean |omega|, relative std, sign flips of the dominant axis
def window_rows_A1(lab, w, dt):
    """cmd_window's rows for one sequence: (lab, dur, |w| mean, std/mean, flips)."""
    rows = []
    for dur in DURS_A1:
        n = int(round(dur / dt))
        wn = w[:n]
        m = np.linalg.norm(wn, axis=1)
        ax = int(np.argmax(np.abs(wn).mean(0)))          # dominant rotation axis
        flips = int((np.diff(np.sign(wn[:, ax])) != 0).sum())
        rows.append((lab, dur, m.mean(), m.std() / m.mean(), flips))
    return rows

rows_A1 = [r for lab, w in w_A1.items() for r in window_rows_A1(lab, w, dt_A1)]
display(pd.DataFrame([(a, b, to_deg_s(c), d, e) for a, b, c, d, e in rows_A1],
                     columns=['sequence', 'duration_s', '|w| deg/s', 'rel_std', 'sign_flips']
                     ).round({'|w| deg/s': 1, 'rel_std': 3}))
'''),

('code', A1, r'''
%%section A1
# step 4 -- the scene motion inside one window (cmd_window's second table)
# rotation in one window = |w| * window; image travel at the centre pixel = f * angle,
# with f = 199 px (cmd_window's constant: the DAVIS240 focal length of calib.txt)
F_PX_A1 = 199.0
ms_A1 = [10, 20, 30, 50] + ([] if dt_A1*1000 in (10, 20, 30, 50) else [round(dt_A1*1000, 3)])
motion_A1 = {lab: {f'{ms:g} ms': f'{np.degrees(wm*ms/1000):4.2f} deg / {F_PX_A1*wm*ms/1000:4.1f} px'
                   for ms in ms_A1}
             for lab, dur, wm, _rs, _fl in rows_A1 if dur == 1.0}
display(pd.DataFrame(motion_A1).T)
'''),

('code', A1, r'''
%%section A1
# step 5 -- text_numbers.py::segment_speeds: is each reported segment the median-speed one?
# gyro |w| over 3 s (150 frames of 20 ms) of every candidate segment of config.DATASET_SEGMENTS
from config import DATASET_SEGMENTS, get_dataset_paths as _paths_A1
from interacting_maps import eval_io as _io_A1
from report_runs import SYNTHETIC as _SYN_A1

def segment_speed_lines_A1(segs, dt):
    lines = []
    real = [s for s in segs if s[3] not in _SYN_A1]
    n = int(round(3.0 / dt))
    for ds in dict.fromkeys(s[0] for s in real):
        imu = _io_A1.load_imu(_paths_A1(ds)['imu'])
        chosen = [s[1] for s in real if s[0] == ds]
        speeds = {}
        for c in DATASET_SEGMENTS[ds]:
            w = np.array([_io_A1.get_gyro_for_frame(imu, c['t_start'] + k*dt, c['t_start'] + (k+1)*dt)
                          for k in range(n)])
            m = to_deg_s(np.linalg.norm(w, axis=1))
            speeds[c['id']] = m.mean()
            lines.append(f"{ds[:14]:14s} {c['id']}{'*' if c['id'] in chosen else ' '} "
                         f"mean |w| {m.mean():6.1f} deg/s, std/mean {m.std() / m.mean():.2f}, "
                         f"range {m.min():5.1f}-{m.max():5.1f}")
        order = sorted(speeds, key=speeds.get)         # the median of the sorted speeds
        k = len(order)
        lines.append(f"   median-speed segment(s): {', '.join(order[(k - 1) // 2: k // 2 + 1])}; "
                     f"reported: {', '.join(chosen)}")
    return lines

speed_lines_A1 = segment_speed_lines_A1(segs_A1, dt_A1)
print(f'== segment speeds (* = reported), {dt_A1*1000:g} ms frames over 3 s')
print('\n'.join(speed_lines_A1) or '(no real sequence with data)')
'''),

('code', A1, r'''
%%section A1
# step 6 -- the window lines of the text: text_numbers.py::intensity (image travel)
# and ::track_length (sign changes). Both read window_choice.csv: Redraw reads the
# report's csv as they do; Live uses this section's rows at the panel's dt.
from report_runs import SYNTHETIC as _SYN_A1
WC_A1 = os.path.join('report', 'experiments', 'window_choice.csv')

def window_text_A1(rows, dt):
    """rows: dicts with the csv's columns. Returns the two lines, as text_numbers says them."""
    real = [s[3] for s in SEGMENTS if s[3] not in _SYN_A1]
    syn = [s[3] for s in SEGMENTS if s[3] in _SYN_A1]
    travel, flips = [], []
    for name, labs in (('real', real), ('synthetic', syn)):       # intensity
        w = [float(r['omega_mean']) for r in rows
             if r['sequence'] in labs and float(r['duration_s']) == 1.0]
        if w:
            travel.append(f'image travel per {dt*1000:g} ms window, centre pixel ({name}): '
                          f'{199.0*np.mean(w)*dt:.1f} px')
    flips.append('sign changes of the dominant axis of the reference (run_experiments window): '
                 + '; '.join(f'{lab} ' + ', '.join(f'{r["duration_s"]} s {r["sign_flips"]}'
                                                  for r in rows if r['sequence'] == lab
                                                  and float(r['duration_s']) in (0.5, 1.5, 3.0))
                             for lab in real))                      # track_length
    return travel, flips

def as_csv_rows_A1(rows):
    """this section's rows as the csv stores them (same rounding, strings)"""
    return [dict(sequence=a, duration_s=str(b), omega_mean=str(round(c, 4)),
                 rel_std=str(round(d, 4)), sign_flips=str(e)) for a, b, c, d, e in rows]

if p['mode'] == 'Redraw':
    src_A1 = list(csv.DictReader(open(WC_A1))) if need([WC_A1], ['window']) else []
else:
    src_A1 = as_csv_rows_A1(rows_A1)
travel_A1, flips_A1 = window_text_A1(src_A1, dt_A1)
print('== text_numbers.intensity (window line)'); print('\n'.join(travel_A1) or '(no rows)')
print('== text_numbers.track_length (window line)'); print('\n'.join(flips_A1))
'''),

('code', A1, r'''
%%section A1
# same-code check -- cmd_window itself (OUTDIR -> walk_dir('A1')/check, only the
# sequences with data), the report's csv, and the text_numbers functions.
# The report's code is fixed at 20 ms, so a Live run at another dt is recomputed
# at 20 ms with the unrolled functions above before comparing.
import io
if dt_A1 == 0.02:
    rows20_A1, speed20_A1 = rows_A1, speed_lines_A1
else:
    print(f'(Live dt {dt_A1*1000:g} ms: comparing a 20 ms recompute of the unrolled steps)')
    rows20_A1 = [r for s in segs_A1
                 for r in window_rows_A1(s[3], reference_series_A1(s, 0.02, 150), 0.02)]
    speed20_A1 = segment_speed_lines_A1(segs_A1, 0.02)
mine_A1 = as_csv_rows_A1(rows20_A1)
num_A1 = lambda rows: [[float(r[k]) for k in ('duration_s', 'omega_mean', 'rel_std', 'sign_flips')]
                       for r in rows]

out_A1 = os.path.join(walk_dir('A1'), 'check')
os.makedirs(out_A1, exist_ok=True)
with redirect(run_experiments, OUTDIR=out_A1, SEGMENTS=segs_A1), \
     contextlib.redirect_stdout(io.StringIO()):
    run_experiments.cmd_window(E, None)
theirs_A1 = list(csv.DictReader(open(os.path.join(out_A1, 'window_choice.csv'))))
check('cmd_window rows', num_A1(mine_A1), num_A1(theirs_A1))

if need([WC_A1], ['window']):                 # the report's csv, sequences with data
    rep_A1 = [r for r in csv.DictReader(open(WC_A1)) if r['sequence'] in {s[3] for s in segs_A1}]
    check('window_choice.csv rows', num_A1(mine_A1), num_A1(rep_A1))
    rep_all_A1 = list(csv.DictReader(open(WC_A1)))
    travel_rep_A1, flips_rep_A1 = window_text_A1(rep_all_A1, 0.02)
else:
    travel_rep_A1, flips_rep_A1 = [], []

# text_numbers: segment_speeds on the sequences with data (REAL redirected)
real_A1 = [s for s in segs_A1 if s[3] not in _SYN_A1]
with redirect(text_numbers, OUT=[], REAL=real_A1), contextlib.redirect_stdout(io.StringIO()):
    text_numbers.segment_speeds()
    ss_A1 = [l for l in text_numbers.OUT if l and not l.startswith('==')]
check('text_numbers.segment_speeds', nums(speed20_A1), nums(ss_A1))

# the window lines: their functions read the report's csv, so the 20 ms lines of that csv
for name, prefix, mine in (('intensity', 'image travel', travel_rep_A1),
                           ('track_length', 'sign changes', flips_rep_A1)):
    with redirect(text_numbers, OUT=[]), contextlib.redirect_stdout(io.StringIO()):
        getattr(text_numbers, name)()
        theirs = [l for l in text_numbers.OUT if l.startswith(prefix)]
    if theirs:
        check(f'text_numbers.{name} window line', nums(mine), nums(theirs))
    else:
        print(f'missing: the window line of text_numbers.{name} -- '
              + ('it is printed only when every 3 s run of every model is on disk '
                 '(run_all.sh jobs main_<seq>_<model>)' if name == 'track_length'
                 else 'it needs report/experiments/window_choice.csv (job window)'))
'''),
]
