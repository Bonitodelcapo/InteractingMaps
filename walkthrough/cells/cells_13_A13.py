A13 = ('A13',)

CELLS = [
('md', A13, r'''
## A13 · Maps on poster (every model stopped at frame 44)

**Question.** What do the maps (V, in-loop I, read-out I, G, F) look like at the
frame each sequence is shown at? poster's last Table III frame falls in the
anchored excursion, so it is shown at frame 43 (counted from 0), the frame of
highest |r|, from runs stopped there: 44 frames = 0.88 s, every model.

**Made by** `report/run_experiments.py::cmd_sweep`
(`--what sweep --name maps_poster --mode coord --duration 0.88 --sequences poster
--vary n_iters=75`), `run_all.sh` jobs `maps_poster_<model>` → runs (with
`maps_final.npz`) and `sweep_<model>.csv` in `experiments/maps_poster/`. The
other four rows of the figures are the last frames of the Table III runs (A3).

**Read by** `report/make_figures.py::compute_maps` → `::fig_maps` (fig:maps,
CMax-anchor) and `::fig_appendix` (fig:appendix_cook, _thesis, _imu,
_cmax_v2), and `report/text_numbers.py::polarity`. `compute_maps` runs no
network: it reads each run's saved `maps_final.npz` and bins the raw events of
the shown frame (V), so it is unrolled and also called here; the binning is
done once per row and shared by the models (the same call, same arguments).
'''),

('code', A13, r'''
# step 1 -- the runs: the saved maps_poster rows and run folders (Redraw), or the panel's model (Live)
p = params()
rows_A13 = run_sweep(p, 'A13', [f'maps_poster_{m}' for m in MODELS])
POSTER_A13 = next(s for s in SEGMENTS if s[3] == 'poster')
if p['mode'] == 'Redraw':
    runs_A13 = {m: run_dir(POSTER_A13, m, 44, root=os.path.join('experiments', 'maps_poster'))
                for m in MODELS}
    for m, d in runs_A13.items():
        print(f'{m:16s} {d}')
    if READY['A13'] and runs_A13['thesis_cmax'] is None:
        READY['A13'] = need([os.path.join('experiments', 'maps_poster', 'poster_rotation', 'thesis_cmax')],
                            ['maps_poster_thesis_cmax'])
else:
    runs_A13 = {LIVE_JOB['A13'].split('maps_poster_')[1]: d for d in LIVE_RUNS.get('A13', {}).values()}
'''),

('code', A13, r'''
%%section A13
# step 2 -- make_figures.py::compute_maps, unrolled: (V, I, read-out, G, F) per MAP_ROWS row
from interacting_maps.data_loader import EventFrameSequence
from interacting_maps.network_dissertation import solve_poisson_exact
from make_figures import MAP_ROWS, APPENDIX_MODELS, TUNED, _MAP_COLUMNS, grid_size
V_A13 = {}                                       # (events, t0, n) -> last binned frame

def last_V_A13(ds, t0, n):
    """compute_maps' V: the last of n event frames from t0 (cached: same for every model)."""
    ev = E.get_dataset_paths(ds)['events']
    if (ev, t0, n) not in V_A13:
        pp = E.get_dataset_paths(ds)
        sq = EventFrameSequence(pp['events'], pp['calib'], frame_duration=0.02, t_start=t0,
                                n_frames=n, clip_value=10.0, undistort=False, sensor_size=(180, 240))
        V_A13[ev, t0, n] = list(sq)[-1]
    return V_A13[ev, t0, n][0]

def compute_maps_A13(model, rows):
    d_anchor = TUNED['delta_IMU'] if model in ('thesis_imu', 'thesis_cmax') else None
    maps = []
    for i, _lab, n, root in rows:
        ds, sid, t0, label = SEGMENTS[i]
        d = make_figures.find_run(ds, sid, t0, model, TUNED['delta_FR'], d_anchor, n_frames=n, root=root)
        if d is None or not os.path.exists(os.path.join(d, 'maps_final.npz')):
            print(f'    (missing {ds}/{model})')
            maps.append(None)
            continue
        if not need([os.path.join('data', ds)]):  # the events of V (the report would stop here)
            maps.append(None)
            continue
        m = np.load(os.path.join(d, 'maps_final.npz'))
        maps.append((last_V_A13(ds, t0, n), m['I'], solve_poisson_exact(m['G'], 'fft'), m['G'], m['F']))
        print(f'    {label}/{model} ({os.path.basename(d)})')
    return maps

if p['mode'] == 'Redraw':
    rows_maps_A13 = MAP_ROWS
    models_A13 = ['thesis_cmax'] + APPENDIX_MODELS
else:                                            # Live: the poster row of the run just made
    nf = int(round(sweep_duration(p, SWEEP_JOBS[LIVE_JOB['A13']]) / 0.02))
    rows_maps_A13 = [(1, f'poster\n(Live, frame {nf - 1})', nf, walk_dir('A13'))]
    models_A13 = list(runs_A13)
maps_A13 = {m: compute_maps_A13(m, rows_maps_A13) for m in models_A13}
for m, mm in maps_A13.items():
    print(f'{m:16s} ' + ', '.join(f'{lab.splitlines()[0]} {"(none)" if x is None else x[0].shape}'
                                  for (_, lab, _, _), x in zip(rows_maps_A13, mm)))
'''),

('md', A13, r'''
### fig:maps, fig:appendix -- `make_figures.py::fig_maps`, `::fig_appendix` (`_draw_maps`)

One row per sequence, columns V, I in-loop, I read-out, G, F. fig_maps is
CMax-anchor; the appendix repeats the frames for the other four models.
'''),

('code', A13, r'''
%%section A13
# step 3 -- _draw_maps, unrolled, for fig_maps (CMax-anchor) and the four appendix figures
from interacting_maps.viz import draw_map
def draw_maps_A13(maps, labels, name):
    labels = [lab for m, lab in zip(maps, labels) if m is not None]
    maps = [m for m in maps if m is not None]
    if not maps:
        print(f'  {name}: no runs found, skipped')
        return None
    with use_style('paper'):
        fig, axes = plt.subplots(len(maps), len(_MAP_COLUMNS),
                                 figsize=grid_size(len(maps), len(_MAP_COLUMNS)))
        axes = np.atleast_2d(axes)
        for r, (arrs, lab) in enumerate(zip(maps, labels)):
            for c, ((kind, ttl), arr) in enumerate(zip(_MAP_COLUMNS, arrs)):
                draw_map(axes[r, c], kind, arr, title=ttl if r == 0 else '')
            axes[r, 0].set_ylabel(lab)
        imgs = fig_images([fig])
        print(name)
        style_save(fig, os.path.join(walk_dir('A13'), f'{name}.png'), show=True)
    return imgs

imgs_A13 = {}
for m in maps_A13:
    name = 'fig_maps' if m == 'thesis_cmax' else f'fig_appendix_maps_{m}'
    imgs_A13[name] = draw_maps_A13(maps_A13[m], [r[1] for r in rows_maps_A13], name)
'''),

('code', A13, r'''
%%section A13
# step 4 -- text_numbers.polarity, unrolled (Redraw): sign of the read-out, unanchored models
def text_polarity_A13():
    from report_runs import aps_frame, SYNTHETIC
    from interacting_maps.metrics import crop_interior
    L = []
    for s in [s for s in SEGMENTS if s[3] not in SYNTHETIC]:      # text_numbers.REAL
        ds, sid, t0, lab = s
        root, nf = ((os.path.join('experiments', 'maps_poster'), 44) if lab == 'poster'
                    else ('results', 75))
        for m in ('cook', 'thesis'):
            d = run_dir(s, m, nf, root=root)
            if d is None or not os.path.exists(os.path.join(d, 'maps_final.npz')):
                continue                         # text_numbers says '(missing maps ...)'
            try:
                aps = np.log(aps_frame(ds, t0 + (nf - 0.5) * 0.02).astype(np.float64) + 1.0)
            except Exception as e:               # no APS images in this checkout
                L.append(f'   (APS frame not available: {e})')
                return L
            img = solve_poisson_exact(np.load(os.path.join(d, 'maps_final.npz'))['G'], 'fft')
            a, b = crop_interior(img).ravel(), crop_interior(aps).ravel()
            a, b = a - a.mean(), b - b.mean()
            r = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
            with open(os.path.join(d, 'tracking.csv')) as f:
                last = list(csv.DictReader(f))[-1]
            e = np.array([float(last[f'est_w{k}']) for k in 'xyz'])
            g = np.array([float(last[f'gt_w{k}']) for k in 'xyz'])
            cos = float(e @ g / (np.linalg.norm(e) * np.linalg.norm(g)))
            L.append(f'{lab:8s} {m:7s} frame {int(last["frame"])}: signed r of the read-out {r:+.2f}, '
                     f'cosine between estimated and true omega {cos:+.2f}')
    return L

lines_A13 = []
if p['mode'] == 'Redraw':
    lines_A13 = text_polarity_A13()
    print('== polarity'); print('\n'.join(lines_A13))
else:
    print('Live: the text numbers read the report runs; switch to Redraw to see them.')
'''),

('code', A13, r'''
%%section A13
# same-code check -- cmd_sweep (Live); stop at 44, compute_maps, fig_maps, fig_appendix, text (Redraw)
import interacting_maps.data_loader as data_loader_A13
def cached_frames_A13(events, calib, frame_duration=None, t_start=None, n_frames=None, **kw):
    """The binning step 2 already did, for compute_maps' identical call; any other call is real."""
    same = (frame_duration == 0.02 and kw == dict(clip_value=10.0, undistort=False, sensor_size=(180, 240)))
    if same and (events, t_start, n_frames) in V_A13:
        return [V_A13[events, t_start, n_frames]]
    return EventFrameSequence(events, calib, frame_duration=frame_duration, t_start=t_start,
                              n_frames=n_frames, **kw)

if p['mode'] == 'Live':
    check_live_sweep(p, 'A13', LIVE_JOB['A13'], rows_A13)
else:
    for m, d in runs_A13.items():       # deterministic: frame 43 is the Table III run's own frame 43
        d75 = run_dir(POSTER_A13, m, 75)
        if d and d75:
            check(f'maps_poster {m} = first 44 frames of its Table III run',
                  tracking(d)['err'], tracking(d75)['err'][:44])
    have = [r for r in MAP_ROWS if os.path.exists(os.path.join('data', SEGMENTS[r[0]][0]))]
    if len(have) < len(MAP_ROWS):       # the report functions run on the rows whose events are here
        need([os.path.join('data', SEGMENTS[r[0]][0]) for r in MAP_ROWS], ['(raw data: download_gdrive.sh)'])
    keep_rows = [k for k, r in enumerate(MAP_ROWS) if r in have]
    with redirect(data_loader_A13, EventFrameSequence=cached_frames_A13), \
         redirect(make_figures, MAP_ROWS=have):
        for m in maps_A13:
            with contextlib.redirect_stdout(io.StringIO()):
                theirs = make_figures.compute_maps(E, m)
            mine = [maps_A13[m][k] for k in keep_rows]
            flat = lambda ms: (np.concatenate([np.asarray(a, float).ravel() for x in ms if x is not None
                                               for a in x]) if any(x is not None for x in ms)
                               else np.array([]))
            check(f'compute_maps {m}', flat(mine), flat(theirs))
        for name, fn in (('fig_maps', make_figures.fig_maps), ('fig_appendix', make_figures.fig_appendix)):
            figs = report_figs(fn, 'A13')
            mine = [imgs_A13.get(n) for n in imgs_A13 if (n == 'fig_maps') == (name == 'fig_maps')]
            mine = np.concatenate([x for x in mine if x is not None]) if any(
                x is not None for x in mine) else np.array([])
            check(f'{name} images', mine, fig_images(figs))
    with redirect(text_numbers, OUT=[]), contextlib.redirect_stdout(io.StringIO()):
        text_numbers.polarity()
        theirs = [l for l in text_numbers.OUT if l and not l.startswith(('==', '   (missing'))]
    check('text_numbers.polarity', nums(lines_A13), nums(theirs))
'''),
]
