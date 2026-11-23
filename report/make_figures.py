"""
make_figures.py — regenerate every figure in the report from committed code.

Run from the repository root:
    python report/make_figures.py              # all figures
    python report/make_figures.py --only maps  # one of: beta_speed tracking
                                               #   maps appendix distortion cmax_baseline
                                               #   converge recon dissociation
                                               #   regularisers image_settings sensitivity

Outputs PDFs (used by the LaTeX) and PNGs (for quick inspection) into
report/figures/ via style.save.

Where the data comes from
-------------------------
beta_speed, tracking   read results/ directly (summary.json / tracking.csv), so
                       they need the corresponding runs to exist.
maps, appendix         read the final maps (maps_final.npz) of the Table III
                       runs in results/: CMax-anchor for maps, one figure per
                       other model for appendix.
distortion             needs only the raw events.
converge               the I snapshots of run_experiments --what converge.
recon                  the runs of run_experiments --what recon.
dissociation           the Table III runs in results/.
regularisers           the ablation runs (experiments/ablation, run_all.sh).
image_settings         the Table III and curl runs, and experiments/best_image.
sensitivity            the sweep and grid runs (experiments/sens, experiments/grid).

Configuration
-------------
Every figure uses the configuration reported in the paper (THESIS_PARAMS in
config.py), 75 iterations, 20 ms windows, distortion carried in the
C matrix (DISTORTION_MODE = 'C_full'). SEGMENTS below are the segments named in
Table II.
"""

import os
import sys
import csv
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import experiment_tracking  # merged run.json reader (after sys.path points at ROOT)
from interacting_maps.style import (MODELS, PAPER_WIDTH, PAPER_WIDE, NEUTRAL_COLOR, LABEL_BETA,
                                    frame_times, to_deg_s, use_style, save)
from interacting_maps.viz import draw_map, draw_omega, map_scale
from interacting_maps.metrics import abs_corr, crop_interior
from report_runs import (SEGMENTS, SYNTHETIC, BETA_SEGMENTS, run_cfg, recon_cfg,
                         converge_path, aps_frame, track_stats, image_search)
sys.argv = [sys.argv[0]] + sys.argv[1:]          # evaluation.py parses argv

FIG = os.path.join(ROOT, 'report', 'figures')
os.makedirs(FIG, exist_ok=True)

# Both match Table III: 75 frames of 20 ms = 1.5 s. The figures and the
# tables must describe the same runs, or a reader comparing them finds a
# disagreement that is only a difference of protocol.
DT, N_TRACK, N_MAPS = 0.02, 75, 75
# beta-vs-speed pools seven segments, of which only the Table II ones were run
# at 1.5 s; it uses the 3 s (150-frame) runs of --what beta throughout instead.
N_BETA = 150
from config import THESIS_PARAMS
TUNED = {k: THESIS_PARAMS[k] for k in ('delta_FR', 'delta_IMU')}   # adopted, config.py

# fig_maps rows (index into SEGMENTS, label, frame shown, run root). Every
# sequence at the last frame of its Table III run, except poster: its last
# frame falls in the anchored excursion, so it is shown at POSTER_FRAME, the
# frame of highest |r| for CMax-anchor and also for the mean over all five
# models, from runs stopped there (run_experiments --what sweep --name
# maps_poster --duration 0.88, i.e. 44 frames). The runs are deterministic, so
# that frame is the Table III run's own frame 43, counted from 0 as
# tracking.csv and the text count frames.
POSTER_FRAME = 44
MAP_ROWS = [(0, 'boxes\n(real)', N_MAPS, 'results'),
            (1, f'poster\n(real, frame {POSTER_FRAME - 1})', POSTER_FRAME,
             'experiments/maps_poster'),
            (2, 'dynamic\n(person in scene)', N_MAPS, 'results'),
            (3, 'bicycle\n(synthetic)', N_MAPS, 'results'),
            (4, 'street\n(synthetic)', N_MAPS, 'results')]
# the appendix repeats those frames for every model but the CMax-anchor one
APPENDIX_MODELS = ['cook', 'thesis', 'thesis_imu', 'thesis_cmax_v2']

# what the reconstruction-section figures show (the report's choices)
CONVERGE = [('dynamic', 45), ('bicycle', 40)]           # (sequence, frame)
RECON = dict(sequences=('boxes', 'poster'), duration=0.8, model='thesis_cmax',
             root=os.path.join('experiments', 'recon_figure'))
DISSOC = dict(sequences=('boxes', 'bicycle', 'street'), duration=1.5,
              models=('thesis', 'thesis_cmax', 'thesis_imu'))
REGULARISERS = dict(sequences=('boxes', 'poster', 'dynamic', 'bicycle'), duration=1.5,
                    model='thesis_cmax', root=os.path.join('experiments', 'ablation'))
IMAGE_DURATION = 1.5


# Every image panel is a sensor frame, 180 x 240 on all five sequences. With a
# fixed height per row the panels keep that aspect and leave blank space between
# the rows, so the image grids take their height from the panel width instead.
SENSOR_ASPECT = 180 / 240
TEXT_LINE = 0.15            # one line of 9 pt text, inches
LABEL_LINE = 0.19           # one line of x-label under a panel (mathtext is taller)


def grid_size(n_rows, n_cols, width=PAPER_WIDE, below=0, label_lines=2, title_lines=1):
    """Figure size of a grid of sensor images whose rows nearly touch:
    `title_lines` lines of column titles on top, `below` lines of x-label under
    every row, and `label_lines` lines of row label on the left."""
    panel_w = (width - label_lines * TEXT_LINE - 0.05) / n_cols - 0.03
    row_h = panel_w * SENSOR_ASPECT + below * LABEL_LINE + 0.03
    top = title_lines * TEXT_LINE
    return width, n_rows * row_h + top


def find_run(ds, sid, t0, model, d_fr, d_anchor=None, n_frames=N_TRACK, root='results'):
    """The run behind a figure (experiment_tracking.find_run), regularisers off as
    everywhere outside their own section, so e.g. the _curl0.6 ablation run is not
    mistaken for the reported one. d_anchor None: the model has no anchor."""
    params = {'delta_FR': d_fr, 'delta_shrinkI': 0.0, 'delta_map': 0.0, 'delta_curl': 0.0}
    if d_anchor is not None:
        params['delta_IMU'] = d_anchor
    return experiment_tracking.find_run(root, ds, model, sid, n_frames, params,
                                        dt=DT, t_start=t0)


def reference_omega(E, ds, t0, n=N_TRACK):
    p = E.get_dataset_paths(ds)
    gt = E.load_groundtruth(p['groundtruth'])
    imu = E.load_imu(p['imu'])
    om = E.load_omega_gt(p.get('omega_gt'))
    refs = [E.get_reference_omega(gt, imu, t0 + k*DT, t0 + (k+1)*DT, om) for k in range(n)]
    return np.array([w for w, _ in refs]), refs[0][1]


# ---------------------------------------------------------------- beta vs speed
def fig_beta_speed(E):
    # (model, delta_FR, delta_anchor, filled marker). Fixed on purpose: the
    # three series are the delta_FR test of the speed dependence, not the
    # adopted values (they are the BETA_CONFIGS runs of --what beta).
    series = {
        'default': ('thesis_cmax',    0.5, 0.3,  False),
        'tuned':   ('thesis_cmax',    0.1, 0.5,  True),
        'inloop':  ('thesis_cmax_v2', 0.5, None, True),
    }
    data = {k: ([], []) for k in series}
    for ds, sid, t0 in BETA_SEGMENTS:
        w = to_deg_s(np.linalg.norm(reference_omega(E, ds, t0, N_BETA)[0], axis=1).mean())
        for key, (model, d_fr, d_a, _) in series.items():
            d = find_run(ds, sid, t0, model, d_fr, d_a, n_frames=N_BETA)
            if d is None:
                print(f'    (missing {ds}/{sid}/{model} dFR={d_fr})')
                continue
            data[key][0].append(w)
            data[key][1].append(experiment_tracking.read_summary(d)['mean_beta'])

    with use_style('paper'):
        fig, ax = plt.subplots(figsize=(PAPER_WIDTH, 2.5))
        for key, (model, d_fr, _, filled) in series.items():
            x, y = np.array(data[key][0]), np.array(data[key][1])
            if len(x) < 2:
                continue
            name, col = MODELS[model]
            r = np.corrcoef(x, y)[0, 1]
            ax.scatter(x, y, s=26, zorder=3, color=col if filled else 'none', edgecolors=col,
                       linewidths=0.8, label=rf'{name}, $\delta_{{FR}}={d_fr}$  ($r={r:+.2f}$)')
            xs = np.linspace(x.min(), x.max(), 20)
            ax.plot(xs, np.polyval(np.polyfit(x, y, 1), xs), color=col, lw=1.0, alpha=0.55,
                    ls='-' if filled else '--', zorder=2)
        ax.axhline(1.0, color=NEUTRAL_COLOR, lw=0.8, ls=':', zorder=1)
        ax.set_xlabel(r'rotation speed $\|\omega\|$ (°/s)')
        ax.set_ylabel(LABEL_BETA)
        ax.legend(loc='upper left')
        save(fig, os.path.join(FIG, 'fig_beta_speed.png'))
    print('  wrote fig_beta_speed.pdf / .png')


# -------------------------------------------------------------------- tracking
def fig_tracking(E):
    ds, sid, t0 = 'poster_rotation', 'seg_C', 8.816
    ref, ref_src = reference_omega(E, ds, t0)
    t = frame_times(DT, N_TRACK)
    runs = [('thesis_cmax', find_run(ds, sid, t0, 'thesis_cmax', TUNED['delta_FR'],
                                     TUNED['delta_IMU'])),
            ('thesis', find_run(ds, sid, t0, 'thesis', TUNED['delta_FR']))]

    with use_style('paper'):
        fig, axes = plt.subplots(3, 1, figsize=(PAPER_WIDTH, 3.6), sharex=True)
        first = True
        for model, d in runs:
            if d is None:
                print(f'    (missing run for {MODELS[model][0]})')
                continue
            rows = list(csv.DictReader(open(os.path.join(d, 'tracking.csv'))))
            est = np.array([[float(r['est_wx']), float(r['est_wy']), float(r['est_wz'])]
                            for r in rows])[:N_TRACK]
            for i in range(3):
                draw_omega(axes[i], t[:len(est)], est[:, i], axis=i, model=model,
                           ref=ref[:len(est), i] if first else None, ref_src=ref_src)
            first = False
        for ax in axes:
            ax.label_outer()
        fig.legend(*axes[0].get_legend_handles_labels(), loc='outside upper center', ncol=3)
        save(fig, os.path.join(FIG, 'fig_tracking.png'))
    print('  wrote fig_tracking.pdf / .png')


# ------------------------------------------------------- CMax alone vs reference
def fig_cmax_baseline(E, labels=('boxes', 'bicycle')):
    """CMax and gyro on their own (no network) vs the reference, plus the CMax
    IWE at the first / middle / last frame. Reads run_experiments --what baseline."""
    with use_style('paper'):
        fig = plt.figure(figsize=(PAPER_WIDE, 5.2))
        for sub, lab in zip(fig.subfigures(1, len(labels)), labels):
            path = os.path.join(ROOT, 'report', 'experiments', f'baseline_dt{int(DT*1000)}ms_{lab}.npz')
            if not os.path.exists(path):
                print(f'    (missing {path})')
                continue
            d = np.load(path)
            t = frame_times(DT, len(d['ref']))
            gs = sub.add_gridspec(4, 3, height_ratios=(1, 1, 1, 1.3))
            for i in range(3):
                ax = sub.add_subplot(gs[i, :])
                draw_omega(ax, t, d['cmax'][:, i], ref=d['ref'][:, i], imu=d['gyro'][:, i],
                           axis=i, model='thesis_cmax', ref_src=str(d['ref_src']),
                           label='CMax alone')
                for k in d['frames']:
                    ax.axvline(t[k], color=NEUTRAL_COLOR, lw=0.6, ls='--')
                if i < 2:
                    ax.tick_params(labelbottom=False)
                    ax.set_xlabel('')
            for j, k in enumerate(d['frames']):
                draw_map(sub.add_subplot(gs[3, j]), 'IWE', d['iwe'][j],
                         title=f'IWE  t = {t[k]:.2f} s')
            sub.suptitle(lab)
            sub.legend(*sub.axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=3)
        save(fig, os.path.join(FIG, 'fig_cmax_baseline.png'))
    print('  wrote fig_cmax_baseline.pdf / .png')


# ------------------------------------------------------------------- map panels
def compute_maps(E, model):
    """Final maps of one model's runs at the MAP_ROWS frames (Table III runs,
    except poster: see MAP_ROWS).

    Read from each run's maps_final.npz rather than re-running the network: a
    re-run here once passed the events without the CMax anchor, so the figure
    showed an unanchored network under a caption naming the anchored one.
    A sequence without a completed run comes back as None.
    """
    from interacting_maps.data_loader import EventFrameSequence
    from interacting_maps.network_dissertation import solve_poisson_exact
    d_anchor = TUNED['delta_IMU'] if model in ('thesis_imu', 'thesis_cmax') else None
    maps = []
    for i, _lab, n, root in MAP_ROWS:
        ds, sid, t0, label = SEGMENTS[i]
        d = find_run(ds, sid, t0, model, TUNED['delta_FR'], d_anchor,
                     n_frames=n, root=root)
        if d is None or not os.path.exists(os.path.join(d, 'maps_final.npz')):
            print(f'    (missing {ds}/{model})')
            maps.append(None)
            continue
        p = E.get_dataset_paths(ds)
        sq = EventFrameSequence(p['events'], p['calib'], frame_duration=DT, t_start=t0,
                                n_frames=n, clip_value=10.0, undistort=False,
                                sensor_size=(180, 240))
        V = list(sq)[-1][0]
        m = np.load(os.path.join(d, 'maps_final.npz'))
        # The exact read-out of Eq. 6.64 as well: every reconstruction number
        # in the report refers to that, not to the in-loop map.
        maps.append((V, m['I'], solve_poisson_exact(m['G'], 'fft'), m['G'], m['F']))
        print(f'    {label}/{model} ({os.path.basename(d)})')
    return maps


_MAP_COLUMNS = [('V', r'events $V$ (input)'), ('I', r'intensity $I$ (in-loop)'),
                ('I', r'intensity $I$ (read-out)'), ('G', r'gradient $G$'),
                ('F', r'flow $F$')]


def _draw_maps(maps, labels, name):
    labels = [lab for m, lab in zip(maps, labels) if m is not None]
    maps = [m for m in maps if m is not None]
    if not maps:
        print(f'  {name}: no runs found, skipped')
        return
    with use_style('paper'):
        fig, axes = plt.subplots(len(maps), len(_MAP_COLUMNS),
                                 figsize=grid_size(len(maps), len(_MAP_COLUMNS)))
        axes = np.atleast_2d(axes)
        for r, (arrs, lab) in enumerate(zip(maps, labels)):
            for c, ((kind, ttl), arr) in enumerate(zip(_MAP_COLUMNS, arrs)):
                draw_map(axes[r, c], kind, arr, title=ttl if r == 0 else '')
            axes[r, 0].set_ylabel(lab)
        save(fig, os.path.join(FIG, f'{name}.png'))
    print(f'  wrote {name}.pdf / .png')


def fig_maps(E):
    _draw_maps(compute_maps(E, 'thesis_cmax'), [r[1] for r in MAP_ROWS], 'fig_maps')


def fig_appendix(E):
    """The frames of fig_maps for every other model, one figure each. Like
    every report figure it has no title of its own: the caption names the model."""
    for model in APPENDIX_MODELS:
        _draw_maps(compute_maps(E, model), [r[1] for r in MAP_ROWS],
                   f'fig_appendix_maps_{model}')


# ------------------------------------------------------------------ distortion
def distortion_numbers(calib, H, W):
    """Where undistortion sends the sensor pixels (numbers of Lens Distortion).

    occupied  distinct frame pixels that the H*W sensor pixels land on after
              undistortion, binned as events_to_vframe bins
    spacing   largest distance between the undistorted positions of two
              neighbouring sensor pixels that both land inside the frame
    Printed and written to report/experiments/distortion_numbers.csv.
    """
    import cv2
    K = np.array([[calib.fx, 0, calib.cx], [0, calib.fy, calib.cy], [0, 0, 1]])
    ys, xs = np.mgrid[0:H, 0:W]
    pts = np.stack([xs.ravel(), ys.ravel()], -1).astype(np.float64).reshape(-1, 1, 2)
    u = cv2.undistortPoints(pts, K, calib.dist.astype(np.float64), P=K)
    u = u[:, 0, :].reshape(H, W, 2)
    c = u.astype(np.int32)
    hit = (c[..., 0] >= 0) & (c[..., 0] < W) & (c[..., 1] >= 0) & (c[..., 1] < H)
    inside = (u[..., 0] >= 0) & (u[..., 0] < W) & (u[..., 1] >= 0) & (u[..., 1] < H)
    dx = np.linalg.norm(u[:, 1:] - u[:, :-1], axis=-1)[inside[:, 1:] & inside[:, :-1]]
    dy = np.linalg.norm(u[1:] - u[:-1], axis=-1)[inside[1:] & inside[:-1]]
    num = {'sensor_pixels': H * W,
           'occupied_undistorted': int(np.unique(c[hit][:, 1] * W + c[hit][:, 0]).size),
           'max_spacing_px': round(float(max(dx.max(), dy.max())), 3)}
    out = os.path.join(ROOT, 'report', 'experiments')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'distortion_numbers.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(num.keys())
        w.writerow(num.values())
    print('   ', ', '.join(f'{k} {v}' for k, v in num.items()))


def fig_distortion(E):
    from interacting_maps.data_loader import (CameraCalibration, load_events_fast,
                             undistort_events, events_to_vframe)
    ds, t0, dur, H, W = 'poster_rotation', 8.816, 0.20, 180, 240
    p = E.get_dataset_paths(ds)
    calib = CameraCalibration(p['calib'])
    ev = load_events_fast(p['events'], t_start=t0, duration=dur + 0.02)
    ev = ev[ev[:, 0] < t0 + dur]
    V_raw = events_to_vframe(ev, H, W, clip_value=40.0, normalise=True)
    V_und = events_to_vframe(undistort_events(ev, calib), H, W,
                             clip_value=40.0, normalise=True)
    distortion_numbers(calib, H, W)

    panels = [(V_raw, 'raw events\n' r'(lens in $\mathcal{C}$)'),
              (V_und, 'undistorted\nevents'),
              (V_und[10:80, 140:230], 'detail\n(undistorted)')]
    scale = map_scale('V', V_und)
    with use_style('paper'):
        # drawn at the column width the report prints it at, so its text
        # prints at the font size of the style
        fig, ax = plt.subplots(1, 3, figsize=(PAPER_WIDTH, 1.25))
        for a, (img, ttl) in zip(ax, panels):
            draw_map(a, 'V', img, title=ttl, scale=scale)
        save(fig, os.path.join(FIG, 'fig_distortion.png'))
    print('  wrote fig_distortion.pdf / .png')


# ----------------------------------------------- convergence within one frame
def fig_converge(E):
    """I across the message-passing iterations of one frame, every model
    (rows) at every checkpoint (columns); from run_experiments --what converge."""
    from report_runs import MODELS as ORDER
    for lab, frame in CONVERGE:
        path = converge_path(lab, frame)
        if not os.path.exists(path):
            print(f'    (missing {path})')
            continue
        z = np.load(path)
        checkpoints = [int(c) for c in z['checkpoints']]
        with use_style('paper'):
            fig, axes = plt.subplots(len(ORDER), len(checkpoints),
                                     figsize=grid_size(len(ORDER), len(checkpoints),
                                                       label_lines=1))
            for r, model in enumerate(ORDER):
                for c, it in enumerate(checkpoints):
                    draw_map(axes[r, c], 'I', z[f'{model}_{it}'],
                             title=f'iter {it}' if r == 0 else '')
                axes[r, 0].set_ylabel(MODELS[model][0])
            save(fig, os.path.join(FIG, f'fig_converge_{lab}.png'))
        print(f'  wrote fig_converge_{lab}.pdf / .png')


# --------------------------------------------- what the intensity map depends on
def fig_recon(E):
    """APS, the iterative update at dt = 20 and 10 ms, and the exact read-out at
    10 ms: halving the window cleans up G, and solving the Poisson equation
    exactly recovers the low frequencies the iterative update leaves out."""
    from interacting_maps.network_dissertation import solve_poisson_exact
    seqs = [s for s in SEGMENTS if s[3] in RECON['sequences']]
    dur = RECON['duration']
    with use_style('paper'):
        fig, axes = plt.subplots(len(seqs), 4, figsize=grid_size(len(seqs), 4, below=1,
                                                                  label_lines=1))
        axes = np.atleast_2d(axes)
        for row, (ds, sid, t0, lab) in enumerate(seqs):
            got = {dt: np.load(os.path.join(recon_cfg(ds, sid, t0, RECON['model'], dt, dur,
                                                      RECON['root']).output_dir,
                                            'maps_final.npz'))
                   for dt in (0.02, 0.01)}
            aps = aps_frame(ds, t0 + dur)
            ref = np.log(aps.astype(np.float64) + 1.0) if aps is not None else None
            panels = [
                ('APS', aps, 'aps'),
                (r'iterative, $\Delta t = 20$ ms', got[0.02]['I'], 'I'),
                (r'iterative, $\Delta t = 10$ ms', got[0.01]['I'], 'I'),
                (r'exact read-out, $\Delta t = 10$ ms',
                 solve_poisson_exact(got[0.01]['G'], 'fft'), 'I'),
            ]
            for col, (ttl, img, kind) in enumerate(panels):
                a = axes[row, col]
                if img is None:
                    a.set_axis_off()
                    continue
                draw_map(a, kind, img, title=ttl if row == 0 else '')
                if col == 0:
                    a.set_ylabel(lab)
                if col > 0 and ref is not None:
                    a.set_xlabel(f'$|r|={abs_corr(crop_interior(img), crop_interior(ref)):.2f}$')
        save(fig, os.path.join(FIG, 'fig_recon.png'))
    print('  wrote fig_recon.pdf / .png')


# --------------------------------------------- good omega is not a good picture
def fig_dissociation(E):
    """The read-out of the Table III runs, each labelled with its track-mean |r|
    and omega error: the unanchored network reconstructs better on the synthetic
    sequences while its omega is an order of magnitude worse."""
    from interacting_maps.network_dissertation import solve_poisson_exact
    nf = int(round(DISSOC['duration'] / DT))
    seqs = [s for s in SEGMENTS if s[3] in DISSOC['sequences']]
    models = DISSOC['models']
    with use_style('paper'):
        fig, axes = plt.subplots(len(seqs), len(models) + 1,
                                 figsize=grid_size(len(seqs), len(models) + 1, below=1,
                                                   label_lines=1))
        axes = np.atleast_2d(axes)
        for r_i, (ds, sid, t0, lab) in enumerate(seqs):
            panels = [('APS', aps_frame(ds, t0 + nf * DT), None, 'aps')]
            for m in models:
                d = run_cfg(ds, sid, t0, m, DT, nf).output_dir
                npz = os.path.join(d, 'maps_final.npz')
                img = st = None
                if os.path.exists(npz):
                    img = solve_poisson_exact(np.load(npz)['G'], 'fft')
                    # |r| from tracking.csv, since older summaries hold a signed mean
                    st = track_stats(d)
                else:
                    print(f'    missing: {npz}', flush=True)
                panels.append((MODELS[m][0], img, st, 'I'))
            for c_i, (title, img, st, kind) in enumerate(panels):
                a = axes[r_i, c_i]
                if img is None:
                    a.set_axis_off()
                    continue
                draw_map(a, kind, img, title=title if r_i == 0 else '')
                if c_i == 0:
                    a.set_ylabel(lab)
                if st is not None:
                    a.set_xlabel(f"$|r|={st[0]:.2f}$,  $\\omega$ err $={st[1]:.1f}$ °/s")
        save(fig, os.path.join(FIG, 'fig_dissociation.png'))
    print('  wrote fig_dissociation.pdf / .png')


# -------------------------------------------------- what each regulariser does
def fig_regularisers(E):
    """The read-out of each ablation run's final G (what the ablation table
    scores), one row per sequence, one column per regulariser."""
    from interacting_maps.network_dissertation import solve_poisson_exact
    seqs = [s for s in SEGMENTS if s[3] in REGULARISERS['sequences']]
    # (column label, the regulariser that ablation job set)
    cols = [('APS', None), ('none', {}),
            (r'$\delta_{\mathrm{shrink}I}{=}1$', {'delta_shrinkI': 1.0}),
            (r'$\delta_{\mathrm{map}}{=}0.03$', {'delta_map': 0.03}),
            (r'$\delta_{\mathrm{curl}}{=}0.6$', {'delta_curl': 0.6})]
    nf = int(round(REGULARISERS['duration'] / DT))
    with use_style('paper'):
        fig, axes = plt.subplots(len(seqs), len(cols),
                                 figsize=grid_size(len(seqs), len(cols), below=1, label_lines=1))
        axes = np.atleast_2d(axes)
        for row, (ds, sid, t0, lab) in enumerate(seqs):
            aps = aps_frame(ds, t0 + nf * DT)
            ref = np.log(aps.astype(np.float64) + 1.0) if aps is not None else None
            for col, (title, deltas) in enumerate(cols):
                a = axes[row, col]
                kind, img = 'aps', aps
                if col > 0:
                    kind, img = 'I', None
                    rc = run_cfg(ds, sid, t0, REGULARISERS['model'], DT, nf,
                                 out_root=REGULARISERS['root'], deltas=dict(deltas))
                    npz = os.path.join(rc.output_dir, 'maps_final.npz')
                    if os.path.exists(npz):
                        img = solve_poisson_exact(np.load(npz)['G'], 'fft')
                if img is None:
                    a.set_axis_off()
                    continue
                draw_map(a, kind, img, title=title if row == 0 else '')
                if col == 0:
                    a.set_ylabel(lab)
                elif ref is not None and ref.shape == img.shape:
                    a.set_xlabel(f'$|r|={abs_corr(crop_interior(img), crop_interior(ref)):.2f}$')
        save(fig, os.path.join(FIG, 'fig_regularisers.png'))
    print('  wrote fig_regularisers.pdf / .png')


# ------------------------------------------------------- settings for the image
def fig_image_settings(E):
    """Every sequence at the omega configuration next to the image configuration
    of its kind (real or synthetic); the search is report_runs.image_search."""
    from interacting_maps.network_dissertation import solve_poisson_exact
    score, best, run_dir = image_search(IMAGE_DURATION)
    print('    best:', best)
    omega_cfg = ('thesis_cmax', 0.02, 0.0)            # Table params
    with use_style('paper'):
        fig, axes = plt.subplots(len(SEGMENTS), 3,
                                 figsize=grid_size(len(SEGMENTS), 3, width=PAPER_WIDTH,
                                                   below=2, title_lines=2))
        for row, (ds, sid, t0, lab) in enumerate(SEGMENTS):
            kind = 'synthetic' if lab in SYNTHETIC else 'real'
            panels = [('APS', aps_frame(ds, t0 + IMAGE_DURATION), None, 'aps')]
            for ttl, cfg in (('configured\n' r'for $\omega$', omega_cfg),
                             ('configured\nfor the image', best[kind])):
                d = run_dir(ds, sid, t0, *cfg)
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
                    # model, window and curl weight of each column are in the caption
                    a.set_xlabel(f'$|r|={r:.2f}$\n$\\omega$ err ${err:.1f}$ °/s')
        save(fig, os.path.join(FIG, 'fig_image_settings.png'))
    print('  wrote fig_image_settings.pdf / .png')


# ----------------------------------------------------------------- sensitivity
# The one-at-a-time sweep (experiments/sens) and the 5x5 grid of delta_FR and
# delta_anchor (experiments/grid), both CMax-anchor over 1.5 s (run_all.sh).
# Each panel shows the mean over the three real and over the two synthetic
# sequences; Table sweep in the appendix gives every sequence.
SENS_REAL = ('boxes_rotation', 'poster_rotation', 'dynamic_rotation')
SENS_SYN = ('ecrot_bicycle', 'ecrot_street')
SENS_BASE = {'dt_ms': 20.0, 'n_iters': 75.0,
             **{k: THESIS_PARAMS[k] for k in ('delta_VFG', 'delta_IG', 'delta_GI', 'delta_RF',
                                              'delta_FR', 'delta_IMU')}}
# (csv column, x-axis label) of every one-at-a-time panel, left to right
SENS_AXES = [('delta_VFG', r'flow constraint $\delta_{VFG}$'),
             ('delta_RF', r'flow from $\omega$, $\delta_{RF}$'),
             ('delta_IG', r'$G$ from $I$, $\delta_{IG}$'),
             ('delta_GI', r'$I$ from $G$, $\delta_{GI}$'),
             ('n_iters', 'iterations per frame'),
             ('dt_ms', r'window $\Delta t$ (ms)')]
SENS_ERR_TOP = 36.0          # °/s; points above it are drawn at the top and labelled


def _csv_rows(pattern):
    import glob
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, pattern))):
        out += list(csv.DictReader(open(p)))
    return out


def sensitivity_data():
    """{panel: {value: {'real'|'syn': (err, |r|)}}} and the no-anchor point.
    'kappa' is delta_FR / delta_anchor over the grid, averaged over the
    settings that share it (they agree to 0.1 °/s and 0.01 in |r|)."""
    def mean(rows):
        out = {}
        for kind, seqs in (('real', SENS_REAL), ('syn', SENS_SYN)):
            sel = [r for r in rows if r['sequence'] in seqs]
            if sel:
                out[kind] = (np.mean([float(r['err']) for r in sel]),
                             np.mean([float(r['recon_mean']) for r in sel]))
        return out
    data = {}
    grid = [r for r in _csv_rows('experiments/grid/*.csv') if r['model'] == 'thesis_cmax']
    by_k = {}
    for r in grid:
        by_k.setdefault(round(float(r['delta_FR']) / float(r['delta_IMU']), 4), []).append(r)
    data['kappa'] = {k: mean(v) for k, v in sorted(by_k.items())}
    sens = [r for r in _csv_rows('experiments/sens/sens_*.csv') if r['model'] == 'thesis_cmax']
    for key, _ in SENS_AXES + [('delta_IMU', '')]:
        by_v = {}
        for r in sens:
            if all(abs(float(r[k]) - v) < 1e-9 for k, v in SENS_BASE.items() if k != key):
                by_v.setdefault(float(r[key]), []).append(r)
        data[key] = {v: mean(rs) for v, rs in sorted(by_v.items())}
    no_anchor = data.pop('delta_IMU').get(0.0)          # delta_anchor = 0 is thesis
    return data, no_anchor


def fig_sensitivity(E):
    data, no_anchor = sensitivity_data()
    name, col = MODELS['thesis_cmax']
    style = {'real': dict(ls='-', marker='o', mfc=col, label='real (mean of 3)'),
             'syn': dict(ls='--', marker='o', mfc='white', label='synthetic (mean of 2)')}
    panels = [('kappa', r'$\kappa=\delta_{FR}/\delta_{\mathrm{anchor}}$')] + SENS_AXES
    adopted = {'kappa': SENS_BASE['delta_FR'] / SENS_BASE['delta_IMU'], **SENS_BASE}
    INF = 12.0                                  # x position of the no-anchor point (kappa = inf)
    with use_style('paper'):
        fig, axes = plt.subplots(2, len(panels), figsize=(PAPER_WIDE, 3.0), sharey='row',
                                 gridspec_kw={'width_ratios': [1.6] + [1] * len(SENS_AXES)})
        for c, (key, xlabel) in enumerate(panels):
            ax_e, ax_r = axes[0, c], axes[1, c]
            pts = data.get(key, {})
            for kind in ('real', 'syn'):
                xs = [v for v in pts if kind in pts[v]]
                for ax, j in ((ax_e, 0), (ax_r, 1)):
                    ys = np.array([pts[v][kind][j] for v in xs])
                    shown = np.minimum(ys, SENS_ERR_TOP) if j == 0 else ys
                    ax.plot(xs, shown, color=col, lw=1.0, ms=3.2, mec=col, mew=0.8,
                            **{k: v for k, v in style[kind].items() if k != 'label'},
                            label=style[kind]['label'] if c == 0 and j == 0 else None)
                    for x, y in zip(xs, ys):
                        if j == 0 and y > SENS_ERR_TOP:
                            ax.annotate(f'{y:.0f}', (x, SENS_ERR_TOP), xytext=(3, -8),
                                        textcoords='offset points', fontsize=6.5)
                if key == 'kappa' and no_anchor and kind in no_anchor:
                    e, r = no_anchor[kind]
                    ax_e.plot([INF], [min(e, SENS_ERR_TOP)], color=col, ls='none', mew=0.8,
                              mec=col, ms=3.2, **{k: v for k, v in style[kind].items()
                                                  if k in ('marker', 'mfc')})
                    ax_r.plot([INF], [r], color=col, ls='none', mew=0.8, mec=col, ms=3.2,
                              **{k: v for k, v in style[kind].items() if k in ('marker', 'mfc')})
            for ax in (ax_e, ax_r):
                ax.set_xscale('log')
                ax.axvline(adopted[key], color=NEUTRAL_COLOR, lw=0.8, ls=':', zorder=0)
                ax.minorticks_off()
            vals = sorted(pts) + ([INF] if key == 'kappa' else [])
            ticks = vals if len(vals) <= 5 else [vals[0], adopted[key], vals[-1]]
            if key == 'kappa':
                ticks = [0.02, 0.2, 1, 5, INF]
            elif key == 'delta_VFG':
                ticks = [0.03, 0.12, 0.48]
            lab = lambda v: r'$\infty$' if v == INF else (f'{v:g}' if v >= 0.1 or key == 'dt_ms'
                                                          else f'{v:g}'.lstrip('0'))
            ax_r.set_xticks(ticks)
            ax_r.set_xticklabels([lab(v) for v in ticks], fontsize=7)
            ax_e.set_xticks(ticks)
            ax_e.set_xticklabels([])
            ax_r.set_xlabel(xlabel, fontsize=7.5)
        axes[0, 0].set_ylim(0, SENS_ERR_TOP)
        axes[1, 0].set_ylim(0, 0.62)
        axes[0, 0].set_ylabel(r'$\omega$ error (°/s)')
        axes[1, 0].set_ylabel(r'$|r|$ of the read-out')
        if no_anchor:
            e = '/'.join(f"{no_anchor[k][0]:.0f}" for k in ('real', 'syn') if k in no_anchor)
            axes[0, 0].annotate(e, (INF, SENS_ERR_TOP), xytext=(-4, -9), ha='right',
                                textcoords='offset points', fontsize=6.5)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside upper center',
                   ncol=2, fontsize=7.5, handlelength=2.4)
        save(fig, os.path.join(FIG, 'fig_sensitivity.png'))
    print('  wrote fig_sensitivity.pdf / .png')


FIGURES = {'beta_speed': fig_beta_speed, 'tracking': fig_tracking,
           'maps': fig_maps, 'appendix': fig_appendix, 'cmax_baseline': fig_cmax_baseline,
           'distortion': fig_distortion, 'converge': fig_converge, 'recon': fig_recon,
           'dissociation': fig_dissociation, 'regularisers': fig_regularisers,
           'image_settings': fig_image_settings, 'sensitivity': fig_sensitivity}


def main():
    ap = argparse.ArgumentParser(description='regenerate the report figures')
    ap.add_argument('--only', choices=sorted(FIGURES), help='one figure only')
    args = ap.parse_args()
    sys.argv = [sys.argv[0]]                     # evaluation.py parses argv
    import evaluation as E

    todo = [args.only] if args.only else list(FIGURES)
    for name in todo:
        print(f'{name}:')
        FIGURES[name](E)
    print(f'\nfigures in {FIG}')


if __name__ == '__main__':
    main()
