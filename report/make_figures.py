"""
make_figures.py — regenerate every figure in the report from committed code.

Run from the repository root:
    python report/make_figures.py              # all figures
    python report/make_figures.py --only maps  # one of: beta_speed tracking
                                               #         maps appendix distortion
                                               #         cmax_baseline

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

# (dataset, segment id, t_start, label) — Table II
SEGMENTS = [
    ('boxes_rotation',    'seg_B', 7.322,  'boxes'),
    ('poster_rotation',   'seg_C', 8.816,  'poster'),
    ('dynamic_rotation',  'seg_D', 1.619,  'dynamic'),
    ('ecrot_bicycle', 'seg_A', 1.001,  'bicycle'),
    ('ecrot_street',  'seg_A', 1.001,  'street'),
]

# fig_maps rows (index into SEGMENTS, label, frame shown, run root). Every
# sequence at the last frame of its Table III run, except poster: its last
# frame falls in the anchored excursion, so it is shown at POSTER_FRAME, the
# frame of highest |r| for CMax-anchor and also for the mean over all five
# models, from runs stopped there (run_experiments --what sweep --name
# maps_poster --duration 0.88). The runs are deterministic, so that frame is
# the Table III run's own frame 44.
POSTER_FRAME = 44
MAP_ROWS = [(0, 'boxes\n(real)', N_MAPS, 'results'),
            (1, f'poster\n(real, frame {POSTER_FRAME})', POSTER_FRAME,
             'experiments/maps_poster'),
            (2, 'dynamic\n(person in scene)', N_MAPS, 'results'),
            (3, 'bicycle\n(synthetic)', N_MAPS, 'results'),
            (4, 'street\n(synthetic)', N_MAPS, 'results')]
# the appendix repeats those frames for every model but the CMax-anchor one
APPENDIX_MODELS = ['cook', 'thesis', 'thesis_imu', 'thesis_cmax_v2']

# every segment of the two real sequences — the beta-vs-speed population
BETA_SEGMENTS = [
    ('boxes_rotation', 'seg_A', 9.972), ('boxes_rotation', 'seg_B', 7.322),
    ('boxes_rotation', 'seg_C', 4.422), ('boxes_rotation', 'seg_D', 1.922),
    ('poster_rotation', 'seg_A', 14.166), ('poster_rotation', 'seg_C', 8.816),
    ('poster_rotation', 'seg_D', 1.866),
]


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


def _draw_maps(maps, labels, name, height_per_row=1.62, title=None):
    labels = [lab for m, lab in zip(maps, labels) if m is not None]
    maps = [m for m in maps if m is not None]
    if not maps:
        print(f'  {name}: no runs found, skipped')
        return
    with use_style('paper'):
        fig, axes = plt.subplots(len(maps), len(_MAP_COLUMNS),
                                 figsize=(PAPER_WIDE, height_per_row * len(maps)))
        axes = np.atleast_2d(axes)
        for r, (arrs, lab) in enumerate(zip(maps, labels)):
            for c, ((kind, ttl), arr) in enumerate(zip(_MAP_COLUMNS, arrs)):
                draw_map(axes[r, c], kind, arr, title=ttl if r == 0 else '')
            axes[r, 0].set_ylabel(lab)
        if title:
            fig.suptitle(title)
        save(fig, os.path.join(FIG, f'{name}.png'))
    print(f'  wrote {name}.pdf / .png')


def fig_maps(E):
    _draw_maps(compute_maps(E, 'thesis_cmax'), [r[1] for r in MAP_ROWS], 'fig_maps')


def fig_appendix(E):
    """The frames of fig_maps for every other model, one figure each."""
    for model in APPENDIX_MODELS:
        _draw_maps(compute_maps(E, model), [r[1] for r in MAP_ROWS],
                   f'fig_appendix_maps_{model}', title=MODELS[model][0])


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

    panels = [(V_raw, r'raw events (lens modelled in $\mathcal{C}$)'),
              (V_und, 'undistorted events'),
              (V_und[10:80, 140:230], 'detail (undistorted)')]
    scale = map_scale('V', V_und)
    with use_style('paper'):
        fig, ax = plt.subplots(1, 3, figsize=(PAPER_WIDE, 2.15))
        for a, (img, ttl) in zip(ax, panels):
            draw_map(a, 'V', img, title=ttl, scale=scale)
        save(fig, os.path.join(FIG, 'fig_distortion.png'))
    print('  wrote fig_distortion.pdf / .png')


FIGURES = {'beta_speed': fig_beta_speed, 'tracking': fig_tracking,
           'maps': fig_maps, 'appendix': fig_appendix, 'cmax_baseline': fig_cmax_baseline,
           'distortion': fig_distortion}


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
