"""
make_figures.py — regenerate every figure in the report from committed code.

Run from the repository root:
    python report/make_figures.py              # all figures
    python report/make_figures.py --only maps  # one of: beta_speed tracking
                                               #         maps appendix distortion

Outputs PDFs (used by the LaTeX) and PNGs (for quick inspection) into
report/figures/ via style.save.

Where the data comes from
-------------------------
beta_speed, tracking   read results/ directly (summary.json / tracking.csv), so
                       they need the corresponding runs to exist.
maps, appendix         re-run the network for 40 frames per sequence, because
                       the map state is not persisted by the harness. Cached in
                       report/figures/.maps_cache.npz -- delete it to recompute.
distortion             needs only the raw events.

Configuration
-------------
Every figure uses the configuration reported in the paper: delta_FR = 0.1,
delta_anchor = 0.5, 75 iterations, 20 ms windows, distortion carried in the
C matrix (DISTORTION_MODE = 'C_full'). SEGMENTS below are the segments named in
Table II.
"""

import os
import sys
import csv
import json
import glob
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
from interacting_maps.viz import draw_map, draw_omega, map_scale, net_maps
sys.argv = [sys.argv[0]] + sys.argv[1:]          # evaluation.py parses argv

FIG = os.path.join(ROOT, 'report', 'figures')
os.makedirs(FIG, exist_ok=True)

DT, N_TRACK, N_MAPS = 0.02, 150, 40
TUNED = dict(delta_FR=0.1, delta_IMU=0.5)

# (dataset, segment id, t_start, label) — Table II
SEGMENTS = [
    ('boxes_rotation',    'seg_B', 7.322,  'boxes'),
    ('poster_rotation',   'seg_C', 8.816,  'poster'),
    ('dynamic_rotation',  'seg_D', 1.619,  'dynamic'),
    ('ecrot_bicycle', 'seg_A', 1.001,  'bicycle'),
    ('ecrot_street',  'seg_A', 1.001,  'street'),
]

# every segment of the two real sequences — the beta-vs-speed population
BETA_SEGMENTS = [
    ('boxes_rotation', 'seg_A', 9.972), ('boxes_rotation', 'seg_B', 7.322),
    ('boxes_rotation', 'seg_C', 4.422), ('boxes_rotation', 'seg_D', 1.922),
    ('poster_rotation', 'seg_A', 14.166), ('poster_rotation', 'seg_C', 8.816),
    ('poster_rotation', 'seg_D', 1.866),
]


def find_run(ds, sid, t0, model, d_fr, d_anchor=None, n_iters=75,
             poisson='iterative', deltas=None):
    """Locate the results/ directory matching this configuration exactly.

    Matching on the step sizes ALONE is not enough. results/ accumulates runs
    from earlier configurations -- different iteration counts, and in particular
    the 'undistort_events' distortion mode, whose directories lack the _C_full
    suffix. Several of those share step sizes with the reported runs, and
    sorted() can return one of them first (e.g. 'i100' sorts before 'i75').
    Every field that changes the result is therefore checked, and an ambiguous
    match is reported rather than silently resolved.

    `poisson` and `deltas` extend that guard to the parameters added for the
    solver comparison: a run differing only in the I-update or in one of the
    remaining relaxation rates must not be picked up as if it were this one.
    Older runs carry no 'poisson' key and are treated as 'iterative'.
    """
    hits = []
    for d in sorted(glob.glob(f'results/{ds}/{model}/{sid}_t{t0}_dt20ms_n150_*')):
        try:
            cfg = experiment_tracking.read_config(d)
            pr = cfg['params']
        except Exception:
            continue
        if cfg.get('distortion_mode') != 'C_full':
            continue
        if int(cfg.get('n_iters', -1)) != n_iters:
            continue
        if abs(pr['delta_FR'] - d_fr) > 1e-9:
            continue
        if d_anchor is not None and abs(pr.get('delta_IMU', -1) - d_anchor) > 1e-9:
            continue
        if cfg.get('poisson', 'iterative') != poisson:
            continue
        if any(abs(pr.get(k, float('nan')) - v) > 1e-9
               for k, v in (deltas or {}).items()):
            continue
        hits.append(d)
    if len(hits) > 1:
        print(f'    AMBIGUOUS ({len(hits)} runs match) {ds}/{sid}/{model} '
              f'dFR={d_fr} anchor={d_anchor}; using {os.path.basename(hits[0])}')
    return hits[0] if hits else None


def reference_omega(E, ds, t0, n=N_TRACK):
    p = E.get_dataset_paths(ds)
    gt = E.load_groundtruth(p['groundtruth'])
    imu = E.load_imu(p['imu'])
    om = E.load_omega_gt(p.get('omega_gt'))
    refs = [E.get_reference_omega(gt, imu, t0 + k*DT, t0 + (k+1)*DT, om) for k in range(n)]
    return np.array([w for w, _ in refs]), refs[0][1]


# ---------------------------------------------------------------- beta vs speed
def fig_beta_speed(E):
    # (model, delta_FR, delta_anchor, filled marker)
    series = {
        'default': ('thesis_cmax',    0.5, 0.3,  False),
        'tuned':   ('thesis_cmax',    0.1, 0.5,  True),
        'inloop':  ('thesis_cmax_v2', 0.5, None, True),
    }
    data = {k: ([], []) for k in series}
    for ds, sid, t0 in BETA_SEGMENTS:
        w = to_deg_s(np.linalg.norm(reference_omega(E, ds, t0)[0], axis=1).mean())
        for key, (model, d_fr, d_a, _) in series.items():
            d = find_run(ds, sid, t0, model, d_fr, d_a)
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
    runs = [('thesis_cmax', find_run(ds, sid, t0, 'thesis_cmax', 0.1, 0.5)),
            ('thesis', find_run(ds, sid, t0, 'thesis', 0.1))]

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


# ------------------------------------------------------------------- map panels
def compute_maps(E):
    """Run the network N_MAPS frames per sequence; cache the final maps."""
    from interacting_maps.data_loader import EventFrameSequence, CameraCalibration, load_events_fast
    cache = os.path.join(FIG, '.maps_cache.npz')
    if os.path.exists(cache):
        z = np.load(cache)
        return [(z[f'V{i}'], z[f'I{i}'], z[f'G{i}'], z[f'F{i}'])
                for i in range(len(SEGMENTS))]
    maps = []
    for ds, sid, t0, label in SEGMENTS:
        p = E.get_dataset_paths(ds)
        calib = CameraCalibration(p['calib'])
        sq = EventFrameSequence(p['events'], p['calib'], frame_duration=DT, t_start=t0,
                                n_frames=N_MAPS, clip_value=10.0, undistort=False,
                                sensor_size=(180, 240))
        H, W = sq.H, sq.W
        rc = E.RunConfig(dataset=ds, model='thesis_cmax',
                         segment={'id': sid, 't_start': t0, 'frame_duration': DT,
                                  'n_frames': N_MAPS, 'initial_R': None,
                                  'sensor_size': (180, 240)},
                         n_frames=N_MAPS, **dict(TUNED))
        net = E.make_network(rc, H, W, calib.fx, calib.fy, calib.cx, calib.cy)
        net.initialize_from_rotation(rc.initial_R)
        ev = load_events_fast(p['events'], t_start=t0, duration=N_MAPS * DT + 0.05)
        V = None
        for k, (V, _tm) in enumerate(sq):
            win = ev[(ev[:, 0] >= t0 + k*DT) & (ev[:, 0] < t0 + (k+1)*DT)]
            net.step(V, n_iters=75, events=win)
        m = net_maps(net, H, W)
        maps.append((V.copy(), m['I'].copy(), m['G'].copy(), m['F'].copy()))
        print(f'    {label} done')
    np.savez_compressed(cache, **{f'{n}{i}': m for i, tup in enumerate(maps)
                                  for n, m in zip('VIGF', tup)})
    return maps


def _draw_maps(maps, labels, name, height_per_row=1.72):
    with use_style('paper'):
        fig, axes = plt.subplots(len(labels), 4, figsize=(PAPER_WIDE, height_per_row * len(labels)))
        axes = np.atleast_2d(axes)
        for r, (arrs, lab) in enumerate(zip(maps, labels)):
            for c, (kind, arr) in enumerate(zip('VIGF', arrs)):
                draw_map(axes[r, c], kind, arr, title=None if r == 0 else '')
            axes[r, 0].set_ylabel(lab)
        save(fig, os.path.join(FIG, f'{name}.png'))
    print(f'  wrote {name}.pdf / .png')


def fig_maps(E):
    maps = compute_maps(E)
    idx = [1, 2, 4]                       # poster, dynamic, street
    _draw_maps([maps[i] for i in idx],
               ['poster (real)', 'dynamic (person in scene)', 'street (synthetic)'],
               'fig_maps')


def fig_appendix(E):
    maps = compute_maps(E)
    labels = [f'{lab}\n{sid}' for _ds, sid, _t0, lab in SEGMENTS]
    _draw_maps(maps, labels, 'fig_appendix_maps', height_per_row=1.62)


# ------------------------------------------------------------------ distortion
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
           'maps': fig_maps, 'appendix': fig_appendix,
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
