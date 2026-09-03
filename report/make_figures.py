"""
make_figures.py — regenerate every figure in the report from committed code.

Run from the repository root:
    python report/make_figures.py              # all figures
    python report/make_figures.py --only maps  # one of: beta_speed tracking
                                               #         maps appendix distortion

Outputs PDFs (used by the LaTeX) and PNGs (for quick inspection) into
report/figures/.

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
import matplotlib.colors as mcolors

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
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
    ('bycicle_sinthetic', 'seg_A', 1.001,  'bicycle'),
    ('street_sinthetic',  'seg_A', 1.001,  'street'),
]

# every segment of the two real sequences — the beta-vs-speed population
BETA_SEGMENTS = [
    ('boxes_rotation', 'seg_A', 9.972), ('boxes_rotation', 'seg_B', 7.322),
    ('boxes_rotation', 'seg_C', 4.422), ('boxes_rotation', 'seg_D', 1.922),
    ('poster_rotation', 'seg_A', 14.166), ('poster_rotation', 'seg_C', 8.816),
    ('poster_rotation', 'seg_D', 1.866),
]

plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.grid': True, 'grid.alpha': 0.3})


def _save(fig, name, **kw):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'{name}.{ext}'), dpi=190,
                    bbox_inches='tight', pad_inches=0.02, **kw)
    print(f'  wrote {name}.pdf / .png')


def find_run(ds, sid, t0, model, d_fr, d_anchor=None, n_iters=75):
    """Locate the results/ directory matching this configuration exactly.

    Matching on the step sizes ALONE is not enough. results/ accumulates runs
    from earlier configurations -- different iteration counts, and in particular
    the 'undistort_events' distortion mode, whose directories lack the _C_full
    suffix. Several of those share step sizes with the reported runs, and
    sorted() can return one of them first (e.g. 'i100' sorts before 'i75').
    Every field that changes the result is therefore checked, and an ambiguous
    match is reported rather than silently resolved.
    """
    hits = []
    for d in sorted(glob.glob(f'results/{ds}/{model}/{sid}_t{t0}_dt20ms_n150_*')):
        try:
            cfg = json.load(open(os.path.join(d, 'params.json')))
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
    return np.array([E.get_reference_omega(gt, imu, t0 + k*DT, t0 + (k+1)*DT, om)[0]
                     for k in range(n)])


# ---------------------------------------------------------------- beta vs speed
def fig_beta_speed(E):
    series = {
        'default': ('thesis_cmax',    0.5, 0.3, 'o', 'tab:red',
                    r'CMax-anchor, $\delta_{FR}=0.5$'),
        'tuned':   ('thesis_cmax',    0.1, 0.5, 's', 'tab:blue',
                    r'CMax-anchor, $\delta_{FR}=0.1$'),
        'inloop':  ('thesis_cmax_v2', 0.5, None, '^', 'tab:grey',
                    r'CMax-inloop'),
    }
    data = {k: ([], []) for k in series}
    for ds, sid, t0 in BETA_SEGMENTS:
        w = np.linalg.norm(reference_omega(E, ds, t0), axis=1).mean()
        for key, (model, d_fr, d_a, _, _, _) in series.items():
            d = find_run(ds, sid, t0, model, d_fr, d_a)
            if d is None:
                print(f'    (missing {ds}/{sid}/{model} dFR={d_fr})')
                continue
            b = json.load(open(os.path.join(d, 'summary.json')))['mean_beta']
            data[key][0].append(w)
            data[key][1].append(b)

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for key, (_, _, _, mk, col, lab) in series.items():
        x, y = np.array(data[key][0]), np.array(data[key][1])
        if len(x) < 2:
            continue
        r = np.corrcoef(x, y)[0, 1]
        ax.scatter(x, y, marker=mk, s=26, color=col, zorder=3,
                   edgecolors='white', linewidths=0.4, label=f'{lab}  ($r={r:+.2f}$)')
        xs = np.linspace(x.min(), x.max(), 20)
        ax.plot(xs, np.polyval(np.polyfit(x, y, 1), xs), color=col, lw=1.0,
                alpha=0.55, zorder=2)
    ax.axhline(1.0, color='k', lw=0.8, ls=':', zorder=1)
    ax.set_xlabel(r'rotation speed $\|\omega\|$  (rad/s)')
    ax.set_ylabel(r'scale factor $\beta$')
    ax.legend(frameon=False, loc='upper left')
    _save(fig, 'fig_beta_speed')
    plt.close(fig)


# -------------------------------------------------------------------- tracking
def fig_tracking(E):
    ds, sid, t0 = 'poster_rotation', 'seg_C', 8.816
    ref = reference_omega(E, ds, t0)
    t = np.arange(N_TRACK) * DT
    runs = [('CMax-anchor', find_run(ds, sid, t0, 'thesis_cmax', 0.1, 0.5), 'tab:blue'),
            ('thesis (no anchor)', find_run(ds, sid, t0, 'thesis', 0.1), 'tab:red')]

    fig, axes = plt.subplots(3, 1, figsize=(3.4, 3.6), sharex=True)
    for i, comp in enumerate('xyz'):
        axes[i].plot(t, np.degrees(ref[:, i]), color='k', lw=1.2, label='reference')
        for name, d, col in runs:
            if d is None:
                print(f'    (missing run for {name})')
                continue
            rows = list(csv.DictReader(open(os.path.join(d, 'tracking.csv'))))
            est = np.array([[float(r['est_wx']), float(r['est_wy']), float(r['est_wz'])]
                            for r in rows])[:N_TRACK]
            axes[i].plot(t, np.degrees(est[:, i]), color=col, lw=0.9, alpha=0.85,
                         label=name)
        axes[i].set_ylabel(rf'$\omega_{comp}$ ($^\circ$/s)')
    axes[2].set_xlabel('time (s)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=3, loc='upper center',
               bbox_to_anchor=(0.55, 1.004), handlelength=1.4, columnspacing=1.0)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.945))
    _save(fig, 'fig_tracking')
    plt.close(fig)


# ------------------------------------------------------------------- map panels
def compute_maps(E):
    """Run the network N_MAPS frames per sequence; cache the final maps."""
    from data_loader import EventFrameSequence, CameraCalibration, load_events_fast
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
        maps.append((V.copy(), net.I[:H, :W].copy(), net.G.copy(), net.F.copy()))
        print(f'    {label} done')
    np.savez_compressed(cache, **{f'{n}{i}': m for i, tup in enumerate(maps)
                                  for n, m in zip('VIGF', tup)})
    return maps


def _draw_maps(maps, labels, name, height_per_row=1.72):
    fig, axes = plt.subplots(len(labels), 4,
                             figsize=(7.05, height_per_row * len(labels)))
    axes = np.atleast_2d(axes)
    for r, ((V, I, G, F), lab) in enumerate(zip(maps, labels)):
        m = np.percentile(np.abs(V), 99.5) or 1e-6
        axes[r, 0].imshow(V, cmap='bwr', vmin=-m, vmax=m)          # V is signed
        lo, hi = np.percentile(I, [1, 99])
        axes[r, 1].imshow(I, cmap='gray', vmin=lo, vmax=max(hi, lo + 1e-6))
        g = np.linalg.norm(G, axis=-1)
        axes[r, 2].imshow(g, cmap='gray', vmin=0, vmax=np.percentile(g, 99) or 1e-6)
        ang = (np.arctan2(F[..., 1], F[..., 0]) + np.pi) / (2 * np.pi)
        mag = np.linalg.norm(F, axis=-1)
        mag = mag / (np.percentile(mag, 99) + 1e-9)
        axes[r, 3].imshow(mcolors.hsv_to_rgb(
            np.stack([ang, np.clip(mag, 0, 1), np.ones_like(ang)], -1)))
        for c in range(4):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            axes[r, c].grid(False)
        axes[r, 0].set_ylabel(lab, fontsize=7)
    for c, ttl in enumerate([r'events  $V$ (input)', r'intensity  $I$',
                             r'gradient  $\|G\|$', r'flow  $F$']):
        axes[0, c].set_title(ttl, fontsize=8, pad=4)
    fig.tight_layout(pad=0.25)
    _save(fig, name)
    plt.close(fig)


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
    from data_loader import (CameraCalibration, load_events_fast,
                             undistort_events, events_to_vframe)
    ds, t0, dur, H, W = 'poster_rotation', 8.816, 0.20, 180, 240
    p = E.get_dataset_paths(ds)
    calib = CameraCalibration(p['calib'])
    ev = load_events_fast(p['events'], t_start=t0, duration=dur + 0.02)
    ev = ev[ev[:, 0] < t0 + dur]
    V_raw = events_to_vframe(ev, H, W, clip_value=40.0, normalise=True)
    V_und = events_to_vframe(undistort_events(ev, calib), H, W,
                             clip_value=40.0, normalise=True)

    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.15))
    panels = [(V_raw, r'raw events (lens modelled in $\mathcal{C}$)'),
              (V_und, 'undistorted events'),
              (V_und[10:80, 140:230], 'detail (undistorted)')]
    for a, (img, ttl) in zip(ax, panels):
        m = np.percentile(np.abs(img), 99.5) or 1e-6
        a.imshow(img, cmap='bwr', vmin=-m, vmax=m, interpolation='nearest')
        a.set_title(ttl, fontsize=8, pad=4)
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.02, wspace=0.06)
    _save(fig, 'fig_distortion')
    plt.close(fig)


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
