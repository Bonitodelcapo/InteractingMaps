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
maps, appendix         read the final maps (maps_final.npz) of the Table III
                       runs in results/: CMax-anchor for maps, one figure per
                       other model for appendix.
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

import provenance  # merged run.json reader (after sys.path points at ROOT)
from viz import grad_to_rgb
sys.argv = [sys.argv[0]] + sys.argv[1:]          # evaluation.py parses argv

FIG = os.path.join(ROOT, 'report', 'figures')
os.makedirs(FIG, exist_ok=True)

# Both match Table III: 75 frames of 20 ms = 1.5 s. The figures and the
# tables must describe the same runs, or a reader comparing them finds a
# disagreement that is only a difference of protocol.
DT, N_TRACK, N_MAPS = 0.02, 75, 75
# beta-vs-speed pools seven segments, of which only the Table II ones were run
# at 1.5 s; it uses the 3 s (150-frame) runs of tab:anchor throughout instead.
N_BETA = 150
TUNED = dict(delta_FR=0.1, delta_IMU=0.5)

# (dataset, segment id, t_start, label) — Table II
SEGMENTS = [
    ('boxes_rotation',    'seg_B', 7.322,  'boxes'),
    ('poster_rotation',   'seg_C', 8.816,  'poster'),
    ('dynamic_rotation',  'seg_D', 1.619,  'dynamic'),
    ('bycicle_sinthetic', 'seg_A', 1.001,  'bicycle'),
    ('street_sinthetic',  'seg_A', 1.001,  'street'),
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
# the appendix repeats those frames for every model but the CMax-anchor one;
# names as the report's Table III writes them
APPENDIX_MODELS = ['cook', 'thesis', 'thesis_imu', 'thesis_cmax_v2']
DISPLAY = {'cook': 'cook', 'thesis': 'thesis (no anchor)',
           'thesis_imu': 'thesis+IMU', 'thesis_cmax_v2': 'CMax-inloop'}

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


def find_run(ds, sid, t0, model, d_fr, d_anchor=None, n_iters=75,
             poisson='iterative', deltas=None, n_frames=N_TRACK, dt=DT,
             root='results'):
    """Locate the run directory under `root` matching this configuration exactly.

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
    The regularisers default to off, as everywhere outside their own section,
    so e.g. the _curl0.6 ablation run is not mistaken for the reported one.
    """
    deltas = {'delta_shrinkI': 0.0, 'delta_map': 0.0, 'delta_curl': 0.0,
              **(deltas or {})}
    hits = []
    pat = f'{root}/{ds}/{model}/{sid}_t{t0}_dt{round(dt * 1000)}ms_n{n_frames}_*'
    for d in sorted(glob.glob(pat)):
        try:
            cfg = provenance.read_config(d)
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
               for k, v in deltas.items()):
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
        w = np.linalg.norm(reference_omega(E, ds, t0, N_BETA), axis=1).mean()
        for key, (model, d_fr, d_a, _, _, _) in series.items():
            d = find_run(ds, sid, t0, model, d_fr, d_a, n_frames=N_BETA)
            if d is None:
                print(f'    (missing {ds}/{sid}/{model} dFR={d_fr})')
                continue
            b = provenance.read_summary(d)['mean_beta']
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
def compute_maps(E, model):
    """Final maps of one model's runs at the MAP_ROWS frames (Table III runs,
    except poster: see MAP_ROWS).

    Read from each run's maps_final.npz rather than re-running the network: a
    re-run here once passed the events without the CMax anchor, so the figure
    showed an unanchored network under a caption naming the anchored one.
    A sequence without a completed run comes back as None.
    """
    from data_loader import EventFrameSequence
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
        maps.append((V, m['I'], m['G'], m['F'], solve_poisson_exact(m['G'], 'fft')))
        print(f'    {label}/{model} ({os.path.basename(d)})')
    return maps


def _direction_key(ax):
    """Colour wheel beside a panel title: the hue each vector direction gets."""
    yy, xx = np.mgrid[-1:1:65j, -1:1:65j]
    wheel = grad_to_rgb(np.stack([xx, yy], -1), pct=100)
    key = ax.inset_axes([0.0, 1.01, 0.2, 0.22])
    key.imshow(np.dstack([wheel, np.hypot(xx, yy) <= 1]))
    key.set_anchor('W')
    key.axis('off')


def _draw_maps(maps, labels, name, height_per_row=1.62, title=None):
    labels = [lab for m, lab in zip(maps, labels) if m is not None]
    maps = [m for m in maps if m is not None]
    fig, axes = plt.subplots(len(labels), 5,
                             figsize=(8.6, height_per_row * len(labels)))
    axes = np.atleast_2d(axes)
    for r, ((V, I, G, F, I_ro), lab) in enumerate(zip(maps, labels)):
        m = np.percentile(np.abs(V), 99.5) or 1e-6
        axes[r, 0].imshow(V, cmap='bwr', vmin=-m, vmax=m)          # V is signed
        lo, hi = np.percentile(I, [1, 99])
        axes[r, 1].imshow(I, cmap='gray', vmin=lo, vmax=max(hi, lo + 1e-6))
        lo, hi = np.percentile(I_ro, [1, 99])
        axes[r, 2].imshow(I_ro, cmap='gray', vmin=lo, vmax=max(hi, lo + 1e-6))
        axes[r, 3].imshow(grad_to_rgb(G))           # hue = direction
        ang = (np.arctan2(F[..., 1], F[..., 0]) + np.pi) / (2 * np.pi)
        mag = np.linalg.norm(F, axis=-1)
        mag = mag / (np.percentile(mag, 99) + 1e-9)
        axes[r, 4].imshow(mcolors.hsv_to_rgb(
            np.stack([ang, np.clip(mag, 0, 1), np.ones_like(ang)], -1)))
        for c in range(5):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            axes[r, c].grid(False)
        axes[r, 0].set_ylabel(lab, fontsize=7)
    for c, ttl in enumerate([r'events  $V$ (input)', r'intensity  $I$ (in-loop)',
                             r'intensity  $I$ (read-out)',
                             r'gradient  $G$', r'flow  $F$']):
        axes[0, c].set_title(ttl, fontsize=8, pad=4, y=1.0)  # y: ignore the key
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout(pad=0.25)
    _direction_key(axes[0, 3])
    _save(fig, name)
    plt.close(fig)


def fig_maps(E):
    _draw_maps(compute_maps(E, 'thesis_cmax'), [r[1] for r in MAP_ROWS],
               'fig_maps')


def fig_appendix(E):
    """The frames of fig_maps for every other model, one figure each."""
    for model in APPENDIX_MODELS:
        _draw_maps(compute_maps(E, model), [r[1] for r in MAP_ROWS],
                   f'fig_appendix_maps_{model}', title=DISPLAY[model])


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
