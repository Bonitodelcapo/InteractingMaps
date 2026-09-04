"""
run_experiments.py — every experiment behind the report, from committed code.

Run from the repository root:
    python report/run_experiments.py --what grid        # 2-D (dFR, dAnchor) grid
    python report/run_experiments.py --what window      # dt / n_frames choice
    python report/run_experiments.py --what main        # Table III, 5 models
    python report/run_experiments.py --what converge    # per-iteration figure data
    python report/run_experiments.py --what baseline    # CMax / gyro alone, no network
    python report/run_experiments.py --what sweep \
        --vary 'delta_GI=0.05,0.5,1.0;poisson=iterative,fft,dct' \
        --base 'delta_FR=0.10,delta_IMU=0.50' --mode coord

Every run uses the reported segments (SEGMENTS below, = Table II) and carries
lens distortion in the C matrix (distortion_mode='C_full'). Nothing is looked up
in results/ -- each cell is computed -- because results/ mixes runs from
configurations that have since changed.

Where the runs land:
    results/                    -- only --what main, the runs the tables cite
    experiments/<name>/         -- everything exploratory (grid, sweep, poisson)
Each search gets its own directory via --name, so its runs and frames can be
browsed, compared, or deleted as a unit without disturbing results/. The csv
and figures for a search go to experiments/<name>/ as well.

Video frames are written for every run unless --no-frames is given, so any
number in the report can be checked against the maps that produced it. Frames
are the slow part; --frame-stride N writes every Nth frame.
"""

import os
import sys
import csv
import json
import time
import argparse
import traceback

import numpy as np

# evaluation.py prints the thesis' symbols; a cp1252 console cannot encode them
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# the segments reported in Table II -- do not substitute others
SEGMENTS = [
    ('boxes_rotation',    'seg_B', 7.322,  'boxes'),
    ('poster_rotation',   'seg_C', 8.816,  'poster'),
    ('dynamic_rotation',  'seg_D', 1.619,  'dynamic'),
    ('bycicle_sinthetic', 'seg_A', 1.001,  'bicycle'),
    ('street_sinthetic',  'seg_A', 1.001,  'street'),
]
MODELS = ['cook', 'thesis', 'thesis_imu', 'thesis_cmax', 'thesis_cmax_v2']
OUTDIR = os.path.join(ROOT, 'report', 'experiments')
os.makedirs(OUTDIR, exist_ok=True)


def _cfg(E, ds, sid, t0, model, dt, nf, out_root=None, **kw):
    seg = {'id': sid, 't_start': t0, 'frame_duration': dt, 'n_frames': nf,
           'initial_R': None, 'sensor_size': (180, 240)}
    return E.RunConfig(dataset=ds, model=model, segment=seg, n_frames=nf,
                       distortion_mode='C_full', out_root=out_root, **kw)


def _root(args, default):
    """Where this experiment's runs go: experiments/<name>/...

    Search runs vastly outnumber reported ones and would otherwise bury
    results/, which holds the runs the report's tables and figures point at.
    Each experiment gets its own directory, named for what it is.
    """
    name = args.name or default
    root = os.path.join('experiments', name)
    os.makedirs(root, exist_ok=True)
    return root


# ------------------------------------------------------------- sweep plumbing
#: every relaxation rate the thesis network exposes, plus the anchor weight.
#: delta_IMU is the anchor (gyro for thesis_imu, CMax for thesis_cmax).
ALL_DELTAS = ['delta_VFG', 'delta_IG', 'delta_GI', 'delta_RF', 'delta_FR',
              'delta_IMU']


def _parse_vary(spec):
    """'delta_GI=0.05,0.2;poisson=fft,dct' -> [('delta_GI',[0.05,0.2]), ...]"""
    axes = []
    for part in spec.split(';'):
        part = part.strip()
        if not part:
            continue
        name, _, vals = part.partition('=')
        name = name.strip()
        if name == 'poisson':
            axes.append((name, [v.strip() for v in vals.split(',')]))
        elif name in ALL_DELTAS:
            axes.append((name, [float(v) for v in vals.split(',')]))
        else:
            raise SystemExit(f"unknown sweep axis {name!r}; "
                             f"choose from {ALL_DELTAS + ['poisson']}")
    return axes


def _points(axes, mode, base):
    """The configurations to run: full factorial, or one axis at a time."""
    if mode == 'grid':
        pts, stack = [], [{}]
        for name, vals in axes:
            stack = [dict(p, **{name: v}) for p in stack for v in vals]
        pts = stack
    else:                                    # 'coord': vary one, hold the rest
        pts, seen = [], set()
        for name, vals in axes:
            for v in vals:
                p = {name: v}
                key = tuple(sorted({**base, **p}.items()))
                if key not in seen:
                    seen.add(key)
                    pts.append(p)
    return [dict(base, **p) for p in pts]


def _split(point):
    """Separate the network kwargs RunConfig takes separately."""
    p = dict(point)
    return p.pop('poisson', None), p


def _run(E, rc, save_frames, stride):
    t = time.time()
    E.experiment_tracking(rc, save_frames=save_frames, frame_stride=stride)
    s = json.load(open(os.path.join(rc.output_dir, 'summary.json')))
    return s, time.time() - t


# --------------------------------------------------------------- window choice
def cmd_window(E, args):
    """How constant is omega, and how far does the scene move inside one window?

    Decides frame_duration and n_frames before the grid is run. Uses ground
    truth only -- no network runs.
    """
    print(f"{'sequence':<10} {'dur':>5} {'|w|':>6} {'rel.std':>8} {'flips':>6}")
    print('-' * 42)
    rows = []
    for ds, sid, t0, lab in SEGMENTS:
        p = E.get_dataset_paths(ds)
        gt, imu = E.load_groundtruth(p['groundtruth']), E.load_imu(p['imu'])
        om = E.load_omega_gt(p.get('omega_gt'))
        for dur in (0.5, 1.0, 1.5, 2.0, 3.0):
            n = int(round(dur / 0.02))
            w = np.array([E.get_reference_omega(gt, imu, t0+k*0.02, t0+(k+1)*0.02,
                                                om)[0] for k in range(n)])
            mag = np.linalg.norm(w, axis=1)
            ax = int(np.argmax(np.abs(w).mean(0)))
            flips = int((np.diff(np.sign(w[:, ax])) != 0).sum())
            rows.append((lab, dur, mag.mean(), mag.std()/mag.mean(), flips))
            print(f"{lab:<10} {dur:5.1f} {mag.mean():6.3f} "
                  f"{mag.std()/mag.mean():8.3f} {flips:6d}")
        print()
    with open(os.path.join(OUTDIR, 'window_choice.csv'), 'w', newline='') as f:
        wr = csv.writer(f); wr.writerow(['sequence', 'duration_s', 'omega_mean',
                                         'rel_std', 'sign_flips'])
        wr.writerows([(a, b, round(c, 4), round(d, 4), e) for a, b, c, d, e in rows])
    print('scene motion inside one accumulation window (centre pixel):')
    print(f"{'sequence':<10} " + ' '.join(f'{d:>12}' for d in (10, 20, 30, 50)))
    for lab, dur, wm, _rs, _fl in [r for r in rows if r[1] == 1.0]:
        cells = [f"{np.degrees(wm*ms/1000):4.2f}d/{199.0*wm*ms/1000:4.1f}px"
                 for ms in (10, 20, 30, 50)]
        print(f"{lab:<10} " + ' '.join(f'{c:>12}' for c in cells))
    print(f"\n-> {os.path.join(OUTDIR, 'window_choice.csv')}")


# ------------------------------------------------------------------- 2-D grid
def cmd_grid(E, args):
    """Joint (delta_FR, delta_anchor) grid -- they interact, so a
    one-at-a-time sweep cannot locate the optimum."""
    d_fr = [float(x) for x in args.d_fr.split(',')]
    d_anc = [float(x) for x in args.d_anchor.split(',')]
    nf = int(round(args.duration / args.dt))
    root = _root(args, 'grid_FR_anchor')
    path = os.path.join(root, f'grid_dt{int(args.dt*1000)}ms.csv')
    f = open(path, 'w', newline='', encoding='utf-8')
    wr = csv.writer(f)
    wr.writerow(['sequence', 'segment', 'model', 'dt_ms', 'n_frames',
                 'delta_FR', 'delta_anchor', 'err', 'median', 'dir', 'beta', 'secs'])
    f.flush()
    print(f"{len(SEGMENTS)*len(d_fr)*len(d_anc)} runs   dt={args.dt*1000:.0f} ms, "
          f"{nf} frames ({args.duration:.1f} s), C_full, {args.model}", flush=True)
    for ds, sid, t0, lab in SEGMENTS:
        print(f"\n### {lab}/{sid}", flush=True)
        print(f"{'dFR':>6} " + ' '.join(f'{a:>8}' for a in d_anc), flush=True)
        for fr in d_fr:
            cells = []
            for anc in d_anc:
                try:
                    rc = _cfg(E, ds, sid, t0, args.model, args.dt, nf,
                              out_root=root, delta_FR=fr, delta_IMU=anc)
                    s, secs = _run(E, rc, not args.no_frames, args.frame_stride)
                    wr.writerow([ds, sid, args.model, int(args.dt*1000), nf, fr, anc,
                                 round(s['mean_err_deg_s'], 2),
                                 round(s['median_err_deg_s'], 2),
                                 round(s['mean_dir_err_deg'], 2),
                                 round(s['mean_beta'], 3), round(secs)])
                    f.flush()
                    cells.append(f"{s['mean_err_deg_s']:8.2f}")
                except Exception:
                    cells.append(f"{'FAIL':>8}"); traceback.print_exc()
            print(f"{fr:6.2f} " + ' '.join(cells), flush=True)
    f.close()
    print(f"\n-> {path}")


# ------------------------------------------------------------------ main table
def cmd_main(E, args):
    """Table III: every model on every reported segment."""
    nf = int(round(args.duration / args.dt))
    path = os.path.join(OUTDIR, f'main_dt{int(args.dt*1000)}ms.csv')
    f = open(path, 'w', newline='', encoding='utf-8')
    wr = csv.writer(f)
    wr.writerow(['sequence', 'segment', 'model', 'err', 'median', 'dir', 'beta',
                 'floor', 'floor_dir', 'secs'])
    f.flush()
    for ds, sid, t0, lab in SEGMENTS:
        p = E.get_dataset_paths(ds)
        gt, imu = E.load_groundtruth(p['groundtruth']), E.load_imu(p['imu'])
        om = E.load_omega_gt(p.get('omega_gt'))
        ref = np.array([E.get_reference_omega(gt, imu, t0+k*args.dt,
                                              t0+(k+1)*args.dt, om)[0]
                        for k in range(nf)])
        gy = np.array([E.get_gyro_for_frame(imu, t0+k*args.dt, t0+(k+1)*args.dt)
                       for k in range(nf)])
        floor = np.degrees(np.linalg.norm(gy-ref, axis=1)).mean()
        fdir = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', gy, ref) /
               (np.linalg.norm(gy, axis=1)*np.linalg.norm(ref, axis=1)), -1, 1))).mean()
        print(f"\n### {lab}/{sid}  floor {floor:.2f} deg/s", flush=True)
        for model in MODELS:
            try:
                rc = _cfg(E, ds, sid, t0, model, args.dt, nf)
                s, secs = _run(E, rc, not args.no_frames, args.frame_stride)
                wr.writerow([ds, sid, model, round(s['mean_err_deg_s'], 2),
                             round(s['median_err_deg_s'], 2),
                             round(s['mean_dir_err_deg'], 2),
                             round(s['mean_beta'], 3), round(floor, 2),
                             round(fdir, 2), round(secs)])
                f.flush()
                print(f"  {model:<16} err={s['mean_err_deg_s']:8.2f} "
                      f"dir={s['mean_dir_err_deg']:6.2f} "
                      f"beta={s['mean_beta']:7.3f}", flush=True)
            except Exception:
                print(f"!!! {lab}/{model} FAILED", flush=True); traceback.print_exc()
    f.close()
    print(f"\n-> {path}")


# ------------------------------------------------- per-iteration convergence
def cmd_converge(E, args):
    """How I sharpens across the message-passing iterations WITHIN one frame.

    Steps the network one iteration at a time via its real step(), so the anchor
    and the clipping behave exactly as in a normal run. (evaluation.py's
    experiment 1 re-implements the loop inline and never sets the anchor target,
    so the anchored models there are pulled towards omega = 0.)

    Rows are models, columns are iteration checkpoints. One real and one
    synthetic sequence.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from data_loader import (EventFrameSequence, CameraCalibration,
                             load_events_fast, undistort_events)
    from cmax import CMaxAngularVelocity

    checkpoints = [int(x) for x in args.checkpoints.split(',')]
    nmax = max(checkpoints)
    cases = [c for c in SEGMENTS if c[3] in args.sequences.split(',')]
    nf = args.frame + 1

    for ds, sid, t0, lab in cases:
        p = E.get_dataset_paths(ds)
        calib = CameraCalibration(p['calib'])
        rows = {}
        for model in MODELS:
            rc = _cfg(E, ds, sid, t0, model, args.dt, nf)
            seq = EventFrameSequence(p['events'], p['calib'],
                                     frame_duration=args.dt, t_start=t0,
                                     n_frames=nf, clip_value=10.0,
                                     undistort=rc.undistort_at_event_level,
                                     sensor_size=(180, 240))
            H, W = seq.H, seq.W
            frames = list(seq)
            V, _tm = frames[args.frame]
            net = E.make_network(rc, H, W, calib.fx, calib.fy, calib.cx, calib.cy)
            if hasattr(net, 'initialize_from_rotation'):
                net.initialize_from_rotation(rc.initial_R)
            else:
                net.R = rc.initial_R.copy()

            lo = t0 + args.frame * args.dt
            hi = lo + args.dt
            kw = {}
            if getattr(rc, 'use_cmax', False) or getattr(rc, 'use_imu', False):
                cev = undistort_events(
                    load_events_fast(p['events'], t_start=lo, duration=args.dt + 0.05),
                    calib)
                win = cev[(cev[:, 0] >= lo) & (cev[:, 0] < hi)]
                kw['events'] = win
                if getattr(rc, 'use_cmax', False) and len(win) > 10:
                    est = CMaxAngularVelocity(H, W, calib.fx, calib.fy,
                                              calib.cx, calib.cy, use_polarity=True)
                    kw['omega_imu'] = est.estimate(win, t_ref=0.5*(lo+hi),
                                                   omega_init=np.zeros(3))
                elif getattr(rc, 'use_imu', False):
                    kw['omega_imu'] = E.get_gyro_for_frame(
                        E.load_imu(p['imu']), lo, hi)

            snaps = {}
            for it in range(1, nmax + 1):
                try:
                    net.step(V, n_iters=1, **kw)
                except TypeError:
                    net.step(V, n_iters=1)
                if it in checkpoints:
                    I = net.I if net.I.shape == (H, W) else net.I[:H, :W]
                    snaps[it] = I.copy()
            rows[model] = snaps
            print(f"  {lab}/{model} done", flush=True)

        fig, axes = plt.subplots(len(MODELS), len(checkpoints),
                                 figsize=(1.8*len(checkpoints), 1.55*len(MODELS)))
        for r, model in enumerate(MODELS):
            for c, it in enumerate(checkpoints):
                I = rows[model][it]
                lo_, hi_ = np.percentile(I, [1, 99])
                axes[r, c].imshow(I, cmap='gray', vmin=lo_, vmax=max(hi_, lo_+1e-9))
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
                if r == 0:
                    axes[r, c].set_title(f'iter {it}', fontsize=8)
            axes[r, 0].set_ylabel(model.replace('thesis_', 'th_'), fontsize=7)
        fig.suptitle(f'{lab} {sid}: intensity map across message-passing '
                     f'iterations (frame {args.frame})', fontsize=10)
        fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.955))
        out = os.path.join(ROOT, 'report', 'figures', f'fig_converge_{lab}.pdf')
        for ext in ('pdf', 'png'):
            fig.savefig(out.replace('.pdf', f'.{ext}'), dpi=150,
                        bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)
        print(f"-> fig_converge_{lab}.pdf / .png")


# -------------------------------------------------------------- generic sweep
def cmd_sweep(E, args):
    """Sweep any of the relaxation rates, and the choice of Poisson solver.

    --mode coord varies one axis at a time around --base (cheap: sum of the
    axis lengths); --mode grid takes the full factorial (the product). Every
    run records the omega error AND two descriptors of the reconstruction, so
    the two metrics can be compared rather than assumed to agree.
    """
    axes = _parse_vary(args.vary)
    base = {}
    for part in filter(None, (s.strip() for s in args.base.split(','))):
        k, _, v = part.partition('=')
        base[k.strip()] = v.strip() if k.strip() == 'poisson' else float(v)
    pts = _points(axes, args.mode, base)
    seqs = [s for s in SEGMENTS
            if not args.sequences or s[3] in args.sequences.split(',')]
    nf = int(round(args.duration / args.dt))

    root = _root(args, f'sweep_{args.mode}')
    path = os.path.join(root, args.out or 'sweep.csv')
    f = open(path, 'w', newline='', encoding='utf-8')
    wr = csv.writer(f)
    cols = ['sequence', 'segment', 'model', 'dt_ms', 'n_frames', 'poisson'] \
        + ALL_DELTAS + ['err', 'median', 'dir', 'beta', 'low_freq', 'contrast', 'secs']
    wr.writerow(cols)
    print(f"{len(pts)} configs x {len(seqs)} sequences = {len(pts)*len(seqs)} runs "
          f"| dt={args.dt*1000:.0f}ms, {nf} frames, C_full, {args.model}", flush=True)
    for i, pt in enumerate(pts):
        print(f"\n### [{i+1}/{len(pts)}] " +
              ' '.join(f'{k}={v}' for k, v in sorted(pt.items())), flush=True)
        for ds, sid, t0, lab in seqs:
            poisson, deltas = _split(pt)
            try:
                rc = _cfg(E, ds, sid, t0, args.model, args.dt, nf,
                          out_root=root, poisson=poisson, deltas=deltas)
                s, secs = _run(E, rc, not args.no_frames, args.frame_stride)
                wr.writerow([ds, sid, args.model, int(args.dt*1000), nf,
                             rc.poisson]
                            + [rc.params.get(d, '') for d in ALL_DELTAS]
                            + [round(s['mean_err_deg_s'], 2),
                               round(s['median_err_deg_s'], 2),
                               round(s['mean_dir_err_deg'], 2),
                               round(s['mean_beta'], 3),
                               round(s.get('low_freq_share', float('nan')), 4),
                               round(s.get('contrast', float('nan')), 4),
                               round(secs)])
                f.flush()
                print(f"    {lab:<9} err={s['mean_err_deg_s']:7.2f}  "
                      f"beta={s['mean_beta']:5.3f}  "
                      f"lowf={s.get('low_freq_share', float('nan')):.3f}  "
                      f"contrast={s.get('contrast', float('nan')):.3f}", flush=True)
            except Exception:
                print(f"    {lab:<9} FAIL"); traceback.print_exc()
    f.close()
    print(f"\n-> {path}")


# ---------------------------------------------- Poisson solver: loop vs readout
def cmd_poisson(E, args):
    """Where should the exact solve of Eq. 6.64-6.65 be applied?

    Two places are possible, and they behave very differently:

    in-loop  — Cost_Spatial solves exactly on every iteration (poisson=fft/dct).
        I then feeds back into G through the delta_IG blend, and because the
        exact solve amplifies each spatial frequency of G by 1/|k|², any
        low-frequency error in G is integrated into large spurious blobs which
        then corrupt G in turn.
    read-out — the network runs with the iterative update (Eq. 6.61) and the
        exact solve is applied once, at the end, to the G it produced. The
        feedback path is absent, so the same closed-form solution recovers the
        low frequencies without the network chasing them.

    Runs both for each sequence and writes a comparison panel, so the choice is
    made on the images rather than on the argument above.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from interacting_maps.network_dissertation import solve_poisson_exact

    nf = int(round(args.duration / args.dt))
    root = _root(args, 'poisson_solvers')
    seqs = [s for s in SEGMENTS
            if not args.sequences or s[3] in args.sequences.split(',')]
    base = {'delta_FR': 0.10, 'delta_IMU': 0.50}
    for part in filter(None, (s.strip() for s in args.base.split(','))):
        k, _, v = part.partition('=')
        base[k.strip()] = float(v)

    for ds, sid, t0, lab in seqs:
        maps = {}
        for mode in args.modes.split(','):
            rc = _cfg(E, ds, sid, t0, args.model, args.dt, nf,
                      out_root=root, poisson=mode, deltas=dict(base))
            npz = os.path.join(rc.output_dir, 'maps_final.npz')
            if not os.path.exists(npz):
                print(f"  running {lab}/{mode} ...", flush=True)
                _run(E, rc, not args.no_frames, args.frame_stride)
            maps[mode] = np.load(npz)

        panels = [('APS reference', _aps(E, ds, t0 + nf * args.dt), None)]
        for m in maps:
            panels.append((f'{m} in-loop' if m != 'iterative'
                           else 'iterative (Eq. 6.61)', maps[m]['I'], None))
        for m in ('fft', 'dct'):
            panels.append((f'{m} read-out',
                           solve_poisson_exact(maps['iterative']['G'], m), None))

        print(f"\n{lab}: {'low_freq':>10} {'contrast':>10}")
        fig, ax = plt.subplots(1, len(panels), figsize=(2.4 * len(panels), 2.6))
        for a, (ttl, img, _) in zip(ax, panels):
            if img is None:
                a.axis('off'); continue
            st = E._intensity_stats(img)
            print(f"  {ttl:<22} {st['low_freq_share']:10.3f} {st['contrast']:10.3f}")
            lo, hi = np.percentile(img, [1, 99])
            a.imshow(img, cmap='gray', vmin=lo, vmax=max(hi, lo + 1e-9))
            a.set_title(ttl, fontsize=8)
            a.set_xticks([]); a.set_yticks([])
        fig.suptitle(f'{lab}: the same G, five ways of turning it into I',
                     fontsize=10)
        fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.9))
        out = os.path.join(root, f'poisson_{lab}{args.tag}.png')
        fig.savefig(out, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f"  -> {out}")


def _aps(E, ds, t):
    """The APS frame nearest time t, for visual reference."""
    import matplotlib.pyplot as plt
    p = E.get_dataset_paths(ds)
    try:
        rows = [l.split() for l in open(p['images']) if l.strip()]
        at = np.array([float(r[0]) for r in rows])
        j = int(np.argmin(np.abs(at - t)))
        img = plt.imread(os.path.join(p['data_dir'], rows[j][1].replace('\\', '/')))
        return img.mean(-1) if img.ndim == 3 else img
    except Exception:
        return None


# ------------------------------------------------------- front-end baselines
def cmd_baseline(E, args):
    """What do the two anchor signals score on their own, without the network?

    thesis_cmax feeds the network a per-frame CMax omega and thesis_imu feeds it
    the gyroscope. Neither table means anything until we know what those inputs
    score by themselves: if the network cannot beat its own anchor, the maps are
    not contributing to the omega estimate. Scored exactly as in
    experiment_tracking -- same undistorted events into CMax, same warm start
    from the previous frame, same reference omega, same compute_metrics.
    """
    from data_loader import load_events_fast, undistort_events, CameraCalibration
    from cmax import CMaxAngularVelocity

    nf = int(round(args.duration / args.dt))
    path = os.path.join(OUTDIR, f'baseline_dt{int(args.dt*1000)}ms.csv')
    f = open(path, 'w', newline='', encoding='utf-8')
    wr = csv.writer(f)
    wr.writerow(['sequence', 'segment', 'source', 'dt_ms', 'n_frames',
                 'err', 'median', 'dir', 'beta', 'secs'])
    print(f"{'sequence':<10} {'source':<6} {'err':>7} {'med':>7} {'dir':>6} "
          f"{'beta':>6}", flush=True)
    print('-' * 46, flush=True)
    for ds, sid, t0, lab in SEGMENTS:
        p = E.get_dataset_paths(ds)
        calib = CameraCalibration(p['calib'])
        gt = E.load_groundtruth(p['groundtruth'])
        imu = E.load_imu(p['imu'])
        om = E.load_omega_gt(p.get('omega_gt'))
        ev = undistort_events(
            load_events_fast(p['events'], t_start=t0,
                             duration=nf * args.dt + 0.1), calib)
        est = CMaxAngularVelocity(180, 240, calib.fx, calib.fy,
                                  calib.cx, calib.cy, use_polarity=True)
        acc = {'cmax': [], 'gyro': []}
        w_prev = np.zeros(3)
        t = time.time()
        for k in range(nf):
            t_lo, t_hi = t0 + k * args.dt, t0 + (k + 1) * args.dt
            w_ref, _ = E.get_reference_omega(gt, imu, t_lo, t_hi, om)
            win = ev[(ev[:, 0] >= t_lo) & (ev[:, 0] < t_hi)]
            if len(win) > 10:
                w_prev = est.estimate(win, t_ref=0.5 * (t_lo + t_hi),
                                      omega_init=w_prev)
            acc['cmax'].append(E.compute_metrics(w_prev, w_ref))
            acc['gyro'].append(E.compute_metrics(
                E.get_gyro_for_frame(imu, t_lo, t_hi), w_ref))
        secs = time.time() - t
        for src in ('cmax', 'gyro'):
            a = np.array(acc[src])                      # (n, 3): err, dir, beta
            wr.writerow([ds, sid, src, int(args.dt * 1000), nf,
                         round(a[:, 0].mean(), 2), round(np.median(a[:, 0]), 2),
                         round(a[:, 1].mean(), 2), round(a[:, 2].mean(), 3),
                         round(secs)])
            print(f"{lab:<10} {src:<6} {a[:, 0].mean():7.2f} "
                  f"{np.median(a[:, 0]):7.2f} {a[:, 1].mean():6.2f} "
                  f"{a[:, 2].mean():6.3f}", flush=True)
        f.flush()
    f.close()
    print(f"\n-> {path}")
    print('gyro on the synthetic sequences is meaningless (ESIM writes no IMU).')


COMMANDS = {'window': cmd_window, 'grid': cmd_grid, 'main': cmd_main,
            'converge': cmd_converge, 'baseline': cmd_baseline,
            'sweep': cmd_sweep, 'poisson': cmd_poisson}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--what', required=True, choices=sorted(COMMANDS))
    ap.add_argument('--dt', type=float, default=0.02, help='frame_duration (s)')
    ap.add_argument('--duration', type=float, default=3.0, help='track length (s)')
    ap.add_argument('--model', default='thesis_cmax')
    ap.add_argument('--d-fr', default='0.02,0.05,0.10,0.30,0.50')
    ap.add_argument('--d-anchor', default='0.1,0.3,0.5,0.7,0.9')
    ap.add_argument('--no-frames', action='store_true',
                    help='skip video_frames (they are the slow part)')
    ap.add_argument('--frame-stride', type=int, default=1,
                    help='write every Nth frame only')
    ap.add_argument('--frame', type=int, default=0,
                    help='converge: which frame to iterate on')
    ap.add_argument('--checkpoints', default='1,3,10,25,50,75',
                    help='converge: iteration checkpoints to show')
    ap.add_argument('--sequences', default='poster,bicycle',
                    help='converge/sweep: which sequences (labels from SEGMENTS)')
    ap.add_argument('--vary', default='',
                    help="sweep axes, e.g. 'delta_GI=0.05,0.2;poisson=fft,dct'")
    ap.add_argument('--base', default='',
                    help="sweep: held-fixed values, e.g. 'delta_FR=0.10,delta_IMU=0.50'")
    ap.add_argument('--mode', default='coord', choices=['coord', 'grid'],
                    help='sweep: one axis at a time, or the full factorial')
    ap.add_argument('--out', default='', help='sweep: output csv name')
    ap.add_argument('--name', default='',
                    help='experiment name; runs go to experiments/<name>/ '
                         'instead of results/ (searches only, not --what main)')
    ap.add_argument('--modes', default='iterative,fft,dct',
                    help='poisson: which in-loop solvers to include')
    ap.add_argument('--tag', default='', help='poisson: suffix for the figure name')
    args = ap.parse_args()
    sys.argv = [sys.argv[0]]              # evaluation.py parses argv
    import evaluation as E
    COMMANDS[args.what](E, args)


if __name__ == '__main__':
    main()
