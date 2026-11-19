"""
run_experiments.py — every experiment behind the report, from committed code.

Run from the repository root:
    python report/run_experiments.py --what window      # dt / n_frames choice
    python report/run_experiments.py --what main        # Table III, 5 models
    python report/run_experiments.py --what converge    # per-iteration figure data
    python report/run_experiments.py --what recon       # runs of the reconstruction figure
    python report/run_experiments.py --what baseline    # CMax / gyro alone, no network
    python report/run_experiments.py --what beta        # runs behind fig_beta_speed
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
import collections
import glob

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

import experiment_tracking  # merged run.json reader (after sys.path points at ROOT)
from interacting_maps.style import PAPER_WIDE, use_style, save
from interacting_maps.viz import draw_map, map_scale, net_maps
from interacting_maps.metrics import abs_corr, crop_interior
from report_runs import (SEGMENTS, MODELS, ADOPTED, BETA_SEGMENTS, BETA_CONFIGS,
                         run_cfg, recon_cfg, converge_path, aps_frame)

OUTDIR = os.path.join(ROOT, 'report', 'experiments')
os.makedirs(OUTDIR, exist_ok=True)


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
              'delta_IMU', 'delta_curl', 'delta_map', 'delta_shrinkI']


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
        elif name == 'n_iters':
            axes.append((name, [int(v) for v in vals.split(',')]))
        elif name in ALL_DELTAS + ['dt']:
            axes.append((name, [float(v) for v in vals.split(',')]))
        else:
            raise SystemExit(f"unknown sweep axis {name!r}; choose from "
                             f"{ALL_DELTAS + ['dt', 'n_iters', 'poisson']}")
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
    """Separate the axes RunConfig takes as its own arguments from the deltas.

    dt is returned separately because changing it also changes n_frames: a
    sweep over the accumulation window must hold the tracked DURATION fixed,
    not the frame count, or it would confound the window size with how much of
    the sequence was seen.
    """
    p = dict(point)
    return p.pop('poisson', None), p.pop('dt', None), p.pop('n_iters', None), p


def _run(E, rc, save_frames, stride):
    t = time.time()
    E.experiment_tracking(rc, save_frames=save_frames, frame_stride=stride)
    s = experiment_tracking.read_summary(rc.output_dir)
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


# ------------------------------------------------------------------ main table
def cmd_main(E, args):
    """Table III: every model on every reported segment.

    --vary accepts the usual axis syntax so the same table can be produced at a
    second configuration (e.g. 'delta_curl=0.6') and compared row for row. With
    no --vary this is the configuration of config.py, i.e. the two published
    models with no regulariser.
    """
    extra = {}
    for name, vals in _parse_vary(args.vary):
        extra[name] = vals[0]
    tag = ''.join(f'_{k.replace("delta_", "")}{v:g}'
                  for k, v in sorted(extra.items()))
    poisson = extra.pop('poisson', None)
    nf = int(round(args.duration / args.dt))
    # --sequences and --out let the table be built in pieces and merged, since
    # 25 runs do not always fit in one session.
    #
    # Validate everything BEFORE opening the output file. Opening it truncates,
    # so an argument error raised afterwards destroys the previous table --
    # which is exactly what happened once, costing 23 completed runs.
    seqs = [x for x in SEGMENTS
            if not args.sequences or x[3] in args.sequences.split(',')]
    if args.sequences:
        unknown = set(args.sequences.split(',')) - {x[3] for x in SEGMENTS}
        if unknown:
            raise SystemExit(f'unknown sequence(s) {sorted(unknown)}; '
                             f'choose from {[x[3] for x in SEGMENTS]}')
    want = args.models.split(',') if args.models else MODELS
    unknown = [m for m in want if m not in MODELS]
    if unknown:
        raise SystemExit(f'unknown model(s) {unknown}; choose from {MODELS}')
    models = [m for m in MODELS if m in want]

    path = os.path.join(OUTDIR,
                        args.out or f'main_dt{int(args.dt*1000)}ms{tag}.csv')
    f = open(path, 'w', newline='', encoding='utf-8')
    wr = csv.writer(f)
    wr.writerow(['sequence', 'segment', 'model', 'err', 'median', 'dir', 'beta',
                 'recon', 'recon_inloop', 'blowup', 'floor', 'floor_dir', 'secs'])
    f.flush()
    for ds, sid, t0, lab in seqs:
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
        for model in models:
            try:
                rc = run_cfg(ds, sid, t0, model, args.dt, nf,
                          poisson=poisson, deltas=dict(extra))
                s, secs = _run(E, rc, not args.no_frames, args.frame_stride)
                wr.writerow([ds, sid, model, round(s['mean_err_deg_s'], 2),
                             round(s['median_err_deg_s'], 2),
                             round(s['mean_dir_err_deg'], 2),
                             round(s['mean_beta'], 3),
                             round(s.get('recon_r_mean', float('nan')), 4),
                             round(s.get('recon_r_mean_inloop', float('nan')), 4),
                             round(s.get('blowup_frac', float('nan')), 4),
                             round(floor, 2),
                             round(fdir, 2), round(secs)])
                f.flush()
                print(f"  {model:<16} err={s['mean_err_deg_s']:8.2f} "
                      f"dir={s['mean_dir_err_deg']:6.2f} "
                      f"beta={s['mean_beta']:7.3f} "
                      f"r={s.get('recon_r_mean', float('nan')):6.3f}", flush=True)
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

    Stores I at every checkpoint, per model, in converge_path(); the figure is
    make_figures.fig_converge (rows: models, columns: checkpoints).
    """
    from interacting_maps.data_loader import (EventFrameSequence, CameraCalibration,
                             load_events_fast, undistort_events)
    from interacting_maps.angular_velocity import CMaxAngularVelocity

    checkpoints = [int(x) for x in args.checkpoints.split(',')]
    nmax = max(checkpoints)
    cases = [c for c in SEGMENTS if c[3] in args.sequences.split(',')]
    nf = args.frame + 1

    for ds, sid, t0, lab in cases:
        p = E.get_dataset_paths(ds)
        calib = CameraCalibration(p['calib'])
        rows = {}
        for model in MODELS:
            rc = run_cfg(ds, sid, t0, model, args.dt, nf)
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
                    snaps[it] = net_maps(net, H, W)['I'].copy()
            rows[model] = snaps
            print(f"  {lab}/{model} done", flush=True)

        out = converge_path(lab, args.frame)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out, checkpoints=checkpoints, segment=sid,
                            **{f'{m}_{it}': rows[m][it] for m in MODELS for it in checkpoints})
        print(f"-> {out}")


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
    cols = ['sequence', 'segment', 'model', 'dt_ms', 'n_frames', 'n_iters',
            'poisson'] + ALL_DELTAS + ['err', 'median', 'dir', 'beta',
            'low_freq', 'contrast', 'curl', 'recon_mean', 'recon_min',
            'recon_inloop', 'recon_inloop_min', 'blowup', 'secs']
    wr.writerow(cols)
    print(f"{len(pts)} configs x {len(seqs)} sequences = {len(pts)*len(seqs)} runs "
          f"| {args.duration:.1f}s tracked, C_full, {args.model}", flush=True)
    for i, pt in enumerate(pts):
        print(f"\n### [{i+1}/{len(pts)}] " +
              ' '.join(f'{k}={v}' for k, v in sorted(pt.items())), flush=True)
        for ds, sid, t0, lab in seqs:
            poisson, dt, n_iters, deltas = _split(pt)
            dt = dt or args.dt
            # hold the tracked duration fixed, not the frame count
            nfr = int(round(args.duration / dt))
            try:
                rc = run_cfg(ds, sid, t0, args.model, dt, nfr,
                          out_root=root, poisson=poisson, deltas=deltas,
                          n_iters=n_iters)
                s, secs = _run(E, rc, not args.no_frames, args.frame_stride)
                wr.writerow([ds, sid, args.model, round(dt*1000, 1), nfr,
                             rc.n_iters, rc.poisson]
                            + [rc.params.get(d, '') for d in ALL_DELTAS]
                            + [round(s['mean_err_deg_s'], 2),
                               round(s['median_err_deg_s'], 2),
                               round(s['mean_dir_err_deg'], 2),
                               round(s['mean_beta'], 3),
                               round(s.get('low_freq_share', float('nan')), 4),
                               round(s.get('contrast', float('nan')), 4),
                               round(s.get('curl_share', float('nan')), 4),
                               round(s.get('recon_r_mean', float('nan')), 4),
                               round(s.get('recon_r_min', float('nan')), 4),
                               round(s.get('recon_r_mean_inloop', float('nan')), 4),
                               round(s.get('recon_r_min_inloop', float('nan')), 4),
                               round(s.get('blowup_frac', float('nan')), 4),
                               round(secs)])
                f.flush()
                print(f"    {lab:<9} err={s['mean_err_deg_s']:7.2f}  "
                      f"beta={s['mean_beta']:5.3f}  "
                      f"lowf={s.get('low_freq_share', float('nan')):.3f}  "
                      f"r_loop={s.get('recon_r_mean_inloop', float('nan')):.3f}  "
                      f"r_out={s.get('recon_r_mean', float('nan')):.3f}  "
                      f"blow={s.get('blowup_frac', float('nan')):.2f}", flush=True)
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
    base = dict(ADOPTED)
    for part in filter(None, (s.strip() for s in args.base.split(','))):
        k, _, v = part.partition('=')
        base[k.strip()] = float(v)

    for ds, sid, t0, lab in seqs:
        maps = {}
        for mode in args.modes.split(','):
            rc = run_cfg(ds, sid, t0, args.model, args.dt, nf,
                      out_root=root, poisson=mode, deltas=dict(base))
            npz = os.path.join(rc.output_dir, 'maps_final.npz')
            print(f"  running {lab}/{mode} ...", flush=True)
            _run(E, rc, not args.no_frames, args.frame_stride)
            maps[mode] = np.load(npz)

        panels = [('APS reference', aps_frame(ds, t0 + nf * args.dt), 'aps')]
        for m in maps:
            panels.append((f'{m} in-loop' if m != 'iterative'
                           else 'iterative (Eq. 6.61)', maps[m]['I'], 'I'))
        for m in ('fft', 'dct'):
            panels.append((f'{m} read-out',
                           solve_poisson_exact(maps['iterative']['G'], m), 'I'))

        print(f"\n{lab}: {'low_freq':>10} {'contrast':>10}")
        out = os.path.join(root, f'poisson_{lab}{args.tag}.png')
        with use_style('paper'):
            fig, ax = plt.subplots(1, len(panels), figsize=(PAPER_WIDE, 1.9))
            for a, (ttl, img, kind) in zip(ax, panels):
                if img is None:
                    a.set_axis_off()
                    continue
                st = E._intensity_stats(img)
                print(f"  {ttl:<22} {st['low_freq_share']:10.3f} {st['contrast']:10.3f}")
                draw_map(a, kind, img, title=ttl)
            fig.suptitle(f'{lab}: the same G, five ways of turning it into I')
            save(fig, out)
        print(f"  -> {out}")


# ------------------------------------------------ runs of the reconstruction figure
def cmd_recon(E, args):
    """Runs behind make_figures.fig_recon: CMax-anchor at dt = 20 and 10 ms.

    The figure separates two effects on the intensity map: halving the
    accumulation window cleans up G, and solving the Poisson equation exactly
    recovers the low frequencies the iterative update leaves out of it.
    """
    root = _root(args, 'recon_figure')
    seqs = [s for s in SEGMENTS
            if not args.sequences or s[3] in args.sequences.split(',')]
    for ds, sid, t0, lab in seqs:
        for dt in (0.02, 0.01):
            rc = recon_cfg(ds, sid, t0, args.model, dt, args.duration, root)
            print(f"  running {lab} dt={dt*1000:.0f}ms ...", flush=True)
            _run(E, rc, not args.no_frames, args.frame_stride)


# ------------------------------------------------- fft vs dct, as a read-out
def cmd_readout(E, args):
    """Which exact solver reconstructs the better image: fft or dct?

    They differ only in what they assume beyond the image border -- fft that it
    wraps around (periodic), dct that it mirrors (Neumann, zero gradient across
    the edge). On a synthetic test dct is exact and fft is off by a wrap-around
    ramp, yet on real runs fft has so far looked better. This decides it on
    data instead of on the argument.

    Scoring: |Pearson r| between the reconstruction and log(1+APS), APS in [0,1] (near-linear).
    Correlation is invariant to scale and offset, which is exactly the freedom
    the reconstruction has -- the beta ambiguity fixes neither the contrast of
    I nor its constant of integration, so any metric that is not invariant to
    both would be measuring the gauge rather than the image.

    Each sequence is scored at several track lengths, because a single
    end-of-run snapshot is not representative: quality varies with where the
    run happens to stop. The comparison is paired -- same run, same G, the two
    solvers applied to it -- so an APS misalignment or a bad G penalises both
    equally and cancels out of the difference.
    """
    from interacting_maps.network_dissertation import solve_poisson_exact

    root = _root(args, 'readout_fft_vs_dct')
    seqs = [s for s in SEGMENTS
            if not args.sequences or s[3] in args.sequences.split(',')]
    durations = [float(d) for d in args.durations.split(',')]
    base = dict(ADOPTED)

    path = os.path.join(root, 'readout.csv')
    f = open(path, 'w', newline='', encoding='utf-8')
    wr = csv.writer(f)
    wr.writerow(['sequence', 'duration_s', 'n_frames', 'variant',
                 'r_full', 'r_crop', 'low_freq', 'contrast'])
    print(f"{'sequence':<9} {'dur':>5} {'variant':<10} {'r_full':>7} {'r_crop':>7}",
          flush=True)
    print('-' * 44, flush=True)
    tally = collections.Counter()
    for ds, sid, t0, lab in seqs:
        for dur in durations:
            nf = int(round(dur / args.dt))
            rc = run_cfg(ds, sid, t0, args.model, args.dt, nf,
                      out_root=root, poisson='iterative', deltas=dict(base))
            npz = os.path.join(rc.output_dir, 'maps_final.npz')
            _run(E, rc, not args.no_frames, args.frame_stride)
            m = np.load(npz)
            aps = aps_frame(ds, t0 + nf * args.dt)
            if aps is None:
                print(f"  {lab}: no APS, skipped"); continue
            ref = np.log(aps.astype(np.float64) + 1.0)
            variants = {'iterative': m['I'],
                        'fft': solve_poisson_exact(m['G'], 'fft'),
                        'dct': solve_poisson_exact(m['G'], 'dct')}
            got = {}
            for name, I in variants.items():
                st = E._intensity_stats(I)
                r_full = abs_corr(I, ref)
                r_crop = abs_corr(crop_interior(I), crop_interior(ref))
                got[name] = r_crop
                wr.writerow([ds, dur, nf, name, round(r_full, 4),
                             round(r_crop, 4), round(st['low_freq_share'], 4),
                             round(st['contrast'], 4)])
                print(f"{lab:<9} {dur:5.2f} {name:<10} {r_full:7.3f} {r_crop:7.3f}",
                      flush=True)
            tally['fft' if got['fft'] > got['dct'] else 'dct'] += 1
            f.flush()
    f.close()
    print(f"\npaired wins on r_crop:  fft {tally['fft']}   dct {tally['dct']}")
    print(f"-> {path}")


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
    from interacting_maps.data_loader import load_events_fast, undistort_events, CameraCalibration
    from interacting_maps.angular_velocity import CMaxAngularVelocity

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
        keep = (0, nf // 2, nf - 1)                     # frames whose IWE is saved
        series = {'cmax': [], 'gyro': [], 'ref': []}
        iwes = []
        t = time.time()
        for k in range(nf):
            t_lo, t_hi = t0 + k * args.dt, t0 + (k + 1) * args.dt
            w_ref, ref_src = E.get_reference_omega(gt, imu, t_lo, t_hi, om)
            win = ev[(ev[:, 0] >= t_lo) & (ev[:, 0] < t_hi)]
            if len(win) > 10:
                w_prev = est.estimate(win, t_ref=0.5 * (t_lo + t_hi),
                                      omega_init=w_prev)
            w_gyro = E.get_gyro_for_frame(imu, t_lo, t_hi)
            acc['cmax'].append(E.compute_metrics(w_prev, w_ref))
            acc['gyro'].append(E.compute_metrics(w_gyro, w_ref))
            for src, w in (('cmax', w_prev), ('gyro', w_gyro), ('ref', w_ref)):
                series[src].append(w)
            if k in keep:
                iwes.append(est.last_iwe if len(win) > 10 else np.zeros((180, 240)))
        secs = time.time() - t
        np.savez_compressed(path.replace('.csv', f'_{lab}.npz'), frames=keep,
                            iwe=np.array(iwes), ref_src=ref_src,
                            **{s: np.array(v) for s, v in series.items()})
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


# ------------------------------------------------------- runs for fig_beta_speed
def cmd_beta(E, args):
    """Runs behind fig_beta_speed: every real-sequence segment, three anchor settings."""
    nf = int(round(args.duration / args.dt))
    runs = [(seg, cfg) for seg in BETA_SEGMENTS for cfg in BETA_CONFIGS]
    if args.index >= 0:
        runs = [runs[args.index]]
    for (ds, sid, t0), (model, d_fr, d_anc) in runs:
        rc = run_cfg(ds, sid, t0, model, args.dt, nf, delta_FR=d_fr, delta_IMU=d_anc)
        s, secs = _run(E, rc, not args.no_frames, args.frame_stride)
        print(f"  {ds}/{sid} {model} dFR={d_fr} anchor={d_anc}: "
              f"beta={s['mean_beta']:.3f} ({secs:.0f}s)", flush=True)


COMMANDS = {'window': cmd_window, 'main': cmd_main,
            'converge': cmd_converge, 'baseline': cmd_baseline,
            'sweep': cmd_sweep, 'poisson': cmd_poisson,
            'readout': cmd_readout, 'recon': cmd_recon, 'beta': cmd_beta}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--what', required=True, choices=sorted(COMMANDS))
    ap.add_argument('--dt', type=float, default=0.02, help='frame_duration (s)')
    ap.add_argument('--duration', type=float, default=3.0, help='track length (s)')
    ap.add_argument('--model', default='thesis_cmax')
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
    ap.add_argument('--out', default='', help='sweep/main: output csv name')
    ap.add_argument('--models', default='',
                    help='main: comma-separated subset of MODELS to run')
    ap.add_argument('--durations', default='0.4,0.8,1.2',
                    help='readout: track lengths to score at (s)')
    ap.add_argument('--name', default='',
                    help='experiment name; runs go to experiments/<name>/ '
                         'instead of results/ (searches only, not --what main)')
    ap.add_argument('--modes', default='iterative,fft,dct',
                    help='poisson: which in-loop solvers to include')
    ap.add_argument('--tag', default='', help='poisson: suffix for the figure name')
    ap.add_argument('--index', type=int, default=-1,
                    help='beta: run only entry N of BETA_SEGMENTS x BETA_CONFIGS (0-20)')
    args = ap.parse_args()
    sys.argv = [sys.argv[0]]              # evaluation.py parses argv
    import evaluation as E
    COMMANDS[args.what](E, args)


if __name__ == '__main__':
    main()
