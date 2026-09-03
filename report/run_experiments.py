"""
run_experiments.py — every experiment behind the report, from committed code.

Run from the repository root:
    python report/run_experiments.py --what grid        # 2-D (dFR, dAnchor) grid
    python report/run_experiments.py --what window      # dt / n_frames choice
    python report/run_experiments.py --what main        # Table III, 5 models
    python report/run_experiments.py --what converge    # per-iteration figure data

Every run uses the reported segments (SEGMENTS below, = Table II) and carries
lens distortion in the C matrix (distortion_mode='C_full'). Nothing is looked up
in results/ -- each cell is computed -- because results/ mixes runs from
configurations that have since changed.

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


def _cfg(E, ds, sid, t0, model, dt, nf, **kw):
    seg = {'id': sid, 't_start': t0, 'frame_duration': dt, 'n_frames': nf,
           'initial_R': None, 'sensor_size': (180, 240)}
    return E.RunConfig(dataset=ds, model=model, segment=seg, n_frames=nf,
                       distortion_mode='C_full', **kw)


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
    path = os.path.join(OUTDIR, f'grid_dt{int(args.dt*1000)}ms.csv')
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
                              delta_FR=fr, delta_IMU=anc)
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


COMMANDS = {'window': cmd_window, 'grid': cmd_grid, 'main': cmd_main,
            'converge': cmd_converge}


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
                    help='converge: which sequences (labels from SEGMENTS)')
    args = ap.parse_args()
    sys.argv = [sys.argv[0]]              # evaluation.py parses argv
    import evaluation as E
    COMMANDS[args.what](E, args)


if __name__ == '__main__':
    main()
