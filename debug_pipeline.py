"""
debug_pipeline.py — shut down individual pieces of the Interacting Maps network
to localize where the problem is (component ablation).

The message passing is a set of coupled COSTS. Each one can be switched off:

  ofce     Cost_OFCE       V + F·G = 0   — the only term that reads events into F & G
  spatial  Cost_Spatial    G = ∇I        — gradient <-> intensity (reconstruction) link
  kin      Cost_Kinematics F = C·R       — rigid-rotation constraint (couples F and R)
  anchor   Cost_IMU        R -> ω_ext    — external anchor, fed the IMU gyro OR CMax

Turn pieces off and watch which output degrades — that localizes the fault, e.g.:
  - drop `anchor`  -> pure vision: R drifts / wrong scale (the β-gauge ambiguity)
  - drop `kin`     -> R decouples from the flow field
  - drop `ofce`    -> events stop driving F/G; only the anchor + geometry remain
  - drop `spatial` -> I never reconstructs (G <-> I broken)

Modes
-----
  default   leave-one-out SWEEP: baseline (all on) + each piece off, tabulated.
  --off …   run one custom config (e.g. --off ofce,anchor).

The anchor source is set by --model:
  thesis      = no anchor (pure vision)
  thesis_imu  = real gyro
  thesis_cmax = CMax estimate (sensor-free)
Shut the anchor off regardless via --off anchor.

Run:
  python debug_pipeline.py --dataset poster_rotation --segment seg_C
  python debug_pipeline.py --dataset dynamic_rotation --segment seg_A --model thesis_cmax
  python debug_pipeline.py --dataset poster_rotation --segment seg_C --off ofce
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from config import DATASET_SEGMENTS
from data_loader import (CameraCalibration, EventFrameSequence,
                         load_events_fast, undistort_events)
from evaluation import (RunConfig, make_network, load_imu, get_gyro_for_frame,
                        gt_omega_body, compute_metrics)
from interacting_maps.network_dissertation import (Cost_OFCE, Cost_Spatial,
                                                   Cost_Kinematics)
from cmax import CMaxAngularVelocity

RAD = 180.0 / np.pi
OUT = 'results/debug'

# component name -> cost class to strip from net.costs ('anchor' handled separately)
PIECES = {'ofce': Cost_OFCE, 'spatial': Cost_Spatial, 'kin': Cost_Kinematics}
ALL_PIECES = ['ofce', 'spatial', 'kin', 'anchor']


def run_ablation(rc, frames, imu, gt, H, W, fx, fy, cx, cy,
                 off, anchor_src, ev_undist, cmax_est):
    """Run tracking with the listed pieces disabled; return a metrics dict.

    off        : set/list of pieces to switch off (subset of ALL_PIECES)
    anchor_src : 'imu' | 'cmax' | 'none' — what feeds Cost_IMU
    """
    net = make_network(rc, H, W, fx, fy, cx, cy)
    drop = tuple(PIECES[p] for p in off if p in PIECES)
    if drop:
        net.costs = [c for c in net.costs if not isinstance(c, drop)]

    anchor_on = ('anchor' not in off) and (anchor_src != 'none')
    dt = rc.frame_duration
    prev = np.zeros(3)
    errs, dirs, ratios = [], [], []
    V_last = None
    for k, (V, _t) in enumerate(frames):
        t_lo = rc.t_start + k * dt
        t_hi = t_lo + dt
        if anchor_on:
            if anchor_src == 'cmax':
                m = (ev_undist[:, 0] >= t_lo) & (ev_undist[:, 0] < t_hi)
                wa = cmax_est.estimate(ev_undist[m], t_ref=0.5 * (t_lo + t_hi),
                                       omega_init=prev)
                prev = wa.copy()
            else:  # imu
                wa = get_gyro_for_frame(imu, t_lo, t_hi)
            net.step(V, n_iters=rc.n_iters, omega_imu=wa)
        else:
            net.step(V, n_iters=rc.n_iters)

        w_est = net.R / dt
        w_gt = gt_omega_body(gt, t_lo, t_hi)
        e, de, _ = compute_metrics(w_est, w_gt)     # scored vs GT (anchor-independent)
        errs.append(e); dirs.append(de)
        ratios.append(np.linalg.norm(w_est) / (np.linalg.norm(w_gt) + 1e-9))
        V_last = V

    return {
        'omega_err': float(np.mean(errs)),
        'dir_err': float(np.mean(dirs)),
        'mag_ratio': float(np.median(ratios)),         # |ω_est|/|ω_gt|; <1 = underestimate
        'res_VFG': float(net.residual_VFG(V_last)),     # |V + F·G| (flow-brightness)
        'res_GI': float(net.residual_GI()),             # |G - ∇I| (reconstruction)
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='poster_rotation')
    ap.add_argument('--segment', default='seg_C')
    ap.add_argument('--model', default='thesis_imu',
                    choices=['thesis', 'thesis_imu', 'thesis_cmax'])
    ap.add_argument('--n_frames', type=int, default=25)
    ap.add_argument('--off', default='',
                    help='comma list of pieces to switch off: ofce,spatial,kin,anchor')
    args = ap.parse_args()

    seg = next(s for s in DATASET_SEGMENTS[args.dataset] if s['id'] == args.segment)
    rc = RunConfig(dataset=args.dataset, model=args.model, segment=args.segment,
                   n_frames=args.n_frames)

    # anchor source is determined by the model
    anchor_src = {'thesis': 'none', 'thesis_imu': 'imu',
                  'thesis_cmax': 'cmax'}[args.model]

    seq = EventFrameSequence(rc.paths['events'], rc.paths['calib'],
                             frame_duration=rc.frame_duration, t_start=rc.t_start,
                             n_frames=rc.n_frames, clip_value=10.0,
                             undistort=rc.undistort_at_event_level,
                             sensor_size=rc.sensor_size)
    frames = list(seq)
    H, W = seq.H, seq.W
    fx, fy, cx, cy = seq.calib.fx, seq.calib.fy, seq.calib.cx, seq.calib.cy
    imu = load_imu(rc.paths['imu'])
    gt = np.loadtxt(rc.paths['groundtruth'], dtype=np.float64)

    # CMax plumbing only if the anchor uses it
    ev_undist = cmax_est = None
    if anchor_src == 'cmax':
        ev_undist = undistort_events(
            load_events_fast(rc.paths['events'], t_start=rc.t_start,
                             duration=rc.n_frames * rc.frame_duration + 0.1),
            CameraCalibration(rc.paths['calib']))
        cmax_est = CMaxAngularVelocity(H, W, fx, fy, cx, cy, use_polarity=True)

    print(f"\n{'='*78}\nABLATION — {args.dataset}/{args.segment}, model={args.model}, "
          f"anchor={anchor_src}, {len(frames)} frames\n"
          f"scored vs GT (anchor-independent).  ω underestimate shows as |ω|ratio<1 (β)\n{'='*78}")

    custom = [p.strip() for p in args.off.split(',') if p.strip()]
    bad = [p for p in custom if p not in ALL_PIECES]
    if bad:
        ap.error(f"unknown piece(s) {bad}; choose from {ALL_PIECES}")

    if custom:
        configs = [('off: ' + ','.join(custom), set(custom))]
    else:  # default: leave-one-out sweep
        configs = [('baseline (all on)', set())] + [(f'-{p}', {p}) for p in ALL_PIECES]

    print(f"\n{'config':<20} {'ω_err°/s':>8} {'dir°':>6} {'|ω|ratio':>8} "
          f"{'res_VFG':>8} {'res_GI':>7}")
    print('-' * 62)
    rows = []
    for name, off in configs:
        r = run_ablation(rc, frames, imu, gt, H, W, fx, fy, cx, cy,
                         off, anchor_src, ev_undist, cmax_est)
        rows.append((name, r))
        print(f"{name:<20} {r['omega_err']:>8.2f} {r['dir_err']:>6.1f} "
              f"{r['mag_ratio']:>8.2f} {r['res_VFG']:>8.3f} {r['res_GI']:>7.3f}")

    # localize: which single removal changes ω_err the most (sweep only)
    if not custom and len(rows) > 1:
        base = rows[0][1]['omega_err']
        deltas = [(n, r['omega_err'] - base) for n, r in rows[1:]]
        worst = max(deltas, key=lambda x: x[1])
        print('-' * 62)
        print(f"baseline ω_err = {base:.2f}°/s. Removing a piece changes it by:")
        for n, d in deltas:
            print(f"   {n:<12} {d:+7.2f} °/s")
        print(f"=> most load-bearing piece here: {worst[0]} "
              f"({worst[1]:+.2f} °/s when removed)")

        # bar chart
        os.makedirs(OUT, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        names = [n for n, _ in rows]
        vals = [r['omega_err'] for _, r in rows]
        ax.bar(names, vals, color=['#4e79a7'] + ['#e15759'] * (len(rows) - 1))
        ax.set_ylabel('mean ω error vs GT (°/s)')
        ax.set_title(f'Component ablation — {args.dataset}/{args.segment} '
                     f'(model={args.model}, anchor={anchor_src})')
        ax.tick_params(axis='x', rotation=20)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, 'debug_ablation.png'), dpi=140)
        plt.close(fig)
        print(f"Figure: {OUT}/debug_ablation.png")


if __name__ == '__main__':
    main()
