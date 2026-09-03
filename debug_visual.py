"""
debug_visual.py — visual + omega-decomposition debugger across models.

Produces, for one dataset/segment:
  results/debug/visual_<ds>_<seg>_montage.png
      rows = models, cols = Events V | Flow F (HSV) | Estimated I | GT APS
  results/debug/visual_<ds>_<seg>_omega.png
      per-axis omega over frames: GT, IMU, CMax(standalone), and each model
      — shows WHERE the estimate diverges (from the anchor? which axis?).
Plus a per-axis error table.

Run:
  python debug_visual.py --dataset bycicle_sinthetic --segment seg_A \
      --models thesis_cmax,thesis_cmax_v2 --cmax_lr 5e-6 --n_frames 40
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
                        gt_omega_body, compute_metrics, _try_load_gt_images,
                        flow_to_rgb, normalise_robust)
from cmax import CMaxAngularVelocity

RAD = 180.0 / np.pi
OUT = 'results/debug'
COLORS = {'thesis': 'C3', 'cook': 'C1', 'thesis_imu': 'C5',
          'thesis_cmax': 'C0', 'thesis_cmax_v2': 'm'}


def run_model(model, ds, seg_id, frames, imu, H, W, fx, fy, cx, cy, snap_k,
              ev_undist, w_cmax, cmax_lr):
    """Run tracking for one model; return (omega_est [N,3], snapshot dict)."""
    rc = RunConfig(dataset=ds, model=model, segment=seg_id, n_frames=len(frames))
    if model == 'thesis_cmax_v2':
        rc.cmax_lr = cmax_lr
    net = make_network(rc, H, W, fx, fy, cx, cy)
    dt = rc.frame_duration
    R_hist, snap = [], {}
    for k, (V, _t) in enumerate(frames):
        t_lo = rc.t_start + k * dt
        t_hi = t_lo + dt
        if model == 'thesis_imu':
            net.step(V, n_iters=rc.n_iters, omega_imu=get_gyro_for_frame(imu, t_lo, t_hi))
        elif model == 'thesis_cmax':
            net.step(V, n_iters=rc.n_iters, omega_imu=w_cmax[k])
        elif model == 'thesis_cmax_v2':
            m = (ev_undist[:, 0] >= t_lo) & (ev_undist[:, 0] < t_hi)
            net.step(V, n_iters=rc.n_iters, events=ev_undist[m])
        else:  # cook, thesis (pure vision)
            net.step(V, n_iters=rc.n_iters)
        R_hist.append(net.R.copy())
        if k == snap_k:
            I = net.I if net.I.shape == (H, W) else net.I[:H, :W]
            snap = dict(V=V.copy(), F=net.F.copy(), I=I.copy())
    return np.array(R_hist) / dt, snap


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='poster_rotation')
    ap.add_argument('--segment', default='seg_C')
    ap.add_argument('--n_frames', type=int, default=25)
    ap.add_argument('--frame', type=int, default=None, help='snapshot frame (default last)')
    ap.add_argument('--models', default='thesis,cook,thesis_cmax',
                    help='comma list: thesis,cook,thesis_imu,thesis_cmax,thesis_cmax_v2')
    ap.add_argument('--cmax_lr', type=float, default=5e-6, help='V2 ascent step')
    args = ap.parse_args()

    ds, seg_id = args.dataset, args.segment
    models = [m.strip() for m in args.models.split(',') if m.strip()]
    rc0 = RunConfig(dataset=ds, model='thesis_imu', segment=seg_id, n_frames=args.n_frames)
    seq = EventFrameSequence(rc0.paths['events'], rc0.paths['calib'],
                             frame_duration=rc0.frame_duration, t_start=rc0.t_start,
                             n_frames=rc0.n_frames, clip_value=10.0,
                             undistort=rc0.undistort_at_event_level,
                             sensor_size=rc0.sensor_size)
    frames = list(seq)
    H, W = seq.H, seq.W
    fx, fy, cx, cy = seq.calib.fx, seq.calib.fy, seq.calib.cx, seq.calib.cy
    snap_k = args.frame if args.frame is not None else len(frames) - 1
    t0, dt = rc0.t_start, rc0.frame_duration

    imu = load_imu(rc0.paths['imu'])
    gt = np.loadtxt(rc0.paths['groundtruth'], dtype=np.float64)
    gt_images = _try_load_gt_images(rc0)
    ev_undist = undistort_events(
        load_events_fast(rc0.paths['events'], t_start=t0, duration=rc0.n_frames * dt + 0.1),
        CameraCalibration(rc0.paths['calib']))

    # references per frame
    w_gt = np.array([gt_omega_body(gt, t0 + k * dt, t0 + (k + 1) * dt) for k in range(len(frames))])
    w_imu = np.array([get_gyro_for_frame(imu, t0 + k * dt, t0 + (k + 1) * dt) for k in range(len(frames))])
    # standalone CMax (also serves as the thesis_cmax anchor), warm-started
    cmax_est = CMaxAngularVelocity(H, W, fx, fy, cx, cy, use_polarity=True)
    w_cmax, prev = [], np.zeros(3)
    for k in range(len(frames)):
        m = (ev_undist[:, 0] >= t0 + k * dt) & (ev_undist[:, 0] < t0 + (k + 1) * dt)
        prev = cmax_est.estimate(ev_undist[m], t_ref=t0 + (k + 0.5) * dt, omega_init=prev)
        w_cmax.append(prev.copy())
    w_cmax = np.array(w_cmax)

    est, snaps = {}, {}
    for m in models:
        est[m], snaps[m] = run_model(m, ds, seg_id, frames, imu, H, W, fx, fy, cx, cy,
                                     snap_k, ev_undist, w_cmax, args.cmax_lr)

    os.makedirs(OUT, exist_ok=True)
    tag = f"{ds}_{seg_id}"

    # -------------------------------------------------- montage
    aps = None
    if gt_images:
        import cv2
        tf = t0 + (snap_k + 0.5) * dt
        _, p = min(gt_images, key=lambda x: abs(x[0] - tf))
        a = plt.imread(p); a = a.mean(-1) if a.ndim == 3 else a
        aps = cv2.resize(a.astype(float), (W, H))

    fig, ax = plt.subplots(len(models), 4, figsize=(14, 3.2 * len(models)), squeeze=False)
    cols = ['Events V', 'Flow F (HSV)', 'Estimated I', 'GT APS']
    for r, m in enumerate(models):
        s = snaps[m]
        ax[r, 0].imshow(s['V'], cmap='RdBu', vmin=-1, vmax=1)
        ax[r, 1].imshow(flow_to_rgb(s['F']))
        ax[r, 2].imshow(normalise_robust(s['I']), cmap='gray', vmin=0, vmax=1)
        if aps is not None:
            ax[r, 3].imshow(normalise_robust(aps), cmap='gray', vmin=0, vmax=1)
        else:
            ax[r, 3].text(0.5, 0.5, 'no APS', ha='center', va='center')
        ax[r, 0].set_ylabel(m, fontsize=11)
        for c in range(4):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(cols[c], fontsize=10)
    fig.suptitle(f'Visual debug — {tag}, frame {snap_k}', fontsize=12)
    fig.tight_layout()
    mfile = os.path.join(OUT, f'visual_{tag}_montage.png')
    fig.savefig(mfile, dpi=140); plt.close(fig)

    # -------------------------------------------------- omega decomposition
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    names = ['omega_x', 'omega_y', 'omega_z']
    x = np.arange(len(frames))
    for i in range(3):
        ax[i].plot(x, w_gt[:, i], 'k-', lw=2.2, label='GT')
        ax[i].plot(x, w_imu[:, i], color='0.6', lw=1.0, label='IMU')
        ax[i].plot(x, w_cmax[:, i], 'g--', lw=1.2, label='CMax')
        for m in models:
            ax[i].plot(x, est[m][:, i], color=COLORS.get(m, 'C7'), lw=1.4, label=f'net {m}')
        ax[i].set_ylabel(f'{names[i]} (rad/s)'); ax[i].grid(alpha=0.3)
        ax[i].legend(loc='upper right', fontsize=7, ncol=3)
    ax[-1].set_xlabel('frame')
    fig.suptitle(f'omega decomposition — {tag}  (cmax_lr={args.cmax_lr:g})', fontsize=12)
    fig.tight_layout()
    ofile = os.path.join(OUT, f'visual_{tag}_omega.png')
    fig.savefig(ofile, dpi=140); plt.close(fig)

    # -------------------------------------------------- stats
    print(f"\n{'='*66}\n{tag}  ({len(frames)} frames)  |omega_gt| median = "
          f"{np.median(np.linalg.norm(w_gt, axis=1))*RAD:.1f} deg/s\n{'='*66}")
    print(f"{'source':<18} {'RMSE_x':>7} {'RMSE_y':>7} {'RMSE_z':>7} {'|err|':>7} {'dir°':>6}")
    print('-' * 60)

    def stats(w):
        rmse = np.sqrt(np.mean((w - w_gt) ** 2, axis=0)) * RAD
        err = np.mean([compute_metrics(a, b)[0] for a, b in zip(w, w_gt)])
        dire = np.mean([compute_metrics(a, b)[1] for a, b in zip(w, w_gt)])
        return rmse, err, dire

    for label, w in [('IMU', w_imu), ('CMax', w_cmax)] + [(f'net {m}', est[m]) for m in models]:
        rmse, err, dire = stats(w)
        print(f"{label:<18} {rmse[0]:>7.1f} {rmse[1]:>7.1f} {rmse[2]:>7.1f} {err:>7.1f} {dire:>6.1f}")
    print(f"\nFigures:\n  {mfile}\n  {ofile}")


if __name__ == '__main__':
    main()
