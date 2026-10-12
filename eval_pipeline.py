"""
eval_pipeline.py — interactive mirror of evaluation.py's Experiment 2 (multi-frame
ω tracking), the core pipeline every other experiment builds on.

Each `# %%` cell is one atomic step of the pipeline (config → data → network →
per-frame loop → save → summary → plot), for stepping through and inspecting
state interactively. Only imports from core modules (config, run_config,
interacting_maps.*), never from evaluation.py or other scripts.

Kept in sync with eval_pipeline.ipynb (generated from this file's `# %%` cells;
edit this file, then regenerate the notebook).
"""
# %%
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from run_config import RunConfig, make_network, _try_load_gt_images
from interacting_maps.data_loader import EventFrameSequence
from interacting_maps.eval_io import (SCORE_AGAINST, GT_DIFF_HALFWIDTH, load_imu,
                                       get_gyro_for_frame, load_groundtruth,
                                       load_omega_gt, get_reference_omega)
from interacting_maps.metrics import compute_metrics

# %%
rc = RunConfig(dataset='poster_rotation', model='thesis_imu', segment='seg_A',
               out_root='experiments/eval_pipeline')   # never write into results/
print(rc)

# %%
seq = EventFrameSequence(
    rc.paths['events'], rc.paths['calib'],
    frame_duration=rc.frame_duration, t_start=rc.t_start,
    n_frames=rc.n_frames, clip_value=10.0,
    undistort=rc.undistort_at_event_level,
    sensor_size=rc.sensor_size,
)
H, W = seq.H, seq.W
fx, fy, cx, cy = seq.calib.fx, seq.calib.fy, seq.calib.cx, seq.calib.cy
print(f"H={H} W={W} fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

# %%
imu_data = load_imu(rc.paths['imu'])                   # model INPUT (gyro)
gt_data = load_groundtruth(rc.paths['groundtruth'])    # SCORING ref (Vicon)
om_data = load_omega_gt(rc.paths.get('omega_gt'))      # exact omega, if shipped

if SCORE_AGAINST == 'groundtruth' and om_data is not None:
    ref_src = 'omega_gt'
elif SCORE_AGAINST == 'groundtruth' and gt_data is not None:
    ref_src = 'groundtruth'
else:
    ref_src = 'imu'
print(f"ref_src = {ref_src}")

# %%
net = make_network(rc, H, W, fx, fy, cx, cy)
gt_images = _try_load_gt_images(rc)

# %%
rows = []
for k, (V, t_mid) in enumerate(seq):
    t_lo = rc.t_start + k * rc.frame_duration
    t_hi = t_lo + rc.frame_duration

    omega_imu = get_gyro_for_frame(imu_data, t_lo, t_hi)
    omega_ref, _ = get_reference_omega(gt_data, imu_data, t_lo, t_hi, om_data)

    if rc.use_thesis and rc.use_imu:
        net.step(V, n_iters=rc.n_iters, omega_imu=omega_imu)
    else:
        net.step(V, n_iters=rc.n_iters)

    omega_est = net.R / rc.frame_duration
    err, dir_err, beta = compute_metrics(omega_est, omega_ref)

    rows.append({
        'frame': k, 'time': t_mid,
        'est_wx': omega_est[0], 'est_wy': omega_est[1], 'est_wz': omega_est[2],
        'gt_wx': omega_ref[0], 'gt_wy': omega_ref[1], 'gt_wz': omega_ref[2],
        'imu_wx': omega_imu[0], 'imu_wy': omega_imu[1], 'imu_wz': omega_imu[2],
        'err_deg_s': err, 'dir_err_deg': dir_err, 'beta': beta,
    })

    if (k + 1) % 10 == 0 or k == 0:
        print(f"  Frame {k+1:4d}/{rc.n_frames} | "
              f"ω_est=({omega_est[0]:+.3f},{omega_est[1]:+.3f},{omega_est[2]:+.3f}) | "
              f"err={err:.1f}°/s | dir={dir_err:.1f}°")

# %%
os.makedirs(rc.output_dir, exist_ok=True)
csv_path = os.path.join(rc.output_dir, 'tracking.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"Saved: {csv_path}")

# %%
err_all = np.array([r['err_deg_s'] for r in rows])
dir_all = np.array([r['dir_err_deg'] for r in rows])
beta_all = np.array([r['beta'] for r in rows])
summary = {
    'mean_err_deg_s': float(np.mean(err_all)),
    'median_err_deg_s': float(np.median(err_all)),
    'mean_dir_err_deg': float(np.mean(dir_all)),
    'median_dir_err_deg': float(np.median(dir_all)),
    'mean_beta': float(np.mean(beta_all)),
    'std_beta': float(np.std(beta_all)),
}
summary

# %%
times = np.array([r['time'] for r in rows]) - rows[0]['time']
omega_est = np.array([[r['est_wx'], r['est_wy'], r['est_wz']] for r in rows])
omega_gt = np.array([[r['gt_wx'], r['gt_wy'], r['gt_wz']] for r in rows])

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
fig.suptitle(f'ω Tracking — {rc.dataset} ({rc.model}), scored vs {ref_src}')
labels, colors = ['ωx', 'ωy', 'ωz'], ['#4e79a7', '#f28e2b', '#e15759']
for i in range(3):
    axes[i].plot(times, omega_gt[:, i], 'k-', lw=1.0, alpha=0.8, label='GT')
    axes[i].plot(times, omega_est[:, i], color=colors[i], lw=1.5, label='Estimated')
    axes[i].set_ylabel(f'{labels[i]} (rad/s)')
    axes[i].legend(loc='upper right', fontsize=8)
    axes[i].grid(True, alpha=0.3)
axes[3].plot(times, err_all, 'r-', lw=1.0)
axes[3].set_ylabel('Error (°/s)')
axes[3].set_xlabel('Time (s)')
axes[3].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
