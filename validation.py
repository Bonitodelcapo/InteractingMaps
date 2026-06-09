"""
Quantitative validation: estimated ω vs IMU gyro ω vs ground-truth ω.

Two reference sources
---------------------
1. groundtruth.txt  [timestamp  tx ty tz  qx qy qz qw]
   External motion capture (Vicon/OptiTrack) — absolute, drift-free pose.
   Angular velocity DERIVED by differencing successive quaternions.
   Advantage: drift-free.  Drawback: finite-difference noise over short windows.

2. imu.txt  [timestamp  ax ay az  gx gy gz]
   On-board IMU — gyroscope DIRECTLY measures ω in the camera body frame.
   Advantage: instantaneous, ~1 kHz, good short-term accuracy.
   Drawback: has sensor noise and a slow bias drift.

Both references measure the same quantity; showing them together reveals
which oscillations are real motion vs. measurement noise.

Angular velocity in camera body frame (what the network estimates)
------------------------------------------------------------------
Given two successive GT poses R1=R_wc(t1) and R2=R_wc(t2):

    dR_body = R1.T @ R2        ← right-invariant (body frame)
    dR_body ≈ expm([ω_body × Δt])  →  extract axis-angle  →  ω_body

Note: R2 @ R1.T gives world-frame angular velocity — wrong here.

Run:
    python validation.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import DATASET_CONFIGS, THESIS_PARAMS, COOK_PARAMS, ITERS_PER_FRAME, F_DECAY, G_DECAY, get_dataset_paths
from data_loader import EventFrameSequence
from interacting_maps.network import InteractingMaps
from interacting_maps.network_dissertation import InteractingMapsThesis

# ---------------------------------------------------------------------------
# Configuration  (change DATASET here to switch datasets)
# ---------------------------------------------------------------------------

DATASET           = 'shapes_rotation'
USE_THESIS_VERSION = True

cfg   = DATASET_CONFIGS[DATASET]
paths = get_dataset_paths(DATASET)

T_START        = cfg['t_start']
FRAME_DURATION = cfg['frame_duration']
N_FRAMES       = cfg['n_frames']
initial_R      = cfg['initial_R']

print(f"Validation: dataset='{DATASET}',  t_start={T_START:.3f}s,  "
      f"{N_FRAMES} frames × {FRAME_DURATION*1000:.0f}ms")

# ---------------------------------------------------------------------------
# Ground-truth helpers
# ---------------------------------------------------------------------------

def load_groundtruth(path: str) -> np.ndarray:
    """
    Load groundtruth.txt.
    Each row: [timestamp, tx, ty, tz, qx, qy, qz, qw]
    Returns (N, 8) float64.
    """
    data = np.loadtxt(path, dtype=np.float64)
    print(f"  GT:  {len(data)} poses,  "
          f"t = [{data[0,0]:.3f}, {data[-1,0]:.3f}] s")
    return data


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """
    Quaternion (qx, qy, qz, qw) → 3×3 rotation matrix R_wc
    (camera-to-world, active rotation convention).
    """
    qx, qy, qz, qw = q / np.linalg.norm(q)   # normalise for safety
    return np.array([
        [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),  2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw),  1 - 2*(qx**2 + qz**2),  2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),      2*(qy*qz + qx*qw),  1 - 2*(qx**2 + qy**2)],
    ])


def rotmat_to_axisangle(R: np.ndarray):
    """
    Extract axis-angle (angle ≥ 0, axis unit-vector) from a rotation matrix.
    Returns (angle_rad, axis_vec3).
    """
    cos_a = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_a)
    if abs(angle) < 1e-10:
        return 0.0, np.array([0.0, 0.0, 1.0])
    skew = (R - R.T) / (2.0 * np.sin(angle) + 1e-15)
    axis = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    return angle, axis


def gt_omega_body(gt_data: np.ndarray, t_mid: float, dt: float) -> np.ndarray:
    """
    Compute ground-truth angular velocity in the CAMERA BODY FRAME.

    Finds the two GT poses that bracket [t_mid - dt/2, t_mid + dt/2],
    computes the incremental rotation in the camera's own frame, and
    divides by the actual time difference.

    Returns
    -------
    omega : (3,) rad/s  — angular velocity in camera body frame
    """
    t_lo = t_mid - dt / 2.0
    t_hi = t_mid + dt / 2.0

    idx1 = int(np.argmin(np.abs(gt_data[:, 0] - t_lo)))
    idx2 = int(np.argmin(np.abs(gt_data[:, 0] - t_hi)))

    if idx1 == idx2:
        idx2 = min(idx1 + 1, len(gt_data) - 1)
    if idx1 == idx2:
        return np.zeros(3)

    actual_dt = gt_data[idx2, 0] - gt_data[idx1, 0]
    if abs(actual_dt) < 1e-10:
        return np.zeros(3)

    R1 = quat_to_rotmat(gt_data[idx1, 4:8])   # R_wc at t1
    R2 = quat_to_rotmat(gt_data[idx2, 4:8])   # R_wc at t2

    # Body-frame incremental rotation: R1.T @ R2
    # (NOT R2 @ R1.T, which would give world-frame angular velocity)
    dR_body = R1.T @ R2

    angle, axis = rotmat_to_axisangle(dR_body)
    return axis * angle / actual_dt            # rad/s in camera body frame


# ---------------------------------------------------------------------------
# IMU helpers
# ---------------------------------------------------------------------------

def load_imu(path: str) -> np.ndarray:
    """
    Load imu.txt.
    Each row: [timestamp, ax, ay, az, gx, gy, gz]
    Returns (N, 7) float64.
    """
    data = np.loadtxt(path, dtype=np.float64)
    print(f"  IMU: {len(data)} samples,  "
          f"t = [{data[0,0]:.3f}, {data[-1,0]:.3f}] s")
    return data


def imu_omega(imu_data: np.ndarray, t_lo: float, t_hi: float) -> np.ndarray:
    """
    Average gyroscope readings (gx, gy, gz) within [t_lo, t_hi].
    Falls back to nearest sample if the window is empty.
    Returns (3,) rad/s in camera body frame.
    """
    mask = (imu_data[:, 0] >= t_lo) & (imu_data[:, 0] < t_hi)
    if np.sum(mask) == 0:
        idx = int(np.argmin(np.abs(imu_data[:, 0] - (t_lo + t_hi) / 2.0)))
        return imu_data[idx, 4:7].copy()
    return np.mean(imu_data[mask, 4:7], axis=0)


# ---------------------------------------------------------------------------
# Network runner
# ---------------------------------------------------------------------------

def run_validation(gt_data: np.ndarray, imu_data: np.ndarray | None = None):
    """Run the network on N_FRAMES event packets and collect results."""

    seq = EventFrameSequence(
        paths['events'], paths['calib'],
        frame_duration=FRAME_DURATION,
        t_start=T_START,
        n_frames=N_FRAMES,
        clip_value=10.0,
    )
    fx, fy = seq.calib.fx, seq.calib.fy
    cx, cy = seq.calib.cx, seq.calib.cy
    H, W   = seq.H, seq.W

    if USE_THESIS_VERSION:
        net = InteractingMapsThesis(H=H, W=W, fx=fx, fy=fy, cx=cx, cy=cy,
                                   **THESIS_PARAMS)
        net.initialize_from_rotation(initial_R)
        label = 'Thesis (Martel 2019)'
    else:
        net = InteractingMaps(H=H, W=W, fx=fx, fy=fy, cx=cx, cy=cy,
                              **COOK_PARAMS)
        net.reset(scale=0.01)
        net.R = initial_R.copy()
        label = 'Cook et al. 2011'

    have_imu = imu_data is not None
    print(f"\n  Network: {label},  {ITERS_PER_FRAME} iters/frame")
    hdr = (f"\n{'Frame':>5}  {'t_mid':>7}  "
           f"{'Est ωx':>8} {'Est ωy':>8} {'Est ωz':>8}  |  "
           f"{'GT  ωx':>8} {'GT  ωy':>8} {'GT  ωz':>8}  |  ")
    if have_imu:
        hdr += f"{'IMU ωx':>8} {'IMU ωy':>8} {'IMU ωz':>8}  |  "
    hdr += f"{'Err°/s':>7}"
    print(hdr)
    print("-" * (85 + (30 if have_imu else 0)))

    times          = []
    omega_est_all  = []
    omega_gt_all   = []
    omega_imu_all  = []

    for k, (V, t_mid) in enumerate(seq):
        net.step(V, n_iters=ITERS_PER_FRAME, f_decay=F_DECAY, g_decay=G_DECAY)

        omega_est = net.R / FRAME_DURATION
        omega_ref = gt_omega_body(gt_data, t_mid, FRAME_DURATION)

        t_lo = T_START + k * FRAME_DURATION
        t_hi = t_lo + FRAME_DURATION
        omega_imu_k = imu_omega(imu_data, t_lo, t_hi) if have_imu else np.zeros(3)

        times.append(t_mid)
        omega_est_all.append(omega_est.copy())
        omega_gt_all.append(omega_ref.copy())
        omega_imu_all.append(omega_imu_k.copy())

        err_gt  = np.degrees(np.linalg.norm(omega_est - omega_ref))
        row = (f"{k+1:5d}  {t_mid:7.3f}  "
               f"{omega_est[0]:+8.4f} {omega_est[1]:+8.4f} {omega_est[2]:+8.4f}  |  "
               f"{omega_ref[0]:+8.4f} {omega_ref[1]:+8.4f} {omega_ref[2]:+8.4f}  |  ")
        if have_imu:
            row += (f"{omega_imu_k[0]:+8.4f} {omega_imu_k[1]:+8.4f} {omega_imu_k[2]:+8.4f}  |  ")
        row += f"{err_gt:7.2f}"
        print(row)

    return (np.array(times),
            np.array(omega_est_all),
            np.array(omega_gt_all),
            np.array(omega_imu_all) if have_imu else None,
            label)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(times, omega_est, omega_gt, omega_imu, label):
    """4-panel figure: one row per ω component + angular error vs both references."""

    fig = plt.figure(figsize=(13, 10))
    fig.suptitle(
        f'Angular Velocity Validation — {label}\n'
        f'Dataset: {DATASET}   '
        f'{FRAME_DURATION*1000:.0f} ms frames   '
        f'{ITERS_PER_FRAME} iters/frame',
        fontsize=11,
    )

    have_imu = omega_imu is not None
    gs     = GridSpec(4, 1, figure=fig, hspace=0.45)
    ylabels = ['ω_x  (rad/s)', 'ω_y  (rad/s)', 'ω_z  (rad/s)']
    colors  = ['#4e79a7', '#f28e2b', '#e15759']

    for i in range(3):
        ax = fig.add_subplot(gs[i])
        # GT: external motion capture, derived by quaternion differencing
        ax.plot(times, omega_gt[:, i], 'k-', lw=1.5, alpha=0.8,
                label='GT (quaternion diff, body frame)')
        # IMU: direct gyroscope measurement in camera body frame
        if have_imu:
            ax.plot(times, omega_imu[:, i], color='#59a14f', lw=1.5, alpha=0.8,
                    label='IMU gyroscope (direct measurement)')
        # Network estimate
        ax.plot(times, omega_est[:, i], color=colors[i], lw=2.0, ls='--',
                label='Network estimate')
        ax.set_ylabel(ylabels[i], fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.axhline(0, color='k', lw=0.4)
        if i == 0:
            ax.legend(fontsize=7, loc='upper right')

    # Angular error panel — compare estimate vs both references
    ax_err = fig.add_subplot(gs[3])
    err_gt = np.degrees(np.linalg.norm(omega_est - omega_gt, axis=1))
    ax_err.plot(times, err_gt, color='purple', lw=2.0, label='Error vs GT')
    ax_err.fill_between(times, 0, err_gt, alpha=0.12, color='purple')
    if have_imu:
        err_imu = np.degrees(np.linalg.norm(omega_est - omega_imu, axis=1))
        ax_err.plot(times, err_imu, color='#59a14f', lw=1.5, ls='--',
                    label='Error vs IMU')

    ax_err.set_ylabel('Angular error  (°/s)', fontsize=8)
    ax_err.set_xlabel('Time  (s)', fontsize=8)
    ax_err.grid(True, alpha=0.25)
    ax_err.set_ylim(bottom=0)

    mean_e_gt   = np.mean(err_gt)
    median_e_gt = np.median(err_gt)
    ax_err.axhline(mean_e_gt, color='purple', ls=':', lw=1.2,
                   label=f'Mean vs GT  {mean_e_gt:.1f}°/s')
    if have_imu:
        mean_e_imu = np.mean(err_imu)
        ax_err.axhline(mean_e_imu, color='#59a14f', ls=':', lw=1.2,
                       label=f'Mean vs IMU {mean_e_imu:.1f}°/s')
    ax_err.legend(fontsize=7)

    # Print summary
    print(f'\n{"="*55}')
    print('VALIDATION SUMMARY')
    print(f'{"="*55}')
    print(f'  Mean   angular error vs GT  : {mean_e_gt:.2f} °/s')
    print(f'  Median angular error vs GT  : {median_e_gt:.2f} °/s')
    if have_imu:
        print(f'  Mean   angular error vs IMU : {mean_e_imu:.2f} °/s')
        print(f'  Mean   |ω_IMU|  : {np.mean(np.linalg.norm(omega_imu, axis=1)):.4f} rad/s')
    print(f'  Mean   |ω_est|  : {np.mean(np.linalg.norm(omega_est, axis=1)):.4f} rad/s')
    print(f'  Mean   |ω_GT|   : {np.mean(np.linalg.norm(omega_gt,  axis=1)):.4f} rad/s')
    print(f'  Expected ω      : {cfg["expected_omega"]}  rad/s')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if not os.path.isfile(paths['events']):
        raise FileNotFoundError(
            f"events.txt not found at {paths['events']}\n"
            f"Download shapes_rotation.zip from "
            f"http://rpg.ifi.uzh.ch/datasets/davis/shapes_rotation.zip"
        )
    if not os.path.isfile(paths['groundtruth']):
        raise FileNotFoundError(
            f"groundtruth.txt not found at {paths['groundtruth']}\n"
            f"Expected: {paths['groundtruth']}"
        )

    gt_data  = load_groundtruth(paths['groundtruth'])
    imu_data = load_imu(paths['imu']) if os.path.isfile(paths['imu']) else None

    times, omega_est, omega_gt, omega_imu, label = run_validation(gt_data, imu_data)
    plot_results(times, omega_est, omega_gt, omega_imu, label)
