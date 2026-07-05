"""
Validate estimated angular velocity against ground truth.

Data files:
  imu.txt:         timestamp ax ay az gx gy gz
  groundtruth.txt: timestamp px py pz qx qy qz qw

The network estimates R in rad/frame.
IMU gyroscope reports angular velocity in rad/s.
Conversion: R_network / frame_duration = omega_rad_s
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import DATASET_CONFIGS, THESIS_PARAMS, COOK_PARAMS, ITERS_PER_FRAME, get_dataset_paths, get_initial_R_from_imu
from data_loader import EventFrameSequence, CameraCalibration
from interacting_maps.network import InteractingMaps
from interacting_maps.network_dissertation import InteractingMapsThesis
from interacting_maps.camera import build_kinematic_matrix

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET = 'boxes_rotation'
USE_THESIS_VERSION = True

cfg = DATASET_CONFIGS[DATASET]
initial_R = cfg['initial_R']
if initial_R is None:
    initial_R = get_initial_R_from_imu(DATASET)
    print(f"Auto-initialized R from IMU: {initial_R}")
paths = get_dataset_paths(DATASET)

EVENTS_FILE = paths['events']
CALIB_FILE = paths['calib']
IMU_FILE = paths['imu']
GT_FILE = paths['groundtruth']

T_START = cfg['t_start']
FRAME_DURATION = cfg['frame_duration']
N_FRAMES = cfg['n_frames']

# ---------------------------------------------------------------------------
# Load IMU data
# ---------------------------------------------------------------------------

def load_imu(path: str) -> np.ndarray:
    """
    Load IMU data: timestamp ax ay az gx gy gz
    Returns (N, 7) array.
    """
    data = np.loadtxt(path, dtype=np.float64)
    print(f"Loaded {len(data)} IMU measurements")
    print(f"  Time range: {data[0,0]:.3f} – {data[-1,0]:.3f} s")
    return data


def get_gyro_for_frame(imu_data: np.ndarray, t_lo: float, t_hi: float) -> np.ndarray:
    """
    Average gyroscope readings within a time window.
    Returns (3,) angular velocity in rad/s.
    """
    mask = (imu_data[:, 0] >= t_lo) & (imu_data[:, 0] < t_hi)
    if np.sum(mask) == 0:
        # No IMU data in this window — interpolate from nearest
        idx = np.argmin(np.abs(imu_data[:, 0] - (t_lo + t_hi) / 2))
        return imu_data[idx, 4:7]
    return np.mean(imu_data[mask, 4:7], axis=0)


# ---------------------------------------------------------------------------
# Load Ground Truth
# ---------------------------------------------------------------------------

def load_groundtruth(path: str) -> np.ndarray:
    """
    Load ground truth: timestamp px py pz qx qy qz qw
    Returns (N, 8) array.
    """
    data = np.loadtxt(path, dtype=np.float64)
    print(f"Loaded {len(data)} ground truth poses")
    print(f"  Time range: {data[0,0]:.3f} – {data[-1,0]:.3f} s")
    return data


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (qx, qy, qz, qw) to 3x3 rotation matrix."""
    qx, qy, qz, qw = q
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ])
    return R


def rotation_matrix_to_angular_velocity(R1, R2, dt):
    """
    Compute angular velocity from two rotation matrices separated by dt.
    Uses: dR = R2 @ R1^T, then log map to get omega.
    """
    dR = R2 @ R1.T

    # Log map of SO(3): extract angular velocity
    cos_angle = (np.trace(dR) - 1) / 2
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    if abs(angle) < 1e-10:
        return np.zeros(3)

    # Axis from skew-symmetric part
    skew = (dR - dR.T) / (2 * np.sin(angle) + 1e-15)
    omega = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])  # (wx, wy, wz)
    omega *= angle / dt

    return omega


def get_gt_omega_for_frame(gt_data: np.ndarray, t_center: float, dt: float) -> np.ndarray:
    """
    Compute angular velocity from ground truth quaternions at t_center.
    Uses finite differences on the closest poses.
    """
    # Find two poses bracketing t_center
    t_lo = t_center - dt / 2
    t_hi = t_center + dt / 2

    idx_lo = np.argmin(np.abs(gt_data[:, 0] - t_lo))
    idx_hi = np.argmin(np.abs(gt_data[:, 0] - t_hi))

    if idx_lo == idx_hi:
        idx_hi = min(idx_lo + 1, len(gt_data) - 1)

    t1 = gt_data[idx_lo, 0]
    t2 = gt_data[idx_hi, 0]
    actual_dt = t2 - t1

    if abs(actual_dt) < 1e-10:
        return np.zeros(3)

    q1 = gt_data[idx_lo, 4:8]  # qx, qy, qz, qw
    q2 = gt_data[idx_hi, 4:8]

    R1 = quaternion_to_rotation_matrix(q1)
    R2 = quaternion_to_rotation_matrix(q2)

    omega = rotation_matrix_to_angular_velocity(R1, R2, actual_dt)
    return omega


# ---------------------------------------------------------------------------
# Run the network and collect estimates
# ---------------------------------------------------------------------------

def run_validation():
    seq = EventFrameSequence(
        EVENTS_FILE, CALIB_FILE,
        frame_duration=FRAME_DURATION,
        t_start=T_START,
        n_frames=N_FRAMES,
        clip_value=10.0,
    )
    fx, fy, cx, cy = seq.calib.fx, seq.calib.fy, seq.calib.cx, seq.calib.cy
    H, W = seq.H, seq.W

    imu_data = load_imu(IMU_FILE)
    gt_data = load_groundtruth(GT_FILE)

    if USE_THESIS_VERSION:
        net = InteractingMapsThesis(H=H, W=W, fx=fx, fy=fy, cx=cx, cy=cy, **THESIS_PARAMS)
        net.initialize_from_rotation(initial_R)
    else:
        net = InteractingMaps(H=H, W=W, fx=fx, fy=fy, cx=cx, cy=cy, **COOK_PARAMS)
        net.reset(scale=0.01)
        net.R = initial_R.copy()

    # Storage for results
    times = []
    R_estimated = []      # rad/frame
    omega_estimated = []  # rad/s (= R / dt)
    omega_imu = []        # rad/s from gyroscope
    omega_gt = []         # rad/s from ground truth

    print(f"\n{'='*70}")
    print(f"{'Frame':>5} | {'t':>7} | {'Est ωx':>8} {'Est ωy':>8} {'Est ωz':>8} | "
          f"{'IMU ωx':>8} {'IMU ωy':>8} {'IMU ωz':>8} | "
          f"{'Err°':>6}")
    print(f"{'-'*70}")

    for k, (V, t_mid) in enumerate(seq):
        # Run inference
        net.step(V, n_iters=ITERS_PER_FRAME)

        # Get estimated angular velocity in rad/s
        if USE_THESIS_VERSION:
            R_est = net.R.copy()
        else:
            R_est = net.R.copy()

        omega_est = R_est / FRAME_DURATION  # rad/frame → rad/s

        # Get IMU gyro for this frame
        t_lo = T_START + k * FRAME_DURATION
        t_hi = t_lo + FRAME_DURATION
        t_center = (t_lo + t_hi) / 2
        gyro = get_gyro_for_frame(imu_data, t_lo, t_hi)

        # Get ground truth angular velocity
        gt_omega = get_gt_omega_for_frame(gt_data, t_center, FRAME_DURATION)

        # Store
        times.append(t_center)
        R_estimated.append(R_est.copy())
        omega_estimated.append(omega_est.copy())
        omega_imu.append(gyro.copy())
        omega_gt.append(gt_omega.copy())

        # Angular error vs IMU (in degrees)
        err_vec = omega_est - gyro
        err_deg = np.linalg.norm(err_vec) * 180 / np.pi

        print(
            f"{k+1:5d} | {t_center:7.3f} | "
            f"{omega_est[0]:8.4f} {omega_est[1]:8.4f} {omega_est[2]:8.4f} | "
            f"{gyro[0]:8.4f} {gyro[1]:8.4f} {gyro[2]:8.4f} | "
            f"{err_deg:6.2f}°"
        )

    # Convert to arrays
    times = np.array(times)
    omega_estimated = np.array(omega_estimated)
    omega_imu = np.array(omega_imu)
    omega_gt = np.array(omega_gt)

    return times, omega_estimated, omega_imu, omega_gt


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(times, omega_est, omega_imu, omega_gt):
    """Plot estimated vs ground truth angular velocity."""

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Angular Velocity Validation\n'
                 f'{"Thesis" if USE_THESIS_VERSION else "Cook"} version, '
                 f'{FRAME_DURATION*1000:.0f}ms frames, {ITERS_PER_FRAME} iters/frame',
                 fontsize=12)

    gs = GridSpec(4, 1, figure=fig, hspace=0.4)

    labels = ['ω_x (rad/s)', 'ω_y (rad/s)', 'ω_z (rad/s)']
    colors = ['#4e79a7', '#f28e2b', '#e15759']

    for i in range(3):
        ax = fig.add_subplot(gs[i])
        ax.plot(times, omega_imu[:, i], 'k-', lw=1.5, alpha=0.7, label='IMU (gyro)')
        ax.plot(times, omega_gt[:, i], 'g--', lw=1.0, alpha=0.7, label='GT (quaternion diff)')
        ax.plot(times, omega_est[:, i], color=colors[i], lw=2.0, label='Estimated')
        ax.set_ylabel(labels[i])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.3)

    # Error plot
    ax_err = fig.add_subplot(gs[3])
    err_imu = np.linalg.norm(omega_est - omega_imu, axis=1) * 180 / np.pi
    err_gt = np.linalg.norm(omega_est - omega_gt, axis=1) * 180 / np.pi
    ax_err.plot(times, err_imu, 'b-', lw=1.5, label='Error vs IMU (°/s)')
    ax_err.plot(times, err_gt, 'g-', lw=1.5, label='Error vs GT (°/s)')
    ax_err.set_ylabel('Angular error (°/s)')
    ax_err.set_xlabel('Time (s)')
    ax_err.legend(fontsize=8)
    ax_err.grid(True, alpha=0.3)

    # Summary statistics
    mean_err_imu = np.mean(err_imu)
    mean_err_gt = np.mean(err_gt)
    median_err_imu = np.median(err_imu)

    print(f"\n{'='*50}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*50}")
    print(f"Mean angular error vs IMU:   {mean_err_imu:.2f} °/s")
    print(f"Median angular error vs IMU: {median_err_imu:.2f} °/s")
    print(f"Mean angular error vs GT:    {mean_err_gt:.2f} °/s")
    print(f"")
    print(f"Mean estimated |ω|: {np.mean(np.linalg.norm(omega_est, axis=1)):.4f} rad/s")
    print(f"Mean IMU |ω|:       {np.mean(np.linalg.norm(omega_imu, axis=1)):.4f} rad/s")
    print(f"Mean GT |ω|:        {np.mean(np.linalg.norm(omega_gt, axis=1)):.4f} rad/s")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    #plt.savefig('validation_result.png', dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    times, omega_est, omega_imu, omega_gt = run_validation()
    plot_results(times, omega_est, omega_imu, omega_gt)