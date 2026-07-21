"""
Step 1a verification — standalone CMax angular-velocity front-end vs ground truth.

The load-bearing question for the whole V1 plan:
    Can CMax, from events alone (no IMU), recover ω well enough to REPLACE the IMU?

Per frame (fixed-time window): warp events, maximize IWE variance (warm-started
from the previous frame), compare ω_cmax against:
  - GT   : groundtruth.txt quaternion differencing (R1ᵀR2, body frame) — the truth
  - IMU  : imu.txt gyro — the sensor we are trying to replace

Run:
    python test_cmax_frontend.py
"""

import sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from config import get_dataset_paths
from data_loader import CameraCalibration, load_events_fast, undistort_events
from evaluation import gt_omega_body, get_gyro_for_frame, load_imu, compute_metrics
from cmax import CMaxAngularVelocity


# ---------------------------------------------------------------------------
# Config (poster seg_C: ω_x-dominant, |ω|≈1.12 rad/s)
# ---------------------------------------------------------------------------
DATASET   = 'poster_rotation'
T_START   = 8.816
DT        = 0.020
N_FRAMES  = 25
SENSOR    = (180, 240)   # H, W


def main():
    paths = get_dataset_paths(DATASET)
    calib = CameraCalibration(paths['calib'])
    H, W  = SENSOR
    fx, fy, cx, cy = calib.fx, calib.fy, calib.cx, calib.cy

    # Load raw events for the whole segment, then undistort ONCE (pipeline-consistent)
    dur = N_FRAMES * DT + 0.1
    ev = load_events_fast(paths['events'], t_start=T_START, duration=dur)
    ev = undistort_events(ev, calib)
    print(f"Loaded {len(ev)} events over {dur:.2f}s  ({DATASET}, t_start={T_START})")

    gt  = np.loadtxt(paths['groundtruth'], dtype=np.float64)
    imu = load_imu(paths['imu'])

    est = CMaxAngularVelocity(H, W, fx, fy, cx, cy,
                              use_polarity=True, blur_sigma=1.0)

    omega_prev = np.zeros(3)   # warm start (first frame from zero)

    print(f"\n{'k':>3} {'n_ev':>7} | "
          f"{'CMax ωx':>8} {'ωy':>8} {'ωz':>8} | "
          f"{'GT ωx':>8} {'ωy':>8} {'ωz':>8} | "
          f"{'err°/s':>7} {'dir°':>6} | {'err_imu':>7}")
    print("-" * 96)

    errs, dirs, errs_imu = [], [], []
    for k in range(N_FRAMES):
        t_lo = T_START + k * DT
        t_hi = t_lo + DT
        m = (ev[:, 0] >= t_lo) & (ev[:, 0] < t_hi)
        win = ev[m]
        t_ref = 0.5 * (t_lo + t_hi)

        omega_cmax = est.estimate(win, t_ref=t_ref, omega_init=omega_prev)
        omega_prev = omega_cmax.copy()

        omega_gt  = gt_omega_body(gt, t_lo, t_hi)
        omega_imu = get_gyro_for_frame(imu, t_lo, t_hi)

        err, dir_err, _ = compute_metrics(omega_cmax, omega_gt)
        err_imu, _, _   = compute_metrics(omega_imu, omega_gt)
        errs.append(err); dirs.append(dir_err); errs_imu.append(err_imu)

        print(f"{k:3d} {len(win):7d} | "
              f"{omega_cmax[0]:+8.3f} {omega_cmax[1]:+8.3f} {omega_cmax[2]:+8.3f} | "
              f"{omega_gt[0]:+8.3f} {omega_gt[1]:+8.3f} {omega_gt[2]:+8.3f} | "
              f"{err:7.2f} {dir_err:6.1f} | {err_imu:7.2f}")

    errs = np.array(errs); dirs = np.array(dirs); errs_imu = np.array(errs_imu)
    print("\n" + "=" * 60)
    print(f"CMax  vs GT : mean err {errs.mean():6.2f}°/s   median {np.median(errs):6.2f}   "
          f"mean dir {dirs.mean():5.1f}°")
    print(f"IMU   vs GT : mean err {errs_imu.mean():6.2f}°/s   median {np.median(errs_imu):6.2f}   "
          f"(reference — the sensor CMax aims to replace)")
    print("=" * 60)
    print("Interpretation: if CMax err ≈ IMU err (both small), CMax can replace "
          "the IMU as the R-anchor for V1.")


if __name__ == '__main__':
    main()
