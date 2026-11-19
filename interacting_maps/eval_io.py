"""
Sensor and reference I/O for the evaluation harness (R3 of architecture_review.md).

Loads the IMU gyro (model input), the Vicon groundtruth and the exact synthetic
omega, and turns them into the body-frame angular velocity used to *score* an
estimate. Extracted verbatim from evaluation.py so the harness file carries only
RunConfig + experiments; evaluation re-exports every name here, so existing
callers (``E.load_imu`` etc.) are unchanged.

The IMU gyro (imu.txt) is the camera's OWN sensor. When it is also fed to the
thesis_imu model as ``omega_imu``, scoring against it is circular (the model is
graded against its own input). groundtruth.txt comes from an INDEPENDENT
motion-capture rig, so scoring against it is unbiased.

omega is recovered by differencing two successive orientation quaternions:
    dR_body = R1.T @ R2      (right-invariant -> CAMERA BODY FRAME)
NOT R2 @ R1.T, which would give world-frame omega. The network and the gyro both
report body-frame omega, so the reference must be body-frame too.
"""

import os
import numpy as np

# Which source to SCORE against: 'groundtruth' (Vicon, independent) or 'imu'
# (gyro -- only use for datasets that ship no groundtruth.txt). Model INPUT for
# thesis_imu is always the gyro regardless of this setting.
SCORE_AGAINST = 'groundtruth'

# Half-width (seconds) of the bracket used to difference Vicon poses into an
# angular velocity. 0.05 -> a +-50 ms bracket around the frame midpoint.
# Set to frame_duration/2 to recover the old (noisy) per-frame differencing.
GT_DIFF_HALFWIDTH = 0.05


def load_imu(path: str) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float64)


def get_gyro_for_frame(imu_data, t_lo, t_hi):
    mask = (imu_data[:, 0] >= t_lo) & (imu_data[:, 0] < t_hi)
    if np.sum(mask) == 0:
        idx = np.argmin(np.abs(imu_data[:, 0] - (t_lo + t_hi) / 2))
        return imu_data[idx, 4:7]
    return np.mean(imu_data[mask, 4:7], axis=0)


def load_groundtruth(path: str):
    """Load groundtruth.txt: [t tx ty tz qx qy qz qw] → (N, 8), or None."""
    if not os.path.exists(path):
        return None
    return np.loadtxt(path, dtype=np.float64)


def _quat_to_rotmat(q):
    """Quaternion (qx, qy, qz, qw) → 3×3 rotation matrix R_wc."""
    qx, qy, qz, qw = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),   2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),   2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ])


def gt_omega_body(gt_data, t_lo, t_hi):
    """
    Body-frame angular velocity (rad/s) from two groundtruth.txt poses that
    bracket the frame window [t_lo, t_hi], via dR_body = R1.T @ R2.
    """
    idx1 = int(np.argmin(np.abs(gt_data[:, 0] - t_lo)))
    idx2 = int(np.argmin(np.abs(gt_data[:, 0] - t_hi)))
    if idx1 == idx2:
        idx2 = min(idx1 + 1, len(gt_data) - 1)
    actual_dt = gt_data[idx2, 0] - gt_data[idx1, 0]
    if abs(actual_dt) < 1e-10:
        return np.zeros(3)

    R1 = _quat_to_rotmat(gt_data[idx1, 4:8])
    R2 = _quat_to_rotmat(gt_data[idx2, 4:8])
    dR = R1.T @ R2                      # body frame (NOT R2 @ R1.T)

    cos_a = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_a)
    if abs(angle) < 1e-10:
        return np.zeros(3)
    skew = (dR - dR.T) / (2.0 * np.sin(angle) + 1e-15)
    axis = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    return axis * angle / actual_dt    # rad/s, body frame


def load_omega_gt(path):
    """
    Load omega_gt.txt (t wx wy wz) if the dataset provides one. Returns None if
    absent. Written by convert_ecrot.py: for the ECRot synthetic bags this is
    the simulator's EXACT angular velocity, taken from the twist topic.
    """
    if not path or not os.path.exists(path):
        return None
    data = np.loadtxt(path, dtype=np.float64)
    return data.reshape(1, -1) if data.ndim == 1 else data


def omega_gt_at(omega_data, t_lo, t_hi):
    """Exact reference omega for a frame window (rad/s).

    Averaged over the SAME +-GT_DIFF_HALFWIDTH bracket the differenced-pose
    reference uses. The bracket exists because Vicon gives poses, not rates, and
    differencing them over one 20 ms frame is dominated by pose noise -- but if
    it were applied only there, the real sequences would be scored against a
    smoothed reference and the synthetic ones against an instantaneous one, and
    the two columns of every table would not be comparable. Applying it to both
    costs nothing where omega is steady (on the ECRot segments it changes the
    error by under 0.01 deg/s) and removes the asymmetry.
    """
    t_mid = 0.5 * (t_lo + t_hi)
    m = (omega_data[:, 0] >= t_mid - GT_DIFF_HALFWIDTH) & \
        (omega_data[:, 0] <= t_mid + GT_DIFF_HALFWIDTH)
    if m.sum() > 0:
        return omega_data[m, 1:4].mean(axis=0)
    i = int(np.argmin(np.abs(omega_data[:, 0] - t_mid)))
    return omega_data[i, 1:4]


def get_reference_omega(gt_data, imu_data, t_lo, t_hi, omega_data=None):
    """
    Angular velocity used to SCORE the estimate (independent of model input).
    Returns (omega_ref, source_str).

    Preference order:
      1. omega_gt.txt, when the dataset ships one. For the ECRot synthetic bags
         this is the simulator's exact angular velocity, so it carries no
         differencing error at all.
      2. groundtruth.txt poses, differenced. Over a single 20 ms frame window
         that difference is dominated by pose noise -- the gyro itself scores
         8-25 deg/s against such a reference -- so we difference over a wider
         bracket CENTRED on the frame midpoint (GT_DIFF_HALFWIDTH). Even then
         the differencing costs ~0.8 deg/s, measured against the exact twist on
         the synthetic sequences.
      3. the gyro, for datasets with neither.
    """
    if SCORE_AGAINST == 'groundtruth' and omega_data is not None:
        return omega_gt_at(omega_data, t_lo, t_hi), 'omega_gt'
    if SCORE_AGAINST == 'groundtruth' and gt_data is not None:
        t_mid = 0.5 * (t_lo + t_hi)
        return (gt_omega_body(gt_data,
                              t_mid - GT_DIFF_HALFWIDTH,
                              t_mid + GT_DIFF_HALFWIDTH),
                'groundtruth')
    return get_gyro_for_frame(imu_data, t_lo, t_hi), 'imu'
