"""
Synthetic event-camera dataset compatible with validation.py
============================================================

Mimics the file format of the poster_rotation dataset (RPG / Mueggler 2017):
    data/synthetic_xrot/
        events.txt        # t  x  y  polarity            (one event per line)
        calib.txt         # fx fy cx cy k1 k2 p1 p2 k3   (one line)
        groundtruth.txt   # t tx ty tz qx qy qz qw       (Vicon-style poses)
        imu.txt           # t ax ay az gx gy gz          (gyro + accel)

But differs from poster_rotation in two intentional ways:

  1. NO LENS DISTORTION
     calib.txt has k1=k2=p1=p2=k3=0. Principal point is at the geometric
     centre of the DAVIS240C-like sensor: cx=W/2=120, cy=H/2=90. This means
     compute_calibration / build_kinematic_matrix produce the exact rays the
     synthetic events were generated from — no model-vs-data mismatch.

  2. ONE-AXIS ROTATION
     ω = (ω_x, 0, 0) with ω_x = 1.121 rad/s — same magnitude as
     poster_rotation's dominant axis, but no contamination from ω_y, ω_z.
     The pinhole kinematic flow for this is purely radial-vertical:
         F_u = fx · x'y' · ω_x
         F_v = fy · (y'² + 1) · ω_x
     so the network sees a textbook X-rotation flow field.

DVS event-generation model
--------------------------
Standard log-intensity contrast model (Lichtsteiner 2008):
- Each pixel maintains a reference log-intensity log_ref.
- At every simulation step dt_sim:
    Δ = log I(pixel, t) − log_ref
    if Δ >  +C:  emit ON  event, log_ref += C
    if Δ <  −C:  emit OFF event, log_ref −= C
- The scene is an equirectangular environment map (assumed at infinity, the
  correct geometry for a purely rotating camera with no parallax).

Output ω convention
-------------------
groundtruth.txt poses are R_wc(t) = R_x(ω_x · t) — camera-to-world quaternion,
matching the RPG convention. validation.gt_omega_body recovers the body-frame
ω = (ω_x, 0, 0) by computing R1ᵀ R2, which IS what the network estimates.

Run
---
    python generate_synthetic_dataset.py

After running, paste the printed DATASET_CONFIGS block into config.py.
"""

import os
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_NAME = 'synthetic_xrot'
OUTPUT_DIR   = Path(__file__).parent / 'data' / DATASET_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sensor / intrinsics  (DAVIS240C-shape but with NO distortion, principal
# point exactly at sensor centre).
H, W   = 180, 240
fx, fy = 199.0924, 198.8288
cx, cy = W / 2.0, H / 2.0    # 120, 90

# Motion: pure X-axis rotation
OMEGA_X = 1.121
OMEGA   = np.array([OMEGA_X, 0.0, 0.0], dtype=np.float64)

# Timing
DURATION  = 0.6      # seconds — > 25 × 20 ms frames + buffer
DT_SIM    = 1e-3     # 1 ms per simulation step → ~0.06° per step

# DVS contrast model
CONTRAST_THR  = 0.50    # ~1.5–2 Mev expected, comparable to poster_rotation (~1.3 Mev / 0.6 s)
INTENSITY_EPS = 1e-3

# Environment map (the "scene")
ENV_H, ENV_W = 720, 1440
RNG_SEED     = 42

# IMU
IMU_RATE       = 1000.0
IMU_GYRO_NOISE = 0.005       # rad/s std (Gaussian)

# Ground-truth pose rate (matches RPG ~200 Hz Vicon)
GT_RATE = 200.0

# Recommended t_start for validation.py (well inside the simulation)
VAL_T_START = 0.10


# ---------------------------------------------------------------------------
# Step 1 — build a band-limited "scene" texture as an equirectangular map.
# Purely rotational motion has no parallax, so the scene lives on the sphere.
# ---------------------------------------------------------------------------

def make_texture(H, W, seed=42):
    """Multi-scale band-limited noise — gives the OFCE rich gradient structure."""
    rng = np.random.default_rng(seed)
    raw_low   = gaussian_filter(rng.standard_normal((H, W)), sigma=4.0)
    raw_mid   = gaussian_filter(rng.standard_normal((H, W)), sigma=1.5)
    raw_high  = gaussian_filter(rng.standard_normal((H, W)), sigma=0.6)
    tex = (0.5
           + 0.30 * raw_low  / raw_low.std()
           + 0.25 * raw_mid  / raw_mid.std()
           + 0.20 * raw_high / raw_high.std())
    return np.clip(tex, 0.05, 1.0)   # avoid log(0)

print(f"Building scene texture ({ENV_H}×{ENV_W}) ...")
ENV = make_texture(ENV_H, ENV_W, seed=RNG_SEED)


# ---------------------------------------------------------------------------
# Step 2 — precompute camera rays (unit direction per pixel, camera frame)
# ---------------------------------------------------------------------------

cols   = np.arange(W, dtype=np.float64)
rows   = np.arange(H, dtype=np.float64)
uu, vv = np.meshgrid(cols, rows)
x_n    = (uu - cx) / fx
y_n    = (vv - cy) / fy
z_n    = np.ones_like(x_n)
norm   = np.sqrt(x_n**2 + y_n**2 + z_n**2)
CAM_RAYS = np.stack([x_n / norm, y_n / norm, z_n / norm], axis=-1)   # (H, W, 3)


# ---------------------------------------------------------------------------
# Step 3 — rotation matrix and texture-sphere lookup
# ---------------------------------------------------------------------------

def R_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,   c,  -s],
                     [0.0,   s,   c]], dtype=np.float64)


def sample_environment(world_rays: np.ndarray, env: np.ndarray) -> np.ndarray:
    """Bilinear lookup in the equirectangular env map."""
    x = world_rays[..., 0]
    y = world_rays[..., 1]
    z = world_rays[..., 2]

    az = np.arctan2(x, z)                              # ∈ [−π, π]
    el = np.arctan2(y, np.sqrt(x**2 + z**2))           # ∈ [−π/2, π/2]

    u = (az + np.pi) / (2.0 * np.pi) * env.shape[1]
    v = (el + np.pi / 2.0) / np.pi   * env.shape[0]

    u0 = np.floor(u).astype(int) % env.shape[1]
    u1 = (u0 + 1) % env.shape[1]
    v0 = np.clip(np.floor(v).astype(int), 0, env.shape[0] - 1)
    v1 = np.clip(v0 + 1,                  0, env.shape[0] - 1)
    du = u - np.floor(u)
    dv = v - np.floor(v)

    I = ((1 - du) * (1 - dv) * env[v0, u0] +
              du  * (1 - dv) * env[v0, u1] +
         (1 - du) *      dv  * env[v1, u0] +
              du  *      dv  * env[v1, u1])
    return I


# ---------------------------------------------------------------------------
# Step 4 — simulate the DVS pixel array
# ---------------------------------------------------------------------------

print(f"Simulating events ({DURATION:.3f} s, dt_sim = {DT_SIM*1e3:.2f} ms) ...")

I_ref   = sample_environment(CAM_RAYS, ENV)            # frame at t = 0
log_ref = np.log(I_ref + INTENSITY_EPS)

n_steps = int(np.round(DURATION / DT_SIM))
event_chunks = []   # list of (n_k, 4) arrays — concatenated at the end

for step in range(1, n_steps + 1):
    t = step * DT_SIM
    R = R_x(OMEGA_X * t)
    # world-frame rays = R @ camera ray (per pixel)
    world_rays = np.einsum('ij,hwj->hwi', R, CAM_RAYS)

    I_now   = sample_environment(world_rays, ENV)
    log_now = np.log(I_now + INTENSITY_EPS)
    delta   = log_now - log_ref

    on_mask  = delta >  CONTRAST_THR
    off_mask = delta < -CONTRAST_THR

    if on_mask.any():
        ys, xs = np.nonzero(on_mask)
        n      = xs.size
        chunk  = np.empty((n, 4), dtype=np.float64)
        chunk[:, 0] = t
        chunk[:, 1] = xs
        chunk[:, 2] = ys
        chunk[:, 3] = 1.0
        event_chunks.append(chunk)
        log_ref[on_mask] += CONTRAST_THR

    if off_mask.any():
        ys, xs = np.nonzero(off_mask)
        n      = xs.size
        chunk  = np.empty((n, 4), dtype=np.float64)
        chunk[:, 0] = t
        chunk[:, 1] = xs
        chunk[:, 2] = ys
        chunk[:, 3] = 0.0
        event_chunks.append(chunk)
        log_ref[off_mask] -= CONTRAST_THR

events_arr = np.concatenate(event_chunks, axis=0)
events_arr = events_arr[np.argsort(events_arr[:, 0], kind='stable')]
n_events   = events_arr.shape[0]
print(f"  {n_events} events generated "
      f"({n_events / DURATION:.0f} ev/s, "
      f"{n_events / (DURATION/0.020):.0f} ev / 20 ms frame on average)")


# ---------------------------------------------------------------------------
# Step 5 — write events.txt
# ---------------------------------------------------------------------------

events_path = OUTPUT_DIR / 'events.txt'
np.savetxt(events_path, events_arr, fmt='%.6f %d %d %d')
print(f"  wrote {events_path}  ({n_events} lines)")


# ---------------------------------------------------------------------------
# Step 6 — write calib.txt   (NO distortion)
# ---------------------------------------------------------------------------

calib_path = OUTPUT_DIR / 'calib.txt'
with open(calib_path, 'w') as f:
    f.write(f"{fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f} 0.0 0.0 0.0 0.0 0.0\n")
print(f"  wrote {calib_path}  (fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}, distortion = 0)")


# ---------------------------------------------------------------------------
# Step 7 — write groundtruth.txt   (R_wc(t) = R_x(ω_x · t),  zero translation)
# ---------------------------------------------------------------------------

gt_times = np.arange(0.0, DURATION + 1.0/GT_RATE, 1.0/GT_RATE)
gt_path  = OUTPUT_DIR / 'groundtruth.txt'

with open(gt_path, 'w') as f:
    for t in gt_times:
        angle = OMEGA_X * t
        qx = np.sin(angle / 2.0)
        qy = 0.0
        qz = 0.0
        qw = np.cos(angle / 2.0)
        f.write(f"{t:.6f} 0.000000 0.000000 0.000000 "
                f"{qx:.8f} {qy:.8f} {qz:.8f} {qw:.8f}\n")
print(f"  wrote {gt_path}  ({len(gt_times)} poses, R_wc = R_x(omega_x * t))")


# ---------------------------------------------------------------------------
# Step 8 — write imu.txt   (constant ω_body + Gaussian gyro noise; gravity in z)
# ---------------------------------------------------------------------------

imu_times = np.arange(0.0, DURATION + 1.0/IMU_RATE, 1.0/IMU_RATE)
rng_imu   = np.random.default_rng(RNG_SEED + 1)
imu_path  = OUTPUT_DIR / 'imu.txt'

with open(imu_path, 'w') as f:
    for t in imu_times:
        noise = rng_imu.standard_normal(3) * IMU_GYRO_NOISE
        gx = OMEGA[0] + noise[0]
        gy = OMEGA[1] + noise[1]
        gz = OMEGA[2] + noise[2]
        f.write(f"{t:.6f} 0.0000 0.0000 9.8100 "
                f"{gx:.6f} {gy:.6f} {gz:.6f}\n")
print(f"  wrote {imu_path}  ({len(imu_times)} samples)")


# ---------------------------------------------------------------------------
# Step 9 — print the DATASET_CONFIGS block to paste into config.py
# ---------------------------------------------------------------------------

initial_R = OMEGA * 0.020   # rad/frame at Δt = 20 ms
print("\n" + "="*72)
print("ADD THIS BLOCK to DATASET_CONFIGS in config.py:")
print("="*72)
print(f"    '{DATASET_NAME}': {{")
print(f"        't_start': {VAL_T_START:.3f},")
print(f"        'frame_duration': 0.020,")
print(f"        'n_frames': 25,")
print(f"        'initial_R':  np.array([{initial_R[0]:.5f}, {initial_R[1]:.5f}, {initial_R[2]:.5f}]),")
print(f"        'expected_omega': np.array([{OMEGA[0]:.3f}, {OMEGA[1]:.3f}, {OMEGA[2]:.3f}]),")
print(f"    }},")
print("="*72)
print(f"Then set DATASET = '{DATASET_NAME}' in validation.py and run it.")
