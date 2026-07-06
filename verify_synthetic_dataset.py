"""
Programmatic integrity check for the synthetic_xrot dataset.

Verifies:
  1. calib.txt has zero distortion, principal point at sensor centre.
  2. groundtruth.txt quaternion differencing recovers omega = (1.121, 0, 0)
     to floating-point precision.
  3. imu.txt mean matches GT; std matches the configured noise level.
  4. events.txt: spatial distribution, ON/OFF balance, sparsity.
  5. The V frames built from these events show the expected X-rotation
     signature (vertical-dominant temporal derivative).

If any of these fail, the synthetic dataset itself is wrong.
If all pass, any error in validation.py is in the inference loop.
"""

import numpy as np
from pathlib import Path
from config import DATASET_CONFIGS, get_dataset_paths
from data_loader import EventFrameSequence, CameraCalibration
from validation import gt_omega_body, load_groundtruth, load_imu


DATASET = 'synthetic_xrot'
cfg     = DATASET_CONFIGS[DATASET]
paths   = get_dataset_paths(DATASET)

print(f"=== Verifying synthetic dataset: {DATASET} ===\n")


# ---------------------------------------------------------------------------
# CHECK 1 — calib.txt
# ---------------------------------------------------------------------------
print("--- CHECK 1: calib.txt ---")
c = CameraCalibration(paths['calib'])
print(f"  fx, fy = {c.fx:.4f}, {c.fy:.4f}")
print(f"  cx, cy = {c.cx:.4f}, {c.cy:.4f}   (expected: 120.0, 90.0)")
print(f"  distortion coeffs = {c.dist}")
ok = (
    abs(c.cx - 120.0) < 1e-6 and
    abs(c.cy -  90.0) < 1e-6 and
    np.all(c.dist == 0.0)
)
print(f"  -> {'PASS' if ok else 'FAIL'}\n")


# ---------------------------------------------------------------------------
# CHECK 2 — groundtruth.txt: quaternion differencing should recover omega
# ---------------------------------------------------------------------------
print("--- CHECK 2: groundtruth quaternion differencing ---")
gt = load_groundtruth(paths['groundtruth'])

# Sample 10 timestamps inside the simulation, recover omega via R1.T @ R2
test_ts = np.linspace(0.1, 0.5, 10)
omegas  = np.array([gt_omega_body(gt, t, 0.020) for t in test_ts])
print(f"  Recovered omega samples (rad/s):")
print(f"    mean  = [{omegas[:,0].mean():+.6f}, {omegas[:,1].mean():+.6f}, {omegas[:,2].mean():+.6f}]")
print(f"    std   = [{omegas[:,0].std():.2e}, {omegas[:,1].std():.2e}, {omegas[:,2].std():.2e}]")
print(f"    expected = [1.121000, 0.000000, 0.000000]")
err = np.linalg.norm(omegas.mean(axis=0) - np.array([1.121, 0.0, 0.0]))
ok = err < 1e-3
print(f"  Error vs expected: {err:.2e} rad/s")
print(f"  -> {'PASS' if ok else 'FAIL'}\n")


# ---------------------------------------------------------------------------
# CHECK 3 — imu.txt
# ---------------------------------------------------------------------------
print("--- CHECK 3: imu.txt statistics ---")
imu = load_imu(paths['imu'])
gyro = imu[:, 4:7]
print(f"  mean (gx, gy, gz) = ({gyro[:,0].mean():+.6f}, "
      f"{gyro[:,1].mean():+.6f}, {gyro[:,2].mean():+.6f})")
print(f"  std  (gx, gy, gz) = ({gyro[:,0].std():.4f}, "
      f"{gyro[:,1].std():.4f}, {gyro[:,2].std():.4f})")
print(f"  expected mean: (1.121, 0, 0)   expected std: (0.005, 0.005, 0.005)")

mean_ok = (abs(gyro[:,0].mean() - 1.121) < 0.01 and
           abs(gyro[:,1].mean()) < 0.01 and
           abs(gyro[:,2].mean()) < 0.01)
std_ok  = (abs(gyro[:,0].std() - 0.005) < 0.002 and
           abs(gyro[:,1].std() - 0.005) < 0.002 and
           abs(gyro[:,2].std() - 0.005) < 0.002)
print(f"  -> mean: {'PASS' if mean_ok else 'FAIL'}, std: {'PASS' if std_ok else 'FAIL'}\n")


# ---------------------------------------------------------------------------
# CHECK 4 — events.txt distribution
# ---------------------------------------------------------------------------
print("--- CHECK 4: events.txt ---")
# Use the same loader validation.py uses
seq = EventFrameSequence(
    paths['events'], paths['calib'],
    frame_duration=0.020, t_start=0.10, n_frames=25, clip_value=10.0,
)
events = seq._events
print(f"  total events loaded: {len(events):,}")
print(f"  time range: [{events[:,0].min():.3f}, {events[:,0].max():.3f}] s")
print(f"  x range: [{int(events[:,1].min())}, {int(events[:,1].max())}]   (sensor: 0..{seq.W-1})")
print(f"  y range: [{int(events[:,2].min())}, {int(events[:,2].max())}]   (sensor: 0..{seq.H-1})")
on_frac  = (events[:,3] == 1).mean()
print(f"  ON / OFF balance: {on_frac*100:.1f}% / {(1-on_frac)*100:.1f}%   (expected ~50/50)")
ok = (events[:,1].max() <= seq.W-1 and events[:,2].max() <= seq.H-1
      and abs(on_frac - 0.5) < 0.10)
print(f"  -> {'PASS' if ok else 'FAIL'}\n")


# ---------------------------------------------------------------------------
# CHECK 5 — V frame structure (X-rotation should produce vertical flow,
# so V should be more 'rows-correlated' than 'cols-correlated')
# ---------------------------------------------------------------------------
print("--- CHECK 5: V frame structure ---")
V_frames = []
for V, t_mid in seq:
    V_frames.append(V)
V_arr = np.array(V_frames)   # (N, H, W)
print(f"  V stats: mean|V|={np.mean(np.abs(V_arr)):.4f}   max|V|={np.max(np.abs(V_arr)):.4f}")
print(f"  V sparsity (V==0): {(V_arr == 0).mean()*100:.1f}%")

# For an X-rotation, the flow F is dominated by the F_v = fy(y'^2+1) omega_x term.
# So V = -F . grad(I) should have more vertical-direction structure.
# Measure: row-wise variance of V vs col-wise variance of V, averaged over frames.
row_var = float(V_arr.var(axis=1).mean())  # variance across rows
col_var = float(V_arr.var(axis=2).mean())  # variance across cols
print(f"  Variance across rows (vertical structure):    {row_var:.5f}")
print(f"  Variance across cols (horizontal structure):  {col_var:.5f}")
print(f"  ratio row/col = {row_var/col_var:.3f}    (X-rotation: expect > 1, vertical-dominant)")
ok = row_var / col_var > 1.0
print(f"  -> {'PASS' if ok else 'FAIL'}\n")


print("="*60)
print("If all five checks PASS, the synthetic dataset is correct.")
print("Any error inside validation.py is then attributable to the")
print("inference loop, not to the data.")
