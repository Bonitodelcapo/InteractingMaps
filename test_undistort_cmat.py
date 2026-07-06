"""
Sanity check for the distorted C_mat path.

Tests:
  1. Zero-distortion calibration -> distorted-mode C_mat == pinhole C_mat.
  2. Synthetic dataset (k_i = 0) builds without error in both modes.
  3. Poster calibration (k1 = -0.37): distorted C_mat differs from pinhole;
     Jacobian J_g should reduce to identity near image centre, diverge at corners.
  4. The undistort iteration is self-consistent (re-distorting recovers input).
"""

import numpy as np
from data_loader import CameraCalibration
from interacting_maps.camera import build_kinematic_matrix, _undistort_points_iterative


print("="*70)
print("TEST 1: zero distortion -> pinhole C_mat reproduced exactly")
print("="*70)
H, W = 180, 240
fx, fy = 199.0924, 198.8288
cx, cy = 120.0, 90.0

C_pin  = build_kinematic_matrix(H, W, fx, fy, cx, cy, dist_coeffs=None)
C_zero = build_kinematic_matrix(H, W, fx, fy, cx, cy,
                                dist_coeffs=np.zeros(5))
err = np.max(np.abs(C_pin - C_zero))
print(f"  max |C_pin - C_zero_dist| = {err:.2e}")
print(f"  -> {'PASS' if err < 1e-9 else 'FAIL'}\n")


print("="*70)
print("TEST 2: poster calibration -> distorted C_mat differs from pinhole")
print("="*70)
calib = CameraCalibration('data/poster_rotation/calib.txt')
print(f"  loaded poster calib: fx={calib.fx:.1f}, cx={calib.cx:.1f}, k1={calib.dist[0]:.4f}")

C_pin_poster   = build_kinematic_matrix(H, W, calib.fx, calib.fy, calib.cx, calib.cy)
C_dist_poster  = build_kinematic_matrix(H, W, calib.fx, calib.fy, calib.cx, calib.cy,
                                        dist_coeffs=calib.dist)
diff = C_dist_poster - C_pin_poster

# Examine error at centre vs corner
cy_int, cx_int = H // 2, W // 2
centre_diff = np.linalg.norm(diff[cy_int, cx_int])
corner_diff = np.linalg.norm(diff[0, 0])
print(f"  ||C_dist - C_pin|| at sensor centre [{cy_int},{cx_int}] : {centre_diff:.4f}")
print(f"  ||C_dist - C_pin|| at sensor corner [0, 0]             : {corner_diff:.4f}")
print(f"  ratio corner/centre = {corner_diff / max(centre_diff, 1e-9):.1f}x")
print(f"  -> {'PASS' if corner_diff > 10 * centre_diff else 'FAIL'} "
      f"(distortion should be larger at corners)\n")


print("="*70)
print("TEST 3: undistort iteration is self-consistent")
print("="*70)
# Take poster's distortion params, apply forward then inverse -> should match input
k1, k2, p1, p2, k3 = calib.dist
cols = np.arange(W, dtype=np.float64)
rows = np.arange(H, dtype=np.float64)
uu, vv = np.meshgrid(cols, rows)
xd_grid = (uu - calib.cx) / calib.fx
yd_grid = (vv - calib.cy) / calib.fy

# Undistort
x, y = _undistort_points_iterative(xd_grid, yd_grid, k1, k2, p1, p2, k3, n_iters=10)
# Re-distort (forward model)
r2     = x*x + y*y
k_rad  = 1 + k1*r2 + k2*r2**2 + k3*r2**3
xd_rec = x * k_rad + 2*p1*x*y + p2*(r2 + 2*x*x)
yd_rec = y * k_rad + p1*(r2 + 2*y*y) + 2*p2*x*y

err_x = np.max(np.abs(xd_grid - xd_rec))
err_y = np.max(np.abs(yd_grid - yd_rec))
print(f"  max |x_d - redistort(undistort(x_d))| = {err_x:.2e}")
print(f"  max |y_d - redistort(undistort(y_d))| = {err_y:.2e}")
print(f"  -> {'PASS' if max(err_x, err_y) < 1e-6 else 'FAIL'}\n")


print("="*70)
print("TEST 4: synthetic dataset (zero distortion) builds correctly")
print("="*70)
calib_syn = CameraCalibration('data/synthetic_xrot/calib.txt')
C_syn_pin  = build_kinematic_matrix(H, W, calib_syn.fx, calib_syn.fy, calib_syn.cx, calib_syn.cy)
C_syn_dist = build_kinematic_matrix(H, W, calib_syn.fx, calib_syn.fy, calib_syn.cx, calib_syn.cy,
                                    dist_coeffs=calib_syn.dist)
err = np.max(np.abs(C_syn_pin - C_syn_dist))
print(f"  synthetic dataset has dist_coeffs = {calib_syn.dist}")
print(f"  max |C_pin - C_dist| (should be 0 since k_i = p_i = 0) = {err:.2e}")
print(f"  -> {'PASS' if err < 1e-9 else 'FAIL'}")
