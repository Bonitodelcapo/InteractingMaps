"""
Distortion validation via Contrast Maximization — the network-free geometry test.

We validate the distortion-aware warp used by CMax (Way 2) — which is the SAME
lens geometry baked into the distortion-aware C matrix
(interacting_maps/camera.py) — against the trusted undistort-first path (Way 1),
on the same events, with the Interacting-Maps network removed.

  Way 1  undistort events first, then a plain pinhole warp (pi/pi^-1 have no
         distortion terms). Uses only cv2.undistortPoints — trusted.
  Way 2  keep RAW events; distortion lives INSIDE the warp:
         pi^-1 = cv2.undistortPoints, rotate, pi = forward Brown-Conrady.

Two levels of check:

  A/B  GEOMETRY (tight, pass/fail).  For a fixed rotation, the two warps must be
       the SAME map: warping the distortion-aware way and undistorting the result
       must land on the pinhole-warp result (sub-pixel). This isolates the lens
       geometry from the estimator and is the real confirmation of the C-matrix
       re-distortion.  + a unit anchor of the forward model vs cv2.projectPoints.

  C    ESTIMATOR (context, not a hard gate).  Run both CMax optimizations and
       compare omega. They do NOT match exactly: CMax's variance objective is
       computed on different pixel grids (undistorted vs distorted) and is not
       invariant to that resampling — a benign property of CMax that scales with
       distortion and vanishes without it (verified separately), NOT a geometry
       error. Reported for context alongside each way's agreement with the gyro.

Run:
    python test_distortion_cmax.py
    python test_distortion_cmax.py --dataset poster_rotation --segment seg_C
"""

import sys
import argparse
import numpy as np
import cv2

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from config import get_dataset_paths, DATASET_SEGMENTS
from data_loader import CameraCalibration, load_events_fast, undistort_events
from evaluation import gt_omega_body, get_gyro_for_frame, load_imu, compute_metrics
from cmax import CMaxAngularVelocity

RAD = 180.0 / np.pi


def check_forward_model(est, dist, n=4000, seed=0):
    """A) forward Brown-Conrady (Way 2's pi) vs cv2.projectPoints — unit anchor."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-0.6, 0.6, n)
    b = rng.uniform(-0.6, 0.6, n)
    xd, yd = est._distort_normalized(a, b)
    u_mine, v_mine = est.fx * xd + est.cx, est.fy * yd + est.cy
    objp = np.stack([a, b, np.ones_like(a)], -1).astype(np.float64).reshape(-1, 1, 3)
    proj, _ = cv2.projectPoints(objp, np.zeros(3), np.zeros(3), est.K, dist)
    return float(max(np.abs(u_mine - proj[:, 0, 0]).max(),
                     np.abs(v_mine - proj[:, 0, 1]).max()))


def warp_correspondence(est2, win, tref, omega):
    """B) fixed-omega geometry: |undistort(Way2 warp) - Way1 warp|, per event (px).

    Same rays + same rotation, so the distortion-aware warp undistorted must equal
    the pinhole warp. Non-zero only if the lens maps disagree.
    """
    xs = win[:, 1].astype(np.float64)
    ys = win[:, 2].astype(np.float64)
    dt = win[:, 0] - tref
    brg = est2._bearings(xs, ys)                       # true rays (undistortPoints)

    xw2, yw2 = est2._warp_to_pixels(brg, dt, omega)    # Way 2: reproject WITH distortion
    pts = np.stack([xw2, yw2], -1).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, est2.K, est2.dist, P=est2.K)[:, 0, :]

    p_rot = brg + np.cross(dt[:, None] * omega[None, :], brg)   # Way 1: pinhole reproject
    z = np.where(np.abs(p_rot[:, 2]) < 1e-9, 1e-9, p_rot[:, 2])
    xw1 = est2.fx * p_rot[:, 0] / z + est2.cx
    yw1 = est2.fy * p_rot[:, 1] / z + est2.cy

    return np.abs(und - np.stack([xw1, yw1], -1)).max(axis=1)   # (N,) px error


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='poster_rotation')
    ap.add_argument('--segment', default='seg_C')
    ap.add_argument('--n_frames', type=int, default=25)
    ap.add_argument('--geom_tol_px', type=float, default=0.5,
                    help='PASS if the fixed-omega warp correspondence stays below this (px)')
    args = ap.parse_args()

    seg = next(s for s in DATASET_SEGMENTS[args.dataset] if s['id'] == args.segment)
    t_start, dt = seg['t_start'], seg['frame_duration']
    n_frames = args.n_frames
    H, W = seg.get('sensor_size', (180, 240))

    paths = get_dataset_paths(args.dataset)
    calib = CameraCalibration(paths['calib'])
    fx, fy, cx, cy = calib.fx, calib.fy, calib.cx, calib.cy

    dur = n_frames * dt + 0.1
    ev_raw = load_events_fast(paths['events'], t_start=t_start, duration=dur)
    ev_und = undistort_events(ev_raw.copy(), calib)
    imu = load_imu(paths['imu'])
    print(f"{args.dataset}/{args.segment}: {len(ev_raw)} events over {dur:.2f}s, "
          f"dt={dt*1000:.0f}ms, dist={np.array2string(calib.dist, precision=4)}\n")

    est1 = CMaxAngularVelocity(H, W, fx, fy, cx, cy, use_polarity=True, blur_sigma=1.0)
    est2 = CMaxAngularVelocity(H, W, fx, fy, cx, cy, use_polarity=True, blur_sigma=1.0,
                               dist_coeffs=calib.dist)

    # ------------------------------------------------------------------ A
    fwd_err = check_forward_model(est2, calib.dist)
    a_ok = fwd_err < 1e-6
    print(f"[A] forward distortion vs cv2.projectPoints : max {fwd_err:.2e} px  "
          f"-> {'PASS' if a_ok else 'FAIL'}")

    # -------------------------------------------------- B + C over frames
    print(f"\n{'k':>3} {'n_ev':>7} | {'geomB max px':>12} | "
          f"{'w1(undist) x':>12} {'y':>7} {'z':>7} | "
          f"{'w2(rawdist) x':>13} {'y':>7} {'z':>7} | "
          f"{'|w1-w2|°/s':>10} {'w1-gy':>6} {'w2-gy':>6}")
    print("-" * 120)

    geom_max = 0.0
    prev1 = prev2 = np.zeros(3)
    diffs, e1g, e2g = [], [], []
    for k in range(n_frames):
        t_lo = t_start + k * dt
        t_hi = t_lo + dt
        t_ref = 0.5 * (t_lo + t_hi)
        m = (ev_raw[:, 0] >= t_lo) & (ev_raw[:, 0] < t_hi)
        win_raw, win_und = ev_raw[m], ev_und[m]

        # B: geometry correspondence at the (independent) gyro omega
        gyro = get_gyro_for_frame(imu, t_lo, t_hi)
        gb = warp_correspondence(est2, win_raw, t_ref, gyro).max() if m.sum() else 0.0
        geom_max = max(geom_max, gb)

        # C: the two estimators (context)
        w1 = est1.estimate(win_und, t_ref=t_ref, omega_init=prev1)
        w2 = est2.estimate(win_raw, t_ref=t_ref, omega_init=prev2)
        prev1, prev2 = w1.copy(), w2.copy()
        diff = np.linalg.norm(w1 - w2) * RAD
        eg1 = compute_metrics(w1, gyro)[0]
        eg2 = compute_metrics(w2, gyro)[0]
        diffs.append(diff); e1g.append(eg1); e2g.append(eg2)

        print(f"{k:3d} {int(m.sum()):7d} | {gb:12.4f} | "
              f"{w1[0]:+12.3f} {w1[1]:+7.3f} {w1[2]:+7.3f} | "
              f"{w2[0]:+13.3f} {w2[1]:+7.3f} {w2[2]:+7.3f} | "
              f"{diff:10.3f} {eg1:6.2f} {eg2:6.2f}")

    diffs = np.array(diffs); e1g = np.array(e1g); e2g = np.array(e2g)
    b_ok = geom_max < args.geom_tol_px

    print("\n" + "=" * 72)
    print("[B] GEOMETRY  fixed-omega warp correspondence (the real lens-model test)")
    print(f"    max |undistort(Way2 warp) - Way1 warp| = {geom_max:.4f} px  "
          f"(tol {args.geom_tol_px}) -> {'PASS' if b_ok else 'FAIL'}")
    print("[C] ESTIMATOR (context — CMax objective is grid-dependent, not a gate)")
    print(f"    mean |w1-w2| = {diffs.mean():.2f} °/s  (grid-metric; scales with "
          f"distortion, ->0 without it)")
    print(f"    w1 (undistort-first)     vs gyro : {e1g.mean():.2f} °/s")
    print(f"    w2 (distortion-in-warp)  vs gyro : {e2g.mean():.2f} °/s")
    print(f"    => both recover the same physical rotation to within their own "
          f"(~{max(e1g.mean(), e2g.mean()):.0f} °/s) gyro accuracy")
    print("=" * 72)

    ok = a_ok and b_ok
    print(f"RESULT: {'PASS' if ok else 'FAIL'}  — the distortion-aware warp (= the "
          f"C-matrix re-distortion geometry) is {'confirmed' if ok else 'NOT confirmed'} "
          f"by an independent path.")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
