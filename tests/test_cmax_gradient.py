"""Check the analytic contrast gradient against finite differences.

CMax-inloop updates the angular velocity with one gradient-ascent step on the
contrast (the variance of the unblurred image of warped events) per iteration,
using the analytic gradient of `CMaxAngularVelocity._contrast_and_grad`. This
script checks that gradient against central finite differences of the same
variance, computed through the independent IWE builder `_contrast`.

    PYTHONPATH=. python tests/test_cmax_gradient.py

1. synthetic — events sampled on a few straight edges, made sharp at a known
   omega; the gradient is checked at random omegas around it.
2. real      — one 20 ms window of poster_rotation seg_C, if the dataset is
   present locally (skipped otherwise). Undistorted events reach past the
   frame border, and the code's gradient drops the term that events leaving
   the frame contribute (it assumes sum_u dI_u/domega = 0, i.e. no voting mass
   is lost). On real windows that costs a relative error of ~1e-3, so the
   tolerance there is 1e-2.

The variance is piecewise quadratic in omega (bilinear voting): its gradient
jumps wherever a warped event crosses a pixel border, and with tens of
thousands of events those kinks lie ~1e-5 rad/s apart. A central difference
that straddles one is off by up to a few percent, whatever the step. A kink can
lie on one side of omega within the step, not on both, so per component the
central, forward and backward differences (step 1e-6 rad/s) are taken and the
closest one is compared. The two gradients then agree to ~1e-4 or better; a
wrong sign, factor or term in the gradient gives an error of order 0.1 to 1.
"""
import os
import sys
import numpy as np

from interacting_maps.angular_velocity import CMaxAngularVelocity

H, W = 180, 240
FX, FY, CX, CY = 200.0, 200.0, 120.0, 90.0
EPS = 1e-6            # rad/s, central-difference step
TOL = 1e-3            # largest relative error accepted, at every omega
TOL_REAL = 1e-2       # real window: events leave the frame (see above)
FAILURES = []


def check(name, ok, detail=''):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ''))
    if not ok:
        FAILURES.append(name)


def fd_error(est, omega, bearings, dt, w, g_an):
    """Largest relative error over the three components. Per component the
    central, forward and backward differences are taken and the closest kept:
    a kink can lie on one side of omega within EPS, not on both, and where
    there is none the central difference is exact (the piece is quadratic)."""
    v0 = est._contrast(omega, bearings, dt, w)
    scale = max(np.max(np.abs(g_an)), 1e-30)
    errs = []
    for j in range(3):
        e = np.zeros(3); e[j] = EPS
        vp = est._contrast(omega + e, bearings, dt, w)
        vm = est._contrast(omega - e, bearings, dt, w)
        diffs = [(vp - vm) / (2 * EPS), (vp - v0) / EPS, (v0 - vm) / EPS]
        errs.append(min(abs(d - g_an[j]) for d in diffs) / scale)
    return max(errs)


def compare(est, omegas, bearings, dt, w, label, tol=TOL):
    errs = []
    for omega in omegas:
        _, g_an = est._contrast_and_grad(omega, bearings, dt, w)
        errs.append(fd_error(est, omega, bearings, dt, w, g_an))
    errs = np.array(errs)
    check(label, errs.max() < tol,
          f"{len(errs)} omegas: median rel err {np.median(errs):.1e}, max {errs.max():.1e}")


def synthetic():
    print("synthetic edges")
    rng = np.random.default_rng(0)
    est = CMaxAngularVelocity(H, W, FX, FY, CX, CY, use_polarity=True, blur_sigma=0.0)
    omega_true = np.array([0.4, -0.8, 0.3])                   # rad/s

    # Points on six random segments, at random times in a 20 ms window.
    n = 60000
    seg = rng.integers(0, 6, n)
    a = rng.uniform(20, W - 20, (6, 2)); b = rng.uniform(20, H - 20, (6, 2))
    s = rng.uniform(0, 1, n)
    x = a[seg, 0] + s * (a[seg, 1] - a[seg, 0])
    y = b[seg, 0] + s * (b[seg, 1] - b[seg, 0])
    p = est._bearings(x, y)
    dt = rng.uniform(-0.01, 0.01, n)
    # Move each point back along the rotation so that omega_true re-aligns it.
    p_t = p - np.cross(dt[:, None] * omega_true[None, :], p)
    p_t /= np.linalg.norm(p_t, axis=1, keepdims=True)
    xs = FX * p_t[:, 0] / p_t[:, 2] + CX
    ys = FY * p_t[:, 1] / p_t[:, 2] + CY
    bearings = est._bearings(xs, ys)
    w = np.where(rng.uniform(size=n) < 0.5, -1.0, 1.0)

    omegas = omega_true + rng.uniform(-1.0, 1.0, (20, 3))
    compare(est, omegas, bearings, dt, w, f"{n} events")


def real():
    print("real window (poster_rotation seg_C)")
    try:
        from config import get_dataset_paths
        from interacting_maps.data_loader import CameraCalibration, load_events_fast, undistort_events
        paths = get_dataset_paths('poster_rotation')
    except Exception as e:  # noqa: BLE001
        print(f"  [skip] {e}")
        return
    if not os.path.exists(paths['events']):
        print("  [skip] data/poster_rotation not present")
        return
    calib = CameraCalibration(paths['calib'])
    est = CMaxAngularVelocity(H, W, calib.fx, calib.fy, calib.cx, calib.cy,
                              use_polarity=True, blur_sigma=0.0)
    t0, dur = 8.816, 0.020
    ev = undistort_events(load_events_fast(paths['events'], t_start=t0, duration=dur), calib)
    t = ev[:, 0].astype(float)
    bearings = est._bearings(ev[:, 1].astype(float), ev[:, 2].astype(float))
    dt = t - 0.5 * (t.min() + t.max())
    w = 2.0 * ev[:, 3].astype(float) - 1.0
    omegas = np.random.default_rng(1).uniform(-2.0, 2.0, (20, 3))
    compare(est, omegas, bearings, dt, w, f"{len(ev)} events", tol=TOL_REAL)


if __name__ == '__main__':
    synthetic()
    real()
    print("\nall ok" if not FAILURES else f"\nFAILED: {FAILURES}")
    sys.exit(1 if FAILURES else 0)
