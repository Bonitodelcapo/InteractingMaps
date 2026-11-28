"""
Checks of the persistence / block-matching network (Martel 2019, §6.5, §6.8.3).

Spec: docs/superpowers/specs/2026-11-28-persistence-block-matching-design.md

Standalone script, like the other test_*.py, run from the repo root:
    PYTHONPATH=. python tests/test_persistence.py
Prints one [ok]/[FAIL] line per check; exits non-zero on any failure.
Needs data/poster_rotation (calibration and APS frames).
"""

import os
import sys
import traceback

os.environ.setdefault('IM_MLFLOW', '0')

import numpy as np

from config import get_dataset_paths
from interacting_maps.camera import build_kinematic_matrix
from interacting_maps.data_loader import CameraCalibration
from interacting_maps.rotation_warp import (Camera, pixel_rays, rotation_matrix,
                                            compose, warp_coords, warp_jacobian,
                                            bilinear_sample)

H, W = 180, 240
DATASET = 'poster_rotation'


def _cams():
    """Pinhole test camera and the poster calibration (distorted)."""
    cam0 = Camera(200.0, 200.0, 120.0, 90.0, None)
    c = CameraCalibration(get_dataset_paths(DATASET)['calib'])
    camd = Camera(c.fx, c.fy, c.cx, c.cy, np.asarray(c.dist, np.float64))
    return cam0, camd


def _grid():
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    return uu, vv


INTERIOR = (slice(10, H - 10), slice(10, W - 10))


# --------------------------------------------------------------- Task 1: warp
def check_warp_identity():
    uu, vv = _grid()
    for cam, tol in zip(_cams(), (1e-6, 1e-3)):
        rays = pixel_rays(H, W, cam)
        u, v = warp_coords(rays, np.zeros(3), cam)
        err = max(np.abs(u - uu).max(), np.abs(v - vv).max())
        assert err < tol, f"identity warp off by {err:.2e} px (dist={cam.dist is not None})"


def check_warp_matches_C():
    """Small R: T_R(p) - p = -C(p)·R, i.e. I(p) = Ǐ(p - F) with F = C·R."""
    uu, vv = _grid()
    R = np.array([2e-4, -3e-4, 1e-4])
    for cam in _cams():
        rays = pixel_rays(H, W, cam)
        u, v = warp_coords(rays, R, cam)
        C = build_kinematic_matrix(H, W, cam.fx, cam.fy, cam.cx, cam.cy,
                                   dist_coeffs=cam.dist)
        F = np.einsum('hwij,j->hwi', C, R)
        d = np.stack([u - uu, v - vv], axis=-1)
        err = np.abs(d + F)[INTERIOR].max()
        scale = np.abs(F)[INTERIOR].max()
        assert err < 1e-2 * scale, f"warp vs -C·R: err {err:.2e}, |F| {scale:.2e}"


def check_compose():
    rng = np.random.default_rng(0)
    for _ in range(5):
        d, R = rng.normal(0, 0.05, 3), rng.normal(0, 0.05, 3)
        lhs = rotation_matrix(compose(d, R))
        rhs = rotation_matrix(d) @ rotation_matrix(R)
        assert np.abs(lhs - rhs).max() < 1e-10, "compose does not match matrix product"


def check_warp_jacobian_fd():
    R = np.array([0.01, -0.02, 0.015])
    eps = 1e-6
    for cam in _cams():
        rays = pixel_rays(H, W, cam)
        J = warp_jacobian(rays, R, cam)
        u0, v0 = warp_coords(rays, R, cam)
        for k in range(3):
            e = np.zeros(3); e[k] = eps
            u1, v1 = warp_coords(rays, compose(e, R), cam)
            fd = np.stack([(u1 - u0) / eps, (v1 - v0) / eps], axis=-1)[INTERIOR]
            an = J[..., k][INTERIOR]
            rel = np.abs(fd - an).max() / np.abs(fd).max()
            assert rel < 1e-3, f"Jacobian col {k}: rel err {rel:.2e}"


def check_warp_outside_masked():
    cam0, _ = _cams()
    rays = pixel_rays(H, W, cam0)
    img = np.random.default_rng(1).random((H, W))
    vals, valid = bilinear_sample(img, *warp_coords(rays, np.array([0.0, 0.5, 0.0]), cam0))
    assert valid.sum() < H * W, "large rotation left every pixel valid"
    assert np.isfinite(vals).all(), "non-finite samples"
    assert (vals[~valid] == 0).all(), "invalid samples not zeroed"


CHECKS = [
    check_warp_identity,
    check_warp_matches_C,
    check_compose,
    check_warp_jacobian_fd,
    check_warp_outside_masked,
]


def main():
    if not os.path.exists(get_dataset_paths(DATASET)['calib']):
        print(f"[skip] {DATASET} not present locally")
        return 0
    fails = 0
    for chk in CHECKS:
        try:
            chk()
            print(f"[ok]   {chk.__name__}")
        except Exception as e:
            fails += 1
            print(f"[FAIL] {chk.__name__}: {e}")
            if not isinstance(e, AssertionError):
                traceback.print_exc()
    print(f"\n{len(CHECKS) - fails}/{len(CHECKS)} checks passed")
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
