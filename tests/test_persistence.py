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
                                            bilinear_sample, solve_rotation_gn,
                                            block_match)
from interacting_maps.eval_io import load_aps_list, nearest_aps
from scipy.ndimage import gaussian_filter

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


# --------------------------------------------------- Task 2: Gauss–Newton solve
def _poster_frame():
    """First poster APS frame, grey in [0, 1], blurred like a network I."""
    imgs = load_aps_list(get_dataset_paths(DATASET)['data_dir'])
    img = np.asarray(nearest_aps(imgs, imgs[0][0]), np.float64)
    img = (img - img.min()) / (img.max() - img.min())
    return gaussian_filter(img, sigma=1.0)


def _gn(I, Ic, cam, R0, mask=None):
    rays = pixel_rays(H, W, cam)
    gy, gx = np.gradient(Ic)
    return solve_rotation_gn(I, Ic, (gx, gy), rays, cam, np.asarray(R0, float), mask=mask)


def check_gn_known_rotation():
    _, camd = _cams()
    Ic = _poster_frame()
    R_true = np.array([0.002, -0.003, 0.002])
    I, valid = bilinear_sample(Ic, *warp_coords(pixel_rays(H, W, camd), R_true, camd))
    for R0 in (np.zeros(3), 2 * R_true):
        R = _gn(I, Ic, camd, R0, mask=valid)
        err = np.linalg.norm(R - R_true)
        assert err < 1e-4, f"from R0={R0}: |R - R_true| = {err:.2e} rad"


def check_gn_identical_images():
    _, camd = _cams()
    Ic = _poster_frame()
    R = _gn(Ic, Ic, camd, [0.003, 0.002, -0.002])
    assert np.linalg.norm(R) < 1e-4, f"I == Ǐ should give R* ≈ 0, got {R}"


def _mean_cost(I, Ic, cam, R):
    vals, valid = bilinear_sample(Ic, *warp_coords(pixel_rays(H, W, cam), R, cam))
    return np.mean((I - vals)[valid] ** 2)


def check_gn_noise_no_cost_increase():
    """Unrelated, noise-like images (a network I early in a frame): the solve
    must not end at a rotation with a higher cost than where it started."""
    _, camd = _cams()
    rng = np.random.default_rng(3)
    I = gaussian_filter(rng.random((H, W)), 1.0)
    Ic = gaussian_filter(rng.random((H, W)), 1.0)
    R0 = np.array([0.002, -0.001, 0.001])
    R = _gn(I, Ic, camd, R0)
    c0, c1 = _mean_cost(I, Ic, camd, R0), _mean_cost(I, Ic, camd, R)
    assert c1 <= c0 + 1e-15, f"cost rose {c0:.6e} -> {c1:.6e}, R moved {np.linalg.norm(R - R0):.2e}"


def check_gn_flat_reference():
    _, camd = _cams()
    flat = np.ones((H, W))
    R0 = np.array([0.001, 0.0, 0.0])
    R = _gn(flat, flat, camd, R0)
    assert np.isfinite(R).all() and np.abs(R - R0).max() < 1e-12, f"flat Ǐ moved R: {R}"


# --------------------------------------------------------- Task 3: block match
def check_bm_shift():
    Ic = _poster_frame()
    I = np.roll(Ic, shift=(2, -3), axis=(0, 1))        # dy = 2, dx = -3
    Fp, valid = block_match(I, Ic, half_block=3, radius=4)
    hit = np.all(Fp[valid] == np.array([-3.0, 2.0]), axis=-1).mean()
    assert hit >= 0.99, f"only {hit:.3f} of valid pixels found the shift (-3, 2)"


def check_bm_flat_zero():
    flat = np.ones((H, W))
    Fp, valid = block_match(flat, flat, half_block=3, radius=4)
    assert valid.any() and (Fp[valid] == 0).all(), "flat image gave non-zero F'"


# ------------------------------------------------------------ Task 4: network
from config import THESIS_PARAMS
from interacting_maps import network_dissertation as ND

_THESIS_INIT, _THESIS_STEP = ND.InteractingMapsThesis.__init__, ND.InteractingMapsThesis.step
from interacting_maps.network_persistence import InteractingMapsPersist

R_INIT = np.array([0.002, -0.001, 0.001])
VARIANTS = [(True, False), (False, True), (True, True)]


def _V():
    rng = np.random.default_rng(0)
    return (rng.random((H, W)) < 0.1) * rng.choice([-1.0, 1.0], (H, W))


def _net(cls, **kw):
    _, camd = _cams()
    net = cls(H=H, W=W, fx=camd.fx, fy=camd.fy, cx=camd.cx, cy=camd.cy,
              frame_duration=0.02, dist_coeffs=camd.dist, **THESIS_PARAMS, **kw)
    net.initialize_from_rotation(R_INIT)
    return net


def check_frame0_equals_thesis():
    V = _V()
    ref = _net(ND.InteractingMapsThesis)
    ref.step(V, n_iters=10)
    for up, ub in VARIANTS:
        net = _net(InteractingMapsPersist, use_persistence=up, use_block_matching=ub)
        net.step(V, n_iters=10)
        for name in ('I', 'G', 'F', 'R'):
            assert np.array_equal(getattr(net, name), getattr(ref, name)),                 f"frame 0 differs from thesis in {name} (persist={up}, bm={ub})"


def check_frame1_reference():
    V = _V()
    for up, ub in VARIANTS:
        net = _net(InteractingMapsPersist, use_persistence=up, use_block_matching=ub)
        net.step(V, n_iters=2)
        I1 = net.I.copy()
        net.step(V, n_iters=1)
        assert np.array_equal(net.q_Icheck.value, I1), "Ǐ is not the previous frame's I"
        for c in net.persist_costs:
            assert c.active, f"{type(c).__name__} inactive on frame 1"
        assert len(net.persist_costs) == up + ub, "wrong set of enabled costs"
        for name in ('I', 'G', 'F', 'R'):
            assert np.isfinite(getattr(net, name)).all(), f"non-finite {name}"


def check_thesis_untouched():
    assert ND.InteractingMapsThesis.__init__ is _THESIS_INIT
    assert ND.InteractingMapsThesis.step is _THESIS_STEP


# ------------------------------------------------------ Task 5: harness wiring
import shutil
from config import PERSIST_MODELS, PERSIST_DELTAS

OUT_ROOT = 'experiments/persist_test'
_RC = dict(dataset=DATASET, segment='seg_C', n_frames=3, n_iters=10, out_root=OUT_ROOT)

# Captured from the code before the wiring (commit ed90de0); must not change.
OLD_DIRS = {
    'thesis': os.path.join(OUT_ROOT, DATASET, 'thesis',
                           'seg_C_t8.816_dt20ms_n3_i10_dFR0.10_C_full'),
    'thesis_imu': os.path.join(OUT_ROOT, DATASET, 'thesis_imu',
                               'seg_C_t8.816_dt20ms_n3_i10_dimu0.50_dFR0.10_C_full'),
}
OLD_SWEEP_COLS = (['sequence', 'segment', 'model', 'dt_ms', 'n_frames', 'n_iters', 'poisson',
                   'delta_VFG', 'delta_IG', 'delta_GI', 'delta_RF', 'delta_FR',
                   'delta_IMU', 'delta_curl', 'delta_map', 'delta_shrinkI']
                  + ['err', 'median', 'dir', 'beta', 'low_freq', 'contrast', 'curl',
                     'recon_mean', 'recon_min', 'recon_inloop', 'recon_inloop_min',
                     'blowup', 'secs'])


def _run_experiments():
    sys.path.insert(0, 'report')
    import run_experiments
    return run_experiments


def check_runconfig_models():
    from run_config import RunConfig
    for m in PERSIST_MODELS:
        rc = RunConfig(model=m, **_RC)
        assert rc.use_thesis and not rc.use_imu, f"{m}: wrong thesis/imu flags"
        assert rc.use_persistence == ('persist' in m), f"{m}: use_persistence"
        assert rc.use_block_matching == m.endswith('bm'), f"{m}: use_block_matching"
        name = os.path.basename(rc.output_dir)
        assert name == 'seg_C_t8.816_dt20ms_n3_i10_dFR0.10_C_full', f"{m}: suffix in {name}"
        assert rc.output_dir.startswith('experiments'), rc.output_dir
    for m in ('thesis', 'thesis_imu', 'cook'):
        rc = RunConfig(model=m, **_RC)
        assert not rc.use_persistence and not rc.use_block_matching, f"{m}: flags set"


def check_old_model_rejects_persist_delta():
    from run_config import RunConfig
    try:
        RunConfig(model='thesis', deltas={'delta_BM': 0.1}, **_RC)
    except ValueError:
        return
    raise AssertionError("thesis accepted delta_BM")


def check_old_output_dir_unchanged():
    from run_config import RunConfig
    for m, d in OLD_DIRS.items():
        got = RunConfig(model=m, **_RC).output_dir
        assert got == d, f"{m}: {got} != {d}"


def check_sweep_columns():
    X = _run_experiments()
    assert X._parse_vary('delta_BM=0.1,0.2') == [('delta_BM', [0.1, 0.2])]
    assert X._sweep_columns('thesis') == OLD_SWEEP_COLS, "old sweep header changed"
    i = OLD_SWEEP_COLS.index('delta_shrinkI') + 1
    want = OLD_SWEEP_COLS[:i] + PERSIST_DELTAS + OLD_SWEEP_COLS[i:]
    assert X._sweep_columns('thesis_persist') == want, "persist sweep header wrong"


def check_exp6_refuses_persist():
    """The basin test steps a fresh network once: frame 0, where the new costs
    are off by design. It must refuse rather than report the thesis basin."""
    import evaluation as E
    from run_config import RunConfig
    for m in PERSIST_MODELS:
        try:
            E.experiment_basin_of_attraction(RunConfig(model=m, **_RC), frame_idx=1)
        except ValueError:
            continue
        finally:
            shutil.rmtree(OUT_ROOT, ignore_errors=True)
        raise AssertionError(f"exp 6 ran {m} (new costs never active)")


def check_smoke_runs():
    import evaluation as E
    from run_config import RunConfig
    try:
        for m in PERSIST_MODELS:
            rc = RunConfig(model=m, **_RC)
            s = E.experiment_tracking(rc, save_frames=False)
            assert np.isfinite(s['mean_err_deg_s']), f"{m}: non-finite error"
            for f in ('run.json', 'tracking.csv'):
                assert os.path.exists(os.path.join(rc.output_dir, f)), f"{m}: no {f}"
    finally:
        shutil.rmtree(OUT_ROOT, ignore_errors=True)


CHECKS = [
    check_warp_identity,
    check_warp_matches_C,
    check_compose,
    check_warp_jacobian_fd,
    check_warp_outside_masked,
    check_gn_known_rotation,
    check_gn_identical_images,
    check_gn_flat_reference,
    check_gn_noise_no_cost_increase,
    check_bm_shift,
    check_bm_flat_zero,
    check_frame0_equals_thesis,
    check_frame1_reference,
    check_thesis_untouched,
    check_runconfig_models,
    check_old_model_rejects_persist_delta,
    check_old_output_dir_unchanged,
    check_sweep_columns,
    check_exp6_refuses_persist,
    check_smoke_runs,
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
