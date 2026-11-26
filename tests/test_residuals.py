"""Checks on the constraint residuals both networks report (residual_VFG, residual_GI,
residual_FCR).

    PYTHONPATH=. python tests/test_residuals.py

1. kinematic residual — 0 when F = C·R exactly; mean |d| after adding a known d to F.
2. both networks — residual_FCR equals camera.kinematic_residual on their own C, F, R.
"""
import sys
import numpy as np

from interacting_maps.camera import build_kinematic_matrix, kinematic_residual
from interacting_maps.network import InteractingMaps
from interacting_maps.network_dissertation import InteractingMapsThesis

H, W, FX, FY, CX, CY = 18, 24, 20.0, 20.0, 12.0, 9.0
FAILURES = []


def check(name, ok, detail=''):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ''))
    if not ok:
        FAILURES.append(name)


def test_kinematic_residual():
    C = build_kinematic_matrix(H, W, FX, FY, CX, CY)
    R = np.array([0.01, -0.02, 0.005])
    F = np.einsum('hwij,j->hwi', C, R)
    check('F = C·R gives 0', kinematic_residual(C, F, R) < 1e-15)
    d = np.random.default_rng(0).standard_normal(F.shape) * 0.1
    r = kinematic_residual(C, F + d, R)
    check('F = C·R + d gives mean |d|', np.isclose(r, np.mean(np.abs(d))), f'{r:.6f}')


def test_networks():
    R = np.array([0.01, -0.02, 0.005])
    rng = np.random.default_rng(1)
    cook = InteractingMaps(H=H, W=W, fx=FX, fy=FY, cx=CX, cy=CY)
    cook.R = R.copy()
    cook.F = rng.standard_normal((H, W, 2)) * 0.1
    check('cook residual_FCR', np.isclose(cook.residual_FCR(),
                                          kinematic_residual(cook._C_mat, cook.F, cook.R)))
    thesis = InteractingMapsThesis(H=H, W=W, fx=FX, fy=FY, cx=CX, cy=CY)
    thesis.initialize_from_rotation(R)
    check('thesis residual_FCR = 0 right after initialize_from_rotation',
          thesis.residual_FCR() < 1e-15)
    thesis.q_F.value = thesis.q_F.value + 0.05
    check('thesis residual_FCR after shifting F by 0.05', np.isclose(thesis.residual_FCR(), 0.05))


if __name__ == '__main__':
    test_kinematic_residual()
    test_networks()
    print('ALL OK' if not FAILURES else f'{len(FAILURES)} FAILED: {FAILURES}')
    sys.exit(1 if FAILURES else 0)
