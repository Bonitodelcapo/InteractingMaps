"""Checks that the plotting conventions in style.py and viz.py hold numerically.

    PYTHONPATH=. python tests/test_viz.py
"""
import os
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from interacting_maps import style

FAILURES = []


def check(name, ok, detail=''):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ''))
    if not ok:
        FAILURES.append(name)


def test_style():
    print('style.py')
    check('five models with name and colour',
          set(style.MODELS) == {'cook', 'thesis', 'thesis_imu', 'thesis_cmax', 'thesis_cmax_v2'})
    check('thesis_imu is IMU-anchor', style.MODELS['thesis_imu'][0] == 'IMU-anchor')
    check('model colours are distinct', len({c for _, c in style.MODELS.values()}) == 5)
    check('frame_times are midpoints',
          np.allclose(style.frame_times(0.02, 3), [0.01, 0.03, 0.05]))
    check('to_deg_s converts pi rad/s', np.isclose(style.to_deg_s(np.pi), 180.0))
    check('ref_label omega_gt', 'omega' in style.ref_label('omega_gt'))
    check('ref_label groundtruth', 'Vicon' in style.ref_label('groundtruth'))
    check('ref_label imu', 'IMU' in style.ref_label('imu'))
    check('omega_label y', style.omega_label(1) == r'$\omega_y$ (°/s)')
    with style.use_style('paper'):
        inside = matplotlib.rcParams['font.size']
    check('use_style is scoped', inside == 8 and matplotlib.rcParams['font.size'] != 8)
    check('pdf only under report/figures',
          style._wants_pdf(os.path.join('report', 'figures', 'x.png'))
          and not style._wants_pdf(os.path.join('experiments', 'x.png')))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'sub', 'x.png')
        fig, _ = plt.subplots()
        style.save(fig, path)
        check('save writes png and closes', os.path.exists(path)
              and not os.path.exists(path[:-4] + '.pdf')
              and not plt.fignum_exists(fig.number))


if __name__ == '__main__':
    test_style()
    print('all checks passed' if not FAILURES else f"FAILED: {', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)
