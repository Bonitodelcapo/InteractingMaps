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
from types import SimpleNamespace
from interacting_maps import viz

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


def test_maps():
    print('viz.py maps')
    white = np.ones(3)
    check('render_vec: zero vector is white',
          np.allclose(viz.render_vec(np.zeros((4, 4, 2)))[0, 0], white))
    plus_x = viz.render_vec(np.array([[[1.0, 0.0]]]), scale=1.0)[0, 0]
    key = viz.vec_key(65)
    check('render_vec: +x matches right-edge centre of key', np.allclose(plus_x, key[32, -1]))
    check('vec_key: centre is white', np.allclose(key[32, 32], white))
    check('render_vec: weak vector is pale (white base)',
          viz.render_vec(np.array([[[0.1, 0.0]]]), scale=1.0)[0, 0].min() > 0.85)
    V = np.zeros((10, 10)); V[0, 0], V[0, 1] = 1.0, -1.0
    rv = viz.render_V(V)
    check('render_V: zero is white', rv[5, 5].min() > 0.99)
    check('render_V: positive is red', rv[0, 0, 0] > 0.9 and rv[0, 0, 2] < 0.1)
    check('render_V: all-zero is white', np.allclose(viz.render_V(np.zeros((3, 3))), 1.0))
    check('render_I: constant is mid-grey',
          np.allclose(viz.render_I(np.full((5, 5), 3.0)), 0.5))
    check('render_I: explicit scale', np.allclose(viz.render_I(np.array([[2.0]]), scale=(0.0, 4.0)), 0.5))
    check('render_IWE: all-zero is black', np.allclose(viz.render_IWE(np.zeros((3, 3))), 0.0))
    sparse = np.zeros((100, 100)); sparse[0, 0] = 5.0
    check('render_IWE: sparse map falls back to max', viz.render_IWE(sparse)[0, 0].min() > 0.99)
    H, W = 6, 8
    cook = SimpleNamespace(I=np.zeros((H + 1, W + 1)), G=np.zeros((H, W, 2)), F=np.zeros((H, W, 2)))
    thesis = SimpleNamespace(I=np.zeros((H, W)), G=np.zeros((H, W, 2)), F=np.zeros((H, W, 2)))
    check('net_maps crops Cook and thesis alike',
          all(viz.net_maps(n, H, W)['I'].shape == (H, W) for n in (cook, thesis)))
    fig, axes = plt.subplots(1, 6)
    arrs = {'V': V, 'I': np.random.rand(10, 10), 'aps': np.random.rand(10, 10, 3),
            'G': np.random.rand(10, 10, 2) - 0.5, 'F': np.random.rand(10, 10, 2) - 0.5,
            'IWE': np.random.rand(10, 10)}
    for ax, (kind, a) in zip(axes, arrs.items()):
        viz.draw_map(ax, kind, a)
    check('draw_map: default titles', [a.get_title() for a in axes] == [style.MAP_TITLES[k] for k in arrs])
    check('draw_map: key inset only for G and F',
          [len(a.child_axes) for a in axes] == [0, 0, 0, 1, 1, 0])
    check('draw_map: no visible spines', not any(s.get_visible() for a in axes for s in a.spines.values()))
    plt.close(fig)


def test_lines():
    print('viz.py line plots')
    t = style.frame_times(0.02, 4)
    est = np.full(4, np.pi)
    fig, ax = plt.subplots()
    viz.draw_omega(ax, t, est, ref=est, imu=est, axis=2, ref_src='groundtruth')
    lines = ax.get_lines()
    check('draw_omega: ref, imu, estimate', len(lines) == 3)
    check('draw_omega: plotted in deg/s', np.allclose(lines[-1].get_ydata(), 180.0))
    check('draw_omega: x is seconds at midpoints', np.allclose(lines[-1].get_xdata(), t))
    check('draw_omega: axis colour when no model',
          lines[-1].get_color() == style.OMEGA_AXIS_COLORS[2])
    check('draw_omega: labels', ax.get_ylabel() == style.omega_label(2)
          and ax.get_xlabel() == style.LABEL_TIME)
    plt.close(fig)
    fig, ax = plt.subplots()
    viz.draw_omega(ax, t, est, ref=est, imu=est, axis=0, model='thesis_cmax', ref_src='imu')
    lines = ax.get_lines()
    check('draw_omega: IMU skipped when it is the reference', len(lines) == 2)
    check('draw_omega: model colour and name',
          lines[-1].get_color() == style.MODELS['thesis_cmax'][1]
          and lines[-1].get_label() == 'CMax-anchor')
    plt.close(fig)
    fig, ax = plt.subplots()
    viz.draw_error(ax, t, np.ones(4))
    viz.draw_series(ax, t, np.ones(4), 'contrast')
    check('draw_error/draw_series: one line each', len(ax.get_lines()) == 2)
    plt.close(fig)


def test_iwe_io():
    print('iwe_io.py')
    from interacting_maps import iwe_io
    with tempfile.TemporaryDirectory() as d:
        iwe = np.zeros((6, 8)); iwe[2, 3] = 4.0
        for k in range(3):
            iwe_io.save_iwe(iwe, np.zeros(3), 1.0 + k, d, k, t_mid=10.0 + (k + 0.5) * 0.02)
        img = plt.imread(os.path.join(d, 'frame_0000.png'))
        check('save_iwe: event pixel white, background black',
              img[2, 3, :3].min() > 0.99 and img[0, 0, :3].max() < 0.01)
        iwe_io.plot_contrast_curve(d, t_start=10.0)
        check('contrast curve written', os.path.exists(os.path.join(d, 'contrast_curve.png')))


if __name__ == '__main__':
    test_style()
    test_maps()
    test_lines()
    test_iwe_io()
    print('all checks passed' if not FAILURES else f"FAILED: {', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)
