"""
Plot conventions shared by every figure: model names and colours, ω-axis
colours, reference/IMU line styles, units, time axis, label text, size presets
and saving. Nothing here draws; viz.py does. Change a convention here and every
figure follows.
"""

import os
import numpy as np
import matplotlib

MODELS = {
    'cook':           ('Cook',        '#E69F00'),
    'thesis':         ('Thesis',      '#D55E00'),
    'thesis_imu':     ('IMU-anchor',  '#009E73'),
    'thesis_cmax':    ('CMax-anchor', '#0072B2'),
    'thesis_cmax_v2': ('CMax-inloop', '#CC79A7'),
}
OMEGA_AXIS_COLORS = ('#4e79a7', '#f28e2b', '#e15759')
REF_STYLE = dict(color='k', lw=1.2, ls='-')
IMU_STYLE = dict(color='0.55', lw=0.9, ls=':', label='IMU gyro (input)')
LINE_COLOR = '0.25'

LABEL_TIME = 'time (s)'
LABEL_OMEGA = r'$\omega$ (°/s)'
LABEL_ERROR = 'error (°/s)'
LABEL_DIR_ERROR = 'direction error (°)'
LABEL_BETA = r'scale $\beta$'
MAP_TITLES = {'V': r'events $V$', 'I': r'intensity $I$', 'aps': 'APS (log)',
              'G': r'gradient $G$', 'F': r'flow $F$', 'IWE': 'IWE'}
_REF_NAMES = {'omega_gt': r'$\omega_{gt}$', 'groundtruth': 'Vicon', 'imu': 'IMU'}

PAPER_WIDTH, PAPER_WIDE, SCREEN_WIDTH = 3.4, 7.05, 12.0
_BASE = {'axes.grid': True, 'grid.alpha': 0.3, 'axes.spines.top': False,
         'axes.spines.right': False, 'legend.frameon': False,
         'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans'}
_PRESETS = {
    'paper':  {'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
               'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7},
    'screen': {'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10,
               'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9},
}
_REPORT_FIGURES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'report', 'figures')


def omega_label(axis):
    return rf'$\omega_{"xyz"[axis]}$ (°/s)'


def ref_label(ref_src):
    return f'reference ({_REF_NAMES[ref_src]})'


def to_deg_s(omega_rad_s):
    return np.degrees(omega_rad_s)


def frame_times(dt, n):
    """Seconds since segment start at each frame's midpoint."""
    return (np.arange(n) + 0.5) * dt


def use_style(preset):
    return matplotlib.rc_context({**_BASE, **_PRESETS[preset]})


def _wants_pdf(path):
    folder = os.path.normcase(os.path.dirname(os.path.abspath(path)))
    return folder == os.path.normcase(_REPORT_FIGURES)


def save(fig, path, show=False):
    """PNG at 200 dpi (+ PDF for report/figures/), then close or show."""
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    if _wants_pdf(path):
        fig.savefig(os.path.splitext(path)[0] + '.pdf', bbox_inches='tight', pad_inches=0.02)
    if show:
        plt.show()
    else:
        plt.close(fig)
