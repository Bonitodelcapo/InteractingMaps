"""
IWE (Image of Warped Events) logging for the CMax variants.

Saves the FINAL contrast-maximized IWE per frame — V1: after the full CMax solve;
V2: after the message-passing iterations — plus a per-frame log. Shared by both
versions so the saved artifact is identical and directly comparable.

Rendering follows viz.render_IWE (grey, |IWE| at the 99th percentile per frame).
"""

import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt

from interacting_maps.style import SCREEN_WIDTH, use_style, save
from interacting_maps.viz import render_IWE, draw_series


def save_iwe(iwe, omega, contrast, out_dir, frame_idx, t_mid=None, n_events=0):
    """Write `iwe/frame_XXXX.png` and append a row to `iwe/iwe_log.csv`."""
    os.makedirs(out_dir, exist_ok=True)
    plt.imsave(os.path.join(out_dir, f'frame_{frame_idx:04d}.png'), render_IWE(iwe))

    log_path = os.path.join(out_dir, 'iwe_log.csv')
    new = not os.path.exists(log_path)
    with open(log_path, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['frame', 't_mid', 'n_events', 'wx', 'wy', 'wz', 'contrast'])
        w.writerow([frame_idx,
                    f'{t_mid:.6f}' if t_mid is not None else '',
                    int(n_events),
                    f'{omega[0]:.6f}', f'{omega[1]:.6f}', f'{omega[2]:.6f}',
                    f'{contrast:.6e}'])


def plot_contrast_curve(out_dir, t_start):
    """IWE contrast against seconds since segment start -> contrast_curve.png."""
    log_path = os.path.join(out_dir, 'iwe_log.csv')
    if not os.path.exists(log_path):
        return
    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    t = np.array([float(r['t_mid']) for r in rows]) - t_start
    contrast = np.array([float(r['contrast']) for r in rows])
    with use_style('screen'):
        fig, ax = plt.subplots(figsize=(SCREEN_WIDTH, 4))
        draw_series(ax, t, contrast, 'IWE contrast (variance)')
        ax.set_title('final-IWE contrast per frame (higher = sharper focus)')
        save(fig, os.path.join(out_dir, 'contrast_curve.png'))


def make_gif(out_dir, fps=15):
    """Assemble frame_*.png -> iwe_sequence.gif (per-frame brightness). Best-effort."""
    files = sorted(glob.glob(os.path.join(out_dir, 'frame_*.png')))
    if not files:
        return
    try:
        try:
            import imageio.v2 as imageio
        except ImportError:
            import imageio
        frames = [imageio.imread(f) for f in files]
        imageio.mimsave(os.path.join(out_dir, 'iwe_sequence.gif'), frames, fps=fps)
    except Exception as e:
        print(f"  (GIF skipped: {e})")
