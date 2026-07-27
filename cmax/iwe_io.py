"""
IWE (Image of Warped Events) logging for the CMax variants.

Saves the FINAL contrast-maximized IWE per frame — V1: after the full CMax solve;
V2: after the message-passing iterations — plus a per-frame log. Shared by both
versions so the saved artifact is identical and directly comparable.

Brightness: per-frame normalisation for the PNGs and the GIF (for now). Global
sequence brightness is a future option.
"""

import os
import csv
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _to_uint8(iwe, pct=99.0):
    """Per-frame normalise |IWE| to uint8 [0,255], clipping at `pct` percentile."""
    a = np.abs(iwe)
    hi = np.percentile(a, pct)
    if hi < 1e-9:
        hi = a.max() if a.max() > 0 else 1.0
    return (np.clip(a / hi, 0.0, 1.0) * 255).astype(np.uint8)


def save_iwe(iwe, omega, contrast, out_dir, frame_idx, t_mid=None, n_events=0):
    """Write `iwe/frame_XXXX.png` and append a row to `iwe/iwe_log.csv`."""
    os.makedirs(out_dir, exist_ok=True)
    plt.imsave(os.path.join(out_dir, f'frame_{frame_idx:04d}.png'),
               _to_uint8(iwe), cmap='gray')

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


def plot_contrast_curve(out_dir):
    """contrast-vs-frame from iwe_log.csv → contrast_curve.png."""
    log_path = os.path.join(out_dir, 'iwe_log.csv')
    if not os.path.exists(log_path):
        return
    rows = list(csv.DictReader(open(log_path)))
    if not rows:
        return
    frames = [int(r['frame']) for r in rows]
    contrast = [float(r['contrast']) for r in rows]
    plt.figure(figsize=(10, 4))
    plt.plot(frames, contrast, '-o', ms=3)
    plt.xlabel('frame'); plt.ylabel('IWE contrast (variance)')
    plt.title('Final-IWE contrast per frame  (higher = sharper focus)')
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'contrast_curve.png'), dpi=120)
    plt.close()


def make_gif(out_dir, fps=15):
    """Assemble frame_*.png → iwe_sequence.gif (per-frame brightness). Best-effort."""
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
