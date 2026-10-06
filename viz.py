"""
Display helpers: map fields to colour, normalise for stable frames
(R3 of architecture_review.md).

Pure, RunConfig-free rendering helpers extracted verbatim from evaluation.py.
The experiment-coupled plotters (_plot_tracking, _save_3col_frame) stay in
evaluation.py because they depend on RunConfig and the run's maps. evaluation
re-exports every name here, so existing callers are unchanged.
"""

import numpy as np


def normalise(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-10)


def normalise_robust(x, lo_pct=1.0, hi_pct=99.0):
    """Percentile min-max to [0,1] for stable frame-to-frame display.

    Plain min-max maps the extreme pixels to 0/1, so a single outlier that
    moves between frames rescales the whole image -> the video flickers darker
    /brighter. Clipping to robust percentiles (1st/99th) fixes the display
    range against outliers, and because I is only defined up to a gauge
    (offset/scale) this also cancels that drift, keeping brightness steady.
    """
    lo, hi = np.percentile(x, [lo_pct, hi_pct])
    if hi - lo < 1e-9:
        return np.full_like(x, 0.5, dtype=np.float64)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def flow_to_rgb(flow):
    from matplotlib.colors import hsv_to_rgb
    fx, fy = flow[..., 0], flow[..., 1]
    angle = (np.arctan2(fy, fx) + np.pi) / (2 * np.pi)
    mag = np.sqrt(fx**2 + fy**2)
    mag_norm = mag / (mag.max() + 1e-10)
    hsv = np.stack([angle, np.ones_like(angle), mag_norm], axis=-1)
    return hsv_to_rgb(hsv)


def grad_to_rgb(G, pct=99.0):
    """
    Spatial gradient as colour: HUE = direction, VALUE = magnitude.

    Showing only |G| hides the thing the gradient map is for -- which way the
    intensity is changing -- so an edge and its mirror image look identical.
    Hue makes direction visible (each orientation its own colour) while
    brightness still carries magnitude. Magnitude is scaled by a percentile so
    one hot pixel cannot black out the rest of the frame.
    """
    from matplotlib.colors import hsv_to_rgb
    gx, gy = G[..., 0], G[..., 1]
    ang = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)     # direction -> hue
    mag = np.hypot(gx, gy)
    hi = np.percentile(mag, pct)
    val = np.clip(mag / (hi + 1e-12), 0.0, 1.0)
    return hsv_to_rgb(np.stack([ang, np.ones_like(ang), val], axis=-1))
