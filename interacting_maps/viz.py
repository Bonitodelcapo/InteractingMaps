"""
Drawing helpers. Every convention (names, colours, units, labels) comes from
style.py; nothing here defines its own.

Maps: renderers turn an array into an RGB (H, W, 3) array in [0, 1] and never
touch figures; draw_map puts one on an axis. Line plots: draw_omega,
draw_error and draw_series fill one axis each, time in seconds.
"""

import numpy as np

from interacting_maps.style import (MAP_TITLES, MODELS, OMEGA_AXIS_COLORS, REF_STYLE,
                                    IMU_STYLE, LINE_COLOR, LABEL_TIME, LABEL_ERROR,
                                    omega_label, ref_label, to_deg_s)


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


def _robust_max(a, pct=99.0):
    hi = float(np.percentile(a, pct))
    return hi if hi > 1e-12 else float(a.max())


def _gray(g):
    return np.repeat(g[..., None], 3, axis=-1)


def render_V(V, scale=None):
    """bwr: red positive, blue negative, white zero; symmetric ±scale (default p99 |V|)."""
    from matplotlib import colormaps
    s = _robust_max(np.abs(V)) if scale is None else scale
    if s <= 1e-12:
        return np.ones(V.shape + (3,))
    return colormaps['bwr'](0.5 + 0.5 * np.clip(V / s, -1.0, 1.0))[..., :3]


def render_I(I, scale=None):
    """Grey; 1-99 percentile by default, or an explicit (lo, hi)."""
    if scale is None:
        return _gray(normalise_robust(I))
    lo, hi = scale
    if hi - lo < 1e-9:
        return _gray(np.full(I.shape, 0.5))
    return _gray(np.clip((I - lo) / (hi - lo), 0.0, 1.0))


def render_vec(v, scale=None):
    """Hue = direction (image coords, y down), saturation = |v|/scale, white base."""
    from matplotlib.colors import hsv_to_rgb
    mag = np.hypot(v[..., 0], v[..., 1])
    s = _robust_max(mag) if scale is None else scale
    if s <= 1e-12:
        return np.ones(v.shape[:2] + (3,))
    hue = (np.arctan2(v[..., 1], v[..., 0]) / (2 * np.pi)) % 1.0
    sat = np.clip(mag / s, 0.0, 1.0)
    return hsv_to_rgb(np.stack([hue, sat, np.ones_like(hue)], axis=-1))


def render_IWE(iwe, scale=None):
    """Grey, white = many events; |IWE| / scale (default p99)."""
    a = np.abs(iwe)
    s = _robust_max(a) if scale is None else scale
    if s <= 1e-12:
        return np.zeros(iwe.shape + (3,))
    return _gray(np.clip(a / s, 0.0, 1.0))


def vec_key(n=65):
    """Square direction key, centre white; coloured by render_vec itself."""
    u = np.linspace(-1.0, 1.0, n)
    X, Y = np.meshgrid(u, u)
    return render_vec(np.stack([X, Y], axis=-1), scale=1.0)


def _render_aps(aps, scale=None):
    a = np.asarray(aps, dtype=np.float64)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=-1)
    return render_I(np.log1p(a), scale)


_RENDERERS = {'V': render_V, 'I': render_I, 'aps': _render_aps,
              'G': render_vec, 'F': render_vec, 'IWE': render_IWE}


def net_maps(net, H, W):
    """I, G, F cropped to the H x W pixel grid (Cook's I is (H+1) x (W+1))."""
    return {'I': net.I[:H, :W], 'G': net.G[:H, :W], 'F': net.F[:H, :W]}


def draw_map(ax, kind, arr, title=None, scale=None):
    ax.imshow(_RENDERERS[kind](arr, scale=scale), interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(MAP_TITLES[kind] if title is None else title)
    if kind in ('G', 'F'):
        key = ax.inset_axes([0.02, 0.02, 0.2, 0.2])
        key.imshow(vec_key(), interpolation='nearest')
        key.set_axis_off()


def _line(model, label):
    if model is None:
        return LINE_COLOR, label
    name, color = MODELS[model]
    return color, label or name


def draw_omega(ax, t, est, ref=None, imu=None, axis=0, model=None, ref_src=None, label=None):
    """One ω component (inputs rad/s, drawn in deg/s) against time in seconds."""
    if ref is not None:
        ax.plot(t, to_deg_s(ref), label=ref_label(ref_src), **REF_STYLE)
    if imu is not None and ref_src != 'imu':
        ax.plot(t, to_deg_s(imu), **IMU_STYLE)
    if model is None:
        color, label = OMEGA_AXIS_COLORS[axis], label or 'estimate'
    else:
        color, label = _line(model, label)
    ax.plot(t, to_deg_s(est), color=color, lw=1.2, label=label)
    ax.set_ylabel(omega_label(axis))
    ax.set_xlabel(LABEL_TIME)


def draw_error(ax, t, err, model=None, label=None):
    color, label = _line(model, label)
    ax.plot(t, err, color=color, lw=1.0, label=label)
    ax.set_ylabel(LABEL_ERROR)
    ax.set_xlabel(LABEL_TIME)


def draw_series(ax, t, y, ylabel, model=None, label=None):
    color, label = _line(model, label)
    ax.plot(t, y, color=color, lw=1.0, marker='.', ms=3, label=label)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(LABEL_TIME)
