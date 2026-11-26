"""Small read-only helpers behind the viewer's tools: where the spikes are,
one pixel's history, and the residuals across the iterations of a replay.
They read saved states / replays (inspect_io.RunView); no model is stepped.
"""
import numpy as np


# ------------------------------------------------------------------ spikes
def worst_frame(y) -> int:
    """Frame of the largest finite value."""
    y = np.asarray(y, dtype=np.float64)
    return int(np.nanargmax(np.where(np.isfinite(y), y, np.nan)))


def next_spike(y, k, threshold, step=+1):
    """Next frame after k (step=+1) or before k (step=-1) with y > threshold; None if none."""
    y = np.asarray(y, dtype=np.float64)
    idx = range(k + 1, len(y)) if step > 0 else range(k - 1, -1, -1)
    return next((i for i in idx if np.isfinite(y[i]) and y[i] > threshold), None)


# ------------------------------------------------------------------ pixel probe
def _at_pixel(maps, x, y):
    return {'I': float(maps['I'][y, x]),
            '|G|': float(np.hypot(*maps['G'][y, x])),
            '|F|': float(np.hypot(*maps['F'][y, x]))}


def pixel_series(run, x, y) -> dict:
    """I, |G|, |F| at pixel (x = column, y = row) after every frame."""
    rows = [_at_pixel(run.state(k, cache=False), x, y) for k in range(run.n_frames)]
    return {key: np.array([r[key] for r in rows]) for key in ('I', '|G|', '|F|')}


def pixel_iterations(run, k, x, y):
    """The same over the iterations of frame k's replay; None without a replay."""
    rep = run.iterations(k)
    if rep is None:
        return None
    return {'I': rep['I'][:, y, x],
            '|G|': np.hypot(rep['G'][:, y, x, 0], rep['G'][:, y, x, 1]),
            '|F|': np.hypot(rep['F'][:, y, x, 0], rep['F'][:, y, x, 1])}


# ------------------------------------------------------------------ residuals
def _grad(I):
    """Forward differences, zero on the last column / row (the networks' ∇I)."""
    gx, gy = np.zeros_like(I), np.zeros_like(I)
    gx[:, :-1] = I[:, 1:] - I[:, :-1]
    gy[:-1, :] = I[1:, :] - I[:-1, :]
    return np.stack([gx, gy], axis=-1)


def iteration_residuals(run, k):
    """Mean |V+F·G| (NaN without the dataset), mean |G−∇I| and |ω| (deg/s) at every
    iteration of frame k's replay; None without a replay. For Cook the ∇I here is
    taken on the cropped H x W intensity, so its last row/column differ slightly."""
    rep = run.iterations(k)
    if rep is None:
        return None
    V = run.raw(k)['V']
    n = len(rep['R'])
    ofce = np.full(n, np.nan)
    if V is not None:
        ofce = np.array([np.mean(np.abs(V + np.einsum('hwk,hwk->hw', rep['F'][i], rep['G'][i])))
                         for i in range(n)])
    gi = np.array([np.mean(np.abs(rep['G'][i] - _grad(rep['I'][i]))) for i in range(n)])
    w = np.degrees(np.linalg.norm(rep['R'], axis=1) / run.dt)
    return {'|V+F·G|': ofce, '|G−∇I|': gi, '|R|': w}
