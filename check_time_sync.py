"""
check_time_sync.py — are events.txt, imu.txt and the reference (groundtruth.txt or
omega_gt.txt) on the same clock? Prints the findings per dataset and writes one CSV.

For every dataset folder under data/ (or --datasets):

1. Files: time range, rate, timestamp jitter, gaps.
2. Gyro vs reference. The reference ω is either the Vicon poses differenced in the body
   frame (R(t−h)ᵀ·R(t+h), the scoring convention of eval_io, h = GT_DIFF_HALFWIDTH) or the
   exact omega_gt.txt averaged over the same window. The gyro is averaged over the same
   window, shifted by a lag; the lag with the smallest RMS |ω_gyro − ω_ref| is the clock
   offset. Also: axis correlation matrix (are the axes the same?), per-axis scale and bias.
   Done on the whole recording, its first and last third (drift), and each config segment.
3. Events vs motion. Event rate (1 ms bins) cross-correlated with |ω| (gyro, or the
   reference if the dataset has no IMU) on each segment. Weak: it only resolves offsets of
   about 10 ms or more, so the width of the peak is printed with it.

Sign convention: lag < 0 means the gyro value that matches a reference timestamp t was
recorded at t + lag, i.e. the reference timestamps are |lag| late against the DAVIS clock.
Event lag > 0 means the event rate follows |ω| by that much.

Run from the repo root (reads only data/, writes only --out):
    python check_time_sync.py                         # all datasets in data/
    python check_time_sync.py --datasets poster_rotation boxes_rotation
"""

import argparse
import os
import sys
from io import BytesIO

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

from config import DATASET_SEGMENTS, get_dataset_paths
from interacting_maps.eval_io import load_imu, load_groundtruth, load_omega_gt, GT_DIFF_HALFWIDTH
from interacting_maps.style import to_deg_s

GRID_DT = 0.005                                 # spacing of the comparison grid (s)
LAGS = np.arange(-0.100, 0.1005, 0.0005)        # gyro-vs-reference lags scanned (s)
EVENT_BIN = 0.001                               # event-rate bin (s)
EVENT_SMOOTH = 10                               # bins in the moving average of rate and |ω|
EVENT_LAGS_MS = np.arange(-60, 61)              # event-vs-motion lags scanned (ms)


# ---------------------------------------------------------------- file stats
def time_stats(name, t):
    """Range, median rate, jitter and gaps (> 2x the median step) of one timestamp column."""
    d = np.diff(t)
    med = np.median(d)
    return {'file': name, 't0': t[0], 't1': t[-1], 'rate_hz': 1.0 / med,
            'jitter_us': np.std(d) * 1e6, 'gaps': int((d > 2 * med).sum()),
            'max_gap_ms': d.max() * 1e3, 'monotonic': bool(np.all(d >= 0))}


# ---------------------------------------------------------------- ω on a common grid
def cumulative(t, w):
    """Trapezoid integral of w (N,3) over t, for window averages by differencing."""
    return np.vstack([np.zeros(3), np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(t)[:, None], axis=0)])


def window_mean(t, cum, centres, h, lag=0.0):
    """Mean of the integrated signal over [c−h+lag, c+h+lag] for every centre c."""
    a = np.stack([np.interp(centres - h + lag, t, cum[:, k]) for k in range(3)], 1)
    b = np.stack([np.interp(centres + h + lag, t, cum[:, k]) for k in range(3)], 1)
    return (b - a) / (2 * h)


def vicon_omega(gt, centres, h):
    """Body-frame ω (rad/s) from poses slerped at c±h: rotvec(R(c−h)ᵀ·R(c+h)) / 2h."""
    t, keep = np.unique(gt[:, 0], return_index=True)
    slerp = Slerp(t, Rotation.from_quat(gt[keep, 4:8]))
    return (slerp(centres - h).inv() * slerp(centres + h)).as_rotvec() / (2 * h)


def reference_omega(paths, centres, h):
    """(ω_ref on the grid, source name, (t, cum) when it is an exact rate) or (None, None, None)."""
    om = load_omega_gt(paths['omega_gt'])
    if om is not None:
        t, keep = np.unique(om[:, 0], return_index=True)
        cum = cumulative(t, om[keep, 1:4])
        return window_mean(t, cum, centres, h), 'omega_gt', (t, cum)
    gt = load_groundtruth(paths['groundtruth'])
    if gt is not None:
        return vicon_omega(gt, centres, h), 'vicon', None
    return None, None, None


def common_grid(ranges, h):
    """Grid of centres where every signal covers c±h even at the largest lag."""
    lo = max(r[0] for r in ranges) + h + LAGS.max()
    hi = min(r[1] for r in ranges) - h - LAGS.max()
    return np.arange(lo, hi, GRID_DT)


# ---------------------------------------------------------------- gyro vs reference
def lag_scan(t_imu, cum_imu, centres, h, w_ref, mask):
    """RMS |ω_gyro(lag) − ω_ref| (°/s) for every lag in LAGS, on the masked centres."""
    c = centres[mask]
    return np.array([np.sqrt(np.mean(np.sum(to_deg_s(window_mean(t_imu, cum_imu, c, h, L) - w_ref[mask]) ** 2, 1)))
                     for L in LAGS])


def _moves(x):
    """False for an axis that (almost) never rotates, where correlation and scale are undefined."""
    return np.std(x) > 1e-9


def axis_corr(w_gyro, w_ref):
    """3x3 correlation, gyro axes (rows) vs reference axes (cols); NaN where an axis does not move."""
    return np.array([[np.corrcoef(w_gyro[:, a], w_ref[:, b])[0, 1]
                      if _moves(w_gyro[:, a]) and _moves(w_ref[:, b]) else np.nan
                      for b in range(3)] for a in range(3)])


def scale_bias(w_gyro, w_ref):
    """Per axis: gyro ≈ scale·ref + bias (bias in °/s); scale NaN and bias = mean difference where ref does not move."""
    scale, bias = [], []
    for a in range(3):
        if _moves(w_ref[:, a]):
            s, b = np.polyfit(w_ref[:, a], w_gyro[:, a], 1)
        else:
            s, b = np.nan, np.mean(w_gyro[:, a] - w_ref[:, a])
        scale.append(s)
        bias.append(float(to_deg_s(b)))
    return scale, bias


def gyro_vs_ref(t_imu, cum_imu, centres, h, w_ref, mask):
    """Best lag, RMS at 0 and at the best lag, axis correlation, scale and bias, on one scope."""
    rms = lag_scan(t_imu, cum_imu, centres, h, w_ref, mask)
    k = int(np.argmin(rms))
    w_best = window_mean(t_imu, cum_imu, centres[mask], h, LAGS[k])
    scale, bias = scale_bias(w_best, w_ref[mask])
    return {'lag_ms': LAGS[k] * 1e3, 'rms_lag0': rms[np.argmin(np.abs(LAGS))], 'rms_best': rms[k],
            'corr': axis_corr(w_best, w_ref[mask]), 'scale': scale, 'bias': bias, 'n': int(mask.sum())}


# ---------------------------------------------------------------- events
def seek_time(f, size, t):
    """Byte offset of the first line of a sorted events.txt with timestamp >= t."""
    lo, hi = 0, size
    while hi - lo > 4096:
        mid = (lo + hi) // 2
        f.seek(mid)
        f.readline()
        line = f.readline()
        if not line or float(line.split()[0]) >= t:
            hi = mid
        else:
            lo = mid
    f.seek(lo)
    if lo:
        f.readline()
    return f.tell()


def event_times(path, t_lo, t_hi):
    """Timestamps of the events in [t_lo, t_hi), read without loading the whole file."""
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        a = seek_time(f, size, t_lo)
        f.seek(min(seek_time(f, size, t_hi) + 8192, size))   # the search stops up to 4 KB early;
        f.readline()                                          # read past it to a whole line
        b = f.tell()
        f.seek(a)
        buf = f.read(b - a)
    t = pd.read_csv(BytesIO(buf), sep=r'\s+', header=None, usecols=[0], dtype=np.float64)[0].values
    return t[(t >= t_lo) & (t < t_hi)]


def zscore_smooth(x):
    x = np.convolve(x, np.ones(EVENT_SMOOTH) / EVENT_SMOOTH, mode='same')
    return (x - x.mean()) / x.std()


def event_rate_lag(t_ev, t_lo, t_hi, speed_fn):
    """Cross-correlation of event rate with |ω| over EVENT_LAGS_MS: peak lag, cc at 0, peak, peak width."""
    edges = np.arange(t_lo, t_hi + EVENT_BIN / 2, EVENT_BIN)
    centres = 0.5 * (edges[1:] + edges[:-1])
    r = zscore_smooth(np.histogram(t_ev, edges)[0].astype(float))
    s = zscore_smooth(speed_fn(centres))
    m = EVENT_LAGS_MS.max()
    cc = np.array([np.mean(r[m + l:len(r) - m + l] * s[m:len(s) - m]) for l in EVENT_LAGS_MS])
    k = int(np.argmax(cc))
    within = EVENT_LAGS_MS[cc >= cc[k] - 0.01]   # lags indistinguishable from the peak
    return {'event_lag_ms': int(EVENT_LAGS_MS[k]), 'event_cc0': cc[m], 'event_cc_peak': cc[k],
            'event_width_ms': int(within.max() - within.min()), 'n_events': len(t_ev)}


def speed_from_gyro(imu):
    return lambda t: np.linalg.norm(np.stack([np.interp(t, imu[:, 0], imu[:, 4 + k]) for k in range(3)], 1), axis=1)


def speed_from_ref(centres, w_ref):
    sp = np.linalg.norm(w_ref, axis=1)
    return lambda t: np.interp(t, centres, sp)


# ---------------------------------------------------------------- per dataset
def scopes(centres, segments):
    """(name, t_lo, t_hi) for the whole grid, its first/last third, and each segment."""
    lo, hi = centres[0], centres[-1]
    third = (hi - lo) / 3
    out = [('all', lo, hi), ('first_third', lo, lo + third), ('last_third', hi - third, hi)]
    out += [(s['id'], s['t_start'], s['t_start'] + s['n_frames'] * s['frame_duration']) for s in segments]
    return out


def print_files(stats):
    for s in stats:
        print(f"  {s['file']:<12} {s['t0']:9.4f} .. {s['t1']:9.4f} s  {s['rate_hz']:8.1f} Hz  jitter {s['jitter_us']:8.1f} us  "
              f"gaps {s['gaps']:5d} (max {s['max_gap_ms']:.1f} ms)  monotonic {s['monotonic']}")


def print_row(row):
    line = f"  {row['scope']:<12}"
    if 'lag_ms' in row:
        line += (f" lag {row['lag_ms']:+6.1f} ms  RMS {row['rms_lag0']:6.2f} → {row['rms_best']:6.2f} °/s  "
                 f"r_diag {row['r_x']:.4f} {row['r_y']:.4f} {row['r_z']:.4f}  "
                 f"scale {row['scale_x']:.3f} {row['scale_y']:.3f} {row['scale_z']:.3f}  "
                 f"bias {row['bias_x']:+.2f} {row['bias_y']:+.2f} {row['bias_z']:+.2f} °/s")
    if 'event_lag_ms' in row:
        line += (f"  | events {row['event_lag_ms']:+d} ms (width {row['event_width_ms']} ms, "
                 f"cc0 {row['event_cc0']:.3f}, peak {row['event_cc_peak']:.3f})")
    print(line)


def check_dataset(ds, h):
    """All checks on one dataset; returns the summary rows."""
    paths = get_dataset_paths(ds)
    has = {k: os.path.exists(paths[k]) for k in ('events', 'imu', 'groundtruth', 'omega_gt')}
    print(f"\n==================== {ds}   (files: {', '.join(k for k, v in has.items() if v)})")
    imu = load_imu(paths['imu']) if has['imu'] else None
    gt = load_groundtruth(paths['groundtruth']) if has['groundtruth'] else None
    om = load_omega_gt(paths['omega_gt']) if has['omega_gt'] else None
    stats = [time_stats(n, a[:, 0]) for n, a in (('imu', imu), ('groundtruth', gt), ('omega_gt', om)) if a is not None]
    print_files(stats)

    ranges = [(s['t0'], s['t1']) for s in stats]
    if not ranges:
        print('  no imu/groundtruth/omega_gt: nothing to compare')
        return []
    centres = common_grid(ranges, h)
    w_ref, src, _ = reference_omega(paths, centres, h)
    if imu is not None:
        t_imu, keep = np.unique(imu[:, 0], return_index=True)
        cum_imu = cumulative(t_imu, imu[keep, 4:7])
    print(f"  reference: {src}, window ±{h * 1e3:.0f} ms, grid {centres[0]:.2f} .. {centres[-1]:.2f} s")

    rows = []
    for name, lo, hi in scopes(centres, DATASET_SEGMENTS.get(ds, [])):
        row = {'dataset': ds, 'scope': name, 't_lo': lo, 't_hi': hi, 'reference': src}
        mask = (centres >= lo) & (centres <= hi)
        if imu is not None and w_ref is not None and mask.sum() > 20:
            r = gyro_vs_ref(t_imu, cum_imu, centres, h, w_ref, mask)
            row.update({k: r[k] for k in ('lag_ms', 'rms_lag0', 'rms_best', 'n')})
            row.update({f'r_{c}': r['corr'][a, a] for a, c in enumerate('xyz')})
            row.update({f'scale_{c}': r['scale'][a] for a, c in enumerate('xyz')})
            row.update({f'bias_{c}': r['bias'][a] for a, c in enumerate('xyz')})
            if name == 'all':
                print('  axis correlation, gyro (rows) vs reference (cols), at the best lag:')
                print('   ' + np.array2string(r['corr'], precision=3).replace('\n', '\n   '))
        if has['events'] and name not in ('all', 'first_third', 'last_third'):
            speed = speed_from_gyro(imu) if imu is not None else speed_from_ref(centres, w_ref)
            row.update(event_rate_lag(event_times(paths['events'], lo, hi), lo, hi, speed))
        rows.append(row)
        print_row(row)
    return rows


def default_datasets():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def main():
    sys.stdout.reconfigure(encoding='utf-8')   # ω, °, → in the output, also on a cp1252 console
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--datasets', nargs='*', default=None, help='default: every folder in data/')
    ap.add_argument('--halfwidth', type=float, default=GT_DIFF_HALFWIDTH,
                    help='half-width h (s) of the ω window; default = eval_io.GT_DIFF_HALFWIDTH')
    ap.add_argument('--out', default=os.path.join('experiments', 'time_sync'))
    args = ap.parse_args()

    rows = []
    for ds in args.datasets or default_datasets():
        rows += check_dataset(ds, args.halfwidth)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, 'summary.csv')
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)

    print('\n==================== summary (whole recording)')
    cols = [c for c in ('dataset', 'reference', 'lag_ms', 'rms_lag0', 'rms_best', 'bias_x', 'bias_y', 'bias_z') if c in df]
    print(df[df['scope'] == 'all'][cols].round(2).to_string(index=False))
    if 'lag_ms' in df:
        drift = df[df['scope'].isin(['first_third', 'last_third'])].pivot(index='dataset', columns='scope', values='lag_ms')
        print('\nlag (ms), first vs last third (a difference = clock drift):')
        print(drift.round(1).to_string())
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
