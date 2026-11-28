"""
check_time_sync.py — clock offset and drift between the event stream, the IMU gyro and the
ω reference, with confidence intervals and an identifiability check. Prints the findings per
dataset and writes one CSV. Design: docs/superpowers/specs/2026-11-28-time-alignment-design.md.

Signals (all body-frame ω, rad/s):
  events  ω_ev  CMax on event windows centred at known times (the reference clock: µs stamps,
                µs latency). angular_velocity.CMaxAngularVelocity, t_ref = window centre.
  gyro    ω_g   imu.txt, 1 kHz.
  ref     ω_r   ECD: OptiTrack poses, centred difference of neighbouring samples (zero phase).
                ECRot: omega_gt.txt (exact).

Model, per pair (d = denser signal, s = sparser), fitted on the sparse timestamps:
    s(t) = A · d(t + δ(t)) + β,     δ(t) = δ0 + ε·(t − t̄)
A (free 3x3: axis misalignment + scale) and β (bias) are solved in closed form for every (δ0, ε)
(variable projection); (δ0, ε) by a grid on δ0 then Nelder–Mead; 95 % intervals by a moving-block
bootstrap. A robust refit drops samples whose residual exceeds median + 5·1.4826·MAD.
Both signals of a pair get the same zero-phase low-pass (default 20 Hz) first: zero phase cannot move
a lag, and without it linear interpolation of a noisy signal (less noisy half-way between samples)
pulls the fit to half-sample shifts. The filter check shows the lag does not depend on the cutoff.

Identifiability: if the motion is too smooth, a time shift is (nearly) the same as a change of A and
β, and the data do not determine the lag. A 95 % profile-likelihood interval of δ0 (ε, A, β re-fitted
at each δ0; likelihood ratio with the effective sample size of the autocorrelated residuals) shows
it: an interval that runs to the edge of the ±PROFILE_HALF_MS scan marks the lag "not identified".
(A Gauss–Newton variance-inflation factor was tried first and rejected: it reads shape information
from the noise of the interpolated signal and calls a flat profile identified.) The variant with
A = I (camera and IMU axes coincide to ≈ 0.7° per the ECD paper's Kalibr result) is reported too.

Reported as "X lags Y by L ms" (L = −δ when X is the sparse signal) and drift in ms/min
(the change of L per minute).

Validation, per dataset:
  closure    L(ev,ref) = L(ev,gyro) − L(ref,gyro)   within the intervals
  injection  ref (and events) timestamps moved by +3 ms + 2 ms/min: L must grow by exactly that
  filter     cutoffs 40/10/5 Hz besides the default: L must not change
  spectral   phase slope of the cross-spectrum per frequency band (independent of the time fit)
  stamps     ECD only: fit with the given vs regularised (5 ms) mocap timestamps; lower RMS = truer
  external   the ECD paper reports the IMU ≈ 2.4 ms behind the events: L(gyro,ev) ≈ +2.4
  ECRot      exact ω and a simulated IMU: every identified L must be 0 (tests the event-ω timing too)

Run from the repo root (reads data/, writes --out). Real runs on the cluster only:
    python check_time_sync.py                              # all datasets in data/
    python check_time_sync.py --datasets poster_rotation --jobs 32
The event-ω estimate is the slow part (≈ 0.7 s per CMax window, parallel over --jobs); it is
cached in --out and reused by later runs with the same range and window settings.
"""

import argparse
import os
import sys
import time
from io import BytesIO
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.linalg import polar
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import butter, coherence, csd, filtfilt
from scipy.spatial.transform import Rotation

from config import get_dataset_paths
from interacting_maps.angular_velocity import CMaxAngularVelocity
from interacting_maps.data_loader import CameraCalibration, undistort_events
from interacting_maps.eval_io import load_imu, load_groundtruth, load_omega_gt
from interacting_maps.style import to_deg_s

SENSOR = (180, 240)                         # H, W of every dataset here (config sensor_size)
MARGIN = 0.15                               # s kept free at both ends for shifts up to ±GRID_MS
GRID_MS = np.arange(-60.0, 60.25, 0.25)     # δ0 grid (ms) before the Nelder–Mead refinement
INJECT = (3.0, 2.0)                         # injected offset (ms) and drift (ms/min)
LOWPASS_HZ = 20.0                           # common zero-phase low-pass on both signals of a pair
CUTOFFS_HZ = (40.0, 10.0, 5.0)              # other cutoffs in the filter check (lag must not move)
PROFILE_HALF_MS = 30.0                      # half-width of the profile-likelihood scan around the optimum
BANDS_HZ = ((0.5, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 40.0))
SPECTRAL_FS = 200.0                         # Hz, common grid of the cross-spectrum


# ================================================================ inputs
def time_stats(name, t):
    """Range, median rate, jitter and gaps (> 2x the median step) of one timestamp column."""
    d = np.diff(t)
    med = np.median(d)
    return {'file': name, 't0': t[0], 't1': t[-1], 'rate_hz': 1.0 / med, 'jitter_us': np.std(d) * 1e6,
            'gaps': int((d > 2 * med).sum()), 'max_gap_ms': d.max() * 1e3}


def regularised(t, step):
    """Timestamps snapped to the nominal grid t0 + k·step (the test of the given mocap stamps)."""
    return t[0] + np.round((t - t[0]) / step) * step


def mocap_omega(gt, stamps=None):
    """Body-frame ω at each pose with two in-time neighbours: rotvec(R(i−1)ᵀ·R(i+1)) / (t(i+1) − t(i−1)).

    Centred, so zero phase. Samples next to a gap (> 1.5x the median step) are dropped."""
    t = gt[:, 0] if stamps is None else stamps
    keep = np.r_[True, np.diff(t) > 0]
    t, q = t[keep], gt[keep, 4:8]
    rot = Rotation.from_quat(q)                 # (qx,qy,qz,qw); Hamilton matrix = R_wc (see eval_io)
    step = np.median(np.diff(t))
    i = np.arange(1, len(t) - 1)
    ok = (t[i] - t[i - 1] < 1.5 * step) & (t[i + 1] - t[i] < 1.5 * step)
    i = i[ok]
    w = (rot[i - 1].inv() * rot[i + 1]).as_rotvec() / (t[i + 1] - t[i - 1])[:, None]
    return t[i], w


def seek_time(f, size, t):
    """Byte offset of the first line of a sorted events.txt with timestamp >= t (to 4 KB)."""
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


def read_events(path, t_lo, t_hi):
    """Events (N, 4) [t, x, y, pol] in [t_lo, t_hi), float64, without loading the whole file."""
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        a = seek_time(f, size, t_lo)
        f.seek(min(seek_time(f, size, t_hi) + 8192, size))   # the search stops up to 4 KB early
        f.readline()
        b = f.tell()
        f.seek(a)
        buf = f.read(b - a)
    if not buf:
        return np.zeros((0, 4))
    ev = pd.read_csv(BytesIO(buf), sep=r'\s+', header=None, dtype=np.float64).values[:, :4]
    return ev[(ev[:, 0] >= t_lo) & (ev[:, 0] < t_hi)]


def _chunk_omega(job):
    """ω_ev for one chunk of window centres, warm-started from `seed`, then from the previous window.

    A window where CMax does not move from its warm start gives no estimate (NaN) and is counted
    with a negative event count."""
    paths, centres, win, min_events, seed = job
    calib = CameraCalibration(paths['calib'])
    est = CMaxAngularVelocity(*SENSOR, calib.fx, calib.fy, calib.cx, calib.cy, use_polarity=True)
    ev = undistort_events(read_events(paths['events'], centres[0] - win / 2, centres[-1] + win / 2), calib)
    omega = np.full((len(centres), 3), np.nan)
    n_ev = np.zeros(len(centres), dtype=int)
    prev = np.asarray(seed, float)
    for j, c in enumerate(centres):
        lo, hi = np.searchsorted(ev[:, 0], [c - win / 2, c + win / 2])
        n_ev[j] = hi - lo
        if n_ev[j] >= min_events:
            w = est.estimate(ev[lo:hi], t_ref=c, omega_init=prev)
            if np.allclose(w, prev, atol=1e-9):
                n_ev[j] = -n_ev[j]
            else:
                omega[j] = prev = w
    return omega, n_ev


def events_omega(paths, t_lo, t_hi, win, stride, min_events, jobs, chunk=1.0):
    """ω_ev (rad/s) by CMax on windows [c − win/2, c + win/2) for centres c every `stride`.

    Warm starts never use the other sensors: a sequential pass estimates the first window of every
    `chunk` seconds from the previous chunk's (zeros at the start); the chunks then run in parallel,
    each window starting from the previous one. Windows with fewer than min_events are NaN."""
    centres = np.arange(t_lo + win / 2, t_hi - win / 2, stride)
    groups = [g for g in np.split(centres, np.searchsorted(centres, np.arange(t_lo + chunk, t_hi, chunk))) if len(g)]
    seeds, prev, t0 = [], np.zeros(3), time.time()
    for g in groups:
        w, _ = _chunk_omega((paths, g[:1], win, min_events, prev))
        prev = w[0] if np.isfinite(w[0, 0]) else prev
        seeds.append(prev)
    print(f'    events ω: {len(groups)} chunk seeds in {time.time() - t0:.0f} s; '
          f'{len(centres)} windows on {jobs} jobs', flush=True)
    work = [(paths, g, win, min_events, sd) for g, sd in zip(groups, seeds)]
    if jobs > 1:
        with Pool(jobs) as pool:
            res = pool.map(_chunk_omega, work)
    else:
        res = [_chunk_omega(j) for j in work]
    print(f'    events ω done in {time.time() - t0:.0f} s', flush=True)
    return centres, np.vstack([r[0] for r in res]), np.concatenate([r[1] for r in res])


def cached_events_omega(paths, ds, out, t_lo, t_hi, win, stride, min_events, jobs):
    """events_omega, cached in out/<ds>_events_omega_w<win>_s<stride>.npz (reused if the range matches)."""
    path = os.path.join(out, f'{ds}_events_omega_w{win * 1e3:g}ms_s{stride * 1e3:g}ms.npz')
    if os.path.exists(path):
        z = np.load(path)
        if np.isclose(z['t_lo'], t_lo) and np.isclose(z['t_hi'], t_hi) and int(z['min_events']) == min_events:
            return z['t'], z['w'], z['n']
    t, w, n = events_omega(paths, t_lo, t_hi, win, stride, min_events, jobs)
    np.savez(path, t=t, w=w, n=n, t_lo=t_lo, t_hi=t_hi, min_events=min_events)
    return t, w, n


# ================================================================ the joint fit
def shifted(t_d, d, t_eval):
    """Dense signal d linearly interpolated at t_eval (per axis)."""
    return np.stack([np.interp(t_eval, t_d, d[:, k]) for k in range(3)], 1)


def linear_map(Y, X, free_A=True):
    """Closed-form X ≈ Y·Aᵀ + β: returns A (3x3), β (3,), residuals (n, 3). free_A=False fixes A = I."""
    if not free_A:
        beta = np.mean(X - Y, axis=0)
        return np.eye(3), beta, X - Y - beta
    D = np.column_stack([Y, np.ones(len(Y))])
    coef = np.linalg.lstsq(D, X, rcond=None)[0]
    return coef[:3].T, coef[3], X - D @ coef


def delta_s(p, t, t_mid):
    """δ(t) in seconds from p = (δ0 in ms, ε in ms/min)."""
    return p[0] * 1e-3 + p[1] * 1e-3 / 60.0 * (t - t_mid)


def residuals(p, t_d, d, t_s, s, t_mid, free_A=True):
    return linear_map(shifted(t_d, d, t_s + delta_s(p, t_s, t_mid)), s, free_A)


def sse(p, t_d, d, t_s, s, t_mid, free_A=True):
    return float(np.sum(residuals(p, t_d, d, t_s, s, t_mid, free_A)[2] ** 2))


def refine(p0, t_d, d, t_s, s, t_mid, free_A=True):
    simplex = np.array([p0, p0 + (0.5, 0.0), p0 + (0.0, 1.0)])   # steps of 0.5 ms and 1 ms/min
    res = minimize(sse, p0, args=(t_d, d, t_s, s, t_mid, free_A), method='Nelder-Mead',
                   options=dict(xatol=1e-3, fatol=0.0, maxiter=800, initial_simplex=simplex))
    return res.x


def estimate(t_d, d, t_s, s, t_mid, p0=None, free_A=True):
    """(δ0 ms, ε ms/min) minimising the SSE: grid on δ0 at ε = 0, then Nelder–Mead in both."""
    if p0 is None:
        vals = [sse((g, 0.0), t_d, d, t_s, s, t_mid, free_A) for g in GRID_MS]
        p0 = (GRID_MS[int(np.argmin(vals))], 0.0)
    return refine(np.asarray(p0, float), t_d, d, t_s, s, t_mid, free_A)


def usable(t_d, d, t_s, s):
    """Sparse samples whose shifted dense value exists and is finite."""
    m = (t_s > t_d[0] + MARGIN) & (t_s < t_d[-1] - MARGIN) & np.all(np.isfinite(s), 1)
    fin = np.all(np.isfinite(d), 1)
    return m, t_d[fin], d[fin]


def robust_mask(r):
    """Samples with residual norm ≤ median + 5·1.4826·MAD."""
    n = np.linalg.norm(r, axis=1)
    med = np.median(n)
    return n <= med + 5 * 1.4826 * np.median(np.abs(n - med))


def block_bootstrap(t_d, d, t_s, s, t_mid, p_hat, n_boot, block_s, rng, free_A=True):
    """Moving-block bootstrap of (δ0, ε): blocks of block_s seconds (at least 10 blocks), refit from p_hat."""
    n = len(t_s)
    nb = min(max(1, int(round(block_s / np.median(np.diff(t_s))))), max(1, n // 10))
    out = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - nb + 1, int(np.ceil(n / nb)))
        idx = np.sort(np.concatenate([np.arange(a, a + nb) for a in starts])[:n])
        out.append(refine(p_hat, t_d, d, t_s[idx], s[idx], t_mid, free_A))
    return np.array(out)


def effective_n(r):
    """Effective number of independent residuals: n / (1 + 2·Σ ρ_k), summed while ρ_k > 0.05
    (per axis; the smallest of the three, i.e. the most conservative)."""
    n = len(r)
    out = []
    for k in range(3):
        x = r[:, k] - r[:, k].mean()
        ac = np.fft.irfft(np.abs(np.fft.rfft(x, 2 * n)) ** 2)[:n]
        rho = ac / ac[0]
        cut = np.argmax(rho[1:] < 0.05) + 1 if np.any(rho[1:] < 0.05) else n // 10
        out.append(n / (1 + 2 * rho[1:cut].sum()))
    return float(max(min(out), 2.0))


def profile_interval(t_d, d, t_s, s, t_mid, p_hat, free_A=True):
    """95 % profile-likelihood interval of δ0 (ms) and whether it is closed within ±PROFILE_HALF_MS.

    For each δ0 on a grid around the optimum, ε (and A, β in closed form) are re-fitted; δ0 is kept
    while the likelihood-ratio statistic n_eff·ln(SSE(δ0)/SSE_min) ≤ 3.84 (χ², 1 dof). n_eff accounts
    for the autocorrelation of the residuals. An interval that reaches the edge of the scan means the
    data do not determine the lag (e.g. a time shift is absorbed by A and β on smooth motion)."""
    sse_min = sse(p_hat, t_d, d, t_s, s, t_mid, free_A)
    n_eff = effective_n(residuals(p_hat, t_d, d, t_s, s, t_mid, free_A)[2])
    grid = p_hat[0] + np.arange(-PROFILE_HALF_MS, PROFILE_HALF_MS + 0.25, 0.5)
    prof = np.array([minimize_scalar(lambda e: sse((g, e), t_d, d, t_s, s, t_mid, free_A),
                                     bounds=(p_hat[1] - 30.0, p_hat[1] + 30.0), method='bounded',
                                     options=dict(xatol=0.05)).fun for g in grid])
    stat = n_eff * np.log(np.maximum(prof, sse_min) / sse_min) - 3.84      # ≤ 0 inside the interval
    i0 = int(np.argmin(stat))
    lo, hi = i0, i0
    while lo > 0 and stat[lo - 1] <= 0:
        lo -= 1
    while hi < len(grid) - 1 and stat[hi + 1] <= 0:
        hi += 1
    closed = bool(lo > 0 and hi < len(grid) - 1)
    # edges by linear interpolation of the statistic between the last inside and first outside point
    e_lo = grid[lo] if lo == 0 else grid[lo - 1] + (grid[lo] - grid[lo - 1]) * stat[lo - 1] / (stat[lo - 1] - stat[lo])
    e_hi = grid[hi] if hi == len(grid) - 1 else grid[hi] + (grid[hi + 1] - grid[hi]) * (-stat[hi]) / (stat[hi + 1] - stat[hi])
    return (-e_hi, -e_lo), closed, n_eff


def describe_map(A, beta, r):
    """Misalignment angle (deg) and scales of A (polar decomposition), bias and RMS in °/s."""
    U, P = polar(A)
    angle = (np.nan if np.linalg.det(U) < 0
             else float(np.degrees(np.linalg.norm(Rotation.from_matrix(U).as_rotvec()))))
    return {'angle_deg': angle, 'scales': np.sort(np.linalg.eigvalsh(P)), 'bias': to_deg_s(beta),
            'rms': float(np.sqrt(np.mean(np.sum(to_deg_s(r) ** 2, 1))))}


def fit_pair(t_d, d, t_s, s, n_boot=0, block_s=1.0, rng=None, robust=True, free_A=True, profile=True):
    """The joint fit on one pair: lag/drift (as "sparse lags dense"), intervals, map, identifiability."""
    m, t_d, d = usable(t_d, d, t_s, s)
    t_s, s = t_s[m], s[m]
    t_mid = float(np.mean(t_s))
    p = estimate(t_d, d, t_s, s, t_mid, free_A=free_A)
    n_drop = 0
    if robust:
        keep = robust_mask(residuals(p, t_d, d, t_s, s, t_mid, free_A)[2])
        n_drop = int((~keep).sum())
        t_s, s = t_s[keep], s[keep]
        p = estimate(t_d, d, t_s, s, t_mid, p0=p, free_A=free_A)
    A, beta, r = residuals(p, t_d, d, t_s, s, t_mid, free_A)
    out = {'lag_ms': -p[0], 'drift_ms_min': -p[1], 'n': len(t_s), 'n_dropped': n_drop, 't_mid': t_mid,
           'edge': bool(abs(p[0]) >= GRID_MS.max() - 0.5), **describe_map(A, beta, r)}
    if profile:
        (out['prof_lo'], out['prof_hi']), out['identified'], out['n_eff'] = profile_interval(
            t_d, d, t_s, s, t_mid, p, free_A)
    if n_boot:
        b = block_bootstrap(t_d, d, t_s, s, t_mid, p, n_boot, block_s, rng or np.random.default_rng(0), free_A)
        (out['lag_lo'], out['lag_hi']), (out['drift_lo'], out['drift_hi']) = (
            np.percentile(-b[:, 0], [2.5, 97.5]), np.percentile(-b[:, 1], [2.5, 97.5]))
    return out


# ================================================================ checks
def lowpass(t, w, cutoff):
    """Zero-phase 4th-order Butterworth on a signal resampled to its median rate.

    Grid points inside gaps of the original sampling (> 1.5 steps from any sample) are dropped."""
    fin = np.all(np.isfinite(w), 1)
    t, w = t[fin], w[fin]
    step = np.median(np.diff(t))
    tu = np.arange(t[0], t[-1], step)
    b, a = butter(4, min(cutoff * 2 * step, 0.99))
    wf = filtfilt(b, a, shifted(t, w, tu), axis=0)
    j = np.clip(np.searchsorted(t, tu), 1, len(t) - 1)
    near = np.minimum(np.abs(t[j] - tu), np.abs(t[j - 1] - tu)) <= 1.5 * step
    return tu[near], wf[near]


def injected(t, c0_ms, drift_ms_min):
    """Timestamps moved later by c0 + drift·(t − t̄): the sparse signal then lags by that much more."""
    return t + c0_ms * 1e-3 + drift_ms_min * 1e-3 / 60.0 * (t - t.mean())


def spectral_lag(t_d, d, t_s, s, fs=SPECTRAL_FS, bands=BANDS_HZ):
    """'sparse lags dense' (ms) per band from the coherence-weighted phase slope of the cross-spectrum.

    Both signals go on one fs grid by the same linear interpolation (zero phase). With
    s(t) = d(t + δ), csd(d, s) has phase +2π·f·δ, so lag = −slope / 2π. Bins with coherence < 0.8
    are ignored; a band with fewer than 4 bins is NaN."""
    t = np.arange(max(t_d[0], t_s[0]) + MARGIN, min(t_d[-1], t_s[-1]) - MARGIN, 1 / fs)
    fin_s = np.all(np.isfinite(s), 1)
    x, y = shifted(t_d, d, t), shifted(t_s[fin_s], s[fin_s], t)
    nper = int(min(len(t) // 8, 2 ** int(np.log2(fs * 10))))
    fs_, ph_, wt_ = [], [], []
    for k in range(3):
        f, pxy = csd(x[:, k], y[:, k], fs=fs, nperseg=nper)
        _, coh = coherence(x[:, k], y[:, k], fs=fs, nperseg=nper)
        fs_.append(f)
        ph_.append(np.unwrap(np.angle(pxy)))
        wt_.append(np.where(coh >= 0.8, coh / np.maximum(1 - coh, 1e-6), 0.0))
    f, ph, wt = np.concatenate(fs_), np.concatenate(ph_), np.concatenate(wt_)
    out = {}
    for lo, hi in bands:
        sel = (f >= lo) & (f < hi) & (wt > 0)
        if sel.sum() < 4:
            out[f'{lo:g}-{hi:g}Hz'] = np.nan
            continue
        sw = np.sqrt(wt[sel])
        D = np.column_stack([f[sel], np.ones(sel.sum())]) * sw[:, None]
        slope = np.linalg.lstsq(D, ph[sel] * sw, rcond=None)[0][0]
        out[f'{lo:g}-{hi:g}Hz'] = -slope / (2 * np.pi) * 1e3
    return out


# ================================================================ per dataset
def fmt_fit(r):
    ci = f" [{r['lag_lo']:+.2f}, {r['lag_hi']:+.2f}]" if 'lag_lo' in r else ''
    cd = f" [{r['drift_lo']:+.2f}, {r['drift_hi']:+.2f}]" if 'drift_lo' in r else ''
    pr = f"  profile [{r['prof_lo']:+.2f}, {r['prof_hi']:+.2f}] (n_eff {r['n_eff']:.0f})" if 'prof_lo' in r else ''
    flags = ('  ** lag NOT identified (profile open) **' if r.get('identified') is False else '') + \
            ('  ** at grid edge **' if r['edge'] else '')
    return (f"lag {r['lag_ms']:+7.2f} ms{ci}{pr}  drift {r['drift_ms_min']:+6.2f} ms/min{cd}  "
            f"misalign {r['angle_deg']:.2f}°  scales {np.array2string(r['scales'], precision=3)}  "
            f"bias {np.array2string(r['bias'], precision=2)} °/s  RMS {r['rms']:.2f} °/s  "
            f"n {r['n']} (−{r['n_dropped']}){flags}")


def row(ds, pair, variant, r, **extra):
    out = {'dataset': ds, 'pair': pair, 'variant': variant}
    for k, v in r.items():
        if k in ('scales', 'bias'):
            out.update({f'{k}_{i}': x for i, x in enumerate(v)})
        else:
            out[k] = v
    return {**out, **extra}


def crop(sig, t_lo, t_hi):
    """(name, t, ω) restricted to [t_lo − MARGIN, t_hi + MARGIN]: every pair is fitted on one range."""
    m = (sig[1] >= t_lo - MARGIN) & (sig[1] <= t_hi + MARGIN)
    return (sig[0], sig[1][m], sig[2][m])


def signals(ds, paths, args):
    """(name, t, ω) for gyro, ref and events on their common range, and the raw poses (or None)."""
    imu = load_imu(paths['imu'])
    om = load_omega_gt(paths['omega_gt']) if os.path.exists(paths['omega_gt']) else None
    gt = load_groundtruth(paths['groundtruth']) if os.path.exists(paths['groundtruth']) else None
    for n, a in (('imu', imu), ('groundtruth', gt), ('omega_gt', om)):
        if a is not None:
            st = time_stats(n, a[:, 0])
            print(f"  {n:<12} {st['t0']:8.3f} .. {st['t1']:8.3f} s  {st['rate_hz']:7.1f} Hz  jitter "
                  f"{st['jitter_us']:8.1f} us  gaps {st['gaps']} (max {st['max_gap_ms']:.1f} ms)")
    if om is not None:                                  # exact rate wins (ECRot), as in eval_io
        t_u, k = np.unique(om[:, 0], return_index=True)
        ref = ('omega_gt', t_u, om[k, 1:4])
    else:
        ref = ('mocap',) + mocap_omega(gt)
    t_u, k = np.unique(imu[:, 0], return_index=True)
    gyro = ('gyro', t_u, imu[k, 4:7])
    t_lo = max(gyro[1][0], ref[1][0]) + MARGIN
    t_hi = min(gyro[1][-1], ref[1][-1]) - MARGIN
    if args.t_range:
        t_lo, t_hi = max(t_lo, args.t_range[0]), min(t_hi, args.t_range[1])
    if t_hi - t_lo < 1.0:
        return None
    gyro, ref = crop(gyro, t_lo, t_hi), crop(ref, t_lo, t_hi)
    t_e, w_e, n_e = cached_events_omega(paths, ds, args.out, t_lo, t_hi, args.event_win,
                                         args.event_stride, args.min_events, args.jobs)
    print(f"  events ω: {np.isfinite(w_e[:, 0]).sum()} of {len(t_e)} windows ({(n_e < 0).sum()} stuck at "
          f"the warm start; win {args.event_win * 1e3:g} ms, stride {args.event_stride * 1e3:g} ms, "
          f"median {np.median(np.abs(n_e)):.0f} events)")
    return gyro, ref, ('events', t_e, w_e), (gt if om is None else None)


def lowpassed(sig, cutoff):
    return (sig[0],) + lowpass(sig[1], sig[2], cutoff)


def check_dataset(ds, args):
    paths = get_dataset_paths(ds)
    if not (os.path.exists(paths['imu']) and os.path.exists(paths['events'])
            and (os.path.exists(paths['groundtruth']) or os.path.exists(paths['omega_gt']))):
        print(f'\n==================== {ds}: needs events, imu and a reference; skipped')
        return []
    print(f'\n==================== {ds}')
    sig = signals(ds, paths, args)
    if sig is None:
        print('  common range of gyro, reference and --t-range is under 1 s; skipped')
        return []
    gyro, ref, ev, gt = sig
    g, rf, e = (lowpassed(x, args.lowpass) for x in (gyro, ref, ev))
    print(f'  (both signals of every pair low-passed at {args.lowpass:g} Hz, zero phase)')
    rng = np.random.default_rng(args.seed)
    rows, L = [], {}
    for dense, sparse in ((g, rf), (g, e), (rf, e)):       # the sparse one is reported as lagging
        pair = f'{sparse[0]}~{dense[0]}'
        for variant, free_A in (('main', True), ('A_identity', False)):
            r = fit_pair(dense[1], dense[2], sparse[1], sparse[2], args.boot, args.block, rng, free_A=free_A)
            head = f'{sparse[0]:>8} lags {dense[0]:<8}' if free_A else f'{"(A = I)":>21}'
            print(f'  {head} {fmt_fit(r)}')
            rows.append(row(ds, pair, variant, r))
            if free_A:
                L[pair] = r

    a, b, c = L[f'events~{ref[0]}'], L['events~gyro'], L[f'{ref[0]}~gyro']
    gap = a['lag_ms'] - (b['lag_ms'] - c['lag_ms'])
    tol = float(np.sqrt(sum(((x['lag_hi'] - x['lag_lo']) / 2) ** 2 for x in (a, b, c)))) if args.boot else np.nan
    print(f'  closure  L(ev,{ref[0]}) − [L(ev,gyro) − L({ref[0]},gyro)] = {gap:+.2f} ms (combined 95 % ±{tol:.2f} ms)')
    rows.append({'dataset': ds, 'pair': 'closure', 'variant': 'check', 'lag_ms': gap, 'tol_ms': tol})
    print(f'  external gyro lags events by {-b["lag_ms"]:+.2f} ms (ECD paper: ≈ +2.4 ms)')

    for dense, sparse in ((g, rf), (g, e)):
        pair = f'{sparse[0]}~{dense[0]}'
        r = fit_pair(dense[1], dense[2], injected(sparse[1], *INJECT), sparse[2], profile=False)
        dl, dd = r['lag_ms'] - L[pair]['lag_ms'], r['drift_ms_min'] - L[pair]['drift_ms_min']
        print(f'  inject   {pair}: +{INJECT[0]:g} ms, +{INJECT[1]:g} ms/min → recovered {dl:+.3f} ms, {dd:+.3f} ms/min')
        rows.append(row(ds, pair, 'inject', r, expected_lag=INJECT[0], expected_drift=INJECT[1],
                        recovered_lag=dl, recovered_drift=dd))

    for fc in CUTOFFS_HZ:
        r = fit_pair(*lowpassed(gyro, fc)[1:], *lowpassed(ref, fc)[1:], profile=False)
        print(f'  lowpass  {fc:4.0f} Hz  {ref[0]} lags gyro by {r["lag_ms"]:+.2f} ms, drift '
              f'{r["drift_ms_min"]:+.2f} ms/min')
        rows.append(row(ds, f'{ref[0]}~gyro', f'lowpass_{fc:g}Hz', r))

    for dense, sparse in ((gyro, ref), (gyro, ev)):           # unfiltered: the spectrum weights by coherence
        sp = spectral_lag(dense[1], dense[2], sparse[1], sparse[2])
        print(f'  spectral {sparse[0]} lags {dense[0]}: ' + '  '.join(f'{k} {v:+.2f}' for k, v in sp.items()) + ' ms')
        rows.append({'dataset': ds, 'pair': f'{sparse[0]}~{dense[0]}', 'variant': 'spectral',
                     **{f'lag_{k}': v for k, v in sp.items()}})

    if gt is not None:                                         # given vs regularised mocap stamps
        step = np.median(np.diff(gt[:, 0]))
        reg = lowpassed(crop(('mocap',) + mocap_omega(gt, regularised(gt[:, 0], step)),
                             gyro[1][0] + MARGIN, gyro[1][-1] - MARGIN), args.lowpass)
        r = fit_pair(g[1], g[2], reg[1], reg[2], profile=False)
        print(f'  stamps   regularised to {step * 1e3:.1f} ms: RMS {r["rms"]:.3f} vs given '
              f'{L["mocap~gyro"]["rms"]:.3f} °/s, lag {r["lag_ms"]:+.2f} ms')
        rows.append(row(ds, 'mocap~gyro', 'regular_stamps', r))
    return rows


def default_datasets():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith('.'))


def main():
    sys.stdout.reconfigure(encoding='utf-8')   # ω, °, → in the output, also on a cp1252 console
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--datasets', nargs='*', default=None, help='default: every folder in data/')
    ap.add_argument('--boot', type=int, default=200, help='bootstrap replicates for the 95 %% intervals')
    ap.add_argument('--block', type=float, default=1.0, help='bootstrap block length (s)')
    ap.add_argument('--lowpass', type=float, default=LOWPASS_HZ, help='common zero-phase low-pass (Hz)')
    ap.add_argument('--event-win', type=float, default=0.02, help='CMax window (s)')
    ap.add_argument('--event-stride', type=float, default=0.01, help='CMax window spacing (s)')
    ap.add_argument('--min-events', type=int, default=1000)
    ap.add_argument('--t-range', type=float, nargs=2, default=None, help='restrict to [t0, t1] s')
    ap.add_argument('--jobs', type=int, default=os.cpu_count(), help='processes for the event-ω estimate')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.join('experiments', 'time_sync'))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for ds in args.datasets or default_datasets():
        rows += check_dataset(ds, args)
        pd.DataFrame(rows).to_csv(os.path.join(args.out, 'alignment.csv'), index=False)   # after each dataset
    df = pd.DataFrame(rows)
    if len(df) and 'variant' in df:
        m = df[df['variant'].isin(['main', 'A_identity'])]
        cols = [c for c in ('dataset', 'pair', 'variant', 'lag_ms', 'lag_lo', 'lag_hi', 'drift_ms_min',
                            'drift_lo', 'drift_hi', 'prof_lo', 'prof_hi', 'identified', 'rms') if c in m]
        print('\n==================== summary: X lags Y (ms, 95 % interval), drift (ms/min)')
        print(m[cols].round(2).to_string(index=False))
    print(f"\nwrote {os.path.join(args.out, 'alignment.csv')}")


if __name__ == '__main__':
    main()
