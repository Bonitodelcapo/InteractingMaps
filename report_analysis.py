"""
Extra report analyses (D5) - pure re-analysis, NO new simulation runs.

Everything here is derived from the per-frame tracking.csv files that
`evaluation.py --exp 2 / --exp 12` already wrote, so it can be re-run as often
as you like while the big batch is still going.

Figures (results/report/analysis/):
  1 error_vs_time_*.png    error and cumulative angular drift vs time
  2 error_decomposition.png  direction error vs scale (beta) error per model
  3 runtime_tradeoff.png     accuracy vs cost per model
  4 error_vs_omega.png       error as a function of |omega_GT|
  5 gt_noise_floor.png       IMU-vs-mocap disagreement = measurable error floor

Usage:
    python report_analysis.py --all
    python report_analysis.py --all --n-frames 25     # only runs of that length
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = os.path.join('results', 'report', 'analysis')
MODEL_ORDER = ['cook', 'thesis', 'thesis_imu', 'thesis_cmax', 'thesis_cmax_v2']
COLORS = {'cook': 'tab:gray', 'thesis': 'tab:orange', 'thesis_imu': 'tab:green',
          'thesis_cmax': 'tab:blue', 'thesis_cmax_v2': 'tab:purple'}


# --------------------------------------------------------------------------
def load_all(root='results', n_frames=None):
    """Collect every tracking.csv under results/<dataset>/<model>/<segdir>/."""
    rows = []
    for path in glob.glob(os.path.join(root, '*', '*', '*', 'tracking.csv')):
        parts = os.path.normpath(path).split(os.sep)
        try:
            dataset, model, segdir = parts[-4], parts[-3], parts[-2]
        except IndexError:
            continue
        if model not in MODEL_ORDER:
            continue
        if 'oat_' in segdir:          # OAT sweeps belong to exp 13, not here
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty or 'err_deg_s' not in df.columns:
            continue
        if n_frames is not None and len(df) != n_frames:
            continue
        df['dataset'] = dataset
        df['model'] = model
        df['segment_id'] = segdir.split('_t')[0]
        df['run_dir'] = os.path.dirname(path)
        rows.append(df)
    if not rows:
        raise SystemExit(f"No tracking.csv found under {root}/. Run exp 12 first.")
    out = pd.concat(rows, ignore_index=True)
    out['model'] = pd.Categorical(out['model'], categories=MODEL_ORDER, ordered=True)
    return out


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, name)
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"  wrote {p}")


def _gt_mag(df):
    return np.linalg.norm(df[['gt_wx', 'gt_wy', 'gt_wz']].values, axis=1)


# --------------------------------------------------------------------------
def fig_error_vs_time(df, max_panels=6):
    """
    Error and CUMULATIVE angular drift vs time, all models overlaid.

    The cumulative curve (integral of |omega_est - omega_ref| dt, in degrees) is
    the clearest way to separate a model that tracks from one that diverges:
    a tracking model's drift grows slowly and linearly, a diverging one bends up.
    """
    pairs = list(df.groupby(['dataset', 'segment_id'], observed=True).groups)[:max_panels]
    if not pairs:
        return
    n = len(pairs)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 6.4), squeeze=False)
    for j, (ds, seg) in enumerate(pairs):
        sub = df[(df['dataset'] == ds) & (df['segment_id'] == seg)]
        for m, g in sub.groupby('model', observed=True):
            g = g.sort_values('time')
            t = g['time'].values - g['time'].values[0]
            e = g['err_deg_s'].values
            dt = np.gradient(t) if len(t) > 1 else np.array([0.02])
            axes[0][j].plot(t, e, label=m, color=COLORS.get(m), lw=1.3)
            axes[1][j].plot(t, np.cumsum(e * dt), label=m, color=COLORS.get(m), lw=1.3)
        axes[0][j].set_title(f'{ds}\n{seg}', fontsize=10)
        axes[0][j].set_ylabel('error (deg/s)')
        axes[1][j].set_ylabel('cumulative drift (deg)')
        axes[1][j].set_xlabel('time in segment (s)')
        for a in (axes[0][j], axes[1][j]):
            a.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    fig.suptitle('Per-frame error and accumulated angular drift', fontsize=12)
    _save(fig, 'error_vs_time.png')


def fig_error_decomposition(df):
    """
    Split the error into a DIRECTION part and a SCALE part.

        direction part = |w_ref| * 2 sin(dir_err / 2)     (rotating w_est onto w_ref)
        scale part     = |w_ref| * |1/beta - 1|           (beta = |w_ref|/|w_est|)

    This is the quantitative form of the beta-ambiguity claim: without an
    external anchor the network should recover the flow DIRECTION but not its
    absolute scale, so the scale bar should dominate for the pure-vision models
    and shrink for the IMU/CMax-anchored ones.
    """
    d = df.copy()
    mag = _gt_mag(d)
    d['dir_part'] = mag * 2 * np.sin(np.radians(d['dir_err_deg'].values) / 2)
    beta = d['beta'].replace(0, np.nan).values
    d['scale_part'] = mag * np.abs(1.0 / beta - 1.0)
    g = d.groupby('model', observed=True)[['dir_part', 'scale_part']].mean()
    g = g.reindex([m for m in MODEL_ORDER if m in g.index])
    g_deg = np.degrees(g)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(g_deg))
    ax.bar(x - 0.2, g_deg['dir_part'], 0.4, label='direction error', color='tab:orange')
    ax.bar(x + 0.2, g_deg['scale_part'], 0.4, label='scale (beta) error', color='tab:blue')
    ax.set_xticks(x); ax.set_xticklabels(g_deg.index, rotation=15)
    ax.set_ylabel('error contribution (deg/s)')
    ax.set_title('Error decomposition: direction vs scale\n'
                 '(scale dominating = the beta ambiguity)', fontsize=11)
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    _save(fig, 'error_decomposition.png')
    return g_deg


def fig_runtime_tradeoff(df):
    """Accuracy against compute cost. Needs the t_frame_s column."""
    if 't_frame_s' not in df.columns:
        print("  (runtime: no t_frame_s column - skipping)")
        return
    g = df.groupby('model', observed=True).agg(
        err=('err_deg_s', 'mean'), t=('t_frame_s', 'mean')).dropna()
    g = g.reindex([m for m in MODEL_ORDER if m in g.index])
    if g.empty:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(range(len(g)), g['t'], color=[COLORS.get(m) for m in g.index])
    ax[0].set_xticks(range(len(g))); ax[0].set_xticklabels(g.index, rotation=15)
    ax[0].set_ylabel('mean time per frame (s)')
    ax[0].set_title('Compute cost'); ax[0].grid(alpha=0.3, axis='y')
    for m in g.index:
        ax[1].scatter(g.loc[m, 't'], g.loc[m, 'err'], s=90,
                      color=COLORS.get(m), label=m)
    ax[1].set_xlabel('mean time per frame (s)')
    ax[1].set_ylabel('mean error (deg/s)')
    ax[1].set_title('Accuracy vs cost (lower-left is better)')
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    _save(fig, 'runtime_tradeoff.png')
    return g


def fig_error_vs_omega(df, nbins=6):
    """Does accuracy degrade at high rotation rate? Pools all frames."""
    d = df.copy()
    d['gt_mag'] = _gt_mag(d)
    d = d[d['gt_mag'] > 1e-6]
    if d.empty:
        return
    edges = np.linspace(d['gt_mag'].min(), d['gt_mag'].max(), nbins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for m, g in d.groupby('model', observed=True):
        idx = np.digitize(g['gt_mag'].values, edges) - 1
        idx = np.clip(idx, 0, nbins - 1)
        means = [g['err_deg_s'].values[idx == b].mean() if (idx == b).any() else np.nan
                 for b in range(nbins)]
        ax.plot(centres, means, 'o-', label=m, color=COLORS.get(m))
    ax.set_xlabel('|omega_GT|  (rad/s)')
    ax.set_ylabel('mean error (deg/s)')
    ax.set_title('Error vs rotation rate')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    _save(fig, 'error_vs_omega.png')


def fig_gt_noise_floor(df):
    """
    IMU-vs-reference disagreement -- a LOWER BOUND on measurable error.

    For the RPG sets the reference comes from differencing mocap quaternions, so
    the gyro and the reference disagree by a non-trivial amount that no model can
    beat. ECRot's reference is the direct twist, so its floor is far lower; that
    contrast is what makes the synthetic numbers worth reporting.
    """
    need = {'imu_wx', 'gt_wx'}
    if not need.issubset(df.columns):
        print("  (noise floor: missing imu_/gt_ columns - skipping)")
        return
    d = df.dropna(subset=['imu_wx', 'gt_wx']).copy()
    if d.empty:
        return
    imu = d[['imu_wx', 'imu_wy', 'imu_wz']].values
    gt = d[['gt_wx', 'gt_wy', 'gt_wz']].values
    d['floor'] = np.degrees(np.linalg.norm(imu - gt, axis=1))
    g = d.groupby(['dataset', 'ref_source'], observed=True)['floor'].mean().reset_index()
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"{r['dataset']}\n({r['ref_source']})" for _, r in g.iterrows()]
    colors = ['tab:green' if r['ref_source'] == 'omega_direct' else 'tab:red'
              for _, r in g.iterrows()]
    ax.bar(range(len(g)), g['floor'], color=colors)
    ax.set_xticks(range(len(g))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('mean |gyro - reference|  (deg/s)')
    ax.set_title('Ground-truth noise floor\n'
                 '(green = clean direct omega; red = differenced mocap poses)',
                 fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    _save(fig, 'gt_noise_floor.png')
    return g


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Report analyses from tracking.csv')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--root', default='results')
    ap.add_argument('--n-frames', type=int, default=None,
                    help='Only include runs with exactly this many frames')
    args = ap.parse_args()

    df = load_all(args.root, n_frames=args.n_frames)
    print(f"  loaded {len(df)} frames from "
          f"{df['run_dir'].nunique()} runs, "
          f"{df['dataset'].nunique()} datasets, {df['model'].nunique()} models")

    for fn in (fig_error_vs_time, fig_error_decomposition, fig_runtime_tradeoff,
               fig_error_vs_omega, fig_gt_noise_floor):
        try:
            fn(df)
        except Exception as e:
            # Never let one figure kill the rest - this is run repeatedly
            # against partial results while the batch is still going.
            print(f"  ({fn.__name__} skipped: {type(e).__name__}: {e})")


if __name__ == '__main__':
    main()
