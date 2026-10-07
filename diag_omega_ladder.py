"""
diag_omega_ladder.py — ω-recovery oracle ladder (rungs 0 / 1a / 1b) over several
synthetic segments, plus comparison plots.

Run directly (NOT pytest). The compute path (--mode run) is cluster-ready: it
writes diag_ladder_results.csv and diag_ladder_frames.csv. The --mode plots path
is local and renders P1/P2/P3 from those CSVs. Read-only on the model: it only
injects ground-truth flow into q_F.

See docs/superpowers/specs/2026-10-07-omega-recovery-diagnostic-design.md and
docs/superpowers/plans/2026-10-07-omega-ladder-multisegment-plots.md.

Kept in sync with diag_omega_ladder.ipynb (generated from this file's `# %%`
cells; edit this file, then regenerate the notebook).
"""
# %%
import argparse
import csv
import glob
import os
import numpy as np
import yaml as _yaml

import config
from config import get_dataset_paths
from eval_io import load_imu, get_gyro_for_frame, load_omega_gt, omega_gt_at
from data_loader import EventFrameSequence
from interacting_maps.network_dissertation import InteractingMapsThesis
from interacting_maps.camera import build_R_normal_equations, solve_R_lstsq
from metrics import compute_metrics, curl_share

# Cap on frames per segment for a fast local smoke test; set from --limit.
LIMIT = None

SEGMENTS = [
    {'name': 'city_short',   'ds': 'ecrot_city',   't_start': 0.05,  'dt': 0.02, 'n_frames': 75,  'sensor_size': (180, 240)},
    {'name': 'city_long',    'ds': 'ecrot_city',   't_start': 0.05,  'dt': 0.02, 'n_frames': 250, 'sensor_size': (180, 240)},
    {'name': 'street_short', 'ds': 'ecrot_street', 't_start': 1.001, 'dt': 0.02, 'n_frames': 75,  'sensor_size': (180, 240)},
    {'name': 'street_long',  'ds': 'ecrot_street', 't_start': 1.001, 'dt': 0.02, 'n_frames': 250, 'sensor_size': (180, 240)},
]


def cap_to_coverage(seg, om):
    """Cap n_frames so the window stays within the available omega_gt extent."""
    n = seg['n_frames']
    if om is None:
        return n
    t_max = float(om[:, 0].max())
    max_f = int((t_max - seg['t_start']) / seg['dt'])
    if max_f < n:
        print(f"[warn] {seg['name']}: omega_gt covers to {t_max:.2f}s; "
              f"capping {n} -> {max_f} frames")
        n = max(max_f, 0)
    return n


def load_calibration(paths):
    """Network intrinsics + distortion, preferring a ROS camera_info *.yaml in
    the dataset dir (e.g. DAVIS240C-synthetic.yaml) over calib.txt.

    Returns {fx, fy, cx, cy, dist_coeffs, H, W, src}. dist_coeffs is None when
    all coefficients are zero (clean pinhole C), else the [k1,k2,p1,p2,k3] array
    in the same order calib.txt / build_kinematic_matrix use.
    """
    yamls = sorted(glob.glob(os.path.join(paths['data_dir'], '*.yaml')))
    if yamls:
        with open(yamls[0]) as f:
            y = _yaml.safe_load(f)
        K = y['camera_matrix']['data']           # row-major 3x3
        fx, fy, cx, cy = K[0], K[4], K[2], K[5]
        dist = np.asarray(
            y.get('distortion_coefficients', {}).get('data', []),
            dtype=np.float64)
        H, W = int(y['image_height']), int(y['image_width'])
        src = os.path.basename(yamls[0])
    else:
        from data_loader import CameraCalibration
        c = CameraCalibration(paths['calib'])
        fx, fy, cx, cy = c.fx, c.fy, c.cx, c.cy
        dist = np.asarray(getattr(c, 'dist', []), dtype=np.float64)
        H = W = None
        src = 'calib.txt'
    dist_coeffs = None if dist.size == 0 or not np.any(dist) else dist
    return {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
            'dist_coeffs': dist_coeffs, 'H': H, 'W': W, 'src': src}


def prep_segment_context(seg):
    """Load all per-segment data ONCE and precompute the frame windows.

    Returns a ctx dict shared read-only across rungs (each rung still rebuilds
    its own network), or None if the data is absent / the window is too short.
    """
    paths = get_dataset_paths(seg['ds'])
    if not os.path.exists(paths['events']):
        print(f"[skip] {seg['name']}: no data at {paths['events']}")
        return None
    calib = load_calibration(paths)
    om = load_omega_gt(paths.get('omega_gt'))
    n = cap_to_coverage(seg, om)
    if LIMIT is not None:
        n = min(n, LIMIT)
    if n < 2:
        print(f"[skip] {seg['name']}: too few frames after coverage cap")
        return None
    sensor = (calib['H'], calib['W']) if calib['H'] else seg['sensor_size']
    seq = EventFrameSequence(
        paths['events'], paths['calib'],
        frame_duration=seg['dt'], t_start=seg['t_start'],
        n_frames=n, clip_value=10.0,
        sensor_size=sensor, undistort=False,
    )
    t0, dt = seg['t_start'], seg['dt']
    return {
        'seg': seg, 'seq': seq, 'calib': calib,
        'imu': load_imu(paths['imu']), 'om': om,
        'frames': list(seq)[:n], 'n': n, 'dt': dt,
        'wins': [(t0 + k * dt, t0 + (k + 1) * dt) for k in range(n)],
    }


def ref_omega(om, t_lo, t_hi):
    return omega_gt_at(om, t_lo, t_hi)


def build_net(ctx):
    """Build a fresh network for one rung (guarantees isolated state)."""
    c = ctx['calib']
    return InteractingMapsThesis(
        H=ctx['seq'].H, W=ctx['seq'].W,
        fx=c['fx'], fy=c['fy'], cx=c['cx'], cy=c['cy'],
        frame_duration=ctx['dt'],
        dist_coeffs=c['dist_coeffs'], poisson=config.POISSON_MODE,
        **config.THESIS_PARAMS,
    )


def gt_flow(net, R_gt):
    return np.einsum('hwij,j->hwi', net._C_mat, R_gt)


def score_frames(est_list, ref_list):
    est, ref = np.asarray(est_list), np.asarray(ref_list)
    errs, dirs, betas = zip(*[compute_metrics(e, r) for e, r in zip(est, ref)])
    axis_r = []
    for a in range(3):
        ea, ra = est[:, a] - est[:, a].mean(), ref[:, a] - ref[:, a].mean()
        den = np.sqrt((ea * ea).sum() * (ra * ra).sum())
        axis_r.append(float((ea * ra).sum() / den) if den > 0 else float('nan'))
    return {
        'mean_err': float(np.mean(errs)), 'median_err': float(np.median(errs)),
        'mean_dir': float(np.mean(dirs)), 'median_dir': float(np.median(dirs)),
        'mean_beta': float(np.mean(betas)),
        'axis_r': axis_r,
        'axis_bias': (est - ref).mean(axis=0).tolist(),
    }


def _selfcheck():
    for seg in SEGMENTS:
        ctx = prep_segment_context(seg)
        if ctx is None:
            continue
        net = build_net(ctx)
        assert net._C_mat.shape == (ctx['seq'].H, ctx['seq'].W, 2, 3)
        w0 = ref_omega(ctx['om'], *ctx['wins'][0])
        c = ctx['calib']
        print(f"[ok] {seg['name']}: {ctx['n']} frames, |omega_gt[0]|="
              f"{np.linalg.norm(w0):.3f} rad/s  [calib {c['src']}: "
              f"fx={c['fx']:.0f} cx={c['cx']:.0f} cy={c['cy']:.0f}"
              f"{' +dist' if c['dist_coeffs'] is not None else ''}]")


# %%
def rung0_baseline(ctx):
    """Vision-only thesis net on one segment; characterize the failure.
    Returns (score, est(N,3), ref(N,3))."""
    net = build_net(ctx)
    # Warm start from the gyro at t_start (mirrors reported runs: initial_R None).
    net.initialize_from_rotation(
        get_gyro_for_frame(ctx['imu'], *ctx['wins'][0]) * ctx['dt'])
    est, ref, curls = [], [], []
    for (V, _t), (t_lo, t_hi) in zip(ctx['frames'], ctx['wins']):
        net.step(V, n_iters=config.ITERS_PER_FRAME)      # vision-only (no anchor)
        est.append(net.R / ctx['dt'])
        ref.append(ref_omega(ctx['om'], t_lo, t_hi))
        curls.append(curl_share(net.G))
    s = score_frames(est, ref)
    s['cond_M'] = float(np.linalg.cond(build_R_normal_equations(net._C_mat)))
    s['mean_curl_share'] = float(np.mean(curls))
    print(f"[rung0] {ctx['seg']['name']:12s} err={s['mean_err']:6.2f} "
          f"dir={s['mean_dir']:6.2f} beta={s['mean_beta']:.3f} "
          f"axis_r={[round(v,2) for v in s['axis_r']]}")
    return s, np.asarray(est), np.asarray(ref)


def rung1a_kinematics(ctx):
    """Perfect flow F*=C*R_gt inverted back through the R least-squares.
    Must return R_gt (dir~0); failure means the C-matrix / solve is broken."""
    net = build_net(ctx)
    C = net._C_mat
    M_inv = build_R_normal_equations(C)
    dt = ctx['dt']
    est, ref = [], []
    for (t_lo, t_hi) in ctx['wins']:
        w = ref_omega(ctx['om'], t_lo, t_hi)
        R_hat = solve_R_lstsq(M_inv, C, gt_flow(net, w * dt))
        est.append(R_hat / dt)
        ref.append(w)
    s = score_frames(est, ref)
    s['cond_M'] = float(np.linalg.cond(M_inv))
    ok = s['mean_dir'] < 1.0 and abs(s['mean_beta'] - 1.0) < 0.05
    print(f"[rung1a] {ctx['seg']['name']:12s} dir={s['mean_dir']:.4f} "
          f"beta={s['mean_beta']:.4f} [{'ok' if ok else 'FAIL'}]")
    return s, ok


def rung1b_clamped_flow(ctx, anchored):
    """Clamp F = F* (perfect flow) through the relaxation; let R evolve.
    anchored=False: pure vision; anchored=True: also anchor R toward omega_gt."""
    net = build_net(ctx)
    dt = ctx['dt']
    net.initialize_from_rotation(
        get_gyro_for_frame(ctx['imu'], *ctx['wins'][0]) * dt)
    net.q_F.update = lambda lr: None         # freeze F (instance-level no-op)
    est, ref = [], []
    max_drift = 0.0
    for (V, _t), (t_lo, t_hi) in zip(ctx['frames'], ctx['wins']):
        w = ref_omega(ctx['om'], t_lo, t_hi)
        F_star = gt_flow(net, w * dt)
        net.q_F.value = F_star.copy()
        net.step(V, n_iters=config.ITERS_PER_FRAME,
                 omega_imu=(w if anchored else None))
        max_drift = max(max_drift, float(np.max(np.abs(net.q_F.value - F_star))))
        est.append(net.R / dt)
        ref.append(w)
    s = score_frames(est, ref)
    s['cond_M'] = float(np.linalg.cond(build_R_normal_equations(net._C_mat)))
    tag = 'anchored' if anchored else 'free'
    print(f"[rung1b/{tag}] {ctx['seg']['name']:12s} err={s['mean_err']:6.2f} "
          f"dir={s['mean_dir']:6.2f} beta={s['mean_beta']:.3f} "
          f"(clamp drift {max_drift:.1e})")
    return s


# %%
RESULTS_CSV = 'diag_ladder_results.csv'
FRAMES_CSV = 'diag_ladder_frames.csv'

RESULTS_HEADER = ['segment', 'ds', 'n_frames', 'rung', 'mean_err', 'median_err',
                  'mean_dir', 'median_dir', 'mean_beta',
                  'axis_r_x', 'axis_r_y', 'axis_r_z', 'cond_M', 'curl_share']


def _row(ctx, rung, s):
    return [ctx['seg']['name'], ctx['seg']['ds'], ctx['n'], rung,
            round(s['mean_err'], 4), round(s['median_err'], 4),
            round(s['mean_dir'], 4), round(s['median_dir'], 4),
            round(s['mean_beta'], 4),
            round(s['axis_r'][0], 4), round(s['axis_r'][1], 4),
            round(s['axis_r'][2], 4),
            round(s.get('cond_M', float('nan')), 3),
            round(s['mean_curl_share'], 4) if 'mean_curl_share' in s else '']


def run_all_segments(out_results=RESULTS_CSV, out_frames=FRAMES_CSV):
    rrows, frows = [], []
    for seg in SEGMENTS:
        ctx = prep_segment_context(seg)
        if ctx is None:
            continue
        s0, est, ref = rung0_baseline(ctx)
        rrows.append(_row(ctx, '0', s0))
        for k in range(ctx['n']):
            frows.append([seg['name'], k,
                          est[k, 0], est[k, 1], est[k, 2],
                          ref[k, 0], ref[k, 1], ref[k, 2]])
        s1a, _ok = rung1a_kinematics(ctx)
        rrows.append(_row(ctx, '1a', s1a))
        rrows.append(_row(ctx, '1b_free', rung1b_clamped_flow(ctx, anchored=False)))
        rrows.append(_row(ctx, '1b_anchored', rung1b_clamped_flow(ctx, anchored=True)))
    with open(out_results, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(RESULTS_HEADER); w.writerows(rrows)
    with open(out_frames, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['segment', 'frame', 'est_x', 'est_y', 'est_z',
                    'gt_x', 'gt_y', 'gt_z'])
        w.writerows(frows)
    print(f"\n[ok] wrote {out_results} ({len(rrows)} rows) and "
          f"{out_frames} ({len(frows)} rows)")


# %%
RUNG_ORDER = ['0', '1a', '1b_free', '1b_anchored']
RUNG_COLORS = {'0': '#2a78d6', '1a': '#eb6834',
               '1b_free': '#1baf7a', '1b_anchored': '#eda100'}
RUNG_LABEL = {'0': 'rung 0 (vision-only)', '1a': 'rung 1a (kinematics)',
              '1b_free': 'rung 1b (flow, free)',
              '1b_anchored': 'rung 1b (flow, anchored)'}


def _read_results(path=RESULTS_CSV):
    rows = list(csv.DictReader(open(path, newline='')))
    segs = []
    for r in rows:
        if r['segment'] not in segs:
            segs.append(r['segment'])
    return rows, segs


def plot_P1(path=RESULTS_CSV, out='diag_P1_dir_beta.png'):
    """Grouped bars: dir (deg) and beta by segment, grouped by rung.
    Shows the oracle rungs collapsing to ~0 across every scene."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows, segs = _read_results(path)
    val = {(r['segment'], r['rung']): r for r in rows}
    x = np.arange(len(segs)); w = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, title in [(axes[0], 'mean_dir', 'Direction error (deg)'),
                              (axes[1], 'mean_beta', 'Scale  beta  (1.0 = exact)')]:
        for i, rung in enumerate(RUNG_ORDER):
            ys = [float(val[(s, rung)][metric]) if (s, rung) in val else 0.0
                  for s in segs]
            ax.bar(x + (i - 1.5) * w, ys, w, color=RUNG_COLORS[rung],
                   label=RUNG_LABEL[rung])
        ax.set_xticks(x); ax.set_xticklabels(segs, rotation=20, ha='right')
        ax.set_title(title); ax.spines[['top', 'right']].set_visible(False)
    axes[1].axhline(1.0, color='#52514e', lw=1, ls='--')
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[ok] {out}")


def plot_P2(path=RESULTS_CSV, out='diag_P2_axis_heatmap.png'):
    """Heatmap segment x axis of rung-0 per-axis correlation axis_r.
    Diverging (-1..1, 0 neutral): reveals a recurring dead axis across scenes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    rows, segs = _read_results(path)
    r0 = {r['segment']: r for r in rows if r['rung'] == '0'}
    segs = [s for s in segs if s in r0]
    M = np.array([[float(r0[s]['axis_r_x']), float(r0[s]['axis_r_y']),
                   float(r0[s]['axis_r_z'])] for s in segs])
    cmap = LinearSegmentedColormap.from_list(
        'div', ['#eb6834', '#f0efec', '#2a78d6'])
    fig, ax = plt.subplots(figsize=(5, 0.7 * len(segs) + 1.5))
    im = ax.imshow(M, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['wx', 'wy', 'wz'])
    ax.set_yticks(range(len(segs))); ax.set_yticklabels(segs)
    for i in range(len(segs)):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:.2f}", ha='center', va='center',
                    color='#0b0b0b', fontsize=9)
    ax.set_title('Rung-0 per-axis correlation of est. with GT omega')
    fig.colorbar(im, ax=ax, label='correlation')
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[ok] {out}")


def plot_P3(path=FRAMES_CSV, out='diag_P3_omega_timeseries.png'):
    """Small multiples: rung-0 estimated vs GT omega per axis over frames,
    one row per segment, one column per axis. Shows which axis diverges."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(path, newline='')))
    segs = []
    for r in rows:
        if r['segment'] not in segs:
            segs.append(r['segment'])
    fig, axes = plt.subplots(len(segs), 3, figsize=(12, 2.6 * len(segs)),
                             squeeze=False)
    axkeys = [('est_x', 'gt_x', 'wx'), ('est_y', 'gt_y', 'wy'),
              ('est_z', 'gt_z', 'wz')]
    for i, seg in enumerate(segs):
        sr = [r for r in rows if r['segment'] == seg]
        k = [int(r['frame']) for r in sr]
        for j, (ek, gk, title) in enumerate(axkeys):
            ax = axes[i][j]
            ax.plot(k, [float(r[gk]) for r in sr], color='#52514e', lw=1.5,
                    label='GT')
            ax.plot(k, [float(r[ek]) for r in sr], color='#2a78d6', lw=1.5,
                    label='est')
            ax.spines[['top', 'right']].set_visible(False)
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel(seg, fontsize=9)
    axes[0][2].legend(frameon=False, fontsize=8)
    fig.suptitle('Rung 0: estimated vs ground-truth omega (rad/s) over frames')
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[ok] {out}")


def make_all_plots():
    plot_P1(); plot_P2(); plot_P3()


# %%
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='run',
                    choices=['setup', 'run', 'plots'])
    ap.add_argument('--limit', type=int, default=None,
                    help='cap frames per segment (fast local smoke test)')
    args = ap.parse_args()
    LIMIT = args.limit
    if args.mode == 'setup':
        _selfcheck()
    elif args.mode == 'run':
        run_all_segments()
    elif args.mode == 'plots':
        make_all_plots()
