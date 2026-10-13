"""
diag_omega_ladder.py — ω-recovery oracle ladder (rungs 0 / 1a / 1b / 1c) over
several synthetic segments, plus comparison plots.

Run directly (NOT pytest). The compute paths are cluster-ready: --mode run
writes diag_ladder_results.csv and diag_ladder_frames.csv; --mode maps writes
diag_maps_<segment>.npz (I, G, F of rows 1c_R / 1b_free / 0 per frame, and per
iteration at frame K). The --mode plots path is local and renders P1/P2/P3 from
the CSVs and, per npz, P4 (maps grid), P5 (flow EPE) and two GIFs. Read-only on
the model: it only injects ground-truth flow into q_F or ground-truth R into q_R.

See docs/superpowers/specs/2026-10-07-omega-recovery-diagnostic-design.md and
docs/superpowers/plans/2026-10-07-omega-ladder-multisegment-plots.md.

Companion walkthrough: diag_omega_ladder.ipynb imports these functions and steps
through the pipeline on one segment, with a figure or log per step. It is
authored, not generated: when a function here changes, update the notebook cells
that call it.
"""
# %%
import argparse
import csv
import glob
import os
import numpy as np

import config
from config import get_dataset_paths
from interacting_maps.eval_io import load_imu, get_gyro_for_frame, load_omega_gt, omega_gt_at
from interacting_maps.data_loader import EventFrameSequence, CameraCalibration
from interacting_maps.network_dissertation import InteractingMapsThesis, Cost_OFCE, Quantity
from interacting_maps.camera import build_R_normal_equations, solve_R_lstsq
from interacting_maps.metrics import compute_metrics, curl_share, flow_epe
from interacting_maps.viz import net_maps, draw_map, draw_series
from interacting_maps.style import frame_times, use_style, save, SCREEN_WIDTH

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
    """Network intrinsics + distortion from calib.txt — the converter writes it
    from the bag's own CameraInfo, i.e. what the simulator rendered with. (The
    ECRot DAVIS240C-synthetic.yaml has a wrong cy=120; the bag says 90.)

    Returns {fx, fy, cx, cy, dist_coeffs, src}. dist_coeffs is None when all
    coefficients are zero (clean pinhole C), else the [k1,k2,p1,p2,k3] array.
    """
    c = CameraCalibration(paths['calib'])
    dist_coeffs = c.dist if np.any(c.dist) else None
    return {'fx': c.fx, 'fy': c.fy, 'cx': c.cx, 'cy': c.cy,
            'dist_coeffs': dist_coeffs, 'src': 'calib.txt'}


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
    sensor = seg['sensor_size']
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
    net = InteractingMapsThesis(
        H=ctx['seq'].H, W=ctx['seq'].W,
        fx=c['fx'], fy=c['fy'], cx=c['cx'], cy=c['cy'],
        frame_duration=ctx['dt'],
        dist_coeffs=c['dist_coeffs'], poisson=config.POISSON_MODE,
        **{**config.THESIS_PARAMS, **ctx.get('params', {})},
    )
    for cost in net.costs:                  # --mode ofce: max_grad / normalize
        if isinstance(cost, Cost_OFCE):
            vars(cost).update(ctx.get('ofce', {}))
    return net


def gt_flow(net, R_gt):
    return np.einsum('hwij,j->hwi', net._C_mat, R_gt)


def clamp_flow(net, F_star):
    """Hold q_F at F_star: an instance-level no-op update freezes it through the
    relaxation, while the other costs still update G, I and R."""
    net.q_F.update = lambda lr: None
    net.q_F.value = F_star.copy()


def clamp_R(net, R_star):
    """Hold q_R at R_star: with the kinematic R-update off (the V2 switch) and
    no IMU anchor, no cost sends gradient to R, so the relaxation keeps it."""
    net.cost_kin.update_r = False
    net.q_R.value = R_star.copy()


def trace_relaxation(net, V, n_iters, dt, omega_imu=None, on_iter=None):
    """Step one frame's relaxation one iteration at a time (same result as a
    single step(V, n_iters)) and record omega (n_iters, 3) in rad/s plus the
    V+F.G and G-grad(I) residuals, for convergence plots. on_iter(net), if
    given, is called after every iteration."""
    om, r_vfg, r_gi = [], [], []
    for _ in range(n_iters):
        net.step(V, n_iters=1, omega_imu=omega_imu)
        om.append(net.R / dt)
        r_vfg.append(net.residual_VFG(V))
        r_gi.append(net.residual_GI())
        if on_iter is not None:
            on_iter(net)
    return np.asarray(om), np.asarray(r_vfg), np.asarray(r_gi)


def advance(net, V, k, dt, omega_imu=None, on_iter=None, iter_frame=None):
    """One frame's relaxation; at frame iter_frame step it one iteration at a
    time through trace_relaxation so on_iter sees every iteration."""
    if on_iter is not None and k == iter_frame:
        trace_relaxation(net, V, config.ITERS_PER_FRAME, dt, omega_imu, on_iter)
    else:
        net.step(V, n_iters=config.ITERS_PER_FRAME, omega_imu=omega_imu)


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


def _selfcheck_maps_io():
    """Maps saved to the float16 npz must load back to finite colour scales
    (np.percentile on float16 overflows its index past 65504 elements)."""
    import tempfile
    rng = np.random.default_rng(0)
    n, H, W = 3, 180, 240
    d = {'V': rng.standard_normal((n, H, W)), 'F_gt': rng.standard_normal((n, H, W, 2)),
         'I': rng.standard_normal((3, n, H, W)), 'G': rng.standard_normal((3, n, H, W, 2))}
    d = {k: v.astype(np.float32) for k, v in d.items()} | {'dt': 0.02, 'K': 1}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'maps.npz')
        np.savez_compressed(path, **_store16(d))
        sc = _map_scales(_load32(path))
    ok = all(np.all(np.isfinite(v)) for v in sc.values())
    print(f"  [{'ok' if ok else 'FAIL'}] npz round trip gives finite colour scales {sc}")


def _ofce_g_only(F_mag, normalize, n_iters=200, V=1.0):
    """One pixel, F fixed along x, only the OFCE step on G (q.update(1.0) as in
    the network). Returns |V + F.G| after n_iters: ~0 iff the step converged."""
    q = {k: Quantity(s, k) for k, s in (('V', (1, 1)), ('F', (1, 1, 2)), ('G', (1, 1, 2)))}
    q['V'].value[:] = V
    q['F'].value[..., 0] = F_mag
    cost = Cost_OFCE(q, config.THESIS_PARAMS['delta_VFG'], normalize=normalize)
    for _ in range(n_iters):
        q['G'].reset_gradient()
        cost.compute_and_send_gradients()
        q['G'].update(1.0)
    return float(abs(V + np.sum(q['F'].value * q['G'].value)))


def _selfcheck_ofce_step():
    """The raw OFCE step on G converges only for |F| < 1/sqrt(delta_VFG); the
    normalised step converges for any |F|."""
    b = 1.0 / np.sqrt(config.THESIS_PARAMS['delta_VFG'])
    cases = [(0.9 * b, False, True), (1.1 * b, False, False), (10.0, True, True)]
    for F_mag, norm, want in cases:
        res = _ofce_g_only(F_mag, norm)
        ok = (res < 1e-6) == want
        print(f"  [{'ok' if ok else 'FAIL'}] OFCE G-step |F|={F_mag:.2f} "
              f"normalize={norm}: residual {res:.2e} (expect {'converged' if want else 'not converged'})")


def _selfcheck():
    _selfcheck_maps_io()
    _selfcheck_ofce_step()
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
        _selfcheck_clamps(ctx)


def _selfcheck_clamps(ctx):
    """clamp_R holds R exactly; trace_relaxation(on_iter) matches net.step."""
    V, dt, n_it = ctx['frames'][0][0], ctx['dt'], config.ITERS_PER_FRAME
    R_star = ref_omega(ctx['om'], *ctx['wins'][0]) * dt
    a = build_net(ctx)
    a.initialize_from_rotation(0.5 * R_star)
    clamp_R(a, R_star)
    a.step(V, n_iters=n_it)
    print(f"  [{'ok' if np.array_equal(a.R, R_star) else 'FAIL'}] clamp_R holds R exactly")
    b, c, seen = build_net(ctx), build_net(ctx), []
    for net in (b, c):
        net.initialize_from_rotation(R_star)
    b.step(V, n_iters=n_it)
    trace_relaxation(c, V, n_it, dt, on_iter=lambda net: seen.append(1))
    same = all(np.array_equal(getattr(b, m), getattr(c, m)) for m in 'IGFR')
    print(f"  [{'ok' if same and len(seen) == n_it else 'FAIL'}] "
          f"trace_relaxation(on_iter) == net.step ({len(seen)} callbacks)")
    print(f"  [{'ok' if flow_epe(a.F, a.F) == 0.0 else 'FAIL'}] flow_epe(F, F) == 0")


# %%
def rung0_baseline(ctx, on_frame=None, on_iter=None, iter_frame=None):
    """Vision-only thesis net on one segment; characterize the failure.
    Returns (score, est(N,3), ref(N,3)). on_frame(k, net) runs after each
    frame; on_iter(net) after each iteration of frame iter_frame."""
    net = build_net(ctx)
    # Warm start from the gyro at t_start (mirrors reported runs: initial_R None).
    net.initialize_from_rotation(
        get_gyro_for_frame(ctx['imu'], *ctx['wins'][0]) * ctx['dt'])
    dt = ctx['dt']
    est, ref, curls, epes = [], [], [], []
    for k, ((V, _t), (t_lo, t_hi)) in enumerate(zip(ctx['frames'], ctx['wins'])):
        advance(net, V, k, dt, on_iter=on_iter, iter_frame=iter_frame)  # vision-only
        w = ref_omega(ctx['om'], t_lo, t_hi)
        est.append(net.R / dt)
        ref.append(w)
        curls.append(curl_share(net.G))
        epes.append(flow_epe(net.F, gt_flow(net, w * dt)))
        if on_frame is not None:
            on_frame(k, net)
    s = score_frames(est, ref)
    s['F_epe'] = float(np.mean(epes))
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


def rung1b_clamped_flow(ctx, anchored, on_frame=None, on_iter=None, iter_frame=None):
    """Clamp F = F* (perfect flow) through the relaxation; let R evolve.
    anchored=False: pure vision; anchored=True: also anchor R toward omega_gt.
    Hooks as in rung0_baseline."""
    net = build_net(ctx)
    dt = ctx['dt']
    net.initialize_from_rotation(
        get_gyro_for_frame(ctx['imu'], *ctx['wins'][0]) * dt)
    est, ref, epes = [], [], []
    max_drift = 0.0
    for k, ((V, _t), (t_lo, t_hi)) in enumerate(zip(ctx['frames'], ctx['wins'])):
        w = ref_omega(ctx['om'], t_lo, t_hi)
        F_star = gt_flow(net, w * dt)
        clamp_flow(net, F_star)
        advance(net, V, k, dt, omega_imu=(w if anchored else None),
                on_iter=on_iter, iter_frame=iter_frame)
        max_drift = max(max_drift, float(np.max(np.abs(net.q_F.value - F_star))))
        est.append(net.R / dt)
        ref.append(w)
        epes.append(flow_epe(net.F, F_star))
        if on_frame is not None:
            on_frame(k, net)
    s = score_frames(est, ref)
    s['F_epe'] = float(np.mean(epes))
    s['cond_M'] = float(np.linalg.cond(build_R_normal_equations(net._C_mat)))
    tag = 'anchored' if anchored else 'free'
    print(f"[rung1b/{tag}] {ctx['seg']['name']:12s} err={s['mean_err']:6.2f} "
          f"dir={s['mean_dir']:6.2f} beta={s['mean_beta']:.3f} "
          f"(clamp drift {max_drift:.1e})")
    return s


def rung1c_clamped_R(ctx, on_frame=None, on_iter=None, iter_frame=None):
    """Clamp R = R_gt through the relaxation; let I, G and F relax. Asks whether
    the maps come out right once omega is right. Scored by flow EPE vs C*R_gt
    (the omega metrics are ~0 by construction). Hooks as in rung0_baseline."""
    net = build_net(ctx)
    dt = ctx['dt']
    net.initialize_from_rotation(
        get_gyro_for_frame(ctx['imu'], *ctx['wins'][0]) * dt)
    est, ref, epes = [], [], []
    for k, ((V, _t), (t_lo, t_hi)) in enumerate(zip(ctx['frames'], ctx['wins'])):
        w = ref_omega(ctx['om'], t_lo, t_hi)
        clamp_R(net, w * dt)
        advance(net, V, k, dt, on_iter=on_iter, iter_frame=iter_frame)
        est.append(net.R / dt)
        ref.append(w)
        epes.append(flow_epe(net.F, gt_flow(net, w * dt)))
        if on_frame is not None:
            on_frame(k, net)
    s = score_frames(est, ref)
    s['F_epe'] = float(np.mean(epes))
    print(f"[rung1c] {ctx['seg']['name']:12s} F_epe={s['F_epe']:.4f} px/frame "
          f"(err={s['mean_err']:.2e}, R clamped)")
    return s


# %%
RESULTS_CSV = 'diag_ladder_results.csv'
FRAMES_CSV = 'diag_ladder_frames.csv'

RESULTS_HEADER = ['segment', 'ds', 'n_frames', 'rung', 'mean_err', 'median_err',
                  'mean_dir', 'median_dir', 'mean_beta',
                  'axis_r_x', 'axis_r_y', 'axis_r_z', 'cond_M', 'curl_share',
                  'F_epe']


def _row(ctx, rung, s):
    return [ctx['seg']['name'], ctx['seg']['ds'], ctx['n'], rung,
            round(s['mean_err'], 4), round(s['median_err'], 4),
            round(s['mean_dir'], 4), round(s['median_dir'], 4),
            round(s['mean_beta'], 4),
            round(s['axis_r'][0], 4), round(s['axis_r'][1], 4),
            round(s['axis_r'][2], 4),
            round(s.get('cond_M', float('nan')), 3),
            round(s['mean_curl_share'], 4) if 'mean_curl_share' in s else '',
            round(s['F_epe'], 4) if 'F_epe' in s else '']


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
        rrows.append(_row(ctx, '1c_R', rung1c_clamped_R(ctx)))
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
               '1b_free': '#1baf7a', '1b_anchored': '#eda100',
               '1c_R': '#8a5cd1'}
RUNG_LABEL = {'0': 'rung 0 (vision-only)', '1a': 'rung 1a (kinematics)',
              '1b_free': 'rung 1b (flow, free)',
              '1b_anchored': 'rung 1b (flow, anchored)',
              '1c_R': 'rung 1c (R clamped)'}


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


# %%
MAP_RUNGS = ('1c_R', '1b_free', '0')


def _map_run(ctx, rung, **hooks):
    if rung == '1c_R':
        return rung1c_clamped_R(ctx, **hooks)
    if rung == '1b_free':
        return rung1b_clamped_flow(ctx, anchored=False, **hooks)
    return rung0_baseline(ctx, **hooks)[0]


def _snap(net, H, W):
    return {m: a.astype(np.float32) for m, a in net_maps(net, H, W).items()}


def collect_maps(ctx, K):
    """Run the MAP_RUNGS rows on one segment. Returns end-of-frame I, G, F per
    row (rung axis first), the per-iteration I, G, F at frame K, V and F_gt."""
    H, W, dt = ctx['seq'].H, ctx['seq'].W, ctx['dt']
    net = build_net(ctx)
    d = {'V': np.stack([V for V, _ in ctx['frames']]).astype(np.float32),
         'F_gt': np.stack([gt_flow(net, ref_omega(ctx['om'], *win) * dt)
                           for win in ctx['wins']]).astype(np.float32)}
    per = {m: [] for m in ('I', 'G', 'F', 'I_it', 'G_it', 'F_it')}
    for rung in MAP_RUNGS:
        fr, it = [], []
        _map_run(ctx, rung, on_frame=lambda k, n: fr.append(_snap(n, H, W)),
                 on_iter=lambda n: it.append(_snap(n, H, W)), iter_frame=K)
        for m in 'IGF':
            per[m].append(np.stack([s[m] for s in fr]))
            per[m + '_it'].append(np.stack([s[m] for s in it]))
    d.update({k: np.stack(v) for k, v in per.items()})
    d.update(rungs=np.array(MAP_RUNGS), dt=dt, t_start=ctx['seg']['t_start'], K=K)
    return d


def maps_path(seg_name):
    return f'diag_maps_{seg_name}.npz'


def _store16(d):
    """float32 maps -> float16, for the npz only (halves its size). Never compute
    on float16: np.percentile overflows its index past 65504 elements."""
    return {k: v.astype(np.float16) if np.asarray(v).dtype == np.float32 else v
            for k, v in d.items()}


def _load32(path):
    """Read a maps npz back to float32, the dtype every plot function expects."""
    with np.load(path) as z:
        return {k: z[k].astype(np.float32) if z[k].dtype == np.float16 else z[k]
                for k in z.files}


def run_maps(seg_name, K=None):
    """Cluster entry for --mode maps: collect_maps on one segment -> npz."""
    seg = next(s for s in SEGMENTS if s['name'] == seg_name)
    ctx = prep_segment_context(seg)
    if ctx is None:
        return
    K = ctx['n'] // 2 if K is None else min(K, ctx['n'] - 1)
    np.savez_compressed(maps_path(seg_name), **_store16(collect_maps(ctx, K)))
    print(f"[ok] wrote {maps_path(seg_name)} ({ctx['n']} frames, K={K})")


def _map_scales(d):
    """Fixed colour scales for every row and frame, set from the clamped-R row
    (index 0) so differences between rows and frames are real."""
    mag = lambda v: np.hypot(v[..., 0], v[..., 1])
    lo, hi = np.percentile(d['I'][0], [1, 99])
    return {'V': float(np.percentile(np.abs(d['V']), 99)),
            'I': (float(lo), float(hi)),
            'G': float(np.percentile(mag(d['G'][0]), 99)),
            'F': float(np.percentile(mag(d['F_gt']), 99))}


def _frame_maps(d, k):
    return {'V': d['V'][k], 'F_gt': d['F_gt'][k],
            **{m: d[m][:, k] for m in 'IGF'}}


def _iter_maps(d, i):
    K = int(d['K'])
    return {'V': d['V'][K], 'F_gt': d['F_gt'][K],
            **{m: d[m + '_it'][:, i] for m in 'IGF'}}


def draw_maps_grid(axes, maps_t, scales, rungs):
    """Fill a (len(rungs), 6) grid: V | I | G | F | F_gt | F - F_gt."""
    for r, rung in enumerate(rungs):
        cells = [('V', maps_t['V'], None), ('I', maps_t['I'][r], None),
                 ('G', maps_t['G'][r], None), ('F', maps_t['F'][r], None),
                 ('F', maps_t['F_gt'], r'$F_{gt} = C R_{gt}$'),
                 ('F', maps_t['F'][r] - maps_t['F_gt'], r'$F - F_{gt}$')]
        for c, (kind, arr, title) in enumerate(cells):
            draw_map(axes[r, c], kind, arr, title=(title if r == 0 else ''),
                     scale=scales[kind])
        axes[r, 0].set_ylabel(RUNG_LABEL[str(rung)], fontsize=8)


def _maps_figure(maps_t, scales, rungs, title):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(rungs), 6, figsize=(15, 1.95 * len(rungs) + 0.5),
                             squeeze=False)
    draw_maps_grid(axes, maps_t, scales, rungs)
    fig.suptitle(title)
    return fig


def plot_P4(d, out):
    """Static maps grid at frame K (end of its relaxation)."""
    K = int(d['K'])
    t = frame_times(float(d['dt']), len(d['V']))[K]
    save(_maps_figure(_frame_maps(d, K), _map_scales(d), list(d['rungs']),
                      f'frame {K}  (t = {t:.3f} s)'), out)
    print(f"[ok] {out}")


def plot_P5(d, out):
    """Per-frame flow EPE vs C*R_gt, one line per row."""
    import matplotlib.pyplot as plt
    n = len(d['V'])
    t = frame_times(float(d['dt']), n)
    with use_style('screen'):
        fig, ax = plt.subplots(figsize=(SCREEN_WIDTH, 3.5))
        for r, rung in enumerate(d['rungs']):
            draw_series(ax, t, [flow_epe(d['F'][r, k], d['F_gt'][k]) for k in range(n)],
                        'flow EPE (px/frame)', label=RUNG_LABEL[str(rung)])
            ax.get_lines()[-1].set_color(RUNG_COLORS[str(rung)])
        ax.legend()
        save(fig, out)
    print(f"[ok] {out}")


def make_maps_gifs(d, out_dir, fps=8):
    """gif_frames.gif (one figure per frame) and gif_iters.gif (one per
    iteration of frame K). PNGs at a fixed size so every GIF frame matches."""
    import matplotlib.pyplot as plt
    from interacting_maps.iwe_io import make_gif
    sc, rungs, K = _map_scales(d), list(d['rungs']), int(d['K'])
    n, T = len(d['V']), d['I_it'].shape[1]
    t = frame_times(float(d['dt']), n)
    jobs = [('frames', n, lambda k: (_frame_maps(d, k), f'frame {k}  (t = {t[k]:.3f} s)')),
            ('iters', T, lambda i: (_iter_maps(d, i), f'frame {K}, iteration {i + 1}/{T}'))]
    for name, count, get in jobs:
        sub = os.path.join(out_dir, name)
        os.makedirs(sub, exist_ok=True)
        for f in glob.glob(os.path.join(sub, 'frame_*.png')):
            os.remove(f)
        for j in range(count):
            maps_t, title = get(j)
            fig = _maps_figure(maps_t, sc, rungs, title)
            fig.savefig(os.path.join(sub, f'frame_{j:04d}.png'), dpi=72)
            plt.close(fig)
        make_gif(sub, fps=fps, name=f'gif_{name}.gif')
        print(f"[ok] {os.path.join(sub, f'gif_{name}.gif')}")


def plot_maps(seg_name):
    d = _load32(maps_path(seg_name))
    plot_P4(d, f'diag_P4_maps_{seg_name}.png')
    plot_P5(d, f'diag_P5_F_epe_{seg_name}.png')
    make_maps_gifs(d, f'diag_maps_{seg_name}')


# %%
# --mode ofce: is the striped I an OFCE step instability? One variant per call
# (one report/run_all.sh job); rows 1c_R (omega exact, isolates the OFCE step) and 0.
OFCE_DIR = 'diag_ofce'
OFCE_VARIANTS = {
    'E0_base':      {},
    'E1_dVFG006':   {'params': {'delta_VFG': 0.06}},
    'E1_dVFG024':   {'params': {'delta_VFG': 0.24}},
    'E2_noclip':    {'ofce': {'max_grad': np.inf}},
    'E3_normalize': {'ofce': {'normalize': True}},
    'E4_dt10':      {'dt': 0.010},
    'E4_dt05':      {'dt': 0.005},
}
FLIP_BINS = (0, 1, 2, 3, 4, 6, np.inf)          # |F| px/frame
OFCE_HEADER = (['variant', 'rung', 'dt_ms', 'n_frames', 'delta_VFG', 'max_grad',
                'normalize', 'bound_px', 'frac_Fgt_above', 'F_epe', 'F_epe_rel',
                'stripe_I', 'G_at_clip', 'finite', 'mean_err', 'mean_dir', 'mean_beta']
               + [f'flip_{lo}_{hi}' for lo, hi in zip(FLIP_BINS[:-1], FLIP_BINS[1:])]
               + ['secs'])


def stability_bound(delta_VFG, delta_IG):
    """|F| (px/frame) above which the explicit OFCE step on G diverges, from
    lambda = 1 - 2*delta_VFG*|F|^2 - delta_IG (see spec, Findings)."""
    return float(np.sqrt((2.0 - delta_IG) / (2.0 * delta_VFG)))


def flip_rate(tail):
    """Per-pixel rate of sign changes of G along F over the recorded
    iterations, and |F| at the last one."""
    par = np.stack([np.sum(G * F, -1) / np.maximum(np.linalg.norm(F, axis=-1), 1e-9)
                    for G, F in tail])
    return (np.sign(par[1:]) != np.sign(par[:-1])).mean(0), np.linalg.norm(tail[-1][1], axis=-1)


def stripe_index(I):
    """Scale-free high-frequency share of I: mean |discrete Laplacian| / std."""
    lap = 4 * I[1:-1, 1:-1] - I[:-2, 1:-1] - I[2:, 1:-1] - I[1:-1, :-2] - I[1:-1, 2:]
    return float(np.mean(np.abs(lap)) / max(float(np.std(I)), 1e-12))


def run_ofce_variant(name, seg_name='city_short'):
    """Run one OFCE_VARIANTS entry on rows 1c_R and 0 over the segment's span;
    write diag_ofce/<name>.csv and diag_ofce/<name>.png (last-frame maps)."""
    import time
    from collections import deque
    from interacting_maps.camera import CLIP_G
    v = OFCE_VARIANTS[name]
    base = next(s for s in SEGMENTS if s['name'] == seg_name)
    dt = v.get('dt', base['dt'])
    seg = {**base, 'dt': dt, 'n_frames': round(base['n_frames'] * base['dt'] / dt)}
    ctx = prep_segment_context(seg)
    if ctx is None:
        return
    ctx.update(params=v.get('params', {}), ofce=v.get('ofce', {}))
    prm = {**config.THESIS_PARAMS, **ctx['params']}
    ofce = {'max_grad': 5.0, 'normalize': False, **ctx['ofce']}
    H, W, n, K = ctx['seq'].H, ctx['seq'].W, ctx['n'], ctx['n'] // 2
    bound = stability_bound(prm['delta_VFG'], prm['delta_IG'])
    net = build_net(ctx)
    F_gt = [gt_flow(net, ref_omega(ctx['om'], *w) * dt) for w in ctx['wins']]
    mag_gt = [np.linalg.norm(f, axis=-1) for f in F_gt]
    rows, last = [], {}
    for rung in ('1c_R', '0'):
        tail, t0 = deque(maxlen=20), time.time()
        s = _map_run(ctx, rung,
                     on_frame=lambda k, nt: last.update({rung: _snap(nt, H, W)}),
                     on_iter=lambda nt: tail.append((nt.G.copy(), nt.F.copy())),
                     iter_frame=K)
        flips, mag = flip_rate(tail)
        m = last[rung]
        bins = []
        for lo, hi in zip(FLIP_BINS[:-1], FLIP_BINS[1:]):
            sel = (mag >= lo) & (mag < hi)
            bins.append(round(float(flips[sel].mean()), 3) if sel.sum() >= 50 else '')
        rows.append([name, rung, dt * 1000, n, prm['delta_VFG'], ofce['max_grad'],
                     ofce['normalize'], round(bound, 3),
                     round(float(np.mean([(g > bound).mean() for g in mag_gt])), 3),
                     round(s['F_epe'], 4),
                     round(s['F_epe'] / float(np.mean([g.mean() for g in mag_gt])), 4),
                     round(stripe_index(m['I']), 4),
                     round(float((np.abs(m['G']) >= CLIP_G - 1e-6).any(-1).mean()), 4),
                     bool(all(np.all(np.isfinite(a)) for a in m.values())),
                     round(s['mean_err'], 3), round(s['mean_dir'], 3), round(s['mean_beta'], 3)]
                    + bins + [round(time.time() - t0)])
    os.makedirs(OFCE_DIR, exist_ok=True)
    with open(os.path.join(OFCE_DIR, f'{name}.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(OFCE_HEADER); w.writerows(rows)
    V_last = ctx['frames'][-1][0].astype(np.float32)
    rungs = ['1c_R', '0']
    maps_t = {'V': V_last, 'F_gt': F_gt[-1].astype(np.float32),
              **{k: np.stack([last[r][k] for r in rungs]) for k in 'IGF'}}
    sc = _map_scales({'V': V_last[None], 'F_gt': maps_t['F_gt'][None],
                      'I': maps_t['I'][:, None], 'G': maps_t['G'][:, None]})
    save(_maps_figure(maps_t, sc, rungs, f'{name}: last frame (dt = {dt * 1000:.0f} ms)'),
         os.path.join(OFCE_DIR, f'{name}.png'))
    print(f"[ok] {OFCE_DIR}/{name}.csv + .png  (bound {bound:.2f} px/frame, {n} frames)")


def ofce_summary(out=os.path.join(OFCE_DIR, 'summary.csv')):
    """Concatenate diag_ofce/E*.csv into summary.csv and print the key columns."""
    rows = []
    for p in sorted(glob.glob(os.path.join(OFCE_DIR, 'E*.csv'))):
        rows += list(csv.DictReader(open(p, newline='')))
    if not rows:
        return
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=OFCE_HEADER); w.writeheader(); w.writerows(rows)
    keys = ['variant', 'rung', 'bound_px', 'frac_Fgt_above', 'F_epe_rel', 'stripe_I',
            'mean_err'] + [k for k in OFCE_HEADER if k.startswith('flip_')]
    print(' | '.join(keys))
    for r in rows:
        print(' | '.join(str(r[k]) for k in keys))
    print(f"[ok] {out}")


def make_all_plots():
    if os.path.isdir(OFCE_DIR):
        ofce_summary()
    if os.path.exists(RESULTS_CSV) and os.path.exists(FRAMES_CSV):
        plot_P1(); plot_P2(); plot_P3()
    for path in sorted(glob.glob('diag_maps_*.npz')):
        plot_maps(path[len('diag_maps_'):-len('.npz')])


# %%
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='run',
                    choices=['setup', 'run', 'maps', 'ofce', 'ofce_summary', 'plots'])
    ap.add_argument('--variant', default='all', choices=list(OFCE_VARIANTS) + ['all'],
                    help='--mode ofce: one OFCE_VARIANTS entry (one report/run_all.sh job)')
    ap.add_argument('--limit', type=int, default=None,
                    help='cap frames per segment (fast local smoke test)')
    ap.add_argument('--segment', default='city_short',
                    help='--mode maps: segment name from SEGMENTS')
    ap.add_argument('--frame', type=int, default=None,
                    help='--mode maps: frame K for the iteration GIF (default n//2)')
    args = ap.parse_args()
    LIMIT = args.limit
    if args.mode == 'setup':
        _selfcheck()
    elif args.mode == 'run':
        run_all_segments()
    elif args.mode == 'maps':
        run_maps(args.segment, args.frame)
    elif args.mode == 'ofce':
        import matplotlib
        matplotlib.use('Agg')
        for name in (OFCE_VARIANTS if args.variant == 'all' else [args.variant]):
            run_ofce_variant(name, args.segment)
        if args.variant == 'all':            # per-variant jobs: --mode ofce_summary
            ofce_summary()
    elif args.mode == 'ofce_summary':
        ofce_summary()
    elif args.mode == 'plots':
        import matplotlib
        matplotlib.use('Agg')                # headless (cluster); notebooks stay inline
        make_all_plots()
