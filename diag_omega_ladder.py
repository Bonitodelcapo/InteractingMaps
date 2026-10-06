"""
diag_omega_ladder.py — ω-recovery oracle ladder, rungs 0 / 1a / 1b.

Run directly (NOT pytest). Prints [ok]/[FAIL] + per-rung tables and a decision
gate. Read-only on the model: it only injects ground-truth flow into q_F.
See docs/superpowers/specs/2026-10-07-omega-recovery-diagnostic-design.md.

Kept in sync with diag_omega_ladder.ipynb (generated from this file's `# %%`
cells; edit this file, then regenerate the notebook).
"""
# %%
import argparse
import numpy as np

import config
from evaluation import get_dataset_paths
from eval_io import load_imu, get_gyro_for_frame, load_omega_gt, omega_gt_at
from data_loader import EventFrameSequence
from interacting_maps.network_dissertation import InteractingMapsThesis
from interacting_maps.camera import build_R_normal_equations, solve_R_lstsq
from metrics import compute_metrics, curl_share

SEG = {'ds': 'ecrot_city', 't_start': 0.05, 'frame_duration': 0.02,
       'n_frames': 75, 'sensor_size': (180, 240)}


# %%
def load_segment():
    paths = get_dataset_paths(SEG['ds'])
    seq = EventFrameSequence(
        paths['events'], paths['calib'],
        frame_duration=SEG['frame_duration'], t_start=SEG['t_start'],
        n_frames=SEG['n_frames'], clip_value=10.0,
        sensor_size=SEG['sensor_size'], undistort=False,
    )
    imu = load_imu(paths['imu'])
    om = load_omega_gt(paths.get('omega_gt'))
    frames = list(seq)
    return seq, paths, imu, om, frames


def frame_windows(n):
    t0, dt = SEG['t_start'], SEG['frame_duration']
    return [(t0 + k * dt, t0 + (k + 1) * dt) for k in range(n)]


def ref_omega(om, t_lo, t_hi):
    return omega_gt_at(om, t_lo, t_hi)


def build_net(seq):
    c = seq.calib
    return InteractingMapsThesis(
        H=seq.H, W=seq.W, fx=c.fx, fy=c.fy, cx=c.cx, cy=c.cy,
        frame_duration=SEG['frame_duration'],
        dist_coeffs=None, poisson=config.POISSON_MODE,
        **config.THESIS_PARAMS,
    )


def gt_flow(net, R_gt):
    return np.einsum('hwij,j->hwi', net._C_mat, R_gt)


def score_frames(est_list, ref_list):
    est = np.asarray(est_list)          # (N, 3) rad/s
    ref = np.asarray(ref_list)          # (N, 3) rad/s
    errs, dirs, betas = [], [], []
    for e, r in zip(est, ref):
        err, d, b = compute_metrics(e, r)
        errs.append(err); dirs.append(d); betas.append(b)
    axis_r = []
    for a in range(3):
        ea, ra = est[:, a], ref[:, a]
        ea = ea - ea.mean(); ra = ra - ra.mean()
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
    seq, paths, imu, om, frames = load_segment()
    assert om is not None, "ecrot_city must ship omega_gt.txt"
    assert len(frames) == SEG['n_frames'], f"got {len(frames)} frames"
    wins = frame_windows(SEG['n_frames'])
    w0 = ref_omega(om, *wins[0])
    assert np.linalg.norm(w0) > 1e-3, "omega_gt should be non-trivial"
    net = build_net(seq)
    assert net._C_mat.shape == (seq.H, seq.W, 2, 3)
    print(f"[ok] setup: {len(frames)} frames, |omega_gt[0]|="
          f"{np.linalg.norm(w0):.3f} rad/s, C={net._C_mat.shape}")


# %%
def rung0_baseline():
    """Vision-only thesis net on the segment; characterize the failure."""
    seq, paths, imu, om, frames = load_segment()
    wins = frame_windows(len(frames))
    net = build_net(seq)
    # Warm start from the gyro at t_start (mirrors reported runs: initial_R None
    # => R_init = omega_gyro(t_start) * dt).
    R_init = get_gyro_for_frame(imu, *wins[0]) * SEG['frame_duration']
    net.initialize_from_rotation(R_init)

    cond_M = float(np.linalg.cond(build_R_normal_equations(net._C_mat)))
    est, ref, curls = [], [], []
    for (V, _t), (t_lo, t_hi) in zip(frames, wins):
        net.step(V, n_iters=config.ITERS_PER_FRAME)      # vision-only (no anchor)
        est.append(net.R / SEG['frame_duration'])
        ref.append(ref_omega(om, t_lo, t_hi))
        curls.append(curl_share(net.G))
    s = score_frames(est, ref)
    s['cond_M'] = cond_M
    s['mean_curl_share'] = float(np.mean(curls))
    print("\n=== RUNG 0  vision-only baseline (ecrot_city seg_A) ===")
    print(f"  err  mean/median : {s['mean_err']:7.2f} / {s['median_err']:7.2f} deg/s")
    print(f"  dir  mean/median : {s['mean_dir']:7.2f} / {s['median_dir']:7.2f} deg")
    print(f"  beta mean        : {s['mean_beta']:7.3f}  (1.0 = perfect scale)")
    print(f"  axis_r  (x,y,z)  : {[round(v,3) for v in s['axis_r']]}")
    print(f"  axis_bias(x,y,z) : {[round(v,4) for v in s['axis_bias']]}")
    print(f"  cond(M)          : {cond_M:7.1f}")
    print(f"  curl_share mean  : {s['mean_curl_share']:7.3f}")
    # Reproduction self-check: the run must complete and produce finite metrics.
    ok = np.isfinite(s['mean_dir']) and np.isfinite(s['mean_err'])
    print(f"[{'ok' if ok else 'FAIL'}] rung0 produced finite metrics")
    return s


# %%
def rung1a_kinematics():
    """Perfect flow F*=C*R_gt inverted back through the R least-squares.
    Must return R_gt (dir~0); failure means the C-matrix / solve is broken."""
    seq, paths, imu, om, frames = load_segment()
    wins = frame_windows(len(frames))
    net = build_net(seq)                      # only for its C matrix
    C = net._C_mat
    M_inv = build_R_normal_equations(C)
    dt = SEG['frame_duration']
    est, ref = [], []
    for (t_lo, t_hi) in wins:
        w = ref_omega(om, t_lo, t_hi)
        R_gt = w * dt
        F_star = gt_flow(net, R_gt)
        R_hat = solve_R_lstsq(M_inv, C, F_star)
        est.append(R_hat / dt)
        ref.append(w)
    s = score_frames(est, ref)
    print("\n=== RUNG 1a  kinematic self-consistency (F*=C.R_gt -> R) ===")
    print(f"  dir  mean/median : {s['mean_dir']:7.4f} / {s['median_dir']:7.4f} deg")
    print(f"  beta mean        : {s['mean_beta']:7.4f}")
    ok = s['mean_dir'] < 1.0 and abs(s['mean_beta'] - 1.0) < 0.05
    print(f"[{'ok' if ok else 'FAIL'}] rung1a: perfect flow recovers R_gt "
          f"(dir<1 deg, beta~1)")
    if not ok:
        print("  >>> GATE: kinematic inversion is broken -> investigate "
              "camera.py (build_kinematic_matrix / solve_R_lstsq).")
    return s, ok


# %%
def rung1b_clamped_flow(anchored):
    """Clamp F = F* (perfect flow) through the relaxation; let R evolve.
    anchored=False: pure vision (R from kinematics only).
    anchored=True : also anchor R toward omega_gt (do flow and anchor agree?)."""
    seq, paths, imu, om, frames = load_segment()
    wins = frame_windows(len(frames))
    net = build_net(seq)
    dt = SEG['frame_duration']
    R_init = get_gyro_for_frame(imu, *wins[0]) * dt
    net.initialize_from_rotation(R_init)
    net.q_F.update = lambda lr: None         # freeze F (instance-level no-op)

    est, ref = [], []
    max_drift = 0.0
    for (V, _t), (t_lo, t_hi) in zip(frames, wins):
        w = ref_omega(om, t_lo, t_hi)
        F_star = gt_flow(net, w * dt)
        net.q_F.value = F_star.copy()
        net.step(V, n_iters=config.ITERS_PER_FRAME,
                 omega_imu=(w if anchored else None))
        max_drift = max(max_drift, float(np.max(np.abs(net.q_F.value - F_star))))
        est.append(net.R / dt)
        ref.append(w)
    s = score_frames(est, ref)
    tag = 'anchored' if anchored else 'anchor-free'
    print(f"\n=== RUNG 1b  clamped perfect flow [{tag}] ===")
    print(f"  err  mean/median : {s['mean_err']:7.2f} / {s['median_err']:7.2f} deg/s")
    print(f"  dir  mean/median : {s['mean_dir']:7.2f} / {s['median_dir']:7.2f} deg")
    print(f"  beta mean        : {s['mean_beta']:7.3f}")
    frozen_ok = max_drift < 1e-9
    print(f"[{'ok' if frozen_ok else 'FAIL'}] F stayed clamped "
          f"(max drift {max_drift:.2e})")
    return s


def _decision_gate(s1a_ok, s1b_free):
    print("\n=== DECISION GATE ===")
    if not s1a_ok:
        print("  1a FAILED -> kinematic inversion broken. Fix camera.py; "
              "the rest of the ladder reorients.")
        return
    if s1b_free['mean_dir'] > 10.0:
        print("  1a ok, 1b(anchor-free) dir high -> relaxation cannot exploit "
              "correct flow. Investigate the R-update / inter-cost coupling.")
    else:
        print("  1a ok, 1b(anchor-free) recovers direction -> correct flow => "
              "correct omega. Failure is UPSTREAM in flow estimation "
              "(V+F.G=0 with a bad G). Rung 2 (GT gradient) is justified.")


def run_all():
    _selfcheck()
    rung0_baseline()
    _, s1a_ok = rung1a_kinematics()
    s1b_free = rung1b_clamped_flow(anchored=False)
    rung1b_clamped_flow(anchored=True)
    _decision_gate(s1a_ok, s1b_free)


# %%
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rung', default='all',
                    choices=['setup', '0', '1a', '1b', 'all'])
    args = ap.parse_args()

    if args.rung == 'setup':
        _selfcheck()
    elif args.rung == '0':
        rung0_baseline()
    elif args.rung == '1a':
        rung1a_kinematics()
    elif args.rung == '1b':
        rung1b_clamped_flow(anchored=False)
        rung1b_clamped_flow(anchored=True)
    elif args.rung == 'all':
        run_all()
