"""
Validation of Interacting Maps implementation.

Three experiments:
  1. Single-frame convergence: maps + R at different iterations (visual)
  2. Multi-frame tracking: estimated ω vs GT over 3 seconds (quantitative)
  3. Parameter influence: iterations and frame duration sweeps

Usage:
    python validation_convergence.py              # Run all
    python validation_convergence.py --exp 1      # Single experiment
    python validation_convergence.py --exp 2 --video  # With frame export
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import csv

from config import (DATASET_CONFIGS, THESIS_PARAMS, COOK_PARAMS,
                    ITERS_PER_FRAME, get_dataset_paths, get_initial_R_from_imu)
from data_loader import EventFrameSequence, CameraCalibration
from interacting_maps.network import InteractingMaps
from interacting_maps.network_dissertation import InteractingMapsThesis
from interacting_maps.camera import build_kinematic_matrix

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET = 'boxes_rotation'
USE_THESIS_VERSION = True
USE_IMU = False  # Only applies when USE_THESIS_VERSION = True

cfg = DATASET_CONFIGS[DATASET]
initial_R = cfg['initial_R']
if initial_R is None:
    initial_R = get_initial_R_from_imu(DATASET)
    print(f"Auto-initialized R from IMU: {initial_R}")
paths = get_dataset_paths(DATASET)

EVENTS_FILE = paths['events']
CALIB_FILE = paths['calib']
IMU_FILE = paths['imu']

T_START = cfg['t_start']
FRAME_DURATION = cfg['frame_duration']
N_FRAMES = cfg['n_frames']

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_imu(path: str) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float64)

def get_gyro_for_frame(imu_data, t_lo, t_hi):
    mask = (imu_data[:, 0] >= t_lo) & (imu_data[:, 0] < t_hi)
    if np.sum(mask) == 0:
        idx = np.argmin(np.abs(imu_data[:, 0] - (t_lo + t_hi) / 2))
        return imu_data[idx, 4:7]
    return np.mean(imu_data[mask, 4:7], axis=0)

def make_network(H, W, fx, fy, cx, cy, R_init, frame_duration=None):
    if frame_duration is None:
        frame_duration = FRAME_DURATION
    if USE_THESIS_VERSION:
        net = InteractingMapsThesis(H=H, W=W, fx=fx, fy=fy, cx=cx, cy=cy,
                                   **THESIS_PARAMS, frame_duration=frame_duration)
        net.initialize_from_rotation(R_init)
    else:
        net = InteractingMaps(H=H, W=W, fx=fx, fy=fy, cx=cx, cy=cy, **COOK_PARAMS)
        # Clean initialization — NOT random noise for G
        net.I = np.random.randn(H+1, W+1) * 0.001  # tiny noise for symmetry breaking
        net.G = np.zeros((H, W, 2), dtype=np.float64)  # G=0: OFCE will bootstrap
        net.F = np.einsum('hwij,j->hwi', net._C_mat, R_init)  # F consistent with R
        net.R = R_init.copy()
    return net

def normalise(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-10)

def flow_to_rgb(flow):
    from matplotlib.colors import hsv_to_rgb
    fx, fy = flow[..., 0], flow[..., 1]
    angle = (np.arctan2(fy, fx) + np.pi) / (2 * np.pi)
    mag = np.sqrt(fx**2 + fy**2)
    mag_norm = mag / (mag.max() + 1e-10)
    hsv = np.stack([angle, np.ones_like(angle), mag_norm], axis=-1)
    return hsv_to_rgb(hsv)


# ===========================================================================
# EXPERIMENT 1: Single-Frame Map Convergence (Visual)
# ===========================================================================

def experiment_single_frame_convergence(frame_idx=0, max_iters=100,
                                        snapshot_iters=None):
    """
    Shows how maps develop iteration-by-iteration within a single frame.
    This is the 'visual inspection' your professor wants.
    
    Saves: one image with V, I, G, F at multiple iteration checkpoints.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Single-Frame Map Convergence")
    print("="*70)

    if snapshot_iters is None:
        snapshot_iters = [1, 3, 5, 10, 25, 50, max_iters]
    snapshot_iters = [i for i in snapshot_iters if i <= max_iters]

    # Load single frame
    seq = EventFrameSequence(
        EVENTS_FILE, CALIB_FILE,
        frame_duration=FRAME_DURATION, t_start=T_START,
        n_frames=frame_idx + 1, clip_value=10.0,
        sensor_size=cfg.get('sensor_size', None),
    )
    frames = list(seq)
    V, t_mid = frames[frame_idx]
    H, W = seq.H, seq.W
    fx, fy, cx, cy = seq.calib.fx, seq.calib.fy, seq.calib.cx, seq.calib.cy

    imu_data = load_imu(IMU_FILE)
    t_lo = T_START + frame_idx * FRAME_DURATION
    t_hi = t_lo + FRAME_DURATION
    gt_omega = get_gyro_for_frame(imu_data, t_lo, t_hi)
    gt_R = gt_omega * FRAME_DURATION

    print(f"  GT ω = ({gt_omega[0]:.3f}, {gt_omega[1]:.3f}, {gt_omega[2]:.3f}) rad/s")

    # Run iteration by iteration, save snapshots
    net = make_network(H, W, fx, fy, cx, cy, gt_R)
    snapshots = {}
    R_history = []
    res_history = []

    if USE_THESIS_VERSION:
        net.q_V.value = V
        for it in range(1, max_iters + 1):
            for q in [net.q_I, net.q_G, net.q_F, net.q_R]:
                q.reset_gradient()
            for cost in net.costs:
                cost.compute_and_send_gradients()
            net.q_I.update(1.0)
            net.q_G.update(1.0)
            net.q_F.update(1.0)
            net.q_R.update(1.0)
            net.q_I.value = np.clip(net.q_I.value, -10.0, 10.0)
            net.q_G.value = np.clip(net.q_G.value, -5.0, 5.0)
            net.q_F.value = np.clip(net.q_F.value, -10.0, 10.0)
            net.q_R.value = np.clip(net.q_R.value, -1.0, 1.0)

            R_history.append(net.R.copy())
            res_history.append(net.residual_VFG(V))

            if it in snapshot_iters:
                snapshots[it] = {
                    'I': net.I.copy(), 'G': net.G.copy(),
                    'F': net.F.copy(), 'R': net.R.copy(),
                }
    else:
        for it in range(1, max_iters + 1):
            net.update_F_from_VG(V)
            net.update_G_from_VF(V)
            net.update_G_from_I()
            net.update_I_from_G()
            net.update_F_from_RC()
            net.update_R_from_FC()

            R_history.append(net.R.copy())
            res_history.append(net.residual_VFG(V))

            if it in snapshot_iters:
                snapshots[it] = {
                    'I': net.I[:H, :W].copy(), 'G': net.G.copy(),
                    'F': net.F.copy(), 'R': net.R.copy(),
                }

    R_history = np.array(R_history)

    # --- PLOT: Map snapshots ---
    n_snaps = len(snapshot_iters)
    fig, axes = plt.subplots(5, n_snaps, figsize=(3 * n_snaps, 14))
    version_str = "Thesis+IMU" if (USE_THESIS_VERSION and USE_IMU) else \
                  "Thesis" if USE_THESIS_VERSION else "Cook"
    fig.suptitle(f'Experiment 1: Map Convergence ({version_str}, {DATASET})\n'
                 f'GT ω = ({gt_omega[0]:.3f}, {gt_omega[1]:.3f}, {gt_omega[2]:.3f}) rad/s',
                 fontsize=11)

    for col, it in enumerate(snapshot_iters):
        snap = snapshots[it]
        omega_it = snap['R'] / FRAME_DURATION

        axes[0, col].imshow(V, cmap='RdBu', vmin=-1, vmax=1)
        axes[0, col].set_title(f'iter {it}', fontsize=9)

        axes[1, col].imshow(normalise(snap['I']), cmap='gray')

        G_mag = np.sqrt(snap['G'][..., 0]**2 + snap['G'][..., 1]**2)
        axes[2, col].imshow(normalise(G_mag), cmap='hot')

        axes[3, col].imshow(flow_to_rgb(snap['F']))

        # R error for this iteration
        err = np.linalg.norm(snap['R'] - gt_R) / FRAME_DURATION * 180 / np.pi
        axes[4, col].bar(['ωx','ωy','ωz'], omega_it, color=['#4e79a7','#f28e2b','#e15759'])
        axes[4, col].bar(['ωx','ωy','ωz'], gt_omega, color='none',
                        edgecolor='black', linewidth=2)
        axes[4, col].set_title(f'err={err:.1f}°/s', fontsize=8)
        axes[4, col].set_ylim(min(gt_omega.min(), omega_it.min()) * 1.3,
                              max(gt_omega.max(), omega_it.max()) * 1.3)

    row_labels = ['V (input)', 'I (intensity)', '|G| (gradient)', 'F (flow)', 'ω (est vs GT)']
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=9)
    for ax in axes[:4].flat:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'exp1_convergence_{DATASET}_{version_str}.png', dpi=150)
    plt.show()

    # --- PLOT: R convergence curve ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    R_err = np.linalg.norm(R_history - gt_R, axis=1) / FRAME_DURATION * 180 / np.pi
    axes[0].semilogy(R_err)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('ω error (°/s)')
    axes[0].set_title('Convergence of ω estimate')
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(res_history)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('|V + F·G|')
    axes[1].set_title('OFCE Residual')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'exp1_convergence_curve_{DATASET}_{version_str}.png', dpi=150)
    plt.show()


# ===========================================================================
# EXPERIMENT 2: Multi-Frame Tracking (3 seconds, with export)
# ===========================================================================

def experiment_tracking(n_frames=None, n_iters=None, save_frames=False):
    """
    Run network over 3 seconds. Compare estimated ω to GT (IMU).
    Optionally save per-frame map images for video assembly.
    
    This is THE main result your professor wants to see.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Multi-Frame Angular Velocity Tracking")
    print("="*70)

    if n_frames is None:
        n_frames = N_FRAMES
    if n_iters is None:
        n_iters = ITERS_PER_FRAME

    version_str = "thesis_imu" if (USE_THESIS_VERSION and USE_IMU) else \
                  "thesis" if USE_THESIS_VERSION else "cook"

    # Output directory
    out_dir = f'results/{DATASET}_{version_str}'
    os.makedirs(out_dir, exist_ok=True)
    if save_frames:
        frames_dir = os.path.join(out_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)

    seq = EventFrameSequence(
        EVENTS_FILE, CALIB_FILE,
        frame_duration=FRAME_DURATION, t_start=T_START,
        n_frames=n_frames, clip_value=10.0,
        sensor_size=cfg.get('sensor_size', None),
    )
    H, W = seq.H, seq.W
    fx, fy, cx, cy = seq.calib.fx, seq.calib.fy, seq.calib.cx, seq.calib.cy
    imu_data = load_imu(IMU_FILE)

    net = make_network(H, W, fx, fy, cx, cy, initial_R)

    # Storage
    rows = []

    for k, (V, t_mid) in enumerate(seq):
        t_lo = T_START + k * FRAME_DURATION
        t_hi = t_lo + FRAME_DURATION
        gt_omega = get_gyro_for_frame(imu_data, t_lo, t_hi)

        # Run
        if USE_THESIS_VERSION and USE_IMU:
            net.step(V, n_iters=n_iters, omega_imu=gt_omega)
        else:
            net.step(V, n_iters=n_iters)

        omega_est = net.R / FRAME_DURATION

        # Metrics
        err = np.linalg.norm(omega_est - gt_omega) * 180 / np.pi
        norm_est = np.linalg.norm(omega_est)
        norm_gt = np.linalg.norm(gt_omega)
        if norm_est > 1e-6 and norm_gt > 1e-6:
            cos_a = np.clip(np.dot(omega_est, gt_omega) / (norm_est * norm_gt), -1, 1)
            dir_err = np.degrees(np.arccos(cos_a))
            beta = norm_gt / norm_est
        else:
            dir_err = 180.0
            beta = 0.0

        rows.append({
            'frame': k, 'time': t_mid,
            'est_wx': omega_est[0], 'est_wy': omega_est[1], 'est_wz': omega_est[2],
            'gt_wx': gt_omega[0], 'gt_wy': gt_omega[1], 'gt_wz': gt_omega[2],
            'err_deg_s': err, 'dir_err_deg': dir_err, 'beta': beta,
        })

        # Save frame images for video
        if save_frames:
            _save_frame(frames_dir, k, V, net, H, W, omega_est, gt_omega)

        # Print progress
        if (k + 1) % 10 == 0 or k == 0:
            print(f"  Frame {k+1:4d}/{n_frames} | "
                  f"ω_est=({omega_est[0]:+.3f},{omega_est[1]:+.3f},{omega_est[2]:+.3f}) | "
                  f"ω_gt=({gt_omega[0]:+.3f},{gt_omega[1]:+.3f},{gt_omega[2]:+.3f}) | "
                  f"err={err:.1f}°/s | dir={dir_err:.1f}°")

    # Save CSV
    csv_path = os.path.join(out_dir, 'tracking.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  CSV saved: {csv_path}")

    # Summary
    err_all = np.array([r['err_deg_s'] for r in rows])
    dir_all = np.array([r['dir_err_deg'] for r in rows])
    beta_all = np.array([r['beta'] for r in rows])

    print(f"\n  {'='*50}")
    print(f"  RESULTS: {version_str}, {DATASET}, {n_frames*FRAME_DURATION:.1f}s")
    print(f"  {'='*50}")
    print(f"  Total error:     mean={np.mean(err_all):.1f}°/s, median={np.median(err_all):.1f}°/s")
    print(f"  Direction error: mean={np.mean(dir_all):.1f}°, median={np.median(dir_all):.1f}°")
    print(f"  Scale factor β:  mean={np.mean(beta_all):.2f} ± {np.std(beta_all):.2f}")

    # --- PLOT: ω tracking ---
    _plot_tracking(rows, out_dir, version_str)

    return rows


def _save_frame(frames_dir, k, V, net, H, W, omega_est, gt_omega):
    """Save V, I, G, F as a single image for video assembly."""
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))

    axes[0].imshow(V, cmap='RdBu', vmin=-1, vmax=1)
    axes[0].set_title('V (input)', fontsize=9)

    I_disp = net.I if net.I.shape == (H, W) else net.I[:H, :W]
    axes[1].imshow(normalise(I_disp), cmap='gray')
    axes[1].set_title('I (inferred)', fontsize=9)

    G_mag = np.sqrt(net.G[..., 0]**2 + net.G[..., 1]**2)
    axes[2].imshow(normalise(G_mag), cmap='hot')
    axes[2].set_title('|G| (gradient)', fontsize=9)

    axes[3].imshow(flow_to_rgb(net.F))
    axes[3].set_title('F (flow)', fontsize=9)

    # ω bar chart
    x = np.arange(3)
    axes[4].bar(x - 0.15, omega_est, 0.3, color=['#4e79a7','#f28e2b','#e15759'], label='Est')
    axes[4].bar(x + 0.15, gt_omega, 0.3, color='gray', alpha=0.6, label='GT')
    axes[4].set_xticks(x)
    axes[4].set_xticklabels(['ωx','ωy','ωz'])
    axes[4].set_title(f'Frame {k}', fontsize=9)
    axes[4].legend(fontsize=7)

    for ax in axes[:4]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(frames_dir, f'frame_{k:04d}.png'), dpi=80,
                bbox_inches='tight')
    plt.close(fig)


def _plot_tracking(rows, out_dir, version_str):
    """The main tracking plot: estimated ω vs GT ω."""
    times = np.array([r['time'] for r in rows]) - rows[0]['time']
    omega_est = np.array([[r['est_wx'], r['est_wy'], r['est_wz']] for r in rows])
    omega_gt = np.array([[r['gt_wx'], r['gt_wy'], r['gt_wz']] for r in rows])
    err = np.array([r['err_deg_s'] for r in rows])

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'Angular Velocity Tracking — {DATASET} ({version_str})\n'
                 f'dt={FRAME_DURATION*1000:.0f}ms, {ITERS_PER_FRAME} iters/frame, '
                 f'{len(rows)*FRAME_DURATION:.1f}s duration', fontsize=11)

    labels = ['ωx', 'ωy', 'ωz']
    colors = ['#4e79a7', '#f28e2b', '#e15759']

    for i in range(3):
        axes[i].plot(times, omega_gt[:, i], 'k-', lw=1.0, alpha=0.8, label='GT (IMU)')
        axes[i].plot(times, omega_est[:, i], color=colors[i], lw=1.5, label='Estimated')
        axes[i].set_ylabel(f'{labels[i]} (rad/s)')
        axes[i].legend(loc='upper right', fontsize=8)
        axes[i].grid(True, alpha=0.3)

    axes[3].plot(times, err, 'r-', lw=1.0)
    axes[3].set_ylabel('Error (°/s)')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_title(f'Error: mean={np.mean(err):.1f}°/s, median={np.median(err):.1f}°/s')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'tracking_plot.png'), dpi=150)
    plt.show()


# ===========================================================================
# EXPERIMENT 3: Parameter Influence (iterations + frame duration)
# ===========================================================================

def experiment_parameter_influence(frame_idx=0):
    """
    Two sweeps in one:
    (a) Number of iterations (single frame, fixed dt)
    (b) Frame duration (single frame, fixed iters)
    
    Shows the operating regime and diminishing returns.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Parameter Influence")
    print("="*70)

    imu_data = load_imu(IMU_FILE)

    # --- (a) Iterations sweep ---
    print("\n  (a) Iterations sweep (dt=20ms):")
    iter_counts = [1, 3, 5, 10, 20, 50, 100]
    iter_results = []

    seq = EventFrameSequence(
        EVENTS_FILE, CALIB_FILE,
        frame_duration=FRAME_DURATION, t_start=T_START,
        n_frames=frame_idx + 1, clip_value=10.0,
        sensor_size=cfg.get('sensor_size', None),
    )
    frames = list(seq)
    V, _ = frames[frame_idx]
    H, W = seq.H, seq.W
    fx, fy, cx, cy = seq.calib.fx, seq.calib.fy, seq.calib.cx, seq.calib.cy

    t_lo = T_START + frame_idx * FRAME_DURATION
    t_hi = t_lo + FRAME_DURATION
    gt_omega = get_gyro_for_frame(imu_data, t_lo, t_hi)
    gt_R = gt_omega * FRAME_DURATION

    for n_iters in iter_counts:
        net = make_network(H, W, fx, fy, cx, cy, gt_R)
        net.step(V, n_iters=n_iters)
        omega_est = net.R / FRAME_DURATION
        err = np.linalg.norm(omega_est - gt_omega) * 180 / np.pi
        iter_results.append({'iters': n_iters, 'err': err})
        print(f"    iters={n_iters:4d} | err={err:.2f}°/s")

    # --- (b) Frame duration sweep ---
    print("\n  (b) Frame duration sweep (iters=6):")
    durations = [0.005, 0.010, 0.015, 0.020, 0.030, 0.050]
    dt_results = []

    for dt in durations:
        seq2 = EventFrameSequence(
            EVENTS_FILE, CALIB_FILE,
            frame_duration=dt, t_start=T_START,
            n_frames=frame_idx + 1, clip_value=10.0,
            sensor_size=cfg.get('sensor_size', None),
        )
        frames2 = list(seq2)
        V2, _ = frames2[frame_idx]
        H2, W2 = seq2.H, seq2.W

        t_lo2 = T_START + frame_idx * dt
        t_hi2 = t_lo2 + dt
        gt_omega2 = get_gyro_for_frame(imu_data, t_lo2, t_hi2)
        gt_R2 = gt_omega2 * dt

        net = make_network(H2, W2, fx, fy, cx, cy, gt_R2, frame_duration=dt)
        net.step(V2, n_iters=ITERS_PER_FRAME)
        omega_est2 = net.R / dt
        err2 = np.linalg.norm(omega_est2 - gt_omega2) * 180 / np.pi
        n_active = np.count_nonzero(V2)
        dt_results.append({'dt_ms': dt*1000, 'err': err2, 'active': n_active})
        print(f"    dt={dt*1000:5.1f}ms | active={n_active:5d} | err={err2:.2f}°/s")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    version_str = "Thesis" if USE_THESIS_VERSION else "Cook"
    fig.suptitle(f'Experiment 3: Parameter Influence ({version_str}, {DATASET})', fontsize=11)

    axes[0].plot([r['iters'] for r in iter_results],
                [r['err'] for r in iter_results], 'bo-', lw=1.5)
    axes[0].set_xlabel('Iterations per frame')
    axes[0].set_ylabel('ω error (°/s)')
    axes[0].set_title('(a) Effect of iterations (dt=20ms)')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')

    axes[1].plot([r['dt_ms'] for r in dt_results],
                [r['err'] for r in dt_results], 'rs-', lw=1.5)
    axes[1].set_xlabel('Frame duration (ms)')
    axes[1].set_ylabel('ω error (°/s)')
    axes[1].set_title(f'(b) Effect of frame duration ({ITERS_PER_FRAME} iters)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'exp3_parameters_{DATASET}_{version_str}.png', dpi=150)
    plt.show()


# ===========================================================================
# ===========================================================================

def experiment_qualitative_video(n_frames=None, n_iters=None):
    """
    Save 3-column comparison frames: [Events V | Inferred I | GT Image (APS)]
    """
    print("\n" + "="*70)
    print("QUALITATIVE: Video Frame Export (Events | Estimated | GT)")
    print("="*70)

    if n_frames is None:
        n_frames = N_FRAMES
    if n_iters is None:
        n_iters = ITERS_PER_FRAME

    version_str = "thesis_imu" if (USE_THESIS_VERSION and USE_IMU) else \
                  "thesis" if USE_THESIS_VERSION else "cook"

    out_dir = f'results/{DATASET}_{version_str}/video_frames'
    os.makedirs(out_dir, exist_ok=True)

    seq = EventFrameSequence(
        EVENTS_FILE, CALIB_FILE,
        frame_duration=FRAME_DURATION, t_start=T_START,
        n_frames=n_frames, clip_value=10.0,
        sensor_size=cfg.get('sensor_size', None),
    )
    H, W = seq.H, seq.W
    fx, fy, cx, cy = seq.calib.fx, seq.calib.fy, seq.calib.cx, seq.calib.cy
    imu_data = load_imu(IMU_FILE)

    net = make_network(H, W, fx, fy, cx, cy, initial_R)

    # Load GT images
    gt_images = _try_load_gt_images(T_START, FRAME_DURATION, n_frames)

    for k, (V, t_mid) in enumerate(seq):
        t_lo = T_START + k * FRAME_DURATION
        t_hi = t_lo + FRAME_DURATION
        gt_omega = get_gyro_for_frame(imu_data, t_lo, t_hi)

        if USE_THESIS_VERSION and USE_IMU:
            net.step(V, n_iters=n_iters, omega_imu=gt_omega)
        else:
            net.step(V, n_iters=n_iters)

        # Save 3-column frame
        _save_3col_frame(out_dir, k, V, net, H, W, gt_images)

        if (k + 1) % 10 == 0 or k == 0:
            print(f"  Saved frame {k+1:4d}/{n_frames}")

    print(f"\n  Frames saved to: {out_dir}/")
    print(f"  Make video: python validation_convergence2.py --exp 5")


def _try_load_gt_images(t_start, dt, n_frames):
    """
    Load ground truth APS images from images.txt.
    Format: timestamp images/frame_00000015.png
    """
    data_dir = paths['data_dir']
    images_file = os.path.join(data_dir, 'images.txt')

    if not os.path.exists(images_file):
        print("  No images.txt found")
        return None

    img_list = []
    t_end = t_start + n_frames * dt + 0.5  # load a bit extra

    with open(images_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            t = float(parts[0])
            fname = parts[1]  # e.g. "images/frame_00000015.png"

            if t < t_start - 0.5:
                continue
            if t > t_end:
                break

            # Build full path — fname is relative to data_dir
            img_path = os.path.join(data_dir, fname)
            if os.path.exists(img_path):
                img_list.append((t, img_path))

    if img_list:
        print(f"  Loaded {len(img_list)} GT images "
              f"(t={img_list[0][0]:.3f} to {img_list[-1][0]:.3f})")
    else:
        print("  No GT images found in time range")
        return None

    return img_list


def _save_3col_frame(out_dir, k, V, net, H, W, gt_images):
    """
    Save exactly 3 columns: Events | Estimated I | GT Image
    FIXED figure size — NO bbox_inches='tight' — ensures all frames same size.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Column 1: Events (V)
    axes[0].imshow(V, cmap='RdBu', vmin=-1, vmax=1)
    axes[0].set_title('Events (V)', fontsize=10)
    axes[0].axis('off')

    # Column 2: Inferred Intensity (I)
    I_disp = net.I if net.I.shape == (H, W) else net.I[:H, :W]
    axes[1].imshow(normalise(I_disp), cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Estimated I', fontsize=10)
    axes[1].axis('off')

    # Column 3: Ground Truth Image (APS)
    if gt_images is not None and len(gt_images) > 0:
        t_frame = T_START + k * FRAME_DURATION + FRAME_DURATION / 2
        closest = min(gt_images, key=lambda x: abs(x[0] - t_frame))
        gt_img = plt.imread(closest[1])
        axes[2].imshow(gt_img, cmap='gray')
        axes[2].set_title('GT Image (APS)', fontsize=10)
    else:
        axes[2].text(0.5, 0.5, 'No GT\nImages', ha='center', va='center',
                     fontsize=14, transform=axes[2].transAxes)
        axes[2].set_facecolor('#f0f0f0')
        axes[2].set_title('GT Image (N/A)', fontsize=10)
    axes[2].axis('off')

    plt.suptitle(f'Frame {k:04d}', fontsize=10)
    plt.tight_layout()
    # FIXED output — no bbox_inches='tight'!
    plt.savefig(os.path.join(out_dir, f'frame_{k:04d}.png'), dpi=100)
    plt.close(fig)


def make_video_imageio(frames_dir, output_path, fps=15):
    """Make video — resizes all frames to match first (handles tiny size variations)."""
    import imageio
    import glob
    from PIL import Image

    frame_files = sorted(glob.glob(os.path.join(frames_dir, 'frame_*.png')))

    if not frame_files:
        print(f"No frames found in {frames_dir}")
        return

    # Get size from first frame, make divisible by 16 (codec requirement)
    first = Image.open(frame_files[0])
    w, h = first.size
    w = (w // 16) * 16
    h = (h // 16) * 16

    writer = imageio.get_writer(output_path, fps=fps)
    for f in frame_files:
        img = Image.open(f).resize((w, h), Image.LANCZOS)
        writer.append_data(np.array(img))
    writer.close()
    print(f"Video saved: {output_path} ({w}×{h}, {len(frame_files)} frames, {fps}fps)")


def make_video_from_frames():
    """Helper to assemble video from already-saved frames."""
    version_str = "thesis_imu" if (USE_THESIS_VERSION and USE_IMU) else \
                  "thesis" if USE_THESIS_VERSION else "cook"
    frames_dir = f'results/{DATASET}_{version_str}/video_frames'
    output_path = f'results/{DATASET}_{version_str}/comparison.mp4'
    make_video_imageio(frames_dir, output_path)
# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Validation Experiments')
    parser.add_argument('--exp', type=int, default=0, help='1, 2, 3, 4=video, or 0=all')
    parser.add_argument('--frame', type=int, default=0, help='Frame index for Exp 1/3')
    parser.add_argument('--video', action='store_true', help='Save frames for video (Exp 2)')
    args = parser.parse_args()

    experiments = {
        1: lambda: experiment_single_frame_convergence(frame_idx=args.frame),
        2: lambda: experiment_tracking(save_frames=args.video),
        3: lambda: experiment_parameter_influence(frame_idx=args.frame),
        4: lambda: experiment_qualitative_video(),
        5: lambda: make_video_from_frames(),

    }

    if args.exp == 0:
        print("Running ALL experiments...")
        for num, func in experiments.items():
            try:
                func()
            except Exception as e:
                print(f"  Experiment {num} failed: {e}")
    else:
        experiments[args.exp]()

'''# Run all (1, 2, 3 — skips 4 since it's slow)
python validation_convergence2.py

# Run only qualitative video export
python validation_convergence2.py --exp 4

# Run tracking with frame export for video
python validation_convergence2.py --exp 2 --video

# Run single-frame convergence for frame 5
python validation_convergence2.py --exp 1 --frame 5'''