"""
Validation strategies:
1. How iterations converge (within a single frame)
2. Show map updates iteration-by-iteration
3. Parameter sweeps:
   a. Initialize near/far from GT angular velocity
   b. Number of iterations vs. quality
   c. Number of events (frame duration)
4. Basin of attraction: start far, check if optimization returns to minimum

Usage:
    python validation_convergence.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import hsv_to_rgb
from copy import deepcopy
import os

from config import DATASET_CONFIGS, THESIS_PARAMS, COOK_PARAMS, ITERS_PER_FRAME, get_dataset_paths, get_initial_R_from_imu
from data_loader import EventFrameSequence, CameraCalibration
from interacting_maps.network import InteractingMaps
from interacting_maps.network_dissertation import InteractingMapsThesis
from interacting_maps.camera import build_kinematic_matrix


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET = 'poster_rotation'  # shapes_rotation, poster_rotation
USE_THESIS_VERSION = True
USE_IMU = True # only when thesis = True

cfg = DATASET_CONFIGS[DATASET]
initial_R = cfg['initial_R']
if initial_R is None:
    initial_R = get_initial_R_from_imu(DATASET)
    print(f"Auto-initialized R from IMU: {initial_R}")
paths = get_dataset_paths(DATASET)


EVENTS_FILE = paths['events']
CALIB_FILE = paths['calib']
IMU_FILE = paths['imu']
GT_FILE = paths['groundtruth']

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


def get_gt_omega(imu_data, t_start, frame_duration, frame_idx):
    """Get ground truth angular velocity for a specific frame."""
    t_lo = t_start + frame_idx * frame_duration
    t_hi = t_lo + frame_duration
    return get_gyro_for_frame(imu_data, t_lo, t_hi)


def make_network(H, W, fx, fy, cx, cy, R_init, frame_duration=None):
    """Create and initialize a network."""
    if frame_duration is None:
        frame_duration = FRAME_DURATION

    if USE_THESIS_VERSION:
        net = InteractingMapsThesis(H=H, W=W, fx=fx, fy=fy, cx=cx, cy=cy,
                                   **THESIS_PARAMS, frame_duration=frame_duration)
        net.initialize_from_rotation(R_init)
    else:
        net = InteractingMaps(H=H, W=W, fx=fx, fy=fy, cx=cx, cy=cy,
                             **COOK_PARAMS)
        
        net.reset(scale=0.5)
        net.R = R_init.copy()
        # Initialize F = C·R so OFCE has something to work with
        net.F = np.einsum('hwij,j->hwi', net._C_mat, R_init)
    return net


def flow_to_rgb(flow):
    fx, fy = flow[..., 0], flow[..., 1]
    angle = (np.arctan2(fy, fx) + np.pi) / (2 * np.pi)
    mag = np.sqrt(fx**2 + fy**2)
    mag_norm = mag / (mag.max() + 1e-10)
    hsv = np.stack([angle, np.ones_like(angle), mag_norm], axis=-1)
    return hsv_to_rgb(hsv)


def normalise(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-10)


def load_single_frame(frame_idx=0):
    """Load events and return a single V frame + metadata."""
    seq = EventFrameSequence(
        EVENTS_FILE, CALIB_FILE,
        frame_duration=FRAME_DURATION,
        t_start=T_START,
        n_frames=frame_idx + 1,
        clip_value=10.0,
        sensor_size=cfg.get('sensor_size', None),
    )
    frames = list(seq)
    V, t_mid = frames[frame_idx]
    return V, t_mid, seq.calib, seq.H, seq.W


# ===========================================================================
# EXPERIMENT 1: Iteration-by-Iteration Convergence (Within a Single Frame)
# ===========================================================================

def experiment_convergence_within_frame(frame_idx=0, max_iters=200):
    """
    Track how residuals and R evolve at EACH iteration within a single frame.
    This directly shows "how iterations converge".
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Convergence within a single frame")
    print("="*70)

    V, t_mid, calib, H, W = load_single_frame(frame_idx)
    fx, fy, cx, cy = calib.fx, calib.fy, calib.cx, calib.cy

    imu_data = load_imu(IMU_FILE)
    gt_omega = get_gt_omega(imu_data, T_START, FRAME_DURATION, frame_idx)
    gt_R = gt_omega * FRAME_DURATION  # rad/s → rad/frame

    print(f"Frame {frame_idx}: GT angular velocity = {gt_omega} rad/s")
    print(f"Frame {frame_idx}: GT R (rad/frame)    = {gt_R}")

    # Initialize near GT
    net = make_network(H, W, fx, fy, cx, cy, gt_R)

    # Track metrics at each iteration
    res_VFG = []
    res_GI = []
    R_history = []
    R_error = []
    total_cost = []

    # Manually iterate (don't use net.step, do it iteration by iteration)
    if USE_THESIS_VERSION:
        net.q_V.value = V
        for it in range(max_iters):
            # One iteration of Algorithm 6.5
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

            # Record
            res_VFG.append(net.residual_VFG(V))
            res_GI.append(net.residual_GI())
            R_history.append(net.R.copy())
            R_error.append(np.linalg.norm(net.R - gt_R))
            total_cost.append(res_VFG[-1] + res_GI[-1])
    else:
        for it in range(max_iters):
            net.update_F_from_VG(V)
            net.update_G_from_VF(V)
            net.update_G_from_I()
            net.update_I_from_G()
            net.update_F_from_RC()
            net.update_R_from_FC()

            res_VFG.append(net.residual_VFG(V))
            res_GI.append(net.residual_GI())
            R_history.append(net.R.copy())
            R_error.append(np.linalg.norm(net.R - gt_R))
            total_cost.append(res_VFG[-1] + res_GI[-1])

    R_history = np.array(R_history)

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Experiment 1: Convergence Within Frame {frame_idx}\n'
                 f'{"Thesis" if USE_THESIS_VERSION else "Cook"} version, '
                 f'initialized near GT', fontsize=12)

    # (a) Residuals vs iteration
    ax = axes[0, 0]
    ax.semilogy(res_VFG, 'b-', label='|V + F·G| (OFCE)')
    ax.semilogy(res_GI, 'r-', label='|G - ∇I| (Spatial)')
    ax.semilogy(total_cost, 'k--', label='Total', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean Absolute Residual')
    ax.set_title('(a) Constraint Residuals vs. Iteration')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) R components vs iteration
    ax = axes[0, 1]
    ax.plot(R_history[:, 0], label='Rx (est)')
    ax.plot(R_history[:, 1], label='Ry (est)')
    ax.plot(R_history[:, 2], label='Rz (est)')
    ax.axhline(gt_R[0], color='C0', ls='--', alpha=0.5, label=f'Rx GT={gt_R[0]:.4f}')
    ax.axhline(gt_R[1], color='C1', ls='--', alpha=0.5, label=f'Ry GT={gt_R[1]:.4f}')
    ax.axhline(gt_R[2], color='C2', ls='--', alpha=0.5, label=f'Rz GT={gt_R[2]:.4f}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('R (rad/frame)')
    ax.set_title('(b) Rotation Estimate vs. Iteration')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (c) R error vs iteration
    ax = axes[1, 0]
    ax.semilogy(R_error)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('||R_est - R_gt||')
    ax.set_title('(c) Angular Velocity Error vs. Iteration')
    ax.grid(True, alpha=0.3)

    # (d) Convergence rate (ratio of successive residuals)
    ax = axes[1, 1]
    ratios = [total_cost[i+1] / (total_cost[i] + 1e-15) for i in range(len(total_cost)-1)]
    ax.plot(ratios, 'g-')
    ax.axhline(1.0, color='k', ls='--', alpha=0.3)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost(i+1) / Cost(i)')
    ax.set_title('(d) Convergence Rate')
    ax.set_ylim(0.8, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exp1_convergence_within_frame.png', dpi=150)
    plt.show()

    print(f"\nFinal R error: {R_error[-1]:.6f} rad/frame")
    print(f"Final residual VFG: {res_VFG[-1]:.6f}")
    print(f"Final residual GI:  {res_GI[-1]:.6f}")


# ===========================================================================
# EXPERIMENT 2: Map Visualization Iteration-by-Iteration
# ===========================================================================

def experiment_map_snapshots(frame_idx=0, max_iters=100,
                             snapshot_iters=None):
    """
    Show I, G, F maps at several iteration checkpoints.
    Visualizes HOW the maps develop, not just the final result.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Map snapshots at different iterations")
    print("="*70)

    if snapshot_iters is None:
        snapshot_iters = [1, 5, 10, 20, 50, 100]
    snapshot_iters = [i for i in snapshot_iters if i <= max_iters]

    V, t_mid, calib, H, W = load_single_frame(frame_idx)
    fx, fy, cx, cy = calib.fx, calib.fy, calib.cx, calib.cy

    imu_data = load_imu(IMU_FILE)
    gt_omega = get_gt_omega(imu_data, T_START, FRAME_DURATION, frame_idx)
    gt_R = gt_omega * FRAME_DURATION

    net = make_network(H, W, fx, fy, cx, cy, gt_R)

    # Collect snapshots
    snapshots = {}  # iter_num -> {'I': ..., 'G': ..., 'F': ..., 'R': ...}

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

            if it in snapshot_iters:
                snapshots[it] = {
                    'I': net.I.copy(),
                    'G': net.G.copy(),
                    'F': net.F.copy(),
                    'R': net.R.copy(),
                }
    else:
        for it in range(1, max_iters + 1):
            net.update_F_from_VG(V)
            net.update_G_from_VF(V)
            net.update_G_from_I()
            net.update_I_from_G()
            net.update_F_from_RC()
            net.update_R_from_FC()

            if it in snapshot_iters:
                I_disp = net.I[:H, :W]
                snapshots[it] = {
                    'I': I_disp.copy(),
                    'G': net.G.copy(),
                    'F': net.F.copy(),
                    'R': net.R.copy(),
                }

    # --- Plot ---
    n_snaps = len(snapshot_iters)
    fig, axes = plt.subplots(4, n_snaps, figsize=(3 * n_snaps, 12))
    fig.suptitle(f'Experiment 2: Map Evolution (Frame {frame_idx})\n'
                 f'GT ω = ({gt_omega[0]:.3f}, {gt_omega[1]:.3f}, {gt_omega[2]:.3f}) rad/s',
                 fontsize=11)

    row_labels = ['V (input)', 'I (intensity)', '|G| (gradient)', 'F (flow HSV)']

    for col, it in enumerate(snapshot_iters):
        snap = snapshots[it]

        # Row 0: V (same for all)
        axes[0, col].imshow(V, cmap='RdBu', vmin=-1, vmax=1)
        axes[0, col].set_title(f'iter={it}', fontsize=9)

        # Row 1: I
        axes[1, col].imshow(normalise(snap['I']), cmap='gray', vmin=0, vmax=1)

        # Row 2: |G|
        G_mag = np.sqrt(snap['G'][..., 0]**2 + snap['G'][..., 1]**2)
        axes[2, col].imshow(normalise(G_mag), cmap='hot')

        # Row 3: F (HSV)
        axes[3, col].imshow(flow_to_rgb(snap['F']))

        # R annotation
        R = snap['R']
        omega = R / FRAME_DURATION
        axes[3, col].set_xlabel(
            f'ω=({omega[0]:.2f},{omega[1]:.2f},{omega[2]:.2f})',
            fontsize=7)

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=9)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig('exp2_map_snapshots.png', dpi=150)
    plt.show()


# ===========================================================================
# EXPERIMENT 3: Number of Iterations vs. Quality
# ===========================================================================

def experiment_iterations_sweep(frame_idx=0,
                                iter_counts=None):
    """
    For a single frame, run with different numbers of iterations.
    Shows: more iterations → better convergence (diminishing returns?).
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Number of Iterations Sweep")
    print("="*70)

    if iter_counts is None:
        iter_counts = [5, 10, 20, 50, 75, 100, 150, 200, 300]

    V, t_mid, calib, H, W = load_single_frame(frame_idx)
    fx, fy, cx, cy = calib.fx, calib.fy, calib.cx, calib.cy

    imu_data = load_imu(IMU_FILE)
    gt_omega = get_gt_omega(imu_data, T_START, FRAME_DURATION, frame_idx)
    gt_R = gt_omega * FRAME_DURATION

    results = []

    for n_iters in iter_counts:
        net = make_network(H, W, fx, fy, cx, cy, gt_R)
        net.step(V, n_iters=n_iters)

        r_err = np.linalg.norm(net.R - gt_R)
        
        omega_est = net.R / FRAME_DURATION
        omega_err = np.linalg.norm(omega_est - gt_omega) * 180 / np.pi
        res_vfg = net.residual_VFG(V)
        res_gi = net.residual_GI()

        # Direction error (meaningful for both versions)
        norm_est = np.linalg.norm(net.R)
        norm_gt = np.linalg.norm(gt_R)
        if norm_est > 1e-8 and norm_gt > 1e-8:
            cos_a = np.clip(np.dot(net.R, gt_R) / (norm_est * norm_gt), -1, 1)
            dir_err = np.degrees(np.arccos(cos_a))
            beta = norm_gt / norm_est
        else:
            dir_err = 180.0
            beta = 0.0

        results.append({
            'n_iters': n_iters,
            'R_error': r_err,
            'omega_error_deg': omega_err,
            'dir_error_deg': dir_err,
            'beta': beta,
            'res_VFG': res_vfg,
            'res_GI': res_gi,
            'R_est': net.R.copy(),
        })

        if USE_THESIS_VERSION:
            print(f"  iters={n_iters:4d} | R_err={r_err:.6f} | "
                  f"ω_err={omega_err:.2f}° | VFG={res_vfg:.5f} | GI={res_gi:.5f}")
        else:
            print(f"  iters={n_iters:4d} | dir_err={dir_err:.1f}° | β={beta:.2f} | "
                  f"VFG={res_vfg:.5f} | GI={res_gi:.5f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Experiment 3: Effect of Number of Iterations (Frame {frame_idx})',
                 fontsize=12)

    iters = [r['n_iters'] for r in results]

    ax = axes[0]
    ax.semilogy(iters, [r['R_error'] for r in results], 'bo-')
    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('||R_est - R_gt|| (rad/frame)')
    ax.set_title('Angular Velocity Error')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(iters, [r['res_VFG'] for r in results], 'b^-', label='OFCE')
    ax.semilogy(iters, [r['res_GI'] for r in results], 'rs-', label='Spatial')
    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Mean Residual')
    ax.set_title('Constraint Residuals')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(iters, [r['omega_error_deg'] for r in results], 'go-')
    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Error (°/s)')
    ax.set_title('Angular Velocity Error (degrees/s)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exp3_iterations_sweep.png', dpi=150)
    plt.show()


# ===========================================================================
# EXPERIMENT 4: Number of Events (Frame Duration) Sweep
# ===========================================================================

def experiment_events_sweep(frame_idx=0, n_iters=None,
                            durations=None):
    """
    Vary the frame duration (= number of events per frame).
    More events → stronger V signal → possibly better convergence.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: Frame Duration (Number of Events) Sweep")
    print("="*70)
    if n_iters is None:
        n_iters = ITERS_PER_FRAME
    if durations is None:
        durations = [0.005, 0.010, 0.015, 0.020, 0.030, 0.050, 0.075, 0.100]

    imu_data = load_imu(IMU_FILE)

    results = []

    for dt in durations:
        # Load events for this duration
        seq = EventFrameSequence(
            EVENTS_FILE, CALIB_FILE,
            frame_duration=dt,
            t_start=T_START,
            n_frames=frame_idx + 1,
            clip_value=10.0,
            sensor_size=cfg.get('sensor_size', None),  

        )
        frames = list(seq)
        V, t_mid = frames[frame_idx]
        H, W = seq.H, seq.W
        fx, fy, cx, cy = seq.calib.fx, seq.calib.fy, seq.calib.cx, seq.calib.cy

        # GT for this frame duration
        t_lo = T_START + frame_idx * dt
        t_hi = t_lo + dt
        gt_omega = get_gyro_for_frame(imu_data, t_lo, t_hi)
        gt_R = gt_omega * dt

        # Count non-zero pixels in V
        n_events_pixels = np.count_nonzero(V)
        v_energy = np.sum(np.abs(V))
        sparsity = (1.0 - n_events_pixels / (H * W)) * 100

        # Run network
        net = make_network(H, W, fx, fy, cx, cy, gt_R)
        if USE_THESIS_VERSION and USE_IMU:
            net.step(V, n_iters=n_iters, omega_imu=gt_omega)
        else:
            net.step(V, n_iters=n_iters)


        omega_est = net.R / dt
        omega_err = np.linalg.norm(omega_est - gt_omega) * 180 / np.pi

        # Direction error (meaningful for Cook)
        norm_est = np.linalg.norm(omega_est)
        norm_gt = np.linalg.norm(gt_omega)
        if norm_est > 1e-8 and norm_gt > 1e-8:
            cos_a = np.clip(np.dot(omega_est, gt_omega) / (norm_est * norm_gt), -1, 1)
            dir_err = np.degrees(np.arccos(cos_a))
            beta = norm_gt / norm_est
        else:
            dir_err = 180.0
            beta = 0.0

        if USE_THESIS_VERSION:
            print(f"  dt={dt*1000:6.1f}ms | active_px={n_events_pixels:6d} | "
                  f"sparsity={sparsity:.0f}% | ω_err={omega_err:.2f}°/s")
        else:
            print(f"  dt={dt*1000:6.1f}ms | active_px={n_events_pixels:6d} | "
                  f"sparsity={sparsity:.0f}% | dir_err={dir_err:.1f}° | β={beta:.1f}")
            

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Experiment 4: Effect of Frame Duration (Event Count)', fontsize=12)

    dts_ms = [r['duration'] * 1000 for r in results]

    ax = axes[0]
    ax.plot(dts_ms, [r['omega_error_deg'] for r in results], 'bo-')
    ax.set_xlabel('Frame Duration (ms)')
    ax.set_ylabel('ω Error (°/s)')
    ax.set_title('Angular Velocity Error vs. Frame Duration')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(dts_ms, [r['n_active_pixels'] for r in results], 'rs-')
    ax.set_xlabel('Frame Duration (ms)')
    ax.set_ylabel('Active Pixels')
    ax.set_title('Event Density vs. Frame Duration')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot([r['n_active_pixels'] for r in results],
            [r['omega_error_deg'] for r in results], 'g^-')
    ax.set_xlabel('Active Pixels (event count proxy)')
    ax.set_ylabel('ω Error (°/s)')
    ax.set_title('Error vs. Event Count')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exp4_events_sweep.png', dpi=150)
    plt.show()


# ===========================================================================
# EXPERIMENT 5: Basin of Attraction (Start Near vs. Far from GT)
# ===========================================================================

def experiment_basin_of_attraction(frame_idx=0, n_iters=None):
    """
    Initialize R at various distances from the GT angular velocity.
    Check: does the optimization converge back to the correct minimum?
    
    This tests the professor's suggestion:
    "start near → then try far → does it go back to minimum?"
    """
    print("\n" + "="*70)
    print("EXPERIMENT 5: Basin of Attraction")
    print("="*70)

    if n_iters is None:
        n_iters = ITERS_PER_FRAME

    V, t_mid, calib, H, W = load_single_frame(frame_idx)
    fx, fy, cx, cy = calib.fx, calib.fy, calib.cx, calib.cy

    imu_data = load_imu(IMU_FILE)
    gt_omega = get_gt_omega(imu_data, T_START, FRAME_DURATION, frame_idx)
    gt_R = gt_omega * FRAME_DURATION

    gt_R_norm = np.linalg.norm(gt_R)
    print(f"GT R = {gt_R}, |GT R| = {gt_R_norm:.6f} rad/frame")
    print(f"GT ω = {gt_omega} rad/s")

    # Perturbation magnitudes (as fraction of |gt_R|, and absolute)
    # We test: exact GT, small perturbation, medium, large, very large, opposite
    scale = max(gt_R_norm, 0.01)  # avoid division by zero
    perturbation_configs = [
        ("Exact GT", gt_R.copy()),
        ("GT + 10% noise", gt_R + 0.1 * scale * np.random.randn(3)),
        ("GT + 50% noise", gt_R + 0.5 * scale * np.random.randn(3)),
        ("GT + 100% noise", gt_R + 1.0 * scale * np.random.randn(3)),
        ("GT + 200% noise", gt_R + 2.0 * scale * np.random.randn(3)),
        ("Opposite direction", -gt_R),
        ("Zero init", np.zeros(3)),
        ("Random large", np.random.randn(3) * 0.05),
    ]

    results = []

    for label, R_init in perturbation_configs:
        init_dist = np.linalg.norm(R_init - gt_R)

        net = make_network(H, W, fx, fy, cx, cy, R_init)

        # Track convergence
        R_trajectory = [R_init.copy()]

        if USE_THESIS_VERSION:
            net.q_V.value = V
            for it in range(n_iters):
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
                R_trajectory.append(net.R.copy())
        else:
            for it in range(n_iters):
                net.update_F_from_VG(V)
                net.update_G_from_VF(V)
                net.update_G_from_I()
                net.update_I_from_G()
                net.update_F_from_RC()
                net.update_R_from_FC()
                R_trajectory.append(net.R.copy())

        R_trajectory = np.array(R_trajectory)
        final_R = R_trajectory[-1]
        final_dist = np.linalg.norm(final_R - gt_R)
        converged = final_dist < 0.5 * scale  # within 50% of GT magnitude

        results.append({
            'label': label,
            'R_init': R_init.copy(),
            'init_dist': init_dist,
            'final_dist': final_dist,
            'converged': converged,
            'trajectory': R_trajectory,
        })

        print(f"  {label:20s} | init_dist={init_dist:.5f} | "
              f"final_dist={final_dist:.5f} | "
              f"{'✓ CONVERGED' if converged else '✗ DID NOT CONVERGE'}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Experiment 5: Basin of Attraction (Frame {frame_idx})\n'
                 f'GT ω = ({gt_omega[0]:.3f}, {gt_omega[1]:.3f}, {gt_omega[2]:.3f}) rad/s',
                 fontsize=11)

    # (a) Distance to GT over iterations for each initialization
    ax = axes[0, 0]
    for r in results:
        traj = r['trajectory']
        dist_curve = np.linalg.norm(traj - gt_R, axis=1)
        style = '-' if r['converged'] else '--'
        ax.semilogy(dist_curve, style, label=r['label'])
    ax.set_xlabel('Iteration')
    ax.set_ylabel('||R - R_gt||')
    ax.set_title('(a) Distance to GT vs. Iteration')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (b) Rx component trajectories
    ax = axes[0, 1]
    for r in results:
        ax.plot(r['trajectory'][:, 0], alpha=0.7, label=r['label'])
    ax.axhline(gt_R[0], color='k', ls='--', lw=2, label='GT Rx')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Rx (rad/frame)')
    ax.set_title('(b) Rx Trajectory')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (c) Initial distance vs. final distance (scatter)
    ax = axes[1, 0]
    for r in results:
        color = 'green' if r['converged'] else 'red'
        ax.scatter(r['init_dist'], r['final_dist'], c=color, s=100, zorder=5)
        ax.annotate(r['label'], (r['init_dist'], r['final_dist']),
                    fontsize=7, ha='left')
    ax.plot([0, max(r['init_dist'] for r in results)],
            [0, max(r['init_dist'] for r in results)], 'k--', alpha=0.3,
            label='no improvement')
    ax.set_xlabel('Initial ||R_init - R_gt||')
    ax.set_ylabel('Final ||R_final - R_gt||')
    ax.set_title('(c) Basin of Attraction Summary')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (d) Ry and Rz
    ax = axes[1, 1]
    for r in results:
        ax.plot(r['trajectory'][:, 1], alpha=0.5, ls='-')
        ax.plot(r['trajectory'][:, 2], alpha=0.5, ls='--')
    ax.axhline(gt_R[1], color='k', ls='-', lw=2, label='GT Ry')
    ax.axhline(gt_R[2], color='k', ls='--', lw=2, label='GT Rz')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('rad/frame')
    ax.set_title('(d) Ry (solid) and Rz (dashed) Trajectories')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exp5_basin_of_attraction.png', dpi=150)
    plt.show()


# ===========================================================================
# EXPERIMENT 6: Multi-Frame Tracking with GT Comparison
# ===========================================================================

def experiment_multiframe_tracking(n_frames=N_FRAMES, n_iters=None):
    """
    Run the network over multiple frames, comparing the estimated ω
    to GT at each frame. This is what's already partly in validation.py
    but enhanced with iteration-level detail.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 6: Multi-Frame Tracking vs. Ground Truth")
    print("="*70)
    print(f"initial_R from config = {initial_R}")
    print(f"|initial_R| = {np.linalg.norm(initial_R)}")
    #print(f"GT R for frame 0 = {gt_R}")
    #print(f"Distance = {np.linalg.norm(initial_R - gt_R)}")
    if n_iters is None:
        n_iters = ITERS_PER_FRAME

    seq = EventFrameSequence(
        EVENTS_FILE, CALIB_FILE,
        frame_duration=FRAME_DURATION,
        t_start=T_START,
        n_frames=n_frames,
        clip_value=10.0,
        sensor_size=cfg.get('sensor_size', None),
    )
    H, W = seq.H, seq.W
    fx, fy, cx, cy = seq.calib.fx, seq.calib.fy, seq.calib.cx, seq.calib.cy

    imu_data = load_imu(IMU_FILE)

    net = make_network(H, W, fx, fy, cx, cy, initial_R)

    omega_est_all = []
    omega_gt_all = []
    times = []

    for k, (V, t_mid) in enumerate(seq):
        # Get IMU reading for this frame
        t_lo = T_START + k * FRAME_DURATION
        t_hi = t_lo + FRAME_DURATION
        #omega_imu = get_gyro_for_frame(imu_data, t_lo, t_hi)
        
        # Pass IMU to network
        if USE_THESIS_VERSION and USE_IMU:
            net.step(V, n_iters=n_iters, omega_imu=gt_omega)
        else:
            net.step(V, n_iters=n_iters)

        omega_est = net.R / FRAME_DURATION
        gt_omega = get_gyro_for_frame(imu_data, t_lo, t_hi)

        omega_est_all.append(omega_est.copy())
        omega_gt_all.append(gt_omega.copy())
        times.append(t_mid)

        err = np.linalg.norm(omega_est - gt_omega) * 180 / np.pi
        if (k + 1) % 5 == 0 or k == 0:
            print(f"  Frame {k+1:3d}/{n_frames} | "
                  f"est=({omega_est[0]:+.3f},{omega_est[1]:+.3f},{omega_est[2]:+.3f}) | "
                  f"gt=({gt_omega[0]:+.3f},{gt_omega[1]:+.3f},{gt_omega[2]:+.3f}) | "
                  f"err={err:.1f}°/s")

    omega_est_all = np.array(omega_est_all)
    omega_gt_all = np.array(omega_gt_all)
    times = np.array(times)

    # --- Plot ---
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    fig.suptitle(f'Experiment 6: Multi-Frame Angular Velocity Tracking\n'
                 f'{"Thesis" if USE_THESIS_VERSION else "Cook"}, '
                 f'{FRAME_DURATION*1000:.0f}ms frames, {n_iters} iters/frame',
                 fontsize=11)

    labels = ['ωx', 'ωy', 'ωz']
    colors = ['#4e79a7', '#f28e2b', '#e15759']

    for i in range(3):
        ax = axes[i]
        ax.plot(times, omega_gt_all[:, i], 'k-', lw=1.5, label=f'{labels[i]} GT (IMU)')
        ax.plot(times, omega_est_all[:, i], color=colors[i], lw=2, label=f'{labels[i]} Estimated')
        ax.set_ylabel(f'{labels[i]} (rad/s)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    # Error
    ax = axes[3]
    err_deg = np.linalg.norm(omega_est_all - omega_gt_all, axis=1) * 180 / np.pi
    ax.plot(times, err_deg, 'r-', lw=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error (°/s)')
    ax.set_title(f'Total Angular Error (mean={np.mean(err_deg):.1f}°/s, '
                 f'median={np.median(err_deg):.1f}°/s)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exp6_multiframe_tracking.png', dpi=150)
    plt.show()

def experiment_multiframe_tracking2(n_frames=N_FRAMES, n_iters=None):
    """Multi-frame tracking — adapts based on USE_THESIS_VERSION."""
    print("\n" + "="*70)
    print("EXPERIMENT 6: Multi-Frame Tracking vs. Ground Truth")
    print("="*70)
    print(f"initial_R from config = {initial_R}")
    print(f"|initial_R| = {np.linalg.norm(initial_R)}")

    if n_iters is None:
        n_iters = ITERS_PER_FRAME

    seq = EventFrameSequence(
        EVENTS_FILE, CALIB_FILE,
        frame_duration=FRAME_DURATION, t_start=T_START,
        n_frames=n_frames, clip_value=10.0,sensor_size=cfg.get('sensor_size', None),
    )
    H, W = seq.H, seq.W
    fx, fy, cx, cy = seq.calib.fx, seq.calib.fy, seq.calib.cx, seq.calib.cy
    imu_data = load_imu(IMU_FILE)

    net = make_network(H, W, fx, fy, cx, cy, initial_R)

    omega_est_all = []
    omega_gt_all = []

    for k, (V, t_mid) in enumerate(seq):
        t_lo = T_START + k * FRAME_DURATION
        t_hi = t_lo + FRAME_DURATION
        gt_omega = get_gyro_for_frame(imu_data, t_lo, t_hi)

        if USE_THESIS_VERSION and USE_IMU:
            # Thesis: USE IMU to fix scale (§6.8.3)
            omega_imu = get_gyro_for_frame(imu_data, t_lo, t_hi)
            net.step(V, n_iters=n_iters, omega_imu=omega_imu)
        else:
            # Cook: NO IMU — β-ambiguity is accepted
            net.step(V, n_iters=n_iters)

        omega_est = net.R / FRAME_DURATION
        omega_est_all.append(omega_est.copy())
        omega_gt_all.append(gt_omega.copy())

        if (k + 1) % 5 == 0 or k == 0:
            if USE_THESIS_VERSION:
                err = np.linalg.norm(omega_est - gt_omega) * 180 / np.pi
                print(f"  Frame {k+1:3d}/{n_frames} | err={err:.1f}°/s")
            else:
                # Direction error only for Cook
                norm_est = np.linalg.norm(omega_est)
                norm_gt = np.linalg.norm(gt_omega)
                if norm_est > 1e-6 and norm_gt > 1e-6:
                    cos_a = np.clip(np.dot(omega_est, gt_omega) / (norm_est * norm_gt), -1, 1)
                    dir_err = np.degrees(np.arccos(cos_a))
                    beta = norm_gt / norm_est
                else:
                    dir_err = 180.0
                    beta = 0.0
                print(f"  Frame {k+1:3d}/{n_frames} | dir_err={dir_err:.1f}° | β={beta:.2f}")

    omega_est_all = np.array(omega_est_all)
    omega_gt_all = np.array(omega_gt_all)

    # Summary
    if USE_THESIS_VERSION:
        err_all = np.linalg.norm(omega_est_all - omega_gt_all, axis=1) * 180 / np.pi
        print(f"\n  WITH IMU: mean_err={np.mean(err_all):.1f}°/s")
    else:
        # Report direction error + β
        norms_est = np.linalg.norm(omega_est_all, axis=1)
        norms_gt = np.linalg.norm(omega_gt_all, axis=1)
        valid = (norms_est > 1e-6) & (norms_gt > 1e-6)
        cos_angles = np.sum(omega_est_all[valid] * omega_gt_all[valid], axis=1) / \
                     (norms_est[valid] * norms_gt[valid])
        cos_angles = np.clip(cos_angles, -1, 1)
        dir_errors = np.degrees(np.arccos(cos_angles))
        betas = norms_gt[valid] / norms_est[valid]
        print(f"\n  NO IMU (Cook): direction_err={np.mean(dir_errors):.1f}°, "
              f"β={np.mean(betas):.2f}±{np.std(betas):.2f}")

# ===========================================================================
# EXPERIMENT 7: Iterations × Frame Duration (2D Sweep)
# ===========================================================================

def experiment_iters_x_duration(frame_idx=0):
    """
    2D parameter sweep: iterations AND frame duration simultaneously.
    Shows the trade-off the professor asked about.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 7: Iterations × Frame Duration (2D Sweep)")
    print("="*70)

    iter_counts = [10, 25, 50, 75, 100, 150, 200]
    durations = [0.005, 0.010, 0.020, 0.030, 0.050]

    imu_data = load_imu(IMU_FILE)

    error_matrix = np.zeros((len(durations), len(iter_counts)))

    for di, dt in enumerate(durations):
        seq = EventFrameSequence(
            EVENTS_FILE, CALIB_FILE,
            frame_duration=dt,
            t_start=T_START,
            n_frames=frame_idx + 1,
            clip_value=10.0,
            sensor_size=cfg.get('sensor_size', None),
        )
        frames = list(seq)
        V, t_mid = frames[frame_idx]
        H, W = seq.H, seq.W
        fx, fy, cx, cy = seq.calib.fx, seq.calib.fy, seq.calib.cx, seq.calib.cy

        t_lo = T_START + frame_idx * dt
        t_hi = t_lo + dt
        gt_omega = get_gyro_for_frame(imu_data, t_lo, t_hi)
        gt_R = gt_omega * dt

        for ii, n_iters in enumerate(iter_counts):
            net = make_network(H, W, fx, fy, cx, cy, gt_R)
            if USE_THESIS_VERSION and USE_IMU:
                net.step(V, n_iters=n_iters, omega_imu=gt_omega)
            else:
                net.step(V, n_iters=n_iters)
            omega_err = np.linalg.norm(net.R / dt - gt_omega) * 180 / np.pi
            error_matrix[di, ii] = omega_err

        print(f"  dt={dt*1000:5.1f}ms done")

    # --- Plot heatmap ---
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(error_matrix, aspect='auto', cmap='viridis_r',
                   origin='lower')
    ax.set_xticks(range(len(iter_counts)))
    ax.set_xticklabels(iter_counts)
    ax.set_yticks(range(len(durations)))
    ax.set_yticklabels([f'{d*1000:.0f}ms' for d in durations])
    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Frame Duration')
    ax.set_title('Experiment 7: Angular Velocity Error (°/s)\nIterations × Frame Duration')
    plt.colorbar(im, ax=ax, label='Error (°/s)')

    # Annotate cells
    for di in range(len(durations)):
        for ii in range(len(iter_counts)):
            ax.text(ii, di, f'{error_matrix[di, ii]:.1f}',
                    ha='center', va='center', fontsize=8,
                    color='white' if error_matrix[di, ii] > error_matrix.mean() else 'black')

    plt.tight_layout()
    plt.savefig('exp7_iters_x_duration.png', dpi=150)
    plt.show()

# ===========================================================================
# EXPERIMENT 8: Scale-Invariant (Direction-Only) Validation
# ===========================================================================

def experiment_scale_invariant(n_frames=N_FRAMES, n_iters=None):
    """
    Validate that the DIRECTION of ω is correct, ignoring magnitude.
    Per Cook et al.: "impossible to distinguish slow-moving high-contrast
    from fast-moving low-contrast" — direction should still be correct.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 8: Scale-Invariant Direction Validation")
    print("="*70)

    if n_iters is None:
        n_iters = ITERS_PER_FRAME

    seq = EventFrameSequence(
        EVENTS_FILE, CALIB_FILE,
        frame_duration=FRAME_DURATION, t_start=T_START,
        n_frames=n_frames, clip_value=10.0,sensor_size=cfg.get('sensor_size', None),
    )
    H, W = seq.H, seq.W
    fx, fy, cx, cy = seq.calib.fx, seq.calib.fy, seq.calib.cx, seq.calib.cy
    imu_data = load_imu(IMU_FILE)

    net = make_network(H, W, fx, fy, cx, cy, initial_R)

    dir_errors = []
    scale_factors = []
    omega_est_all = []
    omega_gt_all = []

    for k, (V, t_mid) in enumerate(seq):
        t_lo = T_START + k * FRAME_DURATION
        t_hi = t_lo + FRAME_DURATION
        gt_omega = get_gyro_for_frame(imu_data, t_lo, t_hi)
        if USE_THESIS_VERSION and USE_IMU:
            net.step(V, n_iters=n_iters, omega_imu=gt_omega)
        else:
            net.step(V, n_iters=n_iters)

        omega_est = net.R / FRAME_DURATION


        # Direction error (scale-invariant)
        norm_est = np.linalg.norm(omega_est)
        norm_gt = np.linalg.norm(gt_omega)

        if norm_est > 1e-6 and norm_gt > 1e-6:
            cos_angle = np.clip(
                np.dot(omega_est, gt_omega) / (norm_est * norm_gt), -1.0, 1.0
            )
            dir_err = np.degrees(np.arccos(cos_angle))
            beta = norm_gt / norm_est
        else:
            dir_err = 180.0
            beta = 1.0

        dir_errors.append(dir_err)
        scale_factors.append(beta)
        omega_est_all.append(omega_est.copy())
        omega_gt_all.append(gt_omega.copy())

        if (k + 1) % 5 == 0 or k == 0:
            print(f"  Frame {k+1:3d}/{n_frames} | "
                  f"dir_err={dir_err:5.1f}° | β={beta:.2f} | "
                  f"|est|={norm_est:.3f} | |gt|={norm_gt:.3f}")

    dir_errors = np.array(dir_errors)
    scale_factors = np.array(scale_factors)

    print(f"\n  Summary:")
    print(f"    Direction error: mean={np.mean(dir_errors):.1f}°, "
          f"median={np.median(dir_errors):.1f}°")
    print(f"    Scale factor β:  mean={np.mean(scale_factors):.2f}, "
          f"std={np.std(scale_factors):.2f}")
    print(f"    (β=1.0 → perfect scale; β>1 → network underestimates magnitude)")

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Experiment 8: Scale-Invariant Validation\n'
                 '(Cook et al.: "up to a factor β")', fontsize=11)

    ax = axes[0, 0]
    ax.plot(dir_errors, 'b-', lw=1.5)
    ax.set_ylabel('Direction Error (°)')
    ax.set_xlabel('Frame')
    ax.set_title(f'Direction Error (median={np.median(dir_errors):.1f}°)')
    ax.axhline(10, color='g', ls='--', alpha=0.5, label='10° threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(scale_factors, 'r-', lw=1.5)
    ax.axhline(1.0, color='k', ls='--', lw=2)
    ax.set_ylabel('β = |ω_gt| / |ω_est|')
    ax.set_xlabel('Frame')
    ax.set_title(f'Scale Factor β (mean={np.mean(scale_factors):.2f})')
    ax.grid(True, alpha=0.3)

    # Scaled estimate vs GT
    ax = axes[1, 0]
    omega_est_all = np.array(omega_est_all)
    omega_gt_all = np.array(omega_gt_all)
    # Apply mean β correction
    mean_beta = np.mean(scale_factors)
    omega_corrected = omega_est_all * mean_beta
    ax.plot(omega_gt_all[:, 0], 'k-', lw=1.5, label='GT ωx')
    ax.plot(omega_corrected[:, 0], 'b--', lw=1.5, label=f'Est ωx × β={mean_beta:.2f}')
    ax.set_xlabel('Frame')
    ax.set_ylabel('rad/s')
    ax.set_title('β-Corrected Estimate vs GT (dominant axis)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Magnitude comparison
    ax = axes[1, 1]
    ax.plot(np.linalg.norm(omega_gt_all, axis=1), 'k-', lw=1.5, label='|ω_gt|')
    ax.plot(np.linalg.norm(omega_est_all, axis=1), 'b-', lw=1.5, label='|ω_est|')
    ax.plot(np.linalg.norm(omega_corrected, axis=1), 'b--', lw=1.0, label='|ω_est| × β')
    ax.set_xlabel('Frame')
    ax.set_ylabel('rad/s')
    ax.set_title('Magnitude Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exp8_scale_invariant.png', dpi=150)
    plt.show()

# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Validation Experiments')
    parser.add_argument('--exp', type=int, default=0,
                        help='Experiment number (1-7, 0=all)')
    parser.add_argument('--frame', type=int, default=0,
                        help='Frame index to use')
    args = parser.parse_args()

    
    experiments = {
        #1: experiment_convergence_within_frame,
        2: experiment_map_snapshots,
        #3: experiment_iterations_sweep,
        #4: experiment_events_sweep,
        5: experiment_basin_of_attraction,
        6: experiment_multiframe_tracking2,
        #7: experiment_iters_x_duration,
        8: experiment_scale_invariant,
    }

    if args.exp == 0:
        print("Running ALL experiments...")
        for num, func in experiments.items():
            try:
                if num in [1, 2, 3, 4, 5, 7]:
                    func(frame_idx=args.frame)
                else:
                    func()
            except Exception as e:
                print(f"  Experiment {num} failed: {e}")
    else:
        func = experiments[args.exp]
        if args.exp in [1, 2, 3, 4, 5, 7]:
            func(frame_idx=args.frame)
        else:
            func()