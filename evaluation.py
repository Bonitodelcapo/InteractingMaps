"""
evaluation.py — Systematic evaluation of Interacting Maps.

Experiments:
  1. Single-frame convergence (visual: maps at iteration checkpoints)
  2. Multi-frame tracking (quantitative: ω_est vs ω_GT over time)
  3. Parameter sweeps (iterations & frame duration)
  4. Qualitative video (Events | Estimated I | GT APS)
  5. Assemble video from frames
  6. Basin of attraction (how far can init be from GT?)
  7. Full evaluation (all datasets × all models × multiple segments)

Usage:
    python evaluation.py --exp 8 --all-segments     # for all segments sweep over
                                                    #  'n_frames':  [25, 50, 150],
                                                    #  'n_iters':   [75, 100],
                                                    #  'delta_FR':  [0.10, 0.20, 0.30, 0.50],
                                                    #  'delta_IMU': [0.10, 0.20, 0.30, 0.50],

    # Alles mit Bildern (Standard):
    python evaluation.py --exp 7 --model all

    # Schneller ohne Bilder (nur Metriken):
    python evaluation.py --exp 7 --model all --no-frames

    # Parameter-Grid, alle Segmente, alle Models, mit Bildern:
    python evaluation.py --exp 8 --all-segments --model all

    # Einzelner Run:
    python evaluation.py --exp 2 --dataset poster_rotation --model thesis_imu
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import json
import time as time_module

from config import (DATASET_CONFIGS, DATASET_SEGMENTS, THESIS_PARAMS, COOK_PARAMS,
                    ITERS_PER_FRAME, get_dataset_paths, get_initial_R_from_imu)
from data_loader import EventFrameSequence
from interacting_maps.network import InteractingMaps
from interacting_maps.network_dissertation import InteractingMapsThesis


# ===========================================================================
# RUN CONFIGURATION (override via CLI args)
# ===========================================================================

class RunConfig:
    """All parameters for a single evaluation run."""
    def __init__(self, dataset='boxes_rotation', model='thesis_imu',
                 segment=None, t_start=None, frame_duration=None, n_frames=None,
                 n_iters=None, delta_IMU=None):
        
        self.dataset = dataset
        self.model = model  # 'cook', 'thesis', 'thesis_imu'
        
        if segment is None:
            # Fallback: erstes Segment bzw. DATASET_CONFIGS
            self.segment = DATASET_CONFIGS[dataset]
            self.segment_id = 'default'
        elif isinstance(segment, str):
            # Lookup by ID (z.B. 'seg_A')
            segs = DATASET_SEGMENTS[dataset]
            match = [s for s in segs if s['id'] == segment]
            if not match:
                raise ValueError(f"Segment '{segment}' not found for {dataset}")
            self.segment = match[0]
            self.segment_id = segment
        else:
            # Direkt ein Dict übergeben
            self.segment = segment
            self.segment_id = segment.get('id', 'unknown')

        # Load dataset defaults
        cfg = self.segment
        self.t_start = t_start if t_start is not None else cfg['t_start']
        self.frame_duration = frame_duration if frame_duration is not None else cfg['frame_duration']
        self.n_frames = n_frames if n_frames is not None else cfg['n_frames']
        self.n_iters = n_iters if n_iters is not None else ITERS_PER_FRAME
        self.sensor_size = cfg.get('sensor_size', (180, 240))
        
        # Model parameters
        if model == 'cook':
            self.params = COOK_PARAMS.copy()
            self.use_thesis = False
            self.use_imu = False
        elif model == 'thesis':
            self.params = THESIS_PARAMS.copy()
            self.use_thesis = True
            self.use_imu = False
        else:  # thesis_imu
            self.params = THESIS_PARAMS.copy()
            self.use_thesis = True
            self.use_imu = True
        
        # Override delta_IMU if specified
        if delta_IMU is not None and self.use_imu:
            self.params['delta_IMU'] = delta_IMU
        
        # Paths
        self.paths = get_dataset_paths(dataset)
        
        # Initial R from IMU
        self.initial_R = self._compute_initial_R()
    
    @property
    def delta_IMU(self):
        return self.params.get('delta_IMU', 0.0)
    
    @property
    def duration_s(self):
        return self.n_frames * self.frame_duration
    
    @property
    def output_dir(self):
        """Folder now includes dt."""
        folder_name = (f"{self.segment_id}_t{self.t_start:.3f}"
                    f"_dt{self.frame_duration*1000:.0f}ms"
                    f"_n{self.n_frames}_i{self.n_iters}")
        if self.use_imu:
            folder_name += f"_dimu{self.delta_IMU:.2f}"
        # Also encode delta_FR to distinguish configs
        folder_name += f"_dFR{self.params.get('delta_FR', 0):.2f}"
        return os.path.join('results', self.dataset, self.model, folder_name)
    
    def to_dict(self):
        """Serialize all parameters for JSON."""
        return {
            'dataset': self.dataset,
            'model': self.model,
            't_start': self.t_start,
            'frame_duration': self.frame_duration,
            'n_frames': self.n_frames,
            'n_iters': self.n_iters,
            'duration_s': self.duration_s,
            'sensor_size': list(self.sensor_size),
            'initial_R': self.initial_R.tolist(),
            'params': self.params,
        }
    
    def save_params(self):
        """Save params.json to output directory."""
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, 'params.json'), 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def __repr__(self):
        return (f"RunConfig({self.dataset}, {self.model}, "
                f"t={self.t_start:.3f}, dt={self.frame_duration*1000:.0f}ms, "
                f"n={self.n_frames}, iters={self.n_iters})")

    def _compute_initial_R(self):
        """Compute initial R from IMU at THIS segment's t_start."""
        imu_data = np.loadtxt(self.paths['imu'], dtype=np.float64)
        t_lo = self.t_start
        t_hi = self.t_start + self.frame_duration
        mask = (imu_data[:, 0] >= t_lo) & (imu_data[:, 0] < t_hi)
        
        if np.sum(mask) > 0:
            omega = np.mean(imu_data[mask, 4:7], axis=0)
        else:
            idx = np.argmin(np.abs(imu_data[:, 0] - self.t_start))
            omega = imu_data[idx, 4:7]
        
        return omega * self.frame_duration  # rad/s → rad/frame


# ===========================================================================
# HELPERS
# ===========================================================================

def load_imu(path: str) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float64)

def get_gyro_for_frame(imu_data, t_lo, t_hi):
    mask = (imu_data[:, 0] >= t_lo) & (imu_data[:, 0] < t_hi)
    if np.sum(mask) == 0:
        idx = np.argmin(np.abs(imu_data[:, 0] - (t_lo + t_hi) / 2))
        return imu_data[idx, 4:7]
    return np.mean(imu_data[mask, 4:7], axis=0)

def make_network(rc: RunConfig, H, W, fx, fy, cx, cy):
    """Create network from RunConfig."""
    if rc.use_thesis:
        net = InteractingMapsThesis(
            H=H, W=W, fx=fx, fy=fy, cx=cx, cy=cy,
            frame_duration=rc.frame_duration, **rc.params
        )
        net.initialize_from_rotation(rc.initial_R)
    else:
        net = InteractingMaps(H=H, W=W, fx=fx, fy=fy, cx=cx, cy=cy, **rc.params)
        net.I = np.random.randn(H+1, W+1) * 0.001
        net.G = np.zeros((H, W, 2), dtype=np.float64)
        net.F = np.einsum('hwij,j->hwi', net._C_mat, rc.initial_R)
        net.R = rc.initial_R.copy()
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

def compute_metrics(omega_est, omega_gt):
    """Compute all metrics for a single frame."""
    err = np.linalg.norm(omega_est - omega_gt) * 180 / np.pi
    norm_est = np.linalg.norm(omega_est)
    norm_gt = np.linalg.norm(omega_gt)
    if norm_est > 1e-6 and norm_gt > 1e-6:
        cos_a = np.clip(np.dot(omega_est, omega_gt) / (norm_est * norm_gt), -1, 1)
        dir_err = np.degrees(np.arccos(cos_a))
        beta = norm_gt / norm_est
    else:
        dir_err = 180.0
        beta = 0.0
    return err, dir_err, beta


# ===========================================================================
# EXPERIMENT 1: Single-Frame Map Convergence
# ===========================================================================

def experiment_single_frame_convergence(rc: RunConfig, frame_idx=0, max_iters=100,
                                         snapshot_iters=None):
    """Maps at iteration checkpoints — visual inspection."""
    print("\n" + "="*70)
    print("EXPERIMENT 1: Single-Frame Map Convergence")
    print(f"  Config: {rc}")
    print("="*70)

    if snapshot_iters is None:
        snapshot_iters = [1, 3, 5, 10, 25, 50, max_iters]
    snapshot_iters = [i for i in snapshot_iters if i <= max_iters]

    seq = EventFrameSequence(
        rc.paths['events'], rc.paths['calib'],
        frame_duration=rc.frame_duration, t_start=rc.t_start,
        n_frames=frame_idx + 1, clip_value=10.0,
        sensor_size=rc.sensor_size,
    )
    frames = list(seq)
    V, t_mid = frames[frame_idx]
    H, W = seq.H, seq.W
    fx, fy, cx, cy = seq.calib.fx, seq.calib.fy, seq.calib.cx, seq.calib.cy

    imu_data = load_imu(rc.paths['imu'])
    t_lo = rc.t_start + frame_idx * rc.frame_duration
    t_hi = t_lo + rc.frame_duration
    gt_omega = get_gyro_for_frame(imu_data, t_lo, t_hi)
    gt_R = gt_omega * rc.frame_duration

    print(f"  GT ω = ({gt_omega[0]:.3f}, {gt_omega[1]:.3f}, {gt_omega[2]:.3f}) rad/s")

    net = make_network(rc, H, W, fx, fy, cx, cy)
    snapshots = {}
    R_history = []
    res_history = []

    if rc.use_thesis:
        net.q_V.value = V
        for it in range(1, max_iters + 1):
            for q in [net.q_I, net.q_G, net.q_F, net.q_R]:
                q.reset_gradient()
            for cost in net.costs:
                if cost is net.cost_imu and not rc.use_imu:
                    continue
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
            net.update_R_from_FC()
            net.update_F_from_VG(V)
            net.update_G_from_VF(V)
            net.update_G_from_I()
            net.update_I_from_G()
            net.update_F_from_RC()

            R_history.append(net.R.copy())
            res_history.append(net.residual_VFG(V))

            if it in snapshot_iters:
                snapshots[it] = {
                    'I': net.I[:H, :W].copy(), 'G': net.G.copy(),
                    'F': net.F.copy(), 'R': net.R.copy(),
                }

    R_history = np.array(R_history)

    # --- Save plot ---
    rc.save_params()
    n_snaps = len(snapshot_iters)
    fig, axes = plt.subplots(5, n_snaps, figsize=(3 * n_snaps, 14))
    fig.suptitle(f'Exp 1: Map Convergence ({rc.model}, {rc.dataset})\n'
                 f'GT ω = ({gt_omega[0]:.3f}, {gt_omega[1]:.3f}, {gt_omega[2]:.3f}) rad/s',
                 fontsize=11)

    for col, it in enumerate(snapshot_iters):
        snap = snapshots[it]
        omega_it = snap['R'] / rc.frame_duration

        axes[0, col].imshow(V, cmap='RdBu', vmin=-1, vmax=1)
        axes[0, col].set_title(f'iter {it}', fontsize=9)
        axes[1, col].imshow(normalise(snap['I']), cmap='gray')
        G_mag = np.sqrt(snap['G'][..., 0]**2 + snap['G'][..., 1]**2)
        axes[2, col].imshow(normalise(G_mag), cmap='hot')
        axes[3, col].imshow(flow_to_rgb(snap['F']))

        err = np.linalg.norm(snap['R'] - gt_R) / rc.frame_duration * 180 / np.pi
        axes[4, col].bar(['ωx','ωy','ωz'], omega_it,
                        color=['#4e79a7','#f28e2b','#e15759'])
        axes[4, col].bar(['ωx','ωy','ωz'], gt_omega,
                        color='none', edgecolor='black', linewidth=2)
        axes[4, col].set_title(f'err={err:.1f}°/s', fontsize=8)

    row_labels = ['V (input)', 'I (intensity)', '|G| (gradient)', 'F (flow)', 'ω']
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=9)
    for ax in axes[:4].flat:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(rc.output_dir, 'exp1_convergence.png'), dpi=150)
    plt.close()
    print(f"  Saved: {rc.output_dir}/exp1_convergence.png")


# ===========================================================================
# EXPERIMENT 2: Multi-Frame Tracking
# ===========================================================================
def experiment_tracking(rc: RunConfig, save_frames=True):
    """
    Main experiment: ω tracking over time.
    ALWAYS saves:
      - tracking.csv (per-frame metrics)
      - tracking_plot.png (ω over time)
      - summary.json
      - params.json
    If save_frames=True (default):
      - video_frames/frame_XXXX.png (3-col: Events | I | GT)
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Multi-Frame Angular Velocity Tracking")
    print(f"  Config: {rc}")
    print("="*70)

    rc.save_params()

    seq = EventFrameSequence(
        rc.paths['events'], rc.paths['calib'],
        frame_duration=rc.frame_duration, t_start=rc.t_start,
        n_frames=rc.n_frames, clip_value=10.0,
        sensor_size=rc.sensor_size,
    )
    H, W = seq.H, seq.W
    fx, fy, cx, cy = seq.calib.fx, seq.calib.fy, seq.calib.cx, seq.calib.cy
    imu_data = load_imu(rc.paths['imu'])

    net = make_network(rc, H, W, fx, fy, cx, cy)

    # ─── Frame saving setup ───────────────────────────────────────────
    frames_dir = None
    gt_images = None
    if save_frames:
        frames_dir = os.path.join(rc.output_dir, 'video_frames')
        os.makedirs(frames_dir, exist_ok=True)
        gt_images = _try_load_gt_images(rc)
        print(f"  Saving frames to: {frames_dir}/")

    rows = []
    for k, (V, t_mid) in enumerate(seq):
        t_lo = rc.t_start + k * rc.frame_duration
        t_hi = t_lo + rc.frame_duration
        gt_omega = get_gyro_for_frame(imu_data, t_lo, t_hi)

        if rc.use_thesis and rc.use_imu:
            net.step(V, n_iters=rc.n_iters, omega_imu=gt_omega)
        else:
            net.step(V, n_iters=rc.n_iters)

        omega_est = net.R / rc.frame_duration
        err, dir_err, beta = compute_metrics(omega_est, gt_omega)

        rows.append({
            'frame': k, 'time': t_mid,
            'est_wx': omega_est[0], 'est_wy': omega_est[1], 'est_wz': omega_est[2],
            'gt_wx': gt_omega[0], 'gt_wy': gt_omega[1], 'gt_wz': gt_omega[2],
            'err_deg_s': err, 'dir_err_deg': dir_err, 'beta': beta,
        })

        # ─── Save 3-column frame ─────────────────────────────────────
        if save_frames:
            _save_3col_frame(frames_dir, k, V, net, H, W, gt_images, rc)

        if (k + 1) % 10 == 0 or k == 0:
            print(f"  Frame {k+1:4d}/{rc.n_frames} | "
                  f"ω_est=({omega_est[0]:+.3f},{omega_est[1]:+.3f},{omega_est[2]:+.3f}) | "
                  f"err={err:.1f}°/s | dir={dir_err:.1f}°")

    # ─── Save CSV ─────────────────────────────────────────────────────
    csv_path = os.path.join(rc.output_dir, 'tracking.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # ─── Summary ──────────────────────────────────────────────────────
    err_all = np.array([r['err_deg_s'] for r in rows])
    dir_all = np.array([r['dir_err_deg'] for r in rows])
    beta_all = np.array([r['beta'] for r in rows])

    summary = {
        'mean_err_deg_s': float(np.mean(err_all)),
        'median_err_deg_s': float(np.median(err_all)),
        'mean_dir_err_deg': float(np.mean(dir_all)),
        'median_dir_err_deg': float(np.median(dir_all)),
        'mean_beta': float(np.mean(beta_all)),
        'std_beta': float(np.std(beta_all)),
    }

    with open(os.path.join(rc.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  {'='*50}")
    print(f"  RESULTS: {rc.model}, {rc.dataset}, {rc.duration_s:.1f}s")
    print(f"  {'='*50}")
    print(f"  Total error:     mean={summary['mean_err_deg_s']:.1f}°/s, "
          f"median={summary['median_err_deg_s']:.1f}°/s")
    print(f"  Direction error: mean={summary['mean_dir_err_deg']:.1f}°, "
          f"median={summary['median_dir_err_deg']:.1f}°")
    print(f"  Scale factor β:  mean={summary['mean_beta']:.2f} ± {summary['std_beta']:.2f}")
    if save_frames:
        print(f"  Frames saved:    {frames_dir}/ ({rc.n_frames} PNGs)")

    _plot_tracking(rows, rc)
    return summary

def _plot_tracking(rows, rc: RunConfig):
    """Tracking plot with ω components + error."""
    times = np.array([r['time'] for r in rows]) - rows[0]['time']
    omega_est = np.array([[r['est_wx'], r['est_wy'], r['est_wz']] for r in rows])
    omega_gt = np.array([[r['gt_wx'], r['gt_wy'], r['gt_wz']] for r in rows])
    err = np.array([r['err_deg_s'] for r in rows])

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'ω Tracking — {rc.dataset} ({rc.model})\n'
                 f'dt={rc.frame_duration*1000:.0f}ms, {rc.n_iters} iters, '
                 f'{rc.duration_s:.1f}s', fontsize=11)

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
    plt.savefig(os.path.join(rc.output_dir, 'tracking_plot.png'), dpi=150)
    plt.close()


# ===========================================================================
# EXPERIMENT 3: Parameter Influence
# ===========================================================================

def experiment_parameter_influence(rc: RunConfig, frame_idx=0):
    """Iterations sweep + frame duration sweep."""
    print("\n" + "="*70)
    print("EXPERIMENT 3: Parameter Influence")
    print(f"  Config: {rc}")
    print("="*70)

    rc.save_params()
    imu_data = load_imu(rc.paths['imu'])

    # (a) Iterations sweep
    print("\n  (a) Iterations sweep:")
    iter_counts = [1, 3, 5, 10, 20, 50, 100]
    iter_results = []

    seq = EventFrameSequence(
        rc.paths['events'], rc.paths['calib'],
        frame_duration=rc.frame_duration, t_start=rc.t_start,
        n_frames=frame_idx + 1, clip_value=10.0, sensor_size=rc.sensor_size,
    )
    frames = list(seq)
    V, _ = frames[frame_idx]
    H, W = seq.H, seq.W
    fx, fy, cx, cy = seq.calib.fx, seq.calib.fy, seq.calib.cx, seq.calib.cy

    t_lo = rc.t_start + frame_idx * rc.frame_duration
    t_hi = t_lo + rc.frame_duration
    gt_omega = get_gyro_for_frame(imu_data, t_lo, t_hi)

    for n_iters in iter_counts:
        net = make_network(rc, H, W, fx, fy, cx, cy)
        if rc.use_thesis and rc.use_imu:
            net.step(V, n_iters=n_iters, omega_imu=gt_omega)
        else:
            net.step(V, n_iters=n_iters)
        omega_est = net.R / rc.frame_duration
        err, dir_err, beta = compute_metrics(omega_est, gt_omega)
        iter_results.append({'iters': n_iters, 'err': err, 'dir': dir_err, 'beta': beta})
        print(f"    iters={n_iters:4d} | err={err:.2f}°/s | dir={dir_err:.1f}°")

    # (b) Frame duration sweep
    print("\n  (b) Frame duration sweep:")
    durations = [0.005, 0.010, 0.015, 0.020, 0.030, 0.050]
    dt_results = []

    for dt in durations:
        seq2 = EventFrameSequence(
            rc.paths['events'], rc.paths['calib'],
            frame_duration=dt, t_start=rc.t_start,
            n_frames=frame_idx + 1, clip_value=10.0, sensor_size=rc.sensor_size,
        )
        frames2 = list(seq2)
        V2, _ = frames2[frame_idx]
        H2, W2 = seq2.H, seq2.W

        t_lo2 = rc.t_start + frame_idx * dt
        t_hi2 = t_lo2 + dt
        gt_omega2 = get_gyro_for_frame(imu_data, t_lo2, t_hi2)

        # Temporarily override frame_duration for make_network
        orig_dt = rc.frame_duration
        rc.frame_duration = dt
        net = make_network(rc, H2, W2, fx, fy, cx, cy)
        rc.frame_duration = orig_dt

        if rc.use_thesis and rc.use_imu:
            net.step(V2, n_iters=rc.n_iters, omega_imu=gt_omega2)
        else:
            net.step(V2, n_iters=rc.n_iters)

        omega_est2 = net.R / dt
        err2, dir2, beta2 = compute_metrics(omega_est2, gt_omega2)
        n_active = np.count_nonzero(V2)
        dt_results.append({'dt_ms': dt*1000, 'err': err2, 'active': n_active})
        print(f"    dt={dt*1000:5.1f}ms | active={n_active:5d} | err={err2:.2f}°/s")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Exp 3: Parameter Influence ({rc.model}, {rc.dataset})', fontsize=11)

    axes[0].plot([r['iters'] for r in iter_results],
                [r['err'] for r in iter_results], 'bo-', lw=1.5)
    axes[0].set_xlabel('Iterations per frame')
    axes[0].set_ylabel('ω error (°/s)')
    axes[0].set_title(f'(a) Iterations (dt={rc.frame_duration*1000:.0f}ms)')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')

    axes[1].plot([r['dt_ms'] for r in dt_results],
                [r['err'] for r in dt_results], 'rs-', lw=1.5)
    axes[1].set_xlabel('Frame duration (ms)')
    axes[1].set_ylabel('ω error (°/s)')
    axes[1].set_title(f'(b) Frame duration ({rc.n_iters} iters)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(rc.output_dir, 'exp3_parameters.png'), dpi=150)
    plt.close()


# ===========================================================================
# EXPERIMENT 4: Qualitative Video
# ===========================================================================

def experiment_qualitative_video(rc: RunConfig):
    print("  (Exp 4 is now integrated into Exp 2 — running tracking with frames)")
    return experiment_tracking(rc, save_frames=True)


def _try_load_gt_images(rc: RunConfig):
    """Load GT APS images from images.txt."""
    data_dir = rc.paths['data_dir']
    images_file = os.path.join(data_dir, 'images.txt')

    if not os.path.exists(images_file):
        print("  No images.txt found")
        return None

    img_list = []
    t_end = rc.t_start + rc.n_frames * rc.frame_duration + 0.5

    with open(images_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            t = float(parts[0])
            fname = parts[1]

            if t < rc.t_start - 0.5:
                continue
            if t > t_end:
                break

            img_path = os.path.join(data_dir, fname)
            if os.path.exists(img_path):
                img_list.append((t, img_path))

    if img_list:
        print(f"  Loaded {len(img_list)} GT images")
    else:
        print("  No GT images found in time range")
        return None
    return img_list


def _save_3col_frame(out_dir, k, V, net, H, W, gt_images, rc: RunConfig):
    """Fixed-size 3-column frame."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(V, cmap='RdBu', vmin=-1, vmax=1)
    axes[0].set_title('Events (V)', fontsize=10)
    axes[0].axis('off')

    I_disp = net.I if net.I.shape == (H, W) else net.I[:H, :W]
    axes[1].imshow(normalise(I_disp), cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Estimated I', fontsize=10)
    axes[1].axis('off')

    if gt_images is not None and len(gt_images) > 0:
        t_frame = rc.t_start + k * rc.frame_duration + rc.frame_duration / 2
        closest = min(gt_images, key=lambda x: abs(x[0] - t_frame))
        gt_img = plt.imread(closest[1])
        axes[2].imshow(gt_img, cmap='gray')
        axes[2].set_title('GT Image (APS)', fontsize=10)
    else:
        axes[2].text(0.5, 0.5, 'No GT', ha='center', va='center',
                     fontsize=14, transform=axes[2].transAxes)
        axes[2].set_facecolor('#f0f0f0')
        axes[2].set_title('GT (N/A)', fontsize=10)
    axes[2].axis('off')

    plt.suptitle(f'Frame {k:04d}', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'frame_{k:04d}.png'), dpi=100)
    plt.close(fig)


# ===========================================================================
# EXPERIMENT 5: Make Video
# ===========================================================================
# ===========================================================================
# EXPERIMENT 5: Make Videos (batch-fähig)
# ===========================================================================

def experiment_make_videos_batch(base_dir='results', fps=15):
    """Find ALL video_frames/ directories and build videos."""
    import glob
    import imageio
    from PIL import Image

    pattern = os.path.join(base_dir, '**', 'video_frames')
    frame_dirs = sorted(glob.glob(pattern, recursive=True))

    if not frame_dirs:
        print(f"No video_frames/ directories found under {base_dir}")
        return

    print(f"Found {len(frame_dirs)} frame directories")

    for frames_dir in frame_dirs:
        frame_files = sorted(glob.glob(os.path.join(frames_dir, 'frame_*.png')))
        if not frame_files:
            continue

        output_path = os.path.join(os.path.dirname(frames_dir), 'comparison.mp4')
        if os.path.exists(output_path):
            print(f"  Skip (exists): {output_path}")
            continue

        first = Image.open(frame_files[0])
        w, h = first.size
        w = (w // 16) * 16
        h = (h // 16) * 16

        writer = imageio.get_writer(output_path, fps=fps)
        for f in frame_files:
            img = Image.open(f).resize((w, h), Image.LANCZOS)
            writer.append_data(np.array(img))
        writer.close()
        print(f"  Video: {output_path} ({len(frame_files)} frames)")

    print("Done.")

def experiment_make_video(rc: RunConfig, fps=15):
    """Assemble video from saved frames."""
    import imageio
    import glob
    from PIL import Image

    frames_dir = os.path.join(rc.output_dir, 'video_frames')
    output_path = os.path.join(rc.output_dir, 'comparison.mp4')

    frame_files = sorted(glob.glob(os.path.join(frames_dir, 'frame_*.png')))
    if not frame_files:
        print(f"No frames found in {frames_dir}")
        return

    first = Image.open(frame_files[0])
    w, h = first.size
    w = (w // 16) * 16
    h = (h // 16) * 16

    writer = imageio.get_writer(output_path, fps=fps)
    for f in frame_files:
        img = Image.open(f).resize((w, h), Image.LANCZOS)
        writer.append_data(np.array(img))
    writer.close()
    print(f"Video saved: {output_path} ({w}×{h}, {len(frame_files)} frames)")


# ===========================================================================
# EXPERIMENT 6: Basin of Attraction
# ===========================================================================

def experiment_basin_of_attraction(rc: RunConfig, frame_idx=0):
    """Initialize R at various distances from GT — does it converge?"""
    print("\n" + "="*70)
    print("EXPERIMENT 6: Basin of Attraction")
    print(f"  Config: {rc}")
    print("="*70)

    rc.save_params()

    seq = EventFrameSequence(
        rc.paths['events'], rc.paths['calib'],
        frame_duration=rc.frame_duration, t_start=rc.t_start,
        n_frames=frame_idx + 1, clip_value=10.0, sensor_size=rc.sensor_size,
    )
    frames = list(seq)
    V, _ = frames[frame_idx]
    H, W = seq.H, seq.W
    fx, fy, cx, cy = seq.calib.fx, seq.calib.fy, seq.calib.cx, seq.calib.cy

    imu_data = load_imu(rc.paths['imu'])
    t_lo = rc.t_start + frame_idx * rc.frame_duration
    t_hi = t_lo + rc.frame_duration
    gt_omega = get_gyro_for_frame(imu_data, t_lo, t_hi)
    gt_R = gt_omega * rc.frame_duration
    scale = max(np.linalg.norm(gt_R), 0.01)

    perturbations = [
        ("Exact GT", gt_R.copy()),
        ("GT + 10%", gt_R + 0.1 * scale * np.random.randn(3)),
        ("GT + 50%", gt_R + 0.5 * scale * np.random.randn(3)),
        ("GT + 100%", gt_R + 1.0 * scale * np.random.randn(3)),
        ("GT + 200%", gt_R + 2.0 * scale * np.random.randn(3)),
        ("Opposite", -gt_R),
        ("Zero", np.zeros(3)),
        ("Random", np.random.randn(3) * 0.05),
    ]

    results = []
    for label, R_init in perturbations:
        # Override initial_R temporarily
        orig_R = rc.initial_R.copy()
        rc.initial_R = R_init
        net = make_network(rc, H, W, fx, fy, cx, cy)
        rc.initial_R = orig_R

        if rc.use_thesis and rc.use_imu:
            net.step(V, n_iters=rc.n_iters, omega_imu=gt_omega)
        else:
            net.step(V, n_iters=rc.n_iters)

        final_R = net.R.copy()
        init_dist = np.linalg.norm(R_init - gt_R)
        final_dist = np.linalg.norm(final_R - gt_R)
        converged = final_dist < 0.5 * scale

        results.append({'label': label, 'init_dist': init_dist,
                       'final_dist': final_dist, 'converged': converged})
        print(f"  {label:15s} | init={init_dist:.5f} | final={final_dist:.5f} | "
              f"{'✓' if converged else '✗'}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results:
        color = 'green' if r['converged'] else 'red'
        ax.scatter(r['init_dist'], r['final_dist'], c=color, s=100, zorder=5)
        ax.annotate(r['label'], (r['init_dist'], r['final_dist']), fontsize=8)
    max_d = max(r['init_dist'] for r in results) * 1.1
    ax.plot([0, max_d], [0, max_d], 'k--', alpha=0.3, label='no improvement')
    ax.set_xlabel('Initial ||R - R_gt||')
    ax.set_ylabel('Final ||R - R_gt||')
    ax.set_title(f'Basin of Attraction ({rc.model}, {rc.dataset})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(rc.output_dir, 'exp6_basin.png'), dpi=150)
    plt.close()


# ===========================================================================
# EXPERIMENT 7: Full Systematic Evaluation
# ===========================================================================

def experiment_full_evaluation(dataset_filter=None, model_filter=None, save_frames=True):
    """
    Läuft über ALL datasets × ALL models × ALL SEGMENTS.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 7: Full Systematic Evaluation (All Segments)")
    print("="*70)

    # Datasets auswählen
    if dataset_filter:
        datasets = {dataset_filter: DATASET_SEGMENTS[dataset_filter]}
    else:
        datasets = DATASET_SEGMENTS

    # Models auswählen
    models = [model_filter] if model_filter else ['cook', 'thesis', 'thesis_imu']

    all_results = []

    for dataset, segments in datasets.items():
        for seg in segments:                          # ← NEU: innere Schleife
            for model in models:
                print(f"\n--- {dataset} / {seg['id']} / {model} ---")
                try:
                    rc = RunConfig(dataset=dataset, model=model, segment=seg)
                    summary = experiment_tracking(rc, save_frames=save_frames)
                    summary['dataset'] = dataset
                    summary['segment_id'] = seg['id']  # ← NEU
                    summary['model'] = model
                    summary['duration_s'] = rc.duration_s
                    all_results.append(summary)
                except Exception as e:
                    print(f"  FAILED: {e}")
                    all_results.append({
                        'dataset': dataset, 'segment_id': seg['id'],
                        'model': model,
                        'mean_err_deg_s': float('nan'),
                        'mean_dir_err_deg': float('nan'),
                        'mean_beta': float('nan'),
                        'duration_s': 0,
                    })

    # Tabelle ausgeben
    print("\n\n" + "="*90)
    print("FULL EVALUATION RESULTS")
    print("="*90)
    print(f"{'Dataset':<18} {'Segment':<8} {'Model':<12} "
          f"{'ω err°/s':<10} {'Dir err°':<10} {'β':<8} {'Dur(s)':<6}")
    print("-"*90)
    for r in all_results:
        print(f"{r['dataset']:<18} {r.get('segment_id','?'):<8} {r['model']:<12} "
              f"{r.get('mean_err_deg_s', float('nan')):>7.1f}   "
              f"{r.get('mean_dir_err_deg', float('nan')):>7.1f}   "
              f"{r.get('mean_beta', float('nan')):>6.2f}  "
              f"{r.get('duration_s', 0):>5.1f}")

    # CSV speichern
    os.makedirs('results', exist_ok=True)
    csv_path = 'results/full_evaluation.csv'
    if all_results:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nMaster table saved: {csv_path}")

    return all_results

# ===========================================================================
# EXPERIMENT 8: Automatic Parameter Grid Sweep
# ===========================================================================
def experiment_parameter_grid(dataset_filter=None, model_filter=None, 
                              all_segments=False, save_frames=True):
    """
    Sweep over:
      - frame_duration: [0.010, 0.020, 0.030]
      - n_frames: [25, 50, 150]
      - n_iters: [75, 100]
      - delta_FR: [0.10, 0.20, 0.30, 0.50]
      - delta_IMU: [0.10, 0.20, 0.30, 0.50]  (only for thesis_imu)
    
    For each config: runs tracking (Exp 2) AND single-frame convergence (Exp 1).
    """
    print("\n" + "="*70)
    print("EXPERIMENT 8: Automatic Parameter Grid Sweep")
    print("="*70)

    # ─── GRID DEFINITION ──────────────────────────────────────────────
    grid = {
        'frame_duration': [0.010, 0.020, 0.030],
        'n_frames':  [25, 50, 150],
        'n_iters':   [75, 100],
        'delta_FR':  [0.10, 0.20, 0.30, 0.50],
        'delta_IMU': [0.10, 0.20, 0.30, 0.50],
    }

    # ─── DATASETS ─────────────────────────────────────────────────────
    if dataset_filter:
        datasets = [dataset_filter]
    else:
        datasets = list(DATASET_SEGMENTS.keys())

    # ─── MODELS ───────────────────────────────────────────────────────
    if model_filter is None:
        models = ['cook', 'thesis', 'thesis_imu']
    else:
        models = [model_filter]

    all_results = []
    run_idx = 0

    # Count total runs
    total_runs = 0
    for dataset in datasets:
        n_segs = len(DATASET_SEGMENTS[dataset]) if all_segments else 1
        for model in models:
            n_imu = len(grid['delta_IMU']) if model == 'thesis_imu' else 1
            total_runs += (n_segs * len(grid['frame_duration']) * 
                          len(grid['n_frames']) * len(grid['n_iters']) * 
                          len(grid['delta_FR']) * n_imu)

    print(f"  Datasets: {datasets}")
    print(f"  Models:   {models}")
    print(f"  Segments: {'ALL' if all_segments else 'first only'}")
    print(f"  Grid axes: {list(grid.keys())}")
    print(f"  Total runs: {total_runs}")
    print()

    for dataset in datasets:
        if all_segments:
            segs = DATASET_SEGMENTS[dataset]
        else:
            segs = [DATASET_SEGMENTS[dataset][0]]

        for seg in segs:
            for model in models:
                imu_values = grid['delta_IMU'] if model == 'thesis_imu' else [None]

                for frame_duration in grid['frame_duration']:
                    for n_frames in grid['n_frames']:
                        for n_iters in grid['n_iters']:
                            for delta_FR in grid['delta_FR']:
                                for delta_IMU in imu_values:
                                    run_idx += 1
                                    imu_str = f", δ_IMU={delta_IMU}" if delta_IMU else ""
                                    print(f"\n{'─'*60}")
                                    print(f"  RUN {run_idx}/{total_runs}: "
                                          f"{dataset}/{seg['id']}/{model}")
                                    print(f"  dt={frame_duration*1000:.0f}ms, "
                                          f"n_frames={n_frames}, n_iters={n_iters}, "
                                          f"δ_FR={delta_FR}{imu_str}")
                                    print(f"{'─'*60}")

                                    try:
                                        rc = RunConfig(
                                            dataset=dataset,
                                            model=model,
                                            segment=seg,
                                            frame_duration=frame_duration,
                                            n_frames=n_frames,
                                            n_iters=n_iters,
                                            delta_IMU=delta_IMU,
                                        )
                                        rc.params['delta_FR'] = delta_FR

                                        # ── Run Exp 2 (tracking) ──
                                        summary = experiment_tracking(
                                            rc, save_frames=save_frames)

                                        # ── Run Exp 1 (convergence diagnostic) ──
                                        if save_frames:
                                            experiment_single_frame_convergence(
                                                rc, frame_idx=0, 
                                                max_iters=n_iters)

                                        result = {
                                            'dataset': dataset,
                                            'segment_id': seg['id'],
                                            'model': model,
                                            'frame_duration': frame_duration,
                                            'n_frames': n_frames,
                                            'n_iters': n_iters,
                                            'delta_FR': delta_FR,
                                            'delta_IMU': delta_IMU if delta_IMU else 0.0,
                                            'duration_s': n_frames * frame_duration,
                                            'mean_err_deg_s': summary['mean_err_deg_s'],
                                            'median_err_deg_s': summary['median_err_deg_s'],
                                            'mean_dir_err_deg': summary['mean_dir_err_deg'],
                                            'mean_beta': summary['mean_beta'],
                                        }
                                        all_results.append(result)

                                    except Exception as e:
                                        print(f"  FAILED: {e}")
                                        all_results.append({
                                            'dataset': dataset,
                                            'segment_id': seg['id'],
                                            'model': model,
                                            'frame_duration': frame_duration,
                                            'n_frames': n_frames,
                                            'n_iters': n_iters,
                                            'delta_FR': delta_FR,
                                            'delta_IMU': delta_IMU if delta_IMU else 0.0,
                                            'duration_s': n_frames * frame_duration,
                                            'mean_err_deg_s': float('nan'),
                                            'median_err_deg_s': float('nan'),
                                            'mean_dir_err_deg': float('nan'),
                                            'mean_beta': float('nan'),
                                        })

    # ─── RESULTS TABLE ─────────────────────────────────────────────────
    print("\n\n" + "="*130)
    print("PARAMETER GRID RESULTS")
    print("="*130)
    print(f"{'Dataset':<16} {'Seg':<6} {'Model':<12} {'dt_ms':>5} {'n_fr':>5} "
          f"{'iters':>5} {'δ_FR':>5} {'δ_IMU':>6} | "
          f"{'err°/s':>7} {'med°/s':>7} {'dir°':>6} {'β':>5}")
    print("-"*130)
    for r in all_results:
        print(f"{r['dataset']:<16} {r['segment_id']:<6} {r['model']:<12} "
              f"{r['frame_duration']*1000:>5.0f} {r['n_frames']:>5} "
              f"{r['n_iters']:>5} {r['delta_FR']:>5.2f} "
              f"{r['delta_IMU']:>6.2f} | "
              f"{r['mean_err_deg_s']:>7.1f} {r['median_err_deg_s']:>7.1f} "
              f"{r['mean_dir_err_deg']:>6.1f} {r['mean_beta']:>5.2f}")

    # ─── SAVE CSV ──────────────────────────────────────────────────────
    os.makedirs('results', exist_ok=True)
    if dataset_filter:
        csv_path = f'results/parameter_grid_{dataset_filter}.csv'
    else:
        csv_path = 'results/parameter_grid.csv'

    if all_results:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nGrid results saved: {csv_path}")

    # ─── FIND BEST PER MODEL ──────────────────────────────────────────
    valid = [r for r in all_results if not np.isnan(r['mean_err_deg_s'])]
    if valid:
        print(f"\n  ★ BEST CONFIG PER MODEL:")
        for m in models:
            m_valid = [r for r in valid if r['model'] == m]
            if m_valid:
                best = min(m_valid, key=lambda r: r['mean_err_deg_s'])
                imu_str = (f", δ_IMU={best['delta_IMU']:.2f}" 
                          if m == 'thesis_imu' else "")
                print(f"    [{m:12s}] {best['dataset']}/{best['segment_id']}, "
                      f"dt={best['frame_duration']*1000:.0f}ms, "
                      f"n={best['n_frames']}, i={best['n_iters']}, "
                      f"δ_FR={best['delta_FR']:.2f}{imu_str} "
                      f"→ {best['mean_err_deg_s']:.1f}°/s")

    return all_results


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Systematic Evaluation')
    parser.add_argument('--exp', type=int, default=2, help='1-7')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--model', type=str, default='thesis_imu',
                        choices=['cook', 'thesis', 'thesis_imu', 'all'])
    parser.add_argument('--segment', type=str, default=None,       # ← NEU
                        help='Segment ID (z.B. seg_A, seg_B)')
    parser.add_argument('--n_frames', type=int, default=None)
    parser.add_argument('--n_iters', type=int, default=None)
    parser.add_argument('--delta_imu', type=float, default=None)
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--all-segments', action='store_true',
                    help='Run grid sweep over ALL segments (slow)')
    parser.add_argument('--no-frames', action='store_true',
                        help='Skip saving 3-col PNGs (faster)')
    args = parser.parse_args()
    save_frames = not args.no_frames  # Default: save images

    # Build config ONLY for experiments that need it (1-6)
    
    # Resolve --model all → None (triggers all-models logic in exp 7/8)
    effective_model = None if args.model == 'all' else args.model

    # Build config ONLY for experiments that need it (1-6)
    rc = None
    if args.exp in (1, 2, 3, 4, 5, 6):
        dataset = args.dataset or 'boxes_rotation'
        model = args.model if args.model != 'all' else 'thesis_imu'
        rc = RunConfig(
            dataset=dataset,
            model=model,
            segment=args.segment,
            n_frames=args.n_frames,
            n_iters=args.n_iters,
            delta_IMU=args.delta_imu,
        )

    experiments = {
        1: lambda: experiment_single_frame_convergence(rc, frame_idx=args.frame),
        2: lambda: experiment_tracking(rc, save_frames=save_frames),
        3: lambda: experiment_parameter_influence(rc, frame_idx=args.frame),
        4: lambda: experiment_tracking(rc, save_frames=True),  # Always with frames
        #5: lambda: experiment_make_video(rc, fps=args.fps),
        5: lambda: experiment_make_videos_batch(fps=args.fps),  # ← Batch!
        6: lambda: experiment_basin_of_attraction(rc, frame_idx=args.frame),
        7: lambda: experiment_full_evaluation(
            dataset_filter=args.dataset,
            model_filter=effective_model,
            save_frames=save_frames,
        ),
        8: lambda: experiment_parameter_grid(
            dataset_filter=args.dataset,
            model_filter=effective_model,
            all_segments=args.all_segments,
            save_frames=save_frames,
        ),
    }

    experiments[args.exp]()