"""
Find time segments with approximately constant angular velocity in imu.txt.

Usage:
    python find_segments.py data/shapes_rotation/imu.txt
    python find_segments.py data/boxes_rotation/imu.txt --min_duration 0.5
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_imu(path: str) -> np.ndarray:
    """Load IMU: timestamp ax ay az gx gy gz"""
    data = np.loadtxt(path, dtype=np.float64)
    print(f"Loaded {len(data)} IMU measurements from {path}")
    print(f"  Time range: {data[0,0]:.3f} – {data[-1,0]:.3f} s")
    return data


def find_constant_velocity_segments(
    imu_data: np.ndarray,
    window_duration: float = 0.5,
    max_std_threshold: float = 0.1,
    step: float = 0.05,
    min_omega: float = 0.05,
):
    """
    Slide a window over the gyroscope data and find segments where
    the angular velocity is approximately constant (low std).

    Parameters
    ----------
    imu_data : (N, 7) — timestamp ax ay az gx gy gz
    window_duration : length of window in seconds
    max_std_threshold : maximum std (rad/s) per axis to count as "constant"
    step : window step size in seconds
    min_omega : minimum |ω| to exclude stationary segments

    Returns
    -------
    segments : list of dicts with keys:
        t_start, t_end, duration, mean_omega, std_omega, quality
    """
    t = imu_data[:, 0]
    gyro = imu_data[:, 4:7]  # gx, gy, gz

    t_min, t_max = t[0], t[-1]
    segments = []

    t_start = t_min
    while t_start + window_duration <= t_max:
        t_end = t_start + window_duration
        mask = (t >= t_start) & (t < t_end)

        if np.sum(mask) < 10:
            t_start += step
            continue

        window_gyro = gyro[mask]
        mean_omega = np.mean(window_gyro, axis=0)
        std_omega = np.std(window_gyro, axis=0)
        max_std = np.max(std_omega)
        omega_magnitude = np.linalg.norm(mean_omega)

        if max_std < max_std_threshold and omega_magnitude > min_omega:
            quality = omega_magnitude / (max_std + 1e-10)  # higher = better
            segments.append({
                't_start': t_start,
                't_end': t_end,
                'duration': window_duration,
                'mean_omega': mean_omega,
                'omega_magnitude': omega_magnitude,
                'std_omega': std_omega,
                'max_std': max_std,
                'quality': quality,
            })

        t_start += step

    # Sort by quality (best first)
    segments.sort(key=lambda s: s['quality'], reverse=True)
    return segments


def print_segments(segments, top_n=20):
    """Print the best segments."""
    print(f"\n{'='*80}")
    print(f"TOP {min(top_n, len(segments))} CONSTANT-VELOCITY SEGMENTS")
    print(f"{'='*80}")
    print(f"{'#':>3} | {'t_start':>8} {'t_end':>8} | "
          f"{'ωx':>7} {'ωy':>7} {'ωz':>7} | "
          f"{'|ω|':>6} {'max_σ':>6} {'quality':>7}")
    print(f"{'-'*80}")

    for i, seg in enumerate(segments[:top_n]):
        omega = seg['mean_omega']
        print(
            f"{i+1:3d} | {seg['t_start']:8.3f} {seg['t_end']:8.3f} | "
            f"{omega[0]:7.3f} {omega[1]:7.3f} {omega[2]:7.3f} | "
            f"{seg['omega_magnitude']:6.3f} {seg['max_std']:6.4f} {seg['quality']:7.1f}"
        )

    return segments[:top_n]


def plot_imu_with_segments(imu_data, segments, top_n=5):
    """Plot gyroscope data with highlighted constant-velocity segments."""
    t = imu_data[:, 0]
    gyro = imu_data[:, 4:7]

    fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
    fig.suptitle('IMU Gyroscope — Constant Velocity Segments Highlighted')

    labels = ['ω_x (rad/s)', 'ω_y (rad/s)', 'ω_z (rad/s)']
    colors = ['#4e79a7', '#f28e2b', '#e15759']

    for i in range(3):
        axes[i].plot(t, gyro[:, i], color=colors[i], lw=0.5, alpha=0.7)
        axes[i].set_ylabel(labels[i])
        axes[i].grid(True, alpha=0.3)

    # |ω|
    omega_mag = np.linalg.norm(gyro, axis=1)
    axes[3].plot(t, omega_mag, 'k-', lw=0.5, alpha=0.7)
    axes[3].set_ylabel('|ω| (rad/s)')
    axes[3].set_xlabel('Time (s)')
    axes[3].grid(True, alpha=0.3)

    # Highlight top segments
    seg_colors = plt.cm.Set2(np.linspace(0, 1, top_n))
    for i, seg in enumerate(segments[:top_n]):
        for ax in axes:
            ax.axvspan(seg['t_start'], seg['t_end'],
                      alpha=0.3, color=seg_colors[i],
                      label=f"#{i+1}: t={seg['t_start']:.2f}s" if ax == axes[0] else None)

    axes[0].legend(fontsize=7, loc='upper right')
    plt.tight_layout()
    #plt.savefig('imu_segments.png', dpi=150)
    plt.show()


def generate_config(segments, dataset_name: str, top_n=5):
    """Generate a config snippet for validation.py."""
    print(f"\n{'='*60}")
    print(f"SUGGESTED CONFIG FOR {dataset_name}")
    print(f"{'='*60}")
    print(f"# Copy one of these into your validation.py or demo.py:\n")

    for i, seg in enumerate(segments[:top_n]):
        omega = seg['mean_omega']
        # Estimate initial_R from mean omega * frame_duration
        dt = 0.020
        initial_R = omega * dt
        print(f"# Segment #{i+1}: quality={seg['quality']:.1f}, |ω|={seg['omega_magnitude']:.3f} rad/s")
        print(f"T_START = {seg['t_start']:.3f}")
        print(f"FRAME_DURATION = 0.020")
        print(f"N_FRAMES = {int(seg['duration'] / dt)}")
        print(f"initial_R = np.array([{initial_R[0]:.5f}, {initial_R[1]:.5f}, {initial_R[2]:.5f}])")
        print(f"# Expected ω ≈ [{omega[0]:.3f}, {omega[1]:.3f}, {omega[2]:.3f}] rad/s")
        print()

# Add at the end of find_segments.py:

def export_to_config(segments, dataset_name, top_n=5, frame_duration=0.020):
    """Print ready-to-paste DATASET_SEGMENTS entry matching config.py format."""
    print(f"\n# Paste into config.py DATASET_SEGMENTS['{dataset_name}']:")
    print(f"'{dataset_name}': [")
    for i, seg in enumerate(segments[:top_n]):
        omega = seg['mean_omega']
        n_frames = int(seg['duration'] / frame_duration)
        print(f"    {{  # quality={seg['quality']:.1f}, |ω|={seg['omega_magnitude']:.3f} rad/s")
        print(f"        'id': 'seg_{chr(65+i)}',")
        print(f"        't_start': {seg['t_start']:.3f},")
        print(f"        'frame_duration': {frame_duration},")
        print(f"        'n_frames': {n_frames},  # {seg['duration']:.1f}s / {frame_duration}s")
        print(f"        'initial_R': None,")
        print(f"        'expected_omega': np.array([{omega[0]:.3f}, {omega[1]:.3f}, {omega[2]:.3f}]),")
        print(f"        'sensor_size': (180, 240),")
        print(f"    }},")
    print(f"],")
    
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Find constant-velocity segments in IMU data')
    parser.add_argument('imu_file', type=str, help='Path to imu.txt')
    parser.add_argument('--window', type=float, default=0.5, help='Window duration (s)')
    parser.add_argument('--threshold', type=float, default=0.15, help='Max std threshold (rad/s)')
    parser.add_argument('--min_omega', type=float, default=0.05, help='Min |ω| to exclude stationary')
    parser.add_argument('--min_duration', type=float, default=0.5, help='Min segment duration (s)')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    parser.add_argument('--export', action='store_true', help='Print config.py snippet')

    args = parser.parse_args()

    imu_data = load_imu(args.imu_file)

    # Try multiple window sizes
    all_segments = []
    for window in [args.min_duration, args.min_duration * 2, args.min_duration * 3]:
        segs = find_constant_velocity_segments(
            imu_data,
            window_duration=window,
            max_std_threshold=args.threshold,
            min_omega=args.min_omega,
        )
        all_segments.extend(segs)

    # Deduplicate (remove overlapping segments, keep best quality)
    all_segments.sort(key=lambda s: s['quality'], reverse=True)
    filtered = []
    for seg in all_segments:
        overlap = False
        for kept in filtered:
            if not (seg['t_end'] < kept['t_start'] or seg['t_start'] > kept['t_end']):
                overlap = True
                break
        if not overlap:
            filtered.append(seg)

    if not filtered:
        print("\nNo constant-velocity segments found!")
        print("Try: --threshold 0.3 --min_omega 0.02")
        sys.exit(1)

    top = print_segments(filtered)
    dataset_name = Path(args.imu_file).parent.name
    generate_config(filtered, dataset_name)
    if args.export:
        export_to_config(filtered, dataset_name, top_n=5)
    if not args.no_plot:
        plot_imu_with_segments(imu_data, filtered)