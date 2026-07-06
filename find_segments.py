"""
Find time segments with approximately constant angular velocity in imu.txt.

Usage:
    python find_segments.py data/boxes_rotation/imu.txt
    python find_segments.py data/dynamic_rotation/imu.txt --top 10
    python find_segments.py data/poster_rotation/imu.txt --export
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


def check_window_quality(gyro_window, min_omega=0.30, max_relative_std=0.25):
    """
    Check if a gyroscope window qualifies as constant-velocity.
    
    Returns (passes, quality, stats_dict) or (False, 0, None).
    """
    if len(gyro_window) < 10:
        return False, 0.0, None
    
    mean_omega = np.mean(gyro_window, axis=0)
    std_omega = np.std(gyro_window, axis=0)
    omega_magnitude = np.linalg.norm(mean_omega)
    
    if omega_magnitude < min_omega:
        return False, 0.0, None
    
    # Dominant axis analysis
    abs_mean = np.abs(mean_omega)
    dominant_axis = np.argmax(abs_mean)
    dominant_magnitude = abs_mean[dominant_axis]
    dominant_std = std_omega[dominant_axis]
    relative_std_dominant = dominant_std / (dominant_magnitude + 1e-10)
    
    # Magnitude stability
    omega_magnitudes = np.linalg.norm(gyro_window, axis=1)
    magnitude_std = np.std(omega_magnitudes)
    relative_std_magnitude = magnitude_std / (omega_magnitude + 1e-10)
    
    # Direction stability: check that the rotation axis doesn't drift
    # Compute angle between each sample's ω and mean ω
    if omega_magnitude > 0.1:
        dots = gyro_window @ mean_omega / (
            np.linalg.norm(gyro_window, axis=1, keepdims=False) * omega_magnitude + 1e-10
        )
        dots = np.clip(dots, -1, 1)
        direction_spread = np.std(np.arccos(dots))  # rad
    else:
        direction_spread = 0.0
    
    # Use the better (lower) of the two relative stds
    relative_std = min(relative_std_dominant, relative_std_magnitude)
    
    if relative_std > max_relative_std:
        return False, 0.0, None
    
    # Also reject if direction drifts too much (>15° std)
    if direction_spread > 0.26:  # ~15 degrees
        return False, 0.0, None
    
    quality = omega_magnitude * (1.0 - relative_std)
    
    stats = {
        'mean_omega': mean_omega,
        'omega_magnitude': omega_magnitude,
        'std_omega': std_omega,
        'dominant_axis': int(dominant_axis),
        'relative_std': relative_std,
        'direction_spread_deg': np.degrees(direction_spread),
    }
    return True, quality, stats


def find_constant_velocity_segments(
    imu_data: np.ndarray,
    window_durations: list = None,
    step: float = 0.05,
    min_omega: float = 0.30,
    max_relative_std: float = 0.25,
    frame_duration: float = 0.020,
):
    """
    Find segments where angular velocity is approximately constant.
    
    For each candidate starting point, finds the LONGEST window that
    still passes the constant-velocity test. This determines max_n_frames.
    
    Parameters
    ----------
    imu_data : (N, 7) — timestamp ax ay az gx gy gz
    window_durations : list of durations to test (longest first internally)
    step : window step size in seconds  
    min_omega : minimum |ω| to exclude near-stationary segments
    max_relative_std : max relative std on dominant axis
    frame_duration : dt for computing max_n_frames
    
    Returns
    -------
    segments : list of dicts sorted by quality (best first)
    """
    if window_durations is None:
        window_durations = [0.5, 1.0, 1.5, 2.0, 3.0]
    
    # Sort longest first (we'll find the max duration that passes)
    window_durations = sorted(window_durations, reverse=True)
    
    t = imu_data[:, 0]
    gyro = imu_data[:, 4:7]
    t_min, t_max = t[0], t[-1]
    
    segments = []
    t_start = t_min
    
    while t_start + window_durations[-1] <= t_max:  # shortest window must fit
        # Try windows from longest to shortest
        best_duration = None
        best_quality = 0
        best_stats = None
        
        for duration in window_durations:
            if t_start + duration > t_max:
                continue
            
            t_end = t_start + duration
            mask = (t >= t_start) & (t < t_end)
            
            if np.sum(mask) < 10:
                continue
            
            window_gyro = gyro[mask]
            passes, quality, stats = check_window_quality(
                window_gyro, min_omega=min_omega, max_relative_std=max_relative_std
            )
            
            if passes:
                best_duration = duration
                best_quality = quality
                best_stats = stats
                break  # longest passing window found
        
        if best_duration is not None:
            max_n_frames = int(best_duration / frame_duration)
            segments.append({
                't_start': t_start,
                't_end': t_start + best_duration,
                'duration': best_duration,
                'max_n_frames': max_n_frames,
                'mean_omega': best_stats['mean_omega'],
                'omega_magnitude': best_stats['omega_magnitude'],
                'std_omega': best_stats['std_omega'],
                'dominant_axis': best_stats['dominant_axis'],
                'relative_std': best_stats['relative_std'],
                'direction_spread_deg': best_stats['direction_spread_deg'],
                'quality': best_quality,
            })
        
        t_start += step
    
    segments.sort(key=lambda s: s['quality'], reverse=True)
    return segments


def deduplicate_segments(segments, min_gap=0.5):
    """Remove overlapping segments, keeping highest quality."""
    filtered = []
    for seg in segments:
        overlap = False
        for kept in filtered:
            if not (seg['t_end'] + min_gap < kept['t_start'] or
                    seg['t_start'] - min_gap > kept['t_end']):
                overlap = True
                break
        if not overlap:
            filtered.append(seg)
    return filtered


def print_segments(segments, top_n=10):
    """Print the best segments."""
    axis_names = ['x', 'y', 'z']
    n = min(top_n, len(segments))
    print(f"\n{'='*100}")
    print(f"TOP {n} CONSTANT-VELOCITY SEGMENTS")
    print(f"{'='*100}")
    print(f"{'#':>3} | {'t_start':>7} {'t_end':>7} {'dur':>5} {'max_n':>5} | "
          f"{'ωx':>7} {'ωy':>7} {'ωz':>7} | "
          f"{'|ω|':>5} {'dom':>3} {'rσ':>5} {'dir°':>5} {'qual':>5}")
    print(f"{'-'*100}")

    for i, seg in enumerate(segments[:n]):
        omega = seg['mean_omega']
        dom = axis_names[seg['dominant_axis']]
        print(
            f"{i+1:3d} | {seg['t_start']:7.3f} {seg['t_end']:7.3f} "
            f"{seg['duration']:5.1f} {seg['max_n_frames']:5d} | "
            f"{omega[0]:7.3f} {omega[1]:7.3f} {omega[2]:7.3f} | "
            f"{seg['omega_magnitude']:5.3f} {dom:>3} "
            f"{seg['relative_std']:5.3f} {seg['direction_spread_deg']:5.1f} "
            f"{seg['quality']:5.2f}"
        )

    return segments[:n]


def generate_config(segments, dataset_name: str, top_n=5, frame_duration=0.020):
    """Generate config snippet."""
    print(f"\n{'='*60}")
    print(f"SUGGESTED CONFIG FOR {dataset_name}")
    print(f"{'='*60}")
    print(f"# Copy one of these into your config.py:\n")

    for i, seg in enumerate(segments[:top_n]):
        omega = seg['mean_omega']
        dt = frame_duration
        initial_R = omega * dt
        axis_names = ['x', 'y', 'z']
        dom = axis_names[seg['dominant_axis']]
        print(f"# Segment #{i+1}: quality={seg['quality']:.2f}, "
              f"|ω|={seg['omega_magnitude']:.3f} rad/s, dominant=ω_{dom}")
        print(f"# Validated duration: {seg['duration']:.1f}s → max_n_frames={seg['max_n_frames']}")
        print(f"T_START = {seg['t_start']:.3f}")
        print(f"FRAME_DURATION = {dt}")
        print(f"N_FRAMES = {seg['max_n_frames']}  # max safe value")
        print(f"initial_R = np.array([{initial_R[0]:.5f}, {initial_R[1]:.5f}, {initial_R[2]:.5f}])")
        print()


def export_to_config(segments, dataset_name, top_n=5, frame_duration=0.020, target_n_frames=150):
    """Print ready-to-paste DATASET_SEGMENTS entry."""
    print(f"\n# Paste into config.py DATASET_SEGMENTS['{dataset_name}']:")
    print(f"'{dataset_name}': [")
    for i, seg in enumerate(segments[:top_n]):
        omega = seg['mean_omega']
        axis_names = ['x', 'y', 'z']
        dom = axis_names[seg['dominant_axis']]
        print(f"    {{  # quality={seg['quality']:.2f}, |ω|={seg['omega_magnitude']:.3f} rad/s, "
              f"dominant=ω_{dom}")
        print(f"        'id': 'seg_{chr(65+i)}',")
        print(f"        't_start': {seg['t_start']:.3f},")
        print(f"        'frame_duration': {frame_duration},")
        print(f"        'n_frames': {target_n_frames},  # {target_n_frames * frame_duration:.1f}s")
        print(f"        'initial_R': None,")
        print(f"        'sensor_size': (180, 240),")
        print(f"    }},")
    print(f"],")


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

    omega_mag = np.linalg.norm(gyro, axis=1)
    axes[3].plot(t, omega_mag, 'k-', lw=0.5, alpha=0.7)
    axes[3].set_ylabel('|ω| (rad/s)')
    axes[3].set_xlabel('Time (s)')
    axes[3].grid(True, alpha=0.3)

    seg_colors = plt.cm.Set2(np.linspace(0, 1, min(top_n, len(segments))))
    for i, seg in enumerate(segments[:top_n]):
        for ax in axes:
            ax.axvspan(seg['t_start'], seg['t_end'],
                      alpha=0.3, color=seg_colors[i],
                      label=(f"#{i+1}: |ω|={seg['omega_magnitude']:.2f}, "
                             f"{seg['duration']:.1f}s")
                      if ax == axes[0] else None)

    axes[0].legend(fontsize=7, loc='upper right')
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Find constant-velocity segments in IMU data')
    parser.add_argument('imu_file', type=str, help='Path to imu.txt')
    parser.add_argument('--dt', type=float, default=0.020, help='Frame duration (s)')
    parser.add_argument('--max-relative-std', type=float, default=0.25,
                        help='Max relative std (default 0.25 = 25%%)')
    parser.add_argument('--min-omega', type=float, default=0.30,
                        help='Min |ω| in rad/s (default 0.30)')
    parser.add_argument('--step', type=float, default=0.05, help='Window step (s)')
    parser.add_argument('--top', type=int, default=10, help='Number of segments')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    parser.add_argument('--export', action='store_true', help='Print config.py snippet')
    parser.add_argument('--min-gap', type=float, default=3.0,
                    help='Min gap between segment starts (s). '
                         'Set to max tested duration to avoid overlap.')

    args = parser.parse_args()

    imu_data = load_imu(args.imu_file)

    segments = find_constant_velocity_segments(
        imu_data,
        window_durations=[0.5, 1.0, 1.5, 2.0, 3.0],
        step=args.step,
        min_omega=args.min_omega,
        max_relative_std=args.max_relative_std,
        frame_duration=args.dt,
    )

    if not segments:
        print(f"\nNo segments found! Try: --min-omega 0.15 --max-relative-std 0.35")
        sys.exit(1)

    segments = deduplicate_segments(segments, min_gap=args.min_gap)
    print(f"\nFound {len(segments)} non-overlapping segments")

    top = print_segments(segments, top_n=args.top)

    dataset_name = Path(args.imu_file).parent.name
    generate_config(segments, dataset_name, top_n=min(args.top, 5), frame_duration=args.dt)

    if args.export:
        export_to_config(segments, dataset_name, top_n=5, frame_duration=args.dt)

    if not args.no_plot:
        plot_imu_with_segments(imu_data, segments, top_n=min(5, len(segments)))