"""
evaluation.py — Systematic evaluation harness for Interacting Maps.

Models (``--model``)
  cook           Cook 2011 network (Gauss-Seidel)
  thesis         Martel 2019 network (Jacobi two-phase)
  thesis_imu     thesis + Cost_IMU: R anchored to the gyro every iteration
  thesis_cmax    V1 — R anchored to a full CMax solve per frame (IMU-free in-loop)
  thesis_cmax_v2 V2 — R driven by one CMax gradient step per iteration (CMax-only)

Angular velocity is scored against groundtruth.txt (quaternion differencing,
body frame — the independent reference; see SCORE_AGAINST) rather than the IMU,
so the IMU-fed models are not graded on their own input. Metrics per frame:
err = ‖ω_est − ω_ref‖ (deg/s), direction error (deg), and β = ‖ω_ref‖/‖ω_est‖.

Experiments (``--exp N``)
  1. Single-frame convergence (maps at iteration checkpoints)
  2. Multi-frame tracking (ω_est vs ω_ref over time)   ← the core; 4/7/8/9 call it
  3. Parameter influence (iterations & frame-duration sweeps for one frame)
  4. Qualitative video (tracking with 3-column frames: Events | I | GT APS)
  5. Assemble MP4s from saved frames (batch)
  6. Basin of attraction (how far can init-R be from GT and still converge?)
  7. Full evaluation (all datasets × models × segments → full_evaluation.csv)
  8. Parameter grid (sweep frame_duration × n_frames × n_iters × delta_FR × delta_IMU)
  9. Distortion ablation (Way-1 undistort-events vs Way-2 distortion-aware C/warp)

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

    # exp 9 
    python evaluation.py --exp 9 --dataset boxes_rotation  --segment all --model thesis_imu --no-frames
    python evaluation.py --exp 9 --dataset poster_rotation --segment all --model thesis_imu --no-frames
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import csv
import json
import time as time_module

# The pipeline prints Unicode (omega, delta, beta, degree signs) throughout. On a
# Windows console (cp1252) those crash with UnicodeEncodeError. Reconfigure stdout
# to UTF-8 once, here at the hub every entry point imports (CLI, tests, report),
# so no caller needs PYTHONUTF8=1. Guarded: never fails, degrades to 'replace'.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from config import DATASET_SEGMENTS, DISTORTION_MODE, get_dataset_paths
from data_loader import EventFrameSequence
import provenance

# Helpers extracted into focused modules (R3/R5 of architecture_review.md) and
# re-exported here so existing callers (incl. report scripts' E.<name>) are
# unchanged. See eval_io.py / metrics.py / viz.py / run_config.py.
from eval_io import (SCORE_AGAINST, GT_DIFF_HALFWIDTH, load_imu,
                     get_gyro_for_frame, load_groundtruth, _quat_to_rotmat,
                     gt_omega_body, load_omega_gt, omega_gt_at,
                     get_reference_omega)
from metrics import (compute_metrics, curl_share, _intensity_stats,
                     _corr_to_aps, _recon_scores)
from viz import normalise, normalise_robust, flow_to_rgb, grad_to_rgb
from run_config import (RunConfig, resolve_best_kwargs, make_network,
                        _try_load_gt_images)

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
        undistort=rc.undistort_at_event_level,
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
def experiment_tracking(rc: RunConfig, save_frames=True, frame_stride=1):
    """
    Main experiment: ω tracking over time.
    ALWAYS saves:
      - tracking.csv (per-frame metrics)
      - tracking_plot.png (ω over time)
      - run.json (provenance + config + summary, one self-describing file)
    Reported runs are also mirrored to the local MLflow store (see provenance.py).
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
        undistort=rc.undistort_at_event_level,
        sensor_size=rc.sensor_size,
    )
    H, W = seq.H, seq.W
    fx, fy, cx, cy = seq.calib.fx, seq.calib.fy, seq.calib.cx, seq.calib.cy
    imu_data = load_imu(rc.paths['imu'])            # model INPUT (gyro)
    gt_data  = load_groundtruth(rc.paths['groundtruth'])  # SCORING ref (Vicon)
    om_data  = load_omega_gt(rc.paths.get('omega_gt'))     # exact omega, if shipped

    if SCORE_AGAINST == 'groundtruth' and om_data is not None:
        ref_src = 'omega_gt'
    elif SCORE_AGAINST == 'groundtruth' and gt_data is not None:
        ref_src = 'groundtruth'
    else:
        ref_src = 'imu'
    if ref_src == 'imu':
        print("  ⚠ Scoring against IMU gyro (no groundtruth.txt). "
              "For thesis_imu this is circular — the model is graded on its own input.")
    elif ref_src == 'omega_gt':
        print("  Scoring against omega_gt.txt (exact angular velocity, no differencing).")
    else:
        print(f"  Scoring against groundtruth.txt (Vicon poses differenced over "
              f"±{GT_DIFF_HALFWIDTH*1000:.0f} ms, independent of the IMU input).")

    net = make_network(rc, H, W, fx, fy, cx, cy)

    # ─── CMax front-end (V1: full CMax per frame → R anchor) ──────────
    # B1: load the raw events separately and slice per window here, leaving
    # EventFrameSequence untouched. The gyro still provides frame-0 init only.
    cmax_est = None
    cmax_events = None
    omega_cmax_prev = np.zeros(3)
    omega_anchor = None
    if getattr(rc, 'use_cmax', False):
        from data_loader import load_events_fast, undistort_events
        dur = rc.n_frames * rc.frame_duration + 0.1
        cmax_events = undistort_events(
            load_events_fast(rc.paths['events'], t_start=rc.t_start, duration=dur),
            seq.calib)
        if getattr(rc, 'use_cmax_v2', False):
            print(f"  CMax V2 active — R driven by 1 CMax step / iteration "
                  f"(lr={rc.cmax_lr}, {len(cmax_events)} events).")
        else:
            # V1: standalone full-solve estimator, feeds the R anchor.
            from cmax import CMaxAngularVelocity
            cmax_est = CMaxAngularVelocity(H, W, fx, fy, cx, cy, use_polarity=True)
            print(f"  CMax V1 active — R anchored to full CMax ω per frame "
                  f"({len(cmax_events)} events).")

    # ─── IWE logging (final contrast-maximized IWE per frame) ─────────
    iwe_dir = None
    if getattr(rc, 'save_iwe', False) and getattr(rc, 'use_cmax', False):
        from cmax.iwe_io import save_iwe, plot_contrast_curve, make_gif
        iwe_dir = os.path.join(rc.output_dir, 'iwe')
        os.makedirs(iwe_dir, exist_ok=True)
        print(f"  Saving final IWE per frame to: {iwe_dir}/")

    # ─── Frame saving setup ───────────────────────────────────────────
    # APS frames are needed for the per-frame reconstruction score whether or
    # not we are writing pictures, so load them unconditionally.
    frames_dir = None
    gt_images = _try_load_gt_images(rc)
    if save_frames:
        frames_dir = os.path.join(rc.output_dir, 'video_frames')
        os.makedirs(frames_dir, exist_ok=True)
        print(f"  Saving frames to: {frames_dir}/")

    rows = []
    for k, (V, t_mid) in enumerate(seq):
        t_lo = rc.t_start + k * rc.frame_duration
        t_hi = t_lo + rc.frame_duration

        # Model INPUT: gyro (the sensor being fused). Independent of scoring.
        omega_imu = get_gyro_for_frame(imu_data, t_lo, t_hi)
        # SCORING reference: Vicon (independent), falls back to gyro if absent.
        omega_ref, _ = get_reference_omega(gt_data, imu_data, t_lo, t_hi, om_data)

        # Raw events for this window (V1 anchor and V2 in-loop both need them).
        win = None
        if cmax_events is not None:
            win = cmax_events[(cmax_events[:, 0] >= t_lo) & (cmax_events[:, 0] < t_hi)]

        if getattr(rc, 'use_cmax_v2', False):
            # V2: CMax drives R inside the message passing.
            net.step(V, n_iters=rc.n_iters, events=win)
        elif cmax_est is not None:
            # V1: full CMax solve → R anchor (via Cost_IMU mechanism).
            omega_anchor = cmax_est.estimate(win, t_ref=0.5 * (t_lo + t_hi),
                                             omega_init=omega_cmax_prev)
            omega_cmax_prev = omega_anchor.copy()
            net.step(V, n_iters=rc.n_iters, omega_imu=omega_anchor)
        elif rc.use_thesis and rc.use_imu:
            net.step(V, n_iters=rc.n_iters, omega_imu=omega_imu)
        else:
            net.step(V, n_iters=rc.n_iters)

        omega_est = net.R / rc.frame_duration
        err, dir_err, beta = compute_metrics(omega_est, omega_ref)

        # ─── Save the final contrast-maximized IWE for this frame ────
        if iwe_dir is not None and win is not None:
            if getattr(rc, 'use_cmax_v2', False):
                iwe, contrast = net.cost_cmax.build_current_iwe()   # at final R
                w_iwe = omega_est
            else:
                iwe = cmax_est.last_iwe                              # at CMax solve ω
                contrast = float(np.var(iwe)) if iwe is not None else 0.0
                w_iwe = omega_anchor
            if iwe is not None:
                save_iwe(iwe, w_iwe, contrast, iwe_dir, k, t_mid, len(win))

        _rr = _recon_scores(net, H, W, gt_images, t_mid)
        rows.append({
            'frame': k, 'time': t_mid,
            'est_wx': omega_est[0], 'est_wy': omega_est[1], 'est_wz': omega_est[2],
            # gt_* is the SCORING reference (Vicon when available)
            'gt_wx': omega_ref[0], 'gt_wy': omega_ref[1], 'gt_wz': omega_ref[2],
            # imu_* is the gyro fed to the model (kept for comparison/debug)
            'imu_wx': omega_imu[0], 'imu_wy': omega_imu[1], 'imu_wz': omega_imu[2],
            'err_deg_s': err, 'dir_err_deg': dir_err, 'beta': beta,
            'ref_source': ref_src,
            # Reconstruction quality AT THIS FRAME. Scored every frame, not
            # once at the end: the intensity map is not uniform along a track
            # and breaks down intermittently, so a single end-of-run snapshot
            # cannot see a change that only affects how often it breaks.
            'recon_r': _rr[1], 'recon_r_inloop': _rr[0],
            'I_std': float(net.I[:H, :W].std()),
        })

        # ─── Save 3-column frame ─────────────────────────────────────
        if save_frames:
            if frame_stride <= 1 or k % frame_stride == 0 \
                    or k == rc.n_frames - 1:
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

    # ─── IWE sequence artefacts (contrast curve + GIF) ────────────────
    if iwe_dir is not None:
        plot_contrast_curve(iwe_dir)
        make_gif(iwe_dir)
        print(f"  IWE: {rc.n_frames} PNGs + iwe_log.csv + contrast_curve.png "
              f"(+gif) → {iwe_dir}/")

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

    # ─── Reconstruction quality: keep the final maps and two cheap ────
    # descriptors of I. These do not replace looking at the frames -- they let
    # many runs be ranked and re-examined without running them again.
    np.savez_compressed(os.path.join(rc.output_dir, 'maps_final.npz'),
                        I=net.I[:H, :W], G=net.G[:H, :W], F=net.F[:H, :W])
    summary.update(_intensity_stats(net.I[:H, :W]))
    summary['curl_share'] = curl_share(net.G[:H, :W])

    # Reconstruction quality over the WHOLE track, not just its final frame.
    # The worst-case matters as much as the mean: a term that stops the map
    # breaking down occasionally shows up in recon_r_min and in how often I
    # blows up, and not at all in a snapshot taken at a good moment.
    for key, pref in (('recon_r', ''), ('recon_r_inloop', '_inloop')):
        a = np.array([r[key] for r in rows], dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size:
            summary.update({
                f'recon_r_mean{pref}': float(a.mean()),
                f'recon_r_min{pref}': float(a.min()),
                f'recon_r_p10{pref}': float(np.percentile(a, 10)),
            })
    istd = np.array([r['I_std'] for r in rows], dtype=np.float64)
    if istd.size:
        med = float(np.median(istd))
        summary.update({
            'I_std_median': med,
            'I_std_max': float(istd.max()),
            # how much of the track the intensity map spends blown up
            'blowup_frac': float((istd > 5.0 * med).mean()) if med > 0 else 0.0,
        })

    # One self-describing file per run: provenance + config + metrics (R1).
    rc.save_run(summary)

    # Mirror to the local MLflow store. Reported runs (out_root == 'results') are
    # logged by default; throwaway sweeps under experiments/ are skipped unless
    # IM_MLFLOW_ALL=1, so the tracking store is not flooded. IM_MLFLOW=0 disables.
    if os.environ.get('IM_MLFLOW_ALL') == '1' or rc.out_root == 'results':
        provenance.mlflow_log_run(rc.to_dict(), summary, output_dir=rc.output_dir)

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

    ref_src = rows[0].get('ref_source', 'imu')
    ref_label = 'GT (Vicon quat)' if ref_src == 'groundtruth' else 'GT (IMU)'
    # Only overlay the gyro separately when it is NOT the scoring reference.
    has_imu_col = 'imu_wx' in rows[0]
    omega_imu = (np.array([[r['imu_wx'], r['imu_wy'], r['imu_wz']] for r in rows])
                 if has_imu_col else None)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'ω Tracking — {rc.dataset} ({rc.model})\n'
                 f'dt={rc.frame_duration*1000:.0f}ms, {rc.n_iters} iters, '
                 f'{rc.duration_s:.1f}s  |  scored vs {ref_label}', fontsize=11)

    labels = ['ωx', 'ωy', 'ωz']
    colors = ['#4e79a7', '#f28e2b', '#e15759']

    for i in range(3):
        axes[i].plot(times, omega_gt[:, i], 'k-', lw=1.0, alpha=0.8, label=ref_label)
        if omega_imu is not None and ref_src == 'groundtruth':
            axes[i].plot(times, omega_imu[:, i], color='#59a14f', lw=0.8, ls=':',
                         alpha=0.7, label='IMU gyro (model input)')
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

def experiment_distortion_ablation(rc: RunConfig, save_frames=False):
    """
    EXPERIMENT 9: run the SAME tracking config under both distortion models
    and tabulate metrics.
        undistort_events : events undistorted, pinhole C        (baseline)
        C_full           : distortion in C, directions + Jacobian
    """
    print("\n" + "="*70)
    print("EXPERIMENT 9: Distortion-Handling Ablation")
    print(f"  Base config: {rc}")
    print("="*70)

    modes = ['undistort_events', 'C_full']
    rows = []
    for mode in modes:
        rc_m = RunConfig(
            dataset=rc.dataset, model=rc.model, segment=rc.segment,
            frame_duration=rc.frame_duration, n_frames=rc.n_frames,
            n_iters=rc.n_iters,
            delta_IMU=(rc.delta_IMU if rc.use_imu else None),
            delta_FR=rc.params.get('delta_FR'),
            distortion_mode=mode,
        )
        summary = experiment_tracking(rc_m, save_frames=save_frames)
        rows.append({
            'mode': mode,
            'mean_err_deg_s': summary['mean_err_deg_s'],
            'median_err_deg_s': summary['median_err_deg_s'],
            'mean_dir_err_deg': summary['mean_dir_err_deg'],
            'mean_beta': summary['mean_beta'],
            'std_beta': summary['std_beta'],
        })

    print("\n\n" + "="*80)
    print(f"DISTORTION ABLATION — {rc.dataset}/{rc.segment_id}, model={rc.model}")
    print("="*80)
    print(f"{'mode':<18} {'err°/s':>8} {'med°/s':>8} {'dir°':>7} {'β':>6} {'σβ':>6}")
    print("-"*80)
    for r in rows:
        print(f"{r['mode']:<18} {r['mean_err_deg_s']:>8.1f} "
              f"{r['median_err_deg_s']:>8.1f} {r['mean_dir_err_deg']:>7.1f} "
              f"{r['mean_beta']:>6.2f} {r['std_beta']:>6.2f}")

    out = os.path.join('results', 'ablation_distortion')
    os.makedirs(out, exist_ok=True)
    csv_path = os.path.join(out, f"{rc.dataset}_{rc.segment_id}_{rc.model}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nAblation CSV saved: {csv_path}")
    return rows

def experiment_distortion_ablation_all_segments(dataset, model, save_frames=False,
                                                use_best=False):
    """
    Run the distortion ablation over EVERY segment of a dataset and emit one
    combined CSV (dataset × segment × mode) plus a grouped summary table.

    If use_best=True, each segment's base config is taken from best_config.py
    (best grid params for that dataset/segment/model) instead of the defaults.
    """
    print("\n" + "="*80)
    print(f"EXPERIMENT 9 (ALL SEGMENTS): Distortion Ablation — {dataset}, model={model}")
    print("="*80)

    segs = DATASET_SEGMENTS[dataset]
    all_rows = []
    for seg in segs:
        best_kw = resolve_best_kwargs(dataset, seg['id'], model) if use_best else {}
        rc = RunConfig(dataset=dataset, model=model, segment=seg['id'],
                       distortion_mode=None, **best_kw)  # per-mode override happens inside
        rows = experiment_distortion_ablation(rc, save_frames=save_frames)
        for r in rows:
            all_rows.append({'dataset': dataset, 'segment_id': seg['id'],
                             'model': model, **r})

    # ─── grouped summary table ──────────────────────────────────────────
    print("\n\n" + "="*92)
    print(f"COMBINED DISTORTION ABLATION — {dataset}, model={model}")
    print("="*92)
    print(f"{'segment':<8} {'mode':<18} {'err°/s':>8} {'med°/s':>8} "
          f"{'dir°':>7} {'β':>6} {'σβ':>6}")
    print("-"*92)
    last_seg = None
    for r in all_rows:
        if r['segment_id'] != last_seg:
            print("-"*92)
            last_seg = r['segment_id']
        print(f"{r['segment_id']:<8} {r['mode']:<18} {r['mean_err_deg_s']:>8.1f} "
              f"{r['median_err_deg_s']:>8.1f} {r['mean_dir_err_deg']:>7.1f} "
              f"{r['mean_beta']:>6.2f} {r['std_beta']:>6.2f}")

    out = os.path.join('results', 'ablation_distortion')
    os.makedirs(out, exist_ok=True)
    csv_path = os.path.join(out, f"{dataset}_ALLSEG_{model}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nCombined ablation CSV saved: {csv_path}")
    return all_rows

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
        n_frames=frame_idx + 1, clip_value=10.0, undistort=rc.undistort_at_event_level, sensor_size=rc.sensor_size,
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
            n_frames=frame_idx + 1, clip_value=10.0, undistort=rc.undistort_at_event_level, sensor_size=rc.sensor_size,
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


def _save_3col_frame(out_dir, k, V, net, H, W, gt_images, rc: RunConfig):
    """
    One inspection frame per time step: events | I | exact read-out of I |
    G (direction-coloured) | flow | ground-truth APS.

    Both intensity panels are shown because they differ and only one of them is
    scored. `Estimated I` is the in-loop map, which the iterative update of
    Eq. 6.61 leaves as an edge map; `I from G (exact)` solves Eq. 6.64 on the
    same G and is what recon_r measures. Judging a run by the first panel while
    the numbers describe the second is how a comparison goes wrong.

    Every quantity the network infers is shown, because the angular-velocity
    number alone cannot tell you whether the interpretation behind it is sound.
    G is coloured by direction rather than shown as magnitude only -- see
    grad_to_rgb.
    """
    fig, axes = plt.subplots(1, 6, figsize=(22, 4))

    axes[0].imshow(V, cmap='RdBu', vmin=-1, vmax=1)
    axes[0].set_title('Events (V)', fontsize=10)
    axes[0].axis('off')

    I_disp = net.I if net.I.shape == (H, W) else net.I[:H, :W]
    # Display range is carried across frames with a slow EMA rather than
    # recomputed per frame. Per-frame percentile normalisation removes flicker
    # but also HIDES loss of contrast: when I collapses (which it does
    # transiently at high rotation speed) a near-flat map gets stretched to full
    # black-to-white, so amplified noise reads as texture. A slowly-adapting
    # range keeps the video steady while letting a genuine collapse show up as
    # what it is -- a washed-out frame. The measured contrast is printed in the
    # title so it can be read off rather than inferred from appearance.
    lo, hi = np.percentile(I_disp, [1.0, 99.0])
    if hi - lo < 1e-9:
        lo, hi = lo - 0.5, hi + 0.5
    prev = getattr(rc, '_I_disp_range', None)
    if prev is None or k == 0:
        rng = [lo, hi]
    else:
        a = 0.05                      # slow: ~20 frames to follow a real change
        rng = [prev[0] + a * (lo - prev[0]), prev[1] + a * (hi - prev[1])]
    rc._I_disp_range = rng
    axes[1].imshow(I_disp, cmap='gray', vmin=rng[0], vmax=rng[1])
    _r_in = _corr_to_aps(np.asarray(I_disp, dtype=np.float64), gt_images,
                         rc.t_start + (k + 0.5) * rc.frame_duration)
    axes[1].set_title(f'Estimated I (in-loop)   std {I_disp.std():.3f}'
                      + ('' if not np.isfinite(_r_in) else f'   r = {_r_in:+.2f}'),
                      fontsize=10)
    axes[1].axis('off')

    # The quantity recon_r actually scores: I recovered from THIS G by the
    # exact solve of Eq. 6.64, not the in-loop iterative map beside it.
    G = net.G[:H, :W]
    try:
        from interacting_maps.network_dissertation import solve_poisson_exact
        I_ro = solve_poisson_exact(G, 'fft')
        lo_r, hi_r = np.percentile(I_ro, [1.0, 99.0])
        axes[2].imshow(I_ro, cmap='gray', vmin=lo_r, vmax=max(hi_r, lo_r + 1e-9))
        r = _corr_to_aps(I_ro, gt_images,
                         rc.t_start + (k + 0.5) * rc.frame_duration)
        axes[2].set_title('I from G (exact)'
                          + ('' if not np.isfinite(r) else f'   r = {r:+.2f}'),
                          fontsize=10)
    except Exception:
        axes[2].set_title('I from G (exact) - failed', fontsize=10)
    axes[2].axis('off')

    axes[3].imshow(grad_to_rgb(net.G))
    axes[3].set_title(r'Gradient G   hue = direction'
                      f'   (max |G| {np.abs(net.G).max():.2f})', fontsize=10)
    axes[3].axis('off')

    axes[4].imshow(flow_to_rgb(net.F))
    axes[4].set_title(r'Flow F   hue = direction'
                      f'   (|w| {np.linalg.norm(net.R)/rc.frame_duration:.2f} rad/s)',
                      fontsize=10)
    axes[4].axis('off')

    if gt_images is not None and len(gt_images) > 0:
        t_frame = rc.t_start + k * rc.frame_duration + rc.frame_duration / 2
        closest = min(gt_images, key=lambda x: abs(x[0] - t_frame))
        gt_img = plt.imread(closest[1])
        axes[5].imshow(gt_img, cmap='gray')
        axes[5].set_title(f'GT Image (APS)  dt {1000*abs(closest[0]-t_frame):.0f} ms',
                          fontsize=10)
    else:
        axes[5].text(0.5, 0.5, 'No GT', ha='center', va='center',
                     fontsize=14, transform=axes[5].transAxes)
        axes[5].set_facecolor('#f0f0f0')
        axes[5].set_title('GT (N/A)', fontsize=10)
    axes[5].axis('off')

    plt.suptitle(f'Frame {k:04d}    t = {rc.t_start + k*rc.frame_duration:.3f} s'
                 f'    {rc.model}   dFR={rc.params["delta_FR"]:.2f}'
                 + (f"  dAnchor={rc.params['delta_IMU']:.2f}"
                    if getattr(rc, 'use_imu', False) else '')
                 + f'   dt={1000*rc.frame_duration:.0f} ms  {rc.distortion_mode}',
                 fontsize=10)
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
        n_frames=frame_idx + 1, clip_value=10.0, undistort=rc.undistort_at_event_level, sensor_size=rc.sensor_size,
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
    models = ([model_filter] if model_filter
              else ['cook', 'thesis', 'thesis_imu', 'thesis_cmax', 'thesis_cmax_v2'])

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
                              all_segments=False, save_frames=True,
                              distortion_mode=None):
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
    # Trimmed per sensitivity analysis of the previous grid:
    #   - n_iters: FIXED at 75 (75 vs 100 moved error by <0.2 deg/s → inert)
    #   - frame_duration: NOW SWEPT (was silently fixed at 0.020 before; it is
    #     the physically most important untested axis — sets flow magnitude/frame)
    #   - delta_FR: [0.1, 0.2, 0.5] (inert for pure vision; matters only for imu)
    #   - delta_IMU: [0.1, 0.2, 0.5] (main knob for thesis_imu)
    grid = {
        'frame_duration': [0.010, 0.020, 0.050],
        'n_frames':  [25, 150],       # short-track accuracy + long-track drift/reversal
        'n_iters':   [75],
        'delta_FR':  [0.10, 0.20, 0.50],    # 0.9 add 
        'delta_IMU': [0.10, 0.20, 0.50],
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
    print(f"  Distortion mode: {distortion_mode or DISTORTION_MODE}")
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
                                            distortion_mode=distortion_mode,
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
                                            'distortion_mode': rc.distortion_mode,
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
                                            'distortion_mode': distortion_mode or DISTORTION_MODE,
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
                        choices=['cook', 'thesis', 'thesis_imu', 'thesis_cmax',
                                 'thesis_cmax_v2', 'all'])
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
    parser.add_argument('--save-iwe', action='store_true',
                        help='Save the final CMax IWE per frame (V1/V2 only)')
    parser.add_argument('--distortion-mode', type=str, default=None,
                        choices=['undistort_events', 'C_full'],
                        help='Override distortion handling (exp 1-6, 8). Exp 9 sweeps both.')
    parser.add_argument('--best', action='store_true',
                        help='Fill unspecified params from best_config.py '
                             '(best grid params for this dataset/segment/model). '
                             'Explicit CLI flags still win.')
    args = parser.parse_args()
    save_frames = not args.no_frames  # Default: save images

    # Build config ONLY for experiments that need it (1-6)
    
    # Resolve --model all → None (triggers all-models logic in exp 7/8)
    effective_model = None if args.model == 'all' else args.model

    # Build config ONLY for experiments that need it (1-6)
    rc = None
    need_rc = args.exp in (1, 2, 3, 4, 5, 6, 9)
    if args.exp == 9 and args.segment == 'all':
        need_rc = False                       # all-segments path builds its own configs
    if need_rc:
        dataset = args.dataset or 'boxes_rotation'
        model = args.model if args.model != 'all' else 'thesis_imu'
        rc_kwargs = dict(
            dataset=dataset,
            model=model,
            segment=args.segment,
            n_frames=args.n_frames,
            n_iters=args.n_iters,
            delta_IMU=args.delta_imu,
            distortion_mode=args.distortion_mode,
        )
        if args.best:
            if args.segment in (None, 'all'):
                print("  [--best] requires a specific --segment (e.g. seg_A); "
                      "ignoring --best.")
            else:
                cli = {'n_frames': args.n_frames, 'n_iters': args.n_iters,
                       'delta_IMU': args.delta_imu}
                rc_kwargs.update(resolve_best_kwargs(dataset, args.segment, model, cli))
        rc = RunConfig(**rc_kwargs)
        rc.save_iwe = args.save_iwe

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
            distortion_mode=args.distortion_mode,
        ),
        9: lambda: (experiment_distortion_ablation_all_segments(
                        dataset=args.dataset or 'boxes_rotation',
                        model=(args.model if args.model != 'all' else 'thesis_imu'),
                        save_frames=save_frames, use_best=args.best)
                    if args.segment == 'all'
                    else experiment_distortion_ablation(rc, save_frames=save_frames)),
        }

    experiments[args.exp]()