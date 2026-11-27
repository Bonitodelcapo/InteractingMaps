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
  2. Multi-frame tracking (ω_est vs ω_ref over time; per-frame panels
     V | I | I read-out | G | F | APS unless --no-frames)   ← the core; 9 calls it
  5. Assemble MP4s from saved frames (batch)
  6. Basin of attraction (how far can init-R be from GT and still converge?)
  9. Distortion ablation (Way-1 undistort-events vs Way-2 distortion-aware C/warp)

Every other study, and every run behind the report, is a command of
report/run_experiments.py (see report/run_all.sh): --what converge replaces the
old single-frame convergence, --what sweep the parameter influence and grid,
--what main the full evaluation.

Usage:
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

from config import DATASET_SEGMENTS, get_dataset_paths
from interacting_maps.data_loader import EventFrameSequence
import experiment_tracking as run_store

# Helpers extracted into focused modules (R3/R5 of architecture_review.md) and
# re-exported here so existing callers (incl. report scripts' E.<name>) are
# unchanged. See eval_io.py / metrics.py / viz.py / run_config.py.
from interacting_maps.eval_io import (SCORE_AGAINST, GT_DIFF_HALFWIDTH, load_imu,
                     get_gyro_for_frame, load_groundtruth, _quat_to_rotmat,
                     gt_omega_body, load_omega_gt, omega_gt_at,
                     get_reference_omega)
from interacting_maps.metrics import (compute_metrics, curl_share, _intensity_stats,
                     _corr_to_aps, _recon_scores)
from interacting_maps.viz import (draw_map, draw_omega, draw_error, net_maps, map_scale,
                                  draw_state, run_figures)
from interacting_maps.style import (MODELS, SCREEN_WIDTH, frame_times,
                                    to_deg_s, ref_label, use_style, save)
from run_config import (RunConfig, make_network,
                        _try_load_gt_images)

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
      - video_frames/frame_XXXX.png (V | I | I read-out | G | F | APS)
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
        from interacting_maps.data_loader import load_events_fast, undistort_events
        dur = rc.n_frames * rc.frame_duration + 0.1
        cmax_events = undistort_events(
            load_events_fast(rc.paths['events'], t_start=rc.t_start, duration=dur),
            seq.calib)
        if getattr(rc, 'use_cmax_v2', False):
            print(f"  CMax V2 active — R driven by 1 CMax step / iteration "
                  f"(lr={rc.cmax_lr}, {len(cmax_events)} events).")
        else:
            # V1: standalone full-solve estimator, feeds the R anchor.
            from interacting_maps.angular_velocity import CMaxAngularVelocity
            cmax_est = CMaxAngularVelocity(H, W, fx, fy, cx, cy, use_polarity=True)
            print(f"  CMax V1 active — R anchored to full CMax ω per frame "
                  f"({len(cmax_events)} events).")

    # ─── IWE logging (final contrast-maximized IWE per frame) ─────────
    iwe_dir = None
    if getattr(rc, 'save_iwe', False) and getattr(rc, 'use_cmax', False):
        from interacting_maps.iwe_io import save_iwe, plot_contrast_curve, make_gif
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
            w_deg = to_deg_s(omega_est)
            print(f"  Frame {k+1:4d}/{rc.n_frames} | "
                  f"ω_est=({w_deg[0]:+.1f},{w_deg[1]:+.1f},{w_deg[2]:+.1f})°/s | "
                  f"err={err:.1f}°/s | dir={dir_err:.1f}°")

    # ─── Save CSV ─────────────────────────────────────────────────────
    csv_path = os.path.join(rc.output_dir, 'tracking.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # ─── IWE sequence artefacts (contrast curve + GIF) ────────────────
    if iwe_dir is not None:
        plot_contrast_curve(iwe_dir, rc.t_start)
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
        run_store.mlflow_log_run(rc.to_dict(), summary, output_dir=rc.output_dir)

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
    run_figures(rc.output_dir, net_maps(net, H, W), rows, gt_images, rc.to_dict())
    return summary

def _plot_tracking(rows, rc: RunConfig):
    """ω components and error against seconds since segment start."""
    t = frame_times(rc.frame_duration, len(rows))
    col = lambda p: np.array([[r[f'{p}_wx'], r[f'{p}_wy'], r[f'{p}_wz']] for r in rows])
    est, ref, imu = col('est'), col('gt'), col('imu')
    ref_src = rows[0]['ref_source']

    with use_style('screen'):
        fig, axes = plt.subplots(4, 1, figsize=(SCREEN_WIDTH, 10), sharex=True)
        fig.suptitle(f'ω tracking — {MODELS[rc.model][0]}   [{rc.dataset} {rc.segment_id}, '
                     f't_start {rc.t_start:.3f} s]\n'
                     f'dt {rc.frame_duration*1000:.0f} ms, {rc.n_iters} iters, '
                     f'{rc.duration_s:.1f} s, scored vs {ref_label(ref_src)}')
        for i in range(3):
            draw_omega(axes[i], t, est[:, i], ref=ref[:, i], imu=imu[:, i], axis=i,
                       ref_src=ref_src)
        axes[0].legend(loc='upper right')
        err = np.array([r['err_deg_s'] for r in rows])
        draw_error(axes[3], t, err)
        axes[3].set_title(f'error: mean {err.mean():.1f} °/s, median {np.median(err):.1f} °/s')
        for ax in axes:
            ax.label_outer()
        save(fig, os.path.join(rc.output_dir, 'tracking_plot.png'))

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

def experiment_distortion_ablation_all_segments(dataset, model, save_frames=False):
    """
    Run the distortion ablation over EVERY segment of a dataset and emit one
    combined CSV (dataset × segment × mode) plus a grouped summary table.
    """
    print("\n" + "="*80)
    print(f"EXPERIMENT 9 (ALL SEGMENTS): Distortion Ablation — {dataset}, model={model}")
    print("="*80)

    segs = DATASET_SEGMENTS[dataset]
    all_rows = []
    for seg in segs:
        rc = RunConfig(dataset=dataset, model=model, segment=seg['id'],
                       distortion_mode=None)  # per-mode override happens inside
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

def _steady_scale(rc, kind, arr, k, a=0.05):
    """Per-frame scale smoothed across frames (EMA), so a fading map shows as fading."""
    s = map_scale(kind, arr)
    store = rc.__dict__.setdefault('_disp_scale', {})
    prev = None if k == 0 else store.get(kind)
    if prev is None:
        blended = s
    else:
        blended = np.asarray(prev) + a * (np.asarray(s) - np.asarray(prev))
        blended = tuple(blended) if kind == 'I' else float(blended)
    store[kind] = blended
    return blended


def _save_3col_frame(out_dir, k, V, net, H, W, gt_images, rc: RunConfig):
    """
    One inspection frame per time step: V | in-loop I | I read out from G by
    the exact solve (what recon_r scores) | G | F | nearest APS frame.

    V, in-loop I, G and F keep a display scale smoothed across frames (EMA),
    so a map that loses contrast shows washed out instead of being
    re-stretched; I's std and correlation with APS are in the title. "I from
    G (exact)" and APS keep per-image scaling.
    """
    from interacting_maps.network_dissertation import solve_poisson_exact
    maps = net_maps(net, H, W)
    I, G, F = maps['I'], maps['G'], maps['F']
    t_rel = (k + 0.5) * rc.frame_duration
    t_mid = rc.t_start + t_rel

    v_scale = _steady_scale(rc, 'V', V, k)
    rng = _steady_scale(rc, 'I', I, k)
    g_scale = _steady_scale(rc, 'G', G, k)
    f_scale = _steady_scale(rc, 'F', F, k)
    I_ro = solve_poisson_exact(G, 'fft')
    r_in = _corr_to_aps(np.asarray(I, dtype=np.float64), gt_images, t_mid)
    r_ro = _corr_to_aps(I_ro, gt_images, t_mid)
    w = to_deg_s(np.linalg.norm(net.R) / rc.frame_duration)
    aps = aps_dt = None
    if gt_images:
        t_aps, path = min(gt_images, key=lambda x: abs(x[0] - t_mid))
        aps, aps_dt = plt.imread(path), abs(t_aps - t_mid)

    with use_style('screen'):
        fig, axes = plt.subplots(1, 6, figsize=(SCREEN_WIDTH, 2.6))
        draw_map(axes[0], 'V', V, scale=v_scale)
        draw_state(axes[1:], I, I_ro, G, F, r_in, r_ro, w, aps, aps_dt,
                   scale_I=rng, scale_G=g_scale, scale_F=f_scale)
        anchor = f'   δ_anchor {rc.params["delta_IMU"]:.2f}' if rc.use_imu else ''
        fig.suptitle(f'{MODELS[rc.model][0]}   frame {k:04d}   t {t_rel:.3f} s '
                     f'(segment start {rc.t_start:.3f} s)   δ_FR {rc.params["delta_FR"]:.2f}'
                     f'{anchor}   dt {1000 * rc.frame_duration:.0f} ms   {rc.distortion_mode}')
        save(fig, os.path.join(out_dir, f'frame_{k:04d}.png'))


# ===========================================================================
# EXPERIMENT 5: Make Videos (batch)
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

    # Plot: distances as angular-velocity error in deg/s; filled = converged
    color = MODELS[rc.model][1]
    to_w = lambda d: to_deg_s(d / rc.frame_duration)
    with use_style('screen'):
        fig, ax = plt.subplots(figsize=(8, 6))
        for r in results:
            x, y = to_w(r['init_dist']), to_w(r['final_dist'])
            ax.scatter(x, y, s=80, zorder=5, color=color if r['converged'] else 'none',
                       edgecolors=color, linewidths=1.5)
            ax.annotate(r['label'], (x, y), fontsize=8)
        max_d = to_w(max(r['init_dist'] for r in results)) * 1.1
        ax.plot([0, max_d], [0, max_d], ls='--', color='0.5', label='no improvement')
        ax.set_xlabel(r'initial $\|\omega - \omega_{ref}\|$ (°/s)')
        ax.set_ylabel(r'final $\|\omega - \omega_{ref}\|$ (°/s)')
        ax.set_title(f'Exp 6: basin of attraction — {MODELS[rc.model][0]}   '
                     f'[{rc.dataset} {rc.segment_id}, t_start {rc.t_start:.3f} s, '
                     f'{ref_label("imu")}; filled = converged]')
        ax.legend()
        save(fig, os.path.join(rc.output_dir, 'exp6_basin.png'))


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Systematic Evaluation')
    parser.add_argument('--exp', type=int, default=2, choices=[2, 5, 6, 9],
                        help='2 tracking, 5 videos, 6 basin, 9 distortion ablation')
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
    parser.add_argument('--no-frames', action='store_true',
                        help='Skip saving 3-col PNGs (faster)')
    parser.add_argument('--save-iwe', action='store_true',
                        help='Save the final CMax IWE per frame (V1/V2 only)')
    parser.add_argument('--distortion-mode', type=str, default=None,
                        choices=['undistort_events', 'C_full'],
                        help='Override distortion handling (exp 2, 6). Exp 9 sweeps both.')
    parser.add_argument('--out-root', type=str, default=None,
                        help="Run folder root (default 'results'); e.g. experiments/<name> for exploratory runs")
    args = parser.parse_args()
    save_frames = not args.no_frames  # Default: save images

    # Build the run config for the experiments that need one
    rc = None
    need_rc = args.exp in (2, 6, 9)
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
            out_root=args.out_root,
        )
        rc = RunConfig(**rc_kwargs)
        rc.save_iwe = args.save_iwe

    experiments = {
        2: lambda: experiment_tracking(rc, save_frames=save_frames),
        5: lambda: experiment_make_videos_batch(fps=args.fps),
        6: lambda: experiment_basin_of_attraction(rc, frame_idx=args.frame),
        9: lambda: (experiment_distortion_ablation_all_segments(
                        dataset=args.dataset or 'boxes_rotation',
                        model=(args.model if args.model != 'all' else 'thesis_imu'),
                        save_frames=save_frames)
                    if args.segment == 'all'
                    else experiment_distortion_ablation(rc, save_frames=save_frames)),
        }

    experiments[args.exp]()