"""
Test the effect of `clip_value` and `normalise` in events_to_vframe()
on validation accuracy.

We sweep two axes:

  AXIS 1 - clip_value:
      Bound applied to the signed event count BEFORE optional normalisation.
      Small clip  -> many pixels saturate, V is sparse and low-magnitude.
      Large clip  -> no saturation, V follows the heavy-tailed raw count.

  AXIS 2 - normalise:
      If True : V /= clip_value  ->  V in [-1, +1].
      If False: V kept as raw counts (or clipped counts).

For each combination we measure:
  - V statistics (mean |V|, max |V|, % of pixels at saturation)
  - Mean angular error vs GT (using RESEED_R_FROM_GT mode for fairness)
  - Mean |omega_est|  (to expose scale collapse)

Uses the same parametrised step() infrastructure as test_clipping_effect.py
(stability clips left at their defaults — we showed they are inert).
"""

import os
import io
import contextlib
import numpy as np

from config import DATASET_CONFIGS, THESIS_PARAMS, ITERS_PER_FRAME, F_DECAY, G_DECAY, get_dataset_paths
from data_loader import load_events_fast, events_to_vframe, CameraCalibration
from interacting_maps.network_dissertation import InteractingMapsThesis

from validation import gt_omega_body, load_groundtruth


DATASET = 'poster_rotation'
cfg     = DATASET_CONFIGS[DATASET]
paths   = get_dataset_paths(DATASET)

T_START        = cfg['t_start']
FRAME_DURATION = cfg['frame_duration']
N_FRAMES       = cfg['n_frames']
initial_R      = cfg['initial_R']

# Sensor size (DAVIS240C, same as data_loader hardcodes)
H, W = 180, 240


# ---------------------------------------------------------------------------
# Pre-load events ONCE so every run uses the same data
# ---------------------------------------------------------------------------

print(f"Pre-loading events for {DATASET} ...")
t_lo_global = T_START
t_hi_global = T_START + N_FRAMES * FRAME_DURATION + 0.1
with contextlib.redirect_stdout(io.StringIO()):
    EVENTS = load_events_fast(paths['events'],
                              t_start=t_lo_global,
                              duration=t_hi_global - t_lo_global)
print(f"  loaded {len(EVENTS)} events over {t_hi_global - t_lo_global:.3f} s")

calib = CameraCalibration(paths['calib'])
fx, fy, cx, cy = calib.fx, calib.fy, calib.cx, calib.cy

gt_data = load_groundtruth(paths['groundtruth'])


# ---------------------------------------------------------------------------
# Build the per-frame V the way the production loader would, but with
# parametrised clip_value and normalise so we can sweep them.
# ---------------------------------------------------------------------------

def make_frame(k, clip_value, normalise):
    t_lo = T_START + k * FRAME_DURATION
    t_hi = t_lo + FRAME_DURATION
    mask = (EVENTS[:, 0] >= t_lo) & (EVENTS[:, 0] < t_hi)
    V = events_to_vframe(EVENTS[mask], H, W,
                        x_offset=0, y_offset=0,
                        clip_value=clip_value, normalise=normalise)
    return V, (t_lo + t_hi) / 2.0


# ---------------------------------------------------------------------------
# One run = one (clip_value, normalise) combination
# ---------------------------------------------------------------------------

def run_one(clip_value, normalise):
    net = InteractingMapsThesis(H=H, W=W, fx=fx, fy=fy, cx=cx, cy=cy, **THESIS_PARAMS)
    net.initialize_from_rotation(initial_R)

    err_list = []
    R_norm_list = []
    V_abs_means = []
    V_abs_maxes = []
    V_sat_pcts = []

    for k in range(N_FRAMES):
        V, t_mid = make_frame(k, clip_value, normalise)

        # V stats: max abs value, mean abs value, % of pixels at saturation
        v_abs = np.abs(V)
        V_abs_means.append(float(v_abs.mean()))
        V_abs_maxes.append(float(v_abs.max()))
        # saturation threshold for the "is this pixel saturated?" check
        sat_thr = 1.0 if normalise else clip_value
        # only count saturation on non-zero pixels (zero events doesn't count)
        n_evt_pixels = int(np.sum(v_abs > 0))
        n_sat = int(np.sum(v_abs >= sat_thr * 0.999))
        V_sat_pcts.append(100.0 * n_sat / max(n_evt_pixels, 1))

        # RESEED R from GT for fairness
        omega_ref = gt_omega_body(gt_data, t_mid, FRAME_DURATION)
        net.q_R.value = (omega_ref * FRAME_DURATION).copy()

        net.step(V, n_iters=ITERS_PER_FRAME, f_decay=F_DECAY, g_decay=G_DECAY)

        omega_est = net.R / FRAME_DURATION
        err_list.append(float(np.degrees(np.linalg.norm(omega_est - omega_ref))))
        R_norm_list.append(float(np.linalg.norm(omega_est)))

    return dict(
        mean_err=float(np.mean(err_list)),
        med_err=float(np.median(err_list)),
        mean_R=float(np.mean(R_norm_list)),
        mean_V_abs=float(np.mean(V_abs_means)),
        max_V_abs=float(np.max(V_abs_maxes)),
        mean_sat_pct=float(np.mean(V_sat_pcts)),
    )


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

print(f"\nGT mean |omega|         : "
      f"{np.mean([np.linalg.norm(gt_omega_body(gt_data, T_START + (k+0.5)*FRAME_DURATION, FRAME_DURATION)) for k in range(N_FRAMES)]):.3f} rad/s")
print(f"Expected omega (config) : {np.linalg.norm(cfg['expected_omega']):.3f} rad/s")

clip_grid = [1.0, 3.0, 10.0, 30.0, 100.0, 1e9]

print("\n" + "="*92)
print(f"AXIS 1 + 2 sweep on {DATASET}  (RESEED_R_FROM_GT, "
      f"{ITERS_PER_FRAME} iters/frame, {N_FRAMES} frames)")
print("="*92)
print(f"{'clip':>8} {'norm':>5} | "
      f"{'mean|V|':>9} {'max|V|':>8} {'sat%':>6} | "
      f"{'mean err':>9} {'med err':>9} | {'mean |R_est|':>13}")
print(f"{'':>8} {'':>5} | "
      f"{'':>9} {'':>8} {'':>6} | "
      f"{'(deg/s)':>9} {'(deg/s)':>9} | {'(rad/s)':>13}")
print("-"*92)

results = []
for normalise in (True, False):
    for cv in clip_grid:
        r = run_one(cv, normalise)
        r['clip_value'] = cv
        r['normalise']  = normalise
        results.append(r)
        cv_str = "inf" if cv >= 1e8 else f"{cv:g}"
        print(f"{cv_str:>8} {str(normalise):>5} | "
              f"{r['mean_V_abs']:9.4f} {r['max_V_abs']:8.3f} {r['mean_sat_pct']:6.2f} | "
              f"{r['mean_err']:9.3f} {r['med_err']:9.3f} | {r['mean_R']:13.4f}")
    print("-"*92)


# ---------------------------------------------------------------------------
# Summary / interpretation hints
# ---------------------------------------------------------------------------

# Find best and worst
results_sorted = sorted(results, key=lambda r: r['mean_err'])
best = results_sorted[0]
worst = results_sorted[-1]

print("\nSUMMARY")
print("-"*92)
print(f"  Best : clip={best['clip_value']:g}  norm={best['normalise']}  "
      f"-> mean err {best['mean_err']:.3f} deg/s   |R_est|={best['mean_R']:.4f}")
print(f"  Worst: clip={worst['clip_value']:g}  norm={worst['normalise']}  "
      f"-> mean err {worst['mean_err']:.3f} deg/s   |R_est|={worst['mean_R']:.4f}")

# Effect of normalisation at the production clip_value=10
prod_n = next(r for r in results if r['normalise'] and r['clip_value'] == 10.0)
prod_r = next(r for r in results if not r['normalise'] and r['clip_value'] == 10.0)
print(f"\n  At validation's clip_value=10:")
print(f"    normalised (current default) : mean err {prod_n['mean_err']:.3f} deg/s   |R_est|={prod_n['mean_R']:.4f}")
print(f"    NOT normalised               : mean err {prod_r['mean_err']:.3f} deg/s   |R_est|={prod_r['mean_R']:.4f}")
print(f"    delta                        : {prod_r['mean_err'] - prod_n['mean_err']:+.3f} deg/s")
