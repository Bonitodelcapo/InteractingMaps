"""
End-to-end pipeline check for the UNDISTORT_MODE flag.

Runs four combinations, all with RESEED_R_FROM_GT=True (per-frame oracle,
matches the harness we've used for F2/F3):

    poster_rotation   x  UNDISTORT_MODE=False  (baseline)
    poster_rotation   x  UNDISTORT_MODE=True   (distortion-aware C_mat)
    synthetic_xrot    x  UNDISTORT_MODE=False  (baseline)
    synthetic_xrot    x  UNDISTORT_MODE=True   (must be identical: k_i=p_i=0)

Reports:
  - mean / median angular error vs GT
  - mean |R_est|
Confirms:
  - validation.py's data loading + step loop still runs after the changes
  - dist_coeffs wiring reaches build_kinematic_matrix correctly
  - synthetic (zero-dist) is a no-op invariant under the flag
"""

import io, contextlib
import numpy as np

from config import (DATASET_CONFIGS, THESIS_PARAMS, ITERS_PER_FRAME,
                    F_DECAY, G_DECAY, get_dataset_paths)
from data_loader import EventFrameSequence
from interacting_maps.network_dissertation import InteractingMapsThesis
from validation import gt_omega_body, load_groundtruth


def run(dataset, undistort_mode):
    cfg   = DATASET_CONFIGS[dataset]
    paths = get_dataset_paths(dataset)

    with contextlib.redirect_stdout(io.StringIO()):
        seq = EventFrameSequence(
            paths['events'], paths['calib'],
            frame_duration=cfg['frame_duration'],
            t_start=cfg['t_start'],
            n_frames=cfg['n_frames'],
            clip_value=10.0,
        )
        gt = np.loadtxt(paths['groundtruth'], dtype=np.float64)

    dist_coeffs = seq.calib.dist if undistort_mode else None

    net = InteractingMapsThesis(
        H=seq.H, W=seq.W,
        fx=seq.calib.fx, fy=seq.calib.fy,
        cx=seq.calib.cx, cy=seq.calib.cy,
        dist_coeffs=dist_coeffs,
        **THESIS_PARAMS,
    )
    net.initialize_from_rotation(cfg['initial_R'])

    errs, R_norms = [], []
    for V, t_mid in seq:
        omega_ref = gt_omega_body(gt, t_mid, cfg['frame_duration'])
        net.q_R.value = (omega_ref * cfg['frame_duration']).copy()   # RESEED

        net.step(V, n_iters=ITERS_PER_FRAME,
                 f_decay=F_DECAY, g_decay=G_DECAY)

        omega_est = net.R / cfg['frame_duration']
        errs.append(float(np.degrees(np.linalg.norm(omega_est - omega_ref))))
        R_norms.append(float(np.linalg.norm(omega_est)))

    return dict(
        mean_err = float(np.mean(errs)),
        med_err  = float(np.median(errs)),
        mean_R   = float(np.mean(R_norms)),
        C_mat_norm = float(np.linalg.norm(net._C_mat)),
    )


rows = []
for dataset in ('poster_rotation', 'synthetic_xrot'):
    for undistort in (False, True):
        r = run(dataset, undistort)
        rows.append(dict(dataset=dataset, undistort=undistort, **r))
        print(f"  {dataset:16s} undistort={str(undistort):5}  "
              f"mean_err={r['mean_err']:7.3f}  med_err={r['med_err']:7.3f}  "
              f"|R_est|={r['mean_R']:.4f}  ||C_mat||={r['C_mat_norm']:.2f}")


print("\n" + "="*78)
print("SUMMARY")
print("="*78)

# poster comparison
p_off = next(r for r in rows if r['dataset'] == 'poster_rotation' and not r['undistort'])
p_on  = next(r for r in rows if r['dataset'] == 'poster_rotation' and     r['undistort'])
print(f"\nposter_rotation:")
print(f"  UNDISTORT_MODE=False : mean err {p_off['mean_err']:.3f} deg/s   "
      f"|R_est|={p_off['mean_R']:.4f}   ||C||={p_off['C_mat_norm']:.2f}")
print(f"  UNDISTORT_MODE=True  : mean err {p_on ['mean_err']:.3f} deg/s   "
      f"|R_est|={p_on ['mean_R']:.4f}   ||C||={p_on ['C_mat_norm']:.2f}")
print(f"  DELTA                : {p_on['mean_err'] - p_off['mean_err']:+.3f} deg/s")
print(f"  C_mat changed?       : {'YES' if abs(p_on['C_mat_norm']-p_off['C_mat_norm']) > 1e-6 else 'NO'}")

# synthetic invariant check
s_off = next(r for r in rows if r['dataset'] == 'synthetic_xrot' and not r['undistort'])
s_on  = next(r for r in rows if r['dataset'] == 'synthetic_xrot' and     r['undistort'])
print(f"\nsynthetic_xrot (zero-distortion invariant):")
print(f"  UNDISTORT_MODE=False : mean err {s_off['mean_err']:.3f} deg/s   "
      f"|R_est|={s_off['mean_R']:.4f}")
print(f"  UNDISTORT_MODE=True  : mean err {s_on ['mean_err']:.3f} deg/s   "
      f"|R_est|={s_on ['mean_R']:.4f}")
print(f"  DELTA                : {s_on['mean_err'] - s_off['mean_err']:+.3f} deg/s   "
      f"({'PASS: no-op invariant holds' if abs(s_on['mean_err']-s_off['mean_err']) < 1e-6 else 'FAIL: should be exactly 0'})")
