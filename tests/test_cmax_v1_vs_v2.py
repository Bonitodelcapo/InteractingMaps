"""
V1 vs V2 side-by-side over a small number of frames.

  V1 = thesis_cmax      (full CMax solve/frame → R anchor; kinematics ALSO on R)
  V2 = thesis_cmax_v2   (1 CMax step/iter drives R; kinematics on F only)

Both are IMU-free in-loop (gyro only for frame-0 init). Scored three ways:
  - GT(20ms)   : raw quaternion differencing (noisy at short windows)
  - GT(smooth) : quaternion differencing over ±50 ms (fair yardstick)
  - IMU        : gyro (independent cross-check)

Usage:
    python test_cmax_v1_vs_v2.py                 # default: poster seg_C, 15 frames
    python test_cmax_v1_vs_v2.py 25              # 25 frames
"""

import sys
import os
import csv
import numpy as np

sys.argv_backup = sys.argv[:]
N_FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 15
sys.argv = ['x']
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import evaluation as E

DATASET = 'poster_rotation'
SEG = {'id': 'seg_C', 't_start': 8.816, 'frame_duration': 0.02,
       'n_frames': N_FRAMES, 'initial_R': None, 'sensor_size': (180, 240)}
T0, DT = SEG['t_start'], SEG['frame_duration']

paths = E.get_dataset_paths(DATASET)
gt  = E.load_groundtruth(paths['groundtruth'])
imu = E.load_imu(paths['imu'])


def run(model):
    rc = E.RunConfig(dataset=DATASET, model=model, segment=dict(SEG), n_frames=N_FRAMES)
    E.experiment_tracking(rc, save_frames=False)
    rows = list(csv.DictReader(open(os.path.join(rc.output_dir, 'tracking.csv'))))
    out = []
    for k, r in enumerate(rows):
        we = np.array([float(r['est_wx']), float(r['est_wy']), float(r['est_wz'])])
        out.append(we)
    return np.array(out)


print(f"\nRunning V1 and V2 on {DATASET}/{SEG['id']}, {N_FRAMES} frames …")
est_v1 = run('thesis_cmax')
est_v2 = run('thesis_cmax_v2')

# references per frame
gt20, gts, wimu = [], [], []
for k in range(N_FRAMES):
    tlo = T0 + k * DT; thi = tlo + DT; tm = 0.5 * (tlo + thi)
    gt20.append(E.gt_omega_body(gt, tlo, thi))
    gts.append(E.gt_omega_body(gt, tm - 0.05, tm + 0.05))
    wimu.append(E.get_gyro_for_frame(imu, tlo, thi))
gt20 = np.array(gt20); gts = np.array(gts); wimu = np.array(wimu)


def err(a, b):  # deg/s
    return np.degrees(np.linalg.norm(a - b, axis=1))


print(f"\n{'k':>3} | {'GT ωx':>7} {'ωy':>7} {'ωz':>7} | "
      f"{'V1 ωx':>7} {'ωy':>7} {'ωz':>7} | {'V2 ωx':>7} {'ωy':>7} {'ωz':>7} | "
      f"{'e_V1':>5} {'e_V2':>5}")
print("-" * 104)
eV1 = err(est_v1, gts); eV2 = err(est_v2, gts)
for k in range(N_FRAMES):
    print(f"{k:3d} | {gts[k,0]:+7.3f} {gts[k,1]:+7.3f} {gts[k,2]:+7.3f} | "
          f"{est_v1[k,0]:+7.3f} {est_v1[k,1]:+7.3f} {est_v1[k,2]:+7.3f} | "
          f"{est_v2[k,0]:+7.3f} {est_v2[k,1]:+7.3f} {est_v2[k,2]:+7.3f} | "
          f"{eV1[k]:5.1f} {eV2[k]:5.1f}")


def summary(name, est):
    print(f"  {name:4s}  vs GT(smooth) mean {err(est,gts).mean():6.2f} med {np.median(err(est,gts)):6.2f}  |  "
          f"vs IMU {err(est,wimu).mean():6.2f}  |  vs GT(20ms) {err(est,gt20).mean():6.2f}")


print("\n" + "=" * 92)
print("SUMMARY  (mean/median deg/s)")
print("=" * 92)
summary('V1', est_v1)
summary('V2', est_v2)
