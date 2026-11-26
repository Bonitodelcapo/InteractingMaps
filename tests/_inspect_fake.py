"""Fake inputs for the viewer tests, fast and independent of data/:
a tiny dataset (a few thousand events, calib, IMU, Vicon, two APS images) and
inspect run folders with synthetic states and a synthetic tracking.csv.
No model is run and no real event file is scanned."""
import csv
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from interacting_maps import states_io

H, W, NF = 180, 240, 3
DATASET = 'tinyset'
T0 = 1.0


def make_dataset(data_root, name=DATASET, seed=0):
    d = os.path.join(data_root, name)
    os.makedirs(os.path.join(d, 'images'), exist_ok=True)
    rng = np.random.default_rng(seed)
    with open(os.path.join(d, 'calib.txt'), 'w') as f:
        f.write('199.0 198.8 132.2 110.7 0 0 0 0 0\n')
    n = 3000
    t = np.sort(rng.uniform(0.0, T0 + 1.0, n))
    ev = np.column_stack([t, rng.integers(0, W, n), rng.integers(0, H, n), rng.integers(0, 2, n)])
    np.savetxt(os.path.join(d, 'events.txt'), ev, fmt=['%.9f', '%d', '%d', '%d'])
    ti = np.arange(0.0, T0 + 1.0, 0.001)
    imu = np.column_stack([ti, np.zeros((len(ti), 3)), np.tile([0.1, -0.2, 0.3], (len(ti), 1))])
    np.savetxt(os.path.join(d, 'imu.txt'), imu, fmt='%.9f')
    tg = np.arange(0.0, T0 + 1.0, 0.005)
    gt = np.column_stack([tg, np.zeros((len(tg), 3)), np.tile([0, 0, 0, 1.0], (len(tg), 1))])
    np.savetxt(os.path.join(d, 'groundtruth.txt'), gt, fmt='%.9f')
    with open(os.path.join(d, 'images.txt'), 'w') as f:
        for i, ta in enumerate((T0 + 0.01, T0 + 0.05)):
            plt.imsave(os.path.join(d, 'images', f'frame_{i:08d}.png'),
                       rng.uniform(0, 1, (H, W)), cmap='gray')
            f.write(f'{ta:.9f} images/frame_{i:08d}.png\n')
    return data_root


def make_run(root, model, dataset=DATASET, t_start=T0, scale=1.0, seed=0, n_frames=NF,
             folder=None, deltas=None):
    d = os.path.join(root, dataset, model, folder or f'seg_C_t{t_start:.3f}_fake')
    rng = np.random.default_rng(seed)
    cfg = {'dataset': dataset, 'model': model, 'segment_id': 'seg_C', 't_start': t_start,
           'frame_duration': 0.02, 'n_frames': n_frames, 'n_iters': 10,
           'distortion_mode': 'C_full', 'poisson': 'iterative', 'out_root': root,
           'sensor_size': [H, W], 'initial_R': [0.0, 0.0, 0.0], 'params': dict(deltas or {})}
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'run.json'), 'w') as f:
        json.dump({'provenance': {}, 'config': cfg, 'summary': {}}, f)
    with open(os.path.join(d, 'tracking.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame', 'time', 'err_deg_s', 'beta', 'ref_source'])
        for k in range(n_frames):
            w.writerow([k, t_start + (k + 0.5) * 0.02, 5.0 + k, 1.0, 'groundtruth'])
    ih = H + 1 if model == 'cook' else H
    iw = W + 1 if model == 'cook' else W
    for k in range(-1, n_frames):
        st = {'I': rng.standard_normal((ih, iw)) * scale,
              'G': rng.standard_normal((H, W, 2)) * scale,
              'F': rng.standard_normal((H, W, 2)) * scale,
              'R': rng.standard_normal(3) * 0.01}
        states_io.save_state(d, k, st, {})
    return d
