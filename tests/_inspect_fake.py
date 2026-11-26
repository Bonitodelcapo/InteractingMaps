"""A fake inspect run folder for the viewer tests: real run.json config of
poster_rotation seg_C (so the raw data can be found), synthetic states and a
synthetic tracking.csv. No model is run."""
import csv
import json
import os

import numpy as np

from interacting_maps import states_io

H, W, NF = 180, 240, 3


def make_run(root, model, dataset='poster_rotation', t_start=8.816, scale=1.0, seed=0):
    d = os.path.join(root, dataset, model, f'seg_C_t{t_start:.3f}_fake')
    rng = np.random.default_rng(seed)
    cfg = {'dataset': dataset, 'model': model, 'segment_id': 'seg_C', 't_start': t_start,
           'frame_duration': 0.02, 'n_frames': NF, 'n_iters': 10,
           'distortion_mode': 'C_full', 'poisson': 'iterative', 'out_root': root,
           'sensor_size': [H, W], 'initial_R': [0.0, 0.0, 0.0], 'params': {}}
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'run.json'), 'w') as f:
        json.dump({'provenance': {}, 'config': cfg, 'summary': {}}, f)
    with open(os.path.join(d, 'tracking.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame', 'time', 'err_deg_s', 'beta', 'ref_source'])
        for k in range(NF):
            w.writerow([k, t_start + (k + 0.5) * 0.02, 5.0 + k, 1.0, 'groundtruth'])
    ih = H + 1 if model == 'cook' else H
    iw = W + 1 if model == 'cook' else W
    for k in range(-1, NF):
        st = {'I': rng.standard_normal((ih, iw)) * scale,
              'G': rng.standard_normal((H, W, 2)) * scale,
              'F': rng.standard_normal((H, W, 2)) * scale,
              'R': rng.standard_normal(3) * 0.01}
        states_io.save_state(d, k, st, {})
    return d
