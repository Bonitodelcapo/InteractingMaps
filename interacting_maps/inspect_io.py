"""Read one run folder for the inspection viewer, lazily.

A RunView gives the run's config and per-frame curves (tracking.csv) at once,
and the heavy parts only when asked: the saved state at frame k, the replayed
iterations of frame k, the raw data of frame k's window (event frame V, nearest
APS image), events per frame and the full-rate IMU / reference |omega|.
The raw parts come back empty (None / NaN / {}) when the dataset is not
present locally, so a copied-back run folder still shows its maps and curves.
"""
import csv
import os

import numpy as np

import experiment_tracking
from config import get_dataset_paths
from interacting_maps import states_io
from interacting_maps.style import frame_times

#: fields compared runs must share
SHARED = ('dataset', 'segment_id', 't_start', 'frame_duration', 'n_frames')
MAX_RUNS = 5
#: frames of states kept in memory per run
CACHE = 8


def _numeric_columns(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    out = {}
    for key in rows[0]:
        try:
            out[key] = np.array([float(r[key]) if r[key] != '' else np.nan for r in rows])
        except ValueError:
            continue                       # text column (ref_source)
    return out


class RunView:
    def __init__(self, run_dir, data_root=None):
        if not states_io.has_states(run_dir):
            states_io.load_state(run_dir, -1)          # raises with the hint
        self.dir = run_dir
        self.config = experiment_tracking.read_run(run_dir)['config']
        c = self.config
        self.label = c['model']
        self.n_frames, self.n_iters = int(c['n_frames']), int(c['n_iters'])
        self.dt, self.t_start = float(c['frame_duration']), float(c['t_start'])
        self.H, self.W = (int(x) for x in c['sensor_size'])
        self.times = frame_times(self.dt, self.n_frames)
        self.rows = _numeric_columns(os.path.join(run_dir, 'tracking.csv'))
        self.paths = get_dataset_paths(c['dataset'], base_dir=data_root)
        self._states, self._V, self._aps, self._curves = {}, None, None, None

    # ------------------------------------------------------------ model
    def state(self, k, cache=True) -> dict:
        """Maps I, G, F cropped to H x W (copies, so the full arrays are freed)
        and R. Up to CACHE frames are kept; cache=False reads without keeping."""
        if k in self._states:
            return self._states[k]
        st, _ = states_io.load_state(self.dir, k)
        H, W = self.H, self.W
        out = {key: st[key][:H, :W].copy() for key in ('I', 'G', 'F')}
        out['R'] = st['R']
        if cache:
            if len(self._states) >= CACHE:
                self._states.pop(next(iter(self._states)))
            self._states[k] = out
        return out

    def iterations(self, k):
        path = states_io.replay_path(self.dir, k)
        if not os.path.exists(path):
            return None
        with np.load(path) as z:
            return {key: z[key] for key in z.files}

    # ------------------------------------------------------------ raw data
    def has_data(self) -> bool:
        return os.path.exists(self.paths['events'])

    def _frames_V(self):
        if self._V is None:
            from interacting_maps.data_loader import EventFrameSequence
            seq = EventFrameSequence(
                self.paths['events'], self.paths['calib'], frame_duration=self.dt,
                t_start=self.t_start, n_frames=self.n_frames, clip_value=10.0,
                undistort=self.config['distortion_mode'] == 'undistort_events',
                sensor_size=(self.H, self.W))
            self._V = [V for V, _ in seq]
        return self._V

    def raw(self, k) -> dict:
        if not self.has_data():
            return {'V': None, 'aps': None}
        if self._aps is None:
            from interacting_maps.eval_io import load_aps_list
            self._aps = load_aps_list(self.paths['data_dir'], self.t_start - 0.5,
                                      self.t_start + self.n_frames * self.dt + 0.5) or []
        aps = None
        if self._aps:
            t = self.t_start + self.times[k]
            ts = np.array([s for s, _ in self._aps])
            i = int(np.argmin(np.abs(ts - t)))
            import matplotlib.pyplot as plt
            img = plt.imread(self._aps[i][1])
            aps = (ts[i] - self.t_start, img.mean(-1) if img.ndim == 3 else img)
        return {'V': self._frames_V()[k], 'aps': aps}

    def curves(self) -> dict:
        if self._curves is None:
            n_ev = np.full(self.n_frames, np.nan)
            if self.has_data():
                from interacting_maps.data_loader import load_events_fast
                ev = load_events_fast(self.paths['events'], t_start=self.t_start,
                                      duration=self.n_frames * self.dt)
                edges = self.t_start + np.arange(self.n_frames + 1) * self.dt
                n_ev = np.histogram(ev[:, 0], bins=edges)[0].astype(float)
            self._curves = {**self.rows, 'n_events': n_ev}
        return self._curves

    def omega_fullrate(self) -> dict:
        """|omega| in deg/s: the gyro at its own rate and the scoring reference on
        a grid of dt/10, both against seconds since segment start."""
        if not os.path.exists(self.paths['imu']):
            return {}
        from interacting_maps.eval_io import (load_imu, load_groundtruth, load_omega_gt,
                                              get_reference_omega)
        imu = load_imu(self.paths['imu'])
        t_end = self.t_start + self.n_frames * self.dt
        m = (imu[:, 0] >= self.t_start) & (imu[:, 0] < t_end)
        gt = load_groundtruth(self.paths['groundtruth'])
        om = load_omega_gt(self.paths.get('omega_gt'))
        h = self.dt / 10
        tg = np.arange(self.t_start, t_end, h)
        ref = np.array([get_reference_omega(gt, imu, t, t + h, om)[0] for t in tg])
        return {'t_imu': imu[m, 0] - self.t_start,
                'imu': np.degrees(np.linalg.norm(imu[m, 4:7], axis=1)),
                't_gt': tg + h / 2 - self.t_start,
                'gt': np.degrees(np.linalg.norm(ref, axis=1))}


def load_run(run_dir, data_root=None) -> RunView:
    """data_root: folder holding <dataset>/ (default: the repository's data/)."""
    return RunView(run_dir, data_root)


def check_comparable(runs) -> None:
    if not 1 <= len(runs) <= MAX_RUNS:
        raise ValueError(f'compare 1 to {MAX_RUNS} runs, got {len(runs)}')
    for key in SHARED:
        vals = {str(r.config[key]) for r in runs}
        if len(vals) > 1:
            raise ValueError(f'runs differ in {key}: {sorted(vals)}')


# ------------------------------------------------------------------ catalog
def segment_label(cfg) -> str:
    """One segment of one dataset at one dt and length: what compared runs share."""
    return (f"{cfg['segment_id']}  t0 {float(cfg['t_start']):.3f} s  "
            f"dt {1000 * float(cfg['frame_duration']):.0f} ms  {int(cfg['n_frames'])} frames")


def scan_runs(root) -> list:
    """Every run under root that has saved states, as dicts with dir, dataset,
    segment (segment_label), model, n_frames, replays (frames), summary, provenance."""
    import glob
    out = []
    for path in sorted(glob.glob(os.path.join(root, '**', 'run.json'), recursive=True)):
        d = os.path.dirname(path)
        if not states_io.has_states(d):
            continue
        rec = experiment_tracking.read_run(d)
        c = rec['config']
        rep = glob.glob(os.path.join(os.path.dirname(states_io.replay_path(d, 0)), 'frame_*.npz'))
        out.append({'dir': d, 'dataset': c['dataset'], 'segment': segment_label(c),
                    'model': c['model'], 'n_frames': int(c['n_frames']),
                    'replays': sorted(int(os.path.basename(p)[6:10]) for p in rep),
                    'summary': rec['summary'], 'provenance': rec['provenance']})
    return out
