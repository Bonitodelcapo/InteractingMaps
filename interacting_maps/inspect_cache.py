"""Everything the inspection app reads per step, for one run, in memory.

RunCache.open(run_dir) decompresses all saved states (I, G, F as float32,
cropped to H x W; R), all replays, the event frames, the APS images, the curves
and the full-rate |omega| once. Stepping through frames afterwards is indexing.

A copy of the decompressed arrays is kept in <run>/.inspect_cache/ and reused
while the run's states/ and replay/ files are unchanged (names, sizes, mtimes
in stamp.json), so reopening a run takes seconds.
"""
import json
import os

import numpy as np

from interacting_maps import states_io
from interacting_maps.inspect_io import load_run
from interacting_maps.inspect_nav import pixel_of_stack, residuals_of_stack

CACHE_DIR = '.inspect_cache'
KINDS = ('I', 'G', 'F')


def _stamp(run_dir) -> list:
    out = []
    for sub in ('states', 'replay'):
        d = os.path.join(run_dir, sub)
        if os.path.isdir(d):
            for name in sorted(os.listdir(d)):
                st = os.stat(os.path.join(d, name))
                out.append([f'{sub}/{name}', st.st_size, st.st_mtime_ns])
    return out


def _replay_frames(run_dir) -> list:
    d = os.path.dirname(states_io.replay_path(run_dir, 0))
    if not os.path.isdir(d):
        return []
    return sorted(int(n[6:10]) for n in os.listdir(d) if n.startswith('frame_'))


class RunCache:
    def __init__(self, run):
        self.run = run
        self.dir = run.dir
        self.config = run.config
        self.n_frames, self.n_iters = run.n_frames, run.n_iters
        self.H, self.W, self.dt, self.times = run.H, run.W, run.dt, run.times
        self.maps, self.replays, self.V = {}, {}, None
        self.aps, self.aps_index = [], None
        self.curves, self.fullrate, self.frame_scales = {}, {}, {}

    # ------------------------------------------------------------ open
    @classmethod
    def open(cls, run_dir, data_root=None, progress=None) -> 'RunCache':
        c = cls(load_run(run_dir, data_root))
        report = progress or (lambda f: None)
        report(0.0)
        cdir = os.path.join(run_dir, CACHE_DIR)
        stamp = _stamp(run_dir)
        if c._cache_valid(cdir, stamp):
            c._read_cache(cdir)
        else:
            c._decompress(report)
            c._write_cache(cdir, stamp)
        report(0.8)
        c._load_raw()
        c._compute_scales()
        report(1.0)
        return c

    def _cache_valid(self, cdir, stamp) -> bool:
        try:
            with open(os.path.join(cdir, 'stamp.json')) as f:
                return json.load(f) == stamp
        except (OSError, ValueError):
            return False

    def _decompress(self, report):
        n = self.n_frames
        frames = _replay_frames(self.dir)
        total = n + len(frames)
        self.maps = {'I': np.empty((n, self.H, self.W), np.float32),
                     'G': np.empty((n, self.H, self.W, 2), np.float32),
                     'F': np.empty((n, self.H, self.W, 2), np.float32),
                     'R': np.empty((n, 3), np.float64)}
        for k in range(n):
            st = self.run.state(k, cache=False)
            for key in self.maps:
                self.maps[key][k] = st[key]
            report(0.8 * (k + 1) / total)
        for i, k in enumerate(frames):
            rep = self.run.iterations(k)
            self.replays[k] = {key: (v.astype(np.float32) if key != 'R' else v)
                               for key, v in rep.items()}
            report(0.8 * (n + i + 1) / total)

    def _write_cache(self, cdir, stamp):
        os.makedirs(cdir, exist_ok=True)
        np.savez(os.path.join(cdir, 'maps.npz'), **self.maps)
        for k, rep in self.replays.items():
            np.savez(os.path.join(cdir, f'replay_{k:04d}.npz'), **rep)
        with open(os.path.join(cdir, 'stamp.json'), 'w') as f:      # last: marks the cache complete
            json.dump(stamp, f)

    def _read_cache(self, cdir):
        with np.load(os.path.join(cdir, 'maps.npz')) as z:
            self.maps = {key: z[key] for key in z.files}
        for k in _replay_frames(self.dir):
            with np.load(os.path.join(cdir, f'replay_{k:04d}.npz')) as z:
                self.replays[k] = {key: z[key] for key in z.files}

    def _load_raw(self):
        run = self.run
        self.curves = run.curves()
        self.fullrate = run.omega_fullrate()
        if not run.has_data():
            return
        self.V = np.stack(run._frames_V()).astype(np.float32)
        run.raw(0)                                   # fills run._aps (APS index)
        import matplotlib.pyplot as plt
        for t, path in run._aps:
            img = plt.imread(path)
            self.aps.append((t - self.t_start, img.mean(-1) if img.ndim == 3 else img))
        if self.aps:
            ts = np.array([t for t, _ in self.aps])
            self.aps_index = np.array([int(np.argmin(np.abs(ts - t))) for t in self.times])

    @property
    def t_start(self):
        return self.run.t_start

    def _compute_scales(self):
        from interacting_maps.inspect_fig import _finite_scale
        for kind in KINDS:
            self.frame_scales[kind] = [_finite_scale(kind, self.maps[kind][k])
                                       for k in range(self.n_frames)]
        self.frame_scales['V'] = ([_finite_scale('V', v) for v in self.V]
                                  if self.V is not None else [])

    # ------------------------------------------------------------ access
    def has_replay(self, k) -> bool:
        return k in self.replays

    def map_at(self, kind, k, it=None) -> np.ndarray:
        """Map after frame k, or after iteration `it` (1-based) of its replay."""
        if it and k in self.replays:
            rep = self.replays[k][kind]
            return rep[min(it, len(rep)) - 1]
        return self.maps[kind][k]

    def aps_at(self, k):
        return None if self.aps_index is None else self.aps[self.aps_index[k]][1]

    def aps_time(self, k):
        return None if self.aps_index is None else self.aps[self.aps_index[k]][0]

    def pixel_series(self, x, y) -> dict:
        return pixel_of_stack(self.maps, x, y)

    def pixel_iterations(self, k, x, y):
        return pixel_of_stack(self.replays[k], x, y) if k in self.replays else None

    def iteration_residuals(self, k):
        if k not in self.replays:
            return None
        V = None if self.V is None else self.V[k].astype(np.float64)
        return residuals_of_stack(self.replays[k], V, self.dt)
