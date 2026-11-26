"""Replay one frame of a saved run, one iteration at a time.

Starts from the state saved after frame k-1 (the initial state for k = 0),
rebuilds frame k's inputs exactly as experiment_tracking does (FrameDriver),
and stores I, G, F (cropped to H x W) and R after every iteration in
states_io.replay_path(run_dir, k). The last iteration equals the saved state k.
"""
import os

import numpy as np

from interacting_maps import states_io
from interacting_maps.data_loader import EventFrameSequence
from interacting_maps.frame_driver import FrameDriver
from interacting_maps.viz import net_maps


def frame_input(rc, k):
    """(V_k, calib, H, W) of frame k, binned exactly as in the tracking run."""
    seq = EventFrameSequence(rc.paths['events'], rc.paths['calib'],
                             frame_duration=rc.frame_duration, t_start=rc.t_start,
                             n_frames=k + 1, clip_value=10.0,
                             undistort=rc.undistort_at_event_level,
                             sensor_size=rc.sensor_size)
    V = None
    for V, _ in seq:
        pass
    return V, seq.calib, seq.H, seq.W


def replay_frame(run_dir: str, k: int) -> str:
    from run_config import rc_from_run, make_network
    states_io.assert_not_results(run_dir)
    rc = rc_from_run(run_dir)
    V, calib, H, W = frame_input(rc, k)
    net = make_network(rc, H, W, calib.fx, calib.fy, calib.cx, calib.cy)
    driver = FrameDriver(rc, net, H, W, calib)
    state, loop = states_io.load_state(run_dir, k - 1)
    net.set_state(state)
    driver.set_loop_state(loop)
    kw = driver.frame_kwargs(k)          # once: V1's CMax solve advances its loop state

    hist = {'I': [], 'G': [], 'F': [], 'R': []}
    for _ in range(rc.n_iters):
        net.step(V, n_iters=1, **kw)
        maps = net_maps(net, H, W)
        for key in ('I', 'G', 'F'):
            hist[key].append(maps[key].copy())
        hist['R'].append(np.array(net.R, dtype=np.float64))

    path = states_io.replay_path(run_dir, k)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **{key: np.stack(v) for key, v in hist.items()})
    return path
