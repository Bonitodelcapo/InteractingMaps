"""The per-frame inputs of a tracking run, in one place.

experiment_tracking (evaluation.py) and the iteration replay (replay.py) both
step a network through frame k. What they hand to net.step besides V depends on
the model: the gyro (thesis_imu), a full CMax solve (thesis_cmax, V1) or the
raw event window (thesis_cmax_v2). FrameDriver builds those inputs, and holds
the only state that lives outside the network: V1's omega_cmax_prev.
"""
import numpy as np

from interacting_maps.eval_io import load_imu, get_gyro_for_frame


class FrameDriver:
    def __init__(self, rc, net, H, W, calib):
        self.rc, self.net = rc, net
        self.imu_data = load_imu(rc.paths['imu'])     # model INPUT (gyro)
        self.cmax_est = None
        self.cmax_events = None
        self.omega_cmax_prev = np.zeros(3)
        self.last_window = None
        self.last_anchor = None
        if getattr(rc, 'use_cmax', False):
            self._load_cmax(H, W, calib)

    def _load_cmax(self, H, W, calib):
        # B1: load the raw events separately and slice per window, leaving
        # EventFrameSequence untouched. The gyro still provides frame-0 init only.
        from interacting_maps.data_loader import load_events_fast, undistort_events
        rc = self.rc
        dur = rc.n_frames * rc.frame_duration + 0.1
        self.cmax_events = undistort_events(
            load_events_fast(rc.paths['events'], t_start=rc.t_start, duration=dur), calib)
        if getattr(rc, 'use_cmax_v2', False):
            print(f"  CMax V2 active — R driven by 1 CMax step / iteration "
                  f"(lr={rc.cmax_lr}, {len(self.cmax_events)} events).")
        else:
            # V1: standalone full-solve estimator, feeds the R anchor.
            from interacting_maps.angular_velocity import CMaxAngularVelocity
            self.cmax_est = CMaxAngularVelocity(H, W, calib.fx, calib.fy, calib.cx,
                                                calib.cy, use_polarity=True)
            print(f"  CMax V1 active — R anchored to full CMax ω per frame "
                  f"({len(self.cmax_events)} events).")

    def window_bounds(self, k):
        lo = self.rc.t_start + k * self.rc.frame_duration
        return lo, lo + self.rc.frame_duration

    def _window(self, lo, hi):
        ev = self.cmax_events
        if ev is None:
            return None
        return ev[(ev[:, 0] >= lo) & (ev[:, 0] < hi)]

    def frame_kwargs(self, k) -> dict:
        """What net.step gets besides V for frame k. V1 runs its CMax solve here
        and advances omega_cmax_prev, so call it once per frame."""
        rc = self.rc
        lo, hi = self.window_bounds(k)
        win = self._window(lo, hi)
        self.last_window = win
        if getattr(rc, 'use_cmax_v2', False):
            return {'events': win}                    # V2: CMax drives R in the loop
        if self.cmax_est is not None:                 # V1: full CMax solve -> R anchor
            anchor = self.cmax_est.estimate(win, t_ref=0.5 * (lo + hi),
                                            omega_init=self.omega_cmax_prev)
            self.omega_cmax_prev = anchor.copy()
            self.last_anchor = anchor
            return {'omega_imu': anchor}
        if rc.use_thesis and rc.use_imu:
            return {'omega_imu': get_gyro_for_frame(self.imu_data, lo, hi)}
        return {}

    def step(self, k, V, n_iters) -> None:
        self.net.step(V, n_iters=n_iters, **self.frame_kwargs(k))

    def loop_state(self) -> dict:
        return {'omega_cmax_prev': self.omega_cmax_prev.copy()} if self.cmax_est is not None else {}

    def set_loop_state(self, d: dict) -> None:
        if 'omega_cmax_prev' in d:
            self.omega_cmax_prev = np.array(d['omega_cmax_prev'], dtype=np.float64)
