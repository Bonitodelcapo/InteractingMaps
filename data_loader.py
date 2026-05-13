"""
Event camera data loader.

Supports the RPG Event Camera Dataset format (Mueggler et al., 2017):
  events.txt  — one event per line: timestamp x y polarity
  calib.txt   — camera intrinsics: fx fy cx cy k1 k2 p1 p2 k3

Reference dataset:
  http://rpg.ifi.uzh.ch/datasets/davis/shapes_rotation.zip
  (Creative Commons CC BY-NC-SA 3.0, non-commercial use only)
"""

import os
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class CameraCalibration:
    """
    Pinhole camera parameters parsed from calib.txt.

    calib.txt format: fx fy cx cy k1 k2 p1 p2 k3
    """

    def __init__(self, path: str):
        vals = np.loadtxt(path, dtype=np.float64)
        self.fx, self.fy = vals[0], vals[1]
        self.cx, self.cy = vals[2], vals[3]
        self.dist = vals[4:]     # k1 k2 p1 p2 k3

    def sensor_size(self) -> tuple[int, int]:
        """Guess sensor size from principal point (approximate)."""
        H = int(round(self.cy * 2))
        W = int(round(self.cx * 2))
        return H, W

    def crop_calibration(
        self,
        x_start: int,
        y_start: int,
    ) -> "CameraCalibration":
        """
        Return a new CameraCalibration for a crop starting at (x_start, y_start).
        """
        c = CameraCalibration.__new__(CameraCalibration)
        c.fx, c.fy = self.fx, self.fy
        c.cx = self.cx - x_start
        c.cy = self.cy - y_start
        c.dist = self.dist.copy()
        return c


# ---------------------------------------------------------------------------
# Event loading
# ---------------------------------------------------------------------------

def load_events(
    events_txt: str,
    t_start: float = 0.0,
    t_end: float | None = None,
    max_events: int | None = None,
) -> np.ndarray:
    """
    Load events from an RPG-format events.txt file.

    Each row of the returned array is [timestamp, x, y, polarity].
    Polarity is stored as-is (0 = off, 1 = on).

    Parameters
    ----------
    events_txt : path to events.txt
    t_start    : only load events with timestamp >= t_start
    t_end      : only load events with timestamp <  t_end  (None = no limit)
    max_events : maximum number of events to return

    Returns
    -------
    events : (N, 4) float32  [t, x, y, polarity]
    """
    events = []
    n = 0
    with open(events_txt, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            t = float(parts[0])
            if t < t_start:
                continue
            if t_end is not None and t >= t_end:
                break
            events.append((t, float(parts[1]), float(parts[2]), float(parts[3])))
            n += 1
            if max_events is not None and n >= max_events:
                break

    if not events:
        return np.empty((0, 4), dtype=np.float32)
    return np.array(events, dtype=np.float32)


def load_events_fast(
    events_txt: str,
    t_start: float = 0.0,
    duration: float = 5.0,
) -> np.ndarray:
    """
    Faster bulk loader using numpy (reads up to duration seconds of events).
    Scans to find the approximate row where t_start begins, then bulk-loads.
    """
    # First pass: count rows to skip (events before t_start)
    skip = 0
    with open(events_txt, 'r') as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if float(parts[0]) >= t_start:
                break
            skip += 1

    # Estimate max rows (events/sec * duration from first scan)
    chunk = np.loadtxt(events_txt, skiprows=skip, max_rows=5_000_000, dtype=np.float32)
    if chunk.ndim == 1:
        chunk = chunk.reshape(1, -1)

    t_end = t_start + duration
    mask = chunk[:, 0] < t_end
    return chunk[mask]


# ---------------------------------------------------------------------------
# Event-to-V frame conversion
# ---------------------------------------------------------------------------

def events_to_vframe(
    events: np.ndarray,
    H: int,
    W: int,
    x_offset: int = 0,
    y_offset: int = 0,
    clip_value: float = 3.0,
    normalise: bool = True,
) -> np.ndarray:
    """
    Bin events in a time window into a scalar V frame.

    Each pixel accumulates the signed count of events:
        +1 per ON event  (polarity = 1)
        -1 per OFF event (polarity = 0)

    Parameters
    ----------
    events   : (N, 4) float32  [t, x, y, polarity]
    H, W     : output frame size
    x_offset : subtract from event x-coordinate (for cropping)
    y_offset : subtract from event y-coordinate
    clip_value : clip the accumulated count to ±clip_value before normalising
    normalise  : divide by clip_value so output is in [-1, 1]

    Returns
    -------
    V : (H, W) float64
    """
    V = np.zeros((H, W), dtype=np.float64)

    if len(events) == 0:
        return V

    xs = events[:, 1].astype(np.int32) - x_offset
    ys = events[:, 2].astype(np.int32) - y_offset
    pol = events[:, 3]                           # 0 or 1

    # Keep only events within the crop
    mask = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xs, ys, pol = xs[mask], ys[mask], pol[mask]

    # Signed accumulation: ON → +1, OFF → -1
    signed = 2.0 * pol - 1.0

    np.add.at(V, (ys, xs), signed)

    V = np.clip(V, -clip_value, clip_value)
    if normalise:
        V /= clip_value

    return V


# ---------------------------------------------------------------------------
# Frame sequence builder
# ---------------------------------------------------------------------------

class EventFrameSequence:
    """
    Iterates over a sequence of V frames built from a loaded events array.

    Usage
    -----
    seq = EventFrameSequence('data/shapes_rotation/events.txt',
                             calib_path='data/shapes_rotation/calib.txt',
                             H=128, W=128)
    for V, t in seq:
        # feed V into the network
    """

    def __init__(
        self,
        events_txt: str,
        calib_path: str,
        H: int = 128,
        W: int = 128,
        frame_duration: float = 0.020,   # seconds per frame (≈ 50 fps)
        t_start: float = 0.5,            # skip first 0.5 s (sensor settling)
        n_frames: int = 50,
        clip_value: float = 3.0,
    ):
        self.H = H
        self.W = W
        self.frame_duration = frame_duration
        self.clip_value = clip_value

        calib_raw = CameraCalibration(calib_path)
        H_sensor, W_sensor = calib_raw.sensor_size()

        # Centre crop to (H, W)
        self.x_offset = max(0, int(calib_raw.cx) - W // 2)
        self.y_offset = max(0, int(calib_raw.cy) - H // 2)
        self.calib = calib_raw.crop_calibration(self.x_offset, self.y_offset)

        print(f"Loading events from {events_txt} …")
        print(f"  Sensor: {W_sensor}×{H_sensor}, crop offset: ({self.x_offset}, {self.y_offset})")
        print(f"  Output frame: {W}×{H},  {n_frames} frames × {frame_duration*1000:.0f} ms")

        t_end = t_start + n_frames * frame_duration + 0.1
        all_events = load_events_fast(events_txt, t_start=t_start, duration=t_end - t_start)
        self._events = all_events
        self._t_start = t_start
        self._n_frames = n_frames

        print(f"  Loaded {len(all_events):,} events covering "
              f"{all_events[-1, 0] - all_events[0, 0]:.3f} s")

    def __iter__(self):
        """Yield (V_frame, frame_midtime) for each time bin."""
        events = self._events
        t0 = self._t_start
        dt = self.frame_duration
        for k in range(self._n_frames):
            t_lo = t0 + k * dt
            t_hi = t_lo + dt
            mask = (events[:, 0] >= t_lo) & (events[:, 0] < t_hi)
            V = events_to_vframe(
                events[mask],
                self.H, self.W,
                x_offset=self.x_offset,
                y_offset=self.y_offset,
                clip_value=self.clip_value,
                normalise=True,
            )
            yield V, (t_lo + t_hi) / 2

    def __len__(self) -> int:
        return self._n_frames

    def calibration_matrix(self) -> np.ndarray:
        """Return the (H, W, 3) unit-vector calibration map C for this crop."""
        from interacting_maps.camera import compute_calibration
        return compute_calibration(self.H, self.W, self.calib.fx)
