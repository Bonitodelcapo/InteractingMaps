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
import cv2

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

    # Estimate max rows (events/sec * duration from first scan).
    # Cap raised 5M→30M: long segments (dt=0.05 × 150 frames = 7.5s at
    # ~2 Mev/s ≈ 16 Mev) were silently truncated at 5M, blanking later frames.
    chunk = np.loadtxt(events_txt, skiprows=skip, max_rows=30_000_000, dtype=np.float32)
    if chunk.ndim == 1:
        chunk = chunk.reshape(1, -1)

    t_end = t_start + duration
    mask = chunk[:, 0] < t_end
    return chunk[mask]

import cv2

def undistort_events(events: np.ndarray, calib: 'CameraCalibration') -> np.ndarray:
    """
    Undistort event (x, y) coordinates using the radial/tangential distortion model.
    
    Parameters
    ----------
    events : (N, 4) [t, x, y, pol]
    calib  : CameraCalibration with fx, fy, cx, cy, dist
    
    Returns
    -------
    events_undist : (N, 4) with corrected x, y (float coords)
    """
    if len(events) == 0:
        return events
    
    # Build intrinsic matrix and distortion vector
    K = np.array([[calib.fx, 0, calib.cx],
                  [0, calib.fy, calib.cy],
                  [0, 0, 1]], dtype=np.float64)
    dist_coeffs = calib.dist.astype(np.float64)  # [k1, k2, p1, p2, k3]
    
    # Event pixel coords → (N, 1, 2) for cv2
    pts = events[:, 1:3].astype(np.float64).reshape(-1, 1, 2)
    
    # Undistort points (keeps them in pixel space, same K)
    pts_undist = cv2.undistortPoints(pts, K, dist_coeffs, P=K)
    
    events_out = events.copy()
    events_out[:, 1] = pts_undist[:, 0, 0]
    events_out[:, 2] = pts_undist[:, 0, 1]
    return events_out

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
    def __init__(
        self,
        events_txt: str,
        calib_path: str,
        frame_duration: float = 0.020,
        t_start: float = 0.5,
        n_frames: int = 50,
        clip_value: float = 3.0,
        sensor_size: tuple = None,
        undistort: bool = True,
    ):
        self.calib = CameraCalibration(calib_path)

        self.H, self.W = sensor_size

        self.x_offset = 0
        self.y_offset = 0

        self.frame_duration = frame_duration
        self.clip_value = clip_value
        self.undistort = undistort

        print(f"Using calib.txt: fx={self.calib.fx:.1f} fy={self.calib.fy:.1f} "
              f"cx={self.calib.cx:.1f} cy={self.calib.cy:.1f}")
        print(f"Sensor size: {self.W}×{self.H}")
        print(f"Principal point offset from center: "
              f"Δx={self.calib.cx - self.W/2:.1f}px, "
              f"Δy={self.calib.cy - self.H/2:.1f}px")

        t_end = t_start + n_frames * frame_duration + 0.1
        self._events = load_events_fast(events_txt, t_start=t_start, duration=t_end - t_start)
        self._t_start = t_start
        self._n_frames = n_frames
        self._dt = frame_duration

    def __iter__(self):
        events = self._events
        for k in range(self._n_frames):
            t_lo = self._t_start + k * self._dt
            t_hi = t_lo + self._dt
            mask = (events[:, 0] >= t_lo) & (events[:, 0] < t_hi)

            frame_events = events[mask]
            if self.undistort:
                frame_events = undistort_events(frame_events, self.calib)
            V = events_to_vframe(
                frame_events, self.H, self.W,
                x_offset=0, y_offset=0,  # no crop
                clip_value=self.clip_value, normalise=True,
            )
            yield V, (t_lo + t_hi) / 2

    def __len__(self) -> int:
        return self._n_frames

