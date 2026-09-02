"""
evaluation.py — Systematic evaluation harness for Interacting Maps.

Models (``--model``)
  cook           Cook 2011 network (Gauss-Seidel)
  thesis         Martel 2019 network (Jacobi two-phase)
  thesis_imu     thesis + Cost_IMU: R anchored to the gyro every iteration
  thesis_cmax    V1 — R anchored to a full CMax solve per frame (IMU-free in-loop)
  thesis_cmax_v2 V2 — R driven by one CMax gradient step per iteration (CMax-only)

Angular velocity is scored against groundtruth.txt (quaternion differencing,
body frame — the independent reference; see SCORE_AGAINST) rather than the IMU,
so the IMU-fed models are not graded on their own input. Metrics per frame:
err = ‖ω_est − ω_ref‖ (deg/s), direction error (deg), and β = ‖ω_ref‖/‖ω_est‖.

Experiments (``--exp N``)
  1. Single-frame convergence (maps at iteration checkpoints)
  2. Multi-frame tracking (ω_est vs ω_ref over time)   ← the core; 4/7/8/9 call it
  3. Parameter influence (iterations & frame-duration sweeps for one frame)
  4. Qualitative video (tracking with 3-column frames: Events | I | GT APS)
  5. Assemble MP4s from saved frames (batch)
  6. Basin of attraction (how far can init-R be from GT and still converge?)
  7. Full evaluation (all datasets × models × segments → full_evaluation.csv)
  8. Parameter grid (sweep frame_duration × n_frames × n_iters × delta_FR × delta_IMU)
  9. Distortion ablation (Way-1 undistort-events vs Way-2 distortion-aware C/warp)

Usage:
    python evaluation.py --exp 8 --all-segments     # for all segments sweep over
                                                    #  'n_frames':  [25, 50, 150],
                                                    #  'n_iters':   [75, 100],
                                                    #  'delta_FR':  [0.10, 0.20, 0.30, 0.50],
                                                    #  'delta_IMU': [0.10, 0.20, 0.30, 0.50],

    # Alles mit Bildern (Standard):
    python evaluation.py --exp 7 --model all

    # Schneller ohne Bilder (nur Metriken):
    python evaluation.py --exp 7 --model all --no-frames

    # Parameter-Grid, alle Segmente, alle Models, mit Bildern:
    python evaluation.py --exp 8 --all-segments --model all

    # Einzelner Run:
    python evaluation.py --exp 2 --dataset poster_rotation --model thesis_imu

    # exp 9 
    python evaluation.py --exp 9 --dataset boxes_rotation  --segment all --model thesis_imu --no-frames
    python evaluation.py --exp 9 --dataset poster_rotation --segment all --model thesis_imu --no-frames
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import json
import time as time_module

from config import (DATASET_CONFIGS, DATASET_SEGMENTS, THESIS_PARAMS, COOK_PARAMS,
                    ITERS_PER_FRAME, DISTORTION_MODE, get_dataset_paths, get_initial_R_from_imu)
from best_config import best_params
from data_loader import EventFrameSequence
from interacting_maps.network import InteractingMaps
from interacting_maps.network_dissertation import InteractingMapsThesis


# ===========================================================================
# MODEL / DATASET RESOLUTION
# ===========================================================================

# The full model roster. Batch runners used to hardcode only the first three,
# which silently dropped both CMax variants from `--model all`.
ALL_MODELS = ['cook', 'thesis', 'thesis_imu', 'thesis_cmax', 'thesis_cmax_v2']
CLASSIC_MODELS = ['cook', 'thesis', 'thesis_imu']


def resolve_models(model_arg=None, models_arg=None):
    """
    Resolve the model list for a batch run.

    models_arg : comma-separated list (wins over model_arg) e.g. 'thesis,cook'
    model_arg  : 'all' -> ALL_MODELS, 'classic' -> CLASSIC_MODELS,
                 None  -> ALL_MODELS, a single name -> [name]
    """
    if models_arg:
        names = [m.strip() for m in models_arg.split(',') if m.strip()]
        unknown = [m for m in names if m not in ALL_MODELS]
        if unknown:
            raise ValueError(f"Unknown model(s) {unknown}; valid: {ALL_MODELS}")
        return names
    if model_arg in (None, 'all'):
        return list(ALL_MODELS)
    if model_arg == 'classic':
        return list(CLASSIC_MODELS)
    return [model_arg]


def available_datasets(dataset_filter=None):
    """
    Datasets from DATASET_SEGMENTS whose events.txt actually exists.

    Missing data is skipped with a warning instead of raising, so an overnight
    batch keeps going when one dataset was not downloaded/converted.
    """
    names = [dataset_filter] if dataset_filter else list(DATASET_SEGMENTS.keys())
    out = []
    for name in names:
        if name not in DATASET_SEGMENTS:
            print(f"  WARNING: '{name}' is not in DATASET_SEGMENTS - skipping")
            continue
        if not os.path.exists(get_dataset_paths(name)['events']):
            print(f"  WARNING: no events.txt for '{name}' - skipping")
            continue
        out.append(name)
    return out


def resolve_segments(dataset, segments_arg=None):
    """Segment dicts for a dataset. segments_arg: None/'all' or 'seg_A,seg_C'."""
    segs = DATASET_SEGMENTS[dataset]
    if segments_arg in (None, 'all'):
        return list(segs)
    wanted = [s.strip() for s in segments_arg.split(',') if s.strip()]
    out = []
    for w in wanted:
        match = [s for s in segs if s['id'] == w]
        if not match:
            print(f"  WARNING: segment '{w}' not found in {dataset} - skipping")
            continue
        out.append(match[0])
    return out


def segment_n_frames(seg, frame_duration, requested=None):
    """
    Frames to run for a segment.

    A segment defines an interval where omega is quasi-constant; n_frames and
    frame_duration are hyperparameters. If the segment carries a validated
    'duration', never run past it -- otherwise the "constant-omega" premise is
    violated (this is how 0.5 s segments ended up being run for 3.0 s).
    """
    n = requested if requested is not None else seg.get('n_frames', 75)
    dur = seg.get('duration')
    if dur:
        n = min(n, int(dur / frame_duration))
    return max(1, int(n))


# ===========================================================================
# RUN CONFIGURATION (override via CLI args)
# ===========================================================================

class RunConfig:
    """All parameters for a single evaluation run."""
    def __init__(self, dataset='boxes_rotation', model='thesis_imu',
                 segment=None, t_start=None, frame_duration=None, n_frames=None,
                 n_iters=None, delta_IMU=None, delta_FR=None, distortion_mode=None,
                 delta_VFG=None, delta_IG=None, delta_GI=None, delta_RF=None,
                 cmax_lr=None, tag=None):

        self.dataset = dataset
        self.model = model  # 'cook', 'thesis', 'thesis_imu', 'thesis_cmax[_v2]'
        self.distortion_mode = distortion_mode or DISTORTION_MODE
        self.tag = tag      # free-form suffix for output_dir (used by OAT sweeps)
        
        if segment is None:
            # Fallback: erstes Segment bzw. DATASET_CONFIGS
            self.segment = DATASET_CONFIGS[dataset]
            self.segment_id = 'default'
        elif isinstance(segment, str):
            # Lookup by ID (z.B. 'seg_A')
            segs = DATASET_SEGMENTS[dataset]
            match = [s for s in segs if s['id'] == segment]
            if not match:
                raise ValueError(f"Segment '{segment}' not found for {dataset}")
            self.segment = match[0]
            self.segment_id = segment
        else:
            # Direkt ein Dict übergeben
            self.segment = segment
            self.segment_id = segment.get('id', 'unknown')

        # Load dataset defaults
        cfg = self.segment
        self.t_start = t_start if t_start is not None else cfg['t_start']
        self.frame_duration = frame_duration if frame_duration is not None else cfg['frame_duration']
        self.n_frames = n_frames if n_frames is not None else cfg['n_frames']
        self.n_iters = n_iters if n_iters is not None else ITERS_PER_FRAME
        self.sensor_size = cfg.get('sensor_size', (180, 240))
        
        # Model parameters
        self.use_cmax = False       # load raw events + compute per-frame CMax ω
        self.use_cmax_v2 = False    # V2: CMax drives R inside the MP loop
        self.save_iwe = False       # save the final CMax IWE per frame (V1/V2)
        self.save_maps = True       # dump maps_{center,final}.npz for the D1 panel
        # V2 ascent step. STABLE range ~[1e-5, 1e-4]; >=5e-4 diverges. Scales
        # with event count^2 - lower it for denser streams (see cmax/cmax.md).
        self.cmax_lr = 1e-4 if cmax_lr is None else cmax_lr
        if model == 'cook':
            self.params = COOK_PARAMS.copy()
            self.use_thesis = False
            self.use_imu = False
        elif model == 'thesis':
            self.params = THESIS_PARAMS.copy()
            self.use_thesis = True
            self.use_imu = False
        elif model == 'thesis_cmax':
            # V1: same as thesis_imu, but the per-frame R anchor comes from a
            # full CMax solve on the events instead of the IMU gyro. Reuses the
            # Cost_IMU mechanism (a generic "pull R toward an external ω").
            # Init still from the gyro at frame 0 (see _compute_initial_R).
            self.params = THESIS_PARAMS.copy()
            self.use_thesis = True
            self.use_imu = True     # Cost_IMU is the anchor; target = ω_cmax
            self.use_cmax = True
        elif model == 'thesis_cmax_v2':
            # V2: CMax is the per-iteration R update INSIDE the message passing.
            # Kinematics updates F only; no IMU. Events loaded per frame (B1).
            self.params = THESIS_PARAMS.copy()
            self.use_thesis = True
            self.use_imu = False
            self.use_cmax = True
            self.use_cmax_v2 = True
        else:  # thesis_imu
            self.params = THESIS_PARAMS.copy()
            self.use_thesis = True
            self.use_imu = True
        
        # ---- Parameter overrides -------------------------------------
        # delta_IMU only applies to models that actually run Cost_IMU.
        if delta_IMU is not None and self.use_imu:
            self.params['delta_IMU'] = delta_IMU
        if delta_FR is not None:
            self.params['delta_FR'] = delta_FR

        # Spatial / OFCE relaxation weights (no CLI flags before; needed by the
        # OAT sweep). `_explicit` records which ones were set on purpose so that
        # output_dir only grows a suffix for non-default runs -> existing result
        # folder names stay byte-identical (resume-safe).
        self._explicit = {}
        for name, val in (('delta_VFG', delta_VFG), ('delta_IG', delta_IG),
                          ('delta_GI', delta_GI), ('delta_RF', delta_RF)):
            if val is not None:
                self.params[name] = val
                self._explicit[name] = val

        # Paths
        self.paths = get_dataset_paths(dataset)
        
        # Initial R from IMU
        self.initial_R = self._compute_initial_R()
    
    @property
    def delta_IMU(self):
        return self.params.get('delta_IMU', 0.0)
    
    @property
    def undistort_at_event_level(self):
        return self.distortion_mode == 'undistort_events'
    
    @property
    def duration_s(self):
        return self.n_frames * self.frame_duration
    
    @property
    def output_dir(self):
        """
        Unique folder per configuration.

        Backwards compatible: the base name is unchanged, and the extra
        suffixes below only appear for NON-default runs, so folders written by
        earlier versions keep their exact names (and --resume still finds them).
        The extra parts exist so that OAT sweeps (which vary one parameter at a
        time) do not overwrite each other -- notably cmax_lr, which was not
        encoded at all before and made two V2 runs collide.
        """
        folder_name = (f"{self.segment_id}_t{self.t_start:.3f}"
                    f"_dt{self.frame_duration*1000:.0f}ms"
                    f"_n{self.n_frames}_i{self.n_iters}")
        if self.use_imu:
            folder_name += f"_dimu{self.delta_IMU:.2f}"
        # Also encode delta_FR to distinguish configs
        folder_name += f"_dFR{self.params.get('delta_FR', 0):.2f}"
        folder_name += f"_{self.distortion_mode}"
        # Non-default spatial/OFCE weights (empty for every pre-existing run).
        for name in ('delta_VFG', 'delta_IG', 'delta_GI', 'delta_RF'):
            if name in self._explicit:
                folder_name += f"_{name.replace('delta_', 'd')}{self._explicit[name]:.3f}"
        # cmax_lr matters only for V2 (the only model that uses it).
        if self.use_cmax_v2:
            folder_name += f"_lr{self.cmax_lr:.0e}"
        if self.tag:
            folder_name += f"_{self.tag}"
        return os.path.join('results', self.dataset, self.model, folder_name)
    
    def to_dict(self):
        """Serialize all parameters for JSON."""
        return {
            'dataset': self.dataset,
            'model': self.model,
            't_start': self.t_start,
            'frame_duration': self.frame_duration,
            'n_frames': self.n_frames,
            'distortion_mode': self.distortion_mode,
            'n_iters': self.n_iters,
            'duration_s': self.duration_s,
            'sensor_size': list(self.sensor_size),
            'initial_R': self.initial_R.tolist(),
            'params': self.params,
        }
    
    def save_params(self):
        """Save params.json to output directory."""
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, 'params.json'), 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def __repr__(self):
        return (f"RunConfig({self.dataset}, {self.model}, "
                f"t={self.t_start:.3f}, dt={self.frame_duration*1000:.0f}ms, "
                f"n={self.n_frames}, iters={self.n_iters})")

    def _compute_initial_R(self):
        """Compute initial R from IMU at THIS segment's t_start."""
        imu_data = np.loadtxt(self.paths['imu'], dtype=np.float64)
        t_lo = self.t_start
        t_hi = self.t_start + self.frame_duration
        mask = (imu_data[:, 0] >= t_lo) & (imu_data[:, 0] < t_hi)
        
        if np.sum(mask) > 0:
            omega = np.mean(imu_data[mask, 4:7], axis=0)
        else:
            idx = np.argmin(np.abs(imu_data[:, 0] - self.t_start))
            omega = imu_data[idx, 4:7]
        
        return omega * self.frame_duration  # rad/s → rad/frame


# ===========================================================================
# HELPERS
# ===========================================================================

def resolve_best_kwargs(dataset, segment_id, model, cli_overrides=None):
    """Return RunConfig kwargs (frame_duration/n_frames/n_iters/delta_FR/delta_IMU)
    from best_config.py for a (dataset, segment_id, model).

    delta_IMU is only included for thesis_imu. `cli_overrides` (a dict of any of
    those keys with non-None values) wins over the stored best values, so an
    explicit CLI flag always takes precedence. Returns {} if no best entry exists.
    """
    bp = best_params(dataset, segment_id, model)
    if bp is None:
        print(f"  [--best] No best config for {dataset}/{segment_id}/{model}; "
              f"using defaults.")
        return {}
    kw = {
        'frame_duration': bp['frame_duration'],
        'n_frames': bp['n_frames'],
        'n_iters': bp['n_iters'],
        'delta_FR': bp['delta_FR'],
    }
    if model == 'thesis_imu':
        kw['delta_IMU'] = bp['delta_IMU']
    if cli_overrides:
        kw.update({k: v for k, v in cli_overrides.items() if v is not None})
    return kw


def load_imu(path: str) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float64)

def get_gyro_for_frame(imu_data, t_lo, t_hi):
    mask = (imu_data[:, 0] >= t_lo) & (imu_data[:, 0] < t_hi)
    if np.sum(mask) == 0:
        idx = np.argmin(np.abs(imu_data[:, 0] - (t_lo + t_hi) / 2))
        return imu_data[idx, 4:7]
    return np.mean(imu_data[mask, 4:7], axis=0)


# ---------------------------------------------------------------------------
# Ground-truth reference from groundtruth.txt (Vicon poses)
#
# The IMU gyro (imu.txt) is the camera's OWN sensor. When it is also fed to the
# thesis_imu model as `omega_imu`, scoring against it is circular (the model is
# graded against its own input). groundtruth.txt comes from an INDEPENDENT
# motion-capture rig, so scoring against it is unbiased.
#
# ω is recovered by differencing two successive orientation quaternions:
#     dR_body = R1.T @ R2      (right-invariant → CAMERA BODY FRAME)
# NOT R2 @ R1.T, which would give world-frame ω. The network and the gyro both
# report body-frame ω, so the reference must be body-frame too.
# ---------------------------------------------------------------------------

# Which source to SCORE against: 'groundtruth' (Vicon, independent) or 'imu'
# (gyro — only use for datasets that ship no groundtruth.txt). Model INPUT for
# thesis_imu is always the gyro regardless of this setting.
SCORE_AGAINST = 'groundtruth'


def load_groundtruth(path: str):
    """Load groundtruth.txt: [t tx ty tz qx qy qz qw] → (N, 8), or None."""
    if not os.path.exists(path):
        return None
    return np.loadtxt(path, dtype=np.float64)


def _quat_to_rotmat(q):
    """Quaternion (qx, qy, qz, qw) → 3×3 rotation matrix R_wc."""
    qx, qy, qz, qw = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),   2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),   2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ])


def gt_omega_body(gt_data, t_lo, t_hi):
    """
    Body-frame angular velocity (rad/s) from two groundtruth.txt poses that
    bracket the frame window [t_lo, t_hi], via dR_body = R1.T @ R2.
    """
    idx1 = int(np.argmin(np.abs(gt_data[:, 0] - t_lo)))
    idx2 = int(np.argmin(np.abs(gt_data[:, 0] - t_hi)))
    if idx1 == idx2:
        idx2 = min(idx1 + 1, len(gt_data) - 1)
    actual_dt = gt_data[idx2, 0] - gt_data[idx1, 0]
    if abs(actual_dt) < 1e-10:
        return np.zeros(3)

    R1 = _quat_to_rotmat(gt_data[idx1, 4:8])
    R2 = _quat_to_rotmat(gt_data[idx2, 4:8])
    dR = R1.T @ R2                      # body frame (NOT R2 @ R1.T)

    cos_a = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_a)
    if abs(angle) < 1e-10:
        return np.zeros(3)
    skew = (dR - dR.T) / (2.0 * np.sin(angle) + 1e-15)
    axis = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    return axis * angle / actual_dt    # rad/s, body frame


def load_omega_gt(path):
    """
    Load omega_gt.txt: [t wx wy wz] → (N, 4), or None. This is the CLEAN direct
    angular-velocity reference produced by convert_ecrot.py for synthetic
    datasets (no quaternion differencing / mocap noise).
    """
    if not path or not os.path.exists(path):
        return None
    return np.loadtxt(path, dtype=np.float64)


def omega_gt_direct(omega_data, t_lo, t_hi):
    """Mean of the direct ω reference over the window [t_lo, t_hi] (rad/s)."""
    mask = (omega_data[:, 0] >= t_lo) & (omega_data[:, 0] < t_hi)
    if np.sum(mask) == 0:
        idx = int(np.argmin(np.abs(omega_data[:, 0] - 0.5 * (t_lo + t_hi))))
        return omega_data[idx, 1:4].copy()
    return np.mean(omega_data[mask, 1:4], axis=0)


def get_reference_omega(gt_data, imu_data, t_lo, t_hi, omega_data=None):
    """
    Angular velocity used to SCORE the estimate (independent of model input).
    Preference: direct ω (omega_gt.txt, cleanest) → groundtruth.txt quaternion
    differencing → gyro. Datasets without omega_gt.txt are unaffected.
    Returns (omega_ref, source_str).
    """
    if omega_data is not None and SCORE_AGAINST in ('groundtruth', 'omega_direct'):
        return omega_gt_direct(omega_data, t_lo, t_hi), 'omega_direct'
    if SCORE_AGAINST == 'groundtruth' and gt_data is not None:
        return gt_omega_body(gt_data, t_lo, t_hi), 'groundtruth'
    return get_gyro_for_frame(imu_data, t_lo, t_hi), 'imu'

def make_network(rc: RunConfig, H, W, fx, fy, cx, cy):
    """Create network from RunConfig; distortion handling from rc.distortion_mode."""
    from data_loader import CameraCalibration
    mode = rc.distortion_mode
    if mode == 'undistort_events':
        dist_coeffs = None                                    # events already undistorted → pinhole C
    elif mode == 'C_full':
        dist_coeffs = CameraCalibration(rc.paths['calib']).dist  # distortion-aware C (+ Jacobian)
    else:
        raise ValueError(f"Unknown distortion_mode: {mode}")

    if rc.use_thesis:
        net = InteractingMapsThesis(
            H=H, W=W, fx=fx, fy=fy, cx=cx, cy=cy,
            frame_duration=rc.frame_duration,
            dist_coeffs=dist_coeffs,
            **rc.params
        )
        # V2: CMax drives R inside the loop (kinematics → F only, no IMU).
        if getattr(rc, 'use_cmax_v2', False):
            from cmax import CMaxAngularVelocity
            est = CMaxAngularVelocity(H, W, fx, fy, cx, cy, use_polarity=True)
            net.enable_cmax_r_update(est, lr=rc.cmax_lr)
        net.initialize_from_rotation(rc.initial_R)
    else:
        net = InteractingMaps(
            H=H, W=W, fx=fx, fy=fy, cx=cx, cy=cy,
            dist_coeffs=dist_coeffs,
            **rc.params
        )
        net.I = np.random.randn(H+1, W+1) * 0.001
        net.G = np.zeros((H, W, 2), dtype=np.float64)
        net.F = np.einsum('hwij,j->hwi', net._C_mat, rc.initial_R)
        net.R = rc.initial_R.copy()
    return net

def normalise(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-10)

def normalise_robust(x, lo_pct=1.0, hi_pct=99.0):
    """Percentile min-max to [0,1] for stable frame-to-frame display.

    Plain min-max maps the extreme pixels to 0/1, so a single outlier that
    moves between frames rescales the whole image -> the video flickers darker
    /brighter. Clipping to robust percentiles (1st/99th) fixes the display
    range against outliers, and because I is only defined up to a gauge
    (offset/scale) this also cancels that drift, keeping brightness steady.
    """
    lo, hi = np.percentile(x, [lo_pct, hi_pct])
    if hi - lo < 1e-9:
        return np.full_like(x, 0.5, dtype=np.float64)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)

def flow_to_rgb(flow, max_mag=None):
    """
    Colour-wheel rendering of a flow field: hue = direction, value = magnitude.

    max_mag fixes the magnitude normalisation. Without it each call normalises by
    its own max, so a sequence of frames (a GIF, or several models side by side)
    flickers and cannot be compared -- pass a shared max_mag in those cases.
    """
    from matplotlib.colors import hsv_to_rgb
    fx, fy = flow[..., 0], flow[..., 1]
    angle = (np.arctan2(fy, fx) + np.pi) / (2 * np.pi)
    mag = np.sqrt(fx**2 + fy**2)
    denom = (mag.max() if max_mag is None else max_mag) + 1e-10
    mag_norm = np.clip(mag / denom, 0.0, 1.0)
    hsv = np.stack([angle, np.ones_like(angle), mag_norm], axis=-1)
    return hsv_to_rgb(hsv)

def compute_metrics(omega_est, omega_gt):
    """Compute all metrics for a single frame."""
    err = np.linalg.norm(omega_est - omega_gt) * 180 / np.pi
    norm_est = np.linalg.norm(omega_est)
    norm_gt = np.linalg.norm(omega_gt)
    if norm_est > 1e-6 and norm_gt > 1e-6:
        cos_a = np.clip(np.dot(omega_est, omega_gt) / (norm_est * norm_gt), -1, 1)
        dir_err = np.degrees(np.arccos(cos_a))
        beta = norm_gt / norm_est
    else:
        dir_err = 180.0
        beta = 0.0
    return err, dir_err, beta


# ===========================================================================
# EXPERIMENT 1: Single-Frame Map Convergence
# ===========================================================================

def experiment_single_frame_convergence(rc: RunConfig, frame_idx=0, max_iters=100,
                                         snapshot_iters=None):
    """Maps at iteration checkpoints — visual inspection."""
    print("\n" + "="*70)
    print("EXPERIMENT 1: Single-Frame Map Convergence")
    print(f"  Config: {rc}")
    print("="*70)

    if snapshot_iters is None:
        snapshot_iters = [1, 3, 5, 10, 25, 50, max_iters]
    snapshot_iters = [i for i in snapshot_iters if i <= max_iters]

    seq = EventFrameSequence(
        rc.paths['events'], rc.paths['calib'],
        frame_duration=rc.frame_duration, t_start=rc.t_start,
        n_frames=frame_idx + 1, clip_value=10.0,
        undistort=rc.undistort_at_event_level,
        sensor_size=rc.sensor_size,
    )
    frames = list(seq)
    V, t_mid = frames[frame_idx]
    H, W = seq.H, seq.W
    fx, fy, cx, cy = seq.calib.fx, seq.calib.fy, seq.calib.cx, seq.calib.cy

    imu_data = load_imu(rc.paths['imu'])
    t_lo = rc.t_start + frame_idx * rc.frame_duration
    t_hi = t_lo + rc.frame_duration
    gt_omega = get_gyro_for_frame(imu_data, t_lo, t_hi)
    gt_R = gt_omega * rc.frame_duration

    print(f"  GT ω = ({gt_omega[0]:.3f}, {gt_omega[1]:.3f}, {gt_omega[2]:.3f}) rad/s")

    net = make_network(rc, H, W, fx, fy, cx, cy)
    snapshots = {}
    R_history = []
    res_history = []

    if rc.use_thesis:
        net.q_V.value = V
        for it in range(1, max_iters + 1):
            for q in [net.q_I, net.q_G, net.q_F, net.q_R]:
                q.reset_gradient()
            for cost in net.costs:
                if cost is net.cost_imu and not rc.use_imu:
                    continue
                cost.compute_and_send_gradients()
            net.q_I.update(1.0)
            net.q_G.update(1.0)
            net.q_F.update(1.0)
            net.q_R.update(1.0)
            net.q_I.value = np.clip(net.q_I.value, -10.0, 10.0)
            net.q_G.value = np.clip(net.q_G.value, -5.0, 5.0)
            net.q_F.value = np.clip(net.q_F.value, -10.0, 10.0)
            net.q_R.value = np.clip(net.q_R.value, -1.0, 1.0)

            R_history.append(net.R.copy())
            res_history.append(net.residual_VFG(V))

            if it in snapshot_iters:
                snapshots[it] = {
                    'I': net.I.copy(), 'G': net.G.copy(),
                    'F': net.F.copy(), 'R': net.R.copy(),
                }
    else:
        for it in range(1, max_iters + 1):
            net.update_R_from_FC()
            net.update_F_from_VG(V)
            net.update_G_from_VF(V)
            net.update_G_from_I()
            net.update_I_from_G()
            net.update_F_from_RC()

            R_history.append(net.R.copy())
            res_history.append(net.residual_VFG(V))

            if it in snapshot_iters:
                snapshots[it] = {
                    'I': net.I[:H, :W].copy(), 'G': net.G.copy(),
                    'F': net.F.copy(), 'R': net.R.copy(),
                }

    R_history = np.array(R_history)

    # --- Save plot ---
    rc.save_params()
    n_snaps = len(snapshot_iters)
    fig, axes = plt.subplots(5, n_snaps, figsize=(3 * n_snaps, 14))
    fig.suptitle(f'Exp 1: Map Convergence ({rc.model}, {rc.dataset})\n'
                 f'GT ω = ({gt_omega[0]:.3f}, {gt_omega[1]:.3f}, {gt_omega[2]:.3f}) rad/s',
                 fontsize=11)

    for col, it in enumerate(snapshot_iters):
        snap = snapshots[it]
        omega_it = snap['R'] / rc.frame_duration

        axes[0, col].imshow(V, cmap='RdBu', vmin=-1, vmax=1)
        axes[0, col].set_title(f'iter {it}', fontsize=9)
        axes[1, col].imshow(normalise(snap['I']), cmap='gray')
        G_mag = np.sqrt(snap['G'][..., 0]**2 + snap['G'][..., 1]**2)
        axes[2, col].imshow(normalise(G_mag), cmap='hot')
        axes[3, col].imshow(flow_to_rgb(snap['F']))

        err = np.linalg.norm(snap['R'] - gt_R) / rc.frame_duration * 180 / np.pi
        axes[4, col].bar(['ωx','ωy','ωz'], omega_it,
                        color=['#4e79a7','#f28e2b','#e15759'])
        axes[4, col].bar(['ωx','ωy','ωz'], gt_omega,
                        color='none', edgecolor='black', linewidth=2)
        axes[4, col].set_title(f'err={err:.1f}°/s', fontsize=8)

    row_labels = ['V (input)', 'I (intensity)', '|G| (gradient)', 'F (flow)', 'ω']
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=9)
    for ax in axes[:4].flat:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(rc.output_dir, 'exp1_convergence.png'), dpi=150)
    plt.close()
    print(f"  Saved: {rc.output_dir}/exp1_convergence.png")


# ===========================================================================
# EXPERIMENT 2: Multi-Frame Tracking
# ===========================================================================
def _net_step(net, rc: RunConfig, V, n_iters=None, omega_imu=None,
              win=None, cmax_est=None, state=None, on_iter=None):
    """
    Run ONE frame through the network, dispatching per model variant.

    This is the single place that knows how each of the five models must be
    driven. It previously lived inline in experiment_tracking while exp 1
    re-implemented the loop by hand -- and that hand-rolled copy never called
    Cost_CMax.set_frame(), so `thesis_cmax_v2` silently produced results with
    the CMax gradient disabled (its `_bearings is None` guard returns early).
    Every experiment now shares this function, so a model is driven identically
    everywhere.

    Parameters
    ----------
    win       : (N,4) raw events for this frame window (CMax variants only)
    cmax_est  : CMaxAngularVelocity for V1 (full solve per frame)
    state     : dict carrying cross-frame state; uses 'omega_cmax_prev' as the
                V1 warm start
    on_iter   : forwarded to net.step() -> per-iteration callback (D2 GIF)

    Returns
    -------
    (omega_est, aux) : omega_est in rad/s; aux carries e.g. the V1 CMax anchor.
    """
    n_iters = rc.n_iters if n_iters is None else n_iters
    state = {} if state is None else state
    aux = {}

    if getattr(rc, 'use_cmax_v2', False):
        # V2: CMax drives R from inside the message passing (needs raw events).
        net.step(V, n_iters=n_iters, events=win, on_iter=on_iter)
    elif cmax_est is not None:
        # V1: full CMax solve -> R anchor (reuses the Cost_IMU mechanism).
        t_ref = state.get('t_ref')
        omega_anchor = cmax_est.estimate(
            win, t_ref=t_ref, omega_init=state.get('omega_cmax_prev'))
        state['omega_cmax_prev'] = omega_anchor.copy()
        aux['omega_anchor'] = omega_anchor
        net.step(V, n_iters=n_iters, omega_imu=omega_anchor, on_iter=on_iter)
    elif rc.use_thesis and rc.use_imu:
        net.step(V, n_iters=n_iters, omega_imu=omega_imu, on_iter=on_iter)
    elif rc.use_thesis:
        net.step(V, n_iters=n_iters, on_iter=on_iter)
    else:
        # Cook network: no omega_imu/events kwargs.
        net.step(V, n_iters=n_iters, on_iter=on_iter)

    return net.R / rc.frame_duration, aux


def experiment_tracking(rc: RunConfig, save_frames=True):
    """
    Main experiment: ω tracking over time.
    ALWAYS saves:
      - tracking.csv (per-frame metrics)
      - tracking_plot.png (ω over time)
      - summary.json
      - params.json
    If save_frames=True (default):
      - video_frames/frame_XXXX.png (3-col: Events | I | GT)
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Multi-Frame Angular Velocity Tracking")
    print(f"  Config: {rc}")
    print("="*70)

    rc.save_params()

    seq = EventFrameSequence(
        rc.paths['events'], rc.paths['calib'],
        frame_duration=rc.frame_duration, t_start=rc.t_start,
        n_frames=rc.n_frames, clip_value=10.0,
        undistort=rc.undistort_at_event_level,
        sensor_size=rc.sensor_size,
    )
    H, W = seq.H, seq.W
    fx, fy, cx, cy = seq.calib.fx, seq.calib.fy, seq.calib.cx, seq.calib.cy
    imu_data = load_imu(rc.paths['imu'])            # model INPUT (gyro)
    gt_data  = load_groundtruth(rc.paths['groundtruth'])       # pose GT (RPG)
    omega_gt_data = load_omega_gt(rc.paths.get('omega_gt'))    # clean direct ω (ECRot)

    if omega_gt_data is not None:
        ref_src = 'omega_direct'
        print("  Scoring against omega_gt.txt (clean direct ω — synthetic, no differencing).")
    elif gt_data is not None and SCORE_AGAINST == 'groundtruth':
        ref_src = 'groundtruth'
        print("  Scoring against groundtruth.txt (poses, quaternion differencing).")
    else:
        ref_src = 'imu'
        print("  ⚠ Scoring against IMU gyro (no GT). For thesis_imu this is "
              "circular — the model is graded on its own input.")

    net = make_network(rc, H, W, fx, fy, cx, cy)

    # ─── CMax front-end (V1: full CMax per frame → R anchor) ──────────
    # B1: load the raw events separately and slice per window here, leaving
    # EventFrameSequence untouched. The gyro still provides frame-0 init only.
    cmax_est = None
    cmax_events = None
    omega_cmax_prev = np.zeros(3)
    omega_anchor = None
    _step_state = {}          # cross-frame state carried through _net_step
    if getattr(rc, 'use_cmax', False):
        from data_loader import load_events_fast, undistort_events
        dur = rc.n_frames * rc.frame_duration + 0.1
        cmax_events = undistort_events(
            load_events_fast(rc.paths['events'], t_start=rc.t_start, duration=dur),
            seq.calib)
        if getattr(rc, 'use_cmax_v2', False):
            print(f"  CMax V2 active — R driven by 1 CMax step / iteration "
                  f"(lr={rc.cmax_lr}, {len(cmax_events)} events).")
        else:
            # V1: standalone full-solve estimator, feeds the R anchor.
            from cmax import CMaxAngularVelocity
            cmax_est = CMaxAngularVelocity(H, W, fx, fy, cx, cy, use_polarity=True)
            print(f"  CMax V1 active — R anchored to full CMax ω per frame "
                  f"({len(cmax_events)} events).")

    # ─── IWE logging (final contrast-maximized IWE per frame) ─────────
    iwe_dir = None
    if getattr(rc, 'save_iwe', False) and getattr(rc, 'use_cmax', False):
        from cmax.iwe_io import save_iwe, plot_contrast_curve, make_gif
        iwe_dir = os.path.join(rc.output_dir, 'iwe')
        os.makedirs(iwe_dir, exist_ok=True)
        print(f"  Saving final IWE per frame to: {iwe_dir}/")

    # ─── Frame saving setup ───────────────────────────────────────────
    frames_dir = None
    gt_images = None
    if save_frames:
        frames_dir = os.path.join(rc.output_dir, 'video_frames')
        os.makedirs(frames_dir, exist_ok=True)
        gt_images = _try_load_gt_images(rc)
        print(f"  Saving frames to: {frames_dir}/")

    rows = []
    for k, (V, t_mid) in enumerate(seq):
        _t_frame0 = time_module.time()   # wall-clock per frame (runtime study)
        t_lo = rc.t_start + k * rc.frame_duration
        t_hi = t_lo + rc.frame_duration

        # Model INPUT: gyro (the sensor being fused). Independent of scoring.
        omega_imu = get_gyro_for_frame(imu_data, t_lo, t_hi)
        # SCORING reference: Vicon (independent), falls back to gyro if absent.
        omega_ref, _ = get_reference_omega(gt_data, imu_data, t_lo, t_hi, omega_gt_data)

        # Raw events for this window (V1 anchor and V2 in-loop both need them).
        win = None
        if cmax_events is not None:
            win = cmax_events[(cmax_events[:, 0] >= t_lo) & (cmax_events[:, 0] < t_hi)]

        # Single shared dispatch for all five model variants (see _net_step).
        _step_state['t_ref'] = 0.5 * (t_lo + t_hi)
        _step_state['omega_cmax_prev'] = omega_cmax_prev
        omega_est, _aux = _net_step(net, rc, V, n_iters=rc.n_iters,
                                    omega_imu=omega_imu, win=win,
                                    cmax_est=cmax_est, state=_step_state)
        if 'omega_anchor' in _aux:
            omega_anchor = _aux['omega_anchor']
            omega_cmax_prev = _step_state['omega_cmax_prev']
        err, dir_err, beta = compute_metrics(omega_est, omega_ref)

        # ─── Save the final contrast-maximized IWE for this frame ────
        if iwe_dir is not None and win is not None:
            if getattr(rc, 'use_cmax_v2', False):
                iwe, contrast = net.cost_cmax.build_current_iwe()   # at final R
                w_iwe = omega_est
            else:
                iwe = cmax_est.last_iwe                              # at CMax solve ω
                contrast = float(np.var(iwe)) if iwe is not None else 0.0
                w_iwe = omega_anchor
            if iwe is not None:
                save_iwe(iwe, w_iwe, contrast, iwe_dir, k, t_mid, len(win))

        rows.append({
            'frame': k, 'time': t_mid,
            'est_wx': omega_est[0], 'est_wy': omega_est[1], 'est_wz': omega_est[2],
            # gt_* is the SCORING reference (Vicon when available)
            'gt_wx': omega_ref[0], 'gt_wy': omega_ref[1], 'gt_wz': omega_ref[2],
            # imu_* is the gyro fed to the model (kept for comparison/debug)
            'imu_wx': omega_imu[0], 'imu_wy': omega_imu[1], 'imu_wz': omega_imu[2],
            'err_deg_s': err, 'dir_err_deg': dir_err, 'beta': beta,
            'ref_source': ref_src,
            't_frame_s': time_module.time() - _t_frame0,
        })

        # ─── Dump the full map set at the centre / last frame (D1) ───
        # ~2 MB per run; lets the maps panel (exp 10) be assembled later without
        # re-running the network.
        if getattr(rc, 'save_maps', True):
            which = None
            if k == rc.n_frames // 2:
                which = 'center'
            elif k == rc.n_frames - 1:
                which = 'final'
            if which is not None:
                _save_maps_npz(rc, which, V, net, H, W,
                               omega_est, omega_ref, t_mid, k)

        # ─── Save 3-column frame ─────────────────────────────────────
        if save_frames:
            _save_3col_frame(frames_dir, k, V, net, H, W, gt_images, rc)

        if (k + 1) % 10 == 0 or k == 0:
            print(f"  Frame {k+1:4d}/{rc.n_frames} | "
                  f"ω_est=({omega_est[0]:+.3f},{omega_est[1]:+.3f},{omega_est[2]:+.3f}) | "
                  f"err={err:.1f}°/s | dir={dir_err:.1f}°")

    # ─── Save CSV ─────────────────────────────────────────────────────
    csv_path = os.path.join(rc.output_dir, 'tracking.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # ─── IWE sequence artefacts (contrast curve + GIF) ────────────────
    if iwe_dir is not None:
        plot_contrast_curve(iwe_dir)
        make_gif(iwe_dir)
        print(f"  IWE: {rc.n_frames} PNGs + iwe_log.csv + contrast_curve.png "
              f"(+gif) → {iwe_dir}/")

    # ─── Summary ──────────────────────────────────────────────────────
    err_all = np.array([r['err_deg_s'] for r in rows])
    dir_all = np.array([r['dir_err_deg'] for r in rows])
    beta_all = np.array([r['beta'] for r in rows])

    summary = {
        'mean_err_deg_s': float(np.mean(err_all)),
        'median_err_deg_s': float(np.median(err_all)),
        'mean_dir_err_deg': float(np.mean(dir_all)),
        'median_dir_err_deg': float(np.median(dir_all)),
        'mean_beta': float(np.mean(beta_all)),
        'std_beta': float(np.std(beta_all)),
        # --- added for the report table (D3 needs min/max across frames) ---
        'min_err_deg_s': float(np.min(err_all)),
        'max_err_deg_s': float(np.max(err_all)),
        'std_err_deg_s': float(np.std(err_all)),
        'p90_err_deg_s': float(np.percentile(err_all, 90)),
        'median_beta': float(np.median(beta_all)),
        # error on the LAST frame: drift indicator (does it diverge over time?)
        'final_err_deg_s': float(err_all[-1]),
        'n_frames_run': int(len(err_all)),
        'ref_source': rows[0].get('ref_source', 'unknown'),
    }
    # Timing (present when the per-frame loop recorded it; see t_frame_s).
    if 't_frame_s' in rows[0]:
        t_all = np.array([r['t_frame_s'] for r in rows])
        summary['mean_frame_time_s'] = float(np.mean(t_all))
        summary['total_runtime_s'] = float(np.sum(t_all))

    with open(os.path.join(rc.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  {'='*50}")
    print(f"  RESULTS: {rc.model}, {rc.dataset}, {rc.duration_s:.1f}s")
    print(f"  {'='*50}")
    print(f"  Total error:     mean={summary['mean_err_deg_s']:.1f}°/s, "
          f"median={summary['median_err_deg_s']:.1f}°/s")
    print(f"  Direction error: mean={summary['mean_dir_err_deg']:.1f}°, "
          f"median={summary['median_dir_err_deg']:.1f}°")
    print(f"  Scale factor β:  mean={summary['mean_beta']:.2f} ± {summary['std_beta']:.2f}")
    if save_frames:
        print(f"  Frames saved:    {frames_dir}/ ({rc.n_frames} PNGs)")

    _plot_tracking(rows, rc)
    return summary

def _plot_tracking(rows, rc: RunConfig):
    """Tracking plot with ω components + error."""
    times = np.array([r['time'] for r in rows]) - rows[0]['time']
    omega_est = np.array([[r['est_wx'], r['est_wy'], r['est_wz']] for r in rows])
    omega_gt = np.array([[r['gt_wx'], r['gt_wy'], r['gt_wz']] for r in rows])
    err = np.array([r['err_deg_s'] for r in rows])

    ref_src = rows[0].get('ref_source', 'imu')
    ref_label = 'GT (Vicon quat)' if ref_src == 'groundtruth' else 'GT (IMU)'
    # Only overlay the gyro separately when it is NOT the scoring reference.
    has_imu_col = 'imu_wx' in rows[0]
    omega_imu = (np.array([[r['imu_wx'], r['imu_wy'], r['imu_wz']] for r in rows])
                 if has_imu_col else None)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'ω Tracking — {rc.dataset} ({rc.model})\n'
                 f'dt={rc.frame_duration*1000:.0f}ms, {rc.n_iters} iters, '
                 f'{rc.duration_s:.1f}s  |  scored vs {ref_label}', fontsize=11)

    labels = ['ωx', 'ωy', 'ωz']
    colors = ['#4e79a7', '#f28e2b', '#e15759']

    for i in range(3):
        axes[i].plot(times, omega_gt[:, i], 'k-', lw=1.0, alpha=0.8, label=ref_label)
        if omega_imu is not None and ref_src == 'groundtruth':
            axes[i].plot(times, omega_imu[:, i], color='#59a14f', lw=0.8, ls=':',
                         alpha=0.7, label='IMU gyro (model input)')
        axes[i].plot(times, omega_est[:, i], color=colors[i], lw=1.5, label='Estimated')
        axes[i].set_ylabel(f'{labels[i]} (rad/s)')
        axes[i].legend(loc='upper right', fontsize=8)
        axes[i].grid(True, alpha=0.3)

    axes[3].plot(times, err, 'r-', lw=1.0)
    axes[3].set_ylabel('Error (°/s)')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_title(f'Error: mean={np.mean(err):.1f}°/s, median={np.median(err):.1f}°/s')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(rc.output_dir, 'tracking_plot.png'), dpi=150)
    plt.close()

def experiment_distortion_ablation(rc: RunConfig, save_frames=False):
    """
    EXPERIMENT 9: run the SAME tracking config under both distortion models
    and tabulate metrics.
        undistort_events : events undistorted, pinhole C        (baseline)
        C_full           : distortion in C, directions + Jacobian
    """
    print("\n" + "="*70)
    print("EXPERIMENT 9: Distortion-Handling Ablation")
    print(f"  Base config: {rc}")
    print("="*70)

    modes = ['undistort_events', 'C_full']
    rows = []
    for mode in modes:
        rc_m = RunConfig(
            dataset=rc.dataset, model=rc.model, segment=rc.segment,
            frame_duration=rc.frame_duration, n_frames=rc.n_frames,
            n_iters=rc.n_iters,
            delta_IMU=(rc.delta_IMU if rc.use_imu else None),
            delta_FR=rc.params.get('delta_FR'),
            distortion_mode=mode,
        )
        summary = experiment_tracking(rc_m, save_frames=save_frames)
        rows.append({
            'mode': mode,
            'mean_err_deg_s': summary['mean_err_deg_s'],
            'median_err_deg_s': summary['median_err_deg_s'],
            'mean_dir_err_deg': summary['mean_dir_err_deg'],
            'mean_beta': summary['mean_beta'],
            'std_beta': summary['std_beta'],
        })

    print("\n\n" + "="*80)
    print(f"DISTORTION ABLATION — {rc.dataset}/{rc.segment_id}, model={rc.model}")
    print("="*80)
    print(f"{'mode':<18} {'err°/s':>8} {'med°/s':>8} {'dir°':>7} {'β':>6} {'σβ':>6}")
    print("-"*80)
    for r in rows:
        print(f"{r['mode']:<18} {r['mean_err_deg_s']:>8.1f} "
              f"{r['median_err_deg_s']:>8.1f} {r['mean_dir_err_deg']:>7.1f} "
              f"{r['mean_beta']:>6.2f} {r['std_beta']:>6.2f}")

    out = os.path.join('results', 'ablation_distortion')
    os.makedirs(out, exist_ok=True)
    csv_path = os.path.join(out, f"{rc.dataset}_{rc.segment_id}_{rc.model}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nAblation CSV saved: {csv_path}")
    return rows

def experiment_distortion_ablation_all_segments(dataset, model, save_frames=False,
                                                use_best=False):
    """
    Run the distortion ablation over EVERY segment of a dataset and emit one
    combined CSV (dataset × segment × mode) plus a grouped summary table.

    If use_best=True, each segment's base config is taken from best_config.py
    (best grid params for that dataset/segment/model) instead of the defaults.
    """
    print("\n" + "="*80)
    print(f"EXPERIMENT 9 (ALL SEGMENTS): Distortion Ablation — {dataset}, model={model}")
    print("="*80)

    segs = DATASET_SEGMENTS[dataset]
    all_rows = []
    for seg in segs:
        best_kw = resolve_best_kwargs(dataset, seg['id'], model) if use_best else {}
        rc = RunConfig(dataset=dataset, model=model, segment=seg['id'],
                       distortion_mode=None, **best_kw)  # per-mode override happens inside
        rows = experiment_distortion_ablation(rc, save_frames=save_frames)
        for r in rows:
            all_rows.append({'dataset': dataset, 'segment_id': seg['id'],
                             'model': model, **r})

    # ─── grouped summary table ──────────────────────────────────────────
    print("\n\n" + "="*92)
    print(f"COMBINED DISTORTION ABLATION — {dataset}, model={model}")
    print("="*92)
    print(f"{'segment':<8} {'mode':<18} {'err°/s':>8} {'med°/s':>8} "
          f"{'dir°':>7} {'β':>6} {'σβ':>6}")
    print("-"*92)
    last_seg = None
    for r in all_rows:
        if r['segment_id'] != last_seg:
            print("-"*92)
            last_seg = r['segment_id']
        print(f"{r['segment_id']:<8} {r['mode']:<18} {r['mean_err_deg_s']:>8.1f} "
              f"{r['median_err_deg_s']:>8.1f} {r['mean_dir_err_deg']:>7.1f} "
              f"{r['mean_beta']:>6.2f} {r['std_beta']:>6.2f}")

    out = os.path.join('results', 'ablation_distortion')
    os.makedirs(out, exist_ok=True)
    csv_path = os.path.join(out, f"{dataset}_ALLSEG_{model}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nCombined ablation CSV saved: {csv_path}")
    return all_rows

# ===========================================================================
# EXPERIMENT 3: Parameter Influence
# ===========================================================================

def experiment_parameter_influence(rc: RunConfig, frame_idx=0):
    """Iterations sweep + frame duration sweep."""
    print("\n" + "="*70)
    print("EXPERIMENT 3: Parameter Influence")
    print(f"  Config: {rc}")
    print("="*70)

    rc.save_params()
    imu_data = load_imu(rc.paths['imu'])

    # (a) Iterations sweep
    print("\n  (a) Iterations sweep:")
    iter_counts = [1, 3, 5, 10, 20, 50, 100]
    iter_results = []

    seq = EventFrameSequence(
        rc.paths['events'], rc.paths['calib'],
        frame_duration=rc.frame_duration, t_start=rc.t_start,
        n_frames=frame_idx + 1, clip_value=10.0, undistort=rc.undistort_at_event_level, sensor_size=rc.sensor_size,
    )
    frames = list(seq)
    V, _ = frames[frame_idx]
    H, W = seq.H, seq.W
    fx, fy, cx, cy = seq.calib.fx, seq.calib.fy, seq.calib.cx, seq.calib.cy

    t_lo = rc.t_start + frame_idx * rc.frame_duration
    t_hi = t_lo + rc.frame_duration
    gt_omega = get_gyro_for_frame(imu_data, t_lo, t_hi)

    for n_iters in iter_counts:
        net = make_network(rc, H, W, fx, fy, cx, cy)
        if rc.use_thesis and rc.use_imu:
            net.step(V, n_iters=n_iters, omega_imu=gt_omega)
        else:
            net.step(V, n_iters=n_iters)
        omega_est = net.R / rc.frame_duration
        err, dir_err, beta = compute_metrics(omega_est, gt_omega)
        iter_results.append({'iters': n_iters, 'err': err, 'dir': dir_err, 'beta': beta})
        print(f"    iters={n_iters:4d} | err={err:.2f}°/s | dir={dir_err:.1f}°")

    # (b) Frame duration sweep
    print("\n  (b) Frame duration sweep:")
    durations = [0.005, 0.010, 0.015, 0.020, 0.030, 0.050]
    dt_results = []

    for dt in durations:
        seq2 = EventFrameSequence(
            rc.paths['events'], rc.paths['calib'],
            frame_duration=dt, t_start=rc.t_start,
            n_frames=frame_idx + 1, clip_value=10.0, undistort=rc.undistort_at_event_level, sensor_size=rc.sensor_size,
        )
        frames2 = list(seq2)
        V2, _ = frames2[frame_idx]
        H2, W2 = seq2.H, seq2.W

        t_lo2 = rc.t_start + frame_idx * dt
        t_hi2 = t_lo2 + dt
        gt_omega2 = get_gyro_for_frame(imu_data, t_lo2, t_hi2)

        # Temporarily override frame_duration for make_network
        orig_dt = rc.frame_duration
        rc.frame_duration = dt
        net = make_network(rc, H2, W2, fx, fy, cx, cy)
        rc.frame_duration = orig_dt

        if rc.use_thesis and rc.use_imu:
            net.step(V2, n_iters=rc.n_iters, omega_imu=gt_omega2)
        else:
            net.step(V2, n_iters=rc.n_iters)

        omega_est2 = net.R / dt
        err2, dir2, beta2 = compute_metrics(omega_est2, gt_omega2)
        n_active = np.count_nonzero(V2)
        dt_results.append({'dt_ms': dt*1000, 'err': err2, 'active': n_active})
        print(f"    dt={dt*1000:5.1f}ms | active={n_active:5d} | err={err2:.2f}°/s")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Exp 3: Parameter Influence ({rc.model}, {rc.dataset})', fontsize=11)

    axes[0].plot([r['iters'] for r in iter_results],
                [r['err'] for r in iter_results], 'bo-', lw=1.5)
    axes[0].set_xlabel('Iterations per frame')
    axes[0].set_ylabel('ω error (°/s)')
    axes[0].set_title(f'(a) Iterations (dt={rc.frame_duration*1000:.0f}ms)')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')

    axes[1].plot([r['dt_ms'] for r in dt_results],
                [r['err'] for r in dt_results], 'rs-', lw=1.5)
    axes[1].set_xlabel('Frame duration (ms)')
    axes[1].set_ylabel('ω error (°/s)')
    axes[1].set_title(f'(b) Frame duration ({rc.n_iters} iters)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(rc.output_dir, 'exp3_parameters.png'), dpi=150)
    plt.close()


# ===========================================================================
# EXPERIMENT 4: Qualitative Video
# ===========================================================================

def experiment_qualitative_video(rc: RunConfig):
    print("  (Exp 4 is now integrated into Exp 2 — running tracking with frames)")
    return experiment_tracking(rc, save_frames=True)


def _try_load_gt_images(rc: RunConfig):
    """Load GT APS images from images.txt."""
    data_dir = rc.paths['data_dir']
    images_file = os.path.join(data_dir, 'images.txt')

    if not os.path.exists(images_file):
        print("  No images.txt found")
        return None

    img_list = []
    t_end = rc.t_start + rc.n_frames * rc.frame_duration + 0.5

    with open(images_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            t = float(parts[0])
            fname = parts[1]

            if t < rc.t_start - 0.5:
                continue
            if t > t_end:
                break

            img_path = os.path.join(data_dir, fname)
            if os.path.exists(img_path):
                img_list.append((t, img_path))

    if img_list:
        print(f"  Loaded {len(img_list)} GT images")
    else:
        print("  No GT images found in time range")
        return None
    return img_list


def _save_maps_npz(rc: RunConfig, which, V, net, H, W,
                   omega_est, omega_ref, t_mid, frame_idx):
    """
    Dump the full inferred map set for one frame -> {output_dir}/maps_{which}.npz.

    Written during experiment_tracking so the maps panel (exp 10) can be
    assembled afterwards without re-running any network. `which` is 'center' or
    'final'. I is cropped to (H, W) because the Cook network carries an
    (H+1, W+1) intensity map (extra row/col for the forward differences).
    """
    os.makedirs(rc.output_dir, exist_ok=True)
    I = net.I[:H, :W] if net.I.shape != (H, W) else net.I
    np.savez_compressed(
        os.path.join(rc.output_dir, f'maps_{which}.npz'),
        V=V.astype(np.float32), I=np.asarray(I, np.float32),
        G=np.asarray(net.G, np.float32), F=np.asarray(net.F, np.float32),
        R=np.asarray(net.R, np.float64),
        omega_est=np.asarray(omega_est, np.float64),
        omega_ref=np.asarray(omega_ref, np.float64),
        t_mid=float(t_mid), frame_idx=int(frame_idx),
        dataset=rc.dataset, model=rc.model, segment_id=rc.segment_id,
    )


def _save_3col_frame(out_dir, k, V, net, H, W, gt_images, rc: RunConfig):
    """Fixed-size 3-column frame."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(V, cmap='RdBu', vmin=-1, vmax=1)
    axes[0].set_title('Events (V)', fontsize=10)
    axes[0].axis('off')

    I_disp = net.I if net.I.shape == (H, W) else net.I[:H, :W]
    axes[1].imshow(normalise_robust(I_disp), cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Estimated I', fontsize=10)
    axes[1].axis('off')

    if gt_images is not None and len(gt_images) > 0:
        t_frame = rc.t_start + k * rc.frame_duration + rc.frame_duration / 2
        closest = min(gt_images, key=lambda x: abs(x[0] - t_frame))
        gt_img = plt.imread(closest[1])
        axes[2].imshow(gt_img, cmap='gray')
        axes[2].set_title('GT Image (APS)', fontsize=10)
    else:
        axes[2].text(0.5, 0.5, 'No GT', ha='center', va='center',
                     fontsize=14, transform=axes[2].transAxes)
        axes[2].set_facecolor('#f0f0f0')
        axes[2].set_title('GT (N/A)', fontsize=10)
    axes[2].axis('off')

    plt.suptitle(f'Frame {k:04d}', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'frame_{k:04d}.png'), dpi=100)
    plt.close(fig)


# ===========================================================================
# EXPERIMENT 5: Make Video
# ===========================================================================
# ===========================================================================
# EXPERIMENT 5: Make Videos (batch-fähig)
# ===========================================================================

def experiment_make_videos_batch(base_dir='results', fps=15):
    """Find ALL video_frames/ directories and build videos."""
    import glob
    import imageio
    from PIL import Image

    pattern = os.path.join(base_dir, '**', 'video_frames')
    frame_dirs = sorted(glob.glob(pattern, recursive=True))

    if not frame_dirs:
        print(f"No video_frames/ directories found under {base_dir}")
        return

    print(f"Found {len(frame_dirs)} frame directories")

    for frames_dir in frame_dirs:
        frame_files = sorted(glob.glob(os.path.join(frames_dir, 'frame_*.png')))
        if not frame_files:
            continue

        output_path = os.path.join(os.path.dirname(frames_dir), 'comparison.mp4')
        if os.path.exists(output_path):
            print(f"  Skip (exists): {output_path}")
            continue

        first = Image.open(frame_files[0])
        w, h = first.size
        w = (w // 16) * 16
        h = (h // 16) * 16

        writer = imageio.get_writer(output_path, fps=fps)
        for f in frame_files:
            img = Image.open(f).resize((w, h), Image.LANCZOS)
            writer.append_data(np.array(img))
        writer.close()
        print(f"  Video: {output_path} ({len(frame_files)} frames)")

    print("Done.")

def experiment_make_video(rc: RunConfig, fps=15):
    """Assemble video from saved frames."""
    import imageio
    import glob
    from PIL import Image

    frames_dir = os.path.join(rc.output_dir, 'video_frames')
    output_path = os.path.join(rc.output_dir, 'comparison.mp4')

    frame_files = sorted(glob.glob(os.path.join(frames_dir, 'frame_*.png')))
    if not frame_files:
        print(f"No frames found in {frames_dir}")
        return

    first = Image.open(frame_files[0])
    w, h = first.size
    w = (w // 16) * 16
    h = (h // 16) * 16

    writer = imageio.get_writer(output_path, fps=fps)
    for f in frame_files:
        img = Image.open(f).resize((w, h), Image.LANCZOS)
        writer.append_data(np.array(img))
    writer.close()
    print(f"Video saved: {output_path} ({w}×{h}, {len(frame_files)} frames)")


# ===========================================================================
# EXPERIMENT 6: Basin of Attraction
# ===========================================================================

def experiment_basin_of_attraction(rc: RunConfig, frame_idx=0):
    """Initialize R at various distances from GT — does it converge?"""
    print("\n" + "="*70)
    print("EXPERIMENT 6: Basin of Attraction")
    print(f"  Config: {rc}")
    print("="*70)

    rc.save_params()

    seq = EventFrameSequence(
        rc.paths['events'], rc.paths['calib'],
        frame_duration=rc.frame_duration, t_start=rc.t_start,
        n_frames=frame_idx + 1, clip_value=10.0, undistort=rc.undistort_at_event_level, sensor_size=rc.sensor_size,
    )
    frames = list(seq)
    V, _ = frames[frame_idx]
    H, W = seq.H, seq.W
    fx, fy, cx, cy = seq.calib.fx, seq.calib.fy, seq.calib.cx, seq.calib.cy

    imu_data = load_imu(rc.paths['imu'])
    t_lo = rc.t_start + frame_idx * rc.frame_duration
    t_hi = t_lo + rc.frame_duration
    gt_omega = get_gyro_for_frame(imu_data, t_lo, t_hi)
    gt_R = gt_omega * rc.frame_duration
    scale = max(np.linalg.norm(gt_R), 0.01)

    perturbations = [
        ("Exact GT", gt_R.copy()),
        ("GT + 10%", gt_R + 0.1 * scale * np.random.randn(3)),
        ("GT + 50%", gt_R + 0.5 * scale * np.random.randn(3)),
        ("GT + 100%", gt_R + 1.0 * scale * np.random.randn(3)),
        ("GT + 200%", gt_R + 2.0 * scale * np.random.randn(3)),
        ("Opposite", -gt_R),
        ("Zero", np.zeros(3)),
        ("Random", np.random.randn(3) * 0.05),
    ]

    results = []
    for label, R_init in perturbations:
        # Override initial_R temporarily
        orig_R = rc.initial_R.copy()
        rc.initial_R = R_init
        net = make_network(rc, H, W, fx, fy, cx, cy)
        rc.initial_R = orig_R

        if rc.use_thesis and rc.use_imu:
            net.step(V, n_iters=rc.n_iters, omega_imu=gt_omega)
        else:
            net.step(V, n_iters=rc.n_iters)

        final_R = net.R.copy()
        init_dist = np.linalg.norm(R_init - gt_R)
        final_dist = np.linalg.norm(final_R - gt_R)
        converged = final_dist < 0.5 * scale

        results.append({'label': label, 'init_dist': init_dist,
                       'final_dist': final_dist, 'converged': converged})
        print(f"  {label:15s} | init={init_dist:.5f} | final={final_dist:.5f} | "
              f"{'✓' if converged else '✗'}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results:
        color = 'green' if r['converged'] else 'red'
        ax.scatter(r['init_dist'], r['final_dist'], c=color, s=100, zorder=5)
        ax.annotate(r['label'], (r['init_dist'], r['final_dist']), fontsize=8)
    max_d = max(r['init_dist'] for r in results) * 1.1
    ax.plot([0, max_d], [0, max_d], 'k--', alpha=0.3, label='no improvement')
    ax.set_xlabel('Initial ||R - R_gt||')
    ax.set_ylabel('Final ||R - R_gt||')
    ax.set_title(f'Basin of Attraction ({rc.model}, {rc.dataset})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(rc.output_dir, 'exp6_basin.png'), dpi=150)
    plt.close()


# ===========================================================================
# EXPERIMENT 7: Full Systematic Evaluation
# ===========================================================================

def experiment_full_evaluation(dataset_filter=None, model_filter=None, save_frames=True):
    """
    Läuft über ALL datasets × ALL models × ALL SEGMENTS.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 7: Full Systematic Evaluation (All Segments)")
    print("="*70)

    # Datasets auswählen
    if dataset_filter:
        datasets = {dataset_filter: DATASET_SEGMENTS[dataset_filter]}
    else:
        datasets = DATASET_SEGMENTS

    # Models auswählen (includes the CMax variants; see resolve_models)
    models = resolve_models(model_filter)

    all_results = []

    for dataset, segments in datasets.items():
        for seg in segments:                          # ← NEU: innere Schleife
            for model in models:
                print(f"\n--- {dataset} / {seg['id']} / {model} ---")
                try:
                    rc = RunConfig(dataset=dataset, model=model, segment=seg)
                    summary = experiment_tracking(rc, save_frames=save_frames)
                    summary['dataset'] = dataset
                    summary['segment_id'] = seg['id']  # ← NEU
                    summary['model'] = model
                    summary['duration_s'] = rc.duration_s
                    all_results.append(summary)
                except Exception as e:
                    print(f"  FAILED: {e}")
                    all_results.append({
                        'dataset': dataset, 'segment_id': seg['id'],
                        'model': model,
                        'mean_err_deg_s': float('nan'),
                        'mean_dir_err_deg': float('nan'),
                        'mean_beta': float('nan'),
                        'duration_s': 0,
                    })

    # Tabelle ausgeben
    print("\n\n" + "="*90)
    print("FULL EVALUATION RESULTS")
    print("="*90)
    print(f"{'Dataset':<18} {'Segment':<8} {'Model':<12} "
          f"{'ω err°/s':<10} {'Dir err°':<10} {'β':<8} {'Dur(s)':<6}")
    print("-"*90)
    for r in all_results:
        print(f"{r['dataset']:<18} {r.get('segment_id','?'):<8} {r['model']:<12} "
              f"{r.get('mean_err_deg_s', float('nan')):>7.1f}   "
              f"{r.get('mean_dir_err_deg', float('nan')):>7.1f}   "
              f"{r.get('mean_beta', float('nan')):>6.2f}  "
              f"{r.get('duration_s', 0):>5.1f}")

    # CSV speichern
    os.makedirs('results', exist_ok=True)
    csv_path = 'results/full_evaluation.csv'
    if all_results:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nMaster table saved: {csv_path}")

    return all_results

# ===========================================================================
# EXPERIMENT 8: Automatic Parameter Grid Sweep
# ===========================================================================
def experiment_parameter_grid(dataset_filter=None, model_filter=None,
                              all_segments=False, save_frames=True,
                              distortion_mode=None):
    """
    Sweep over:
      - frame_duration: [0.010, 0.020, 0.030]
      - n_frames: [25, 50, 150]
      - n_iters: [75, 100]
      - delta_FR: [0.10, 0.20, 0.30, 0.50]
      - delta_IMU: [0.10, 0.20, 0.30, 0.50]  (only for thesis_imu)
    
    For each config: runs tracking (Exp 2) AND single-frame convergence (Exp 1).
    """
    print("\n" + "="*70)
    print("EXPERIMENT 8: Automatic Parameter Grid Sweep")
    print("="*70)

    # ─── GRID DEFINITION ──────────────────────────────────────────────
    # Trimmed per sensitivity analysis of the previous grid:
    #   - n_iters: FIXED at 75 (75 vs 100 moved error by <0.2 deg/s → inert)
    #   - frame_duration: NOW SWEPT (was silently fixed at 0.020 before; it is
    #     the physically most important untested axis — sets flow magnitude/frame)
    #   - delta_FR: [0.1, 0.2, 0.5] (inert for pure vision; matters only for imu)
    #   - delta_IMU: [0.1, 0.2, 0.5] (main knob for thesis_imu)
    grid = {
        'frame_duration': [0.010, 0.020, 0.050],
        'n_frames':  [25, 150],       # short-track accuracy + long-track drift/reversal
        'n_iters':   [75],
        'delta_FR':  [0.10, 0.20, 0.50],
        'delta_IMU': [0.10, 0.20, 0.50],
    }

    # ─── DATASETS ─────────────────────────────────────────────────────
    if dataset_filter:
        datasets = [dataset_filter]
    else:
        datasets = list(DATASET_SEGMENTS.keys())

    # ─── MODELS ───────────────────────────────────────────────────────
    # Grid sweeps stay on the classic 3 by default: the CMax models are ~1.6x
    # slower and delta_IMU/delta_FR are not their governing knobs (use exp 13).
    models = resolve_models(model_filter or 'classic')

    all_results = []
    run_idx = 0

    # Count total runs
    total_runs = 0
    for dataset in datasets:
        n_segs = len(DATASET_SEGMENTS[dataset]) if all_segments else 1
        for model in models:
            n_imu = len(grid['delta_IMU']) if model == 'thesis_imu' else 1
            total_runs += (n_segs * len(grid['frame_duration']) * 
                          len(grid['n_frames']) * len(grid['n_iters']) * 
                          len(grid['delta_FR']) * n_imu)

    print(f"  Datasets: {datasets}")
    print(f"  Models:   {models}")
    print(f"  Segments: {'ALL' if all_segments else 'first only'}")
    print(f"  Distortion mode: {distortion_mode or DISTORTION_MODE}")
    print(f"  Grid axes: {list(grid.keys())}")
    print(f"  Total runs: {total_runs}")
    print()

    for dataset in datasets:
        if all_segments:
            segs = DATASET_SEGMENTS[dataset]
        else:
            segs = [DATASET_SEGMENTS[dataset][0]]

        for seg in segs:
            for model in models:
                imu_values = grid['delta_IMU'] if model == 'thesis_imu' else [None]

                for frame_duration in grid['frame_duration']:
                    for n_frames in grid['n_frames']:
                        for n_iters in grid['n_iters']:
                            for delta_FR in grid['delta_FR']:
                                for delta_IMU in imu_values:
                                    run_idx += 1
                                    imu_str = f", δ_IMU={delta_IMU}" if delta_IMU else ""
                                    print(f"\n{'─'*60}")
                                    print(f"  RUN {run_idx}/{total_runs}: "
                                          f"{dataset}/{seg['id']}/{model}")
                                    print(f"  dt={frame_duration*1000:.0f}ms, "
                                          f"n_frames={n_frames}, n_iters={n_iters}, "
                                          f"δ_FR={delta_FR}{imu_str}")
                                    print(f"{'─'*60}")

                                    try:
                                        rc = RunConfig(
                                            dataset=dataset,
                                            model=model,
                                            segment=seg,
                                            frame_duration=frame_duration,
                                            n_frames=n_frames,
                                            n_iters=n_iters,
                                            delta_IMU=delta_IMU,
                                            distortion_mode=distortion_mode,
                                        )
                                        rc.params['delta_FR'] = delta_FR

                                        # ── Run Exp 2 (tracking) ──
                                        summary = experiment_tracking(
                                            rc, save_frames=save_frames)

                                        # ── Run Exp 1 (convergence diagnostic) ──
                                        if save_frames:
                                            experiment_single_frame_convergence(
                                                rc, frame_idx=0, 
                                                max_iters=n_iters)

                                        result = {
                                            'dataset': dataset,
                                            'segment_id': seg['id'],
                                            'model': model,
                                            'distortion_mode': rc.distortion_mode,
                                            'frame_duration': frame_duration,
                                            'n_frames': n_frames,
                                            'n_iters': n_iters,
                                            'delta_FR': delta_FR,
                                            'delta_IMU': delta_IMU if delta_IMU else 0.0,
                                            'duration_s': n_frames * frame_duration,
                                            'mean_err_deg_s': summary['mean_err_deg_s'],
                                            'median_err_deg_s': summary['median_err_deg_s'],
                                            'mean_dir_err_deg': summary['mean_dir_err_deg'],
                                            'mean_beta': summary['mean_beta'],
                                        }
                                        all_results.append(result)

                                    except Exception as e:
                                        print(f"  FAILED: {e}")
                                        all_results.append({
                                            'dataset': dataset,
                                            'segment_id': seg['id'],
                                            'model': model,
                                            'distortion_mode': distortion_mode or DISTORTION_MODE,
                                            'frame_duration': frame_duration,
                                            'n_frames': n_frames,
                                            'n_iters': n_iters,
                                            'delta_FR': delta_FR,
                                            'delta_IMU': delta_IMU if delta_IMU else 0.0,
                                            'duration_s': n_frames * frame_duration,
                                            'mean_err_deg_s': float('nan'),
                                            'median_err_deg_s': float('nan'),
                                            'mean_dir_err_deg': float('nan'),
                                            'mean_beta': float('nan'),
                                        })

    # ─── RESULTS TABLE ─────────────────────────────────────────────────
    print("\n\n" + "="*130)
    print("PARAMETER GRID RESULTS")
    print("="*130)
    print(f"{'Dataset':<16} {'Seg':<6} {'Model':<12} {'dt_ms':>5} {'n_fr':>5} "
          f"{'iters':>5} {'δ_FR':>5} {'δ_IMU':>6} | "
          f"{'err°/s':>7} {'med°/s':>7} {'dir°':>6} {'β':>5}")
    print("-"*130)
    for r in all_results:
        print(f"{r['dataset']:<16} {r['segment_id']:<6} {r['model']:<12} "
              f"{r['frame_duration']*1000:>5.0f} {r['n_frames']:>5} "
              f"{r['n_iters']:>5} {r['delta_FR']:>5.2f} "
              f"{r['delta_IMU']:>6.2f} | "
              f"{r['mean_err_deg_s']:>7.1f} {r['median_err_deg_s']:>7.1f} "
              f"{r['mean_dir_err_deg']:>6.1f} {r['mean_beta']:>5.2f}")

    # ─── SAVE CSV ──────────────────────────────────────────────────────
    os.makedirs('results', exist_ok=True)
    if dataset_filter:
        csv_path = f'results/parameter_grid_{dataset_filter}.csv'
    else:
        csv_path = 'results/parameter_grid.csv'

    if all_results:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nGrid results saved: {csv_path}")

    # ─── FIND BEST PER MODEL ──────────────────────────────────────────
    valid = [r for r in all_results if not np.isnan(r['mean_err_deg_s'])]
    if valid:
        print(f"\n  ★ BEST CONFIG PER MODEL:")
        for m in models:
            m_valid = [r for r in valid if r['model'] == m]
            if m_valid:
                best = min(m_valid, key=lambda r: r['mean_err_deg_s'])
                imu_str = (f", δ_IMU={best['delta_IMU']:.2f}" 
                          if m == 'thesis_imu' else "")
                print(f"    [{m:12s}] {best['dataset']}/{best['segment_id']}, "
                      f"dt={best['frame_duration']*1000:.0f}ms, "
                      f"n={best['n_frames']}, i={best['n_iters']}, "
                      f"δ_FR={best['delta_FR']:.2f}{imu_str} "
                      f"→ {best['mean_err_deg_s']:.1f}°/s")

    return all_results


# ===========================================================================
# Main
# ===========================================================================


# ===========================================================================
# EXPERIMENT 12: report results table (all models x datasets x segments)
# ===========================================================================

# Fixed CSV header. Written up-front and used with extrasaction='ignore' so a
# failure on the FIRST run can no longer truncate the header (the old exp 7
# derived fieldnames from all_results[0].keys(), which then made every later
# full row raise).
RUN_FIELDS = (
    'dataset', 'segment_id', 'model', 'status',
    'n_frames', 'frame_duration', 'duration_s', 'n_iters',
    'delta_FR', 'delta_IMU', 'cmax_lr', 'distortion_mode', 'ref_source',
    'omega_gt_mag_mean',
    'mean_err_deg_s', 'median_err_deg_s', 'std_err_deg_s',
    'min_err_deg_s', 'max_err_deg_s', 'p90_err_deg_s', 'final_err_deg_s',
    'mean_dir_err_deg', 'median_dir_err_deg',
    'mean_beta', 'std_beta', 'median_beta',
    'mean_frame_time_s', 'total_runtime_s',
)


def _omega_gt_mag_mean(out_dir):
    """Mean |omega_ref| over the run, read back from tracking.csv."""
    path = os.path.join(out_dir, 'tracking.csv')
    if not os.path.exists(path):
        return float('nan')
    try:
        with open(path) as f:
            mags = [np.linalg.norm([float(r['gt_wx']), float(r['gt_wy']),
                                    float(r['gt_wz'])])
                    for r in csv.DictReader(f)]
        return float(np.mean(mags)) if mags else float('nan')
    except Exception:
        return float('nan')


def experiment_report_table(dataset_filter=None, models=None, segments='all',
                            n_frames=25, save_frames=False, resume=True,
                            distortion_mode=None, out_csv=None):
    """
    D3: the main report table. All models x datasets x segments.

    Every cell uses IDENTICAL settings (dt, n_iters, deltas, distortion mode) so
    the comparison is fair; the only thing that varies is model/dataset/segment.
    Deliberately does NOT use --best: BEST_CONFIGS covers one dataset and three
    models, so it would tune a few cells and leave the rest at defaults.

    n_frames is clamped per segment by segment_n_frames() so no run ever extends
    past the segment's validated constant-omega duration.

    Loops dataset -> segment -> model, appending and flushing one row per run, so
    an interrupted overnight job still leaves a valid CSV. With resume=True a run
    whose summary.json already exists is loaded instead of recomputed.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 12: report results table")
    print("=" * 70)

    models = models or list(ALL_MODELS)
    datasets = available_datasets(dataset_filter)
    if not datasets:
        print("  No datasets with data available - nothing to do.")
        return []

    out_csv = out_csv or os.path.join('results', 'report', 'report_runs.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # Keep rows from a previous invocation so re-runs accumulate rather than
    # clobber (resume across nights).
    existing = {}
    if resume and os.path.exists(out_csv):
        with open(out_csv) as f:
            for r in csv.DictReader(f):
                existing[(r['dataset'], r['segment_id'], r['model'])] = r

    rows = []
    f_out = open(out_csv, 'w', newline='')
    writer = csv.DictWriter(f_out, fieldnames=RUN_FIELDS, extrasaction='ignore',
                            restval='')
    writer.writeheader()
    f_out.flush()

    n_total = sum(len(resolve_segments(d, segments)) for d in datasets) * len(models)
    n_done = 0

    for dataset in datasets:
        for seg in resolve_segments(dataset, segments):
            for model in models:
                n_done += 1
                key = (dataset, seg['id'], model)
                tag = f"[{n_done}/{n_total}] {dataset}/{seg['id']}/{model}"

                row = {'dataset': dataset, 'segment_id': seg['id'],
                       'model': model, 'status': 'ok'}
                try:
                    nf = segment_n_frames(seg, seg.get('frame_duration', 0.02),
                                          requested=n_frames)
                    rc = RunConfig(dataset=dataset, model=model, segment=seg,
                                   n_frames=nf, distortion_mode=distortion_mode)

                    summ = None
                    if resume:
                        sp = os.path.join(rc.output_dir, 'summary.json')
                        if os.path.exists(sp):
                            try:
                                summ = json.load(open(sp))
                                print(f"  {tag}: cached")
                            except Exception:
                                summ = None
                    if summ is None:
                        print(f"  {tag}: running ({nf} frames)")
                        summ = experiment_tracking(rc, save_frames=save_frames)

                    row.update(summ)
                    row.update({
                        'n_frames': rc.n_frames,
                        'frame_duration': rc.frame_duration,
                        'duration_s': rc.duration_s,
                        'n_iters': rc.n_iters,
                        'delta_FR': rc.params.get('delta_FR'),
                        'delta_IMU': rc.delta_IMU if rc.use_imu else '',
                        'cmax_lr': rc.cmax_lr if rc.use_cmax_v2 else '',
                        'distortion_mode': rc.distortion_mode,
                        'omega_gt_mag_mean': _omega_gt_mag_mean(rc.output_dir),
                    })
                except Exception as e:
                    # Never let one bad cell kill an overnight batch.
                    row['status'] = f'failed: {type(e).__name__}: {e}'[:200]
                    print(f"  {tag}: FAILED - {e}")

                writer.writerow(row)
                f_out.flush()
                rows.append(row)

    # Carry over rows from earlier invocations that this batch did not touch,
    # so running one dataset per night accumulates into a single table instead
    # of each run clobbering the previous night's results.
    done = {(r['dataset'], r['segment_id'], r['model']) for r in rows}
    carried = 0
    for key, old in existing.items():
        if key not in done:
            writer.writerow(old)
            carried += 1
    f_out.flush()
    f_out.close()

    ok = sum(1 for r in rows if r['status'] == 'ok')
    extra = f", {carried} carried over from previous runs" if carried else ""
    print(f"\n  {ok}/{len(rows)} runs ok{extra}  ->  {out_csv}")
    return rows



# ===========================================================================
# SHARED MAP RENDERING (used by exp 10 maps panel and exp 11 iteration GIF)
# ===========================================================================

def _draw_maps_row(axes, V, I, G, F, omega_est, omega_ref=None,
                   max_mag=None, gmax=None, titles=False, row_label=None,
                   note=None):
    """
    Draw one row of the map set: V | I | |G| | F | omega bars.

    max_mag / gmax fix the F and |G| colour scales. Pass shared values when
    several rows (models) or frames (GIF iterations) must be comparable --
    otherwise every panel self-normalises and neither comparison nor animation
    is meaningful.
    """
    ax = axes[0]
    ax.imshow(V, cmap='RdBu', vmin=-1, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    if titles:
        ax.set_title('V  (events)', fontsize=10)
    if row_label:
        ax.set_ylabel(row_label, fontsize=10, rotation=90, labelpad=6)

    ax = axes[1]
    ax.imshow(normalise_robust(I), cmap='gray', vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    if titles:
        ax.set_title('I  (intensity)', fontsize=10)

    ax = axes[2]
    gm = np.linalg.norm(G, axis=-1)
    ax.imshow(gm, cmap='hot', vmin=0,
              vmax=(gmax if gmax else gm.max() + 1e-10))
    ax.set_xticks([]); ax.set_yticks([])
    if titles:
        ax.set_title('|G|  (gradient)', fontsize=10)

    ax = axes[3]
    ax.imshow(flow_to_rgb(F, max_mag=max_mag))
    ax.set_xticks([]); ax.set_yticks([])
    if titles:
        ax.set_title('F  (flow)', fontsize=10)

    ax = axes[4]
    x = np.arange(3)
    ax.bar(x - 0.2, omega_est, 0.4, label='est', color='tab:blue')
    if omega_ref is not None:
        ax.bar(x + 0.2, omega_ref, 0.4, label='GT', color='none',
               edgecolor='black', linewidth=1.4)
    ax.set_xticks(x); ax.set_xticklabels(['wx', 'wy', 'wz'], fontsize=9)
    ax.axhline(0, color='0.6', linewidth=0.8)
    ax.grid(alpha=0.25, axis='y')
    if titles:
        ax.set_title('omega (rad/s)', fontsize=10)
    if note:
        ax.set_xlabel(note, fontsize=8)
    if omega_ref is not None:
        err, dir_err, beta = compute_metrics(omega_est, omega_ref)
        ax.text(0.02, 0.97, f'err {err:.1f}\ndir {dir_err:.1f}\nb {beta:.2f}',
                transform=ax.transAxes, va='top', ha='left', fontsize=7,
                bbox=dict(fc='white', alpha=0.7, lw=0))


# ===========================================================================
# EXPERIMENT 10: final maps panel (D1)  -- assembles maps_*.npz, no re-runs
# ===========================================================================

def experiment_maps_panel(dataset, segment_id, models=None, which='center',
                          n_frames=25, run_missing=True):
    """
    One figure: rows = models, columns = V | I | |G| | F | omega.

    Reads the maps_{which}.npz that experiment_tracking dumps, so after the
    report-table run this costs seconds and needs no network execution.
    |G| and F share a colour scale across rows so the models are comparable.
    """
    import glob as _glob
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 10: maps panel  {dataset}/{segment_id}  ({which})")
    print("=" * 70)

    models = models or list(ALL_MODELS)
    data = {}
    for m in models:
        pat = os.path.join('results', dataset, m, f'{segment_id}_*',
                           f'maps_{which}.npz')
        hits = sorted(_glob.glob(pat))
        if not hits and run_missing:
            print(f"  {m}: no maps found - running it")
            try:
                segs = resolve_segments(dataset, segment_id)
                if not segs:
                    print(f"  {m}: segment '{segment_id}' unknown - skipping")
                    continue
                nf = segment_n_frames(segs[0], segs[0].get('frame_duration', 0.02),
                                      requested=n_frames)
                rc = RunConfig(dataset=dataset, model=m, segment=segs[0], n_frames=nf)
                experiment_tracking(rc, save_frames=False)
                hits = sorted(_glob.glob(pat))
            except Exception as e:
                print(f"  {m}: failed ({e}) - skipping")
                continue
        if hits:
            # Newest by mtime, NOT lexicographic: '..._n25_...' sorts before
            # '..._n3_...', so sorted()[-1] would pick a stale 3-frame run.
            data[m] = np.load(max(hits, key=os.path.getmtime), allow_pickle=True)
        else:
            print(f"  {m}: no maps - skipping")

    if not data:
        print("  Nothing to plot.")
        return None

    # Shared scales (99th percentile: robust to a few hot pixels).
    gmax = max(float(np.percentile(np.linalg.norm(d['G'], axis=-1), 99))
               for d in data.values()) + 1e-10
    max_mag = max(float(np.percentile(np.linalg.norm(d['F'], axis=-1), 99))
                  for d in data.values()) + 1e-10

    names = [m for m in models if m in data]
    fig, axes = plt.subplots(len(names), 5,
                             figsize=(15, 2.9 * len(names)), squeeze=False)
    for i, m in enumerate(names):
        d = data[m]
        _draw_maps_row(axes[i], d['V'], d['I'], d['G'], d['F'],
                       d['omega_est'], d['omega_ref'],
                       max_mag=max_mag, gmax=gmax,
                       titles=(i == 0), row_label=m)
    fig.suptitle(f'Maps after relaxation - {dataset} / {segment_id} '
                 f'({which} frame; |G| and F share a scale across rows)',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_dir = os.path.join('results', 'report')
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f'maps_panel_{dataset}_{segment_id}_{which}')
    fig.savefig(stem + '.png', dpi=150)
    fig.savefig(stem + '.pdf')
    plt.close(fig)
    print(f"  wrote {stem}.png / .pdf   ({len(names)} models)")
    return stem + '.png'


# ===========================================================================
# EXPERIMENT 11: map evolution across relaxation iterations (D2)
# ===========================================================================

def experiment_iteration_gif(rc: RunConfig, max_iters=100, fps=10, every=1):
    """
    Render one PNG per relaxation iteration for the frame at the SEGMENT CENTRE,
    then assemble a GIF.

    Cold start: a fresh network relaxes just that frame, which is the
    "convergence from scratch" story the animation is meant to tell.

    Runs the relaxation TWICE: pass 1 measures the final |G| and |F| magnitudes
    so pass 2 can render every iteration on a FIXED colour scale. Without that
    each frame self-normalises and the GIF flickers instead of showing the maps
    actually forming. One frame, so the extra pass is cheap.
    """
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 11: iteration GIF  {rc.dataset}/{rc.segment_id}/{rc.model}")
    print("=" * 70)

    k_center = rc.n_frames // 2
    seq = EventFrameSequence(
        rc.paths['events'], rc.paths['calib'],
        frame_duration=rc.frame_duration, t_start=rc.t_start,
        n_frames=k_center + 1, clip_value=10.0,
        undistort=rc.undistort_at_event_level, sensor_size=rc.sensor_size,
    )
    frames = list(seq)
    V, t_mid = frames[k_center]
    H, W = seq.H, seq.W
    fx, fy, cx, cy = seq.calib.fx, seq.calib.fy, seq.calib.cx, seq.calib.cy

    t_lo = rc.t_start + k_center * rc.frame_duration
    t_hi = t_lo + rc.frame_duration
    imu_data = load_imu(rc.paths['imu'])
    gt_data = load_groundtruth(rc.paths['groundtruth'])
    omega_gt_data = load_omega_gt(rc.paths.get('omega_gt'))
    omega_ref, ref_src = get_reference_omega(gt_data, imu_data, t_lo, t_hi,
                                             omega_gt_data)
    omega_imu = get_gyro_for_frame(imu_data, t_lo, t_hi)
    print(f"  centre frame {k_center}, t={t_mid:.3f}s, ref={ref_src}")

    # Raw events for the CMax variants.
    win, cmax_est = None, None
    if getattr(rc, 'use_cmax', False):
        from data_loader import load_events_fast, undistort_events
        ev = undistort_events(
            load_events_fast(rc.paths['events'], t_start=t_lo,
                             duration=rc.frame_duration + 0.02), seq.calib)
        win = ev[(ev[:, 0] >= t_lo) & (ev[:, 0] < t_hi)] if len(ev) else ev
        if not getattr(rc, 'use_cmax_v2', False):
            from cmax import CMaxAngularVelocity
            cmax_est = CMaxAngularVelocity(H, W, fx, fy, cx, cy, use_polarity=True)

    def fresh_net():
        return make_network(rc, H, W, fx, fy, cx, cy)

    # ---- pass 1: fix the colour scales -------------------------------
    net = fresh_net()
    _net_step(net, rc, V, n_iters=max_iters, omega_imu=omega_imu, win=win,
              cmax_est=cmax_est, state={'t_ref': 0.5 * (t_lo + t_hi)})
    gmax = float(np.percentile(np.linalg.norm(net.G, axis=-1), 99)) + 1e-10
    max_mag = float(np.percentile(np.linalg.norm(net.F, axis=-1), 99)) + 1e-10

    # ---- pass 2: render one PNG per iteration ------------------------
    out_dir = os.path.join(rc.output_dir, 'iter_frames')
    os.makedirs(out_dir, exist_ok=True)
    for old in os.listdir(out_dir):
        if old.startswith('frame_') and old.endswith('.png'):
            os.remove(os.path.join(out_dir, old))

    hist = []
    net = fresh_net()

    def cb(it, n):
        w = n.R / rc.frame_duration
        err, dir_err, beta = compute_metrics(w, omega_ref)
        try:
            res = n.residual_VFG(V)
        except Exception:
            res = float('nan')
        hist.append((it, err, dir_err, beta, res))
        if it % every:
            return
        I = n.I[:H, :W] if n.I.shape != (H, W) else n.I
        # Fixed figsize + dpi and NO tight_layout: every frame must have
        # identical pixel dimensions or the GIF jitters.
        fig, axes = plt.subplots(1, 5, figsize=(15, 3.1))
        fig.subplots_adjust(left=0.04, right=0.99, top=0.86, bottom=0.12,
                            wspace=0.18)
        _draw_maps_row(axes, V, I, n.G, n.F, w, omega_ref,
                       max_mag=max_mag, gmax=gmax, titles=True)
        fig.suptitle(f'{rc.model} - iteration {it + 1}/{max_iters}   '
                     f'err {err:.1f} deg/s   residual {res:.4f}', fontsize=11)
        fig.savefig(os.path.join(out_dir, f'frame_{it:04d}.png'), dpi=90)
        plt.close(fig)

    _net_step(net, rc, V, n_iters=max_iters, omega_imu=omega_imu, win=win,
              cmax_est=cmax_est, state={'t_ref': 0.5 * (t_lo + t_hi)},
              on_iter=cb)

    from cmax.iwe_io import make_gif
    make_gif(out_dir, fps=fps, name='iteration_evolution.gif')

    # Quantitative companion: the per-iteration histories exp 1 used to discard.
    h = np.array(hist)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(h[:, 0] + 1, h[:, 1], color='tab:red')
    ax[0].set_xlabel('iteration'); ax[0].set_ylabel('error (deg/s)')
    ax[0].set_title('omega error vs iteration'); ax[0].grid(alpha=0.3)
    ax[1].plot(h[:, 0] + 1, h[:, 4], color='tab:blue')
    ax[1].set_xlabel('iteration'); ax[1].set_ylabel('residual |V + F.G|')
    ax[1].set_title('OFCE residual vs iteration'); ax[1].grid(alpha=0.3)
    fig.suptitle(f'{rc.dataset}/{rc.segment_id}/{rc.model} - convergence')
    fig.tight_layout()
    fig.savefig(os.path.join(rc.output_dir, 'exp11_iter_convergence.png'), dpi=140)
    plt.close(fig)

    np.savetxt(os.path.join(rc.output_dir, 'exp11_iter_history.csv'), h,
               delimiter=',', header='iter,err_deg_s,dir_err_deg,beta,residual',
               comments='')
    n_png = len([f for f in os.listdir(out_dir) if f.endswith('.png')])
    print(f"  {n_png} frames -> {os.path.join(out_dir, 'iteration_evolution.gif')}")
    print(f"  err {h[0, 1]:.1f} -> {h[-1, 1]:.1f} deg/s over {max_iters} iterations")
    return out_dir


# ===========================================================================
# EXPERIMENT 13: one-at-a-time parameter sweep (D4)
# ===========================================================================

OAT_AXES = {
    'n_iters':         [10, 25, 50, 75, 100, 150],
    'delta_FR':        [0.05, 0.10, 0.20, 0.30, 0.50, 0.80],
    'delta_IMU':       [0.05, 0.10, 0.20, 0.30, 0.50, 0.80],
    'frame_duration':  [0.005, 0.010, 0.020, 0.030, 0.050],
    'delta_VFG':       [0.04, 0.08, 0.12, 0.20, 0.30],
    'delta_IG':        [0.03, 0.10, 0.20, 0.30],
    'delta_GI':        [0.02, 0.05, 0.10, 0.20],
    'delta_RF':        [0.01, 0.03, 0.05, 0.10, 0.20],
    'cmax_lr':         [1e-5, 3e-5, 1e-4, 3e-4],
    'distortion_mode': ['undistort_events', 'C_full'],
}


def clone_rc(rc: RunConfig, tag=None, **overrides):
    """
    Copy a RunConfig, changing only the named fields.

    The old distortion-ablation clone passed just a handful of fields, silently
    resetting delta_VFG/IG/GI/RF and cmax_lr to defaults. This carries every
    parameter across. Spatial deltas equal to the model default are passed as
    None so they do not add a suffix to output_dir (keeping baseline folder
    names stable); only genuinely non-default values are marked explicit.
    """
    base = COOK_PARAMS if rc.model == 'cook' else THESIS_PARAMS
    kw = dict(dataset=rc.dataset, model=rc.model, segment=rc.segment,
              t_start=rc.t_start, frame_duration=rc.frame_duration,
              n_frames=rc.n_frames, n_iters=rc.n_iters,
              delta_IMU=(rc.params.get('delta_IMU') if rc.use_imu else None),
              delta_FR=rc.params.get('delta_FR'),
              cmax_lr=rc.cmax_lr, distortion_mode=rc.distortion_mode)
    for k in ('delta_VFG', 'delta_IG', 'delta_GI', 'delta_RF'):
        v = rc.params.get(k)
        d = base.get(k)
        kw[k] = v if (v is not None and d is not None
                      and abs(v - d) > 1e-12) else None
    kw.update(overrides)
    out = RunConfig(**kw)
    out.tag = tag
    return out


def experiment_oat_sweep(rc_base: RunConfig, axes=None, save_frames=False):
    """
    D4: hold everything fixed, vary ONE parameter at a time.

    Two rules that make the results interpretable:
      * every clone gets a unique tag -> unique output_dir, so runs cannot
        overwrite each other (this is what previously made cmax_lr sweeps
        silently produce identical rows);
      * the frame_duration axis keeps the PHYSICAL duration constant by
        rescaling n_frames, otherwise the axis confounds "shorter time window"
        with "fewer events per bin".
    """
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 13: OAT sweep  {rc_base.dataset}/{rc_base.segment_id}"
          f"/{rc_base.model}")
    print("=" * 70)

    if axes is None:
        axes = ['n_iters', 'delta_FR', 'delta_IMU', 'frame_duration',
                'delta_VFG', 'delta_IG', 'delta_GI', 'cmax_lr']
    base_duration = rc_base.n_frames * rc_base.frame_duration
    out_root = os.path.join('results', 'oat',
                            f'{rc_base.dataset}_{rc_base.segment_id}_{rc_base.model}')
    os.makedirs(out_root, exist_ok=True)
    all_rows = []

    for axis in axes:
        if axis not in OAT_AXES:
            print(f"  unknown axis '{axis}' - skipping")
            continue
        if axis == 'delta_IMU' and not rc_base.use_imu:
            print(f"  {axis}: n/a for {rc_base.model} - skipping")
            continue
        if axis == 'cmax_lr' and not getattr(rc_base, 'use_cmax_v2', False):
            print(f"  {axis}: n/a for {rc_base.model} - skipping")
            continue

        print(f"\n  --- axis: {axis} ---")
        rows = []
        for v in OAT_AXES[axis]:
            ov = {axis: v}
            if axis == 'frame_duration':
                ov['n_frames'] = max(1, int(round(base_duration / v)))
            try:
                rc = clone_rc(rc_base, tag=f'oat_{axis}_{v}', **ov)
                summ = experiment_tracking(rc, save_frames=save_frames)
                row = {'axis': axis, 'value': v, 'n_frames': rc.n_frames,
                       'frame_duration': rc.frame_duration}
                row.update(summ)
                rows.append(row)
                print(f"    {axis}={v}: mean_err={summ['mean_err_deg_s']:.2f} "
                      f"beta={summ['mean_beta']:.3f}")
            except Exception as e:
                print(f"    {axis}={v}: FAILED - {e}")

        if not rows:
            continue
        all_rows.extend(rows)

        csv_p = os.path.join(out_root, f'{axis}.csv')
        keys = sorted({k for r in rows for k in r})
        with open(csv_p, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            w.writeheader()
            for r in rows:
                w.writerow(r)

        xs = [r['value'] for r in rows]
        ys = [r['mean_err_deg_s'] for r in rows]
        bs = [r['mean_beta'] for r in rows]
        fig, ax = plt.subplots(figsize=(6, 3.8))
        numeric = all(isinstance(x, (int, float)) for x in xs)
        if numeric:
            ax.plot(xs, ys, 'o-', color='tab:red', label='mean err')
            if axis in ('n_iters', 'cmax_lr'):
                ax.set_xscale('log')
        else:
            ax.bar([str(x) for x in xs], ys, color='tab:red', alpha=0.8)
        ax.set_xlabel(axis); ax.set_ylabel('mean error (deg/s)', color='tab:red')
        ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        if numeric:
            ax2.plot(xs, bs, 's--', color='tab:blue', alpha=0.8)
        else:
            ax2.plot([str(x) for x in xs], bs, 's--', color='tab:blue', alpha=0.8)
        ax2.set_ylabel('mean beta', color='tab:blue')
        ax2.axhline(1.0, color='tab:blue', ls=':', alpha=0.5)
        ax.set_title(f'{rc_base.model} - sensitivity to {axis}')
        fig.tight_layout()
        fig.savefig(os.path.join(out_root, f'oat_{axis}.png'), dpi=140)
        plt.close(fig)

    if all_rows:
        summary_p = os.path.join(out_root, 'oat_all.csv')
        keys = sorted({k for r in all_rows for k in r})
        with open(summary_p, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        print(f"\n  wrote {out_root}/  ({len(all_rows)} runs)")
    return all_rows


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Systematic Evaluation')
    parser.add_argument('--exp', type=int, default=2, help='1-7')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--model', type=str, default='thesis_imu',
                        choices=['cook', 'thesis', 'thesis_imu', 'thesis_cmax',
                                 'thesis_cmax_v2', 'all'])
    parser.add_argument('--segment', type=str, default=None,       # ← NEU
                        help='Segment ID (z.B. seg_A, seg_B)')
    parser.add_argument('--n_frames', type=int, default=None)
    parser.add_argument('--n_iters', type=int, default=None)
    parser.add_argument('--delta_imu', type=float, default=None)
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--all-segments', action='store_true',
                    help='Run grid sweep over ALL segments (slow)')
    parser.add_argument('--no-frames', action='store_true',
                        help='Skip saving 3-col PNGs (faster)')
    parser.add_argument('--save-iwe', action='store_true',
                        help='Save the final CMax IWE per frame (V1/V2 only)')
    parser.add_argument('--distortion-mode', type=str, default=None,
                        choices=['undistort_events', 'C_full'],
                        help='Override distortion handling (exp 1-6, 8). Exp 9 sweeps both.')
    # --- parameter overrides (needed by the OAT sweep, exp 13) ---
    parser.add_argument('--frame_duration', type=float, default=None,
                        help='Event window length dt in seconds')
    parser.add_argument('--t_start', type=float, default=None)
    parser.add_argument('--delta_fr', type=float, default=None)
    parser.add_argument('--delta_vfg', type=float, default=None)
    parser.add_argument('--delta_ig', type=float, default=None)
    parser.add_argument('--delta_gi', type=float, default=None)
    parser.add_argument('--delta_rf', type=float, default=None)
    parser.add_argument('--cmax_lr', type=float, default=None,
                        help='V2 CMax ascent step (stable ~1e-5..1e-4)')
    parser.add_argument('--tag', type=str, default=None,
                        help='Suffix appended to output_dir')
    # --- batch selection ---
    parser.add_argument('--models', type=str, default=None,
                        help="Comma list, e.g. 'cook,thesis_imu'. Overrides --model.")
    parser.add_argument('--segments', type=str, default='all',
                        help="Comma list of segment ids, or 'all'")
    parser.add_argument('--no-resume', action='store_true',
                        help='Recompute even when summary.json already exists')
    parser.add_argument('--max-iters', type=int, default=100,
                        help='exp 11: relaxation iterations to animate')
    parser.add_argument('--gif-fps', type=int, default=10)
    parser.add_argument('--every', type=int, default=1,
                        help='exp 11: render every Nth iteration')
    parser.add_argument('--maps-which', type=str, default='center',
                        choices=['center', 'final'], help='exp 10: which frame')
    parser.add_argument('--oat-axes', type=str, default=None,
                        help='exp 13: comma list of axes to sweep')
    parser.add_argument('--best', action='store_true',
                        help='Fill unspecified params from best_config.py '
                             '(best grid params for this dataset/segment/model). '
                             'Explicit CLI flags still win.')
    args = parser.parse_args()
    save_frames = not args.no_frames  # Default: save images

    # Build config ONLY for experiments that need it (1-6)
    
    # Resolve --model all → None (triggers all-models logic in exp 7/8)
    effective_model = None if args.model == 'all' else args.model

    # Build config ONLY for experiments that need it (1-6)
    rc = None
    need_rc = args.exp in (1, 2, 3, 4, 5, 6, 9, 11, 13)
    if args.exp == 9 and args.segment == 'all':
        need_rc = False                       # all-segments path builds its own configs
    if need_rc:
        dataset = args.dataset or 'boxes_rotation'
        model = args.model if args.model != 'all' else 'thesis_imu'
        rc_kwargs = dict(
            dataset=dataset,
            model=model,
            segment=args.segment,
            n_frames=args.n_frames,
            n_iters=args.n_iters,
            delta_IMU=args.delta_imu,
            distortion_mode=args.distortion_mode,
            t_start=args.t_start,
            frame_duration=args.frame_duration,
            delta_FR=args.delta_fr,
            delta_VFG=args.delta_vfg,
            delta_IG=args.delta_ig,
            delta_GI=args.delta_gi,
            delta_RF=args.delta_rf,
            cmax_lr=args.cmax_lr,
            tag=args.tag,
        )
        if args.best:
            if args.segment in (None, 'all'):
                print("  [--best] requires a specific --segment (e.g. seg_A); "
                      "ignoring --best.")
            else:
                cli = {'n_frames': args.n_frames, 'n_iters': args.n_iters,
                       'delta_IMU': args.delta_imu}
                rc_kwargs.update(resolve_best_kwargs(dataset, args.segment, model, cli))
        rc = RunConfig(**rc_kwargs)
        rc.save_iwe = args.save_iwe

    experiments = {
        10: lambda: experiment_maps_panel(
                dataset=args.dataset or 'poster_rotation',
                segment_id=args.segment or 'seg_A',
                models=resolve_models(effective_model, args.models),
                which=args.maps_which,
                n_frames=args.n_frames if args.n_frames else 25),
        11: lambda: experiment_iteration_gif(
                rc, max_iters=args.max_iters, fps=args.gif_fps, every=args.every),
        13: lambda: experiment_oat_sweep(
                rc,
                axes=[a.strip() for a in args.oat_axes.split(',')]
                     if args.oat_axes else None,
                save_frames=False),
        12: lambda: experiment_report_table(
                dataset_filter=args.dataset,
                models=resolve_models(effective_model, args.models),
                segments=args.segments,
                n_frames=args.n_frames if args.n_frames else 25,
                save_frames=save_frames,
                resume=not args.no_resume,
                distortion_mode=args.distortion_mode),
        1: lambda: experiment_single_frame_convergence(rc, frame_idx=args.frame),
        2: lambda: experiment_tracking(rc, save_frames=save_frames),
        3: lambda: experiment_parameter_influence(rc, frame_idx=args.frame),
        4: lambda: experiment_tracking(rc, save_frames=True),  # Always with frames
        #5: lambda: experiment_make_video(rc, fps=args.fps),
        5: lambda: experiment_make_videos_batch(fps=args.fps),  # ← Batch!
        6: lambda: experiment_basin_of_attraction(rc, frame_idx=args.frame),
        7: lambda: experiment_full_evaluation(
            dataset_filter=args.dataset,
            model_filter=effective_model,
            save_frames=save_frames,
        ),
        8: lambda: experiment_parameter_grid(
            dataset_filter=args.dataset,
            model_filter=effective_model,
            all_segments=args.all_segments,
            save_frames=save_frames,
            distortion_mode=args.distortion_mode,
        ),
        9: lambda: (experiment_distortion_ablation_all_segments(
                        dataset=args.dataset or 'boxes_rotation',
                        model=(args.model if args.model != 'all' else 'thesis_imu'),
                        save_frames=save_frames, use_best=args.best)
                    if args.segment == 'all'
                    else experiment_distortion_ablation(rc, save_frames=save_frames)),
        }

    experiments[args.exp]()