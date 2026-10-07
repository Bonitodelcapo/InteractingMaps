"""
run_config.py — Run configuration and network factory for Interacting Maps.

Extracted from evaluation.py (architecture_review.md R5 / §"Concern" table:
RunConfig, resolve_best_kwargs, make_network) so scripts and tests can import
run configuration without pulling in the whole experiment-dispatch module.
"""

import os
import numpy as np

from config import (DATASET_CONFIGS, DATASET_SEGMENTS, THESIS_PARAMS, COOK_PARAMS,
                    ITERS_PER_FRAME, DISTORTION_MODE, POISSON_MODE,
                    get_dataset_paths)
from best_config import best_params
import provenance
from interacting_maps.network import InteractingMaps
from interacting_maps.network_dissertation import InteractingMapsThesis


# ===========================================================================
# RUN CONFIGURATION (override via CLI args)
# ===========================================================================

class RunConfig:
    """All parameters for a single evaluation run."""
    def __init__(self, dataset='boxes_rotation', model='thesis_imu',
                 segment=None, t_start=None, frame_duration=None, n_frames=None,
                 n_iters=None, delta_IMU=None, delta_FR=None, distortion_mode=None,
                 poisson=None, deltas=None, out_root=None):

        # Where this run's directory goes. 'results' holds the runs behind the
        # reported tables; a sweep passes e.g. 'experiments/sweep_deltas' so its
        # hundreds of throwaway runs stay out of the way and can be deleted as
        # a unit without touching anything the report depends on.
        self.out_root = out_root or 'results'
        self.dataset = dataset
        self.model = model  # 'cook', 'thesis', 'thesis_imu'
        self.distortion_mode = distortion_mode or DISTORTION_MODE
        # Which of the thesis' two I-updates to use: 'iterative' (Eq. 6.61) or
        # the exact frequency-domain solve, 'fft'/'dct' (Eq. 6.64-6.65).
        # Thesis network only; Cook's variant has its own intensity update.
        self.poisson = poisson or POISSON_MODE

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
        self.cmax_lr = 1e-4         # V2 ascent step. STABLE range ~[1e-5, 1e-4];
                                    # ≳5e-4 diverges. Scales with event count² —
                                    # lower it for denser streams (see cmax.md).
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

        # Override delta_IMU if specified
        if delta_IMU is not None and self.use_imu:
            self.params['delta_IMU'] = delta_IMU

        # Override delta_FR if specified
        if delta_FR is not None:
            self.params['delta_FR'] = delta_FR

        # Generic override for any relaxation rate, so a sweep can vary the
        # deltas the two named arguments above do not cover. Unknown keys are
        # rejected here rather than surfacing as a TypeError from the network.
        for k, v in (deltas or {}).items():
            if k not in self.params:
                raise ValueError(f"{model!r} has no parameter {k!r} "
                                 f"(has: {sorted(self.params)})")
            self.params[k] = v

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
        """Folder now includes dt."""
        folder_name = (f"{self.segment_id}_t{self.t_start:.3f}"
                    f"_dt{self.frame_duration*1000:.0f}ms"
                    f"_n{self.n_frames}_i{self.n_iters}")
        if self.use_imu:
            folder_name += f"_dimu{self.delta_IMU:.2f}"
        # Also encode delta_FR to distinguish configs
        folder_name += f"_dFR{self.params.get('delta_FR', 0):.2f}"
        folder_name += f"_{self.distortion_mode}"
        # Any OTHER relaxation rate that deviates from this model's defaults,
        # so a sweep over the remaining deltas cannot overwrite a run that
        # differs only in a parameter the name above does not carry.
        base = COOK_PARAMS if self.model == 'cook' else THESIS_PARAMS
        extra = [f"{k.replace('delta_', '')}{v:g}"
                 for k, v in sorted(self.params.items())
                 if k not in ('delta_FR', 'delta_IMU') and base.get(k) != v]
        if extra:
            folder_name += '_' + '-'.join(extra)
        if self.use_thesis and self.poisson != 'iterative':
            folder_name += f"_{self.poisson}"
        return os.path.join(self.out_root, self.dataset, self.model, folder_name)

    def to_dict(self):
        """Serialize all parameters for JSON."""
        return {
            'dataset': self.dataset,
            'model': self.model,
            'segment_id': self.segment_id,
            't_start': self.t_start,
            'frame_duration': self.frame_duration,
            'n_frames': self.n_frames,
            'distortion_mode': self.distortion_mode,
            'poisson': self.poisson,
            'out_root': self.out_root,
            'n_iters': self.n_iters,
            'duration_s': self.duration_s,
            'sensor_size': list(self.sensor_size),
            'initial_R': self.initial_R.tolist(),
            'params': self.params,
        }

    def save_run(self, summary=None):
        """Write the merged run.json (provenance + config + summary).

        Called once early with summary=None, so config and provenance survive a
        crash, and again at the end with the metrics filled in. Replaces the old
        split params.json/summary.json (read_run still falls back to those for
        runs written before this change)."""
        return provenance.save_run(self.output_dir, self.to_dict(), summary)

    def save_params(self):
        """Back-compat: write the early run.json (config + provenance, no metrics
        yet). Kept because exp 1/3/6 call it and have no summary to record."""
        self.save_run()

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
            poisson=getattr(rc, 'poisson', 'iterative'),
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
        # Seeded (matches the thesis init's default_rng(42)) so Cook runs are
        # reproducible; the global np.random here previously made them vary
        # run-to-run. The I-noise only seeds ∇I, so fixing the seed is strictly a
        # reproducibility gain, not a tuning change.
        net.I = np.random.default_rng(42).standard_normal((H+1, W+1)) * 0.001
        net.G = np.zeros((H, W, 2), dtype=np.float64)
        net.F = np.einsum('hwij,j->hwi', net._C_mat, rc.initial_R)
        net.R = rc.initial_R.copy()
    return net


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
