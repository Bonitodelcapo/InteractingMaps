"""
Demo for the Interacting Maps network (Cook et al., IJCNN 2011)
If USE_THESIS = True, uses the energy-based network from Martel 2019 Thesis (Chapter 6)).
Run:
    python demo.py
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

from interacting_maps.camera import compute_calibration
from interacting_maps.network_dissertation import InteractingMapsThesis
from interacting_maps.network import InteractingMaps


# ---------------------------------------------------------------------------
# Configuration — DATASET SPECIFIC
# ---------------------------------------------------------------------------

# Choose your dataset:
DATASET = 'poster_rotation'   # or 'boxes_rotation', 'dynamic_rotation', etc.

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', DATASET)
EVENTS_FILE = os.path.join(DATA_DIR, 'events.txt')
CALIB_FILE = os.path.join(DATA_DIR, 'calib.txt')
IMU_FILE = os.path.join(DATA_DIR, 'imu.txt')
GT_FILE = os.path.join(DATA_DIR, 'groundtruth.txt')

# Per-dataset configurations (from find_segments.py output):
DATASET_CONFIGS = {
    'shapes_rotation': {
        't_start': 2.416,       
        'frame_duration': 0.020,
        'n_frames': 25,
        'initial_R':  np.array([0.00079, -0.01494, 0.00237]),
        'expected_omega': np.array([0.040, -0.747, 0.118]),
    },
    'boxes_rotation': {
        't_start': 10.872,
        'frame_duration': 0.020,
        'n_frames': 25,
        'initial_R': np.array([0.00308, 0.00203, -0.02959]),
        'expected_omega': np.array([0.154, 0.102, -1.479]),
    },
    'poster_rotation':{
        't_start':  8.816,
        'frame_duration': 0.020,
        'n_frames': 25,
        'initial_R': np.array([0.02242, 0.00101, 0.00055]),
        'expected_omega': np.array([1.121, 0.051, 0.027]),
    }
}

# Load config for chosen dataset
cfg = DATASET_CONFIGS[DATASET]
t_start = cfg['t_start']
frame_duration = cfg['frame_duration']
n_frames = cfg['n_frames']
initial_R = cfg['initial_R']

ITERS_PER_FRAME = 75
USE_THESIS_VERSION = False

# ---------------------------------------------------------------------------
# Choose data source
# ---------------------------------------------------------------------------

def make_real_source():
    from data_loader import EventFrameSequence
    seq = EventFrameSequence(
        EVENTS_FILE, CALIB_FILE,
        frame_duration=cfg['frame_duration'],
        t_start=cfg['t_start'],        
        n_frames=cfg['n_frames'],
        clip_value=10.0,
    )
    return list(seq), seq.calib, seq.H, seq.W

use_real = os.path.isfile(EVENTS_FILE) and os.path.isfile(CALIB_FILE)
print(f"Using real event-camera data ({DATA_DIR} dataset).")
frames, calib, H, W = make_real_source()
fx, fy, cx, cy = calib.fx, calib.fy, calib.cx, calib.cy


# ---------------------------------------------------------------------------
# Build network
# ---------------------------------------------------------------------------

if USE_THESIS_VERSION:
    NET_PARAMS = dict(
        delta_VFG=0.20,    # OFCE: strong (must compete with kinematics) 0.15
        delta_IG=0.10,     # G from I: moderate
        delta_GI=0.05,     # I from G: gentle (Poisson step, needs stability)
        delta_RF=0.01,     # F from R: WEAK (let OFCE build local structure) 0.03
        delta_FR=0.80,     # R from F: strong (global aggregate, stable)  0.50
    )

    ITERS_PER_FRAME = 50   # Thesis uses 50-75 (Section 6.8)
    net = InteractingMapsThesis(H=H, W=W, fx=fx, fy=fy, cx=cx, cy=cy, **NET_PARAMS)
    net.initialize_from_rotation(initial_R)

else:
    NET_PARAMS = dict(
            delta_VFG=0.08, delta_IG=0.12, delta_GI=0.08,
            delta_RF=0.10, delta_FR=0.50,
        )
    net = InteractingMaps(H=H, W=W, fx=fx, fy=fy, cx=cx, cy=cy, **NET_PARAMS)
    net.reset(scale=0.01)
    net.R = initial_R.copy()

# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def normalise(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-10)


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """HSV: hue = direction, value = magnitude."""
    fx, fy = flow[..., 0], flow[..., 1]
    angle = (np.arctan2(fy, fx) + np.pi) / (2 * np.pi)
    mag   = np.sqrt(fx ** 2 + fy ** 2)
    hsv   = np.stack([angle, np.ones_like(angle), normalise(mag)], axis=-1)
    return matplotlib.colors.hsv_to_rgb(hsv)


def flow_norm(field: np.ndarray) -> np.ndarray:
    return np.sqrt(field[..., 0] ** 2 + field[..., 1] ** 2)


# ---------------------------------------------------------------------------
# Figure setup
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(14, 8))
title_src = "RPG shapes_rotation dataset" if use_real else "Synthetic simulation"
fig.suptitle(
    f'Interacting Maps — Cook et al., IJCNN 2011  [{title_src}]\n'
    'Sole input: V (temporal derivative). All other maps are inferred.',
    fontsize=10,
)

gs = GridSpec(2, 4, figure=fig, hspace=0.40, wspace=0.30)
ax_V   = fig.add_subplot(gs[0, 0])
ax_I   = fig.add_subplot(gs[0, 1])
ax_G   = fig.add_subplot(gs[0, 2])
ax_F   = fig.add_subplot(gs[0, 3])
ax_R   = fig.add_subplot(gs[1, 0])
ax_res = fig.add_subplot(gs[1, 1:])

blank  = np.zeros((H, W))
blank3 = np.zeros((H, W, 3))

im_V = ax_V.imshow(blank,  cmap='RdBu',  vmin=-1, vmax=1)
im_I = ax_I.imshow(blank,  cmap='gray',  vmin=0,  vmax=1)
im_G = ax_G.imshow(blank,  cmap='hot',   vmin=0,  vmax=1)
im_F = ax_F.imshow(blank3)

for ax, t in [(ax_V, 'V: Input (temporal deriv.)'),
              (ax_I, 'I: Inferred intensity'),
              (ax_G, '|G|: Inferred gradient'),
              (ax_F, 'F: Inferred flow (HSV)')]:
    ax.set_title(t, fontsize=8)
    ax.axis('off')

# R bar chart — x/y/z components
r_idx   = np.arange(3)
bw      = 0.6
bars_R  = ax_R.bar(r_idx, [0, 0, 0], bw, color=['#4e79a7', '#f28e2b', '#e15759'])
ax_R.set_xticks(r_idx)
ax_R.set_xticklabels(['Rx', 'Ry', 'Rz'])
ax_R.set_title('Estimated camera rotation R', fontsize=8)
ax_R.axhline(0, color='k', linewidth=0.5)
ax_R.set_ylim(-0.05, 0.05)

# Residual curves
res_VFG_hist: list[float] = []
res_GI_hist:  list[float] = []
line_VFG, = ax_res.semilogy([], [], 'b-', lw=1.5, label='|V + F·G|  (flow constraint)')
line_GI,  = ax_res.semilogy([], [], 'r-', lw=1.5, label='|G − ∇I|   (gradient constraint)')
ax_res.set_title('Constraint residuals over frames', fontsize=8)
ax_res.set_xlabel('Frame')
ax_res.legend(fontsize=7)
ax_res.set_xlim(0, n_frames)
ax_res.set_ylim(1e-5, 2)

frame_idx = [0]


# ---------------------------------------------------------------------------
# Animation update
# ---------------------------------------------------------------------------

def update(_):
    t = frame_idx[0]
    if t >= len(frames):
        return im_V, im_I, im_G, im_F, *bars_R, line_VFG, line_GI

    V, _ = frames[t]
    frame_idx[0] += 1

    net.step(V, n_iters=ITERS_PER_FRAME)

    # V display
    im_V.set_data(V)
    vabs = max(np.percentile(np.abs(V), 95), 1e-3)
    im_V.set_clim(-vabs, vabs)

    # Inferred I
    I_disp = net.I[:H, :W]
    im_I.set_data(normalise(I_disp))

    # |G|
    im_G.set_data(normalise(flow_norm(net.G)))

    # F (HSV colour)
    im_F.set_data(flow_to_rgb(net.F))

    # R bars
    R_scale = np.linalg.norm(net.R) + 1e-9
    for bar, val in zip(bars_R, net.R):
        bar.set_height(val)
    ax_R.set_ylim(
        min(-0.05, net.R.min() * 1.5),
        max( 0.05, net.R.max() * 1.5),
    )

    # Residuals
    res_VFG_hist.append(net.residual_VFG(V) + 1e-9)
    res_GI_hist.append(net.residual_GI()  + 1e-9)
    xs = list(range(len(res_VFG_hist)))
    line_VFG.set_data(xs, res_VFG_hist)
    line_GI.set_data(xs,  res_GI_hist)

    sparsity = np.mean(V == 0) * 100
    print(
        f'Frame {t+1:3d}/{n_frames}  '
        f'res_VFG={res_VFG_hist[-1]:.4f}  '
        f'res_GI={res_GI_hist[-1]:.5f}  '
        f'V_sparsity={sparsity:.0f}%  '
        f'|R|={np.linalg.norm(net.R):.3e}'
    )

    return im_V, im_I, im_G, im_F, *bars_R, line_VFG, line_GI


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

ani = animation.FuncAnimation(
    fig, update,
    frames=n_frames,
    interval=100,
    blit=False,
    repeat=False,
)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()

print('\nFinal estimated rotation R:')
print(f'  R = {net.R}')
print(f'  |R| = {np.linalg.norm(net.R):.4e}')
