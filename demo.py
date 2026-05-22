"""
Demo for the Interacting Maps network (Cook et al., IJCNN 2011).

Modes
-----
1. Real event-camera data (default if data/shapes_rotation/ exists):
   Uses the RPG shapes_rotation dataset (DAVIS 240C sensor, 128×128 crop).
   Download: http://rpg.ifi.uzh.ch/datasets/davis/shapes_rotation.zip
   Extract events.txt and calib.txt to  data/shapes_rotation/

2. Synthetic fallback:
   A smooth natural-looking image rotated in front of a virtual DVS sensor.

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

from interacting_maps import InteractingMaps
from interacting_maps.camera import compute_calibration

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR       = os.path.join(os.path.dirname(__file__), 'data', 'shapes_rotation')
EVENTS_FILE    = os.path.join(DATA_DIR, 'events.txt')
CALIB_FILE     = os.path.join(DATA_DIR, 'calib.txt')

H, W          = 128, 128          # crop / output resolution
N_FRAMES      = 40
ITERS_PER_FRAME = 30
FRAME_DURATION  = 0.020           # seconds per frame for real data

# Network step sizes
NET_PARAMS = dict(
    delta_VFG=0.08,
    delta_IG=0.12,
    delta_GI=0.08,
    delta_RF=0.10,
    delta_FR=0.50,
)


# ---------------------------------------------------------------------------
# Choose data source
# ---------------------------------------------------------------------------

def make_real_source():
    """Load the RPG shapes_rotation dataset and return a frame iterator."""
    from data_loader import EventFrameSequence
    seq = EventFrameSequence(
        EVENTS_FILE,
        CALIB_FILE,
        H=H, W=W,
        frame_duration=FRAME_DURATION,
        t_start=8.6,
        n_frames=N_FRAMES,
    )
    focal_length = seq.calib.fx
    frames = list(seq)          # (V, t) pairs
    return frames, focal_length


def make_synthetic_source():
    """Generate synthetic event frames from a natural-looking image."""
    from simulation import DVSSimulator
    R_true = np.array([0.008, 0.015, 0.003])
    sim = DVSSimulator(
        H=H, W=W, f=64.0,
        image_kind='random',
        noise_std=0.003,
        rng_seed=42,
    )
    frames = [(sim.frame_from_rotation(R_true), None) for _ in range(N_FRAMES)]
    return frames, 64.0


use_real = os.path.isfile(EVENTS_FILE) and os.path.isfile(CALIB_FILE)
if use_real:
    print("Using real event-camera data (shapes_rotation dataset).")
    frames, focal_length = make_real_source()
else:
    print("Real data not found. Using synthetic simulation.")
    print(f"  To use real data: download shapes_rotation.zip from")
    print(f"  http://rpg.ifi.uzh.ch/datasets/davis/shapes_rotation.zip")
    print(f"  and extract events.txt + calib.txt to  {DATA_DIR}/")
    frames, focal_length = make_synthetic_source()

C = compute_calibration(H, W, focal_length)


# ---------------------------------------------------------------------------
# Build network
# ---------------------------------------------------------------------------

net = InteractingMaps(H=H, W=W, f=focal_length, **NET_PARAMS)
net.reset(scale=0.01)

initial_R = np.array([0.0, 0.0, 0.026])
 
USE_THESIS_VERSION = True

if USE_THESIS_VERSION:
    from interacting_maps.network_dissertation import InteractingMapsThesis
    net = InteractingMapsThesis(H=H, W=W, f=focal_length, **NET_PARAMS)
    net.q_R.value = initial_R
else:
    from interacting_maps.network import InteractingMaps
    net = InteractingMaps(H=H, W=W, f=focal_length, **NET_PARAMS)
    net.R = initial_R

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
ax_res.set_xlim(0, N_FRAMES)
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
        f'Frame {t+1:3d}/{N_FRAMES}  '
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
    frames=N_FRAMES,
    interval=100,
    blit=False,
    repeat=False,
)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()

print('\nFinal estimated rotation R:')
print(f'  R = {net.R}')
print(f'  |R| = {np.linalg.norm(net.R):.4e}')
