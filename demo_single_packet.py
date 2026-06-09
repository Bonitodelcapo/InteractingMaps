"""
Single Event Packet — Iteration-by-Iteration Visualization.

Loads ONE event packet (one fixed V frame) and runs the network one iteration
at a time. The animation shows how maps I, G, F, R emerge and interact:

  - Early iterations:  F starts to align with V via OFCE (V + F·G = 0)
  - Mid iterations:    G couples F to I via the spatial constraint (G = ∇I)
  - Late iterations:   I reveals scene structure; R converges to rotation

The residual plot (bottom right) is the clearest convergence signal.

Run:
    python demo_single_packet.py
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

from config import DATASET_CONFIGS, THESIS_PARAMS, F_DECAY, G_DECAY, get_dataset_paths

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET          = 'poster_rotation'   # which dataset to pull the packet from
MAX_ITERS        = 100                 # inference iterations to run
ANIM_INTERVAL_MS = 120                 # ms per animation frame (one iteration)
CLIP_VALUE       = 10.0               # event accumulation clip for real data

# ---------------------------------------------------------------------------
# Init mode:
#   USE_GT_INIT = False  →  use the approximate initial_R from config.py
#   USE_GT_INIT = True   →  compute the EXACT omega from groundtruth.txt and
#                            use it as R_init (perfect-conditions test)
#                            Requires groundtruth.txt in the dataset folder.
# ---------------------------------------------------------------------------
USE_GT_INIT = False

cfg   = DATASET_CONFIGS[DATASET]
paths = get_dataset_paths(DATASET)

# ---------------------------------------------------------------------------
# Load ONE event packet (one time-window → one V frame)
# ---------------------------------------------------------------------------

use_real = os.path.isfile(paths['events']) and os.path.isfile(paths['calib'])

if use_real:
    from data_loader import EventFrameSequence
    seq = EventFrameSequence(
        paths['events'], paths['calib'],
        frame_duration=cfg['frame_duration'],
        t_start=cfg['t_start'],
        n_frames=1,
        clip_value=CLIP_VALUE,
    )
    frames = list(seq)
    V, t_mid = frames[0]
    fx, fy = seq.calib.fx, seq.calib.fy
    cx, cy = seq.calib.cx, seq.calib.cy
    H, W   = seq.H, seq.W
    print(f"Loaded 1 event packet at t={t_mid:.3f} s  ({DATASET})")
    print(f"  V: max={np.abs(V).max():.3f},  sparsity={np.mean(V == 0)*100:.1f}%")
else:
    from simulation import DVSSimulator
    H, W    = 128, 128
    fx = fy = 64.0
    cx, cy  = W / 2.0, H / 2.0
    R_synth = np.array([0.008, 0.015, 0.003])
    sim = DVSSimulator(H=H, W=W, f=64.0, image_kind='random',
                       noise_std=0.003, rng_seed=42)
    V, t_mid = sim.frame_from_rotation(R_synth), None
    print("Real data not found — using synthetic event packet.")
    print(f"  Ground-truth R = {R_synth}")

# ---------------------------------------------------------------------------
# Ground-truth init helpers (inline to avoid import-time side effects)
# ---------------------------------------------------------------------------

def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Quaternion (qx,qy,qz,qw) → 3×3 rotation matrix R_wc."""
    qx, qy, qz, qw = q / np.linalg.norm(q)
    return np.array([
        [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),  2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2),  2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
    ])

def _gt_omega_body(gt_data: np.ndarray, t_mid: float, dt: float) -> np.ndarray:
    """
    Ground-truth angular velocity in camera body frame from two quaternion poses.
    dR_body = R1.T @ R2  (right-invariant / body-frame convention).
    """
    t_lo, t_hi = t_mid - dt/2.0, t_mid + dt/2.0
    i1 = int(np.argmin(np.abs(gt_data[:, 0] - t_lo)))
    i2 = int(np.argmin(np.abs(gt_data[:, 0] - t_hi)))
    if i1 == i2:
        i2 = min(i1 + 1, len(gt_data) - 1)
    actual_dt = gt_data[i2, 0] - gt_data[i1, 0]
    if abs(actual_dt) < 1e-10:
        return np.zeros(3)
    R1 = _quat_to_rotmat(gt_data[i1, 4:8])
    R2 = _quat_to_rotmat(gt_data[i2, 4:8])
    dR = R1.T @ R2
    cos_a = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_a)
    if abs(angle) < 1e-10:
        return np.zeros(3)
    skew = (dR - dR.T) / (2.0 * np.sin(angle) + 1e-15)
    axis = np.array([skew[2,1], skew[0,2], skew[1,0]])
    return axis * angle / actual_dt   # rad/s in body frame

# ---------------------------------------------------------------------------
# Initialize network
# ---------------------------------------------------------------------------

from interacting_maps.network_dissertation import InteractingMapsThesis

net = InteractingMapsThesis(H=H, W=W, fx=fx, fy=fy, cx=cx, cy=cy, **THESIS_PARAMS)

gt_omega_true = None   # will be set if USE_GT_INIT succeeds

if USE_GT_INIT and use_real and os.path.isfile(paths['groundtruth']) and t_mid is not None:
    gt_data   = np.loadtxt(paths['groundtruth'], dtype=np.float64)
    gt_omega_true = _gt_omega_body(gt_data, t_mid, cfg['frame_duration'])
    R_init    = gt_omega_true * cfg['frame_duration']  # rad/s → rad/frame
    net.initialize_from_rotation(R_init)
    print(f"\nNetwork initialised with GT omega (USE_GT_INIT=True)")
    print(f"  GT  ω  = {gt_omega_true}  rad/s")
    print(f"  R_init = {R_init}  rad/frame")
else:
    R_init = cfg['initial_R']
    net.initialize_from_rotation(R_init)
    if USE_GT_INIT:
        print("\nUSE_GT_INIT=True but groundtruth.txt not found — using config init.")
    print(f"\nNetwork initialised with config initial_R")
    print(f"  R_init = {R_init}  rad/frame")

print(f"  OFCE residual at t=0: {net.residual_VFG(V):.4f}")
print(f"\nRunning {MAX_ITERS} iterations on the SAME event packet …\n")

# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def normalise(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-10)

def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """HSV: hue = direction, brightness = magnitude."""
    fu, fv = flow[..., 0], flow[..., 1]
    angle  = (np.arctan2(fv, fu) + np.pi) / (2.0 * np.pi)
    mag    = np.sqrt(fu**2 + fv**2)
    hsv    = np.stack([angle, np.ones_like(angle), normalise(mag)], axis=-1)
    return matplotlib.colors.hsv_to_rgb(hsv)

def flow_norm(field: np.ndarray) -> np.ndarray:
    return np.sqrt(field[..., 0]**2 + field[..., 1]**2)

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
#
#  ┌───────┬───────┬───────┬───────┐
#  │  V    │  I    │  |G|  │  F    │   ← spatial maps (top row)
#  │(fixed)│(iter) │(iter) │(iter) │
#  ├───────┴───────┼───────┴───────┤
#  │  R over iters │ Residuals     │   ← convergence signals (bottom row)
#  └───────────────┴───────────────┘

fig = plt.figure(figsize=(16, 8))
gs  = GridSpec(2, 4, figure=fig, hspace=0.42, wspace=0.30)

ax_V   = fig.add_subplot(gs[0, 0])
ax_I   = fig.add_subplot(gs[0, 1])
ax_G   = fig.add_subplot(gs[0, 2])
ax_F   = fig.add_subplot(gs[0, 3])
ax_R   = fig.add_subplot(gs[1, :2])
ax_res = fig.add_subplot(gs[1, 2:])

# ---- Static V panel (shown once, never updated) ----
vabs = max(float(np.percentile(np.abs(V), 95)), 1e-3)
im_V = ax_V.imshow(V, cmap='RdBu', vmin=-vabs, vmax=vabs)
ax_V.set_title('V  (input, fixed)', fontsize=8)
ax_V.axis('off')

# ---- Evolving map panels ----
blank  = np.zeros((H, W))
blank3 = np.zeros((H, W, 3))

im_I = ax_I.imshow(blank, cmap='gray', vmin=0, vmax=1)
im_G = ax_G.imshow(blank, cmap='hot',  vmin=0, vmax=1)
im_F = ax_F.imshow(blank3)

for ax, title in [(ax_I, 'I  (intensity)'),
                  (ax_G, '|G|  (gradient mag.)'),
                  (ax_F, 'F  (flow, HSV)')]:
    ax.set_title(title, fontsize=8)
    ax.axis('off')

# ---- R history (line plot) ----
ax_R.set_title('R components over iterations', fontsize=8)
ax_R.set_xlabel('Iteration', fontsize=7)
ax_R.set_ylabel('rad/frame', fontsize=7)
ax_R.axhline(0, color='k', linewidth=0.5)
ax_R.set_xlim(0, MAX_ITERS)
ax_R.grid(True, alpha=0.25)
line_Rx, = ax_R.plot([], [], color='#4e79a7',  lw=1.5, label='Rx')
line_Ry, = ax_R.plot([], [], color='#f28e2b',  lw=1.5, label='Ry')
line_Rz, = ax_R.plot([], [], color='#e15759',  lw=1.5, label='Rz')
ax_R.legend(fontsize=7, loc='upper right')

# ---- Residual history (semilogy) ----
ax_res.set_title('Constraint residuals over iterations', fontsize=8)
ax_res.set_xlabel('Iteration', fontsize=7)
ax_res.set_xlim(0, MAX_ITERS)
ax_res.set_ylim(1e-5, 2.0)
ax_res.grid(True, alpha=0.25, which='both')
line_ofce, = ax_res.semilogy([], [], 'b-', lw=1.5, label='|V + F·G|  (OFCE)')
line_spat, = ax_res.semilogy([], [], 'r-', lw=1.5, label='|G − ∇I|   (Spatial)')
ax_res.legend(fontsize=7)

_init_label = "GT init" if (USE_GT_INIT and os.path.isfile(paths.get('groundtruth', ''))) \
              else "config init"
fig.suptitle(
    f'Single Event Packet — Iteration 0 / {MAX_ITERS}   [{DATASET}]   [{_init_label}]\n'
    f'V is fixed.  Watch I, G, F, R emerge from the inference loop.',
    fontsize=10,
)

# ---- History buffers ----
iters        = []
R_hist       = []
res_ofce_buf = []
res_spat_buf = []

# ---------------------------------------------------------------------------
# Animation callback — ONE inference iteration per frame
# ---------------------------------------------------------------------------

def update(i: int):
    if i >= MAX_ITERS:
        return im_I, im_G, im_F, line_Rx, line_Ry, line_Rz, line_ofce, line_spat

    # --- ONE inference iteration ---
    net.step(V, n_iters=1, f_decay=F_DECAY, g_decay=G_DECAY)

    # --- Record history ---
    iters.append(i + 1)
    R_hist.append(net.R.copy())
    res_ofce_buf.append(net.residual_VFG(V) + 1e-9)
    res_spat_buf.append(net.residual_GI()   + 1e-9)

    # --- Update spatial map images ---
    im_I.set_data(normalise(net.I))
    im_G.set_data(normalise(flow_norm(net.G)))
    im_F.set_data(flow_to_rgb(net.F))

    # --- Update R line plot ---
    R_arr = np.array(R_hist)   # (i+1, 3)
    line_Rx.set_data(iters, R_arr[:, 0])
    line_Ry.set_data(iters, R_arr[:, 1])
    line_Rz.set_data(iters, R_arr[:, 2])
    r_lo = min(-0.005, R_arr.min() * 1.3)
    r_hi = max( 0.005, R_arr.max() * 1.3)
    ax_R.set_ylim(r_lo, r_hi)

    # --- Update residual plot ---
    line_ofce.set_data(iters, res_ofce_buf)
    line_spat.set_data(iters, res_spat_buf)

    # --- Update title ---
    fig.suptitle(
        f'Single Event Packet — Iteration {i+1} / {MAX_ITERS}   [{DATASET}]\n'
        f'OFCE: {res_ofce_buf[-1]:.4f}   Spatial: {res_spat_buf[-1]:.5f}   '
        f'|R| = {np.linalg.norm(net.R):.4f} rad/frame',
        fontsize=10,
    )

    # --- Console progress every 50 iterations ---
    if (i + 1) % 50 == 0:
        omega = net.R / cfg['frame_duration']
        print(f"  iter {i+1:4d}  OFCE={res_ofce_buf[-1]:.4f}  "
              f"Spat={res_spat_buf[-1]:.5f}  "
              f"ω=[{omega[0]:+.3f} {omega[1]:+.3f} {omega[2]:+.3f}] rad/s")

    return im_I, im_G, im_F, line_Rx, line_Ry, line_Rz, line_ofce, line_spat


ani = animation.FuncAnimation(
    fig, update,
    frames=MAX_ITERS,
    interval=ANIM_INTERVAL_MS,
    blit=False,
    repeat=False,
)

plt.tight_layout(rect=[0, 0, 1, 0.88])
plt.show()

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

omega_est = net.R / cfg['frame_duration']
omega_exp = cfg['expected_omega']

print(f'\n{"="*60}')
print(f'Final state after {MAX_ITERS} iterations on ONE event packet')
init_mode = "GT init" if (USE_GT_INIT and gt_omega_true is not None) else "config init"
print(f'Init mode: {init_mode}')
print(f'{"="*60}')
print(f'  Estimated R   = {net.R}  rad/frame')
print(f'  Estimated ω   = {omega_est}  rad/s')
if gt_omega_true is not None:
    err_gt = np.linalg.norm(omega_est - gt_omega_true)
    print(f'  GT        ω   = {gt_omega_true}  rad/s')
    print(f'  Error vs GT   = {err_gt:.4f} rad/s  ({np.degrees(err_gt):.1f}°/s)')
err_exp = np.linalg.norm(omega_est - omega_exp)
print(f'  Expected  ω   = {omega_exp}  rad/s')
print(f'  Error vs exp  = {err_exp:.4f} rad/s  ({np.degrees(err_exp):.1f}°/s)')
print(f'  OFCE residual = {res_ofce_buf[-1]:.4e}')
print(f'  Spatial resid = {res_spat_buf[-1]:.4e}')
