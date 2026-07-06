"""
Test whether the stability clipping in InteractingMapsThesis.step()
is affecting validation accuracy.

Strategy
--------
1. Load shapes_rotation events + groundtruth.
2. For each frame: reseed R from GT (perfect oracle), run inference,
   compare omega_est vs omega_GT.
3. Do this twice:
   - Run A: default clips  (I±10, G±5, F±10, R±1)
   - Run B: loose clips    (everything ±1e6 — effectively off)
4. Also report per-frame saturation counts: how many pixels actually hit
   the clip limit. If saturation count is 0 in Run A, the clipping is
   inert and not causing the validation gap.
"""

import os
import numpy as np

# Suppress the EventFrameSequence print noise
import io
import contextlib

from config import DATASET_CONFIGS, THESIS_PARAMS, ITERS_PER_FRAME, F_DECAY, G_DECAY, get_dataset_paths
from data_loader import EventFrameSequence
from interacting_maps.network_dissertation import InteractingMapsThesis

# Reuse the GT helpers from validation.py
from validation import gt_omega_body, load_groundtruth


DATASET = 'poster_rotation'
cfg     = DATASET_CONFIGS[DATASET]
paths   = get_dataset_paths(DATASET)

T_START        = cfg['t_start']
FRAME_DURATION = cfg['frame_duration']
N_FRAMES       = cfg['n_frames']
initial_R      = cfg['initial_R']


# ---------------------------------------------------------------------------
# Monkey-patched step(): parametrised clip bounds + saturation counters
# ---------------------------------------------------------------------------

def make_clipped_step(clip_I, clip_G, clip_F, clip_R):
    """Return a step() implementation that uses the supplied clip bounds."""

    def step(self, V, n_iters=50, f_decay=0.5, g_decay=0.7):
        self.q_V.value = V

        if f_decay > 0.0:
            F_kin = np.einsum('hwij,j->hwi', self._C_mat, self.q_R.value)
            self.q_F.value = (1.0 - f_decay) * self.q_F.value + f_decay * F_kin
        if g_decay < 1.0:
            self.q_G.value *= g_decay

        sat_I = sat_G = sat_F = sat_R = 0

        for _ in range(n_iters):
            for q in [self.q_I, self.q_G, self.q_F, self.q_R]:
                q.reset_gradient()
            for cost in self.costs:
                cost.compute_and_send_gradients()
            self.q_I.update(self.lr_I)
            self.q_G.update(self.lr_G)
            self.q_F.update(self.lr_F)
            self.q_R.update(self.lr_R)

            # Count saturations BEFORE clipping
            sat_I += int(np.sum(np.abs(self.q_I.value) >= clip_I))
            sat_G += int(np.sum(np.abs(self.q_G.value) >= clip_G))
            sat_F += int(np.sum(np.abs(self.q_F.value) >= clip_F))
            sat_R += int(np.sum(np.abs(self.q_R.value) >= clip_R))

            self.q_I.value = np.clip(self.q_I.value, -clip_I, clip_I)
            self.q_G.value = np.clip(self.q_G.value, -clip_G, clip_G)
            self.q_F.value = np.clip(self.q_F.value, -clip_F, clip_F)
            self.q_R.value = np.clip(self.q_R.value, -clip_R, clip_R)

        # Stash the per-frame totals on the network for the caller to read
        self._sat = (sat_I, sat_G, sat_F, sat_R)
    return step


# ---------------------------------------------------------------------------
# Single-run helper
# ---------------------------------------------------------------------------

def run_one(label, clip_I, clip_G, clip_F, clip_R, gt_data):
    print(f"\n{'='*78}")
    print(f"RUN: {label}   (clips: I±{clip_I}, G±{clip_G}, F±{clip_F}, R±{clip_R})")
    print(f"{'='*78}")

    # Silence the EventFrameSequence prints
    with contextlib.redirect_stdout(io.StringIO()):
        seq = EventFrameSequence(
            paths['events'], paths['calib'],
            frame_duration=FRAME_DURATION,
            t_start=T_START,
            n_frames=N_FRAMES,
            clip_value=10.0,
        )

    H, W = seq.H, seq.W
    fx, fy = seq.calib.fx, seq.calib.fy
    cx, cy = seq.calib.cx, seq.calib.cy

    net = InteractingMapsThesis(H=H, W=W, fx=fx, fy=fy, cx=cx, cy=cy, **THESIS_PARAMS)
    net.initialize_from_rotation(initial_R)

    # Bind the parametrised step to this instance
    net.step = make_clipped_step(clip_I, clip_G, clip_F, clip_R).__get__(net, InteractingMapsThesis)

    n_pixels = H * W

    print(f"{'k':>3}  {'t_mid':>7}  {'Est wx':>8} {'Est wy':>8} {'Est wz':>8}   "
          f"{'GT  wx':>8} {'GT  wy':>8} {'GT  wz':>8}   "
          f"{'err deg/s':>9}   "
          f"{'satI%':>6} {'satG%':>6} {'satF%':>6} {'satR':>4}")

    err_list = []
    sat_totals = [0, 0, 0, 0]

    for k, (V, t_mid) in enumerate(seq):
        omega_ref = gt_omega_body(gt_data, t_mid, FRAME_DURATION)

        # RESEED R from GT each frame (per-frame oracle)
        net.q_R.value = (omega_ref * FRAME_DURATION).copy()

        net.step(V, n_iters=ITERS_PER_FRAME, f_decay=F_DECAY, g_decay=G_DECAY)

        omega_est = net.R / FRAME_DURATION
        err = float(np.degrees(np.linalg.norm(omega_est - omega_ref)))
        err_list.append(err)

        sI, sG, sF, sR = net._sat
        # Express I/G/F saturation as percent of (pixels × iters) opportunities
        denom = n_pixels * ITERS_PER_FRAME
        denom_R = 3 * ITERS_PER_FRAME
        for i, s in enumerate((sI, sG, sF, sR)):
            sat_totals[i] += s

        print(f"{k+1:3d}  {t_mid:7.3f}  "
              f"{omega_est[0]:+8.4f} {omega_est[1]:+8.4f} {omega_est[2]:+8.4f}   "
              f"{omega_ref[0]:+8.4f} {omega_ref[1]:+8.4f} {omega_ref[2]:+8.4f}   "
              f"{err:9.2f}   "
              f"{100*sI/denom:6.3f} {100*sG/denom:6.3f} {100*sF/denom:6.3f} {sR:4d}")

    err_arr = np.array(err_list)
    print(f"\n  Mean error vs GT   : {err_arr.mean():.3f} deg/s")
    print(f"  Median error vs GT : {np.median(err_arr):.3f} deg/s")
    print(f"  Total saturation hits over the whole run:")
    print(f"    I: {sat_totals[0]:>10d}   G: {sat_totals[1]:>10d}   "
          f"F: {sat_totals[2]:>10d}   R: {sat_totals[3]:>10d}")
    return err_arr, sat_totals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    gt_data = load_groundtruth(paths['groundtruth'])

    err_A, sat_A = run_one('A — DEFAULT clips',
                           clip_I=10.0, clip_G=5.0, clip_F=10.0, clip_R=1.0,
                           gt_data=gt_data)

    err_B, sat_B = run_one('B — LOOSE clips (effectively disabled)',
                           clip_I=1e6,  clip_G=1e6, clip_F=1e6, clip_R=1e6,
                           gt_data=gt_data)

    print(f"\n{'='*78}")
    print("COMPARISON")
    print(f"{'='*78}")
    print(f"  Mean err  (A default) : {err_A.mean():.4f} deg/s")
    print(f"  Mean err  (B loose)   : {err_B.mean():.4f} deg/s")
    print(f"  Delta mean err        : {err_B.mean() - err_A.mean():+.4f} deg/s")
    print(f"  Max |delta per-frame| : {np.max(np.abs(err_B - err_A)):.4f} deg/s")
    print(f"  Frames where loose was better : "
          f"{int(np.sum(err_B < err_A))} / {len(err_A)}")
    print(f"\n  Clip activity in run A:")
    print(f"    I: {sat_A[0]:>10d}   G: {sat_A[1]:>10d}   "
          f"F: {sat_A[2]:>10d}   R: {sat_A[3]:>10d}")
    if sum(sat_A) == 0:
        print("\n  → Run A had ZERO clip hits → the stability bounds are inert.")
        print("    The validation gap is NOT caused by clipping.")
    else:
        print("\n  → Clip hits in run A.  If err_A > err_B by a meaningful amount,")
        print("    the clip bounds are biasing the estimate.")
