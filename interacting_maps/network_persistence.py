"""
network_persistence.py — the thesis network plus Martel's cross-frame terms.

Spec: docs/superpowers/specs/2026-11-28-persistence-block-matching-design.md

The base thesis network (Martel 2019 §6.4, our `thesis`) has no notion of time
between frames, so the gauge F → sF, G → G/s is free and vision-only ω stays at
its warm start. Martel fixes this with terms that compare the image ACROSS
frames, which brings the real time step in:

- persistence (§6.5, Eqs. 6.68–6.100): I must equal the previous image Ǐ
  warped by the inter-frame rotation R_δ = exp([R]×);
- block-matching flow (§6.8.3, Eqs. 6.138–6.139, Fig. 6.17): a flow F′ found
  by matching I against Ǐ, tied to F.

InteractingMapsPersist subclasses InteractingMapsThesis; the base network is
not edited. Deviations from Martel (spec, "Deviations from Martel"):
  1. no world map W, no absolute orientation (Eqs. 6.101–6.111);
  2. R is our per-frame rotation vector (Ω·Δt), so R_δ = exp([R]×) and the
     Ω–R relation is an identity;
  3. the warp goes through our camera model incl. lens distortion (C_full);
  4. persistence and block matching can be combined (thesis_persist_bm);
  5. frame-0 R comes from the harness (gyro warm start or R = 0), which
     Martel does not specify.
"""

import numpy as np

from interacting_maps.network_dissertation import Cost, Quantity, InteractingMapsThesis
from interacting_maps.rotation_warp import (Camera, pixel_rays, warp_coords,
                                            bilinear_sample, solve_rotation_gn,
                                            block_match)


class Cost_Persistence(Cost):
    """
    Persistence: I(p) = Ǐ(T_R(p))  (Eq. 6.72), quantities I, Ǐ, R.

    R side (Eqs. 6.73–6.98): Gauss–Newton from the current R to convergence
    gives R*; R takes a step toward it, gradient δ_persist_R·(R − R*) — the
    "candidate, then small step" pattern of Cost_Kinematics.
    I side (Eq. 6.100): I ← I − η·(I − Ǐ∘T_R), with the current R.
    Ǐ is not updated (Martel: "we will not update Ǐ"). Pixels warped outside
    the image are excluded from both sides. Inactive until set_reference().
    """

    def __init__(self, quantities_dict, rays, cam, delta_persist_R, eta_persist_I,
                 gn_iters=20, gn_tol=1e-8):
        super().__init__(quantities_dict)
        self.rays, self.cam = rays, cam
        self.delta_persist_R = delta_persist_R
        self.eta_persist_I = eta_persist_I
        self.gn_iters, self.gn_tol = gn_iters, gn_tol
        self.active = False
        self._grad = None

    def set_reference(self, Icheck):
        """New frame: Ǐ is fixed for the frame, so its gradient is computed once."""
        gy, gx = np.gradient(Icheck)
        self._grad = (gx, gy)
        self.active = True

    def compute_and_send_gradients(self):
        if not self.active:
            return
        I = self.q['I'].value
        Ic = self.q['Icheck'].value
        R = self.q['R'].value
        R_star = solve_rotation_gn(I, Ic, self._grad, self.rays, self.cam, R,
                                   n_max=self.gn_iters, tol=self.gn_tol)
        self.q['R'].add_gradient(self.delta_persist_R * (R - R_star))
        Iw, valid = bilinear_sample(Ic, *warp_coords(self.rays, R, self.cam))
        self.q['I'].add_gradient(self.eta_persist_I * (I - Iw) * valid)


class Cost_BlockMatching(Cost):
    """
    Block-matching flow (Eq. 6.138–6.139 + the F′–F link of Fig. 6.17),
    quantities I, Ǐ, F: F′ = block_match(I, Ǐ), then F steps toward it,
    gradient δ_BM·(F − F′) on pixels where the search fits in the image.
    I is not updated by this cost. Inactive until set_reference().
    """

    def __init__(self, quantities_dict, delta_BM, half_block=3, radius=4):
        super().__init__(quantities_dict)
        self.delta_BM = delta_BM
        self.half_block, self.radius = half_block, radius
        self.active = False

    def set_reference(self, Icheck):
        self.active = True

    def compute_and_send_gradients(self):
        if not self.active:
            return
        F_prime, valid = block_match(self.q['I'].value, self.q['Icheck'].value,
                                     self.half_block, self.radius)
        self.q['F'].add_gradient(self.delta_BM * (self.q['F'].value - F_prime)
                                 * valid[..., None])


class InteractingMapsPersist(InteractingMapsThesis):
    """
    Thesis network + persistence and/or block-matching costs (models
    thesis_persist, thesis_bm, thesis_persist_bm). Frame 0 behaves exactly
    like the thesis network: there is no previous image yet.
    """

    def __init__(self, H, W, fx, fy, cx, cy, frame_duration=0.005, *,
                 use_persistence=True, use_block_matching=False,
                 delta_persist_R=0.10, eta_persist_I=0.05, delta_BM=0.05,
                 bm_half_block=3, bm_radius=4, gn_iters=20, gn_tol=1e-8,
                 dist_coeffs=None, include_jacobian=True, poisson='iterative',
                 **thesis_deltas):
        super().__init__(H, W, fx, fy, cx, cy, frame_duration=frame_duration,
                         dist_coeffs=dist_coeffs, include_jacobian=include_jacobian,
                         poisson=poisson, **thesis_deltas)
        self.q_Icheck = Quantity((H, W), "Intensity_previous")
        self.q_dict['Icheck'] = self.q_Icheck
        cam = Camera(fx, fy, cx, cy,
                     None if dist_coeffs is None else np.asarray(dist_coeffs, np.float64))
        self.persist_costs = []
        if use_persistence:
            self.persist_costs.append(Cost_Persistence(
                self.q_dict, pixel_rays(H, W, cam), cam, delta_persist_R,
                eta_persist_I, gn_iters, gn_tol))
        if use_block_matching:
            self.persist_costs.append(Cost_BlockMatching(
                self.q_dict, delta_BM, bm_half_block, bm_radius))
        self.costs.extend(self.persist_costs)
        self._n_frames_seen = 0

    def step(self, V, n_iters=50, omega_imu=None, events=None):
        """One FRAME: Ǐ ← I (from the second call on), then n_iters iterations.

        Unlike the base network, every call starts a new frame. Do not step this
        network one iteration per call (`step(V, n_iters=1)` in a loop, as the
        iteration-snapshot diagnostics do): Ǐ would be reset every iteration.
        """
        if self._n_frames_seen > 0:
            # Martel: Ǐ = I(t − δt); I is NOT pre-warped — at the first
            # iteration of a frame I equals Ǐ, then the events and the costs
            # move it (spec, "Frame start").
            self.q_Icheck.value = self.q_I.value.copy()
            for c in self.persist_costs:
                c.set_reference(self.q_Icheck.value)
        self._n_frames_seen += 1
        super().step(V, n_iters=n_iters, omega_imu=omega_imu, events=events)
