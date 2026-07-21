"""
CMax angular-velocity estimator (front-end), ported from Gallego's `cmax_slam`.

Reference implementation (C++): tub-rip/cmax_slam, src/frontend/
  - ang_vel_estimator.cpp        (warm-start = previous ω; reference time = window midpoint)
  - local_image_warped_events.cpp(linear warp p_rot = p3D + (ω·dt)×p3D; bilinear voting)
  - local_focus_funcs.cpp        (objective = variance of the IWE, maximized)
  - local_optim_contrast_gsl.cpp (GSL conjugate-gradient with analytic gradient)

Adaptations for InteractingMaps (V1):
  - Fixed-TIME window (one frame = `frame_duration`), NOT a fixed event count.
    One ω is produced per frame; warm-started from the previous frame's ω.
  - Optimizer: SciPy Nelder-Mead (derivative-free) for a first, dependency-light
    port. The objective is smoothed by a Gaussian blur of the IWE (standard in
    CMax) so the landscape is well-behaved without an analytic gradient.

Everything is in the CAMERA BODY FRAME, ω in rad/s — the same convention as the
network's R (rad/frame = ω·dt) and as `evaluation.gt_omega_body`.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize


class CMaxAngularVelocity:
    """
    Estimate camera angular velocity ω (rad/s) from a window of events by
    maximizing the contrast (variance) of the Image of Warped Events (IWE).

    Parameters
    ----------
    H, W          : sensor size (pixels)
    fx, fy, cx, cy: pinhole intrinsics
    use_polarity  : accumulate signed polarity (True) or event count (False)
    blur_sigma    : Gaussian blur (px) applied to the IWE before variance;
                    smooths the objective (Gallego blurs the IWE too)
    optimizer     : SciPy method ('Nelder-Mead' default)
    """

    def __init__(self, H, W, fx, fy, cx, cy,
                 use_polarity=True, blur_sigma=1.0, optimizer='Nelder-Mead',
                 use_analytic_gradient=False):
        self.H, self.W = H, W
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.use_polarity = use_polarity
        self.blur_sigma = blur_sigma
        self.optimizer = optimizer

        # When True: use the analytic gradient of the (UNBLURRED) IWE variance
        # with a gradient-based optimizer (conjugate gradient). When False:
        # derivative-free Nelder-Mead on the blurred variance (default, robust).
        #
        # Performance: analytic+CG is usually FASTER *and* more accurate — CG
        # converges in ~10-30 iters using the gradient, vs Nelder-Mead's
        # ~100-400 objective evals. Each gradient eval costs a bit more (warp
        # Jacobians), but far fewer are needed. The trade-off is robustness:
        # the unblurred variance has a rougher landscape, so a good warm-start
        # (previous ω) matters more.
        self.use_analytic_gradient = use_analytic_gradient
        if use_analytic_gradient and optimizer == 'Nelder-Mead':
            self.optimizer = 'CG'

        # Diagnostics from the last estimate()
        self.last_iwe = None
        self.last_result = None

    # ------------------------------------------------------------------
    # Event → bearing vector (undistorted pinhole)
    # ------------------------------------------------------------------
    def _bearings(self, xs, ys):
        """
        Undistorted pixel (xs, ys) → unit bearing vectors (N, 3) in the camera
        frame. Assumes events are already undistorted (as the pipeline delivers
        them); a pure pinhole back-projection.
        """
        x_n = (xs - self.cx) / self.fx
        y_n = (ys - self.cy) / self.fy
        p = np.stack([x_n, y_n, np.ones_like(x_n)], axis=-1)   # (N, 3)
        p /= np.linalg.norm(p, axis=-1, keepdims=True)
        return p

    # ------------------------------------------------------------------
    # Warp + IWE
    # ------------------------------------------------------------------
    def _warp_to_pixels(self, bearings, dt, omega):
        """
        Linear (first-order) rotational warp of bearing vectors, then reproject.
            p_rot = p + (ω·dt) × p           (Gallego's linear warp)
            (x', y') = intrinsics( p_rot / p_rot_z )
        dt : (N,) time offset of each event from the reference time (s)
        omega : (3,) rad/s
        Returns warped pixel coords (xw, yw), each (N,).
        """
        delta_rot = dt[:, None] * omega[None, :]          # (N, 3) = ω·dt
        p_rot = bearings + np.cross(delta_rot, bearings)  # (N, 3)
        z = p_rot[:, 2]
        z = np.where(np.abs(z) < 1e-9, 1e-9, z)
        xw = self.fx * (p_rot[:, 0] / z) + self.cx
        yw = self.fy * (p_rot[:, 1] / z) + self.cy
        return xw, yw

    def _accumulate_iwe(self, xw, yw, weights):
        """Bilinear voting of warped events into an (H, W) IWE."""
        iwe = np.zeros((self.H, self.W), dtype=np.float64)
        x0 = np.floor(xw).astype(np.int64)
        y0 = np.floor(yw).astype(np.int64)
        dx = xw - x0
        dy = yw - y0
        for ox, wx in ((0, 1.0 - dx), (1, dx)):
            for oy, wy in ((0, 1.0 - dy), (1, dy)):
                xi = x0 + ox
                yi = y0 + oy
                w = weights * wx * wy
                m = (xi >= 0) & (xi < self.W) & (yi >= 0) & (yi < self.H)
                np.add.at(iwe, (yi[m], xi[m]), w[m])
        return iwe

    def _build_iwe(self, bearings, dt, weights, omega):
        xw, yw = self._warp_to_pixels(bearings, dt, omega)
        return self._accumulate_iwe(xw, yw, weights)

    def _contrast(self, omega, bearings, dt, weights):
        """Variance of the (blurred) IWE — the quantity to MAXIMIZE."""
        iwe = self._build_iwe(bearings, dt, weights, omega)
        if self.blur_sigma > 0:
            iwe = gaussian_filter(iwe, self.blur_sigma)
        return float(np.var(iwe))

    # ------------------------------------------------------------------
    # Analytic gradient of the (unblurred) IWE variance
    # ------------------------------------------------------------------
    def _warp_and_jacobian(self, bearings, dt, omega):
        """
        Warp bearing vectors and also return the per-event Jacobian
        Jₑ = ∂(x'ₑ, y'ₑ)/∂ω  (N, 2, 3).

        p_rot = p + dt·(ω × p)        →  ∂p_rot/∂ω = -dt·[p]×
        (x', y') = intrinsics(p_rot/z) →  ∂(x',y')/∂p_rot = P_proj(p_rot)
        Jₑ = P_proj(p_rot) · (-dt·[p]×)
        """
        delta_rot = dt[:, None] * omega[None, :]
        p_rot = bearings + np.cross(delta_rot, bearings)     # (N,3)
        px, py, pz = p_rot[:, 0], p_rot[:, 1], p_rot[:, 2]
        pz = np.where(np.abs(pz) < 1e-9, 1e-9, pz)
        xw = self.fx * (px / pz) + self.cx
        yw = self.fy * (py / pz) + self.cy

        N = bearings.shape[0]
        # Projection Jacobian P_proj = ∂(x',y')/∂p_rot   (N,2,3)
        Pp = np.zeros((N, 2, 3))
        Pp[:, 0, 0] = self.fx / pz
        Pp[:, 0, 2] = -self.fx * px / (pz * pz)
        Pp[:, 1, 1] = self.fy / pz
        Pp[:, 1, 2] = -self.fy * py / (pz * pz)
        # Skew of the ORIGINAL bearing p (∂p_rot/∂ω = -dt·[p]×)   (N,3,3)
        bx, by, bz = bearings[:, 0], bearings[:, 1], bearings[:, 2]
        skew = np.zeros((N, 3, 3))
        skew[:, 0, 1] = -bz; skew[:, 0, 2] = by
        skew[:, 1, 0] = bz;  skew[:, 1, 2] = -bx
        skew[:, 2, 0] = -by; skew[:, 2, 1] = bx
        dprot_dw = -dt[:, None, None] * skew                 # (N,3,3)
        J = np.einsum('nij,njk->nik', Pp, dprot_dw)          # (N,2,3)
        return xw, yw, J

    def _contrast_and_grad(self, omega, bearings, dt, weights):
        """
        (value, gradient) of the UNBLURRED IWE variance w.r.t. ω, computed
        consistently with the bilinear voting (exact for the discrete IWE).

        IWE:  I_u = Σ_e w_e K(u − x'ₑ),  K = bilinear kernel.
        Deriv image:  D_j(u) = ∂I_u/∂ω_j = Σ_e w_e (∂K/∂x'·Jₑ[0,j] + ∂K/∂y'·Jₑ[1,j])
        Since Σ_u D_j = ∂(Σ_u I_u)/∂ω_j = 0 (voting mass is conserved):
            dVar/dω_j = (2/P) Σ_u I_u · D_j(u)
        """
        xw, yw, J = self._warp_and_jacobian(bearings, dt, omega)
        H, W = self.H, self.W
        iwe = np.zeros((H, W))
        D = np.zeros((3, H, W))

        x0 = np.floor(xw).astype(np.int64)
        y0 = np.floor(yw).astype(np.int64)
        dx = xw - x0
        dy = yw - y0
        Jx = J[:, 0, :]      # ∂x'/∂ω  (N,3)
        Jy = J[:, 1, :]      # ∂y'/∂ω  (N,3)

        # 4 bilinear corners: (offset_x, offset_y, K, ∂K/∂x', ∂K/∂y')
        corners = [
            (0, 0, (1 - dx) * (1 - dy), -(1 - dy), -(1 - dx)),
            (1, 0, dx * (1 - dy),        (1 - dy), -dx),
            (0, 1, (1 - dx) * dy,       -dy,        (1 - dx)),
            (1, 1, dx * dy,              dy,         dx),
        ]
        for ox, oy, kw, dkx, dky in corners:
            xi = x0 + ox
            yi = y0 + oy
            m = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            ym, xm = yi[m], xi[m]
            np.add.at(iwe, (ym, xm), (weights * kw)[m])
            # ∂K/∂ω_j = ∂K/∂x'·Jx_j + ∂K/∂y'·Jy_j   (N,3)
            dKdw = dkx[:, None] * Jx + dky[:, None] * Jy
            contrib = (weights[:, None] * dKdw)[m]           # (Nm,3)
            for j in range(3):
                np.add.at(D[j], (ym, xm), contrib[:, j])

        var = float(np.var(iwe))
        P = iwe.size
        grad = np.array([(2.0 / P) * np.sum(iwe * D[j]) for j in range(3)])
        return var, grad

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def estimate(self, events, t_ref=None, omega_init=None):
        """
        Estimate ω (rad/s) for one window of events by maximizing IWE variance.

        Parameters
        ----------
        events : (N, 4) array [t, x, y, pol]  (x, y already undistorted)
        t_ref  : reference time to warp to. Default = window midpoint
                 (Gallego uses the packet midpoint).
        omega_init : (3,) warm-start (rad/s). Default = previous estimate / zeros.

        Returns
        -------
        omega : (3,) rad/s in the camera body frame.
        """
        if len(events) < 10:
            # Too few events to form a meaningful IWE — return the warm start.
            return np.zeros(3) if omega_init is None else np.asarray(omega_init, float)

        t = events[:, 0].astype(np.float64)
        xs = events[:, 1].astype(np.float64)
        ys = events[:, 2].astype(np.float64)
        pol = events[:, 3].astype(np.float64)

        if t_ref is None:
            t_ref = 0.5 * (t.min() + t.max())          # window midpoint
        dt = t - t_ref                                  # (N,) seconds

        bearings = self._bearings(xs, ys)
        weights = (2.0 * pol - 1.0) if self.use_polarity else np.ones_like(pol)

        x0 = np.zeros(3) if omega_init is None else np.asarray(omega_init, float).copy()

        if self.use_analytic_gradient:
            # Gradient-based: minimize -Var with the analytic gradient (CG).
            def neg_val_grad(w):
                v, g = self._contrast_and_grad(w, bearings, dt, weights)
                return -v, -g
            res = minimize(neg_val_grad, x0, method=self.optimizer, jac=True,
                           options=dict(maxiter=200))
        elif self.optimizer == 'Nelder-Mead':
            # Derivative-free on the blurred variance. Initial simplex scaled to
            # a plausible ω range so it works even from a zero warm-start.
            neg = lambda w: -self._contrast(w, bearings, dt, weights)
            span = 0.5
            init_simplex = np.vstack([x0, x0 + span * np.eye(3)])
            res = minimize(neg, x0, method='Nelder-Mead',
                           options=dict(initial_simplex=init_simplex,
                                        xatol=1e-3, fatol=1e-9, maxiter=400))
        else:
            neg = lambda w: -self._contrast(w, bearings, dt, weights)
            res = minimize(neg, x0, method=self.optimizer,
                           options=dict(maxiter=400))

        self.last_result = res
        self.last_iwe = self._build_iwe(bearings, dt, weights, res.x)
        return res.x
