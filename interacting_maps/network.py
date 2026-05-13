"""
InteractingMaps — Cook et al., IJCNN 2011.

Six maps: V (input), I, G, F, C (constant), R.
Three constraints:
    1.  -V  = F · G                  (optical flow constraint, Eq. 1)
    2.   G  = ∇I                     (gradient definition,     Eq. 2)
    3.   F  = m32(R × C)             (rotation → flow,         Eq. 3)

Each update rule takes a small relaxation step (delta) towards satisfying
its constraint, while leaving all other maps unchanged.
"""

import numpy as np
from .camera import compute_calibration, m32, m23


class InteractingMaps:
    def __init__(
        self,
        H: int = 128,
        W: int = 128,
        f: float = 64.0,
        delta_VFG: float = 0.1,
        delta_IG: float = 0.1,
        delta_GI: float = 0.1,
        delta_RF: float = 0.1,
        delta_FR: float = 0.5,
    ):
        self.H = H
        self.W = W
        self.f = f

        # Relaxation step sizes
        self.delta_VFG = delta_VFG  # optical flow constraint (updates F and G)
        self.delta_IG = delta_IG    # G from I (Eq. 6)
        self.delta_GI = delta_GI    # I from G (Eq. 9)
        self.delta_RF = delta_RF    # F from R,C (Eq. 10)
        self.delta_FR = delta_FR    # R from F,C (Eq. 13)

        # Constant calibration map
        self.C = compute_calibration(H, W, f)  # (H, W, 3)

        # Mutable maps — initialised by reset()
        self.I = np.zeros((H + 1, W + 1), dtype=np.float64)
        self.G = np.zeros((H, W, 2), dtype=np.float64)
        self.F = np.zeros((H, W, 2), dtype=np.float64)
        self.R = np.zeros(3, dtype=np.float64)

        # Pre-build the least-squares A matrix for the R update (constant)
        self._build_R_lsq_matrix()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def reset(self, scale: float = 0.01) -> None:
        """Randomly initialise all inferred maps."""
        rng = np.random.default_rng()
        self.I = rng.standard_normal((self.H + 1, self.W + 1)) * scale
        self.G = rng.standard_normal((self.H, self.W, 2)) * scale
        self.F = rng.standard_normal((self.H, self.W, 2)) * scale
        self.R = np.zeros(3, dtype=np.float64)

    # ------------------------------------------------------------------
    # Constraint 1:  -V = F · G  (Eq. 1 / Eq. 5)
    # ------------------------------------------------------------------

    def update_F_from_VG(self, V: np.ndarray) -> None:
        """
        Gradient-descent step on Q_{V,G}(F) = (V + F·G)²  (Eq. 4-5).
        dQ/dF = 2 G (V + F·G)
        """
        e = V + np.einsum('hwk,hwk->hw', self.F, self.G)  # residual (H,W)
        self.F -= self.delta_VFG * 2.0 * self.G * e[..., np.newaxis]

    def update_G_from_VF(self, V: np.ndarray) -> None:
        """
        Analogous gradient-descent step for G.
        dQ/dG = 2 F (V + F·G)
        """
        e = V + np.einsum('hwk,hwk->hw', self.F, self.G)
        self.G -= self.delta_VFG * 2.0 * self.F * e[..., np.newaxis]

    # ------------------------------------------------------------------
    # Constraint 2:  G = ∇I  (Eqs. 2, 6-9)
    # ------------------------------------------------------------------

    def _grad_I(self) -> np.ndarray:
        """Forward-difference gradient of I, size (H, W, 2)."""
        dIx = self.I[:self.H, 1:self.W + 1] - self.I[:self.H, :self.W]
        dIy = self.I[1:self.H + 1, :self.W] - self.I[:self.H, :self.W]
        return np.stack([dIx, dIy], axis=-1)  # (H, W, 2)

    def update_G_from_I(self) -> None:
        """
        Relaxation step towards G = ∇I  (Eq. 6).
        G ← (1-δ)·G + δ·∇I
        """
        self.G = (1.0 - self.delta_IG) * self.G + self.delta_IG * self._grad_I()

    def update_I_from_G(self) -> None:
        """
        Update I so that ∇I is closer to G  (Eqs. 7-9).

        Ψ      = G - ∇I                              (residual)
        Ψ̂_x[v,u] = Ψ_x[v,u] - Ψ_x[v,u-1]           (boundary = 0)
        Ψ̂_y[v,u] = Ψ_y[v,u] - Ψ_y[v-1,u]
        I[v,u] ← I[v,u] - δ·(Ψ̂_x[v,u] + Ψ̂_y[v,u])
        """
        Psi = self.G - self._grad_I()  # (H, W, 2)

        Psi_x = Psi[..., 0]  # (H, W)
        Psi_y = Psi[..., 1]

        # Ψ̂_x[v, u] = Ψ_x[v, u] − Ψ_x[v, u-1]   (Eq. 8, out-of-bounds = 0)
        Psi_hat_x = np.zeros((self.H, self.W), dtype=np.float64)
        Psi_hat_x[:, 0] = Psi_x[:, 0]
        Psi_hat_x[:, 1:] = Psi_x[:, 1:] - Psi_x[:, :-1]

        Psi_hat_y = np.zeros((self.H, self.W), dtype=np.float64)
        Psi_hat_y[0, :] = Psi_y[0, :]
        Psi_hat_y[1:, :] = Psi_y[1:, :] - Psi_y[:-1, :]

        self.I[:self.H, :self.W] = (
            (1.0 - self.delta_GI) * self.I[:self.H, :self.W]
            + self.delta_GI * (
                self.I[:self.H, :self.W] - Psi_hat_x - Psi_hat_y
            )
        )

    # ------------------------------------------------------------------
    # Constraint 3:  F = m32(R × C)  (Eqs. 3, 10-13)
    # ------------------------------------------------------------------

    def update_F_from_RC(self) -> None:
        """
        Relaxation step: F ← (1-δ)·F + δ·m32(R×C)  (Eq. 10).
        """
        R_bc = np.broadcast_to(self.R, (self.H, self.W, 3))  # (H,W,3)
        RxC = np.cross(R_bc, self.C)                          # (H,W,3)
        F_candidate = m32(RxC, self.C, self.f)                # (H,W,2)
        self.F = (1.0 - self.delta_RF) * self.F + self.delta_RF * F_candidate

    def _build_R_lsq_matrix(self) -> None:
        """
        Pre-build the constant part of the linear system for the R update.

        R × C = F3d  is linear in R.  Writing it out component-wise:
            (R × C)_x = Ry*Cz - Rz*Cy = [0,  Cz, -Cy] · R = F3d_x
            (R × C)_y = Rz*Cx - Rx*Cz = [-Cz, 0,  Cx] · R = F3d_y

        A_lsq has shape (2·H·W, 3) and is constant (depends only on C).
        """
        C_flat = self.C.reshape(-1, 3)            # (N, 3)
        cx, cy, cz = C_flat[:, 0], C_flat[:, 1], C_flat[:, 2]
        N = C_flat.shape[0]
        zeros = np.zeros(N, dtype=np.float64)

        # Row for x-component: [0, Cz, -Cy]
        A_row_x = np.stack([zeros, cz, -cy], axis=-1)    # (N, 3)
        # Row for y-component: [-Cz, 0, Cx]
        A_row_y = np.stack([-cz, zeros, cx], axis=-1)    # (N, 3)

        self._A_lsq = np.vstack([A_row_x, A_row_y])      # (2N, 3)
        self._N_pixels = N

    def update_R_from_FC(self) -> None:
        """
        Least-squares estimate of R from all optical-flow vectors  (Eq. 13).

        R × C = F3d  →  A_lsq · R = b
        R ← (1-δ)·R + δ·R_new
        """
        F3d = m23(self.F, self.C, self.f)               # (H,W,3)
        F3d_flat = F3d.reshape(-1, 3)                   # (N, 3)

        # Right-hand side: stacked x then y components of F3d
        b = np.concatenate([F3d_flat[:, 0], F3d_flat[:, 1]])  # (2N,)

        R_new, _, _, _ = np.linalg.lstsq(self._A_lsq, b, rcond=None)
        self.R = (1.0 - self.delta_FR) * self.R + self.delta_FR * R_new

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def step(self, V: np.ndarray, n_iters: int = 20) -> None:
        """
        Process one input frame V by running n_iters relaxation cycles.

        Parameters
        ----------
        V       : (H, W) temporal intensity derivative (the sole input)
        n_iters : number of update cycles per frame
        """
        for _ in range(n_iters):
            self.update_F_from_VG(V)
            self.update_G_from_VF(V)
            self.update_G_from_I()
            self.update_I_from_G()
            self.update_F_from_RC()
            self.update_R_from_FC()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def residual_VFG(self, V: np.ndarray) -> float:
        """Mean absolute residual of the optical flow constraint -V = F·G."""
        return float(
            np.mean(np.abs(V + np.einsum('hwk,hwk->hw', self.F, self.G)))
        )

    def residual_GI(self) -> float:
        """Mean absolute residual of the gradient constraint G = ∇I."""
        return float(np.mean(np.abs(self.G - self._grad_I())))
