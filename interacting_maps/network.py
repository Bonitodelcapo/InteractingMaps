"""
InteractingMaps — Cook et al., IJCNN 2011.

Six maps: V (input), I, G, F, C (constant), R.
Three constraints:
    1.  V + F · G = 0               (optical flow constraint, Eq. 1)
    2.   G  = ∇I                     (gradient definition,     Eq. 2)
    3.   F  = C_mat @ R              (rotation → flow,         Eq. 3)

Each update rule takes a small relaxation step (delta) towards satisfying
its constraint, while leaving all other maps unchanged.

Update strategy: Sequential (Gauss-Seidel) — each map is updated immediately,
so subsequent updates see the latest values. This is the key difference from
the thesis version (network_dissertation.py) which uses simultaneous updates.
"""

import numpy as np
from .camera import compute_calibration, build_kinematic_matrix


class InteractingMaps:
    def __init__(
        self,
        H: int,
        W: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        delta_VFG: float = 0.1,
        delta_IG: float = 0.1,
        delta_GI: float = 0.1,
        delta_RF: float = 0.1,
        delta_FR: float = 0.5,
        dist_coeffs: np.ndarray | None = None,
    ):
        self.H = H
        self.W = W
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        # Relaxation step sizes
        self.delta_VFG = delta_VFG  # optical flow constraint (updates F and G)
        self.delta_IG = delta_IG    # G from I (Eq. 6)
        self.delta_GI = delta_GI    # I from G (Eq. 9)
        self.delta_RF = delta_RF    # F from R,C (Eq. 10)
        self.delta_FR = delta_FR    # R from F,C (Eq. 13)

        # Constant calibration map (unit direction per pixel)
        self.C = compute_calibration(H, W, fx, fy, cx, cy)  # (H, W, 3)

        # Precompute the perspective-correct kinematic matrix (Thesis Eq. 6.37).
        # When dist_coeffs is set, C_mat is built in distorted pixel space.
        self._C_mat = build_kinematic_matrix(H, W, fx, fy, cx, cy,
                                             dist_coeffs=dist_coeffs)  # (H, W, 2, 3)

        # Mutable maps — initialised by reset()
        self.I = np.zeros((H + 1, W + 1), dtype=np.float64)
        self.G = np.zeros((H, W, 2), dtype=np.float64)
        self.F = np.zeros((H, W, 2), dtype=np.float64)
        self.R = np.zeros(3, dtype=np.float64)

        # Pre-build the normal equations matrix for the R least-squares update
        self._build_R_normal_equations()

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
    # Constraint 1:  V + F · G = 0  (Eq. 1 / Eq. 5)
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
        """
        Forward-difference gradient of I, size (H, W, 2).

        Convention:
            G[..., 0] = dI/dx  (horizontal, along columns)
            G[..., 1] = dI/dy  (vertical, along rows)

        I is (H+1, W+1) so forward differences yield a full (H, W) map.
        """
        dIx = self.I[:self.H, 1:self.W + 1] - self.I[:self.H, :self.W]  # horizontal
        dIy = self.I[1:self.H + 1, :self.W] - self.I[:self.H, :self.W]  # vertical
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

        Psi_x = Psi[..., 0]  # (H, W) — horizontal component
        Psi_y = Psi[..., 1]  # (H, W) — vertical component

        # Ψ̂_x[v, u] = Ψ_x[v, u] − Ψ_x[v, u-1]   (Eq. 8, boundary = 0)
        Psi_hat_x = np.zeros((self.H, self.W), dtype=np.float64)
        Psi_hat_x[:, 0] = Psi_x[:, 0]
        Psi_hat_x[:, 1:] = Psi_x[:, 1:] - Psi_x[:, :-1]

        # Ψ̂_y[v, u] = Ψ_y[v, u] − Ψ_y[v-1, u]
        Psi_hat_y = np.zeros((self.H, self.W), dtype=np.float64)
        Psi_hat_y[0, :] = Psi_y[0, :]
        Psi_hat_y[1:, :] = Psi_y[1:, :] - Psi_y[:-1, :]

        # I[v,u] ← I[v,u] - δ·(Ψ̂_x + Ψ̂_y)
        self.I[:self.H, :self.W] -= self.delta_GI * (Psi_hat_x + Psi_hat_y)

    # ------------------------------------------------------------------
    # Constraint 3:  F = C_mat @ R  (Eqs. 3, 10-13)
    # ------------------------------------------------------------------

    def update_F_from_RC(self) -> None:
        """
        Relaxation step: F ← (1-δ)·F + δ·(C_mat @ R)  (Eq. 10).

        Uses the perspective-correct kinematic matrix instead of m32(R×C).
        """
        F_candidate = np.einsum('hwij,j->hwi', self._C_mat, self.R)  # (H,W,2)
        self.F = (1.0 - self.delta_RF) * self.F + self.delta_RF * F_candidate

    def _build_R_normal_equations(self) -> None:
        """
        Pre-build the constant (C_mat^T @ C_mat) matrix for the R update.

        The least-squares problem is:
            argmin_R  Σ_{x,y} || F[x,y] - C_mat[x,y] @ R ||²

        Normal equations: (Σ C_mat^T C_mat) R = Σ C_mat^T F

        The left-hand side matrix M = Σ C_mat^T @ C_mat is constant (3x3).
        """
        # C_mat is (H, W, 2, 3)
        # M = Σ C_mat[h,w]^T @ C_mat[h,w]  → sum of (3,2)@(2,3) = (3,3)
        self._M_normal = np.einsum('hwji,hwjk->ik', self._C_mat, self._C_mat)  # (3, 3)
        self._M_inv = np.linalg.inv(self._M_normal)  # (3, 3)

    def update_R_from_FC(self) -> None:
        """
        Least-squares estimate of R from all optical-flow vectors (Eq. 13).

        Normal equations:  M @ R_new = Σ C_mat^T @ F
        R ← (1-δ)·R + δ·R_new
        """
        # Right-hand side: v = Σ C_mat^T @ F  → (3,)
        # C_mat is (H,W,2,3):  subscript 'hwji' → j=flow(2), i=rot(3)
        # F is (H,W,2):        subscript 'hwj'  → j=flow(2)
        # Contract over h,w,j; output i → (3,)
        v = np.einsum('hwji,hwj->i', self._C_mat, self.F)  # ← FIX: 'hwi->j' → 'hwj->i'

        # Solve: R_new = M^{-1} @ v
        R_new = self._M_inv @ v

        # Blend toward new estimate
        self.R = (1.0 - self.delta_FR) * self.R + self.delta_FR * R_new

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def step(self, V: np.ndarray, n_iters: int = 20) -> None:
        """
        Process one input frame V by running n_iters relaxation cycles.

        Uses sequential (Gauss-Seidel) updates: each map update immediately
        sees the latest values from previous updates in the same iteration.

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
        """Mean absolute residual of the optical flow constraint V + F·G = 0."""
        return float(
            np.mean(np.abs(V + np.einsum('hwk,hwk->hw', self.F, self.G)))
        )

    def residual_GI(self) -> float:
        """Mean absolute residual of the gradient constraint G = ∇I."""
        return float(np.mean(np.abs(self.G - self._grad_I())))