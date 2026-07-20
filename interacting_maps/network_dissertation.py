"""
InteractingMaps — Energy-Based Message Passing, Martel 2019 Thesis (Chapter 6).

Faithfully implements Algorithm 6.5 with:
- BLEND updates for linear relations (Eq. 6.140-6.141):
    q ← (1-η)q + η·target  =  q - η·(q - target)
- GRADIENT updates for nonlinear relations (Eq. 6.12-6.14):
    q ← q - η·∂C/∂q

Phase 1: All costs compute "gradients" (messages) from current state.
Phase 2: All quantities update simultaneously.

Key implementation detail:
    The simultaneous (Jacobi) update requires quantities to start in a
    MUTUALLY CONSISTENT state. If R is known, F must be initialized as
    F = C·R (not zero), otherwise the kinematic cost crushes R to zero
    before OFCE can develop structure.
"""

import numpy as np
from .camera import compute_calibration, build_kinematic_matrix

# ---------------------------------------------------------------------------
# 1. THE CORE ARCHITECTURE
# ---------------------------------------------------------------------------

class Quantity:
    """Represents a Map (Intensity, Gradient, Flow, Rotation) in the network."""
    def __init__(self, shape, name):
        self.name = name
        self.shape = shape
        self.value = np.zeros(shape, dtype=np.float64)
        self.gradient_accumulator = np.zeros(shape, dtype=np.float64)

    def reset_gradient(self):
        self.gradient_accumulator.fill(0)

    def add_gradient(self, grad):
        self.gradient_accumulator += grad

    def update(self, learning_rate):
        self.value -= learning_rate * self.gradient_accumulator


class Cost:
    """Base class for relations between quantities."""
    def __init__(self, quantities_dict):
        self.q = quantities_dict

    def compute_and_send_gradients(self):
        raise NotImplementedError()


# ---------------------------------------------------------------------------
# 2. THE COSTS (Thesis Table 6.1, Section 6.4.2)
# ---------------------------------------------------------------------------

class Cost_OFCE(Cost):
    """
    Optical Flow Constraint: V + F·G = 0  (Thesis Eq. 6.54-6.55, Table 6.1)

    Cost C₂ = Σ (V + F·G)²

    NONLINEAR relation → gradient descent updates:
        ∂C₂/∂F = 2(V+F·G)·G
        ∂C₂/∂G = 2(V+F·G)·F

    Per-pixel gradient clipping prevents the cubic instability that arises
    when |F| and |G| are both large.
    """
    def __init__(self, quantities_dict, delta_VFG, max_grad=5.0):
        super().__init__(quantities_dict)
        self.delta_VFG = delta_VFG
        self.max_grad = max_grad

    def compute_and_send_gradients(self):
        v = self.q['V'].value   # (H, W)
        f = self.q['F'].value   # (H, W, 2)
        g = self.q['G'].value   # (H, W, 2)

        # Constraint residual (scalar per pixel)
        error = v + np.sum(f * g, axis=-1)  # (H, W)

        # Raw gradients from Table 6.1
        grad_F = 2.0 * error[..., np.newaxis] * g  # (H, W, 2)
        grad_G = 2.0 * error[..., np.newaxis] * f  # (H, W, 2)

        # Per-pixel clip to bound cubic growth
        grad_F = np.clip(grad_F, -self.max_grad, self.max_grad)
        grad_G = np.clip(grad_G, -self.max_grad, self.max_grad)

        self.q['F'].add_gradient(grad_F * self.delta_VFG)
        self.q['G'].add_gradient(grad_G * self.delta_VFG)


class Cost_Spatial(Cost):
    """
    Spatial Gradient Relation: G = ∇I  (Thesis Eq. 6.56-6.65)

    LINEAR relation → BLEND updates (Eq. 6.141):
        G: blend toward ∇I
        I: iterative PDE step (Eq. 6.61)

    Convention:
        G[..., 0] = dI/dx  (horizontal, along columns)
        G[..., 1] = dI/dy  (vertical, along rows)
    """
    def __init__(self, quantities_dict, delta_IG, delta_GI):
        super().__init__(quantities_dict)
        self.delta_IG = delta_IG
        self.delta_GI = delta_GI

    def compute_and_send_gradients(self):
        i_map = self.q['I'].value   # (H, W)
        g = self.q['G'].value       # (H, W, 2)

        # Compute ∇I using forward differences
        grad_I_x = np.zeros_like(i_map)
        grad_I_x[:, :-1] = i_map[:, 1:] - i_map[:, :-1]

        grad_I_y = np.zeros_like(i_map)
        grad_I_y[:-1, :] = i_map[1:, :] - i_map[:-1, :]

        grad_I_stack = np.stack([grad_I_x, grad_I_y], axis=-1)
        error = g - grad_I_stack  # (H, W, 2)

        # G: blend toward ∇I (Eq. 6.57)
        self.q['G'].add_gradient(error * self.delta_IG)

        # I: negative divergence (Eq. 6.61)
        grad_I_update = np.zeros_like(i_map)
        grad_I_update += error[:, :, 0] + error[:, :, 1]
        grad_I_update[:, 1:] -= error[:, :-1, 0]
        grad_I_update[1:, :] -= error[:-1, :, 1]

        self.q['I'].add_gradient(grad_I_update * self.delta_GI)


class Cost_Kinematics(Cost):
    """
    Camera Kinematics: F = C·Ω  (Thesis Eq. 6.36-6.50)

    LINEAR relation → BLEND updates:
        F: blend toward C·Ω (Eq. 6.40)
        Ω: blend toward Ω* = M⁻¹v (Eq. 6.50)

    The Ω update uses the thesis's RECOMMENDED approach (Eq. 6.50, footnote 18):
    precompute M⁻¹ and blend toward the closed-form optimal Ω*.
    """
    def __init__(self, quantities_dict, delta_RF, delta_FR, C_mat):
        super().__init__(quantities_dict)
        self.delta_RF = delta_RF
        self.delta_FR = delta_FR
        self.C_mat = C_mat  # (H, W, 2, 3)

        # Precompute M = Σ C^T·C and M⁻¹ (Eq. 6.48, footnote 18)
        self._M = np.einsum('hwji,hwjk->ik', C_mat, C_mat)  # (3, 3)
        self._M_inv = np.linalg.inv(self._M)  # (3, 3)

    def compute_and_send_gradients(self):
        f = self.q['F'].value   # (H, W, 2)
        r = self.q['R'].value   # (3,)

        # Target flow from current R
        f_target = np.einsum('hwij,j->hwi', self.C_mat, r)  # (H, W, 2)

        # F: blend toward C·Ω (Eq. 6.40)
        error_F = f - f_target
        self.q['F'].add_gradient(error_F * self.delta_RF)

        # Ω: blend toward Ω* = M⁻¹·v (Eq. 6.49-6.50)
        v = np.einsum('hwji,hwj->i', self.C_mat, f)  # (3,)
        R_target = self._M_inv @ v  # (3,)
        error_R = r - R_target
        self.q['R'].add_gradient(error_R * self.delta_FR)

class Cost_IMU(Cost):
    """
    IMU Soft Constraint (Thesis Section 6.8.3).
    
    Gently pulls R toward the IMU gyroscope reading each frame.
    This provides the "tracking signal" that pure OFCE at short dt
    cannot supply — the visual signal is too weak to detect 0.05 pixel
    flow changes between frames.
    
    With delta_IMU=0.3, the IMU provides the coarse tracking while
    OFCE refines the estimate (correcting IMU drift/bias).
    """
    def __init__(self, quantities_dict, delta_IMU):
        super().__init__(quantities_dict)
        self.delta_IMU = delta_IMU
        self.R_imu = np.zeros(3)  # Set externally each frame (in rad/frame units)

    def compute_and_send_gradients(self):
        r = self.q['R'].value
        error_R = r - self.R_imu
        self.q['R'].add_gradient(error_R * self.delta_IMU)

# ---------------------------------------------------------------------------
# 3. THE API WRAPPER
# ---------------------------------------------------------------------------

class InteractingMapsThesis:
    """
    Energy-Based Graphical Model for 3-DoF rotation estimation.
    Implements Algorithm 6.5 (two-phase simultaneous message passing).
    """
    def __init__(self, H, W, fx, fy, cx, cy, frame_duration=0.005,
                 delta_VFG=0.15, delta_IG=0.10, delta_GI=0.05,
                 delta_RF=0.03, delta_FR=0.50, delta_IMU=0.3,
                 dist_coeffs=None, include_jacobian=True):

        self.H = H
        self.W = W
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy

        self.frame_duration = frame_duration


        # Build kinematic matrix from real intrinsics (Eq. 6.38)
        self._C_mat = build_kinematic_matrix(H, W, fx, fy, cx, cy,
                                             dist_coeffs=dist_coeffs,
                                             include_jacobian=include_jacobian)

        # Initialize Quantities
        self.q_V = Quantity((H, W), "Input_V")
        self.q_I = Quantity((H, W), "Intensity")
        self.q_G = Quantity((H, W, 2), "Spatial_Gradient")
        self.q_F = Quantity((H, W, 2), "Optic_Flow")
        self.q_R = Quantity((3,), "Rotation")

        # Initialize Costs (Table 6.1)
        q_dict = {
            'V': self.q_V, 'I': self.q_I, 'G': self.q_G,
            'F': self.q_F, 'R': self.q_R,
        }
        self.costs = [
            Cost_OFCE(q_dict, delta_VFG, max_grad=5.0),
            Cost_Spatial(q_dict, delta_IG, delta_GI),
            Cost_Kinematics(q_dict, delta_RF, delta_FR, self._C_mat),
        ]
        self.cost_imu = Cost_IMU(q_dict, delta_IMU)
        self.costs.append(self.cost_imu)

    # Properties for demo.py
    @property
    def I(self): return self.q_I.value
    @property
    def G(self): return self.q_G.value
    @property
    def F(self): return self.q_F.value
    @property
    def R(self): return self.q_R.value

    def initialize_from_rotation(self, R_init: np.ndarray) -> None:
        """
        Set R and initialize F = C·R for a mutually consistent starting state.

        This is ESSENTIAL for the Jacobi (simultaneous) update scheme:
        without it, the kinematic cost sees F=0 and crushes R to zero
        before OFCE can develop any structure.

        Also initializes I with small noise so that ∇I provides initial
        gradient structure for G to bootstrap from.
        """
        # Set rotation
        self.q_R.value = R_init.copy()

        # Set F = C·R (mutually consistent with R)
        self.q_F.value = np.einsum('hwij,j->hwi', self._C_mat, R_init)

        # Small random noise for I (provides initial ∇I for G bootstrap)
        rng = np.random.default_rng(42)
        self.q_I.value = rng.standard_normal((self.H, self.W)) * 0.01

        # G and other quantities start at zero — they'll develop from OFCE

    def reset(self, scale=0.01):
        """Randomly initialise all inferred maps (without R/F consistency)."""
        rng = np.random.default_rng()
        self.q_I.value = rng.standard_normal((self.H, self.W)) * scale
        self.q_G.value = rng.standard_normal((self.H, self.W, 2)) * scale
        self.q_F.value = rng.standard_normal((self.H, self.W, 2)) * scale
        self.q_R.value = np.zeros(3, dtype=np.float64)

    def step(self, V: np.ndarray, n_iters: int = 50, omega_imu: np.ndarray = None):
        """
        Two-Phase Message Passing (Algorithm 6.5) with inter-frame flow decay.
        Thesis uses 50-75 iterations per time-slice (Section 6.8).

        The decay breaks the kinematic-flow feedback lock that
        prevents the network from tracking time-varying rotations.
        
        """
        self.q_V.value = V

        # Set IMU target
        if omega_imu is not None:
            self.cost_imu.R_imu = omega_imu * self.frame_duration
            self._use_imu = True
        else:
            self._use_imu = False
            
        for _ in range(n_iters):
            # PHASE 1: All costs compute gradients
            for q in [self.q_I, self.q_G, self.q_F, self.q_R]:
                q.reset_gradient()
            for cost in self.costs:
                if cost is self.cost_imu and not self._use_imu:
                    continue  # ← SKIP IMU cost entirely when no IMU
                cost.compute_and_send_gradients()

            # PHASE 2: All quantities update simultaneously
            self.q_I.update(1.0)
            self.q_G.update(1.0)
            self.q_F.update(1.0)
            self.q_R.update(1.0)

            # Stability clipping
            self.q_I.value = np.clip(self.q_I.value, -10.0, 10.0)
            self.q_G.value = np.clip(self.q_G.value, -5.0, 5.0)
            self.q_F.value = np.clip(self.q_F.value, -10.0, 10.0)
            self.q_R.value = np.clip(self.q_R.value, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def residual_VFG(self, V: np.ndarray) -> float:
        return float(np.mean(np.abs(
            V + np.einsum('hwk,hwk->hw', self.F, self.G)
        )))

    def residual_GI(self) -> float:
        dIx = np.zeros_like(self.I)
        dIy = np.zeros_like(self.I)
        dIx[:, :-1] = self.I[:, 1:] - self.I[:, :-1]
        dIy[:-1, :] = self.I[1:, :] - self.I[:-1, :]
        grad_I = np.stack([dIx, dIy], axis=-1)
        return float(np.mean(np.abs(self.G - grad_I)))