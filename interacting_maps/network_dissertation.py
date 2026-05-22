"""
InteractingMaps — Message Passing (Energy-Based) Version based on Martel's 2019 Thesis.
Phase 1: Compute all gradients (messages) based on the current state.
Phase 2: Simultaneously update all maps.
"""

import numpy as np
from .camera import compute_calibration

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
        """Phase 1 prep: Clear out old messages."""
        self.gradient_accumulator.fill(0)
        
    def add_gradient(self, grad):
        """Phase 1 action: Costs push their errors here."""
        self.gradient_accumulator += grad
        
    def update(self, learning_rate):
        """Phase 2 action: Step in the opposite direction of the gradient."""
        self.value -= learning_rate * self.gradient_accumulator


class Cost:
    """Base class for mathematical relations."""
    def __init__(self, quantities_dict):
        self.q = quantities_dict
        
    def compute_and_send_gradients(self):
        raise NotImplementedError()


# ---------------------------------------------------------------------------
# 2. THE MATHEMATICAL RELATIONS (COSTS)
# ---------------------------------------------------------------------------
class Cost_OFCE(Cost):
    """Optical Flow Constraint: V + F * G = 0"""
    def __init__(self, quantities_dict, delta_VFG):
        super().__init__(quantities_dict)
        self.delta_VFG = delta_VFG

    def compute_and_send_gradients(self):
        v = self.q['V'].value
        f = self.q['F'].value
        g = self.q['G'].value

        # THE MISSING THESIS DETAIL: Gradient Normalization
        # To prevent the cubic NaN explosion, we divide the error 
        # by the squared magnitude of the interacting maps.
        error = v + np.sum(f * g, axis=-1) 
        f_mag_sq = np.sum(f**2, axis=-1)
        g_mag_sq = np.sum(g**2, axis=-1)
        
        # Multiply the gradient by delta_VFG before sending!
        grad_F = (error / (1.0 + g_mag_sq))[..., np.newaxis] * g  
        self.q['F'].add_gradient(grad_F * self.delta_VFG)
        
        grad_G = (error / (1.0 + f_mag_sq))[..., np.newaxis] * f  
        self.q['G'].add_gradient(grad_G * self.delta_VFG)


class Cost_Spatial(Cost):
    """Spatial Gradient constraint: G = Gradient(I)"""
    def __init__(self, quantities_dict, delta_IG, delta_GI):
        super().__init__(quantities_dict)
        self.delta_IG = delta_IG
        self.delta_GI = delta_GI

    def compute_and_send_gradients(self):
        i = self.q['I'].value
        g = self.q['G'].value
        
        # Compute math gradient of I using Forward Difference
        grad_I_x = np.zeros_like(i)
        grad_I_y = np.zeros_like(i)
        grad_I_x[:-1, :] = i[1:, :] - i[:-1, :] 
        grad_I_y[:, :-1] = i[:, 1:] - i[:, :-1] 
        grad_I_stack = np.stack([grad_I_x, grad_I_y], axis=-1)
        
        error = g - grad_I_stack
        
        # Multiply by respective deltas
        self.q['G'].add_gradient(error * self.delta_IG)
        
        # Gradient w.r.t I (Negative divergence)
        grad_I_update = np.zeros_like(i)
        grad_I_update[1:, :] -= error[:-1, :, 0]
        grad_I_update[:, 1:] -= error[:, :-1, 1]
        grad_I_update += error[:, :, 0] + error[:, :, 1]
        
        self.q['I'].add_gradient(grad_I_update * self.delta_GI)


class Cost_Kinematics(Cost):
    """Camera Kinematics constraint: F = R x C"""
    def __init__(self, quantities_dict, delta_RF, delta_FR):
        super().__init__(quantities_dict)
        self.delta_RF = delta_RF
        self.delta_FR = delta_FR

    def compute_and_send_gradients(self):
        f = self.q['F'].value
        r = self.q['R'].value
        c = self.q['C'].value

        # Target Flow (Cross product projected to 2D)
        r_cross_c = np.cross(r, c) 
        f_target = r_cross_c[..., :2] 
        error = f - f_target
        
        # F is influenced by Kinematics according to delta_RF!
        self.q['F'].add_gradient(error * self.delta_RF)
        
        # Gradient w.r.t R (Global Rotation requires summing over all pixels)
        error_3d = np.pad(error, ((0,0), (0,0), (0,1))) 
        grad_r_map = np.cross(error_3d, c) 
        grad_r_global = np.mean(grad_r_map, axis=(0, 1))
        
        self.q['R'].add_gradient(grad_r_global * self.delta_FR)

# ---------------------------------------------------------------------------
# 3. THE API WRAPPER (To match your colleague's demo.py)
# ---------------------------------------------------------------------------

class InteractingMapsThesis:
    def __init__(self, H=128, W=128, f=64.0, 
                 delta_VFG=0.08, delta_IG=0.12, delta_GI=0.08, delta_RF=0.10, delta_FR=0.50):
        self.H = H
        self.W = W
        self.f = f
        
        # Initialize Quantities
        self.q_V = Quantity((H, W), "Input_V")
        self.q_I = Quantity((H, W), "Intensity")
        self.q_G = Quantity((H, W, 2), "Spatial_Gradient")
        self.q_F = Quantity((H, W, 2), "Optic_Flow")
        self.q_R = Quantity((3,), "Rotation")
        self.q_C = Quantity((H, W, 3), "Camera")
        
        self.q_C.value = compute_calibration(H, W, f)
        
        # Initialize Costs WITH THEIR SPECIFIC DELTAS
        q_dict = {'V': self.q_V, 'I': self.q_I, 'G': self.q_G, 'F': self.q_F, 'R': self.q_R, 'C': self.q_C}
        self.costs = [
            Cost_OFCE(q_dict, delta_VFG), 
            Cost_Spatial(q_dict, delta_IG, delta_GI), 
            Cost_Kinematics(q_dict, delta_RF, delta_FR)
        ]

    # Provide Properties to match demo.py visualization API
    @property
    def I(self): return self.q_I.value
    @property
    def G(self): return self.q_G.value
    @property
    def F(self): return self.q_F.value
    @property
    def R(self): return self.q_R.value

    def reset(self, scale=0.01):
        """Randomly initialise all inferred maps."""
        rng = np.random.default_rng()
        self.q_I.value = rng.standard_normal((self.H, self.W)) * scale
        self.q_G.value = rng.standard_normal((self.H, self.W, 2)) * scale
        self.q_F.value = rng.standard_normal((self.H, self.W, 2)) * scale
        self.q_R.value = np.zeros(3, dtype=np.float64)

    def step(self, V: np.ndarray, n_iters: int = 20):
        """The Two-Phase Message Passing Loop."""
        self.q_V.value = V
        
        for _ in range(n_iters):
            # PHASE 1: Reset accumulators & compute all gradients
            for q in [self.q_I, self.q_G, self.q_F, self.q_R]:
                q.reset_gradient()
                
            for cost in self.costs:
                cost.compute_and_send_gradients()
                
            # PHASE 2: Apply updates simultaneously
            # The deltas are already baked into the gradient accumulators!
            # So we just tell the maps to apply the accumulated changes (learning rate = 1.0).
            self.q_I.update(1.0)
            self.q_G.update(1.0)
            self.q_F.update(1.0)
            self.q_R.update(1.0)
            
            # Stabilization (Clipping)
            self.q_F.value = np.clip(self.q_F.value, -5.0, 5.0)
            self.q_G.value = np.clip(self.q_G.value, -5.0, 5.0)
            self.q_R.value = np.clip(self.q_R.value, -0.5, 0.5)
            self.q_I.value = np.clip(self.q_I.value, -5.0, 5.0) # Added safety clip for Image!

    # Diagnostics to match demo.py graph
    def residual_VFG(self, V: np.ndarray) -> float:
        return float(np.mean(np.abs(V + np.einsum('hwk,hwk->hw', self.F, self.G))))

    def residual_GI(self) -> float:
        # Match colleague's forward-difference check
        dIx = np.zeros_like(self.I)
        dIy = np.zeros_like(self.I)
        dIx[:-1, :] = self.I[1:, :] - self.I[:-1, :]
        dIy[:, :-1] = self.I[:, 1:] - self.I[:, :-1]
        grad_I = np.stack([dIx, dIy], axis=-1)
        return float(np.mean(np.abs(self.G - grad_I)))