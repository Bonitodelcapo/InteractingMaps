"""
Synthetic event (DVS) simulation utilities.

Provides:
  - make_synthetic_image : create a test image (checkerboard / gradient / noise)
  - rotation_flow        : ground-truth optical flow from camera rotation R
  - compute_V            : temporal intensity derivative V = −F · ∇I
  - DVSSimulator         : wraps a sequence of warped frames into V frames
"""

import numpy as np
from scipy.ndimage import map_coordinates

from interacting_maps.camera import compute_calibration, m32


# ---------------------------------------------------------------------------
# Synthetic image generation
# ---------------------------------------------------------------------------

def make_synthetic_image(
    H: int = 128,
    W: int = 128,
    kind: str = 'checkerboard',
    tile: int = 16,
) -> np.ndarray:
    """
    Create a float64 image in [0, 1].

    Parameters
    ----------
    kind : 'checkerboard' | 'gradient' | 'random'
    tile : checkerboard tile size in pixels
    """
    if kind == 'checkerboard':
        u = np.arange(W)
        v = np.arange(H)
        uu, vv = np.meshgrid(u, v)
        img = ((uu // tile + vv // tile) % 2).astype(np.float64)
    elif kind == 'gradient':
        u = np.linspace(0.0, 1.0, W)
        v = np.linspace(0.0, 1.0, H)
        uu, vv = np.meshgrid(u, v)
        img = 0.5 * uu + 0.5 * vv
    elif kind == 'random':
        from scipy.ndimage import gaussian_filter
        img = gaussian_filter(np.random.default_rng(0).random((H, W)), sigma=4)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    else:
        raise ValueError(f"Unknown kind '{kind}'")
    return img


# ---------------------------------------------------------------------------
# Ground-truth optical flow from camera rotation
# ---------------------------------------------------------------------------

def rotation_flow(
    R_vec: np.ndarray,
    C: np.ndarray,
    f: float,
) -> np.ndarray:
    """
    Compute the ground-truth optical flow (pixels/frame) produced by
    camera rotation R_vec (3-D angular velocity vector, rad/frame).

    F_{x,y} = m32(R × C_{x,y})

    Parameters
    ----------
    R_vec : (3,)
    C     : (H, W, 3)  calibration map
    f     : focal length

    Returns
    -------
    F : (H, W, 2)
    """
    H, W = C.shape[:2]
    R_bc = np.broadcast_to(R_vec, (H, W, 3))
    RxC = np.cross(R_bc, C)       # (H, W, 3)
    return m32(RxC, C, f)         # (H, W, 2)


# ---------------------------------------------------------------------------
# Temporal intensity derivative from optical flow
# ---------------------------------------------------------------------------

def compute_V(
    image: np.ndarray,
    flow: np.ndarray,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Compute the temporal intensity derivative V = −F · ∇I.

    Uses the brightness constancy assumption (optical flow constraint):
        dI/dt = −F · ∇I

    Parameters
    ----------
    image     : (H, W) float image
    flow      : (H, W, 2) optical flow (pixels/frame)
    noise_std : std-dev of additive Gaussian noise (0 = no noise)
    rng       : optional numpy Generator for reproducibility

    Returns
    -------
    V : (H, W)
    """
    H, W = image.shape

    # Forward finite-difference gradient — matches the network's G = ∇I (Eq. 2)
    # G_x[v, u] = I[v, u+1] - I[v, u]   (clamped at right boundary)
    # G_y[v, u] = I[v+1, u] - I[v, u]   (clamped at bottom boundary)
    grad_x = np.zeros((H, W), dtype=np.float64)
    grad_y = np.zeros((H, W), dtype=np.float64)

    grad_x[:, :-1] = image[:, 1:] - image[:, :-1]
    grad_y[:-1, :] = image[1:, :] - image[:-1, :]

    V = -(flow[..., 0] * grad_x + flow[..., 1] * grad_y)

    if noise_std > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        V = V + rng.standard_normal(V.shape) * noise_std

    return V


# ---------------------------------------------------------------------------
# DVSSimulator — sequences of frames
# ---------------------------------------------------------------------------

class DVSSimulator:
    """
    Simulate a sequence of DVS-like V frames by warping a synthetic image
    with a given (possibly varying) camera rotation.

    Usage
    -----
    sim = DVSSimulator(H=128, W=128, f=64.0, image_kind='checkerboard')
    for t in range(n_frames):
        R_t = np.array([0.01, 0.02, 0.0]) * (t + 1)
        V   = sim.next_frame(R_t)
    """

    def __init__(
        self,
        H: int = 128,
        W: int = 128,
        f: float = 64.0,
        image_kind: str = 'checkerboard',
        noise_std: float = 0.005,
        rng_seed: int = 42,
    ):
        self.H = H
        self.W = W
        self.f = f
        self.noise_std = noise_std
        self.rng = np.random.default_rng(rng_seed)

        self.C = compute_calibration(H, W, f)
        self.image = make_synthetic_image(H, W, kind=image_kind)

    def frame_from_rotation(self, R_vec: np.ndarray) -> np.ndarray:
        """
        Return V for a single rotation R_vec (rad/frame) applied to the
        stored reference image.
        """
        flow = rotation_flow(R_vec, self.C, self.f)
        return compute_V(self.image, flow, noise_std=self.noise_std, rng=self.rng)

    def warp_image(self, R_vec: np.ndarray) -> np.ndarray:
        """
        Warp the reference image by the rotation R_vec and return the
        temporal difference as V.  Provides a more accurate V for large
        rotations (uses bilinear interpolation).
        """
        flow = rotation_flow(R_vec, self.C, self.f)

        v_coords, u_coords = np.meshgrid(
            np.arange(self.H, dtype=np.float64),
            np.arange(self.W, dtype=np.float64),
            indexing='ij',
        )
        src_u = u_coords - flow[..., 0]
        src_v = v_coords - flow[..., 1]

        # Clamp source coordinates to image bounds
        src_u = np.clip(src_u, 0, self.W - 1)
        src_v = np.clip(src_v, 0, self.H - 1)

        warped = map_coordinates(
            self.image, [src_v, src_u], order=1, mode='nearest'
        )
        V = warped - self.image

        if self.noise_std > 0.0:
            V = V + self.rng.standard_normal(V.shape) * self.noise_std

        return V
