"""
Camera geometry for the InteractingMaps pipeline.

Provides:
  - compute_calibration    : per-pixel unit ray directions C (H,W,3), the
                             general-calibration map used by the Cook network.
  - build_kinematic_matrix : the (H,W,2,3) matrix that maps angular velocity ω
                             to pixel flow (F = C_mat·ω), thesis Eq. 6.37/6.38.

Distortion (build_kinematic_matrix, dist_coeffs)
  - dist_coeffs=None            → ideal pinhole (events assumed pre-undistorted).
  - dist_coeffs=[k1,k2,p1,p2,k3]→ "Way 2": the native (distorted) pixel grid is
    back-projected to true undistorted rays, and (if include_jacobian) the flow
    is mapped into distorted-pixel space via the Brown–Conrady Jacobian J_D:
        C_mat = diag(fx,fy) · J_D(x',y') · A(x',y')
    This matches the distortion-aware warp in cmax/angular_velocity.py.
"""

import numpy as np
import cv2

def compute_calibration(H: int, W: int, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    Compute the camera calibration map C using real intrinsics from calib.txt.

    Each pixel (col, row) maps to a unit direction vector:
        C[row, col] = normalize( (col - cx)/fx, (row - cy)/fy, 1 )

    Returns
    -------
    C : (H, W, 3) float64, unit 3-D direction per pixel.
    """
    cols = np.arange(W, dtype=np.float64)
    rows = np.arange(H, dtype=np.float64)
    uu, vv = np.meshgrid(cols, rows)  # uu = col, vv = row

    xn = (uu - cx) / fx
    yn = (vv - cy) / fy
    zn = np.ones_like(xn)

    norm = np.sqrt(xn**2 + yn**2 + zn**2)
    C = np.stack([xn / norm, yn / norm, zn / norm], axis=-1)
    return C

def build_kinematic_matrix(H, W, fx, fy, cx, cy,
                           dist_coeffs=None, include_jacobian=True):
    """
    (H, W, 2, 3) matrix mapping ω -> pixel flow.

    dist_coeffs=None      -> ideal pinhole
    dist_coeffs=[k1,k2,p1,p2,k3]:
        x', y' become the true UNDISTORTED normalized coords of each native
        (distorted) pixel; if include_jacobian, the flow is mapped into
        distorted-pixel space via the distortion Jacobian J_D:
            C_mat = diag(fx,fy) @ J_D(x',y') @ A(x',y')
    """
    cols = np.arange(W, dtype=np.float64)
    rows = np.arange(H, dtype=np.float64)
    uu, vv = np.meshgrid(cols, rows)

    if dist_coeffs is None:
        xp = (uu - cx) / fx
        yp = (vv - cy) / fy
        Jxx = Jyy = np.ones_like(xp)
        Jxy = np.zeros_like(xp)
    else:
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.asarray(dist_coeffs, dtype=np.float64).ravel()
        pts = np.stack([uu.ravel(), vv.ravel()], axis=-1).reshape(-1, 1, 2)
        norm = cv2.undistortPoints(pts, K, dist)   # no P -> normalized undistorted
        xp = norm[:, 0, 0].reshape(H, W)
        yp = norm[:, 0, 1].reshape(H, W)

        if include_jacobian:
            k1, k2, p1, p2, k3 = dist[:5]
            r2 = xp**2 + yp**2
            krad = 1 + k1*r2 + k2*r2**2 + k3*r2**3
            s = k1 + 2*k2*r2 + 3*k3*r2**2
            Jxx = krad + 2*xp**2*s + 2*p1*yp + 6*p2*xp
            Jxy = 2*xp*yp*s + 2*p1*xp + 2*p2*yp
            Jyy = krad + 2*yp**2*s + 6*p1*yp + 2*p2*xp
        else:
            Jxx = Jyy = np.ones_like(xp)
            Jxy = np.zeros_like(xp)

    # Rotational-flow matrix A(x',y'), rows = [Fx'; Fy']
    A = np.empty((H, W, 2, 3), dtype=np.float64)
    A[..., 0, 0] = xp * yp
    A[..., 0, 1] = -(xp**2 + 1.0)
    A[..., 0, 2] = yp
    A[..., 1, 0] = yp**2 + 1.0
    A[..., 1, 1] = -xp * yp
    A[..., 1, 2] = -xp

    # C_mat = diag(fx,fy) @ J_D @ A   (J_D symmetric: Jyx = Jxy)
    C_mat = np.empty((H, W, 2, 3), dtype=np.float64)
    C_mat[..., 0, :] = fx * (Jxx[..., None] * A[..., 0, :] + Jxy[..., None] * A[..., 1, :])
    C_mat[..., 1, :] = fy * (Jxy[..., None] * A[..., 0, :] + Jyy[..., None] * A[..., 1, :])
    return C_mat


# ===========================================================================
# Shared between the Cook and thesis networks (R4 of architecture_review.md)
# ===========================================================================
#
# Both networks estimate R by the same least squares and clip the maps to the
# same bounds; keeping those here means the two implementations cannot drift
# apart, so "the two networks differ only in the update scheme" (config.py) is
# structural, not a comment. They still differ in HOW they blend toward R_target
# (Cook: Gauss-Seidel direct blend; thesis: a Jacobi gradient step) -- that is
# the one intended difference, and it stays in each network.

# Stability clip bounds applied after each iteration. I/G/F are shared by both
# networks; R is clipped only by the thesis (Jacobi) network -- Cook's
# Gauss-Seidel update does not clip R.
CLIP_I = 10.0
CLIP_G = 5.0
CLIP_F = 10.0
CLIP_R = 1.0  # thesis network only


def build_R_normal_equations(C_mat):
    """M⁻¹ for the R least squares, where M = Σ_{x,y} C_matᵀ·C_mat (3×3, constant).

    Solves argmin_R Σ ‖F − C_mat·R‖²; precomputed once since C_mat is constant
    (thesis Eq. 6.48, footnote 18). Used by both networks."""
    M = np.einsum('hwji,hwjk->ik', C_mat, C_mat)   # (3, 3)
    return np.linalg.inv(M)


def solve_R_lstsq(M_inv, C_mat, F):
    """Closed-form least-squares R* = M⁻¹·(Σ C_matᵀ·F) from the current flow F.

    Both networks then *blend* toward this R* in their own update scheme
    (Cook directly, thesis via a gradient step)."""
    v = np.einsum('hwji,hwj->i', C_mat, F)         # (3,)
    return M_inv @ v
