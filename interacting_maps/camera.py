import numpy as np


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


def _undistort_points_iterative(x_d, y_d, k1, k2, p1, p2, k3, n_iters=10):
    """
    Invert OpenCV's plumb-bob distortion model on a grid of distorted normalised
    coordinates.  Same fixed-point iteration as cv2.undistortPoints (no Jacobian
    needed — converges in ~5 iterations for typical |k1| < 1).

    Parameters
    ----------
    x_d, y_d : ndarray, distorted normalised coords (i.e. (col-cx)/fx, (row-cy)/fy)
    k1..k3, p1, p2 : Brown-Conrady coefficients (radial + tangential)
    n_iters : number of fixed-point iterations

    Returns
    -------
    x, y : ndarray, undistorted normalised coords such that the forward distortion
           of (x, y) reproduces (x_d, y_d).
    """
    x, y = x_d.copy(), y_d.copy()   # initial guess = distorted coords themselves
    for _ in range(n_iters):
        r2     = x * x + y * y
        k_rad  = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
        x_tang = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        y_tang = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x = (x_d - x_tang) / k_rad
        y = (y_d - y_tang) / k_rad
    return x, y


def build_kinematic_matrix(H: int, W: int, fx: float, fy: float, cx: float, cy: float,
                           dist_coeffs: np.ndarray | None = None) -> np.ndarray:
    """
    Precompute the (H, W, 2, 3) matrix that maps angular velocity ω → pixel flow.

    Pinhole case (dist_coeffs is None or all zeros) — Thesis Eq. 6.38:
        x' = (col - cx) / fx,  y' = (row - cy) / fy
        F_u = fx · [x'y' ωx - (x'²+1) ωy + y' ωz]
        F_v = fy · [(y'²+1) ωx - x'y' ωy - x' ωz]

    Distortion case (dist_coeffs = [k1, k2, p1, p2, k3], OpenCV plumb-bob):
        The pixel grid (col, row) is at DISTORTED locations on the sensor.
        For each pixel:
          1. Recover the undistorted normalised coords (x, y) by inverting
             the plumb-bob model via fixed-point iteration.
          2. Build the pinhole flow matrix M_ω(x, y) at the UNDISTORTED point.
          3. Multiply by the Jacobian J_g of the distortion mapping
             (x, y) → (x_d, y_d), which transforms velocity in undistorted
             coords to velocity in distorted coords.
          4. Scale by diag(fx, fy) to convert distorted-normalised velocity
             to distorted-pixel velocity.

        So C_mat_distorted[row, col] = diag(fx, fy) · J_g(x, y) · M_ω(x, y).

    The pure-rotation assumption (no translation, no parallax) is what lets us
    write this as a per-pixel 2×3 matrix that depends only on the calibration
    — no scene-depth dependence.
    """
    cols = np.arange(W, dtype=np.float64)
    rows = np.arange(H, dtype=np.float64)
    uu, vv = np.meshgrid(cols, rows)

    # Distorted normalised coordinates (always — the pixel grid is what it is).
    xd = (uu - cx) / fx
    yd = (vv - cy) / fy

    # Detect "no distortion" so we can fall through to the cheap pinhole path.
    has_dist = (dist_coeffs is not None) and np.any(np.asarray(dist_coeffs) != 0.0)

    if not has_dist:
        # ---- Pure pinhole: (x, y) = (x_d, y_d), J_g = identity ----
        x, y = xd, yd

        C_mat = np.zeros((H, W, 2, 3), dtype=np.float64)
        C_mat[:, :, 0, 0] = fx * (x * y)
        C_mat[:, :, 0, 1] = fx * (-(x ** 2 + 1.0))
        C_mat[:, :, 0, 2] = fx * y
        C_mat[:, :, 1, 0] = fy * (y ** 2 + 1.0)
        C_mat[:, :, 1, 1] = fy * (-x * y)
        C_mat[:, :, 1, 2] = fy * (-x)
        return C_mat

    # ---- Plumb-bob distortion path ----
    k1, k2, p1, p2, k3 = (float(dist_coeffs[i]) for i in range(5))

    # 1. Undistort the pixel grid to get the true ray coords (x, y).
    x, y = _undistort_points_iterative(xd, yd, k1, k2, p1, p2, k3)

    # 2. Jacobian J_g of (x, y) → (x_d, y_d).
    #    Using k_rad = 1 + k1 r² + k2 r⁴ + k3 r⁶ ;  dk_rad/dr² = k1 + 2k2 r² + 3k3 r⁴
    r2          = x * x + y * y
    k_rad       = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    dk_rad_dr2  = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r2 ** 2

    # ∂x_d/∂x and ∂y_d/∂y are the diagonal terms; ∂x_d/∂y = ∂y_d/∂x by symmetry.
    Jxx = k_rad + 2.0 * (x ** 2) * dk_rad_dr2 + 2.0 * p1 * y + 6.0 * p2 * x
    Jyy = k_rad + 2.0 * (y ** 2) * dk_rad_dr2 + 6.0 * p1 * y + 2.0 * p2 * x
    Jxy = 2.0 * x * y * dk_rad_dr2 + 2.0 * p1 * x + 2.0 * p2 * y     # = Jyx

    # 3. Pinhole flow Jacobian M_ω(x, y), evaluated at UNDISTORTED coords.
    M00 = x * y
    M01 = -(x ** 2 + 1.0)
    M02 = y
    M10 = (y ** 2 + 1.0)
    M11 = -x * y
    M12 = -x

    # 4. C_mat_distorted = diag(fx, fy) · J_g · M_ω
    #    Per-pixel:   [a b]   [M00 M01 M02]
    #                 [b d] · [M10 M11 M12]
    a = Jxx
    b = Jxy
    d = Jyy

    C_mat = np.empty((H, W, 2, 3), dtype=np.float64)
    C_mat[:, :, 0, 0] = fx * (a * M00 + b * M10)
    C_mat[:, :, 0, 1] = fx * (a * M01 + b * M11)
    C_mat[:, :, 0, 2] = fx * (a * M02 + b * M12)
    C_mat[:, :, 1, 0] = fy * (b * M00 + d * M10)
    C_mat[:, :, 1, 1] = fy * (b * M01 + d * M11)
    C_mat[:, :, 1, 2] = fy * (b * M02 + d * M12)
    return C_mat

#def compute_calibration(H: int, W: int, f: float) -> np.ndarray:
#    """
#    Compute the camera calibration map C.
#
#    Each pixel (u, v) maps to a unit vector on the sphere pointing in the
#    direction of that pixel.  C[v, u] = normalize((u-cx)/f, (v-cy)/f, 1).
#
#    Returns
#    -------
#    C : (H, W, 3)  float64, each row-vector is a unit 3-D direction.
#    """
#    cx, cy = W / 2.0, H / 2.0
#    u_coords = np.arange(W, dtype=np.float64)          # (W,)
#    v_coords = np.arange(H, dtype=np.float64)          # (H,)
#    uu, vv = np.meshgrid(u_coords, v_coords)            # (H, W)
#
#    xn = (uu - cx) / f
#    yn = (vv - cy) / f
#    zn = np.ones_like(xn)
#
#    norm = np.sqrt(xn ** 2 + yn ** 2 + zn ** 2)        # (H, W)
#    C = np.stack([xn / norm, yn / norm, zn / norm], axis=-1)  # (H, W, 3)
#    return C
#
#
#def m32(v3d: np.ndarray, C: np.ndarray, f: float) -> np.ndarray:
#    """
#    Project 3-D velocity vectors to 2-D image-plane flow (pixels/frame).
#
#    The tangential component of v3d (perpendicular to C) is projected onto
#    the image plane using a perspective division by C_z.
#
#    Parameters
#    ----------
#    v3d : (H, W, 3)
#    C   : (H, W, 3)  unit vectors (calibration map)
#    f   : focal length in pixels
#
#    Returns
#    -------
#    F2d : (H, W, 2)
#    """
#    # Tangential component: remove the radial (along-C) part
#    v_dot_C = np.einsum('hwk,hwk->hw', v3d, C)          # (H, W)
#    v_tang = v3d - v_dot_C[..., np.newaxis] * C          # (H, W, 3)
#
#    # Perspective division by C_z, scale by focal length
#    Cz = C[..., 2]                                        # (H, W)
#    eps = 1e-8
#    F2d = v_tang[..., :2] / (Cz[..., np.newaxis] + eps) * f  # (H, W, 2)
#    return F2d
#
#
#def m23(F2d: np.ndarray, C: np.ndarray, f: float) -> np.ndarray:
#    """
#    Unproject 2-D image-plane flow to an approximate 3-D velocity vector.
#
#    The inverse of m32: lift (F_u, F_v) to 3-D and make the result
#    tangential to C.
#
#    Parameters
#    ----------
#    F2d : (H, W, 2)
#    C   : (H, W, 3)
#    f   : focal length in pixels
#
#    Returns
#    -------
#    v3d : (H, W, 3)  tangential to C at each pixel
#    """
#    eps = 1e-8
#    Cz = C[..., 2]  # (H, W)
#
#    # Lift: reverse the perspective division
#    vx = F2d[..., 0] / f * Cz   # (H, W)
#    vy = F2d[..., 1] / f * Cz   # (H, W)
#    # z-component so that v · C = 0  (tangential constraint)
#    Cx, Cy = C[..., 0], C[..., 1]
#    vz = -(Cx * vx + Cy * vy) / (Cz + eps)  # (H, W)
#
#    v3d = np.stack([vx, vy, vz], axis=-1)    # (H, W, 3)
#
#    # Enforce tangential (remove residual radial component)
#    v_dot_C = np.einsum('hwk,hwk->hw', v3d, C)
#    v3d = v3d - v_dot_C[..., np.newaxis] * C
#    return v3d
#