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


def build_kinematic_matrix(H: int, W: int, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    Precompute the (H, W, 2, 3) matrix that maps angular velocity R → pixel flow.

    From Thesis Eq. 6.38 (general intrinsics):
        x' = (col - cx) / fx
        y' = (row - cy) / fy

        F_u = fx*[x'y' ωx - (x'²+1) ωy + y' ωz]
        F_v = fy*[(y'²+1) ωx - x'y' ωy - x' ωz]

    Parameters come directly from calib.txt.
    """
    cols = np.arange(W, dtype=np.float64)
    rows = np.arange(H, dtype=np.float64)
    uu, vv = np.meshgrid(cols, rows)

    xp = (uu - cx) / fx  # normalized x'
    yp = (vv - cy) / fy  # normalized y'

    C_mat = np.zeros((H, W, 2, 3), dtype=np.float64)

    # Horizontal flow (F_u), scaled by fx:
    C_mat[:, :, 0, 0] = fx * (xp * yp)
    C_mat[:, :, 0, 1] = fx * (-(xp**2 + 1.0))
    C_mat[:, :, 0, 2] = fx * yp

    # Vertical flow (F_v), scaled by fy:
    C_mat[:, :, 1, 0] = fy * (yp**2 + 1.0)
    C_mat[:, :, 1, 1] = fy * (-xp * yp)
    C_mat[:, :, 1, 2] = fy * (-xp)

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