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