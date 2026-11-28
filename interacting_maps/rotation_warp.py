"""
rotation_warp.py — exact rotational image warp for the persistence network.

Martel 2019 (§6.5, Eqs. 6.68–6.98) compares the current intensity I with the
previous one Ǐ warped by the inter-frame rotation R_δ = exp([R]×) (Rodrigues,
Eq. 6.91). This module holds the camera geometry of that warp, kept apart from
the network so it can be tested on its own:

    T_R(p) = π( exp([R]×) · π⁻¹(p) )

π⁻¹ turns a native (distorted) pixel into a unit ray, π projects a ray back
through the same lens (Brown–Conrady [k1,k2,p1,p2,k3], as in
camera.build_kinematic_matrix). T_R(p) is the position in Ǐ that pixel p of I
came from, so I(p) ≈ Ǐ(T_R(p)) and, to first order, T_R(p) = p − C(p)·R with
the network's kinematic matrix C — the same flow convention as F.

Deviation from Martel (spec, deviation 3): his warp uses a pinhole K; ours
carries the lens distortion, because I lives in distorted pixels (C_full).
"""

from collections import namedtuple

import cv2
import numpy as np

#: Intrinsics + distortion. dist is None (pinhole) or the (5,) Brown–Conrady vector.
Camera = namedtuple('Camera', 'fx fy cx cy dist')


def pixel_rays(H, W, cam):
    """(H, W, 3) unit rays π⁻¹(p) of every native pixel."""
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    if cam.dist is None:
        a = (uu - cam.cx) / cam.fx
        b = (vv - cam.cy) / cam.fy
    else:
        K = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]], np.float64)
        pts = np.stack([uu.ravel(), vv.ravel()], axis=-1).reshape(-1, 1, 2)
        n = cv2.undistortPoints(pts, K, np.asarray(cam.dist, np.float64))
        a = n[:, 0, 0].reshape(H, W)
        b = n[:, 0, 1].reshape(H, W)
        a, b = _refine_undistort(a, b, (uu - cam.cx) / cam.fx, (vv - cam.cy) / cam.fy,
                                 cam.dist)
    p = np.stack([a, b, np.ones_like(a)], axis=-1)
    return p / np.linalg.norm(p, axis=-1, keepdims=True)


def _distort(a, b, dist):
    """Forward Brown–Conrady: ideal normalized (a, b) → distorted (x_d, y_d)."""
    k1, k2, p1, p2, k3 = np.asarray(dist, np.float64)[:5]
    r2 = a * a + b * b
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    return (a * rad + 2 * p1 * a * b + p2 * (r2 + 2 * a * a),
            b * rad + p1 * (r2 + 2 * b * b) + 2 * p2 * a * b)


def _distort_jacobian(a, b, dist):
    """Symmetric J_D = ∂(x_d, y_d)/∂(a, b), as in camera.build_kinematic_matrix."""
    k1, k2, p1, p2, k3 = np.asarray(dist, np.float64)[:5]
    r2 = a ** 2 + b ** 2
    krad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    s = k1 + 2 * k2 * r2 + 3 * k3 * r2 ** 2
    return (krad + 2 * a ** 2 * s + 2 * p1 * b + 6 * p2 * a,
            2 * a * b * s + 2 * p1 * a + 2 * p2 * b,
            krad + 2 * b ** 2 * s + 6 * p1 * b + 2 * p2 * a)


def _refine_undistort(a, b, xd, yd, dist, n_iter=10):
    """Newton steps on distort(a, b) = (xd, yd) from cv2's estimate.

    cv2.undistortPoints stops after 5 fixed-point iterations, which leaves
    ~0.03 px at the corners — a spurious warp at R = 0 without this."""
    for _ in range(n_iter):
        ex, ey = _distort(a, b, dist)
        ex, ey = ex - xd, ey - yd
        jxx, jxy, jyy = _distort_jacobian(a, b, dist)
        det = jxx * jyy - jxy * jxy
        a = a - (jyy * ex - jxy * ey) / det
        b = b - (jxx * ey - jxy * ex) / det
    return a, b


def rotation_matrix(R):
    """exp([R]×), the rotation the warp applies to the rays (Eq. 6.91)."""
    return cv2.Rodrigues(np.asarray(R, np.float64).reshape(3, 1))[0]


def compose(delta, R):
    """R' with exp([R']×) = exp([delta]×)·exp([R]×): the so(3) update of Eq. 6.98."""
    return cv2.Rodrigues(rotation_matrix(delta) @ rotation_matrix(R))[0].ravel()


def _rotated(rays, R):
    return np.einsum('ij,hwj->hwi', rotation_matrix(R), rays)


def _project(q, cam):
    """Ray q → (normalized a, b) and pixel (u, v), through the lens."""
    qz = np.where(np.abs(q[..., 2]) < 1e-9, 1e-9, q[..., 2])
    a, b = q[..., 0] / qz, q[..., 1] / qz
    if cam.dist is None:
        xd, yd = a, b
    else:
        xd, yd = _distort(a, b, cam.dist)
    return a, b, qz, cam.fx * xd + cam.cx, cam.fy * yd + cam.cy


def warp_coords(rays, R, cam):
    """T_R(p) as pixel arrays (u, v), each (H, W)."""
    _, _, _, u, v = _project(_rotated(rays, R), cam)
    return u, v


def warp_jacobian(rays, R, cam):
    """(H, W, 2, 3) ∂T_{compose(Δ,R)}(p)/∂Δ at Δ = 0 (Eqs. 6.81–6.96).

    exp([Δ]×)·q ≈ q − [q]×Δ, so J = diag(fx,fy) · J_D(a,b) · ∂(a,b)/∂q · (−[q]×).
    """
    q = _rotated(rays, R)
    a, b, qz, _, _ = _project(q, cam)
    # ∂(a,b)/∂q  (H, W, 2, 3)
    P = np.zeros(q.shape[:2] + (2, 3))
    P[..., 0, 0] = 1 / qz
    P[..., 0, 2] = -a / qz
    P[..., 1, 1] = 1 / qz
    P[..., 1, 2] = -b / qz
    # −[q]×
    S = np.zeros(q.shape[:2] + (3, 3))
    S[..., 0, 1], S[..., 0, 2] = q[..., 2], -q[..., 1]
    S[..., 1, 0], S[..., 1, 2] = -q[..., 2], q[..., 0]
    S[..., 2, 0], S[..., 2, 1] = q[..., 1], -q[..., 0]
    J_ab = np.einsum('hwij,hwjk->hwik', P, S)
    # distortion Jacobian J_D (symmetric), same formulas as build_kinematic_matrix
    if cam.dist is None:
        Jxx = Jyy = np.ones_like(a)
        Jxy = np.zeros_like(a)
    else:
        Jxx, Jxy, Jyy = _distort_jacobian(a, b, cam.dist)
    J = np.empty_like(J_ab)
    J[..., 0, :] = cam.fx * (Jxx[..., None] * J_ab[..., 0, :] + Jxy[..., None] * J_ab[..., 1, :])
    J[..., 1, :] = cam.fy * (Jxy[..., None] * J_ab[..., 0, :] + Jyy[..., None] * J_ab[..., 1, :])
    return J


def bilinear_sample(img, u, v):
    """img at (u, v) by bilinear interpolation → (vals, valid).

    valid is False outside [0, W-1] × [0, H-1]; vals is 0 there."""
    H, W = img.shape
    valid = (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    uc = np.clip(u, 0, W - 1)
    vc = np.clip(v, 0, H - 1)
    x0 = np.minimum(np.floor(uc).astype(int), W - 2)
    y0 = np.minimum(np.floor(vc).astype(int), H - 2)
    fx, fy = uc - x0, vc - y0
    vals = ((1 - fx) * (1 - fy) * img[y0, x0] + fx * (1 - fy) * img[y0, x0 + 1]
            + (1 - fx) * fy * img[y0 + 1, x0] + fx * fy * img[y0 + 1, x0 + 1])
    return np.where(valid, vals, 0.0), valid


def solve_rotation_gn(I, Icheck, grad_Icheck, rays, cam, R0, n_max=20, tol=1e-8,
                      mask=None):
    """Gauss–Newton on Σ ρ_i², ρ_i(R) = I(p_i) − Ǐ(T_R(p_i))  (Eqs. 6.72–6.98).

    Starts at R0 and iterates to convergence (|Δ| < tol) or n_max steps, each
    step linearised in so(3) and applied as R ← compose(Δ, R). grad_Icheck is
    (gx, gy) of Ǐ, precomputed by the caller (Ǐ is fixed within a frame).
    Pixels whose warped position leaves the image, or outside `mask`, are
    excluded. A rank-deficient system (e.g. a textureless Ǐ) gives the
    minimum-norm step, so R does not move along unobserved directions; with
    fewer than 3 usable pixels, or a non-finite result, R0 is returned.
    """
    gx, gy = grad_Icheck
    R = np.asarray(R0, np.float64).copy()
    for _ in range(n_max):
        u, v = warp_coords(rays, R, cam)
        Iw, valid = bilinear_sample(Icheck, u, v)
        if mask is not None:
            valid &= mask
        if valid.sum() < 3:
            return np.asarray(R0, np.float64).copy()
        gxw, _ = bilinear_sample(gx, u, v)
        gyw, _ = bilinear_sample(gy, u, v)
        Jt = warp_jacobian(rays, R, cam)                       # (H, W, 2, 3)
        J = -(gxw[..., None] * Jt[..., 0, :] + gyw[..., None] * Jt[..., 1, :])
        rho = (I - Iw)[valid]
        delta = np.linalg.lstsq(J[valid], -rho, rcond=1e-10)[0]
        R = compose(delta, R)
        if not np.isfinite(R).all():
            return np.asarray(R0, np.float64).copy()
        if np.linalg.norm(delta) < tol:
            break
    return R


def block_match(I, Icheck, half_block=3, radius=4):
    """Block-matching flow F′ of I against Ǐ (Eq. 6.138) → (F_prime, valid).

    For each pixel p, the integer d = (dx, dy), |dx|, |dy| ≤ radius, minimising
    the sum of absolute differences between the (2·half_block+1)² block of I at
    p and the block of Ǐ at p − d — the flow convention of F, I(p) ≈ Ǐ(p − F).
    Candidates are visited in order of |d| with a strict '<', so ties (flat
    regions) keep the smallest shift, zero first. valid is False within
    half_block + radius of the border, where block or search leave the image.
    """
    from scipy.ndimage import uniform_filter
    H, W = I.shape
    size = 2 * half_block + 1
    cands = [(dx, dy) for dy in range(-radius, radius + 1)
             for dx in range(-radius, radius + 1)]
    cands.sort(key=lambda d: d[0] ** 2 + d[1] ** 2)
    best = np.full((H, W), np.inf)
    F = np.zeros((H, W, 2))
    for dx, dy in cands:
        shifted = np.roll(Icheck, shift=(dy, dx), axis=(0, 1))   # Ǐ(p − d)
        sad = uniform_filter(np.abs(I - shifted), size=size, mode='nearest')
        better = sad < best
        best[better] = sad[better]
        F[better] = (dx, dy)
    m = half_block + radius
    valid = np.zeros((H, W), bool)
    valid[m:H - m, m:W - m] = True
    return F, valid
