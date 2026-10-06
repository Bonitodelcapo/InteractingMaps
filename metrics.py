"""
Metrics and reconstruction-quality descriptors (R3 of architecture_review.md).

Pure scoring helpers extracted verbatim from evaluation.py: the per-frame omega
metrics, and the descriptors that let many runs be ranked without re-running them
(APS correlation, curl share of G, intensity-map statistics). evaluation
re-exports every name here, so existing callers are unchanged.
"""

import numpy as np


def compute_metrics(omega_est, omega_gt):
    """Compute all metrics for a single frame."""
    err = np.linalg.norm(omega_est - omega_gt) * 180 / np.pi
    norm_est = np.linalg.norm(omega_est)
    norm_gt = np.linalg.norm(omega_gt)
    if norm_est > 1e-6 and norm_gt > 1e-6:
        cos_a = np.clip(np.dot(omega_est, omega_gt) / (norm_est * norm_gt), -1, 1)
        dir_err = np.degrees(np.arccos(cos_a))
        beta = norm_gt / norm_est
    else:
        dir_err = 180.0
        beta = 0.0
    return err, dir_err, beta


def _corr_to_aps(img, gt_images, t_mid, crop_frac=0.1):
    """Pearson r between an intensity map and the nearest APS frame.

    Invariant to scale and offset, the two freedoms the reconstruction has.
    Interior only: the border is where the Poisson boundary assumption is
    weakest. nan when the dataset ships no APS.
    """
    if not gt_images or img is None:
        return float('nan')
    try:
        import matplotlib.pyplot as plt
        times = np.array([t for t, _ in gt_images])
        aps = plt.imread(gt_images[int(np.argmin(np.abs(times - t_mid)))][1])
        if aps.ndim == 3:
            aps = aps.mean(-1)
        ref = np.log(aps.astype(np.float64) + 1.0)
        if ref.shape != img.shape:
            return float('nan')
        h, w = img.shape
        dy, dx = int(h * crop_frac), int(w * crop_frac)
        a = img[dy:h - dy, dx:w - dx].ravel()
        b = ref[dy:h - dy, dx:w - dx].ravel()
        a = a - a.mean()
        b = b - b.mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / d) if d > 0 else float('nan')
    except Exception:
        return float('nan')


def _recon_scores(net, H, W, gt_images, t_mid):
    """Reconstruction quality of BOTH intensity maps the pipeline produces.

    in-loop : net.I, carried across frames by the warm start, which is what the
              saved frames show.
    read-out: the exact solve of Eq. 6.64 applied to the current G alone.

    They are different images and they do not rank runs the same way -- a term
    that shrinks G weakens the read-out while leaving the accumulated in-loop
    map intact. Scoring only one of them silently picks a winner.
    """
    from interacting_maps.network_dissertation import solve_poisson_exact
    inloop = _corr_to_aps(np.asarray(net.I[:H, :W], dtype=np.float64),
                          gt_images, t_mid)
    try:
        ro = _corr_to_aps(solve_poisson_exact(net.G[:H, :W], 'fft'),
                          gt_images, t_mid)
    except Exception:
        ro = float('nan')
    return inloop, ro


def curl_share(G):
    """Fraction of G's energy that is NOT any image's gradient.

    G is a gradient field only if it is curl-free: walking a closed loop and
    summing the steps must return zero, since the two endpoints are the same
    pixel and must have the same intensity. Nothing in the model enforces that.
    Cost_Spatial pulls G towards grad(I), but Cost_OFCE only needs F.G = -V at
    each pixel separately and couples no neighbours, so the two costs settle on
    a compromise that is generally not curl-free -- and the aperture problem
    leaves the component of G perpendicular to F free to drift.

    Recovering I from G is a least-squares projection onto the gradient fields,
    so whatever curl G carries is silently discarded. This measures how much
    that is, by Helmholtz decomposition: project G onto the gradient subspace
    and return the relative energy of the residual. 0 means G is exactly some
    image's gradient; 0.4 means nearly half of it describes no image at all.
    """
    G = np.asarray(G, dtype=np.float64)
    H, W = G.shape[:2]
    gx, gy = G[..., 0], G[..., 1]
    dx = np.exp(2j * np.pi * np.fft.fftfreq(W)[None, :]) - 1.0
    dy = np.exp(2j * np.pi * np.fft.fftfreq(H)[:, None]) - 1.0
    den = np.abs(dx) ** 2 + np.abs(dy) ** 2
    den[0, 0] = 1.0
    phi = (np.conj(dx) * np.fft.fft2(gx) + np.conj(dy) * np.fft.fft2(gy)) / den
    px = np.real(np.fft.ifft2(dx * phi))
    py = np.real(np.fft.ifft2(dy * phi))
    tot = float((gx ** 2 + gy ** 2).sum())
    if tot <= 0:
        return 0.0
    return float(((gx - px) ** 2 + (gy - py) ** 2).sum() / tot)


def _intensity_stats(I, radius=10):
    """Two numbers that separate a photograph from an edge map.

    low_freq_share : fraction of |FFT(I)| inside a disc of `radius` cycles.
        The iterative Poisson update (Eq. 6.61) cannot build these modes, so
        an edge map scores near zero while a real intensity image does not.
    contrast       : std of I, the quantity the beta scale ambiguity acts on.
    """
    I = np.asarray(I, dtype=np.float64)
    F = np.abs(np.fft.fftshift(np.fft.fft2(I - I.mean())))
    h, w = I.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - h // 2, xx - w // 2)
    tot = F.sum()
    return {'low_freq_share': float(F[r < radius].sum() / tot) if tot > 0 else 0.0,
            'contrast': float(I.std())}
