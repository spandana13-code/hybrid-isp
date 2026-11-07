"""Full-reference quality metrics: PSNR, SSIM, CIEDE2000 Delta E, perceptual (LPIPS or GMSD)."""
from __future__ import annotations

import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2lab, deltaE_ciede2000

_LPIPS = None
_LPIPS_TRIED = False


def psnr(ref, img):
    return float(peak_signal_noise_ratio(ref, np.clip(img, 0, 1), data_range=1.0))


def ssim(ref, img):
    return float(structural_similarity(ref, np.clip(img, 0, 1), channel_axis=2, data_range=1.0))


def delta_e(ref, img):
    """Mean CIEDE2000 between reference and output (sRGB D65)."""
    return float(deltaE_ciede2000(rgb2lab(ref), rgb2lab(np.clip(img, 0, 1))).mean())


def gmsd(ref, img):
    """Gradient Magnitude Similarity Deviation (Xue et al. 2014). Lower is better, 0 = identical."""
    def y(x):
        x = (np.clip(x, 0, 1) * 255).astype(np.float32)
        y = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
        return cv2.resize(cv2.blur(y, (2, 2)), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    a, b = y(ref), y(img)
    k = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], np.float32) / 3
    def gm(z):
        return np.sqrt(cv2.filter2D(z, -1, k) ** 2 + cv2.filter2D(z, -1, k.T) ** 2)
    ga, gb = gm(a), gm(b)
    c = 170.0
    gms = (2 * ga * gb + c) / (ga ** 2 + gb ** 2 + c)
    return float(gms.std())


def lpips_fn():
    """LPIPS(alex) if its backbone weights can be loaded (needs internet once), else None."""
    global _LPIPS, _LPIPS_TRIED
    if not _LPIPS_TRIED:
        _LPIPS_TRIED = True
        try:
            import torch, lpips  # noqa
            _LPIPS = lpips.LPIPS(net="alex", verbose=False).eval()
        except Exception:
            _LPIPS = None
    return _LPIPS


def lpips(ref, img):
    fn = lpips_fn()
    if fn is None:
        return float("nan")
    import torch
    t = lambda x: torch.from_numpy(np.clip(x, 0, 1).astype(np.float32)).permute(2, 0, 1)[None] * 2 - 1
    with torch.no_grad():
        return float(fn(t(ref), t(img)).item())


def all_metrics(ref, img, with_lpips=True):
    m = dict(psnr=psnr(ref, img), ssim=ssim(ref, img), delta_e=delta_e(ref, img), gmsd=gmsd(ref, img))
    if with_lpips:
        m["lpips"] = lpips(ref, img)
    return m
