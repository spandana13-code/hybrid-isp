"""Classical ISP stages + the adaptive controller.

Every stage is a small pure function so it can be reused by the classical baseline,
the hybrid front-end and the ablation study.
"""
from __future__ import annotations

import time
import numpy as np
import cv2

from .raw_sim import BLACK_LEVEL, WHITE_LEVEL, CAM2RGB_NOMINAL, linear_to_srgb, pack_rggb


# ----------------------------------------------------------------------------- stage 1
def preprocess(raw_dn: np.ndarray, black=BLACK_LEVEL, white=WHITE_LEVEL) -> np.ndarray:
    """Black-level subtraction + normalisation to [0,1] + hot-pixel suppression."""
    x = (raw_dn.astype(np.float32) - black) / (white - black)
    x = np.clip(x, 0.0, 1.0)
    # hot/dead pixel suppression per CFA plane: clamp to 3x3 same-colour max of neighbours
    p = pack_rggb(x)
    out = np.empty_like(p)
    for c in range(4):
        ch = p[..., c]
        k = np.ones((3, 3), np.uint8); k[1, 1] = 0
        nmax = cv2.dilate(ch, k)
        out[..., c] = np.minimum(ch, nmax * 1.5 + 0.02)
    x[0::2, 0::2], x[0::2, 1::2], x[1::2, 0::2], x[1::2, 1::2] = (out[..., i] for i in range(4))
    return x


# ----------------------------------------------------------------------------- analysis
def estimate_noise_profile(raw: np.ndarray, bins: int = 12) -> tuple[float, float]:
    """Fit var(x) = a*x + b from the two green planes of the CFA.

    Gr and Gb sample (almost) the same scene, so (Gr - Gb)/sqrt(2) is dominated by noise.
    A robust (MAD) variance is computed in brightness bins and a line is fitted.
    """
    p = pack_rggb(raw)
    gr, gb = p[..., 1], p[..., 2]
    d = (gr - gb) / np.sqrt(2)
    m = (gr + gb) / 2
    # Gr/Gb are offset by one diagonal pixel, so edges leak into d. Keep only flat regions:
    # local gradient of the (smoothed) green below the 40th percentile.
    ms = cv2.GaussianBlur(m, (0, 0), 1.0)
    grad = np.abs(cv2.Sobel(ms, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(ms, cv2.CV_32F, 0, 1))
    flat = grad <= np.quantile(grad, 0.4)
    m, d = ms[flat].ravel(), d[flat].ravel()
    valid = m < 0.9
    m, d = m[valid], d[valid]
    if m.size < 100:
        return 1e-3, 1e-5
    qs = np.quantile(m, np.linspace(0, 1, bins + 1))
    xs, vs = [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        sel = (m >= lo) & (m <= hi)
        if sel.sum() < 50:
            continue
        dd = d[sel]
        sig = 1.4826 * np.median(np.abs(dd - np.median(dd)))
        xs.append(m[sel].mean()); vs.append(sig ** 2)
    xs, vs = np.array(xs), np.array(vs)
    if len(xs) < 2:
        return 1e-3, float(vs.mean() if len(vs) else 1e-5)
    a, b = np.polyfit(xs, vs, 1)
    return float(max(a, 1e-6)), float(max(b, 1e-8))


def estimate_wb(raw: np.ndarray, p: float = 6.0) -> tuple[float, float]:
    """Shades-of-grey (Minkowski p-norm) illuminant estimate on the packed CFA."""
    pk = pack_rggb(raw)
    r, g, b = pk[..., 0], (pk[..., 1] + pk[..., 2]) / 2, pk[..., 3]
    mx = np.maximum(np.maximum(r, g), b)
    lo = np.quantile(mx, 0.02)
    mask = (mx < 0.95) & (mx > lo)
    if mask.sum() < 100:
        mask = np.ones_like(mx, bool)
    def norm(c):
        return np.power(np.mean(np.power(np.maximum(c[mask], 0), p)), 1 / p) + 1e-8
    nr, ng, nb = norm(r), norm(g), norm(b)
    return float(np.clip(ng / nr, 0.5, 5.0)), float(np.clip(ng / nb, 0.5, 5.0))


def auto_exposure_gain(raw: np.ndarray, wb: tuple[float, float]) -> float:
    """Digital gain so that the bright end (99.5th pct of max channel) lands near 0.9."""
    pk = pack_rggb(raw)
    r, g, b = pk[..., 0] * wb[0], (pk[..., 1] + pk[..., 2]) / 2, pk[..., 3] * wb[1]
    mx = np.maximum(np.maximum(r, g), b)
    hi = np.quantile(mx, 0.995)
    mean = mx.mean()
    g1 = 0.92 / max(hi, 1e-4)
    g2 = 0.22 / max(mean, 1e-5)                     # don't over-brighten scenes with a light source
    return float(np.clip(min(g1, max(g2, g1 * 0.5)), 1.0, 40.0))


def analyze(raw: np.ndarray, wb=None, gain=None) -> dict:
    """Adaptive controller: measures the frame and chooses per-stage parameters.

    wb   : as-shot white-balance gains from metadata (R/G, B/G); estimated when None.
    gain : digital exposure gain; 1.0 for real camera RAW (the camera already exposed it),
           auto-exposure when None (simulated / under-exposed input).
    """
    a, b = estimate_noise_profile(raw)
    wb_src = "metadata" if wb is not None else "estimated"
    wb = tuple(float(v) for v in wb) if wb is not None else estimate_wb(raw)
    gain = float(gain) if gain is not None else auto_exposure_gain(raw, wb)
    mean = float(raw.mean())
    # effective noise std at mid-grey after gain, in the *output* (gamma) domain
    x_mid = 0.18 / gain
    sig_lin = np.sqrt(a * x_mid + b) * gain
    sig_out = float(sig_lin * 0.5 * (0.18 ** (-0.58)))     # d(sRGB)/dx around mid-grey ~ 1.36
    if gain > 6 and sig_out > 0.06:
        scene = "low-light + high noise"
    elif gain > 6:
        scene = "low light"
    elif sig_out > 0.05:
        scene = "high noise"
    else:
        scene = "normal"
    return dict(
        shot=a, read=b, wb_gains=wb, wb_source=wb_src, gain=gain, mean_raw=mean, noise_sigma=sig_out, scene=scene,
        denoise_h=float(np.clip(sig_out * 255 * 0.9, 0, 40)),
        sharpen=float(np.clip(0.7 - 6.0 * sig_out, 0.0, 0.7)),
    )


# ----------------------------------------------------------------------------- stages
def apply_wb(raw: np.ndarray, wb: tuple[float, float]) -> np.ndarray:
    out = raw.copy()
    out[0::2, 0::2] *= wb[0]
    out[1::2, 1::2] *= wb[1]
    return out


def demosaic(raw: np.ndarray, method: str = "ea") -> np.ndarray:
    """RGGB Bayer -> RGB. OpenCV names CFAs by the 2x2 block starting at (1,1), so RGGB == BayerBG."""
    u16 = (np.clip(raw, 0, 1) * 65535).astype(np.uint16)
    code = {"bilinear": cv2.COLOR_BayerBG2RGB, "ea": cv2.COLOR_BayerBG2RGB_EA}[method]
    return cv2.cvtColor(u16, code).astype(np.float32) / 65535.0


def apply_ccm(rgb: np.ndarray, ccm: np.ndarray = CAM2RGB_NOMINAL) -> np.ndarray:
    return np.clip(rgb @ ccm.T, 0, None).astype(np.float32)


def tone_map(lin: np.ndarray, knee: float = 0.8) -> np.ndarray:
    """Identity below `knee`, smooth roll-off above (highlight compression), then sRGB gamma.
    knee >= 1 gives the plain sRGB curve (what camera-referred ground truth like SIDD uses)."""
    if knee >= 1:
        return linear_to_srgb(lin)
    x = lin.copy()
    hi = x > knee
    t = (x[hi] - knee) / (1 - knee)
    x[hi] = knee + (1 - knee) * t / (1 + t)
    return linear_to_srgb(x)


def denoise(srgb: np.ndarray, h: float, method: str = "nlm") -> np.ndarray:
    """Edge-preserving denoise. NLM works in CIELab internally (luma h, chroma 1.5h)."""
    if h < 0.5:
        return srgb
    u8 = (np.clip(srgb, 0, 1) * 255 + 0.5).astype(np.uint8)
    if method == "nlm":
        out = cv2.fastNlMeansDenoisingColored(u8, None, h, h * 1.5, 5, 15)
    else:  # bilateral in YCrCb: light on luma, strong on chroma
        ycc = cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = (np.ascontiguousarray(ycc[..., i]) for i in range(3))
        ycc = np.stack([cv2.bilateralFilter(y, 7, h * 2, 5),
                        cv2.bilateralFilter(cr, 9, h * 4, 7),
                        cv2.bilateralFilter(cb, 9, h * 4, 7)], -1)
        out = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)
    return out.astype(np.float32) / 255.0


def sharpen(srgb: np.ndarray, amount: float, sigma: float = 1.0) -> np.ndarray:
    """Unsharp mask on the luma channel only (YCrCb) so colour noise is not amplified."""
    if amount <= 0:
        return srgb
    ycc = cv2.cvtColor(np.clip(srgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2YCrCb)
    y = ycc[..., 0]
    blur = cv2.GaussianBlur(y, (0, 0), sigma)
    detail = y - blur
    # soft threshold on detail to avoid sharpening residual noise
    thr = 0.004
    detail = np.sign(detail) * np.maximum(np.abs(detail) - thr, 0)
    ycc[..., 0] = y + amount * detail
    return np.clip(cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB), 0, 1)


# ----------------------------------------------------------------------------- pipelines
class Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, name):
        timer = self
        class _C:
            def __enter__(s):
                s.t0 = time.perf_counter()
            def __exit__(s, *a):
                timer.t[name] = timer.t.get(name, 0) + (time.perf_counter() - s.t0) * 1000
        return _C()


def front_end(raw_dn: np.ndarray, info: dict | None = None, timer: Timer | None = None,
              use_wb=True, use_ccm=True, use_gain=True, demosaic_method="ea", ccm=None,
              black=BLACK_LEVEL, white=WHITE_LEVEL, wb=None, gain=None, knee=0.8):
    """Shared classical front-end: preprocess -> analyse -> WB -> demosaic -> CCM -> gain -> tone.

    Returns (gamma-encoded RGB, linear RGB after gain, info).
    """
    timer = timer or Timer()
    with timer("preprocess"):
        raw = preprocess(raw_dn, black, white)
    with timer("analyze"):
        info = info or analyze(raw, wb=wb, gain=gain)
    with timer("white_balance"):
        wb = info["wb_gains"] if use_wb else (1.0, 1.0)
        raw_wb = apply_wb(raw, wb)
    with timer("demosaic"):
        rgb = demosaic(np.clip(raw_wb, 0, 1), demosaic_method)
    with timer("ccm"):
        if use_ccm:
            rgb = apply_ccm(rgb, CAM2RGB_NOMINAL if ccm is None else ccm)
    with timer("gain_tonemap"):
        lin = rgb * (info["gain"] if use_gain else 1.0)
        out = tone_map(lin, knee)
    return out, lin, info, timer


def classical_isp(raw_dn: np.ndarray, use_wb=True, use_ccm=True, use_denoise=True,
                  use_sharpen=True, use_gain=True, denoise_method="nlm", demosaic_method="ea", **fe):
    timer = Timer()
    out, _, info, timer = front_end(raw_dn, timer=timer, use_wb=use_wb, use_ccm=use_ccm,
                                    use_gain=use_gain, demosaic_method=demosaic_method, **fe)
    with timer("denoise"):
        if use_denoise:
            out = denoise(out, info["denoise_h"], denoise_method)
    with timer("sharpen"):
        if use_sharpen:
            out = sharpen(out, info["sharpen"])
    return np.clip(out, 0, 1), info, timer.t
