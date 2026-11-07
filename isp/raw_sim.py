"""Physically-motivated RAW simulation ("unprocessing", after Brooks et al., CVPR 2019).

Turns a clean sRGB image into a degraded 10-bit RGGB Bayer RAW frame with:
  * inverse gamma (sRGB -> linear)
  * inverse colour correction (sRGB -> camera RGB) with a per-shot perturbation (colour distortion)
  * inverse white balance with a random illuminant (white-balance error)
  * exposure scaling (varying illumination / low light)
  * mosaicking to RGGB
  * heteroscedastic shot + read noise, black level, 10-bit quantisation and clipping

The clean sRGB image is the ground truth target, so every pipeline can be scored with
full-reference metrics (PSNR / SSIM / Delta E / perceptual).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
import numpy as np

BLACK_LEVEL = 64.0
WHITE_LEVEL = 1023.0

# Nominal sensor cam->sRGB colour correction matrix (rows sum to 1). This is the
# "factory calibration" every pipeline is allowed to know about.
CAM2RGB_NOMINAL = np.array([
    [1.72, -0.58, -0.14],
    [-0.24, 1.56, -0.32],
    [0.02, -0.55, 1.53],
], dtype=np.float32)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


@dataclass
class Degradation:
    exposure: float = 1.0          # scene brightness multiplier (lower = darker / low light)
    r_gain: float = 2.0            # true white-balance gains of the illuminant
    b_gain: float = 1.6
    shot_noise: float = 0.002      # variance coefficient (Poisson part)
    read_noise: float = 1e-5       # variance (Gaussian part)
    ccm_jitter: float = 0.0        # magnitude of unknown per-shot CCM error
    seed: int = 0
    cam2rgb: list = field(default_factory=lambda: CAM2RGB_NOMINAL.tolist())

    def to_dict(self):
        return asdict(self)


# Named presets used by the app and the condition-wise evaluation.
CONDITIONS = {
    "normal":     dict(exposure=(0.7, 1.0),  shot=(1e-4, 1e-3), wb_strength=0.3),
    "low_light":  dict(exposure=(0.08, 0.2), shot=(1e-3, 4e-3), wb_strength=0.3),
    "high_noise": dict(exposure=(0.4, 0.8),  shot=(6e-3, 1.5e-2), wb_strength=0.3),
    "wb_cast":    dict(exposure=(0.5, 1.0),  shot=(5e-4, 2e-3), wb_strength=1.0),
    "extreme":    dict(exposure=(0.05, 0.1), shot=(8e-3, 1.5e-2), wb_strength=1.0),
}


def sample_degradation(rng: np.random.Generator, condition: str | None = None) -> Degradation:
    """Random degradation. `condition=None` samples the full training distribution."""
    if condition is None:
        exposure = float(np.exp(rng.uniform(np.log(0.05), np.log(1.0))))
        log_shot = rng.uniform(np.log(1e-4), np.log(1.5e-2))
        wb_strength = rng.uniform(0.0, 1.0)
    else:
        c = CONDITIONS[condition]
        exposure = float(rng.uniform(*c["exposure"]))
        log_shot = rng.uniform(np.log(c["shot"][0]), np.log(c["shot"][1]))
        wb_strength = c["wb_strength"]
    shot = float(np.exp(log_shot))
    # read-noise is correlated with shot-noise (same ISO), as in Brooks et al.
    read = float(np.exp(2.18 * log_shot + 1.20 + rng.normal(0, 0.26)))
    # illuminant: neutral daylight gains (2.0, 1.6) pushed towards tungsten / shade
    t = rng.uniform(-1, 1) * wb_strength
    r_gain = float(2.0 + 0.9 * t + rng.normal(0, 0.08))
    b_gain = float(1.6 - 0.6 * t + rng.normal(0, 0.08))
    jitter = float(rng.uniform(0.0, 0.12))
    return Degradation(exposure=exposure, r_gain=r_gain, b_gain=b_gain, shot_noise=shot,
                       read_noise=read, ccm_jitter=jitter, seed=int(rng.integers(1 << 31)))


def mosaic_rggb(rgb: np.ndarray) -> np.ndarray:
    h, w, _ = rgb.shape
    raw = np.empty((h, w), np.float32)
    raw[0::2, 0::2] = rgb[0::2, 0::2, 0]
    raw[0::2, 1::2] = rgb[0::2, 1::2, 1]
    raw[1::2, 0::2] = rgb[1::2, 0::2, 1]
    raw[1::2, 1::2] = rgb[1::2, 1::2, 2]
    return raw


def pack_rggb(raw: np.ndarray) -> np.ndarray:
    """HxW Bayer -> (H/2)x(W/2)x4 [R, Gr, Gb, B]."""
    return np.stack([raw[0::2, 0::2], raw[0::2, 1::2], raw[1::2, 0::2], raw[1::2, 1::2]], -1)


def simulate_raw(srgb: np.ndarray, deg: Degradation) -> np.ndarray:
    """srgb float32 HxWx3 in [0,1] (H, W even) -> uint16 10-bit Bayer RGGB with black level."""
    rng = np.random.default_rng(deg.seed)
    lin = srgb_to_linear(srgb)
    cam2rgb = np.asarray(deg.cam2rgb, np.float32)
    if deg.ccm_jitter > 0:
        e = rng.normal(0, deg.ccm_jitter, (3, 3)).astype(np.float32)
        e -= e.mean(1, keepdims=True)                   # keep rows summing to 1 (greys stay grey)
        cam2rgb = cam2rgb + e
    rgb2cam = np.linalg.inv(cam2rgb)
    cam = np.clip(lin @ rgb2cam.T, 0, 1)
    cam = cam / np.array([deg.r_gain, 1.0, deg.b_gain], np.float32)  # illuminant
    cam = cam * deg.exposure
    raw = mosaic_rggb(cam)
    var = np.maximum(raw * deg.shot_noise + deg.read_noise, 0)
    raw = raw + rng.normal(0, 1, raw.shape).astype(np.float32) * np.sqrt(var)
    dn = raw * (WHITE_LEVEL - BLACK_LEVEL) + BLACK_LEVEL
    return np.clip(np.round(dn), 0, WHITE_LEVEL).astype(np.uint16)


def even_crop(img: np.ndarray, mult: int = 2) -> np.ndarray:
    h, w = img.shape[:2]
    return img[: h - h % mult, : w - w % mult]
