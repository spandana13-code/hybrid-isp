"""Model inputs, procedural scenes and the training-patch bank."""
from __future__ import annotations

import glob
import os
import numpy as np
import cv2

from .raw_sim import (Degradation, sample_degradation, simulate_raw, even_crop, pack_rggb,
                      srgb_to_linear)
from .classical import preprocess, analyze, front_end, Timer

NOISE_SCALE = 5.0


def load_rgb(path: str) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"cannot read image {path}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def hybrid_inputs(raw_dn: np.ndarray, timer: Timer | None = None, **fe_kwargs):
    """Classical front-end output + per-pixel output-domain noise-sigma map."""
    x, lin, info, timer = front_end(raw_dn, timer=timer, **fe_kwargs)
    with timer("noise_map"):
        gain = info["gain"]
        raw_equiv = np.clip(lin.mean(2) / gain, 0, 1)                   # approx. raw level per pixel
        sig_lin = np.sqrt(info["shot"] * raw_equiv + info["read"]) * gain * 1.8  # 1.8 ~ CCM noise gain
        slope = np.where(lin.mean(2) <= 0.0031308, 12.92,
                         (1.055 / 2.4) * np.power(np.clip(lin.mean(2), 1e-4, 1), 1 / 2.4 - 1))
        nm = np.clip(sig_lin * slope, 0, 1)[..., None].astype(np.float32) * NOISE_SCALE
        thumb = make_thumb(x, nm)
    return x.astype(np.float32), nm, thumb, info, timer


def wb_flag(info: dict) -> float:
    """1 if the white balance came from metadata (trustworthy), 0 if it was estimated.
    Lets the networks learn when to correct the illuminant and when to leave it alone."""
    return 1.0 if info.get("wb_source") == "metadata" else 0.0


THUMB = 64


def make_thumb(x: np.ndarray, nm: np.ndarray, flag: float = 0.0) -> np.ndarray:
    """Fixed-size whole-frame thumbnail (area-averaged) for the global colour/exposure head:
    RGB + noise map + constant WB-source flag channel (5 channels)."""
    t = np.concatenate([x, nm], 2).astype(np.float32)
    return cv2.resize(t, (THUMB, THUMB), interpolation=cv2.INTER_AREA)


def learned_inputs(raw_dn: np.ndarray, info: dict | None = None, black=None, white=None, wb=None, gain=None):
    """Packed RGGB with the same scalar white-balance gains and exposure gain the controller
    chose (metadata or estimated) - the end-to-end net learns demosaic, colour, tone, denoise."""
    raw = preprocess(raw_dn) if black is None else preprocess(raw_dn, black, white)
    info = info or analyze(raw, wb=wb, gain=gain)
    pk = pack_rggb(raw) * np.array([info["wb_gains"][0], 1, 1, info["wb_gains"][1]], np.float32) * info["gain"]
    sig = np.sqrt(info["shot"] * pack_rggb(raw).mean(2) + info["read"]) * info["gain"]
    nm = np.clip(sig, 0, 1)[..., None].astype(np.float32) * NOISE_SCALE
    return np.clip(pk, 0, 1.5).astype(np.float32), nm, info


# ----------------------------------------------------------------------------- procedural scenes
def procedural_scene(rng: np.random.Generator, size=384) -> np.ndarray:
    """Random natural-ish synthetic scene: gradients, shapes, textures, thin lines, text."""
    img = np.zeros((size, size, 3), np.float32)
    c0, c1 = rng.uniform(0, 1, 3), rng.uniform(0, 1, 3)
    t = np.linspace(0, 1, size)[None, :, None]
    ang = rng.uniform(0, 1)
    tt = (t * ang + np.linspace(0, 1, size)[:, None, None] * (1 - ang))
    img[:] = c0 * (1 - tt) + c1 * tt
    for _ in range(rng.integers(8, 25)):
        col = tuple(float(v) for v in rng.uniform(0, 1, 3))
        kind = rng.integers(0, 4)
        p = tuple(int(v) for v in rng.integers(0, size, 2))
        if kind == 0:
            cv2.circle(img, p, int(rng.integers(5, size // 4)), col, -1, cv2.LINE_AA)
        elif kind == 1:
            q = tuple(int(v) for v in rng.integers(0, size, 2))
            cv2.rectangle(img, p, q, col, -1)
        elif kind == 2:
            q = tuple(int(v) for v in rng.integers(0, size, 2))
            cv2.line(img, p, q, col, int(rng.integers(1, 4)), cv2.LINE_AA)
        else:
            mask = np.zeros((size, size), np.uint8)
            cv2.putText(mask, "ISP%d" % rng.integers(100), p, cv2.FONT_HERSHEY_SIMPLEX,
                        float(rng.uniform(0.4, 2)), 255, int(rng.integers(1, 3)), cv2.LINE_AA)
            a = (mask.astype(np.float32) / 255)[..., None]
            img = img * (1 - a) + np.array(col, np.float32) * a
    # texture
    tex = cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32), (0, 0), rng.uniform(0.6, 3))
    img += 0.05 * tex[..., None] * rng.uniform(0, 1)
    return np.clip(img, 0, 1)


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    s = rng.uniform(0.5, 1.0)
    if min(img.shape[:2]) * s >= 160:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        img = img[:, ::-1]
    if rng.random() < 0.5:
        img = img[::-1]
    if rng.random() < 0.5:
        img = np.rot90(img)
    return np.ascontiguousarray(img)


def build_bank(image_dir: str, n_patches: int, patch: int = 128, crops_per_frame: int = 8,
               procedural_frac: float = 0.3, seed: int = 0) -> dict:
    """Pre-computes (raw-packed, hybrid-input, noise maps, target) patch tuples."""
    rng = np.random.default_rng(seed)
    files = sorted(glob.glob(os.path.join(image_dir, "*")))
    imgs = [load_rgb(f) for f in files]
    keys = ["packed", "nm_l", "front", "nm_h", "thumb", "target"]
    bank = {k: [] for k in keys}
    while len(bank["target"]) < n_patches:
        if rng.random() < procedural_frac or not imgs:
            img = procedural_scene(rng)
        else:
            img = augment(imgs[rng.integers(len(imgs))], rng)
        img = even_crop(img, 4)
        h, w = img.shape[:2]
        if h < patch or w < patch:
            continue
        deg = sample_degradation(rng)
        raw = simulate_raw(img, deg)
        front, nm_h, thumb, info, _ = hybrid_inputs(raw)
        packed, nm_l, _ = learned_inputs(raw, info)
        for _ in range(crops_per_frame):
            y = int(rng.integers(0, (h - patch) // 2 + 1)) * 2
            x = int(rng.integers(0, (w - patch) // 2 + 1)) * 2
            f16 = lambda a: np.ascontiguousarray(a, dtype=np.float16)
            bank["target"].append(f16(img[y:y + patch, x:x + patch]))
            bank["front"].append(f16(front[y:y + patch, x:x + patch]))
            bank["nm_h"].append(f16(nm_h[y:y + patch, x:x + patch]))
            bank["thumb"].append(f16(thumb))
            bank["packed"].append(f16(packed[y // 2:(y + patch) // 2, x // 2:(x + patch) // 2]))
            bank["nm_l"].append(f16(nm_l[y // 2:(y + patch) // 2, x // 2:(x + patch) // 2]))
    out = {}
    for k in list(bank):
        out[k] = np.stack(bank.pop(k)[:n_patches])
    return out
