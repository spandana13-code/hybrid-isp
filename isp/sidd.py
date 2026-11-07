"""Real smartphone RAW from SIDD (Abdelhamed, Lin & Brown, CVPR 2018), via the SIDD-CC release.

* Inputs  : real noisy Bayer RAW from 5 phones (Galaxy S6, Pixel, Nexus 6, iPhone 7, LG G4),
            black-level subtracted and normalised, with DNG metadata (AsShotNeutral, ColorMatrix2,
            CFA pattern).
* Targets : SIDD's clean sRGB ground truth (mean of ~150 aligned frames, rendered by the
            dataset's camera pipeline), pixel-aligned with the RAW.

Every Bayer layout is converted to RGGB with lossless flips (and flipped back afterwards), so
the whole ISP only ever sees RGGB.
"""
from __future__ import annotations

import glob
import json
import os

import cv2
import numpy as np
import scipy.io

from .raw_sim import BLACK_LEVEL, WHITE_LEVEL, srgb_to_linear, pack_rggb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIDD_DIR = os.environ.get("SIDD_DIR", os.path.join(os.path.dirname(ROOT), "sidd"))
CALIB_PATH = os.path.join(ROOT, "isp", "camera_calib.json")

CFA = {"S6": "grbg", "GP": "bggr", "N6": "bggr", "G4": "bggr", "IP": "rggb"}
CAMERA_BY_SHAPE = {"3000x5328": "S6", "3036x4048": "GP", "3120x4208": "N6",
                   "2988x5312": "G4", "3024x4032": "IP"}
SRGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                     [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]], np.float32)


# ----------------------------------------------------------------------------- CFA / metadata
def to_rggb(a: np.ndarray, cfa: str) -> np.ndarray:
    """Flip an image (Bayer or aligned RGB) so that the Bayer pattern reads RGGB.
    Flips are their own inverse, so the same call maps RGGB results back."""
    cfa = cfa.lower()
    if cfa == "rggb":
        return a
    if cfa == "grbg":
        return a[:, ::-1]
    if cfa == "gbrg":
        return a[::-1]
    if cfa == "bggr":
        return a[::-1, ::-1]
    raise ValueError(f"unsupported CFA {cfa}")


def ccm_from_colormatrix(cm) -> np.ndarray:
    """DNG ColorMatrix (XYZ->camera) -> camera->linear-sRGB matrix, rows normalised to keep grey."""
    a = np.asarray(cm, np.float32).reshape(3, 3) @ SRGB2XYZ
    a = a / a.sum(1, keepdims=True)
    return np.linalg.inv(a).astype(np.float32)


def wb_from_neutral(n) -> tuple[float, float]:
    n = np.asarray(n, np.float64).ravel()
    return float(n[1] / n[0]), float(n[1] / n[2])


def norm_to_dn(x: np.ndarray) -> np.ndarray:
    """Normalised, black-subtracted RAW -> the ISP's 10-bit DN convention (black 64, white 1023).
    Kept as float so low-light precision is not lost; sub-black noise down to -64 DN survives."""
    return np.clip(BLACK_LEVEL + x * (WHITE_LEVEL - BLACK_LEVEL), 0, WHITE_LEVEL).astype(np.float32)


def camera_of(name: str) -> str:
    return name.split("_")[2]


def load_calibration() -> dict:
    if os.path.exists(CALIB_PATH):
        return json.load(open(CALIB_PATH))
    return {}


def camera_ccm(cam: str, calib: dict | None = None) -> np.ndarray | None:
    calib = calib if calib is not None else load_calibration()
    return np.asarray(calib[cam]["ccm"], np.float32) if cam in calib else None


def generic_calibration() -> np.ndarray:
    """Average of the per-phone calibration matrices, for cameras not in SIDD (user DNGs)."""
    calib = load_calibration()
    if not calib:
        return np.eye(3, dtype=np.float32)
    return np.mean([np.asarray(v["M"], np.float32) for v in calib.values()], 0)


def fe_settings(ccm=None, wb=None) -> dict:
    """Front-end settings for real camera RAW: the camera already chose the exposure (gain 1)
    and the ground truth uses the plain sRGB curve (knee 1)."""
    return dict(ccm=ccm, wb=wb, gain=1.0, knee=1.0)


# ----------------------------------------------------------------------------- train frames
def train_names() -> list[str]:
    return sorted(os.path.basename(f)[:-len("_NOISY.MAT")] for f in glob.glob(os.path.join(SIDD_DIR, "raw", "*_NOISY.MAT")))


def split_names(names: list[str], val_frac=0.13):
    """Split by *capture instance*, stratified per camera. SIDD has only 10 physical scenes,
    re-shot by every phone at many ISO / illuminant settings, so (like the official SIDD split)
    instances - not scenes - are the unit; each instance is a separate capture."""
    rng = np.random.default_rng(42)
    va = []
    for cam in sorted({camera_of(n) for n in names}):
        group = [n for n in names if camera_of(n) == cam]
        k = max(1, int(round(len(group) * val_frac)))
        va += list(rng.choice(group, k, replace=False))
    tr = [n for n in names if n not in va]
    return tr, sorted(va)


def load_frame(name: str) -> dict:
    meta = scipy.io.loadmat(os.path.join(SIDD_DIR, "raw", f"{name}_META.MAT"))
    raw = scipy.io.loadmat(os.path.join(SIDD_DIR, "raw", f"{name}_NOISY.MAT"))["data"].astype(np.float32)
    gt = cv2.cvtColor(cv2.imread(os.path.join(SIDD_DIR, "srgb", f"{name}_GT.PNG")), cv2.COLOR_BGR2RGB)
    cfa = str(meta["cfa_str"][0])
    h, w = raw.shape[0] // 2 * 2, raw.shape[1] // 2 * 2
    raw, gt = to_rggb(raw[:h, :w], cfa), to_rggb(gt[:h, :w], cfa)
    return dict(name=name, cam=camera_of(name), raw=np.ascontiguousarray(raw),
                gt=np.ascontiguousarray(gt), wb=wb_from_neutral(meta["AsShotNeutral"]),
                ccm_meta=ccm_from_colormatrix(meta["ColorMatrix2"]), cfa=cfa)


def fit_calibration(names: list[str]) -> dict:
    """Per-camera 3x3 colour calibration (least squares, linear light) on training frames only:
    final CCM = M_cam @ CCM_metadata. This is the factory colour calibration step of a real ISP."""
    acc = {}
    for n in names:
        f = load_frame(n)
        pk = pack_rggb(f["raw"])
        cam_rgb = np.stack([pk[..., 0] * f["wb"][0], (pk[..., 1] + pk[..., 2]) / 2, pk[..., 3] * f["wb"][1]], -1)
        small = lambda a: cv2.resize(a, (a.shape[1] // 8, a.shape[0] // 8), interpolation=cv2.INTER_AREA)
        lin = small(cam_rgb) @ f["ccm_meta"].T
        gt = srgb_to_linear(f["gt"].astype(np.float32) / 255)
        gt = cv2.resize(gt, (lin.shape[1], lin.shape[0]), interpolation=cv2.INTER_AREA)
        ok = (gt.min(2) > 0.01) & (gt.max(2) < 0.85) & (lin.max(2) < 0.95) & (lin.min(2) > 0)
        a, b = acc.setdefault(f["cam"], ([], []))
        a.append(lin[ok]); b.append(gt[ok])
        acc[f["cam"]] = (a, b)
    calib = {}
    for cam, (a, b) in acc.items():
        X, Y = np.concatenate(a), np.concatenate(b)
        M, *_ = np.linalg.lstsq(X, Y, rcond=None)          # Y ~= X @ M
        meta_ccm = ccm_from_colormatrix(scipy.io.loadmat(os.path.join(
            SIDD_DIR, "raw", f"{[n for n in names if camera_of(n) == cam][0]}_META.MAT"))["ColorMatrix2"])
        calib[cam] = dict(M=M.T.tolist(), ccm=(M.T @ meta_ccm).tolist(), n_pixels=int(len(X)),
                          cfa=CFA[cam])
    json.dump(calib, open(CALIB_PATH, "w"), indent=1)
    return calib


def build_sidd_bank(names: list[str], crops_per_frame: int, crop_sizes=(256,), patch: int = 128,
                    subcrops: int = 2, wb_meta_prob: float = 0.5, seed: int = 0) -> dict:
    """Each RAW crop (256, 512 or 768 px, so the global head sees thumbnails at several scene
    scales) is treated as a complete frame, like a SIDD benchmark block or an app upload: the
    adaptive controller analyses it, white balance is estimated from it or, with probability
    `wb_meta_prob`, taken from metadata (as on a phone, flagged to the networks), the front-end
    renders it, then `subcrops` random 128x128 patches are stored with the crop's thumbnail."""
    from .data import hybrid_inputs, learned_inputs
    rng = np.random.default_rng(seed)
    calib = load_calibration()
    keys = ["packed", "nm_l", "front", "nm_h", "thumb", "target"]
    bank = {k: [] for k in keys}
    f16 = lambda a: np.ascontiguousarray(a, dtype=np.float16)
    for i, n in enumerate(names):
        f = load_frame(n)
        H, W = f["raw"].shape
        ccm = camera_ccm(f["cam"], calib)
        for _ in range(crops_per_frame):
            crop = int(rng.choice(crop_sizes))
            y = int(rng.integers(0, (H - crop) // 2)) * 2
            x = int(rng.integers(0, (W - crop) // 2)) * 2
            raw = norm_to_dn(f["raw"][y:y + crop, x:x + crop])
            gt = f["gt"][y:y + crop, x:x + crop].astype(np.float32) / 255
            wb = f["wb"] if rng.random() < wb_meta_prob else None
            fe = fe_settings(ccm, wb)
            front, nm_h, thumb, info, _ = hybrid_inputs(raw, **fe)
            packed, nm_l, _ = learned_inputs(raw, info)
            for _ in range(subcrops):
                py = int(rng.integers(0, (crop - patch) // 2 + 1)) * 2
                px = int(rng.integers(0, (crop - patch) // 2 + 1)) * 2
                bank["target"].append(f16(gt[py:py + patch, px:px + patch]))
                bank["front"].append(f16(front[py:py + patch, px:px + patch]))
                bank["nm_h"].append(f16(nm_h[py:py + patch, px:px + patch]))
                bank["thumb"].append(f16(thumb))
                bank["packed"].append(f16(packed[py // 2:(py + patch) // 2, px // 2:(px + patch) // 2]))
                bank["nm_l"].append(f16(nm_l[py // 2:(py + patch) // 2, px // 2:(px + patch) // 2]))
        print(f"  bank: {i + 1}/{len(names)} {n}", flush=True)
    return {k: np.stack(bank.pop(k)) for k in list(bank)}


# ----------------------------------------------------------------------------- official test set
def load_test_blocks() -> list[dict]:
    """The official SIDD benchmark: 29 scenes x 32 blocks of 256x256 (noisy RAW + clean sRGB).
    The camera (hence CFA and calibrated CCM) is identified from the source resolution; no
    per-image metadata is published for the test set, so white balance must be estimated."""
    t = os.path.join(SIDD_DIR, "test")
    raw = scipy.io.loadmat(os.path.join(t, "BenchmarkNoisyBlocksRaw.mat"))["data"]
    gt = scipy.io.loadmat(os.path.join(t, "BenchmarkGtBlocksSrgb.mat"))["data"]
    info = [l.strip().split("\t") for l in open(os.path.join(SIDD_DIR, "test_scenes_info.txt")).readlines()[1:]]
    calib = load_calibration()
    out = []
    for i, (sid, img, shape) in enumerate(info):
        cam = CAMERA_BY_SHAPE[shape]
        for j in range(raw.shape[1]):
            out.append(dict(scene=sid, block=j, cam=cam, cfa=CFA[cam],
                            raw=np.ascontiguousarray(to_rggb(raw[i, j].astype(np.float32), CFA[cam])),
                            gt=np.ascontiguousarray(to_rggb(gt[i, j], CFA[cam])).astype(np.float32) / 255,
                            ccm=camera_ccm(cam, calib)))
    return out
