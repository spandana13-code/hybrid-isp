"""Input handling: real RAW (DNG/CR2/NEF/ARW... via rawpy), .npy Bayer, or sRGB photos to degrade."""
from __future__ import annotations

import os
import numpy as np
import cv2

from .raw_sim import BLACK_LEVEL, WHITE_LEVEL, even_crop, pack_rggb

RAW_EXT = {".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf", ".rw2", ".pef", ".srw"}
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MAX_PIXELS = 1_200_000          # keeps the hosted demo comfortably under ~10 s


class InputError(ValueError):
    pass


def _srgb2xyz():
    return np.array([[0.4124564, 0.3575761, 0.1804375],
                     [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]], np.float32)


def unpack_rggb(p: np.ndarray) -> np.ndarray:
    h, w, _ = p.shape
    raw = np.empty((2 * h, 2 * w), p.dtype)
    raw[0::2, 0::2], raw[0::2, 1::2], raw[1::2, 0::2], raw[1::2, 1::2] = (p[..., i] for i in range(4))
    return raw


def bin_bayer(raw: np.ndarray, max_pixels=MAX_PIXELS) -> np.ndarray:
    """Downscale an RGGB mosaic by averaging 2x2 same-colour samples (keeps the Bayer layout)."""
    while raw.shape[0] * raw.shape[1] > max_pixels:
        p = pack_rggb(even_crop(raw, 4)).astype(np.float32)
        h, w, _ = p.shape
        p = p.reshape(h // 2, 2, w // 2, 2, 4).mean((1, 3))
        raw = unpack_rggb(p)
    return raw


def load_camera_raw(path: str) -> dict:
    """Reads a camera RAW with rawpy and converts it to the internal 10-bit RGGB convention."""
    import rawpy
    with rawpy.imread(path) as r:
        if r.raw_type != rawpy.RawType.Flat or r.num_colors != 3:
            raise InputError("Only standard Bayer RAW files are supported (not Foveon/X-Trans/linear DNG).")
        raw = r.raw_image_visible.astype(np.float32)
        pattern = r.raw_pattern[:2, :2]
        black = float(np.mean(r.black_level_per_channel))
        white = float(r.white_level)
        xyz2cam = np.asarray(r.rgb_xyz_matrix[:3, :3], np.float32)
        desc = r.color_desc.decode()
        cwb = np.asarray(r.camera_whitebalance, np.float64)
    # shift so that the top-left 2x2 block reads R G / G B
    for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
        if desc[pattern[dy, dx]] == "R" and desc[pattern[1 - dy, 1 - dx]] == "B":
            raw = raw[dy:, dx:]
            break
    else:
        raise InputError("Unrecognised CFA pattern.")
    raw = even_crop(raw, 2)
    norm = np.clip((raw - black) / (white - black), 0, 1)
    raw = bin_bayer(norm)
    dn = np.round(raw * (WHITE_LEVEL - BLACK_LEVEL) + BLACK_LEVEL).astype(np.uint16)
    from .sidd import ccm_from_colormatrix, fe_settings, generic_calibration
    ccm = None
    if np.abs(xyz2cam).sum() > 0:
        # DNG colour matrix + the average rendering calibration learned on SIDD phones
        ccm = (generic_calibration() @ ccm_from_colormatrix(xyz2cam)).astype(np.float32)
    wb = None
    if cwb[1] > 0 and cwb[0] > 0 and cwb[2] > 0:
        wb = (float(cwb[0] / cwb[1]), float(cwb[2] / cwb[1]))      # as-shot white balance
    return dict(raw=dn, gt=None, fe=fe_settings(ccm, wb), cfa="rggb",
                source="camera RAW (" + os.path.splitext(path)[1][1:].upper() + "), as-shot WB" + ("" if wb else " missing → AWB"))


def load_npz_frame(path: str) -> dict:
    """Packaged real-RAW example (e.g. a SIDD benchmark block): normalised RGGB raw,
    aligned ground-truth sRGB, camera id and original CFA (to restore orientation)."""
    from .sidd import norm_to_dn, fe_settings, camera_ccm, CFA
    z = np.load(path, allow_pickle=False)
    if "raw" not in z:
        raise InputError(".npz must contain a 'raw' array (normalised RGGB Bayer).")
    raw = z["raw"].astype(np.float32)
    if raw.ndim != 2 or min(raw.shape) < 32:
        raise InputError(f"'raw' must be a 2-D Bayer mosaic of at least 32x32, got {raw.shape}")
    raw = even_crop(raw, 2)
    cam = str(z["cam"]) if "cam" in z else ""
    gt = z["gt"].astype(np.float32) / 255 if "gt" in z else None
    if gt is not None:
        gt = even_crop(gt, 2)[: raw.shape[0], : raw.shape[1]]
    label = {"GP": "Google Pixel", "IP": "iPhone 7", "N6": "Nexus 6", "G4": "LG G4", "S6": "Galaxy S6"}.get(cam, "camera")
    wb = tuple(float(v) for v in z["wb"]) if "wb" in z else None
    return dict(raw=norm_to_dn(raw), gt=gt, fe=fe_settings(camera_ccm(cam) if cam else None, wb),
                cfa=str(z["cfa"]) if "cfa" in z else "rggb",
                source=f"real {label} RAW, " + (str(z["desc"]) if "desc" in z else "SIDD") +
                       (", as-shot WB" if wb else ", AWB estimated"))


def load_npy_bayer(path: str) -> dict:
    a = np.load(path)
    if a.ndim != 2:
        raise InputError(f".npy must be a 2-D RGGB Bayer mosaic, got shape {a.shape}")
    a = even_crop(a.astype(np.float32), 2)
    if a.max() <= 1.0:
        a = a * (WHITE_LEVEL - BLACK_LEVEL) + BLACK_LEVEL
    elif a.max() > WHITE_LEVEL:
        a = (a / a.max()) * (WHITE_LEVEL - BLACK_LEVEL) + BLACK_LEVEL
    a = bin_bayer((a - BLACK_LEVEL) / (WHITE_LEVEL - BLACK_LEVEL))
    from .sidd import fe_settings
    return dict(raw=np.round(a * (WHITE_LEVEL - BLACK_LEVEL) + BLACK_LEVEL).astype(np.uint16),
                gt=None, fe=fe_settings(None, None), cfa="rggb", source="Bayer .npy (RGGB, 10-bit assumed)")


def load_srgb(path_or_array) -> np.ndarray:
    if isinstance(path_or_array, np.ndarray):
        im = path_or_array
        if im.dtype != np.float32:
            im = im.astype(np.float32) / (255.0 if im.max() > 1 else 1.0)
        if im.ndim == 2:
            im = np.repeat(im[..., None], 3, 2)
        im = im[..., :3]
    else:
        im = cv2.imread(path_or_array, cv2.IMREAD_COLOR)
        if im is None:
            raise InputError("Could not decode the image file.")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = im.shape[:2]
    if min(h, w) < 32:
        raise InputError(f"Image too small ({w}x{h}); need at least 32x32.")
    s = min(1.0, np.sqrt(MAX_PIXELS / (h * w)))
    if s < 1:
        im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return even_crop(im, 8).astype(np.float32)


def load_any(path: str) -> dict:
    ext = os.path.splitext(path)[1].lower()
    if ext in RAW_EXT:
        return load_camera_raw(path)
    if ext == ".npy":
        return load_npy_bayer(path)
    if ext == ".npz":
        return load_npz_frame(path)
    if ext in IMG_EXT:
        return dict(srgb=load_srgb(path), source="sRGB photo -> simulated sensor RAW")
    raise InputError(f"Unsupported file type '{ext}'. Use a camera RAW/DNG, a .npz/.npy Bayer array, or a JPG/PNG.")
