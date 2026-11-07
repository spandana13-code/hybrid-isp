"""Figure of the hybrid pipeline's intermediate representations -> docs/intermediates.jpg

    python scripts/make_intermediates.py

Columns: visualised Bayer RAW | classical front-end output (WB + demosaic + CCM + gamma, the
network's input) | measured per-pixel noise map (network conditioning) | learned residual
(network output - its input, amplified) | hybrid output | ground truth.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isp.engine import ISPEngine, _pad          # noqa: E402
from isp.data import hybrid_inputs               # noqa: E402
from isp.io import load_any                      # noqa: E402
from isp.sidd import to_rggb                     # noqa: E402
from isp.raw_sim import pack_rggb, BLACK_LEVEL, WHITE_LEVEL  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = ["sidd_test_N6_0096_b07.npz",   # dark, high noise
            "sidd_test_S6_0102_b05.npz",   # well lit
            "sidd_test_GP_0020_b09.npz"]   # Google Pixel
TITLES = ["RAW (visualised)", "Classical front-end", "Noise map", "Learned residual x4",
          "Hybrid output", "Ground truth"]


def vis_raw(raw):
    p = pack_rggb((raw.astype(np.float32) - BLACK_LEVEL) / (WHITE_LEVEL - BLACK_LEVEL))
    rgb = np.stack([p[..., 0], (p[..., 1] + p[..., 2]) / 2, p[..., 3]], -1)
    rgb = np.clip(rgb * 4, 0, 1) ** (1 / 2.2)                     # x4 so dark RAW is visible
    return cv2.resize(rgb, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST)


def main():
    eng = ISPEngine("int8")
    rows = []
    for name in EXAMPLES:
        d = load_any(os.path.join(ROOT, "examples", name))
        raw, fe, gt, cfa = d["raw"], d["fe"], d["gt"], d["cfa"]
        x, nm, thumb, info, _ = hybrid_inputs(raw, **fe)
        xp, h, w = _pad(x, 8)
        np_, _, _ = _pad(nm, 8)
        net = np.clip(eng.models["hybrid"](xp, np_, thumb)[:h, :w], 0, 1)
        out, _, _ = eng.hybrid(raw, **fe)
        nmv = cv2.applyColorMap((np.clip(nm[..., 0] / max(nm.max(), 1e-6), 0, 1) * 255).astype(np.uint8),
                                cv2.COLORMAP_INFERNO)[..., ::-1] / 255.0
        resid = np.clip(0.5 + 4 * (net - x), 0, 1)
        tiles = [vis_raw(raw), x, nmv, resid, out, gt]
        rows.append(np.concatenate([to_rggb(np.clip(t, 0, 1).astype(np.float32), cfa) for t in tiles], 1))
    grid = np.concatenate(rows, 0)
    th = 28
    tw = grid.shape[1] // len(TITLES)
    header = np.full((th, grid.shape[1], 3), 255, np.uint8)
    for i, t in enumerate(TITLES):
        cv2.putText(header, t, (i * tw + 6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    fig = np.concatenate([header, (grid * 255 + 0.5).astype(np.uint8)], 0)
    out = os.path.join(ROOT, "docs", "intermediates.jpg")
    cv2.imwrite(out, cv2.cvtColor(fig, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("wrote", out, fig.shape)


if __name__ == "__main__":
    main()
