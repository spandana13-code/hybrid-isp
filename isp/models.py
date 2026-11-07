"""Lightweight learned components.

LiteUNet   - a 3-level U-Net body (depthwise-separable blocks) shared by both models.
LearnedISP - pure end-to-end RAW->sRGB baseline (PyNET / U-Net ISP style): packed RGGB in,
             sRGB out at full resolution via PixelShuffle. Must learn demosaic, WB, CCM,
             tone and denoise on its own.
HybridNet  - the learned stage of the hybrid ISP. Takes the classical front-end output
             (demosaiced, white-balanced, colour-corrected, gamma-encoded) plus a per-pixel
             noise map and predicts
               * a local residual (denoise / detail / artefact removal), and
               * a global 3x3 colour matrix + exposure correction applied in *linear* light
                 (learned colour transformation that fixes residual WB / CCM / AE errors).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def srgb_to_linear_t(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).clamp_min(1e-6) ** 2.4)


def linear_to_srgb_t(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp_min(1e-6) ** (1 / 2.4) - 0.055)


class DSBlock(nn.Module):
    """Depthwise-separable residual block: dw3x3 -> pw1x1 -> act -> dw3x3 -> pw1x1."""

    def __init__(self, cin, cout):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cin, 3, padding=1, groups=cin), nn.Conv2d(cin, cout, 1), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, groups=cout), nn.Conv2d(cout, cout, 1),
        )
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class LiteUNet(nn.Module):
    def __init__(self, cin, cout, ch=(24, 48, 96)):
        super().__init__()
        c1, c2, c3 = ch
        self.inc = nn.Sequential(nn.Conv2d(cin, c1, 3, padding=1), nn.ReLU(inplace=True), DSBlock(c1, c1))
        self.d1 = nn.Sequential(nn.Conv2d(c1, c2, 2, stride=2), DSBlock(c2, c2))
        self.d2 = nn.Sequential(nn.Conv2d(c2, c3, 2, stride=2), DSBlock(c3, c3), DSBlock(c3, c3))
        self.u2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.b2 = DSBlock(2 * c2, c2)
        self.u1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.b1 = DSBlock(2 * c1, c1)
        self.out = nn.Conv2d(c1, cout, 3, padding=1)
        self.bottleneck_ch = c3

    def forward(self, x):
        e1 = self.inc(x)
        e2 = self.d1(e1)
        e3 = self.d2(e2)
        d2 = self.b2(torch.cat([self.u2(e3), e2], 1))
        d1 = self.b1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1), e3


class GlobalHead(nn.Module):
    """Whole-frame colour/exposure estimator (HDRNet-style low-res stream).

    Runs on a fixed 64x64 thumbnail of the *entire* frame (front-end RGB + noise map), so its
    statistics are identical in training (128-px patches) and inference (full frames).
    Predicts a 3x3 colour matrix (initialised to identity) and a log exposure gain.
    """

    def __init__(self, cin=4, ch=(16, 32, 64)):
        super().__init__()
        c1, c2, c3 = ch
        self.feat = nn.Sequential(
            nn.Conv2d(cin, c1, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, 3, 2, 1), nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(nn.Linear(c3 * 2, 64), nn.ReLU(inplace=True), nn.Linear(64, 10))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, thumb):
        f = self.feat(thumb)
        f = torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)
        p = self.mlp(f)
        mat = torch.eye(3, device=thumb.device).unsqueeze(0) + 0.5 * torch.tanh(p[:, :9]).view(-1, 3, 3)
        log_gain = 1.5 * torch.tanh(p[:, 9:10])
        return mat, log_gain


class HybridNet(nn.Module):
    def __init__(self, ch=(24, 48, 96), use_noise_map=True, use_global=True):
        super().__init__()
        self.use_noise_map, self.use_global = use_noise_map, use_global
        # works at half resolution (PixelUnshuffle) for mobile efficiency
        self.unet = LiteUNet(4 * (3 + int(use_noise_map)), 12, ch)
        self.down, self.up = nn.PixelUnshuffle(2), nn.PixelShuffle(2)
        self.head = GlobalHead() if use_global else None

    def forward(self, x, noise_map, thumb):
        inp = torch.cat([x, noise_map], 1) if self.use_noise_map else x
        res, _ = self.unet(self.down(inp))
        y = x + self.up(res)
        if self.use_global:
            mat, lg = self.head(thumb)
            lin = srgb_to_linear_t(y)
            lin = torch.einsum("bij,bjhw->bihw", mat, lin) * torch.exp(lg)[..., None, None]
            y = linear_to_srgb_t(lin)
        return y.clamp(0, 1)


class LearnedISP(nn.Module):
    """Pure learned RAW->RGB: packed RGGB (+noise map) at H/2 -> sRGB at H via PixelShuffle."""

    def __init__(self, ch=(24, 48, 96)):
        super().__init__()
        self.unet = LiteUNet(5, 12, ch)
        self.ps = nn.PixelShuffle(2)

    def forward(self, packed, noise_map):
        out, _ = self.unet(torch.cat([packed, noise_map], 1))
        return torch.sigmoid(self.ps(out))


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
