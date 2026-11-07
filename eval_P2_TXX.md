1RUA24CSE0507,1RUA24CSE0509,1RUA24CSE0510,1RUA24CSE0511

# Evaluation – Problem P2 (Hybrid Smartphone ISP) – Team TXX

**Test set: official SIDD benchmark** – 928 blocks of 256×256 real noisy RAW (29 capture instances, 5 smartphones: Google Pixel, iPhone 7, Nexus 6, LG G4, Galaxy S6) with SIDD's clean sRGB ground truth. No test instance is used in training, colour calibration or model selection. The test set ships without per-image metadata, so white balance is *estimated* by every pipeline; the camera colour matrix comes from the per-camera calibration fitted on training captures.

## 1. Metrics (mean over all blocks)

| Pipeline | PSNR (dB) ↑ | SSIM ↑ | ΔE2000 ↓ | LPIPS ↓ | Params | Model size | Latency 1 MP (ms) | Peak NN memory 1 MP |
|---|---|---|---|---|---|---|---|---|
| Classical ISP | 26.25 | 0.7241 | 10.65 | 0.4475 | 0 | – | 536 | – |
| Learned ISP (FP32) | 29.22 | 0.8828 | 8.07 | 0.2716 | 115,212 | 463 KB | 200 | 290 MB |
| Learned ISP (INT8) | 28.69 | 0.8637 | 8.83 | 0.2874 | 115,212 | 202 KB | 180 | 209 MB |
| Hybrid ISP (FP32) | 29.17 | 0.8675 | 7.85 | 0.2773 | 187,150 | 752 KB | 346 | 364 MB |
| Hybrid ISP (INT8) | 29.12 | 0.8647 | 7.86 | 0.2855 | 187,150 | 488 KB | 285 | 225 MB |

Latency = full pipeline (all classical stages + network) on a 1024×1024 crop of a real SIDD frame, ONNX Runtime on 2 CPU threads. Memory = peak resident memory of the neural stage at 1024×1024.

## 2. Robustness across real conditions (PSNR dB / ΔE2000)

Blocks are split into thirds by the noise level the adaptive controller *measured* (σ ≤ 0.026 / ≤ 0.036 / above) and by scene brightness (mean GT intensity ≤ 0.19 / ≤ 0.37 / above).

| Pipeline | low noise | medium noise | high noise | dark | mid | bright |
|---|---|---|---|---|---|---|
| Classical ISP | 26.11 / 10.78 | 27.82 / 9.52 | 24.81 / 11.63 | 26.37 / 10.66 | 25.49 / 11.55 | 26.88 / 9.73 |
| Learned ISP (FP32) | 28.10 / 8.80 | 31.50 / 6.52 | 28.07 / 8.89 | 32.08 / 6.21 | 28.25 / 8.71 | 27.33 / 9.30 |
| Learned ISP (INT8) | 27.60 / 9.74 | 30.85 / 7.25 | 27.62 / 9.49 | 31.42 / 6.82 | 27.51 / 9.98 | 27.14 / 9.69 |
| Hybrid ISP (FP32) | 28.31 / 8.41 | 31.27 / 6.53 | 27.93 / 8.62 | 31.98 / 6.13 | 28.47 / 8.30 | 27.05 / 9.14 |
| Hybrid ISP (INT8) | 28.26 / 8.40 | 31.22 / 6.54 | 27.88 / 8.63 | 31.92 / 6.14 | 28.41 / 8.29 | 27.02 / 9.15 |

### By phone (PSNR dB)

| Pipeline | Google Pixel | iPhone 7 | Nexus 6 | LG G4 | Galaxy S6 |
|---|---|---|---|---|---|
| Classical ISP | 24.88 | 26.35 | 27.12 | 26.91 | 28.06 |
| Learned ISP (FP32) | 27.22 | 29.81 | 30.26 | 30.19 | 30.78 |
| Learned ISP (INT8) | 26.85 | 29.02 | 29.92 | 29.58 | 29.76 |
| Hybrid ISP (FP32) | 27.06 | 30.41 | 30.02 | 29.83 | 31.00 |
| Hybrid ISP (INT8) | 26.99 | 30.36 | 29.99 | 29.79 | 30.89 |

## 3. Optimisation (neural stage only, ONNX Runtime CPU, 2 threads)

| Network | Precision | Size | Latency 512² (ms) | Latency 1024² (ms) | Peak mem 1024² (MB) |
|---|---|---|---|---|---|
| hybrid | FP32 | 752 KB | 43 | 162 | 364 |
| hybrid | FP16 | 395 KB | 41 | 163 | 359 |
| hybrid | INT8 | 488 KB | 32 | 153 | 225 |
| learned | FP32 | 463 KB | 32 | 132 | 290 |
| learned | FP16 | 242 KB | 26 | 120 | 296 |
| learned | INT8 | 202 KB | 20 | 107 | 209 |

FP16 = weights in half precision (reduced precision). INT8 = post-training static quantisation (ONNX Runtime QDQ, per-channel INT8 weights, UINT8 activations, percentile calibration on 96 SIDD training patches; the hybrid colour head stays FP32 — mixed precision).

## 4. Stage latency breakdown (ms, 1 MP real frame)

- **Classical ISP** (536 ms): preprocess 8.4, analyze 16.2, white_balance 1.0, demosaic 5.7, ccm 8.0, gain_tonemap 13.8, denoise 465.1, sharpen 13.4
- **Learned ISP (FP32)** (200 ms): preprocess+analyze 25.5, neural_net 173.4
- **Learned ISP (INT8)** (180 ms): preprocess+analyze 25.7, neural_net 153.5
- **Hybrid ISP (FP32)** (346 ms): preprocess 8.0, analyze 12.0, white_balance 0.8, demosaic 6.1, ccm 4.9, gain_tonemap 13.8, noise_map 28.0, neural_net 251.0, adaptive_sharpen 21.0
- **Hybrid ISP (INT8)** (285 ms): preprocess 7.2, analyze 11.9, white_balance 0.8, demosaic 5.9, ccm 4.9, gain_tonemap 13.0, noise_map 26.4, neural_net 195.0, adaptive_sharpen 19.6

## 5. Ablation study (mean over all blocks)

| Variant | PSNR ↑ | SSIM ↑ | ΔE2000 ↓ | GMSD ↓ |
|---|---|---|---|---|
| Classical ISP | 26.25 | 0.7241 | 10.65 | 0.0637 |
| Classical: no white balance | 19.81 | 0.6623 | 19.86 | 0.0595 |
| Classical: no camera CCM (nominal matrix) | 23.94 | 0.6459 | 14.17 | 0.0699 |
| Classical: no denoise | 23.79 | 0.4321 | 13.71 | 0.0887 |
| Classical: no sharpen | 26.39 | 0.7344 | 10.64 | 0.0581 |
| Classical: bilinear demosaic | 26.47 | 0.7441 | 10.58 | 0.0558 |
| Classical: bilateral instead of NLM | 27.35 | 0.8027 | 9.99 | 0.0393 |
| Classical: non-adaptive (fixed params) | 26.41 | 0.7467 | 10.62 | 0.0664 |
| Hybrid ISP (FP32) | 29.17 | 0.8675 | 7.85 | 0.0289 |
| Hybrid: front-end only (no network) | 23.93 | 0.4453 | 13.69 | 0.0786 |
| Hybrid: no post-sharpen | 29.19 | 0.8695 | 7.85 | 0.0288 |
| Hybrid: no noise-map conditioning | 29.27 | 0.8681 | 7.93 | 0.0281 |
| Hybrid: no global colour/exposure head | 29.02 | 0.8403 | 8.26 | 0.0276 |

## 6. Failure cases (lowest hybrid PSNR)

- instance 0020 block 21 (Google Pixel, measured σ 0.025): hybrid 10.72 dB / ΔE 25.90; classical 10.62 dB; learned 10.68 dB
- instance 0099 block 21 (LG G4, measured σ 0.022): hybrid 11.21 dB / ΔE 19.61; classical 11.15 dB; learned 11.08 dB
- instance 0123 block 0 (LG G4, measured σ 0.023): hybrid 11.39 dB / ΔE 23.64; classical 11.28 dB; learned 11.35 dB
- instance 0123 block 23 (LG G4, measured σ 0.023): hybrid 12.92 dB / ΔE 21.64; classical 12.30 dB; learned 12.58 dB
- instance 0099 block 23 (LG G4, measured σ 0.023): hybrid 13.14 dB / ΔE 17.62; classical 12.84 dB; learned 12.79 dB
- instance 0099 block 0 (LG G4, measured σ 0.023): hybrid 13.20 dB / ΔE 19.19; classical 12.91 dB; learned 12.83 dB

Panels (reference | classical | learned | hybrid) for blocks 0 and 13 of every test instance are in `results/failure_cases/`; the analysis is in `docs/DESIGN.md` §10.
