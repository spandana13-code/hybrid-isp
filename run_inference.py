"""Batch inference CLI: run the ISP pipelines on every supported file in a folder.

    python run_inference.py --input_dir ./examples/ --output_dir ./results/inference/
    python run_inference.py --input_dir ./examples/ --output_dir ./out --pipeline all --precision fp32

Inputs: phone RAW/DNG (as-shot WB + colour matrix from metadata), SIDD-style .npz blocks
(with ground truth -> metrics are reported), RGGB .npy Bayer arrays, or JPG/PNG
(converted to a simulated sensor RAW). Outputs: one PNG per input and pipeline, plus
metrics.csv (quality metrics only where ground truth exists, latency always).
Uses the exported ONNX models in models/ - no PyTorch needed.
"""
import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

from isp.engine import ISPEngine
from isp.io import load_any, InputError, RAW_EXT, IMG_EXT
from isp.sidd import to_rggb
from isp.metrics import psnr, ssim, delta_e
from isp.raw_sim import sample_degradation, simulate_raw

SUPPORTED = RAW_EXT | IMG_EXT | {".npz", ".npy"}


def to_u8(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--pipeline", default="hybrid", choices=["hybrid", "learned", "classical", "all"])
    ap.add_argument("--precision", default="int8", choices=["int8", "fp16", "fp32"])
    ap.add_argument("--condition", default="normal",
                    help="simulated-sensor condition for JPG/PNG inputs: normal, low_light, high_noise, wb_cast")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not os.path.isdir(args.input_dir):
        sys.exit(f"input_dir not found: {args.input_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(args.input_dir) if os.path.splitext(f)[1].lower() in SUPPORTED)
    if not files:
        sys.exit(f"no supported files in {args.input_dir} (supported: {', '.join(sorted(SUPPORTED))})")

    eng = ISPEngine(args.precision)
    pipes = ["classical", "learned", "hybrid"] if args.pipeline == "all" else [args.pipeline]
    for p in pipes:
        if p != "classical" and not eng.available(p):
            sys.exit(f"model models/{p}_{args.precision}.onnx not found - run optimize.py first")

    rows = []
    for f in files:
        path = os.path.join(args.input_dir, f)
        try:
            d = load_any(path)
        except InputError as e:
            print(f"[skip] {f}: {e}")
            continue
        if "raw" in d:
            raw, fe, gt, cfa = d["raw"], d["fe"], d["gt"], d["cfa"]
        else:  # sRGB photo -> simulated sensor RAW; the original photo serves as reference
            deg = sample_degradation(np.random.default_rng(args.seed), args.condition)
            raw, fe, gt, cfa = simulate_raw(d["srgb"], deg), {}, d["srgb"], "rggb"

        stem = os.path.splitext(f)[0]
        info = None
        for p in pipes:
            t0 = time.perf_counter()
            if p == "classical":
                out, info, _ = eng.classical(raw, **fe)
            elif p == "learned":
                out, info2, _ = eng.learned(raw, info, **fe)
                info = info or info2
            else:
                out, info2, _ = eng.hybrid(raw, **fe)
                info = info or info2
            ms = (time.perf_counter() - t0) * 1000
            out_path = os.path.join(args.output_dir, f"{stem}_{p}.png")
            cv2.imwrite(out_path, cv2.cvtColor(to_u8(to_rggb(out, cfa)), cv2.COLOR_RGB2BGR))
            row = dict(file=f, pipeline=p, precision=args.precision, h=raw.shape[0], w=raw.shape[1],
                       latency_ms=round(ms, 1), psnr="", ssim="", delta_e2000="")
            if gt is not None:
                row.update(psnr=round(psnr(gt, out), 2), ssim=round(ssim(gt, out), 4),
                           delta_e2000=round(delta_e(gt, out), 2))
            rows.append(row)
            q = f"  PSNR {row['psnr']} dB  SSIM {row['ssim']}  dE {row['delta_e2000']}" if gt is not None else ""
            print(f"{f:40s} {p:9s} {ms:7.0f} ms{q}")

    if rows:
        with open(os.path.join(args.output_dir, "metrics.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n{len(rows)} outputs + metrics.csv written to {args.output_dir}")


if __name__ == "__main__":
    main()
