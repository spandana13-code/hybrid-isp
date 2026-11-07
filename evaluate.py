"""Comparative evaluation on the official SIDD benchmark (real smartphone RAW).

928 blocks (29 capture instances x 32 blocks of 256x256) of real noisy RAW from 5 phones, with
SIDD's clean sRGB ground truth. Every pipeline sees exactly the same blocks; no test instance is
used for training, calibration or model selection. Writes eval_P2_<TEAM>.md and results/.

    python evaluate.py --team T01 --usns "1XX22CS001,1XX22CS002"
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from isp.classical import classical_isp, sharpen, front_end, denoise
from isp.data import hybrid_inputs
from isp.engine import ISPEngine, _pad
from isp.metrics import all_metrics, lpips_fn
from isp.sidd import (load_test_blocks, norm_to_dn, fe_settings, load_frame, camera_ccm, to_rggb,
                      split_names, train_names)

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
MAIN = ["Classical ISP", "Learned ISP (FP32)", "Learned ISP (INT8)", "Hybrid ISP (FP32)", "Hybrid ISP (INT8)"]
CAM_NAMES = {"GP": "Google Pixel", "IP": "iPhone 7", "N6": "Nexus 6", "G4": "LG G4", "S6": "Galaxy S6"}


class TorchVariant:
    """Runs an ablation checkpoint (PyTorch) inside the hybrid pipeline."""

    def __init__(self, name):
        import torch
        from train import make_model
        self.torch = torch
        self.m = make_model(name)
        self.m.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", f"{name}.pt"), map_location="cpu"))
        self.m.eval()

    def __call__(self, raw, **fe):
        x, nm, th, info, _ = hybrid_inputs(raw, **fe)
        xp, h, w = _pad(x, 8); npd, _, _ = _pad(nm, 8)
        t = lambda a: self.torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None]
        with self.torch.no_grad():
            y = self.m(t(xp), t(npd), t(th))[0].permute(1, 2, 0).numpy()[:h, :w]
        return sharpen(np.clip(y, 0, 1), 0.5 * info["sharpen"])


def u8(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="TXX")
    ap.add_argument("--usns", default="USN1,USN2,USN3,USN4")
    ap.add_argument("--stride", type=int, default=1, help="evaluate every n-th block (1 = all 928)")
    args = ap.parse_args()
    os.makedirs(os.path.join(RES, "failure_cases"), exist_ok=True)

    e32, e8 = ISPEngine("fp32"), ISPEngine("int8")
    variants = {v: TorchVariant(v) for v in ("hybrid_nonoise", "hybrid_noglobal")
                if os.path.exists(os.path.join(ROOT, "checkpoints", f"{v}.pt"))}
    use_lpips = lpips_fn() is not None

    methods = {
        "Classical ISP": lambda r, fe: classical_isp(r, **fe)[0],
        "Learned ISP (FP32)": lambda r, fe: e32.learned(r, **fe)[0],
        "Learned ISP (INT8)": lambda r, fe: e8.learned(r, **fe)[0],
        "Hybrid ISP (FP32)": lambda r, fe: e32.hybrid(r, **fe)[0],
        "Hybrid ISP (INT8)": lambda r, fe: e8.hybrid(r, **fe)[0],
    }
    ablations = {
        "Classical: no white balance": lambda r, fe: classical_isp(r, use_wb=False, **fe)[0],
        "Classical: no camera CCM (nominal matrix)": lambda r, fe: classical_isp(r, **{**fe, "ccm": None})[0],
        "Classical: no denoise": lambda r, fe: classical_isp(r, use_denoise=False, **fe)[0],
        "Classical: no sharpen": lambda r, fe: classical_isp(r, use_sharpen=False, **fe)[0],
        "Classical: bilinear demosaic": lambda r, fe: classical_isp(r, demosaic_method="bilinear", **fe)[0],
        "Classical: bilateral instead of NLM": lambda r, fe: classical_isp(r, denoise_method="bilateral", **fe)[0],
        "Classical: non-adaptive (fixed params)": lambda r, fe: sharpen(denoise(front_end(r, **fe)[0], 10.0), 0.5),
        "Hybrid: front-end only (no network)": lambda r, fe: hybrid_inputs(r, **fe)[0],
        "Hybrid: no post-sharpen": lambda r, fe: e32.hybrid(r, post_sharpen=False, **fe)[0],
    }
    labels = {"hybrid_nonoise": "Hybrid: no noise-map conditioning",
              "hybrid_noglobal": "Hybrid: no global colour/exposure head"}
    for v, fn in variants.items():
        ablations[labels[v]] = (lambda f: (lambda r, fe: f(r, **fe)))(fn)

    blocks = load_test_blocks()[::args.stride]
    rows, t0 = [], time.time()
    for k, b in enumerate(blocks):
        raw = norm_to_dn(b["raw"])
        fe = fe_settings(b["ccm"])                   # test set: no metadata -> AWB is estimated
        info = front_end(raw, **fe)[2]
        outs = {}
        for m, fn in {**methods, **ablations}.items():
            out = fn(raw, fe)
            met = all_metrics(b["gt"], out, with_lpips=use_lpips and m in MAIN)
            rows.append(dict(scene=b["scene"], block=b["block"], cam=b["cam"], method=m,
                             noise=info["noise_sigma"], bright=float(b["gt"].mean()), **met))
            outs[m] = out
        if b["block"] in (0, 13):
            # flip back from RGGB working orientation to the camera's original orientation
            panel = np.hstack([to_rggb(im, b["cfa"]) for im in
                               [b["gt"]] + [outs[m] for m in ("Classical ISP", "Learned ISP (FP32)", "Hybrid ISP (FP32)")]])
            cv2.imwrite(os.path.join(RES, "failure_cases", f"sidd_{b['scene']}_{b['block']:02d}.jpg"),
                        cv2.cvtColor(u8(panel), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if (k + 1) % 100 == 0:
            print(f"{k + 1}/{len(blocks)} blocks ({time.time() - t0:.0f}s)", flush=True)
    json.dump(rows, open(os.path.join(RES, "eval_rows.json"), "w"))

    # ------------------------------------------------------------------ latency on a real 12 MP-class frame crop
    opt = json.load(open(os.path.join(RES, "optimization.json")))
    f = load_frame(split_names(train_names())[1][0])          # a held-out validation capture
    H, W = f["raw"].shape
    raw = norm_to_dn(f["raw"][H // 2 - 512:H // 2 + 512, W // 2 - 512:W // 2 + 512])   # 1 MP
    fe = fe_settings(camera_ccm(f["cam"]), f["wb"])
    lat = {}
    for label, fn in (("Classical ISP", lambda: e32.classical(raw, **fe)[2]),
                      ("Learned ISP (FP32)", lambda: e32.learned(raw, **fe)[2]),
                      ("Learned ISP (INT8)", lambda: e8.learned(raw, **fe)[2]),
                      ("Hybrid ISP (FP32)", lambda: e32.hybrid(raw, **fe)[2]),
                      ("Hybrid ISP (INT8)", lambda: e8.hybrid(raw, **fe)[2])):
        fn()
        runs = [fn() for _ in range(3)]
        tot = [sum(r.values()) for r in runs]
        best = runs[int(np.argmin(tot))]
        lat[label] = dict(total_ms=float(np.median(tot)), stages={k: round(v, 1) for k, v in best.items()})
    json.dump(lat, open(os.path.join(RES, "latency_pipeline.json"), "w"), indent=1)
    write_report(rows, opt, lat, args, use_lpips, list(ablations))
    print("done in", round(time.time() - t0))


def agg(rows, method, key="psnr", **flt):
    v = [r[key] for r in rows if r["method"] == method and all(f(r) for f in flt.values())]
    v = [x for x in v if x == x]
    return float(np.mean(v)) if v else float("nan")


def write_report(rows, opt, lat, args, use_lpips, ablations):
    pk = "lpips" if use_lpips else "gmsd"
    pname = "LPIPS ↓" if use_lpips else "GMSD ↓"

    def eff(m):
        if m.startswith("Classical"):
            return "0", "–", "–"
        net = "hybrid" if m.startswith("Hybrid") else "learned"
        prec = "int8" if "INT8" in m else "fp32"
        o = opt[net]
        return f'{o["params"]:,}', f'{o[prec]["size_mb"] * 1024:.0f} KB', f'{o[prec]["1024x1024"]["peak_mem_mb"]:.0f} MB'

    n_blocks = len({(r["scene"], r["block"]) for r in rows})
    L = [args.usns, "", f"# Evaluation – Problem P2 (Hybrid Smartphone ISP) – Team {args.team}", "",
         f"**Test set: official SIDD benchmark** – {n_blocks} blocks of 256×256 real noisy RAW "
         f"({len({r['scene'] for r in rows})} capture instances, 5 smartphones: Google Pixel, iPhone 7, Nexus 6, LG G4, "
         "Galaxy S6) with SIDD's clean sRGB ground truth. No test instance is used in training, colour calibration "
         "or model selection. The test set ships without per-image metadata, so white balance is *estimated* by every "
         "pipeline; the camera colour matrix comes from the per-camera calibration fitted on training captures.", "",
         "## 1. Metrics (mean over all blocks)", "",
         f"| Pipeline | PSNR (dB) ↑ | SSIM ↑ | ΔE2000 ↓ | {pname} | Params | Model size | Latency 1 MP (ms) | Peak NN memory 1 MP |",
         "|---|---|---|---|---|---|---|---|---|"]
    for m in MAIN:
        p, s, mem = eff(m)
        L.append(f"| {m} | {agg(rows, m):.2f} | {agg(rows, m, 'ssim'):.4f} | {agg(rows, m, 'delta_e'):.2f} | "
                 f"{agg(rows, m, pk):.4f} | {p} | {s} | {lat[m]['total_ms']:.0f} | {mem} |")
    L += ["", "Latency = full pipeline (all classical stages + network) on a 1024×1024 crop of a real SIDD frame, "
          "ONNX Runtime on 2 CPU threads. Memory = peak resident memory of the neural stage at 1024×1024.", ""]

    # robustness by condition (real data)
    noise = np.array([r["noise"] for r in rows if r["method"] == MAIN[0]])
    bright = np.array([r["bright"] for r in rows if r["method"] == MAIN[0]])
    n1, n2 = np.quantile(noise, [1 / 3, 2 / 3])
    b1, b2 = np.quantile(bright, [1 / 3, 2 / 3])
    conds = {
        "low noise": lambda r: r["noise"] <= n1, "medium noise": lambda r: n1 < r["noise"] <= n2,
        "high noise": lambda r: r["noise"] > n2,
        "dark": lambda r: r["bright"] <= b1, "mid": lambda r: b1 < r["bright"] <= b2, "bright": lambda r: r["bright"] > b2,
    }
    L += ["## 2. Robustness across real conditions (PSNR dB / ΔE2000)", "",
          "Blocks are split into thirds by the noise level the adaptive controller *measured* "
          f"(σ ≤ {n1:.3f} / ≤ {n2:.3f} / above) and by scene brightness (mean GT intensity ≤ {b1:.2f} / ≤ {b2:.2f} / above).", "",
          "| Pipeline | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    for m in MAIN:
        L.append(f"| {m} | " + " | ".join(f"{agg(rows, m, c=f):.2f} / {agg(rows, m, 'delta_e', c=f):.2f}" for f in conds.values()) + " |")
    L += ["", "### By phone (PSNR dB)", "", "| Pipeline | " + " | ".join(CAM_NAMES.values()) + " |",
          "|---|" + "---|" * len(CAM_NAMES)]
    for m in MAIN:
        L.append(f"| {m} | " + " | ".join(f"{agg(rows, m, c=(lambda cc: (lambda r: r['cam'] == cc))(c)):.2f}" for c in CAM_NAMES) + " |")

    L += ["", "## 3. Optimisation (neural stage only, ONNX Runtime CPU, 2 threads)", "",
          "| Network | Precision | Size | Latency 512² (ms) | Latency 1024² (ms) | Peak mem 1024² (MB) |",
          "|---|---|---|---|---|---|"]
    for net in ("hybrid", "learned"):
        for prec in ("fp32", "fp16", "int8"):
            if prec in opt[net]:
                o = opt[net][prec]
                L.append(f"| {net} | {prec.upper()} | {o['size_mb'] * 1024:.0f} KB | {o['512x512']['latency_ms']:.0f} | "
                         f"{o['1024x1024']['latency_ms']:.0f} | {o['1024x1024']['peak_mem_mb']:.0f} |")
    L += ["", "FP16 = weights in half precision (reduced precision). INT8 = post-training static quantisation "
          "(ONNX Runtime QDQ, per-channel INT8 weights, UINT8 activations, percentile calibration on 96 SIDD training patches; the hybrid colour head stays FP32 — mixed precision).", "",
          "## 4. Stage latency breakdown (ms, 1 MP real frame)", ""]
    for m, v in lat.items():
        L.append(f"- **{m}** ({v['total_ms']:.0f} ms): " + ", ".join(f"{k} {s}" for k, s in v["stages"].items()))
    L += ["", "## 5. Ablation study (mean over all blocks)", "", "| Variant | PSNR ↑ | SSIM ↑ | ΔE2000 ↓ | GMSD ↓ |", "|---|---|---|---|---|"]
    for m in ["Classical ISP"] + [a for a in ablations if a.startswith("Classical")] + ["Hybrid ISP (FP32)"] + [a for a in ablations if a.startswith("Hybrid")]:
        L.append(f"| {m} | {agg(rows, m):.2f} | {agg(rows, m, 'ssim'):.4f} | {agg(rows, m, 'delta_e'):.2f} | {agg(rows, m, 'gmsd'):.4f} |")
    L += ["", "## 6. Failure cases (lowest hybrid PSNR)", ""]
    hy = sorted([r for r in rows if r["method"] == "Hybrid ISP (FP32)"], key=lambda r: r["psnr"])[:6]
    for r in hy:
        g = {x["method"]: x for x in rows if x["scene"] == r["scene"] and x["block"] == r["block"]}
        L.append(f"- instance {r['scene']} block {r['block']} ({CAM_NAMES[r['cam']]}, measured σ {r['noise']:.3f}): "
                 f"hybrid {r['psnr']:.2f} dB / ΔE {r['delta_e']:.2f}; classical {g['Classical ISP']['psnr']:.2f} dB; "
                 f"learned {g['Learned ISP (FP32)']['psnr']:.2f} dB")
    L += ["", "Panels (reference | classical | learned | hybrid) for blocks 0 and 13 of every test instance are in "
          "`results/failure_cases/`; the analysis is in `docs/DESIGN.md` §10."]
    open(os.path.join(ROOT, f"eval_P2_{args.team}.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
