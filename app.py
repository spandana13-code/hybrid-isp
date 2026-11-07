"""Hybrid Smartphone ISP - web demo (Gradio).

    python app.py            # http://localhost:7860
"""
import os
import time
import traceback

import cv2
import gradio as gr
import numpy as np

from isp.engine import ISPEngine
from isp.io import load_any, load_srgb, InputError
from isp.sidd import to_rggb
from isp.metrics import psnr, ssim, delta_e, gmsd
from isp.raw_sim import (CONDITIONS, Degradation, sample_degradation, simulate_raw, pack_rggb,
                         BLACK_LEVEL, WHITE_LEVEL)

ROOT = os.path.dirname(os.path.abspath(__file__))
ENGINES = {"INT8 (quantised)": ISPEngine("int8"), "FP32": ISPEngine("fp32")}
EXAMPLES = sorted(os.path.join(ROOT, "examples", f) for f in os.listdir(os.path.join(ROOT, "examples")))


def visualize_raw(raw):
    p = pack_rggb((raw.astype(np.float32) - BLACK_LEVEL) / (WHITE_LEVEL - BLACK_LEVEL))
    rgb = np.stack([p[..., 0], (p[..., 1] + p[..., 2]) / 2, p[..., 3]], -1)
    rgb = np.clip(rgb, 0, 1) ** (1 / 2.2)
    return cv2.resize(rgb, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST)


def to_u8(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def build_raw(file, condition, exposure, noise, wb_cast, seed):
    """Returns (raw, front-end settings, ground truth or None, CFA for display, source text, degradation or None)."""
    if file is None:
        raise InputError("Upload a file or pick one of the examples first.")
    path = file if isinstance(file, str) else file.name
    d = load_any(path)
    if "raw" in d:
        return d["raw"], d["fe"], d["gt"], d["cfa"], d["source"], None
    img = d["srgb"]
    rng = np.random.default_rng(int(seed))
    if condition == "custom":
        log_shot = np.log(noise)
        deg = Degradation(exposure=float(exposure), r_gain=2.0 + 0.9 * wb_cast, b_gain=1.6 - 0.6 * wb_cast,
                          shot_noise=float(noise), read_noise=float(np.exp(2.18 * log_shot + 1.2)),
                          ccm_jitter=0.05, seed=int(seed))
    else:
        deg = sample_degradation(rng, condition)
    # simulated sensor: unknown exposure -> auto-exposure + highlight knee, nominal CCM
    return simulate_raw(img, deg), {}, img, "rggb", d["source"], deg.to_dict()


def run(file, condition, exposure, noise, wb_cast, seed, precision, progress=gr.Progress()):
    try:
        progress(0.05, desc="Reading input")
        raw, fe, gt, cfa, source, deg = build_raw(file, condition, exposure, noise, wb_cast, seed)
        eng = ENGINES[precision]
        progress(0.2, desc="Classical ISP")
        t0 = time.perf_counter()
        res, info = {}, None
        out, info, t = eng.classical(raw, **fe)
        res["Classical"] = (out, t)
        progress(0.5, desc="Learned ISP")
        out, _, t = eng.learned(raw, info, **fe)
        res["Learned"] = (out, t)
        progress(0.75, desc="Hybrid ISP")
        out, _, t = eng.hybrid(raw, **fe)
        res["Hybrid"] = (out, t)
        total = (time.perf_counter() - t0) * 1000

        rows = []
        for k, (img, t) in res.items():
            row = [k, round(sum(t.values()), 0)]
            if gt is not None:
                row += [round(psnr(gt, img), 2), round(ssim(gt, img), 4), round(delta_e(gt, img), 2), round(gmsd(gt, img), 4)]
            else:
                row += ["-", "-", "-", "-"]
            rows.append(row)

        stages = []
        for k, (_, t) in res.items():
            for s, ms in t.items():
                stages.append([k, s, round(ms, 1)])

        h, w = raw.shape
        ctrl = (f"**Input:** {source} · {w}×{h} px  \n"
                f"**Adaptive controller:** scene = `{info['scene']}` · est. noise σ = {info['noise_sigma']:.3f} · "
                f"digital gain = {info['gain']:.2f}× · WB ({info['wb_source']}) R {info['wb_gains'][0]:.2f} / B {info['wb_gains'][1]:.2f} · "
                f"denoise strength h = {info['denoise_h']:.1f} · sharpen = {info['sharpen']:.2f}  \n"
                f"**Precision:** {precision} · total processing {total:.0f} ms")
        if deg is not None:
            ctrl += (f"  \n**Simulated sensor:** exposure {deg['exposure']:.2f} · shot-noise {deg['shot_noise']:.4f} · "
                     f"illuminant gains R {deg['r_gain']:.2f} / B {deg['b_gain']:.2f}")
        if gt is None:
            ctrl += "  \n_No ground truth for real RAW files, so quality metrics are not shown._"
        show = lambda im: to_u8(to_rggb(im, cfa))     # back to the camera's original orientation
        gt_img = show(gt) if gt is not None else None
        return (show(visualize_raw(raw)), show(res["Classical"][0]), show(res["Learned"][0]),
                show(res["Hybrid"][0]), gt_img, rows, stages, ctrl)
    except InputError as e:
        raise gr.Error(str(e))
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Processing failed: {type(e).__name__}: {e}")


CSS = """
.gradio-container {max-width: 1280px !important}
#title h1 {margin-bottom: 0}
"""

with gr.Blocks(title="Hybrid Smartphone ISP") as demo:
    gr.Markdown("# Hybrid Smartphone ISP\nReal phone RAW → sRGB with a classical ISP, a pure learned ISP and a hybrid "
                "(classical front-end + 0.19 M-param INT8 network), trained on the SIDD smartphone dataset. "
                "Try the **SIDD examples** (real noisy RAW from 5 phones, never seen in training, with ground truth), "
                "upload your own phone **DNG**, or any **JPG/PNG** (turned into a simulated sensor RAW).",
                elem_id="title")
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            f = gr.File(label="Input (phone RAW/DNG, .npz/.npy Bayer, or JPG/PNG)", type="filepath")
            gr.Examples(EXAMPLES, inputs=f, label="Examples")
            cond = gr.Dropdown(list(CONDITIONS) + ["custom"], value="low_light",
                               label="Simulated sensor condition (JPG/PNG inputs only)")
            with gr.Accordion("Custom degradation", open=False):
                exp = gr.Slider(0.03, 1.0, 0.25, label="Exposure (scene brightness)")
                noi = gr.Slider(0.0001, 0.02, 0.004, label="Shot-noise level")
                wbc = gr.Slider(-1.0, 1.0, 0.6, label="Illuminant cast (− shade · + tungsten)")
            seed = gr.Number(7, label="Noise seed", precision=0)
            prec = gr.Radio(list(ENGINES), value="INT8 (quantised)", label="Neural precision")
            btn = gr.Button("Process", variant="primary")
            ctrl = gr.Markdown()
        with gr.Column(scale=3):
            with gr.Row():
                o_raw = gr.Image(label="Visualised RAW (no processing)", height=260)
                o_gt = gr.Image(label="Reference (ground truth)", height=260)
            with gr.Row():
                o_cl = gr.Image(label="Classical ISP", height=260)
                o_le = gr.Image(label="Learned ISP", height=260)
                o_hy = gr.Image(label="Hybrid ISP", height=260)
            tbl = gr.Dataframe(headers=["Pipeline", "Latency (ms)", "PSNR ↑", "SSIM ↑", "ΔE2000 ↓", "GMSD ↓"],
                               label="Quality vs reference", interactive=False)
            with gr.Accordion("Per-stage latency", open=False):
                stg = gr.Dataframe(headers=["Pipeline", "Stage", "ms"], interactive=False)
    btn.click(run, [f, cond, exp, noi, wbc, seed, prec], [o_raw, o_cl, o_le, o_hy, o_gt, tbl, stg, ctrl])

if __name__ == "__main__":
    demo.queue(max_size=16).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), css=CSS)
