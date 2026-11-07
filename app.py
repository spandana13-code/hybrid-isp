"""Hybrid Smartphone ISP - web demo (Gradio).

    python app.py            # http://localhost:7860

Lab tab: pick an example (or upload a phone DNG / JPG), compare any two pipeline outputs with a
before/after slider, read the headline gain, metrics and the adaptive controller's decisions.
Results tab: the official SIDD benchmark numbers, robustness, optimisation and figures.
"""
import os
import time
import traceback

import cv2
import gradio as gr
import numpy as np

from isp.engine import ISPEngine
from isp.io import load_any, InputError
from isp.sidd import to_rggb
from isp.metrics import psnr, ssim, delta_e, gmsd
from isp.raw_sim import (CONDITIONS, Degradation, sample_degradation, simulate_raw, pack_rggb,
                         BLACK_LEVEL, WHITE_LEVEL)

ROOT = os.path.dirname(os.path.abspath(__file__))
ENGINES = {"INT8 (quantised)": ISPEngine("int8"), "FP32": ISPEngine("fp32")}
PHONES = {"GP": "Google Pixel", "IP": "iPhone 7", "N6": "Nexus 6", "G4": "LG G4", "S6": "Galaxy S6"}
VIEWS = ["Visualised RAW", "Classical ISP", "Learned ISP", "Hybrid ISP", "Ground truth"]


# ----------------------------------------------------------------------------- helpers
def visualize_raw(raw):
    p = pack_rggb((raw.astype(np.float32) - BLACK_LEVEL) / (WHITE_LEVEL - BLACK_LEVEL))
    rgb = np.stack([p[..., 0], (p[..., 1] + p[..., 2]) / 2, p[..., 3]], -1)
    rgb = np.clip(rgb, 0, 1) ** (1 / 2.2)
    return cv2.resize(rgb, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST)


def to_u8(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def for_display(im):
    """Small benchmark blocks (256 px) are enlarged with nearest-neighbour so pixels stay honest."""
    h, w = im.shape[:2]
    k = max(1, int(np.ceil(640 / max(h, w))))
    return cv2.resize(im, (w * k, h * k), interpolation=cv2.INTER_NEAREST) if k > 1 else im


def build_raw(path, condition, exposure, noise, wb_cast, seed):
    """Returns (raw, front-end settings, ground truth or None, CFA for display, source text, degradation or None)."""
    if not path:
        raise InputError("Pick an example on the left or upload a file first.")
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


# ----------------------------------------------------------------------------- example cards
def _example_meta(path):
    name = os.path.basename(path)
    stem, ext = os.path.splitext(name)
    if ext == ".npz":
        z = np.load(path, allow_pickle=False)
        cam = str(z["cam"]) if "cam" in z else ""
        cfa = str(z["cfa"]) if "cfa" in z else "rggb"
        gt = z["gt"].astype(np.float32) / 255
        thumb = to_rggb(gt, cfa)
        phone = PHONES.get(cam, "Phone")
        light = "low light" if gt.mean() < 0.2 else "normal light"
        kind = "test block" if "_test_" in name else "full frame"
        title = f"{phone} · {light}"
        order = (1 if "_test_" in name else 0, name)
    else:
        im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        thumb = im.astype(np.float32) / 255
        title = stem.replace("_", " ").title()
        kind = "photo → sim. RAW"
        order = (2, name)
    h, w = thumb.shape[:2]
    s = min(h, w)
    thumb = thumb[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    thumb = cv2.resize(to_u8(thumb), (160, 160), interpolation=cv2.INTER_AREA)
    return order, path, thumb, f"{title} · {kind}"


EXAMPLES = sorted((_example_meta(os.path.join(ROOT, "examples", f))
                   for f in os.listdir(os.path.join(ROOT, "examples"))
                   if os.path.splitext(f)[1].lower() in {".npz", ".npy", ".png", ".jpg", ".jpeg", ".dng"}),
                  key=lambda e: e[0])
EX_PATHS = [e[1] for e in EXAMPLES]
EX_GALLERY = [(e[2], e[3]) for e in EXAMPLES]
EX_TITLES = [e[3] for e in EXAMPLES]


# ----------------------------------------------------------------------------- processing
def _headline(rows, gt, title, total_ms):
    if gt is None:
        hy = rows["Hybrid ISP"]
        return (f"<div class='hl'><div class='hl-title'>{title}</div>"
                f"<div class='hl-main'>Processed in <b>{hy['ms']:.0f} ms</b> (hybrid). "
                f"No ground truth for your own file, so compare the pipelines visually with the slider.</div></div>")
    cl, le, hy = rows["Classical ISP"], rows["Learned ISP"], rows["Hybrid ISP"]
    d_cl = hy["psnr"] - cl["psnr"]
    d_le = hy["psnr"] - le["psnr"]
    return (f"<div class='hl'><div class='hl-title'>{title}</div>"
            f"<div class='hl-main'><span class='hl-big'>Hybrid {d_cl:+.1f} dB PSNR</span> vs classical ISP · "
            f"colour error ΔE {cl['de']:.1f} → <b>{hy['de']:.1f}</b> · "
            f"{d_le:+.1f} dB vs the pure learned ISP · {hy['ms']:.0f} ms</div></div>")


def _metrics_html(rows, gt):
    keys = [("ms", "Latency (ms)", min, "{:.0f}")]
    if gt is not None:
        keys = [("psnr", "PSNR ↑", max, "{:.2f}"), ("ssim", "SSIM ↑", max, "{:.4f}"),
                ("de", "ΔE2000 ↓", min, "{:.2f}"), ("gmsd", "GMSD ↓", min, "{:.4f}")] + keys
    best = {k: f(r[k] for r in rows.values()) for k, _, f, _ in keys}
    head = "".join(f"<th>{lab}</th>" for _, lab, _, _ in keys)
    body = ""
    for name, r in rows.items():
        cls = " class='hy'" if name == "Hybrid ISP" else ""
        cells = "".join((f"<td><b>{fmt.format(r[k])}</b></td>" if r[k] == best[k] else f"<td>{fmt.format(r[k])}</td>")
                        for k, _, _, fmt in keys)
        body += f"<tr{cls}><td class='nm'>{name}</td>{cells}</tr>"
    note = "" if gt is not None else "<div class='note'>Quality metrics need a ground-truth reference.</div>"
    return f"<table class='mt'><tr><th>Pipeline</th>{head}</tr>{body}</table>{note}"


def _controller_html(info, source, w, h, precision, deg):
    items = [
        ("Input", f"{source} · {w}×{h}"),
        ("Scene", f"{info['scene']}"),
        ("Measured noise σ", f"{info['noise_sigma']:.3f}"),
        ("White balance", f"{info['wb_source']} · R {info['wb_gains'][0]:.2f} / B {info['wb_gains'][1]:.2f}"),
        ("Digital gain", f"{info['gain']:.2f}×"),
        ("Denoise strength h", f"{info['denoise_h']:.1f}"),
        ("Sharpen amount", f"{info['sharpen']:.2f}"),
        ("Neural precision", precision),
    ]
    if deg is not None:
        items.append(("Simulated sensor", f"exposure {deg['exposure']:.2f} · shot noise {deg['shot_noise']:.4f} · "
                                          f"illuminant R {deg['r_gain']:.2f} / B {deg['b_gain']:.2f}"))
    return "<div class='ctl'>" + "".join(f"<div><span>{k}</span><b>{v}</b></div>" for k, v in items) + "</div>"


def process(path, title, condition, exposure, noise, wb_cast, seed, precision, left, right, progress=gr.Progress()):
    try:
        progress(0.05, desc="Reading input")
        raw, fe, gt, cfa, source, deg = build_raw(path, condition, exposure, noise, wb_cast, seed)
        eng = ENGINES[precision]
        t0 = time.perf_counter()
        outs, times = {}, {}
        progress(0.2, desc="Classical ISP")
        outs["Classical ISP"], info, times["Classical ISP"] = eng.classical(raw, **fe)
        progress(0.5, desc="Learned ISP")
        outs["Learned ISP"], _, times["Learned ISP"] = eng.learned(raw, info, **fe)
        progress(0.75, desc="Hybrid ISP")
        outs["Hybrid ISP"], _, times["Hybrid ISP"] = eng.hybrid(raw, **fe)
        total = (time.perf_counter() - t0) * 1000

        rows = {}
        for k, im in outs.items():
            r = dict(ms=sum(times[k].values()))
            if gt is not None:
                r.update(psnr=psnr(gt, im), ssim=ssim(gt, im), de=delta_e(gt, im), gmsd=gmsd(gt, im))
            rows[k] = r

        show = lambda im: for_display(to_u8(to_rggb(im, cfa)))     # back to the camera's original orientation
        views = {"Visualised RAW": show(visualize_raw(raw)), **{k: show(v) for k, v in outs.items()}}
        if gt is not None:
            views["Ground truth"] = show(gt)
        stages = [[k, s, round(ms, 1)] for k, t in times.items() for s, ms in t.items()]
        h, w = raw.shape
        title = title or os.path.basename(path)
        left = left if left in views else "Classical ISP"
        right = right if right in views else "Hybrid ISP"
        return (views, (views[left], views[right]), _headline(rows, gt, title, total), _metrics_html(rows, gt),
                _controller_html(info, source, w, h, precision, deg), stages,
                [(views[k], k) for k in VIEWS if k in views])
    except InputError as e:
        raise gr.Error(str(e))
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Processing failed: {type(e).__name__}: {e}")


def change_view(views, left, right):
    if not views:
        return None
    return (views.get(left, views.get("Classical ISP")), views.get(right, views.get("Hybrid ISP")))


def pick_example(evt: gr.SelectData):
    return EX_PATHS[evt.index], EX_TITLES[evt.index]


def uploaded(path):
    return path, (os.path.basename(path) if path else "")


# ----------------------------------------------------------------------------- results tab
RESULTS_MD = """
### Official SIDD benchmark — 928 real phone RAW blocks, never seen in training

| Pipeline | PSNR (dB) ↑ | SSIM ↑ | ΔE2000 ↓ | LPIPS ↓ | Params | Size | Latency 1 MP |
|---|---|---|---|---|---|---|---|
| Classical ISP | 26.25 | 0.724 | 10.65 | 0.448 | 0 | – | 536 ms |
| Learned ISP (FP32) | **29.22** | **0.883** | 8.07 | **0.272** | 115 k | 463 KB | 200 ms |
| Learned ISP (INT8) | 28.69 | 0.864 | 8.83 | 0.287 | 115 k | 202 KB | 180 ms |
| Hybrid ISP (FP32) | 29.17 | 0.868 | **7.85** | 0.277 | 187 k | 752 KB | 346 ms |
| **Hybrid ISP (INT8, deployed)** | **29.12** | 0.865 | **7.86** | 0.286 | 187 k | 488 KB | 285 ms |

* **+2.9 dB over the classical ISP**, with the largest gains on **dark scenes (+5.6 dB)** and **high noise (+3.1 dB)**.
* **Best colour accuracy** (ΔE 7.86) of any deployable model.
* **Quantisation-robust:** mixed-precision INT8 costs the hybrid only 0.05 dB, but the pure learned ISP 0.53 dB.

### Robustness (PSNR dB, hybrid INT8 vs classical)

| | Dark | Mid | Bright | Low noise | Medium noise | High noise |
|---|---|---|---|---|---|---|
| Classical | 26.37 | 25.49 | 26.88 | 26.11 | 27.82 | 24.81 |
| Hybrid INT8 | **31.92** | **28.41** | **27.02** | **28.26** | **31.22** | **27.88** |

| | Google Pixel | iPhone 7 | Nexus 6 | LG G4 | Galaxy S6 |
|---|---|---|---|---|---|
| Classical | 24.88 | 26.35 | 27.12 | 26.91 | 28.06 |
| Hybrid INT8 | **26.99** | **30.36** | **29.99** | **29.79** | **30.89** |

Full tables, ablation study and failure analysis: `eval_P2_T42.md` and `docs/DESIGN.md` in the repository.
"""
FIGS = [(os.path.join(ROOT, "docs", f), cap) for f, cap in [
    ("architecture.png", "System architecture"),
    ("comparison.jpg", "SIDD test blocks — ground truth | classical | learned | hybrid (last row: failure case)"),
    ("intermediates.jpg", "Intermediate states — RAW | classical front-end | noise map | learned residual | hybrid | GT"),
]]
FIGS = [(p, c) for p, c in FIGS if os.path.exists(p)]


# ----------------------------------------------------------------------------- layout
CSS = """
.gradio-container {max-width: 1360px !important}
#hdr {display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap; padding:6px 2px 2px}
#hdr .brand {font-size:26px; font-weight:800; letter-spacing:-0.3px}
#hdr .brand small {font-size:14px; font-weight:400; opacity:.7; margin-left:10px}
#hdr .badge {font-size:13px; padding:6px 12px; border-radius:999px; border:1px solid #16a34a55;
             background:#16a34a1a; color:#16a34a; white-space:nowrap}
#hdr .badge:before {content:'● '}
.hl {padding:14px 18px; border-radius:12px; border:1px solid var(--border-color-primary); background:var(--block-background-fill)}
.hl-title {font-size:22px; font-weight:800; margin-bottom:4px}
.hl-main {font-size:15px; line-height:1.5}
.hl-big {font-weight:800; color:#4f46e5}
.mt {width:100%; border-collapse:collapse; font-size:14px}
.mt th, .mt td {padding:7px 10px; border-bottom:1px solid var(--border-color-primary); text-align:right}
.mt th:first-child, .mt td.nm {text-align:left}
.mt tr.hy td {background:#4f46e514}
.mt th {font-weight:600; opacity:.8}
.note {font-size:13px; opacity:.7; margin-top:6px}
.ctl {display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:8px}
.ctl div {padding:8px 10px; border-radius:8px; background:var(--block-background-fill); border:1px solid var(--border-color-primary)}
.ctl span {display:block; font-size:12px; opacity:.65}
.ctl b {font-size:14px; font-weight:600}
#examples .caption-label {font-size:11px !important; line-height:1.25 !important; padding:2px 6px !important;
  max-width:92% !important; white-space:normal !important; opacity:.92}
#swap {max-width:56px; min-width:56px}
"""
HEADER = ("<div id='hdr'><div class='brand'>Hybrid ISP<small>classical + learned smartphone camera pipeline · "
          "Problem P2 · Team Rebels (T42)</small></div>"
          "<div class='badge'>Hybrid core: HybridNet · INT8 · 187 k params · 488 KB</div></div>")

THEME = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")

with gr.Blocks(title="Hybrid Smartphone ISP") as demo:
    gr.HTML(HEADER)
    views_state = gr.State({})
    path_state = gr.State(EX_PATHS[0] if EX_PATHS else None)
    title_state = gr.State(EX_TITLES[0] if EX_TITLES else "")

    with gr.Tabs():
        with gr.Tab("Lab"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=320):
                    gr.Markdown("#### Input — real phone RAW or a photo")
                    gal = gr.Gallery(EX_GALLERY, columns=3, height=420, allow_preview=False, show_label=False,
                                     elem_id="examples", object_fit="cover")
                    up = gr.File(label="Or upload your own (phone DNG/RAW, .npz, JPG/PNG)", type="filepath")
                    with gr.Accordion("Settings", open=False):
                        prec = gr.Radio(list(ENGINES), value="INT8 (quantised)", label="Neural precision")
                        cond = gr.Dropdown(list(CONDITIONS) + ["custom"], value="low_light",
                                           label="Simulated sensor condition (JPG/PNG only)")
                        exp = gr.Slider(0.03, 1.0, 0.25, label="Custom: exposure")
                        noi = gr.Slider(0.0001, 0.02, 0.004, label="Custom: shot-noise level")
                        wbc = gr.Slider(-1.0, 1.0, 0.6, label="Custom: illuminant cast (− shade · + tungsten)")
                        seed = gr.Number(7, label="Noise seed", precision=0)
                    btn = gr.Button("Run pipelines", variant="primary", size="lg")
                with gr.Column(scale=3):
                    head = gr.HTML("<div class='hl'><div class='hl-title'>Pick an example to start</div>"
                                   "<div class='hl-main'>Each input runs through the classical ISP, the pure "
                                   "learned ISP and the proposed hybrid. Drag the slider to compare any two.</div></div>")
                    with gr.Row():
                        left = gr.Dropdown(VIEWS, value="Classical ISP", show_label=False, scale=4)
                        swap = gr.Button("⇄", elem_id="swap", scale=0)
                        right = gr.Dropdown(VIEWS, value="Hybrid ISP", show_label=False, scale=4)
                    slider = gr.ImageSlider(show_label=False, height=560, max_height=560,
                                            buttons=["download", "fullscreen"])
                    strip = gr.Gallery(columns=5, height=170, allow_preview=True, show_label=False,
                                       object_fit="cover")
                    gr.Markdown("#### Quality vs reference")
                    metrics = gr.HTML()
                    gr.Markdown("#### Adaptive controller decisions")
                    ctl = gr.HTML()
                    with gr.Accordion("Per-stage latency", open=False):
                        stg = gr.Dataframe(headers=["Pipeline", "Stage", "ms"], interactive=False)

        with gr.Tab("Results"):
            gr.Markdown(RESULTS_MD)
            for p, cap in FIGS:
                gr.Image(p, label=cap, show_label=True, interactive=False, buttons=["fullscreen"])

    inputs = [path_state, title_state, cond, exp, noi, wbc, seed, prec, left, right]
    outputs = [views_state, slider, head, metrics, ctl, stg, strip]
    sp = dict(show_progress_on=[slider])
    gal.select(pick_example, None, [path_state, title_state]).then(process, inputs, outputs, **sp)
    up.upload(uploaded, up, [path_state, title_state]).then(process, inputs, outputs, **sp)
    btn.click(process, inputs, outputs, **sp)
    prec.change(process, inputs, outputs, **sp)
    left.change(change_view, [views_state, left, right], slider)
    right.change(change_view, [views_state, left, right], slider)
    swap.click(lambda a, b: (b, a), [left, right], [left, right])
    demo.load(process, inputs, outputs, **sp)

if __name__ == "__main__":
    demo.queue(max_size=16).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)),
                                   css=CSS, theme=THEME)
