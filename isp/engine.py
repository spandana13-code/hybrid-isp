"""Inference engine: runs the classical, learned and hybrid ISPs on one RAW frame.

Neural stages run with ONNX Runtime (FP32 or INT8-quantised graphs in ./models), so the
deployed app does not need PyTorch.
"""
from __future__ import annotations

import os
import time
import numpy as np

from .classical import classical_isp, sharpen, Timer
from .data import hybrid_inputs, learned_inputs
from .raw_sim import BLACK_LEVEL, WHITE_LEVEL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")


def _pad(x: np.ndarray, mult: int):
    h, w = x.shape[:2]
    ph, pw = (-h) % mult, (-w) % mult
    if ph or pw:
        x = np.pad(x, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    return x, h, w


class OnnxModel:
    def __init__(self, path: str, threads: int = 0):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.names = [i.name for i in self.sess.get_inputs()]
        self.path = path
        self.size_mb = os.path.getsize(path) / 2 ** 20

    def __call__(self, *arrays: np.ndarray) -> np.ndarray:
        feed = {n: a.transpose(2, 0, 1)[None].astype(np.float32) for n, a in zip(self.names, arrays)}
        return self.sess.run(None, feed)[0][0].transpose(1, 2, 0)


class ISPEngine:
    def __init__(self, precision: str = "int8", model_dir: str = MODEL_DIR, threads: int = 0):
        self.precision = precision
        self.models = {}
        for name in ("hybrid", "learned"):
            p = os.path.join(model_dir, f"{name}_{precision}.onnx")
            if os.path.exists(p):
                self.models[name] = OnnxModel(p, threads)

    def available(self, name):
        return name in self.models

    # ------------------------------------------------------------------ pipelines
    # `fe` = front-end settings: ccm, wb, gain, knee (see isp.sidd.fe_settings for camera RAW)
    def classical(self, raw, **fe):
        return classical_isp(raw, **fe)

    def learned(self, raw, info=None, **fe):
        t = Timer()
        with t("preprocess+analyze"):
            packed, nm, info = learned_inputs(raw, info, wb=fe.get("wb"), gain=fe.get("gain"))
        with t("neural_net"):
            pp, h, w = _pad(packed, 4)
            nn_, _, _ = _pad(nm, 4)
            y = self.models["learned"](pp, nn_)[: 2 * h, : 2 * w]
        return np.clip(y, 0, 1), info, t.t

    def hybrid(self, raw, post_sharpen=True, **fe):
        t = Timer()
        x, nm, thumb, info, t = hybrid_inputs(raw, timer=t, **fe)
        with t("neural_net"):
            xp, h, w = _pad(x, 8)
            np_, _, _ = _pad(nm, 8)
            y = self.models["hybrid"](xp, np_, thumb)[:h, :w]
        with t("adaptive_sharpen"):
            if post_sharpen:
                # the network already restores detail; add only a light, noise-aware touch
                y = sharpen(np.clip(y, 0, 1), 0.5 * info["sharpen"])
        return np.clip(y, 0, 1), info, t.t
