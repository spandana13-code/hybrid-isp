"""Model optimisation + efficiency measurement.

1. Export FP32 ONNX graphs (dynamic H/W).
2. Reduced precision: FP16 weights (size) .
3. Post-training static INT8 quantisation (QDQ, per-channel weights) with calibration
   patches drawn from the training distribution.
4. Measure params, file size, latency (median of N runs) and peak memory per precision.

    python optimize.py
"""
import json
import os
import subprocess
import sys

import numpy as np
import torch

from isp.models import HybridNet, LearnedISP, count_params

ROOT = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(ROOT, "models")
CK = os.path.join(ROOT, "checkpoints")


def export(name, model):
    model.load_state_dict(torch.load(os.path.join(CK, f"{name}.pt"), map_location="cpu"))
    model.eval()
    if name == "learned":
        args = (torch.rand(1, 4, 64, 64), torch.rand(1, 1, 64, 64))
        names = ["packed", "noise"]
    else:
        args = (torch.rand(1, 3, 128, 128), torch.rand(1, 1, 128, 128), torch.rand(1, 4, 64, 64))
        names = ["front", "noise", "thumb"]
    path = os.path.join(MD, f"{name}_fp32.onnx")
    dyn = {n: {2: "h", 3: "w"} for n in names[:2]}
    dyn["out"] = {2: "H", 3: "W"}
    torch.onnx.export(model, args, path, input_names=names, output_names=["out"],
                      dynamic_axes=dyn, opset_version=17, dynamo=False)
    return path, count_params(model)


def fp16(src, dst):
    import onnx
    from onnx import numpy_helper
    m = onnx.load(src)
    # weight-only FP16 storage: initialisers stored in half precision, cast back at load time
    from onnxconverter_common import float16
    m16 = float16.convert_float_to_float16(m, keep_io_types=True)
    onnx.save(m16, dst)


class Calib:
    def __init__(self, name, n=96):
        from train import get_bank
        bank = get_bank("train")
        rng = np.random.default_rng(7)
        idx = rng.choice(len(bank["target"]), n, replace=False)
        if name == "learned":
            a, b, names = bank["packed"][idx], bank["nm_l"][idx], ["packed", "noise"]
        else:
            a, b, names = bank["front"][idx], bank["nm_h"][idx], ["front", "noise"]
        self.items = [{names[0]: a[i:i + 1].astype(np.float32).transpose(0, 3, 1, 2),
                       names[1]: b[i:i + 1].astype(np.float32).transpose(0, 3, 1, 2)} for i in range(n)]
        if name != "learned":
            th = bank["thumb"][idx]
            for i, it in enumerate(self.items):
                it["thumb"] = th[i:i + 1].astype(np.float32).transpose(0, 3, 1, 2)
        self.i = 0

    def get_next(self):
        if self.i >= len(self.items):
            return None
        self.i += 1
        return self.items[self.i - 1]

    def rewind(self):
        self.i = 0


def int8(name, src, dst):
    from onnxruntime.quantization import (quantize_static, QuantType, QuantFormat,
                                          CalibrationDataReader, CalibrationMethod)
    from onnxruntime.quantization.shape_inference import quant_pre_process
    pre = src.replace(".onnx", "_pre.onnx")
    quant_pre_process(src, pre, skip_symbolic_shape=True)

    class R(CalibrationDataReader):
        def __init__(self):
            self.c = Calib(name)

        def get_next(self):
            return self.c.get_next()

    import onnx
    # Mixed precision: INT8 for the U-Net (>99 % of the compute); the tiny global colour head
    # and the linear-light colour transform stay FP32 because a 3x3 colour matrix is very
    # sensitive to quantisation error (measured: -0.45 dB when fully quantised).
    names = [n.name for n in onnx.load(pre).graph.node]
    exclude = [n for n in names if not n.startswith("/unet/")]
    quantize_static(pre, dst, R(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.Percentile, nodes_to_exclude=exclude,
                    extra_options={"CalibPercentile": 99.99})
    os.remove(pre)


BENCH = r'''
import sys, time, json, threading, psutil, numpy as np, onnxruntime as ort
ort.set_default_logger_severity(3)
proc = psutil.Process()
peak = [0]
def watch():
    while True:
        peak[0] = max(peak[0], proc.memory_info().rss); time.sleep(0.002)
threading.Thread(target=watch, daemon=True).start()
path, kind, H, W, threads = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
so = ort.SessionOptions(); so.intra_op_num_threads = threads
time.sleep(0.05); base = proc.memory_info().rss
s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
names = [i.name for i in s.get_inputs()]
if kind == "learned":
    feed = {names[0]: np.random.rand(1, 4, H // 2, W // 2).astype(np.float32), names[1]: np.random.rand(1, 1, H // 2, W // 2).astype(np.float32)}
else:
    feed = {names[0]: np.random.rand(1, 3, H, W).astype(np.float32), names[1]: np.random.rand(1, 1, H, W).astype(np.float32), names[2]: np.random.rand(1, 4, 64, 64).astype(np.float32)}
for _ in range(2): s.run(None, feed)
ts = []
for _ in range(7):
    t = time.perf_counter(); s.run(None, feed); ts.append((time.perf_counter() - t) * 1000)
time.sleep(0.05)
print(json.dumps(dict(latency_ms=float(np.median(ts)), peak_mem_mb=(peak[0] - base) / 2**20)))
'''


def bench(path, kind, H, W, threads=2):
    out = subprocess.check_output([sys.executable, "-c", BENCH, path, kind, str(H), str(W), str(threads)])
    return json.loads(out.decode().strip().splitlines()[-1])


def main():
    os.makedirs(MD, exist_ok=True)
    report = {}
    for name, model in (("hybrid", HybridNet()), ("learned", LearnedISP())):
        p32, params = export(name, model)
        p16 = os.path.join(MD, f"{name}_fp16.onnx")
        p8 = os.path.join(MD, f"{name}_int8.onnx")
        try:
            fp16(p32, p16)
        except Exception as e:  # onnxconverter-common missing
            print("fp16 export skipped:", e)
            p16 = None
        int8(name, p32, p8)
        report[name] = dict(params=params)
        for prec, p in (("fp32", p32), ("fp16", p16), ("int8", p8)):
            if p is None:
                continue
            r = dict(size_mb=os.path.getsize(p) / 2 ** 20)
            for H, W in ((512, 512), (1024, 1024)):
                r[f"{H}x{W}"] = bench(p, name, H, W)
            report[name][prec] = r
            print(name, prec, json.dumps(r), flush=True)
    json.dump(report, open(os.path.join(ROOT, "results", "optimization.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
