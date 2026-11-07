"""Train the learned components.

    python train.py --model hybrid   --iters 3000
    python train.py --model learned  --iters 3000
    python train.py --model hybrid_nonoise / hybrid_noglobal   (ablations)

Data: real smartphone RAW from SIDD (see isp/sidd.py). Train and validation use different
capture instances (101 / 14, stratified per phone); the official SIDD benchmark blocks are
used only by evaluate.py.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from isp.sidd import build_sidd_bank, split_names, train_names
from isp.models import HybridNet, LearnedISP, count_params

ROOT = os.path.dirname(os.path.abspath(__file__))


def ssim_t(x, y, C1=0.01 ** 2, C2=0.03 ** 2):
    w = torch.ones(3, 1, 7, 7, device=x.device) / 49
    mu_x, mu_y = F.conv2d(x, w, groups=3), F.conv2d(y, w, groups=3)
    sxx = F.conv2d(x * x, w, groups=3) - mu_x ** 2
    syy = F.conv2d(y * y, w, groups=3) - mu_y ** 2
    sxy = F.conv2d(x * y, w, groups=3) - mu_x * mu_y
    s = ((2 * mu_x * mu_y + C1) * (2 * sxy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sxx + syy + C2))
    return s.mean()


def make_model(name):
    if name == "learned":
        return LearnedISP()
    if name == "hybrid":
        return HybridNet()
    if name == "hybrid_nonoise":
        return HybridNet(use_noise_map=False)
    if name == "hybrid_noglobal":
        return HybridNet(use_global=False)
    raise ValueError(name)


def get_bank(split, n=None, seed=0):
    """SIDD patch bank (real phone RAW -> clean sRGB), cached as .npy and memory-mapped.
    train: 101 capture instances x 30 crops x 2 patches; val: 14 held-out instances x 8 x 2."""
    d = os.path.join(ROOT, "checkpoints", f"bank_sidd_{split}")
    keys = ["packed", "nm_l", "front", "nm_h", "thumb", "target"]
    if all(os.path.exists(os.path.join(d, k + ".npy")) for k in keys):
        return {k: np.load(os.path.join(d, k + ".npy"), mmap_mode="r") for k in keys}
    t = time.time()
    tr, va = split_names(train_names())
    if split == "train":
        bank = build_sidd_bank(tr, crops_per_frame=30, seed=seed)
    else:
        bank = build_sidd_bank(va, crops_per_frame=8, seed=seed)
    os.makedirs(d, exist_ok=True)
    for k in keys:
        np.save(os.path.join(d, k + ".npy"), bank[k])
    print(f"built {split} bank ({len(bank['target'])} patches) in {time.time() - t:.0f}s", flush=True)
    return bank


def to_t(a, idx):
    return torch.from_numpy(a[idx].astype(np.float32)).permute(0, 3, 1, 2)


def forward(model, name, bank, idx):
    if name == "learned":
        return model(to_t(bank["packed"], idx), to_t(bank["nm_l"], idx))
    return model(to_t(bank["front"], idx), to_t(bank["nm_h"], idx), to_t(bank["thumb"], idx))


def evaluate(model, name, bank, bs=32):
    model.eval()
    ps = []
    with torch.no_grad():
        for i in range(0, len(bank["target"]), bs):
            idx = np.arange(i, min(i + bs, len(bank["target"])))
            y = forward(model, name, bank, idx)
            t = to_t(bank["target"], idx)
            mse = ((y - t) ** 2).mean((1, 2, 3))
            ps += (-10 * torch.log10(mse + 1e-10)).tolist()
    model.train()
    return float(np.mean(ps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hybrid")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    os.makedirs(os.path.join(ROOT, "checkpoints"), exist_ok=True)

    train = get_bank("train", seed=0)
    val = get_bank("val", seed=123)
    model = make_model(args.model)
    print(f"{args.model}: {count_params(model):,} params", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.iters, pct_start=0.1)
    rng = np.random.default_rng(0)
    best, log, t0 = -1, [], time.time()
    ckpt = os.path.join(ROOT, "checkpoints", f"{args.model}.pt")
    for it in range(1, args.iters + 1):
        idx = rng.integers(0, len(train["target"]), args.batch)
        y = forward(model, args.model, train, idx)
        t = to_t(train["target"], idx)
        loss = F.l1_loss(y, t) + 0.15 * (1 - ssim_t(y, t))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if it % 250 == 0 or it == args.iters:
            vp = evaluate(model, args.model, val)
            log.append(dict(iter=it, loss=float(loss), val_psnr=vp, sec=time.time() - t0))
            print(f"[{args.model}] it {it} loss {loss.item():.4f} val_psnr {vp:.2f} ({time.time()-t0:.0f}s)", flush=True)
            if vp > best:
                best = vp
                torch.save(model.state_dict(), ckpt)
    json.dump(dict(model=args.model, params=count_params(model), best_val_psnr=best, log=log, args=vars(args)),
              open(os.path.join(ROOT, "results", f"train_{args.model}.json"), "w"), indent=1)
    print("best val psnr", best)


if __name__ == "__main__":
    main()
