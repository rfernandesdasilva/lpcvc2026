"""Local (PyTorch) screening: fractional Recall@10 on a retrieval set + local GPU latency.

Local latency is only for screening. Every claim about speed must come from AI Hub on the S22
(scripts/aihub_run.py), because rankings can flip between GPU and Snapdragon.

  python scripts/eval_local.py --models clip-b16 clip-b16-norm mobileclip2-s0 mobileclip2-s2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lpcvc import metrics  # noqa: E402
from lpcvc.data import load_lpcvc_sample  # noqa: E402
from lpcvc.models import build  # noqa: E402


@torch.no_grad()
def embed(img_enc, txt_enc, images, tokens, device, bs=64):
    ie = torch.cat([img_enc(torch.from_numpy(images[i:i + bs]).to(device)) for i in range(0, len(images), bs)])
    te = torch.cat([txt_enc(torch.from_numpy(tokens[i:i + bs]).to(device)) for i in range(0, len(tokens), bs)])
    return ie.float().cpu().numpy(), te.float().cpu().numpy()


@torch.no_grad()
def time_batch1(mod, x, warmup=20, iters=100):
    for _ in range(warmup):
        mod(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mod(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["clip-b16", "clip-b16-norm", "mobileclip2-s0"])
    ap.add_argument("--data", default=str(ROOT / "data" / "lpcvc_sample"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "runs"))
    args = ap.parse_args()

    ds = load_lpcvc_sample(args.data)
    images, tokens = ds.images(), ds.tokens()
    print(f"{ds.name}: {len(ds)} images, {len(ds.captions)} captions")

    rows, per_image = [], {}
    for name in args.models:
        img_enc, txt_enc = build(name, args.device)
        ie, te = embed(img_enc, txt_enc, images, tokens, args.device)
        r = metrics.report(ie, te, ds.gt, ds.txt_ids)
        per_image[name] = r.pop("per_image")
        row = {"model": name, **r,
               "img_params_M": sum(p.numel() for p in img_enc.parameters()) / 1e6,
               "txt_params_M": sum(p.numel() for p in txt_enc.parameters()) / 1e6}
        if not args.no_timing:
            row["gpu_img_ms"] = time_batch1(img_enc, torch.from_numpy(images[:1]).to(args.device))
            row["gpu_txt_ms"] = time_batch1(txt_enc, torch.from_numpy(tokens[:1]).to(args.device))
        rows.append(row)
        lo, hi = row["frac_R@10_ci95"]
        print(f"{name:24s} fracR@10={row['frac_R@10']*100:6.2f}  [{lo*100:5.1f},{hi*100:5.1f}]  "
              f"hit@10={row['hit@10']*100:6.2f}  hit@1={row['hit@1']*100:6.2f}"
              + (f"  gpu img/txt ms={row['gpu_img_ms']:.2f}/{row['gpu_txt_ms']:.2f}" if 'gpu_img_ms' in row else ""))
        del img_enc, txt_enc
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    base = args.models[0]
    for name in args.models[1:]:
        d, lo, hi = metrics.paired_bootstrap_delta(per_image[base], per_image[name])
        print(f"  {name} - {base}: {d*100:+.2f} pts  (95% paired CI {lo*100:+.1f} .. {hi*100:+.1f})")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"dataset": ds.name, "n_images": len(ds), "n_captions": len(ds.captions), "notes": ds.notes,
            "device": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
            "torch": torch.__version__}
    (out / f"{stamp}_local.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    np.savez(out / f"{stamp}_local_per_image.npz", **per_image)
    print(f"saved {out / (stamp + '_local.json')}")


if __name__ == "__main__":
    main()
