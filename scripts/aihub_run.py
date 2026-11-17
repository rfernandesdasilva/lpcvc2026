"""Compile -> profile -> on-device inference -> official score, on Qualcomm AI Hub. One command per model.

  python scripts/aihub_run.py --model mobileclip2-s0            # Galaxy S22, QNN (same options as the sample)
  python scripts/aihub_run.py --model clip-b16 --profile-runs 3 # repeat profiles to get latency spread

Prereq (once per machine): `qai-hub configure --api_token <YOUR_TOKEN>` (token from https://app.aihub.qualcomm.com).
The sample's image/text inputs are model-independent, so the datasets are uploaded once and their IDs
cached in results/aihub_datasets.json. Every run writes results/runs/<stamp>_aihub_<model>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import qai_hub as hub

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lpcvc import metrics  # noqa: E402
from lpcvc.data import load_lpcvc_sample  # noqa: E402

DEVICE = "Samsung Galaxy S22 (Family)"
COMPILE_OPTS = "--target_runtime qnn_dlc --truncate_64bit_io"   # identical to the official sample
DATASET_CACHE = ROOT / "results" / "aihub_datasets.json"


def get_datasets(ds) -> tuple[hub.Dataset, hub.Dataset]:
    cache = json.loads(DATASET_CACHE.read_text()) if DATASET_CACHE.exists() else {}
    if ds.name not in cache:
        images = [x[None] for x in ds.images()]           # list of (1,3,224,224) float32
        tokens = [x[None] for x in ds.tokens()]           # list of (1,77) int32
        cache[ds.name] = {"image": hub.upload_dataset({"image": images}).dataset_id,
                          "text": hub.upload_dataset({"text": tokens}).dataset_id}
        DATASET_CACHE.parent.mkdir(parents=True, exist_ok=True)
        DATASET_CACHE.write_text(json.dumps(cache, indent=2))
    return hub.get_dataset(cache[ds.name]["image"]), hub.get_dataset(cache[ds.name]["text"])


def profile(target_model, device, runs: int) -> list[dict]:
    out = []
    for _ in range(runs):
        job = hub.submit_profile_job(model=target_model, device=device, options="--max_profiler_iterations 100")
        summ = job.download_profile()["execution_summary"]
        out.append({"job_id": job.job_id,
                    "ms": summ["estimated_inference_time"] / 1000.0,
                    "peak_mem_MiB": summ.get("estimated_inference_peak_memory", 0) / 2**20})
    return out


def run_one(onnx_path: Path, input_name: str, spec, device, dataset, runs: int) -> dict:
    cjob = hub.submit_compile_job(model=str(onnx_path), device=device,
                                  input_specs={input_name: spec}, options=COMPILE_OPTS)
    target = cjob.get_target_model()
    if target is None:
        raise RuntimeError(f"compile failed: {cjob.url}")
    profs = profile(target, device, runs)
    ijob = hub.submit_inference_job(model=target, device=device, inputs=dataset)
    outputs = ijob.download_output_data()
    emb = np.concatenate([np.asarray(x).reshape(1, -1) for x in next(iter(outputs.values()))])
    return {"compile_job": cjob.job_id, "inference_job": ijob.job_id, "profiles": profs, "embeddings": emb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--onnx-dir", default=None, help="defaults to exported/<model>; exported on the fly if missing")
    ap.add_argument("--data", default=str(ROOT / "data" / "lpcvc_sample"))
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--profile-runs", type=int, default=1)
    args = ap.parse_args()

    onnx_dir = Path(args.onnx_dir or ROOT / "exported" / args.model.replace("@", "_"))
    if not (onnx_dir / "image_encoder.onnx").exists():
        from export_onnx import export
        export(args.model, onnx_dir)

    device = hub.Device(args.device)
    ds = load_lpcvc_sample(args.data)
    img_ds, txt_ds = get_datasets(ds)

    img = run_one(onnx_dir / "image_encoder.onnx", "image", (1, 3, 224, 224), device, img_ds, args.profile_runs)
    txt = run_one(onnx_dir / "text_encoder.onnx", "text", ((1, 77), "int64"), device, txt_ds, args.profile_runs)

    r = metrics.report(img.pop("embeddings"), txt.pop("embeddings"), ds.gt, ds.txt_ids)
    per = r.pop("per_image")
    img_ms = float(np.median([p["ms"] for p in img["profiles"]]))
    txt_ms = float(np.median([p["ms"] for p in txt["profiles"]]))
    total = img_ms + txt_ms
    row = {"model": args.model, "device": args.device, "compile_opts": COMPILE_OPTS, **r,
           "img_ms": img_ms, "txt_ms": txt_ms, "total_ms": total, "valid_under_35ms": total < 35.0,
           "onnx_MiB": {p.name: p.stat().st_size / 2**20 for p in onnx_dir.glob("*.onnx")},
           "image": img, "text": txt}
    lo, hi = r["frac_R@10_ci95"]
    print(f"{args.model} on {args.device}: fracR@10={r['frac_R@10']*100:.2f} [{lo*100:.1f},{hi*100:.1f}]  "
          f"img {img_ms:.3f} ms + txt {txt_ms:.3f} ms = {total:.3f} ms  ({'VALID' if total < 35 else 'TOO SLOW'})")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = ROOT / "results" / "runs"
    out.mkdir(parents=True, exist_ok=True)
    tag = args.model.replace("@", "_")
    (out / f"{stamp}_aihub_{tag}.json").write_text(json.dumps(row, indent=2))
    np.save(out / f"{stamp}_aihub_{tag}_per_image.npy", per)
    print(f"saved {out / f'{stamp}_aihub_{tag}.json'}")


if __name__ == "__main__":
    main()
