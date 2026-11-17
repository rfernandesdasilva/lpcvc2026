"""On-device measurement on Qualcomm AI Hub: compile -> profile (xN) -> inference -> score, for several models.

  python scripts/aihub_run.py --exp s22-profile-candidates --models clip-b16 mobileclip2-s0 --profile-runs 3

All jobs are submitted up front, so AI Hub runs them in parallel. Per model and encoder this stores the
compile/profile/inference job IDs and URLs, every downloaded profile (per-layer detail included), and the
device embeddings, all under experiments/NNN_<exp>/aihub/<model>/.

What each number means:
  img_ms / txt_ms   median over profile runs of AI Hub `estimated_inference_time` (100 iterations per run), and the
                    LPCVC latency rule applies to their sum. It excludes external preprocessing and ranking, and
                    includes anything inside the graph (resize, normalization, token re-padding).
  device frac R@10  official score on the LPCVC sample, computed from embeddings produced ON the device.
  fidelity          cosine(device embedding, local PyTorch FP32 embedding) on the same inputs; it shows whether
                    local accuracy numbers (e.g. RefCOCOg) carry over to the deployed model.

Prereq (once per machine): `& .\\scripts\\configure_aihub.ps1` (stores your AI Hub token, hidden input).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import qai_hub as hub
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from export_onnx import export  # noqa: E402
from lpcvc import metrics  # noqa: E402
from lpcvc.data import load_lpcvc_sample  # noqa: E402
from lpcvc.experiment import EXP_DIR, Experiment  # noqa: E402
from lpcvc.models import build  # noqa: E402

DEVICE = "Samsung Galaxy S22 (Family)"
COMPILE_OPTS = "--target_runtime qnn_dlc --truncate_64bit_io"   # identical to the official sample
DATASET_CACHE = ROOT / ".cache" / "aihub_datasets.json"
ENCODERS = {"image": ("image_encoder.onnx", (1, 3, 224, 224)), "text": ("text_encoder.onnx", ((1, 77), "int64"))}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_datasets(ds) -> dict[str, str]:
    cache = json.loads(DATASET_CACHE.read_text()) if DATASET_CACHE.exists() else {}
    if ds.name not in cache:
        print(f"uploading {ds.name} to AI Hub ...")
        cache[ds.name] = {"image": hub.upload_dataset({"image": [x[None] for x in ds.images()]}).dataset_id,
                          "text": hub.upload_dataset({"text": [x[None] for x in ds.tokens()]}).dataset_id}
        DATASET_CACHE.parent.mkdir(parents=True, exist_ok=True)
        DATASET_CACHE.write_text(json.dumps(cache, indent=2))
    return cache[ds.name]


@torch.no_grad()
def local_embeddings(model: str, images: np.ndarray, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ie, te = build(model, dev)
    a = torch.cat([ie(torch.from_numpy(images[i:i + 32]).to(dev)) for i in range(0, len(images), 32)])
    b = torch.cat([te(torch.from_numpy(tokens[i:i + 32]).to(dev)) for i in range(0, len(tokens), 32)])
    return a.cpu().numpy(), b.cpu().numpy()


def local_refcocog_scores() -> dict[str, float]:
    """Pick up local RefCOCOg-LPCVC scores from the latest local screening experiment, for the summary table."""
    out = {}
    for p in sorted(EXP_DIR.glob("[0-9][0-9][0-9]_*/results.json")):
        for r in json.loads(p.read_text(encoding="utf-8")).get("rows", []):
            if r.get("set") == "refcocog":
                out[r["model"]] = (r["frac_R@10"], p.parent.name)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--profile-runs", type=int, default=3)
    ap.add_argument("--data", default=str(ROOT / "data" / "lpcvc_sample"))
    ap.add_argument("--exp", default="s22-profile")
    ap.add_argument("--title", default="On-device profiling on the Galaxy S22")
    ap.add_argument("--question", default="What are the real S22 latencies and on-device accuracy of these models?")
    args = ap.parse_args()
    with Experiment(args.exp, args.title, args.question, args) as exp:
        run(args, exp)


def run(args, exp):
    device = hub.Device(args.device)
    ds = load_lpcvc_sample(args.data)
    images, tokens = ds.images(), ds.tokens()
    dataset_ids = get_datasets(ds)
    exp.manifest["aihub"] = {"device": args.device, "compile_options": COMPILE_OPTS, "datasets": dataset_ids}
    print(f"device={args.device}  datasets={dataset_ids}")

    # 1) export + 2) submit compile jobs (identical ONNX files compile once)
    plan, compile_jobs = {}, {}
    for m in args.models:
        tag = m.replace("@", "_")
        onnx_dir = ROOT / "exported" / tag
        if not all((onnx_dir / f).exists() for f, _ in ENCODERS.values()):
            print(f"exporting {m} ...")
            export(m, onnx_dir)
        plan[m] = {}
        for enc, (fname, spec) in ENCODERS.items():
            p = onnx_dir / fname
            digest = sha256(p)
            if digest not in compile_jobs:
                compile_jobs[digest] = hub.submit_compile_job(
                    model=str(p), device=device, input_specs={enc: spec}, options=COMPILE_OPTS,
                    name=f"{tag}_{enc}")
                print(f"compile {tag}/{enc}: {compile_jobs[digest].url}")
            plan[m][enc] = {"onnx": str(p.relative_to(ROOT)), "onnx_MiB": p.stat().st_size / 2**20,
                            "onnx_sha256": digest, "compile_job": compile_jobs[digest]}

    # 3) as each compile finishes: submit profile x N + inference (also deduped by compiled model)
    downstream = {}
    for m in args.models:
        for enc, e in plan[m].items():
            cj = e["compile_job"]
            if e["onnx_sha256"] in downstream:
                e.update(downstream[e["onnx_sha256"]])
                continue
            target = cj.get_target_model()
            if target is None:
                e["error"] = f"compile failed: {cj.get_status().message} ({cj.url})"
                print(f"!! {m}/{enc}: {e['error']}")
                downstream[e["onnx_sha256"]] = {"error": e["error"]}
                continue
            d = {"profile_jobs": [hub.submit_profile_job(model=target, device=device,
                                                         options="--max_profiler_iterations 100",
                                                         name=f"{m}_{enc}_profile{i}")
                                  for i in range(args.profile_runs)],
                 "inference_job": hub.submit_inference_job(model=target, device=device,
                                                           inputs=hub.get_dataset(dataset_ids[enc]),
                                                           name=f"{m}_{enc}_inference")}
            e.update(d)
            downstream[e["onnx_sha256"]] = d

    # 4) collect, 5) score
    rows, refc = [], local_refcocog_scores()
    for m in args.models:
        tag = m.replace("@", "_")
        row = {"model": m}
        emb = {}
        try:
            for enc, e in plan[m].items():
                if "error" in e:
                    raise RuntimeError(e["error"])
                times, mems = [], []
                for i, pj in enumerate(e["profile_jobs"]):
                    prof = pj.download_profile()
                    exp.save_json(f"aihub/{tag}/{enc}_profile{i}.json", prof)
                    s = prof["execution_summary"]
                    times.append(s["estimated_inference_time"] / 1000.0)
                    mems.append(s.get("estimated_inference_peak_memory", 0) / 2**20)
                    units = [d.get("compute_unit") for d in prof.get("execution_detail", [])]
                out = e["inference_job"].download_output_data()
                emb[enc] = np.concatenate([np.asarray(x).reshape(1, -1) for x in next(iter(out.values()))])
                row[enc] = {
                    "ms_median": float(np.median(times)), "ms_runs": times, "peak_mem_MiB": float(np.max(mems)),
                    "layers_by_unit": {u: units.count(u) for u in sorted(set(units), key=str)},
                    "onnx_MiB": e["onnx_MiB"], "onnx_sha256": e["onnx_sha256"],
                    "compile_job": e["compile_job"].job_id, "compile_url": e["compile_job"].url,
                    "profile_jobs": [pj.job_id for pj in e["profile_jobs"]],
                    "inference_job": e["inference_job"].job_id,
                }
            exp.save_npz(f"aihub/{tag}/device_embeddings.npz", image=emb["image"], text=emb["text"])
            r = metrics.report(emb["image"], emb["text"], ds.gt, ds.txt_ids)
            per = r.pop("per_image")
            li, lt = local_embeddings(m, images, tokens)
            fid_i = (metrics.l2norm(li) * metrics.l2norm(emb["image"])).sum(1)
            fid_t = (metrics.l2norm(lt) * metrics.l2norm(emb["text"])).sum(1)
            r_local = metrics.fractional_recall_at_k(metrics.similarity(li, lt), ds.gt, ds.txt_ids)
            total = row["image"]["ms_median"] + row["text"]["ms_median"]
            row.update({"total_ms": total, "valid_under_35ms": total < 35.0,
                        "device_sample": r, "device_sample_per_image": per,
                        "local_sample_frac_R@10": r_local,
                        "fidelity_cos": {"image_mean": float(fid_i.mean()), "image_min": float(fid_i.min()),
                                         "text_mean": float(fid_t.mean()), "text_min": float(fid_t.min())},
                        "local_refcocog": refc.get(m)})
            print(f"{m:22s} img {row['image']['ms_median']:.3f} + txt {row['text']['ms_median']:.3f} = {total:.3f} ms "
                  f"{'VALID' if total < 35 else 'TOO SLOW'} | device sample R@10 {r['frac_R@10']*100:.2f} "
                  f"(local {r_local*100:.2f}) | fidelity img/txt min {fid_i.min():.4f}/{fid_t.min():.4f}")
        except Exception as ex:
            row["error"] = f"{type(ex).__name__}: {ex}"
            print(f"!! {m}: {row['error']}")
            traceback.print_exc()
        rows.append(row)

    exp.save_json("results.json", {"device": args.device, "compile_options": COMPILE_OPTS,
                                   "datasets": dataset_ids, "rows": rows})

    lines = ["| Model | Image ms | Text ms | **Total ms** | <35 ms | Peak mem MiB (img/txt) | ONNX MiB (img/txt) "
             "| Device frac R@10, sample | RefCOCOg-LPCVC frac R@10 (local) | Fidelity cos min (img/txt) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['model']} | failed: {r['error'][:80]} |" + " |" * 8)
            continue
        i, t = r["image"], r["text"]
        rng = lambda x: f"{x['ms_median']:.2f} ({min(x['ms_runs']):.2f}-{max(x['ms_runs']):.2f})"  # noqa: E731
        refc_s = f"{r['local_refcocog'][0]*100:.2f}" if r.get("local_refcocog") else "n/a"
        lines.append(f"| {r['model']} | {rng(i)} | {rng(t)} | **{r['total_ms']:.2f}** | "
                     f"{'yes' if r['valid_under_35ms'] else 'NO'} | {i['peak_mem_MiB']:.0f}/{t['peak_mem_MiB']:.0f} | "
                     f"{i['onnx_MiB']:.0f}/{t['onnx_MiB']:.0f} | {r['device_sample']['frac_R@10']*100:.2f} | {refc_s} | "
                     f"{r['fidelity_cos']['image_min']:.4f}/{r['fidelity_cos']['text_min']:.4f} |")
    exp.add_table("\n".join(lines) + f"\n\nLatency: median of {args.profile_runs} AI Hub profile runs "
                  "(min-max in parentheses), 100 iterations each. Job IDs and links are in `results.json`.")
    ok = [r for r in rows if "error" not in r and r["valid_under_35ms"]]
    if ok:
        best = max(ok, key=lambda r: (r.get("local_refcocog") or (0,))[0])
        exp.headline = f"best valid by RefCOCOg: {best['model']} ({best['total_ms']:.1f} ms)"


if __name__ == "__main__":
    main()
