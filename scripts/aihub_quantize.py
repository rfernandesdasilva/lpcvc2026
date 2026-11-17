"""Post-training quantization on Qualcomm AI Hub, measured on the S22 and evaluated on RefCOCOg-LPCVC.

  python scripts/aihub_quantize.py --exp s22-int8-quant --models mobileclip2-b mobileclip2-s0 --precisions w8a8 w8a16

For each model x precision x encoder:
  1. AI Hub quantize job (calibration: data/calib_refcocog_train, 256 images / 512 captions, never validation images)
     -> QDQ ONNX, downloaded to exported/<model>/<precision>/
  2. compile (QNN, same options as the sample; float/int IO kept, so the official input interface is unchanged)
  3. profile x N on the S22 + on-device inference on the LPCVC sample
  4. local accuracy: the downloaded QDQ ONNX run in onnxruntime on all RefCOCOg-LPCVC folds + the sample
Then every (image precision, text precision) combination, including fp32, is scored from the stored embeddings,
because image and text encoders tolerate quantization differently and the latencies simply add.
"""
from __future__ import annotations

import argparse
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
from aihub_run import COMPILE_OPTS, DATASET_CACHE, DEVICE, ENCODERS, get_datasets, sha256  # noqa: E402
from eval_local import PRESETS, resolve_sets  # noqa: E402
from export_onnx import export  # noqa: E402
from lpcvc import metrics  # noqa: E402
from lpcvc.data import load_lpcvc_sample, process_image, tokenize  # noqa: E402
from lpcvc.experiment import EXP_DIR, Experiment  # noqa: E402
from lpcvc.models import build  # noqa: E402

PRECISIONS = {"w8a8": (hub.QuantizeDtype.INT8, hub.QuantizeDtype.INT8),
              "w8a16": (hub.QuantizeDtype.INT8, hub.QuantizeDtype.INT16)}
CALIB = ROOT / "data" / "calib_refcocog_train"


def calib_datasets() -> dict[str, str]:
    cache = json.loads(DATASET_CACHE.read_text()) if DATASET_CACHE.exists() else {}
    if "calib_refcocog_train" not in cache:
        names = (CALIB / "images.txt").read_text(encoding="utf-8").split()
        caps = [c for c in (CALIB / "captions.txt").read_text(encoding="utf-8").splitlines() if c.strip()]
        imgs = [process_image(CALIB / "images" / n) for n in names]
        toks = [t[None].astype(np.int64) for t in tokenize(caps)]     # ONNX text input is int64
        print(f"uploading calibration data ({len(imgs)} images, {len(toks)} captions) ...")
        cache["calib_refcocog_train"] = {"image": hub.upload_dataset({"image": imgs}).dataset_id,
                                         "text": hub.upload_dataset({"text": toks}).dataset_id}
        DATASET_CACHE.parent.mkdir(parents=True, exist_ok=True)
        DATASET_CACHE.write_text(json.dumps(cache, indent=2))
    return cache["calib_refcocog_train"]


def find_onnx(path: Path) -> Path:
    if path.is_file() and path.suffix == ".onnx":
        return path
    hits = sorted(Path(path).rglob("*.onnx"))
    if not hits:
        raise FileNotFoundError(f"no .onnx in downloaded {path}")
    return hits[0]


def ort_embed(onnx_path: Path, enc: str, arrays: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    want_int = "int" in sess.get_inputs()[0].type
    out = []
    for i in range(len(arrays)):
        x = arrays[i:i + 1]
        x = x.astype(np.int64) if want_int else x.astype(np.float32)
        out.append(sess.run(None, {name: x})[0].reshape(1, -1))
    return np.concatenate(out)


@torch.no_grad()
def torch_embed(model: str, enc: str, arrays: np.ndarray) -> np.ndarray:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ie, te = build(model, dev)
    m = ie if enc == "image" else te
    return torch.cat([m(torch.from_numpy(arrays[i:i + 64]).to(dev)) for i in range(0, len(arrays), 64)]).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--precisions", nargs="+", default=["w8a8", "w8a16"], choices=list(PRECISIONS))
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--profile-runs", type=int, default=3)
    ap.add_argument("--data", nargs="+", default=["refcocog", "sample"])
    ap.add_argument("--exp", default="s22-quant")
    ap.add_argument("--title", default="Post-training quantization on the S22")
    ap.add_argument("--question", default="How much latency does INT8 PTQ save on the S22, and what does it cost in accuracy?")
    args = ap.parse_args()
    with Experiment(args.exp, args.title, args.question, args) as exp:
        run(args, exp)


def run(args, exp):
    device = hub.Device(args.device)
    sample = load_lpcvc_sample(ROOT / "data" / "lpcvc_sample")
    sample_ids = get_datasets(sample)
    calib_ids = calib_datasets()
    exp.manifest["aihub"] = {"device": args.device, "compile_options": COMPILE_OPTS,
                             "sample_datasets": sample_ids, "calibration_datasets": calib_ids,
                             "calibration": "data/calib_refcocog_train (splits/calib/), 256 images / 512 captions"}

    # ---- 1) submit quantize jobs ------------------------------------------------------------------
    jobs = {}   # (model, precision, enc) -> dict
    for m in args.models:
        tag = m.replace("@", "_")
        onnx_dir = ROOT / "exported" / tag
        if not all((onnx_dir / f).exists() for f, _ in ENCODERS.values()):
            export(m, onnx_dir)
        for p in args.precisions:
            for enc, (fname, spec) in ENCODERS.items():
                wd, ad = PRECISIONS[p]
                qj = hub.submit_quantize_job(model=str(onnx_dir / fname), calibration_data=hub.get_dataset(calib_ids[enc]),
                                             weights_dtype=wd, activations_dtype=ad, name=f"{tag}_{enc}_{p}")
                print(f"quantize {tag}/{enc}/{p}: {qj.url}")
                jobs[(m, p, enc)] = {"quantize_job": qj, "spec": spec, "float_onnx_sha256": sha256(onnx_dir / fname)}

    # ---- 2) download QDQ ONNX, compile, profile, infer --------------------------------------------
    for (m, p, enc), j in jobs.items():
        try:
            qmodel = j["quantize_job"].get_target_model()
            if qmodel is None:
                raise RuntimeError(f"quantize failed: {j['quantize_job'].get_status().message}")
            qdir = ROOT / "exported" / m.replace("@", "_") / p
            qdir.mkdir(parents=True, exist_ok=True)
            j["qdq_onnx"] = find_onnx(Path(qmodel.download(str(qdir / f"{enc}_encoder.onnx"))))
            cj = hub.submit_compile_job(model=qmodel, device=device, input_specs={enc: j["spec"]},
                                        options=COMPILE_OPTS, name=f"{m}_{enc}_{p}")
            print(f"compile {m}/{enc}/{p}: {cj.url}")
            j["compile_job"] = cj
        except Exception as ex:
            j["error"] = f"{type(ex).__name__}: {ex}"
            print(f"!! {m}/{enc}/{p}: {j['error']}")
    for (m, p, enc), j in jobs.items():
        if "error" in j:
            continue
        target = j["compile_job"].get_target_model()
        if target is None:
            j["error"] = f"compile failed: {j['compile_job'].get_status().message} ({j['compile_job'].url})"
            print(f"!! {m}/{enc}/{p}: {j['error']}")
            continue
        j["profile_jobs"] = [hub.submit_profile_job(model=target, device=device, options="--max_profiler_iterations 100",
                                                    name=f"{m}_{enc}_{p}_profile{i}") for i in range(args.profile_runs)]
        j["inference_job"] = hub.submit_inference_job(model=target, device=device,
                                                      inputs=hub.get_dataset(sample_ids[enc]), name=f"{m}_{enc}_{p}_inf")

    # ---- 3) local embeddings for every set: fp32 (PyTorch) and each QDQ ONNX (onnxruntime) --------
    sets = []
    for name, path in resolve_sets(args.data):
        ds = load_lpcvc_sample(path)
        ds.name = name
        sets.append(ds)
    groups = {g: PRESETS.get(g, [g]) for g in args.data}
    emb = {}   # (model, precision, enc, set) -> array
    for m in args.models:
        for ds in sets:
            imgs, toks = ds.images(), ds.tokens()
            emb[(m, "fp32", "image", ds.name)] = torch_embed(m, "image", imgs)
            emb[(m, "fp32", "text", ds.name)] = torch_embed(m, "text", toks)
            for p in args.precisions:
                for enc, arr in (("image", imgs), ("text", toks)):
                    j = jobs[(m, p, enc)]
                    if "qdq_onnx" in j:
                        emb[(m, p, enc, ds.name)] = ort_embed(j["qdq_onnx"], enc, arr)
        print(f"local embeddings done for {m}")

    # ---- 4) collect device results ------------------------------------------------------------------
    enc_rows = []
    fp_lat = float_latencies()
    for (m, p, enc), j in jobs.items():
        row = {"model": m, "precision": p, "encoder": enc}
        if "error" in j:
            row["error"] = j["error"]
            enc_rows.append(row)
            continue
        try:
            tag = m.replace("@", "_")
            times, mems = [], []
            for i, pj in enumerate(j["profile_jobs"]):
                prof = pj.download_profile()
                exp.save_json(f"aihub/{tag}/{p}/{enc}_profile{i}.json", prof)
                s = prof["execution_summary"]
                times.append(s["estimated_inference_time"] / 1000.0)
                mems.append(s.get("estimated_inference_peak_memory", 0) / 2**20)
                units = [d.get("compute_unit") for d in prof.get("execution_detail", [])]
            out = j["inference_job"].download_output_data()
            dev_emb = np.concatenate([np.asarray(x).reshape(1, -1) for x in next(iter(out.values()))])
            exp.save_npz(f"aihub/{tag}/{p}/{enc}_device_embeddings.npz", emb=dev_emb)
            ref_fp = emb[(m, "fp32", enc, "lpcvc_sample")]
            ref_q = emb.get((m, p, enc, "lpcvc_sample"))
            cos = lambda a, b: (metrics.l2norm(a) * metrics.l2norm(b)).sum(1)  # noqa: E731
            row.update({
                "ms_median": float(np.median(times)), "ms_runs": times, "peak_mem_MiB": float(max(mems)),
                "layers_by_unit": {u: units.count(u) for u in sorted(set(units), key=str)},
                "qdq_onnx_MiB": sum(f.stat().st_size for f in j["qdq_onnx"].parent.glob("*")) / 2**20,
                "cos_device_vs_fp32_min": float(cos(dev_emb, ref_fp).min()),
                "cos_device_vs_fp32_mean": float(cos(dev_emb, ref_fp).mean()),
                "cos_device_vs_ort_qdq_min": float(cos(dev_emb, ref_q).min()) if ref_q is not None else None,
                "fp32_ms": fp_lat.get((m, enc)),
                "quantize_job": j["quantize_job"].job_id, "compile_job": j["compile_job"].job_id,
                "profile_jobs": [pj.job_id for pj in j["profile_jobs"]], "inference_job": j["inference_job"].job_id,
            })
            emb[(m, p, enc, "lpcvc_sample@device")] = dev_emb
            print(f"{m:16s} {p:6s} {enc:5s} {row['ms_median']:.3f} ms (fp32 {row['fp32_ms']}) "
                  f"cos(device,fp32) min {row['cos_device_vs_fp32_min']:.4f}  cos(device,ORT-QDQ) min {row['cos_device_vs_ort_qdq_min']}")
        except Exception as ex:
            row["error"] = f"{type(ex).__name__}: {ex}"
            traceback.print_exc()
        enc_rows.append(row)

    # ---- 5) score every image x text precision combination -----------------------------------------
    lat = {(r["model"], r["precision"], r["encoder"]): r["ms_median"] for r in enc_rows if "ms_median" in r}
    for (m, enc), v in fp_lat.items():
        lat[(m, "fp32", enc)] = v
    combos, per_image = [], {}
    for m in args.models:
        precs = ["fp32", *args.precisions]
        base = None
        for pi in precs:
            for pt in precs:
                row = {"model": m, "image": pi, "text": pt,
                       "total_ms": (lat.get((m, pi, "image")) or np.nan) + (lat.get((m, pt, "text")) or np.nan)}
                ok = True
                for g, members in groups.items():
                    if not all((m, pi, "image", s) in emb and (m, pt, "text", s) in emb for s in members):
                        ok = False
                        break
                    per = np.concatenate([metrics.per_image_recall(
                        metrics.similarity(emb[(m, pi, "image", s)], emb[(m, pt, "text", s)]),
                        next(d for d in sets if d.name == s).gt, next(d for d in sets if d.name == s).txt_ids)
                        for s in members])
                    row[f"{g}_frac_R@10"] = float(per.mean())
                    row[f"{g}_ci95"] = metrics.bootstrap_ci(per)
                    per_image[f"{m}|{pi}|{pt}|{g}"] = per
                    if g == args.data[0]:
                        if pi == pt == "fp32":
                            base = per
                        if base is not None:
                            row["delta_vs_fp32"] = metrics.paired_bootstrap_delta(base, per)
                if not ok:
                    continue
                if all((m, pi if e == "image" else pt, e, "lpcvc_sample@device") in emb or
                       (pi if e == "image" else pt) == "fp32" for e in ("image", "text")):
                    ie = emb.get((m, pi, "image", "lpcvc_sample@device"), emb[(m, "fp32", "image", "lpcvc_sample")])
                    te = emb.get((m, pt, "text", "lpcvc_sample@device"), emb[(m, "fp32", "text", "lpcvc_sample")])
                    row["sample_device_frac_R@10"] = metrics.fractional_recall_at_k(
                        metrics.similarity(ie, te), sample.gt, sample.txt_ids)
                combos.append(row)
    exp.save_json("results.json", {"encoders": enc_rows, "combinations": combos})
    exp.save_npz("per_image_recall.npz", **per_image)

    g0 = args.data[0]
    t1 = ["| Model | Precision | Encoder | S22 ms (fp32 ms) | Peak MiB | QDQ ONNX MiB | cos(device, fp32) min / mean "
          "| cos(device, ORT QDQ) min |", "|---|---|---|---|---|---|---|---|"]
    for r in enc_rows:
        if "error" in r:
            t1.append(f"| {r['model']} | {r['precision']} | {r['encoder']} | failed: {r['error'][:90]} | | | | |")
            continue
        cq = "n/a" if r["cos_device_vs_ort_qdq_min"] is None else f"{r['cos_device_vs_ort_qdq_min']:.4f}"
        fp = "n/a" if r["fp32_ms"] is None else f"{r['fp32_ms']:.2f}"
        t1.append(f"| {r['model']} | {r['precision']} | {r['encoder']} | {r['ms_median']:.2f} ({fp}) | "
                  f"{r['peak_mem_MiB']:.0f} | {r['qdq_onnx_MiB']:.0f} | {r['cos_device_vs_fp32_min']:.4f} / "
                  f"{r['cos_device_vs_fp32_mean']:.4f} | {cq} |")
    t2 = [f"| Model | Image | Text | **S22 total ms** | {g0} frac R@10 [95% CI] | Δ vs fp32 [95% CI] | sample (device emb) |",
          "|---|---|---|---|---|---|---|"]
    for r in combos:
        lo, hi = r[f"{g0}_ci95"]
        d = r.get("delta_vs_fp32")
        ds_ = "-" if r["image"] == r["text"] == "fp32" or d is None else \
            f"{d[0]*100:+.2f} [{d[1]*100:+.1f}, {d[2]*100:+.1f}]{' *' if d[1] > 0 or d[2] < 0 else ''}"
        sd = f"{r['sample_device_frac_R@10']*100:.2f}" if "sample_device_frac_R@10" in r else "n/a"
        t2.append(f"| {r['model']} | {r['image']} | {r['text']} | **{r['total_ms']:.2f}** | "
                  f"{r[f'{g0}_frac_R@10']*100:.2f} [{lo*100:.1f}, {hi*100:.1f}] | {ds_} | {sd} |")
    exp.add_table("### Per encoder\n\n" + "\n".join(t1))
    exp.add_table("### Image × text precision combinations\n\n" + "\n".join(t2) +
                  "\n\nAccuracy on RefCOCOg uses onnxruntime on the downloaded QDQ ONNX (simulated quantization); "
                  "the `cos(device, ORT QDQ)` column shows how closely that simulation matches the phone. "
                  "`sample (device emb)` uses real on-device embeddings. `*` = paired CI excludes 0.")
    valid = [r for r in combos if r["total_ms"] < 35]
    if valid:
        b = max(valid, key=lambda r: r[f"{g0}_frac_R@10"])
        exp.headline = (f"best: {b['model']} img={b['image']} txt={b['text']} "
                        f"{b[f'{g0}_frac_R@10']*100:.2f} @ {b['total_ms']:.1f} ms")


def float_latencies() -> dict[tuple[str, str], float]:
    """fp32 S22 latencies from the latest AI Hub profiling experiment that measured each model."""
    out = {}
    for p in sorted(EXP_DIR.glob("[0-9][0-9][0-9]_*/results.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        for r in data.get("rows", []):
            if "image" in r and isinstance(r["image"], dict) and "ms_median" in r["image"]:
                out[(r["model"], "image")] = r["image"]["ms_median"]
                out[(r["model"], "text")] = r["text"]["ms_median"]
    return out


if __name__ == "__main__":
    main()
