"""Local (PyTorch) screening: fractional Recall@10 on a retrieval set + local GPU latency.

Local latency is only for screening. Every claim about speed must come from AI Hub on the S22
(scripts/aihub_run.py), because rankings can flip between GPU and Snapdragon.

  python scripts/eval_local.py --exp local-screen-xyz --title "..." --question "..."       --models clip-b16 mobileclip2-s0 --data refcocog sample
Results go to experiments/NNN_<exp>/ (see lpcvc/experiment.py).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lpcvc import metrics  # noqa: E402
from lpcvc.data import load_lpcvc_sample  # noqa: E402
from lpcvc.experiment import Experiment  # noqa: E402
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


PRESETS = {
    "sample": ["lpcvc_sample"],
    "refcocog": [f"refcocog_val/fold{k}" for k in range(5)],
    "test": ["refcoco_test/fold0"],   # held-out; final report only (scripts/build_test_set.py)
}


def resolve_sets(specs: list[str], final: bool = False) -> list[tuple[str, Path]]:
    out = []
    for s in specs:
        for rel in PRESETS.get(s, [s]):
            if "refcoco_test" in rel and not final:
                sys.exit("the held-out test fold is scored only for the final report: eval_local.py --data test --final")
            p = Path(rel) if Path(rel).is_absolute() else ROOT / "data" / rel
            out.append((rel, p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["clip-b16", "clip-b16-norm", "mobileclip2-s0"])
    ap.add_argument("--data", nargs="+", default=["refcocog", "sample"],
                    help="presets (sample, refcocog, test) or dirs under data/ in the official layout")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--final", action="store_true", help="allow the held-out test fold (final report only)")
    ap.add_argument("--exp", default="local-screen", help="experiment slug (folder experiments/NNN_<slug>)")
    ap.add_argument("--title", default="Local screening")
    ap.add_argument("--question", default="How do these models rank on the validation sets (local PyTorch FP32)?")
    args = ap.parse_args()
    resolve_sets(args.data, args.final)   # refuse the test fold before an experiment folder is created
    with Experiment(args.exp, args.title, args.question, args) as exp:
        run(args, exp)


def run(args, exp):

    sets = []
    for name, path in resolve_sets(args.data, args.final):
        ds = load_lpcvc_sample(path)
        ds.name = name
        sets.append((ds, ds.images(), ds.tokens()))
        print(f"{name}: {len(ds)} images, {len(ds.captions)} captions")
    groups = {}  # report group -> list of set names (folds of one preset are pooled)
    for spec in args.data:
        groups[spec] = PRESETS.get(spec, [spec])

    rows, per_image = [], {}   # per_image[model][set name] = per-image fractional recall
    for name in args.models:
        img_enc, txt_enc = build(name, args.device)
        per_image[name] = {}
        for ds, images, tokens in sets:
            ie, te = embed(img_enc, txt_enc, images, tokens, args.device)
            r = metrics.report(ie, te, ds.gt, ds.txt_ids)
            per_image[name][ds.name] = r.pop("per_image")
            rows.append({"model": name, "set": ds.name, **r})
        timing = {}
        if not args.no_timing:
            _, images, tokens = sets[0]
            timing = {"gpu_img_ms": time_batch1(img_enc, torch.from_numpy(images[:1]).to(args.device)),
                      "gpu_txt_ms": time_batch1(txt_enc, torch.from_numpy(tokens[:1]).to(args.device))}
        params = {"img_params_M": sum(p.numel() for p in img_enc.parameters()) / 1e6,
                  "txt_params_M": sum(p.numel() for p in txt_enc.parameters()) / 1e6}
        for g, members in groups.items():
            per = np.concatenate([per_image[name][m] for m in members])
            lo, hi = metrics.bootstrap_ci(per)
            fold_means = [float(per_image[name][m].mean()) for m in members]
            rows.append({"model": name, "set": g, "frac_R@10": float(per.mean()), "frac_R@10_ci95": (lo, hi),
                         "fold_means": fold_means, "n_images": len(per), **params, **timing})
            spread = f" folds {min(fold_means)*100:.1f}-{max(fold_means)*100:.1f}" if len(members) > 1 else ""
            print(f"{name:22s} {g:10s} fracR@10={per.mean()*100:6.2f} [{lo*100:5.1f},{hi*100:5.1f}] n={len(per)}{spread}"
                  + (f"  gpu img/txt ms={timing['gpu_img_ms']:.2f}/{timing['gpu_txt_ms']:.2f}" if timing else ""))
        del img_enc, txt_enc
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    base = args.models[0]
    for g, members in groups.items():
        print(f"paired deltas vs {base} on {g}:")
        a = np.concatenate([per_image[base][m] for m in members])
        for name in args.models[1:]:
            b = np.concatenate([per_image[name][m] for m in members])
            d, lo, hi = metrics.paired_bootstrap_delta(a, b)
            sig = "*" if lo > 0 or hi < 0 else " "
            print(f"  {sig} {name:22s} {d*100:+.2f} pts  (95% CI {lo*100:+.1f} .. {hi*100:+.1f})")

    meta = {"sets": {ds.name: {"n_images": len(ds), "n_captions": len(ds.captions), "notes": ds.notes}
                     for ds, _, _ in sets}}
    exp.save_json("results.json", {"meta": meta, "rows": rows})
    exp.save_npz("per_image_recall.npz", **{f"{m}|{s}": v for m, d in per_image.items() for s, v in d.items()})

    # README table: one row per model, one column per report group, plus paired delta on the first group
    g0 = list(groups)[0]
    a = np.concatenate([per_image[base][m] for m in groups[g0]])
    hdr = "| Model | " + " | ".join(f"{g} frac R@10 [95% CI]" for g in groups) + f" | delta vs {base} on {g0} [95% CI] |"
    lines = [hdr, "|" + "---|" * (len(groups) + 2)]
    for name in args.models:
        cells = []
        for g in groups:
            r = next(r for r in rows if r["model"] == name and r["set"] == g)
            lo, hi = r["frac_R@10_ci95"]
            cells.append(f"{r['frac_R@10']*100:.2f} [{lo*100:.1f}, {hi*100:.1f}]")
        b = np.concatenate([per_image[name][m] for m in groups[g0]])
        d, lo, hi = metrics.paired_bootstrap_delta(a, b)
        delta = "-" if name == base else f"{d*100:+.2f} [{lo*100:+.1f}, {hi*100:+.1f}]{' *' if lo > 0 or hi < 0 else ''}"
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {delta} |")
    exp.add_table("\n".join(lines) + "\n\n`*` = paired 95% CI excludes 0. Folds of a preset are pooled.")
    best = max((r for r in rows if r["set"] == g0), key=lambda r: r["frac_R@10"])
    exp.headline = f"best on {g0}: {best['model']} {best['frac_R@10']*100:.2f}"


if __name__ == "__main__":
    main()
