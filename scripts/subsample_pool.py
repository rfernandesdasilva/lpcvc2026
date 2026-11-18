"""Pool-size control: score models on random 56-image subsets of each validation fold (candidate pool = only the
captions of those 56 images), so they can be compared with the 56-image LPCVC sample at a matched pool size.

  python scripts/subsample_pool.py --out experiments/021_grouped-val-screen/pool56.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from eval_local import embed, resolve_sets  # noqa: E402
from lpcvc import metrics  # noqa: E402
from lpcvc.data import load_lpcvc_sample  # noqa: E402
from lpcvc.models import build  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["clip-b16", "mobileclip2-b",
                                                     "mobileclip2-b#checkpoints/010_ft-b-contrastive/model.pt",
                                                     "mobileclip2-b#checkpoints/020_ft-b-cococaptions/model.pt"])
    ap.add_argument("--data", nargs="+", default=["grouped", "refcocog"])
    ap.add_argument("--n", type=int, default=56)
    ap.add_argument("--draws", type=int, default=200, help="subsets per fold")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sets = [(spec, load_lpcvc_sample(p)) for spec in args.data for _, p in resolve_sets([spec])]
    rng = np.random.default_rng(args.seed)
    draws = {id(ds): [rng.choice(len(ds), args.n, replace=False) for _ in range(args.draws)] for _, ds in sets}
    out = {}
    for m in args.models:
        img_enc, txt_enc = build(m, "cuda")
        res = {spec: [] for spec in args.data}
        pool = {spec: [] for spec in args.data}
        with torch.no_grad():
            for spec, ds in sets:
                ie, te = embed(img_enc, txt_enc, ds.images(), ds.tokens(), "cuda")
                sim = metrics.similarity(ie, te)
                tid = np.asarray(ds.txt_ids)
                for S in draws[id(ds)]:
                    gt = [ds.gt[i] for i in S]
                    cols = np.flatnonzero(np.isin(tid, sorted({c for g in gt for c in g})))
                    res[spec].append(metrics.fractional_recall_at_k(sim[np.ix_(S, cols)], gt, tid[cols]))
                    pool[spec].append(len(cols))
        out[m] = {s: {"frac_R@10": float(np.mean(v)), "p5_p95": [float(np.percentile(v, 5)), float(np.percentile(v, 95))],
                      "mean_pool": float(np.mean(pool[s]))} for s, v in res.items()}
        print(m, {s: f"{d['frac_R@10'] * 100:.2f} (5-95%: {d['p5_p95'][0] * 100:.1f}-{d['p5_p95'][1] * 100:.1f}, "
                     f"pool {d['mean_pool']:.0f})" for s, d in out[m].items()})
        del img_enc, txt_enc
        torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps({"args": vars(args), "results": out}, indent=1))


if __name__ == "__main__":
    main()
