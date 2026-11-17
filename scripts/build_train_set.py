"""LPCVC-style TRAINING set from the train splits of RefCOCO, RefCOCO+ and RefCOCOg (pinned HF revisions).

Same caption filtering/normalization as the validation set (scripts/build_val_set.py): no people refs, no
location words, "The ..." style, >= 2 words. Every image listed in splits/reserved_coco_train2014_ids.txt
(the RefCOCOg val/test images our validation folds come from) is removed, and the script asserts zero overlap.
RefCOCO and RefCOCO+ train images are partly the same COCO images as RefCOCOg val/test, so this filter matters.

  python scripts/build_train_set.py               # ~15.9k images, ~76k captions
Output: data/train_refmix/{images/, train.jsonl}   one line per image: {"image": ..., "captions": [...]}
        data/train_refmix/caption_sources.json     caption -> source datasets (for cross-source ablations)
        splits/train_refmix/images.txt            committed image list (+ sha256 of train.jsonl printed)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_val_set import captions_per_image, download  # noqa: E402

SOURCES = {  # repo: (pinned revision, train parquet)
    "jxu124/refcocog": ("55319436ad54b9480cefda6b9d64397de92456dd", "data/train-00000-of-00001-4fe3e6340cfb69ed.parquet"),
    "jxu124/refcocoplus": ("12e20b0b6039fbf656e89e2f26597e84c1037847", "data/train-00000-of-00001-7294665695c630ee.parquet"),
    "jxu124/refcoco": ("9fba8200c5326e996f789191f095bd464ef1d09e", "data/train-00000-of-00001-94431d5f4bd5b93f.parquet"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-ref", type=int, default=3)
    ap.add_argument("--max-per-img", type=int, default=30)
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    reserved = {int(x) for x in (ROOT / "splits" / "reserved_coco_train2014_ids.txt").read_text().split()}
    caps: dict[str, list[str]] = {}
    sources: dict[str, set[str]] = {}
    for repo, (rev, f) in SOURCES.items():
        df = pd.read_parquet(hf_hub_download(repo, f, repo_type="dataset", revision=rev))
        df = df[~df.image_id.isin(reserved)]
        c = captions_per_image(df, args.per_ref, args.max_per_img)
        for k, v in c.items():
            cur = caps.setdefault(k, [])
            cur += [s for s in v if s not in cur]
            for s in v:
                sources.setdefault(s, set()).add(repo.split("/")[1])
        print(f"{repo}: {len(c)} images, {sum(map(len, c.values()))} captions")
    names = sorted(caps)
    overlap = {int(n.split("_")[-1].split(".")[0]) for n in names} & reserved
    assert not overlap, f"training images overlap reserved validation images: {sorted(overlap)[:10]}"

    out = ROOT / "data" / "train_refmix"
    (out / "images").mkdir(parents=True, exist_ok=True)
    with open(out / "train.jsonl", "w", encoding="utf-8") as fh:
        for n in names:
            fh.write(json.dumps({"image": n, "captions": caps[n]}, ensure_ascii=False) + "\n")
    (out / "caption_sources.json").write_text(json.dumps({k: sorted(v) for k, v in sorted(sources.items())},
                                                         ensure_ascii=False), encoding="utf-8")
    split = ROOT / "splits" / "train_refmix"
    split.mkdir(parents=True, exist_ok=True)
    (split / "images.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    digest = hashlib.sha256((out / "train.jsonl").read_bytes()).hexdigest()
    n_caps = sum(map(len, caps.values()))
    print(f"train_refmix: {len(names)} images, {n_caps} captions ({n_caps / len(names):.2f}/img), "
          f"0 overlap with reserved IDs, train.jsonl sha256={digest}")
    if not args.no_download:
        with ThreadPoolExecutor(24) as ex:
            list(ex.map(lambda f: download(f, out / "images"), names))
        print(f"images in {out / 'images'}")


if __name__ == "__main__":
    main()
