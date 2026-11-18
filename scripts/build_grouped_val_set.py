"""LPCVC-like "grouped" validation folds: every image shows several objects of ONE category, and its captions tell
those objects apart by appearance ("The orange ping pong ball", "The ping pong ball with no text").

That is the structure of the LPCVC sample: each image has 2-6 instances of one category, each caption names the
category plus an attribute, and the category nouns recur across images (monitors, pigs, cars...). The standard
folds (build_val_set.py) instead mix whatever objects RefCOCOg annotated, often one per category per image.

Source: only the reserved images (splits/reserved_coco_train2014_ids.txt = RefCOCOg val/test, never trained on),
with the referring expressions of RefCOCO, RefCOCO+ and RefCOCOg merged per object (same COCO ann_id). RefCOCO and
RefCOCO+ were collected on images with several same-category objects, so merging them gives ~3x more such images
than RefCOCOg alone. Their train-split refs on these images are unused by training, because training filters by image.
Filtering and normalization match build_val_set.py (no people refs, no location words, "The ..." style, >= 2 words).
Per image: the category with the most described objects (>= 2 required), up to 2 sentences per object (closest to 5
words first), at most 10 captions. Folds: 4 x 300 images, stratified by category, identical captions merged per fold.

  python scripts/build_grouped_val_set.py
Output: data/grouped_val/fold{k}/{images/, img_list.csv, txt_list.csv}, splits/grouped_val/fold{k}/ (committed)
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_train_set import SOURCES  # noqa: E402  (same pinned revisions)
from build_val_set import PERSON, SPATIAL, TARGET_WORDS, download, normalize, write_fold  # noqa: E402


def load_reserved_refs(reserved: set[int]) -> pd.DataFrame:
    dfs = []
    for repo, (rev, _) in SOURCES.items():
        for f in sorted(HfApi().list_repo_files(repo, repo_type="dataset", revision=rev)):
            if f.endswith(".parquet"):
                d = pd.read_parquet(hf_hub_download(repo, f, repo_type="dataset", revision=rev))
                dfs.append(d[d.image_id.isin(reserved)])
    return pd.concat(dfs, ignore_index=True)


def grouped_captions(df: pd.DataFrame, per_obj: int, max_per_img: int):
    objs: dict[int, dict[int, dict[int, set[str]]]] = defaultdict(lambda: defaultdict(dict))  # img -> cat -> ann -> sents
    fname = {}
    for r in df.itertuples():
        fname[r.image_id] = r.image_path.split("/")[-1]
        if r.category_id == PERSON:
            continue
        sents = {normalize(s["raw"]) for s in r.sentences if not SPATIAL.search(s["raw"])}
        sents = {s for s in sents if len(s.split()) >= 2}
        if sents:
            objs[r.image_id][r.category_id].setdefault(r.ann_id, set()).update(sents)
    caps, cat_of = {}, {}
    for im in sorted(objs):
        cat, anns = max(sorted(objs[im].items()), key=lambda kv: len(kv[1]))
        if len(anns) < 2:
            continue
        out: list[str] = []
        for ann in sorted(anns):
            ranked = sorted(anns[ann], key=lambda s: (abs(len(s.split()) - TARGET_WORDS), s))
            for s in [s for s in ranked if s not in out][:per_obj]:
                if len(out) < max_per_img:
                    out.append(s)
        caps[fname[im]], cat_of[fname[im]] = out, cat
    return caps, cat_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--fold-size", type=int, default=300)
    ap.add_argument("--per-obj", type=int, default=2)
    ap.add_argument("--max-per-img", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    reserved = {int(x) for x in (ROOT / "splits" / "reserved_coco_train2014_ids.txt").read_text().split()}
    caps, cat_of = grouped_captions(load_reserved_refs(reserved), args.per_obj, args.max_per_img)
    assert all(int(f.split("_")[-1].split(".")[0]) in reserved for f in caps)
    print(f"{len(caps)} reserved images with >= 2 described objects of one category, {len(set(cat_of.values()))} categories")

    # stratified deal: shuffle within each category, then deal images round-robin over the folds
    rng = random.Random(args.seed)
    by_cat = defaultdict(list)
    for f in sorted(caps):
        by_cat[cat_of[f]].append(f)
    order = []
    for c in sorted(by_cat):
        rng.shuffle(by_cat[c])
        order += by_cat[c]
    folds = [order[k::args.folds] for k in range(args.folds)]
    for k in range(args.folds):
        if len(folds[k]) < args.fold_size:
            sys.exit(f"fold {k} has only {len(folds[k])} images")
        rng.shuffle(folds[k])
        imgs = sorted(folds[k][:args.fold_size])
        data_dir = ROOT / "data" / "grouped_val" / f"fold{k}"
        n_img, n_txt = write_fold(data_dir, imgs, caps)
        split_dir = ROOT / "splits" / "grouped_val" / f"fold{k}"
        split_dir.mkdir(parents=True, exist_ok=True)
        for f in ("img_list.csv", "txt_list.csv"):
            shutil.copy(data_dir / f, split_dir / f)
        n_caps = sum(len(caps[f]) for f in imgs)
        words = sum(len(c.split()) for f in imgs for c in caps[f]) / n_caps
        cats = Counter(cat_of[f] for f in imgs)
        same = sum(cats[cat_of[f]] - 1 for f in imgs) / n_img
        print(f"fold{k}: {n_img} images, {n_txt} unique captions, {n_caps / n_img:.2f} caps/img, {words:.2f} words/cap, "
              f"{len(cats)} categories, {same:.1f} other same-category images per image")
        if not args.no_download:
            img_dir = data_dir / "images"
            img_dir.mkdir(exist_ok=True)
            with ThreadPoolExecutor(16) as ex:
                list(ex.map(lambda f: download(f, img_dir), imgs))


if __name__ == "__main__":
    main()
