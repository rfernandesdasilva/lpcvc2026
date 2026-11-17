"""Build LPCVC-style validation folds from RefCOCOg (UMD split, val + test), deterministically.

Why RefCOCOg: like the LPCVC captions, its sentences describe single objects ("the brown dog lying down"),
and each image has several of them. Raw RefCOCOg sentences are longer (8.4 vs 5.0 words), often spatial
("the sink on the left"), and 35% about people. The LPCVC sample is attribute-based and object-centric. So we:
  1. drop references to people (COCO category 1); the people can still appear in the images,
  2. drop sentences with location words (LPCVC captions describe appearance, not position),
  3. keep up to 2 sentences per object, preferring lengths close to 5 words,
  4. normalize the style: strip, drop a trailing period, turn a leading "a"/"an" into "The", capitalize,
  5. merge identical strings into one caption ID inside a fold (LPCVC txt_list holds *unique* texts),
  6. keep images with >=1 caption, shuffle with a fixed seed, cut into disjoint folds of 300 images
     (about the size of the LPCVC test set, so the candidate-pool difficulty is comparable).

Output (same format as the official sample, so every evaluator works unchanged):
  data/refcocog_val/fold{k}/{images/, img_list.csv, txt_list.csv}
  splits/refcocog_val/fold{k}/{img_list.csv, txt_list.csv}     (committed copy of the definition)
  splits/reserved_coco_train2014_ids.txt                        (all RefCOCOg val/test images: NEVER train on these)

  python scripts/build_val_set.py              # 5 folds x 300 images
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
REPO, REV = "jxu124/refcocog", "55319436ad54b9480cefda6b9d64397de92456dd"   # pinned revision
FILES = ["data/validation-00000-of-00001-15168dfe7b5961e5.parquet",
         "data/test-00000-of-00001-2316f36b19cd7f72.parquet"]
COCO_URL = "http://images.cocodataset.org/train2014/{}"
PERSON = 1
SPATIAL = re.compile(r"\b(left|right|top|bottom|front|behind|background|foreground|middle|center|centre|"
                     r"closest|closer|nearest|near|farthest|further|far|corner|side|above|below|next)\b", re.I)
TARGET_WORDS = 5


def normalize(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip()).rstrip(".").strip()
    s = re.sub(r"^(a|an)\s+", "the ", s, flags=re.I)
    return s[:1].upper() + s[1:]


def load_refs() -> pd.DataFrame:
    dfs = [pd.read_parquet(hf_hub_download(REPO, f, repo_type="dataset", revision=REV)) for f in FILES]
    return pd.concat(dfs, ignore_index=True)


def captions_per_image(df: pd.DataFrame, per_ref: int, max_per_img: int) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for _, r in df.sort_values(["image_id", "ref_id"]).iterrows():
        fname = r.image_path.split("/")[-1]
        out.setdefault(fname, [])
        if r.category_id == PERSON:
            continue
        sents = sorted({normalize(s["raw"]) for s in r.sentences if not SPATIAL.search(s["raw"])},
                       key=lambda s: (abs(len(s.split()) - TARGET_WORDS), s))
        for s in sents[:per_ref]:
            if s not in out[fname] and len(out[fname]) < max_per_img and len(s.split()) >= 2:
                out[fname].append(s)
    return {k: v for k, v in out.items() if v}


def write_fold(fold_dir: Path, imgs: list[str], caps: dict[str, list[str]]) -> tuple[int, int]:
    fold_dir.mkdir(parents=True, exist_ok=True)
    text_ids: dict[str, int] = {}
    rows = []
    for f in imgs:
        ids = []
        for c in caps[f]:
            text_ids.setdefault(c, len(text_ids) + 1)
            ids.append(text_ids[c])
        rows.append((f, ";".join(map(str, ids))))
    with open(fold_dir / "img_list.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["Image_names", "Text_nums"]); w.writerows(rows)
    with open(fold_dir / "txt_list.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["Text_nums", "Unique_Texts"]); w.writerows((i, t) for t, i in text_ids.items())
    return len(rows), len(text_ids)


def download(fname: str, dest: Path) -> None:
    p = dest / fname
    if p.exists() and p.stat().st_size > 0:
        return
    with urlopen(COCO_URL.format(fname), timeout=60) as r:
        data = r.read()
    p.write_bytes(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold-size", type=int, default=300)
    ap.add_argument("--per-ref", type=int, default=2)
    ap.add_argument("--max-per-img", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    df = load_refs()
    reserved = sorted({int(x) for x in df.image_id})
    (ROOT / "splits").mkdir(exist_ok=True)
    (ROOT / "splits" / "reserved_coco_train2014_ids.txt").write_text("\n".join(map(str, reserved)) + "\n")

    caps = captions_per_image(df, args.per_ref, args.max_per_img)
    eligible = sorted(caps)
    random.Random(args.seed).shuffle(eligible)
    need = args.folds * args.fold_size
    if len(eligible) < need:
        sys.exit(f"only {len(eligible)} eligible images, need {need}")
    print(f"RefCOCOg val+test: {df.image_id.nunique()} images, {len(eligible)} eligible after filtering")

    for k in range(args.folds):
        imgs = sorted(eligible[k * args.fold_size:(k + 1) * args.fold_size])
        data_dir = ROOT / "data" / "refcocog_val" / f"fold{k}"
        n_img, n_txt = write_fold(data_dir, imgs, caps)
        split_dir = ROOT / "splits" / "refcocog_val" / f"fold{k}"
        split_dir.mkdir(parents=True, exist_ok=True)
        for f in ("img_list.csv", "txt_list.csv"):
            shutil.copy(data_dir / f, split_dir / f)
        n_caps = sum(len(caps[f]) for f in imgs)
        words = sum(len(c.split()) for f in imgs for c in caps[f]) / n_caps
        print(f"fold{k}: {n_img} images, {n_txt} unique captions, {n_caps / n_img:.2f} caps/img, {words:.2f} words/cap")
        if not args.no_download:
            img_dir = data_dir / "images"
            img_dir.mkdir(exist_ok=True)
            with ThreadPoolExecutor(16) as ex:
                list(ex.map(lambda f: download(f, img_dir), imgs))
            print(f"  images in {img_dir}")


if __name__ == "__main__":
    main()
