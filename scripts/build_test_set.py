"""Held-out TEST fold, built once and scored only for the final report (never for any decision).

Images come from the RefCOCO and RefCOCO+ (UNC) validation/test/testB splits, pinned HF revisions. RefCOCO's
splits partition COCO differently from RefCOCOg's, so many of these images are in our training data or in the
RefCOCOg validation folds; every such image is removed, as is every calibration image, and the script asserts it.
Caption filtering/normalization is identical to the validation folds (scripts/build_val_set.py): no people refs,
no location words, "The ..." style, >= 2 words, up to 2 sentences per object. RefCOCO+ refs were collected without
location words, so this fold is closer to the attribute-only LPCVC captions than RefCOCOg is.

  python scripts/build_test_set.py               # 1 fold x 300 images
Output (official layout): data/refcoco_test/fold0/{images/, img_list.csv, txt_list.csv}
                          splits/refcoco_test/fold0/{img_list.csv, txt_list.csv}   (committed definition)
Score it only with  eval_local.py --data test --final
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_val_set import captions_per_image, download, write_fold  # noqa: E402

SOURCES = {  # repo: (pinned revision, held-out parquets)
    "jxu124/refcoco": ("9fba8200c5326e996f789191f095bd464ef1d09e", [
        "data/validation-00000-of-00001-bfeafdc84ca37aa2.parquet",
        "data/test-00000-of-00001-82af0c1b600890ac.parquet",
        "data/testB-00000-of-00001-60990e4598892dc1.parquet"]),
    "jxu124/refcocoplus": ("12e20b0b6039fbf656e89e2f26597e84c1037847", [
        "data/validation-00000-of-00001-8c57d66282bc60c9.parquet",
        "data/test-00000-of-00001-2b8e5d26906553b9.parquet",
        "data/testB-00000-of-00001-4f1178d399f1874a.parquet"]),
}


def coco_id(name: str) -> int:
    return int(name.strip().split("_")[-1].split(".")[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-size", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    used = {int(x) for x in (ROOT / "splits" / "reserved_coco_train2014_ids.txt").read_text().split()}
    for f in ("train_refmix/images.txt", "calib/images.txt"):
        used |= {coco_id(l) for l in (ROOT / "splits" / f).read_text().splitlines() if l.strip()}
    df = pd.concat([pd.read_parquet(hf_hub_download(r, f, repo_type="dataset", revision=rev))
                    for r, (rev, fs) in SOURCES.items() for f in fs], ignore_index=True)
    n_all = df.image_id.nunique()
    df = df[~df.image_id.isin(used)]
    caps = captions_per_image(df, per_ref=2, max_per_img=10)   # merges RefCOCO and RefCOCO+ refs per image
    eligible = sorted(caps)
    print(f"RefCOCO/RefCOCO+ held-out splits: {n_all} images, {df.image_id.nunique()} unused by train/val/calib, "
          f"{len(eligible)} eligible after filtering")
    if len(eligible) < args.fold_size:
        sys.exit(f"only {len(eligible)} eligible images, need {args.fold_size}")
    random.Random(args.seed).shuffle(eligible)
    imgs = sorted(eligible[:args.fold_size])
    assert not {coco_id(f) for f in imgs} & used

    data_dir = ROOT / "data" / "refcoco_test" / "fold0"
    n_img, n_txt = write_fold(data_dir, imgs, caps)
    split_dir = ROOT / "splits" / "refcoco_test" / "fold0"
    split_dir.mkdir(parents=True, exist_ok=True)
    for f in ("img_list.csv", "txt_list.csv"):
        shutil.copy(data_dir / f, split_dir / f)
    n_caps = sum(len(caps[f]) for f in imgs)
    words = sum(len(c.split()) for f in imgs for c in caps[f]) / n_caps
    print(f"test fold0: {n_img} images, {n_txt} unique captions, {n_caps / n_img:.2f} caps/img, {words:.2f} words/cap")
    if not args.no_download:
        img_dir = data_dir / "images"
        img_dir.mkdir(exist_ok=True)
        with ThreadPoolExecutor(16) as ex:
            list(ex.map(lambda f: download(f, img_dir), imgs))
        print(f"images in {img_dir}")


if __name__ == "__main__":
    main()
