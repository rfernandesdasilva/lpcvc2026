"""Calibration set for post-training quantization, from RefCOCOg TRAIN (never the reserved val/test images).

Same caption filtering/normalization as the validation set (scripts/build_val_set.py), so the activation
ranges seen during calibration match the LPCVC caption style.

  python scripts/build_calib_set.py              # 256 images, 512 captions
Output: data/calib_refcocog_train/{images/, captions.txt, images.txt} (+ committed copy of the lists in splits/calib/)
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
from build_val_set import REPO, REV, captions_per_image, download  # noqa: E402

TRAIN_FILE = "data/train-00000-of-00001-4fe3e6340cfb69ed.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=256)
    ap.add_argument("--captions", type=int, default=512)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    df = pd.read_parquet(hf_hub_download(REPO, TRAIN_FILE, repo_type="dataset", revision=REV))
    reserved = {int(x) for x in (ROOT / "splits" / "reserved_coco_train2014_ids.txt").read_text().split()}
    df = df[~df.image_id.isin(reserved)]
    caps = captions_per_image(df, per_ref=2, max_per_img=10)
    names = sorted(caps)
    rng = random.Random(args.seed)
    rng.shuffle(names)
    imgs = sorted(names[:args.images])
    # captions from *other* train images too, so text calibration is not tied to the calibration images
    pool = sorted({c for n in names[:args.images * 4] for c in caps[n]})
    rng.shuffle(pool)
    captions = pool[:args.captions]

    out = ROOT / "data" / "calib_refcocog_train"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "images.txt").write_text("\n".join(imgs) + "\n", encoding="utf-8")
    (out / "captions.txt").write_text("\n".join(captions) + "\n", encoding="utf-8")
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda f: download(f, out / "images"), imgs))
    split = ROOT / "splits" / "calib"
    split.mkdir(parents=True, exist_ok=True)
    for f in ("images.txt", "captions.txt"):
        shutil.copy(out / f, split / f)
    overlap = {int(n.split("_")[-1].split(".")[0]) for n in imgs} & reserved
    assert not overlap, f"calibration images overlap reserved validation images: {overlap}"
    print(f"calibration: {len(imgs)} images, {len(captions)} captions -> {out} (0 overlap with reserved IDs)")


if __name__ == "__main__":
    main()
