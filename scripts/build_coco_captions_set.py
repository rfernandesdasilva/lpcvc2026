"""COCO Captions control set: the SAME 15,905 training images as data/train_refmix, with their COCO captions.

This isolates the caption type (whole-scene COCO sentences, the proposal's original dataset, vs object-level
RefCOCO referring expressions) from everything else: same images, same recipe, same number of steps.

captions_train2014.json is read straight out of the official annotations_trainval2014.zip with HTTP range
requests (about 12 MB transferred instead of the 253 MB zip) and cached in data/coco_captions/.
Normalization matches the RefCOCO sets (strip, drop trailing period, leading "a"/"an" -> "The", capitalize) and
sentences containing "left"/"right" are dropped, because training uses horizontal flips. People are kept: they
are what COCO captions describe, and removing them is not part of "using COCO Captions".

  python scripts/build_coco_captions_set.py
Output: data/train_coco_captions/train.jsonl   (images are read from data/train_refmix/images/)
Train:  python scripts/train.py --train-file data/train_coco_captions/train.jsonl ...
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_val_set import normalize  # noqa: E402

URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
MEMBER = "annotations/captions_train2014.json"
EXPECTED = "abd52e33fdb5097617233a5b0b4a65e73465f9bb0d94ddf59808db6644f9090f"   # train.jsonl of experiment 020
FLIP = re.compile(r"\b(left|right)\b", re.I)


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file, so zipfile fetches only the bytes it needs."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        with urlopen(Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size or not len(b):
            return 0
        end = min(self.pos + len(b), self.size) - 1
        with urlopen(Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"}), timeout=120) as r:
            data = r.read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def coco_captions() -> dict:
    cache = ROOT / "data" / "coco_captions" / "captions_train2014.json"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(URL), buffer_size=1 << 22)) as z:
            cache.write_bytes(z.read(MEMBER))
    print(f"{cache.relative_to(ROOT).as_posix()} sha256={hashlib.sha256(cache.read_bytes()).hexdigest()}")
    return json.loads(cache.read_text(encoding="utf-8"))


def main():
    ann = coco_captions()
    by_id: dict[int, list[str]] = {}
    for a in sorted(ann["annotations"], key=lambda a: a["id"]):
        by_id.setdefault(a["image_id"], []).append(a["caption"])
    names = [json.loads(l)["image"] for l in open(ROOT / "data" / "train_refmix" / "train.jsonl", encoding="utf-8")]
    out = ROOT / "data" / "train_coco_captions"
    out.mkdir(parents=True, exist_ok=True)
    n_caps = n_drop = 0
    words = []
    with open(out / "train.jsonl", "w", encoding="utf-8") as fh:
        for n in names:
            caps = []
            for s in by_id[int(n.split("_")[-1].split(".")[0])]:
                s = normalize(s)
                if FLIP.search(s) or len(s.split()) < 2:
                    n_drop += 1
                elif s not in caps:
                    caps.append(s)
            assert caps, n
            n_caps += len(caps)
            words += [len(c.split()) for c in caps]
            fh.write(json.dumps({"image": n, "captions": caps}, ensure_ascii=False) + "\n")
    digest = hashlib.sha256((out / "train.jsonl").read_bytes()).hexdigest()
    print(f"train_coco_captions: {len(names)} images, {n_caps} captions ({n_caps / len(names):.2f}/img, "
          f"{sum(words) / len(words):.2f} words/cap), {n_drop} dropped (left/right), train.jsonl sha256={digest} "
          + ("(matches experiment 020)" if digest == EXPECTED else "(DIFFERS from experiment 020)"))


if __name__ == "__main__":
    main()
