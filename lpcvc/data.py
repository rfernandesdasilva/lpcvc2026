"""LPCVC 2026 Track 1 data loading, matching the official sample solution exactly.

Official interface (https://lpcv.ai/2026LPCVC/tracks/track1):
  image: float32 (1, 3, 224, 224), PIL resize to 224x224, /255, HWC->CHW, NO mean/std normalization
  text:  int32   (1, 77), CLIPTokenizer("openai/clip-vit-base-patch32"), padding="max_length",
         truncation=True, max_length=77  (padding token is EOT=49407, repeated)

Anything else a model needs (mean/std, resizing, token re-padding) must live INSIDE the model graph.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

IMG_EXTS = (".jpg", ".png", ".jpeg", ".webp")
EOT_TOKEN = 49407
SOT_TOKEN = 49406


def process_image(image_path, target_size=(224, 224)) -> np.ndarray:
    """Verbatim copy of the official preprocessing. Returns (1, 3, H, W) float32 in [0, 1]."""
    image = Image.open(image_path).convert("RGB").resize(target_size)
    image_array = np.array(image, dtype=np.float32) / 255.0
    return np.transpose(image_array, (2, 0, 1))[np.newaxis, :]


_TOKENIZER = None


def tokenize(prompts: list[str]) -> np.ndarray:
    """Official tokenization. Returns (N, 77) int32."""
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import CLIPTokenizer

        _TOKENIZER = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    out = _TOKENIZER(prompts, padding="max_length", truncation=True, max_length=77, return_tensors="np")
    return out["input_ids"].astype(np.int32)


@dataclass
class RetrievalSet:
    """A fixed image query set and a fixed caption candidate pool."""

    name: str
    image_paths: list[Path]          # query images, in evaluation order
    gt: list[list[int]]              # gt[i] = caption IDs relevant to image i
    txt_ids: list[int]               # txt_ids[j] = caption ID of candidate j
    captions: list[str]              # captions[j] = text of candidate j
    notes: list[str] = field(default_factory=list)

    def __len__(self):
        return len(self.image_paths)

    def images(self) -> np.ndarray:
        return np.concatenate([process_image(p) for p in self.image_paths], axis=0)

    def tokens(self) -> np.ndarray:
        return tokenize(self.captions)


def load_lpcvc_sample(root: str | os.PathLike) -> RetrievalSet:
    """Load the official sample (images/, img_list.csv, txt_list.csv).

    The official pipeline uploads images *sorted by filename* but reads ground truth in *CSV row
    order*. We pair images with ground truth by filename (column 0 of img_list.csv) and warn if the
    two orders differ, because then the official scorer would be misaligned.
    """
    root = Path(root)
    df_img = pd.read_csv(root / "img_list.csv")
    df_txt = pd.read_csv(root / "txt_list.csv")
    notes: list[str] = []

    img_dir = root / "images"
    sorted_names = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS))
    csv_names = [str(x).strip() for x in df_img.iloc[:, 0].tolist()]
    if all((img_dir / n).exists() for n in csv_names):
        image_paths = [img_dir / n for n in csv_names]
        if csv_names != sorted_names:
            notes.append("img_list.csv row order != sorted filename order; the official scorer would misalign.")
    else:
        image_paths = [img_dir / n for n in sorted_names]
        notes.append("Column 0 of img_list.csv is not a filename; falling back to sorted filename order (official behavior).")

    gt = [[int(x) for x in str(s).split(";")] for s in df_img.iloc[:, 1].dropna().tolist()]

    df_txt = df_txt.dropna(subset=[df_txt.columns[0], df_txt.columns[1]])
    txt_ids = df_txt.iloc[:, 0].astype(int).tolist()
    captions = df_txt.iloc[:, 1].astype(str).tolist()

    missing = sorted({g for gs in gt for g in gs} - set(txt_ids))
    if missing:
        notes.append(f"GT references caption IDs absent from txt_list.csv: {missing} (still counted in the denominator, as officially).")
    if len(gt) != len(image_paths):
        raise ValueError(f"{len(gt)} GT rows but {len(image_paths)} images")
    for n in notes:
        warnings.warn(n)
    return RetrievalSet("lpcvc_sample", image_paths, gt, txt_ids, captions, notes)
