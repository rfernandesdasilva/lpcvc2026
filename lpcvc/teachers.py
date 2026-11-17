"""Large open_clip models used only as offline TEACHERS (never deployed, so no latency limit).

A teacher uses its own tokenizer and normalization, because only its embeddings are used. Images are squash-resized
to the teacher's native resolution, like the official LPCVC preprocessing (no center crop that could cut the
described object out of the frame).

  t = Teacher("ViT-H-14-378:dfn5b"); ie = t.encode_images(paths); te = t.encode_texts(captions)
Name format: "<open_clip arch>:<pretrained tag>", or a mobileclip2-* name from lpcvc.models.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .models import MOBILECLIP2

ALIASES = {  # short names used in experiment tables
    "vit-l14-dfn2b": "ViT-L-14-quickgelu:dfn2b",
    "vit-h14-378-dfn5b": "ViT-H-14-378-quickgelu:dfn5b",
    "siglip2-so400m-384": "ViT-SO400M-16-SigLIP2-384:webli",
    "pe-core-l14-336": "PE-Core-L-14-336:meta",
    "mobileclip2-l14": "MobileCLIP2-L-14:dfndr2b",
}


def resolve(name: str) -> tuple[str, str]:
    name = ALIASES.get(name, name)
    if name in MOBILECLIP2:
        return MOBILECLIP2[name], "dfndr2b"
    arch, tag = name.split(":")
    return arch, tag


class Teacher:
    def __init__(self, name: str, device="cuda", dtype=torch.float16):
        import open_clip

        arch, tag = resolve(name)
        self.name, self.device, self.dtype = name, device, dtype
        self.model = open_clip.create_model(arch, pretrained=tag).eval().to(device, dtype)
        self.tokenizer = open_clip.get_tokenizer(arch)
        cfg = self.model.visual.preprocess_cfg
        size = cfg["size"]
        self.res = size[0] if isinstance(size, (tuple, list)) else size
        self.mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)

    def load(self, path, flip=False) -> torch.Tensor:
        im = Image.open(path).convert("RGB").resize((self.res, self.res), Image.BICUBIC)
        if flip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        return torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1)

    @torch.no_grad()
    def encode_images(self, paths, bs=64, flip=False, workers=8) -> np.ndarray:
        from concurrent.futures import ThreadPoolExecutor

        out = []
        with ThreadPoolExecutor(workers) as ex:
            for i in range(0, len(paths), bs):
                x = torch.stack(list(ex.map(lambda p: self.load(p, flip), paths[i:i + bs]))).to(self.device)
                x = ((x.float() / 255 - self.mean) / self.std).to(self.dtype)
                out.append(F.normalize(self.model.encode_image(x).float(), dim=-1).cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def encode_texts(self, texts, bs=256) -> np.ndarray:
        out = []
        for i in range(0, len(texts), bs):
            t = self.tokenizer(list(texts[i:i + bs])).to(self.device)
            out.append(F.normalize(self.model.encode_text(t).float(), dim=-1).cpu())
        return torch.cat(out).numpy()

    @property
    def logit_scale(self) -> float:
        return float(self.model.logit_scale.exp())
