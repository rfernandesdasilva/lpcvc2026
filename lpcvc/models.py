"""Encoder wrappers that accept the OFFICIAL LPCVC input interface and hide everything else in-graph.

Every model returned by `build()` is a pair (image_encoder, text_encoder) of nn.Modules:
  image_encoder(images: float32 (B, 3, 224, 224) in [0, 1]) -> (B, D) L2-normalized
  text_encoder(tokens: int (B, 77), official CLIP tokenizer w/ EOT padding) -> (B, D) L2-normalized

Model names:
  clip-b16            Official baseline as shipped: OpenAI CLIP ViT-B/16, image fed WITHOUT mean/std
                      normalization (the sample exports `clip.visual` directly on [0,1] pixels).
  clip-b16-norm       Same weights, but with CLIP mean/std normalization inside the graph.
  mobileclip2-s0|s2|b MobileCLIP2 (open_clip, 'dfndr2b'), reparameterized, resize-in-graph.

Append `@<res>` to change the internal image resolution, e.g. `mobileclip2-s0@224`.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import EOT_TOKEN

OPENAI_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_STD = (0.26862954, 0.26130258, 0.27577711)

MOBILECLIP2 = {
    "mobileclip2-s0": "MobileCLIP2-S0",
    "mobileclip2-s2": "MobileCLIP2-S2",
    "mobileclip2-b": "MobileCLIP2-B",
    "mobileclip2-s3": "MobileCLIP2-S3",
    "mobileclip2-s4": "MobileCLIP2-S4",
}


def reparameterize(model: nn.Module) -> nn.Module:
    """Fold MobileOne/RepMixer train-time branches into single convs (same as ml-mobileclip)."""
    model = copy.deepcopy(model)
    for m in model.modules():
        if hasattr(m, "reparameterize"):
            m.reparameterize()
    return model


class ImageEncoder(nn.Module):
    def __init__(self, clip, res: int, mean, std, normalize_input: bool = True):
        super().__init__()
        self.clip = clip
        self.res = res
        self.normalize_input = normalize_input
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = image
        if x.shape[-1] != self.res or x.shape[-2] != self.res:
            x = F.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False)
        if self.normalize_input:
            x = (x - self.mean) / self.std
        z = self.clip.encode_image(x)
        return F.normalize(z, dim=-1)


class TextEncoder(nn.Module):
    """Optionally converts official EOT-padding into the zero-padding MobileCLIP2 was trained with.

    MobileCLIP2 text towers attend bidirectionally (open_clip cfg `no_causal_mask=True`) with no
    padding mask, so tokens after EOT change the embedding; CLIP's causal transformer ignores them.
    """

    def __init__(self, clip, zero_pad_after_eot: bool):
        super().__init__()
        self.clip = clip
        self.zero_pad_after_eot = zero_pad_after_eot

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        t = text.long()
        if self.zero_pad_after_eot:
            eot_pos = (t == EOT_TOKEN).long().argmax(dim=-1, keepdim=True)          # first EOT
            pos = torch.arange(t.shape[-1], device=t.device).unsqueeze(0)
            t = t * (pos <= eot_pos).long()
        z = self.clip.encode_text(t)
        return F.normalize(z, dim=-1)


def parse_name(name: str) -> tuple[str, int | None]:
    if "@" in name:
        base, res = name.split("@")
        return base, int(res)
    return name, None


def build(name: str, device="cpu") -> tuple[nn.Module, nn.Module]:
    import open_clip

    base, res = parse_name(name)
    if base in ("clip-b16", "clip-b16-norm"):
        clip = open_clip.create_model("ViT-B-16", pretrained="openai")
        img = ImageEncoder(clip, res or 224, OPENAI_MEAN, OPENAI_STD, normalize_input=(base == "clip-b16-norm"))
        txt = TextEncoder(clip, zero_pad_after_eot=False)
    elif base in MOBILECLIP2:
        # open_clip's pretrained cfg already carries the right mean/std (0/1 for S0/S2/B, per ml-mobileclip).
        clip = open_clip.create_model(MOBILECLIP2[base], pretrained="dfndr2b")
        clip.eval()
        clip = reparameterize(clip)
        cfg = clip.visual.preprocess_cfg
        native = cfg["size"][0] if isinstance(cfg["size"], (tuple, list)) else cfg["size"]
        img = ImageEncoder(clip, res or native, cfg["mean"], cfg["std"], normalize_input=True)
        txt = TextEncoder(clip, zero_pad_after_eot=True)
    else:
        raise KeyError(f"unknown model {name!r}")
    return img.eval().to(device), txt.eval().to(device)


def count_params(module: nn.Module, prefix: str) -> int:
    return sum(p.numel() for n, p in module.named_parameters() if n.startswith(prefix))
