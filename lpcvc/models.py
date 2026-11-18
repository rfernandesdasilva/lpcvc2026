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
Append `#<checkpoint>` to load fine-tuned weights (path relative to the repo, written by scripts/train.py), and
`~<a>` after it to blend them with the pretrained weights (WiSE-FT): theta = (1 - a) * pretrained + a * finetuned.
  e.g. `mobileclip2-b#checkpoints/010_ft-b-contrastive/model.pt~0.5`
Append `+<flag>` (before `@`/`#`) for graph variants that keep the weights (see FLAGS), e.g.
  `mobileclip2-b+fmask+gelutanh@224#checkpoints/010_ft-b-contrastive/model.pt`
"""
from __future__ import annotations

import copy
import re

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


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def split_ckpt(name: str) -> tuple[str, str | None, float]:
    """'base@res#path~a' -> ('base@res', 'path', a)."""
    if "#" not in name:
        return name, None, 1.0
    name, ckpt = name.split("#", 1)
    alpha = 1.0
    if "~" in ckpt:
        ckpt, a = ckpt.rsplit("~", 1)
        alpha = float(a)
    return name, ckpt, alpha


def load_finetuned(clip: nn.Module, ckpt: str, alpha: float = 1.0) -> nn.Module:
    sd = torch.load(ROOT / ckpt, map_location="cpu", weights_only=True)
    if alpha != 1.0:
        zs = clip.state_dict()
        sd = {k: ((1 - alpha) * zs[k].float() + alpha * v.float()) if v.is_floating_point() else v for k, v in sd.items()}
    clip.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
    return clip


FINITE_MASK = -100.0   # exp(-100) underflows to 0 in FP16/FP32 softmax, yet the range stays quantizable


class QuickGELU(nn.Module):
    """x * sigmoid(1.702 x): sigmoid approximation of GELU (OpenAI CLIP's activation)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


def _swap_gelu(clip: nn.Module, make) -> int:
    n = 0
    for parent in list(clip.modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.GELU):
                setattr(parent, child_name, make())
                n += 1
    return n


def _finite_mask(clip: nn.Module) -> int:
    """Replace the -inf additive causal mask by FINITE_MASK. Min-max quantizers cannot represent -inf: AI Hub gave
    the -inf mask a scale of 4.8e-10, which clamps the mask and the masked attention scores to ~0 (exp 008/022)."""
    n = 0
    for mod in clip.modules():
        m = getattr(mod, "attn_mask", None)
        if isinstance(m, torch.Tensor) and m.is_floating_point() and torch.isinf(m).any():
            m.copy_(m.clamp(min=FINITE_MASK))
            n += 1
    return n


FLAGS = {
    "fmask": _finite_mask,                                                 # finite causal mask (text tower)
    "gelutanh": lambda c: _swap_gelu(c, lambda: nn.GELU(approximate="tanh")),   # tanh GELU instead of erf
    "gelusig": lambda c: _swap_gelu(c, QuickGELU),                         # sigmoid GELU (QuickGELU)
}


def split_flags(name: str) -> tuple[str, list[str]]:
    """'mobileclip2-b+fmask+gelutanh@224' -> ('mobileclip2-b@224', ['fmask', 'gelutanh']) (checkpoint already split)."""
    flags = re.findall(r"\+([A-Za-z0-9]+)", name)
    unknown = [f for f in flags if f not in FLAGS]
    if unknown:
        raise KeyError(f"unknown model flag(s) {unknown}; known: {sorted(FLAGS)}")
    return re.sub(r"\+[A-Za-z0-9]+", "", name), flags


def apply_flags(clip: nn.Module, flags: list[str]) -> None:
    for f in flags:
        if FLAGS[f](clip) == 0:
            raise ValueError(f"model flag +{f} changed nothing in this model")


def parse_name(name: str) -> tuple[str, int | None]:
    if "@" in name:
        base, res = name.split("@")
        return base, int(res)
    return name, None


def build(name: str, device="cpu") -> tuple[nn.Module, nn.Module]:
    import open_clip

    name, ckpt, alpha = split_ckpt(name)
    name, flags = split_flags(name)
    base, res = parse_name(name)
    if base in ("clip-b16", "clip-b16-norm"):
        clip = open_clip.create_model("ViT-B-16", pretrained="openai")
        if ckpt:
            load_finetuned(clip, ckpt, alpha)
        apply_flags(clip, flags)
        img = ImageEncoder(clip, res or 224, OPENAI_MEAN, OPENAI_STD, normalize_input=(base == "clip-b16-norm"))
        txt = TextEncoder(clip, zero_pad_after_eot=False)
    elif base in MOBILECLIP2:
        # open_clip's pretrained cfg already carries the right mean/std (0/1 for S0/S2/B, per ml-mobileclip).
        clip = open_clip.create_model(MOBILECLIP2[base], pretrained="dfndr2b")
        if ckpt:
            load_finetuned(clip, ckpt, alpha)
        clip.eval()
        clip = reparameterize(clip)
        apply_flags(clip, flags)
        cfg = clip.visual.preprocess_cfg
        native = cfg["size"][0] if isinstance(cfg["size"], (tuple, list)) else cfg["size"]
        img = ImageEncoder(clip, res or native, cfg["mean"], cfg["std"], normalize_input=True)
        txt = TextEncoder(clip, zero_pad_after_eot=True)
    else:
        raise KeyError(f"unknown model {name!r}")
    return img.eval().to(device), txt.eval().to(device)


def count_params(module: nn.Module, prefix: str) -> int:
    return sum(p.numel() for n, p in module.named_parameters() if n.startswith(prefix))
