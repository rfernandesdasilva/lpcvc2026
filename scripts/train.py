"""Fine-tune a deployable student (MobileCLIP2-*) on the LPCVC-style training set, optionally with distillation.

Training runs THROUGH the deployment wrappers (lpcvc.models.ImageEncoder/TextEncoder): images enter as the official
[0,1] 224x224 squash-resized tensors, text as official EOT-padded tokens, so whatever the phone runs is what we train.

Loss (per batch of B images, each with up to K sampled captions):
  contrastive   multi-positive InfoNCE in both directions; any caption string that belongs to an image is a
                positive for it (identical strings of different images are merged, so no false negatives).
  distill       TinyCLIP-style affinity mimicking: KL(teacher || student) over the in-batch image->text and
                text->image similarity distributions. Teacher embeddings are precomputed once (both flips) and cached.
  total = (1 - alpha) * contrastive + alpha * distill        (alpha = 0 -> plain fine-tuning control)

--chunk N computes the same full-batch loss with gradient caching (GradCache, Gao et al. 2021): embeddings are
computed without graphs, the loss is backpropagated to them, then each chunk is re-forwarded with its cached
gradient. Exact up to bf16 rounding (no dropout; BatchNorm stats frozen), so the recipe is unchanged.

No early stopping or checkpoint selection on validation: the recipe is fixed before the run and the LAST epoch is
the result. Per-epoch validation is logged only to draw learning curves.

  python scripts/train.py --exp ft-b-contrastive --student mobileclip2-b
  python scripts/train.py --exp distill-b-l14 --student mobileclip2-b --teacher mobileclip2-l14 --alpha 0.5
Checkpoint: checkpoints/<experiment dir>/model.pt (fp16, not reparameterized). Evaluate/deploy it with
  --models "mobileclip2-b#checkpoints/<experiment dir>/model.pt"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_local import PRESETS, embed, resolve_sets  # noqa: E402
from lpcvc import metrics  # noqa: E402
from lpcvc.data import load_lpcvc_sample, tokenize  # noqa: E402
from lpcvc.experiment import Experiment  # noqa: E402
from lpcvc.models import MOBILECLIP2, ImageEncoder, TextEncoder  # noqa: E402

TRAIN = ROOT / "data" / "train_refmix"
CACHE = ROOT / ".cache"


# ---- data -----------------------------------------------------------------------------------------
def load_train(sources=None):
    items = [json.loads(l) for l in open(TRAIN / "train.jsonl", encoding="utf-8")]
    if sources:   # keep only captions that come from the given source datasets (cross-source ablation)
        src = json.loads((TRAIN / "caption_sources.json").read_text(encoding="utf-8"))
        items = [{"image": it["image"], "captions": [c for c in it["captions"] if set(src[c]) & set(sources)]}
                 for it in items]
        items = [it for it in items if it["captions"]]
    caps = sorted({c for it in items for c in it["captions"]})
    cid = {c: i for i, c in enumerate(caps)}
    img_caps = [[cid[c] for c in it["captions"]] for it in items]
    paths = [str(TRAIN / "images" / it["image"]) for it in items]
    # cache key = the (filtered) training content itself, so --sources subsets never reuse a wrong cache
    digest = hashlib.sha256(json.dumps([[it["image"] for it in items], caps]).encode()).hexdigest()[:16]
    f = CACHE / f"train_tokens_{digest}.npy"
    if f.exists():
        tokens = np.load(f)
    else:
        tokens = np.concatenate([tokenize(caps[i:i + 4096]) for i in range(0, len(caps), 4096)])
        np.save(f, tokens)
    return paths, caps, img_caps, tokens, digest


class ImageDS(Dataset):
    """Official preprocessing (PIL resize to 224x224, no crop) + random horizontal flip (location words are
    filtered out of the captions, so flipping never contradicts a caption)."""

    def __init__(self, paths, flip=True):
        self.paths, self.flip = paths, flip

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB").resize((224, 224))
        f = int(self.flip and torch.rand(1).item() < 0.5)
        if f:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        return torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1), i, f


# ---- teacher --------------------------------------------------------------------------------------
def teacher_embeddings(name, paths, caps, digest, device):
    """(img[N, 2, D] for no-flip/flip, txt[C, D], logit_scale) for one teacher, cached on disk (fp16)."""
    f = CACHE / "teacher_emb" / name / f"train_{digest}.npz"
    if f.exists():
        z = np.load(f)
        return z["img"], z["txt"], float(z["scale"])
    from lpcvc.teachers import Teacher

    t = Teacher(name, device)
    t0 = time.time()
    img = np.stack([t.encode_images(paths, flip=False), t.encode_images(paths, flip=True)], axis=1)
    txt = t.encode_texts(caps)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, img=img.astype(np.float16), txt=txt.astype(np.float16), scale=t.logit_scale)
    print(f"teacher {name}: embedded {len(paths)} images x2 + {len(caps)} captions in {time.time() - t0:.0f}s")
    scale = t.logit_scale
    del t
    torch.cuda.empty_cache()
    return img.astype(np.float16), txt.astype(np.float16), scale


# ---- losses ---------------------------------------------------------------------------------------
def multi_pos_ce(logits, pos):
    return (logits.logsumexp(1) - logits.masked_fill(~pos, float("-inf")).logsumexp(1)).mean()


def kl(s, t, T):
    return F.kl_div(F.log_softmax(s / T, 1), F.log_softmax(t / T, 1), log_target=True, reduction="batchmean") * T * T


# ---- eval -----------------------------------------------------------------------------------------
class Evaluator:
    def __init__(self, specs, device):
        self.device = device
        self.groups = {s: PRESETS.get(s, [s]) for s in specs}
        self.sets = []
        for name, path in resolve_sets(specs):
            ds = load_lpcvc_sample(path)
            ds.name = name
            self.sets.append((ds, ds.images(), ds.tokens()))

    @torch.no_grad()
    def __call__(self, img_enc, txt_enc):
        img_enc.eval(); txt_enc.eval()
        per = {}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for ds, images, tokens in self.sets:
                ie, te = embed(img_enc, txt_enc, images, tokens, self.device, bs=128)
                per[ds.name] = metrics.per_image_recall(ie @ te.T, ds.gt, ds.txt_ids)
        return per, {g: float(np.concatenate([per[m] for m in ms]).mean()) for g, ms in self.groups.items()}


# ---- main -----------------------------------------------------------------------------------------
def build_student(name, device):
    import open_clip

    clip = open_clip.create_model(MOBILECLIP2[name], pretrained="dfndr2b")   # NOT reparameterized (train-time)
    cfg = clip.visual.preprocess_cfg
    native = cfg["size"][0] if isinstance(cfg["size"], (tuple, list)) else cfg["size"]
    img = ImageEncoder(clip, native, cfg["mean"], cfg["std"], normalize_input=True)
    txt = TextEncoder(clip, zero_pad_after_eot=True)
    return clip.to(device), img.to(device), txt.to(device)


def set_train(clip):
    clip.train()
    for m in clip.modules():          # keep BatchNorm statistics frozen (S0/S2 conv towers); no-op for ViTs
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="mobileclip2-b", choices=list(MOBILECLIP2))
    ap.add_argument("--teacher", default=None, help="teacher name, or '+'-joined ensemble (sims averaged)")
    ap.add_argument("--alpha", type=float, default=0.0, help="distillation weight (0 = contrastive only)")
    ap.add_argument("--tau", type=float, default=1.0, help="distillation temperature (on top of logit scales)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=256, help="images per batch")
    ap.add_argument("--k", type=int, default=2, help="captions sampled per image per step")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=float, default=0.1, help="fraction of steps")
    ap.add_argument("--sources", nargs="*", default=None, choices=["refcoco", "refcocoplus", "refcocog"],
                    help="train only on captions from these datasets (default: all)")
    ap.add_argument("--freeze", choices=["none", "image", "text"], default="none")
    ap.add_argument("--chunk", type=int, default=0,
                    help="gradient caching chunk size (0 = off). Same loss and gradients as the full batch, "
                         "less memory, ~1.3x more compute; for students whose full batch does not fit on the GPU")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-data", nargs="+", default=["refcocog", "sample"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--question", default="Does fine-tuning on LPCVC-style captions improve frac R@10?")
    args = ap.parse_args()
    title = args.title or f"Train {args.student}" + (f" (distill from {args.teacher}, alpha={args.alpha})" if args.alpha else " (contrastive)")
    with Experiment(args.exp, title, args.question, args) as exp:
        run(args, exp)


def run(args, exp):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = args.device
    paths, caps, img_caps, tokens, digest = load_train(args.sources)
    n_caps_img = np.array([len(c) for c in img_caps])
    print(f"train: {len(paths)} images, {len(caps)} unique captions ({n_caps_img.mean():.2f}/img), data sha {digest}")
    exp.manifest["train_data"] = {"images": len(paths), "captions": len(caps), "content_sha256_prefix": digest,
                                  "train_jsonl_sha256": hashlib.sha256((TRAIN / "train.jsonl").read_bytes()).hexdigest(),
                                  "sources": args.sources or "all"}

    teachers = []
    if args.alpha > 0:
        assert args.teacher, "--alpha > 0 needs --teacher"
        for t in args.teacher.split("+"):
            ti, tt, ts = teacher_embeddings(t, paths, caps, digest, dev)
            teachers.append((torch.from_numpy(ti).to(dev), torch.from_numpy(tt).to(dev), ts))
        t_scale = float(np.mean([t[2] for t in teachers]))
        print(f"teacher(s) {args.teacher}: logit scale {t_scale:.1f}")

    clip, img_enc, txt_enc = build_student(args.student, dev)
    if hasattr(clip, "set_grad_checkpointing"):
        clip.set_grad_checkpointing(True)
    clip.logit_scale.requires_grad_(False)
    scale = float(clip.logit_scale.exp())
    for n, p in clip.named_parameters():
        if (args.freeze == "image" and n.startswith("visual.")) or (args.freeze == "text" and not n.startswith("visual.")):
            p.requires_grad_(False)
    decay = [p for n, p in clip.named_parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for n, p in clip.named_parameters() if p.requires_grad and p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.98), eps=1e-6)
    loader = DataLoader(ImageDS(paths), batch_size=args.bs, shuffle=True, drop_last=True,
                        num_workers=args.workers, persistent_workers=True, pin_memory=True)
    steps = args.epochs * len(loader)
    warm = max(1, int(args.warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    tok = torch.from_numpy(tokens)

    evaluator = Evaluator(args.eval_data, dev)
    per0, m0 = evaluator(img_enc, txt_enc)
    print(f"epoch 0 (pretrained): " + "  ".join(f"{g}={v*100:.2f}" for g, v in m0.items()))
    curve = [{"epoch": 0, "step": 0, **{f"val_{g}": v for g, v in m0.items()}}]

    step, t0 = 0, time.time()
    for ep in range(1, args.epochs + 1):
        set_train(clip)
        run_l = {"loss": 0.0, "con": 0.0, "dist": 0.0}
        for x, idx, flip in loader:
            idx = idx.numpy()
            # sample up to K distinct captions per image
            tcap, owner = [], []
            for b, i in enumerate(idx):
                c = img_caps[i]
                pick = rng.choice(len(c), size=min(args.k, len(c)), replace=False)
                tcap += [c[p] for p in pick]; owner += [b] * len(pick)
            batch_caps = {}
            for b, i in enumerate(idx):
                for c in img_caps[i]:
                    batch_caps.setdefault(c, []).append(b)
            pos = torch.zeros(len(idx), len(tcap), dtype=torch.bool)
            for j, c in enumerate(tcap):
                pos[batch_caps[c], j] = True
            pos = pos.to(dev, non_blocking=True)
            x = x.to(dev, non_blocking=True).float().div_(255)
            t = tok[tcap].to(dev, non_blocking=True)

            C = args.chunk
            with torch.autocast("cuda", dtype=torch.bfloat16), torch.set_grad_enabled(not C):
                zi = torch.cat([img_enc(x[i:i + C]) for i in range(0, len(x), C)]) if C else img_enc(x)
                zt = torch.cat([txt_enc(t[i:i + C]) for i in range(0, len(t), C)]) if C else txt_enc(t)
            if C:   # gradient caching: loss on detached embeddings first, then re-forward chunks with their grads
                zi = zi.float().detach().requires_grad_()
                zt = zt.float().detach().requires_grad_()
            logits = scale * zi.float() @ zt.float().T
            l_con = 0.5 * (multi_pos_ce(logits, pos) + multi_pos_ce(logits.T, pos.T))
            loss = (1 - args.alpha) * l_con
            l_dist = torch.zeros((), device=dev)
            if args.alpha > 0:
                ii, ff = torch.from_numpy(idx).to(dev), flip.to(dev)
                tc = torch.tensor(tcap, device=dev)
                tl = sum(ti[ii, ff].float() @ tt[tc].float().T for ti, tt, _ in teachers) / len(teachers) * t_scale
                l_dist = 0.5 * (kl(logits, tl, args.tau) + kl(logits.T, tl.T, args.tau))
                loss = loss + args.alpha * l_dist
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if C:
                for enc, inp, z in ((img_enc, x, zi), (txt_enc, t, zt)):
                    for i in range(0, len(inp), C):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            zc = enc(inp[i:i + C])
                        zc.float().backward(z.grad[i:i + C])
            torch.nn.utils.clip_grad_norm_([p for p in clip.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); step += 1
            run_l["loss"] += loss.item(); run_l["con"] += l_con.item(); run_l["dist"] += l_dist.item()
            if step % 20 == 0:
                print(f"  step {step}/{steps} loss {loss.item():.4f} con {l_con.item():.4f} dist {l_dist.item():.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s")
        n = len(loader)
        per, m = evaluator(img_enc, txt_enc)
        curve.append({"epoch": ep, "step": step, **{k: v / n for k, v in run_l.items()}, **{f"val_{g}": v for g, v in m.items()}})
        print(f"epoch {ep}: loss {run_l['loss']/n:.4f}  " + "  ".join(f"{g}={v*100:.2f}" for g, v in m.items())
              + f"  ({time.time() - t0:.0f}s)")

    ck = ROOT / "checkpoints" / exp.dir.name / "model.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.half() if v.is_floating_point() else v for k, v in clip.state_dict().items()}, ck)
    ck_rel = ck.relative_to(ROOT).as_posix()
    print(f"checkpoint: {ck_rel}")

    # report: last epoch vs. pretrained (same pipeline), paired bootstrap per group
    lines = ["| Model | " + " | ".join(f"{g} frac R@10 [95% CI]" for g in evaluator.groups) + " | delta vs pretrained [95% CI] (first group) |",
             "|" + "---|" * (len(evaluator.groups) + 2)]
    rows = []
    g0 = list(evaluator.groups)[0]
    for label, p in ((f"{args.student} (pretrained)", per0), (f"{args.student}#{ck_rel}", per)):
        cells = []
        for g, ms in evaluator.groups.items():
            v = np.concatenate([p[x] for x in ms]); lo, hi = metrics.bootstrap_ci(v)
            cells.append(f"{v.mean()*100:.2f} [{lo*100:.1f}, {hi*100:.1f}]")
            rows.append({"model": label, "set": g, "frac_R@10": float(v.mean()), "frac_R@10_ci95": (lo, hi),
                         "fold_means": [float(p[x].mean()) for x in ms]})
        a = np.concatenate([per0[x] for x in evaluator.groups[g0]]); b = np.concatenate([p[x] for x in evaluator.groups[g0]])
        d, lo, hi = metrics.paired_bootstrap_delta(a, b)
        delta = "-" if p is per0 else f"{d*100:+.2f} [{lo*100:+.1f}, {hi*100:+.1f}]{' *' if lo > 0 or hi < 0 else ''}"
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {delta} |")
    curve_md = ["| Epoch | " + " | ".join(f"val {g}" for g in evaluator.groups) + " | train loss |", "|" + "---|" * (len(evaluator.groups) + 2)]
    for c in curve:
        curve_md.append(f"| {c['epoch']} | " + " | ".join(f"{c[f'val_{g}']*100:.2f}" for g in evaluator.groups)
                        + f" | {c.get('loss', float('nan')):.4f} |")
    exp.add_table("\n".join(lines) + "\n\n`*` = paired 95% CI excludes 0. Last epoch reported (no selection on validation).")
    exp.add_table("Learning curve (logged only, not used for selection):\n\n" + "\n".join(curve_md))
    exp.save_json("results.json", {"rows": rows, "curve": curve, "checkpoint": ck_rel})
    exp.save_npz("per_image_recall.npz", **{f"pretrained|{s}": v for s, v in per0.items()},
                 **{f"finetuned|{s}": v for s, v in per.items()})
    d = (np.concatenate([per[x] for x in evaluator.groups[g0]]).mean()
         - np.concatenate([per0[x] for x in evaluator.groups[g0]]).mean())
    exp.headline = f"{args.student} {g0}: {curve[0][f'val_{g0}']*100:.2f} -> {curve[-1][f'val_{g0}']*100:.2f} ({d*100:+.2f})"


if __name__ == "__main__":
    main()
