"""Retrieval metrics. `fractional_recall_at_k` reproduces the official `evaluate_track1` exactly
(see tests/test_metrics.py, which checks it against a verbatim copy of the official code)."""
from __future__ import annotations

import numpy as np


def l2norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def similarity(img_emb: np.ndarray, txt_emb: np.ndarray) -> np.ndarray:
    return l2norm(img_emb) @ l2norm(txt_emb).T


def _topk(sim: np.ndarray, k: int) -> np.ndarray:
    # Same tie-breaking as the official np.argsort(-sim)[:k]
    return np.argsort(-sim, axis=1)[:, :k]


def fractional_recall_at_k(sim, gt, txt_ids, k=10) -> float:
    """Official LPCVC Track 1 score: mean over images of |top-k ∩ GT| / |GT| (as a fraction)."""
    top = _topk(sim, k)
    txt_ids = np.asarray(txt_ids)
    scores = [len(set(txt_ids[top[i]].tolist()) & set(gt[i])) / len(gt[i]) for i in range(len(gt))]
    return float(np.mean(scores))


def hit_at_k(sim, gt, txt_ids, k=10) -> float:
    """Fraction of images with at least one relevant caption in the top k."""
    top = _topk(sim, k)
    txt_ids = np.asarray(txt_ids)
    return float(np.mean([bool(set(txt_ids[top[i]].tolist()) & set(gt[i])) for i in range(len(gt))]))


def per_image_recall(sim, gt, txt_ids, k=10) -> np.ndarray:
    top = _topk(sim, k)
    txt_ids = np.asarray(txt_ids)
    return np.array([len(set(txt_ids[top[i]].tolist()) & set(gt[i])) / len(gt[i]) for i in range(len(gt))])


def bootstrap_ci(per_image: np.ndarray, n=2000, alpha=0.05, seed=0) -> tuple[float, float]:
    """Percentile bootstrap CI over query images. Use this before claiming a difference on ~50 images."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_image), size=(n, len(per_image)))
    means = per_image[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_delta(a: np.ndarray, b: np.ndarray, n=2000, alpha=0.05, seed=0):
    """CI of mean(b - a) with images resampled jointly. If it excludes 0, b reliably beats a."""
    rng = np.random.default_rng(seed)
    d = b - a
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def report(img_emb, txt_emb, gt, txt_ids, k=10) -> dict:
    sim = similarity(img_emb, txt_emb)
    per = per_image_recall(sim, gt, txt_ids, k)
    lo, hi = bootstrap_ci(per)
    return {
        f"frac_R@{k}": float(per.mean()),
        f"frac_R@{k}_ci95": (lo, hi),
        f"hit@{k}": hit_at_k(sim, gt, txt_ids, k),
        "hit@1": hit_at_k(sim, gt, txt_ids, 1),
        "per_image": per,
    }
