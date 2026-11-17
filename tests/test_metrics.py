"""Our metric must equal the official scorer bit-for-bit, otherwise every number we report is suspect."""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lpcvc import metrics  # noqa: E402


def official_evaluate(img_embeds, txt_embeds, txt_id, gt, k=10):
    """Verbatim logic of evaluate_track1 in third_party/lpcvc_sample/inference.py (CSV parsing removed)."""
    img_embeds = img_embeds / np.linalg.norm(img_embeds, axis=1, keepdims=True)
    txt_embeds = txt_embeds / np.linalg.norm(txt_embeds, axis=1, keepdims=True)
    sim_matrix = cosine_similarity(img_embeds, txt_embeds)
    recalls = []
    for i in range(len(img_embeds)):
        gt_ids = [int(x) for x in gt[i].split(';')]
        top_k = np.argsort(-sim_matrix[i])[:k]
        predicted_txt_ids = [txt_id[idx] for idx in top_k]
        matched = len(set(predicted_txt_ids) & set(gt_ids))
        recalls.append(matched / len(gt_ids))
    return np.mean(recalls)


@pytest.mark.parametrize("seed", range(5))
def test_matches_official(seed):
    rng = np.random.default_rng(seed)
    n_img, n_txt, d = 56, 211, 512
    img, txt = rng.normal(size=(n_img, d)), rng.normal(size=(n_txt, d))
    txt_id = list(range(1, n_txt + 1))
    gt_str = [";".join(map(str, rng.choice(n_txt + 3, size=rng.integers(1, 7), replace=False) + 1)) for _ in range(n_img)]
    gt = [[int(x) for x in s.split(";")] for s in gt_str]
    ours = metrics.fractional_recall_at_k(metrics.similarity(img, txt), gt, txt_id, 10)
    assert ours == pytest.approx(official_evaluate(img, txt, txt_id, gt_str), abs=1e-12)


def test_hit_vs_fractional():
    sim = np.array([[0.9, 0.8, 0.1]])
    assert metrics.fractional_recall_at_k(sim, [[1, 3]], [1, 2, 3], k=1) == 0.5
    assert metrics.hit_at_k(sim, [[1, 3]], [1, 2, 3], k=1) == 1.0
