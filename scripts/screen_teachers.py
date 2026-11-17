"""Zero-shot screening of large TEACHER models (and similarity-averaged ensembles) on the validation folds.

Teachers are never deployed, so latency does not matter; we only need the most accurate one(s) on LPCVC-style
captions to distill from. Ensembles average the cosine-similarity matrices of their members.

  python scripts/screen_teachers.py --exp teacher-screen --teachers mobileclip2-b mobileclip2-s4 mobileclip2-l14 \
      --ensembles mobileclip2-l14+mobileclip2-s4
Sims are cached in .cache/teacher_sims/<teacher>/<set>.npy so ensembles and reruns are free.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_local import PRESETS, resolve_sets  # noqa: E402
from lpcvc import metrics  # noqa: E402
from lpcvc.data import load_lpcvc_sample  # noqa: E402
from lpcvc.experiment import Experiment  # noqa: E402
from lpcvc.teachers import Teacher  # noqa: E402

CACHE = ROOT / ".cache" / "teacher_sims"


def sims_for(name, sets, device):
    out, teacher = {}, None
    for ds in sets:
        f = CACHE / name / (ds.name.replace("/", "_") + ".npy")
        if f.exists():
            out[ds.name] = np.load(f)
            continue
        if teacher is None:
            teacher = Teacher(name, device)
            print(f"loaded {name} (res {teacher.res})")
        ie = teacher.encode_images([str(p) for p in ds.image_paths])
        te = teacher.encode_texts(ds.captions)
        f.parent.mkdir(parents=True, exist_ok=True)
        np.save(f, ie @ te.T)
        out[ds.name] = ie @ te.T
    del teacher
    torch.cuda.empty_cache()
    return out


def per_image_from_sims(sim, ds):
    return metrics.per_image_recall(sim, ds.gt, ds.txt_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teachers", nargs="+", required=True)
    ap.add_argument("--ensembles", nargs="*", default=[], help="'+'-joined teacher names")
    ap.add_argument("--data", nargs="+", default=["refcocog", "sample"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--exp", default="teacher-screen")
    ap.add_argument("--title", default="Teacher screening (zero-shot, local)")
    ap.add_argument("--question", default="Which large model (or ensemble) is the most accurate teacher on LPCVC-style captions?")
    args = ap.parse_args()
    with Experiment(args.exp, args.title, args.question, args) as exp:
        sets = []
        for name, path in resolve_sets(args.data):
            ds = load_lpcvc_sample(path)
            ds.name = name
            sets.append(ds)
        sims = {t: sims_for(t, sets, args.device) for t in args.teachers}
        for e in args.ensembles:
            members = e.split("+")
            sims[e] = {ds.name: np.mean([sims[m][ds.name] for m in members], axis=0) for ds in sets}
        groups = {spec: PRESETS.get(spec, [spec]) for spec in args.data}
        per = {m: {ds.name: per_image_from_sims(s[ds.name], ds) for ds in sets} for m, s in sims.items()}
        names = list(sims)
        base = names[0]
        rows, lines = [], ["| Teacher | " + " | ".join(f"{g} frac R@10 [95% CI]" for g in groups)
                           + f" | delta vs {base} on {list(groups)[0]} |", "|" + "---|" * (len(groups) + 2)]
        g0 = list(groups)[0]
        a = np.concatenate([per[base][m] for m in groups[g0]])
        for n in names:
            cells = []
            for g, members in groups.items():
                p = np.concatenate([per[n][m] for m in members])
                lo, hi = metrics.bootstrap_ci(p)
                rows.append({"model": n, "set": g, "frac_R@10": float(p.mean()), "frac_R@10_ci95": (lo, hi),
                             "fold_means": [float(per[n][m].mean()) for m in members]})
                cells.append(f"{p.mean()*100:.2f} [{lo*100:.1f}, {hi*100:.1f}]")
            b = np.concatenate([per[n][m] for m in groups[g0]])
            d, lo, hi = metrics.paired_bootstrap_delta(a, b)
            delta = "-" if n == base else f"{d*100:+.2f} [{lo*100:+.1f}, {hi*100:+.1f}]{' *' if lo > 0 or hi < 0 else ''}"
            lines.append(f"| {n} | " + " | ".join(cells) + f" | {delta} |")
            print(lines[-1])
        exp.add_table("\n".join(lines) + "\n\n`*` = paired 95% CI excludes 0.")
        exp.save_json("results.json", {"rows": rows})
        exp.save_npz("per_image_recall.npz", **{f"{m}|{s}": v for m, d in per.items() for s, v in d.items()})
        best = max((r for r in rows if r["set"] == g0), key=lambda r: r["frac_R@10"])
        exp.headline = f"best teacher on {g0}: {best['model']} {best['frac_R@10']*100:.2f}"


if __name__ == "__main__":
    main()
