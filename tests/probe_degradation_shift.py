"""How far does a degradation move the embedding, in the same units as the anchor margins?

The robustness gap is not explained yet. Two hypotheses died:
  - "tight anchors cause the collapse" — omniaug has the tightest anchors measured
    (1.7°) and is the most robust;
  - "widening the margin cures it" — Run C has 4x sweepwin's margin (8.8° vs 2.2°)
    and collapses identically under JPEG (AUC 0.637 vs 0.643).

What survives is a quantitative version: a margin only helps if it is larger than the
displacement the degradation induces. This measures that displacement directly — the
angle between the embedding of the SAME image, clean and degraded — and puts it next
to the anchor margins from tests/probe_anchor_spread.py.

    displacement >> margin, for both hyperbolic runs  -> it IS a threshold story, and
        8.8° was simply too small; only the euclidean's 65° clears it.
    displacement small (a few degrees)                -> the margin is irrelevant and
        the euclidean's advantage lives in the ENCODER, not in the anchors.

Predictions are `argmax_c cos(x, a_c)` for both geometries: on the euclidean model that
IS the decision rule, and on the hyperbolic one it agrees with `argmin_c ξ_c` on 0.9998
of images (tests/probe_cone_vs_cosine.py), so the flip rate is the model's own.

    IAB_EXCLUDE_GENERATORS=dalle3 python -m tests.probe_degradation_shift \\
        $CK/attribution_22cls_sweepwin_vitl14.pt \\
        $CK/attribution_22cls_promptsC_ce_vitl14.pt \\
        $CK/attribution_22cls_euclidean_d128_vitl14.pt

Levels are the eval's: 1=DS0.5 2=DS0.25 3=JPEG65 4=JPEG30 5=Blur3 6=Blur5. GPU node.
"""
import argparse
import math
import os

os.environ.setdefault("IAB_EXCLUDE_GENERATORS", "dalle3")

import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from comparison.training.test_hypclip import harness_class_names
from data.degradations import LEVEL_LABELS
from data.iab_clip_dataset import IABCLIPDataset
from tests.probe_anchor_spread import load_any

SEMANTICS = ["COCO", "cat", "dog", "wild", "FFHQ", "celebahq", "bedroom", "church",
             "classroom", "ImageNet-1k"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--levels", type=int, nargs="+", default=[3, 6],
                   help="degradation levels to measure (default JPEG65 and Blur5)")
    p.add_argument("--dataset_path", default=os.environ.get("FAST", "") + "/datasets/iab_dataset")
    p.add_argument("--captions_dir", default=os.environ.get("WORK", "") + "/hyp_fine_tuning/iab_captions")
    p.add_argument("--manifest", default=os.environ.get("WORK", "") + "/hyp_fine_tuning/split_manifest_22cls.json")
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    return p.parse_args()


_CACHE = {}


def paired_datasets(args, names, clip_name, level):
    """The same images twice, clean and degraded — identical order, asserted.

    Cached: enumerating 437k files takes about as long as the forward passes, and every
    checkpoint here shares the same clip_name and levels.
    """
    key = (clip_name, level)
    if key in _CACHE:
        return _CACHE[key]
    with open(args.manifest) as f:
        val_paths = set(json.load(f)["val"])
    common = dict(root=args.dataset_path, captions_dir=args.captions_dir,
                  generators=names, semantics=SEMANTICS, processor_name=clip_name,
                  split="all", include_paths=val_paths, require_caption=False,
                  include_uncaptioned=True)
    clean = IABCLIPDataset(**common, degraded=0)
    degraded = IABCLIPDataset(**common, degraded=level)
    assert [s[0] for s in clean.samples] == [s[0] for s in degraded.samples], \
        "the two datasets are not aligned image-by-image"

    g = torch.Generator().manual_seed(0)
    keep = torch.randperm(len(clean), generator=g)[:args.n].tolist()
    _CACHE[key] = (Subset(clean, keep), Subset(degraded, keep))
    return _CACHE[key]


@torch.no_grad()
def embed(model, ds, args, device, desc):
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    out = []
    for batch in tqdm(loader, desc=desc, leave=False):
        x = model.encode_image(batch["pixel_values"].to(device))
        out.append((x[0] if isinstance(x, tuple) else x).cpu())   # hyperbolic returns a tuple
    return torch.cat(out)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = harness_class_names()
    deg = lambda c: math.degrees(math.acos(min(max(float(c), -1.0), 1.0)))
    rows = []

    for path in args.checkpoints:
        model, x_anc, _, desc, clip_name = load_any(path, device)
        if x_anc.shape[0] != len(names):
            raise ValueError(f"{os.path.basename(path)}: {x_anc.shape[0]} anchors vs "
                             f"{len(names)} active classes — fix IAB_EXCLUDE_GENERATORS")
        anc = F.normalize(x_anc, dim=-1)
        A = (anc @ anc.T).clamp(-1.0, 1.0)
        i, j = torch.triu_indices(len(names), len(names), offset=1)
        margin = deg(A[i, j].max())          # the closest pair: the tightest boundary

        print(f"\n═════ {os.path.basename(path)}   ({desc})")
        print(f"  closest anchor pair: {margin:.1f}°")

        for level in args.levels:
            clean_ds, deg_ds = paired_datasets(args, names, clip_name, level)
            x0 = embed(model, clean_ds, args, device, "clean")
            x1 = embed(model, deg_ds, args, device, LEVEL_LABELS.get(level, str(level)))

            cos = F.cosine_similarity(x0, x1, dim=-1).clamp(-1.0, 1.0)
            shift = torch.rad2deg(cos.arccos())
            p50, p90, p99 = shift.quantile(torch.tensor([0.5, 0.9, 0.99])).tolist()
            pred0 = (F.normalize(x0, dim=-1) @ anc.cpu().T).argmax(1)
            pred1 = (F.normalize(x1, dim=-1) @ anc.cpu().T).argmax(1)
            flipped = (pred0 != pred1).float().mean().item()
            over = (shift > margin).float().mean().item()

            label = LEVEL_LABELS.get(level, str(level))
            print(f"  {label:7s} shift p50={p50:6.1f}°  p90={p90:6.1f}°  p99={p99:6.1f}°   "
                  f"over margin {over:6.1%}   class flipped {flipped:6.1%}")
            rows.append((os.path.basename(path), label, margin, p50, over, flipped))

    print(f"\n{'checkpoint':44s} {'level':7s} {'margin':>7s} {'shift p50':>10s} "
          f"{'>margin':>8s} {'flipped':>8s}")
    for name, label, margin, p50, over, flipped in rows:
        print(f"{name:44s} {label:7s} {margin:6.1f}° {p50:9.1f}° {over:7.1%} {flipped:7.1%}")
    print("\nIf 'shift p50' dwarfs 'margin' everywhere, robustness is a threshold problem and\n"
          "only a margin bigger than the shift can help. If the shift is small and the class\n"
          "still flips, the anchors are not what separates the robust model from the fragile one.")



if __name__ == "__main__":
    main()
