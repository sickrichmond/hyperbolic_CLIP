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

`shift / minimum margin` turned out to order every model measured so far, with a sharp
threshold at 1 (below it AUC 0.94-1.00, above it 0.64-0.66). --styled asks whether that
is a law about DEGRADATION or a law about SHIFT, by swapping the degraded view for the
same content rendered in another style — the styled sets append --style to the same
dense caption, so the pairing is exact:

    IAB_EXCLUDE_GENERATORS=dalle3 python -m tests.probe_degradation_shift \\
        --styled $FAST/datasets/iab_recap_dataset_v2 $FAST/datasets/iab_recap_cartoon_v2 \\
        $CK/attribution_22cls_sweepwin_vitl14.pt $CK/attribution_22cls_euclidean_d128_vitl14.pt
"""
import argparse
import math
import os

os.environ.setdefault("IAB_EXCLUDE_GENERATORS", "dalle3")

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from comparison.training.test_hypclip import harness_class_names
from data.degradations import LEVEL_LABELS
from data.iab_clip_dataset import IABCLIPDataset
from data.image_io import open_image_retry
from tests.probe_anchor_spread import load_any

SEMANTICS = ["COCO", "cat", "dog", "wild", "FFHQ", "celebahq", "bedroom", "church",
             "classroom", "ImageNet-1k"]
EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--levels", type=int, nargs="+", default=[3, 6],
                   help="degradation levels to measure (default JPEG65 and Blur5)")
    p.add_argument("--dataset_path", default=os.environ.get("FAST", "") + "/datasets/iab_dataset")
    p.add_argument("--captions_dir", default=os.environ.get("WORK", "") + "/hyp_fine_tuning/iab_captions")
    p.add_argument("--manifest", default=os.environ.get("WORK", "") + "/hyp_fine_tuning/split_manifest_22cls.json")
    p.add_argument("--styled", nargs=2, metavar=("ROOT_A", "ROOT_B"),
                   help="measure the SEMANTIC shift instead of a degradation: two styled "
                        "roots, e.g. .../iab_recap_dataset_v2 .../iab_recap_cartoon_v2")
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


class StyledView(Dataset):
    """Just enough of IABCLIPDataset for embed(): open the file, CLIP-preprocess it.

    The styled sets carry no captions on disk and none of their four generators is
    grok3, so the caption, degradation and watermark-crop branches would all be
    dead code here.
    """

    def __init__(self, paths, clip_name):
        from transformers import CLIPImageProcessor
        self.paths = paths
        self.processor = CLIPImageProcessor.from_pretrained(clip_name)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = open_image_retry(str(self.paths[i]))
        return {"pixel_values":
                self.processor(images=img, return_tensors="pt")["pixel_values"][0]}


def styled_pair(args, clip_name):
    """The same CONTENT in two styles — the semantic analogue of clean vs degraded.

    Styled sets are produced by appending --style to the SAME dense caption
    (dataset_rebuilding/generate_fakes.py:286), so a given relative path
    <generator>/<semantic>/<stem> names the same source image in every style.
    Intersecting the two file trees is therefore an exact content pairing, and the
    resulting angle is directly comparable with the degradation shifts and with the
    anchor margins.
    """
    key = (clip_name, tuple(args.styled))
    if key in _CACHE:
        return _CACHE[key]
    roots = [Path(r) for r in args.styled]
    rel = [{p.relative_to(r) for p in r.rglob("*") if p.suffix.lower() in EXTENSIONS}
           for r in roots]
    shared = sorted(rel[0] & rel[1])
    print(f"  styled pairing: {len(rel[0])} / {len(rel[1])} files, "
          f"{len(shared)} shared relative paths")

    g = torch.Generator().manual_seed(0)
    keep = torch.randperm(len(shared), generator=g)[:args.n].tolist()
    picked = [shared[i] for i in keep]
    _CACHE[key] = (StyledView([roots[0] / p for p in picked], clip_name),
                   StyledView([roots[1] / p for p in picked], clip_name))
    return _CACHE[key]


def pair_sources(args, names, clip_name):
    """(label, view_a, view_b) for every shift axis this run measures."""
    if args.styled:
        yield "→".join(Path(r).name.replace("iab_recap_", "") for r in args.styled), \
            *styled_pair(args, clip_name)
        return
    for level in args.levels:
        yield LEVEL_LABELS.get(level, str(level)), *paired_datasets(args, names, clip_name, level)


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

        for label, ds_a, ds_b in pair_sources(args, names, clip_name):
            x0 = embed(model, ds_a, args, device, "view A")
            x1 = embed(model, ds_b, args, device, label)

            cos = F.cosine_similarity(x0, x1, dim=-1).clamp(-1.0, 1.0)
            shift = torch.rad2deg(cos.arccos())
            p50, p90, p99 = shift.quantile(torch.tensor([0.5, 0.9, 0.99])).tolist()
            pred0 = (F.normalize(x0, dim=-1) @ anc.cpu().T).argmax(1)
            pred1 = (F.normalize(x1, dim=-1) @ anc.cpu().T).argmax(1)
            flipped = (pred0 != pred1).float().mean().item()
            over = (shift > margin).float().mean().item()

            print(f"  {label:18s} shift p50={p50:6.1f}°  p90={p90:6.1f}°  p99={p99:6.1f}°   "
                  f"over margin {over:6.1%}   class flipped {flipped:6.1%}   "
                  f"shift/margin {p50 / margin:.2f}")
            rows.append((os.path.basename(path), label, margin, p50, over, flipped))

    print(f"\n{'checkpoint':44s} {'axis':18s} {'margin':>7s} {'shift p50':>10s} "
          f"{'>margin':>8s} {'flipped':>8s} {'ratio':>7s}")
    for name, label, margin, p50, over, flipped in rows:
        print(f"{name:44s} {label:18s} {margin:6.1f}° {p50:9.1f}° {over:7.1%} "
              f"{flipped:7.1%} {p50 / margin:6.2f}")
    print("\n'ratio' is shift/margin, the quantity that orders every model measured so far\n"
          "with a threshold at 1 (below: AUC 0.94-1.00, above: 0.64-0.66). --styled asks\n"
          "whether that threshold also governs SEMANTIC shift, not just degradation.")


if __name__ == "__main__":
    main()
