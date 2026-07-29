"""Is there ANY class signal in CLIP(spectrum)? A linear probe, ~5 minutes.

The centroid caches only hold the 22 class MEANS, so they show the between-class
separation but say nothing about the within-class scatter — and separability
depends on the ratio. This encodes a small subset of real images through both
branches of the frozen backbone and fits a logistic regression on each:

    pixel probe   = the control. It must score high; the full model does.
    spectral probe = the answer. Near 1/22 = chance means the FFT-into-CLIP
                     embedding carries no linearly decodable class signal, and no
                     training budget will rescue that branch.

LoRA is zero-initialised, so the frozen backbone here is exactly what the
spectral branch starts from.

Usage (GPU node, few minutes):
    python -m tests.probe_spectral \\
        --dataset_path $FAST/datasets/iab_dataset \\
        --captions_dir $WORK/hyp_fine_tuning/iab_captions \\
        --split_manifest $WORK/hyp_fine_tuning/split_manifest_22cls.json \\
        --generators real 4o ... --max_per_class 20
"""
import argparse
import json
from collections import Counter

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.iab_clip_dataset import IABCLIPDataset
from patch_freq_attribution.model import PatchFreqAttributionCLIP


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--captions_dir", required=True)
    p.add_argument("--split_manifest", required=True)
    p.add_argument("--generators", nargs="+", required=True)
    p.add_argument("--semantics", nargs="+",
                   default=["COCO", "cat", "dog", "wild", "FFHQ", "celebahq",
                            "bedroom", "church", "classroom", "ImageNet-1k"])
    p.add_argument("--clip_name", default="openai/clip-vit-large-patch14")
    p.add_argument("--max_per_class", type=int, default=20,
                   help="per (generator, semantic) — x10 semantics ≈ 200 per class")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.split_manifest) as f:
        train_paths = set(json.load(f)["train"])
    ds = IABCLIPDataset(
        root=args.dataset_path, captions_dir=args.captions_dir,
        generators=args.generators, semantics=args.semantics,
        processor_name=args.clip_name, max_per_class=args.max_per_class,
        split="all", require_caption=False, include_paths=train_paths,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = PatchFreqAttributionCLIP(clip_name=args.clip_name).to(device).eval()

    pix, spec, labels = [], [], []
    for batch in tqdm(loader, desc="encoding"):
        x = batch["pixel_values"].to(device)
        with autocast("cuda"):
            pix.append(model._clip_image(x).float().cpu())
            spec.append(model._clip_spectrum(x).float().cpu())
        labels.extend(batch["generator"])

    names = sorted(set(labels))
    y = torch.tensor([names.index(g) for g in labels]).numpy()
    print(f"\n{len(y)} images, {len(names)} classes: {dict(Counter(labels))}")

    for tag, feats in (("pixel   ", torch.cat(pix)), ("spectral", torch.cat(spec))):
        X = feats.numpy()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42,
                                              stratify=y)
        clf = LogisticRegression(max_iter=2000, n_jobs=-1).fit(Xtr, ytr)
        acc = balanced_accuracy_score(yte, clf.predict(Xte))
        print(f"  {tag} linear probe: balanced acc = {100*acc:5.1f}%  "
              f"(chance {100/len(names):.1f}%)")

    print("\nIf the spectral probe is near chance while the pixel one is not, the "
          "spectral branch is dead as constructed — change the representation, do "
          "not spend a 23h job on it.")


if __name__ == "__main__":
    main()
