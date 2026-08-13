"""Phase A of the linear probe: cache image features once, probe them many times.

Three feature sources, all consumed by the same train_linear_probe.py, which is how
we separate "CLIP already knows" from "the LoRA learned it" from "the cones did it":

  (default)          frozen CLIP                 -> DetectorDF._clip_image
  --checkpoint X     CLIP + the trained LoRA     -> AttributionCLIP._clip_image
  --checkpoint X --features projection
                     the projection head output  -> the tangent vectors themselves

If a linear probe on the LoRA features already reaches the full model's accuracy,
the hyperbolic geometry is not buying accuracy and the paper has to say so.

--split_manifest puts the probe on the SAME images as everything in the results
tables (the harness split); without it the split is the legacy caption-based one
and the numbers are not comparable.

Usage:
    python -m scripts.extract_clip_features \\
        --dataset_path $FAST/datasets/iab_dataset --captions_dir $CAPS \\
        --generators real 4o ... --split_manifest $MANIFEST \\
        --out_dir $WORK/hyp_fine_tuning/clip_features_frozen
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.iab_clip_dataset import IABCLIPDataset
from models.det_on_frozen_CLIP import DetectorDF


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_path",   required=True)
    p.add_argument("--captions_dir",   required=True)
    p.add_argument("--clip_name",      default="openai/clip-vit-large-patch14")
    p.add_argument("--generators",     nargs="+", default=["real", "FLUX"])
    p.add_argument("--semantics",      nargs="+",
                   default=["COCO", "cat", "dog", "wild", "FFHQ", "celebahq",
                            "bedroom", "church", "classroom", "ImageNet-1k"])
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--val_frac",       type=float, default=0.2)
    p.add_argument("--max_per_class",  type=int,   default=None)
    p.add_argument("--num_workers",    type=int,   default=8)
    p.add_argument("--batch_size",     type=int,   default=256)
    p.add_argument("--split_manifest", default=None,
                   help="Harness split JSON (train/val/test path lists). Without it the "
                        "legacy caption-driven val_frac split is used and the probe is "
                        "NOT comparable with the tables.")
    p.add_argument("--checkpoint",     default=None,
                   help="Extract from a TRAINED AttributionCLIP instead of frozen CLIP.")
    p.add_argument("--features", choices=["clip", "projection"], default="clip",
                   help="--checkpoint only: 'clip' = the LoRA'd CLIP embedding, "
                        "'projection' = the tangent vector out of the projection head.")
    p.add_argument("--out_dir",   required=True)
    return p.parse_args()


def build_extractor(args, device):
    """Returns (fn: pixel_values -> features, description)."""
    if args.checkpoint is None:
        model = DetectorDF(clip_name=args.clip_name,
                           num_classes=len(args.generators)).to(device).eval()
        return model._clip_image, f"frozen CLIP ({args.clip_name})"

    from models.attribution_clip import AttributionCLIP
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = AttributionCLIP(
        clip_name=ckpt["clip_name"],
        lora_r=ckpt.get("lora_r", 8),
        lora_alpha=ckpt.get("lora_alpha", 16),
        hyperbolic_dim=ckpt.get("hyperbolic_dim", 128),
        curv=ckpt.get("curv", 1.0),
    ).to(device)
    model.clip.load_state_dict(ckpt["lora_state"])
    model.projection.load_state_dict(ckpt["projection"])
    model.eval()
    if args.features == "clip":
        return model._clip_image, f"LoRA CLIP from {Path(args.checkpoint).name}"
    return (lambda pixel: model.to_hyperbolic(model._clip_image(pixel))[1],
            f"projection head from {Path(args.checkpoint).name}")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name_to_idx = {n: i for i, n in enumerate(args.generators)}

    common = dict(root=args.dataset_path, captions_dir=args.captions_dir,
                  generators=args.generators, semantics=args.semantics,
                  processor_name=args.clip_name, max_per_class=args.max_per_class,
                  seed=args.seed)

    if args.split_manifest:
        # Same two datasets train_attribution.py builds with a manifest, so the probe
        # trains and validates on exactly the images the tabled models did.
        with open(args.split_manifest) as f:
            man = json.load(f)
        print(f"Split manifest: {len(man['train'])} train + {len(man['val'])} val")
        train_ds = IABCLIPDataset(**common, split="all",
                                  include_paths=set(man["train"]), require_caption=False)
        val_ds = IABCLIPDataset(**common, split="all", include_paths=set(man["val"]),
                                require_caption=False, include_uncaptioned=True)
    else:
        train_ds = IABCLIPDataset(**common, split="train", val_frac=args.val_frac)
        val_ds = IABCLIPDataset(**common, split="val", val_frac=args.val_frac,
                                include_uncaptioned=True)

    extract_fn, desc = build_extractor(args, device)
    print(f"Features: {desc}")

    loaders = {
        split: DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)
        for split, ds in (("train", train_ds), ("val", val_ds))
    }
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for split, loader in loaders.items():
        X, y = extract_features(extract_fn, loader, name_to_idx, device, split)
        torch.save({"X": X, "y": y, "classes": args.generators, "source": desc},
                   Path(args.out_dir) / f"clip_features_{split}.pt")
        print(f"  {split}: {tuple(X.shape)} → {args.out_dir}/clip_features_{split}.pt")


@torch.no_grad()
def extract_features(extract_fn, loader, name_to_idx, device, desc="embedding"):
    all_img, all_labels = [], []
    for batch in tqdm(loader, desc=desc):
        feats = extract_fn(batch["pixel_values"].to(device))
        all_img.append(feats.float().cpu())
        all_labels.extend([name_to_idx[g] for g in batch["generator"]])
    return torch.cat(all_img, dim=0), torch.tensor(all_labels)


if __name__ == "__main__":
    main()
