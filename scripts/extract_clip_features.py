import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
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
    p.add_argument("--out_dir",   required=True)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    name_to_idx = {n: i for i, n in enumerate(args.generators)}

    train_ds = IABCLIPDataset(
        root=args.dataset_path,
        captions_dir=args.captions_dir,
        generators=args.generators,
        semantics=args.semantics,
        processor_name=args.clip_name,
        max_per_class=args.max_per_class,
        split="train",
        val_frac=args.val_frac,
        seed=args.seed,
    )

    val_ds = IABCLIPDataset(
        root=args.dataset_path,
        captions_dir=args.captions_dir,
        generators=args.generators,
        semantics=args.semantics,
        processor_name=args.clip_name,
        max_per_class=args.max_per_class,
        split="val",
        val_frac=args.val_frac,
        seed=args.seed,
        include_uncaptioned=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = DetectorDF(
        clip_name = args.clip_name,
        num_classes = len(args.generators)
    )

    model.to(device).eval()

    X_train, y_train = extract_features(model, train_loader, name_to_idx, device)
    X_val, y_val = extract_features(model, val_loader, name_to_idx, device)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    torch.save({"X": X_train, "y": y_train, "classes": args.generators}, Path(args.out_dir) / "clip_features_train.pt")
    torch.save({"X": X_val, "y": y_val, "classes": args.generators}, Path(args.out_dir) / "clip_features_val.pt")



@torch.no_grad()
def extract_features(model, loader, name_to_idx, device):
    all_img, all_labels = [], []
    for batch in tqdm(loader, desc="embedding"):
        pixel = batch["pixel_values"].to(device)
        feats = model._clip_image(pixel)
        all_img.append(feats.cpu())
        all_labels.extend([name_to_idx[g] for g in batch["generator"]])
    x = torch.cat(all_img, dim=0)
    y = torch.tensor(all_labels) 
    return x, y


if __name__ == "__main__": 
    main()