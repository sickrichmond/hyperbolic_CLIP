"""Plot all non-DALL-E IAB images from a checkpoint with Euclidean t-SNE."""

import argparse
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from models.attribution_clip import AttributionCLIP
from data.iab_clip_dataset import IABCLIPDataset
from training.poincare import extract_embeddings, lorentz_to_poincare


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint",    required=True)
    p.add_argument("--dataset_path",  required=True)
    p.add_argument("--captions_dir",  required=True)
    p.add_argument("--semantics",     nargs="+",
                   default=["COCO", "cat", "dog", "wild", "FFHQ", "celebahq",
                            "bedroom", "church", "classroom", "ImageNet-1k"])
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--batch_size",    type=int,   default=128)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--output_dir",    required=True)

    return p.parse_args()


def plot_tsne(embeddings: np.ndarray, labels: list[str], output_path: Path, seed=42):
    res = TSNE(n_components=2, random_state=seed).fit_transform(embeddings)
    plt.figure(figsize=(12, 10))
    labels_array = np.asarray(labels)
    for name in dict.fromkeys(labels):
        mask = labels_array == name
        plt.scatter(res[mask, 0], res[mask, 1], label=name,
                    s=2, alpha=0.3, rasterized=True)

    plt.title(f"t-SNE of Poincaré coordinates ({len(embeddings)} images)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(markerscale=3, fontsize=8, ncol=2)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    clip_name = ckpt["clip_name"]
    class_names = ckpt["class_names"]
    curv = ckpt.get("curv", 1.0)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"  classes: {class_names}")
    print(f"  curv={curv}")

    root = Path(args.dataset_path)
    generators = sorted(
        (p.name for p in root.iterdir() if p.is_dir()
         and not re.sub(r"[^a-z0-9]", "", p.name.lower()).startswith("dalle")),
        key=lambda name: (name != "real", name),
    )
    if not generators:
        raise ValueError(f"No non-DALL-E generator directories in {root}")
    print(f"  plotting all images for: {generators}")

    model = AttributionCLIP.from_checkpoint(ckpt).to(device)
    model.clip.load_state_dict(ckpt["lora_state"])
    model.projection.load_state_dict(ckpt["projection"])
    model.eval()

    dataset = IABCLIPDataset(
        root=args.dataset_path,
        captions_dir=args.captions_dir,
        generators=generators,
        semantics=args.semantics,
        processor_name=clip_name,
        split="all",
        seed=args.seed,
        require_caption=False,
    )
    if not dataset:
        raise ValueError(f"No images found in {root} for the selected semantics")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    image_embeddings, labels, _ = extract_embeddings(model, loader, device)
    poincare_points = lorentz_to_poincare(image_embeddings, curv)
    plot_tsne(poincare_points, labels, output_path=out_dir / "tsne_plot.png", seed=args.seed)


if __name__ == "__main__":
    main()

