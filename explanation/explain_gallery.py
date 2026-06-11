"""
Per-class explanation gallery for AttributionCLIP.

Unlike explain_image.py (which explains ONE image against all class cones),
this script picks one representative image *per class* — a real FLUX sample for
the FLUX class, a real SD3 sample for the SD3 class, etc. — explains each with
its own class heatmap, and assembles a side-by-side comparison grid.

This answers the question: "How does the model look at a genuine sample of each
generator?", which is what you want for comparing attribution behaviour across
generators rather than dissecting a single image.

Usage
-----
    python -m explanation.explain_gallery \\
        --checkpoint    $WORK/checkpoints/attribution_all_no_dalle_d16.pt \\
        --dataset_path  $WORK/iab_dataset \\
        --semantic      COCO \\
        --method        agcam \\
        --output_dir    $WORK/outputs/gallery/d16

Notes
-----
* One semantic is fixed (default COCO) so every class is shown on the same kind
  of content — a fair comparison. Override with --semantic.
* --image_index selects which sample per class (default 0 = first file).
* The model's predicted class is annotated per tile; a green title means the
  prediction matches the tile's true class, red means it does not.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPTokenizer

import matplotlib
matplotlib.use("Agg")  # headless: compute nodes have no display
import matplotlib.pyplot as plt

from losses.attribution_loss import predict_class
from data.iab_dataset import SEMANTIC_TO_SUPER, _images_in
from explanation.explain_image import load_checkpoint, heatmap_to_pil, overlay_heatmap
from explanation.agcam_guided import (
    encode_anchors,
    compute_agcam_heatmap,
    compute_guided_heatmap,
)


# ---------------------------------------------------------------------------
# Dataset path resolution (mirrors IABDataset)
# ---------------------------------------------------------------------------

def resolve_class_dir(root: Path, cls: str, semantic: str) -> Path:
    """Return the image directory for a (class, semantic) pair."""
    super_cat = SEMANTIC_TO_SUPER.get(semantic, semantic)
    if super_cat == semantic:
        return root / cls / semantic
    return root / cls / super_cat / semantic


def pick_image(root: Path, cls: str, semantic: str, index: int) -> Path | None:
    """Pick the index-th image for a class, or None if unavailable."""
    img_dir = resolve_class_dir(root, cls, semantic)
    if not img_dir.exists():
        print(f"  [skip] directory not found: {img_dir}")
        return None
    imgs = _images_in(img_dir)
    if not imgs:
        print(f"  [skip] no images in: {img_dir}")
        return None
    if index >= len(imgs):
        print(f"  [warn] index {index} >= {len(imgs)} for {cls}; using last image")
        return imgs[-1]
    return imgs[index]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint",  type=Path, required=True,
                   help="AttributionCLIP checkpoint (.pt).")
    p.add_argument("--dataset_path", type=Path, required=True,
                   help="IAB dataset root.")
    p.add_argument("--semantic",    type=str, default="COCO",
                   help="Semantic class to sample from for every generator.")
    p.add_argument("--image_index", type=int, default=0,
                   help="Which sample (by sorted order) to pick per class.")
    p.add_argument("--output_dir",  type=Path, default=Path("outputs/gallery"),
                   help="Directory where outputs are written.")
    p.add_argument("--method",
                   choices=["agcam", "guided"], default="agcam",
                   help="Explanation method.")
    p.add_argument("--score_mode",
                   choices=["angle", "margin"], default="margin",
                   help="Score used for backpropagation.")
    p.add_argument("--classes",     type=str, nargs="+", default=None,
                   help="Subset of classes to include. Default: all in checkpoint.")
    p.add_argument("--head_fusion",
                   choices=["sum", "mean", "max"], default="sum")
    p.add_argument("--layer_fusion",
                   choices=["sum", "mean", "max"], default="sum",
                   help="(AGCAM only).")
    p.add_argument("--no_sigmoid",  action="store_true",
                   help="(AGCAM only) Disable sigmoid on attention maps.")
    p.add_argument("--overlay_alpha", type=float, default=0.50)
    p.add_argument("--ncols",       type=int, default=6,
                   help="Columns in the comparison grid.")
    p.add_argument("--device",
                   choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


def resolve_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Grid assembly
# ---------------------------------------------------------------------------

def save_grid(tiles: list[dict], output_path: Path, ncols: int, title: str) -> None:
    """
    tiles: list of {"class": str, "pred": str, "image": PIL.Image}.
    Draws a labelled grid; green title = prediction matches the tile class.
    """
    n = len(tiles)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 2.9))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, tile in zip(axes, tiles):
        ax.imshow(tile["image"])
        correct = tile["pred"] == tile["class"]
        color = "green" if correct else "red"
        ax.set_title(
            f"{tile['class']}\n→ {tile['pred']}",
            fontsize=8, color=color,
        )
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grid → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, class_names, anchor_texts, curv = load_checkpoint(args.checkpoint, device)

    ckpt_meta = torch.load(args.checkpoint, map_location="cpu")
    clip_name = ckpt_meta["clip_name"]

    classes = args.classes if args.classes is not None else class_names
    unknown = [c for c in classes if c not in class_names]
    if unknown:
        raise ValueError(f"Unknown classes {unknown}; available: {class_names}")
    print(f"Explaining {len(classes)} classes on semantic {args.semantic!r}")

    # Shared, loaded once.
    processor = CLIPProcessor.from_pretrained(clip_name)
    tokenizer = CLIPTokenizer.from_pretrained(clip_name)
    x_anchors = encode_anchors(model, anchor_texts, tokenizer, device)

    fn = compute_agcam_heatmap if args.method == "agcam" else compute_guided_heatmap
    extra = dict(head_fusion=args.head_fusion)
    if args.method == "agcam":
        extra["layer_fusion"] = args.layer_fusion
        extra["apply_sigmoid"] = not args.no_sigmoid

    args.output_dir.mkdir(parents=True, exist_ok=True)

    heatmap_tiles: list[dict] = []
    overlay_tiles: list[dict] = []
    records: list[dict] = []

    for cls in classes:
        img_path = pick_image(args.dataset_path, cls, args.semantic, args.image_index)
        if img_path is None:
            continue

        pil_image = Image.open(img_path).convert("RGB")
        pixel_values = processor(images=pil_image, return_tensors="pt")["pixel_values"].to(device)

        # Predicted class (sanity annotation).
        with torch.no_grad():
            x_hyp, _ = model.encode_image(pixel_values)
        pred_idx = int(predict_class(x_hyp, x_anchors, curv=curv).item())
        pred_class = class_names[pred_idx]

        # Heatmap for this tile's TRUE class.
        target_idx = class_names.index(cls)
        heatmap = fn(
            model=model,
            pixel_values=pixel_values,
            x_anchors=x_anchors,
            target_class=target_idx,
            score_mode=args.score_mode,
            curv=curv,
            **extra,
        )

        heat_pil = heatmap_to_pil(heatmap, pil_image.size)
        over_pil = overlay_heatmap(pil_image, heatmap, alpha=args.overlay_alpha)

        safe = cls.replace("/", "_").replace(" ", "_")
        heat_path = args.output_dir / f"{safe}_{args.method}_heatmap.png"
        over_path = args.output_dir / f"{safe}_{args.method}_overlay.png"
        heat_pil.save(heat_path)
        over_pil.save(over_path)

        mark = "✓" if pred_class == cls else "✗"
        print(f"  [{cls:>14}]  pred={pred_class:<14} {mark}  ({img_path.name})")

        heatmap_tiles.append({"class": cls, "pred": pred_class, "image": heat_pil})
        overlay_tiles.append({"class": cls, "pred": pred_class, "image": over_pil})
        records.append({
            "class": cls,
            "predicted": pred_class,
            "correct": pred_class == cls,
            "image": str(img_path),
            "heatmap": str(heat_path),
            "overlay": str(over_path),
        })

    if not overlay_tiles:
        raise RuntimeError("No tiles were produced — check dataset paths/semantic.")

    n_correct = sum(r["correct"] for r in records)
    print(f"\nModel predicted the true class on {n_correct}/{len(records)} tiles.")

    title_suffix = (
        f"{args.method.upper()} · {args.score_mode} · {args.semantic} · "
        f"{Path(args.checkpoint).stem}"
    )
    save_grid(
        overlay_tiles,
        args.output_dir / f"gallery_{args.method}_overlays.png",
        ncols=args.ncols,
        title=f"Overlays — {title_suffix}",
    )
    save_grid(
        heatmap_tiles,
        args.output_dir / f"gallery_{args.method}_heatmaps.png",
        ncols=args.ncols,
        title=f"Heatmaps — {title_suffix}",
    )

    summary = {
        "checkpoint":  str(args.checkpoint),
        "semantic":    args.semantic,
        "method":      args.method,
        "score_mode":  args.score_mode,
        "image_index": args.image_index,
        "n_correct":   n_correct,
        "n_total":     len(records),
        "tiles":       records,
    }
    json_path = args.output_dir / f"gallery_{args.method}_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON → {json_path}")


if __name__ == "__main__":
    main()
