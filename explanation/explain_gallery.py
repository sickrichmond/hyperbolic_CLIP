"""
Per-class explanation gallery for AttributionCLIP.

Unlike explain_image.py (which explains ONE image against all class cones),
this script picks one representative image *per class* — a real FLUX sample for
the FLUX class, a real SD3 sample for the SD3 class, etc. — explains each with
its own class heatmap, and assembles a side-by-side comparison grid.

By default it runs all three explanation methods (AGCAM, Guided and Chefer) and
lays them out next to the original image, one row per class:

    class │ Original │ AGCAM │ GUIDED │ CHEFER

so you can compare, on the same genuine sample of each generator, what the
methods highlight and how they differ from the raw image.

Usage
-----
    python -m explanation.explain_gallery \\
        --checkpoint    $WORK/checkpoints/attribution_all_no_dalle_d16.pt \\
        --dataset_path  $WORK/iab_dataset \\
        --semantic      COCO \\
        --output_dir    $WORK/outputs/gallery/d16          # AGCAM + Chefer

    # Restrict to a single method if you only want one:
    python -m explanation.explain_gallery ... --methods chefer

Notes
-----
* One semantic is fixed (default COCO) so every class is shown on the same kind
  of content — a fair comparison. Override with --semantic.
* --image_index selects which sample per class (default 0 = first file).
* The model's predicted class is annotated per row; a green label means the
  prediction matches the row's true class, red means it does not.
"""
from __future__ import annotations

import argparse
import json
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
    HEATMAP_METHODS,
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
    p.add_argument("--methods", nargs="+",
                   choices=list(HEATMAP_METHODS), default=["agcam", "guided", "chefer"],
                   help="Explanation method(s) to run and show side by side. "
                        "Default runs AGCAM, Guided (last-layer variant) and "
                        "Chefer (relevance rollout, Chefer et al. 2021).")
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
                   help="(AGCAM/Guided) Disable sigmoid on attention maps.")
    p.add_argument("--start_layer", type=int, default=0,
                   help="(Chefer only) First transformer layer of the rollout.")
    p.add_argument("--overlay_alpha", type=float, default=0.60,
                   help="Opacity cap for the most-salient pixels in overlays.")
    p.add_argument("--cmap", type=str, default="inferno",
                   help="Matplotlib colormap for heatmaps/overlays "
                        "(sequential, e.g. inferno/magma/viridis).")
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

def save_comparison_grid(
    rows: list[dict],
    col_labels: list[str],
    output_path: Path,
    title: str,
) -> None:
    """
    Draw a class × (original + methods) comparison grid.

    rows:       one dict per class, each with
                    {"class": str, "pred": str, "cells": {label: PIL.Image}}.
    col_labels: ordered column headers, e.g. ["Original", "AGCAM", "CHEFER"];
                every row's "cells" must provide an image for each label.

    The left column header carries the class → prediction annotation for its
    row (green when the prediction matches the class, red otherwise); the top
    row carries the method column headers.
    """
    nrows = len(rows)
    ncols = len(col_labels)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 2.6, nrows * 2.9),
        squeeze=False,
    )

    for r, row in enumerate(rows):
        correct = row["pred"] == row["class"]
        color = "green" if correct else "red"
        for c, label in enumerate(col_labels):
            ax = axes[r][c]
            ax.imshow(row["cells"][label])
            ax.axis("off")
            if c == 0:
                # Class label on the leftmost (Original) column, per row.
                head = f"{label}\n" if r == 0 else ""
                ax.set_title(
                    f"{head}{row['class']} → {row['pred']}",
                    fontsize=8, color=color,
                )
            elif r == 0:
                # Method column header on the top row only.
                ax.set_title(label, fontsize=10)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grid → {output_path}")


def _method_kwargs(method: str, args: argparse.Namespace) -> dict:
    """Keyword args each heatmap method understands, from the parsed CLI args."""
    if method == "agcam":
        return dict(
            head_fusion=args.head_fusion,
            layer_fusion=args.layer_fusion,
            apply_sigmoid=not args.no_sigmoid,
        )
    if method == "guided":
        return dict(head_fusion=args.head_fusion, apply_sigmoid=not args.no_sigmoid)
    return dict(start_layer=args.start_layer)  # chefer


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

    methods = args.methods
    method_extra = {m: _method_kwargs(m, args) for m in methods}
    print(f"Methods: {', '.join(m.upper() for m in methods)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Column layout of the comparison grids: original first, then one per method.
    overlay_cols = ["Original"] + [m.upper() for m in methods]
    heatmap_cols = ["Original"] + [m.upper() for m in methods]

    overlay_rows: list[dict] = []
    heatmap_rows: list[dict] = []
    records: list[dict] = []

    for cls in classes:
        img_path = pick_image(args.dataset_path, cls, args.semantic, args.image_index)
        if img_path is None:
            continue

        pil_image = Image.open(img_path).convert("RGB")
        pixel_values = processor(images=pil_image, return_tensors="pt")["pixel_values"].to(device)

        # Predicted class (annotation).
        with torch.no_grad():
            x_hyp, _ = model.encode_image(pixel_values)
        pred_idx = int(predict_class(x_hyp, x_anchors, curv=curv).item())
        pred_class = class_names[pred_idx]

        # One heatmap per method, all for this row's TRUE class.
        target_idx = class_names.index(cls)
        safe = cls.replace("/", "_").replace(" ", "_")

        overlay_cells = {"Original": pil_image}
        heatmap_cells = {"Original": pil_image}
        method_outputs: dict[str, dict] = {}

        for m in methods:
            heatmap = HEATMAP_METHODS[m](
                model=model,
                pixel_values=pixel_values,
                x_anchors=x_anchors,
                target_class=target_idx,
                score_mode=args.score_mode,
                curv=curv,
                **method_extra[m],
            )
            heat_pil = heatmap_to_pil(heatmap, pil_image.size, cmap_name=args.cmap)
            over_pil = overlay_heatmap(
                pil_image, heatmap, alpha=args.overlay_alpha, cmap_name=args.cmap
            )

            heat_path = args.output_dir / f"{safe}_{m}_heatmap.png"
            over_path = args.output_dir / f"{safe}_{m}_overlay.png"
            heat_pil.save(heat_path)
            over_pil.save(over_path)

            overlay_cells[m.upper()] = over_pil
            heatmap_cells[m.upper()] = heat_pil
            method_outputs[m] = {"heatmap": str(heat_path), "overlay": str(over_path)}

        mark = "✓" if pred_class == cls else "✗"
        print(f"  [{cls:>14}]  pred={pred_class:<14} {mark}  ({img_path.name})")

        overlay_rows.append({"class": cls, "pred": pred_class, "cells": overlay_cells})
        heatmap_rows.append({"class": cls, "pred": pred_class, "cells": heatmap_cells})
        records.append({
            "class": cls,
            "predicted": pred_class,
            "correct": pred_class == cls,
            "image": str(img_path),
            "methods": method_outputs,
        })

    if not records:
        raise RuntimeError("No tiles were produced — check dataset paths/semantic.")

    n_correct = sum(r["correct"] for r in records)
    print(f"\nModel predicted the true class on {n_correct}/{len(records)} tiles.")

    tag = "_".join(methods)
    title_suffix = (
        f"{'+'.join(m.upper() for m in methods)} · {args.score_mode} · "
        f"{args.semantic} · {Path(args.checkpoint).stem}"
    )
    save_comparison_grid(
        overlay_rows, overlay_cols,
        args.output_dir / f"gallery_{tag}_overlays.png",
        title=f"Overlays — {title_suffix}",
    )
    save_comparison_grid(
        heatmap_rows, heatmap_cols,
        args.output_dir / f"gallery_{tag}_heatmaps.png",
        title=f"Heatmaps — {title_suffix}",
    )

    summary = {
        "checkpoint":  str(args.checkpoint),
        "semantic":    args.semantic,
        "methods":     methods,
        "score_mode":  args.score_mode,
        "image_index": args.image_index,
        "n_correct":   n_correct,
        "n_total":     len(records),
        "tiles":       records,
    }
    json_path = args.output_dir / f"gallery_{tag}_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON → {json_path}")


if __name__ == "__main__":
    main()
